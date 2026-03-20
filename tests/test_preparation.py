from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset, load_dataset, load_from_disk
from hydra_zen import builds

from medrap.configs import (
    LoadHFDatasetFromDiskConfig,
    OrderedFieldDocumentRendererConfig,
    PrepareRetrievalDatasetAppConfig,
    PrepareRetrievalDatasetConfig,
    RetrievalDatasetOutputConfig,
    prepare_retrieval_dataset_from_config,
)
from medrap.preparation import OrderedFieldDocumentRenderer, prepare_retrieval_dataset
from medrap.retrievers import load_hf_dataset_retriever


class StubTokenizer:
    def __call__(
        self,
        texts: list[str],
        *,
        truncation: bool,
        padding: str,
        max_length: int,
    ) -> dict[str, list[list[int]]]:
        assert truncation is True
        assert padding == "max_length"
        return {
            "input_ids": [[idx + 1] * max_length for idx, _text in enumerate(texts)],
            "attention_mask": [[1] * max_length for _text in texts],
        }


class StubEmbedder:
    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        assert batch_size >= 1
        assert convert_to_numpy is True
        assert show_progress_bar is False
        rows = []
        for text in texts:
            rows.append([1.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)


def test_load_hf_dataset_source_supports_local_data_files(tmp_path: Path) -> None:
    data_file = tmp_path / "records.jsonl"
    data_file.write_text('{"question": "alpha"}\n{"question": "beta"}\n')

    dataset = load_dataset(path="json", split="train", data_files=str(data_file))

    assert len(dataset) == 2
    assert dataset["question"] == ["alpha", "beta"]


def test_ordered_field_document_renderer_rejects_empty_fields() -> None:
    with pytest.raises(ValueError, match="fields must contain at least one field name"):
        OrderedFieldDocumentRenderer(fields=[])


def test_load_hf_dataset_from_disk_round_trips_saved_dataset(tmp_path: Path) -> None:
    source_path = tmp_path / "source"
    Dataset.from_dict({"text": ["alpha", "beta"]}).save_to_disk(str(source_path))

    dataset = load_from_disk(str(source_path))

    assert len(dataset) == 2
    assert dataset["text"] == ["alpha", "beta"]


def test_prepare_retrieval_dataset_saves_static_artifact(tmp_path: Path) -> None:
    source = Dataset.from_dict(
        {
            "title": ["Alpha title", "Beta title"],
            "body": ["alpha body", "beta body"],
            "source_id": [11, 22],
        }
    )

    output_dir = prepare_retrieval_dataset(
        dataset=source,
        renderer=OrderedFieldDocumentRenderer(fields=["title", "body"], separator=" | "),
        tokenizer=StubTokenizer(),
        embedder=StubEmbedder(),
        output_dir=tmp_path / "prepared",
        source_id_column="source_id",
        max_length=4,
    )

    assert output_dir == tmp_path / "prepared"
    assert (output_dir / "retrieval.faiss").exists()

    reloaded = load_from_disk(str(output_dir))
    assert reloaded["doc_text"] == ["Alpha title | alpha body", "Beta title | beta body"]
    assert reloaded["doc_ids"] == [11, 22]
    assert reloaded["doc_tokens"] == [[1, 1, 1, 1], [2, 2, 2, 2]]
    assert reloaded["doc_attention_mask"] == [[1, 1, 1, 1], [1, 1, 1, 1]]
    assert reloaded["doc_key_embeddings"] == [[1.0, 0.0], [0.0, 1.0]]


def test_prepare_retrieval_dataset_rejects_non_integer_source_ids(tmp_path: Path) -> None:
    source = Dataset.from_dict(
        {
            "text": ["alpha", "beta"],
            "source_id": ["doc-a", "doc-b"],
        }
    )

    with pytest.raises(ValueError, match="doc_ids values must be integer-like"):
        prepare_retrieval_dataset(
            dataset=source,
            renderer=OrderedFieldDocumentRenderer(fields=["text"]),
            tokenizer=StubTokenizer(),
            embedder=StubEmbedder(),
            output_dir=tmp_path / "prepared",
            source_id_column="source_id",
            max_length=4,
        )


def test_prepare_retrieval_dataset_from_config_runs_end_to_end(tmp_path: Path) -> None:
    source_path = tmp_path / "source"
    Dataset.from_dict({"text": ["alpha", "beta"]}).save_to_disk(str(source_path))

    tokenizer_cfg = builds(StubTokenizer, zen_dataclass={"cls_name": "StubTokenizerConfig"})
    embedder_cfg = builds(StubEmbedder, zen_dataclass={"cls_name": "StubEmbedderConfig"})
    cfg = PrepareRetrievalDatasetAppConfig(
        prep=PrepareRetrievalDatasetConfig(
            source=LoadHFDatasetFromDiskConfig(dataset_path=str(source_path)),
            document=OrderedFieldDocumentRendererConfig(fields=["text"]),
            tokenizer=tokenizer_cfg,
            embedder=embedder_cfg,
            output=RetrievalDatasetOutputConfig(output_dir=str(tmp_path / "prepared")),
        )
    )

    output_dir = prepare_retrieval_dataset_from_config(cfg)

    assert output_dir.endswith("/prepared")
    assert Path(output_dir, "retrieval.faiss").exists()


def test_prepared_artifact_loads_into_hf_dataset_retriever(tmp_path: Path) -> None:
    source = Dataset.from_dict({"text": ["alpha", "beta"]})
    output_dir = prepare_retrieval_dataset(
        dataset=source,
        renderer=OrderedFieldDocumentRenderer(fields=["text"]),
        tokenizer=StubTokenizer(),
        embedder=StubEmbedder(),
        output_dir=tmp_path / "prepared",
        max_length=3,
    )

    retriever = load_hf_dataset_retriever(
        dataset_path=str(output_dir),
        index_name="retrieval",
        doc_tokens_column="doc_tokens",
        doc_attention_mask_column="doc_attention_mask",
        doc_ids_column="doc_ids",
        doc_key_embeddings_column="doc_key_embeddings",
        k=1,
    )

    out = retriever.retrieve(torch.FloatTensor([[[1.0, 0.0]], [[0.0, 1.0]]]))

    assert tuple(out.doc_tokens.shape) == (2, 1, 1, 3)
    assert out.doc_ids is not None
    assert out.doc_ids.tolist() == [[[0]], [[1]]]
    assert out.doc_key_embeddings is not None
    assert out.doc_key_embeddings.tolist() == [[[[1.0, 0.0]]], [[[0.0, 1.0]]]]
