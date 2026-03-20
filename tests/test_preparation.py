from __future__ import annotations

from pathlib import Path

from datasets import Dataset
from hydra_zen import builds

from medrap.configs import (
    LoadHFDatasetFromDiskConfig,
    OrderedFieldDocumentRendererConfig,
    PrepareRetrievalDatasetAppConfig,
    PrepareRetrievalDatasetConfig,
    RetrievalDatasetOutputConfig,
    prepare_retrieval_dataset_from_config,
)


class StubTokenizer:
    def __call__(
        self,
        texts: list[str],
        *,
        truncation: bool,
        padding: str,
        max_length: int,
    ) -> dict[str, list[list[int]]]:
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
    ) -> list[list[float]]:
        rows = []
        for text in texts:
            rows.append([1.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0])
        return rows


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
