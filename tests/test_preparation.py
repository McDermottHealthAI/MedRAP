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


class _Tokenizer:
    def __call__(self, texts, *, truncation, padding, max_length):
        return {
            "input_ids": [[idx + 1] * max_length for idx, _ in enumerate(texts)],
            "attention_mask": [[1] * max_length for _ in texts],
        }


class _Embedder:
    def encode(self, texts, *, batch_size, convert_to_numpy, show_progress_bar):
        return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]][: len(texts)]


TokenizerConfig = builds(_Tokenizer, populate_full_signature=False)
EmbedderConfig = builds(_Embedder, populate_full_signature=False)


def test_prepare_retrieval_dataset_from_config_runs_end_to_end(tmp_path: Path) -> None:
    source_path = tmp_path / "source"
    Dataset.from_dict({"text": ["alpha", "beta"]}).save_to_disk(str(source_path))

    cfg = PrepareRetrievalDatasetAppConfig(
        prep=PrepareRetrievalDatasetConfig(
            source=LoadHFDatasetFromDiskConfig(dataset_path=str(source_path)),
            document=OrderedFieldDocumentRendererConfig(fields=["text"]),
            tokenizer=TokenizerConfig(),
            embedder=EmbedderConfig(),
            output=RetrievalDatasetOutputConfig(output_dir=str(tmp_path / "prepared")),
        )
    )

    output_dir = prepare_retrieval_dataset_from_config(cfg)

    assert output_dir.endswith("/prepared")
    assert Path(output_dir, "retrieval.faiss").exists()
