from __future__ import annotations

from pathlib import Path

from datasets import Dataset

from medrap.configs import (
    HFTokenizerConfig,
    LoadHFDatasetFromDiskConfig,
    OrderedFieldDocumentRendererConfig,
    PrepareRetrievalDatasetAppConfig,
    PrepareRetrievalDatasetConfig,
    RetrievalDatasetOutputConfig,
    SentenceTransformerEmbedderConfig,
    prepare_retrieval_dataset_from_config,
)

TINY_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers-testing/stsb-bert-tiny-safetensors"


def test_prepare_retrieval_dataset_from_config_runs_end_to_end(tmp_path: Path) -> None:
    source_path = tmp_path / "source"
    Dataset.from_dict({"text": ["alpha", "beta"]}).save_to_disk(str(source_path))

    cfg = PrepareRetrievalDatasetAppConfig(
        prep=PrepareRetrievalDatasetConfig(
            source=LoadHFDatasetFromDiskConfig(dataset_path=str(source_path)),
            document=OrderedFieldDocumentRendererConfig(fields=["text"]),
            tokenizer=HFTokenizerConfig(pretrained_model_name_or_path=TINY_SENTENCE_TRANSFORMER_MODEL),
            embedder=SentenceTransformerEmbedderConfig(
                model_name_or_path=TINY_SENTENCE_TRANSFORMER_MODEL,
                device="cpu",
            ),
            output=RetrievalDatasetOutputConfig(output_dir=str(tmp_path / "prepared")),
        )
    )

    output_dir = prepare_retrieval_dataset_from_config(cfg)

    assert output_dir.endswith("/prepared")
    assert Path(output_dir, "retrieval.faiss").exists()
