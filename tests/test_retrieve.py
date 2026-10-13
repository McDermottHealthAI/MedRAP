from datetime import datetime
from pathlib import Path

import polars as pl
from omegaconf import OmegaConf

import medrap.retrieve.retrieval as retrieval_module
from medrap.retrieve.retrieval import run_retrieval


def test_run_retrieval_backfills_null_columns_when_every_split_is_empty(monkeypatch, tmp_path) -> None:
    """When no split produces any rows, the output still has null-typed doc_ids/doc_scores columns."""
    index_only = pl.DataFrame({"subject_id": [123], "prediction_time": [datetime(2020, 1, 1)]})
    monkeypatch.setattr(
        retrieval_module, "run_indexed_inference", lambda cfg, *, module, trainer, extract: index_only
    )

    cfg = OmegaConf.create({"output_dir": str(tmp_path)})
    output_path = run_retrieval(cfg, module=object(), trainer=object())

    assert output_path == Path(tmp_path) / "retrieved_documents.parquet"
    result = pl.read_parquet(output_path)
    assert result.columns == ["subject_id", "prediction_time", "doc_ids", "doc_scores"]
    assert result["doc_ids"].to_list() == [None]
    assert result["doc_scores"].to_list() == [None]
