from datetime import datetime
from pathlib import Path

import polars as pl
from omegaconf import OmegaConf

import medrap.get_embeddings.embeddings as embeddings_module
from medrap.get_embeddings.embeddings import run_get_embeddings


def test_run_get_embeddings_backfills_null_column_when_every_split_is_empty(monkeypatch, tmp_path) -> None:
    """When no split produces any rows, the output still has a null-typed ``embedding`` column."""
    index_only = pl.DataFrame({"subject_id": [123], "prediction_time": [datetime(2020, 1, 1)]})
    monkeypatch.setattr(
        embeddings_module, "run_indexed_inference", lambda cfg, *, module, trainer, extract: index_only
    )

    cfg = OmegaConf.create({"output_dir": str(tmp_path)})
    output_path = run_get_embeddings(cfg, module=object(), trainer=object())

    assert output_path == Path(tmp_path) / "embeddings.parquet"
    result = pl.read_parquet(output_path)
    assert result.columns == ["subject_id", "prediction_time", "embedding"]
    assert result["embedding"].to_list() == [None]
