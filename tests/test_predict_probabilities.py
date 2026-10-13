from datetime import datetime
from pathlib import Path

import polars as pl
from omegaconf import OmegaConf

import medrap.predict_probabilities.probabilities as probabilities_module
from medrap.predict_probabilities.probabilities import run_predict_probabilities


def test_run_predict_probabilities_backfills_null_column_when_every_split_is_empty(
    monkeypatch, tmp_path
) -> None:
    """When no split produces any rows, the output still has a null-typed probabilities column."""
    index_only = pl.DataFrame({"subject_id": [123], "prediction_time": [datetime(2020, 1, 1)]})
    monkeypatch.setattr(
        probabilities_module,
        "run_indexed_inference",
        lambda cfg, *, module, trainer, extract: index_only,
    )

    cfg = OmegaConf.create({"output_dir": str(tmp_path)})
    output_path = run_predict_probabilities(cfg, module=object(), trainer=object())

    assert output_path == Path(tmp_path) / "probabilities.parquet"
    result = pl.read_parquet(output_path)
    assert result.columns == ["subject_id", "prediction_time", "probabilities"]
    assert result["probabilities"].to_list() == [None]
