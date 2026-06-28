"""Multi-task binary label creation from MEDS cohort data.

Prediction time is sampled uniformly at random from the per-subject window
``[first_event + min_history_days, last_event - horizon_days]``.
Subjects whose timeline is shorter than ``min_history_days + horizon_days`` are excluded.

Task codes are sampled randomly from codes present in the train split, excluding
synthetic time tokens (``TIMELINE//`` prefix) added by the MEDS-transforms pipeline.
Rare-code and sparse-subject filtering is handled upstream by the MEDS-transforms
pipeline stage, so no frequency filtering is applied here.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl


def _sample_task_codes(meds_data_dir: str | Path, num_tasks: int, seed: int) -> list[str]:
    """Sample ``num_tasks`` codes at random from the train split.

    Excludes synthetic ``TIMELINE//`` tokens added by the MEDS-transforms pipeline.

    Args:
        meds_data_dir: Root of a MEDS dataset (``data/train/*.parquet``).
        num_tasks: Number of codes to sample.
        seed: Random seed.

    Returns:
        List of sampled code strings (length ``<= num_tasks``).

    Raises:
        ValueError: If no eligible codes are found.

    Examples:
        >>> import tempfile
        >>> from datetime import datetime
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shard_dir = Path(tmpdir) / "data" / "train"
        ...     shard_dir.mkdir(parents=True)
        ...     pl.DataFrame(
        ...         {
        ...             "subject_id": [1, 1, 2],
        ...             "code": ["DIAG//A", "TIMELINE//DELTA//years", "DIAG//A"],
        ...             "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 1)],
        ...         }
        ...     ).write_parquet(shard_dir / "0.parquet")
        ...     codes = _sample_task_codes(tmpdir, num_tasks=5, seed=0)
        ...     codes == ["DIAG//A"]
        True
    """
    data = pl.scan_parquet(Path(meds_data_dir) / "data" / "train" / "*.parquet")
    eligible = (
        data.filter(pl.col("time").is_not_null())
        .filter(~pl.col("code").str.starts_with("TIMELINE//"))
        .select("code")
        .unique()
        .collect()["code"]
        .to_list()
    )
    if not eligible:
        raise ValueError(f"No eligible task codes found in {meds_data_dir}/data/train/.")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=min(num_tasks, len(eligible)), replace=False)
    return chosen.tolist()


def _generate_labels_shard(
    shard_path: Path,
    codes: list[str],
    horizon_days: float,
    min_history_days: float,
    rng: np.random.Generator,
) -> pl.DataFrame | None:
    """Build multi-task label rows for one shard with a random prediction time per subject.

    Prediction time is sampled uniformly from
    ``[first_event + min_history_days, last_event - horizon_days]``.
    Subjects whose window is empty are dropped.
    """
    df = pl.read_parquet(shard_path, columns=["subject_id", "time", "code"]).filter(
        pl.col("time").is_not_null()
    )
    if df.is_empty():
        return None

    min_history_us = int(min_history_days * 86_400 * 1_000_000)
    horizon_us = int(horizon_days * 86_400 * 1_000_000)

    bounds = (
        df.group_by("subject_id")
        .agg(
            pl.col("time").min().dt.epoch(time_unit="us").alias("first_us"),
            pl.col("time").max().dt.epoch(time_unit="us").alias("last_us"),
        )
        .with_columns(
            (pl.col("first_us") + min_history_us).alias("earliest_us"),
            (pl.col("last_us") - horizon_us).alias("latest_us"),
        )
        .filter(pl.col("earliest_us") <= pl.col("latest_us"))
    )

    if bounds.is_empty():
        return None

    earliest = bounds["earliest_us"].to_numpy()
    window = (bounds["latest_us"] - bounds["earliest_us"]).to_numpy()
    offsets = (rng.random(len(bounds)) * window).astype(np.int64)

    anchors = (
        pl.DataFrame({"subject_id": bounds["subject_id"], "prediction_us": earliest + offsets})
        .with_columns(pl.from_epoch("prediction_us", time_unit="us").alias("prediction_time"))
        .select(["subject_id", "prediction_time"])
    )

    joined = anchors.join(df, on="subject_id", how="left").with_columns(
        ((pl.col("time") - pl.col("prediction_time")).dt.total_seconds() / 86400.0).alias("delta_days")
    )

    in_window = (
        joined.filter(
            (pl.col("delta_days") > 0)
            & (pl.col("delta_days") <= horizon_days)
            & pl.col("code").is_in(codes)
        )
        .group_by(["subject_id", "prediction_time", "code"])
        .agg(pl.len().alias("_n"))
        .with_columns(pl.lit(1.0).alias("occurred"))
        .drop("_n")
    )

    if in_window.is_empty():
        result = anchors
    else:
        wide = in_window.pivot(
            values="occurred",
            index=["subject_id", "prediction_time"],
            on="code",
            aggregate_function="first",
        )
        result = anchors.join(wide, on=["subject_id", "prediction_time"], how="left")

    for i, code in enumerate(codes):
        col = f"task_{i}"
        if code in result.columns:
            result = result.rename({code: col}).with_columns(
                pl.col(col).fill_null(0.0).cast(pl.Float32)
            )
        else:
            result = result.with_columns(pl.lit(0.0).cast(pl.Float32).alias(col))

    return result.select(["subject_id", "prediction_time", *[f"task_{i}" for i in range(len(codes))]])


def generate_labels(
    meds_data_dir: str | Path,
    split: str,
    codes: list[str],
    horizon_days: float,
    min_history_days: float,
    seed: int,
) -> pl.DataFrame:
    """Generate multi-task binary labels for all shards of one split.

    Args:
        meds_data_dir: Root of a MEDS dataset.
        split: Split name (e.g. ``"train"``).
        codes: Task codes from :func:`_sample_task_codes`.
        horizon_days: Days after prediction time to look for code occurrence.
        min_history_days: Minimum days of history before the prediction anchor.
        seed: Random seed for reproducible anchor sampling.

    Returns:
        DataFrame with columns ``subject_id``, ``prediction_time``, ``task_0``, …

    Examples:
        >>> import tempfile
        >>> from datetime import datetime
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shard_dir = Path(tmpdir) / "data" / "train"
        ...     shard_dir.mkdir(parents=True)
        ...     pl.DataFrame(
        ...         {
        ...             "subject_id": [1, 1, 1, 1],
        ...             "code": ["A", "B", "A", "B"],
        ...             "time": [
        ...                 datetime(2020, 1, 1),
        ...                 datetime(2020, 1, 5),
        ...                 datetime(2020, 1, 10),
        ...                 datetime(2020, 1, 15),
        ...             ],
        ...         }
        ...     ).write_parquet(shard_dir / "0.parquet")
        ...     df = generate_labels(tmpdir, "train", ["A", "B"], horizon_days=7.0, min_history_days=1.0, seed=0)
        ...     set(df.columns) == {"subject_id", "prediction_time", "task_0", "task_1"}
        True
    """
    shard_dir = Path(meds_data_dir) / "data" / split
    if not shard_dir.exists():
        raise FileNotFoundError(f"No shard directory: {shard_dir}")

    rng = np.random.default_rng(seed)
    shards = [
        s
        for f in sorted(shard_dir.glob("*.parquet"))
        if (s := _generate_labels_shard(f, codes, horizon_days, min_history_days, rng)) is not None
    ]
    if not shards:
        return pl.DataFrame()
    return pl.concat(shards)


def generate_tasks(
    meds_data_dir: str | Path,
    output_dir: str | Path,
    *,
    num_tasks: int = 25,
    horizon_days: float = 7.0,
    min_history_days: float = 1.0,
    seed: int = 42,
    splits: tuple[str, ...] = ("train", "tuning", "held_out"),
) -> Path:
    """Create multi-task binary labels from a MEDS cohort.

    Task codes are sampled randomly from codes present in the train split
    (excluding synthetic ``TIMELINE//`` tokens). Rare-code filtering is handled
    upstream by the MEDS-transforms pipeline.

    Args:
        meds_data_dir: Root of a MEDS dataset (``data/{split}/*.parquet``).
        output_dir: Directory to write output files into.
        num_tasks: Number of task codes to sample.
        horizon_days: Days after prediction time to look for code occurrence.
        min_history_days: Minimum days of history before the prediction anchor.
        seed: Random seed for task selection and anchor sampling.
        splits: Splits to generate labels for.

    Returns:
        ``output_dir`` as a ``Path``.

    Examples:
        >>> import tempfile
        >>> from datetime import datetime
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     for split in ("train", "tuning"):
        ...         shard_dir = Path(tmpdir) / "data" / split
        ...         shard_dir.mkdir(parents=True)
        ...         pl.DataFrame(
        ...             {
        ...                 "subject_id": [1, 1, 1, 2, 2, 2],
        ...                 "code": ["X", "Y", "X", "X", "Z", "Y"],
        ...                 "time": [
        ...                     datetime(2020, 1, 1), datetime(2020, 1, 10), datetime(2020, 1, 20),
        ...                     datetime(2020, 2, 1), datetime(2020, 2, 10), datetime(2020, 2, 20),
        ...                 ],
        ...             }
        ...         ).write_parquet(shard_dir / "0.parquet")
        ...     out_dir = Path(tmpdir) / "tasks"
        ...     returned = generate_tasks(
        ...         tmpdir, out_dir, num_tasks=3, seed=0, splits=("train", "tuning")
        ...     )
        ...     returned == out_dir and (out_dir / "train.parquet").exists() and (out_dir / "code_index.json").exists()
        True
    """
    meds_data_dir = Path(meds_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    codes = _sample_task_codes(meds_data_dir, num_tasks=num_tasks, seed=seed)

    for split in splits:
        df = generate_labels(meds_data_dir, split, codes, horizon_days, min_history_days, seed)
        if df.is_empty():
            continue
        df.write_parquet(output_dir / f"{split}.parquet")

    (output_dir / "code_index.json").write_text(
        json.dumps({str(i): code for i, code in enumerate(codes)}, indent=2)
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "num_tasks": len(codes),
                "horizon_days": horizon_days,
                "min_history_days": min_history_days,
                "seed": seed,
                "codes": codes,
            },
            indent=2,
        )
    )

    return output_dir
