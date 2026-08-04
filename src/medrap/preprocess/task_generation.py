"""Multi-task binary label creation from MEDS cohort data.

Prediction time is sampled uniformly at random from the per-subject window
``[first_event + min_history_days, last_event - horizon_days]``.
Subjects whose timeline is shorter than ``min_history_days + horizon_days`` are excluded.

Task codes are sampled uniformly at random from codes present in the train split,
excluding synthetic time tokens (``TIMELINE//`` prefix) added by the MEDS-transforms
pipeline (see :func:`_sample_task_codes`). There is no positive-rate or count
filtering -- a randomly chosen code can turn out rare or degenerate (all-positive or
all-negative) on a given split; that is a property of the sampled task, not something
this module tries to correct for.
"""

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


def _sample_task_codes(meds_data_dir: str | Path, num_tasks: int, seed: int) -> list[str]:
    """Sample ``num_tasks`` codes uniformly at random from the train split.

    Every code present in the train split (excluding synthetic ``TIMELINE//``
    tokens added by the MEDS-transforms pipeline) is equally eligible; no
    positive-rate or count filtering is applied. A sampled code may turn out
    rare or degenerate (all-positive or all-negative) on a given split.

    Args:
        meds_data_dir: Root of a MEDS dataset (``data/train/*.parquet``).
        num_tasks: Number of codes to sample.
        seed: Random seed.

    Returns:
        List of ``num_tasks`` code strings, sampled without replacement.

    Raises:
        ValueError: If no eligible codes are found, or fewer than ``num_tasks``
            eligible codes exist.

    Examples:
        >>> import tempfile
        >>> from datetime import datetime
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shard_dir = Path(tmpdir) / "data" / "train"
        ...     shard_dir.mkdir(parents=True)
        ...     pl.DataFrame(
        ...         {
        ...             "subject_id": [1, 1, 1],
        ...             "code": ["DIAG//A", "DIAG//B", "TIMELINE//DELTA//years"],
        ...             "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        ...         }
        ...     ).write_parquet(shard_dir / "0.parquet")
        ...     sorted(_sample_task_codes(tmpdir, num_tasks=2, seed=0))
        ['DIAG//A', 'DIAG//B']

        Raises when there aren't enough eligible codes:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shard_dir = Path(tmpdir) / "data" / "train"
        ...     shard_dir.mkdir(parents=True)
        ...     pl.DataFrame(
        ...         {
        ...             "subject_id": [1],
        ...             "code": ["TIMELINE//DELTA//years"],
        ...             "time": [datetime(2020, 1, 1)],
        ...         }
        ...     ).write_parquet(shard_dir / "0.parquet")
        ...     _sample_task_codes(tmpdir, num_tasks=5, seed=0)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: No eligible task codes found in .../data/train/.
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
    if len(eligible) < num_tasks:
        raise ValueError(
            f"Only {len(eligible)} eligible task codes found in {meds_data_dir}/data/train/, "
            f"fewer than the requested num_tasks={num_tasks}."
        )

    rng = np.random.default_rng(seed)
    return rng.choice(eligible, size=num_tasks, replace=False).tolist()


def _sample_prediction_anchors(
    df: pl.DataFrame, horizon_days: float, min_history_days: float, rng: np.random.Generator
) -> pl.DataFrame | None:
    """Sample one random prediction anchor per subject within their valid window.

    The anchor is drawn uniformly at random from the subject's own distinct
    *real event timestamps* falling in
    ``[first_event + min_history_days, last_event - horizon_days]`` -- not
    from any continuous point in that range. Clinical data is bursty (dense
    during an encounter, long silent gaps between them), so a continuous
    draw frequently lands in a gap where nothing was recorded; restricting to
    real event times guarantees the anchor always coincides with an actual
    observation (port of `EveryQuery <https://github.com/payalchandak/EveryQuery>`_'s
    ``build_prediction_times`` index-sampling idea -- see
    ``generate_tasks/sample_tasks.py``). Subjects with no qualifying event
    time are dropped.

    Args:
        df: Shard events with ``subject_id`` and ``time`` columns.
        horizon_days: Days after prediction time to look for code occurrence.
        min_history_days: Minimum days of history before the prediction anchor.
        rng: Random generator; consumes one ``rng.random(n_subjects)`` draw.

    Returns:
        DataFrame with columns ``subject_id``, ``prediction_time``, or ``None``
        if no subject has a qualifying event time.

    Examples:
        Only real event timestamps are eligible anchors -- day 0 (before the
        window) and day 20 (after it) are never selected; day 5 and day 8
        (both inside ``[first + min_history, last - horizon]``) are the only
        two candidates, so the result always lands exactly on one of them,
        never a continuous point like day 6.5:

        >>> import numpy as np
        >>> from datetime import datetime, timedelta
        >>> base = datetime(2020, 1, 1)
        >>> df = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 1, 1],
        ...         "time": [
        ...             base,
        ...             base + timedelta(days=5),
        ...             base + timedelta(days=8),
        ...             base + timedelta(days=20),
        ...         ],
        ...     }
        ... )
        >>> result = _sample_prediction_anchors(
        ...     df, horizon_days=7.0, min_history_days=1.0, rng=np.random.default_rng(0)
        ... )
        >>> result["prediction_time"][0] in (base + timedelta(days=5), base + timedelta(days=8))
        True
    """
    if df.is_empty():
        return None

    min_history_us = int(min_history_days * 86_400 * 1_000_000)
    horizon_us = int(horizon_days * 86_400 * 1_000_000)

    bounds = df.group_by("subject_id").agg(
        pl.col("time").min().dt.epoch(time_unit="us").alias("first_us"),
        pl.col("time").max().dt.epoch(time_unit="us").alias("last_us"),
    )

    # Candidate anchors: each subject's own distinct event timestamps that fall
    # inside the valid window -- sorted by subject_id then time so per-subject
    # index assignment (and the rng draw below) is deterministic regardless of
    # groupby/join execution order.
    candidates = (
        df.select("subject_id", pl.col("time").dt.epoch(time_unit="us").alias("time_us"))
        .unique()
        .join(bounds, on="subject_id")
        .filter(
            (pl.col("time_us") >= pl.col("first_us") + min_history_us)
            & (pl.col("time_us") <= pl.col("last_us") - horizon_us)
        )
        .sort(["subject_id", "time_us"])
        .with_columns(pl.int_range(pl.len()).over("subject_id").alias("candidate_idx"))
    )
    if candidates.is_empty():
        return None

    counts = candidates.group_by("subject_id").agg(pl.len().alias("n")).sort("subject_id")
    chosen_idx = (rng.random(len(counts)) * counts["n"].to_numpy()).astype(np.int64)
    chosen = pl.DataFrame({"subject_id": counts["subject_id"], "candidate_idx": chosen_idx})

    return (
        candidates.join(chosen, on=["subject_id", "candidate_idx"], how="inner")
        .with_columns(pl.from_epoch("time_us", time_unit="us").alias("prediction_time"))
        .select(["subject_id", "prediction_time"])
    )


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
    anchors = _sample_prediction_anchors(df, horizon_days, min_history_days, rng)
    if anchors is None:
        return None

    joined = anchors.join(df, on="subject_id", how="left").with_columns(
        ((pl.col("time") - pl.col("prediction_time")).dt.total_seconds() / 86400.0).alias("delta_days")
    )

    in_window = (
        joined.filter(
            (pl.col("delta_days") > 0) & (pl.col("delta_days") <= horizon_days) & pl.col("code").is_in(codes)
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
            result = result.rename({code: col}).with_columns(pl.col(col).fill_null(0.0).cast(pl.Float32))
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
        ...     df = generate_labels(
        ...         tmpdir, "train", ["A", "B"], horizon_days=7.0, min_history_days=1.0, seed=0
        ...     )
        ...     set(df.columns) == {"subject_id", "prediction_time", "task_0", "task_1"}
        True
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     generate_labels(
        ...         tmpdir, "train", ["A"], horizon_days=7.0, min_history_days=1.0, seed=0
        ...     )  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        FileNotFoundError: No shard directory: .../data/train
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

    Task codes are sampled uniformly at random from codes present in the train
    split (excluding synthetic ``TIMELINE//`` tokens); see
    :func:`_sample_task_codes`. No positive-rate or count filtering is applied.
    Prediction time is sampled independently per split, per subject; see
    :func:`_sample_prediction_anchors`.

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
        >>> from datetime import datetime, timedelta
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     for split in ("train", "tuning"):
        ...         shard_dir = Path(tmpdir) / "data" / split
        ...         shard_dir.mkdir(parents=True)
        ...         rows = {"subject_id": [], "code": [], "time": []}
        ...         for subject_id in range(5):
        ...             start = datetime(2020, 1, 1) + timedelta(days=subject_id)
        ...             rows["subject_id"] += [subject_id, subject_id, subject_id]
        ...             rows["code"] += ["TIMELINE//DELTA//years", "DIAG//A", "TIMELINE//DELTA//years"]
        ...             rows["time"] += [start, start + timedelta(days=5), start + timedelta(days=20)]
        ...         pl.DataFrame(rows).write_parquet(shard_dir / "0.parquet")
        ...     out_dir = Path(tmpdir) / "tasks"
        ...     returned = generate_tasks(tmpdir, out_dir, num_tasks=1, seed=0, splits=("train", "tuning"))
        ...     codes = json.loads((out_dir / "code_index.json").read_text())
        ...     is_ok = returned == out_dir and (out_dir / "train.parquet").exists()
        ...     is_ok and codes == {"0": "DIAG//A"}
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
