"""Multi-task binary label creation from MEDS cohort data.

Prediction time is drawn per subject from the window
``[first_event + min_history_days, last_event - horizon_days]``; subjects whose
timeline is shorter than ``min_history_days + horizon_days`` are excluded. How the
point inside that window is drawn is set by ``anchor_strategy`` (see
:func:`_sample_prediction_anchors`): ``"uniform_lifetime"`` samples a uniformly random
timestamp, while ``"uniform_event"`` samples uniformly over the subject's real clinical
events, so every anchor lands on actual clinical activity.

Task codes are sampled uniformly at random from codes present in the train split,
excluding synthetic time tokens and birth (see :func:`_clinical_events`). There is no
positive-rate or count filtering -- a randomly chosen code can turn out rare or
degenerate (all-positive or all-negative) on a given split; that is a property of the
sampled task, not something this module tries to correct for.
"""

import json
import logging
from pathlib import Path

import meds
import numpy as np
import polars as pl

log = logging.getLogger(__name__)


def _clinical_events[FrameT: (pl.DataFrame, pl.LazyFrame)](df: FrameT) -> FrameT:
    """Restrict ``df`` to real clinical events.

    Drops synthetic ``TIMELINE//`` tokens added by the MEDS-transforms pipeline and
    ``meds.birth_code``. Birth is structurally degenerate as a task: it is the first
    event on a subject's timeline and prediction anchors are always sampled after it,
    so it can never fall inside a prediction window. ``TIMELINE//START`` sits at the
    *same* timestamp as birth, so dropping only one of the two would leave the other
    as a back door to that instant.

    Task-code selection and anchor sampling share this one filter: a code eligible as
    a *task* must also be eligible as an *anchor*, or the label semantics diverge.

    Args:
        df: Frame with a ``code`` column.

    Returns:
        ``df`` restricted to rows whose ``code`` is neither ``meds.birth_code`` nor a
        ``TIMELINE//`` token, in the input row order and of the input frame type.

    Examples:
        >>> _clinical_events(pl.DataFrame({"code": [meds.birth_code, "TIMELINE//START", "DIAG//A"]}))[
        ...     "code"
        ... ].to_list()
        ['DIAG//A']
    """
    return df.filter(
        ~pl.col("code").str.starts_with("TIMELINE//"),
        pl.col("code") != meds.birth_code,
    )


def _sample_task_codes(meds_data_dir: str | Path, num_tasks: int, seed: int) -> list[str]:
    """Sample ``num_tasks`` codes uniformly at random from the train split.

    Every code present in the train split is equally eligible, except those
    :func:`_clinical_events` drops (synthetic ``TIMELINE//`` tokens and
    ``meds.birth_code``); no positive-rate or count filtering is applied. A sampled
    code may turn out rare or degenerate (all-positive or all-negative) on a split.

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

        ``meds.birth_code`` is never eligible, even though every subject has one:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shard_dir = Path(tmpdir) / "data" / "train"
        ...     shard_dir.mkdir(parents=True)
        ...     pl.DataFrame(
        ...         {
        ...             "subject_id": [1, 1],
        ...             "code": [meds.birth_code, "DIAG//A"],
        ...             "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        ...         }
        ...     ).write_parquet(shard_dir / "0.parquet")
        ...     _sample_task_codes(tmpdir, num_tasks=1, seed=0)
        ['DIAG//A']

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

        ...and when there are some, but fewer than requested:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shard_dir = Path(tmpdir) / "data" / "train"
        ...     shard_dir.mkdir(parents=True)
        ...     pl.DataFrame(
        ...         {
        ...             "subject_id": [1],
        ...             "code": ["DIAG//A"],
        ...             "time": [datetime(2020, 1, 1)],
        ...         }
        ...     ).write_parquet(shard_dir / "0.parquet")
        ...     _sample_task_codes(tmpdir, num_tasks=5, seed=0)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Only 1 eligible task codes found in .../data/train/, fewer than the requested num_tasks=5.
    """
    data = pl.scan_parquet(Path(meds_data_dir) / "data" / "train" / "*.parquet")
    eligible = (
        _clinical_events(data.filter(pl.col("time").is_not_null()))
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
    df: pl.DataFrame,
    horizon_days: float,
    min_history_days: float,
    rng: np.random.Generator,
    anchor_strategy: str = "uniform_lifetime",
) -> pl.DataFrame | None:
    """Sample one random prediction anchor per subject within their valid window.

    The valid window is ``[first_event + min_history_days, last_event - horizon_days]``
    either way; ``anchor_strategy`` decides how a point inside it is drawn. Subjects
    whose window is empty are dropped.

    ``"uniform_lifetime"`` samples a uniformly random *timestamp* in that window. On a
    record spanning decades most draws land in stretches with no clinical activity
    nearby, so the prediction window is usually empty and labels are overwhelmingly
    negative. ``"uniform_event"`` instead samples uniformly over the subject's real
    clinical events (:func:`_clinical_events`) inside the window, so every anchor sits
    on actual clinical activity.

    Args:
        df: Shard events with ``subject_id``, ``time`` and ``code`` columns.
        horizon_days: Days after prediction time to look for code occurrence.
        min_history_days: Minimum days of history before the prediction anchor.
        rng: Random generator; consumes one ``rng.random(n_subjects)`` draw either way.
        anchor_strategy: ``"uniform_lifetime"`` (default) or ``"uniform_event"``.

    Returns:
        DataFrame with columns ``subject_id``, ``prediction_time``, or ``None``
        if no subject has a nonempty window.

    Raises:
        ValueError: If ``anchor_strategy`` is not one of the two known values.

    Examples:
        Two subjects, each with one clinical event mid-window and a long empty tail.
        ``"uniform_event"`` must land exactly on that event; ``"uniform_lifetime"``
        almost surely does not.

        >>> from datetime import datetime
        >>> events = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 1, 2, 2, 2],
        ...         "code": ["DIAG//A", "DIAG//B", "DIAG//A", "DIAG//A", "DIAG//B", "DIAG//A"],
        ...         "time": [
        ...             datetime(2000, 1, 1),
        ...             datetime(2010, 6, 1),
        ...             datetime(2020, 1, 1),
        ...             datetime(2000, 1, 1),
        ...             datetime(2010, 6, 1),
        ...             datetime(2020, 1, 1),
        ...         ],
        ...     }
        ... )
        >>> rng = np.random.default_rng(0)
        >>> anchors = _sample_prediction_anchors(events, 7.0, 1.0, rng, anchor_strategy="uniform_event")
        >>> sorted(anchors["prediction_time"].dt.date().unique().to_list())
        [datetime.date(2010, 6, 1)]

        Both strategies are reproducible for a fixed seed. This is not automatic:
        ``group_by`` row order is unspecified, so the draw is zipped positionally
        against explicitly sorted rows.

        >>> def anchors_for(strategy):
        ...     return _sample_prediction_anchors(
        ...         events, 7.0, 1.0, np.random.default_rng(0), anchor_strategy=strategy
        ...     ).sort("subject_id")
        >>> all(anchors_for(s).equals(anchors_for(s)) for s in ("uniform_lifetime", "uniform_event"))
        True

        An unknown strategy is rejected rather than silently falling back:

        >>> _sample_prediction_anchors(events, 7.0, 1.0, rng, anchor_strategy="nearest")
        Traceback (most recent call last):
            ...
        ValueError: anchor_strategy must be 'uniform_lifetime' or 'uniform_event', got 'nearest'

        ``None`` is returned when no anchor can be placed. An empty shard:

        >>> empty = events.clear()
        >>> _sample_prediction_anchors(empty, 7.0, 1.0, rng) is None
        True

        A timeline too short to hold ``min_history_days + horizon_days``:

        >>> short = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 1],
        ...         "code": ["DIAG//A", "DIAG//B"],
        ...         "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        ...     }
        ... )
        >>> _sample_prediction_anchors(short, 7.0, 1.0, rng) is None
        True

        And, under ``"uniform_event"`` only, a window that contains no *clinical*
        event -- here the sole in-window row is a synthetic ``TIMELINE//`` token, so
        there is nothing to anchor on even though the window itself is valid:

        >>> sparse = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 1],
        ...         "code": ["DIAG//A", "TIMELINE//DELTA//years", "DIAG//B"],
        ...         "time": [datetime(2000, 1, 1), datetime(2005, 6, 1), datetime(2010, 1, 1)],
        ...     }
        ... )
        >>> _sample_prediction_anchors(sparse, 7.0, 1.0, rng, anchor_strategy="uniform_event") is None
        True
    """
    if anchor_strategy not in ("uniform_lifetime", "uniform_event"):
        raise ValueError(
            f"anchor_strategy must be 'uniform_lifetime' or 'uniform_event', got {anchor_strategy!r}"
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
        # Load-bearing: group_by's row order is unspecified, and the draw below is
        # zipped positionally against these rows, so without an explicit sort the
        # same seed pairs different random numbers with different subjects run to run.
        .sort("subject_id")
    )

    if bounds.is_empty():
        return None

    if anchor_strategy == "uniform_event":
        candidates = (
            _clinical_events(df)
            .select("subject_id", pl.col("time").dt.epoch(time_unit="us").alias("event_us"))
            .join(bounds.select("subject_id", "earliest_us", "latest_us"), on="subject_id")
            .filter(pl.col("event_us").is_between(pl.col("earliest_us"), pl.col("latest_us")))
            .sort(["subject_id", "event_us"])
        )
        if candidates.is_empty():
            return None

        # Both .sort()s are load-bearing: group_by's row order is unspecified, so
        # without them row i of the rng draw would not always mean the same subject.
        counts = candidates.group_by("subject_id").agg(pl.len().alias("n_events")).sort("subject_id")
        # .astype(int64) is required: pl.len() is UInt32, and mixing it with the
        # int64 offsets below would promote the index array to float.
        n_events = counts["n_events"].to_numpy().astype(np.int64)
        block_starts = np.cumsum(n_events) - n_events  # exclusive prefix sum
        picks = block_starts + (rng.random(len(counts)) * n_events).astype(np.int64)
        subject_ids = counts["subject_id"]
        prediction_us = candidates["event_us"].to_numpy()[picks]
    else:
        earliest = bounds["earliest_us"].to_numpy()
        window = (bounds["latest_us"] - bounds["earliest_us"]).to_numpy()
        subject_ids = bounds["subject_id"]
        prediction_us = earliest + (rng.random(len(bounds)) * window).astype(np.int64)

    return (
        pl.DataFrame({"subject_id": subject_ids, "prediction_us": prediction_us})
        .with_columns(pl.from_epoch("prediction_us", time_unit="us").alias("prediction_time"))
        .select(["subject_id", "prediction_time"])
    )


def _generate_labels_shard(
    shard_path: Path,
    codes: list[str],
    horizon_days: float,
    min_history_days: float,
    rng: np.random.Generator,
    anchor_strategy: str = "uniform_lifetime",
) -> pl.DataFrame | None:
    """Build multi-task label rows for one shard with a random prediction time per subject.

    Prediction time is drawn from
    ``[first_event + min_history_days, last_event - horizon_days]`` per
    ``anchor_strategy`` (see :func:`_sample_prediction_anchors`). Subjects whose
    window is empty are dropped.

    Examples:
        Returns ``None`` when no subject in the shard has a usable window, so the
        caller can skip the shard entirely:

        >>> import tempfile
        >>> from datetime import datetime
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shard = Path(tmpdir) / "0.parquet"
        ...     pl.DataFrame(
        ...         {
        ...             "subject_id": [1, 1],
        ...             "code": ["DIAG//A", "DIAG//B"],
        ...             "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        ...         }
        ...     ).write_parquet(shard)
        ...     _generate_labels_shard(shard, ["DIAG//A"], 7.0, 1.0, np.random.default_rng(0)) is None
        True
    """
    df = pl.read_parquet(shard_path, columns=["subject_id", "time", "code"]).filter(
        pl.col("time").is_not_null()
    )
    anchors = _sample_prediction_anchors(
        df, horizon_days, min_history_days, rng, anchor_strategy=anchor_strategy
    )
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
    anchor_strategy: str = "uniform_lifetime",
) -> pl.DataFrame:
    """Generate multi-task binary labels for all shards of one split.

    Args:
        meds_data_dir: Root of a MEDS dataset.
        split: Split name (e.g. ``"train"``).
        codes: Task codes from :func:`_sample_task_codes`.
        horizon_days: Days after prediction time to look for code occurrence.
        min_history_days: Minimum days of history before the prediction anchor.
        seed: Random seed for reproducible anchor sampling.
        anchor_strategy: ``"uniform_lifetime"`` (default) or ``"uniform_event"``;
            see :func:`_sample_prediction_anchors`.

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

        An empty frame comes back when the split exists but no shard yields a
        usable anchor:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shard_dir = Path(tmpdir) / "data" / "train"
        ...     shard_dir.mkdir(parents=True)
        ...     pl.DataFrame(
        ...         {
        ...             "subject_id": [1, 1],
        ...             "code": ["A", "B"],
        ...             "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        ...         }
        ...     ).write_parquet(shard_dir / "0.parquet")
        ...     generate_labels(
        ...         tmpdir, "train", ["A"], horizon_days=7.0, min_history_days=1.0, seed=0
        ...     ).is_empty()
        True
    """
    shard_dir = Path(meds_data_dir) / "data" / split
    if not shard_dir.exists():
        raise FileNotFoundError(f"No shard directory: {shard_dir}")

    rng = np.random.default_rng(seed)
    shards = [
        s
        for f in sorted(shard_dir.glob("*.parquet"))
        if (
            s := _generate_labels_shard(
                f, codes, horizon_days, min_history_days, rng, anchor_strategy=anchor_strategy
            )
        )
        is not None
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
    anchor_strategy: str = "uniform_lifetime",
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
        anchor_strategy: ``"uniform_lifetime"`` (default) samples a uniformly random
            timestamp in each subject's valid window; ``"uniform_event"`` samples
            uniformly over their real clinical events inside it, so every anchor lands
            on actual clinical activity. See :func:`_sample_prediction_anchors`.

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
        ...             rows["subject_id"] += [subject_id, subject_id]
        ...             rows["code"] += ["DIAG//A", "TIMELINE//DELTA//years"]
        ...             rows["time"] += [start, start + timedelta(days=20)]
        ...         pl.DataFrame(rows).write_parquet(shard_dir / "0.parquet")
        ...     out_dir = Path(tmpdir) / "tasks"
        ...     returned = generate_tasks(tmpdir, out_dir, num_tasks=1, seed=0, splits=("train", "tuning"))
        ...     codes = json.loads((out_dir / "code_index.json").read_text())
        ...     is_ok = returned == out_dir and (out_dir / "train.parquet").exists()
        ...     is_ok and codes == {"0": "DIAG//A"}
        True

        A split whose subjects all have unusable windows is skipped rather than
        written as an empty parquet -- here ``tuning`` has one 2-day timeline, too
        short to hold ``min_history_days + horizon_days``:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     for split, days in (("train", 20), ("tuning", 1)):
        ...         shard_dir = Path(tmpdir) / "data" / split
        ...         shard_dir.mkdir(parents=True)
        ...         rows = {"subject_id": [], "code": [], "time": []}
        ...         for subject_id in range(5):
        ...             start = datetime(2020, 1, 1) + timedelta(days=subject_id)
        ...             rows["subject_id"] += [subject_id, subject_id]
        ...             rows["code"] += ["DIAG//A", "DIAG//B"]
        ...             rows["time"] += [start, start + timedelta(days=days)]
        ...         pl.DataFrame(rows).write_parquet(shard_dir / "0.parquet")
        ...     out_dir = Path(tmpdir) / "tasks"
        ...     _ = generate_tasks(tmpdir, out_dir, num_tasks=1, seed=0, splits=("train", "tuning"))
        ...     (out_dir / "train.parquet").exists(), (out_dir / "tuning.parquet").exists()
        (True, False)
    """
    meds_data_dir = Path(meds_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    codes = _sample_task_codes(meds_data_dir, num_tasks=num_tasks, seed=seed)

    for split in splits:
        df = generate_labels(
            meds_data_dir,
            split,
            codes,
            horizon_days,
            min_history_days,
            seed,
            anchor_strategy=anchor_strategy,
        )
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
                "anchor_strategy": anchor_strategy,
                "codes": codes,
            },
            indent=2,
        )
    )

    return output_dir
