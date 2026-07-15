"""Multi-task binary label creation from MEDS cohort data.

Prediction time is sampled uniformly at random from the per-subject window
``[first_event + min_history_days, last_event - horizon_days]``.
Subjects whose timeline is shorter than ``min_history_days + horizon_days`` are excluded.

Task codes are sampled randomly from codes present in the train split, excluding
synthetic time tokens (``TIMELINE//`` prefix) added by the MEDS-transforms pipeline,
then filtered to codes with an *in-window* positive rate within
``[min_positive_rate, max_positive_rate]`` and an absolute count of at least
``min_positive_count``, all three checked independently on *every* split that will
be generated -- not just train (see :func:`_sample_task_codes`). The upstream
MEDS-transforms ``min_subjects_per_code`` filter only guarantees a code occurred
somewhere in a subject's lifetime record; it says nothing about how often the code
falls inside one subject's randomly-sampled ``horizon_days``-wide prediction window,
which is what the task label actually measures. A code can also clear an
absolute-count bar on the (typically much larger) train split while its positive
rate is low enough that the same rate, applied to a smaller tuning/held_out split,
rounds down to zero or one positive subject there -- collapsing that split's label
to a single class, with no learning signal and an undefined/dropped validation
AUROC for that task. Checking rate + count on every split at selection time (using
the same windowed-occurrence definition and anchor sampling that will actually be
used to build the labels) guarantees a chosen code cannot collapse in any split
that ships.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl


def _sample_task_codes(
    meds_data_dir: str | Path,
    num_tasks: int,
    seed: int,
    *,
    horizon_days: float = 7.0,
    min_history_days: float = 1.0,
    min_positive_count: int = 10,
    min_positive_rate: float = 0.01,
    max_positive_rate: float = 0.5,
    splits: tuple[str, ...] = ("train", "tuning", "held_out"),
) -> list[str]:
    """Sample ``num_tasks`` codes whose in-window positive rate is learnable on every split.

    Every code present in the train split (excluding synthetic ``TIMELINE//``
    tokens) is windowed-tested via :func:`_count_in_window_positive_subjects`
    (a memory-efficient group-by, not a wide pivot, so this scales to testing
    every eligible code regardless of how many there are) on *each* split in
    ``splits``, then ``num_tasks`` codes are sampled uniformly from those that
    pass, on *every* split, both an absolute floor (``min_positive_count``) and
    a rate band (``min_positive_rate`` to ``max_positive_rate``) of in-window
    positive subjects.

    Checking every split -- not just train -- matters because splits differ in
    size: a code can clear an absolute count on a large train split while its
    positive *rate*, applied to a much smaller tuning/held_out split, implies an
    expected positive count near zero there, collapsing that split's label to a
    single class (no learning signal, dropped/undefined AUROC for that task). A
    rate-based bound scales with split size in a way an absolute count alone
    cannot; the absolute floor still guards against a code that clears the rate
    band by chance in a nearly-empty split.

    On a real long-tailed clinical vocabulary, only a small fraction of
    lifetime-frequent codes also clear the much narrower windowed bar, so
    testing every eligible code (rather than a random subsample) avoids
    under-sampling the pass rate and failing to find enough tasks.

    Args:
        meds_data_dir: Root of a MEDS dataset (``data/train/*.parquet``).
        num_tasks: Number of codes to sample.
        seed: Random seed.
        horizon_days: Days after prediction time to look for code occurrence
            (must match the value passed to label generation).
        min_history_days: Minimum days of history before the prediction anchor
            (must match the value passed to label generation).
        min_positive_count: Minimum number of in-window positive subjects a code
            must have on every split in ``splits`` to be eligible as a task.
        min_positive_rate: Minimum in-window positive rate (positive subjects /
            subjects with a valid prediction window) a code must have on every
            split in ``splits``.
        max_positive_rate: Maximum in-window positive rate a code may have on
            every split in ``splits`` (guards against near-constant "always
            occurs" codes, symmetric to the rare-code guard above).
        splits: Splits the code must pass the rate/count bounds on; should match
            the splits label generation will actually produce.

    Returns:
        List of ``num_tasks`` code strings, each within the rate/count bounds
        on every split in ``splits``.

    Raises:
        ValueError: If no eligible codes are found, or fewer than ``num_tasks``
            eligible codes meet the rate/count thresholds on every split.

    Examples:
        >>> import tempfile
        >>> from datetime import datetime, timedelta
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shard_dir = Path(tmpdir) / "data" / "train"
        ...     shard_dir.mkdir(parents=True)
        ...     rows = {"subject_id": [], "code": [], "time": []}
        ...     for subject_id in range(40):
        ...         start = datetime(2020, 1, 1) + timedelta(days=subject_id)
        ...         rows["subject_id"] += [subject_id, subject_id, subject_id]
        ...         # NEVER sits at t=0 (always before the prediction window, so it never
        ...         # counts as an in-window positive). The code at t=5 days always falls
        ...         # inside the window given the default horizon_days=7, min_history_days=1;
        ...         # the TIMELINE marker at t=10 days opens a wide enough window to begin with.
        ...         # COMMON gets it for 10/40 subjects (rate 0.25, passes); RARE gets it for
        ...         # the other 30/40 (rate 0.75, fails max_positive_rate=0.5).
        ...         mid_code = "DIAG//COMMON" if subject_id < 10 else "DIAG//RARE"
        ...         rows["code"] += ["DIAG//NEVER", mid_code, "TIMELINE//DELTA//years"]
        ...         rows["time"] += [start, start + timedelta(days=5), start + timedelta(days=10)]
        ...     pl.DataFrame(rows).write_parquet(shard_dir / "0.parquet")
        ...     codes = _sample_task_codes(
        ...         tmpdir, num_tasks=1, seed=0, min_positive_count=10, splits=("train",)
        ...     )
        ...     codes == ["DIAG//COMMON"]
        True
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
        ...     _sample_task_codes(
        ...         tmpdir, num_tasks=1, seed=0, min_positive_count=10, splits=("train",)
        ...     )  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: Only 0/1 eligible codes had >= 10 in-window positive subjects and rate...
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
        ...     _sample_task_codes(tmpdir, num_tasks=5, seed=0, splits=("train",))  # doctest: +ELLIPSIS
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

    per_split = {
        split: _count_in_window_positive_subjects(
            meds_data_dir, split, eligible, horizon_days, min_history_days, seed
        )
        for split in splits
    }

    def _passes(code: str) -> bool:
        for counts, total in per_split.values():
            if total == 0:
                return False
            count = counts.get(code, 0)
            rate = count / total
            if count < min_positive_count or not (min_positive_rate <= rate <= max_positive_rate):
                return False
        return True

    passing = [code for code in eligible if _passes(code)]
    if len(passing) < num_tasks:
        raise ValueError(
            f"Only {len(passing)}/{num_tasks} eligible codes had >= {min_positive_count} "
            f"in-window positive subjects and rate in [{min_positive_rate}, {max_positive_rate}] "
            f"on every split in {splits} (horizon_days={horizon_days}, "
            f"min_history_days={min_history_days}) among {len(eligible)} eligible codes. "
            "Lower min_positive_count/min_positive_rate, raise max_positive_rate, or lower num_tasks."
        )

    rng = np.random.default_rng(seed)
    chosen = rng.choice(passing, size=num_tasks, replace=False)
    return chosen.tolist()


def _sample_prediction_anchors(
    df: pl.DataFrame, horizon_days: float, min_history_days: float, rng: np.random.Generator
) -> pl.DataFrame | None:
    """Sample one random prediction anchor per subject within their valid window.

    Prediction time is sampled uniformly from
    ``[first_event + min_history_days, last_event - horizon_days]``. Subjects
    whose window is empty are dropped.

    Args:
        df: Shard events with ``subject_id`` and ``time`` columns.
        horizon_days: Days after prediction time to look for code occurrence.
        min_history_days: Minimum days of history before the prediction anchor.
        rng: Random generator; consumes one ``rng.random(n_subjects)`` draw.

    Returns:
        DataFrame with columns ``subject_id``, ``prediction_time``, or ``None``
        if no subject has a nonempty window.
    """
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

    return (
        pl.DataFrame({"subject_id": bounds["subject_id"], "prediction_us": earliest + offsets})
        .with_columns(pl.from_epoch("prediction_us", time_unit="us").alias("prediction_time"))
        .select(["subject_id", "prediction_time"])
    )


def _count_in_window_positive_subjects(
    meds_data_dir: str | Path,
    split: str,
    candidate_codes: list[str],
    horizon_days: float,
    min_history_days: float,
    seed: int,
) -> tuple[dict[str, int], int]:
    """Count in-window positive subjects per candidate code, without a wide pivot.

    Uses the same windowed-occurrence definition and anchor sampling as
    :func:`generate_labels` (same seed reproduces identical anchors, since anchor
    sampling depends only on each subject's event bounds, not on which codes are
    being tested), but aggregates counts as a long ``code -> count`` table instead
    of a ``(subject, code)`` wide matrix. This lets it scale to testing every
    eligible code at once, which a wide pivot cannot do memory-efficiently.

    Args:
        meds_data_dir: Root of a MEDS dataset.
        split: Split name (e.g. ``"train"``).
        candidate_codes: Codes to count.
        horizon_days: Days after prediction time to look for code occurrence.
        min_history_days: Minimum days of history before the prediction anchor.
        seed: Random seed for reproducible anchor sampling.

    Returns:
        Tuple of (dict mapping code -> number of distinct subjects with an
        in-window occurrence, omitting codes with zero; total number of subjects
        with a valid prediction window on this split, i.e. the rate denominator).

    Examples:
        >>> import tempfile
        >>> from datetime import datetime, timedelta
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shard_dir = Path(tmpdir) / "data" / "train"
        ...     shard_dir.mkdir(parents=True)
        ...     rows = {"subject_id": [], "code": [], "time": []}
        ...     for subject_id in range(3):
        ...         start = datetime(2020, 1, 1) + timedelta(days=subject_id)
        ...         rows["subject_id"] += [subject_id, subject_id, subject_id]
        ...         rows["code"] += ["DIAG//RARE", "DIAG//COMMON", "TIMELINE//DELTA//years"]
        ...         rows["time"] += [start, start + timedelta(days=5), start + timedelta(days=10)]
        ...     pl.DataFrame(rows).write_parquet(shard_dir / "0.parquet")
        ...     _count_in_window_positive_subjects(
        ...         tmpdir,
        ...         "train",
        ...         ["DIAG//RARE", "DIAG//COMMON"],
        ...         horizon_days=7.0,
        ...         min_history_days=1.0,
        ...         seed=0,
        ...     )
        ({'DIAG//COMMON': 3}, 3)
    """
    shard_dir = Path(meds_data_dir) / "data" / split
    if not shard_dir.exists():
        raise FileNotFoundError(f"No shard directory: {shard_dir}")

    rng = np.random.default_rng(seed)
    counts: dict[str, int] = {}
    total_subjects = 0
    for shard_path in sorted(shard_dir.glob("*.parquet")):
        df = pl.read_parquet(shard_path, columns=["subject_id", "time", "code"]).filter(
            pl.col("time").is_not_null()
        )
        anchors = _sample_prediction_anchors(df, horizon_days, min_history_days, rng)
        if anchors is None:
            continue
        total_subjects += anchors.height

        events = df.filter(pl.col("code").is_in(candidate_codes))
        joined = anchors.join(events, on="subject_id", how="inner").with_columns(
            ((pl.col("time") - pl.col("prediction_time")).dt.total_seconds() / 86400.0).alias("delta_days")
        )
        in_window = joined.filter((pl.col("delta_days") > 0) & (pl.col("delta_days") <= horizon_days))
        if in_window.is_empty():
            continue

        grouped = in_window.group_by("code").agg(pl.col("subject_id").n_unique().alias("n"))
        for code, n in zip(grouped["code"], grouped["n"], strict=False):
            counts[code] = counts.get(code, 0) + n

    return counts, total_subjects


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
    min_positive_count: int = 10,
    min_positive_rate: float = 0.01,
    max_positive_rate: float = 0.5,
    splits: tuple[str, ...] = ("train", "tuning", "held_out"),
) -> Path:
    """Create multi-task binary labels from a MEDS cohort.

    Task codes are sampled randomly from codes present in the train split
    (excluding synthetic ``TIMELINE//`` tokens), then filtered to codes with an
    in-window positive rate/count that's learnable on *every* split in
    ``splits``; see :func:`_sample_task_codes`.

    Args:
        meds_data_dir: Root of a MEDS dataset (``data/{split}/*.parquet``).
        output_dir: Directory to write output files into.
        num_tasks: Number of task codes to sample.
        horizon_days: Days after prediction time to look for code occurrence.
        min_history_days: Minimum days of history before the prediction anchor.
        seed: Random seed for task selection and anchor sampling.
        min_positive_count: Minimum in-window positive subjects a candidate code
            must have on every split in ``splits`` to be selected as a task.
        min_positive_rate: Minimum in-window positive rate a candidate code must
            have on every split in ``splits``.
        max_positive_rate: Maximum in-window positive rate a candidate code may
            have on every split in ``splits``.
        splits: Splits to generate labels for, and to require the rate/count
            bounds hold on.

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
        ...         for subject_id in range(40):
        ...             start = datetime(2020, 1, 1) + timedelta(days=subject_id)
        ...             rows["subject_id"] += [subject_id, subject_id, subject_id]
        ...             # See _sample_task_codes docstring: COMMON gets a 0.25 in-window
        ...             # rate (passes), RARE gets 0.75 (fails max_positive_rate=0.5).
        ...             mid_code = "DIAG//COMMON" if subject_id < 10 else "DIAG//RARE"
        ...             rows["code"] += ["DIAG//NEVER", mid_code, "TIMELINE//DELTA//years"]
        ...             rows["time"] += [start, start + timedelta(days=5), start + timedelta(days=10)]
        ...         pl.DataFrame(rows).write_parquet(shard_dir / "0.parquet")
        ...     out_dir = Path(tmpdir) / "tasks"
        ...     returned = generate_tasks(
        ...         tmpdir, out_dir, num_tasks=1, seed=0, min_positive_count=10, splits=("train", "tuning")
        ...     )
        ...     codes = json.loads((out_dir / "code_index.json").read_text())
        ...     is_ok = returned == out_dir and (out_dir / "train.parquet").exists()
        ...     is_ok and codes == {"0": "DIAG//COMMON"}
        True
    """
    meds_data_dir = Path(meds_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    codes = _sample_task_codes(
        meds_data_dir,
        num_tasks=num_tasks,
        seed=seed,
        horizon_days=horizon_days,
        min_history_days=min_history_days,
        min_positive_count=min_positive_count,
        min_positive_rate=min_positive_rate,
        max_positive_rate=max_positive_rate,
        splits=splits,
    )

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
