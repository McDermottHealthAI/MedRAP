"""Multi-task binary label creation from MEDS cohort data.

Prediction time is sampled uniformly at random from the per-subject window
``[first_event + min_history_days, last_event - anchor_horizon_days]``, where
``anchor_horizon_days`` is the longest of the (possibly per-task) durations below.
Subjects whose timeline is shorter than ``min_history_days + anchor_horizon_days``
are excluded.

Task codes are selected from codes present in the train split, excluding synthetic
time tokens (``TIMELINE//`` prefix) added by the MEDS-transforms pipeline and
``meds.birth_code`` (structurally always before the prediction window, see
:func:`_select_task_codes`); see that function for the two selection strategies
(``"random"``, ``"most_frequent"``). There is no positive-rate or count filtering
beyond that -- a selected code can still turn out rare or degenerate (all-positive
or all-negative) on a given split; that is a property of the selected task, not
something this module tries to correct for.

Each task's occurrence window ("how many days after prediction_time to look for
the code") defaults to a single shared ``horizon_days`` for every task
(``duration_distribution="fixed"``), but can instead be sampled independently
per task from ``[min_duration_days, max_duration_days]`` via
``duration_distribution="uniform"`` or ``"log-uniform"``; see
:func:`_sample_task_durations`. This is a port of the duration-sampling formula
from `EveryQuery <https://github.com/payalchandak/EveryQuery>`_
(``generate_tasks/sample_tasks.py``'s ``QueryDistribution.sample``) -- only that
formula, not the package itself: EveryQuery's task-label schema is a long table
(one row per ``(subject, query, duration)`` with a three-valued censored label),
fundamentally different from this module's wide per-subject ``task_0..task_{N-1}``
columns with a single shared ``prediction_time``, so nothing else from its
pipeline is directly reusable here.
"""

import json
import logging
from pathlib import Path

import meds
import numpy as np
import polars as pl

log = logging.getLogger(__name__)


def _select_task_codes(
    meds_data_dir: str | Path, num_tasks: int, seed: int, code_selection: str = "random"
) -> list[str]:
    """Select ``num_tasks`` codes from the train split, per ``code_selection``.

    Every code present in the train split is eligible, excluding synthetic
    ``TIMELINE//`` tokens added by the MEDS-transforms pipeline and
    ``meds.birth_code`` (structurally always before the prediction window,
    since it's the first event on a subject's timeline and prediction times
    are sampled after ``first_event + min_history_days`` -- see
    :func:`_sample_prediction_anchors`). No positive-rate or count filtering
    is applied beyond that and the selection strategy itself; a selected code
    may still turn out rare or degenerate (all-positive or all-negative) on a
    given split.

    Args:
        meds_data_dir: Root of a MEDS dataset (``data/train/*.parquet``).
        num_tasks: Number of codes to select.
        seed: Random seed; only used when ``code_selection="random"``.
        code_selection: ``"random"`` (uniform, without replacement) or
            ``"most_frequent"`` (the ``num_tasks`` codes with the highest
            *distinct-subject* count in the train split -- not event-row count,
            since a code measured repeatedly on a small subject subset can have
            a huge row count while still being near-zero prevalence in the
            per-subject labels this module produces; ties broken by code string).

    Returns:
        List of ``num_tasks`` code strings.

    Raises:
        ValueError: If no eligible codes are found, fewer than ``num_tasks``
            eligible codes exist, or ``code_selection`` is not recognized.

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
        ...     sorted(_select_task_codes(tmpdir, num_tasks=2, seed=0))
        ['DIAG//A', 'DIAG//B']

        ``most_frequent`` picks the codes with the most distinct subjects, not
        the most event rows: ``DIAG//A`` has 5 rows but only 1 subject
        (repeated measurements), while ``DIAG//B`` has 3 rows across 3
        subjects -- ``B`` wins despite fewer rows:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shard_dir = Path(tmpdir) / "data" / "train"
        ...     shard_dir.mkdir(parents=True)
        ...     pl.DataFrame(
        ...         {
        ...             "subject_id": [1, 1, 1, 1, 1, 2, 3, 4],
        ...             "code": ["DIAG//A"] * 5 + ["DIAG//B"] * 3,
        ...             "time": [datetime(2020, 1, i + 1) for i in range(8)],
        ...         }
        ...     ).write_parquet(shard_dir / "0.parquet")
        ...     _select_task_codes(tmpdir, num_tasks=1, seed=0, code_selection="most_frequent")
        ['DIAG//B']

        ``meds.birth_code`` is never eligible, even though every subject has
        exactly one -- it's always the first event, so it can never fall
        inside a prediction window:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shard_dir = Path(tmpdir) / "data" / "train"
        ...     shard_dir.mkdir(parents=True)
        ...     pl.DataFrame(
        ...         {
        ...             "subject_id": [1, 2],
        ...             "code": [meds.birth_code, "DIAG//A"],
        ...             "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        ...         }
        ...     ).write_parquet(shard_dir / "0.parquet")
        ...     _select_task_codes(tmpdir, num_tasks=1, seed=0)
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
        ...     _select_task_codes(tmpdir, num_tasks=5, seed=0)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: No eligible task codes found in .../data/train/.
    """
    if code_selection not in ("random", "most_frequent"):
        raise ValueError(
            f"Unrecognized code_selection={code_selection!r}; expected 'random' or 'most_frequent'."
        )

    # meds.birth_code ("MEDS_BIRTH") is excluded because it's structurally
    # degenerate for every task: it's the first event on a subject's timeline,
    # and prediction_time is always sampled after first_event + min_history_days
    # (see _sample_prediction_anchors), so it can never fall inside a prediction
    # window -- pos_rate is guaranteed 0 regardless of code_selection.
    # meds.death_code is NOT excluded: a death near the end of a timeline can
    # legitimately fall inside a window, so it's a valid (if rare) task.
    data = pl.scan_parquet(Path(meds_data_dir) / "data" / "train" / "*.parquet").filter(
        pl.col("time").is_not_null(),
        ~pl.col("code").str.starts_with("TIMELINE//"),
        pl.col("code") != meds.birth_code,
    )

    if code_selection == "most_frequent":
        # Ranked by distinct-subject count, not event-row count: a code measured
        # repeatedly on a small subset of subjects (e.g. hourly ICU labs) can have
        # a huge row count while still being near-zero prevalence in the per-subject
        # labels this module produces (one label per subject). Subject count is what
        # min_subjects_per_code filtering already uses elsewhere in this pipeline.
        counts = (
            data.group_by("code")
            .agg(pl.col("subject_id").n_unique().alias("n_subjects"))
            .sort(["n_subjects", "code"], descending=[True, False])
            .collect()
        )
        eligible = counts["code"].to_list()
    else:
        eligible = data.select("code").unique().collect()["code"].to_list()

    if not eligible:
        raise ValueError(f"No eligible task codes found in {meds_data_dir}/data/train/.")
    if len(eligible) < num_tasks:
        raise ValueError(
            f"Only {len(eligible)} eligible task codes found in {meds_data_dir}/data/train/, "
            f"fewer than the requested num_tasks={num_tasks}."
        )

    if code_selection == "most_frequent":
        return eligible[:num_tasks]

    rng = np.random.default_rng(seed)
    return rng.choice(eligible, size=num_tasks, replace=False).tolist()


def _sample_task_durations(
    num_tasks: int, min_duration_days: float, max_duration_days: float, duration_distribution: str, seed: int
) -> list[float]:
    """Sample ``num_tasks`` per-task occurrence-window durations, in days.

    Port of `EveryQuery <https://github.com/payalchandak/EveryQuery>`_'s
    ``QueryDistribution.sample`` duration draw (``generate_tasks/sample_tasks.py``) --
    only this formula, not the package (see module docstring for why).

    Args:
        num_tasks: Number of durations to draw, one per task code.
        min_duration_days: Lower bound in days (must be > 0).
        max_duration_days: Upper bound in days (must be >= ``min_duration_days``).
        duration_distribution: ``"uniform"`` or ``"log-uniform"``. Log-uniform draws
            ``exp(uniform(log(min), log(max)))``, biasing toward shorter durations
            while still covering the full range.
        seed: Random seed.

    Returns:
        List of ``num_tasks`` duration floats (no day-rounding), each in
        ``[min_duration_days, max_duration_days]``.

    Raises:
        ValueError: If ``min_duration_days <= 0``, ``max_duration_days <
            min_duration_days``, or ``duration_distribution`` is not recognized.

    Examples:
        >>> durations = _sample_task_durations(5, 1.0, 365.0, "log-uniform", seed=0)
        >>> len(durations)
        5
        >>> all(1.0 <= d <= 365.0 for d in durations)
        True

        Durations are floats, not rounded to whole days:

        >>> any(d != round(d) for d in durations)
        True

        Same seed gives identical output:

        >>> _sample_task_durations(3, 1.0, 30.0, "uniform", seed=7) == _sample_task_durations(
        ...     3, 1.0, 30.0, "uniform", seed=7
        ... )
        True

        >>> _sample_task_durations(1, 0.0, 10.0, "uniform", seed=0)
        Traceback (most recent call last):
            ...
        ValueError: min_duration_days must be > 0 (got 0.0)

        >>> _sample_task_durations(1, 10.0, 5.0, "uniform", seed=0)
        Traceback (most recent call last):
            ...
        ValueError: max_duration_days (5.0) must be >= min_duration_days (10.0)

        >>> _sample_task_durations(1, 1.0, 10.0, "bogus", seed=0)
        Traceback (most recent call last):
            ...
        ValueError: duration_distribution must be 'uniform' or 'log-uniform', got 'bogus'
    """
    if min_duration_days <= 0:
        raise ValueError(f"min_duration_days must be > 0 (got {min_duration_days})")
    if max_duration_days < min_duration_days:
        raise ValueError(
            f"max_duration_days ({max_duration_days}) must be >= min_duration_days ({min_duration_days})"
        )
    if duration_distribution not in ("uniform", "log-uniform"):
        raise ValueError(
            f"duration_distribution must be 'uniform' or 'log-uniform', got {duration_distribution!r}"
        )

    rng = np.random.default_rng(seed)
    if duration_distribution == "log-uniform":
        durations = np.exp(rng.uniform(np.log(min_duration_days), np.log(max_duration_days), size=num_tasks))
    else:
        durations = rng.uniform(min_duration_days, max_duration_days, size=num_tasks)
    return [float(d) for d in durations]


def _sample_prediction_anchors(
    df: pl.DataFrame, anchor_horizon_days: float, min_history_days: float, rng: np.random.Generator
) -> pl.DataFrame | None:
    """Sample one random prediction anchor per subject within their valid window.

    Prediction time is sampled uniformly from
    ``[first_event + min_history_days, last_event - anchor_horizon_days]``. Subjects
    whose window is empty are dropped.

    Args:
        df: Shard events with ``subject_id`` and ``time`` columns.
        anchor_horizon_days: Days of trailing room required after the prediction
            anchor. When task durations vary per task (see
            :func:`_sample_task_durations`), pass the *longest* sampled duration
            here, so the single shared anchor per subject has enough room for
            every task's window -- no task is ever dropped for a given subject
            due to insufficient trailing data.
        min_history_days: Minimum days of history before the prediction anchor.
        rng: Random generator; consumes one ``rng.random(n_subjects)`` draw.

    Returns:
        DataFrame with columns ``subject_id``, ``prediction_time``, or ``None``
        if no subject has a nonempty window.
    """
    if df.is_empty():
        return None

    min_history_us = int(min_history_days * 86_400 * 1_000_000)
    horizon_us = int(anchor_horizon_days * 86_400 * 1_000_000)

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


def _generate_labels_shard(
    shard_path: Path,
    codes: list[str],
    durations: list[float],
    min_history_days: float,
    rng: np.random.Generator,
) -> pl.DataFrame | None:
    """Build multi-task label rows for one shard with a random prediction time per subject.

    Prediction time is sampled uniformly from
    ``[first_event + min_history_days, last_event - max(durations)]``.
    Subjects whose window is empty are dropped.

    Args:
        shard_path: Path to one MEDS event shard.
        codes: Task codes, in the same order as ``durations``.
        durations: Per-task occurrence-window length in days, one per code
            (``durations[i]`` is code ``codes[i]``'s window). All equal when
            ``duration_distribution="fixed"``; independent draws otherwise --
            see :func:`_sample_task_durations`.
        min_history_days: Minimum days of history before the prediction anchor.
        rng: Random generator, forwarded to :func:`_sample_prediction_anchors`.

    Raises:
        ValueError: If ``len(codes) != len(durations)``.
    """
    if len(codes) != len(durations):
        raise ValueError(
            f"codes and durations must be the same length, got {len(codes)} and {len(durations)}"
        )

    df = pl.read_parquet(shard_path, columns=["subject_id", "time", "code"]).filter(
        pl.col("time").is_not_null()
    )
    anchor_horizon_days = max(durations)
    anchors = _sample_prediction_anchors(df, anchor_horizon_days, min_history_days, rng)
    if anchors is None:
        return None

    joined = anchors.join(df, on="subject_id", how="left").with_columns(
        ((pl.col("time") - pl.col("prediction_time")).dt.total_seconds() / 86400.0).alias("delta_days")
    )

    # Inner join against the code->duration map both restricts to task codes (like the
    # old `code.is_in(codes)` filter) and attaches each row's own occurrence window, so
    # a code with a shorter sampled duration doesn't pick up occurrences past its own
    # window just because another task's longer duration widened the anchor's trailing room.
    code_durations = pl.DataFrame({"code": codes, "duration_days": durations})
    in_window = (
        joined.join(code_durations, on="code", how="inner")
        .filter((pl.col("delta_days") > 0) & (pl.col("delta_days") <= pl.col("duration_days")))
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
    durations: list[float],
    min_history_days: float,
    seed: int,
) -> pl.DataFrame:
    """Generate multi-task binary labels for all shards of one split.

    Args:
        meds_data_dir: Root of a MEDS dataset.
        split: Split name (e.g. ``"train"``).
        codes: Task codes from :func:`_select_task_codes`.
        durations: Per-task occurrence-window length in days, one per code
            (same length and order as ``codes``). Pass ``[horizon_days] * len(codes)``
            for a single shared window, or the output of
            :func:`_sample_task_durations` for independent per-task windows.
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
        ...         tmpdir, "train", ["A", "B"], durations=[7.0, 7.0], min_history_days=1.0, seed=0
        ...     )
        ...     set(df.columns) == {"subject_id", "prediction_time", "task_0", "task_1"}
        True

        Independent per-task durations -- events chosen so the anchor window
        collapses to a single deterministic point (``first_event + min_history_days
        == last_event - max(durations)``): ``B``'s narrower 2-day window misses its
        day-5 occurrence (4 days after the day-1 anchor > 2), while ``A``'s wider
        20-day window catches its day-10 occurrence (9 days after the anchor <= 20):

        >>> from datetime import timedelta
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shard_dir = Path(tmpdir) / "data" / "train"
        ...     shard_dir.mkdir(parents=True)
        ...     base = datetime(2020, 1, 1)
        ...     pl.DataFrame(
        ...         {
        ...             "subject_id": [1, 1, 1, 1],
        ...             "code": ["A", "B", "A", "TIMELINE//DELTA//years"],
        ...             "time": [
        ...                 base,
        ...                 base + timedelta(days=5),
        ...                 base + timedelta(days=10),
        ...                 base + timedelta(days=21),
        ...             ],
        ...         }
        ...     ).write_parquet(shard_dir / "0.parquet")
        ...     df = generate_labels(
        ...         tmpdir, "train", ["A", "B"], durations=[20.0, 2.0], min_history_days=1.0, seed=0
        ...     )
        ...     df.select("task_0", "task_1").to_dicts()
        [{'task_0': 1.0, 'task_1': 0.0}]
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     generate_labels(
        ...         tmpdir, "train", ["A"], durations=[7.0], min_history_days=1.0, seed=0
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
        if (s := _generate_labels_shard(f, codes, durations, min_history_days, rng)) is not None
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
    code_selection: str = "random",
    duration_distribution: str = "fixed",
    min_duration_days: float | None = None,
    max_duration_days: float | None = None,
) -> Path:
    """Create multi-task binary labels from a MEDS cohort.

    Task codes are selected from codes present in the train split (excluding
    synthetic ``TIMELINE//`` tokens and ``meds.birth_code``) per
    ``code_selection``; see :func:`_select_task_codes`. No positive-rate or
    count filtering is applied beyond that. Prediction time is sampled
    independently per split, per subject, regardless of ``code_selection``;
    see :func:`_sample_prediction_anchors`.

    Each task's occurrence-window duration is either a single shared
    ``horizon_days`` (``duration_distribution="fixed"``, the default -- matches
    this function's behavior before per-task durations existed) or sampled
    independently per task from ``[min_duration_days, max_duration_days]``
    (``duration_distribution="uniform"`` or ``"log-uniform"``); see
    :func:`_sample_task_durations`.

    Args:
        meds_data_dir: Root of a MEDS dataset (``data/{split}/*.parquet``).
        output_dir: Directory to write output files into.
        num_tasks: Number of task codes to select.
        horizon_days: Days after prediction time to look for code occurrence.
            Used as every task's duration when ``duration_distribution="fixed"``;
            ignored otherwise.
        min_history_days: Minimum days of history before the prediction anchor.
        seed: Random seed for task selection (when ``code_selection="random"``),
            duration sampling (when ``duration_distribution != "fixed"``), and
            anchor sampling.
        splits: Splits to generate labels for.
        code_selection: ``"random"`` or ``"most_frequent"``; see
            :func:`_select_task_codes`.
        duration_distribution: ``"fixed"`` (default), ``"uniform"``, or
            ``"log-uniform"``; see :func:`_sample_task_durations`.
        min_duration_days: Lower duration bound in days. Required (and only
            used) when ``duration_distribution != "fixed"``.
        max_duration_days: Upper duration bound in days. Required (and only
            used) when ``duration_distribution != "fixed"``.

    Returns:
        ``output_dir`` as a ``Path``.

    Raises:
        ValueError: If ``duration_distribution`` is not ``"fixed"``,
            ``"uniform"``, or ``"log-uniform"``, or if it is not ``"fixed"``
            and ``min_duration_days``/``max_duration_days`` are not both set.

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

        ``duration_distribution="log-uniform"`` samples one duration per task and
        records it in ``metadata.json``:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     for split in ("train",):
        ...         shard_dir = Path(tmpdir) / "data" / split
        ...         shard_dir.mkdir(parents=True)
        ...         rows = {"subject_id": [], "code": [], "time": []}
        ...         for subject_id in range(5):
        ...             start = datetime(2020, 1, 1) + timedelta(days=subject_id)
        ...             rows["subject_id"] += [subject_id, subject_id]
        ...             rows["code"] += ["DIAG//A", "TIMELINE//DELTA//years"]
        ...             rows["time"] += [start, start + timedelta(days=100)]
        ...         pl.DataFrame(rows).write_parquet(shard_dir / "0.parquet")
        ...     out_dir = Path(tmpdir) / "tasks"
        ...     _ = generate_tasks(
        ...         tmpdir,
        ...         out_dir,
        ...         num_tasks=1,
        ...         seed=0,
        ...         splits=("train",),
        ...         duration_distribution="log-uniform",
        ...         min_duration_days=1.0,
        ...         max_duration_days=90.0,
        ...     )
        ...     meta = json.loads((out_dir / "metadata.json").read_text())
        ...     meta["duration_distribution"], len(meta["durations"])
        ('log-uniform', 1)
        >>> 1.0 <= meta["durations"][0] <= 90.0
        True

        ``min_duration_days``/``max_duration_days`` are required outside
        ``"fixed"`` mode:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     generate_tasks(
        ...         tmpdir, Path(tmpdir) / "tasks", num_tasks=1, duration_distribution="uniform"
        ...     )  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: min_duration_days and max_duration_days are required when duration_distribution='uniform'
    """
    if duration_distribution not in ("fixed", "uniform", "log-uniform"):
        raise ValueError(
            "duration_distribution must be 'fixed', 'uniform', or 'log-uniform', "
            f"got {duration_distribution!r}"
        )
    if duration_distribution != "fixed" and (min_duration_days is None or max_duration_days is None):
        raise ValueError(
            "min_duration_days and max_duration_days are required when "
            f"duration_distribution={duration_distribution!r}"
        )

    meds_data_dir = Path(meds_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    codes = _select_task_codes(meds_data_dir, num_tasks=num_tasks, seed=seed, code_selection=code_selection)

    if duration_distribution == "fixed":
        durations = [float(horizon_days)] * len(codes)
    else:
        assert min_duration_days is not None and max_duration_days is not None  # checked above
        durations = _sample_task_durations(
            len(codes), min_duration_days, max_duration_days, duration_distribution, seed
        )

    for split in splits:
        df = generate_labels(meds_data_dir, split, codes, durations, min_history_days, seed)
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
                "code_selection": code_selection,
                "duration_distribution": duration_distribution,
                "min_duration_days": min_duration_days,
                "max_duration_days": max_duration_days,
                "durations": durations,
                "codes": codes,
            },
            indent=2,
        )
    )

    return output_dir
