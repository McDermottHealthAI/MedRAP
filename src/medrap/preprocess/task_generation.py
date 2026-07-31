"""Multi-task binary label creation from MEDS cohort data.

Prediction time is anchored before ``last_event - anchor_horizon_days``, where
``anchor_horizon_days`` is the longest of the (possibly per-task) durations below,
and after ``first_event + min_history_days``. Subjects left with an empty window
are excluded. ``anchor_strategy`` selects *how* the anchor is drawn -- uniformly
over calendar time (``"uniform_lifetime"``, the default) or uniformly over the
subject's real clinical events (``"uniform_event"``, which additionally measures
``min_history_days`` from the first *clinical* event, since ``first_event`` is
birth); see :func:`_sample_prediction_anchors` for why this matters so much in
practice.

Task codes are selected from codes present in the train split, excluding synthetic
time tokens (``TIMELINE//`` prefix) added by the MEDS-transforms pipeline and
``meds.birth_code`` (structurally always before the prediction window, see
:func:`_select_task_codes`); see that function for the two selection strategies
(``"random"``, ``"most_frequent"``).

When ``duration_distribution="fixed"`` (the default), a selected code is also
required to have both classes present (at least one positive, at least one
negative) in every generated split: see :func:`_select_valid_task_codes_and_labels`.
A code that comes up degenerate is discarded and replaced with a fresh draw,
repeating until ``num_tasks`` valid codes are found or the eligible pool is
exhausted. This is only safe when every draw shares the same ``horizon_days``
(so prediction anchors -- and thus the subject population checked for
validity -- never shift between rounds); ``duration_distribution="uniform"``/
``"log-uniform"`` samples an independent duration per task and does not get
this guarantee -- a selected code can still turn out rare or degenerate there.

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


def _clinical_events[FrameT: (pl.DataFrame, pl.LazyFrame)](df: FrameT) -> FrameT:
    """Rows representing real clinical contact.

    Excludes BOTH ``meds.birth_code`` AND ``TIMELINE//*`` -- ``TIMELINE//START``
    shares the birth timestamp for every subject, so excluding birth alone
    changes nothing. Both exclusions are needed for
    ``anchor_strategy="uniform_event"`` to actually land on clinical activity
    rather than on the birth timestamp (see :func:`_sample_prediction_anchors`).

    This is the single definition of "eligible event" for the whole module:
    :func:`_select_task_codes` filters the task-code pool through it and
    :func:`_sample_prediction_anchors` filters anchor candidates through it. The
    two must agree -- a code eligible as a *task* must also be eligible as an
    *anchor*, or the label semantics shift -- so they share this one filter
    rather than repeating it.

    Args:
        df: Shard events with at least a ``code`` column, eager or lazy.

    Returns:
        ``df`` restricted to rows whose ``code`` is neither ``meds.birth_code``
        nor a ``TIMELINE//`` token, in the input row order and of the input
        frame type.

    Examples:
        Birth and ``TIMELINE//START`` sit at the *same* timestamp, so dropping
        only ``meds.birth_code`` would leave the timeline's start intact --
        both go, and only the real code survives:

        >>> from datetime import datetime
        >>> events = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 1],
        ...         "code": [meds.birth_code, "TIMELINE//START", "DIAG//A"],
        ...         "time": [datetime(1961, 1, 1), datetime(1961, 1, 1), datetime(2020, 1, 1)],
        ...     }
        ... )
        >>> _clinical_events(events).to_dicts()
        [{'subject_id': 1, 'code': 'DIAG//A', 'time': datetime.datetime(2020, 1, 1, 0, 0)}]

        A ``LazyFrame`` in gives a ``LazyFrame`` back, which is what lets
        :func:`_select_task_codes` share this filter without collecting:

        >>> _clinical_events(events.lazy()).collect()["code"].to_list()
        ['DIAG//A']
    """
    return df.filter(
        ~pl.col("code").str.starts_with("TIMELINE//"),
        pl.col("code") != meds.birth_code,
    )


def _select_task_codes(
    meds_data_dir: str | Path,
    num_tasks: int,
    seed: int,
    code_selection: str = "random",
    exclude: set[str] | None = None,
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
        exclude: Codes to drop from the eligible pool before selecting, e.g. codes
            already tried and rejected by :func:`_select_valid_task_codes_and_labels`.

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

        ``exclude`` drops codes from the eligible pool before selecting:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shard_dir = Path(tmpdir) / "data" / "train"
        ...     shard_dir.mkdir(parents=True)
        ...     pl.DataFrame(
        ...         {
        ...             "subject_id": [1, 2],
        ...             "code": ["DIAG//A", "DIAG//B"],
        ...             "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        ...         }
        ...     ).write_parquet(shard_dir / "0.parquet")
        ...     _select_task_codes(tmpdir, num_tasks=1, seed=0, exclude={"DIAG//A"})
        ['DIAG//B']

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

    # The TIMELINE// and meds.birth_code exclusions live in _clinical_events, shared
    # with anchor sampling: a code eligible as a *task* must also be eligible as an
    # *anchor*, or the label semantics shift, so these two filters must never diverge.
    # meds.birth_code ("MEDS_BIRTH") is excluded because it's structurally degenerate
    # for every task: it's the first event on a subject's timeline, and prediction_time
    # is always sampled after it (see _sample_prediction_anchors), so it can never fall
    # inside a prediction window -- pos_rate is guaranteed 0 regardless of
    # code_selection. meds.death_code is NOT excluded: a death near the end of a
    # timeline can legitimately fall inside a window, so it's a valid (if rare) task.
    data = _clinical_events(pl.scan_parquet(Path(meds_data_dir) / "data" / "train" / "*.parquet")).filter(
        pl.col("time").is_not_null()
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

    if exclude:
        eligible = [c for c in eligible if c not in exclude]

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
    df: pl.DataFrame,
    anchor_horizon_days: float,
    min_history_days: float,
    rng: np.random.Generator,
    anchor_strategy: str = "uniform_lifetime",
) -> pl.DataFrame | None:
    """Sample one random prediction anchor per subject within their valid window.

    Both strategies bound the anchor above by ``last_event - anchor_horizon_days``
    and below by at least ``first_event + min_history_days``; they differ in the
    measure they sample against and in how the lower bound is computed. Subjects
    left with an empty window are dropped.

    ``"uniform_lifetime"`` samples uniformly over *calendar time* in the half-open
    window ``[first_event + min_history_days, last_event - anchor_horizon_days)``.
    Because ``first_event`` is effectively the subject's birth, the window spans
    the subject's whole life -- decades -- while their clinical activity is
    typically confined to a window of months. On a real MIMIC-IV shard this
    yields a ~0.1% positive rate.

    ``"uniform_event"`` samples uniformly over the subject's
    :func:`_clinical_events` timestamps falling in the *closed* window
    ``[max(first_event, first_clinical_event) + min_history_days,
    last_event - anchor_horizon_days]``, and measured ~348x higher: on the full
    MIMIC-IV ``held_out`` split with the 128 most frequent codes at a 7-day
    horizon, median in-window positive rate 0.0004 -> 0.1380. (The closed/half-open
    asymmetry between the two strategies is a consequence of drawing from a
    discrete set rather than a continuous interval, and is harmless: an anchor
    exactly at ``last_event - anchor_horizon_days`` still has its full horizon.)

    Two properties of the lower bound are worth stating explicitly, because both
    are non-obvious:

    - Excluding ``meds.birth_code`` alone would be a **no-op**: ``TIMELINE//START``
      carries the identical timestamp for 100% of subjects, which is why
      :func:`_clinical_events` drops both.
    - Consequently ``first_event + min_history_days`` alone imposes *no clinical
      history requirement whatsoever* -- it is ``min_history_days`` after birth.
      ``"uniform_event"`` therefore measures ``min_history_days`` from the first
      **clinical** event as well, so an anchor always has at least that much real
      clinical history behind it. Without this the parameter is inert (measured:
      byte-identical anchors at ``min_history_days`` of 1, 30 and 365 days) and a
      sizeable share of anchors land on the subject's very first clinical contact,
      leaving nothing but ``MEDS_BIRTH`` and ``TIMELINE//`` tokens as context.

    Args:
        df: Shard events with ``subject_id``, ``time`` and (for
            ``anchor_strategy="uniform_event"``) ``code`` columns.
        anchor_horizon_days: Days of trailing room required after the prediction
            anchor. When task durations vary per task (see
            :func:`_sample_task_durations`), pass the *longest* sampled duration
            here, so the single shared anchor per subject has enough room for
            every task's window -- no task is ever dropped for a given subject
            due to insufficient trailing data.
        min_history_days: Minimum days of history before the prediction anchor.
            Measured from ``first_event`` (i.e. birth) under
            ``"uniform_lifetime"``, and from the later of ``first_event`` and the
            subject's first clinical event under ``"uniform_event"``.
        rng: Random generator. Either strategy consumes exactly one vectorised
            ``rng.random(n)`` draw, taken *after* sorting by ``subject_id``, so
            the same seed always maps the same subject to the same anchor:
            ``n`` is the number of subjects with a nonempty window under
            ``"uniform_lifetime"``, and the number of subjects with at least one
            eligible clinical event under ``"uniform_event"`` (where draw ``u``
            picks event index ``floor(u * n_events)``). The two strategies
            therefore consume different amounts of randomness from the same
            generator -- expected, since they are different samplers.
        anchor_strategy: ``"uniform_lifetime"`` (default) or ``"uniform_event"``.

    Returns:
        DataFrame with columns ``subject_id``, ``prediction_time``, or ``None``
        if no subject has an eligible anchor.

    Raises:
        ValueError: If ``anchor_strategy`` is not recognized.

    Examples:
        ``"uniform_event"`` always lands *on* one of the subject's clinical
        event timestamps. Subject 1's first clinical event is the day-10
        ``DIAG//A``, so with ``min_history_days=1`` the eligible anchors start at
        day 11 and the day-20 ``DIAG//B`` is the only one -- note the anchor is
        nowhere near the day-0 birth that ``"uniform_lifetime"`` anchors around:

        >>> from datetime import datetime, timedelta
        >>> base = datetime(2020, 1, 1)
        >>> def subject(sid, offset):
        ...     "Birth + TIMELINE// bookends around clinical events at +10 and +20 days."
        ...     return {
        ...         "subject_id": [sid] * 5,
        ...         "code": [
        ...             meds.birth_code,
        ...             "TIMELINE//START",
        ...             "DIAG//A",
        ...             "DIAG//B",
        ...             "TIMELINE//DELTA//years",
        ...         ],
        ...         "time": [
        ...             base + timedelta(days=offset + d) for d in (0, 0, 10, 20, 40)
        ...         ],
        ...     }
        >>> events = pl.DataFrame(subject(1, 0))
        >>> _sample_prediction_anchors(
        ...     events, 7.0, 1.0, np.random.default_rng(0), anchor_strategy="uniform_event"
        ... ).to_dicts()
        [{'subject_id': 1, 'prediction_time': datetime.datetime(2020, 1, 21, 0, 0)}]

        Each subject's anchor must come from *that subject's own* events. These
        three subjects are offset by 1000 days each, so their event timestamps are
        disjoint and any cross-subject mix-up would change the printed output.
        The rows are deliberately shuffled and the ``subject_id`` values are
        non-contiguous, since the pick indexing relies on an explicit sort rather
        than on input order:

        >>> rows = pl.concat([pl.DataFrame(subject(sid, off))
        ...                   for sid, off in [(70, 2000), (5, 0), (42, 1000)]])
        >>> shuffled = rows.sample(fraction=1.0, shuffle=True, seed=7)
        >>> _sample_prediction_anchors(
        ...     shuffled, 7.0, 1.0, np.random.default_rng(0), anchor_strategy="uniform_event"
        ... ).to_dicts()
        [{'subject_id': 5, 'prediction_time': datetime.datetime(2020, 1, 21, 0, 0)},
         {'subject_id': 42, 'prediction_time': datetime.datetime(2022, 10, 17, 0, 0)},
         {'subject_id': 70, 'prediction_time': datetime.datetime(2025, 7, 13, 0, 0)}]

        ``min_history_days`` is measured from the first *clinical* event, not from
        birth -- so raising it past the span of a subject's clinical activity drops
        them, rather than silently changing nothing:

        >>> _sample_prediction_anchors(
        ...     events, 7.0, 15.0, np.random.default_rng(0), anchor_strategy="uniform_event"
        ... ) is None
        True

        A subject whose only clinical events fall outside the window is dropped
        entirely, rather than falling back to a calendar-time anchor:

        >>> only_edges = events.filter(~pl.col("code").str.starts_with("DIAG//"))
        >>> _sample_prediction_anchors(
        ...     only_edges, 7.0, 1.0, np.random.default_rng(0), anchor_strategy="uniform_event"
        ... ) is None
        True

        >>> _sample_prediction_anchors(events, 7.0, 1.0, np.random.default_rng(0), anchor_strategy="x")
        Traceback (most recent call last):
            ...
        ValueError: anchor_strategy must be 'uniform_lifetime' or 'uniform_event', got 'x'
    """
    if anchor_strategy not in ("uniform_lifetime", "uniform_event"):
        raise ValueError(
            f"anchor_strategy must be 'uniform_lifetime' or 'uniform_event', got {anchor_strategy!r}"
        )

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
        # group_by's output row order is not guaranteed stable across calls (polars can use a
        # multi-threaded hash aggregation), so without this sort the same rng.random(len(bounds))
        # draw below can land on a *different* subject each call even with an identical seed --
        # silently reshuffling everyone's prediction_time. Explicit sort makes row i always mean
        # the same subject_id, so the same seed always produces the same anchors.
        .sort("subject_id")
    )

    if bounds.is_empty():
        return None

    if anchor_strategy == "uniform_lifetime":
        earliest = bounds["earliest_us"].to_numpy()
        window = (bounds["latest_us"] - bounds["earliest_us"]).to_numpy()
        offsets = (rng.random(len(bounds)) * window).astype(np.int64)
        subject_ids = bounds["subject_id"]
        prediction_us = earliest + offsets
    else:
        # Candidate anchors are the subject's real clinical events (birth and
        # TIMELINE// tokens dropped -- see _clinical_events) that fall inside the
        # window. The left edge is pushed out to min_history_days after the first
        # *clinical* event: `earliest_us` is min_history_days after birth, which on
        # MEDS-transforms output is no history requirement at all, and without this
        # an anchor can land on the subject's first clinical contact with nothing
        # but MEDS_BIRTH and TIMELINE// tokens preceding it. Taking the max of the
        # two keeps the lifetime bound as a floor rather than replacing it.
        clinical = _clinical_events(df).select(
            "subject_id", pl.col("time").dt.epoch(time_unit="us").alias("event_us")
        )
        clinical_earliest = clinical.group_by("subject_id").agg(
            (pl.col("event_us").min() + min_history_us).alias("clinical_earliest_us")
        )
        # Sorting by (subject_id, event_us) makes each subject's candidates one
        # contiguous, deterministically ordered block, so the per-subject counts
        # below line up with offsets into the flat event array.
        candidates = (
            clinical.join(clinical_earliest, on="subject_id", how="inner")
            .join(bounds.select("subject_id", "earliest_us", "latest_us"), on="subject_id", how="inner")
            .filter(
                pl.col("event_us") >= pl.max_horizontal("earliest_us", "clinical_earliest_us"),
                pl.col("event_us") <= pl.col("latest_us"),
            )
            .sort(["subject_id", "event_us"])
        )
        if candidates.is_empty():
            return None

        # Same reason for the explicit .sort("subject_id") as above: group_by's row
        # order is not stable, so without it row i of the rng draw would not always
        # mean the same subject.
        counts = candidates.group_by("subject_id").agg(pl.len().alias("n_events")).sort("subject_id")
        n_events = counts["n_events"].to_numpy().astype(np.int64)
        block_starts = np.cumsum(n_events) - n_events  # exclusive prefix sum
        picks = block_starts + (rng.random(len(counts)) * n_events).astype(np.int64)
        subject_ids = counts["subject_id"]
        prediction_us = candidates["event_us"].to_numpy()[picks]

    return (
        pl.DataFrame({"subject_id": subject_ids, "prediction_us": prediction_us})
        .with_columns(pl.from_epoch("prediction_us", time_unit="us").alias("prediction_time"))
        .select(["subject_id", "prediction_time"])
    )


def _generate_labels_shard(
    shard_path: Path,
    codes: list[str],
    durations: list[float],
    min_history_days: float,
    rng: np.random.Generator,
    anchor_strategy: str = "uniform_lifetime",
) -> pl.DataFrame | None:
    """Build multi-task label rows for one shard with a random prediction time per subject.

    Prediction time is anchored inside
    ``[first_event + min_history_days, last_event - max(durations)]`` per
    ``anchor_strategy``. Subjects whose window is empty are dropped.

    Args:
        shard_path: Path to one MEDS event shard.
        codes: Task codes, in the same order as ``durations``.
        durations: Per-task occurrence-window length in days, one per code
            (``durations[i]`` is code ``codes[i]``'s window). All equal when
            ``duration_distribution="fixed"``; independent draws otherwise --
            see :func:`_sample_task_durations`.
        min_history_days: Minimum days of history before the prediction anchor.
        rng: Random generator, forwarded to :func:`_sample_prediction_anchors`.
        anchor_strategy: ``"uniform_lifetime"`` (default) or ``"uniform_event"``;
            forwarded to :func:`_sample_prediction_anchors`.

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
    anchors = _sample_prediction_anchors(df, anchor_horizon_days, min_history_days, rng, anchor_strategy)
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
    anchor_strategy: str = "uniform_lifetime",
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
        anchor_strategy: ``"uniform_lifetime"`` (default, uniform over calendar
            time) or ``"uniform_event"`` (uniform over the subject's clinical
            events); see :func:`_sample_prediction_anchors`.

    Returns:
        DataFrame with columns ``subject_id``, ``prediction_time``, ``task_0``, …

    Raises:
        ValueError: If ``anchor_strategy`` is not ``"uniform_lifetime"`` or
            ``"uniform_event"``.
        FileNotFoundError: If ``split`` has no shard directory.

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

        Same seed gives identical anchors across repeated calls, even with many
        subjects -- ``_sample_prediction_anchors``'s internal ``group_by`` sorts by
        ``subject_id`` before consuming the RNG draw, so row order can't reshuffle
        which subject gets which offset (this matters for
        :func:`_select_valid_task_codes_and_labels`, which calls :func:`generate_labels`
        multiple times with the same seed and requires consistent anchors each time):

        >>> import numpy as np
        >>> from datetime import timedelta
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     shard_dir = Path(tmpdir) / "data" / "train"
        ...     shard_dir.mkdir(parents=True)
        ...     rng = np.random.default_rng(0)
        ...     subject_ids = list(range(200))
        ...     rows = {"subject_id": [], "code": [], "time": []}
        ...     for sid in subject_ids:
        ...         start = datetime(2020, 1, 1) + timedelta(days=int(rng.integers(0, 50)))
        ...         rows["subject_id"] += [sid, sid, sid]
        ...         rows["code"] += ["A", "A", "TIMELINE//DELTA//years"]
        ...         rows["time"] += [start, start + timedelta(days=10), start + timedelta(days=20)]
        ...     pl.DataFrame(rows).write_parquet(shard_dir / "0.parquet")
        ...     a = generate_labels(tmpdir, "train", ["A"], [7.0], min_history_days=1.0, seed=42)
        ...     b = generate_labels(tmpdir, "train", ["A"], [7.0], min_history_days=1.0, seed=42)
        ...     event_a = generate_labels(
        ...         tmpdir, "train", ["A"], [7.0], 1.0, seed=42, anchor_strategy="uniform_event"
        ...     )
        ...     event_b = generate_labels(
        ...         tmpdir, "train", ["A"], [7.0], 1.0, seed=42, anchor_strategy="uniform_event"
        ...     )
        ...     a.sort("subject_id")["prediction_time"].equals(b.sort("subject_id")["prediction_time"])
        True

        ``anchor_strategy="uniform_event"`` is idempotent under the same seed for
        the same reason:

        >>> event_a.sort("subject_id")["prediction_time"].equals(
        ...     event_b.sort("subject_id")["prediction_time"]
        ... )
        True

        …and every anchor is one of **that subject's own** ``"A"`` timestamps,
        not just some subject's. The per-subject form matters: the ``uniform_event``
        pick is an index into one flat array of every subject's events, so a
        misaligned block offset would still return a real ``"A"`` timestamp and
        pass a pooled membership check while handing each subject somebody else's
        anchor (which is exactly the class of bug the ``sort("subject_id")`` above
        exists to prevent):

        >>> own = {sid: set() for sid in subject_ids}
        >>> for sid, code, t in zip(rows["subject_id"], rows["code"], rows["time"]):
        ...     if code == "A":
        ...         own[sid].add(t)
        >>> all(r["prediction_time"] in own[r["subject_id"]] for r in event_a.to_dicts())
        True

        An unrecognized ``anchor_strategy`` is rejected here rather than deep inside
        anchor sampling, which is never reached when a split has no shards:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     (Path(tmpdir) / "data" / "train").mkdir(parents=True)
        ...     generate_labels(tmpdir, "train", ["A"], [7.0], 1.0, 0, anchor_strategy="bogus")
        Traceback (most recent call last):
            ...
        ValueError: anchor_strategy must be 'uniform_lifetime' or 'uniform_event', got 'bogus'

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
    # Validated here, not only inside _sample_prediction_anchors: that is never
    # reached for a split with no shards, so a typo'd strategy would otherwise
    # silently return an empty frame instead of raising.
    if anchor_strategy not in ("uniform_lifetime", "uniform_event"):
        raise ValueError(
            f"anchor_strategy must be 'uniform_lifetime' or 'uniform_event', got {anchor_strategy!r}"
        )

    shard_dir = Path(meds_data_dir) / "data" / split
    if not shard_dir.exists():
        raise FileNotFoundError(f"No shard directory: {shard_dir}")

    rng = np.random.default_rng(seed)
    shards = [
        s
        for f in sorted(shard_dir.glob("*.parquet"))
        if (s := _generate_labels_shard(f, codes, durations, min_history_days, rng, anchor_strategy))
        is not None
    ]
    if not shards:
        return pl.DataFrame()
    return pl.concat(shards)


def _select_valid_task_codes_and_labels(
    meds_data_dir: str | Path,
    num_tasks: int,
    horizon_days: float,
    min_history_days: float,
    seed: int,
    splits: tuple[str, ...],
    code_selection: str,
    anchor_strategy: str,
) -> tuple[list[str], dict[str, pl.DataFrame]]:
    """Select ``num_tasks`` codes whose labels have both classes in every split.

    Draws candidate codes via :func:`_select_task_codes`, generates their labels
    for every split in ``splits``, and keeps only codes where each split has at
    least one positive and one negative label. Rejected codes are excluded and
    replaced with fresh draws (same ``code_selection`` strategy), repeating
    until ``num_tasks`` valid codes are found or the eligible pool is exhausted
    (in which case :func:`_select_task_codes` raises ``ValueError``).

    Only safe for a single shared ``horizon_days`` (i.e. ``duration_distribution
    ="fixed"`` in :func:`generate_tasks`): every round -- and the final label
    generation -- uses the same ``horizon_days``, ``seed`` and
    ``anchor_strategy``, so prediction anchors (and thus the subject population
    checked for validity) are identical across rounds; a code accepted in an
    earlier round stays valid when the final combined label set is generated.
    ``anchor_strategy`` must be threaded in for exactly this reason: validating
    candidates against anchors other than the ones the final labels use would
    silently accept codes that are degenerate under the real anchors.

    Args:
        meds_data_dir: Root of a MEDS dataset.
        num_tasks: Number of valid codes to find.
        horizon_days: Shared occurrence-window length in days for every task.
        min_history_days: Minimum days of history before the prediction anchor.
        seed: Random seed, forwarded to code selection and label generation.
        splits: Splits a code's labels must be valid in.
        code_selection: ``"random"`` or ``"most_frequent"``.
        anchor_strategy: ``"uniform_lifetime"`` or ``"uniform_event"``; forwarded
            unchanged to every :func:`generate_labels` call, candidate rounds and
            final generation alike.

    Returns:
        ``(codes, split_frames)`` where ``codes`` has ``num_tasks`` entries and
        ``split_frames`` maps each split in ``splits`` to its label
        ``DataFrame`` (columns ``subject_id``, ``prediction_time``,
        ``task_0``, ...), built from exactly those codes.

    Examples:
        ``"BAD"`` occurs for every subject (degenerate, all-positive) while
        ``"GOOD"`` splits 2-2 -- regardless of which code the RNG draws
        first, only ``"GOOD"`` survives:

        >>> import tempfile
        >>> from datetime import datetime, timedelta
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     base = datetime(2020, 1, 1)
        ...     records = []
        ...     for subject_id in range(1, 5):
        ...         records.append({"subject_id": subject_id, "code": "TIMELINE//DELTA//years", "time": base})
        ...         records.append(
        ...             {"subject_id": subject_id, "code": "BAD", "time": base + timedelta(days=3)}
        ...         )
        ...         if subject_id <= 2:
        ...             records.append(
        ...                 {"subject_id": subject_id, "code": "GOOD", "time": base + timedelta(days=3)}
        ...             )
        ...         records.append(
        ...             {
        ...                 "subject_id": subject_id,
        ...                 "code": "TIMELINE//DELTA//years",
        ...                 "time": base + timedelta(days=8),
        ...             }
        ...         )
        ...     for split in ("train", "tuning", "held_out"):
        ...         shard_dir = Path(tmpdir) / "data" / split
        ...         shard_dir.mkdir(parents=True)
        ...         pl.DataFrame(records).write_parquet(shard_dir / "0.parquet")
        ...     codes, frames = _select_valid_task_codes_and_labels(
        ...         tmpdir,
        ...         num_tasks=1,
        ...         horizon_days=7.0,
        ...         min_history_days=1.0,
        ...         seed=0,
        ...         splits=("train", "tuning", "held_out"),
        ...         code_selection="random",
        ...         anchor_strategy="uniform_lifetime",
        ...     )
        ...     codes
        ['GOOD']
        >>> sorted(frames)
        ['held_out', 'train', 'tuning']
        >>> sorted(frames["train"]["task_0"].to_list())
        [0.0, 0.0, 1.0, 1.0]
    """
    exclude: set[str] = set()
    valid_codes: list[str] = []
    while len(valid_codes) < num_tasks:
        candidates = _select_task_codes(
            meds_data_dir,
            num_tasks=num_tasks - len(valid_codes),
            seed=seed,
            code_selection=code_selection,
            exclude=exclude,
        )
        durations = [float(horizon_days)] * len(candidates)
        frames = {
            split: generate_labels(
                meds_data_dir, split, candidates, durations, min_history_days, seed, anchor_strategy
            )
            for split in splits
        }
        for i, code in enumerate(candidates):
            col = f"task_{i}"
            is_valid = all(
                not frames[split].is_empty() and 0 < frames[split][col].sum() < frames[split].height
                for split in splits
            )
            if is_valid:
                valid_codes.append(code)
            exclude.add(code)

    durations = [float(horizon_days)] * len(valid_codes)
    final_frames = {
        split: generate_labels(
            meds_data_dir, split, valid_codes, durations, min_history_days, seed, anchor_strategy
        )
        for split in splits
    }
    return valid_codes, final_frames


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
    anchor_strategy: str = "uniform_lifetime",
    duration_distribution: str = "fixed",
    min_duration_days: float | None = None,
    max_duration_days: float | None = None,
) -> Path:
    """Create multi-task binary labels from a MEDS cohort.

    Task codes are selected from codes present in the train split (excluding
    synthetic ``TIMELINE//`` tokens and ``meds.birth_code``) per
    ``code_selection``; see :func:`_select_task_codes`. Prediction time is
    sampled independently per split, per subject, regardless of
    ``code_selection``, per ``anchor_strategy``; see
    :func:`_sample_prediction_anchors`. ``anchor_strategy="uniform_lifetime"``
    (the default, preserved byte-for-byte for reproducibility of existing label
    sets) spreads anchors over the subject's whole lifetime -- which starts at
    birth -- and produces ~0.1% positive labels on real data;
    ``"uniform_event"`` anchors on actual clinical events instead, and is the
    only strategy under which ``min_history_days`` requires any *clinical*
    history (under ``"uniform_lifetime"`` it is measured from birth, so on
    MEDS-transforms output it is effectively inert).

    When ``duration_distribution="fixed"`` (the default), a selected code is
    also required to have both classes present in every generated split --
    degenerate draws are discarded and replaced until ``num_tasks`` valid
    codes are found; see :func:`_select_valid_task_codes_and_labels`. This
    guarantee does not extend to ``duration_distribution="uniform"``/
    ``"log-uniform"`` (independent per-task durations mean anchors shift
    between candidate draws, so validity can't be checked consistently) --
    a selected code can still turn out rare or degenerate there.

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
        anchor_strategy: ``"uniform_lifetime"`` (default) or ``"uniform_event"``;
            see :func:`_sample_prediction_anchors`. Recorded in
            ``metadata.json``.
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
            ``"uniform"``, or ``"log-uniform"``, if it is not ``"fixed"``
            and ``min_duration_days``/``max_duration_days`` are not both set,
            or if ``anchor_strategy`` is not ``"uniform_lifetime"`` or
            ``"uniform_event"``.

    Examples:
        >>> import tempfile
        >>> from datetime import datetime, timedelta
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     base = datetime(2020, 1, 1)
        ...     for split in ("train", "tuning"):
        ...         shard_dir = Path(tmpdir) / "data" / split
        ...         shard_dir.mkdir(parents=True)
        ...         records = []
        ...         for subject_id in range(5):
        ...             records.append(
        ...                 {"subject_id": subject_id, "code": "TIMELINE//DELTA//years", "time": base}
        ...             )
        ...             if subject_id < 2:  # "DIAG//A" occurs for 2/5 subjects: mixed, not degenerate
        ...                 records.append(
        ...                     {
        ...                         "subject_id": subject_id,
        ...                         "code": "DIAG//A",
        ...                         "time": base + timedelta(days=3),
        ...                     }
        ...                 )
        ...             records.append(
        ...                 {
        ...                     "subject_id": subject_id,
        ...                     "code": "TIMELINE//DELTA//years",
        ...                     "time": base + timedelta(days=8),
        ...                 }
        ...             )
        ...         pl.DataFrame(records).write_parquet(shard_dir / "0.parquet")
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

        ``anchor_strategy`` is recorded in ``metadata.json``. Each of these 5
        subjects is born on day 0 and has clinical events on days 3000 and 3005,
        so under ``"uniform_event"`` with ``min_history_days=1`` the eligible
        anchors start on day 3001 (a day after the first *clinical* event, not a
        day after birth) and end on day 3005 (``last_event`` 3012 minus the 7-day
        horizon) -- leaving day 3005 as the only candidate, and therefore every
        subject's anchor. ``VISIT`` never recurs in the following 7 days, so it is
        all-negative and gets rejected and replaced by ``DIAG//A``, which 3 of the
        5 subjects have on day 3010:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     base = datetime(2020, 1, 1)
        ...     shard_dir = Path(tmpdir) / "data" / "train"
        ...     shard_dir.mkdir(parents=True)
        ...     records = []
        ...     for subject_id in range(5):
        ...         for code, day in [(meds.birth_code, 0), ("TIMELINE//START", 0)]:
        ...             records.append({"subject_id": subject_id, "code": code, "time": base})
        ...         for day in (3000, 3005):
        ...             records.append(
        ...                 {
        ...                     "subject_id": subject_id,
        ...                     "code": "VISIT",
        ...                     "time": base + timedelta(days=day),
        ...                 }
        ...             )
        ...         if subject_id < 3:  # "DIAG//A" occurs for 3/5 subjects: mixed, not degenerate
        ...             records.append(
        ...                 {
        ...                     "subject_id": subject_id,
        ...                     "code": "DIAG//A",
        ...                     "time": base + timedelta(days=3010),
        ...                 }
        ...             )
        ...         records.append(
        ...             {
        ...                 "subject_id": subject_id,
        ...                 "code": "TIMELINE//DELTA//years",
        ...                 "time": base + timedelta(days=3012),
        ...             }
        ...         )
        ...     pl.DataFrame(records).write_parquet(shard_dir / "0.parquet")
        ...     out_dir = Path(tmpdir) / "tasks"
        ...     _ = generate_tasks(
        ...         tmpdir,
        ...         out_dir,
        ...         num_tasks=1,
        ...         seed=0,
        ...         splits=("train",),
        ...         code_selection="most_frequent",
        ...         anchor_strategy="uniform_event",
        ...     )
        ...     meta = json.loads((out_dir / "metadata.json").read_text())
        ...     labels = pl.read_parquet(out_dir / "train.parquet").sort("subject_id")
        ...     (meta["anchor_strategy"], meta["codes"])
        ...     labels["prediction_time"].unique().to_list()
        ...     labels["task_0"].to_list()
        ('uniform_event', ['DIAG//A'])
        [datetime.datetime(2028, 3, 15, 0, 0)]
        [1.0, 1.0, 1.0, 0.0, 0.0]

        ``anchor_strategy`` is validated up front rather than deep inside label
        generation:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     generate_tasks(tmpdir, Path(tmpdir) / "tasks", num_tasks=1, anchor_strategy="bogus")
        Traceback (most recent call last):
            ...
        ValueError: anchor_strategy must be 'uniform_lifetime' or 'uniform_event', got 'bogus'
    """
    if anchor_strategy not in ("uniform_lifetime", "uniform_event"):
        raise ValueError(
            f"anchor_strategy must be 'uniform_lifetime' or 'uniform_event', got {anchor_strategy!r}"
        )
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

    if duration_distribution == "fixed":
        codes, split_frames = _select_valid_task_codes_and_labels(
            meds_data_dir,
            num_tasks,
            horizon_days,
            min_history_days,
            seed,
            splits,
            code_selection,
            anchor_strategy,
        )
        durations = [float(horizon_days)] * len(codes)
    else:
        codes = _select_task_codes(
            meds_data_dir, num_tasks=num_tasks, seed=seed, code_selection=code_selection
        )
        assert min_duration_days is not None and max_duration_days is not None  # checked above
        durations = _sample_task_durations(
            len(codes), min_duration_days, max_duration_days, duration_distribution, seed
        )
        split_frames = {
            split: generate_labels(
                meds_data_dir, split, codes, durations, min_history_days, seed, anchor_strategy
            )
            for split in splits
        }

    for split in splits:
        df = split_frames[split]
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
                "anchor_strategy": anchor_strategy,
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
