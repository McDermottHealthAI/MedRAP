"""Compute multi-task binary labels from MEDS cohort data.

For every patient the prediction anchor is sampled uniformly at random from the
window ``[first_event + min_history_days, last_event - horizon_days]``.  This
guarantees the model sees a non-trivial history before the anchor and that a full
label horizon exists after it.  Patients whose total timeline is shorter than
``min_history_days + horizon_days`` are excluded.

Codes are selected by a two-pass procedure that avoids trivially easy tasks:

  Pass 1 (train split only)
    - Count distinct patients and numeric-value fraction per code.
    - Drop *measurement* codes (those with ``numeric_value`` on more than
      ``--measurement_threshold`` of rows, e.g. vitals, lab results).
    - Rank remaining candidates by distinct-patient count (not raw row count)
      and keep the top ``--num_candidates``.

  Pass 2a (train split)
    - Generate binary labels for the candidate codes.
    - Compute positive rate = fraction of patients where the code occurred
      within ``--horizon_days``.
    - Keep only codes whose positive rate is in
      [``--min_positive_rate``, ``--max_positive_rate``].
    - Take the top ``--num_tasks`` by patient count.

  Pass 2b (tuning / held_out)
    - Generate labels for the final selected codes only.

Outputs
-------
{output_dir}/{split}.parquet
    Schema: subject_id | prediction_time | task_0 | ... | task_{N-1}
    task_i is 1.0 if the code occurred within horizon_days, else 0.0.

{output_dir}/code_index.json
    Maps task index (str) -> MEDS code string.

{output_dir}/metadata.json
    Records parameters and per-code positive rates.

Usage
-----
python scripts/prepare_multi_task_labels.py \\
    --meds_cohort_dir   /path/to/MEDS_cohort \\
    --output_dir        /path/to/mt_labels \\
    --num_tasks         25 \\
    --horizon_days      7 \\
    --min_history_days  1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SPLITS = ("train", "tuning", "held_out")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shard_files(cohort_dir: Path, split: str) -> list[Path]:
    shard_dir = cohort_dir / "data" / split
    if not shard_dir.exists():
        raise FileNotFoundError(f"No shard directory: {shard_dir}")
    files = sorted(shard_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {shard_dir}")
    return files


# ---------------------------------------------------------------------------
# Pass 1: code statistics
# ---------------------------------------------------------------------------


def _compute_code_stats(cohort_dir: Path) -> pl.DataFrame:
    """Return per-code (distinct_patients, value_frac) from the train split."""
    log.info("Pass 1: computing code statistics on train split ...")
    files = _shard_files(cohort_dir, "train")
    shards: list[pl.DataFrame] = []
    for f in files:
        df = pl.read_parquet(f, columns=["subject_id", "time", "code", "numeric_value"]).filter(
            pl.col("time").is_not_null() & ~pl.col("code").str.starts_with("MEDS_BIRTH")
        )
        if df.is_empty():
            continue
        shards.append(
            df.group_by("code").agg(
                pl.len().alias("total_rows"),
                pl.col("numeric_value").is_not_null().sum().alias("rows_with_value"),
                pl.col("subject_id").n_unique().alias("distinct_patients"),
            )
        )
    if not shards:
        raise RuntimeError("No usable events found in train split.")
    stats = (
        pl.concat(shards)
        .group_by("code")
        .agg(
            pl.col("total_rows").sum(),
            pl.col("rows_with_value").sum(),
            pl.col("distinct_patients").sum(),
        )
        .with_columns((pl.col("rows_with_value") / pl.col("total_rows")).alias("value_frac"))
        .sort("distinct_patients", descending=True)
    )
    log.info("Found %d distinct codes in train split.", len(stats))
    return stats


def _select_candidates(
    stats: pl.DataFrame,
    measurement_threshold: float,
    min_patients: int,
    num_candidates: int,
) -> list[str]:
    """Filter out measurement codes; return top candidates by patient count."""
    df = stats.filter(
        (pl.col("value_frac") <= measurement_threshold) & (pl.col("distinct_patients") >= min_patients)
    ).head(num_candidates)
    codes = df["code"].to_list()
    log.info(
        "Candidates after measurement filter: %d  (threshold=%.2f, min_patients=%d)",
        len(codes),
        measurement_threshold,
        min_patients,
    )
    if codes:
        log.info("  Top-3: %s", codes[:3])
    return codes


# ---------------------------------------------------------------------------
# Pass 2: label generation
# ---------------------------------------------------------------------------


def _process_shard(
    f: Path,
    codes: list[str],
    horizon_days: float,
    min_history_days: float,
    rng: np.random.Generator,
) -> pl.DataFrame | None:
    """Build binary label rows for one shard.

    Patients whose timeline is shorter than min_history_days + horizon_days are
    excluded.  For the rest, prediction_time is sampled uniformly from
    [first_event + min_history_days, last_event - horizon_days].

    Returns None if no usable events survive filtering.
    """
    df = pl.read_parquet(f, columns=["subject_id", "time", "code"]).filter(pl.col("time").is_not_null())
    if df.is_empty():
        return None

    clinical = df.filter(~pl.col("code").str.starts_with("MEDS_BIRTH"))
    if clinical.is_empty():
        return None

    min_history_us = int(min_history_days * 86_400 * 1_000_000)
    horizon_us = int(horizon_days * 86_400 * 1_000_000)

    bounds = (
        clinical.group_by("subject_id")
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
            (pl.col("delta_days") > 0) & (pl.col("delta_days") <= horizon_days) & pl.col("code").is_in(codes)
        )
        .group_by(["subject_id", "prediction_time", "code"])
        .agg(pl.len().alias("n"))
        .with_columns(pl.lit(1.0).alias("occurred"))
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

    task_cols = [f"task_{i}" for i in range(len(codes))]
    return result.select(["subject_id", "prediction_time", *task_cols])


def _generate_labels(
    cohort_dir: Path,
    split: str,
    codes: list[str],
    horizon_days: float,
    min_history_days: float,
    seed: int,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    shards = [
        s
        for f in _shard_files(cohort_dir, split)
        if (s := _process_shard(f, codes, horizon_days, min_history_days, rng)) is not None
    ]
    if not shards:
        return pl.DataFrame()
    result = pl.concat(shards)
    log.info("Split %s: %d patients.", split, len(result))
    return result


# ---------------------------------------------------------------------------
# Positive-rate filtering
# ---------------------------------------------------------------------------


def _filter_by_positive_rate(
    train_labels: pl.DataFrame,
    candidates: list[str],
    stats: pl.DataFrame,
    min_rate: float,
    max_rate: float,
    num_tasks: int,
    random_tasks: bool = False,
    seed: int = 42,
) -> list[str]:
    """Keep codes whose train positive rate is in [min_rate, max_rate].

    If random_tasks is True, sample num_tasks uniformly at random from the
    passing codes instead of taking the top-N by patient count.
    """
    n = len(train_labels)
    if n == 0:
        return candidates[:num_tasks]

    positive_rates: dict[str, float] = {}
    for i, code in enumerate(candidates):
        col = f"task_{i}"
        if col in train_labels.columns:
            positive_rates[code] = float(train_labels[col].mean())
        else:
            positive_rates[code] = 0.0

    passing = [c for c in candidates if min_rate <= positive_rates[c] <= max_rate]
    if len(passing) < num_tasks:
        log.warning(
            "Only %d codes pass positive-rate filter [%.2f, %.2f]; requested %d. "
            "Consider relaxing --min_positive_rate / --max_positive_rate.",
            len(passing),
            min_rate,
            max_rate,
            num_tasks,
        )

    if random_tasks:
        rng = np.random.default_rng(seed)
        rng.shuffle(passing)
    else:
        # Sort by patient count (preserve original ranking)
        order = {c: i for i, c in enumerate(stats["code"].to_list())}
        passing.sort(key=lambda c: order.get(c, 999_999))
    selected = passing[:num_tasks]

    if selected:
        rates = [positive_rates[c] for c in selected]
        log.info(
            "Selected %d codes  positive_rate: min=%.3f  max=%.3f  mean=%.3f",
            len(selected),
            min(rates),
            max(rates),
            sum(rates) / len(rates),
        )
    for c in selected:
        log.info("  %-60s  positive_rate=%.3f", c, positive_rates[c])
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare multi-task binary labels from MEDS cohort.")
    parser.add_argument("--meds_cohort_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--num_tasks", type=int, default=25)
    parser.add_argument(
        "--horizon_days",
        type=float,
        default=7.0,
        help="Days after prediction_time to look for code occurrence. Default: 7.",
    )
    parser.add_argument(
        "--min_history_days",
        type=float,
        default=1.0,
        help="Minimum days of history required before the prediction anchor. "
        "Patients with total timeline < min_history_days + horizon_days are excluded. Default: 1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible anchor sampling and (if --random_tasks) task selection. Default: 42.",
    )
    parser.add_argument(
        "--random_tasks",
        action="store_true",
        help="Sample num_tasks randomly from passing codes instead of taking the top-N by patient count.",
    )
    parser.add_argument(
        "--measurement_threshold",
        type=float,
        default=0.3,
        help="Exclude codes where >X fraction of rows carry a numeric value. Default: 0.3.",
    )
    parser.add_argument(
        "--min_patients",
        type=int,
        default=50,
        help="Minimum distinct training patients a code must appear in.",
    )
    parser.add_argument(
        "--min_positive_rate",
        type=float,
        default=0.05,
        help="Minimum fraction of patients with code in horizon window. Default: 0.05.",
    )
    parser.add_argument(
        "--max_positive_rate",
        type=float,
        default=0.90,
        help="Maximum fraction of patients with code in horizon window. Default: 0.90.",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=500,
        help="Candidate codes carried through to positive-rate filtering.",
    )
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: code statistics → candidates
    stats = _compute_code_stats(args.meds_cohort_dir)
    candidates = _select_candidates(
        stats,
        measurement_threshold=args.measurement_threshold,
        min_patients=args.min_patients,
        num_candidates=args.num_candidates,
    )
    if not candidates:
        raise RuntimeError(
            "No candidate codes survived measurement filtering. Try raising --measurement_threshold."
        )

    # Pass 2a: train labels for candidates → positive-rate filtering
    log.info("Pass 2a: generating candidate labels for train split ...")
    train_labels_raw = _generate_labels(
        args.meds_cohort_dir, "train", candidates, args.horizon_days, args.min_history_days, args.seed
    )
    if train_labels_raw.is_empty():
        raise RuntimeError("No train labels generated. Check --meds_cohort_dir.")

    final_codes = _filter_by_positive_rate(
        train_labels_raw,
        candidates,
        stats,
        min_rate=args.min_positive_rate,
        max_rate=args.max_positive_rate,
        num_tasks=args.num_tasks,
        random_tasks=args.random_tasks,
        seed=args.seed,
    )
    if not final_codes:
        raise RuntimeError(
            "No codes passed positive-rate filtering. Adjust --min_positive_rate / --max_positive_rate."
        )

    # Rename columns from candidate indices to final indices and save train split
    old_to_new = {f"task_{candidates.index(c)}": f"task_{i}" for i, c in enumerate(final_codes)}
    final_cols = [f"task_{i}" for i in range(len(final_codes))]
    train_labels = (
        train_labels_raw.select(
            ["subject_id", "prediction_time"] + [f"task_{candidates.index(c)}" for c in final_codes]
        )
        .rename(old_to_new)
        .select(["subject_id", "prediction_time", *final_cols])
    )
    if "train" in args.splits:
        out = args.output_dir / "train.parquet"
        train_labels.write_parquet(out)
        log.info("Saved train.parquet (%d rows, %d tasks).", len(train_labels), len(final_codes))

    # Pass 2b: remaining splits with final codes only
    for split in args.splits:
        if split == "train":
            continue
        log.info("Pass 2b: generating labels for split %s ...", split)
        df = _generate_labels(
            args.meds_cohort_dir, split, final_codes, args.horizon_days, args.min_history_days, args.seed
        )
        if df.is_empty():
            log.warning("No labels for split %s — skipping.", split)
            continue
        out = args.output_dir / f"{split}.parquet"
        df.write_parquet(out)
        log.info("Saved %s (%d rows, %d tasks).", out.name, len(df), len(final_codes))

    # Metadata
    positive_rates = {}
    for i, code in enumerate(final_codes):
        col = f"task_{i}"
        positive_rates[code] = float(train_labels[col].mean()) if col in train_labels.columns else 0.0

    code_index = {str(i): code for i, code in enumerate(final_codes)}
    (args.output_dir / "code_index.json").write_text(json.dumps(code_index, indent=2))

    metadata = {
        "num_tasks": len(final_codes),
        "horizon_days": args.horizon_days,
        "min_history_days": args.min_history_days,
        "seed": args.seed,
        "measurement_threshold": args.measurement_threshold,
        "min_positive_rate": args.min_positive_rate,
        "max_positive_rate": args.max_positive_rate,
        "codes": final_codes,
        "positive_rates": positive_rates,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log.info("Saved code_index.json and metadata.json. Done.")


if __name__ == "__main__":
    main()
