"""Compute multi-task binary labels from MEDS cohort data.

For every patient in the cohort, a single prediction_time anchor is derived
directly from the EHR: the patient's first dynamic (non-static) event time
plus ``--anchor_offset_hours`` hours (default 24h).  No external task-label
file is required.

For each of the top-N codes (by corpus frequency) the script labels whether
the code appears within ``--horizon_days`` after prediction_time.

Outputs
-------
{output_dir}/{split}.parquet
    Schema: subject_id | prediction_time | task_0 | task_1 | ... | task_{N-1}
    task_i is 1.0 if the code occurred within horizon_days, else 0.0.

{output_dir}/code_index.json
    Maps task index (str) -> MEDS code string.

{output_dir}/metadata.json
    Records num_tasks, horizon_days, anchor_offset_hours.

Usage
-----
python scripts/prepare_multi_task_labels.py \\
    --meds_cohort_dir  /path/to/MEDS_cohort \\
    --output_dir       /path/to/mt_labels \\
    --num_tasks        25 \\
    --horizon_days     30
"""

import argparse
import json
import logging
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SPLITS = ("train", "tuning", "held_out")


def _read_meds_split(cohort_dir: Path, split: str) -> pl.DataFrame:
    shard_dir = cohort_dir / "data" / split
    if not shard_dir.exists():
        raise FileNotFoundError(f"No shard directory: {shard_dir}")
    files = list(shard_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {shard_dir}")
    return pl.concat([pl.read_parquet(f, columns=["subject_id", "time", "code"]) for f in files])


def _select_top_codes(cohort_dir: Path, num_tasks: int) -> list[str]:
    """Identify top-N codes by occurrence count across the train split."""
    log.info("Counting code frequencies in train split ...")
    df = _read_meds_split(cohort_dir, "train")
    counts = (
        df.filter(pl.col("time").is_not_null())
        .group_by("code")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    top = counts.head(num_tasks)["code"].to_list()
    log.info("Selected %d codes, top-3: %s", len(top), top[:3])
    return top


def _build_anchors(events: pl.DataFrame, anchor_offset_hours: float) -> pl.DataFrame:
    """Derive one prediction_time per patient from their first dynamic event.

    prediction_time = time of first non-null event + anchor_offset_hours. Patients whose entire record has
    null timestamps are dropped.
    """
    offset_us = int(anchor_offset_hours * 3600 * 1_000_000)
    anchors = (
        events.filter(pl.col("time").is_not_null())
        .group_by("subject_id")
        .agg(pl.col("time").min().alias("first_event_time"))
        .with_columns(
            (pl.col("first_event_time") + pl.duration(microseconds=offset_us)).alias("prediction_time")
        )
        .select(["subject_id", "prediction_time"])
    )
    log.info("Built %d anchors from EHR data.", len(anchors))
    return anchors


def _compute_labels_for_split(
    cohort_dir: Path,
    split: str,
    codes: list[str],
    horizon_days: float,
    anchor_offset_hours: float,
) -> pl.DataFrame:
    events = _read_meds_split(cohort_dir, split).filter(pl.col("time").is_not_null())
    anchors = _build_anchors(events, anchor_offset_hours)
    log.info("Split %s: %d patients -> %d anchors", split, events["subject_id"].n_unique(), len(anchors))

    joined = anchors.join(events, on="subject_id", how="left")
    joined = joined.with_columns(
        ((pl.col("time") - pl.col("prediction_time")).dt.total_seconds() / 86400.0).alias("delta_days")
    )

    in_window = (
        joined.filter((pl.col("delta_days") > 0) & (pl.col("delta_days") <= horizon_days))
        .filter(pl.col("code").is_in(codes))
        .group_by(["subject_id", "prediction_time", "code"])
        .agg(pl.len().alias("n"))
        .with_columns(pl.lit(1.0).alias("occurred"))
    )

    wide = in_window.pivot(
        values="occurred", index=["subject_id", "prediction_time"], on="code", aggregate_function="first"
    )

    result = anchors.join(wide, on=["subject_id", "prediction_time"], how="left")

    code_to_task = {code: f"task_{i}" for i, code in enumerate(codes)}
    for code in codes:
        col = code_to_task[code]
        if code in result.columns:
            result = result.rename({code: col}).with_columns(pl.col(col).fill_null(0.0).cast(pl.Float32))
        else:
            result = result.with_columns(pl.lit(0.0).cast(pl.Float32).alias(col))

    task_cols = [f"task_{i}" for i in range(len(codes))]
    return result.select(["subject_id", "prediction_time", *task_cols])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare multi-task binary labels from MEDS cohort.")
    parser.add_argument("--meds_cohort_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--num_tasks", type=int, default=25)
    parser.add_argument(
        "--horizon_days",
        type=float,
        default=30.0,
        help="Days after prediction_time to look for code occurrence.",
    )
    parser.add_argument(
        "--anchor_offset_hours",
        type=float,
        default=24.0,
        help="Hours after a patient's first event to set as prediction_time.",
    )
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    codes = _select_top_codes(args.meds_cohort_dir, args.num_tasks)

    code_index = {str(i): code for i, code in enumerate(codes)}
    (args.output_dir / "code_index.json").write_text(json.dumps(code_index, indent=2))
    log.info("Saved code_index.json")

    for split in args.splits:
        log.info("Processing split: %s", split)
        df = _compute_labels_for_split(
            args.meds_cohort_dir, split, codes, args.horizon_days, args.anchor_offset_hours
        )
        if df.is_empty():
            continue
        out_file = args.output_dir / f"{split}.parquet"
        df.write_parquet(out_file)
        log.info("Saved %s (%d rows, %d task columns)", out_file, len(df), args.num_tasks)

    metadata = {
        "num_tasks": args.num_tasks,
        "horizon_days": args.horizon_days,
        "anchor_offset_hours": args.anchor_offset_hours,
        "codes": codes,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log.info("Done.")


if __name__ == "__main__":
    main()
