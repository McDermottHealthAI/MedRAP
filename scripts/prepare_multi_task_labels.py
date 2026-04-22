"""Compute multi-task binary labels from MEDS cohort data.

For each patient × prediction_time anchor (taken from an existing task label file),
and for each of the top-N codes selected by corpus frequency, labels whether the code
appears within a fixed horizon (days) after the prediction_time.

Outputs
-------
{output_dir}/{split}.parquet
    Schema: subject_id | prediction_time | task_0 | task_1 | ... | task_{N-1}
    task_i is 1.0 if the code occurred within horizon_days, else 0.0.

{output_dir}/code_index.json
    Maps task index (str) -> MEDS code string.

{output_dir}/metadata.json
    Records num_tasks, horizon_days, split list used.

Usage
-----
python scripts/prepare_multi_task_labels.py \\
    --meds_cohort_dir  /path/to/MEDS_cohort \\
    --task_labels_dir  /path/to/task_labels/in_hospital_mortality \\
    --output_dir       /path/to/tte_labels \\
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


def _compute_labels_for_split(
    cohort_dir: Path,
    task_labels_path: Path,
    split: str,
    codes: list[str],
    horizon_days: float,
) -> pl.DataFrame:
    label_file = task_labels_path / f"{split}.parquet"
    if not label_file.exists():
        log.warning("No task label file for split %s, skipping.", split)
        return pl.DataFrame()

    anchors = pl.read_parquet(label_file, columns=["subject_id", "prediction_time"])
    log.info("Split %s: %d anchors", split, len(anchors))

    events = _read_meds_split(cohort_dir, split).filter(pl.col("time").is_not_null())

    # Join events to anchors on subject_id
    joined = anchors.join(events, on="subject_id", how="left")

    # Days from prediction_time to event time
    joined = joined.with_columns(
        ((pl.col("time") - pl.col("prediction_time")).dt.total_seconds() / 86400.0).alias("delta_days")
    )

    # For each (subject_id, prediction_time, code): did code appear in (0, horizon_days]?
    in_window = (
        joined.filter((pl.col("delta_days") > 0) & (pl.col("delta_days") <= horizon_days))
        .filter(pl.col("code").is_in(codes))
        .group_by(["subject_id", "prediction_time", "code"])
        .agg(pl.len().alias("n"))
        .with_columns(pl.lit(1.0).alias("occurred"))
    )

    # Pivot to wide: one column per code
    wide = in_window.pivot(
        values="occurred", index=["subject_id", "prediction_time"], on="code", aggregate_function="first"
    )

    # Left-join back to anchors so every anchor row is present
    result = anchors.join(wide, on=["subject_id", "prediction_time"], how="left")

    # Rename code columns to task_0, task_1, ...; fill missing with 0.0
    code_to_task = {code: f"task_{i}" for i, code in enumerate(codes)}
    for code in codes:
        col = code_to_task[code]
        if code in result.columns:
            result = result.rename({code: col}).with_columns(pl.col(col).fill_null(0.0).cast(pl.Float32))
        else:
            result = result.with_columns(pl.lit(0.0).cast(pl.Float32).alias(col))

    # Ensure column order: subject_id, prediction_time, task_0, ..., task_{N-1}
    task_cols = [f"task_{i}" for i in range(len(codes))]
    return result.select(["subject_id", "prediction_time", *task_cols])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare multi-task binary labels from MEDS cohort.")
    parser.add_argument("--meds_cohort_dir", required=True, type=Path)
    parser.add_argument("--task_labels_dir", required=True, type=Path,
                        help="Dir containing train.parquet, tuning.parquet, held_out.parquet")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--num_tasks", type=int, default=25)
    parser.add_argument("--horizon_days", type=float, default=30.0,
                        help="Days after prediction_time to look for code occurrence.")
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
            args.meds_cohort_dir, args.task_labels_dir, split, codes, args.horizon_days
        )
        if df.is_empty():
            continue
        out_file = args.output_dir / f"{split}.parquet"
        df.write_parquet(out_file)
        log.info("Saved %s (%d rows, %d task columns)", out_file, len(df), args.num_tasks)

    metadata = {
        "num_tasks": args.num_tasks,
        "horizon_days": args.horizon_days,
        "splits": args.splits,
        "codes": codes,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log.info("Done.")


if __name__ == "__main__":
    main()
