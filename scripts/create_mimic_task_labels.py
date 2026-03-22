#!/usr/bin/env -S uv run
"""Create binary classification task labels from a MIMIC-IV MEDS cohort.

This script reads the merged MEDS parquet shards and produces task label files in the
standard MEDS label format (``subject_id``, ``prediction_time``, ``boolean_value``).
The labels can then be used with ``meds_torchdata``'s ``task_labels_dir`` configuration.

Supported tasks
---------------
``in_hospital_mortality``
    Labels each hospital admission as positive if the patient has a ``DEATH``-prefixed
    event during the visit window, and negative otherwise. The ``prediction_time`` is
    set to the admission event time so that only pre-admission data is visible to the
    model.

Usage
-----
::

    uv run scripts/create_mimic_task_labels.py \\
        --meds-dir mimic/MEDS_cohort \\
        --output-dir mimic/task_labels/in_hospital_mortality \\
        --task in_hospital_mortality

The output directory will contain one parquet file per split (``train.parquet``,
``tuning.parquet``, ``held_out.parquet``).

These label files are consumed by ``meds_torchdata.MEDSTorchDataConfig`` via its
``task_labels_dir`` parameter.

Examples
--------
Demonstrate the label schema on a tiny synthetic MEDS-like dataframe:

>>> import polars as pl
>>> from datetime import datetime
>>> events = pl.DataFrame({
...     "subject_id": [1, 1, 1, 1, 2, 2, 2],
...     "time": [
...         datetime(2020, 1, 1),
...         datetime(2020, 1, 1),
...         datetime(2020, 1, 2),
...         datetime(2020, 1, 3),
...         datetime(2020, 2, 1),
...         datetime(2020, 2, 2),
...         datetime(2020, 2, 3),
...     ],
...     "code": [
...         "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM",
...         "HR",
...         "TEMP",
...         "MEDS_DEATH",
...         "HOSPITAL_ADMISSION//ELECTIVE//PHYSICIAN REFERRAL",
...         "HR",
...         "HOSPITAL_DISCHARGE//HOME",
...     ],
...     "numeric_value": [
...         None,
...         80.0,
...         37.5,
...         None,
...         None,
...         90.0,
...         None,
...     ],
... })
>>> labels = _build_in_hospital_mortality_labels(events)
>>> labels.sort("subject_id")
shape: (2, 3)
┌────────────┬─────────────────────┬───────────────┐
│ subject_id ┆ prediction_time     ┆ boolean_value │
│ ---        ┆ ---                 ┆ ---           │
│ i64        ┆ datetime[μs]        ┆ bool          │
╞════════════╪═════════════════════╪═══════════════╡
│ 1          ┆ 2020-01-01 00:00:00 ┆ true          │
│ 2          ┆ 2020-02-01 00:00:00 ┆ false         │
└────────────┴─────────────────────┴───────────────┘
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl


def _build_in_hospital_mortality_labels(events: pl.DataFrame) -> pl.DataFrame:
    """Build in-hospital mortality labels from MEDS events.

    For each hospital admission (identified by ``ADMISSION`` codes), the label is
    ``True`` if the patient has any ``DEATH``-prefixed event after the admission time,
    and ``False`` otherwise.

    Args:
        events: MEDS events dataframe with columns ``subject_id``, ``time``, ``code``.

    Returns:
        Labels dataframe with columns ``subject_id``, ``prediction_time``,
        ``boolean_value``.
    """
    admissions = events.filter(pl.col("code").str.starts_with("HOSPITAL_ADMISSION"))
    deaths = events.filter(
        pl.col("code").str.starts_with("MEDS_DEATH")
        | pl.col("code").str.starts_with("HOSPITAL_DISCHARGE//DIED")
    )

    death_subjects = set(deaths["subject_id"].unique().to_list())

    labels = admissions.select(
        pl.col("subject_id"),
        pl.col("time").alias("prediction_time"),
        pl.col("subject_id").is_in(death_subjects).alias("boolean_value"),
    ).unique(subset=["subject_id", "prediction_time"])

    return labels


def _find_meds_shards(meds_dir: Path, split: str) -> list[Path]:
    """Find all MEDS parquet shards for a given split.

    Searches in ``meds_dir/data/{split}/`` for parquet files.

    Args:
        meds_dir: Root MEDS cohort directory.
        split: Data split name (``train``, ``tuning``, ``held_out``).

    Returns:
        Sorted list of shard paths.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     d = Path(tmpdir) / "data" / "train"
        ...     d.mkdir(parents=True)
        ...     _ = (d / "0.parquet").write_bytes(b"")
        ...     _ = (d / "1.parquet").write_bytes(b"")
        ...     _find_meds_shards(Path(tmpdir), "train")
        [PosixPath('...data/train/0.parquet'), PosixPath('...data/train/1.parquet')]
    """
    data_dir = meds_dir / "data" / split
    if not data_dir.is_dir():
        return []
    return sorted(data_dir.glob("*.parquet"))


def _process_split(
    meds_dir: Path,
    output_dir: Path,
    split: str,
    task: str,
) -> int:
    """Process a single split and write the label parquet file.

    Returns:
        Number of labels written.
    """
    shards = _find_meds_shards(meds_dir, split)
    if not shards:
        print(f"  {split}: no shards found, skipping")
        return 0

    dfs = [pl.read_parquet(shard) for shard in shards]
    events = pl.concat(dfs)

    if task == "in_hospital_mortality":
        labels = _build_in_hospital_mortality_labels(events)
    else:
        raise ValueError(f"Unknown task: {task!r}")

    if len(labels) == 0:
        print(f"  {split}: no labels generated, skipping")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.parquet"
    labels.write_parquet(output_path)
    n_pos = labels.filter(pl.col("boolean_value")).height
    n_neg = labels.filter(~pl.col("boolean_value")).height
    print(f"  {split}: {len(labels)} labels ({n_pos} positive, {n_neg} negative) -> {output_path}")
    return len(labels)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for task label creation.

    Examples:
        >>> main(["--help"])  # doctest: +SKIP
    """
    parser = argparse.ArgumentParser(
        description="Create binary task labels from a MIMIC-IV MEDS cohort.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--meds-dir",
        type=Path,
        required=True,
        help="Path to the MEDS cohort directory (e.g. mimic/MEDS_cohort).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for task label parquet files.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="in_hospital_mortality",
        choices=["in_hospital_mortality"],
        help="Task to extract labels for (default: in_hospital_mortality).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "tuning", "held_out"],
        help="Splits to process (default: train tuning held_out).",
    )
    args = parser.parse_args(argv)

    print(f"Creating {args.task!r} labels from {args.meds_dir}")
    total = 0
    for split in args.splits:
        total += _process_split(args.meds_dir, args.output_dir, split, args.task)

    if total == 0:
        print("Warning: no labels were created. Check that your MEDS cohort has data.")
        return 1
    print(f"Done. Total: {total} labels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
