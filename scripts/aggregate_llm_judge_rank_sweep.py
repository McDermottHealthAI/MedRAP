"""Aggregate per-(task, rank) LLM-judge outputs into a single CSV.

The SLURM array job at ``scripts/run_llm_judge_rank_sweep_array_slurm.sh``
launches 200 invocations of ``scripts/run_llm_judge.py``, one per
(task, rank) cell. Each writes to its own ``task<K>_rank<R>/`` subdir
under ``<run_dir>/llm_judge/``. This script stitches them together into
``<run_dir>/llm_judge/all_task_winrates.csv`` with the same schema as
the in-process writer in ``scripts/run_llm_judge.py``
(``_write_cross_task_summary``) so the downstream plot script can read
it unchanged.

Usage::

    python scripts/aggregate_llm_judge_rank_sweep.py \\
        --run_dir outputs/mt_rope_cross_attention
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl


def _collect_long_rows(judge_dir: Path) -> list[dict]:
    """Walk ``judge_dir/task<K>_rank<R>/`` cells, return long-format rows."""
    long_rows: list[dict] = []
    cell_dirs = sorted(judge_dir.glob("task*_rank*/"))
    if not cell_dirs:
        return long_rows

    for cell_dir in cell_dirs:
        cfg_path = cell_dir / "run_config.json"
        csv_path = cell_dir / "family_winrates.csv"
        if not cfg_path.is_file() or not csv_path.is_file():
            print(
                f"WARN: skipping {cell_dir.name}: missing "
                f"{'run_config.json' if not cfg_path.is_file() else 'family_winrates.csv'}.",
                file=sys.stderr,
            )
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"WARN: skipping {cell_dir.name}: could not read run_config.json ({exc}).",
                file=sys.stderr,
            )
            continue

        task_idx = cfg.get("target_task")
        task_code = cfg.get("task_code")
        task_label = cfg.get("task_label")
        n_filtered = cfg.get("task_filtered_n")
        if task_idx is None:
            print(
                f"WARN: skipping {cell_dir.name}: run_config.json has no target_task.",
                file=sys.stderr,
            )
            continue

        try:
            df = pl.read_csv(csv_path)
        except Exception as exc:
            print(
                f"WARN: skipping {cell_dir.name}: could not parse family_winrates.csv ({exc}).",
                file=sys.stderr,
            )
            continue

        for row in df.iter_rows(named=True):
            long_rows.append(
                {
                    "task_idx": task_idx,
                    "task_code": task_code,
                    "task_label": task_label,
                    "n_filtered_patients": n_filtered,
                    **row,
                }
            )

    return long_rows


def aggregate(run_dir: Path) -> Path:
    """Aggregate (task, rank) cells under ``run_dir/llm_judge/``.

    Returns the path to the written CSV.
    """
    judge_dir = run_dir / "llm_judge"
    if not judge_dir.is_dir():
        raise FileNotFoundError(f"{judge_dir} not found")

    long_rows = _collect_long_rows(judge_dir)
    if not long_rows:
        raise RuntimeError(
            f"No (task, rank) cells found under {judge_dir}. Expected subdirs "
            "named task<K>_rank<R>/ produced by the rank-sweep array job."
        )

    long_df = pl.DataFrame(long_rows)
    out_path = judge_dir / "all_task_winrates.csv"
    long_df.write_csv(out_path)
    print(f"Wrote {out_path} ({long_df.height} rows from {len({r['task_idx'] for r in long_rows})} tasks)")

    # Surface missing cells: expect every task_idx to appear once per
    # target_rank value observed across the data.
    if "target_rank" in long_df.columns:
        observed_ranks = sorted(long_df["target_rank"].unique().to_list())
        observed_tasks = sorted(long_df["task_idx"].unique().to_list())
        expected_pairs = {(t, r) for t in observed_tasks for r in observed_ranks}
        actual_pairs = {(row["task_idx"], row["target_rank"]) for row in long_rows}
        missing = sorted(expected_pairs - actual_pairs)
        if missing:
            print(
                f"WARN: {len(missing)} (task, rank) cell(s) missing from the "
                f"aggregate: {missing[:10]}{'...' if len(missing) > 10 else ''}",
                file=sys.stderr,
            )

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate per-(task, rank) LLM-judge results into one CSV.")
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Training run directory containing llm_judge/task<K>_rank<R>/ subdirs.",
    )
    args = parser.parse_args()
    try:
        aggregate(args.run_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
