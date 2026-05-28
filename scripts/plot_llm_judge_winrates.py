"""Plot per-task LLM-judge win-rate with bootstrap standard errors.

Reads ``<run_dir>/llm_judge/all_task_winrates.csv`` produced by the
multitask sweep in :mod:`scripts.run_llm_judge` and writes a single PDF
figure: x-axis = tasks, y-axis = ``target_preferred_rate ± standard_error``,
with a red dashed reference line at 0.5 (chance under
``half_credit_ties``).

Tasks where the judge tied on most pairs (``n_invalid / n_pairs >= 0.95``)
are drawn in gray to flag that their rate is mostly driven by ties — the
narrow error bar reflects "the judge kept saying tie," not strong signal.

Usage::

    python scripts/plot_llm_judge_winrates.py \\
        --run_dir outputs/mt_rope_cross_attention

By default the plot covers family F1 (retrieved vs random doc). Pass
``--family`` to render a different counterfactual when the sweep also
ran F2/F3/F4.

Pass ``--rank_sweep`` to emit one PDF per retrieval rank present in the
CSV (one rank per file: ``winrates_per_task_F1_rank{1..k}.pdf``). This
requires the upstream multitask sweep to have been run with
``run_llm_judge.py --rank_sweep``, which populates the ``target_rank``
column.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

_HIGH_TIE_THRESHOLD = 0.95


def _shorten_label(label: str, *, max_chars: int = 30) -> str:
    s = str(label)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _plot_family(
    fam_df: pl.DataFrame,
    *,
    family: str,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    fam_df = fam_df.sort("task_idx")
    rates = fam_df["target_preferred_rate"].to_numpy()
    ses = fam_df["standard_error"].to_numpy()
    n_invalid = fam_df["n_invalid"].to_numpy()
    n_pairs = np.maximum(fam_df["n_pairs"].to_numpy(), 1)
    high_tie_mask = (n_invalid / n_pairs) >= _HIGH_TIE_THRESHOLD
    labels = [_shorten_label(label) for label in fam_df["task_label"].to_list()]
    n_tasks = rates.size

    fig, ax = plt.subplots(figsize=(max(8.0, n_tasks * 0.5), 5.5))
    x = np.arange(n_tasks)

    # Plot solid-color bars first, then overlay gray for high-tie tasks so the
    # error bars are still drawn in the same call.
    colors = ["lightgray" if is_high_tie else "steelblue" for is_high_tie in high_tie_mask]
    ax.bar(
        x,
        rates,
        yerr=ses,
        capsize=3,
        color=colors,
        error_kw={"linewidth": 1, "ecolor": "black"},
    )

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="chance (0.5)")

    n_high_tie = int(high_tie_mask.sum())
    if n_high_tie > 0:
        # Single proxy handle for the legend (matplotlib doesn't legend bars by
        # color natively).
        ax.bar(
            [],
            [],
            color="lightgray",
            label=f"≥{int(_HIGH_TIE_THRESHOLD * 100)}% ties ({n_high_tie}/{n_tasks} tasks)",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Task")
    ax.set_ylabel(f"{family} win-rate (±1 SE)")
    ax.set_ylim(0, 1)
    title = f"Per-task LLM-judge win-rate — {family}"
    if title_suffix:
        title += f" {title_suffix}"
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")

    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path} ({n_tasks} tasks; {n_high_tie} marked high-tie)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot per-task LLM-judge win-rate with bootstrap SE bars.")
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Training run directory. Reads <run_dir>/llm_judge/all_task_winrates.csv.",
    )
    parser.add_argument(
        "--family",
        type=str,
        default="F1",
        help="Counterfactual family to plot (default: F1).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help=(
            "Output PDF path (single-rank mode only). Defaults to "
            "<run_dir>/llm_judge/winrates_per_task_<family>.pdf. Ignored when "
            "--rank_sweep is set; rank-sweep PDFs land at "
            "<run_dir>/llm_judge/winrates_per_task_<family>_rank{1..k}.pdf."
        ),
    )
    parser.add_argument(
        "--rank_sweep",
        action="store_true",
        help=(
            "Emit one PDF per ``target_rank`` value present in the CSV "
            "(produced by ``run_llm_judge.py --rank_sweep``). Each PDF is a "
            "clone of the single-rank plot restricted to that rank's slice."
        ),
    )
    args = parser.parse_args()

    csv_path = args.run_dir / "llm_judge" / "all_task_winrates.csv"
    if not csv_path.is_file():
        print(
            f"Error: {csv_path} not found. Run the multitask sweep first.",
            file=sys.stderr,
        )
        return 1

    df = pl.read_csv(csv_path)
    fam_df = df.filter(pl.col("family") == args.family)
    if fam_df.height == 0:
        available = sorted(df["family"].unique().to_list())
        print(
            f"Error: no rows for family={args.family!r} in {csv_path}. Available families: {available}.",
            file=sys.stderr,
        )
        return 2

    if args.rank_sweep:
        if "target_rank" not in fam_df.columns:
            print(
                f"Error: --rank_sweep requested but {csv_path} has no "
                "'target_rank' column. Re-run the sweep with "
                "`run_llm_judge.py --rank_sweep` first.",
                file=sys.stderr,
            )
            return 3
        ranks = sorted({int(r) for r in fam_df["target_rank"].to_list()})
        for rank in ranks:
            rank_df = fam_df.filter(pl.col("target_rank") == rank)
            output_path = args.run_dir / "llm_judge" / f"winrates_per_task_{args.family}_rank{rank + 1}.pdf"
            _plot_family(
                rank_df,
                family=args.family,
                output_path=output_path,
                title_suffix=f"(top-{rank + 1})",
            )
        return 0

    output_path = args.output_path or args.run_dir / "llm_judge" / f"winrates_per_task_{args.family}.pdf"
    _plot_family(fam_df, family=args.family, output_path=output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
