"""Command-line entrypoints for the scaling-law sweep: grid generation and results fitting.

Neither entrypoint submits or runs training jobs -- ``medrap-scaling-laws-grid`` only
writes the list of ``medrap-train`` override lines for a grid spec, and
``medrap-scaling-laws-fit`` only fits/compares arms from an already-aggregated results
table. See ``src/medrap/scaling_laws/README.md`` for the end-to-end workflow.
"""

import argparse
import json
from pathlib import Path

import polars as pl

from medrap.scaling_laws.fit import compare_arms, fit_arms
from medrap.scaling_laws.grid import build_grid, cell_to_cli_overrides


def grid_main() -> None:
    """Write a sweep grid's ``medrap-train`` overrides, one line per run, to ``--output``.

    Takes a JSON grid spec with keys ``capacities``, ``dataset_fractions``,
    ``arm_overrides``, ``n_repeats``, and optionally ``base_seed`` (see
    :func:`medrap.scaling_laws.grid.build_grid`). Each output line is
    ``<training_run_id>\\t<override1> <override2> ...``, so launching the sweep is
    looping over the file with whatever submission mechanism the target cluster uses.
    """
    parser = argparse.ArgumentParser(description=grid_main.__doc__)
    parser.add_argument("spec", type=Path, help="JSON file with the grid spec.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    cells = build_grid(
        capacities=spec["capacities"],
        dataset_fractions=spec["dataset_fractions"],
        arm_overrides=spec["arm_overrides"],
        n_repeats=spec["n_repeats"],
        base_seed=spec.get("base_seed", 0),
    )
    lines = [f"{cell.training_run_id}\t{' '.join(cell_to_cli_overrides(cell))}" for cell in cells]
    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(cells)} sweep cells to {args.output}")


def fit_main() -> None:
    """Fit + compare retrieval arms from an aggregated results table, writing the comparison table.

    ``results`` must match :func:`medrap.scaling_laws.schema.build_results_table`'s
    output (an ``arm`` column plus one ``test_loss__<arm>`` column per arm). ``--points``
    is a JSON file of ``point_id``/``model_size__n_params``/``dataset_size__n_subjects``
    lists, passed through to ``simple_scaling_laws.compare``.
    """
    parser = argparse.ArgumentParser(description=fit_main.__doc__)
    parser.add_argument("results", type=Path, help="Parquet file of per-run results.")
    parser.add_argument("--points", type=Path, required=True, help="JSON file of query points.")
    parser.add_argument("--reference", required=True, help="Arm to compare others against.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = pl.read_parquet(args.results)
    arms = sorted(results["arm"].unique())
    models = fit_arms(results, arms=arms, n_draws=args.n_draws, seed=args.seed)
    points = json.loads(args.points.read_text())
    table = compare_arms(models, points, reference=args.reference)
    table.write_parquet(args.output)
    print(f"wrote comparison for {len(arms)} arms to {args.output}")
