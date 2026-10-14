"""Fit a separable-power scaling law per retrieval arm and compare them.

Thin wrapper around ``simple_scaling_laws``: it owns none of the statistics (see that
package's README for what ``fit``/``compare`` actually do -- wild cluster bootstrap
over configurations, logit-scale fitting for unit-interval metrics, paired draws for
comparisons). This module's only job is isolating, per arm, the rows and the
``test_loss__<arm>`` column :func:`medrap.scaling_laws.schema.build_results_table`
wrote for that arm -- ``simple_scaling_laws`` discovers targets from every
``test_loss__*``/``test_metric__*`` column present in the frame regardless of what's
passed via ``targets``, so another arm's (all-null, for these rows) loss column has to
be dropped rather than merely left out of ``targets``, and ``test_metric__auroc`` (the
one column comparable across arms) has to be set as ``primary_target`` explicitly,
since the package's "defaults to the single test loss" rule doesn't apply once more
than one loss column exists in the frame.
"""

from collections.abc import Sequence
from typing import Any

import polars as pl
from simple_scaling_laws import ScalingLawModel, compare, fit


def fit_arms(
    results: pl.DataFrame, arms: Sequence[str], n_draws: int = 1000, seed: int = 0
) -> dict[str, ScalingLawModel]:
    """Fit one scaling-law model per arm, from that arm's own rows and loss column.

    ``results`` is expected to hold every arm's rows together, as
    :func:`medrap.scaling_laws.schema.build_results_table` produces it, with an
    ``"arm"`` column and one ``test_loss__<arm>`` column per arm (null outside that
    arm's own rows). Each arm is fit from only its own rows, with every other arm's
    loss column dropped and ``test_metric__auroc`` (comparable across arms) set as the
    primary target.

    Examples:
        >>> from simple_scaling_laws.simulate import simulate_runs
        >>> runs = simulate_runs(
        ...     {
        ...         "test_metric__auroc": {"E": 0.9, "A": -0.1, "alpha": 0.3, "B": -0.05, "beta": 0.2},
        ...     },
        ...     runs_per_config=2,
        ...     evaluations_per_run=4,
        ...     run_sd=0.01,
        ...     eval_sd=0.02,
        ...     seed=0,
        ... ).with_columns(arm=pl.lit("only_arm"))
        >>> models = fit_arms(runs, arms=["only_arm"], n_draws=50, seed=0)
        >>> sorted(models)
        ['only_arm']
        >>> models["only_arm"].primary_target
        'test_metric__auroc'

        Two arms, each with their own (otherwise all-null) loss column:

        >>> import polars as pl
        >>> patient_only = runs.with_columns(arm=pl.lit("patient_only"), test_loss__patient_only=pl.lit(0.4))
        >>> retrieval = runs.with_columns(arm=pl.lit("retrieval"), test_loss__retrieval=pl.lit(0.3))
        >>> both = pl.concat([patient_only, retrieval], how="diagonal")
        >>> models = fit_arms(both, arms=["patient_only", "retrieval"], n_draws=50, seed=0)
        >>> sorted(models)
        ['patient_only', 'retrieval']
        >>> {name: model.primary_target for name, model in models.items()}
        {'patient_only': 'test_metric__auroc', 'retrieval': 'test_metric__auroc'}
    """
    models = {}
    for arm in arms:
        arm_rows = results.filter(pl.col("arm") == arm)
        loss_column = f"test_loss__{arm}"
        other_loss_columns = [c for c in arm_rows.columns if c.startswith("test_loss__") and c != loss_column]
        arm_rows = arm_rows.drop(other_loss_columns)
        models[arm] = fit(arm_rows, primary_target="test_metric__auroc", n_draws=n_draws, seed=seed)
    return models


def compare_arms(
    models: dict[str, ScalingLawModel],
    points: Any,
    reference: str,
    target: str = "test_metric__auroc",
) -> pl.DataFrame:
    """Compare arms' fitted AUROC at a set of (model_size, dataset_size) points.

    Returns ``simple_scaling_laws.compare``'s long table (one row per point per arm),
    with ``p_better_than_reference`` giving the probability each non-reference arm beats
    ``reference`` at that scale -- the signal for "does retrieval's value grow with scale".

    Examples:
        >>> from simple_scaling_laws.simulate import simulate_runs
        >>> def arm_runs(a, seed):
        ...     return simulate_runs(
        ...         {"test_metric__auroc": {"E": 0.9, "A": -0.1, "alpha": a, "B": -0.05, "beta": 0.2}},
        ...         runs_per_config=2,
        ...         evaluations_per_run=4,
        ...         run_sd=0.01,
        ...         eval_sd=0.02,
        ...         seed=seed,
        ...     )
        >>> models = {
        ...     "patient_only": fit(arm_runs(0.3, 0), n_draws=100, seed=0),
        ...     "retrieval": fit(arm_runs(0.4, 1), n_draws=100, seed=0),
        ... }
        >>> points = {
        ...     "point_id": ["now"],
        ...     "model_size__n_params": [1e8],
        ...     "dataset_size__n_subjects": [1e5],
        ... }
        >>> table = compare_arms(models, points, reference="patient_only")
        >>> set(table["system"]) == {"patient_only", "retrieval"}
        True
        >>> "p_better_than_reference" in table.columns
        True
    """
    return compare(models, points, target=target, reference=reference)
