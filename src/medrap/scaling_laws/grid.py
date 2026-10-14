"""Build the (capacity, dataset-size, retrieval-arm, repeat) grid for a scaling-law sweep.

Pure combinatorics: this module decides which runs exist, their deterministic ids, and
their Hydra overrides. It carries no assumptions about which fusion/retriever/
query_projector architecture a given retrieval arm actually uses -- those are supplied
by the caller via ``arm_overrides`` (see :func:`build_grid`), since that choice depends
on whatever the retrieval/no-retrieval setup the cluster-side experiment is actually
configured to run.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScalingLawCell:
    """One training run in the scaling-law sweep grid."""

    capacity: str
    dataset_fraction: float
    arm: str
    repeat: int
    seed: int
    training_run_id: str
    train_set_id: str
    overrides: dict[str, str]


def build_grid(
    capacities: list[str],
    dataset_fractions: list[float],
    arm_overrides: dict[str, dict[str, str]],
    n_repeats: int,
    base_seed: int = 0,
) -> list[ScalingLawCell]:
    """Enumerate every (capacity, dataset_fraction, arm, repeat) cell.

    Each repeat gets its own seed (``base_seed + repeat``), reused both as the training
    run's optimizer seed and as the seed for sampling that repeat's task subset / data
    subsample -- so the same seed drives every random choice that distinguishes one
    repeat from another at a fixed (capacity, dataset_fraction, arm).

    Args:
        capacities: ``conf/encoder`` config names, e.g. ``["rope_dim16", "rope_dim128"]``.
            Merged in as ``encoder=<capacity>``.
        dataset_fractions: Fractions of the training set to use, e.g. ``[0.01, 0.1, 1.0]``.
        arm_overrides: Maps an arm name (e.g. ``"patient_only"``, ``"retrieval"``) to its
            *fixed* Hydra overrides (task, loss, fusion, retriever, query_projector, ...).
            This function does not guess those; it only plugs them in.
        n_repeats: Number of independently-seeded repeats per (capacity, fraction, arm) cell.
        base_seed: Seed for repeat 0; repeat ``r`` uses ``base_seed + r``.

    Examples:
        >>> cells = build_grid(
        ...     capacities=["rope_dim16", "rope_dim128"],
        ...     dataset_fractions=[0.1, 1.0],
        ...     arm_overrides={
        ...         "patient_only": {"task": "binary_classification", "loss": "binary_bce"},
        ...         "retrieval": {"task": "marginalized_binary", "loss": "marginalized_retrieval"},
        ...     },
        ...     n_repeats=2,
        ... )
        >>> len(cells)  # 2 capacities * 2 fractions * 2 arms * 2 repeats
        16
        >>> cells[0].training_run_id
        'rope_dim16-frac0.1-patient_only-rep0'
        >>> cells[0].overrides["encoder"]
        'rope_dim16'
        >>> cells[0].overrides["task"]
        'binary_classification'
        >>> cells[0].seed == cells[8].seed  # same repeat index, different capacity -> same seed
        True
        >>> {cell.training_run_id for cell in cells} == {
        ...     f"{c}-frac{f}-{a}-rep{r}"
        ...     for c in ("rope_dim16", "rope_dim128")
        ...     for f in (0.1, 1.0)
        ...     for a in ("patient_only", "retrieval")
        ...     for r in (0, 1)
        ... }
        True
    """
    cells = []
    for capacity in capacities:
        for dataset_fraction in dataset_fractions:
            for arm, fixed_overrides in arm_overrides.items():
                for repeat in range(n_repeats):
                    seed = base_seed + repeat
                    training_run_id = f"{capacity}-frac{dataset_fraction}-{arm}-rep{repeat}"
                    train_set_id = f"frac{dataset_fraction}-rep{repeat}"
                    overrides = {
                        "encoder": capacity,
                        **fixed_overrides,
                        "seed": str(seed),
                    }
                    cells.append(
                        ScalingLawCell(
                            capacity=capacity,
                            dataset_fraction=dataset_fraction,
                            arm=arm,
                            repeat=repeat,
                            seed=seed,
                            training_run_id=training_run_id,
                            train_set_id=train_set_id,
                            overrides=overrides,
                        )
                    )
    return cells


def cell_to_cli_overrides(cell: ScalingLawCell) -> list[str]:
    """Render a cell's Hydra overrides as ``key=value`` CLI args for ``medrap-train``.

    Examples:
        >>> cell = build_grid(
        ...     ["rope_dim16"],
        ...     [0.1],
        ...     {"patient_only": {"task": "binary_classification"}},
        ...     n_repeats=1,
        ... )[0]
        >>> cell_to_cli_overrides(cell)
        ['encoder=rope_dim16', 'task=binary_classification', 'seed=0']
    """
    return [f"{key}={value}" for key, value in cell.overrides.items()]
