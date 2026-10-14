# medrap.scaling_laws

Supports the scaling-laws experiment: fit `y = A + B * x^-alpha` independently over
dataset size and model capacity, with and without retrieval, and check whether the gap
to the fitted asymptote predicts how much retrieval helps. Uses
[`simple-scaling-laws`](https://github.com/mmcdermott/simple-scaling-laws) for the
actual fitting/comparison statistics (wild cluster bootstrap over configurations,
paired draws for arm comparisons -- see that package's README).

This package only builds and analyzes the sweep -- it does not submit or run training
jobs. That's intentional: which cluster, which submission mechanism, and the exact
fusion/retriever/query_projector architecture for each retrieval arm are all left to the
caller, since they depend on infrastructure outside this repo.

## Workflow

1. **Pick the grid.** Capacities are `conf/encoder` config names --
    `rope_dim16`/`rope_dim64`/`rope.yaml` (128-D)/`rope_dim256` span the capacity axis
    for `TimeDeltaRoPEPatientEncoder`. Dataset-size fractions are handled by
    subsampling `task_labels_dir`'s `train.parquet` (see below), not a training flag.

2. **Subsample the training labels** for each dataset-size grid point:

    ```python
    from medrap.scaling_laws.subsample_labels import write_subsampled_task_labels_dir

    write_subsampled_task_labels_dir(
        source_dir="<path/to/tasks>", output_dir="<path/to/tasks_frac0.1>", fraction=0.1, seed=0
    )
    ```

    `tuning.parquet`/`held_out.parquet` are copied through unchanged -- only `train.parquet`
    shrinks, so every grid point is evaluated on the same resample
    (`simple-scaling-laws` relies on this: reusing `test_set_id` across models is what
    lets it tell shared evaluation noise apart from a real difference between arms).

3. **Generate the sweep grid**, given a JSON spec naming each arm's *fixed* Hydra
    overrides (this module doesn't guess those):

    ```json
    {
      "capacities": ["rope_dim16", "rope_dim64", "rope", "rope_dim256"],
      "dataset_fractions": [0.01, 0.03, 0.1, 0.3, 1.0],
      "n_repeats": 3,
      "arm_overrides": {
        "patient_only": {
          "task": "binary_classification",
          "loss": "binary_bce",
          "marginalized_retrieval": "false"
        },
        "retrieval": {
          "task": "marginalized_binary",
          "loss": "marginalized_retrieval",
          "marginalized_retrieval": "true"
        }
      }
    }
    ```

    ```bash
    medrap-scaling-laws-grid grid_spec.json --output sweep_cells.tsv
    ```

    Each line of `sweep_cells.tsv` is `<training_run_id>\t<override1> <override2> ...`
    -- pass `query_projector.in_dim=<embedding_dim>` and
    `training.datamodule.config.task_labels_dir=<path/to/tasks_frac...>` per cell on top
    of these, since those depend on paths/dims this module doesn't know about. Loop over
    the file with whatever launches `medrap-train` on the target cluster (ssh+sbatch,
    local subprocess, ...).

4. **Aggregate results** from each run's eval metrics (e.g. pulled from W&B, one row
    per evaluation resample) into `simple-scaling-laws`'s input format:

    ```python
    from medrap.scaling_laws.schema import build_results_table

    table = build_results_table(records)  # each record: see schema.py's docstring
    table.write_parquet("results.parquet")
    ```

5. **Fit and compare arms**:

    ```bash
    medrap-scaling-laws-fit results.parquet \
      --points query_points.json \
      --reference patient_only \
      --output comparison.parquet
    ```

    `comparison.parquet` has `p_better_than_reference` per arm per point -- whether
    retrieval's advantage over `patient_only` grows with scale is read directly off this
    column across the `query_points.json` grid.
