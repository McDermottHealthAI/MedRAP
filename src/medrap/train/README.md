# medrap.train

`medrap-train` and `medrap-eval` are CLI commands for supervised training and
evaluation of retrieval-augmented models on MEDS cohorts.

## How to run

### Training

```bash
medrap-train \
  output_dir=<path/to/run> \
  training/datamodule=meds \
  training.datamodule.config.tensorized_cohort_dir=<path/to/tensorized> \
  training.datamodule.config.max_seq_len=512 \
  training.datamodule.config.task_labels_dir=<path/to/tasks>
```

### Evaluation

```bash
medrap-eval \
  output_dir=<path/to/eval_run> \
  checkpoint_path=<path/to/run/checkpoints/best.ckpt> \
  training/datamodule=meds \
  training.datamodule.config.tensorized_cohort_dir=<path/to/tensorized> \
  training.datamodule.config.max_seq_len=512 \
  training.datamodule.config.task_labels_dir=<path/to/tasks>
```

## Key options

| Option           | Default                  | Description                                                          |
| ---------------- | ------------------------ | -------------------------------------------------------------------- |
| `output_dir`     | **required**             | Directory where checkpoints, logs, and configs are saved             |
| `seed`           | `42`                     | Global random seed (passed to `lightning.seed_everything`)           |
| `do_resume`      | `false`                  | Resume from the latest checkpoint in `output_dir/checkpoints/`       |
| `do_overwrite`   | `false`                  | Wipe and recreate `output_dir` if it already exists                  |
| `wandb_project`  | `medrap`                 | W&B project name (only used with `training/trainer=lightning_wandb`) |
| `wandb_run_name` | `medrap-train`           | W&B run name (only used with `training/trainer=lightning_wandb`)     |
| `eval_mode`      | `validate` *(eval only)* | `validate` runs the val split; `test` runs the held-out split        |

### Config groups

Each of these can be swapped by passing `training/<group>=<option>` on the command line.

| Group                 | Options                                                                                             | Default                 |
| --------------------- | --------------------------------------------------------------------------------------------------- | ----------------------- |
| `training/task`       | `binary_classification`, `marginalized_binary`, `multitask_binary`                                  | `binary_classification` |
| `training/loss`       | `binary_bce`, `marginalized_retrieval`, `multitask_binary_bce`, `multitask_binary_bce_marginalized` | `binary_bce`            |
| `training/datamodule` | `meds`, `meds_multitask`, `synthetic`, `synthetic_marginalized`                                     | `meds`                  |
| `training/trainer`    | `lightning_default`, `lightning_wandb`, `lightning_demo`                                            | `lightning_default`     |

### Resuming a run

Set `do_resume=true` and point `output_dir` at an existing run. The CLI finds
the latest checkpoint in `output_dir/checkpoints/` (preferring `last.ckpt`,
then the highest `epoch=*-step=*.ckpt`) and validates that the saved config
matches the current one before resuming.

```bash
medrap-train \
  output_dir=<path/to/existing_run> \
  do_resume=true \
  training/datamodule=meds \
  ...
```

### Logging to W&B

Switch the trainer config group to `lightning_wandb`:

```bash
medrap-train \
  training/trainer=lightning_wandb \
  wandb_project=my_project \
  wandb_run_name=my_run \
  output_dir=<path/to/run> \
  ...
```

The W&B trainer also enables `GradientNormCallback` and `LearningRateMonitor`
and evaluates on a 200-batch validation subset every 20% of each epoch.

## What happens under the hood

### Training (`medrap-train`)

1. **Seed** — `lightning.seed_everything(seed, workers=True)` is called first
    so all sources of randomness (data workers included) are deterministic.

2. **Output directory** — `output_dir` is created (or wiped if
    `do_overwrite=true`). The raw Hydra config is written to `config.yaml` and
    the fully resolved config (all interpolations expanded) to
    `resolved_config.yaml`.

3. **Module construction** — `instantiate_training_module` builds a
    `RetrievalAugmentedModel` from the encoder / retriever / fusion / head
    config groups, then wraps it in `MedRAPSupervisedLightningModule` together
    with the configured task and loss.

4. **Trainer construction** — the configured `lightning.Trainer` is instantiated
    with paths rebound to `output_dir` (checkpoints, log dirs).

5. **Fit** — `trainer.fit(module, datamodule=datamodule, ckpt_path=...)`.
    Checkpoints are saved by `ModelCheckpoint` (monitoring `val/loss`) and
    `ProgressCheckpointCallback` (saves at 10%, 50%, and 90% of training).

6. **Best checkpoint** — after fit completes, the checkpoint with the lowest
    `val/loss` is copied to `output_dir/best.ckpt`.

### Evaluation (`medrap-eval`)

1. `checkpoint_path` is loaded via `MedRAPSupervisedLightningModule.load_from_checkpoint`.
2. `trainer.validate` or `trainer.test` is called depending on `eval_mode`.
3. Metrics are written to the CSV logger under `output_dir/loggers/`.

## Output layout

```
<output_dir>/
  config.yaml            # raw Hydra config as passed
  resolved_config.yaml   # fully resolved config (all interpolations expanded)
  best.ckpt              # checkpoint with lowest val/loss (written after fit)
  checkpoints/
    last.ckpt            # latest epoch checkpoint (for resuming)
    epoch=N-step=M.ckpt  # periodic checkpoints
  loggers/
    csv/
      version_0/
        metrics.csv      # per-step/epoch metrics
```

## Multi-task training

To train on N simultaneous binary prediction tasks (produced by
`scripts/prepare_multi_task_labels.py`), swap three config groups:

```bash
medrap-train \
  training/task=multitask_binary \
  training/loss=multitask_binary_bce \
  training/datamodule=meds_multitask \
  training.task.num_tasks=<N> \
  training.datamodule.num_tasks=<N> \
  training.datamodule.mt_labels_dir=<path/to/mt_labels> \
  training.datamodule.config.tensorized_cohort_dir=<path/to/tensorized> \
  training.datamodule.config.max_seq_len=512 \
  training.datamodule.config.task_labels_dir=<path/to/mt_labels> \
  output_dir=<path/to/run>
```

To also train the retriever end-to-end via REALM-style marginalization:

```bash
  training/task=multitask_binary \
  training/loss=multitask_binary_bce_marginalized \
  marginalized_retrieval=true \
  ...
```

## Module reference

| Module                    | Contents                                                                                                                              |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `lightning_module.py`     | `MedRAPSupervisedLightningModule` — wraps a `RetrievalAugmentedModel` for supervised training                                         |
| `factory.py`              | `instantiate_training_module` — wires encoder / retriever / fusion / task / loss into a Lightning module from a Hydra config          |
| `task.py`                 | Task definitions: `BinaryClassificationTask`, `MarginalizedBinaryClassificationTask`, `MultiTaskBinaryClassificationTask`             |
| `losses.py`               | Loss functions: `BinaryClassificationLoss`, `MarginalizedRetrievalSupervisedLoss`, `MultiTaskBCELoss`, `MultiTaskBCEMarginalizedLoss` |
| `callbacks.py`            | `EndOfFitValAUROCCallback`, `GradientNormCallback`, `ProgressCheckpointCallback`                                                      |
| `metrics.py`              | AUROC and other scalar metrics used during training                                                                                   |
| `retrieval_logging.py`    | Batch-level retrieval diagnostics logged to W&B / Lightning                                                                           |
| `datamodule.py`           | `SyntheticSupervisedDatamodule` for smoke-testing without real data                                                                   |
| `multitask_datamodule.py` | `MultiTaskMEDSDatamodule` and `MultiTaskMEDSDataset` for simultaneous multi-label prediction                                          |
