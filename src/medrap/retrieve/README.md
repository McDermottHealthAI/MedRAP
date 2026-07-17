# medrap.retrieve

`medrap-retrieve` takes a trained model checkpoint, a retrieval dataset, and an
*index dataframe* naming a set of patient-timepoints in an already-preprocessed
MEDS dataset, and retrieves the configured retriever's top-K documents (or null)
for each one, saving document IDs and scores to disk.

An index dataframe is a directory of parquet file(s) with `subject_id` and
`prediction_time` columns -- the same shape as the `subject_id, prediction_time, task_0, ...` labels `medrap-preprocess` writes to `output_dir/tasks/`, minus the
task columns. It is the `meds_torchdata` "task index" concept
(`MEDSTorchDataConfig(task_labels_dir=...)` without a label column).

## How to run

```bash
medrap-retrieve \
  output_dir=<path/to/retrieve_run> \
  checkpoint_path=<path/to/train_run/checkpoints/best.ckpt> \
  training/datamodule=meds \
  training.datamodule.config.tensorized_cohort_dir=<path/to/tensorized> \
  training.datamodule.config.max_seq_len=512 \
  index_dataframe_dir=<path/to/index>
```

`tensorized_cohort_dir`/`max_seq_len` must match the MEDS dataset the checkpoint
was trained on. `index_dataframe_dir` may be any set of patient-timepoints --
`medrap-preprocess`'s task labels, a custom cohort, or a different one than was
used at training time.

### Using a different retrieval dataset

A different retrieval dataset may be used, provided it was built with the same
embedding model as the one used at training time. Override the retriever's
dataset path directly:

```bash
medrap-retrieve \
  ... \
  retriever.dataset_path=<path/to/different_retrieval_dataset>
```

### Key options

| Option                | Default                     | Description                                                                              |
| --------------------- | --------------------------- | ---------------------------------------------------------------------------------------- |
| `output_dir`          | **required**                | Directory where the retrieved-document parquet and configs are saved                     |
| `checkpoint_path`     | **required**                | Path to a `.ckpt` written by `medrap-train`                                              |
| `index_dataframe_dir` | **required**                | Directory of parquet file(s) with `subject_id`/`prediction_time` columns                 |
| `splits`              | `[train, tuning, held_out]` | MEDS splits to run retrieval over (a patient-timepoint's subject belongs to exactly one) |
| `batch_size`          | `32`                        | Batch size for the prediction dataloader                                                 |
| `do_overwrite`        | `false`                     | Overwrite an existing output directory                                                   |

## What happens under the hood

1. **Model reconstruction** -- the exact trained architecture (encoder / query
    projector / retriever / retrieval encoder / fusion / pooling / head) is
    rebuilt from config and the checkpoint `state_dict` is loaded, the same way
    `medrap-eval` does.
2. **Per-split retrieval** -- because a patient-timepoint's subject belongs to
    exactly one MEDS split, for each split in `splits` that has data in the
    tensorized cohort a `MEDSPytorchDataset` is built with
    `task_labels_dir=index_dataframe_dir` (`meds_torchdata` keeps only the rows
    belonging to that split), and `trainer.predict()` collects
    `doc_ids`/`doc_scores` for every row. A split with no data in the cohort
    (e.g. no `held_out` split) is skipped rather than treated as an error.
3. **Left join** -- the per-split results are concatenated and left-joined
    against the *full* input index dataframe, so every input patient-timepoint is
    represented in the output -- with null `doc_ids`/`doc_scores` for any row
    meds_torchdata could not build a sample for (e.g. a subject outside every
    configured split, or insufficient history before its prediction time).

## Output layout

```
<output_dir>/
  config.yaml                  # raw Hydra config as passed
  resolved_config.yaml         # fully resolved config (all interpolations expanded)
  retrieved_documents.parquet  # subject_id, prediction_time, doc_ids (list[K]), doc_scores (list[K])
```
