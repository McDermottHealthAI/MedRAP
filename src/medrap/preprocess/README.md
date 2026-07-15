# medrap.preprocess

`medrap-preprocess` is a single CLI command that takes a raw MEDS dataset and
produces two things: a tensorized dataset ready for training, and binary task
labels for evaluation. It has two stages; both can be skipped if you already
have tensorized data.

## How to run

```bash
medrap-preprocess \
  meds_data_dir=<path/to/MEDS_cohort> \
  output_dir=<path/to/output>
```

Key options (all have defaults):

| Option                   | Default | Description                                                                                          |
| ------------------------ | ------- | ---------------------------------------------------------------------------------------------------- |
| `min_subjects_per_code`  | 100     | Drop codes appearing in fewer than this many subjects                                                |
| `min_events_per_subject` | 10      | Drop subjects with fewer distinct events than this                                                   |
| `num_tasks`              | 25      | Number of prediction tasks to generate                                                               |
| `horizon_days`           | 7.0     | How far ahead to look for a code occurrence                                                          |
| `min_history_days`       | 1.0     | Minimum history required before a prediction time                                                    |
| `seed`                   | 42      | Random seed for task sampling and prediction times                                                   |
| `min_positive_count`     | 10      | Minimum in-window positive subjects, on every split, a candidate code needs to be selected as a task |
| `min_positive_rate`      | 0.01    | Minimum in-window positive rate, on every split, a candidate code needs to be selected as a task     |
| `max_positive_rate`      | 0.5     | Maximum in-window positive rate, on every split, a candidate code may have to be selected as a task  |

### Skipping stage 1 (use existing tensorized data)

If you already have a tensorized dataset (e.g. from a previous run), pass its
path to skip stages 1 and 2 and go straight to task generation:

```bash
medrap-preprocess \
  meds_data_dir=<path/to/MEDS_cohort> \
  output_dir=<path/to/output> \
  tensorized_dir=<path/to/existing/tensorized>
```

`meds_data_dir` is still required because task generation reads the MEDS
parquet files directly to sample codes and build labels.

## What happens under the hood

### Stage 1 — MEDS-transforms pipeline

Runs `MEDS_transform-pipeline` on the raw MEDS dataset. This applies six
transformations in sequence:

1. **add_time_derived_measurements** — inserts `TIMELINE//DELTA//years//...`
    tokens between consecutive events, encoding the elapsed time as a binned
    code. These tokens are what enable time-aware models (e.g. ROPE attention).
2. **count_codes** — counts how many subjects each code appears in (train split
    only).
3. **filter_measurements** — drops codes below `min_subjects_per_code`.
4. **filter_subjects** — drops subjects with fewer than `min_events_per_subject`
    events.
5. **fit_quantile_binning** — computes per-code quantile breakpoints from the
    train split.
6. **bin_numeric_values** — replaces raw numeric values with a bin token (e.g.
    `LAB//50912//mg/dL//value_[0.9,1.1)`). The original `numeric_value` is
    dropped; all information is now in the code string.

Output goes to `output_dir/intermediate/`.

### Stage 2 — MTD_preprocess

Runs `MTD_preprocess` on the intermediate dataset. This tokenizes and
tensorizes the data into `.nrt` binary files that `meds-torch-data` loads at
training time. Output goes to `output_dir/tensorized/`.

### Stage 3 — generate_tasks (always runs)

Randomly samples `num_tasks` codes from the train split (excluding
`TIMELINE//` tokens, which are synthetic) and creates binary prediction labels
for each subject in every split:

- A single random **prediction time** is drawn per subject, uniformly from
    `[first_event + min_history_days, last_event - horizon_days]`. Subjects
    whose timeline is too short for this window are excluded.
- For each task code, the label is `1.0` if the code appears within
    `horizon_days` after the prediction time, otherwise `0.0`.

Every eligible code is windowed-tested (a memory-efficient group-by, not a
wide pivot, so this scales to the full vocabulary) and only kept as a
candidate task if it clears `min_positive_count` **and** a
`[min_positive_rate, max_positive_rate]` in-window positive-rate band — on
**every** split that will be generated (train, tuning, held_out), not just
train. Checking every split matters because splits differ in size: a code can
clear an absolute count on a large train split while its positive *rate*,
applied to a much smaller tuning/held_out split, implies an expected positive
count near zero there — collapsing that split's label to a single class, with
no learning signal and an undefined/dropped validation AUROC for that task
specifically. This is on top of the general problem that
`min_subjects_per_code` (stage 1) only guarantees a code occurred *somewhere*
in a subject's lifetime record — it says nothing about whether the code falls
inside any single subject's much narrower `horizon_days`-wide prediction
window. On a long-tailed clinical vocabulary, only a small fraction of
lifetime-frequent codes also clear this much narrower windowed bar; without
the filter, uniformly-sampled codes frequently produce a degenerate task with
no learning signal. If fewer than `num_tasks` codes pass on every split,
`generate_tasks` raises a `ValueError` rather than silently returning
degenerate tasks — loosen `min_positive_count`/`min_positive_rate`/
`max_positive_rate` or lower `num_tasks` in response.

Output goes to `output_dir/tasks/` and contains one parquet per split plus
`code_index.json` (mapping task index → code string) and `metadata.json`.
