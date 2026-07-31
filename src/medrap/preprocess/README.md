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

| Option                   | Default | Description                                           |
| ------------------------ | ------- | ----------------------------------------------------- |
| `min_subjects_per_code`  | 100     | Drop codes appearing in fewer than this many subjects |
| `min_events_per_subject` | 10      | Drop subjects with fewer distinct events than this    |
| `num_tasks`              | 25      | Number of prediction tasks to generate                |
| `horizon_days`           | 7.0     | How far ahead to look for a code occurrence           |
| `min_history_days`       | 1.0     | Minimum history required before a prediction time     |
| `seed`                   | 42      | Random seed for task sampling and prediction times    |
| `code_selection`         | random  | Task code strategy: `random` or `most_frequent`       |
| `anchor_strategy`        | uniform_lifetime | Prediction-time strategy: `uniform_lifetime` or `uniform_event` |
| `duration_distribution`  | fixed   | Occurrence-window strategy: `fixed`, `uniform`, or `log-uniform` |
| `min_duration_days`      | null    | Lower duration bound; required (and only used) when `duration_distribution != fixed` |
| `max_duration_days`      | null    | Upper duration bound; required (and only used) when `duration_distribution != fixed` |

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

Selects `num_tasks` codes from the train split (excluding `TIMELINE//`
tokens, which are synthetic, and `meds.birth_code`, which is structurally
always before the prediction window -- see below) per `code_selection`, and
creates binary prediction labels for each subject in every split:

- A single random **prediction time** is drawn per subject from
    `[first_event + min_history_days, last_event - anchor_horizon_days]`
    (`anchor_horizon_days` is the longest occurrence-window duration across
    all sampled tasks — see `duration_distribution` below), per
    `anchor_strategy`. Subjects whose timeline is too short for this window
    are excluded. This is independent of `code_selection` — prediction times
    are always random.
- For each task code, the label is `1.0` if the code appears within that
    task's own occurrence-window duration after the prediction time,
    otherwise `0.0`.

`meds.birth_code` ("MEDS_BIRTH") is always excluded from eligible codes: it's
the first event on a subject's timeline, and prediction time is always
sampled after `first_event + min_history_days`, so it can never fall inside a
prediction window — it would be guaranteed `pos_rate=0` on every split, for
any `code_selection`. `meds.death_code` is not excluded, since a death near
the end of a timeline can legitimately fall inside a window.

`code_selection` controls which of the remaining codes are chosen:

- `random` (default) — uniform, without replacement, from every eligible code
    in the train split. There is no positive-rate or count filtering, so a
    sampled code can turn out rare or degenerate (all-positive or
    all-negative) on a given split; that's a property of the sampled task,
    not something `generate_tasks` tries to correct for.
- `most_frequent` — the `num_tasks` codes with the highest **distinct-subject**
    count in the train split (ties broken by code string), not event-row
    count — a code measured repeatedly on a small subject subset (e.g. hourly
    ICU labs) can dominate row count while still being near-zero prevalence
    in the per-subject labels this module produces. Deterministic; `seed` is
    ignored for code selection (it still seeds prediction-time sampling).

Raises a `ValueError` only if the train split has fewer than `num_tasks`
eligible codes to select from.

`anchor_strategy` controls **how** the prediction time is drawn from the
per-subject window. Both strategies use the same window; they differ only in
the measure they sample against.

- `uniform_lifetime` (default) — uniform over *calendar time* in the window.
    This is the historical behaviour, kept as the default and unchanged
    byte-for-byte so that every previously generated label set stays
    reproducible.
- `uniform_event` — uniform over the subject's real **clinical events** inside
    the window (see `_clinical_events`, which drops both `meds.birth_code` and
    every `TIMELINE//` token). Subjects with no clinical event in the window
    are dropped, exactly as subjects with an empty window already were.

#### Why `uniform_lifetime` produces ~0.1% positive labels

`first_event` is effectively the subject's **birth**. MIMIC-IV subjects are
born decades before they ever touch the hospital, so the anchor window spans a
whole lifetime (~59 years on the measured shard) while the subject's actual
clinical activity is confined to roughly a year. A uniformly drawn calendar
timestamp therefore lands, almost always, in an empty stretch of that lifetime
where nothing is going to occur within `horizon_days` — so the label is `0.0`.

The obvious-looking fix — excluding `meds.birth_code` from the anchor bounds —
is a **no-op**. The MEDS-transforms pipeline (stage 1) inserts a
`TIMELINE//START` token at the *identical* timestamp as `MEDS_BIRTH` for 100%
of subjects, so dropping birth alone leaves the window's left edge exactly
where it was. This is only visible on the `intermediate/` dataset that task
generation actually reads, not on the raw `MEDS_cohort` input:

| cohort dir                                       | `TIMELINE//` rows | median(first non-birth event - birth) |
| ------------------------------------------------ | ----------------- | ------------------------------------- |
| `MEDS_cohort` (raw)                              | 0                 | 17,287 days                           |
| `intermediate/` (what `medrap-preprocess` reads) | 211,488           | **0.000 days**                        |

Both exclusions together are what make `uniform_event` work. Measured on a
999-subject shard, median in-window positive rate over the 8 most frequent
codes at a 7-day horizon:

| `anchor_strategy`                     | positive rate |
| ------------------------------------- | ------------- |
| `uniform_lifetime` (uniform over life) | 0.0010        |
| `uniform_event` (uniform over events)  | **0.3493**    |

— a **349x** increase. Note that this also affects which codes survive the
degenerate-code rejection in `fixed` mode: candidate codes are validated with
the same `anchor_strategy` used for the final labels, so a code accepted as
non-degenerate is non-degenerate under the anchors that actually ship.

`anchor_strategy` is recorded in `metadata.json`.

`duration_distribution` controls each task's occurrence-window length (how
many days after the prediction time to look for that task's code):

- `fixed` (default) — every task shares the single `horizon_days` window,
    exactly as before this option existed.
- `uniform` / `log-uniform` — each task's duration is sampled independently
    from `[min_duration_days, max_duration_days]` (both required in this
    mode; `horizon_days` is ignored). `log-uniform` draws
    `exp(uniform(log(min), log(max)))`, biasing toward shorter durations
    while still covering the full range; `uniform` draws linearly. This is a
    port of the duration-sampling formula from
    [EveryQuery](https://github.com/payalchandak/EveryQuery)'s
    `generate_tasks/sample_tasks.py` (`QueryDistribution.sample`) — only
    that formula, not the package: EveryQuery's task-label schema is a long
    table (one row per `(subject, query, duration)` with a three-valued
    censored label — `True`/`False`/`null` for "window extends past the
    subject's last recorded time"), fundamentally different from this
    module's wide per-subject `task_0..task_{N-1}` columns with a single
    shared `prediction_time` and hard exclusion (not censoring) of subjects
    whose window doesn't fit, so nothing else from its pipeline carries over
    directly.

When durations vary per task, the single shared prediction-time window uses
the *longest* sampled duration (`anchor_horizon_days = max(durations)`), so
every subject who gets a prediction time has enough trailing data for every
task's window — no task is ever dropped for a subject due to insufficient
history. `metadata.json` records `duration_distribution`,
`min_duration_days`, `max_duration_days`, and the resolved per-task
`durations` list (parallel to `codes`).

Output goes to `output_dir/tasks/` and contains one parquet per split plus
`code_index.json` (mapping task index → code string) and `metadata.json`.
