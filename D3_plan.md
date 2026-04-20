# D3 — LLM-as-a-Judge Plan

## Context

The boss wants an **LLM-as-a-judge patient-level retrieval-relevance evaluation** as a paper-grade result for the NeurIPS push. Existing retrieval diagnostics (doc_score splits, demographic heatmaps, PCA) say *what* is retrieved but not *whether it's relevant to this patient for this outcome*. This deliverable fills that gap.

Core claim the table will support: (1) retrieved documents beat random, (2) similarity scores have ranking meaning, (3) retrieval is patient-specific, not just label-generic.

Scope is a self-contained new module `src/medrap/llm_judge.py` plus a CLI script `scripts/run_llm_judge.py` that consumes an existing trained run's extraction artifacts (no retraining, no changes to the model).

## Framing — 4 comparison families

The original screenshot listed 5 conditions. We collapse the confusing ii.1/ii.2 pair into a single same-patient high-rank-vs-low-rank family. Each row in the final table names its **target document** so interpretation never flips.

| Family | Target doc | Other doc | What it tests |
|---|---|---|---|
| **F1** retrieved-vs-random | patient top-1 retrieved | uniform random doc from corpus | retrieval beats random |
| **F2** high-rank-vs-low-rank (same patient) | patient top-1 retrieved | patient top-`j` retrieved, j∈{2..k} | similarity ranking is meaningful |
| **F3** retrieved-vs-same-label-other-patient | patient top-1 retrieved | top-1 retrieved for a *different* patient with the **same** binary label | retrieval is patient-specific |
| **F4** retrieved-vs-opposite-label-other-patient | patient top-1 retrieved | top-1 retrieved for a *different* patient with the **opposite** binary label | sanity: label-generic distractors don't tie |

Headline metric: **target-document-preferred (%)** per family, with 95% patient-cluster-bootstrap CI.

## Decisions already made with the user

1. **Patient timeline = MEDS code strings annotated with `description` from `codes.parquet`, in sequence, no time, no numerics.** Same ordered code sequence the encoder consumes (only `batch.code`). `codes.parquet` is a true ontology-level dictionary — a **1-to-1 mapping** from `code` to `description` (verified on `/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/metadata/codes.parquet`: 45,105 rows, 45,105 unique codes, 0 null codes, 0 null descriptions; columns `[code, description, parent_codes, itemid, valueuom, possibly_cpt_code]`). Descriptions come from the MEDS ETL's upstream ontology (ICD-10-CM, LOINC, RxNorm), not from events, so the annotation is a legibility layer over the same fact the encoder sees — not extra signal. Render format: `"{code} — {description}"` when description is non-null, else `"{code}"` alone.

   **Invariant enforced in code** (new test `test_codes_parquet_is_one_to_one` at load time): `PatientTimelineRenderer.__init__` asserts `codes.parquet` has unique `code` values and raises with a loud, actionable error if it doesn't (so a future cohort that accidentally ships an event-level or ambiguous metadata file fails fast instead of silently leaking information).
2. **Patient is the sampling unit**, stratified by label (50/50). Default `n_patients=100`, `pairs_per_patient_per_family=1`, `k_families=4` → 400 API calls.
3. **Pairs are built once and frozen**, then the patient-cluster bootstrap resamples patients (not pairs) for CI.
4. **Within-patient averaging first** if multiple pairs per patient per family, then average across patients (patient-cluster estimator — see "Aggregation and inference" below).
5. **F3/F4 dedupe on** — if other patient's top-1 == anchor's top-1, redraw (≤10 tries) then skip; log the collision rate.
6. **F2 soft-skip** when checkpoint's `k=1`; emit a single warning, don't hard-fail.
7. **`task_description` is a required CLI arg** (`--task_description STR | --task_description_file PATH`). No auto-derivation in v1.
8. **Cost guardrail**: hard cap `max_total_calls_cap=1000`; `--dry_run` estimates tokens/$ and exits before any API hit.
9. **Determinism**: `temperature=0`, per-call `seed`, OpenAI **Structured Outputs** (`response_format={"type":"json_schema", ...}`). Default model `gpt-4o-mini`.
10. **Concurrency**: `concurrent.futures.ThreadPoolExecutor(max_workers=8)` — no async.
11. **TDD per AGENTS.md**: failing tests checked in first.

## Concrete example — what the judge sees

User prompt (illustrative MIMIC-style, built by `JudgePromptBuilder.build(pair, patient_timeline)`):

```
TASK: Predict in-ICU mortality within the first 24 hours of ICU admission.

PATIENT (sequence of MEDS codes up to prediction time):
MEDS_BIRTH
GENDER//M
RACE//WHITE
HOSPITAL_ADMISSION
DIAGNOSIS//ICD//10//C7800 — Secondary malignant neoplasm of unspecified site
DIAGNOSIS//ICD//10//I4891 — Unspecified atrial fibrillation
DIAGNOSIS//ICD//10//N179 — Acute kidney failure, unspecified
LAB//50811//g/dL — Hemoglobin
LAB//50931//mg/dL — Glucose [Mass/volume] in Serum or Plasma
ICU_ADMISSION
MEDICATION//ATENOLOL
MEDICATION//FUROSEMIDE
PROCEDURE//ICD//10//0T1807C — Bypass Bilateral Ureters to Ileum with Autologous Tissue Substitute
... (up to last 150 events)

DOCUMENT A:
<retrieval-corpus text for doc A, truncated at 4000 chars>

DOCUMENT B:
<retrieval-corpus text for doc B, truncated at 4000 chars>

Which document (A or B) is more relevant? Respond with
{"winner": "A"|"B"|"tie", "confidence": 0..1, "rationale": "<=1 sentence"}.
```

The system prompt is a fixed clinical-reviewer instruction + JSON schema. Target vs other placement in slot A vs B is dictated by `pair.target_position` (randomized at pair build time).

## Aggregation and inference (explicit)

### Within-patient averaging first — why and numeric example

Pairs from the same patient share a timeline and an anchor doc — they are **not independent**. If we treat every pair as an i.i.d. Bernoulli trial, we both (a) over-weight patients who happened to contribute more pairs and (b) understate variance.

Toy family (`F1`, 3 patients, 2 pairs each):

| patient | outcomes | patient mean |
|---|---|---|
| A | win, win | 1.0 |
| B | win, loss | 0.5 |
| C | loss, loss | 0.0 |

- **Naive pair-level mean**: (2+1+0)/6 = 0.500
- **Within-patient first, then across**: (1.0+0.5+0.0)/3 = 0.500

Identical when pair counts are balanced. Unbalanced case — A: 10 pairs all wins, B: 1 loss, C: 1 loss:

- **Naive**: 10/12 = 0.833
- **Within-patient first**: (1.0+0.0+0.0)/3 = 0.333

Within-patient first gives every patient equal weight regardless of how many pairs happen to be drawn from them — the correct estimator under cluster sampling. With the default `pairs_per_patient_per_family=1` this is a no-op, but the code path is kept to make raising the setting later safe.

### Patient-cluster bootstrap for SE + 95% CI

For each family:

1. Collapse raw verdicts to `per_patient[i] = mean(target_won for pairs of patient i in this family)`. N values where N = #patients in the family.
2. **Point estimate** = `mean(per_patient)`.
3. For r = 1..B (`B = 2000`): sample N patients with replacement from `per_patient`, compute `p_r = mean(resampled)`.
4. **Standard error** = `np.std(p_r, ddof=1)` over the B replicates.
5. **95% CI** = percentile method: `[np.quantile(p_r, 0.025), np.quantile(p_r, 0.975)]`.

Resampling at the **patient** level (not the pair level) respects within-patient correlation. Pairs are frozen at construction and carried with their patient into every replicate.

## Hypothetical results — what the `family_winrates` sheet will look like

The headline deliverable for the paper is a single four-row table. Below is a **hypothetical, illustrative** rendering of `<out_dir>/llm_judge_results.xlsx` sheet `family_winrates` for `n_patients=100`, `pairs_per_patient_per_family=1`, `k=5`, `B=2000`, judge=`gpt-4o-mini`. Numbers are plausible placeholders — not real outputs.

| family | description                                      | n_patients | n_pairs | n_invalid | target_preferred_rate | standard_error | ci_low | ci_high | ci_95               | bootstrap_mean |
|--------|--------------------------------------------------|-----------:|--------:|----------:|----------------------:|---------------:|-------:|--------:|---------------------|---------------:|
| F1     | retrieved vs. random                             |        100 |     100 |         1 |                 0.912 |          0.029 |  0.850 |   0.960 | [0.850, 0.960]      |          0.911 |
| F2     | same-patient top-1 vs. top-j (j ∈ {2..4})        |        100 |     100 |         2 |                 0.684 |          0.047 |  0.590 |   0.770 | [0.590, 0.770]      |          0.683 |
| F3     | retrieved vs. same-label other-patient top-1     |         98 |      98 |         0 |                 0.615 |          0.049 |  0.520 |   0.710 | [0.520, 0.710]      |          0.614 |
| F4     | retrieved vs. opposite-label other-patient top-1 |         97 |      97 |         1 |                 0.798 |          0.041 |  0.720 |   0.880 | [0.720, 0.880]      |          0.797 |

### Why aren't all the counts 100?

Anchor patients are sampled **once**, stratified by label, and the same 100-patient set is reused across all four families. But two things can still reduce a family's row count below 100:

| Drop mechanism                        | Where it happens            | Which families affected | Hypothetical effect above |
|---------------------------------------|-----------------------------|-------------------------|---------------------------|
| **Pair-construction drop** (dedupe)   | `build_pairs()`             | F3, F4 only             | F3: 100→98, F4: 100→97    |
| **F2 soft-skip** when checkpoint k=1  | `build_pairs()`             | F2 only                 | Not triggered here (k=5)  |
| **Judge-failure drop** (via n_invalid)| `summarize_winrates(...drop)` | All                   | See n_invalid column      |

**F3/F4 dedupe drops** (why F3=98, F4=97): F3 draws the "other" doc as the top-1 retrieval of a *different* same-label patient. If that doc happens to equal the anchor's own top-1 (common when the retriever concentrates mass on a small set of popular docs), we'd be asking the judge "is A more relevant than A?" — a degenerate pair. The builder retries up to 10 times from the same-label pool, then skips that patient for that family and logs a collision-rate diagnostic. F4 does the same against the opposite-label pool. F1 and F2 have no equivalent drop mechanism, so they stay at 100.

**Sanity bound**: if F3 or F4 drops ≫ 5/100 in a real run, the retriever is collapsing — that's a model-health signal, not a statistical nuisance, and the plan's CLI prints the collision rate to stderr for exactly this reason.

**`n_patients` vs `n_pairs`**: identical under the default `pairs_per_patient_per_family=1` (one pair per patient means the two counts always match). They diverge only if the user raises that flag — e.g., at `pairs_per_patient_per_family=3`, a family with 98 patients has 294 pairs. Both columns are kept so the table stays honest when the setting is raised.

### What is n_invalid for?

`n_invalid` counts verdicts the judge could not resolve into a clean `A`/`B` winner. Three sources funnel into it:

1. **Judge said `"tie"`** — the structured-output schema allows it.
2. **JSON parse failure** — Structured Outputs makes this rare but not impossible (e.g., content-filter refusal returns no JSON at all).
3. **API error swallowed by `OpenAIJudge`** — per plan, the judge never raises; it returns `winner_position="invalid"` and preserves `raw_response` for forensic review. A single transient 5xx shouldn't crash a 400-call run.

We surface the count for **three operational reasons**:

- **Audit trail**: the reviewer can see, in one glance, what fraction of the numerator/denominator was dropped. A rate like "0.912 over 99/100" is more credible than "0.912 over 100/100" that silently dropped 12 ties.
- **Sensitivity toggle**: `summarize_winrates(..., invalid_policy="count_as_loss")` re-runs the summary counting invalids as losses. If the point estimate shifts materially between `"drop"` and `"count_as_loss"`, the result is fragile and we flag it in the paper; if it doesn't shift, `n_invalid` is small enough to ignore and the default `"drop"` policy is fine.
- **Operational health signal**: a sudden spike in `n_invalid` (e.g., 30/100) means the judge model was swapped, the prompt is malformed, or the API is flaking. Catching that before reading the rates prevents chasing a ghost scientific finding.

In the hypothetical above, `n_invalid ≤ 2` across all families — the results are not policy-sensitive, the rates are trustworthy, and the table is paper-ready.

**How to read each column (recap)**
- `target_preferred_rate`: point estimate. The across-patient mean of each patient's within-family win-rate (so a patient with two pairs (win, loss) contributes 0.5, not two rows).
- `standard_error`: `np.std(p_r, ddof=1)` over B=2000 patient-cluster bootstrap replicates.
- `ci_low` / `ci_high`: percentile-method 95% CI (2.5th / 97.5th quantile of bootstrap distribution). `ci_95` is a formatted merged cell `"[low, high]"` for paper-table legibility.
- `bootstrap_mean`: mean of the B bootstrap replicates — should track `target_preferred_rate` closely; a large gap is a red flag for severe imbalance or a coding bug.

**Narrative read of the hypothetical row pattern** (this is the story the boss wants the table to tell):
- **F1 ≈ 0.91** — retrieval beats random handily; if this drops below ~0.7 the retriever is broken.
- **F4 ≈ 0.80** — opposite-label distractors are usually easy to beat; sanity-check, not the paper's headline.
- **F2 ≈ 0.68** — similarity ranking is meaningful (top-1 beats top-3), but not dominant.
- **F3 ≈ 0.62** — this is the **key scientific result**. Even against a document retrieved for a *different* patient with the *same* label, the judge still prefers this patient's own top-1 more often than chance. That's the evidence that retrieval is patient-specific rather than label-generic.

**Sibling sheets in the same workbook** (names + one-line contents):
- `family_winrates` — the table above (headline, bold header, `ci_95` rendered as merged text column for easy paper copy-paste).
- `per_patient_results` — one row per patient, all four families side-by-side. Columns include `anchor_subject_id, anchor_label, predicted_prob, prediction_correct, gender, race, age_years_at_prediction, age_bin, patient_timeline, target_doc_id, target_doc_title, target_doc_score, target_doc_text_preview, F1_target_won, F1_other_doc_title, F1_other_doc_score, …` through `F4_*`. This is the human-review artifact when the rates look wrong.
- `pairs_verdicts` — raw per-pair judge output (one row per API call). Long-form diagnostic.
- `human_validation` — a randomized, anonymized subset (default n=50, proportional across families) with `target_doc_id`/`target_position`/`target_won` stripped out and blank columns `human_winner, human_confidence, human_notes` for the rater to fill in. Row order shuffled with a separate seed so the rater can't infer target position from sheet order.

CSV twins of all four sheets are written alongside the `.xlsx` for diff-friendly version control.

## Module layout — `src/medrap/llm_judge.py`

### Dataclasses
- `JudgePair(pair_id, family, anchor_row_idx, anchor_subject_id, anchor_label, target_doc_id, other_doc_id, target_position, other_source_row_idx, other_source_subject_id, other_rank, rng_seed)` — frozen, slots.
- `Verdict(pair_id, winner_position, target_won, confidence, rationale, raw_response, model, prompt_tokens, completion_tokens)` — `winner_position` ∈ {"A","B","tie","invalid"}; `target_won` is `None` on tie/invalid.

### Protocols + implementations
- `Judge` (Protocol, runtime_checkable) — method `judge(system_prompt, user_prompt, seed) -> Verdict`.
- `OpenAIJudge(model="gpt-4o-mini", client=None, temperature=0.0)` — lazy-imports `openai`, reads `OPENAI_API_KEY`. Never raises on parser failure (returns `winner_position="invalid"`, preserves `raw_response`).
- `FakeJudge(rule)` — test-only. `FakeJudge.always_A()`, `FakeJudge.always_target(pair_lookup)`, `FakeJudge.flaky(seed)`.

### Patient timeline
```python
class PatientTimelineRenderer:
    def __init__(self, *, codes_parquet: Path, max_events: int = 150,
                 include_description: bool = True): ...
    def render(self, subject_id: int, prediction_time: datetime, meds_cohort_dir: Path) -> str: ...
```
- Loads `codes.parquet` once into a `code_string → description` dict (via `polars`; `description` may be null for some codes). Reads `code, description` columns only.
- For each call: lazy-scan `{meds_cohort_dir}/data/*/*.parquet`, filter `subject_id == X AND time <= prediction_time`, sort by `time`, take **last 150 events**. For each event: emit `"{code} — {description}"` when description is non-null, else the bare `"{code}"`. No absolute time, no numeric, no time_delta.
- `include_description=False` kill-switch for cohorts where `codes.parquet` lacks a description column (we degrade to code-only rendering with a warning).
- Cache per subject via `functools.lru_cache`.

### Prompt builder
```python
class JudgePromptBuilder:
    def __init__(self, *, task_description: str, timeline_renderer: PatientTimelineRenderer,
                 retrieval_ds, doc_text_column="doc_text", doc_id_to_row=None, max_doc_chars=4000): ...
    def build(self, pair: JudgePair, patient_timeline: str) -> tuple[str, str]: ...
```
- **System prompt** (fixed): "You are a clinical research assistant judging which of two reference documents is more relevant for predicting a specific clinical outcome for a specific patient. Base your decision on the patient's clinical presentation and the documents' content — not on writing style or length. Respond with a JSON object matching the provided schema."
- **User prompt** (fixed layout, identical across families):
  - `TASK: {task_description}`
  - `PATIENT (sequence of MEDS codes up to prediction time):\n{timeline}`
  - `DOCUMENT A:\n{doc_a_text[:4000]}`
  - `DOCUMENT B:\n{doc_b_text[:4000]}`
  - `Which document (A or B) is more relevant? Respond with {"winner": "A"|"B"|"tie", "confidence": 0..1, "rationale": "<=1 sentence"}.`
- Target placement in slot A vs B is dictated by `pair.target_position`.

### Pair construction
```python
def build_pairs(*, artifacts: dict, val_schema: pl.DataFrame, labels: np.ndarray,
                families: Sequence[str] = ("F1","F2","F3","F4"),
                n_patients: int = 100, pairs_per_patient_per_family: int = 1,
                corpus_size: int, k: int, seed: int = 42,
                dedupe_identical_docs: bool = True,
                skip_missing_families: bool = True) -> list[JudgePair]: ...
```
- Stratified sample: `n_patients//2` from positives + `n_patients//2` from negatives (clamp if imbalanced). Same anchor set across families.
- **F1**: target = `doc_ids[i,0,0]`; other = `rng.integers(corpus_size)`; redraw on collision.
- **F2**: requires `k≥2`. target = `doc_ids[i,0,0]`; other = `doc_ids[i,0,j]`, j∈{1..k-1}. Skip silently if `k==1`.
- **F3**: pool = rows with same label as anchor, minus anchor itself. Draw `other_row`; `other_doc = doc_ids[other_row,0,0]`. Redraw on `target==other`.
- **F4**: same as F3, opposite-label pool.
- A/B randomization: per-pair `rng.random() < 0.5`; store `rng_seed = int(rng.integers(1<<30))` for deterministic OpenAI calls.

### Runner + aggregation
```python
def run_judge(pairs, *, judge: Judge, prompt_builder: JudgePromptBuilder,
              max_workers: int = 8, progress: bool = True) -> pl.DataFrame: ...
```
Output columns: `pair_id, family, anchor_subject_id, anchor_row_idx, anchor_label, target_doc_id, other_doc_id, target_position, other_rank, other_source_subject_id, winner_position, target_won, confidence, rationale, model, prompt_tokens, completion_tokens, raw_response`.

```python
def summarize_winrates(df, *, n_bootstrap: int = 2000, seed: int = 42,
                       ci_level: float = 0.95,
                       invalid_policy: Literal["drop","count_as_loss"] = "drop") -> pl.DataFrame: ...
```
Algorithm: (1) drop or downgrade invalid rows per `invalid_policy`; (2) `per_patient = df.group_by(["family","anchor_subject_id"]).agg(mean("target_won"))`; (3) point estimate = `per_patient.group_by("family").agg(mean)`; (4) patient-cluster bootstrap — `n_bootstrap` resamples of the per-patient means within each family, percentile CI + SE.
Output columns: `family, n_patients, n_pairs, n_invalid, target_preferred_rate, standard_error, ci_low, ci_high, bootstrap_mean`.

```python
def build_per_patient_rollup(pairs: Sequence[JudgePair], verdicts: pl.DataFrame,
                             *,
                             logits: np.ndarray, targets: np.ndarray,
                             artifacts: dict,
                             timeline_renderer: PatientTimelineRenderer,
                             val_schema: pl.DataFrame,
                             demographics: pl.DataFrame,
                             retrieval_ds, doc_id_to_row: dict,
                             doc_text_column: str = "doc_text",
                             doc_metadata_columns: Sequence[str] = ("title",),
                             doc_text_preview_chars: int = 300) -> pl.DataFrame: ...
```
**One row per sampled patient.** Columns (intentionally wide — the workbook is a human-review artifact):

**Patient-level** (from `val_schema` + `load_subject_demographics` + artifacts):
- `anchor_subject_id`, `prediction_time`, `anchor_label`, `predicted_label` (argmax of `artifacts["logits"]`), `prediction_correct`, `predicted_prob` (softmax of positive class)
- `gender`, `race`, `age_years_at_prediction` (computed from `birth_time` and `prediction_time`), `age_bin`
- `patient_timeline` (rendered string — last N MEDS code strings)

**Target document** (same across all families for a given patient — their top-1 retrieved doc):
- `target_doc_id`, `target_doc_title` (from `retrieval_ds[row]["title"]` when present), `target_doc_score` (`artifacts["doc_scores"][i,0,0]`), `target_doc_text_preview` (first `doc_text_preview_chars` chars of `doc_text`)
- Additional `target_doc_{col}` columns for every name in `doc_metadata_columns` that actually exists on the retrieval ds (e.g. `title`, `source`, `category` — skipped silently if absent)

**Per-family** — for each `F` in {F1, F2, F3, F4}:
- Outcome: `{F}_target_won`, `{F}_winner_position`, `{F}_confidence`, `{F}_rationale`
- Other doc: `{F}_other_doc_id`, `{F}_other_doc_title`, `{F}_other_doc_score` (for F2 this is the lower-rank score; for F3/F4 it's the *other patient's* top-1 score), `{F}_other_doc_text_preview`
- F2 only: `F2_other_rank` (the `j` index, 1-indexed)
- F3/F4 only: `F3_other_source_subject_id` / `F4_other_source_subject_id` + that patient's `_gender`, `_race`, `_anchor_label` (so the rater can sanity-check "did the same/opposite-label patient come from a plausibly similar cohort?")

Rows with multiple pairs per family collapse via: `target_won` → mean; `confidence` → mean; `rationale`, `winner_position`, `other_doc_*` → taken from the first pair with a `F_n_pairs` companion column when >1.

```python
def build_human_validation_subset(df, *, n: int = 50, seed: int = 42,
                                  retrieval_ds, doc_id_to_row: dict,
                                  doc_metadata_columns: Sequence[str] = ("title",)) -> pl.DataFrame: ...
```
Proportional allocation across families (min 5 each). **Drop** (these would leak the answer): `target_doc_id, target_position, target_won, winner_position, other_source_subject_id, other_rank, model, raw_response, confidence, rationale`. **Keep**: `pair_id, family, anchor_subject_id, anchor_label`, patient `gender`/`race`/`age_bin` (patient-level, not answer-revealing), plus re-materialized `patient_timeline, doc_a_text, doc_b_text, doc_a_title, doc_b_title` (titles are displayed because the rater is going to read the docs anyway — anonymization applies only to which of A/B was the target). **Add blank columns** for the rater: `human_winner, human_confidence (1–5 Likert), human_notes`. Row order shuffled with a separate seed.

```python
def write_results_workbook(path: Path, *, family_winrates: pl.DataFrame,
                           per_patient: pl.DataFrame, pairs_verdicts: pl.DataFrame,
                           human_validation: pl.DataFrame) -> None: ...
```
Writes a single multi-sheet `.xlsx` via `polars.DataFrame.write_excel(..., worksheet=...)` (uses `xlsxwriter`). Sheets: `family_winrates` (headline, formatted — bold header, CI as a merged `"[low, high]"` column alongside the numeric columns), `per_patient_results`, `pairs_verdicts`, `human_validation`. CSV twins are written alongside for diffability.

## CLI — `scripts/run_llm_judge.py`

Mirrors [scripts/run_demographic_heatmap.py](scripts/run_demographic_heatmap.py) argparse + artifact-reuse pattern.

```
--run_dir PATH                       (required)
--retrieval_db PATH                  (required)
--meds_cohort PATH                   (required)
--task_description STR | --task_description_file PATH   (required, mutex)
--families F1,F2,F3,F4                (default all)
--n_patients INT                      (default 100)
--pairs_per_patient_per_family INT    (default 1)
--model STR                           (default gpt-4o-mini)
--seed INT                            (default 42)
--n_bootstrap INT                     (default 2000)
--max_workers INT                     (default 8)
--out_dir PATH                        (default <run_dir>/llm_judge)
--human_validation_n INT              (default 50)
--max_total_calls_cap INT             (default 1000, hard)
--timeline_max_events INT             (default 150)
--doc_max_chars INT                   (default 4000)
--dry_run                             (flag)
--overwrite                           (flag)
```

Pipeline:
1. `cfg = OmegaConf.load(run_dir/"config.yaml")`; resolve `tensorized_cohort_dir` from `cfg.training.datamodule.config.tensorized_cohort_dir`; derive `codes_parquet = tensorized_cohort/"metadata"/"codes.parquet"`.
2. Reuse-or-extract `extraction/extraction_artifacts.pt` via `extract_artifacts()` (same cache block as `run_demographic_heatmap.py`).
3. `val_schema = extract_val_schema(datamodule)`; assert row-count match with `artifacts["doc_ids"].shape[0]`; fail fast on mismatch.
4. `labels = artifacts["targets"].int().numpy()`; assert both classes present.
5. Load retrieval corpus (`load_from_disk(args.retrieval_db)`); compute `corpus_size` and `doc_id_to_row` (reuse `demographic_analysis._build_doc_id_to_row_map`). Also record which of `("title", "source", "category", ...)` are actually present as columns on the ds — this is what `doc_metadata_columns` picks up.
6. Load `demographics = load_subject_demographics(meds_cohort, val_schema["subject_id"].unique())` (reuses [demographic_analysis.py:392](src/medrap/demographic_analysis.py)).
7. `pairs = build_pairs(...)`.
8. **Cost estimate** (always printed): use `tiktoken.encoding_for_model(args.model)` (or a coarse `len(text)//4` fallback) × per-pair token count × price constants for `gpt-4o-mini` (document as "verify before paper submission"). Assert `len(pairs) <= max_total_calls_cap`. If `--dry_run`: exit 0.
9. `OpenAIJudge(model=args.model)`; `results_df = run_judge(...)`.
10. `summary_df = summarize_winrates(...)`; `per_patient_df = build_per_patient_rollup(pairs, results_df, logits=artifacts["logits"].numpy(), targets=labels, artifacts=artifacts, timeline_renderer=..., val_schema=val_schema, demographics=demographics, retrieval_ds=retrieval_ds, doc_id_to_row=doc_id_to_row)`; `human_df = build_human_validation_subset(results_df, retrieval_ds=retrieval_ds, doc_id_to_row=doc_id_to_row, ...)`.
11. Write **CSV twins**: `<out_dir>/{pairs_verdicts.csv, per_patient_results.csv, family_winrates.csv, human_validation.csv, run_config.json}`.
12. Write **multi-sheet workbook**: `<out_dir>/llm_judge_results.xlsx` (sheets: `family_winrates`, `per_patient_results`, `pairs_verdicts`, `human_validation`).

## Test layout — `tests/test_llm_judge.py`

All tests use `FakeJudge` + in-memory fixtures (tiny polars frames + `Dataset.from_dict`). No live OpenAI.

1. `test_build_pairs_f1_uses_top1_and_random_other`
2. `test_build_pairs_f2_samples_lower_rank_same_patient`
3. `test_build_pairs_f2_skipped_silently_when_k_equals_one`
4. `test_build_pairs_f3_other_patient_has_same_label`
5. `test_build_pairs_f4_other_patient_has_opposite_label`
6. `test_build_pairs_deduplicates_identical_target_and_other_doc`
7. `test_build_pairs_is_deterministic_with_seed`
8. `test_build_pairs_ab_position_roughly_balanced` (within [0.4, 0.6] over 1000 pairs)
9. `test_stratified_patient_sample_respects_label_balance`
10. `test_run_judge_roundtrips_fake_verdicts_to_dataframe`
11. `test_run_judge_marks_target_won_based_on_target_position`
12. `test_prompt_builder_places_target_according_to_position`
13. `test_prompt_builder_truncates_doc_texts_to_max_chars`
14. `test_timeline_renderer_returns_last_n_events_before_prediction_time`
15. `test_timeline_renderer_annotates_codes_with_descriptions_from_parquet` (fixture codes.parquet with `code` + `description`; assert rendered line matches `"{code} — {description}"` for codes with a description and `"{code}"` alone for null-description codes)
15a. `test_codes_parquet_must_be_one_to_one` (inject a codes.parquet fixture with a duplicated `code` row; `PatientTimelineRenderer.__init__` raises with a clear message naming the offending code)
16. `test_within_patient_averaging_before_aggregation` (patient with (win,loss) contributes 0.5, not two rows)
17. `test_summarize_winrates_point_estimate_matches_hand_computed`
18. `test_summarize_winrates_bootstrap_ci_covers_point_estimate`
19. `test_summarize_winrates_drop_vs_count_as_loss_policy`
20. `test_human_validation_subset_strips_target_and_position_columns`
21. `test_human_validation_subset_preserves_pair_id_for_rejoining`
22. `test_openai_judge_never_raises_on_parser_failure` (inject mock client returning malformed JSON)
23. `test_fake_judge_implements_judge_protocol`
24. `test_summarize_winrates_standard_error_matches_bootstrap_std` (synthetic fixture with known per-patient values; assert returned `standard_error` == `np.std(bootstrap_replicates, ddof=1)`)
25. `test_build_per_patient_rollup_one_row_per_patient_with_all_families` (4 families × N patients → N rows with F1_…F4_ columns present)
26. `test_build_per_patient_rollup_merges_multiple_pairs_into_mean_target_won` (within-patient aggregation path)
27. `test_write_results_workbook_produces_all_four_sheets` (open with `openpyxl`, assert sheet names + header row)
28. `test_per_patient_rollup_joins_doc_metadata_columns` (retrieval ds has `title`; assert `target_doc_title`, `F1_other_doc_title` columns populated from the right doc rows)
29. `test_per_patient_rollup_joins_demographics` (stub `load_subject_demographics` return; assert `gender`, `race`, `age_years_at_prediction` columns aligned to correct subject_ids)
30. `test_per_patient_rollup_skips_missing_doc_metadata_columns` (retrieval ds without `title`; call with `doc_metadata_columns=("title",)` and assert no error, column just absent)
31. `test_human_validation_subset_keeps_doc_titles_and_timeline_but_not_target_position` (explicit anonymization contract: titles visible, target_position/target_won absent)

## `pyproject.toml` changes

Add optional extra (mirroring the existing `[prep]`/`[wandb]`/`[viz]` pattern at [pyproject.toml:34-44](pyproject.toml)):
```toml
[project.optional-dependencies]
llm_judge = [
  "openai>=1.0.0",
  "tiktoken>=0.7.0",
  "xlsxwriter>=3.2",   # used by polars.DataFrame.write_excel for the results workbook
  "openpyxl>=3.1",     # used by tests to verify the produced workbook
]
```
`openai` and `xlsxwriter` imports are lazy (inside `OpenAIJudge.__init__` / `write_results_workbook`) so doctest collection still works without the extra installed. All doctest examples use `FakeJudge` and produce CSV-only output. No `--ignore` needed.

## Critical files

- [D3_plan.md](D3_plan.md) *(new, repo root — copy of this plan; first step on exit from plan mode)*
- [src/medrap/llm_judge.py](src/medrap/llm_judge.py) *(new)*
- [scripts/run_llm_judge.py](scripts/run_llm_judge.py) *(new)*
- [tests/test_llm_judge.py](tests/test_llm_judge.py) *(new)*
- [pyproject.toml](pyproject.toml) — add `[llm_judge]` optional-dependencies
- [src/medrap/demographic_analysis.py](src/medrap/demographic_analysis.py) — reuse `extract_val_schema`, `_build_doc_id_to_row_map`, lazy-scan pattern (no edits)
- [src/medrap/preparation.py](src/medrap/preparation.py) — reuse `OrderedFieldDocumentRenderer` as a reference pattern (no edits)
- [scripts/run_demographic_heatmap.py](scripts/run_demographic_heatmap.py) — mirror argparse + artifact-reuse block (no edits)

## Commit sequencing (TDD)

0. **Commit 0** — save this plan to `/groups/mm6677_gp/zzw2102/MedRAP/D3_plan.md` (alongside existing [overall_plan.md](overall_plan.md)). This is the very first action on exiting plan mode so the plan travels with the repo. The plan-mode file (`/users/zzw2102/.claude/plans/i-was-given-the-compressed-wind.md`) remains the editing source during planning; `D3_plan.md` is the repo-tracked copy for the PR description and review.
1. **Commit 1** — checked-in failing tests: `tests/test_llm_judge.py` (all 31 tests), `pyproject.toml` optional extra, skeleton `src/medrap/llm_judge.py` with dataclasses + Protocol + stubbed functions returning `NotImplementedError`. Check in with user at this point per AGENTS.md.
2. **Commit 2** — implementation for T1–T15 (pair construction, prompt builder, timeline renderer, codes.parquet 1-to-1 invariant).
3. **Commit 3** — implementation for T16–T27 (aggregation, bootstrap, anonymization, OpenAI error handling, Excel writer).
4. **Commit 4** — implementation for T28–T31 (per-patient rollup with demographics + doc metadata join) + `scripts/run_llm_judge.py` + documentation of a budgeted-dry-run smoke test in the PR description. No live API in CI.

## Verification

**Unit tests** (no API):
```
pytest tests/test_llm_judge.py -v
pytest src/medrap/llm_judge.py -v      # doctests
```
All 23 tests green, coverage ≥ existing threshold.

**Dry-run** (no API, validates cost + plumbing):
```
python scripts/run_llm_judge.py \
  --run_dir outputs/mimic_run_retrieval_only \
  --retrieval_db data/retrieval_db \
  --meds_cohort /groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort \
  --task_description "Predict in-ICU mortality within the first 24 hours of ICU admission." \
  --n_patients 100 --dry_run
```
Expected: prints `Pairs: 400 | Est total tokens ~XM | Est cost ~$0.10`, exits 0.

**Budgeted live run** (n=20 patients, ~80 API calls, ~$0.02):
```
OPENAI_API_KEY=... python scripts/run_llm_judge.py \
  --run_dir outputs/mimic_run_retrieval_only \
  --retrieval_db data/retrieval_db \
  --meds_cohort /groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort \
  --task_description "Predict in-ICU mortality within the first 24 hours of ICU admission." \
  --n_patients 20
```
Writes to `<run_dir>/llm_judge/`:
- `llm_judge_results.xlsx` (4 sheets: `family_winrates`, `per_patient_results`, `pairs_verdicts`, `human_validation`)
- CSV twins: `pairs_verdicts.csv`, `per_patient_results.csv`, `family_winrates.csv`, `human_validation.csv`
- `run_config.json`

Rough sanity check on outputs:
- F1 target_preferred_rate ≫ 0.5 (retrieval should beat random by a lot).
- F4 ≳ 0.5 (opposite-label distractors should be easy to beat).
- F3 near 0.5 or just above (the interesting paper signal).
- F2 ≳ 0.5 if similarity ranking is meaningful.
- Collision-rate diagnostic for F3/F4 printed to stderr.

## Out of scope for this plan (deferred)

- Batch API variant (`OpenAIBatchJudge`) — 50% cost, 24h latency; add post-v1 once design stabilizes.
- Ontology-based code-description join — superseded: `codes.parquet` already carries `description` as a 1-to-1 mapping.
- Sampling stratified by prediction-correctness (not just label) — add in a v2 flag if the initial run shows flat rates.
- D6/D7 tasks and synthetic experiments — separate deliverables.

## Appendix — full example prompts

### Fixed system prompt (every call)

```
You are a clinical research assistant judging which of two reference documents
is more relevant for predicting a specific clinical outcome for a specific
patient. You will receive (1) a one-sentence task description, (2) a compact
event-by-event timeline for the patient up to the prediction time, and (3) two
candidate documents labeled DOCUMENT A and DOCUMENT B. Decide which document
would be more useful to a clinician reasoning about the outcome for THIS
patient. Base your decision only on the patient's clinical presentation and
the documents' content — not on writing style or length. Respond with a single
JSON object matching the provided schema:
  {"winner": "A" | "B" | "tie", "confidence": 0..1, "rationale": "<=1 sentence"}
```

The JSON schema is enforced via OpenAI Structured Outputs (`response_format={"type": "json_schema", ...}`).

### F1 — retrieved vs. random (anchor label = 1, target in slot A)

```
TASK: Predict in-ICU mortality within the first 24 hours of ICU admission.

PATIENT (sequence of MEDS codes up to prediction time):
MEDS_BIRTH
GENDER//M
RACE//WHITE
HOSPITAL_ADMISSION
DIAGNOSIS//ICD//10//I509 — Heart failure, unspecified
DIAGNOSIS//ICD//10//N179 — Acute kidney failure, unspecified
DIAGNOSIS//ICD//10//J9601 — Acute respiratory failure with hypoxia
LAB//50811//g/dL — Hemoglobin
LAB//50931//mg/dL — Glucose [Mass/volume] in Serum or Plasma
LAB//51006//mg/dL — Urea Nitrogen [Mass/volume] in Serum or Plasma
MEDICATION//NOREPINEPHRINE
MEDICATION//FUROSEMIDE
ICU_ADMISSION
PROCEDURE//ICD//10//5A1935Z — Respiratory Ventilation, Greater than 96 Consecutive Hours

DOCUMENT A:
Acute respiratory failure is a life-threatening condition characterized by
inadequate gas exchange. In critically ill patients, predictors of short-term
mortality include the need for mechanical ventilation, vasopressor support,
acute kidney injury, and advanced heart failure ...

DOCUMENT B:
Benign skin lesions such as seborrheic keratoses are common in older adults.
Diagnosis is typically clinical; biopsy is reserved for atypical features ...

Which document (A or B) is more relevant? Respond with
{"winner":"A"|"B"|"tie","confidence":0..1,"rationale":"<=1 sentence"}.
```

Expected judge output (illustrative): `{"winner":"A","confidence":0.97,"rationale":"Doc A covers ICU mortality predictors matching this patient's vent/vasopressor/AKI/HF profile; Doc B is unrelated dermatology."}` → `target_won=True`.

### F2 — same patient, top-1 vs. top-j (anchor label = 0, target in slot B)

```
TASK: Predict in-ICU mortality within the first 24 hours of ICU admission.

PATIENT (sequence of MEDS codes up to prediction time):
MEDS_BIRTH
GENDER//F
RACE//BLACK/AFRICAN AMERICAN
HOSPITAL_ADMISSION
DIAGNOSIS//ICD//10//E1165 — Type 2 diabetes mellitus with hyperglycemia
DIAGNOSIS//ICD//10//I10 — Essential (primary) hypertension
LAB//50931//mg/dL — Glucose [Mass/volume] in Serum or Plasma
LAB//50852//% — Hemoglobin A1c/Hemoglobin.total in Blood
MEDICATION//METFORMIN
MEDICATION//LISINOPRIL
ICU_ADMISSION

DOCUMENT A:
Venous thromboembolism prophylaxis in hospitalized patients reduces the
incidence of deep vein thrombosis and pulmonary embolism. Risk stratification
tools (Padua, IMPROVE) guide pharmacologic vs. mechanical prophylaxis ...

DOCUMENT B:
Diabetic ketoacidosis and hyperosmolar hyperglycemic state are the two most-
serious acute metabolic complications of diabetes mellitus. ICU mortality is
driven largely by precipitating illness, age, and altered mental status ...

Which document (A or B) is more relevant? ...
```

Doc B is the patient's top-1 retrieval; Doc A is the same patient's top-3. Tests whether similarity ranking is meaningful.

### F3 — retrieved vs. same-label-different-patient (anchor label = 1, target in slot A)

```
TASK: Predict in-ICU mortality within the first 24 hours of ICU admission.

PATIENT (sequence of MEDS codes up to prediction time):
MEDS_BIRTH
GENDER//M
RACE//HISPANIC/LATINO
HOSPITAL_ADMISSION
DIAGNOSIS//ICD//10//A419 — Sepsis, unspecified organism
DIAGNOSIS//ICD//10//R6521 — Severe sepsis with septic shock
LAB//51301//K/uL — Leukocytes [#/volume] in Blood by Automated count
LAB//50813//mmol/L — Lactate [Moles/volume] in Blood
LAB//50971//mEq/L — Potassium [Moles/volume] in Serum or Plasma
MEDICATION//VANCOMYCIN
MEDICATION//PIPERACILLIN/TAZOBACTAM
MEDICATION//NOREPINEPHRINE
ICU_ADMISSION

DOCUMENT A:
Septic shock carries a 30–50% short-term mortality despite aggressive
resuscitation. Early broad-spectrum antibiotics, source control, and
norepinephrine-first vasopressor strategy are cornerstones of care ...

DOCUMENT B:
Acute coronary syndromes encompass unstable angina and MI with or without
ST elevation. Troponin kinetics, ECG changes, and early coronary angiography
drive management ...

Which document (A or B) is more relevant? ...
```

Both docs match the anchor label (high-mortality ICU topics). Doc A was retrieved for *this* septic patient; Doc B was retrieved for a different label-1 patient (ACS). Whether the judge still prefers Doc A is the "retrieval is patient-specific vs. just label-generic" signal.

### F4 — retrieved vs. opposite-label-different-patient (anchor label = 0, target in slot B)

```
TASK: Predict in-ICU mortality within the first 24 hours of ICU admission.

PATIENT (sequence of MEDS codes up to prediction time):
MEDS_BIRTH
GENDER//F
RACE//WHITE
HOSPITAL_ADMISSION
DIAGNOSIS//ICD//10//K802 — Calculus of gallbladder without cholecystitis
DIAGNOSIS//ICD//10//R109 — Unspecified abdominal pain
LAB//50878//U/L — Aspartate aminotransferase [Enzymatic activity/volume] in Serum
LAB//51464//mg/dL — Bilirubin.total [Mass/volume] in Serum or Plasma
MEDICATION//ONDANSETRON
PROCEDURE//ICD//10//0FB44ZZ — Excision of Gallbladder, Percutaneous Endoscopic Approach
ICU_ADMISSION

DOCUMENT A:
Septic shock carries a 30–50% short-term mortality despite aggressive
resuscitation ...

DOCUMENT B:
Elective laparoscopic cholecystectomy is the treatment of choice for
symptomatic cholelithiasis. ICU-level care after uncomplicated operation is
uncommon and short; inpatient mortality is well under 1% ...

Which document (A or B) is more relevant? ...
```

Anchor has a benign surgical course (label = 0). Doc B was retrieved for *this* patient; Doc A came from a label-1 sepsis patient. If the judge prefers Doc B, that's evidence retrieval tracks the patient's actual clinical picture even against a generic "high-mortality" distractor.
