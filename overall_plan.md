# MedRAP — Retrieval Assessment & NeurIPS Ablations Plan

## Context

The boss gave Zach three screenshots of improvements needed to bring MedRAP to paper/NeurIPS level. Two themes:

1. **Document-Patient Alignment Assessment** — understand *what* is being retrieved and *whether* it is relevant to the patient and the prediction task. This is *insight work*, not a proxy for real performance.
2. **Retrieval-Value Ablations** — show *when and why* retrieval adds value, with a minimum viable experimental design for the NeurIPS paper.

Branch `zach/retrieval_assessment_codes` is clean (nothing diverged from `main` yet). An OpenAI API key is available for LLM-as-a-judge work.

A lot of scaffolding already exists:
- [src/medrap/extraction.py](src/medrap/extraction.py) — `extract_artifacts()` produces a `.pt` file with `query_embeddings`, `doc_ids`, `doc_scores`, `doc_key_embeddings`, `per_doc_logits`, `targets`, `logits`.
- [src/medrap/demographic_analysis.py](src/medrap/demographic_analysis.py) — pluggable `DocKeywordProvider` (title, LDA, static), patient demographic joins (age/race/gender), softmax-weighted keyword × demographic heatmap tables.
- [src/medrap/retrieval_logging.py](src/medrap/retrieval_logging.py) — `retrieval_diagnostic_scalars()` already logs `doc_score_mean/std`, positive-vs-negative doc_score splits, and `unique_docs_per_batch` via `self.log()` in training/validation steps.
- [scripts/run_demographic_heatmap.py](scripts/run_demographic_heatmap.py) + [scripts/extract_and_visualize.py](scripts/extract_and_visualize.py) — invokable with `--run_dir`, already do PCA plots of key/query embeddings.
- Hydra retriever group at [src/medrap/conf/retriever/](src/medrap/conf/retriever/) — eval can be run with a *different* retriever than training via `retriever=<name>` override; the checkpoint load path ([cli.py:_run_eval](src/medrap/cli.py)) instantiates the retriever fresh from the config.

## Expected Outputs (from screenshots), ranked

| # | Deliverable | Source | Tier |
|---|---|---|---|
| D1 | Random-document ablation at eval (headline paper result) | 2.b.i.1 + "Immediate action item" 2.b.iii | **1 — urgent** |
| D2 | Patient × Document viz with **label + clinical stratification** | 2.a.i | **1 — urgent** (small lift on existing code) |
| D3 | LLM-as-a-judge (patient, doc) 5-way win-rate + human-validation spreadsheet | 2.a.ii + checkbox 2 + screenshot-3 item 1a | **2** |
| D4 | Keyword-based extrinsic alignment metric, wandb-logged during training | checkbox 1 + screenshot-3 item 1b | **3** (1h time-box) |
| D5 | Embedding-space (key vs query) dynamic wandb visualization | checkbox 3 + screenshot-3 item 1c | **3** (1h time-box) |
| D6 | Rare-outcome / long-horizon task ablations | 2.b.i.2 | **4** (new MEDS-DEV tasks required) |
| D7 | Controllable synthetic experiments for retrieval success | 2.b.ii | **4** |

## Implementation Approach

### D1 — Random-document ablation at eval `[Tier 1]`

**Result the paper wants:** "Trained model + random docs → X% AUROC drop vs trained retrieval."

Reuse Hydra retriever swap. No change to the model, no retraining needed.

- Add `src/medrap/retrievers.py :: RandomRetriever` — subclass `Retriever`, hold the same corpus as `InMemoryRetriever` but return a random subset of k indices per query (seeded). It must still populate `doc_tokens`, `doc_attention_mask`, `doc_key_embeddings`, `doc_ids`, and `doc_scores` with the *actually-retrieved* keys' embeddings so fusion/pooling behaves identically.
- **Doc scores for the random arm:** recompute via [`differentiable_retrieval_scores`](src/medrap/retrieval_scoring.py) with the same `similarity` the trained retriever uses (`dot` or `cosine`). Keeps downstream fusion/pooling numerics on-distribution so only the doc identity is ablated.
- Add `src/medrap/conf/retriever/random_in_memory.yaml` mirroring the matching trained retriever config.
- Also add `shuffled_in_memory.yaml` variant that **keeps which docs are retrieved but shuffles the assignment across patients** — isolates "is it the doc identity or the doc set?"
- Eval invocation:
  ```bash
  medrap eval checkpoint_path=<best.ckpt> retriever=random_in_memory output_dir=outputs/eval_random +seed=...
  ```
- Add a small sweep script `scripts/ablation_random_docs.sh` that runs: {trained-retriever, random, shuffled} × {seed1,2,3} and collects metrics.csv via existing [summarize_sweep.py](scripts/summarize_sweep.py) pattern.
- Tests: extend [tests/test_module_composition.py](tests/test_module_composition.py) to cover `RandomRetriever` output shape/contract. Reuse [conftest.py](conftest.py) fixtures.

### D2 — Patient × Document viz with label + clinical stratification `[Tier 1]`

Existing [build_keyword_demographic_table()](src/medrap/demographic_analysis.py) does age/race/gender heatmaps. Extend, don't rebuild.

- Add a `label_column` input path: `extract_val_schema()` already returns subject_id + prediction_time — join task labels from the datamodule's label frame (available via `datamodule.labels` / MEDS-DEV parquet).
- Generalize `build_patient_demographic_frame()` to accept an *arbitrary list of stratification columns* (label, clinical comorbidity flags, age-bin, race, gender). Make a `StratificationSpec` dataclass so new axes are additive.
- Add two new heatmap panels to [render_demographic_heatmaps()](src/medrap/demographic_analysis.py): (a) `label` axis (positive/negative outcome), (b) user-supplied "clinical" axis (e.g., comorbidity cluster) via a callable `PatientStratifier`.
- CLI: extend `scripts/run_demographic_heatmap.py` with `--stratify_by label,age_bin,race,gender,clinical` (comma-list).
- Optional: add a **per-patient row "evidence" viewer** (markdown or HTML) that lists: raw MEDS codes, task label, top-k retrieved doc texts w/ score. Needed anyway for D3 spreadsheet. Put in `src/medrap/patient_report.py` with a `build_patient_report_rows(artifacts, retrieval_ds, meds_cohort_dir) -> pl.DataFrame` that D3 can consume directly.
- Tests: extend [tests/test_demographic_analysis.py](tests/test_demographic_analysis.py) with a label-stratified fixture.

### D3 — LLM-as-a-judge (patient, doc) 5-way win-rate `[Tier 2]`

**This is the most substantive new module.** Scope:

- Given a raw-MEDS patient, a task description, and **two documents A/B** (randomized), ask an LLM which is more relevant for task prediction.
- Five comparison conditions against the "document chosen for that patient" (the anchor):
  1. random corpus document
  2. anchor vs lower-similarity doc from same patient's top-k
  3. anchor vs higher-similarity doc from same patient's top-k
  4. doc retrieved for a *different* patient with the *same* task label
  5. doc retrieved for a *different* patient with the *opposite* task label
- Expected output: per-condition **win-rate** (anchor wins / total), plus a spreadsheet dumped to CSV for a human-validation subset (~50 pairs).

Design:

- New module `src/medrap/llm_judge.py`:
  - `SamplePair` dataclass: `patient_id`, `patient_meds_text`, `task_description`, `doc_a`, `doc_b`, `condition`, `is_anchor_a`.
  - `JudgePromptBuilder` — renders patient MEDS as a compact readable timeline (reuse `preparation.OrderedFieldDocumentRenderer` pattern).
  - `OpenAIJudge` — thin async wrapper over `openai` SDK. Reads key from `OPENAI_API_KEY` env var. **Default model: `gpt-4o-mini`** (cheap enough to run full 5-condition × 100-pair sweeps). Expose `--model` so a small validation subset can be re-scored on `gpt-4o` if mini-vs-big judge agreement is in question. Parses A/B + confidence. Deterministic `seed` + `temperature=0`.
  - `build_pairs(artifacts, retrieval_ds, labels, condition, n_per_condition, seed) -> list[SamplePair]` — for each condition, sample anchors + counterfactual docs from the extraction artifacts.
  - `run_judge(pairs, judge) -> pl.DataFrame` with per-pair verdict + confidence + rationale.
  - `summarize_winrates(df) -> pl.DataFrame` — per-condition win-rate + 95% CI (bootstrap).
- New script `scripts/run_llm_judge.py` — `--run_dir`, `--retrieval_db`, `--meds_cohort`, `--n_per_condition`, `--model`, `--out_csv`, `--human_validation_subset N` (writes an extra CSV with A/B already anonymized for blind human rater).
- Cost guardrail: hard cap `n_per_condition` default to 100; token/cost budget printed at start.
- Tests: mock the OpenAI client (use a `FakeJudge` implementing the same Protocol) to test pair construction, win-rate stats, and spreadsheet shape. **Do not call the live API in unit tests.**
- Optional: add `preparation.py`-style doctest showing the 5-condition contract.

### D4 — Keyword-based extrinsic alignment metric during training `[Tier 3, 1h box]`

Cheap, proxies real alignment, uses corpus fields already available in `retrieval_ds["doc_text"]`.

- New `src/medrap/keyword_alignment.py`:
  - `extract_patient_keywords(batch, code_vocab, meds_cohort_dir=None)` — for MIMIC: map codes → {ICD descriptions, drug names} using existing [scripts/debug_vocab.py](scripts/debug_vocab.py) patterns. For MedRAG textbooks: use title + first 32 tokens of content.
  - `extract_doc_keywords(doc_text)` — noun-chunk / MeSH-like; v1 = lowercased title tokens minus stopwords.
  - `alignment_score(patient_kw, doc_kw) -> float` — Jaccard or TF-IDF cosine.
- Hook into training: new `KeywordAlignmentCallback(LightningCallback)` in [src/medrap/callbacks.py](src/medrap/callbacks.py). On `on_validation_batch_end()`, compute a running mean over ~256 patients and `self.log("val/keyword_alignment", score)`. Wandb auto-picks it up.
- Tests: unit-test the scoring functions; mock batch + retrieval_ds for the callback.

### D5 — Embedding-space dynamic wandb viz `[Tier 3, 1h box]`

- Extend existing PCA plotting in [scripts/extract_and_visualize.py](scripts/extract_and_visualize.py) into a reusable function `plot_key_query_pca(query_embeddings, doc_key_embeddings, patient_labels, doc_labels)` in a new `src/medrap/embedding_viz.py`.
- Add `EmbeddingSnapshotCallback` — at `on_validation_epoch_end()` every `log_every_n_epochs`, pull cached batch (small N ~256), PCA or t-SNE to 2D, scatter with `patient_label` colors and `doc_cluster` markers, log as `wandb.Image`.
- Also log at `on_fit_start()` (pre-training baseline) and `on_fit_end()` for the "before vs after" figure the boss asked for.
- Tests: unit-test the plotting function with tiny fake tensors (matplotlib figure non-empty, no crash).

### D6 — Rare-outcome / long-horizon tasks `[Tier 4]`

Current setup only has `mortality/in_icu/first_24h` (see [scripts/create_tasks.sh](scripts/create_tasks.sh)).

- Add new MEDS-DEV task extractions (already installable via the `task_venv`): e.g., `readmission/30d`, `mortality/in_hospital/horizon_1y`, a rare phenotype (sepsis or AKI). Extend [scripts/create_tasks.sh](scripts/create_tasks.sh).
- Rerun the train → eval → D1 ablation matrix per task; add a `task` axis to summary CSVs.
- Correlate task base-rate / horizon with "Δ AUROC from random-doc ablation."
- Defer until D1 is working end-to-end on the existing task.

### D7 — Controllable synthetic experiments `[Tier 4]`

Extend the existing synthetic sanity check (commit `fdd4d36`) at [src/medrap/conf/training/datamodule/synthetic_marginalized.yaml](src/medrap/conf/training/datamodule/synthetic_marginalized.yaml).

- Parameterize: (a) sequence length per patient, (b) class balance (`p_positive`), (c) retrieval-signal strength (overlap between code ranges & doc embeddings), (d) noise / distractor docs.
- Add a small `synthetic_sweep.py` that runs the grid and produces "performance vs signal strength" curve — the classic retrieval-value figure for the paper.
- Defer until D1 is working and the paper figure skeleton is chosen.

## Critical Files

- [src/medrap/retrievers.py](src/medrap/retrievers.py) — add `RandomRetriever`, `ShuffledRetriever`
- [src/medrap/conf/retriever/](src/medrap/conf/retriever/) — new yaml configs
- [src/medrap/demographic_analysis.py](src/medrap/demographic_analysis.py) — add label/clinical stratification
- [src/medrap/llm_judge.py](src/medrap/llm_judge.py) *(new)* — LLM-as-a-judge module
- [src/medrap/patient_report.py](src/medrap/patient_report.py) *(new)* — per-patient evidence rows
- [src/medrap/keyword_alignment.py](src/medrap/keyword_alignment.py) *(new)*
- [src/medrap/embedding_viz.py](src/medrap/embedding_viz.py) *(new)*
- [src/medrap/callbacks.py](src/medrap/callbacks.py) — add `KeywordAlignmentCallback`, `EmbeddingSnapshotCallback`
- [scripts/ablation_random_docs.sh](scripts/) *(new)*
- [scripts/run_llm_judge.py](scripts/) *(new)*
- Tests in [tests/](tests/) mirroring every new module

## Reused Building Blocks

- `extract_artifacts()` ([extraction.py](src/medrap/extraction.py)) for all offline analyses
- `DocKeywordProvider` + `build_patient_demographic_frame()` ([demographic_analysis.py](src/medrap/demographic_analysis.py)) for D2
- `retrieval_diagnostic_scalars()` pattern ([retrieval_logging.py](src/medrap/retrieval_logging.py)) for D4 callback
- Hydra retriever group override mechanic ([cli.py](src/medrap/cli.py) `_run_eval`) for D1
- `OrderedFieldDocumentRenderer` ([preparation.py](src/medrap/preparation.py)) for patient-timeline rendering in D3
- `summarize_sweep.py` for D1 result aggregation

## Verification

- **D1:** `pytest tests/test_retrievers.py::test_random_retriever`; then run `medrap eval retriever=random_in_memory` on an existing checkpoint and confirm AUROC is lower than with the trained retriever.
- **D2:** Run `scripts/run_demographic_heatmap.py --run_dir <existing run> --stratify_by label`; confirm new label panel in PNG + CSV.
- **D3:** Unit tests with `FakeJudge` green. Then a **budgeted** live run (n=20 per condition, cheapest model) on one existing run dir; sanity-check win-rates (expect: anchor ≫ random; anchor ≈ same-label-different-patient).
- **D4:** Training run with callback enabled; confirm `val/keyword_alignment` appears in wandb panel; correlates with `val_auroc` across epochs.
- **D5:** Training run; confirm epoch-0 scatter (random) vs final-epoch scatter (clustered) rendered in wandb.
- **D6/D7:** Defer spec until D1 numbers land.

## Order of Attack

User confirmed scope: **all seven deliverables (D1–D7)**. Recommended sequencing:

1. **D1** (random-doc ablation) — unblocks the paper's headline result.
2. **D2** (label + clinical stratification in heatmaps) — minimal effort on existing code; fills the "what are we retrieving" section.
3. **D3** (LLM-as-a-judge, default `gpt-4o-mini`) — the big new module; run end-to-end on one checkpoint. Also produces the per-patient evidence rows D2 reuses.
4. **D4** (keyword alignment, 1h box) + **D5** (embedding-viz wandb callback, 1h box) — reuse D3's per-patient rendering.
5. **D6** (rare-outcome / long-horizon MEDS-DEV tasks) — rerun D1's ablation matrix across tasks once D1 is stable.
6. **D7** (controllable synthetic experiments) — final figure-polish step once the paper skeleton is chosen.
