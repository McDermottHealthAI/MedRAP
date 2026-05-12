"""Render keyword x demographic heatmaps from a trained MedRAP run.

If extraction artifacts already exist under ``<run_dir>/extraction/``, they
are reused; otherwise we run extraction first via the same code path as
``scripts/extract_and_visualize.py``.

Usage::

    python scripts/run_demographic_heatmap.py \\
        --run_dir outputs/mimic_run_retrieval_only \\
        --retrieval_db data/retrieval_db \\
        --meds_cohort /groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort \\
        --keyword_provider title
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightning
import torch
from omegaconf import OmegaConf

# Make src/ importable when invoked from the repo root.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from medrap.comorbidity import (  # noqa: E402
    CHARLSON_CATEGORIES,
    assign_patient_charlson,
    load_charlson_lookup,
)
from medrap.configs import instantiate_datamodule, instantiate_training_module  # noqa: E402
from medrap.demographic_analysis import (  # noqa: E402
    LDATopicProvider,
    TitleKeywordProvider,
    aggregate_race,
    build_patient_demographic_frame,
    extract_val_schema,
    load_subject_demographics,
    render_demographic_heatmaps,
)
from medrap.extraction import extract_artifacts  # noqa: E402

# Reuse the checkpoint resolver from extract_and_visualize.
sys.path.insert(0, str(_repo_root / "scripts"))
from extract_and_visualize import _find_checkpoint  # noqa: E402


def _build_provider(name: str, retrieval_db: Path, *, n_topics: int = 30):
    if name == "title":
        return TitleKeywordProvider(retrieval_db)
    if name == "lda":
        return LDATopicProvider(retrieval_db, n_topics=n_topics)
    raise ValueError(f"Unknown --keyword_provider: {name!r}. Supported: title, lda")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render keyword x demographic heatmaps for a MedRAP MIMIC run."
    )
    parser.add_argument(
        "--run_dir", type=Path, required=True, help="Training run directory (must contain config.yaml)."
    )
    parser.add_argument(
        "--retrieval_db", type=Path, required=True, help="HF dataset directory used as the retrieval corpus."
    )
    parser.add_argument(
        "--meds_cohort",
        type=Path,
        required=True,
        help="MEDS cohort root containing data/{train,tuning,held_out}/*.parquet.",
    )
    parser.add_argument(
        "--keyword_provider",
        type=str,
        default="lda",
        help="Doc → keyword provider: 'lda' (default) or 'title'.",
    )
    parser.add_argument(
        "--n_topics",
        type=int,
        default=30,
        help="Number of LDA topics (only used with --keyword_provider lda).",
    )
    parser.add_argument(
        "--top_n_keywords", type=int, default=20, help="Cap on number of keywords shown per heatmap."
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    if not (run_dir / "config.yaml").is_file():
        print(f"Error: {run_dir / 'config.yaml'} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Run directory: {run_dir}")

    cfg = OmegaConf.load(run_dir / "config.yaml")

    # Step 1: Reuse existing artifacts if present, else extract.
    extract_dir = run_dir / "extraction"
    artifact_path = extract_dir / "extraction_artifacts.pt"

    datamodule = instantiate_datamodule(cfg)

    artifacts = None
    if artifact_path.is_file():
        cached = torch.load(artifact_path, weights_only=True)
        if "doc_ids" in cached and "differentiable_doc_scores" in cached:
            print(f"Reusing existing artifacts at {artifact_path}")
            artifacts = cached
        else:
            print(
                f"Cached artifacts at {artifact_path} are missing required keys "
                f"(have {sorted(cached.keys())}); re-extracting."
            )

    if artifacts is None:
        print(f"Running extraction for {run_dir}")
        ckpt_path = _find_checkpoint(run_dir)
        module = instantiate_training_module(cfg)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        module.load_state_dict(checkpoint["state_dict"])

        datamodule.setup("fit")
        dataloader = datamodule.val_dataloader()

        trainer = lightning.Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        artifact_path = extract_artifacts(module, dataloader, trainer, output_dir=extract_dir)
        artifacts = torch.load(artifact_path, weights_only=True)

    # Step 2: Recover val-split (subject_id, end_event_index, prediction_time).
    val_schema = extract_val_schema(datamodule)
    print(f"Val schema rows: {val_schema.height}")
    print(f"Artifact rows:   {artifacts['doc_ids'].shape[0]}")
    if val_schema.height != artifacts["doc_ids"].shape[0]:
        print(
            "ERROR: row counts disagree — the dataloader-order alignment "
            "assumption is broken. Capture subject_ids inside predict_step instead.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Step 3: Pull demographics for the unique subject ids in the val split.
    subject_ids = val_schema["subject_id"].unique().to_list()
    print(f"Loading demographics for {len(subject_ids)} unique subjects...")
    demographics = load_subject_demographics(args.meds_cohort, subject_ids)

    patient_frame = build_patient_demographic_frame(val_schema, demographics)
    n_with_age = int(patient_frame["age_years"].is_not_null().sum())
    n_with_gender = int(patient_frame["gender"].is_not_null().sum())
    n_with_race = int(patient_frame["race"].is_not_null().sum())
    print(f"  age:    {n_with_age}/{patient_frame.height}")
    print(f"  gender: {n_with_gender}/{patient_frame.height}")
    print(f"  race:   {n_with_race}/{patient_frame.height}")

    # Step 3b: Per-patient Charlson comorbidity flags (multi-membership,
    # with the canonical hierarchy de-duplication). Row order matches
    # val_schema by construction.
    print("Computing per-patient Charlson comorbidities from MEDS diagnoses...")
    charlson_lookup = load_charlson_lookup()
    comorbidity_frame = assign_patient_charlson(
        args.meds_cohort, val_schema, lookup=charlson_lookup
    )
    if comorbidity_frame.height != patient_frame.height:
        print(
            f"ERROR: comorbidity_frame rows ({comorbidity_frame.height}) != "
            f"patient_frame rows ({patient_frame.height}).",
            file=sys.stderr,
        )
        sys.exit(2)
    total_n = comorbidity_frame.height
    n_any = int(comorbidity_frame["any_charlson"].sum())
    print(
        f"  Charlson prevalence in val split (N={total_n}, any flag: "
        f"{n_any} = {100 * n_any / max(total_n, 1):.1f}%):"
    )
    prevalences = []
    for cat in CHARLSON_CATEGORIES:
        n_cat = int(comorbidity_frame[cat].sum())
        prevalences.append((cat, n_cat, 100 * n_cat / max(total_n, 1)))
    # Sort descending so the most prevalent categories print first.
    for cat, n_cat, pct in sorted(prevalences, key=lambda x: -x[1]):
        if n_cat > 0:
            print(f"    {cat:<45}  {n_cat:>6}  ({pct:5.1f}%)")

    # Step 4: Build provider and render heatmaps.
    provider = _build_provider(args.keyword_provider, args.retrieval_db, n_topics=args.n_topics)
    print(f"Keyword vocab size: {len(provider.vocab)}")

    # Diagnostic: demographic distributions.
    from collections import Counter

    gender_dist = Counter(g if g is not None else "unknown" for g in patient_frame["gender"].to_list())
    race_dist = Counter(aggregate_race(r) for r in patient_frame["race"].to_list())
    age_dist = Counter(patient_frame["age_bin"].to_list())
    print(f"\n  Gender distribution: {dict(gender_dist)}")
    print(f"  Race/Ethnicity distribution: {dict(race_dist)}")
    print(f"  Age bin distribution: {dict(age_dist)}")

    # Diagnostic: retrieval diversity.
    import numpy as np

    doc_ids_np = artifacts["doc_ids"].numpy()
    if doc_ids_np.ndim == 3 and doc_ids_np.shape[1] == 1:
        doc_ids_flat = doc_ids_np[:, 0, :].reshape(-1)
    else:
        doc_ids_flat = doc_ids_np.reshape(-1)
    n_unique_docs = len(np.unique(doc_ids_flat))
    doc_freq = Counter(doc_ids_flat.tolist())
    top5 = doc_freq.most_common(5)
    total_retrievals = len(doc_ids_flat)
    unique_keywords = set()
    for did in np.unique(doc_ids_flat):
        for kw, _ in provider.keywords_for(int(did)):
            unique_keywords.add(kw)
    print(f"\n  Unique docs retrieved: {n_unique_docs}")
    n_unique_kws = len(unique_keywords)
    n_vocab = len(provider.vocab)
    print(f"  Unique keywords (book titles) among retrieved docs: {n_unique_kws} / {n_vocab}")
    if len(unique_keywords) < len(provider.vocab):
        missing = sorted(set(provider.vocab) - unique_keywords)
        print(f"  Books NEVER retrieved: {missing}")
    print(f"  Top-5 doc_ids by frequency (of {total_retrievals} total retrievals):")
    for did, cnt in top5:
        title = provider.keywords_for(int(did))[0][0]
        print(f"    doc_id={did} ({cnt} times, {100 * cnt / total_retrievals:.1f}%): {title[:60]}")

    tables = render_demographic_heatmaps(
        artifacts=artifacts,
        provider=provider,
        patient_frame=patient_frame,
        output_dir=extract_dir,
        top_n_keywords=args.top_n_keywords,
        comorbidity_frame=comorbidity_frame,
        comorbidity_categories=CHARLSON_CATEGORIES,
    )

    # Diagnostic: check if table values differ across demographic bins.
    print("\n=== Invariance diagnostics ===")
    for axis_name, info in tables.items():
        tbl = info["table"]
        bins = info["bins"]
        if tbl.shape[0] < 2:
            print(f"  {axis_name}: only {len(bins)} bin(s), cannot compare")
            continue
        max_diff = float((tbl.max(axis=0) - tbl.min(axis=0)).max()) if tbl.size > 0 else 0.0
        print(f"  {axis_name}: {len(bins)} bins, max abs diff across bins = {max_diff:.8f}")
        if max_diff == 0.0:
            print("    *** VALUES ARE EXACTLY IDENTICAL — likely a bug ***")
        # Print first 3 keyword columns for each bin.
        kws = info["keywords"][:3]
        for i, b in enumerate(bins):
            vals = ", ".join(f"{tbl[i, j]:.6f}" for j in range(min(3, tbl.shape[1])))
            print(f"    {b}: [{vals}]")


if __name__ == "__main__":
    main()
