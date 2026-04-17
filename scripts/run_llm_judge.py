"""Run the LLM-as-a-judge patient-level retrieval-relevance evaluation.

Reuses extraction artifacts from a trained MedRAP run (same cache block as
``scripts/run_demographic_heatmap.py``). See ``D3_plan.md`` for the method.

Usage::

    python scripts/run_llm_judge.py \\
        --run_dir outputs/mimic_run_retrieval_only \\
        --retrieval_db data/retrieval_db \\
        --meds_cohort /groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort \\
        --task_description "Predict in-ICU mortality within the first 24 hours of ICU admission." \\
        --n_patients 100 --dry_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightning
import numpy as np
import polars as pl
import torch
from datasets import load_from_disk
from omegaconf import OmegaConf

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from medrap.configs import instantiate_datamodule, instantiate_training_module  # noqa: E402
from medrap.demographic_analysis import (  # noqa: E402
    _build_doc_id_to_row_map,
    extract_val_schema,
    load_subject_demographics,
)
from medrap.extraction import extract_artifacts  # noqa: E402
from medrap.llm_judge import (  # noqa: E402
    JudgePromptBuilder,
    OpenAIJudge,
    PatientTimelineRenderer,
    build_human_validation_subset,
    build_pairs,
    build_per_patient_rollup,
    run_judge,
    summarize_winrates,
    write_results_workbook,
)

sys.path.insert(0, str(_repo_root / "scripts"))
from extract_and_visualize import _find_checkpoint  # noqa: E402


# Rough public pricing per 1K tokens for gpt-4o-mini (verify before paper
# submission — these drift). Used only in the dry-run cost estimate.
_PRICE_PER_1K = {
    "gpt-4o-mini": (0.00015, 0.0006),  # (input, output)
    "gpt-4o": (0.0025, 0.01),
}


def _estimate_tokens(s: str) -> int:
    """Rough token count fallback if tiktoken is unavailable (~4 chars/token)."""
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        return len(enc.encode(s))
    except Exception:
        return max(1, len(s) // 4)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-as-a-judge patient-level retrieval-relevance evaluation."
    )
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--retrieval_db", type=Path, required=True)
    parser.add_argument("--meds_cohort", type=Path, required=True)

    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task_description", type=str)
    task_group.add_argument("--task_description_file", type=Path)

    parser.add_argument("--families", type=str, default="F1,F2,F3,F4")
    parser.add_argument("--n_patients", type=int, default=100)
    parser.add_argument("--pairs_per_patient_per_family", type=int, default=1)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--human_validation_n", type=int, default=50)
    parser.add_argument("--max_total_calls_cap", type=int, default=1000)
    parser.add_argument("--timeline_max_events", type=int, default=150)
    parser.add_argument("--doc_max_chars", type=int, default=4000)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_task_description(args: argparse.Namespace) -> str:
    if args.task_description is not None:
        return args.task_description
    return Path(args.task_description_file).read_text().strip()


def _load_or_extract_artifacts(run_dir: Path, cfg, datamodule) -> dict:
    """Mirror the cache pattern in ``scripts/run_demographic_heatmap.py``."""
    extract_dir = run_dir / "extraction"
    artifact_path = extract_dir / "extraction_artifacts.pt"

    if artifact_path.is_file():
        cached = torch.load(artifact_path, weights_only=True)
        if "doc_ids" in cached:
            print(f"Reusing existing artifacts at {artifact_path}")
            return cached
        print(f"Cached artifacts missing required keys; re-extracting.")

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
    return torch.load(artifact_path, weights_only=True)


def _as_numpy(x):
    return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def main() -> None:  # noqa: C901 - CLI pipeline, branches are linear
    args = _parse_args()

    run_dir: Path = args.run_dir
    if not (run_dir / "config.yaml").is_file():
        print(f"Error: {run_dir / 'config.yaml'} not found.", file=sys.stderr)
        sys.exit(1)

    task_description = _resolve_task_description(args)
    families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    out_dir: Path = args.out_dir or (run_dir / "llm_judge")
    if out_dir.exists() and not args.overwrite and not args.dry_run:
        contents = [p.name for p in out_dir.iterdir()]
        if contents:
            print(
                f"Error: {out_dir} is not empty (pass --overwrite to proceed). "
                f"Contents: {contents[:5]}",
                file=sys.stderr,
            )
            sys.exit(2)

    cfg = OmegaConf.load(run_dir / "config.yaml")
    tensorized_cohort_dir = Path(cfg.training.datamodule.config.tensorized_cohort_dir)
    codes_parquet = tensorized_cohort_dir / "metadata" / "codes.parquet"
    if not codes_parquet.is_file():
        print(
            f"Error: codes.parquet not found at {codes_parquet} — required for "
            f"patient-timeline annotation.",
            file=sys.stderr,
        )
        sys.exit(3)

    datamodule = instantiate_datamodule(cfg)
    artifacts = _load_or_extract_artifacts(run_dir, cfg, datamodule)

    val_schema = extract_val_schema(datamodule)
    doc_ids_tensor = artifacts["doc_ids"]
    if val_schema.height != doc_ids_tensor.shape[0]:
        print(
            "ERROR: val_schema rows != artifact rows — alignment assumption broken.",
            file=sys.stderr,
        )
        sys.exit(2)

    labels = _as_numpy(artifacts["targets"]).astype(int)
    unique_labels = set(np.unique(labels).tolist())
    if not {0, 1}.issubset(unique_labels):
        print(
            f"ERROR: expected both classes {{0, 1}} in targets; saw {unique_labels}.",
            file=sys.stderr,
        )
        sys.exit(2)

    retrieval_ds = load_from_disk(str(args.retrieval_db))
    corpus_size = len(retrieval_ds)
    doc_id_to_row = _build_doc_id_to_row_map(retrieval_ds)
    if doc_id_to_row is None:
        doc_id_to_row = {i: i for i in range(corpus_size)}

    doc_ids_np = _as_numpy(doc_ids_tensor)
    k = int(doc_ids_np.shape[2]) if doc_ids_np.ndim == 3 else 1

    print(f"Corpus size:     {corpus_size}")
    print(f"Artifact rows:   {doc_ids_np.shape[0]}")
    print(f"Retrieval k:     {k}")
    print(f"Families:        {families}")

    subject_ids = val_schema["subject_id"].unique().to_list()
    demographics = load_subject_demographics(args.meds_cohort, subject_ids)

    artifacts_np = {
        "doc_ids": doc_ids_np,
        "doc_scores": _as_numpy(artifacts["doc_scores"]),
        "targets": labels,
        "logits": _as_numpy(artifacts.get("logits", np.zeros((doc_ids_np.shape[0], 2)))),
    }

    pairs = build_pairs(
        artifacts=artifacts_np,
        val_schema=val_schema,
        labels=labels,
        families=families,
        n_patients=args.n_patients,
        pairs_per_patient_per_family=args.pairs_per_patient_per_family,
        corpus_size=corpus_size,
        k=k,
        seed=args.seed,
    )
    print(f"Pairs built:     {len(pairs)}")

    if len(pairs) > args.max_total_calls_cap:
        print(
            f"ERROR: {len(pairs)} pairs exceeds --max_total_calls_cap="
            f"{args.max_total_calls_cap}. Lower --n_patients or raise the cap.",
            file=sys.stderr,
        )
        sys.exit(4)

    timeline_renderer = PatientTimelineRenderer(
        codes_parquet=codes_parquet,
        max_events=args.timeline_max_events,
    )

    # Pre-render a representative timeline for cost estimation.
    anchor_sid_sample = pairs[0].anchor_subject_id if pairs else None
    if anchor_sid_sample is not None:
        pred_t = (
            val_schema.filter(pl.col("subject_id") == anchor_sid_sample)["prediction_time"].to_list()[0]
        )
        sample_timeline = timeline_renderer.render(
            anchor_sid_sample, pred_t, args.meds_cohort
        )
    else:
        sample_timeline = ""

    prompt_builder = JudgePromptBuilder(
        task_description=task_description,
        timeline_renderer=timeline_renderer,
        retrieval_ds=retrieval_ds,
        doc_id_to_row=doc_id_to_row,
        max_doc_chars=args.doc_max_chars,
    )

    sample_sys, sample_user = (
        prompt_builder.build(pairs[0], patient_timeline=sample_timeline) if pairs else ("", "")
    )
    sample_tokens = _estimate_tokens(sample_sys) + _estimate_tokens(sample_user)
    # Completions are small (single JSON object) — budget ~50 tokens.
    est_output_tokens = 50
    total_input = sample_tokens * len(pairs)
    total_output = est_output_tokens * len(pairs)
    in_price, out_price = _PRICE_PER_1K.get(args.model, (0.0, 0.0))
    est_cost = (total_input / 1000.0) * in_price + (total_output / 1000.0) * out_price
    print(
        f"Est tokens:      ~{total_input / 1e6:.2f}M input + "
        f"~{total_output / 1e3:.1f}K output | est cost ~${est_cost:.2f}"
    )

    if args.dry_run:
        print("Dry-run: exiting before API calls.")
        return

    # Render timelines for every unique anchor once, then reuse.
    unique_anchor_sids = {p.anchor_subject_id for p in pairs}
    print(f"Rendering timelines for {len(unique_anchor_sids)} unique anchors...")
    schema_lookup = {
        int(r["subject_id"]): r["prediction_time"]
        for r in val_schema.iter_rows(named=True)
        if int(r["subject_id"]) in unique_anchor_sids
    }
    timelines: dict[int, str] = {}
    for sid in unique_anchor_sids:
        pred_t = schema_lookup.get(int(sid))
        if pred_t is None:
            timelines[int(sid)] = ""
            continue
        timelines[int(sid)] = timeline_renderer.render(int(sid), pred_t, args.meds_cohort)

    judge = OpenAIJudge(model=args.model)
    verdicts_df = run_judge(
        pairs,
        judge=judge,
        prompt_builder=prompt_builder,
        timelines_by_subject_id=timelines,
        max_workers=args.max_workers,
    )

    summary_df = summarize_winrates(verdicts_df, n_bootstrap=args.n_bootstrap, seed=args.seed)
    per_patient_df = build_per_patient_rollup(
        pairs,
        verdicts_df,
        logits=artifacts_np["logits"],
        targets=labels,
        artifacts=artifacts_np,
        timeline_renderer=timeline_renderer,
        val_schema=val_schema,
        demographics=demographics,
        retrieval_ds=retrieval_ds,
        doc_id_to_row=doc_id_to_row,
        timelines_by_subject_id=timelines,
        families=families,
    )
    human_df = build_human_validation_subset(
        verdicts_df,
        n=args.human_validation_n,
        seed=args.seed,
        retrieval_ds=retrieval_ds,
        doc_id_to_row=doc_id_to_row,
    )

    # Collision-rate diagnostic for F3/F4.
    for fam in ("F3", "F4"):
        if fam not in families:
            continue
        expected = args.n_patients * args.pairs_per_patient_per_family
        fam_row = summary_df.filter(pl.col("family") == fam)
        if fam_row.height == 0:
            continue
        actual = int(fam_row["n_pairs"][0])
        dropped = expected - actual
        if dropped > 0:
            print(
                f"[{fam}] collision/dedupe drops: {dropped}/{expected} "
                f"({100 * dropped / max(expected, 1):.1f}%). If ≫ 5%, retriever may be "
                f"collapsing onto a small doc set.",
                file=sys.stderr,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    verdicts_df.write_csv(out_dir / "pairs_verdicts.csv")
    per_patient_df.write_csv(out_dir / "per_patient_results.csv")
    summary_df.write_csv(out_dir / "family_winrates.csv")
    human_df.write_csv(out_dir / "human_validation.csv")

    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "retrieval_db": str(args.retrieval_db),
                "meds_cohort": str(args.meds_cohort),
                "task_description": task_description,
                "families": list(families),
                "n_patients": args.n_patients,
                "pairs_per_patient_per_family": args.pairs_per_patient_per_family,
                "model": args.model,
                "seed": args.seed,
                "n_bootstrap": args.n_bootstrap,
                "n_pairs_actual": len(pairs),
                "corpus_size": corpus_size,
                "k": k,
            },
            indent=2,
            default=str,
        )
    )

    workbook_path = out_dir / "llm_judge_results.xlsx"
    write_results_workbook(
        workbook_path,
        family_winrates=summary_df,
        per_patient=per_patient_df,
        pairs_verdicts=verdicts_df,
        human_validation=human_df,
    )
    print(f"\nWrote results to {out_dir}")
    print(summary_df)


if __name__ == "__main__":
    main()
