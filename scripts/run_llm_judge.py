"""Run the LLM-as-a-judge patient-level retrieval-relevance evaluation.

Reuses extraction artifacts from a trained MedRAP run (same cache block as
``scripts/run_demographic_heatmap.py``). See ``D3_plan.md`` for the method.

Auto-detects binary vs. multitask runs from
``artifacts['targets'].ndim`` so the same command works on both training
setups; pass ``--task_mode binary|multitask`` to override.

In multitask mode, a single invocation sweeps all 25 tasks (or just one
when ``--target_task K`` is set). Per-task outputs land at
``<run_dir>/llm_judge/task<K>/``; a cross-task summary is written at
``<run_dir>/llm_judge/all_task_winrates{,_wide}.csv``.

Auto-generated multitask task descriptions use the readable task name
(via the same ``d_labitems`` lookup ``scripts/extract_and_visualize.py``
uses) — raw ``LAB//<itemid>//<unit>`` codes never appear in prompts.

Usage::

    # Binary single-task run
    uv run python scripts/run_llm_judge.py \\
        --run_dir outputs/mimic_run_rope_cross_attention \\
        --retrieval_db data/retrieval_db \\
        --meds_cohort /groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort \\
        --task_description "Predict in-ICU mortality within the first 24 hours of ICU admission." \\
        --max_total_calls_cap 500

    # Multitask sweep — auto-detects, auto-generates per-task descriptions
    uv run python scripts/run_llm_judge.py \\
        --run_dir outputs/mt_rope_cross_attention \\
        --retrieval_db data/retrieval_db \\
        --meds_cohort /groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort \\
        --max_total_calls_cap 500
"""

from __future__ import annotations

import argparse
import json
import os
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
    compute_patient_clinical_summary,
    run_judge,
    summarize_winrates,
    write_results_workbook,
)

sys.path.insert(0, str(_repo_root / "scripts"))
from extract_and_visualize import (  # noqa: E402
    _DEFAULT_LABITEMS_PATH,
    _find_checkpoint,
    _humanize_meds_code,
    _load_lab_label_lookup,
    _load_task_codes,
)


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


def _format_sample_prompt(
    sample_sys: str,
    sample_user: str,
    sample_timeline: str,
    sample_timeline_chrono_len: int | None,
) -> str:
    """Render SYSTEM + USER prompt for pair 0 as a single text block with separators."""
    bar = "=" * 80
    header = f"SAMPLE PROMPT (pair 0) — USER  ({len(sample_user)} chars"
    if sample_timeline_chrono_len is not None and sample_timeline_chrono_len > 0:
        # Prompt length - timeline length + chrono length = chrono-equivalent prompt length.
        chrono_equiv = len(sample_user) - len(sample_timeline) + sample_timeline_chrono_len
        saved = chrono_equiv - len(sample_user)
        pct = 100 * saved / chrono_equiv if chrono_equiv > 0 else 0
        header += f"; chronological would be ~{chrono_equiv} chars, saved {saved} / {pct:.0f}%"
    header += ")"
    parts = [
        bar,
        "SAMPLE PROMPT (pair 0) — SYSTEM",
        bar,
        sample_sys,
        "",
        bar,
        header,
        bar,
        sample_user,
        bar,
    ]
    return "\n".join(parts) + "\n"


def _resolve_task_mode(mode: str, raw_targets: np.ndarray) -> str:
    """Map ``--task_mode auto|binary|multitask`` to ``binary`` or ``multitask``."""
    if mode == "auto":
        return "multitask" if raw_targets.ndim == 2 else "binary"
    if mode in ("binary", "multitask"):
        return mode
    raise ValueError(f"unknown task_mode {mode!r}; expected one of auto/binary/multitask")


def _auto_task_description(
    task_code: str,
    lab_lookup: dict[int, tuple[str, str]] | None,
    horizon_days: float,
    anchor_offset_hours: float,
) -> str:
    """Return a natural-language one-sentence description for a multitask code.

    Always uses ``_humanize_meds_code`` for the noun phrase. Raw ``//``-delimited
    MEDS codes never appear in the output — when a code falls outside the
    handled prefixes, the function emits a generic-but-readable phrase rather
    than leaking the raw code.
    """
    parts = task_code.split("//")
    head = parts[0]
    horizon_text = f"{horizon_days:g} days"
    anchor_text = f"first event + {anchor_offset_hours:g} h"

    if head == "MEDICATION" and len(parts) >= 3:
        action = parts[1].strip()
        third = parts[2].strip()
        if action in ("START", "STOP"):
            verb = "started" if action == "START" else "stopped"
            if third == "UNK":
                return (
                    f"Predict whether a medication (whose name was not normalized "
                    f"in the MEDS dataset) will be {verb} for this patient within "
                    f"{horizon_text} after the patient's first clinical event "
                    f"(anchor: {anchor_text})."
                )
            return (
                f"Predict whether {third} will be {verb} for this patient within "
                f"{horizon_text} after the patient's first clinical event "
                f"(anchor: {anchor_text})."
            )
        if third == "Administered":
            return (
                f"Predict whether {action} will be administered to this patient within "
                f"{horizon_text} after the patient's first clinical event "
                f"(anchor: {anchor_text})."
            )

    if head == "LAB" and len(parts) >= 2:
        try:
            itemid = int(parts[1])
        except ValueError:
            itemid = None
        if itemid is not None and lab_lookup and itemid in lab_lookup:
            label, fluid = lab_lookup[itemid]
            test_name = f"'{label} [{fluid}]'" if fluid else f"'{label}'"
        elif itemid is not None:
            test_name = f"with internal item id {itemid}"
        else:
            test_name = "of unknown name"
        return (
            f"Predict whether the lab test {test_name} will be ordered for this "
            f"patient within {horizon_text} after the patient's first clinical event "
            f"(anchor: {anchor_text})."
        )

    if head == "TRANSFER_TO" and len(parts) >= 2:
        target = parts[1].strip().lower()
        if target == "discharge":
            return (
                f"Predict whether the patient will be discharged from the hospital "
                f"within {horizon_text} after the patient's first clinical event "
                f"(anchor: {anchor_text})."
            )
        return (
            f"Predict whether the patient will be transferred to {target} within "
            f"{horizon_text} after the patient's first clinical event "
            f"(anchor: {anchor_text})."
        )

    if head == "ED_OUT":
        return (
            f"Predict whether the patient's ED departure will occur within "
            f"{horizon_text} after the patient's first clinical event "
            f"(anchor: {anchor_text})."
        )

    # Fallback for unknown prefixes — humanize what we can, but make absolutely
    # sure no `//` leaks into the prompt (the project requires natural-language
    # task descriptions).
    humanized = _humanize_meds_code(task_code, lab_lookup)
    if "//" in humanized:
        humanized = "an unrecognized clinical event"
    return (
        f"Predict whether {humanized} will occur for this patient within "
        f"{horizon_text} after the patient's first clinical event "
        f"(anchor: {anchor_text})."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-as-a-judge patient-level retrieval-relevance evaluation."
    )
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--retrieval_db", type=Path, required=True)
    parser.add_argument("--meds_cohort", type=Path, required=True)

    # Binary mode requires --task_description (or _file). Multitask mode auto-
    # generates it per task. We relax the mutually-exclusive group to optional
    # here and validate per-mode below.
    task_group = parser.add_mutually_exclusive_group(required=False)
    task_group.add_argument("--task_description", type=str)
    task_group.add_argument("--task_description_file", type=Path)

    parser.add_argument(
        "--task_mode",
        choices=("auto", "binary", "multitask"),
        default="auto",
        help=(
            "Default 'auto' inspects artifacts['targets'].ndim — 1-D → binary, "
            "2-D → multitask. Use 'binary' / 'multitask' to override."
        ),
    )
    parser.add_argument(
        "--target_task",
        type=int,
        default=None,
        help=(
            "Multitask only: run a single task index instead of sweeping all. "
            "Rejected for binary."
        ),
    )
    parser.add_argument(
        "--mimic_labitems_path",
        type=Path,
        default=_DEFAULT_LABITEMS_PATH,
        help=(
            "Multitask only: path to MIMIC-IV's d_labitems CSV (gzip OK). Used "
            f"to humanize LAB tasks. Default: {_DEFAULT_LABITEMS_PATH}."
        ),
    )

    parser.add_argument("--families", type=str, default="F1,F2,F3,F4")
    parser.add_argument("--n_patients", type=int, default=100)
    parser.add_argument("--pairs_per_patient_per_family", type=int, default=1)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--human_validation_n", type=int, default=50)
    parser.add_argument(
        "--max_total_calls_cap",
        type=int,
        default=100,
        help=(
            "Cap on number of API calls. Applies PER TASK in multitask mode "
            "(so a 25-task sweep at default --n_patients=100 caps at "
            "100 × 25 = 2500). Lower --n_patients or raise the cap if the "
            "per-task pair count exceeds it."
        ),
    )
    parser.add_argument("--timeline_max_events", type=int, default=20)
    parser.add_argument("--doc_max_chars", type=int, default=4000)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--save_sample_prompt",
        type=Path,
        default=None,
        help=(
            "Override path for the saved SYSTEM+USER prompt. By default, the "
            "sample prompt for pair 0 is always saved to <out_dir>/sample_prompt.txt "
            "(binary) or <out_root>/task<K>/sample_prompt.txt (each task in "
            "multitask). Use this flag only to redirect the binary save."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_task_description(args: argparse.Namespace) -> str | None:
    if args.task_description is not None:
        return args.task_description
    if args.task_description_file is not None:
        return Path(args.task_description_file).read_text().strip()
    return None


def _load_or_extract_artifacts(run_dir: Path, cfg, datamodule) -> dict:
    """Mirror the cache pattern in ``scripts/run_demographic_heatmap.py``."""
    extract_dir = run_dir / "extraction"
    artifact_path = extract_dir / "extraction_artifacts.pt"

    if artifact_path.is_file():
        cached = torch.load(artifact_path, weights_only=True)
        if "doc_ids" in cached:
            print(f"Reusing existing artifacts at {artifact_path}")
            return cached
        print("Cached artifacts missing required keys; re-extracting.")

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


def _select_target_task(
    raw_targets: np.ndarray,
    target_task: int,
    task_codes: dict[int, str] | None,
    lab_lookup: dict[int, tuple[str, str]] | None,
    horizon_days: float,
    anchor_offset_hours: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Slice 2-D ``raw_targets`` to one task and drop NaN-labeled rows.

    Returns:
        labels: ``(n_valid,)`` int (NaN-free, 0/1).
        valid_indices: ``(n_valid,)`` int positional indices into the original N.
        task_meta: dict with keys ``target_task``, ``task_code``, ``task_label``,
            ``task_description``, ``n_valid``, ``n_pos``, ``n_neg``,
            ``horizon_days``, ``anchor_offset_hours``.
    """
    if raw_targets.ndim != 2:
        raise ValueError(f"expected 2-D targets, got shape {raw_targets.shape}")
    n_tasks = raw_targets.shape[1]
    if not 0 <= target_task < n_tasks:
        raise ValueError(f"target_task={target_task} out of range [0, {n_tasks})")

    col = raw_targets[:, target_task].astype(float, copy=False)
    valid = ~np.isnan(col)
    valid_indices = np.where(valid)[0]
    labels = col[valid_indices].astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    task_code = task_codes.get(target_task) if task_codes else None
    if task_code is None:
        task_label = f"task {target_task}"
        task_description = (
            f"Predict task {target_task} within {horizon_days:g} days after the "
            f"patient's first clinical event (anchor: first event + "
            f"{anchor_offset_hours:g} h)."
        )
    else:
        task_label = _humanize_meds_code(task_code, lab_lookup)
        task_description = _auto_task_description(
            task_code, lab_lookup, horizon_days, anchor_offset_hours
        )

    return labels, valid_indices, {
        "target_task": target_task,
        "task_code": task_code,
        "task_label": task_label,
        "task_description": task_description,
        "n_valid": int(labels.size),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "horizon_days": horizon_days,
        "anchor_offset_hours": anchor_offset_hours,
    }


def _check_out_dir_empty(out_dir: Path, *, overwrite: bool) -> bool:
    """Return True if ``out_dir`` is OK to write into. Prints stderr and returns
    False if it exists, is non-empty, and ``--overwrite`` was not passed."""
    if not out_dir.exists():
        return True
    if overwrite:
        return True
    contents = [p.name for p in out_dir.iterdir()]
    if not contents:
        return True
    print(
        f"Note: {out_dir} is not empty; skipping (pass --overwrite to redo). "
        f"Contents: {contents[:5]}",
        file=sys.stderr,
    )
    return False


def _run_one_task(
    *,
    task_label_for_log: str,           # human-friendly: "Heparin (start)" or e.g. "binary"
    task_description: str,
    target_task: int | None,           # multitask task idx, or None for binary
    task_code: str | None,             # raw MEDS code for multitask, or None
    horizon_days: float | None,
    anchor_offset_hours: float | None,
    labels: np.ndarray,                # 1-D int
    artifacts_np: dict,                # ALREADY ROW-ALIGNED to labels
    val_schema: pl.DataFrame,          # ALREADY ROW-ALIGNED to labels
    retrieval_ds,
    doc_id_to_row: dict,
    corpus_size: int,
    k_docs: int,
    timeline_renderer: PatientTimelineRenderer,
    demo_by_sid: dict[int, dict],
    timelines_cache: dict[int, str],
    clinical_summaries_cache: dict[int, dict],
    out_dir: Path,
    families: tuple[str, ...],
    args: argparse.Namespace,
    save_sample_prompt_to: Path | None,
    show_sample_prompt: bool,
) -> dict | None:
    """Run the per-task LLM-judge pipeline. Returns a per-task summary row for
    the cross-task aggregator, or ``None`` if the task was skipped.
    """
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        print(
            f"[{task_label_for_log}] skipped: need both classes present "
            f"(n_pos={n_pos}, n_neg={n_neg}, n_valid={labels.size}).",
            file=sys.stderr,
        )
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    if not _check_out_dir_empty(out_dir, overwrite=args.overwrite or args.dry_run):
        return None

    print(
        f"\n=== [{task_label_for_log}] n_valid={labels.size} "
        f"(pos={n_pos}, neg={n_neg}) ==="
    )
    print(f"task_description: {task_description}")

    # --- Build pairs for this task --------------------------------------
    pairs = build_pairs(
        artifacts=artifacts_np,
        val_schema=val_schema,
        labels=labels,
        families=families,
        n_patients=args.n_patients,
        pairs_per_patient_per_family=args.pairs_per_patient_per_family,
        corpus_size=corpus_size,
        k=k_docs,
        seed=args.seed,
    )
    print(f"pairs built: {len(pairs)}")
    if len(pairs) > args.max_total_calls_cap:
        print(
            f"[{task_label_for_log}] ERROR: {len(pairs)} pairs exceeds "
            f"--max_total_calls_cap={args.max_total_calls_cap}; lower --n_patients "
            f"or raise the cap.",
            file=sys.stderr,
        )
        return None
    if not pairs:
        print(f"[{task_label_for_log}] skipped: build_pairs returned 0.", file=sys.stderr)
        return None

    # --- Sample timeline + cost estimate --------------------------------
    anchor_sid_sample = pairs[0].anchor_subject_id
    pred_t = (
        val_schema.filter(pl.col("subject_id") == anchor_sid_sample)["prediction_time"]
        .to_list()[0]
    )
    if int(anchor_sid_sample) in clinical_summaries_cache:
        sample_summary = clinical_summaries_cache[int(anchor_sid_sample)]
    else:
        sample_summary = compute_patient_clinical_summary(
            int(anchor_sid_sample), pred_t, args.meds_cohort
        )
        clinical_summaries_cache[int(anchor_sid_sample)] = sample_summary
    if int(anchor_sid_sample) in timelines_cache:
        sample_timeline = timelines_cache[int(anchor_sid_sample)]
    else:
        sample_timeline = timeline_renderer.render_categorical(
            anchor_sid_sample,
            pred_t,
            args.meds_cohort,
            demographics=demo_by_sid.get(int(anchor_sid_sample)),
            clinical_summary=sample_summary,
        )
        timelines_cache[int(anchor_sid_sample)] = sample_timeline
    sample_timeline_chrono_len = len(
        timeline_renderer.render(
            anchor_sid_sample,
            pred_t,
            args.meds_cohort,
            demographics=demo_by_sid.get(int(anchor_sid_sample)),
            clinical_summary=sample_summary,
        )
    )

    prompt_builder = JudgePromptBuilder(
        task_description=task_description,
        timeline_renderer=timeline_renderer,
        retrieval_ds=retrieval_ds,
        doc_id_to_row=doc_id_to_row,
        max_doc_chars=args.doc_max_chars,
    )
    sample_sys, sample_user = prompt_builder.build(pairs[0], patient_timeline=sample_timeline)
    sample_tokens = _estimate_tokens(sample_sys) + _estimate_tokens(sample_user)
    est_output_tokens = 50
    total_input = sample_tokens * len(pairs)
    total_output = est_output_tokens * len(pairs)
    in_price, out_price = _PRICE_PER_1K.get(args.model, (0.0, 0.0))
    est_cost = (total_input / 1000.0) * in_price + (total_output / 1000.0) * out_price
    print(
        f"est tokens: ~{total_input / 1e6:.2f}M input + ~{total_output / 1e3:.1f}K output | "
        f"est cost ~${est_cost:.2f}"
    )

    sample_prompt_text = _format_sample_prompt(
        sample_sys, sample_user, sample_timeline, sample_timeline_chrono_len
    )
    # Always save the sample prompt for pair 0; explicit override path wins.
    sample_prompt_path = save_sample_prompt_to or (out_dir / "sample_prompt.txt")
    sample_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    sample_prompt_path.write_text(sample_prompt_text)
    print(f"Saved sample prompt to {sample_prompt_path}")

    if args.dry_run:
        if show_sample_prompt:
            print()
            print(sample_prompt_text, end="")
        return {
            "target_task": target_task,
            "task_code": task_code,
            "task_label": task_label_for_log,
            "n_filtered": int(labels.size),
            "n_pairs": len(pairs),
            "est_cost_usd": est_cost,
            "summary_df": None,
            "dry_run": True,
        }

    # --- Render timelines for unique anchors ----------------------------
    unique_anchor_sids = {p.anchor_subject_id for p in pairs}
    print(f"rendering timelines for {len(unique_anchor_sids)} unique anchors...")
    schema_lookup = {
        int(r["subject_id"]): r["prediction_time"]
        for r in val_schema.iter_rows(named=True)
        if int(r["subject_id"]) in unique_anchor_sids
    }
    for sid in unique_anchor_sids:
        sid_int = int(sid)
        if sid_int in timelines_cache:
            continue
        pred_t = schema_lookup.get(sid_int)
        if pred_t is None:
            timelines_cache[sid_int] = ""
            continue
        summary = clinical_summaries_cache.get(sid_int)
        if summary is None:
            summary = compute_patient_clinical_summary(sid_int, pred_t, args.meds_cohort)
            clinical_summaries_cache[sid_int] = summary
        timelines_cache[sid_int] = timeline_renderer.render_categorical(
            sid_int,
            pred_t,
            args.meds_cohort,
            demographics=demo_by_sid.get(sid_int),
            clinical_summary=summary,
        )

    # --- Run the judge --------------------------------------------------
    judge = OpenAIJudge(model=args.model)
    verdicts_df = run_judge(
        pairs,
        judge=judge,
        prompt_builder=prompt_builder,
        timelines_by_subject_id=timelines_cache,
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
        demographics=None,  # demographics already inlined into rollup via cache; pass None to keep API
        retrieval_ds=retrieval_ds,
        doc_id_to_row=doc_id_to_row,
        timelines_by_subject_id=timelines_cache,
        clinical_summaries_by_subject_id=clinical_summaries_cache,
        families=families,
    )
    human_df = build_human_validation_subset(
        verdicts_df,
        n=args.human_validation_n,
        seed=args.seed,
        retrieval_ds=retrieval_ds,
        doc_id_to_row=doc_id_to_row,
    )

    # F3/F4 collision diagnostic
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
                f"[{task_label_for_log}][{fam}] collision/dedupe drops: "
                f"{dropped}/{expected} ({100 * dropped / max(expected, 1):.1f}%).",
                file=sys.stderr,
            )

    verdicts_df.write_csv(out_dir / "pairs_verdicts.csv")
    per_patient_df.write_csv(out_dir / "per_patient_results.csv")
    summary_df.write_csv(out_dir / "family_winrates.csv")
    human_df.write_csv(out_dir / "human_validation.csv")

    run_config = {
        "task_mode": "multitask" if target_task is not None else "binary",
        "task_description": task_description,
        "families": list(families),
        "n_patients": args.n_patients,
        "pairs_per_patient_per_family": args.pairs_per_patient_per_family,
        "model": args.model,
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
        "n_pairs_actual": len(pairs),
        "k": k_docs,
    }
    if target_task is not None:
        run_config.update(
            {
                "target_task": target_task,
                "task_code": task_code,
                "task_label": task_label_for_log,
                "task_filtered_n": int(labels.size),
                "task_n_pos": n_pos,
                "task_n_neg": n_neg,
                "horizon_days": horizon_days,
                "anchor_offset_hours": anchor_offset_hours,
            }
        )
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str))

    write_results_workbook(
        out_dir / "llm_judge_results.xlsx",
        family_winrates=summary_df,
        per_patient=per_patient_df,
        pairs_verdicts=verdicts_df,
        human_validation=human_df,
    )
    print(f"Wrote {task_label_for_log} results to {out_dir}")

    return {
        "target_task": target_task,
        "task_code": task_code,
        "task_label": task_label_for_log,
        "n_filtered": int(labels.size),
        "n_pairs": len(pairs),
        "est_cost_usd": est_cost,
        "summary_df": summary_df,
        "dry_run": False,
    }


def _write_cross_task_summary(
    per_task: list[dict],
    out_root: Path,
    *,
    args: argparse.Namespace,
    n_tasks_attempted: int,
) -> None:
    """Write ``all_task_winrates{,_wide}.csv`` and ``sweep_config.json``."""
    completed = [t for t in per_task if not t.get("dry_run") and t.get("summary_df") is not None]
    if not completed:
        print("No completed tasks; skipping cross-task summary.", file=sys.stderr)
        return

    # --- Long format ----------------------------------------------------
    long_rows: list[dict] = []
    for t in completed:
        for row in t["summary_df"].iter_rows(named=True):
            long_rows.append(
                {
                    "task_idx": t["target_task"],
                    "task_code": t["task_code"],
                    "task_label": t["task_label"],
                    "n_filtered_patients": t["n_filtered"],
                    **row,
                }
            )
    long_df = pl.DataFrame(long_rows)
    long_df.write_csv(out_root / "all_task_winrates.csv")
    print(f"Wrote {out_root / 'all_task_winrates.csv'} ({long_df.height} rows)")

    # --- Wide format (paper-figure friendly) ---------------------------
    families_observed = sorted(long_df["family"].unique().to_list())
    wide_rows: list[dict] = []
    for t in completed:
        row: dict = {
            "task_idx": t["target_task"],
            "task_code": t["task_code"],
            "task_label": t["task_label"],
            "n_filtered_patients": t["n_filtered"],
        }
        summary_df = t["summary_df"]
        for fam in families_observed:
            fam_row = summary_df.filter(pl.col("family") == fam)
            if fam_row.height == 0:
                row[f"{fam}_rate"] = None
                row[f"{fam}_ci_low"] = None
                row[f"{fam}_ci_high"] = None
                row[f"{fam}_n_pairs"] = None
            else:
                row[f"{fam}_rate"] = float(fam_row["target_preferred_rate"][0])
                row[f"{fam}_ci_low"] = float(fam_row["ci_low"][0])
                row[f"{fam}_ci_high"] = float(fam_row["ci_high"][0])
                row[f"{fam}_n_pairs"] = int(fam_row["n_pairs"][0])
        wide_rows.append(row)
    wide_df = pl.DataFrame(wide_rows)
    wide_df.write_csv(out_root / "all_task_winrates_wide.csv")
    print(f"Wrote {out_root / 'all_task_winrates_wide.csv'} ({wide_df.height} rows)")

    # --- Sweep config ---------------------------------------------------
    sweep_config = {
        "task_mode": "multitask",
        "n_tasks_attempted": n_tasks_attempted,
        "n_tasks_completed": len(completed),
        "n_tasks_skipped": n_tasks_attempted - len(completed),
        "skipped_task_idx": [
            t["target_task"] for t in per_task
            if t.get("dry_run") is False and t.get("summary_df") is None
        ],
        "model": args.model,
        "seed": args.seed,
        "max_total_calls_cap_per_task": args.max_total_calls_cap,
        "n_patients": args.n_patients,
        "pairs_per_patient_per_family": args.pairs_per_patient_per_family,
        "families": [f.strip() for f in args.families.split(",") if f.strip()],
        "total_est_cost_usd": float(sum(t.get("est_cost_usd", 0.0) for t in completed)),
    }
    (out_root / "sweep_config.json").write_text(json.dumps(sweep_config, indent=2, default=str))
    print(f"Wrote {out_root / 'sweep_config.json'}")


def main() -> None:  # noqa: C901 - CLI orchestration, branches are linear
    args = _parse_args()

    run_dir: Path = args.run_dir
    if not (run_dir / "config.yaml").is_file():
        print(f"Error: {run_dir / 'config.yaml'} not found.", file=sys.stderr)
        sys.exit(1)

    families = tuple(f.strip() for f in args.families.split(",") if f.strip())

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY is not set. Export it before running, or pass "
            "--dry_run to estimate cost without calling the API.",
            file=sys.stderr,
        )
        sys.exit(5)

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

    raw_targets = _as_numpy(artifacts["targets"])
    task_mode = _resolve_task_mode(args.task_mode, raw_targets)
    print(f"Task mode: {task_mode} (requested: {args.task_mode})")

    if task_mode == "binary":
        if args.target_task is not None:
            print("Error: --target_task is rejected for binary runs.", file=sys.stderr)
            sys.exit(2)
        binary_task_description = _resolve_task_description(args)
        if binary_task_description is None:
            print(
                "Error: --task_description (or --task_description_file) is required for "
                "binary runs.",
                file=sys.stderr,
            )
            sys.exit(2)
        labels = raw_targets.astype(int)
        unique_labels = set(np.unique(labels).tolist())
        if not {0, 1}.issubset(unique_labels):
            print(
                f"ERROR: expected both classes {{0, 1}} in binary targets; saw "
                f"{unique_labels}.",
                file=sys.stderr,
            )
            sys.exit(2)
    else:  # multitask
        if raw_targets.ndim != 2:
            print(
                f"Error: --task_mode multitask requires 2-D targets; got shape "
                f"{raw_targets.shape}.",
                file=sys.stderr,
            )
            sys.exit(2)

    # Common loads (used by both modes).
    retrieval_ds = load_from_disk(str(args.retrieval_db))
    corpus_size = len(retrieval_ds)
    doc_id_to_row = _build_doc_id_to_row_map(retrieval_ds)
    if doc_id_to_row is None:
        doc_id_to_row = {i: i for i in range(corpus_size)}

    doc_ids_np = _as_numpy(doc_ids_tensor)
    k_docs = int(doc_ids_np.shape[2]) if doc_ids_np.ndim == 3 else 1

    print(f"Corpus size:     {corpus_size}")
    print(f"Artifact rows:   {doc_ids_np.shape[0]}")
    print(f"Retrieval k:     {k_docs}")
    print(f"Families:        {families}")

    subject_ids = val_schema["subject_id"].unique().to_list()
    demographics = load_subject_demographics(args.meds_cohort, subject_ids)

    timeline_renderer = PatientTimelineRenderer(
        codes_parquet=codes_parquet,
        max_events=args.timeline_max_events,
    )
    demo_by_sid: dict[int, dict] = {}
    if demographics is not None and demographics.height > 0:
        for rec in demographics.iter_rows(named=True):
            demo_by_sid[int(rec["subject_id"])] = rec
    timelines_cache: dict[int, str] = {}
    clinical_summaries_cache: dict[int, dict] = {}

    artifacts_np_full = {
        "doc_ids": doc_ids_np,
        "doc_scores": _as_numpy(artifacts["doc_scores"]),
        "logits": _as_numpy(artifacts.get("logits", np.zeros((doc_ids_np.shape[0], 2)))),
    }

    if task_mode == "binary":
        out_dir = args.out_dir or (run_dir / "llm_judge")
        artifacts_np = {**artifacts_np_full, "targets": labels}
        result = _run_one_task(
            task_label_for_log="binary",
            task_description=binary_task_description,
            target_task=None,
            task_code=None,
            horizon_days=None,
            anchor_offset_hours=None,
            labels=labels,
            artifacts_np=artifacts_np,
            val_schema=val_schema,
            retrieval_ds=retrieval_ds,
            doc_id_to_row=doc_id_to_row,
            corpus_size=corpus_size,
            k_docs=k_docs,
            timeline_renderer=timeline_renderer,
            demo_by_sid=demo_by_sid,
            timelines_cache=timelines_cache,
            clinical_summaries_cache=clinical_summaries_cache,
            out_dir=out_dir,
            families=families,
            args=args,
            save_sample_prompt_to=args.save_sample_prompt,
            show_sample_prompt=True,
        )
        if result is None:
            sys.exit(2)
        if not args.dry_run and result.get("summary_df") is not None:
            print()
            print(result["summary_df"])
        return

    # ---- Multitask sweep ----------------------------------------------
    mt_labels_dir_str = OmegaConf.select(cfg, "training.datamodule.mt_labels_dir")
    if mt_labels_dir_str is None:
        print(
            "Error: multitask run config missing training.datamodule.mt_labels_dir; "
            "cannot find code_index.json / metadata.json.",
            file=sys.stderr,
        )
        sys.exit(2)
    mt_labels_dir = Path(mt_labels_dir_str)
    metadata_path = mt_labels_dir / "metadata.json"
    if not metadata_path.is_file():
        print(f"Error: missing {metadata_path}.", file=sys.stderr)
        sys.exit(2)
    metadata = json.loads(metadata_path.read_text())
    n_tasks = int(metadata["num_tasks"])
    horizon_days = float(metadata["horizon_days"])
    anchor_offset_hours = float(metadata["anchor_offset_hours"])
    if raw_targets.shape[1] != n_tasks:
        print(
            f"Error: artifact targets has {raw_targets.shape[1]} columns but "
            f"metadata says num_tasks={n_tasks}.",
            file=sys.stderr,
        )
        sys.exit(2)

    lab_lookup = _load_lab_label_lookup(args.mimic_labitems_path)
    if lab_lookup is None:
        print(
            f"Note: d_labitems lookup not loaded from {args.mimic_labitems_path}. "
            "LAB tasks will be described as 'lab test with internal item id <N>'.",
            file=sys.stderr,
        )
    else:
        print(f"Loaded d_labitems lookup ({len(lab_lookup)} entries) from {args.mimic_labitems_path}.")

    task_codes = _load_task_codes(run_dir)
    if task_codes is None:
        print(
            "Note: code_index.json not found at the multitask labels dir; tasks "
            "will be labeled by index only.",
            file=sys.stderr,
        )

    if args.target_task is not None:
        if not 0 <= args.target_task < n_tasks:
            print(
                f"Error: --target_task={args.target_task} out of range [0, {n_tasks}).",
                file=sys.stderr,
            )
            sys.exit(2)
        tasks_to_run = [args.target_task]
    else:
        tasks_to_run = list(range(n_tasks))

    out_root = args.out_dir or (run_dir / "llm_judge")
    out_root.mkdir(parents=True, exist_ok=True)

    # User-provided --task_description overrides auto-gen for every task in the
    # sweep — useful for ablations.
    user_override_description = _resolve_task_description(args)

    print(f"\nMultitask sweep: {len(tasks_to_run)} task(s) → {out_root}")
    if user_override_description is not None:
        print(f"NOTE: --task_description overrides auto-generated descriptions for ALL tasks.")

    per_task_results: list[dict] = []
    for i, task_idx in enumerate(tasks_to_run):
        labels, valid_indices, task_meta = _select_target_task(
            raw_targets,
            task_idx,
            task_codes=task_codes,
            lab_lookup=lab_lookup,
            horizon_days=horizon_days,
            anchor_offset_hours=anchor_offset_hours,
        )
        # Subset row-aligned arrays.
        artifacts_np = {
            "doc_ids": artifacts_np_full["doc_ids"][valid_indices],
            "doc_scores": artifacts_np_full["doc_scores"][valid_indices],
            "logits": artifacts_np_full["logits"][valid_indices],
            "targets": labels,
        }
        # Subset val_schema (polars positional row index).
        val_schema_subset = val_schema[valid_indices.tolist()]

        task_dir = out_root / f"task{task_idx}"
        # In multitask mode, `--save_sample_prompt` is ignored — every task
        # saves to its own subdir to keep them from clobbering one another.
        sample_prompt_path: Path | None = task_dir / "sample_prompt.txt"

        description = user_override_description or task_meta["task_description"]

        result = _run_one_task(
            task_label_for_log=f"task {task_idx} ({task_meta['task_label']})",
            task_description=description,
            target_task=task_idx,
            task_code=task_meta["task_code"],
            horizon_days=horizon_days,
            anchor_offset_hours=anchor_offset_hours,
            labels=labels,
            artifacts_np=artifacts_np,
            val_schema=val_schema_subset,
            retrieval_ds=retrieval_ds,
            doc_id_to_row=doc_id_to_row,
            corpus_size=corpus_size,
            k_docs=k_docs,
            timeline_renderer=timeline_renderer,
            demo_by_sid=demo_by_sid,
            timelines_cache=timelines_cache,
            clinical_summaries_cache=clinical_summaries_cache,
            out_dir=task_dir,
            families=families,
            args=args,
            save_sample_prompt_to=sample_prompt_path,
            show_sample_prompt=(i == 0),  # only the first task in a dry run
        )
        if result is not None:
            per_task_results.append(result)
        if args.dry_run and i == 0:
            # Show one sample prompt and exit-summary; don't run the loop further.
            n_remaining = len(tasks_to_run) - 1
            if n_remaining > 0 and result is not None:
                approx_total = result["est_cost_usd"] * len(tasks_to_run)
                print(
                    f"\nDry-run: showed prompt for the first task. Sweep would run "
                    f"{len(tasks_to_run)} tasks; approximate total cost (assuming "
                    f"similar per-task) ~${approx_total:.2f}."
                )
            print("Dry-run: exiting before API calls.")
            return

    if not args.dry_run:
        _write_cross_task_summary(
            per_task_results,
            out_root,
            args=args,
            n_tasks_attempted=len(tasks_to_run),
        )


if __name__ == "__main__":
    main()
