#!/usr/bin/env bash
# ============================================================
# RoPE encoder + Cross-Attention fusion + Token-Feature retrieval (MIMIC)
# ------------------------------------------------------------
# The current "frontier" stack on main as of PR #39 (cross-attention)
# and PR #42 (time-delta RoPE). Patient sequence cross-attends to
# retrieved document tokens, so marginalized retrieval is NOT used
# here (cross-attention produces a single fused state, not per-doc
# predictions).
#
# Requires a prepared HF dataset + FAISS under data/retrieval_db with
# doc_tokens / doc_attention_mask / doc_key_embeddings.
#
# Usage:
#   sbatch scripts/run_rope_cross_attention.sh
#   sbatch scripts/run_rope_cross_attention.sh training.trainer.max_epochs=10
#
# Extra arguments are forwarded to `medrap train` as Hydra overrides.
# ============================================================

#SBATCH --job-name=medrap-rope-cross-attn
#SBATCH --partition=gpu
#SBATCH --account=mm6677_gp
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"
RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
# MEDS cohort root for meds_torchdata: must contain data/{split}/*.nrt, tokenization/schemas, metadata/, etc.
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
# Task label parquet(s) live under the repo.
TASK_LABELS_DIR="${REPO_DIR}/data/task_labels/mortality/in_icu/first_24h"
OUTPUT_DIR="${REPO_DIR}/outputs/mimic_run_rope_cross_attention"

echo "=== Job info ==="
echo "  Job ID   : ${SLURM_JOB_ID:-local}"
echo "  Node     : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)   : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  Started  : $(date)"
echo ""

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"

mkdir -p logs "${OUTPUT_DIR}"

echo "=== Starting medrap train ==="

medrap train \
    encoder=rope \
    query_projector=sequence_mean_1024 \
    query_projector.in_dim=128 \
    retriever=hf_dataset \
    "retriever.dataset_path=${RETRIEVAL_DB}" \
    retriever.doc_ids_column=null \
    retriever.k=8 \
    retrieval_encoder=token_feature \
    retrieval_encoder.vocab_size=151936 \
    retrieval_encoder.embedding_dim=64 \
    fusion=cross_attention_medium \
    fusion.d_in_patient=128 \
    fusion.d_in_doc=64 \
    pooling=masked_mean \
    head=linear \
    head.in_dim=256 \
    head.out_dim=1 \
    training/task=binary_classification \
    training/loss=binary_bce \
    training/datamodule=meds \
    "training.datamodule.config.tensorized_cohort_dir=${TENSORIZED_DIR}" \
    training.datamodule.config.max_seq_len=256 \
    "training.datamodule.config.task_labels_dir=${TASK_LABELS_DIR}" \
    training.datamodule.batch_size=32 \
    training.datamodule.config.seq_sampling_strategy=to_end \
    training/trainer=lightning_wandb \
    training.trainer.max_epochs=5 \
    training.trainer.accelerator=gpu \
    training.trainer.devices=1 \
    training.trainer.gradient_clip_val=1.0 \
    training.trainer.log_every_n_steps=10 \
    training.module.lr=1e-4 \
    training.module.warmup_steps=200 \
    "wandb_run_name=rope-cross-attn-${SLURM_JOB_ID:-local}" \
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true \
    "$@"

echo ""
echo "=== Generating keyword x demographic heatmap ==="
uv run python scripts/run_demographic_heatmap.py \
    --run_dir "${OUTPUT_DIR}" \
    --retrieval_db "${RETRIEVAL_DB}" \
    --meds_cohort /groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort

echo ""
echo "=== Done: $(date) ==="
