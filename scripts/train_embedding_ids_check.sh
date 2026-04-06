#!/bin/bash
# ============================================================
# Synthetic marginalized retrieval sanity check.
#
# Trains a small model on structured synthetic data, then
# extracts retrieval artifacts and generates diagnostic plots.
#
# The synthetic setup: 16 patients (8 pos, 8 neg) with 8 docs
# (4 positive-relevant, 4 negative-relevant). If the algorithm
# works, the model learns class-conditional retrieval and the
# diagnostic plots show clear separation.
#
# Usage:
#   bash scripts/train_embedding_ids_check.sh
#   sbatch scripts/train_embedding_ids_check.sh
# ============================================================

#SBATCH --job-name=medrap-margin-check
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/margin_check_%j.out
#SBATCH --error=logs/margin_check_%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO_DIR}"

echo "=== Marginalized Retrieval Sanity Check ==="
echo "  Node:  $(hostname)"
echo "  Start: $(date)"
echo "  Repo:  ${REPO_DIR}"
echo ""

source .venv/bin/activate

RUN_ID="${SLURM_JOB_ID:-$(date +%s)}"
TRAIN_DIR="outputs/margin_check_${RUN_ID}"

mkdir -p logs

# ------------------------------------------------------------------
# Phase 1: Train
# ------------------------------------------------------------------
echo "=== Phase 1: Training ==="

medrap train \
    training/datamodule=synthetic_marginalized \
    retriever=in_memory_sanity \
    retrieval_encoder=key_embedding \
    training/task=marginalized_binary \
    training/loss=marginalized_retrieval \
    marginalized_retrieval=true \
    marginalized_score_similarity=dot \
    query_projector.in_dim=1 \
    query_projector.out_dim=8 \
    head.in_dim=8 \
    head.out_dim=2 \
    training/trainer=lightning_default \
    training.trainer.max_epochs=100 \
    training.trainer.accelerator=cpu \
    training.trainer.log_every_n_steps=1 \
    "output_dir=${TRAIN_DIR}" \
    do_overwrite=true

echo ""
echo "=== Phase 1 complete. Training output: ${TRAIN_DIR} ==="
echo ""

# ------------------------------------------------------------------
# Phase 2: Extract + Visualize
# ------------------------------------------------------------------
echo "=== Phase 2: Extraction + Visualization ==="

python scripts/extract_and_visualize.py --run_dir "${TRAIN_DIR}"

echo ""
echo "=== Done: $(date) ==="
echo "  Training output:  ${TRAIN_DIR}"
echo "  Artifacts:        ${TRAIN_DIR}/extraction/extraction_artifacts.pt"
echo "  Plots:            ${TRAIN_DIR}/extraction/extraction_diagnostics.png"
