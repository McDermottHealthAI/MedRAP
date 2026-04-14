#!/usr/bin/env bash
# ============================================================
# Patient-only prediction (no retrieval)
# ------------------------------------------------------------
# Predicts from patient encodings only. Uses a small transformer encoder
# (2-layer, 4-head, 128-D, pre-norm) + sequence_mean_1024. Each token
# representation is contextualised via self-attention before mean-pooling,
# giving the query projector richer per-event information than raw embeddings.
# The query embedding (B, 1024) is fed straight to the head,
# bypassing retrieval, fusion, and pooling entirely.
#
# Usage:
#   sbatch scripts/run_patient_only.sh
#   sbatch scripts/run_patient_only.sh training.trainer.max_epochs=20
#
# Extra arguments are forwarded to `medrap train` as Hydra overrides.
# ============================================================

#SBATCH --job-name=medrap-patient-only
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
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
TASK_LABELS_DIR="${REPO_DIR}/data/task_labels/mortality/in_icu/first_24h"
OUTPUT_DIR="${REPO_DIR}/outputs/mimic_run_patient_only"

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

echo "=== Starting medrap train (patient-only) ==="

medrap train \
    patient_only=true \
    encoder=transformer_128 \
    encoder.dropout=0.3 \
    query_projector=sequence_mean_1024 \
    query_projector.in_dim=128 \
    head=linear_1024_to_2 \
    training/task=binary_classification \
    training/loss=binary_bce \
    training.module.lr=3e-4 \
    training.module.weight_decay=0.1 \
    training/datamodule=meds \
    "training.datamodule.config.tensorized_cohort_dir=${TENSORIZED_DIR}" \
    training.datamodule.config.max_seq_len=128 \
    "training.datamodule.config.task_labels_dir=${TASK_LABELS_DIR}" \
    training.datamodule.batch_size=32 \
    training.datamodule.config.seq_sampling_strategy=to_end \
    training/trainer=lightning_wandb \
    training.trainer.max_epochs=10 \
    training.trainer.accelerator=gpu \
    training.trainer.devices=1 \
    training.trainer.log_every_n_steps=10 \
    "+training.trainer.callbacks=[{_target_: lightning.pytorch.callbacks.EarlyStopping, monitor: val/loss, patience: 3, mode: min}]" \
    "wandb_run_name=patient-only-${SLURM_JOB_ID:-local}" \
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true \
    "$@"

echo ""
echo "=== Done: $(date) ==="
