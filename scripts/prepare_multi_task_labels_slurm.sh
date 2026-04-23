#!/usr/bin/env bash
# ============================================================
# SLURM: Prepare multi-task binary labels from MEDS cohort.
# ------------------------------------------------------------
# Run this ONCE before training. Derives one prediction_time
# anchor per patient directly from the EHR (first event + 24h),
# then for each of the top-N codes labels whether the code
# appears within horizon_days after prediction_time.
#
# No external task-label file required — works on the full
# patient population, not just those with mortality labels.
#
# Outputs per-split parquets to ${MT_LABELS_DIR}:
#   subject_id | prediction_time | task_0 | ... | task_{N-1}
#
# Usage:
#   sbatch scripts/prepare_multi_task_labels_slurm.sh
#   sbatch scripts/prepare_multi_task_labels_slurm.sh --num_tasks 100
#
# Extra arguments are forwarded to prepare_multi_task_labels.py.
# ============================================================

#SBATCH --job-name=medrap-prep-mt-labels
#SBATCH --partition=cpu
#SBATCH --account=mm6677_gp
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"

MEDS_COHORT_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort"
MT_LABELS_DIR="${REPO_DIR}/data/mt_labels/top25_7d"

NUM_TASKS=25
HORIZON_DAYS=7
ANCHOR_OFFSET_HOURS=24

echo "=== Job info ==="
echo "  Job ID  : ${SLURM_JOB_ID:-local}"
echo "  Node    : ${SLURMD_NODENAME:-$(hostname)}"
echo "  Started : $(date)"
echo ""

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"

mkdir -p logs "${MT_LABELS_DIR}"

echo "=== Preparing multi-task labels ==="
echo "  MEDS cohort          : ${MEDS_COHORT_DIR}"
echo "  Output dir           : ${MT_LABELS_DIR}"
echo "  num_tasks            : ${NUM_TASKS}"
echo "  horizon_days         : ${HORIZON_DAYS}"
echo "  anchor_offset_hours  : ${ANCHOR_OFFSET_HOURS}"
echo ""

python scripts/prepare_multi_task_labels.py \
    --meds_cohort_dir     "${MEDS_COHORT_DIR}" \
    --output_dir          "${MT_LABELS_DIR}" \
    --num_tasks           "${NUM_TASKS}" \
    --horizon_days        "${HORIZON_DAYS}" \
    --anchor_offset_hours "${ANCHOR_OFFSET_HOURS}" \
    "$@"

echo ""
echo "=== Done: $(date) ==="
echo "Labels saved to ${MT_LABELS_DIR}"
