#!/usr/bin/env bash
# ============================================================
# SLURM: aggregate the 200 (task, rank) cells written by
#         run_llm_judge_rank_sweep_array_slurm.sh, then render
#         the per-rank plots.
# ------------------------------------------------------------
# This script is the second half of the rank-sweep pipeline. It
# expects all 200 array elements to have completed; submit it
# with --dependency=afterok:<array_job_id> via the launcher
# (scripts/submit_llm_judge_rank_sweep.sh).
#
# Outputs:
#   <run_dir>/llm_judge/all_task_winrates.csv  (200 rows: 25 tasks × 8 ranks)
#   <run_dir>/llm_judge/winrates_per_task_F1_rank{1..8}.pdf
# ============================================================

#SBATCH --job-name=medrap-llm-judge-aggregate
#SBATCH --partition=cpu
#SBATCH --account=mm6677_gp
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"

RUN_DIR="${REPO_DIR}/outputs/mt_rope_cross_attention"

echo "=== Job info ==="
echo "  Job ID  : ${SLURM_JOB_ID:-local}"
echo "  Node    : ${SLURMD_NODENAME:-$(hostname)}"
echo "  Started : $(date)"
echo ""

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"

mkdir -p logs

echo "=== Step 1/2: aggregate per-(task, rank) CSVs ==="
uv run python scripts/aggregate_llm_judge_rank_sweep.py \
    --run_dir "${RUN_DIR}"

echo ""
echo "=== Step 2/2: render 8 per-rank PDFs ==="
uv run python scripts/plot_llm_judge_winrates.py \
    --run_dir "${RUN_DIR}" \
    --rank_sweep

echo ""
echo "=== Done: $(date) ==="
echo "  See ${RUN_DIR}/llm_judge/winrates_per_task_F1_rank{1..8}.pdf"
