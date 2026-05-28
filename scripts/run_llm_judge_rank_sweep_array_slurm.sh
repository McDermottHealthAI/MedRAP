#!/usr/bin/env bash
# ============================================================
# SLURM: LLM-judge F1 rank sweep — 200-element ARRAY (one per
#         (task, rank) cell). Pair with the aggregator below.
# ------------------------------------------------------------
# Each element of the array runs ONE (task, rank) combination
# of the F1 random-doc counterfactual:
#   TASK = SLURM_ARRAY_TASK_ID / 8     (0..24)
#   RANK = SLURM_ARRAY_TASK_ID % 8     (0..7)
#
# Each element writes to its own subdir to avoid race conditions:
#   <run_dir>/llm_judge/task<K>_rank<R>/
#
# Aggregation + plotting happens in the dependent companion job
# (scripts/aggregate_llm_judge_rank_sweep_slurm.sh). Use the
# launcher (scripts/submit_llm_judge_rank_sweep.sh) to submit both
# together with the right dependency.
#
# Prerequisites:
#   - Trained multitask checkpoint at outputs/mt_rope_cross_attention/.
#   - OPENAI_API_KEY exported in the submitting shell.
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   bash scripts/submit_llm_judge_rank_sweep.sh
# ============================================================

#SBATCH --job-name=medrap-llm-judge-rank-array
#SBATCH --partition=cpu
#SBATCH --account=mm6677_gp
#SBATCH --array=0-199%50
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"

RUN_DIR="${REPO_DIR}/outputs/mt_rope_cross_attention"
RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
MEDS_COHORT="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort"

TASK_IDX=$((SLURM_ARRAY_TASK_ID / 8))
RANK_IDX=$((SLURM_ARRAY_TASK_ID % 8))

echo "=== Job info ==="
echo "  Array Job : ${SLURM_ARRAY_JOB_ID}"
echo "  Array ID  : ${SLURM_ARRAY_TASK_ID}"
echo "  Task / Rank: ${TASK_IDX} / ${RANK_IDX}"
echo "  Node      : ${SLURMD_NODENAME:-$(hostname)}"
echo "  Started   : $(date)"
echo ""

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set. Export it before sbatch." >&2
    exit 1
fi

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "ERROR: ${RUN_DIR} not found. Train the multitask model first." >&2
    exit 1
fi

cd "${REPO_DIR}"
# shellcheck source=/dev/null
source "${VENV}"

mkdir -p logs

uv run python scripts/run_llm_judge.py \
    --run_dir "${RUN_DIR}" \
    --retrieval_db "${RETRIEVAL_DB}" \
    --meds_cohort "${MEDS_COHORT}" \
    --task_mode multitask \
    --target_task "${TASK_IDX}" \
    --target_rank "${RANK_IDX}" \
    --rank_sweep \
    --overwrite \
    "$@"

echo ""
echo "=== Done: $(date) ==="
