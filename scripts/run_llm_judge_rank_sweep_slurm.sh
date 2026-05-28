#!/usr/bin/env bash
# ============================================================
# SLURM: LLM-judge F1 rank sweep (top-1..top-8 vs random)
# ------------------------------------------------------------
# Runs the multitask sweep with --rank_sweep so F1 emits one pair
# per (patient, retrieval rank), then plots one PDF per rank as a
# diagnostic for whether the judge's tie behavior is rank-dependent.
#
# Outputs land at outputs/mt_rope_cross_attention/llm_judge/:
#   - all_task_winrates.csv          (25 tasks × 8 ranks = 200 rows)
#   - winrates_per_task_F1_rank{1..8}.pdf
#   - task{0..24}/{family_winrates,pairs_verdicts,...}.csv
#
# Costs ~$3-4 at gpt-4o-mini (25 tasks × 8 ranks × 100 patients
# = 20,000 judge calls).
#
# Prerequisites:
#   - A trained multitask checkpoint at outputs/mt_rope_cross_attention/
#     (see scripts/train_multitask_rope_cross_attention_slurm.sh).
#   - OPENAI_API_KEY exported in the submitting shell — SLURM
#     propagates the env, so `export OPENAI_API_KEY=sk-...`
#     before `sbatch` is enough.
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   sbatch scripts/run_llm_judge_rank_sweep_slurm.sh
#
# Extra arguments are forwarded to `run_llm_judge.py` (e.g. to
# override --n_patients or --model).
# ============================================================

#SBATCH --job-name=medrap-llm-judge-rank-sweep
#SBATCH --partition=cpu
#SBATCH --account=mm6677_gp
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"

RUN_DIR="${REPO_DIR}/outputs/mt_rope_cross_attention"
RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
MEDS_COHORT="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort"

echo "=== Job info ==="
echo "  Job ID   : ${SLURM_JOB_ID:-local}"
echo "  Node     : ${SLURMD_NODENAME:-$(hostname)}"
echo "  CPUs     : ${SLURM_CPUS_PER_TASK:-<unset>}"
echo "  Started  : $(date)"
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

echo "=== Step 1/2: run_llm_judge.py --rank_sweep ==="
uv run python scripts/run_llm_judge.py \
    --run_dir "${RUN_DIR}" \
    --retrieval_db "${RETRIEVAL_DB}" \
    --meds_cohort "${MEDS_COHORT}" \
    --rank_sweep \
    --overwrite \
    "$@"

echo ""
echo "=== Step 2/2: plot_llm_judge_winrates.py --rank_sweep ==="
uv run python scripts/plot_llm_judge_winrates.py \
    --run_dir "${RUN_DIR}" \
    --rank_sweep

echo ""
echo "=== Done: $(date) ==="
echo "  See ${RUN_DIR}/llm_judge/winrates_per_task_F1_rank{1..8}.pdf"
