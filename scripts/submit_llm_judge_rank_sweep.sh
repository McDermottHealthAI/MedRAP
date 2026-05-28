#!/usr/bin/env bash
# ============================================================
# Launcher: submit the rank-sweep array job + dependent
#            aggregator/plot job in one shot.
# ------------------------------------------------------------
# Workflow:
#   1. Submit scripts/run_llm_judge_rank_sweep_array_slurm.sh
#      (200 elements, one per (task, rank) cell).
#   2. Submit scripts/aggregate_llm_judge_rank_sweep_slurm.sh
#      with --dependency=afterok:<array_id> so it runs only after
#      every array element succeeds.
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   bash scripts/submit_llm_judge_rank_sweep.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set. Export it before submitting." >&2
    exit 1
fi

echo "Submitting rank-sweep array job (200 elements)..."
ARRAY_ID=$(sbatch --parsable \
    --export=ALL \
    "${SCRIPT_DIR}/run_llm_judge_rank_sweep_array_slurm.sh")
echo "  Array job ID: ${ARRAY_ID}"

echo "Submitting aggregator job (depends on array completion)..."
AGG_ID=$(sbatch --parsable \
    --dependency=afterok:"${ARRAY_ID}" \
    --export=ALL \
    "${SCRIPT_DIR}/aggregate_llm_judge_rank_sweep_slurm.sh")
echo "  Aggregator job ID: ${AGG_ID}"

echo ""
echo "Monitor with:"
echo "  squeue -j ${ARRAY_ID},${AGG_ID}"
echo ""
echo "Final outputs (after aggregator completes):"
echo "  outputs/mt_rope_cross_attention/llm_judge/all_task_winrates.csv"
echo "  outputs/mt_rope_cross_attention/llm_judge/winrates_per_task_F1_rank{1..8}.pdf"
