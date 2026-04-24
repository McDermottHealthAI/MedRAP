#!/usr/bin/env bash
# ============================================================
# Local hyperparameter sweep runner
# ------------------------------------------------------------
# Runs a named set of retrieval-only experiments sequentially for
# one task using the same entrypoint as scripts/run_retrieval_only_local.sh.
#
# Required environment variables:
#   TENSORIZED_DIR   Root tensorized MEDS cohort directory
#   TASK_LABELS_DIR  Directory containing train/tuning/held_out parquet labels
#
# Optional environment variables:
#   SWEEP_ROOT       Output root (default: REPO_DIR/outputs/sweep/<task_name>)
#   WANDB_PROJECT    WandB project name (default: medrap)
#   NUM_WORKERS      Datamodule num_workers override (default: 8)
#
# Usage:
#   export TENSORIZED_DIR=/data/tensorized
#   export TASK_LABELS_DIR=/data/task_labels/dementia
#   export RETRIEVAL_DB=/data/retrieval_db
#   scripts/sweep_local.sh baseline enc_32 ep_1
#
#   scripts/sweep_local.sh --all
#
# Extra Hydra overrides can be passed after `--` and are forwarded
# to each run. Example:
#   scripts/sweep_local.sh baseline k_32 -- training.trainer.enable_progress_bar=true
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

TASK_LABELS_DIR="${TASK_LABELS_DIR:-}"
TASK_NAME="${TASK_NAME:-}"
NUM_WORKERS="${NUM_WORKERS:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-medrap}"

if [[ -z "${TASK_LABELS_DIR}" || ! -d "${TASK_LABELS_DIR}" ]]; then
  echo "TASK_LABELS_DIR is required and must be a directory." >&2
  exit 1
fi

if [[ -z "${TASK_NAME}" ]]; then
  TASK_NAME="$(basename "${TASK_LABELS_DIR}")"
fi

SWEEP_ROOT="${SWEEP_ROOT:-${REPO_DIR}/outputs/sweep/${TASK_NAME}}"
mkdir -p "${SWEEP_ROOT}"

ALL_VARIANTS=(
  baseline
  enc_32
  enc_512
  ep_1
  ep_2
  ep_4
  k_01
  k_08
  k_16
  k_32
  k_64
  k_128
  lr_1e-4
  lr_3e-3
  seed_s12
  seed_s123
  seed_s456
  seed_s789
)

declare -a SELECTED_VARIANTS=()
declare -a EXTRA_ARGS=()
parse_extra=false

for arg in "$@"; do
  if [[ "${arg}" == "--" ]]; then
    parse_extra=true
    continue
  fi
  if [[ "${parse_extra}" == true ]]; then
    EXTRA_ARGS+=("${arg}")
  elif [[ "${arg}" == "--all" ]]; then
    SELECTED_VARIANTS=("${ALL_VARIANTS[@]}")
  else
    SELECTED_VARIANTS+=("${arg}")
  fi
done

if [[ ${#SELECTED_VARIANTS[@]} -eq 0 ]]; then
  SELECTED_VARIANTS=(baseline)
fi

variant_overrides() {
  local name="$1"
  case "${name}" in
    baseline)
      ;;
    enc_32)
      printf '%s\n' encoder.embedding_dim=32 query_projector.in_dim=32
      ;;
    enc_512)
      printf '%s\n' encoder.embedding_dim=512 query_projector.in_dim=512
      ;;
    ep_1)
      printf '%s\n' training.trainer.max_epochs=1
      ;;
    ep_2)
      printf '%s\n' training.trainer.max_epochs=2
      ;;
    ep_4)
      printf '%s\n' training.trainer.max_epochs=4
      ;;
    k_01)
      printf '%s\n' retriever.k=1
      ;;
    k_08)
      printf '%s\n' retriever.k=8
      ;;
    k_16)
      printf '%s\n' retriever.k=16
      ;;
    k_32)
      printf '%s\n' retriever.k=32
      ;;
    k_64)
      printf '%s\n' retriever.k=64
      ;;
    k_128)
      printf '%s\n' retriever.k=128
      ;;
    lr_1e-4)
      printf '%s\n' training.module.lr=1e-4
      ;;
    lr_3e-3)
      printf '%s\n' training.module.lr=3e-3
      ;;
    seed_s12)
      printf '%s\n' seed=12 training.trainer.max_epochs=1
      ;;
    seed_s123)
      printf '%s\n' seed=123 training.trainer.max_epochs=1
      ;;
    seed_s456)
      printf '%s\n' seed=456 training.trainer.max_epochs=1
      ;;
    seed_s789)
      printf '%s\n' seed=789 training.trainer.max_epochs=1
      ;;
    *)
      echo "Unknown sweep variant: ${name}" >&2
      return 1
      ;;
  esac
}

run_variant() {
  local name="$1"
  local output_dir="${SWEEP_ROOT}/${name}"
  local wandb_run_name="sweep-${TASK_NAME}-${name}"
  mapfile -t overrides < <(variant_overrides "${name}")

  echo "=== Sweep run: ${name} ==="
  echo "  Task        : ${TASK_NAME}"
  echo "  Output dir  : ${output_dir}"
  echo "  Started     : $(date)"
  echo ""

  OUTPUT_DIR="${output_dir}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_RUN_NAME="${wandb_run_name}" \
    "${SCRIPT_DIR}/run_retrieval_only_local.sh" \
      training.datamodule.num_workers="${NUM_WORKERS}" \
      "${overrides[@]}" \
      "${EXTRA_ARGS[@]}"

  echo ""
}

for variant in "${SELECTED_VARIANTS[@]}"; do
  run_variant "${variant}"
done
