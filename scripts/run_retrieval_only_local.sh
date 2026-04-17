#!/usr/bin/env bash
# ============================================================
# Local retrieval-only fusion + marginalized loss run
# ------------------------------------------------------------
# Predicts from retrieved documents only (ReplaceFusion). Patient
# state is used only to build the query embedding. Uses the same
# Hydra overrides as the SLURM runner, but removes scheduler
# assumptions and defaults WandB/HF to offline mode.
#
# Required environment variables:
#   TENSORIZED_DIR   Root tensorized MEDS cohort directory
#   TASK_LABELS_DIR  Directory containing train/tuning/held_out parquet labels
#
# Optional environment variables:
#   RETRIEVAL_DB     Retrieval artifact directory (default: REPO_DIR/data/retrieval_db)
#   OUTPUT_DIR       Run output directory (default: REPO_DIR/outputs/retrieval_only_local)
#   ACCELERATOR      Lightning accelerator (default: gpu)
#   DEVICES          Number of devices (default: 1)
#   MAX_EPOCHS       Training epochs (default: 3)
#   BATCH_SIZE       Datamodule batch size (default: 32)
#   MAX_SEQ_LEN      Datamodule max seq len (default: 128)
#   RETRIEVER_K      Retrieved docs for training/inference (default: 4)
#   VOCAB_SIZE       Patient encoder vocab size. Defaults to metadata-derived max code index + 1.
#   WANDB_PROJECT    WandB project name (default: medrap)
#   WANDB_RUN_NAME   WandB run name (default: retrieval-only-local-<timestamp>)
#   WANDB_DIR        WandB output directory (default: OUTPUT_DIR/wandb)
#   HF_HUB_OFFLINE   Default 1; set to 0 to allow HF downloads
#   TRANSFORMERS_OFFLINE  Default 1; set to 0 to allow model/tokenizer downloads
#
# Usage:
#   export TENSORIZED_DIR=/path/to/tensorized
#   export TASK_LABELS_DIR=/path/to/task_labels/in_hospital_mortality
#   export RETRIEVAL_DB=/path/to/retrieval_db
#   scripts/run_retrieval_only_local.sh
#
# Extra CLI arguments are forwarded to `medrap train` as Hydra overrides.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

RETRIEVAL_DB="${RETRIEVAL_DB:-${REPO_DIR}/data/retrieval_db}"
TENSORIZED_DIR="${TENSORIZED_DIR:-}"
TASK_LABELS_DIR="${TASK_LABELS_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/retrieval_only_local}"

ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}"
RETRIEVER_K="${RETRIEVER_K:-4}"
VOCAB_SIZE="${VOCAB_SIZE:-}"

WANDB_PROJECT="${WANDB_PROJECT:-medrap}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-retrieval-only-local-$(date +%Y%m%d-%H%M%S)}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${OUTPUT_DIR}/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${OUTPUT_DIR}/wandb_config}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${OUTPUT_DIR}/wandb_data}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

require_dir() {
  local path="$1"
  local label="$2"
  if [[ -z "${path}" ]]; then
    echo "Missing required ${label}. Set ${label} in the environment." >&2
    exit 1
  fi
  if [[ ! -d "${path}" ]]; then
    echo "${label} does not exist or is not a directory: ${path}" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

infer_vocab_size() {
  local metadata_fp="$1"
  local py_bin=""
  if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
    py_bin="${REPO_DIR}/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    py_bin="python3"
  else
    echo "Could not find Python to infer vocab size." >&2
    exit 1
  fi

  "${py_bin}" - "${metadata_fp}" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq

metadata_fp = Path(sys.argv[1])
table = pq.read_table(metadata_fp, columns=["code/vocab_index"])
col = table.column("code/vocab_index")
max_idx = 0
for chunk in col.chunks:
    values = chunk.to_pylist()
    if values:
        max_idx = max(max_idx, max(values))
print(int(max_idx) + 1)
PY
}

resolve_runner() {
  if [[ -x "${REPO_DIR}/.venv/bin/medrap" ]]; then
    RUNNER=("${REPO_DIR}/.venv/bin/medrap")
    return
  fi
  if command -v medrap >/dev/null 2>&1; then
    RUNNER=("medrap")
    return
  fi
  if command -v uv >/dev/null 2>&1; then
    RUNNER=("uv" "run" "--project" "${REPO_DIR}" "medrap")
    return
  fi
  echo "Could not find a usable medrap runner. Expected .venv/bin/medrap, medrap on PATH, or uv." >&2
  exit 1
}

require_dir "${TENSORIZED_DIR}" "TENSORIZED_DIR"
require_dir "${TASK_LABELS_DIR}" "TASK_LABELS_DIR"
require_dir "${RETRIEVAL_DB}" "RETRIEVAL_DB"
require_file "${TENSORIZED_DIR}/metadata/codes.parquet" "tensorized metadata codes"
require_file "${TASK_LABELS_DIR}/train.parquet" "train labels"
require_file "${TASK_LABELS_DIR}/tuning.parquet" "tuning labels"
require_file "${TASK_LABELS_DIR}/held_out.parquet" "held_out labels"
require_file "${RETRIEVAL_DB}/retrieval.faiss" "FAISS retrieval index"

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${WANDB_DATA_DIR}"

if [[ -z "${VOCAB_SIZE}" ]]; then
  VOCAB_SIZE="$(infer_vocab_size "${TENSORIZED_DIR}/metadata/codes.parquet")"
fi

resolve_runner

echo "=== Local retrieval-only run ==="
echo "  Repo dir        : ${REPO_DIR}"
echo "  Runner          : ${RUNNER[*]}"
echo "  Tensorized dir  : ${TENSORIZED_DIR}"
echo "  Task labels dir : ${TASK_LABELS_DIR}"
echo "  Retrieval DB    : ${RETRIEVAL_DB}"
echo "  Output dir      : ${OUTPUT_DIR}"
echo "  Accelerator     : ${ACCELERATOR}"
echo "  Devices         : ${DEVICES}"
echo "  Vocab size      : ${VOCAB_SIZE}"
echo "  WandB mode      : ${WANDB_MODE}"
echo "  WandB dir       : ${WANDB_DIR}"
echo "  HF offline      : ${HF_HUB_OFFLINE}"
echo "  Started         : $(date)"
echo ""

cd "${REPO_DIR}"

"${RUNNER[@]}" train \
    marginalized_retrieval=true \
    marginalized_score_similarity=dot \
    retriever=hf_dataset \
    "retriever.dataset_path=${RETRIEVAL_DB}" \
    retriever.doc_ids_column=null \
    retriever.doc_key_embeddings_column=doc_key_embeddings \
    "retriever.k=${RETRIEVER_K}" \
    encoder=token_embedding_128 \
    "encoder.vocab_size=${VOCAB_SIZE}" \
    query_projector=sequence_mean_1024 \
    query_projector.in_dim=128 \
    retrieval_encoder=key_embedding \
    fusion=replace \
    head=linear_1024_to_2 \
    training/task=marginalized_binary \
    training/loss=marginalized_retrieval \
    training/datamodule=meds \
    "training.datamodule.config.tensorized_cohort_dir=${TENSORIZED_DIR}" \
    "training.datamodule.config.max_seq_len=${MAX_SEQ_LEN}" \
    "training.datamodule.config.task_labels_dir=${TASK_LABELS_DIR}" \
    "training.datamodule.batch_size=${BATCH_SIZE}" \
    training.datamodule.config.seq_sampling_strategy=to_end \
    training/trainer=lightning_wandb \
    "training.trainer.max_epochs=${MAX_EPOCHS}" \
    "training.trainer.accelerator=${ACCELERATOR}" \
    "training.trainer.devices=${DEVICES}" \
    training.trainer.log_every_n_steps=10 \
    "wandb_project=${WANDB_PROJECT}" \
    "wandb_run_name=${WANDB_RUN_NAME}" \
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true \
    "$@"

echo ""
echo "=== Done: $(date) ==="
