#!/usr/bin/env bash
# ============================================================
# Retrieval-only with separate keys and values (MIMIC example)
# ------------------------------------------------------------
# Like run_retrieval_only.sh but uses PerDocMeanPooledRetrievalEncoder
# so keys (Qwen3 doc_key_embeddings, used for FAISS scoring) and
# values (learned token embeddings, used for prediction) are different.
# Patient state is used only to build the query. ReplaceFusion discards
# the patient state and predicts from retrieved documents only.
#
# Usage:
#   sbatch scripts/run_retrieval_only_per_doc.sh
#   sbatch scripts/run_retrieval_only_per_doc.sh retriever.k=64
#
# Extra arguments are forwarded to `medrap train` as Hydra overrides.
# ============================================================

#SBATCH --job-name=medrap-retrieval-only-per-doc
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
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
TASK_LABELS_DIR="${REPO_DIR}/data/task_labels/mortality/in_icu/first_24h"
OUTPUT_DIR="${REPO_DIR}/outputs/mimic_run_retrieval_only_per_doc"

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
    marginalized_retrieval=true \
    marginalized_score_similarity=dot \
    retriever=hf_dataset \
    "retriever.dataset_path=${RETRIEVAL_DB}" \
    retriever.doc_ids_column=null \
    retriever.doc_key_embeddings_column=doc_key_embeddings \
    retriever.k=4 \
    encoder=token_embedding_1024 \
    encoder.vocab_size=65536 \
    encoder.embedding_dim=128 \
    query_projector=sequence_mean_1024 \
    query_projector.in_dim=128 \
    query_projector.out_dim=1024 \
    retrieval_encoder=per_doc_mean_pooled \
    retrieval_encoder.vocab_size=151669 \
    retrieval_encoder.embedding_dim=1024 \
    fusion=replace \
    head=linear_1024_to_2 \
    head.in_dim=1024 \
    training/task=marginalized_binary \
    training/loss=marginalized_retrieval \
    training/datamodule=meds \
    "training.datamodule.config.tensorized_cohort_dir=${TENSORIZED_DIR}" \
    training.datamodule.config.max_seq_len=128 \
    "training.datamodule.config.task_labels_dir=${TASK_LABELS_DIR}" \
    training.datamodule.batch_size=32 \
    training.datamodule.config.seq_sampling_strategy=to_end \
    training/trainer=lightning_wandb \
    training.trainer.max_epochs=1 \
    training.trainer.accelerator=gpu \
    training.trainer.devices=1 \
    training.trainer.log_every_n_steps=10 \
    "wandb_run_name=retrieval-only-per-doc-${SLURM_JOB_ID:-local}" \
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
