#!/usr/bin/env bash
# ============================================================
# Hyperparameter sweep via SLURM job array
# ------------------------------------------------------------
# Sweeps k (# retrieved docs), learning rate, and encoder
# embedding dimension around the fast-run baseline. All runs
# share the same data/step constraints as the ~15-min baseline.
#
# Usage:
#   sbatch --array=0-7 scripts/sweep_slurm.sh
#
# Extra arguments are forwarded to `medrap train` as Hydra overrides.
# ============================================================

#SBATCH --job-name=medrap-sweep
#SBATCH --partition=gpu
#SBATCH --account=mm6677_gp
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR}"
VENV="${REPO_DIR}/.venv/bin/activate"
RETRIEVAL_DB="${REPO_DIR}/data/retrieval_db"
TENSORIZED_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort/processed"
TASK_LABELS_DIR="${REPO_DIR}/data/task_labels/mortality/in_icu/first_24h"

# ---- Sweep configs ----
NAMES=(   k_32  k_64  k_128  k_256  ep_1  ep_2  ep_3  ep_4)
KS=(      32    64    128    256    4     4     4     4   )
LRS=(     1e-3  1e-3  1e-3   1e-3   1e-3  1e-3  1e-3  1e-3)
ENC_DIMS=(128   128   128    128    128   128   128   128  )
EPOCHS=(  3     3     3      3      1     2     3     4   )

IDX=${SLURM_ARRAY_TASK_ID}
NAME=${NAMES[$IDX]}
K=${KS[$IDX]}
LR=${LRS[$IDX]}
ENC_DIM=${ENC_DIMS[$IDX]}
EPOCH=${EPOCHS[$IDX]}

OUTPUT_DIR="${REPO_DIR}/outputs/sweep/${NAME}"

echo "=== Job info ==="
echo "  Array task : ${IDX} / ${NAME}"
echo "  Job ID     : ${SLURM_JOB_ID:-local}"
echo "  Node       : ${SLURMD_NODENAME:-$(hostname)}"
echo "  GPU(s)     : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  k          : ${K}"
echo "  lr         : ${LR}"
echo "  enc_dim    : ${ENC_DIM}"
echo "  epochs     : ${EPOCH}"
echo "  Started    : $(date)"
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
    retriever.k=${K} \
    encoder=token_embedding_1024 \
    encoder.vocab_size=65536 \
    encoder.embedding_dim=${ENC_DIM} \
    query_projector=sequence_mean_1024 \
    query_projector.in_dim=${ENC_DIM} \
    query_projector.out_dim=1024 \
    retrieval_encoder=key_embedding \
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
    training.trainer.max_epochs=${EPOCH} \
    training.trainer.accelerator=gpu \
    training.trainer.devices=1 \
    training.trainer.log_every_n_steps=10 \
    training.module.lr=${LR} \
    "wandb_run_name=sweep-${NAME}-${SLURM_JOB_ID:-local}" \
    "output_dir=${OUTPUT_DIR}" \
    do_overwrite=true \
    "$@"

echo ""
echo "=== Done: $(date) ==="
