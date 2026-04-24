#!/bin/bash
# ============================================================
# SLURM: build HF retrieval corpus + FAISS index on a GPU node.
#
# Usage:
#   sbatch scripts/prepare_retrieval_slurm.sh
#
# To build a random subset of N documents from the source corpus:
#   sbatch scripts/prepare_retrieval_slurm.sh --num-docs 1000
#   sbatch scripts/prepare_retrieval_slurm.sh --num-docs 10000
#
# Subset sizes intended for ablation: 10, 100, 1000, 10000, full.
# ============================================================

#SBATCH --job-name=medrap-prep-retrieval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=prepare_retrieval_%j.out
#SBATCH --error=prepare_retrieval_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# --- parse optional --num-docs N argument ---
NUM_DOCS=""
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-docs)
      NUM_DOCS="$2"; shift 2 ;;
    *)
      PASSTHROUGH+=("$1"); shift ;;
  esac
done

# Determine output directory: subdirectory when a subset is requested.
PREP_NUM_DOCS_ARG=()
if [[ -n "${NUM_DOCS}" ]]; then
  OUTPUT_DIR="data/retrieval_db_${NUM_DOCS}docs"
  PREP_NUM_DOCS_ARG=( "prep.num_docs=${NUM_DOCS}" )
else
  OUTPUT_DIR="data/retrieval_db"
fi

echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo n/a)"
echo "Num docs:  ${NUM_DOCS:-all}"
echo "Output:    ${OUTPUT_DIR}"
echo "Start:     $(date)"

uv run medrap prepare-retrieval-dataset \
  prep.source.path=MedRAG/textbooks prep.source.split=train \
  prep.document.fields='[title,content]' \
  prep.tokenizer.pretrained_model_name_or_path=Qwen/Qwen3-Embedding-0.6B \
  prep.embedder.model_name_or_path=Qwen/Qwen3-Embedding-0.6B prep.embedder.device=cuda \
  prep.index.source_id_column=id \
  prep.output.output_dir="${OUTPUT_DIR}" \
  prep.index.max_length=256 \
  prep.index.tokenization_batch_size=512 \
  prep.index.embedding_batch_size=256 \
  prep.index.encode_batch_size=32 \
  "${PREP_NUM_DOCS_ARG[@]}" \
  "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"

echo "Done:      $(date)"
