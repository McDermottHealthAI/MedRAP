#!/bin/bash
# ============================================================
# SLURM: build HF retrieval corpus + FAISS index on a GPU node.
#
# Usage:
#   sbatch scripts/prepare_retrieval_slurm.sh
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

MEDRAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${MEDRAP_ROOT}"

echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo n/a)"
echo "Start:  $(date)"

uv run medrap prepare-retrieval-dataset \
  prep.source.path=MedRAG/textbooks prep.source.split=train \
  prep.document.fields='[title,content]' \
  prep.tokenizer.pretrained_model_name_or_path=Qwen/Qwen3-Embedding-0.6B \
  prep.embedder.model_name_or_path=Qwen/Qwen3-Embedding-0.6B prep.embedder.device=cuda \
  prep.index.source_id_column=id \
  prep.output.output_dir=data/retrieval_db \
  prep.index.max_length=256 \
  prep.index.tokenization_batch_size=512 \
  prep.index.embedding_batch_size=256 \
  prep.index.encode_batch_size=32 \
  "$@"

echo "Done:   $(date)"
