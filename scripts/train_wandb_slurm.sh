#!/bin/bash
# SLURM: train with W&B. Synthetic datamodule (no MEDS paths).
# Usage: sbatch scripts/train_wandb_slurm.sh "$(pwd)"


#SBATCH --job-name=medrap-train-wandb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
# Used when you submit from the repo (pattern A). Overridden by sbatch -o/-e (pattern B).
#SBATCH --output=train_wandb_%j.out
#SBATCH --error=train_wandb_%j.err

set -euo pipefail

: "${1:?Pass path to MedRAP repo root. See usage at top of scripts/train_wandb_slurm.sh}"
MEDRAP_ROOT="$(cd "$1" && pwd)" || exit 1
cd "${MEDRAP_ROOT}"

echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Start:  $(date)"
echo "REPO:   ${MEDRAP_ROOT}"

source .venv/bin/activate

export WANDB_PROJECT=medrap

medrap train \
  training/datamodule=synthetic \
  training/trainer=lightning_wandb \
  wandb_project=medrap \
  wandb_run_name=medrap-train-job-${SLURM_JOB_ID} \
  output_dir=outputs/train_wandb_${SLURM_JOB_ID}

echo "Done:   $(date)"
