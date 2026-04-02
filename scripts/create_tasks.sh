#!/bin/bash
# ============================================================
# Extract MEDS-DEV task labels from the shared MIMIC-IV cohort.
#
# Reads from: /groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort  (read-only)
# Writes to:  mimic/task_labels/mortality/in_icu/first_24h    (this repo)
#
# Usage: sbatch scripts/create_tasks.sh
# ============================================================

#SBATCH --job-name=medrap-task-labels
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/groups/mm6677_gp/hs3627/MedRAP/logs/mimic_task_labels_%j.out
#SBATCH --error=/groups/mm6677_gp/hs3627/MedRAP/logs/mimic_task_labels_%j.err

set -euo pipefail

mkdir -p /groups/mm6677_gp/hs3627/MedRAP/logs

cd /groups/mm6677_gp/hs3627/MedRAP

echo "Node:   $(hostname)"
echo "Start:  $(date)"

TASK_VENV=task_venv
MEDS_DEV_DIR=MEDS-DEV

# Set up isolated venv for task label extraction.
# MEDS-DEV brings in ACES as a dependency.
if [ ! -d "$TASK_VENV" ]; then
    echo "Creating task venv..."
    /usr/bin/python3.11 -m venv "$TASK_VENV"
fi
source "$TASK_VENV/bin/activate"
pip install --upgrade pip --quiet

# Clone MEDS-DEV if not already present, then install in editable mode.
if [ ! -d "$MEDS_DEV_DIR" ]; then
    echo "Cloning MEDS-DEV..."
    git clone https://github.com/Medical-Event-Data-Standard/MEDS-DEV.git "$MEDS_DEV_DIR"
fi
pip install --quiet -e "$MEDS_DEV_DIR"

# Source cohort: shared read-only MIMIC-IV MEDS cohort (292 train + 37 tuning + 37 held_out shards).
# Output labels: written inside this repo only — the source cohort is never modified.
MEDS_COHORT_DIR="/groups/mm6677_gp/data/MIMIC_MEDS/MEDS_cohort"
LABELS_DIR="/groups/mm6677_gp/hs3627/MedRAP/mimic/task_labels_full"
TASK_NAME="mortality/in_icu/first_24h"

echo "Extracting task labels for: ${TASK_NAME}"
echo "MEDS cohort:   ${MEDS_COHORT_DIR}  (read-only)"
echo "Labels output: ${LABELS_DIR}/${TASK_NAME}"

# meds-dev-task merges the MIMIC-IV dataset predicates with the task config
# and runs aces-cli --multirun over all shards.
meds-dev-task \
    task="${TASK_NAME}" \
    dataset="MIMIC-IV" \
    dataset_dir="${MEDS_COHORT_DIR}" \
    output_dir="${LABELS_DIR}/${TASK_NAME}" \
    do_overwrite=True

deactivate
echo "Done:   $(date)"
echo "Labels: ${LABELS_DIR}/${TASK_NAME}"
