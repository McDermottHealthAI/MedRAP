#!/bin/bash
# SLURM: launch JupyterLab on a compute node.
# Usage: sbatch scripts/jupyter.slurm
# Then tail the log file printed by sbatch to get the SSH tunnel command and URL.

#SBATCH --job-name=medrap-jupyter
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=jupyter_%j.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

NODE=$(hostname)
PORT=8889

source .venv/bin/activate

echo "=============================================="
echo "Node:   ${NODE}"
echo "Port:   ${PORT}"
echo "Start:  $(date)"
echo ""
echo "From your laptop, run this SSH tunnel:"
echo "  ssh -N -L ${PORT}:${NODE}:${PORT} zzw2102@hpc.c2b2.columbia.edu"
echo ""
echo "Then open the http://127.0.0.1:${PORT}/lab?token=... URL printed below"
echo "in your browser, or paste it into Cursor's 'Existing Jupyter Server' picker."
echo "=============================================="

jupyter lab --ip=0.0.0.0 --port="${PORT}" --no-browser
