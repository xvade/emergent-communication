#!/bin/bash
# Compare JAX toy vs the Lua toy (Melting Pot container, CPU). From repo root:
#   sbatch agents/experiments/commsjax/compare_lua.sbatch
#SBATCH --job-name=cjax-lua
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=agents/experiments/commsjax/logs/compare-lua-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source agents/bin/_common.sh
mp_python agents/commsjax/compare_lua.py
