#!/bin/bash
#SBATCH --job-name=mpviz-smoke
#SBATCH --account=stf
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# Submit from the repo root: sbatch agents/tests/smoke.sbatch
set -euo pipefail
source agents/bin/_common.sh
mp_python agents/tests/smoke_mpviz.py
