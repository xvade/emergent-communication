#!/bin/bash
#SBATCH --job-name=mp-test-comms
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
# Submitted by agents/bin/mp-test-comms (which sets account, partition, time
# and the log path). Arguments: optional config names to test.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source agents/bin/_common.sh
mp_python agents/tests/communication/run_all.py "$@"
