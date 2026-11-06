#!/bin/bash
#SBATCH --job-name=mp-exp
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# Run the zap-ablation experiment in the Melting Pot container.
# Submit from the repo root:  sbatch agents/experiments/run.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
apptainer run --cleanenv --bind "$PWD" agents/apptainer/meltingpot.sif \
  agents/experiments/random_zap_ablation.py \
  --substrate commons_harvest__open --episodes 3 --steps 1000 \
  --out agents/experiments/results "$@"
