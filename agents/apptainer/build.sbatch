#!/bin/bash
#SBATCH --job-name=mp-build
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log
# Build the Melting Pot Apptainer image. Submit from the repo root:
#   sbatch agents/apptainer/build.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# Keep the layer cache and build temp off the 10 GB home quota.
export APPTAINER_CACHEDIR=/gscratch/scrubbed/$USER/.apptainer-cache
export APPTAINER_TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$APPTAINER_CACHEDIR"
apptainer build --force agents/apptainer/meltingpot.sif agents/apptainer/meltingpot.def
ls -lh agents/apptainer/meltingpot.sif
