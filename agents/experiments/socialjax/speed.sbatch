#!/bin/bash
# 15-min single-GPU test of SocialJax env throughput (random actions).
# For an end-to-end training check use train_check.sbatch.
# Needs setup.sbatch first. Usage, from repo root:
#   sbatch agents/experiments/socialjax/speed.sbatch
#SBATCH --job-name=sjax-speed
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=agents/experiments/socialjax/logs/speed-%j.log
set -uo pipefail

SJ_DIR=${SJ_DIR:-/gscratch/amath/slszeto/SocialJax}
SJ_ENV=${SJ_ENV:-/gscratch/amath/slszeto/envs/socialjax}
HERE=$(pwd)/agents/experiments/socialjax
OUT=$HERE/results; mkdir -p "$OUT"
export PYTHONPATH=$SJ_DIR${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
PY=$SJ_ENV/bin/python

echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "=== 1. env throughput (random actions) ==="
timeout 420 $PY "$HERE/speed.py" --out "$OUT/speed-$SLURM_JOB_ID.json"
echo "speed.py exit=$?"

echo "end=$(date)"
