#!/bin/bash
# 30-min single-GPU check that SocialJax IPPO trains end to end on klone.
# Needs setup.sbatch first. Usage, from repo root:
#   sbatch agents/experiments/socialjax/train_check.sbatch [extra Hydra overrides]
#SBATCH --job-name=sjax-train
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=agents/experiments/socialjax/logs/train-%j.log
set -uo pipefail

SJ_DIR=${SJ_DIR:-/gscratch/amath/slszeto/SocialJax}
SJ_ENV=${SJ_ENV:-/gscratch/amath/slszeto/envs/socialjax}
HERE=$(pwd)/agents/experiments/socialjax
export SJ_DIR
export PYTHONPATH=$SJ_DIR${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled

echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
timeout 1620 "$SJ_ENV/bin/python" -u "$HERE/train_check.py" --env coins \
  --out "$HERE/results/train-$SLURM_JOB_ID" \
  TOTAL_TIMESTEPS=1e7 NUM_ENVS=256 NUM_STEPS=500 NUM_MINIBATCHES=8 \
  PARAMETER_SHARING=False "$@"
echo "exit=$? end=$(date)"
