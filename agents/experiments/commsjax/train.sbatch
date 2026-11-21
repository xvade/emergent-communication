#!/bin/bash
# Train IPPO on the JAX comms toy (one GPU). From repo root:
#   sbatch agents/experiments/commsjax/train.sbatch [ppo.py flags]
# e.g. sbatch agents/experiments/commsjax/train.sbatch --separate --fixed-roles --seeds 8
# (--separate alone no longer implies fixed roles; at most 8 seeds per GPU)
#SBATCH --job-name=cjax-train
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=agents/experiments/commsjax/logs/train-%j.log
set -uo pipefail
SJ_ENV=${SJ_ENV:-/gscratch/amath/slszeto/envs/socialjax}
export PYTHONPATH=$(pwd)/agents
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
nvidia-smi --query-gpu=name --format=csv,noheader
"$SJ_ENV/bin/python" -u -m commsjax.ppo --out "$(pwd)/agents/experiments/commsjax/results/$SLURM_JOB_ID" "$@"
echo "exit=$? end=$(date)"
