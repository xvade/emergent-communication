#!/bin/bash
# 2x2 sweep (shared/separate networks x random/fixed roles), 32 seeds per cell
# at 2e7 steps, plus a no-comm baseline per cell (8 seeds, 5e6 steps).
# 8 seeds per task: 8 fit on a 48 GB L40S, 16 run out of memory (9-15 untested). %1 keeps the sweep on one GPU at a time. From repo root:
#   sbatch agents/experiments/commsjax/sweep.sbatch
# Results: agents/experiments/commsjax/results/sweep-<arrayjobid>/<cell>-s<seed>/
# Then:    python -m commsjax.aggregate agents/experiments/commsjax/results/sweep-<id>/*
#SBATCH --job-name=cjax-sweep
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --array=0-19%1
#SBATCH --output=agents/experiments/commsjax/logs/sweep-%A_%a.log
set -uo pipefail
SJ_ENV=${SJ_ENV:-/gscratch/amath/slszeto/envs/socialjax}
export PYTHONPATH=$(pwd)/agents
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
i=$SLURM_ARRAY_TASK_ID
CELLS=("" "--fixed-roles" "--separate" "--separate --fixed-roles")
NAMES=(shared-random shared-fixed separate-random separate-fixed)
if (( i < 4 )); then            # tasks 0-3: no-comm baselines (run first)
  cell=$i; seed=0; extra="--no-comm --total-steps 5e6"; tag="nocomm"
else                            # tasks 4-19: 4 cells x 4 chunks of 8 seeds
  cell=$(( (i - 4) % 4 )); seed=$(( ((i - 4) / 4) * 100 )); extra="--total-steps 2e7"; tag="comm"
fi
out="$(pwd)/agents/experiments/commsjax/results/sweep-$SLURM_ARRAY_JOB_ID/${NAMES[$cell]}-$tag-s$seed"
echo "host=$(hostname) job=$SLURM_JOB_ID task=$i cell=${NAMES[$cell]} $tag seed=$seed start=$(date)"
"$SJ_ENV/bin/python" -u -m commsjax.ppo --seeds 8 --seed $seed ${CELLS[$cell]} $extra --out "$out" 2>&1 \
  | grep -v "slow_operation_alarm\|Trying algorithm\|^$"
echo "exit=${PIPESTATUS[0]} end=$(date)"
