#!/bin/bash
# Unit checks for the JAX comms toy (CPU). From repo root:
#   sbatch agents/experiments/commsjax/test.sbatch
#SBATCH --job-name=cjax-test
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=agents/experiments/commsjax/logs/test-%j.log
set -uo pipefail
SJ_ENV=${SJ_ENV:-/gscratch/amath/slszeto/envs/socialjax}
export PYTHONPATH=$(pwd)/agents
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK:-1}"
"$SJ_ENV/bin/python" -u -m commsjax.test_toy_env
