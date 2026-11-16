#!/bin/bash
# One-time setup: clone SocialJax (pinned) and build its Python env.
# Installs OFFLINE from wheels staged on a login node, because downloads through
# the compute-node proxy time out. Stage first (login node, from repo root):
#   uv python install 3.10
#   W=/gscratch/amath/slszeto/wheels/socialjax; mkdir -p $W
#   uvx --python 3.10 pip download -d $W -r /gscratch/amath/slszeto/SocialJax/requirements.txt setuptools wheel
# then:  sbatch agents/experiments/socialjax/setup.sbatch
#SBATCH --job-name=sjax-setup
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=agents/experiments/socialjax/logs/setup-%j.log
set -euo pipefail

SJ_DIR=${SJ_DIR:-/gscratch/amath/slszeto/SocialJax}
SJ_ENV=${SJ_ENV:-/gscratch/amath/slszeto/envs/socialjax}
SJ_COMMIT=${SJ_COMMIT:-18e21718e52e659bd4cf9c29df2ddfc03dbaca3c}  # main, 2026-10-17
SJ_WHEELS=${SJ_WHEELS:-/gscratch/amath/slszeto/wheels/socialjax}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/gscratch/amath/slszeto/.uv_cache}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-/gscratch/amath/slszeto/uv-python}
export UV_CONCURRENT_DOWNLOADS=4   # keep proxy connections low
export UV_CONCURRENT_BUILDS=${SLURM_CPUS_PER_TASK:-1}

echo "host=$(hostname) start=$(date)"
if [[ ! -d "$SJ_DIR/.git" ]]; then
  git clone https://github.com/cooperativex/SocialJax.git "$SJ_DIR"
fi
git -C "$SJ_DIR" fetch -q origin
git -C "$SJ_DIR" checkout -q "$SJ_COMMIT"
git -C "$SJ_DIR" log -1 --oneline

if [[ ! -x "$SJ_ENV/bin/python" ]]; then
  uv venv --python 3.10 --python-preference only-managed --no-python-downloads "$SJ_ENV"
fi
uv pip install --python "$SJ_ENV/bin/python" --offline --no-index \
  --find-links "$SJ_WHEELS" -r "$SJ_DIR/requirements.txt"
"$SJ_ENV/bin/python" -c "import jax, flax, socialjax; print('jax', jax.__version__, 'flax', flax.__version__)" \
  || PYTHONPATH="$SJ_DIR" "$SJ_ENV/bin/python" -c "import jax, flax, socialjax; print('jax', jax.__version__, 'flax', flax.__version__, '(socialjax via PYTHONPATH)')"
du -sh "$SJ_ENV"
echo "done $(date)"
