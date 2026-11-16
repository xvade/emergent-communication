"""End-to-end check of SocialJax IPPO training on klone.

Loads SocialJax's own Hydra config for an IPPO env, runs its make_train under
jit, prints progress from the metric callback (wandb.log is redirected to
stdout), reports compile vs. train time, and saves the final parameters. Skips
SocialJax's GIF evaluation.

Usage (SocialJax repo on PYTHONPATH):
  python train_check.py --env coins TOTAL_TIMESTEPS=1e7 PARAMETER_SHARING=False
"""
import argparse
import json
import os
import pickle
import time

import jax
import numpy as np

import wandb
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--env', default='coins')
  p.add_argument('--out', default='results')
  p.add_argument('--print-every', type=int, default=5)
  a, overrides = p.parse_known_args()

  sj_dir = os.environ.get('SJ_DIR', '/gscratch/amath/slszeto/SocialJax')
  cfg_dir = os.path.join(sj_dir, 'algorithms', 'IPPO', 'config')
  with initialize_config_dir(config_dir=cfg_dir, version_base=None):
    cfg = compose(config_name=f'ippo_cnn_{a.env}', overrides=overrides)
  config = OmegaConf.to_container(cfg)
  config['WANDB_MODE'] = 'disabled'
  print('config:', json.dumps({k: config[k] for k in (
      'ENV_NAME', 'TOTAL_TIMESTEPS', 'NUM_ENVS', 'NUM_STEPS', 'NUM_MINIBATCHES',
      'UPDATE_EPOCHS', 'PARAMETER_SHARING', 'REWARD') if k in config}), flush=True)
  print('devices:', jax.devices(), flush=True)

  mod = __import__(f'algorithms.IPPO.ippo_cnn_{a.env}', fromlist=['make_train'])

  t_start = [None]
  history = []

  def log(metric):
    m = {k: float(np.asarray(v)) for k, v in metric.items()
         if np.asarray(v).size == 1}
    history.append(m)
    step = int(m.get('update_step', len(history)))
    if t_start[0] is None:
      t_start[0] = time.perf_counter()
    if step == 1 or step % a.print_every == 0:
      el = time.perf_counter() - t_start[0]
      keys = [k for k in ('env_step', 'returned_episode_returns', 'loss',
                          'eat_own_coins') if k in m]
      print(f'[upd {step}] t={el:.1f}s ' +
            ' '.join(f'{k}={m[k]:.4g}' for k in keys), flush=True)

  wandb.log = log  # make_train's callback calls wandb.log(metric)

  t0 = time.perf_counter()
  train = jax.jit(mod.make_train(config)).lower(jax.random.PRNGKey(config['SEED'])).compile()
  compile_s = time.perf_counter() - t0
  print(f'compiled in {compile_s:.1f}s', flush=True)

  t0 = time.perf_counter()
  out = jax.block_until_ready(train(jax.random.PRNGKey(config['SEED'])))
  train_s = time.perf_counter() - t0
  steps = len(history) * config['NUM_STEPS'] * config['NUM_ENVS']
  print(f'trained {len(history)} updates, {steps:.3g} env steps in {train_s:.1f}s '
        f'-> {steps / train_s:,.0f} env steps/s (incl. learning)', flush=True)

  os.makedirs(a.out, exist_ok=True)
  tag = f"ippo_{a.env}_{'shared' if config['PARAMETER_SHARING'] else 'separate'}"
  params = jax.device_get(jax.tree.map(lambda x: x, out['runner_state'][0]))
  params = ([ts.params for ts in params] if isinstance(params, list)
            else params.params)
  with open(os.path.join(a.out, f'{tag}_params.pkl'), 'wb') as f:
    pickle.dump(params, f)
  with open(os.path.join(a.out, f'{tag}_metrics.json'), 'w') as f:
    json.dump(dict(config=config, compile_s=compile_s, train_s=train_s,
                   env_steps=steps, history=history), f)
  print('saved', os.path.join(a.out, tag + '_*'), flush=True)


if __name__ == '__main__':
  main()
