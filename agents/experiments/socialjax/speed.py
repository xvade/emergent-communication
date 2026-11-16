"""Throughput test of SocialJax envs on one GPU (random actions, no learner).

Adapted from SocialJax's speed_test/speed_test_random.py: one jitted
lax.scan of NUM_STEPS env steps over NUM_ENVS vmapped envs. Reports env
steps/s (one step = all agents act once) and agent-steps/s.
"""
import argparse
import json
import time

import jax
import socialjax


def make_benchmark(env, num_envs, num_steps):
  def benchmark(rng):
    rng, r = jax.random.split(rng)
    _, state = jax.vmap(env.reset)(jax.random.split(r, num_envs))

    def step(carry, _):
      state, rng = carry
      rng, ra, rs = jax.random.split(rng, 3)
      ras = jax.random.split(ra, env.num_agents * num_envs).reshape(
          env.num_agents, num_envs, -1)
      actions = [jax.vmap(env.action_space(a).sample)(ras[i])
                 for i, a in enumerate(env.agents)]
      _, state, _, _, _ = jax.vmap(env.step)(
          jax.random.split(rs, num_envs), state, actions)
      return (state, rng), None

    (state, _), _ = jax.lax.scan(step, (state, rng), None, num_steps)
    return state
  return benchmark


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--envs', default='coin_game,harvest_common_open,clean_up')
  p.add_argument('--num-envs', default='1,128,1024')
  p.add_argument('--steps', type=int, default=1000)
  p.add_argument('--out', default=None)
  a = p.parse_args()
  print('devices:', jax.devices(), flush=True)
  rows = []
  for name in a.envs.split(','):
    env = socialjax.make(name)
    for n in map(int, a.num_envs.split(',')):
      t0 = time.perf_counter()
      fn = jax.jit(make_benchmark(env, n, a.steps)).lower(
          jax.random.PRNGKey(0)).compile()
      compile_s = time.perf_counter() - t0
      t0 = time.perf_counter()
      jax.block_until_ready(fn(jax.random.PRNGKey(1)))
      run_s = time.perf_counter() - t0
      sps = n * a.steps / run_s
      row = dict(env=name, agents=env.num_agents, num_envs=n, steps=a.steps,
                 compile_s=round(compile_s, 2), run_s=round(run_s, 3),
                 env_sps=round(sps), agent_sps=round(sps * env.num_agents))
      rows.append(row)
      print(json.dumps(row), flush=True)
  if a.out:
    with open(a.out, 'w') as f:
      json.dump(rows, f, indent=1)


if __name__ == '__main__':
  main()
