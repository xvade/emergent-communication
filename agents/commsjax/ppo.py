"""IPPO for the JAX communication toy (PureJaxRL-style, fully jitted).

* Shared parameters (default): one network acts for both players and works out
  its role from what it sees. That matches the substrate, where rooms are
  assigned at random.
* Separate parameters (`--separate`): each agent has its own network and
  optimizer.
* `--fixed-roles`: agent 0 is always the sender (apple room) and agent 1 the
  receiver (pad room). Independent of `--separate`, so the 2x2 of
  shared/separate x random/fixed roles can be run. `--separate --fixed-roles`
  is the setup a misaligned sender/receiver needs.
* `--no-comm`: the channel is removed from the env (bit always 0, COMMUNICATE
  does nothing). This trains the no-communication baseline pi_R0 for the
  reliance gain Delta = J(pi_R) - J(pi_R0); `aggregate.py` computes Delta
  from a comm run and a matched no-comm run.
* Several seeds are trained in parallel with vmap (`--seeds`).

Logged per update: success rate (correct pad / episodes), episode length, and
how often the sender communicates when the apple is present vs. absent.
After training, each seed is evaluated twice: normally, and with the comm bit
zeroed in both players' observations. The difference is reported as
`ablation_drop`. It is not the one-pager's Delta, which compares against a
receiver trained without the channel (`--no-comm`).

Usage (SocialJax env, which has jax/flax/optax/distrax):
  python -m commsjax.ppo --total-steps 2e7 --seeds 4 --out results/run1
"""
import argparse
import json
import os
import pickle
import time
from functools import partial

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal

from commsjax.toy_env import COMM, ToyCommsEnv


class ActorCritic(nn.Module):
  num_actions: int

  @nn.compact
  def __call__(self, x):
    x = nn.relu(nn.Conv(16, (3, 3), kernel_init=orthogonal(np.sqrt(2)))(x))
    x = nn.relu(nn.Conv(32, (3, 3), kernel_init=orthogonal(np.sqrt(2)))(x))
    x = x.reshape((x.shape[0], -1))
    x = nn.relu(nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)))(x))
    logits = nn.Dense(self.num_actions, kernel_init=orthogonal(0.01),
                      bias_init=constant(0.0))(x)
    v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
    return distrax.Categorical(logits=logits), v.squeeze(-1)


def make_train(cfg):
  env = ToyCommsEnv(exists_prob=cfg["exists_prob"], max_steps=cfg["max_steps"],
                    fixed_roles=cfg["fixed_roles"], comm_enabled=not cfg["no_comm"])
  E, T, A = cfg["num_envs"], cfg["num_steps"], env.num_agents
  num_updates = int(cfg["total_steps"] // (E * T))
  net = ActorCritic(env.num_actions)
  tx = optax.chain(optax.clip_by_global_norm(cfg["max_grad_norm"]),
                   optax.adam(cfg["lr"], eps=1e-5))
  sep = cfg["separate"]

  def policy(params, obs):
    """obs (E, A, ...) -> pi over (E, A), value (E, A)."""
    if sep:  # params stacked (A, ...); agent axis first
      o = jnp.swapaxes(obs, 0, 1)                      # (A, E, ...)
      logits, v = jax.vmap(lambda p, x: _logits_v(p, x))(params, o)
      return jnp.swapaxes(logits, 0, 1), jnp.swapaxes(v, 0, 1)
    logits, v = _logits_v(params, obs.reshape((E * A,) + obs.shape[2:]))
    return logits.reshape(E, A, -1), v.reshape(E, A)

  def _logits_v(p, x):
    pi, v = net.apply(p, x)
    return pi.logits, v

  def init(rng):
    dummy = jnp.zeros((1,) + env.obs_shape)
    if sep:
      params = jax.vmap(lambda k: net.init(k, dummy))(jax.random.split(rng, A))
      opt = jax.vmap(tx.init)(params)
    else:
      params = net.init(rng, dummy)
      opt = tx.init(params)
    return params, opt

  def ppo_update(params, opt, batch, rng):
    """One network's PPO update on a flat batch of transitions."""
    obs, act, logp_old, adv, ret, v_old = batch
    n = obs.shape[0]
    mb = n // cfg["num_minibatches"]

    def loss_fn(p, b):
      o, a, lp0, ad, rt, v0 = b
      pi, v = net.apply(p, o)
      lp = pi.log_prob(a)
      ratio = jnp.exp(lp - lp0)
      ad = (ad - ad.mean()) / (ad.std() + 1e-8)
      pg = -jnp.minimum(ratio * ad, jnp.clip(ratio, 1 - cfg["clip"], 1 + cfg["clip"]) * ad).mean()
      v_clip = v0 + jnp.clip(v - v0, -cfg["clip"], cfg["clip"])
      vl = 0.5 * jnp.maximum((v - rt) ** 2, (v_clip - rt) ** 2).mean()
      ent = pi.entropy().mean()
      return pg + cfg["vf_coef"] * vl - cfg["ent_coef"] * ent, (pg, vl, ent)

    def epoch(carry, _):
      p, o_s, rng = carry
      rng, k = jax.random.split(rng)
      perm = jax.random.permutation(k, n)[: mb * cfg["num_minibatches"]]
      shuf = jax.tree.map(lambda x: x[perm].reshape((cfg["num_minibatches"], mb) + x.shape[1:]), batch)

      def minibatch(c, b):
        p, o_s = c
        (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, b)
        upd, o_s = tx.update(g, o_s, p)
        return (optax.apply_updates(p, upd), o_s), jnp.stack([l, *aux])

      (p, o_s), losses = jax.lax.scan(minibatch, (p, o_s), shuf)
      return (p, o_s, rng), losses.mean(0)

    (params, opt, _), losses = jax.lax.scan(epoch, (params, opt, rng), None, cfg["update_epochs"])
    return params, opt, losses.mean(0)

  def gae(rew, val, done, last_val):
    def f(carry, x):
      g, nv = carry
      r, v, d = x
      delta = r + cfg["gamma"] * nv * (1 - d) - v
      g = delta + cfg["gamma"] * cfg["gae_lambda"] * (1 - d) * g
      return (g, v), g
    _, adv = jax.lax.scan(f, (jnp.zeros_like(last_val), last_val),
                          (rew, val, done), reverse=True)
    return adv, adv + val

  def train(rng):
    rng, k_init, k_env = jax.random.split(rng, 3)
    params, opt = init(k_init)
    obs, st = jax.vmap(env.reset)(jax.random.split(k_env, E))

    def update(carry, _):
      params, opt, obs, st, rng, n_upd = carry

      def rollout(c, _):
        obs, st, rng = c
        rng, k_a, k_s = jax.random.split(rng, 3)
        logits, v = policy(params, obs)
        pi = distrax.Categorical(logits=logits)
        act = pi.sample(seed=k_a)
        logp = pi.log_prob(act)
        sender = jax.vmap(ToyCommsEnv.in_apple_room)(st)          # (E, A)
        nobs, nst, rew, done, info = jax.vmap(env.step)(jax.random.split(k_s, E), st, act)
        tr = dict(obs=obs, act=act, logp=logp, v=v, rew=rew,
                  done=jnp.broadcast_to(done[:, None], (E, A)).astype(jnp.float32),
                  ep_done=done, success=info["success"], ep_len=info["episode_length"],
                  apple=info["apple"],
                  s_comm=jnp.sum((act == COMM) & sender, 1),
                  r_comm=jnp.sum((act == COMM) & ~sender, 1))
        return (nobs, nst, rng), tr

      (obs, st, rng), tr = jax.lax.scan(rollout, (obs, st, rng), None, T)
      _, last_v = policy(params, obs)
      adv, ret = gae(tr["rew"], tr["v"], tr["done"], last_v)       # (T, E, A)

      rng, k_u = jax.random.split(rng)
      if sep:
        to_agent = lambda x: jnp.moveaxis(x, 2, 0).reshape((A, T * E) + x.shape[3:])
        batch = tuple(to_agent(x) for x in (tr["obs"], tr["act"], tr["logp"], adv, ret, tr["v"]))
        params, opt, losses = jax.vmap(ppo_update, in_axes=(0, 0, 0, None))(params, opt, batch, k_u)
        losses = losses.mean(0)
      else:
        flat = lambda x: x.reshape((T * E * A,) + x.shape[3:])
        batch = tuple(flat(x) for x in (tr["obs"], tr["act"], tr["logp"], adv, ret, tr["v"]))
        params, opt, losses = ppo_update(params, opt, batch, k_u)

      d = tr["ep_done"].astype(jnp.float32)
      n_ep = d.sum()
      apple = tr["apple"].astype(jnp.float32)
      m = dict(
          update=n_upd + 1,
          env_steps=(n_upd + 1) * T * E,
          episodes=n_ep,
          success=(tr["success"] * d).sum() / jnp.maximum(n_ep, 1),
          ep_len=(tr["ep_len"] * d).sum() / jnp.maximum(n_ep, 1),
          sender_comm_apple=(tr["s_comm"] * apple).sum() / jnp.maximum(apple.sum(), 1),
          sender_comm_noapple=(tr["s_comm"] * (1 - apple)).sum() / jnp.maximum((1 - apple).sum(), 1),
          receiver_comm=tr["r_comm"].mean(),
          loss=losses[0], pg_loss=losses[1], v_loss=losses[2], entropy=losses[3],
      )
      if cfg.get("print_every"):
        jax.debug.callback(_print_progress, m, cfg["print_every"])
      return (params, opt, obs, st, rng, n_upd + 1), m

    (params, *_), metrics = jax.lax.scan(update, (params, opt, obs, st, rng, 0), None, num_updates)
    return params, metrics

  def evaluate(params, rng, zero_comm, steps=cfg["eval_steps"]):
    obs, st = jax.vmap(env.reset)(jax.random.split(rng, E))

    def f(c, _):
      obs, st, rng = c
      rng, k_a, k_s = jax.random.split(rng, 3)
      o = obs.at[..., -1].multiply(1.0 - zero_comm)   # zero_comm: 0.0 or 1.0
      logits, _ = policy(params, o)
      act = distrax.Categorical(logits=logits).sample(seed=k_a)
      obs, st, _, done, info = jax.vmap(env.step)(jax.random.split(k_s, E), st, act)
      return (obs, st, rng), (done, info["success"])

    _, (done, succ) = jax.lax.scan(f, (obs, st, rng), None, steps)
    return (succ & done).sum() / jnp.maximum(done.sum(), 1), done.sum()

  return train, evaluate, num_updates


def _print_progress(m, every):
  u = int(m["update"])
  if u == 1 or u % every == 0:
    print(f"[upd {u:5d} | {float(m['env_steps']):.3g} steps] "
          f"success={float(m['success']):.3f} ep_len={float(m['ep_len']):.1f} "
          f"sender_comm apple/no={float(m['sender_comm_apple']):.2f}/"
          f"{float(m['sender_comm_noapple']):.2f} "
          f"recv_comm={float(m['receiver_comm']):.2f} ent={float(m['entropy']):.2f}",
          flush=True)


DEFAULTS = dict(
    total_steps=2e7, num_envs=512, num_steps=128, lr=3e-4, update_epochs=4,
    num_minibatches=8, gamma=0.99, gae_lambda=0.95, clip=0.2, ent_coef=0.01,
    vf_coef=0.5, max_grad_norm=0.5, exists_prob=0.5, max_steps=100,
    separate=False, fixed_roles=False, no_comm=False, eval_steps=500,
)


def main():
  p = argparse.ArgumentParser()
  for k, v in DEFAULTS.items():
    if isinstance(v, bool):
      p.add_argument(f"--{k.replace('_', '-')}", action="store_true", default=v)
    else:
      p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
  p.add_argument("--seeds", type=int, default=4)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--out", default=None)
  a = p.parse_args()
  cfg = {k: getattr(a, k) for k in DEFAULTS}
  cfg["total_steps"] = float(cfg["total_steps"])
  print("devices:", jax.devices(), flush=True)
  print("config:", json.dumps(cfg), f"seeds={a.seeds}", flush=True)

  # Progress printing only for single-seed runs (callbacks under vmap would
  # print once per seed and interleave).
  cfg["print_every"] = 0 if a.seeds > 1 else 20
  train, evaluate, num_updates = make_train(cfg)
  keys = jax.random.split(jax.random.PRNGKey(a.seed), a.seeds)

  t0 = time.perf_counter()
  train_v = jax.jit(jax.vmap(train)).lower(keys).compile()
  print(f"compiled in {time.perf_counter() - t0:.1f}s; {num_updates} updates/seed", flush=True)
  t0 = time.perf_counter()
  params, metrics = jax.block_until_ready(train_v(keys))
  dt = time.perf_counter() - t0
  total = num_updates * cfg["num_envs"] * cfg["num_steps"] * a.seeds
  print(f"trained in {dt:.1f}s -> {total / dt:,.0f} env steps/s across {a.seeds} seeds "
        "(incl. learning)", flush=True)

  try:
    ms = jax.devices()[0].memory_stats() or {}
    print(f"peak GPU memory: {ms.get('peak_bytes_in_use', 0) / 2**30:.1f} GiB", flush=True)
  except Exception as e:  # CPU backend has no stats
    print("peak GPU memory: n/a", e, flush=True)

  metrics = jax.tree.map(np.asarray, metrics)
  every = max(1, num_updates // 10)
  for s in range(a.seeds):
    print(f"-- seed {s}")
    for u in list(range(0, num_updates, every)) + [num_updates - 1]:
      print(f"  upd {u + 1:5d}: success={metrics['success'][s, u]:.3f} "
            f"ep_len={metrics['ep_len'][s, u]:.1f} "
            f"sender_comm apple/no={metrics['sender_comm_apple'][s, u]:.2f}/"
            f"{metrics['sender_comm_noapple'][s, u]:.2f} "
            f"ent={metrics['entropy'][s, u]:.2f}")

  t0 = time.perf_counter()
  ev = jax.jit(jax.vmap(evaluate, in_axes=(0, 0, None)))
  ek = jax.random.split(jax.random.PRNGKey(a.seed + 1000), a.seeds)
  s_on, n_on = ev(params, ek, 0.0)
  s_off, n_off = ev(params, ek, 1.0)
  s_on, s_off = np.asarray(s_on), np.asarray(s_off)
  print(f"evaluated in {time.perf_counter() - t0:.1f}s", flush=True)
  results = []
  print("== evaluation (sampled actions) ==")
  for s in range(a.seeds):
    r = dict(seed=a.seed, index=s, success=float(s_on[s]),
             success_comm_zeroed=float(s_off[s]),
             ablation_drop=float(s_on[s] - s_off[s]), episodes=int(n_on[s]),
             train_success_last10=float(metrics["success"][s, -10:].mean()))
    results.append(r)
    print(json.dumps(r))
  learned = sum(r["success"] > 0.9 for r in results)
  print(f"seeds with success > 0.9: {learned}/{a.seeds}", flush=True)

  if a.out:
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "metrics.npz"), "wb") as f:
      np.savez(f, **metrics)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
      json.dump(dict(config=cfg, seeds=a.seeds, train_s=dt, eval=results), f, indent=1)
    with open(os.path.join(a.out, "params.pkl"), "wb") as f:
      pickle.dump(jax.device_get(params), f)
    print("saved to", a.out)


if __name__ == "__main__":
  main()
