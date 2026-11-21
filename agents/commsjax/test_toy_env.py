"""Checks for the JAX toy env (run inside a Slurm job):
  python -m commsjax.test_toy_env
Prints PASS/FAIL per check; exit status 1 if anything failed.
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

from commsjax import toy_env as T

env = T.ToyCommsEnv()
CH = {n: i for i, n in enumerate(T.CHANNELS)}
failed = []


def check(cond, name, detail=""):
  print(("PASS" if cond else "FAIL"), name, detail, flush=True)
  if not cond:
    failed.append(name)


def states(n, seed=0):
  ks = jax.random.split(jax.random.PRNGKey(seed), n)
  return jax.vmap(env.reset)(ks)


# --- reset: rooms, orientation, apple frequency --------------------------
obs, st = states(4000)
in_apple = np.asarray(jax.vmap(T.ToyCommsEnv.in_apple_room)(st))
check(np.all(in_apple.sum(1) == 1), "one player per room")
frac0 = in_apple[:, 0].mean()
check(abs(frac0 - 0.5) < 0.05, "rooms assigned at random", f"P(agent0 in apple room)={frac0:.3f}")
check(abs(np.asarray(st.apple).mean() - 0.5) < 0.05, "apple exists ~50%",
      f"{np.asarray(st.apple).mean():.3f}")
dirs = np.bincount(np.asarray(st.dir).ravel(), minlength=4) / st.dir.size
check(np.all(np.abs(dirs - 0.25) < 0.03), "orientation uniform", str(np.round(dirs, 3)))

# --- no leak: pad-room view identical with and without apple -------------
flip = st._replace(apple=~st.apple)
o1, o2 = jax.vmap(env.get_obs)(st), jax.vmap(env.get_obs)(flip)
pad_agent = np.argmax(~in_apple, 1)
idx = np.arange(len(pad_agent))
same = np.all(np.asarray(o1)[idx, pad_agent] == np.asarray(o2)[idx, pad_agent])
check(same, "pad-room observation independent of apple (no leak)")
# Also after random walks (positions all over the pad room).
def random_walk(st, key, n=30):
  def f(c, k):
    s = c
    a = jax.random.randint(k, (s.pos.shape[0], 2), 1, 7)   # moves/turns only
    _, s2, _, _, _ = jax.vmap(env.step_env)(s, a)
    # Undo episode ends: keep stepping state even past pads for this check.
    return s2._replace(t=jnp.zeros_like(s2.t)), None
  s, _ = jax.lax.scan(f, st, jax.random.split(key, n))
  return s
sw = random_walk(st, jax.random.PRNGKey(1))
o1 = np.asarray(jax.vmap(env.get_obs)(sw))
o2 = np.asarray(jax.vmap(env.get_obs)(sw._replace(apple=~sw.apple)))
in_apple_w = np.asarray(jax.vmap(T.ToyCommsEnv.in_apple_room)(sw))
pa = np.argmax(~in_apple_w, 1)
check(np.all(o1[idx, pa] == o2[idx, pa]), "no leak after random walks")

# --- rooms can't see each other -----------------------------------------
aa = np.argmax(in_apple_w, 1)
check(o1[idx, aa][..., [CH["upper_pad"], CH["lower_pad"], CH["other_player"]]].sum() == 0,
      "apple-room player never sees pads or the other player")
check(o1[idx, pa][..., [CH["apple"], CH["other_player"]]].sum() == 0,
      "pad-room player never sees the apple or the other player")
check(np.all(o1[..., CH["wall"]].reshape(len(idx), 2, -1).sum(-1) > 0), "walls visible")

# --- walls block --------------------------------------------------------
s0 = jax.tree.map(lambda x: x[0], st)
s0 = s0._replace(pos=jnp.array([list(T.APPLE_ROOM_SPAWN), list(T.PAD_ROOM_SPAWN)]),
                 dir=jnp.array([3, 1]))   # agent0 faces W, agent1 faces E
s = s0
for _ in range(5):
  _, s, _, _, _ = env.step_env(s, jnp.array([T.FORWARD, T.NOOP]))
check(tuple(np.asarray(s.pos[0])) == (1, 2), "wall stops movement", str(np.asarray(s.pos[0])))

# --- comm timing ---------------------------------------------------------
o, s, _, _, _ = env.step_env(s0, jnp.array([T.COMM, T.NOOP]))
check(np.all(np.asarray(o)[..., CH["comm_bit"]] == 1), "COMMUNICATE -> bit=1 in same step's obs (both players)")
o, s, _, _, _ = env.step_env(s, jnp.array([T.NOOP, T.NOOP]))
check(np.all(np.asarray(o)[..., CH["comm_bit"]] == 0), "bit resets next step when nobody talks")
o, _ = env.reset(jax.random.PRNGKey(3))
check(np.all(np.asarray(o)[..., CH["comm_bit"]] == 0), "bit=0 at reset")

env_nc = T.ToyCommsEnv(comm_enabled=False)
o, s, _, _, info = env_nc.step_env(s0, jnp.array([T.COMM, T.COMM]))
check(np.all(np.asarray(o)[..., CH["comm_bit"]] == 0) and int(info["comm"]) == 0,
      "comm_enabled=False: bit stays 0 after COMMUNICATE")

# --- pads: reward and termination for each (apple, pad) ------------------
for apple in (True, False):
  for pad, d in (("upper", 0), ("lower", 2)):
    s = s0._replace(apple=jnp.array(apple), dir=jnp.array([0, d]))
    steps, done = 0, False
    while not done and steps < 5:
      _, s, r, done, info = env.step_env(s, jnp.array([T.NOOP, T.FORWARD]))
      steps += 1
    want = 1.0 if (apple == (pad == "upper")) else 0.0
    check(bool(done) and np.all(np.asarray(r) == want),
          f"apple={apple} pad={pad}: ends, both get {want}",
          f"steps={steps} r={np.asarray(r)}")

# --- time limit ---------------------------------------------------------
s, done, n = s0, False, 0
while not done:
  _, s, _, done, _ = env.step_env(s, jnp.array([T.NOOP, T.NOOP]))
  n += 1
check(n == env.max_steps, "episode ends at max_steps", f"n={n}")

# --- auto-reset in step() ------------------------------------------------
s = s0._replace(dir=jnp.array([0, 0]))
o, s, r, done, _ = env.step(jax.random.PRNGKey(5), s, jnp.array([T.NOOP, T.FORWARD]))
check(bool(done) and int(s.t) == 0, "step() auto-resets after a pad")


# --- solvability: memoryless scripted policies ---------------------------
def scripted(obs, use_bit=True):
  """obs (2, 11, 11, C) -> actions (2,). No memory, obs only."""
  def one(o):
    sees_pads = o[..., CH["upper_pad"]].sum() > 0
    bit = o[0, 0, CH["comm_bit"]] if use_bit else 0.0
    # Sender: hold COMMUNICATE while the apple is visible.
    sender_act = jnp.where(o[..., CH["apple"]].sum() > 0, T.COMM, T.NOOP)
    # Receiver: walk to upper pad if bit else lower pad.
    tgt = jnp.where(bit > 0, o[..., CH["upper_pad"]], o[..., CH["lower_pad"]])
    r, c = jnp.unravel_index(jnp.argmax(tgt), tgt.shape)
    fwd, right = T.VIEW - r, c - T.VIEW
    recv_act = jnp.select([fwd > 0, fwd < 0, right > 0], [T.FORWARD, T.BACKWARD, T.STEP_RIGHT],
                          T.STEP_LEFT)
    return jnp.where(sees_pads, recv_act, sender_act)
  return jax.vmap(one)(obs)


def run_scripted(use_bit, n_envs=2000, steps=200):
  obs, st = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(11), n_envs))
  def f(c, k):
    obs, st = c
    act = jax.vmap(lambda o: scripted(o, use_bit))(obs)
    obs, st, _, done, info = jax.vmap(env.step)(jax.random.split(k, n_envs), st, act)
    return (obs, st), (done, info["success"], info["episode_length"])
  _, (d, s, l) = jax.lax.scan(f, (obs, st), jax.random.split(jax.random.PRNGKey(12), steps))
  d, s, l = map(np.asarray, (d, s, l))
  return s[d].mean(), d.sum(), l[d].mean()

rate, n, ln = run_scripted(True)
check(rate > 0.99, "memoryless scripted pair solves it", f"success={rate:.4f} over {n} eps, len={ln:.2f}")
rate, n, ln = run_scripted(False)
check(abs(rate - 0.5) < 0.03, "bit-ignoring receiver gets ~50%", f"success={rate:.4f} over {n} eps")

print("\nALL PASSED" if not failed else f"\nFAILED: {failed}")
sys.exit(1 if failed else 0)
