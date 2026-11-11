"""Simple Melting Pot experiment: does zapping change collective return?

Runs uniformly-random agents in a substrate under two conditions:
  * "random":  sample uniformly from the full action set (includes FIRE_ZAP)
  * "no_zap":  same, but FIRE_ZAP (the last action) is never chosen
and reports per-episode collective and per-agent returns.

Usage (inside the container):
  python random_zap_ablation.py --substrate commons_harvest__open \
      --episodes 3 --steps 500 --out results/
"""

import argparse
import csv
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np  # pylint: disable=g-import-not-at-top
from meltingpot import substrate  # pylint: disable=g-import-not-at-top

try:  # agents/ on PYTHONPATH (agents/bin launchers set this).
  from mpviz import recording  # pylint: disable=g-import-not-at-top
except ImportError:
  import sys  # pylint: disable=g-import-not-at-top
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
  from mpviz import recording  # pylint: disable=g-import-not-at-top


def run_episode(env, rng, num_actions, allowed, max_steps):
  timestep = env.reset()
  n = len(timestep.observation)
  returns = np.zeros(n)
  steps = 0
  while not timestep.last() and steps < max_steps:
    actions = rng.choice(allowed, size=n).tolist()
    timestep = env.step(actions)
    returns += np.asarray(timestep.reward, dtype=float)
    steps += 1
  frame = timestep.observation[0].get("WORLD.RGB")
  return returns, steps, frame


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--substrate", default="commons_harvest__open")
  p.add_argument("--episodes", type=int, default=3)
  p.add_argument("--steps", type=int, default=500)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--out", default="results")
  p.add_argument("--record", type=int, default=0,
                 help="save the first N episodes per condition for replay "
                      "(to <out>/recordings; view with agents/bin/mp-replay)")
  p.add_argument("--record-player-views", action="store_true",
                 help="also record each player's egocentric view")
  args = p.parse_args()
  os.makedirs(args.out, exist_ok=True)

  config = substrate.get_config(args.substrate)
  roles = config.default_player_roles
  env = substrate.build(args.substrate, roles=roles)
  num_actions = env.action_spec()[0].num_values
  print(f"substrate={args.substrate} players={len(roles)} "
        f"actions={num_actions}", flush=True)

  conditions = {
      "random": np.arange(num_actions),
      "no_zap": np.arange(num_actions - 1),  # FIRE_ZAP is last in ACTION_SET
  }
  rows = []
  rng = np.random.default_rng(args.seed)
  saved = []
  for cond, allowed in conditions.items():
    rec_env = None
    if args.record:
      rec_env = recording.wrap(
          env, os.path.join(args.out, "recordings"), substrate=args.substrate,
          tag=cond, max_episodes=args.record,
          record_player_views=args.record_player_views,
          extra_meta={"condition": cond, "seed": args.seed})
    for ep in range(args.episodes):
      t0 = time.time()
      returns, steps, frame = run_episode(rec_env or env, rng, num_actions,
                                          allowed, args.steps)
      dt = time.time() - t0
      rows.append({"condition": cond, "episode": ep, "steps": steps,
                   "collective_return": returns.sum(),
                   "per_agent": " ".join(f"{r:g}" for r in returns),
                   "steps_per_sec": steps / dt})
      print(f"[{cond}] ep={ep} steps={steps} collective={returns.sum():g} "
            f"({steps / dt:.0f} steps/s)", flush=True)
      if frame is not None and ep == 0:
        import cv2  # pylint: disable=g-import-not-at-top
        cv2.imwrite(os.path.join(args.out, f"final_frame_{cond}.png"),
                    cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    if rec_env is not None:
      rec_env.flush()  # episodes cut off by --steps never reach last()
      saved += rec_env.saved
  env.close()

  path = os.path.join(args.out, f"{args.substrate}.csv")
  with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

  print("\nSummary (collective return, mean ± std over episodes):")
  for cond in conditions:
    vals = np.array([r["collective_return"] for r in rows
                     if r["condition"] == cond])
    print(f"  {cond:7s} {vals.mean():8.2f} ± {vals.std():.2f}")
  print(f"wrote {path}")
  for s in saved:
    print(f"recorded {s}")


if __name__ == "__main__":
  main()
