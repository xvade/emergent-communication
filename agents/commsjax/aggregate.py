"""Aggregate commsjax training runs into per-setup statistics.

  python -m commsjax.aggregate RESULT_DIR [RESULT_DIR ...] [--out FILE.json]

Each RESULT_DIR is one `ppo.py --out` directory (summary.json, metrics.npz).
Runs are grouped by setup: (shared|separate, random|fixed roles, comm|no-comm,
total steps). Seeds from several jobs with the same setup are pooled.

Per setup it reports:
* seeds that learned: final-parameter eval success > 0.9 AND mean training
  success over the last 10 updates > 0.9 (a snapshot eval alone can land in
  one of the temporary dips), with a 95% Wilson interval;
* J: mean eval success (= mean return per episode) over ALL seeds;
* the message convention of each learned seed (talk when the apple is there,
  or the inverse), from the last 10 updates;
* median env steps to first reach 0.99 training success (learned seeds);
* ablation drop: eval success minus eval success with the bit zeroed;
* Delta = J(comm) - J(no-comm) for the matching no-comm setup, with a 95%
  bootstrap interval. Seeds are independent runs, so they're resampled
  separately on each side, not paired.

Light enough for a login node (numpy only, small files).
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

LEARN = 0.9


def wilson(k, n, z=1.96):
  if n == 0:
    return (float("nan"),) * 2
  p = k / n
  d = 1 + z * z / n
  c = (p + z * z / (2 * n)) / d
  h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return max(0.0, c - h), min(1.0, c + h)


def boot_diff(a, b, n=10000, seed=0):
  rng = np.random.default_rng(seed)
  a, b = np.asarray(a), np.asarray(b)
  d = (rng.choice(a, (n, len(a))).mean(1) - rng.choice(b, (n, len(b))).mean(1))
  return np.percentile(d, [2.5, 97.5])


def load(dirs):
  cells = defaultdict(lambda: defaultdict(list))
  for d in dirs:
    with open(os.path.join(d, "summary.json")) as f:
      s = json.load(f)
    c = s["config"]
    m = np.load(os.path.join(d, "metrics.npz"))
    key = ("separate" if c["separate"] else "shared",
           "fixed" if c.get("fixed_roles", c["separate"]) else "random",
           "no-comm" if c.get("no_comm", False) else "comm",
           float(c["total_steps"]))
    cell = cells[key]
    steps_per_update = c["num_envs"] * c["num_steps"]
    for i, e in enumerate(s["eval"]):
      succ = m["success"][i]
      last10 = float(succ[-10:].mean())
      learned = e["success"] > LEARN and last10 > LEARN
      hit = np.flatnonzero(succ >= 0.99)
      on, off = (float(m["sender_comm_apple"][i, -10:].mean()),
                 float(m["sender_comm_noapple"][i, -10:].mean()))
      cell["J"].append(e["success"])
      cell["J_zeroed"].append(e["success_comm_zeroed"])
      cell["last10"].append(last10)
      cell["learned"].append(bool(learned))
      cell["steps_to_99"].append(float((hit[0] + 1) * steps_per_update) if len(hit) else None)
      cell["convention"].append(("apple" if on > off else "no-apple") if learned else None)
      cell["dips"].append(int((succ[hit[0]:] < 0.95).sum()) if len(hit) else None)
      cell["ep_len"].append(float(m["ep_len"][i, -10:].mean()))
      cell["src"].append(f"{os.path.basename(os.path.normpath(d))}:{i}")
  return cells


def main():
  p = argparse.ArgumentParser()
  p.add_argument("dirs", nargs="+")
  p.add_argument("--out")
  a = p.parse_args()
  cells = load(a.dirs)
  rows = []
  for key in sorted(cells):
    c = cells[key]
    n, k = len(c["J"]), sum(c["learned"])
    lo, hi = wilson(k, n)
    s99 = [x for x, l in zip(c["steps_to_99"], c["learned"]) if l and x]
    dips = [x for x, l in zip(c["dips"], c["learned"]) if l and x is not None]
    row = dict(
        params=key[0], roles=key[1], channel=key[2], total_steps=key[3], n=n,
        learned=k, learned_ci=[lo, hi],
        J_mean=float(np.mean(c["J"])), J_min=float(np.min(c["J"])), J_max=float(np.max(c["J"])),
        ablation_drop_mean=float(np.mean(np.array(c["J"]) - np.array(c["J_zeroed"]))),
        conv_apple=sum(x == "apple" for x in c["convention"]),
        conv_noapple=sum(x == "no-apple" for x in c["convention"]),
        median_steps_to_99=float(np.median(s99)) if s99 else None,
        dips_median=float(np.median(dips)) if dips else None,
        dips_max=int(max(dips)) if dips else None,
        not_learned=[(s, round(j, 3), round(l, 3)) for s, j, l, ok in
                     zip(c["src"], c["J"], c["last10"], c["learned"]) if not ok],
    )
    rows.append(row)
  # Delta against the matching no-comm cell (same params/roles; any step budget).
  base = {(r["params"], r["roles"]): r for r in rows if r["channel"] == "no-comm"}
  for r in rows:
    b = base.get((r["params"], r["roles"]))
    if r["channel"] == "comm" and b:
      kb = (b["params"], b["roles"], "no-comm", b["total_steps"])
      kr = (r["params"], r["roles"], "comm", r["total_steps"])
      lo, hi = boot_diff(cells[kr]["J"], cells[kb]["J"])
      r["delta"] = r["J_mean"] - b["J_mean"]
      r["delta_ci"] = [float(lo), float(hi)]

  print("| Networks | Roles | Channel | Steps | Seeds | Learned (95% CI) | J mean [min, max] "
        "| Δ vs no-comm (95% CI) | Ablation drop | Convention apple / no-apple "
        "| Median steps to 0.99 | Dips <0.95 after 0.99: median / max |")
  print("|" + "---|" * 12)
  for r in rows:
    d = (f"{r['delta']:.3f} ({r['delta_ci'][0]:.3f}–{r['delta_ci'][1]:.3f})"
         if "delta" in r else "–")
    s99 = f"{r['median_steps_to_99']:.2g}" if r["median_steps_to_99"] else "–"
    dips = f"{r['dips_median']:.0f} / {r['dips_max']}" if r["dips_max"] is not None else "–"
    print(f"| {r['params']} | {r['roles']} | {r['channel']} | {r['total_steps']:.0e} | {r['n']} "
          f"| {r['learned']}/{r['n']} ({r['learned_ci'][0]:.2f}–{r['learned_ci'][1]:.2f}) "
          f"| {r['J_mean']:.3f} [{r['J_min']:.3f}, {r['J_max']:.3f}] | {d} "
          f"| {r['ablation_drop_mean']:.3f} | {r['conv_apple']} / {r['conv_noapple']} "
          f"| {s99} | {dips} |")
  for r in rows:
    if r["not_learned"] and r["channel"] == "comm":
      print(f"\nnot learned, {r['params']}/{r['roles']} (job:index, eval J, last-10 train):",
            r["not_learned"])
  if a.out:
    with open(a.out, "w") as f:
      json.dump(rows, f, indent=1)


if __name__ == "__main__":
  main()
