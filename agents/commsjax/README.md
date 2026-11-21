# commsjax: JAX version of the `communication/toy` substrate, plus IPPO

| File | What it is |
|---|---|
| `toy_env.py` | JAX reimplementation of the toy (fork commit `390e2b9`). Symbolic 11×11×7 egocentric view (walls, apple, pads, other player, off-map, comm bit) instead of RGB. |
| `test_toy_env.py` | 21 checks: no information leak, rooms can't see each other, rewards, comm timing, time limit, scripted solvability. |
| `compare_lua.py` | Runs the same probes on the real Lua toy (Melting Pot container). |
| `aggregate.py` | Pools seeds across runs: learned rate, J, Δ against the no-comm baseline, conventions, dips. |
| `ppo.py` | IPPO, fully jitted, seeds in parallel with vmap. Shared network (default) or `--separate` (one network per agent); random rooms (default) or `--fixed-roles`; `--no-comm` trains the no-communication baseline. Reports success, sender message rates by apple state, and success with the comm bit zeroed at evaluation ("ablation drop"). |

Run from the repo root (each launcher is a Slurm job):

```bash
sbatch agents/experiments/commsjax/test.sbatch            # env checks, CPU, ~2 min
sbatch agents/experiments/commsjax/compare_lua.sbatch     # vs. Lua toy, CPU, ~4 min
sbatch agents/experiments/commsjax/train.sbatch --seeds 8 [--separate] [--fixed-roles] [--no-comm]
sbatch agents/experiments/commsjax/sweep.sbatch           # 2x2 sweep + baselines, ~5.5 GPU-hours
```

Training writes `agents/experiments/commsjax/results/<jobid>/` (`summary.json`,
`metrics.npz`, `params.pkl`). The log is in `agents/experiments/commsjax/logs/`.
The env is the SocialJax one built by `agents/experiments/socialjax/setup.sbatch`.
The launchers default to the copy at `/gscratch/amath/slszeto/envs/socialjax`,
which members of the `amath` group can read. Otherwise build your own with
`setup.sbatch` (see its header for staging the wheels) and point the launchers
at it with `SJ_ENV=/path/to/env sbatch ...`.

## JAX vs. Lua (measured 2026-11-13)

Re-run 2026-11-18 against the refactored fork (`db80057`, where the one-bit
game is `toy_binary.py`): same Lua numbers on all five probes (job 43364024).
The JAX port still uses the old action order (NOOP = 0, COMMUNICATE = 7);
`toy_binary.py` puts COMMUNICATE at 0 and NOOP at 1.

| Probe | Lua toy | JAX toy |
|---|---|---|
| Room assignment / starting orientation | random / random | random / random |
| Comm bit appears | same step, both players | same step, both players |
| Episode cap, nobody moving | 100 steps | 100 steps |
| Scripted memoryless pair, bit used | 1.000 success, 2.55 steps | 1.000, 2.50 steps |
| Scripted pair, bit ignored | 0.487 | 0.500 |

The observation format differs: symbolic grid here, RGB in Lua. Policies
trained here therefore don't run in the Lua toy without an adapter.

## Training results: 2x2 sweep (2026-11-16, job 43299285)

Each setup with communication has 32 seeds at 2e7 steps. Each no-communication
baseline has 8 seeds at 5e6 steps. J is the mean success rate (mean return per
episode) of the final parameters. Δ = J(comm) − J(no-comm) for the same setup.
Intervals are 95% (Wilson for counts, bootstrap for Δ).

| Networks | Roles | Seeds at ≈1.0 at the end | J with comm, mean [min, max] | J without comm | Δ (95% CI) |
|---|---|---|---|---|---|
| shared | random | 31/32 (0.84–0.99) | 0.981 [0.560, 1.000] | 0.500 | 0.481 (0.451–0.499) |
| shared | fixed | 31/32 (0.84–0.99) | 0.985 [0.566, 1.000] | 0.500 | 0.486 (0.458–0.500) |
| separate | random | 30/32 (0.80–0.98) | 0.978 [0.736, 1.000] | 0.500 | 0.478 (0.454–0.495) |
| separate | fixed | 32/32 (0.89–1.00) | 1.000 [1.000, 1.000] | 0.500 | 0.500 (0.499–0.500) |

A seed counts as "at ≈1.0 at the end" if its final evaluation and its mean
training success over the last 10 updates both exceed 0.9.

* **The four setups don't differ measurably in final success.** The
  intervals overlap. They do differ in how fast they learn (below).
* **Every no-communication seed scores 0.50** (range 0.498–0.502 over 32
  seeds). Δ is therefore the success rate with communication minus 0.5.
* **No seed babbled for the whole 2e7 steps.** 126 of 128 seeds reached at
  least 0.98 training success at some point, and the other two reached 0.74.
  In the first runs (4 seeds per setup, jobs 43244543 and 43244544) one of 8
  seeds did stay at 0.50 throughout (peak 0.51). Babbling at this budget is
  probably the slow tail of learning times: the slowest shared-network seeds
  here first reached 0.99 at 1.7e7–2e7 steps. The four seeds below 1.0 at the
  end failed in two other ways:
  * *Late collapse (2 seeds, both shared-network).* The pair reached 1.00,
    then fell back to 0.50 in the last updates while the sender kept
    signalling correctly: the receiver stopped following the bit. One fell
    2 updates before the end (`shared-random-comm-s0` index 4), the other
    within the last 30 (`shared-fixed-comm-s100` index 0). Training ended
    before either could recover, so I can't say whether they would have.
  * *Half-learned (2 seeds, both separate/random).* Success sits at 0.74 from
    about 8e6 steps on. A likely reading is that the pair communicates in one
    room assignment and guesses in the other (0.5·1 + 0.5·0.5 = 0.75). I
    haven't checked this with a per-assignment evaluation.
* **Most dips are short, but a few are long.** After first reaching 0.99, a
  seed that ends at ≈1.0 falls below 0.95 on a median of 0–1 later updates
  (maximum 38 of about 230). Counting all seeds, three stayed below 0.6 for 10 or more updates in a row; two of
  them recovered before the end. A final-parameter snapshot can therefore
  land in a collapse, so keep a checkpoint from a stable stretch if the
  policies are reused.
* **"Talk when the apple is absent" is the more common convention**: 93 of
  the 124 seeds that ended at ≈1.0, against 31 for "talk when the apple is
  there". I don't know why. A guess: the lower pad is one step further, and
  the idle receiver drifts to the nearer upper pad, so silence comes to mean
  "upper".
* **Separate networks with fixed roles learn fastest.** Steps to first reach
  0.99 training success, over all 32 seeds:

  | Setup | Median | Slowest | Never reached 0.99 |
  |---|---|---|---|
  | separate, fixed | 4.6e6 | 7.5e6 | 0 |
  | shared, random | 8.7e6 | 1.7e7 | 3 |
  | shared, fixed | 8.8e6 | 2.0e7 | 1 |
  | separate, random | 1.3e7 | 1.8e7 | 5 |

  Seven of the nine seeds that never reached 0.99 peaked at 0.98–0.99. The
  other two are the half-learned seeds.

Reproduce: `sbatch agents/experiments/commsjax/sweep.sbatch`, then
`python -m commsjax.aggregate agents/experiments/commsjax/results/sweep-<id>/*`.
8 seeds per job fit on a 48 GB L40S and 16 run out of memory (9–15 untested).
One GPU does about 150,000 env steps/s in total, so the sweep took 5.4 GPU-hours.

**Ablation drop vs. Δ.** Zeroing the bit at evaluation on a pair trained with
communication ("ablation drop") isn't the reliance gain Δ = J(π_R) − J(π_R0),
which needs a baseline trained without the channel (`--no-comm`). In this game
the two agree (ablation drop 0.478–0.500 across setups), because the baseline
can't beat 0.5 either way.

Feed-forward policies are enough here, because the toy is solvable without
memory.

The Lua comparison covers five probes. The scripted pair only walks along the
pad column, so turning, strafing, wall blocking and some pad/apple
combinations haven't been compared against Lua directly. The JAX unit tests
cover them against my reading of the Lua code. Replaying recorded Lua
trajectories through the JAX env would close that gap.
