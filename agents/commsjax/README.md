# commsjax: JAX version of the `communication/toy` substrate, plus IPPO

| File | What it is |
|---|---|
| `toy_env.py` | JAX reimplementation of the toy (fork commit `390e2b9`). Symbolic 11×11×7 egocentric view (walls, apple, pads, other player, off-map, comm bit) instead of RGB. |
| `test_toy_env.py` | 21 checks: no information leak, rooms can't see each other, rewards, comm timing, time limit, scripted solvability. |
| `compare_lua.py` | Runs the same probes on the real Lua toy (Melting Pot container). |
| `ppo.py` | IPPO, fully jitted, seeds in parallel with vmap. Shared network (default) or `--separate` (fixed sender/receiver roles, one network each). Reports success, sender message rates by apple state, and success with the comm bit zeroed at evaluation ("ablation drop"). |

Run from the repo root (each launcher is a Slurm job):

```bash
sbatch agents/experiments/commsjax/test.sbatch            # env checks, CPU, ~2 min
sbatch agents/experiments/commsjax/compare_lua.sbatch     # vs. Lua toy, CPU, ~4 min
sbatch agents/experiments/commsjax/train.sbatch --seeds 4 [--separate] [--total-steps 2e7]
```

Training writes `agents/experiments/commsjax/results/<jobid>/` (`summary.json`,
`metrics.npz`, `params.pkl`). The log is in `agents/experiments/commsjax/logs/`.
The env is the SocialJax one built by `agents/experiments/socialjax/setup.sbatch`.
The launchers default to the copy at `/gscratch/amath/slszeto/envs/socialjax`,
which members of the `amath` group can read. Otherwise build your own with
`setup.sbatch` (see its header for staging the wheels) and point the launchers
at it with `SJ_ENV=/path/to/env sbatch ...`.

## JAX vs. Lua (measured 2026-11-13)

| Probe | Lua toy | JAX toy |
|---|---|---|
| Room assignment / starting orientation | random / random | random / random |
| Comm bit appears | same step, both players | same step, both players |
| Episode cap, nobody moving | 100 steps | 100 steps |
| Scripted memoryless pair, bit used | 1.000 success, 2.55 steps | 1.000, 2.50 steps |
| Scripted pair, bit ignored | 0.487 | 0.500 |

The observation format differs: symbolic grid here, RGB in Lua. Policies
trained here therefore don't run in the Lua toy without an adapter.

## First training results (2026-11-13, one L40S, 4 seeds × 2e7 steps, about 10 min + 2.5 min compile)

n = 4 seeds per setup.

| Setup | Seeds that learn (final success ≈ 1.0) | Success with comm zeroed at eval | Ablation drop |
|---|---|---|---|
| Shared network, random roles (job 43244543) | 3/4. Seed 2 stays at 0.50 (babbling). | ≈0.50 | ≈0.50 (seed 2: 0.00) |
| Separate networks, fixed roles (job 43244544) | 4/4 | ≈0.50 | ≈0.50 |

`--separate` changes two things at once (separate parameters and fixed roles),
so 4/4 vs 3/4 is not a controlled comparison.

**Learning isn't monotone.** After first reaching 0.99 success, seeds dip below
0.95 on 0–26 of about 230 later updates (worst single update 0.47), then recover.
In both runs every learned seed ends at 1.000 over its last 10 updates.
Evaluation is a snapshot of the final parameters.

**About "ablation drop" vs. the one-pager's Δ:** zeroing the bit on a pair
trained with communication isn't the reliance gain Δ = J(π_R) − J(π_R0). That
needs a baseline receiver *trained* without the channel. The two coincide here
only because the best return without communication is also 0.5. A `--no-comm`
training flag would give the proper baseline (not implemented yet).

Two message conventions emerged across seeds: "talk when the apple is there"
(episode length 2.5) and its inverse, "talk when it isn't" (3.6). In both, the
receiver follows the bit. The bit's meaning is arbitrary, as cheap talk
predicts. Feed-forward policies are enough here, because the toy is solvable
without memory.

The Lua comparison covers five probes. The scripted pair only walks along the
pad column, so turning, strafing, wall blocking and some pad/apple
combinations haven't been compared against Lua directly. The JAX unit tests
cover them against my reading of the Lua code. Replaying recorded Lua
trajectories through the JAX env would close that gap.
