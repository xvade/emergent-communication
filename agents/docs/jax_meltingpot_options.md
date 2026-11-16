# JAX implementations of Melting Pot: options for training

Written 2026-11-12. This covers how to train on Melting Pot-style environments
for many steps cheaply, and what that means for our communication substrates
(currently `communication/toy` in the `xvade/meltingpot` fork).

**What was checked, and how:** I read shallow clones of the SocialJax,
JaxMARL and marl-jax repositories and of the fork's toy substrate. The papers
I know only from search-result summaries.

> **Update 2026-11-13:** SocialJax is now installed and measured on klone (see
> "Measured on klone" below). Path A has been carried out for the toy:
> `agents/commsjax/` is a JAX port, checked against the Lua toy and trained
> with IPPO (see its README). Some statements below were written before those
> runs and are corrected inline.

## Main finding

**There is no JAX version of the Melting Pot *engine*.** Melting Pot games are
Lua scripts running on dmlab2d, a C++ CPU engine. Nothing compiles or translates
an arbitrary substrate into JAX. Each "JAX Melting Pot" project rewrites a fixed
set of games by hand, at about 1.4k–2.5k lines of `jax.numpy` per game.

That leaves two options:

* **Run any substrate unchanged** (including our Lua toy) on CPU dmlab2d.
  Measured on klone: **about 1,400–2,150 env steps/s** for one process.
  Conditions: Commons Harvest, 7 players, random actions, 1000-step episodes
  including reset, no learner (`mp-exp-42938292.log`). That's about 10⁸ env
  steps/day per core before any network forward or backward passes.
* **Run fast in JAX on the GPU**, but only on games that someone has written in
  JAX. That someone is us for any game SocialJax doesn't already include.

## What exists

| Project | What it has | Status | Use for us |
|---|---|---|---|
| **SocialJax** ([paper](https://arxiv.org/abs/2503.14576), [code](https://github.com/cooperativex/SocialJax), ICLR 2026) | JAX rewrites of Melting Pot 2.0 games: Coins, Harvest (Open/Closed/Partnership), Clean Up, Territory, Coop Mining, Mushrooms, Gift, PD Arena. (Territory has env code but no training config.) Algorithms: IPPO, MAPPO, VDN, SVO, reward-transfer. | Active (last commit Oct 2026). Apache-2.0. Pins `jax[cuda12]==0.6.2`, `jaxmarl==0.0.7`. | Best starting point. It has its own `MultiAgentEnv` base class, modelled on JaxMARL's: `reset(key)` and `step(key, state, actions)`, with dicts keyed by agent. |
| **JaxMARL** ([paper](https://arxiv.org/abs/2311.10090), [code](https://github.com/FLAIROx/JaxMARL)) | STORM, a grid version of Melting Pot's "*_in_the_matrix" games. Also Coin Game, MPE (which includes communication tasks), Switch Riddle (a communication debugging game), Hanabi, Overcooked. Baselines include `ippo_rnn.py` (recurrent IPPO). | Active (last commit Oct 2026). | Recurrent PPO baseline to borrow. STORM is a candidate mixed-motive game. |
| **marl-jax** ([paper](https://arxiv.org/abs/2303.13808)) | JAX *learner* on top of real (CPU) Melting Pot, so it handles any substrate. | Stale (last commit Oct 2023, `jax==0.4.1`, CUDA 11). | Not recommended. Its paper describes Melting Pot support, but its `requirements.txt` doesn't list Melting Pot, so I'm not sure how complete that support is. You'd also have to revive it. |

Reported speed: the SocialJax paper reports at least a 50× wall-clock speed-up
over Melting Pot's RLlib baselines. I haven't verified that on klone (see Next
steps).

### SocialJax details that matter for this project

These come from reading `algorithms/IPPO/` and the env files.

* **The code has a branch for separate sender and receiver networks.**
  `PARAMETER_SHARING: True` is the default in `ippo_base.yaml`. With `False`,
  the code builds one `ActorCritic` and `TrainState` per agent. *Verified:*
  job 43244241 trained Coins this way, 1e7 steps in 84 s after a 210 s
  compile, with returns rising from 0 to 84. A
  misaligned sender and receiver need this.
* **Reward mode.** The base config uses individual rewards
  (`shared_rewards: False`); the `reward=common` override shares them. For
  partially aligned incentives you'd add a custom reward mix, such as
  `r_S = α·r_R + (1-α)·b(state)`, inside the env's `step_env`.
* **Policies are feed-forward CNNs; no RNN.** *(For the toy, memory turned out
  to be unnecessary: memoryless policies solve it, see `agents/commsjax/`.)* A grep for GRU, LSTM and
  ScannedRNN found nothing in SocialJax's algorithms. The receiver has to
  remember messages, and our credibility estimate is a belief over time, so we'd
  take `ScannedRNN` / `ippo_rnn.py` from JaxMARL.
* **Fixed-length episodes** (`num_inner_steps`, default 1000). Ending an episode
  early on a terminal reward (as in our toy) is a small change to `step_env`.
* **None of the envs has a communication channel.** A grep for comm,
  message, utter and communication found nothing. It would have to be added to
  the action space and the observation (see path B).
* Reproduce paper curves with release `socialjax_v1.1.0`. Later versions fix an
  agent-overlap bug.

## Our toy substrate, and what porting it would take

At fork `main` (`390e2b9`, "Added a toy environment for binary communication")
the toy consists of `configs/substrates/communication/toy.py` (509 lines) and
`lua/levels/communication/toy/{init,components}.lua` (204 lines). Note that the
superproject still pins the submodule at `bc456f6`, which doesn't include the
toy.

The game: two walled 5×4 rooms, one player in each. One room holds an apple
with probability `existsProb`=0.5. The other has an upper (green) pad and a
lower (blue) pad. The upper pad pays out if the apple exists, otherwise the
lower one does. Stepping on either pad ends the episode, and both players get
1 if it was the correct pad. Communication is a single **global** bit: it is
"loud" this step if *any* player chose `COMMUNICATE`, a separate action that
can't be combined with moving. Every player observes it as `BINARY_COMMS` for
that step only. The observation is a scalar `Int32` read from the manager's
state, so it doesn't say who spoke.

**Porting estimate:** about 150–300 lines of JAX. There are 2 agents,
walls/apple/pads, moving and turning, a global comm bit, a terminal reward and
an egocentric observation. Without sprite rendering, the observation can be
symbolic one-hot channels, as SocialJax does with `cnn=True`. Correctness can be
checked against the Lua version by running both on the same action sequences.
`agents/tests/communication/checks_toy.py` already lists the behaviours to
match.

Design points to settle before porting. The JAX rewrite is a chance to make
these deliberate:

* With one shared bit, a player can't tell who spoke unless it remembers its own
  action. A per-sender channel (`N×K` message tensor) generalises better to more
  players and larger vocabularies.
* Since talking costs a movement step, cheap talk isn't free. That may be
  intended, but it changes the game theory compared with Crawford–Sobel.
* Rewards are fully shared right now, so the game is fully cooperative. The
  project needs a misalignment knob, such as a bias in the sender's reward.

## Three paths

| | Path | Cost | What you get | Risk |
|---|---|---|---|---|
| **A** | **Port the toy (and later games) to JAX** on the JaxMARL/SocialJax `MultiAgentEnv` interface. Fork SocialJax IPPO with `PARAMETER_SHARING=False` plus a JaxMARL `ScannedRNN`. | Toy: a few days, env plus tests. Each larger game: 1–3 weeks (SocialJax envs are 1.4–2.5k lines). | 10⁸–10⁹ steps feasible on one GPU. Full control over the message channel and the misalignment knob. | Every new game is a rewrite. The Lua and JAX versions can drift apart. |
| **B** | **Add a message channel to an existing SocialJax game** (for example Harvest or Clean Up). Add an `utter` action component and append the other agents' last messages to the observation. | About 1 week. | Real mixed-motive Melting Pot dynamics at JAX speed. Closest to the "extend to Melting Pot" part of the one-pager. | Misalignment comes from the game rather than a clean knob, so credibility is harder to interpret. |
| **C** | **Keep real Melting Pot on CPU** (dmlab2d, our Apptainer container, the Lua toy as is). Use vectorised CPU actors with a PyTorch or JAX PPO learner. | Hours to days. Our fork removed upstream's `examples/` (RLlib, PettingZoo, Gymnasium wrappers) and kept only the tutorial, so take those wrappers from [upstream](https://github.com/google-deepmind/meltingpot/tree/main/examples). | Works on *any* substrate, including future Lua edits, and stays in sync with `mp-play`/replays. | About 1.5k env steps/s per core without a learner (measured, see above). Policy inference and CPU↔GPU transfers add to that, and the toy's short episodes mean frequent resets. It scales only by adding cores (cpu-g2 has about 20 free right now). |

## Recommendation

Do **A for the toy**, then **B** when moving to Melting Pot-scale games.
Treat that as provisional until step 1 below confirms the JAX speed-up on
klone hardware. Keep the
Lua toy as the human-playable, replayable reference.

* The toy is small enough that a JAX port is cheap. The credibility experiments
  (sweeping misalignment, tracking reliance gain Δ over training, comparing
  receivers) need many seeds × many conditions × long runs, which is the
  workload JAX is good at.
* SocialJax gives the env interface and the IPPO training loop. JaxMARL gives
  the recurrent policy. Both are maintained and have similar `MultiAgentEnv`
  APIs. Combining them is untested: SocialJax pins `jaxmarl==0.0.7`, while
  `ippo_rnn.py` comes from current JaxMARL.
* Use C only for a final sanity check, for example running the JAX-trained
  policies in the real Lua substrate for replays. That requires the JAX
  observation format to match, or a small adapter.

This is a recommendation. Which path to take is your call.

## Measured on klone (2026-11-13, job 43243804, one L40S)

`agents/experiments/socialjax/speed.py`: random actions, no learner, 1000-step
jitted scan. Compile time is excluded from steps/s.

| env | agents | parallel envs | env steps/s | compile |
|---|---|---|---|---|
| coin_game | 2 | 1 / 128 / 1024 | 14k / 1.6M / **9.8M** | 4–10 s |
| harvest_common_open | 7 | 1 / 128 / 1024 | 6.5k / 0.7M / **4.5M** | 7–11 s |
| clean_up | 7 | 1 | 5.4k | 21 s |

For comparison, real Melting Pot Commons Harvest (7 players, one CPU process)
runs at about 1.5k env steps/s. SocialJax's harvest at 1024 envs is about
3000× faster. It isn't the same game, though: SocialJax uses a symbolic
observation and its own reimplemented rules.

Two things in that job didn't finish:
* `clean_up` with 128 envs hadn't finished when the 420 s limit for step 1
  hit, so no result.
* The short IPPO run printed nothing before its 300 s timeout. It was still
  compiling: cuDNN autotuning takes about 210 s. A rerun with a longer limit
  (job 43244241, `agents/experiments/socialjax/train_check.sbatch`) trained
  end to end: 1e7 steps in 84 s, about 119k env steps/s including learning.

## Next steps (not done yet)

1. ~~**Measure throughput on klone.**~~ Done for env stepping, see above. A 15-minute single-GPU sbatch on
   `gpu-l40s` (account `amath` has one free L40S right now) running
   `speed_test/speed_test_random.py`. The environment would go in
   `/gscratch/amath/slszeto/`, not `~`. `jax[cuda12]==0.6.2` should be
   fine on L40S, but that's unverified.
2. ~~Port the toy and train it.~~ Done: `agents/commsjax/` (feed-forward
   IPPO; no RNN needed for the toy).
