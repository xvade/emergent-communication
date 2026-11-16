"""Compare the JAX toy (commsjax.toy_env) against the real Lua substrate.

Runs in the Melting Pot container (uses the checkout's toy config and Lua):
  sbatch agents/experiments/commsjax/compare_lua.sbatch

Measures, on the Lua toy:
  1. spawn: room assignment and orientation frequencies
  2. comm timing: does COMMUNICATE show up in BINARY_COMMS on the same step?
  3. episode cap: steps until the episode ends when nobody moves
  4. the memoryless scripted pair used in test_toy_env.py (sender holds
     COMMUNICATE while the apple is visible; receiver walks to the upper pad
     if the bit is 1, otherwise the lower pad), with and without the bit.
For comparison, the JAX env gives success 1.000 (bit used), 0.500 (bit
ignored) and mean episode length 2.50.
"""
import collections
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'agents/tests/communication'))
from mpviz import source  # noqa: E402
import check_substrate  # noqa: E402
import checks_toy  # noqa: E402

TOY = REPO / 'meltingpot/meltingpot/configs/substrates/communication/toy.py'
DIRS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}


def main():
  module = source.load_module(TOY, 'comm_toy')
  config = module.get_config()
  level = checks_toy.Level(check_substrate.raw_settings(module, config))
  game = checks_toy.Game(level, seed=123)
  P = game.players

  def act(moves=None, comm=()):
    a = {k: 0 for k in game.env.action_spec()}
    for p, d in (moves or {}).items():
      rel = (d - game.facing(p)) % 4
      a[f'{p}.move'] = (1, 2, 3, 4)[rel]
    for p in comm:
      a[f'{p}.binaryComms'] = 1
    game.ts = game.env.step(a)
    game.steps += 1
    rew = np.array([float(game.ts.observation.get(f'{p}.REWARD', 0.0)) for p in P])
    return game.ts.last(), rew

  def bit(p):
    return int(np.asarray(game.ts.observation[f'{p}.BINARY_COMMS']))

  # 1. spawn statistics
  rooms, dirs = collections.Counter(), collections.Counter()
  for _ in range(400):
    game.reset()
    rooms[game.room_of(P[0])] += 1
    for p in P:
      dirs[game.facing(p)] += 1
  print('SPAWN player0 room counts:', dict(rooms))
  print('SPAWN orientation counts (0=N..3=W):', dict(sorted(dirs.items())))

  # 2. comm timing
  game.reset()
  sender = next(p for p in P if game.room_of(p) == 'apple')
  b0 = [bit(p) for p in P]
  act(comm=[sender])
  b1 = [bit(p) for p in P]
  act()
  b2 = [bit(p) for p in P]
  print(f'COMM bit at reset={b0} after COMMUNICATE step={b1} after quiet step={b2}')

  # 3. episode cap with nobody moving
  game.reset()
  n, done = 0, False
  while not done and n < 1000:
    done, _ = act()
    n += 1
  print(f'CAP episode ended after {n} steps with no movement')

  # 4. scripted pair
  def run(use_bit, episodes=300):
    succ, lens = [], []
    for _ in range(episodes):
      game.reset()
      s = next(p for p in P if game.room_of(p) == 'apple')
      r = next(p for p in P if game.room_of(p) == 'pad')
      apple = game.apple_present()
      done, total, n = False, 0.0, 0
      while not done and n < 200:
        b = bit(r) if use_bit else 0
        tgt = level.upper if b else level.lower
        x, y = game.pos(r)
        dy = np.sign(tgt[1] - y)
        dx = np.sign(tgt[0] - x)
        d = 0 if dy < 0 else 2 if dy > 0 else (1 if dx > 0 else 3)
        done, rew = act(moves={r: d}, comm=[s] if apple else [])
        total += rew[0]
        n += 1
      succ.append(total > 0)
      lens.append(n)
    return np.mean(succ), np.mean(lens)

  for use_bit in (True, False):
    sr, ln = run(use_bit)
    print(f'SCRIPTED use_bit={use_bit}: success={sr:.3f} mean_len={ln:.2f}')


if __name__ == '__main__':
  main()
