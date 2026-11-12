"""Game-logic checks for configs/substrates/communication/toy.py.

Called by check_substrate.py as run(report, module, config, settings).

The toy, as built: two walled rooms. The apple room holds an apple that exists
with probability `existsProb`; the pad room holds an upper and a lower pad.
Stepping on a pad ends the episode; every player gets the pad's reward if it
is "active" (upper pad when the apple exists, lower pad when it doesn't),
otherwise nothing. Players can't see into the other room.

Everything is read from the config (prefabs, map, kwargs), not hard-coded, so
the checks keep working as the map and parameters change. Positions come from
a LocationObserver that the test adds to the avatars (your config is not
modified).
"""

import collections
import copy
import math

import dm_env
import numpy as np

from mpviz import source
from meltingpot.utils.substrates import builder

N_RESETS = 40           # resets for spawn and apple-frequency checks
MAX_PAD_EPISODES = 60   # episodes to try to see every (apple, pad) combo
DIRS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}  # N E S W (x, y)


# --------------------------------------------------------------------------
# Reading the level from the config
# --------------------------------------------------------------------------
def _plain(x):
  return x.to_dict() if hasattr(x, 'to_dict') else copy.deepcopy(x)


def _components(obj):
  return {c['component']: c.get('kwargs', {}) for c in obj.get('components', [])}


class Level:
  """Map, rooms, and object locations of the toy, from its settings."""

  def __init__(self, settings):
    self.settings = _plain(settings)
    sim = self.settings['simulation']
    self.prefabs = sim.get('prefabs', {})
    text = sim['map'].lstrip('\n')  # Lua strips leading newlines too.
    self.grid = {}
    for y, line in enumerate(text.split('\n')):
      for x, ch in enumerate(line):
        self.grid[(x, y)] = ch
    char_map = sim.get('charPrefabMap', {})
    self.char_prefab = {c: p for c, p in char_map.items() if isinstance(p, str)}

    def tiles_with(pred):
      return [xy for xy, ch in self.grid.items()
              if ch in self.char_prefab and pred(self.char_prefab[ch])]

    def has(component):
      return lambda name: component in _components(self.prefabs.get(name, {}))

    apples = tiles_with(has('DisappearingApple'))
    assert len(apples) == 1, f'expected 1 apple tile, found {len(apples)}'
    self.apple = apples[0]
    apple_kw = _components(self.prefabs[self.char_prefab[self.grid[
        self.apple]]])['DisappearingApple']
    self.exists_prob = float(apple_kw['existsProb'])
    self.upper_name = apple_kw['upperPadName']
    self.lower_name = apple_kw['lowerPadName']
    self.upper = tiles_with(lambda n: n == self.upper_name)
    self.lower = tiles_with(lambda n: n == self.lower_name)
    assert len(self.upper) == 1 and len(self.lower) == 1, 'expected 1 of each pad'
    self.upper, self.lower = self.upper[0], self.lower[0]
    self.reward = float(_components(self.prefabs[self.upper_name])
                        ['RewardWhenActiveAndEnd']['reward'])
    self.walls = {xy for xy, ch in self.grid.items()
                  if 'wall' in self.char_prefab.get(ch, '')}
    self.spawns = set(tiles_with(lambda n: 'spawn' in n))
    self.apple_room = self._region(self.apple)
    self.pad_room = self._region(self.upper)
    self.max_frames = int(self.settings.get('maxEpisodeLengthFrames', 0))

  def _region(self, start):
    """Tiles reachable from `start` without crossing walls (a room)."""
    seen, todo = {start}, [start]
    while todo:
      x, y = todo.pop()
      for dx, dy in DIRS.values():
        nxt = (x + dx, y + dy)
        if nxt in self.grid and nxt not in self.walls and nxt not in seen:
          seen.add(nxt)
          todo.append(nxt)
    return seen

  def path_step(self, start, goal, room):
    """First absolute direction (0..3) on a shortest path, or None."""
    prev = {start: None}
    todo = collections.deque([start])
    while todo:
      cur = todo.popleft()
      if cur == goal:
        break
      for d, (dx, dy) in DIRS.items():
        nxt = (cur[0] + dx, cur[1] + dy)
        if nxt in room and nxt not in prev:
          prev[nxt] = (cur, d)
          todo.append(nxt)
    if goal not in prev:
      return None
    node, d = goal, None
    while prev[node] is not None:
      node, d = prev[node]
    return d

  def avatar_views(self):
    """Max view reach (tiles) of each avatar definition, any rotation."""
    reaches = []
    for obj in self._avatar_objects():
      v = _components(obj)['Avatar'].get('view', {})
      reaches.append(max(v.get(k, 0) for k in
                         ('left', 'right', 'forward', 'backward')))
    return reaches

  def _avatar_objects(self):
    sim = self.settings['simulation']
    objs = [o for o in sim.get('gameObjects', []) if 'Avatar' in _components(o)]
    if 'avatar' in self.prefabs:
      objs.append(self.prefabs['avatar'])
    return objs

  def instrumented(self):
    """Settings with a LocationObserver on every avatar (positions)."""
    s = copy.deepcopy(self.settings)
    sim = s['simulation']
    targets = [o for o in sim.get('gameObjects', [])
               if 'Avatar' in _components(o)]
    if 'avatar' in sim.get('prefabs', {}):
      targets.append(sim['prefabs']['avatar'])
    for obj in targets:
      if 'LocationObserver' in _components(obj):  # e.g. debug observations on
        kw = _components(obj)['LocationObserver']
        kw['alsoReportOrientation'] = True
        continue
      obj['components'].append({
          'component': 'LocationObserver',
          'kwargs': {'objectIsAvatar': True, 'alsoReportOrientation': True}})
    return s


# --------------------------------------------------------------------------
# Driving the game
# --------------------------------------------------------------------------
class Game:
  """Raw dmlab2d env of the toy with positions, for scripted play."""

  def __init__(self, level, seed=7):
    self.level = level
    with source.source_levels():
      self.env = builder.builder(lab2d_settings=level.instrumented(),
                                 env_seed=seed)
    self.players = sorted({k.split('.', 1)[0] for k in self.env.action_spec()},
                          key=int)
    self.offset = None  # (dx, dy) from Lua positions to map coordinates

  def reset(self):
    self.ts = self.env.reset()
    self.steps, self.rewards = 0, np.zeros(len(self.players))
    if self.offset is None:
      self.offset = self._find_offset()
    return self.ts

  def _raw_pos(self, p):
    x, y = (int(v) for v in np.asarray(self.ts.observation[f'{p}.POSITION']))
    return x, y

  def _find_offset(self):
    """Lua positions vs. map coordinates: pick the offset that puts every
    player on a spawn tile (normally (0, 0))."""
    raw = [self._raw_pos(p) for p in self.players]
    for off in [(0, 0), (-1, -1), (1, 1), (-1, 0), (0, -1)]:
      if all((x + off[0], y + off[1]) in self.level.spawns for x, y in raw):
        return off
    raise AssertionError(f'player positions {raw} are not on spawn tiles '
                         f'{sorted(self.level.spawns)}')

  def pos(self, p):
    x, y = self._raw_pos(p)
    return x + self.offset[0], y + self.offset[1]

  def facing(self, p):
    return int(np.asarray(self.ts.observation[f'{p}.ORIENTATION']))

  def apple_present(self):
    """Apple tile in WORLD.RGB differs from an empty tile in the same room."""
    world = np.asarray(self.ts.observation['WORLD.RGB'])
    k = int(self.level.settings.get('spriteSize', 8))
    def tile(xy):
      x, y = xy
      return world[y * k:(y + 1) * k, x * k:(x + 1) * k]
    occupied = {self.pos(p) for p in self.players}
    empty = [xy for xy in self.level.apple_room
             if self.level.grid[xy] == ' ' and xy not in occupied]
    return not np.array_equal(tile(self.level.apple), tile(empty[0]))

  def step(self, moves):
    """moves: {player: absolute direction 0..3 or None (stand still)}."""
    action = {k: 0 for k in self.env.action_spec()}
    for p, d in moves.items():
      if d is not None:
        rel = (d - self.facing(p)) % 4  # egocentric: fwd, right, back, left
        action[f'{p}.move'] = (1, 2, 3, 4)[rel]
    self.ts = self.env.step(action)
    self.steps += 1
    self.rewards += [float(self.ts.observation.get(f'{p}.REWARD', 0.0))
                     for p in self.players]
    return self.ts.step_type == dm_env.StepType.LAST

  def room_of(self, p):
    xy = self.pos(p)
    if xy in self.level.apple_room:
      return 'apple'
    if xy in self.level.pad_room:
      return 'pad'
    return 'outside'


# --------------------------------------------------------------------------
def run(r, module, config, settings):
  del module
  try:
    level = Level(settings)
  except AssertionError as e:
    r.fail('read toy layout from config', str(e))
    return
  r.ok('read toy layout from config',
       f'apple {level.apple}, upper pad {level.upper}, lower pad '
       f'{level.lower}, existsProb {level.exists_prob}, '
       f'reward {level.reward:g}, time limit {level.max_frames}')

  # Static: players can't see into the other room from anywhere in theirs.
  reach = max(level.avatar_views() or [0])
  def cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
  apple_gap = min(cheb(t, level.apple) for t in level.pad_room)
  pad_gap = min(cheb(t, pad) for t in level.apple_room
                for pad in (level.upper, level.lower))
  r.check(apple_gap > reach,
          'pad-room player can never see the apple',
          f'closest pad-room tile is {apple_gap} tiles from the apple; '
          f'view reach {reach}')
  r.check(pad_gap > reach,
          'apple-room player can never see the pads',
          f'closest apple-room tile is {pad_gap} tiles from a pad; '
          f'view reach {reach}')

  game = Game(level)
  try:
    _dynamic_checks(r, config, level, game)
  finally:
    game.env.close()


def _dynamic_checks(r, config, level, game):
  del config  # unused
  # Spawning and apple frequency over many resets. Which player lands in
  # which room may vary between episodes; only one-per-room is required.
  rooms_ok, apple_seen = 0, 0
  for _ in range(N_RESETS):
    game.reset()
    rooms = {p: game.room_of(p) for p in game.players}
    if sorted(rooms.values()) == ['apple', 'pad']:
      rooms_ok += 1
    apple_seen += game.apple_present()
  r.check(rooms_ok == N_RESETS, 'one player spawns in each room',
          f'{rooms_ok}/{N_RESETS} resets')
  p, n = level.exists_prob, N_RESETS
  tol = 3.5 * math.sqrt(p * (1 - p) / n)
  frac = apple_seen / n
  r.check(abs(frac - p) <= tol + 1e-9,
          'apple appears with probability existsProb',
          f'{apple_seen}/{n} = {frac:.2f} (expected {p:.2f} +/- {tol:.2f})')

  # Pads: each (apple present?, pad) combination pays the right reward and
  # ends the episode on arrival.
  seen = {}
  problems = []
  for _ in range(MAX_PAD_EPISODES):
    if len(seen) == 4:
      break
    game.reset()
    apple = game.apple_present()
    walker = next((p for p in game.players if game.room_of(p) == 'pad'), None)
    if walker is None:
      problems.append('no player in the pad room')
      break
    target_name = 'upper' if ((apple, 'upper') not in seen) else 'lower'
    if (apple, target_name) in seen:
      continue
    target = level.upper if target_name == 'upper' else level.lower
    ended = False
    for _ in range(4 * len(level.pad_room)):
      d = level.path_step(game.pos(walker), target, level.pad_room)
      if d is None:
        break
      ended = game.step({walker: d})
      if ended:
        break
    arrived = game.pos(walker) == target
    correct = (target_name == 'upper') == apple
    expected = level.reward if correct else 0.0
    seen[(apple, target_name)] = (arrived, ended, game.rewards.tolist(),
                                  expected, game.steps)
  for (apple, pad), (arrived, ended, rewards, expected, steps) in sorted(
      seen.items()):
    label = (f'apple {"present" if apple else "absent "}, {pad} pad: '
             f'{"correct" if expected else "wrong"} choice')
    r.check(arrived and ended and all(abs(x - expected) < 1e-6
                                      for x in rewards),
            label,
            f'reached pad: {arrived}, episode ended: {ended} '
            f'(step {steps}), rewards {rewards}, expected {expected:g} each')
  missing = {(a, p) for a in (True, False) for p in ('upper', 'lower')} - set(
      seen)
  r.check(not missing and not problems, 'all 4 (apple, pad) cases covered',
          '; '.join(problems) or (f'not seen in {MAX_PAD_EPISODES} episodes: '
                                  f'{sorted(missing)}' if missing else ''))

  # Time limit: nobody moves -> ends at maxEpisodeLengthFrames, no reward.
  game.reset()
  ended = False
  while not ended and game.steps < level.max_frames + 5:
    ended = game.step({})
  r.check(ended and game.steps == level.max_frames and
          not game.rewards.any(),
          'standing still: episode ends at the time limit with 0 reward',
          f'ended {ended} at step {game.steps} (limit {level.max_frames}), '
          f'rewards {game.rewards.tolist()}')
