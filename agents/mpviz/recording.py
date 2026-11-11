"""Record Melting Pot episodes for later replay.

Format: `<name>.npz` (compressed arrays) + `<name>.json` (metadata).

Arrays (T = number of frames = steps + 1; row 0 is the reset frame and its
actions/rewards are zero):
  world      uint8 (T, H, W, 3)       WORLD.RGB
  players    uint8 (T, N, h, w, 3)    per-player RGB views (optional)
  rewards    float32 (T, N)           reward received on arriving at frame t
  actions    int32 (T, N, A)          action that led to frame t
  active     int16 (T,)               human-controlled player index, -1 if none

Two timestep formats are supported:
  * Substrate API (`meltingpot.substrate.build`): observation is a list of
    per-player dicts, reward is a sequence, action is a list of ints.
  * Raw dmlab2d API (used by the human-play builder): observation is a flat
    dict with keys like '1.RGB', '1.REWARD', 'WORLD.RGB'; action is a dict
    like {'1.move': 3, '1.turn': 0, ...}.

Typical experiment use:

    from mpviz import recording
    env = recording.wrap(substrate.build(name, roles=roles), out_dir,
                         substrate=name, max_episodes=2)
    ... use env exactly as before; each finished episode is saved ...
"""

import datetime
import json
import os
import re
from typing import Any, Optional, Sequence

import numpy as np


def _jsonable(x: Any) -> Any:
  if isinstance(x, np.ndarray):
    return x.tolist() if x.size <= 64 else f'<array {x.shape} {x.dtype}>'
  if isinstance(x, (np.integer, np.floating, np.bool_)):
    return x.item()
  if isinstance(x, bytes):
    return x.decode('utf-8', 'replace')
  if isinstance(x, dict):
    return {str(k): _jsonable(v) for k, v in x.items()}
  if isinstance(x, (list, tuple)):
    return [_jsonable(v) for v in x]
  if isinstance(x, (str, int, float, bool)) or x is None:
    return x
  return repr(x)


def _player_prefixes(obs: dict) -> list[str]:
  prefixes = {k.split('.', 1)[0] for k in obs if re.match(r'^\d+\.', k)}
  return sorted(prefixes, key=int)


class EpisodeRecorder:
  """Accumulates one episode in memory; call save() at the end."""

  def __init__(self, *, substrate: str, record_player_views: bool = False,
               fps: int = 8, source: str = 'experiment',
               extra_meta: Optional[dict] = None):
    self.substrate = substrate
    self.record_player_views = record_player_views
    self.meta = dict(substrate=substrate, fps=fps, source=source,
                     created=datetime.datetime.now().isoformat(
                         timespec='seconds'))
    if extra_meta:
      self.meta.update(extra_meta)
    self.clear()

  def clear(self):
    self._world, self._players = [], []
    self._rewards, self._actions, self._active = [], [], []
    self._events: list[list] = []
    self.action_names: Optional[list[str]] = None

  def __len__(self):
    return len(self._world)

  # -- generic entry point ------------------------------------------------
  def add(self, world, rewards, actions, players=None, active=-1,
          events=None):
    self._world.append(np.asarray(world, dtype=np.uint8))
    self._rewards.append(np.asarray(rewards, dtype=np.float32))
    # actions=None (e.g. the reset frame) is filled with zeros in save(),
    # once the action width is known.
    self._actions.append(None if actions is None else np.asarray(
        actions, dtype=np.int32).reshape(len(rewards), -1))
    if self.record_player_views and players is not None:
      self._players.append(np.stack(players).astype(np.uint8))
    self._active.append(active)
    self._events.append(_jsonable(events) if events else [])

  # -- Substrate API --------------------------------------------------------
  def add_substrate_timestep(self, timestep, actions=None, events=None):
    obs = timestep.observation
    n = len(obs)
    if self.action_names is None:
      self.action_names = ['action']
    rewards = (np.zeros(n) if timestep.reward is None
               else np.asarray(timestep.reward, dtype=np.float32))
    acts = None if actions is None else np.asarray(actions)
    players = [o['RGB'] for o in obs] if 'RGB' in obs[0] else None
    self.add(obs[0]['WORLD.RGB'], rewards, acts, players, events=events)

  # -- raw dmlab2d API ------------------------------------------------------
  def add_lab2d_timestep(self, timestep, actions: Optional[dict] = None,
                         active=-1, events=None):
    obs = timestep.observation
    prefixes = _player_prefixes(obs)
    if self.action_names is None and actions:
      self.action_names = sorted({k.split('.', 1)[1] for k in actions})
    names = self.action_names or []
    rewards = [float(obs.get(f'{p}.REWARD', 0.0)) for p in prefixes]
    acts = None
    if actions:
      acts = [[actions.get(f'{p}.{a}', 0) for a in names] for p in prefixes]
    players = None
    if all(f'{p}.RGB' in obs for p in prefixes):
      players = [obs[f'{p}.RGB'] for p in prefixes]
    self.add(obs['WORLD.RGB'], rewards, acts, players, active, events)

  # -- output ---------------------------------------------------------------
  def save(self, path_stem: str) -> str:
    """Writes <path_stem>.npz and <path_stem>.json; returns the .npz path."""
    if not self._world:
      raise ValueError('Nothing recorded.')
    os.makedirs(os.path.dirname(os.path.abspath(path_stem)), exist_ok=True)
    rewards = np.stack(self._rewards)
    width = next((a.shape[1] for a in self._actions if a is not None), 1)
    zeros = np.zeros((rewards.shape[1], width), dtype=np.int32)
    actions = np.stack([zeros if a is None else a for a in self._actions])
    rewards[0] = 0
    actions[0] = 0
    arrays = dict(world=np.stack(self._world), rewards=rewards,
                  actions=actions,
                  active=np.asarray(self._active, dtype=np.int16))
    if self._players and len(self._players) == len(self._world):
      arrays['players'] = np.stack(self._players)
    np.savez_compressed(path_stem + '.npz', **arrays)
    meta = dict(self.meta, steps=len(self._world) - 1,
                num_players=int(rewards.shape[1]),
                action_names=self.action_names or ['action'],
                total_rewards=rewards.sum(0).tolist(),
                events=self._events)
    with open(path_stem + '.json', 'w') as f:
      json.dump(meta, f)
    return path_stem + '.npz'


def default_stem(out_dir: str, substrate: str, tag: str = '') -> str:
  ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
  return os.path.join(out_dir, f'{substrate}{"-" + tag if tag else ""}-{ts}')


class RecordingSubstrate:
  """Wraps a Substrate; records and saves episodes transparently.

  Saves each episode when it ends (timestep.last()) or when reset() is
  called mid-episode. Stops recording after `max_episodes` (None = all).
  `self.saved` lists the .npz paths written.
  """

  def __init__(self, env, out_dir: str, *, substrate: str, tag: str = '',
               max_episodes: Optional[int] = None,
               record_player_views: bool = False, fps: int = 8,
               extra_meta: Optional[dict] = None):
    self._env = env
    self._out_dir = out_dir
    self._tag = tag
    self._max = max_episodes
    self._rec = EpisodeRecorder(substrate=substrate, fps=fps,
                                record_player_views=record_player_views,
                                extra_meta=extra_meta)
    self.saved: list[str] = []

  def __getattr__(self, name):
    return getattr(self._env, name)

  def _active(self):
    return self._max is None or len(self.saved) < self._max

  def _events(self):
    try:
      return self._env.events() if hasattr(self._env, 'events') else None
    except Exception:  # pylint: disable=broad-except
      return None

  def flush(self):
    if len(self._rec) > 1 and self._active():
      self.saved.append(self._rec.save(
          default_stem(self._out_dir, self._rec.substrate, self._tag)))
    self._rec.clear()

  def reset(self, *a, **kw):
    self.flush()
    ts = self._env.reset(*a, **kw)
    if self._active():
      self._rec.add_substrate_timestep(ts, events=self._events())
    return ts

  def step(self, actions: Sequence[int]):
    ts = self._env.step(actions)
    if self._active() and len(self._rec):
      self._rec.add_substrate_timestep(ts, actions, events=self._events())
      if ts.last():
        self.flush()
    return ts

  def close(self):
    self.flush()
    return self._env.close()


def wrap(env, out_dir, **kw) -> RecordingSubstrate:
  return RecordingSubstrate(env, out_dir, **kw)
