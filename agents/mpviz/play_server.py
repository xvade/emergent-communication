"""Play a Melting Pot substrate as a human, in your browser.

Run on a compute node (see agents/bin/mp-play), then open the printed URL
through an SSH tunnel. Controls are exactly those of Melting Pot's own
`meltingpot/human_players/play_*.py` scripts: we reuse their action maps and
feed them the keys your browser reports instead of a pygame window. Players
you are not controlling take no-op actions, as in the original scripts.

  python -m mpviz.play_server --substrate commons_harvest__open
"""

import argparse
import base64
import collections
import glob
import importlib
import inspect
import json
import os
import pathlib
import pkgutil
import re
import secrets
import signal
import socket
import sys
import threading
import time
import traceback
import types
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

import cv2  # pylint: disable=g-import-not-at-top
import dm_env  # pylint: disable=g-import-not-at-top
from meltingpot import human_players  # pylint: disable=g-import-not-at-top
from meltingpot.human_players import level_playing_utils  # pylint: disable=g-import-not-at-top
from meltingpot.utils.substrates import builder  # pylint: disable=g-import-not-at-top
from ml_collections import config_dict  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=g-import-not-at-top
import pygame  # pylint: disable=g-import-not-at-top

from mpviz import recording  # pylint: disable=g-import-not-at-top
from mpviz import replay_html  # pylint: disable=g-import-not-at-top
from mpviz import source  # pylint: disable=g-import-not-at-top


# --------------------------------------------------------------------------
# Browser keys -> pygame.key.get_pressed()
# --------------------------------------------------------------------------
class _BrowserKeys:
  """Stands in for pygame's ScancodeWrapper: keys[pygame.K_w] -> bool."""

  def __init__(self):
    self.pressed: set[int] = set()

  def set(self, names):
    codes = set()
    for name in names:
      code = getattr(pygame, f'K_{name}', None)
      if isinstance(code, int):
        codes.add(code)
    self.pressed = codes

  def __getitem__(self, code):
    return code in self.pressed


KEYS = _BrowserKeys()
# Every play_*.py action function calls pygame.key.get_pressed() at step
# time, so replacing it routes browser input through Melting Pot's own maps.
pygame.key.get_pressed = lambda: KEYS


# --------------------------------------------------------------------------
# Registry of playable substrates (from meltingpot/human_players/play_*.py)
# --------------------------------------------------------------------------
def discover():
  registry = {}
  for info in pkgutil.iter_modules(human_players.__path__):
    if not info.name.startswith('play_') or info.name.endswith('_test'):
      continue
    mod = importlib.import_module(f'meltingpot.human_players.{info.name}')
    if not hasattr(mod, 'environment_configs') or not hasattr(
        mod, '_ACTION_MAP'):
      continue
    for level, env_module in mod.environment_configs.items():
      registry[level] = (mod, env_module)
  return dict(sorted(registry.items()))


# --------------------------------------------------------------------------
# How-to-play text, derived from the code that actually reads the keys
# --------------------------------------------------------------------------
_KEY_LABELS = {
    'SPACE': 'Space', 'UP': '\u2191', 'DOWN': '\u2193', 'LEFT': '\u2190',
    'RIGHT': '\u2192', 'LSHIFT': 'Left Shift', 'RSHIFT': 'Right Shift',
    'LCTRL': 'Left Ctrl', 'RCTRL': 'Right Ctrl', 'DELETE': 'Delete',
    'PAGEDOWN': 'Page Down', 'TAB': 'Tab'}

# level_playing_utils helpers whose return values need a human explanation.
_CURATED_KEYS = {
    'get_direction_pressed': [
        ('W / \u2191', 'forward'), ('S / \u2193', 'backward'),
        ('A / \u2190', 'step left'), ('D / \u2192', 'step right')],
    'get_turn_pressed': [('Q', 'turn left'), ('E', 'turn right')],
    'get_key_number_pressed': [('0\u20139', 'sends the number pressed')],
}

_ACTION_LABELS = {
    'move': 'Move', 'turn': 'Turn', 'fireZap': 'Fire zap beam',
    'fireClean': 'Fire cleaning beam', 'fireClaim': 'Fire claiming beam',
    'fire_1': 'Planting beam 1', 'fire_2': 'Planting beam 2',
    'fire_3': 'Planting beam 3',
}

# Named meanings of signed action values, where Melting Pot's configs name
# them (e.g. fruit_market.py: SHOVE = shove 1, PULL = shove -1).
_VALUE_NAMES = {'shove': {'1': 'push', '-1': 'pull'}}


def _key_label(name):
  return _KEY_LABELS.get(name, name.upper())


def _humanize(action):
  words = re.sub(r'([a-z])([A-Z0-9])', r'\1 \2', action).replace('_', ' ')
  return words[:1].upper() + words[1:].lower()


def _resolve(name, play_mod):
  return (getattr(play_mod, name, None) or
          getattr(level_playing_utils, name, None))


def _keys_on_line(line, play_mod, depth=0):
  """Key names a line of source reads, following calls to get_* helpers."""
  keys = re.findall(r'K_(\w+)', line)
  if depth < 3:
    for fname in re.findall(r'\b(get_\w+)\(', line):
      fn = _resolve(fname, play_mod)
      if fn is not None and fn.__name__ not in _CURATED_KEYS:
        for src_line in inspect.getsource(fn).splitlines():
          keys += _keys_on_line(src_line, play_mod, depth + 1)
  return list(dict.fromkeys(keys))


def _describe_keys(fn, play_mod):
  """[(keys text, effect text or None)] for one action-map function."""
  if fn.__name__ in _CURATED_KEYS:
    return _CURATED_KEYS[fn.__name__]
  lines = inspect.getsource(fn).splitlines()[1:]  # skip the def line
  pairs = []
  for i, line in enumerate(lines):
    if not re.match(r'\s*(el)?if\b', line):
      continue
    keys = _keys_on_line(line, play_mod)
    ret = next((re.search(r'return\s+(.+)', l) for l in lines[i + 1:i + 3]
                if 'return' in l), None)
    if keys and ret:
      pairs.append((' / '.join(map(_key_label, keys)), ret.group(1).strip()))
  if pairs:
    return pairs
  keys = []
  for line in lines:
    keys += _keys_on_line(line, play_mod)
  keys = list(dict.fromkeys(keys))
  return [(' / '.join(map(_key_label, keys)) or '?', None)]


def describe_controls(play_mod):
  rows = []
  for action, fn in play_mod._ACTION_MAP.items():  # pylint: disable=protected-access
    try:
      keys = _describe_keys(fn, play_mod)
    except (OSError, TypeError):
      keys = [('?', None)]
    names = _VALUE_NAMES.get(action, {})
    keys = [(k, names.get(e, f'sends {e}') if e and e.lstrip('-').isdigit()
             else e) for k, e in keys]
    rows.append(dict(action=action,
                     label=_ACTION_LABELS.get(action, _humanize(action)),
                     keys=[dict(keys=k, effect=e) for k, e in keys]))
  return rows


def describe_substrate(env_module):
  """The substrate's own description (its config module docstring)."""
  doc = inspect.cleandoc(env_module.__doc__ or '')
  doc = re.sub(r'^Configuration for [^\n]*\n+', '', doc)
  # Undo the docstring's hard wrapping: one line per paragraph.
  paras = [' '.join(p.split()) for p in re.split(r'\n\s*\n', doc)]
  return '\n\n'.join(p for p in paras if p)


def build_env_config(play_mod, env_module):
  """Mirrors the env construction in each play_*.py main()."""
  if hasattr(play_mod, '_build_environment_config'):
    return play_mod._build_environment_config(env_module)  # pylint: disable=protected-access
  env_config = env_module.get_config()
  with config_dict.ConfigDict(env_config).unlocked() as env_config:
    roles = env_config.default_player_roles
    env_config.lab2d_settings = env_module.build(roles, env_config)
  return env_config


# --------------------------------------------------------------------------
# Custom configs (e.g. the substrate tutorial), loaded from the meltingpot/
# source checkout and re-read from disk on every (re)load.
# --------------------------------------------------------------------------
MP_SOURCE = source.MP_SOURCE

TUTORIAL_CONFIGS = (
    'examples.tutorial.harvest.configs.environment.harvest',
    'examples.tutorial.harvest.configs.environment.harvest_finished',
)


class CustomConfig:
  """A game defined by a `get_config()` module, e.g. the tutorial's harvest.py.

  Configs like the tutorial's return a complete config (lab2d_settings
  included) and point `levelDirectory` at Lua files relative to the
  meltingpot source root, exactly as `play_harvest.py` expects.
  """

  def __init__(self, spec, play_module=None):
    self.spec = spec
    self.path = self._to_path(spec)
    self.play_path = self._to_path(play_module) if play_module else None
    self._play_mod = None

  _to_path = staticmethod(source.to_path)
  _exec = staticmethod(source.load_module)

  def load_config(self):
    """Re-reads the config file from disk; returns (module, config).

    Two config styles are supported:
      * tutorial style: get_config() returns lab2d_settings directly;
      * substrate style (like Melting Pot's own configs): get_config() plus
        build(roles, config), called here with the default player roles.
    """
    if not self.path.is_file():
      raise FileNotFoundError(f'config file not found: {self.path}')
    mod = self._exec(self.path, f'mpviz_custom_{self.path.stem}')
    config = mod.get_config()
    if 'lab2d_settings' not in config:
      if not callable(getattr(mod, 'build', None)):
        raise ValueError(
            f'{self.path.name}: get_config() has no lab2d_settings and the '
            'module has no build(roles, config) function to make them.')
      if 'default_player_roles' not in config:
        raise ValueError(
            f'{self.path.name}: has build() but get_config() sets no '
            'default_player_roles, so the number of players is unknown.')
      roles = tuple(config.default_player_roles)
      with config_dict.ConfigDict(config).unlocked() as config:
        config.lab2d_settings = mod.build(roles=roles, config=config)
    return mod, config

  def play_mod(self):
    """Action map holder: --play-module if given, else WASD + Q/E."""
    if self._play_mod is None:
      if self.play_path:
        self._play_mod = self._exec(self.play_path,
                                    f'mpviz_play_{self.play_path.stem}')
      else:
        # Same map as examples/tutorial/harvest/play_harvest.py.
        self._play_mod = types.SimpleNamespace(
            __name__='default_controls', FRAMES_PER_SECOND=8,
            _ACTION_MAP={
                'move': level_playing_utils.get_direction_pressed,
                'turn': level_playing_utils.get_turn_pressed,
            })
    return self._play_mod

  def build(self, env_config):
    # The builder resolves `levelDirectory` against the root of the
    # *installed* meltingpot package. Point it at the source checkout while
    # building, so the edited tutorial Lua files (and their `require`s) load.
    with source.source_levels():
      return builder.builder(**env_config)


def _png(frame, level=1):
  ok, buf = cv2.imencode('.png', cv2.cvtColor(np.asarray(frame),
                                              cv2.COLOR_RGB2BGR),
                         [cv2.IMWRITE_PNG_COMPRESSION, level])
  assert ok
  return base64.b64encode(buf.tobytes()).decode('ascii')


def _text(value):
  """WORLD.TEXT (a string/bytes scalar or array) -> str."""
  if value is None:
    return ''
  if isinstance(value, np.ndarray):
    if value.ndim == 0:
      value = value.item()
    elif value.dtype.kind in 'US':
      value = '\n'.join(_text(v) for v in value.ravel())
    else:
      value = bytes(value.astype(np.uint8).ravel())
  if isinstance(value, bytes):
    value = value.decode('utf-8', 'replace')
  return str(value)


# --------------------------------------------------------------------------
# Game session
# --------------------------------------------------------------------------
class Game:
  """One substrate instance driven by browser key state."""

  def __init__(self, registry, record_dir, player_views=True):
    # registry: name -> (play_mod, env_module) for built-in games, or
    #           name -> CustomConfig for configs loaded from files.
    self.registry = registry
    self.record_dir = record_dir
    self.player_views = player_views
    self.lock = threading.RLock()
    self.env = None
    self.level = None
    self.error = None
    self.timestep = None
    self.prefixes = []
    self.unbound = []
    self.rec = None
    self.saving = {}  # replay name -> status

  # -- lifecycle -----------------------------------------------------------
  def load(self, level, save=True):
    """(Re)builds `level` from disk. Errors are kept, not raised."""
    with self.lock:
      if level not in self.registry:
        raise KeyError(f'unknown substrate {level!r}')
      self._close_env(save=save)
      self.rec = None
      self.level = level
      self.error = None
      self.timestep = None
      self.prefixes = []
      entry = self.registry[level]
      try:
        if isinstance(entry, CustomConfig):
          self.play_mod = entry.play_mod()
          self.about_mod, self.env_config = entry.load_config()
          self.env = entry.build(self.env_config)
        else:
          self.play_mod, self.about_mod = entry
          self.env_config = build_env_config(*entry)
          self.env = builder.builder(**self.env_config)
        self.fps = int(getattr(self.play_mod, 'FRAMES_PER_SECOND', 8))
        # Actions the game has but no key is bound to always send 0.
        action_map = dict(self.play_mod._ACTION_MAP)  # pylint: disable=protected-access
        names = {k.split('.', 1)[1] for k in self.env.action_spec()}
        self.unbound = sorted(names - set(action_map))
        for name in self.unbound:
          action_map[name] = lambda: 0
        self.reader = level_playing_utils.ActionReader(self.env, action_map)
        self.reset()
      except Exception:  # pylint: disable=broad-except
        self._fail()
      return self.info()

  def _fail(self):
    # dmlab2d colors its Lua error lines with ANSI codes; strip them for HTML.
    self.error = re.sub(r'\x1b\[[0-9;]*m', '', traceback.format_exc())
    print(f'[mp-play] error loading {self.level}:\n{self.error}', flush=True)
    self.rec = None
    if self.env is not None:
      try:
        self.env.close()
      except Exception:  # pylint: disable=broad-except
        pass
      self.env = None

  def reset(self):
    with self.lock:
      if self.env is None:
        return self.state(0)
      self._save_recording()
      try:
        self.timestep = self.env.reset()  # Lua runs here: errors surface now.
      except Exception:  # pylint: disable=broad-except
        self._fail()
        return self.state(0)
      self.prefixes = recording._player_prefixes(self.timestep.observation)  # pylint: disable=protected-access
      self.scores = collections.defaultdict(float)
      self.steps = 0
      self.done = False
      self.end_reason = None
      self.rec = recording.EpisodeRecorder(
          substrate=self.level.split('.')[-1], fps=self.fps, source='human',
          record_player_views=self.player_views)
      self.rec.add_lab2d_timestep(self.timestep, active=-1,
                                  events=self._events())
      return self.state(0)

  def _events(self):
    try:
      return self.env.events()
    except Exception:  # pylint: disable=broad-except
      return None

  def _clamp(self, player):
    return max(0, min(len(self.prefixes) - 1, int(player)))

  def step(self, keys, player):
    with self.lock:
      if self.env is None or self.done:
        return self.state(player)
      player = self._clamp(player)
      prefix = self.prefixes[player] if self.prefixes else ''
      KEYS.set(keys)
      try:
        actions = self.reader.step(prefix)
      finally:
        KEYS.set(())
      try:
        self.timestep = self.env.step(actions)
      except Exception:  # pylint: disable=broad-except  (e.g. a Lua error)
        self._fail()
        return self.state(player)
      self.steps += 1
      for p, r in level_playing_utils._get_rewards(self.timestep).items():  # pylint: disable=protected-access
        self.scores[p] += float(r)
      if self.rec is not None:
        self.rec.add_lab2d_timestep(self.timestep, actions,
                                    active=player if self.prefixes else -1,
                                    events=self._events())
      if self.timestep.step_type == dm_env.StepType.LAST:
        self.done = True
        self.end_reason = self._why_ended()
        self._save_recording()
      return self.state(player)

  def _why_ended(self):
    try:
      limit = int(self.env_config.lab2d_settings['maxEpisodeLengthFrames'])
    except Exception:  # pylint: disable=broad-except
      limit = None
    if limit and self.steps >= limit:
      return (f'Time limit reached: {self.steps} steps '
              f'(maxEpisodeLengthFrames = {limit}).')
    return (f'The game ended the episode at step {self.steps}'
            + (f', before its {limit}-step limit' if limit else '')
            + ' (e.g. simulation:endEpisode() or a win condition).')

  def view(self, player):
    """Current frame from `player`'s perspective, without stepping."""
    with self.lock:
      return self.state(player)

  def state(self, player):
    base = dict(level=self.level, step=getattr(self, 'steps', 0),
                done=getattr(self, 'done', False), error=self.error,
                end_reason=getattr(self, 'end_reason', None),
                saving=self.saving, scores=[], player=None, world=None,
                text='')
    if self.timestep is None:
      return base
    obs = self.timestep.observation
    world = obs.get('WORLD.RGB')
    view = world
    if self.prefixes:
      view = obs.get(f'{self.prefixes[self._clamp(player)]}.RGB', world)
    base.update(
        player=_png(view) if view is not None else None,
        world=_png(world) if world is not None else None,
        scores=[self.scores[p] for p in self.prefixes],
        text=_text(obs.get('WORLD.TEXT')))
    return base

  # -- recording ------------------------------------------------------------
  def _save_recording(self):
    rec = getattr(self, 'rec', None)
    if rec is None or len(rec) <= 1:
      return None
    self.rec = None
    stem = recording.default_stem(self.record_dir, rec.substrate, 'human')
    name = os.path.basename(stem) + '.html'
    self.saving[name] = 'saving'

    def work():
      try:
        npz = rec.save(stem)
        data, meta = replay_html.load(npz)
        with open(stem + '.html', 'w') as f:
          f.write(replay_html.build_html(data, meta))
        self.saving[name] = 'ready'
        print(f'[mp-play] saved {npz} and {stem}.html', flush=True)
      except Exception:  # pylint: disable=broad-except
        self.saving[name] = 'failed'
        traceback.print_exc()

    t = threading.Thread(target=work, daemon=False)
    t.start()
    return name

  def save_now(self):
    with self.lock:
      name = self._save_recording()
      if not getattr(self, 'done', False):
        self.end_reason = f'You ended the episode at step {self.steps}.'
      self.done = True
      return name

  def _close_env(self, save):
    if save:
      self._save_recording()
    if self.env is not None:
      try:
        self.env.close()
      except Exception:  # pylint: disable=broad-except
        pass
      self.env = None

  def info(self):
    entry = self.registry[self.level]
    custom = isinstance(entry, CustomConfig)
    controls, about = [], ''
    if self.error is None:
      try:
        controls = describe_controls(self.play_mod)
        controls += [dict(action=a, label=_ACTION_LABELS.get(a, _humanize(a)),
                          keys=[dict(keys='—',
                                     effect='no key bound (always 0)')])
                     for a in self.unbound]
        about = describe_substrate(self.about_mod)
      except Exception:  # pylint: disable=broad-except
        traceback.print_exc()
    return dict(level=self.level, levels=list(self.registry),
                fps=getattr(self, 'fps', 8), players=len(self.prefixes),
                controls=controls, about=about, error=self.error,
                custom=custom,
                source=str(entry.path) if custom else None)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
  game: Game = None
  token: str = ''
  last_activity = time.time()

  def log_message(self, *args):  # quiet
    pass

  def _authorized(self, query):
    tok = query.get('token', [''])[0] or self.headers.get('X-Token', '')
    return secrets.compare_digest(tok, self.token)

  def _send(self, code, body, ctype='application/json'):
    if isinstance(body, (dict, list)):
      body = json.dumps(body)
    if isinstance(body, str):
      body = body.encode()
    self.send_response(code)
    self.send_header('Content-Type', ctype)
    self.send_header('Content-Length', str(len(body)))
    self.send_header('Cache-Control', 'no-store')
    self.end_headers()
    self.wfile.write(body)

  def _route(self, method):
    url = urllib.parse.urlparse(self.path)
    query = urllib.parse.parse_qs(url.query)
    if not self._authorized(query):
      return self._send(HTTPStatus.FORBIDDEN, {'error': 'bad token'})
    Handler.last_activity = time.time()
    body = {}
    if method == 'POST':
      n = int(self.headers.get('Content-Length', 0) or 0)
      body = json.loads(self.rfile.read(n) or b'{}')
    g = self.game
    try:
      if method == 'GET' and url.path == '/':
        return self._send(200, PAGE, 'text/html; charset=utf-8')
      if method == 'GET' and url.path == '/info':
        return self._send(200, g.info())
      if method == 'POST' and url.path == '/load':
        return self._send(200, g.load(body['level']))
      if method == 'POST' and url.path == '/reload':
        # Re-read config + Lua from disk; don't save the discarded episode.
        return self._send(200, g.load(g.level, save=False))
      if method == 'POST' and url.path == '/reset':
        return self._send(200, g.reset())
      if method == 'POST' and url.path == '/step':
        return self._send(200, g.step(body.get('keys', []),
                                      body.get('player', 0)))
      if method == 'POST' and url.path == '/view':
        return self._send(200, g.view(body.get('player', 0)))
      if method == 'POST' and url.path == '/save':
        return self._send(200, {'replay': g.save_now(), 'saving': g.saving})
      if method == 'GET' and url.path in ('/replays', '/replays/'):
        return self._send(200, _replay_index(g.record_dir, self.token),
                          'text/html; charset=utf-8')
      if method == 'GET' and url.path.startswith('/replays/'):
        name = os.path.basename(urllib.parse.unquote(url.path))
        path = os.path.join(g.record_dir, name)
        if name.endswith('.html') and os.path.isfile(path):
          with open(path, 'rb') as f:
            return self._send(200, f.read(), 'text/html; charset=utf-8')
      return self._send(404, {'error': 'not found'})
    except Exception as e:  # pylint: disable=broad-except
      traceback.print_exc()
      return self._send(500, {'error': f'{type(e).__name__}: {e}'})

  def do_GET(self):  # pylint: disable=invalid-name
    self._route('GET')

  def do_POST(self):  # pylint: disable=invalid-name
    self._route('POST')


def _replay_index(record_dir, token):
  files = sorted(glob.glob(os.path.join(record_dir, '*.html')),
                 key=os.path.getmtime, reverse=True)
  items = ''.join(
      f'<li><a href="/replays/{urllib.parse.quote(os.path.basename(p))}'
      f'?token={token}" target="_blank">{os.path.basename(p)}</a> '
      f'<small>({os.path.getsize(p) / 1e6:.1f} MB)</small></li>'
      for p in files)
  return (f'<!doctype html><meta charset=utf-8><title>Replays</title>'
          f'<body style="font:14px system-ui;padding:16px"><h1>Replays in '
          f'{record_dir}</h1><ul>{items or "<li>none yet</li>"}</ul>')


def _cluster_ip():
  host = socket.gethostname()
  try:
    return host, socket.gethostbyname(host)
  except OSError:
    return host, '127.0.0.1'


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawTextHelpFormatter)
  ap.add_argument('--substrate', default='commons_harvest__open')
  ap.add_argument('--port', type=int, default=0,
                  help='0 = pick a free port')
  ap.add_argument('--local-port', type=int, default=8765,
                  help='port to suggest for your end of the SSH tunnel')
  ap.add_argument('--record-dir', default='agents/recordings')
  ap.add_argument('--no-player-views', action='store_true',
                  help='do not record per-player views (smaller files)')
  ap.add_argument('--idle-minutes', type=float, default=30,
                  help='exit after this long with no browser activity')
  ap.add_argument('--login-host', default='klone.hyak.uw.edu')
  ap.add_argument('--list', action='store_true', help='list substrates')
  ap.add_argument('--tutorial', action='store_true',
                  help='load the substrate tutorial (examples/tutorial/'
                       'harvest): harvest.py, plus harvest_finished.py')
  ap.add_argument('--config', action='append', default=[],
                  help='a game config module with get_config(): dotted path\n'
                       'relative to meltingpot/ (e.g. examples.tutorial.\n'
                       'harvest.configs.environment.harvest) or a .py path.\n'
                       'Re-read from disk on Reload. Repeatable.')
  ap.add_argument('--play-module',
                  help='module whose _ACTION_MAP gives the controls for\n'
                       '--config games (default: WASD move, Q/E turn)')
  args = ap.parse_args()

  registry = {}
  for spec in (TUTORIAL_CONFIGS if args.tutorial else ()) + tuple(args.config):
    registry[spec] = CustomConfig(spec, args.play_module)
  first = next(iter(registry), None)
  registry.update(discover())
  if args.list:
    print('\n'.join(registry))
    return
  level = first if first and args.substrate == ap.get_default('substrate') \
      else args.substrate
  os.makedirs(args.record_dir, exist_ok=True)
  game = Game(registry, os.path.abspath(args.record_dir),
              player_views=not args.no_player_views)
  game.load(level)
  if game.error:
    print('[mp-play] (the page will show this error; fix the file and click'
          ' Reload)', flush=True)

  host, ip = _cluster_ip()
  Handler.game = game
  Handler.token = secrets.token_urlsafe(18)
  # Listen only on the node's cluster-internal address (never 0.0.0.0).
  server = ThreadingHTTPServer((ip, args.port), Handler)
  port = server.server_address[1]
  user = os.environ.get('USER', '<you>')
  lp = args.local_port
  url = f'http://localhost:{lp}/?token={Handler.token}'
  print(f"""
==========================================================================
 Melting Pot play server: {level}  (on {host}:{port})

 1) On YOUR computer (Mac/Linux terminal, or Windows PowerShell), run:

      ssh -N -L {lp}:{host}:{port} {user}@{args.login_host}

    (VS Code Remote-SSH instead: Ports panel -> Forward a Port ->
     enter {host}:{port}, then use the local address it shows.)

 2) Open in your browser:

      {url}

 Recordings + replays are written to {game.record_dir}
 Stops after {args.idle_minutes:g} idle minutes, or press Ctrl-C.
==========================================================================
""", flush=True)

  def idle_watch():
    while True:
      time.sleep(30)
      if time.time() - Handler.last_activity > args.idle_minutes * 60:
        print('[mp-play] idle timeout, shutting down', flush=True)
        server.shutdown()
        return
  threading.Thread(target=idle_watch, daemon=True).start()

  # agents/bin launchers turn Ctrl-C into SIGUSR1 (srun passes it through);
  # scancel and the job time limit send SIGTERM. Handle both like Ctrl-C so
  # the current episode gets saved.
  def on_signal(signum, frame):
    del signum, frame
    raise KeyboardInterrupt
  signal.signal(signal.SIGUSR1, on_signal)
  signal.signal(signal.SIGTERM, on_signal)

  try:
    server.serve_forever(poll_interval=1.0)
  except KeyboardInterrupt:
    pass
  finally:
    with game.lock:
      game._close_env(save=True)  # pylint: disable=protected-access
    for t in threading.enumerate():
      if t is not threading.current_thread() and not t.daemon:
        t.join()
    server.server_close()
    print('[mp-play] stopped', flush=True)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Melting Pot play</title>
<style>
:root{--bg:#f6f6f4;--panel:#fff;--fg:#1d1d1b;--muted:#6b6b66;--line:#deded9;
--accent:#2f5fd0;--chip:#efefeb}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--panel:#1d1d1d;
--fg:#e9e9e6;--muted:#9a9a94;--line:#333;--accent:#7aa2ff;--chip:#2a2a2a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;
gap:12px;align-items:center;flex-wrap:wrap}
header h1{font-size:16px;margin:0 8px 0 0}
main{display:grid;grid-template-columns:auto minmax(260px,1fr);gap:16px;
padding:16px;max-width:1400px;margin:0 auto;align-items:start}
@media (max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;
padding:12px}
canvas{image-rendering:pixelated;image-rendering:crisp-edges;background:#000;
border-radius:6px;display:block}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}
button,select{font:inherit;color:var(--fg);background:var(--chip);
border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.muted{color:var(--muted)}
#status{font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
td{padding:2px 6px;border-bottom:1px solid var(--line);font-size:13px}
td.r{text-align:right} tr.me td{font-weight:600}
.how{margin:0;padding-left:18px;font-size:13px}
.how li{margin:2px 0}
table.ctl td{vertical-align:top}
table.ctl td.k{white-space:nowrap;font-weight:600}
kbd{font:12px ui-monospace,Menlo,monospace;background:var(--chip);
border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;
padding:0 5px}
summary{cursor:pointer;list-style:none}
summary h2{display:inline}
summary::before{content:"\u25b8 ";color:var(--muted)}
details[open] summary::before{content:"\u25be "}
#about{font-size:13px;white-space:pre-line;color:var(--fg)}
.ended{border:1px solid var(--accent);border-left-width:4px;border-radius:6px;
padding:10px 12px;margin:0 0 10px;background:var(--panel)}
.err{background:#fde8e8;color:#8a1111;border:1px solid #f3b4b4;
border-radius:6px;padding:8px;margin:0 0 10px;max-height:320px;overflow:auto;
font-size:12px}
@media (prefers-color-scheme:dark){.err{background:#3a1616;color:#ffb3b3;
border-color:#6b2a2a}}
.wtext{color:var(--fg);background:var(--chip);border-radius:6px;padding:6px}
code{font:12px ui-monospace,Menlo,monospace}
pre{white-space:pre-wrap;font:12px/1.4 ui-monospace,Menlo,monospace;
color:var(--muted);margin:0}
h2{font-size:13px;color:var(--muted);margin:14px 0 6px;font-weight:500;
text-transform:uppercase;letter-spacing:.04em}
#focus{position:fixed;inset:auto 16px 16px auto;background:var(--accent);
color:#fff;padding:6px 10px;border-radius:6px;display:none}
a{color:var(--accent)}
</style></head><body tabindex="0">
<header>
 <h1>Melting Pot</h1>
 <select id="level" title="Substrate"></select>
 <button id="reload" title="Re-read the config and Lua files from disk and restart the episode">&#8635; Reload files</button>
 <span class="muted">playing as</span>
 <select id="player" title="Tab cycles players"></select>
 <label class="muted"><input type="checkbox" id="turn"> turn-based</label>
 <span id="status" class="muted"></span>
</header>
<main>
 <section class="card">
  <div id="source" class="muted" style="margin-bottom:8px" hidden></div>
  <pre id="error" class="err" hidden></pre>
  <div id="ended" class="ended" hidden>
   <div><b>Episode over.</b> <span id="end-reason"></span>
    <span id="end-scores" class="muted"></span></div>
   <div class="row" style="margin-top:8px">
    <button id="restart" class="primary">Restart (Enter)</button>
    <button id="restart-reload">Reload files &amp; restart</button>
    <span id="end-replay" class="muted"></span>
   </div>
  </div>
  <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start">
   <div id="pvbox"><div class="muted" style="margin-bottom:4px">Your view</div>
    <canvas id="pv"></canvas></div>
   <div><div class="muted" style="margin-bottom:4px">World</div>
    <canvas id="wv"></canvas></div>
  </div>
  <div id="wtextbox" hidden><div class="muted" style="margin:8px 0 4px">
   WORLD.TEXT</div><pre id="wtext" class="wtext"></pre></div>
  <div class="row">
   <button id="pause" class="primary">Pause</button>
   <button id="reset">New episode</button>
   <button id="save">End &amp; save replay</button>
   <a id="replays" target="_blank">Replays</a>
   <span id="saved" class="muted"></span>
  </div>
 </section>
 <aside class="card">
  <h2 style="margin-top:0">Scores</h2>
  <table><tbody id="scores"></tbody></table>
  <h2>How to play</h2>
  <ol class="how">
   <li>Click the game once so it receives your key presses.</li>
   <li>You control the highlighted player. Your view is
    egocentric: you face up, so <b>W</b> always moves you up the screen.</li>
   <li><b>Tab</b> (or the dropdown) switches which player you control. Every
    other player stands still (no-op).</li>
   <li>Keys are held, not typed: hold a key to keep doing it. In
    <b>turn-based</b> mode, each key press advances exactly one step.</li>
  </ol>
  <h2>Controls for <span id="ctl-level"></span></h2>
  <table class="ctl"><tbody id="controls"></tbody></table>
  <details open id="about-box"><summary><h2>About this substrate</h2></summary>
   <div id="about"></div></details>
 </aside>
</main>
<div id="focus">Click the page to give it keyboard focus</div>
<script>
const TOKEN = new URLSearchParams(location.search).get('token') || '';
const $ = id => document.getElementById(id);
const CODE = {Space:'SPACE', ArrowUp:'UP', ArrowDown:'DOWN', ArrowLeft:'LEFT',
  ArrowRight:'RIGHT', ShiftLeft:'LSHIFT', ShiftRight:'RSHIFT',
  ControlLeft:'LCTRL', ControlRight:'RCTRL', Delete:'DELETE',
  PageDown:'PAGEDOWN', PageUp:'PAGEUP', Enter:'RETURN', Backspace:'DELETE'};
function keyName(e) {
  if (CODE[e.code]) return CODE[e.code];
  let m = /^Key([A-Z])$/.exec(e.code); if (m) return m[1].toLowerCase();
  m = /^(?:Digit|Numpad)([0-9])$/.exec(e.code); if (m) return m[1];
  return null;
}
let info = null, player = 0, paused = false, inflight = false, done = false;
const held = new Set(), tapped = new Set();
const pv = $('pv'), wv = $('wv');

async function api(path, body) {
  const r = await fetch(path + '?token=' + encodeURIComponent(TOKEN), {
    method: body === undefined ? 'GET' : 'POST',
    headers: {'Content-Type': 'application/json'},
    body: body === undefined ? undefined : JSON.stringify(body)});
  if (!r.ok) throw new Error((await r.json()).error || r.status);
  return r.json();
}
function paint(canvas, b64, maxW, maxH) {
  const im = new Image();
  im.onload = () => {
    const s = Math.max(1, Math.floor(Math.min(maxW / im.width,
                                              maxH / im.height)));
    if (canvas.width !== im.width * s) {
      canvas.width = im.width * s; canvas.height = im.height * s; }
    const c = canvas.getContext('2d'); c.imageSmoothingEnabled = false;
    c.drawImage(im, 0, 0, canvas.width, canvas.height);
  };
  im.src = 'data:image/png;base64,' + b64;
}
function render(s) {
  const H = Math.max(240, window.innerHeight - 220);
  if (s.player) paint(pv, s.player, 560, H);
  if (s.world) paint(wv, s.world, Math.max(240, window.innerWidth - 1000), H);
  showError(s.error);
  $('wtextbox').hidden = !s.text;
  $('wtext').textContent = s.text || '';
  done = s.done || !!s.error;
  const over = s.done && !s.error;
  $('ended').hidden = !over;
  if (over) {
    $('end-reason').textContent = s.end_reason || '';
    $('end-scores').textContent = s.scores.length ? 'Final scores: ' +
        s.scores.map((v, i) => 'P' + (i + 1) + ' ' + (+v.toFixed(2)))
            .join(', ') : '';
    $('restart-reload').hidden = !(info && info.custom);
  }
  $('status').textContent = 'step ' + s.step + (s.done ? ' · episode over'
      : paused ? ' · paused' : '');
  $('scores').innerHTML = s.scores.map((v, i) =>
      '<tr class="' + (i === player ? 'me' : '') + '"><td>Player ' + (i + 1) +
      (i === player ? ' (you)' : '') + '</td><td class="r">' +
      (+v.toFixed(2)) + '</td></tr>').join('');
  const names = Object.keys(s.saving || {});
  if (names.length) {
    const last = names[names.length - 1], st = s.saving[last];
    $('saved').innerHTML = st === 'ready'
      ? 'Saved: <a target="_blank" href="/replays/' + encodeURIComponent(last) +
        '?token=' + encodeURIComponent(TOKEN) + '">' + last + '</a>'
      : 'Replay ' + st + '…';
    $('end-replay').innerHTML = over ? (st === 'ready' ? '<a target="_blank" ' +
        'href="/replays/' + encodeURIComponent(last) + '?token=' +
        encodeURIComponent(TOKEN) + '">Open the replay</a>' :
        'Replay ' + st + '…') : '';
    // The replay is written in the background; refresh until it's ready.
    if (over && st === 'saving') setTimeout(showView, 1000);
  }
}
function showError(err) {
  $('error').hidden = !err;
  $('error').textContent = err ? 'The game hit an error. ' +
      (info && info.custom ? 'Fix the file, then click Reload files.'
                           : 'Click Rebuild to start it again.') +
      ' The server is still running.\n\n' + err : '';
}
let viewSeq = 0;
async function showView() {
  // Every player's view is in the current timestep, so no step is needed.
  const seq = ++viewSeq;
  try { const s = await api('/view', {player});
        if (seq === viewSeq) render(s); }
  catch (e) { $('status').textContent = 'error: ' + e.message; }
}
function switchPlayer(p) {
  if (!info || !info.players) return;
  player = p; $('player').value = p; showView();
}
async function step() {
  if (inflight || done) return;
  inflight = true;
  const keys = [...new Set([...held, ...tapped])]; tapped.clear();
  const p = player;
  try {
    const s = await api('/step', {keys, player: p});
    // If the player was switched mid-request, re-fetch the right view.
    p === player ? render(s) : showView();
  }
  catch (e) { $('status').textContent = 'error: ' + e.message; }
  finally { inflight = false; }
}
let timer = null;
function schedule() {
  clearInterval(timer);
  if (!paused && !$('turn').checked && info)
    timer = setInterval(step, 1000 / info.fps);
}
function setPlayers(n) {
  $('player').innerHTML = Array.from({length: n}, (_, i) =>
      '<option value="' + i + '">Player ' + (i + 1) + '</option>').join('');
  $('player').value = player = Math.max(0, Math.min(player, n - 1));
}
async function loadInfo(i) {
  info = i;
  $('level').innerHTML = i.levels.map(l => '<option' + (l === i.level ?
      ' selected' : '') + '>' + l + '</option>').join('');
  $('ctl-level').textContent = i.level;
  const esc = t => String(t).replace(/[&<>]/g,
      c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;'}[c]));
  $('controls').innerHTML = i.controls.map(c =>
      '<tr><td>' + esc(c.label) + '</td><td class="k">' +
      c.keys.map(k => '<kbd>' + esc(k.keys) + '</kbd>' +
                 (k.effect ? ' <span class="muted">' + esc(k.effect) +
                  '</span>' : '')).join('<br>') + '</td></tr>').join('');
  $('about').textContent = i.about || '(no description)';
  $('source').hidden = !i.source;
  $('source').innerHTML = i.source ? 'Loaded from <code>' + esc(i.source) +
      '</code> &middot; edit it (and its Lua level), then click ' +
      '<b>Reload files</b>.' : '';
  $('reload').innerHTML = i.custom ? '&#8635; Reload files' : '&#8635; Rebuild';
  $('reload').title = i.custom
      ? 'Re-read the config and Lua files from disk and restart the episode'
      : 'Rebuild the game from scratch (recovers from errors)';
  $('pvbox').hidden = !i.players;
  setPlayers(i.players);
  showError(i.error);
  if (!i.error) render(await api('/reset', {}));
  else { done = true; $('status').textContent = 'load error'; }
  schedule();
}
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT') return;
  if (e.key === 'Enter' && !$('ended').hidden) {
    e.preventDefault(); restart(); return;
  }
  if (e.code === 'Tab') {
    e.preventDefault();
    if (info && info.players) switchPlayer((player + 1) % info.players);
    return;
  }
  const k = keyName(e); if (!k) return;
  e.preventDefault();
  if (!held.has(k)) tapped.add(k);
  held.add(k);
  if ($('turn').checked && !e.repeat) step();
});
window.addEventListener('keyup', e => { const k = keyName(e);
                                        if (k) held.delete(k); });
window.addEventListener('blur', () => { held.clear();
                                        $('focus').style.display = 'block'; });
window.addEventListener('focus', () => $('focus').style.display = 'none');
$('pause').onclick = () => { paused = !paused;
  $('pause').textContent = paused ? 'Resume' : 'Pause'; schedule(); };
$('turn').onchange = () => { schedule(); document.body.focus(); };
$('player').onchange = e => { switchPlayer(+e.target.value);
                              document.body.focus(); };
async function restart() {
  try { render(await api('/reset', {})); }
  catch (e) { $('status').textContent = 'error: ' + e.message; }
  document.body.focus();
}
$('reset').onclick = restart;
$('restart').onclick = restart;
$('restart-reload').onclick = () => $('reload').onclick();
$('save').onclick = async () => { const r = await api('/save', {});
  done = true; render(await api('/step', {keys: [], player})); };
$('level').onchange = async e => {
  $('status').textContent = 'loading ' + e.target.value + '…';
  clearInterval(timer);
  await loadInfo(await api('/load', {level: e.target.value}));
  document.body.focus();
};
$('reload').onclick = async () => {
  $('status').textContent = 'reloading…';
  clearInterval(timer);
  try { await loadInfo(await api('/reload', {})); }
  catch (e) { $('status').textContent = 'error: ' + e.message; }
  document.body.focus();
};
$('replays').href = '/replays/?token=' + encodeURIComponent(TOKEN);
api('/info').then(loadInfo).catch(e =>
    $('status').textContent = 'error: ' + e.message);
document.body.focus();
</script></body></html>
"""

if __name__ == '__main__':
  sys.exit(main())
