"""Smoke test for mpviz (play server, recorder, replay). Run inside a job:

  agents/bin/mp-python agents/tests/smoke_mpviz.py   (or see smoke.sbatch)
"""
import base64
import glob
import json
import os
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

from mpviz import play_server as ps
from mpviz import recording, replay_html

import numpy as np

failures = []
def check(cond, msg):
  print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
  if not cond:
    failures.append(msg)

tmp = tempfile.mkdtemp(prefix='mpviz-smoke-', dir=os.environ.get('TMPDIR'))
registry = ps.discover()
print(f'{len(registry)} playable substrates')

# --- 1. play server over HTTP ---------------------------------------------
game = ps.Game(registry, tmp, player_views=True)
game.load('commons_harvest__open')
ps.Handler.game, ps.Handler.token = game, 'secret-token'
host, ip = ps._cluster_ip()
srv = ps.ThreadingHTTPServer((ip, 0), ps.Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f'http://{ip}:{srv.server_address[1]}'
check(ip != '0.0.0.0', f'listening on cluster address {ip}, not 0.0.0.0')

def call(path, body=None, token='secret-token'):
  req = urllib.request.Request(
      f'{base}{path}?token={token}',
      data=None if body is None else json.dumps(body).encode(),
      headers={'Content-Type': 'application/json'})
  with urllib.request.urlopen(req) as r:
    data = r.read()
  return json.loads(data) if path != '/' else data.decode()

try:
  call('/info', token='wrong'); check(False, 'bad token rejected')
except urllib.error.HTTPError as e:
  check(e.code == 403, f'bad token rejected (HTTP {e.code})')
check('<canvas' in call('/'), 'play page served')
info = call('/info')
check(info['players'] == 7 and info['controls'][0]['action'] == 'move' and
      'Apples' in info['about'],
      f"info: {info['players']} players, controls + about present")
cu = {c['action']: c['keys'][0]['keys'] for c in
      ps.describe_controls(registry['clean_up'][0])}
check(cu.get('fireZap') == '1' and cu.get('fireClean') == '2',
      f'clean_up controls from action map: {cu}')
fm = {c['action']: c['keys'] for c in
      ps.describe_controls(registry['fruit_market__concentric_rivers'][0])}
check([k['effect'] for k in fm['shove']] == ['push', 'pull'],
      f"fruit_market shove keys: {fm['shove']}")
s0 = call('/reset', {})
def img(b64):
  import cv2
  return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), 1)
p0, w0 = img(s0['player']), img(s0['world'])
s = None
for _ in range(30):
  s = call('/step', {'keys': ['w'], 'player': 0})
check(not np.array_equal(img(s['world']), w0), 'world view changes when moving')
check(not np.array_equal(img(s['player']), p0), 'player-1 view changes when moving')
for _ in range(10):
  s = call('/step', {'keys': ['e'], 'player': 0})  # turn
for _ in range(10):
  s = call('/step', {'keys': ['SPACE'], 'player': 0})  # zap
check(s['step'] == 50, f"50 steps taken (step={s['step']})")
v0 = call('/view', {'player': 0})
v3 = call('/view', {'player': 3})
check(v0['step'] == v3['step'] == 50, 'view switch does not step the env')
check(not np.array_equal(img(v0['player']), img(v3['player'])) and
      np.array_equal(img(v0['world']), img(v3['world'])),
      'view switch returns another player\'s view of the same frame')
acts = np.stack([a if a is not None else np.zeros((7, 3), int)
                 for a in game.rec._actions])
names = game.rec.action_names
check(acts[1:31, 0, names.index('move')].all() and
      not acts[1:31, 1:, :].any(),
      f'P1 moved forward, others no-op (action names {names})')
check(acts[41:51, 0, names.index('fireZap')].all(), 'P1 fired zapper')
saved = call('/save', {})
name = saved['replay']
for _ in range(120):
  if game.saving.get(name) != 'saving':
    break
  time.sleep(1)
check(game.saving.get(name) == 'ready', f'replay saved: {name}')
check('/replays/' + name in urllib.request.urlopen(
    f'{base}/replays/?token=secret-token').read().decode(),
    'replay listed in /replays/')
npz = glob.glob(os.path.join(tmp, '*.npz'))
d = np.load(npz[0])
check(d['world'].shape[0] == 51 and d['players'].shape[:2] == (51, 7),
      f"npz shapes world={d['world'].shape} players={d['players'].shape}")
check(d['active'][1] == 0 and d['active'][0] == -1, 'active player recorded')
srv.shutdown()
game._close_env(save=False)

# --- 2. every registered substrate builds and steps -----------------------
for level in registry:
  try:
    g = ps.Game(registry, tmp, player_views=False)
    g.load(level)
    for k in (['w'], ['SPACE'], ['1'], [], ['e']):
      st = g.step(k, 0)
    g.rec = None  # don't save
    g._close_env(save=False)
    check(st['step'] == 5, f'substrate {level}')
  except Exception as e:  # pylint: disable=broad-except
    check(False, f'substrate {level}: {type(e).__name__}: {e}')

# --- 2b. substrate tutorial: custom configs, reload, errors ---------------
tut = {spec: ps.CustomConfig(spec) for spec in ps.TUTORIAL_CONFIGS}
g = ps.Game(dict(tut, **registry), tmp, player_views=True)
# The tutorial's own harvest.py / levels/harvest are the user's work in
# progress (possibly mid-edit), so tests never load them. The 0-player
# starting point is tested with a pristine config on the untouched
# harvest_finished level.
empty_cfg = os.path.join(tmp, 'empty_harvest.py')
with open(empty_cfg, 'w') as f:
  f.write('''from ml_collections import config_dict

def get_config():
  config = config_dict.ConfigDict()
  config.individual_observation_names = []
  config.global_observation_names = ["WORLD.RGB"]
  config.lab2d_settings = {
      "levelName": "harvest_finished",
      "levelDirectory": "examples/tutorial/harvest/levels",
      "maxEpisodeLengthFrames": 100,
      "numPlayers": 0,
      "spriteSize": 8,
      "simulation": {"map": " ", "prefabs": {}, "charPrefabMap": {}},
  }
  return config
''')
g0 = ps.Game({empty_cfg: ps.CustomConfig(empty_cfg)}, tmp)
inf = g0.load(empty_cfg)
st = g0.step(['w'], 0)
check(inf['error'] is None and inf['players'] == 0 and st['world'] and
      st['step'] == 1,
      f"tutorial starting point (empty, 0 players) loads: "
      f"players={inf['players']} error={(inf['error'] or '')[-200:]!r}")
g0._close_env(save=False)
inf = g.load(ps.TUTORIAL_CONFIGS[1])  # harvest_finished
w0 = g.state(0)['player']
moved = False
for key in ['w', 'd', 's', 'a'] * 3:  # random spawn may face a wall
  st = g.step([key], 0)
  moved = moved or st['player'] != w0
check(inf['error'] is None and inf['players'] == 5 and
      [c['action'] for c in inf['controls']] == ['move', 'turn'] and moved,
      f"tutorial harvest_finished plays: players={inf['players']}, "
      f"controls={[c['action'] for c in inf['controls']]}")
g._close_env(save=False)

# Edit -> Reload loop on a scratch copy of the finished config.
src = (ps.MP_SOURCE / 'examples/tutorial/harvest/configs/environment/'
       'harvest_finished.py').read_text()
cfg = os.path.join(tmp, 'my_harvest.py')
broken = re.sub(r'("levelName":\s*)"harvest_finished"', r'\g<1>"no_such_level"',
                src)
check(broken != src, 'scratch config can be broken for the error test')
with open(cfg, 'w') as f:
  f.write(broken)
g = ps.Game({cfg: ps.CustomConfig(cfg)}, tmp)
inf = g.load(cfg)
st = g.step(['w'], 0)
check(inf['error'] is not None and st['error'] is not None and
      st['player'] is None,
      'broken config reports an error instead of crashing')
lua_bad = src.replace('"component": "DensityRegrow"',
                      '"component": "NoSuchComponent"')
check(lua_bad != src, 'scratch config can reference a missing Lua component')
with open(cfg, 'w') as f:
  f.write(lua_bad)
inf = g.load(cfg, save=False)
err = inf['error'] or ''
check('NoSuchComponent' in err and '\x1b' not in err,
      f'Lua error reaches the page without terminal color codes: '
      f'{err.strip().splitlines()[-1][:160] if err else None!r}')
with open(cfg, 'w') as f:
  edited = re.sub(r'("maxEpisodeLengthFrames":\s*)1000', r'\g<1>3', src)
  check(edited != src, 'scratch config edit applied')
  f.write(edited)
inf = g.load(cfg, save=False)  # what the Reload button does
for _ in range(3):
  st = g.step([], 0)
check(inf['error'] is None and st['done'],
      f"after fixing the file, Reload picks up the edit (3-step episode "
      f"done={st['done']}, step={st['step']})")
check('Time limit reached' in (st['end_reason'] or ''),
      f"end reason for time limit: {st['end_reason']!r}")
g.steps = 2  # as if the game had ended early, e.g. simulation:endEpisode()
check('game ended the episode at step 2' in g._why_ended(),
      f'end reason for early end: {g._why_ended()!r}')
st = g.reset()
check(not st['done'] and st['end_reason'] is None and st['step'] == 0,
      'restart clears the episode-over state')
g.step([], 0)
g.save_now()
st = g.state(0)
check(st['done'] and 'You ended the episode at step 1' in st['end_reason'],
      f"end reason for End & save: {st['end_reason']!r}")
g._close_env(save=False)

# --- 2c. substrate-style custom config (get_config + build) -------------
sub_cfg = str(ps.MP_SOURCE / 'meltingpot/configs/substrates/clean_up.py')
g = ps.Game({sub_cfg: ps.CustomConfig(sub_cfg)}, tmp)
inf = g.load(sub_cfg)
st = g.step(['w'], 0)
check(inf['error'] is None and inf['players'] == 7 and st['step'] == 1,
      f"--config with a get_config()+build() substrate loads: "
      f"players={inf['players']} error={(inf['error'] or '')[-200:]!r}")
g._close_env(save=False)

# --- 3. experiment recording -> HTML replay --------------------------------
from meltingpot import substrate
env = substrate.build('commons_harvest__open',
                      roles=substrate.get_config('commons_harvest__open')
                      .default_player_roles)
renv = recording.wrap(env, tmp, substrate='commons_harvest__open', tag='exp',
                      max_episodes=1, record_player_views=True)
ts = renv.reset()
for t in range(100):
  ts = renv.step([t % 8] * 7)
renv.close()
check(len(renv.saved) == 1, f'experiment episode recorded: {renv.saved}')
data, meta = replay_html.load(renv.saved[0])
page = replay_html.build_html(data, meta)
check(data['world'].shape[0] == 101 and data['actions'][5, 0, 0] == 4,
      'experiment frames/actions aligned')
# ':' never occurs in base64, so any URL here is a real external reference.
check(not re.search(r'https?://', page), 'replay HTML is self-contained')
print(f'replay html {len(page) / 1e6:.2f} MB for 101 frames x (1 + 7 views)')

import shutil
shutil.rmtree(tmp, ignore_errors=True)
print('\nFAILURES:' if failures else '\nALL PASSED', *failures, sep='\n  ')
sys.exit(1 if failures else 0)
