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
