"""Checks one substrate config from meltingpot/.../configs/substrates/communication.

  python agents/tests/communication/check_substrate.py <config.py>

Prints one PASS / FAIL / SKIP line per check and exits 1 if anything failed.
Normally run for every config by run_all.py (see agents/bin/mp-test-comms),
which runs each config in its own process so a crash in the game (e.g. a Lua
error inside a callback, which aborts the process) is reported, not fatal.

Checks, in order (later ones are skipped when what they need is missing):
  Level        the config loads and its Lua level runs full episodes with
               random actions (works for tutorial-style configs too)
  Substrate    the fields + build() Melting Pot needs, then the same checks
               as meltingpot/substrate_test.py (specs match what the
               environment actually produces), plus every action index
  Registration how the config is listed in configs/substrates/__init__.py
  Game         game-specific checks from checks_<name>.py next to this file
"""

import os
import pathlib
import re
import sys
import traceback

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # agents/ (for mpviz)
sys.path.insert(0, str(HERE))             # checks_<name>.py

import dm_env  # pylint: disable=g-import-not-at-top
from meltingpot import substrate  # pylint: disable=g-import-not-at-top
from meltingpot.utils.substrates import builder  # pylint: disable=g-import-not-at-top
from meltingpot.utils.substrates import specs  # pylint: disable=g-import-not-at-top
from ml_collections import config_dict  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=g-import-not-at-top

from mpviz import source  # pylint: disable=g-import-not-at-top

REQUIRED_FIELDS = ('action_set', 'action_spec', 'timestep_spec', 'valid_roles',
                   'default_player_roles', 'individual_observation_names',
                   'global_observation_names')


class Report:
  """Collects PASS / FAIL / SKIP results and prints them as they happen."""

  def __init__(self):
    self.failed = 0

  def _emit(self, status, name, detail=''):
    print(f'{status:<4}  {name}' + (f': {detail}' if detail else ''),
          flush=True)

  def ok(self, name, detail=''):
    self._emit('PASS', name, detail)

  def fail(self, name, detail=''):
    self.failed += 1
    self._emit('FAIL', name, detail)

  def skip(self, name, why):
    self._emit('SKIP', name, why)

  def check(self, cond, name, detail=''):
    (self.ok if cond else self.fail)(name, detail)
    return cond

  def section(self, title):
    print(f'\n-- {title}', flush=True)


def _last_line(exc_text):
  lines = [l for l in re.sub(r'\x1b\[[0-9;]*m', '', exc_text).splitlines()
           if l.strip()]
  # Prefer the most informative line: a Lua error message if present.
  for l in lines:
    if 'ERROR:' in l and ('attempt' in l or 'error' in l.lower()) \
        and 'stack' not in l:
      return l.strip()[:300]
  return lines[-1].strip()[:300] if lines else '?'


def raw_settings(module, config):
  """The lab2d settings: from build() if the config has one, else the
  tutorial-style config.lab2d_settings."""
  if hasattr(module, 'build') and 'default_player_roles' in config:
    return module.build(roles=tuple(config.default_player_roles),
                        config=config)
  return config.lab2d_settings


def build_raw_env(settings, seed=None):
  """A raw dmlab2d env (all observations, dict actions) from the checkout."""
  with source.source_levels():
    return builder.builder(lab2d_settings=settings, env_seed=seed)


def random_action(action_spec, rng):
  return {k: int(rng.integers(s.minimum, s.maximum + 1))
          for k, s in action_spec.items()}


# ---------------------------------------------------------------------------
def check_level(r, module, config):
  r.section('Level (config + Lua)')
  try:
    settings = raw_settings(module, config)
  except Exception:  # pylint: disable=broad-except
    r.fail('lab2d settings', _last_line(traceback.format_exc()))
    traceback.print_exc()
    return None
  try:
    env = build_raw_env(settings, seed=1)
    ts = env.reset()
  except Exception:  # pylint: disable=broad-except
    r.fail('level loads (Lua init, prefabs, map)',
           _last_line(traceback.format_exc()))
    traceback.print_exc()
    return None
  num_players = len({k.split('.', 1)[0] for k in env.action_spec()})
  r.ok('level loads (Lua init, prefabs, map)',
       f'{num_players} players, observations: '
       f'{sorted(ts.observation)[:8]}'
       + (' ...' if len(ts.observation) > 8 else ''))
  # Full episodes with random actions: catches Lua runtime errors in
  # updaters and callbacks (onEnter, ...), which only fire during play.
  rng = np.random.default_rng(0)
  max_steps = 5000
  lengths = []
  try:
    for _ in range(3):
      steps = 0
      while True:
        ts = env.step(random_action(env.action_spec(), rng))
        steps += 1
        if ts.step_type == dm_env.StepType.LAST or steps >= max_steps:
          break
      lengths.append(steps)
      ts = env.reset()
    r.check(all(n < max_steps for n in lengths),
            '3 episodes with random actions run to the end',
            f'episode lengths {lengths}')
  except Exception:  # pylint: disable=broad-except
    r.fail('3 episodes with random actions run to the end',
           _last_line(traceback.format_exc()))
    traceback.print_exc()
  finally:
    env.close()
  return settings


# ---------------------------------------------------------------------------
def check_substrate(r, name, module, config):
  r.section('Substrate (what substrate.build needs; as in substrate_test.py)')
  missing = [f for f in REQUIRED_FIELDS if f not in config]
  has_build = callable(getattr(module, 'build', None))
  if not has_build:
    missing.append('build(roles, config) function')
  if not r.check(not missing, 'config has everything substrate.build needs',
                 f'missing: {", ".join(missing)}' if missing else ''):
    r.skip('substrate checks', 'add the missing pieces first')
    return
  r.check(config.action_spec == specs.action(len(config.action_set)),
          'action_spec == specs.action(len(action_set))',
          f'{config.action_spec} vs {len(config.action_set)} actions')
  invalid = set(config.default_player_roles) - set(config.valid_roles)
  r.check(not invalid, 'default_player_roles are all in valid_roles',
          f'invalid: {invalid}' if invalid else
          f'{tuple(config.default_player_roles)}')

  config = config_dict.ConfigDict(config)
  with config.unlocked():
    config.lab2d_settings_builder = lambda *, config, roles: module.build(
        roles=roles, config=config)
  roles = tuple(config.default_player_roles)
  try:
    with source.source_levels():
      env = substrate.build_from_config(config, roles=roles)
  except Exception:  # pylint: disable=broad-except
    r.fail('substrate.build_from_config(config, default roles)',
           _last_line(traceback.format_exc()))
    traceback.print_exc()
    return
  r.ok('substrate.build_from_config(config, default roles)',
       f'{len(roles)} players')
  try:
    env.reset()
    n = len(roles)
    action = [s.maximum for s in env.action_spec()]
    ts = env.step(action)
    r.ok('step with every player taking the highest action')
    exp_obs = dict(config.timestep_spec.observation)
    exp_obs['COLLECTIVE_REWARD'] = dm_env.specs.Array(
        shape=(), dtype=np.float64, name='COLLECTIVE_REWARD')
    r.check(list(env.action_spec()) == [config.action_spec] * n,
            'action spec matches config.action_spec')
    r.check(list(env.reward_spec()) == [config.timestep_spec.reward] * n,
            'reward spec matches')
    r.check(env.discount_spec() == config.timestep_spec.discount,
            'discount spec matches')
    actual = env.observation_spec()[0]
    diff = []
    for key in sorted(set(actual) | set(exp_obs)):
      a, e = actual.get(key), exp_obs.get(key)
      if a is None:
        diff.append(f'{key}: in timestep_spec but not produced')
      elif e is None:
        diff.append(f'{key}: produced but not in timestep_spec '
                    f'({a.shape} {a.dtype})')
      elif (a.shape, a.dtype) != (e.shape, e.dtype):
        diff.append(f'{key}: spec {e.shape} {e.dtype}, actual '
                    f'{a.shape} {a.dtype}')
    r.check(not diff and list(env.observation_spec()) == [exp_obs] * n,
            'observation spec matches timestep_spec (+ COLLECTIVE_REWARD)',
            '; '.join(diff))
    bad = []
    for p, (obs, spec) in enumerate(zip(ts.observation,
                                        env.observation_spec())):
      for key, s in spec.items():
        try:
          s.validate(obs[key])
        except Exception as e:  # pylint: disable=broad-except
          bad.append(f'player {p} {key}: {e}')
    for p, rew in enumerate(ts.reward):
      try:
        env.reward_spec()[p].validate(rew)
      except Exception as e:  # pylint: disable=broad-except
        bad.append(f'player {p} reward: {e}')
    r.check(not bad, 'step output validates against the specs',
            '; '.join(bad[:3]))
    failed_actions = []
    for i in range(len(config.action_set)):
      try:
        env.step([i] * n)
      except Exception:  # pylint: disable=broad-except
        failed_actions.append(f'{i} {dict(config.action_set[i])}: '
                              f'{_last_line(traceback.format_exc())}')
    r.check(not failed_actions,
            f'every action in action_set works ({len(config.action_set)})',
            '; '.join(failed_actions[:3]))
  except Exception:  # pylint: disable=broad-except
    r.fail('substrate steps', _last_line(traceback.format_exc()))
    traceback.print_exc()
  finally:
    env.close()


# ---------------------------------------------------------------------------
def check_registration(r, name):
  r.section('Registration (configs/substrates/__init__.py, source checkout)')
  init = source.MP_SOURCE / 'meltingpot/configs/substrates/__init__.py'
  block = re.search(r'SUBSTRATES[^=]*=\s*frozenset\(\{(.*?)\}\)',
                    init.read_text(), re.S)
  entries = re.findall(r"""['"]([^'"]+)['"]""", block.group(1)) if block \
      else []
  want = f'communication.{name}'
  similar = [e for e in entries if e.replace('/', '.') == want]
  if want in entries:
    r.ok('registered in SUBSTRATES', repr(want))
  elif similar:
    r.fail('registered in SUBSTRATES',
           f'listed as {similar[0]!r}, which is not an importable module '
           f'name; use {want!r} (substrate.get_config imports '
           f'meltingpot.configs.substrates.<name>)')
  else:
    r.skip('registered in SUBSTRATES',
           f'not listed (optional; add {want!r} to use substrate.build)')


# ---------------------------------------------------------------------------
def main():
  if len(sys.argv) != 2:
    sys.exit(__doc__)
  path = pathlib.Path(sys.argv[1]).resolve()
  name = path.stem
  r = Report()
  print(f'== {name}  ({path})', flush=True)

  r.section('Config')
  try:
    module = source.load_module(path, f'comm_{name}')
    r.ok('config file imports')
  except Exception:  # pylint: disable=broad-except
    r.fail('config file imports', _last_line(traceback.format_exc()))
    traceback.print_exc()
    return r
  try:
    config = module.get_config()
    r.ok('get_config() runs')
  except Exception:  # pylint: disable=broad-except
    r.fail('get_config() runs', _last_line(traceback.format_exc()))
    traceback.print_exc()
    return r

  settings = check_level(r, module, config)
  check_substrate(r, name, module, config)
  check_registration(r, name)

  game_checks = HERE / f'checks_{name}.py'
  if game_checks.exists():
    r.section(f'Game logic ({game_checks.name})')
    if settings is None:
      r.skip('game checks', 'the level does not load')
    else:
      try:
        source.load_module(game_checks, f'checks_{name}').run(
            r, module, config, settings)
      except Exception:  # pylint: disable=broad-except
        r.fail('game checks', _last_line(traceback.format_exc()))
        traceback.print_exc()
  else:
    r.section('Game logic')
    r.skip('game checks', f'no {game_checks.name} (optional)')
  return r


if __name__ == '__main__':
  report = main()
  print(f'\nRESULT {"FAIL" if report.failed else "PASS"} '
        f'({report.failed} failed)', flush=True)
  sys.exit(1 if report.failed else 0)
