"""Runs check_substrate.py for every config in the communication folder.

  python agents/tests/communication/run_all.py [name ...]

Each config runs in its own process: a game that crashes the engine (e.g. a
Lua error inside onEnter aborts the whole process) is reported as CRASH and
the remaining configs still run. Exit status is 1 if anything failed.
"""

import pathlib
import signal
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
COMM = REPO / 'meltingpot/meltingpot/configs/substrates/communication'
TIMEOUT = 900  # seconds per config


def main(names):
  configs = sorted(p for p in COMM.glob('*.py')
                   if p.name != '__init__.py' and not p.stem.endswith('_test'))
  if names:
    configs = [p for p in configs if p.stem in names]
    unknown = set(names) - {p.stem for p in configs}
    if unknown:
      print(f'No such config(s) in {COMM}: {sorted(unknown)}')
      return 1
  if not configs:
    print(f'No configs found in {COMM}')
    return 1
  print(f'Testing {len(configs)} config(s) in {COMM}\n', flush=True)
  results = {}
  for path in configs:
    t0 = time.time()
    try:
      proc = subprocess.run(
          [sys.executable, str(HERE / 'check_substrate.py'), str(path)],
          timeout=TIMEOUT, check=False)
      code = proc.returncode
      if code == 0:
        status = 'PASS'
      elif code < 0:
        sig = signal.Signals(-code).name
        status = f'CRASH ({sig})'
        print(f'\nCRASH  {path.stem}: the game process died with {sig}. '
              'This is almost always a Lua error inside a callback such as '
              'onEnter/onExit (the engine aborts instead of raising); look '
              'for "ERROR:" lines above to find the file and line.',
              flush=True)
      else:
        status = 'FAIL'
    except subprocess.TimeoutExpired:
      status = f'TIMEOUT (>{TIMEOUT}s)'
    results[path.stem] = (status, time.time() - t0)
    print('\n' + '=' * 70 + '\n', flush=True)

  print('SUMMARY')
  for name, (status, secs) in results.items():
    print(f'  {status:<16} {name}  ({secs:.0f}s)')
  ok = all(s == 'PASS' for s, _ in results.values())
  print(f'\nOVERALL {"PASS" if ok else "FAIL"}', flush=True)
  return 0 if ok else 1


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
