"""Load game configs and Lua levels from the meltingpot/ source checkout.

The container has its own *installed* copy of Melting Pot (from image build
time). Configs and Lua levels you write in the `meltingpot/` git submodule are
not part of it, so anything that loads them goes through here:

  * `load_module(path)` executes a .py file fresh from disk (no stale
    bytecode), so edits are always picked up.
  * `source_levels()` makes Melting Pot's builder resolve `levelDirectory`
    against the checkout instead of the installed package, so your Lua
    (and the `meltingpot.lua.modules` it requires) are read from the checkout.
"""

import contextlib
import pathlib
import sys
import types

from meltingpot.utils.substrates import builder

# <repo>/meltingpot: the git submodule, not the copy installed in the image.
MP_SOURCE = pathlib.Path(__file__).resolve().parents[2] / 'meltingpot'


def to_path(spec):
  """Dotted module path (relative to meltingpot/) or file path -> Path."""
  if spec.endswith('.py') or '/' in spec:
    return pathlib.Path(spec).resolve()
  return MP_SOURCE / (spec.replace('.', '/') + '.py')


def forget_config_modules(*extra_dirs):
  """Drops cached imports of config modules so the next import re-reads them.

  A config file is executed fresh on every load, but modules *it* imports
  (e.g. `from meltingpot.configs.substrates.communication import toy`) live in
  Python's import cache, so edits to them would be ignored until a restart.
  This forgets every cached module whose file is under the checkout's
  meltingpot/configs/ or under any of `extra_dirs`.

  Only plain modules are forgotten, never packages (`__init__.py`, or
  namespace folders without one): packages hold the import machinery's state
  for their children, and dropping e.g. `meltingpot.configs.substrates` breaks
  imports from namespace subfolders like `.../substrates/communication/`.
  The forgotten module is also detached from its parent package, otherwise
  `from package import module` would keep returning the old one.
  """
  roots = [MP_SOURCE / 'meltingpot' / 'configs'] + [
      pathlib.Path(d).resolve() for d in extra_dirs]
  for name, mod in list(sys.modules.items()):
    f = getattr(mod, '__file__', None)
    if not f or hasattr(mod, '__path__'):  # namespace package / package
      continue
    f = pathlib.Path(f).resolve()
    if not any(f.is_relative_to(r) for r in roots):
      continue
    del sys.modules[name]
    parent_name, _, child = name.rpartition('.')
    parent = sys.modules.get(parent_name)
    if parent is not None and getattr(parent, child, None) is mod:
      delattr(parent, child)


def load_module(path, name=None):
  """Executes a module from its source file and returns it.

  Compiles the source directly instead of importing, so no stale bytecode
  cache can be used (__pycache__ validation is by mtime-in-seconds + size,
  which a quick, same-length edit can fool).
  """
  path = pathlib.Path(path)
  if not path.is_file():
    raise FileNotFoundError(f'file not found: {path}')
  mod = types.ModuleType(name or f'mpviz_source_{path.stem}')
  mod.__file__ = str(path)
  exec(compile(path.read_text(), str(path), 'exec'), mod.__dict__)  # pylint: disable=exec-used
  return mod


@contextlib.contextmanager
def source_levels():
  """While active, builder.builder() loads levels from the checkout.

  The builder rewrites `levelDirectory` relative to the root of the installed
  meltingpot package; point that root at the checkout instead. Only needed
  while the environment is being built (the resolved path is captured then).
  """
  old = builder._MELTINGPOT_ROOT  # pylint: disable=protected-access
  builder._MELTINGPOT_ROOT = str(MP_SOURCE)  # pylint: disable=protected-access
  try:
    yield
  finally:
    builder._MELTINGPOT_ROOT = old  # pylint: disable=protected-access
