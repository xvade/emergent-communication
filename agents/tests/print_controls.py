"""Print the generated how-to-play controls for every playable substrate."""
from mpviz import play_server as ps

seen = set()
for level, (play_mod, env_module) in ps.discover().items():
  if play_mod.__name__ in seen:
    continue
  seen.add(play_mod.__name__)
  print(f'== {play_mod.__name__.split(".")[-1]} (e.g. {level})')
  for row in ps.describe_controls(play_mod):
    keys = '; '.join(k['keys'] + (f" ({k['effect']})" if k['effect'] else '')
                     for k in row['keys'])
    print(f"   {row['label']:<22} {keys}")
print()
print(ps.describe_substrate(ps.discover()['clean_up'][1])[:600])
