# Agent-managed tooling for Melting Pot on klone

| Path | What it is |
|---|---|
| `apptainer/meltingpot.def`, `build.sbatch` | Container with Melting Pot + deps (`sbatch agents/apptainer/build.sbatch`) |
| `bin/mp-play` | Play any substrate as a human, in your browser (`--tutorial` for the substrate tutorial) |
| `bin/mp-replay` | Turn recorded episodes into standalone HTML replays |
| `mpviz/` | Python package behind both (`recording`, `replay_html`, `play_server`) |
| `experiments/` | Experiments (e.g. `random_zap_ablation.py`, `run.sbatch`) |
| `tests/smoke_mpviz.py` | End-to-end check of mpviz (`sbatch agents/tests/smoke.sbatch`) |

All commands below are run **from the repo root on a klone login node**. The
launchers request their own small Slurm allocation; nothing heavy runs on the
login node.

> **Slurm account / partition:** everything defaults to `--account=stf
> --partition=compute`. Override per command with flags:
>
> ```bash
> agents/bin/mp-play  -p cpu-g2 -A stf  --substrate clean_up
> agents/bin/mp-replay --partition ckpt --account stf  run.npz
> sbatch -p cpu-g2 -A stf agents/experiments/run.sbatch   # sbatch flags override #SBATCH lines
> ```
>
> or set a personal default in `~/.bashrc`:
> `export MP_ACCOUNT=stf MP_PARTITION=cpu-g2`. Run `hyakalloc` to see what you
> can use. `-t/--minutes` sets the time limit. If `compute` is full, `-p ckpt`
> uses idle nodes cluster-wide. Jobs there can be preempted, which is fine
> for short play or replay sessions.

> **Storage:** this repo lives on `/gscratch/scrubbed`, which **deletes files
> not used for ~21 days**. Recordings and replays you want to keep should be
> copied to your lab's `/gscratch/<group>/` or your laptop.

---

## 1. Play a substrate as a human

```bash
agents/bin/mp-play                                   # commons_harvest__open
agents/bin/mp-play --substrate clean_up
agents/bin/mp-play -t 120 --substrate territory__rooms   # 2-hour job
agents/bin/mp-play --list                            # all playable substrates
```

After a few seconds (and any queue wait) it prints something like:

```
 1) On YOUR computer ... run:
      ssh -N -L 8765:n3456:41234 sgvtc@klone.hyak.uw.edu
 2) Open in your browser:
      http://localhost:8765/?token=Xyz...
```

Do step 1 in a **new terminal on your own computer** (leave it running; it
prints nothing), then open the URL. Click the page once so it has keyboard
focus.

- **Controls** are the ones from Melting Pot's own `human_players/play_*.py`
  scripts and are shown in the side panel (typically WASD move, Q/E turn,
  Space zap/fire; some substrates use 1–5, Shift, C/Z/X). On a Mac, the
  "delete" key acts as Delete.
- **Tab** (or the dropdown) switches which player you control. Other players
  take no-op actions.
- **Real-time** by default at the substrate's fps (usually 8 steps/s).
  Tick **turn-based** to advance one step per key press — easier for careful
  exploration. **Pause** stops the clock.
- **Substrate dropdown** switches substrate without restarting the job.
- Every episode is **recorded automatically** when it ends, when you click
  **End & save replay**, start a **New episode**, or switch substrate. The
  replay opens from the link that appears, or from **Replays** (lists all
  replays in `agents/recordings/`).
- **Ctrl-C** (once) in the klone terminal saves the current episode, stops
  the server and releases the allocation. Press it again to force-stop
  without saving. The server also saves and exits after 30 idle minutes
  (`--idle-minutes`). When the `-t/--minutes` time limit runs out it tries
  to save, but Slurm allows only about 12 s before killing the job, so save
  long episodes yourself first.

### Connecting from different setups

The server listens only on the compute node's internal cluster address and
requires the random token in the URL. You reach it through an SSH tunnel via
the login node.

**Mac / Linux terminal** — use the printed `ssh -N -L ...` command. To avoid a
second Duo prompt for each tunnel, add this to `~/.ssh/config` on your
computer (replace the NetID):

```
Host klone
    HostName klone.hyak.uw.edu
    User YOUR_NETID
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 4h
```

Then `ssh klone` once (Duo), and tunnels like
`ssh -N -L 8765:n3456:41234 klone` reuse that connection with no prompt.

**Windows (PowerShell / Windows Terminal)** — the same `ssh -N -L ...` command
works with the built-in OpenSSH client. (ControlMaster isn't supported on
Windows, so expect a Duo prompt per tunnel.)

**Windows (PuTTY / MobaXterm)** — Connection → SSH → Tunnels: source port
`8765`, destination `n3456:41234` (from the printed message), click *Add*,
then connect to `klone.hyak.uw.edu`.

**VS Code Remote-SSH** — run `mp-play` in the VS Code terminal, then in the
**Ports** panel click *Forward a Port* and enter the `node:port` shown (e.g.
`n3456:41234`). Open the forwarded address VS Code shows, adding the
`/?token=...` part from the printed URL.

**Port 8765 already in use locally?** Pick any other local port, e.g.
`ssh -N -L 9000:n3456:41234 ...` and open `http://localhost:9000/?token=...`
(or pass `--local-port 9000` so the printed instructions match).

---

## 2. Replay any run

Replays are single self-contained `.html` files (no internet needed). Open
one in any browser:

- scrubber, play/pause, speed (0.25×–8×)
- ←/→ step one frame, Shift+←/→ ten, Home/End
- **World** view or any **player's egocentric view** (keys 0–9 or buttons)
- per-step table of each player's action and reward; cumulative reward chart
  (click it to jump); dmlab2d events for the current step

**Getting a replay onto your computer:**

```bash
# Mac/Linux/Windows PowerShell, on your computer:
scp klone:/gscratch/scrubbed/sgvtc/emergent-communication/agents/recordings/<file>.html .
open <file>.html      # Mac;  xdg-open on Linux;  start on Windows
```

Or right-click the file in VS Code's explorer → *Download*, or open it from
the **Replays** link on a running `mp-play` page.

### Recording your own experiments

Wrap any substrate; every episode is saved as `<name>.npz` + `<name>.json`:

```python
from mpviz import recording
env = recording.wrap(substrate.build(name, roles=roles), 'agents/recordings',
                     substrate=name, max_episodes=2,
                     record_player_views=True)   # optional, larger files
... use env as usual; call env.close() (or env.flush()) at the end ...
```

Then build the HTML replays (runs as a short Slurm job):

```bash
agents/bin/mp-replay agents/recordings/*.npz
agents/bin/mp-replay big_run.npz --stride 4 --no-player-views   # smaller file
```

Example: the zap ablation with recording on:

```bash
sbatch agents/experiments/run.sbatch --record 1 --record-player-views
agents/bin/mp-replay agents/experiments/results/recordings/*.npz
```

Size guide (measured, Commons Harvest, 7 players, 1,000 steps): about 6 MB
`.npz` and 11 MB `.html` with player views. Use `--stride` for longer runs.

---

## 3. Work through the substrate tutorial

`meltingpot/docs/substrate_tutorial/index.md` builds a game step by step by
editing files and looking at the result. `mp-play` can load those files
straight from the `meltingpot/` checkout and re-read them on demand:

```bash
agents/bin/mp-play --tutorial
```

This opens the tutorial's game
(`examples/tutorial/harvest/configs/environment/harvest.py`) first, with the
finished version (`harvest_finished.py`) next in the substrate dropdown for
comparison. Then repeat:

1. Edit the files the tutorial mentions. The compute node sees saved changes
   immediately. Use VS Code Remote-SSH or an editor on klone.
   - `meltingpot/examples/tutorial/harvest/configs/environment/harvest.py`
   - `meltingpot/examples/tutorial/harvest/levels/harvest/init.lua`, and the
     `components.lua` you create next to it
2. Click **Reload files** on the page. It re-reads the Python config and the
   Lua level from disk and starts a fresh episode in about a second, with no
   job restart. The discarded episode is not saved.
3. If something is wrong (a Python exception, a Lua error such as
   `The component DensityRegrow does not exist in the registry`), the page
   shows the full error in red and the server keeps running. Fix the file and
   click **Reload files** again.

The tutorial's command-line options map to the page as follows:
- `--observation WORLD.RGB` vs. the default first-person view: the page
  always shows both side by side. With 0 players (the tutorial's first steps)
  it shows the world view only.
- `--display_text`: if the game has a `WORLD.TEXT` observation, the page
  shows it under the world view.
- `--verbose`: use a replay instead. Each step's actions, rewards and events
  are listed there.
- Controls are the tutorial's (WASD move, Q/E turn). An action your config
  adds without a key, e.g. a new beam, is listed as "no key bound" and always
  sends 0.

For your own game, `--config` takes any module with `get_config()`, as a
dotted path relative to `meltingpot/` or a `.py` file path. `--play-module`
takes a module whose `_ACTION_MAP` binds keys to its actions:

```bash
agents/bin/mp-play --config path/to/my_game.py --play-module path/to/play_my_game.py
```

> Tip: the tutorial edits files inside the `meltingpot/` submodule (your
> fork). Work on a branch there, e.g. `git -C meltingpot switch -c tutorial`,
> to keep its `main` clean.

## Troubleshooting

- **Browser shows "connection refused"**: the tunnel isn't running, or the job
  ended (check `squeue -u $USER`). Node and port change on every `mp-play`.
- **403 bad token**: the URL is missing `?token=...`.
- **Keys do nothing**: click the page (a blue "Click the page..." badge shows
  when it lacks focus).
- **Laggy**: each step is one round trip through the tunnel, so real-time
  play needs a latency under ~100 ms. If it feels sluggish, use turn-based
  mode.
- **`Container not found`**: build it with `sbatch agents/apptainer/build.sbatch`.
