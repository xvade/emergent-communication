"""Build a self-contained HTML replay from a recording (.npz + .json).

  python -m mpviz.replay_html run.npz [more.npz ...] [-o out.html]
         [--stride K] [--no-player-views] [--scale S]

The output has no external dependencies: open it in any browser, offline.
Controls: Space play/pause, <-/-> step, Shift+<-/-> step 10, 0-9 pick view
(0 = world), Home/End jump.
"""

import argparse
import base64
import html
import json
import os

import cv2
import numpy as np

try:
  from meltingpot.utils.substrates import colors as _mp_colors
  PLAYER_COLORS = [tuple(c[:3]) for c in _mp_colors.human_readable]
except Exception:  # pylint: disable=broad-except
  PLAYER_COLORS = [(45, 110, 220), (205, 80, 60), (60, 160, 90),
                   (180, 90, 200), (220, 170, 40), (40, 170, 190),
                   (200, 90, 140), (120, 120, 120)]


def _png_b64(frame: np.ndarray) -> str:
  ok, buf = cv2.imencode('.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                         [cv2.IMWRITE_PNG_COMPRESSION, 9])
  if not ok:
    raise RuntimeError('PNG encode failed')
  return base64.b64encode(buf.tobytes()).decode('ascii')


def load(npz_path: str):
  stem = npz_path[:-4] if npz_path.endswith('.npz') else npz_path
  data = dict(np.load(stem + '.npz'))
  meta = {}
  if os.path.exists(stem + '.json'):
    with open(stem + '.json') as f:
      meta = json.load(f)
  return data, meta


def build_html(data: dict, meta: dict, *, stride: int = 1,
               player_views: bool = True, scale: int = 0) -> str:
  idx = np.arange(0, len(data['world']), stride)
  if idx[-1] != len(data['world']) - 1:
    idx = np.append(idx, len(data['world']) - 1)
  rewards = data['rewards'].astype(float)
  cum = np.cumsum(rewards, axis=0)
  n = rewards.shape[1]
  actions = data['actions']
  # Within a stride window, report the summed reward over the window.
  win_rew = np.diff(np.vstack([np.zeros((1, n)), cum[idx]]), axis=0)
  payload = dict(
      meta={k: v for k, v in meta.items() if k != 'events'},
      steps=idx.tolist(),
      world=[_png_b64(data['world'][i]) for i in idx],
      players=([[_png_b64(data['players'][i, p]) for p in range(n)]
                for i in idx]
               if player_views and 'players' in data else None),
      rewards=np.round(win_rew, 4).tolist(),
      cum=np.round(cum[idx], 4).tolist(),
      cum_full=np.round(cum, 4).tolist(),
      actions=actions[idx].tolist(),
      active=data['active'][idx].tolist() if 'active' in data else None,
      events=([meta['events'][i] for i in idx]
              if len(meta.get('events', [])) == len(data['world']) else None),
      colors=['#%02x%02x%02x' % PLAYER_COLORS[p % len(PLAYER_COLORS)]
              for p in range(n)],
      scale=scale,
  )
  blob = json.dumps(payload, separators=(',', ':')).replace('</', '<\\/')
  title = html.escape(f"{meta.get('substrate', 'Melting Pot')} replay")
  return _TEMPLATE.replace('__TITLE__', title).replace('__DATA__', blob)


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawTextHelpFormatter)
  ap.add_argument('inputs', nargs='+', help='.npz recordings')
  ap.add_argument('-o', '--output', help='output .html (single input only)')
  ap.add_argument('--stride', type=int, default=1,
                  help='keep every K-th frame (shrinks long runs)')
  ap.add_argument('--no-player-views', action='store_true',
                  help='omit per-player views even if recorded')
  ap.add_argument('--scale', type=int, default=0,
                  help='display scale (0 = fit to window)')
  args = ap.parse_args()
  if args.output and len(args.inputs) > 1:
    ap.error('-o only works with a single input')
  for path in args.inputs:
    data, meta = load(path)
    out = args.output or (path[:-4] if path.endswith('.npz') else path) + '.html'
    page = build_html(data, meta, stride=args.stride,
                      player_views=not args.no_player_views, scale=args.scale)
    with open(out, 'w') as f:
      f.write(page)
    print(f'wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB, '
          f'{len(data["world"])} frames)')


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f6f4;--panel:#fff;--fg:#1d1d1b;--muted:#6b6b66;--line:#deded9;
--accent:#2f5fd0;--chip:#efefeb}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--panel:#1d1d1d;
--fg:#e9e9e6;--muted:#9a9a94;--line:#333;--accent:#7aa2ff;--chip:#2a2a2a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;
gap:16px;align-items:baseline;flex-wrap:wrap}
header h1{font-size:16px;margin:0}
header .m{color:var(--muted);font-size:13px}
main{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:16px;
padding:16px;max-width:1500px;margin:0 auto}
@media (max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;
padding:12px}
#stage{display:flex;justify-content:center;align-items:center;
background:#000;border-radius:6px;min-height:200px;overflow:hidden}
#view{image-rendering:pixelated;image-rendering:crisp-edges;max-width:100%}
.controls{display:flex;gap:8px;align-items:center;margin-top:10px;
flex-wrap:wrap}
button,select{font:inherit;color:var(--fg);background:var(--chip);
border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
button.on{outline:2px solid var(--accent)}
input[type=range]{flex:1;min-width:160px;accent-color:var(--accent)}
#counter{font-variant-numeric:tabular-nums;color:var(--muted);min-width:9em;
text-align:right}
.views{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;
margin-right:5px;vertical-align:-1px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
td,th{padding:3px 6px;border-bottom:1px solid var(--line);text-align:left;
font-size:13px}
th{color:var(--muted);font-weight:500}
td.r{text-align:right}
tr.act td{font-weight:600}
.pos{color:#2b9348}.neg{color:#d00000}
@media (prefers-color-scheme:dark){.pos{color:#6fd08c}.neg{color:#ff7b7b}}
h2{font-size:13px;color:var(--muted);margin:14px 0 6px;font-weight:500;
text-transform:uppercase;letter-spacing:.04em}
#chart{width:100%;height:180px;display:block;cursor:crosshair}
#events{font:12px/1.4 ui-monospace,Menlo,monospace;max-height:160px;
overflow:auto;white-space:pre-wrap;color:var(--muted)}
.help{color:var(--muted);font-size:12px;margin-top:8px}
</style></head><body>
<header><h1 id="title"></h1><span class="m" id="sub"></span></header>
<main>
<section>
 <div class="card">
  <div class="views" id="views"></div>
  <div id="stage"><canvas id="view"></canvas></div>
  <div class="controls">
   <button id="play" title="Space">&#9654; Play</button>
   <button id="prev" title="Left arrow">&#9664;</button>
   <button id="next" title="Right arrow">&#9654;|</button>
   <select id="speed" title="Playback speed">
    <option value="0.25">0.25&times;</option><option value="0.5">0.5&times;</option>
    <option value="1" selected>1&times;</option><option value="2">2&times;</option>
    <option value="4">4&times;</option><option value="8">8&times;</option>
   </select>
   <input type="range" id="slider" min="0" value="0">
   <span id="counter"></span>
  </div>
  <div class="help">Space play/pause &middot; &larr;/&rarr; step &middot;
   Shift+&larr;/&rarr; &plusmn;10 &middot; 0&ndash;9 switch view (0 = world)
   &middot; click the chart to jump</div>
 </div>
 <div class="card" style="margin-top:16px">
  <h2 style="margin-top:0">Cumulative reward</h2>
  <canvas id="chart"></canvas>
 </div>
</section>
<aside class="card">
 <h2 style="margin-top:0">This step</h2>
 <table><thead><tr><th>Player</th><th>Action</th><th class="r">Reward</th>
 <th class="r">Total</th></tr></thead><tbody id="tbl"></tbody></table>
 <h2>Events</h2><div id="events">&mdash;</div>
</aside>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const M = D.meta || {};
const T = D.world.length, N = D.cum[0].length;
const fps = M.fps || 8;
let cur = 0, view = -1, playing = false, timer = null;
const cache = new Map();
const $ = id => document.getElementById(id);
const canvas = $('view'), ctx = canvas.getContext('2d');
ctx.imageSmoothingEnabled = false;

$('title').textContent = (M.substrate || 'Melting Pot') + ' replay';
$('sub').textContent = [M.source, M.steps != null ? M.steps + ' steps' : '',
  N + ' players', M.condition, M.created].filter(Boolean).join(' · ');

function img(src) {
  let im = cache.get(src);
  if (!im) { im = new Image(); im.src = 'data:image/png;base64,' + src;
             cache.set(src, im); }
  return im;
}
function frameSrc(i) {
  return view < 0 || !D.players ? D.world[i] : D.players[i][view];
}
function fit(im) {
  const stage = $('stage');
  let s = D.scale;
  if (!s) {
    const maxW = stage.clientWidth || 800, maxH = window.innerHeight * 0.62;
    s = Math.max(1, Math.floor(Math.min(maxW / im.naturalWidth,
                                        maxH / im.naturalHeight)));
  }
  const w = im.naturalWidth * s, h = im.naturalHeight * s;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h; ctx.imageSmoothingEnabled = false;
  }
}
function draw() {
  const im = img(frameSrc(cur));
  const paint = () => { fit(im); ctx.drawImage(im, 0, 0, canvas.width,
                                               canvas.height); };
  im.complete ? paint() : (im.onload = paint);
  // Prefetch a few frames ahead.
  for (let k = 1; k <= 8 && cur + k < T; k++) img(frameSrc(cur + k));
  $('slider').value = cur;
  $('counter').textContent = 'step ' + D.steps[cur] + ' / ' + D.steps[T - 1];
  const names = M.action_names || ['action'];
  const act = D.active ? D.active[cur] : -1;
  let rows = '';
  for (let p = 0; p < N; p++) {
    const r = D.rewards[cur][p], a = D.actions[cur][p];
    const aTxt = names.length === 1 ? String(a[0])
        : names.map((n, j) => a[j] ? n + '=' + a[j] : '').filter(Boolean)
              .join(' ') || '·';
    rows += '<tr class="' + (p === act ? 'act' : '') + '"><td>' +
      '<span class="sw" style="background:' + D.colors[p] + '"></span>P' +
      (p + 1) + (p === act ? ' (you)' : '') + '</td><td>' + aTxt +
      '</td><td class="r ' + (r > 0 ? 'pos' : r < 0 ? 'neg' : '') + '">' +
      (r ? (+r.toFixed(3)) : '') + '</td><td class="r">' +
      (+D.cum[cur][p].toFixed(3)) + '</td></tr>';
  }
  $('tbl').innerHTML = rows;
  const ev = D.events ? D.events[cur] : null;
  $('events').textContent = ev && ev.length
      ? ev.map(e => JSON.stringify(e)).join('\n') : '—';
  drawChart();
}
function drawChart() {
  const c = $('chart'), dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight;
  if (c.width !== W * dpr) { c.width = W * dpr; c.height = H * dpr; }
  const g = c.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const F = D.cum_full, L = F.length;
  let lo = 0, hi = 0;
  for (const row of F) for (const v of row) { lo = Math.min(lo, v);
                                              hi = Math.max(hi, v); }
  if (hi === lo) hi = lo + 1;
  const pad = 28, x = i => pad + (W - pad - 6) * i / Math.max(1, L - 1);
  const y = v => 6 + (H - 24) * (1 - (v - lo) / (hi - lo));
  const css = getComputedStyle(document.documentElement);
  g.strokeStyle = css.getPropertyValue('--line'); g.lineWidth = 1;
  g.beginPath(); g.moveTo(pad, y(0)); g.lineTo(W - 6, y(0)); g.stroke();
  g.fillStyle = css.getPropertyValue('--muted');
  g.font = '11px system-ui'; g.fillText(String(+hi.toFixed(1)), 2, 14);
  g.fillText(String(+lo.toFixed(1)), 2, H - 18);
  g.fillText('0', pad, H - 4);
  g.fillText(String(L - 1), W - 34, H - 4);
  for (let p = 0; p < N; p++) {
    g.strokeStyle = D.colors[p];
    g.lineWidth = (D.active && D.active[cur] === p) ? 2.5 : 1.5;
    g.beginPath();
    for (let i = 0; i < L; i++) {
      i ? g.lineTo(x(i), y(F[i][p])) : g.moveTo(x(i), y(F[i][p]));
    }
    g.stroke();
  }
  g.strokeStyle = css.getPropertyValue('--accent'); g.lineWidth = 1;
  g.beginPath(); g.moveTo(x(D.steps[cur]), 4);
  g.lineTo(x(D.steps[cur]), H - 16); g.stroke();
}
function go(i) { cur = Math.max(0, Math.min(T - 1, i)); draw(); }
function setPlaying(on) {
  playing = on; clearInterval(timer);
  $('play').innerHTML = on ? '&#10074;&#10074; Pause' : '&#9654; Play';
  if (on) {
    if (cur >= T - 1) cur = 0;
    const stride = T > 1 ? (D.steps[1] - D.steps[0]) : 1;
    timer = setInterval(() => {
      if (cur >= T - 1) return setPlaying(false);
      go(cur + 1);
    }, 1000 / (fps * +$('speed').value / stride));
  }
}
function setView(v) {
  view = (v >= 0 && D.players && v < N) ? v : -1;
  document.querySelectorAll('.views button').forEach(
      b => b.classList.toggle('on', +b.dataset.v === view));
  draw();
}
// View buttons.
{
  let h = '<button data-v="-1">World</button>';
  if (D.players) for (let p = 0; p < N; p++)
    h += '<button data-v="' + p + '"><span class="sw" style="background:' +
         D.colors[p] + '"></span>P' + (p + 1) + '</button>';
  $('views').innerHTML = h;
  $('views').onclick = e => { const b = e.target.closest('button');
                              if (b) setView(+b.dataset.v); };
}
$('slider').max = T - 1;
$('slider').oninput = e => go(+e.target.value);
$('play').onclick = () => setPlaying(!playing);
$('prev').onclick = () => go(cur - 1);
$('next').onclick = () => go(cur + 1);
$('speed').onchange = () => playing && setPlaying(true);
$('chart').onclick = e => {
  const r = e.target.getBoundingClientRect(), pad = 28;
  const frac = (e.clientX - r.left - pad) / (r.width - pad - 6);
  const step = Math.round(frac * (D.cum_full.length - 1));
  let best = 0;
  for (let i = 0; i < T; i++) if (D.steps[i] <= step) best = i;
  go(best);
};
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT') return;
  const k = e.key;
  if (k === ' ') { setPlaying(!playing); e.preventDefault(); }
  else if (k === 'ArrowRight') { go(cur + (e.shiftKey ? 10 : 1));
                                 e.preventDefault(); }
  else if (k === 'ArrowLeft') { go(cur - (e.shiftKey ? 10 : 1));
                                e.preventDefault(); }
  else if (k === 'Home') go(0);
  else if (k === 'End') go(T - 1);
  else if (/^[0-9]$/.test(k)) setView(+k - 1);
});
window.addEventListener('resize', draw);
setView(D.active && D.active[0] >= 0 && D.players ? D.active[0] : -1);
</script></body></html>
"""

if __name__ == '__main__':
  main()
