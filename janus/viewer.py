"""The viewer process: a side-by-side page served from the head node.

Three panels, left to right:

  REALITY    the frame the inferencer just saw
  PREDICTED  what the model said, half a second ago, this moment would look like
             (its latent prediction, decoded to pixels by the probe decoder)
  DREAM      a free-running rollout: seeded from reality once, then fed only its
             own output, one horizon per step. Never trained on. Its head is the
             dream's belief about the next horizon boundary; when reality gets
             there the head is scored and the dream rolls on.

Nothing here touches the frame loop. The inferencer drops panels + a stats
vector onto the DisplayBus every few frames; this process turns them into
MJPEG streams and a JSON endpoint with the standard library only.
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import numpy as np

from .bus import DisplayBus

STATS_KEYS = [
    "t", "frame", "fps", "version", "surprise", "z", "copy_err", "skill",
    "dream_steps", "dream_err", "dream_copy", "dream_skill", "dream_ahead_s", "dream_age_s",
    "regime", "anomaly", "frame_ms", "swaps", "dream_seeds",
]
REGIMES = ["bouncing blobs", "rotating grating", "scrolling bars", "lissajous swarm",
           "expanding rings"]
PANEL_PX = 288


def _learner_tail(run_dir: str) -> dict:
    """Last complete line of learn.jsonl, or {}. Cheap: seeks to the end."""
    path = os.path.join(run_dir, "learn.jsonl")
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4096))
            lines = f.read().splitlines()
        for line in reversed(lines):
            if line.strip():
                d = json.loads(line)
                if "step" in d:
                    return d
    except (OSError, ValueError):
        pass
    return {}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Janus</title>
<style>
:root{--bg:#0e1116;--panel:#161b22;--ink:#e6edf3;--mute:#8b98a5;--rule:#2a323c;
      --real:#e6edf3;--pred:#4cc2ff;--dream:#ffb454;--bad:#ff6b6b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:14px 12px 40px}
header{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:10px}
h1{font-size:18px;margin:0;letter-spacing:.02em}
h1 small{color:var(--mute);font-weight:400;margin-left:8px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
.panels{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:4px;padding:6px}
.panel img{display:block;width:100%;aspect-ratio:1/1;image-rendering:pixelated;background:#000;border-radius:2px}
.panel .lab{display:flex;justify-content:space-between;align-items:baseline;gap:6px;margin:0 2px 6px;font-size:12px;letter-spacing:.08em;text-transform:uppercase}
.panel .sub{font-size:12px;color:var(--mute);margin:6px 2px 0;min-height:1.2em}
.real .lab{color:var(--real)} .pred .lab{color:var(--pred)} .dream .lab{color:var(--dream)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin-top:12px}
.stat{background:var(--panel);border:1px solid var(--rule);border-radius:4px;padding:8px 10px}
.stat .k{font-size:11px;color:var(--mute);letter-spacing:.06em;text-transform:uppercase}
.stat .v{font-size:20px;margin-top:2px}
.stat .v.small{font-size:15px}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:12px}
button{background:var(--panel);color:var(--ink);border:1px solid var(--rule);border-radius:4px;padding:8px 14px;font:inherit;cursor:pointer}
button:hover{border-color:var(--dream)} button:focus-visible{outline:2px solid var(--dream)}
canvas{width:100%;height:120px;display:block;background:var(--panel);border:1px solid var(--rule);border-radius:4px;margin-top:12px}
.legend{display:flex;gap:16px;font-size:12px;color:var(--mute);margin-top:6px}
.legend i{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:6px}
p.note{color:var(--mute);font-size:13px;max-width:70ch;margin:14px 0 0}
.flag{color:var(--bad);font-weight:600}
</style></head><body><main>
<header><h1>Janus <small>learning while serving</small></h1>
<div class="mono" id="hdr">connecting…</div></header>

<div class="panels">
 <div class="panel real"><div class="lab"><span>Reality</span><span class="mono" id="real-t"></span></div>
  <img src="/stream/real.mjpg" alt="live frame"><div class="sub" id="real-sub">now</div></div>
 <div class="panel pred"><div class="lab"><span>Predicted</span><span class="mono">−0.5 s</span></div>
  <img src="/stream/pred.mjpg" alt="decoded prediction"><div class="sub" id="pred-sub">what the model said this moment would look like</div></div>
 <div class="panel dream"><div class="lab"><span>Dream</span><span class="mono" id="dream-k"></span></div>
  <img src="/stream/dream.mjpg" alt="decoded dream head"><div class="sub" id="dream-sub">free-running, fed only its own output</div></div>
</div>

<div class="stats">
 <div class="stat"><div class="k">surprise z</div><div class="v mono" id="s-z">–</div></div>
 <div class="stat"><div class="k">served skill vs copy</div><div class="v mono" id="s-skill">–</div></div>
 <div class="stat"><div class="k">dream error / copy</div><div class="v mono" id="s-derr">–</div></div>
 <div class="stat"><div class="k">dream skill</div><div class="v mono" id="s-dskill">–</div></div>
 <div class="stat"><div class="k">weight version</div><div class="v mono" id="s-ver">–</div></div>
 <div class="stat"><div class="k">learner steps / s</div><div class="v mono" id="s-sps">–</div></div>
 <div class="stat"><div class="k">embedding rank</div><div class="v mono" id="s-erank">–</div></div>
 <div class="stat"><div class="k">decoder L1</div><div class="v mono" id="s-rec">–</div></div>
 <div class="stat"><div class="k">frame time</div><div class="v mono" id="s-ms">–</div></div>
 <div class="stat"><div class="k">regime</div><div class="v small" id="s-regime">–</div></div>
</div>

<canvas id="chart" width="1000" height="120"></canvas>
<div class="legend"><span><i style="background:var(--pred)"></i>surprise z (served)</span><span><i style="background:var(--dream)"></i>dream error</span><span><i style="background:var(--mute)"></i>dream copy baseline</span></div>

<div class="row"><button id="resync">Resync dream to reality</button><span class="mono" id="seeds"></span></div>

<p class="note">The model predicts embeddings, never pixels. Both decoded panels come from a probe decoder trained only on detached embeddings, so it can show what an embedding holds but cannot shape it. The dream is inference only: nothing it produces is ever used as a training target. Its error is scored each time reality reaches the frame the head was a belief about; the copy baseline is “the world still looks like when the dream started”.</p>
</main>
<script>
const $=id=>document.getElementById(id);
const hist={z:[],derr:[],dcopy:[],t:[]};const HORIZON_S=90;
const f=(x,d=2)=>Number.isFinite(x)?x.toFixed(d):'–';
let lastDreamStep=-1;
async function tick(){
  try{
    const r=await fetch('/stats.json',{cache:'no-store'});const s=await r.json();
    $('hdr').textContent=`${f(s.fps,1)} fps · v${s.version|0} · ${f(s.t,0)} s`;
    $('real-t').textContent=`#${s.frame|0}`;
    $('real-sub').innerHTML=s.anomaly?'<span class="flag">anomaly injected</span>':'now';
    $('pred-sub').textContent=`surprise ${f(s.surprise,3)} · copy ${f(s.copy_err,3)}`;
    $('dream-k').textContent=`step ${s.dream_steps|0}`;
    $('dream-sub').textContent=`believes +${f(s.dream_ahead_s,1)} s · ${f(s.dream_age_s,1)} s since seed · resyncs ${s.dream_seeds|0}`;
    $('s-z').textContent=f(s.z,2);
    $('s-skill').textContent=f(s.skill,3);
    $('s-derr').textContent=`${f(s.dream_err,3)} / ${f(s.dream_copy,3)}`;
    $('s-dskill').textContent=f(s.dream_skill,3);
    $('s-ver').textContent=`${s.version|0}`;
    $('s-sps').textContent=f(s.learner.steps_per_s,1);
    $('s-erank').textContent=f(s.learner.erank,1);
    $('s-rec').textContent=f(s.learner.rec,3);
    $('s-ms').textContent=`${f(s.frame_ms,2)} ms`;
    $('s-regime').textContent=s.regime_name||'–';
    $('seeds').textContent='';
    const now=s.t;hist.t.push(now);hist.z.push(s.z);
    if((s.dream_steps|0)!==lastDreamStep){lastDreamStep=s.dream_steps|0;hist.derr.push([now,s.dream_err,s.dream_copy]);}
    while(hist.t.length&&now-hist.t[0]>HORIZON_S){hist.t.shift();hist.z.shift();}
    while(hist.derr.length&&now-hist.derr[0][0]>HORIZON_S){hist.derr.shift();}
    draw(now);
  }catch(e){$('hdr').textContent='waiting for the inferencer…';}
}
function draw(now){
  const c=$('chart'),ctx=c.getContext('2d'),W=c.width,H=c.height;ctx.clearRect(0,0,W,H);
  const css=getComputedStyle(document.documentElement);
  const x=t=>W-((now-t)/HORIZON_S)*W;
  ctx.strokeStyle=css.getPropertyValue('--rule');ctx.lineWidth=1;
  for(const g of[0.25,0.5,0.75]){ctx.beginPath();ctx.moveTo(0,H*g);ctx.lineTo(W,H*g);ctx.stroke();}
  // surprise z: scale -3..+5 onto the canvas
  const yz=z=>H-((Math.max(-3,Math.min(5,z))+3)/8)*H;
  ctx.strokeStyle=css.getPropertyValue('--pred');ctx.lineWidth=1.5;ctx.beginPath();
  hist.t.forEach((t,i)=>{const X=x(t),Y=yz(hist.z[i]);i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});ctx.stroke();
  // dream error and its copy baseline: cosine distance 0..1
  const ye=e=>H-Math.max(0,Math.min(1,e))*H;
  const series=(k,col)=>{ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.beginPath();
    hist.derr.forEach((d,i)=>{const X=x(d[0]),Y=ye(d[k]);i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});ctx.stroke();};
  series(2,css.getPropertyValue('--mute'));series(1,css.getPropertyValue('--dream'));
}
$('resync').addEventListener('click',async()=>{await fetch('/resync',{method:'POST'});});
setInterval(tick,200);tick();
</script></body></html>
"""


def _jpeg(panel_u8: np.ndarray) -> bytes:
    import cv2
    big = cv2.resize(panel_u8, (PANEL_PX, PANEL_PX), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".jpg", big[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return buf.tobytes() if ok else b""


def make_handler(cfg, display: DisplayBus, run_dir: str, stop_event):
    panel_index = {name: i for i, name in enumerate(DisplayBus.PANELS)}

    class Handler(BaseHTTPRequestHandler):
        server_version = "janus-viewer/0.1"

        def log_message(self, fmt, *args):      # keep the run's stdout for the loop
            pass

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, "text/html; charset=utf-8", PAGE.encode())
            if self.path.startswith("/stats.json"):
                seq, _, stats = display.read()
                d = dict(zip(STATS_KEYS, stats))
                d["seq"] = seq
                d["regime_name"] = (REGIMES[int(d.get("regime", -1))]
                                    if 0 <= int(d.get("regime", -1)) < len(REGIMES)
                                    else ("desktop" if cfg.source == "x11" else cfg.source))
                d["learner"] = _learner_tail(run_dir)
                d["horizon_s"] = cfg.horizon_frames / cfg.fps
                body = json.dumps({k: (None if isinstance(v, float) and v != v else v)
                                   for k, v in d.items()}).encode()
                return self._send(200, "application/json", body)
            if self.path.startswith("/stream/") and self.path.endswith(".mjpg"):
                name = self.path[len("/stream/"):-len(".mjpg")]
                if name not in panel_index:
                    return self._send(404, "text/plain", b"no such panel")
                return self._stream(panel_index[name])
            return self._send(404, "text/plain", b"not found")

        def do_POST(self):
            if self.path == "/resync":
                display.request(DisplayBus.CMD_RESYNC)
                return self._send(200, "application/json", b'{"ok":true}')
            return self._send(404, "text/plain", b"not found")

        def _stream(self, panel: int) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = -1
            try:
                while not stop_event.is_set():
                    seq = display.current_seq()
                    if seq == last:
                        time.sleep(0.01)
                        continue
                    last = seq
                    _, frames, _ = display.read()
                    jpg = _jpeg(frames[panel].numpy())
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

    return Handler


def viewer_main(cfg, display: DisplayBus, stop_event, run_dir: str) -> None:
    handler = make_handler(cfg, display, run_dir, stop_event)
    httpd = ThreadingHTTPServer((cfg.viewer_host, cfg.viewer_port), handler)
    httpd.daemon_threads = True
    t = threading.Thread(target=httpd.serve_forever, name="viewer-http", daemon=True)
    t.start()
    print(f"[viewer] http://{cfg.viewer_host}:{cfg.viewer_port}/  (panels: "
          f"{', '.join(DisplayBus.PANELS)})", flush=True)
    try:
        while not stop_event.is_set():
            time.sleep(0.25)
    finally:
        httpd.shutdown()
        httpd.server_close()
        print("[viewer] stopped", flush=True)
