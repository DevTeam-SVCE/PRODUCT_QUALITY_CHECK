"""Live camera UI (browser) and the capture.py command line.

Standard library only: an HTTP server streams the annotated preview as MJPEG and serves a small status page,
so no GUI toolkit is needed (OpenCV here is the headless build) and the UI also works from a tablet or phone
on the same network (--host 0.0.0.0).
"""
import argparse
import json
import logging
import mimetypes
import os
import re
import ssl
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config
from .camera import CameraApp
from .refframe import load_intrinsics

log = logging.getLogger("partvision.camera")

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Camera · Product Quality Check</title>
<style>
:root{--bg:#0e1012;--panel:#16191c;--line:#24282d;--text:#e8eaed;--muted:#9aa0a6;--ok:#4cc38a;--warn:#f0a83a;
--bad:#ef5f5f;--idle:#5f6368;--accent:#5b9cff}
*{box-sizing:border-box}html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
-webkit-font-smoothing:antialiased}
.app{display:flex;flex-direction:column;min-height:100%}
header{display:flex;align-items:center;justify-content:space-between;padding:12px 20px;border-bottom:1px solid var(--line)}
.brand{font-weight:600;letter-spacing:.2px}.brand span{color:var(--muted);font-weight:400;margin-left:6px}
.cam{color:var(--muted);font-size:12.5px;display:flex;align-items:center;gap:8px;font-variant-numeric:tabular-nums}
.cam i{width:8px;height:8px;border-radius:50%;background:var(--idle);display:inline-block}
.cam.live i{background:var(--ok)}.cam.down i{background:var(--bad)}
main{flex:1;display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:16px;padding:16px 20px 20px}
.stage{display:flex;flex-direction:column;align-items:center;justify-content:flex-start;min-width:0}
.viewport{position:relative;width:100%;max-height:calc(100vh - 120px);display:flex;justify-content:center;
background:#000;border-radius:12px;overflow:hidden;border:1px solid var(--line)}
#feed{display:block;max-width:100%;max-height:calc(100vh - 122px);object-fit:contain}
.flash{position:absolute;inset:0;background:#fff;opacity:0;pointer-events:none;transition:opacity .35s ease-out}
.flash.on{opacity:.55;transition:none}
.banner{position:absolute;left:50%;bottom:22px;transform:translateX(-50%);display:flex;align-items:center;gap:10px;
padding:10px 18px;border-radius:999px;background:rgba(14,16,18,.82);backdrop-filter:blur(6px);
border:1px solid rgba(255,255,255,.08);font-weight:500;white-space:nowrap;max-width:calc(100% - 32px);
transition:border-color .2s}
.banner .dot{width:9px;height:9px;border-radius:50%;background:var(--idle);flex:none;transition:background .2s}
.banner #msg{overflow:hidden;text-overflow:ellipsis}
.banner.ok{border-color:rgba(76,195,138,.55)}.banner.ok .dot{background:var(--ok)}
.banner.warn .dot{background:var(--warn)}.banner.bad .dot{background:var(--bad)}
.hold{position:absolute;left:0;right:0;bottom:-1px;height:3px;border-radius:0 0 999px 999px;overflow:hidden}
.hold div{height:100%;width:0;background:var(--ok);transition:width .12s linear}
.hint{margin-top:10px;color:var(--muted);min-height:20px;text-align:center;font-size:13px}
aside{display:flex;flex-direction:column;gap:12px;min-width:0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.card h3{margin:0 0 10px;font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}
.row{display:flex;gap:12px;padding:7px 0;align-items:flex-start}
.row+.row{border-top:1px solid var(--line)}
.ind{width:10px;height:10px;border-radius:50%;margin-top:5px;flex:none;background:var(--idle);transition:background .2s}
.ind.ok{background:var(--ok)}.ind.fail{background:var(--warn)}.ind.pending{background:var(--accent)}
.row .t{font-weight:500}.row .d{color:var(--muted);font-size:12.5px;font-variant-numeric:tabular-nums;min-height:18px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px 14px}
.grid div span{display:block;color:var(--muted);font-size:11.5px}
.grid div b{font-weight:500;font-variant-numeric:tabular-nums}
.cap{display:none}.cap.show{display:block;animation:in .3s ease-out}
@keyframes in{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.caphead{display:flex;justify-content:space-between;align-items:baseline}
.caphead small{color:var(--muted);font-variant-numeric:tabular-nums}
.thumb{display:block;margin:4px 0 12px;background:#0b0c0d;border-radius:8px;border:1px solid var(--line);overflow:hidden}
.thumb img{display:block;width:100%;max-height:190px;object-fit:contain}
dl{display:grid;grid-template-columns:auto 1fr;gap:5px 12px;margin:0;font-size:13px}
dt{color:var(--muted)}dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
.insp{margin-top:12px;padding-top:12px;border-top:1px solid var(--line);font-size:13px}
.insp .counts{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.pill{padding:2px 9px;border-radius:999px;background:#22262b;font-size:12px}
.files{margin-top:12px;display:flex;flex-wrap:wrap;gap:6px 14px;font-size:12.5px}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.actions{display:flex;gap:10px;margin-top:14px}
button{font:inherit;font-weight:500;border-radius:10px;padding:10px 14px;cursor:pointer;border:1px solid var(--line);
background:var(--panel);color:var(--text);flex:1}
button.primary{background:var(--text);color:#111;border-color:var(--text)}
button.link{background:none;border:none;color:var(--muted);padding:6px;font-weight:400;font-size:12.5px}
button.link:hover{color:var(--text)}
button:disabled{opacity:.4;cursor:default}
button:not(:disabled):hover{filter:brightness(1.1)}
.err{color:var(--bad);font-size:12.5px;min-height:0}
@media (max-width:900px){main{grid-template-columns:1fr;padding:12px 16px}
#feed,.viewport{max-height:none}.banner{font-size:13px;padding:8px 14px;bottom:14px;white-space:normal;
border-radius:14px;width:max-content}}
</style></head><body><div class="app">
<header><div class="brand">Product Quality Check<span>Camera</span></div>
<div class="cam" id="cam"><i></i><span id="camtxt">Connecting…</span></div></header>
<main>
<section class="stage"><div class="viewport"><img id="feed" src="/stream.mjpg" alt="Live camera">
<div class="flash" id="flash"></div>
<div class="banner" id="banner"><span class="dot"></span><span id="msg">Starting…</span>
<div class="hold"><div id="hold"></div></div></div></div>
<div class="hint" id="hint"></div></section>
<aside>
<div class="card"><h3>Status</h3>
<div class="row" id="g-frame"><span class="ind"></span><div><div class="t">Reference frame</div><div class="d"></div></div></div>
<div class="row" id="g-alignment"><span class="ind"></span><div><div class="t">Camera alignment</div><div class="d"></div></div></div>
<div class="row" id="g-stability"><span class="ind"></span><div><div class="t">Stability &amp; image</div><div class="d"></div></div></div>
<div class="err" id="err"></div></div>
<div class="card cap" id="cap"><div class="caphead"><h3>Captured</h3><small id="c-time"></small></div>
<a class="thumb" id="c-link" target="_blank"><img id="c-img" alt="Rectified capture"></a>
<dl><dt>Scale</dt><dd id="c-scale"></dd><dt>Rectified</dt><dd id="c-size"></dd><dt>Tilt</dt><dd id="c-tilt"></dd>
<dt>Alignment score</dt><dd id="c-align"></dd><dt>Sharpness score</dt><dd id="c-sharp"></dd>
<dt>Capture time</dt><dd id="c-lat"></dd></dl>
<div class="insp" id="insp"></div><div class="files" id="files"></div>
<div class="actions"><button class="primary" id="retake" disabled>Retake</button></div></div>
<div class="card"><h3>Live geometry</h3><div class="grid">
<div><span>Tilt</span><b id="m-tilt">–</b></div><div><span>Keystone</span><b id="m-key">–</b></div>
<div><span>Sharpness</span><b id="m-sharp">–</b></div><div><span>Scale (est.)</span><b id="m-scale">–</b></div>
</div></div>
<button class="link" id="manual" title="Capture now, even if not every check passes (recorded in the metadata)">Capture manually</button>
</aside></main></div>
<script>
const $=id=>document.getElementById(id);
const tone={search:'',partial:'warn',detecting:'',adjust:'warn',steady:'warn',ready:'ok',capturing:'ok',captured:'ok',
 disconnected:'bad',starting:''};
let lastCap=null,busy=false;
const f=(v,d=1)=>v==null?'–':Number(v).toFixed(d);
function group(id,g){const r=$(id);const s=(g&&g.status)||'idle';r.querySelector('.ind').className='ind '+s;
 r.querySelector('.d').textContent=(g&&g.detail)||(s==='idle'?'Waiting for the reference frame':'');}
function render(s){
 const c=s.camera||{};$('cam').className='cam '+(c.connected?'live':'down');
 $('camtxt').textContent=c.connected?`Live · ${c.size?c.size.join('×'):''} · ${f(c.fps,0)} fps · ${c.lens} lens`:'Camera not connected';
 const b=$('banner');b.className='banner '+(tone[s.state]||'');$('msg').textContent=s.message||'';
 $('hint').textContent=s.hint||'';$('err').textContent=s.error||'';
 const g=s.groups||{};group('g-frame',g.frame);group('g-alignment',g.alignment);group('g-stability',g.stability);
 const m=s.metrics||{};
 $('m-tilt').textContent=m.tilt_deg==null?'–':f(m.tilt_deg)+'°';
 $('m-key').textContent=m.keystone==null?'–':f(m.keystone*100)+' %';
 $('m-sharp').textContent=m.edge_width_px==null?'–':`${f(m.edge_width_px)} / ${f(m.edge_width_limit_px)} px`;
 $('m-scale').textContent=m.mm_per_px==null?'–':f(m.mm_per_px,4)+' mm/px';
 const st=(s.gates||{}).stability;let hold=0;
 if(st&&st.detail){const x=/held (\d+)\/(\d+)/.exec(st.detail);if(x)hold=Math.min(1,x[1]/x[2]);}
 if(['ready','capturing','captured'].includes(s.state))hold=1;else if(s.state!=='steady')hold=0;
 $('hold').style.width=(hold*100)+'%';
 const cap=s.capture;$('retake').disabled=!cap;
 if(cap){if(!lastCap||lastCap.id!==cap.id){const fl=$('flash');fl.classList.add('on');
   requestAnimationFrame(()=>requestAnimationFrame(()=>fl.classList.remove('on')));
   $('c-img').src='/files/'+cap.files.rectified;$('c-link').href='/files/'+cap.files.rectified;}
  $('cap').classList.add('show');$('c-time').textContent=new Date(cap.timestamp).toLocaleTimeString();
  $('c-scale').textContent=`${f(cap.mm_per_px,4)} mm/px · ${f(cap.px_per_mm,2)} px/mm`;
  $('c-size').textContent=`${cap.rectified_wh.join(' × ')} px · ${cap.orientation}`;
  $('c-tilt').textContent=f(cap.tilt_deg,2)+'°'+(cap.tilt_estimate.startsWith('assumed')?' (est.)':'');
  $('c-align').textContent=f(cap.alignment_score,2);$('c-sharp').textContent=f(cap.sharpness_score,2);
  $('c-lat').textContent=cap.finalize_ms+' ms'+(cap.manual?' · manual':'');
  const fs=cap.files;$('files').innerHTML=['rectified','original','detection','calibration'].map(k=>
   `<a href="/files/${fs[k]}" target="_blank">${k[0].toUpperCase()+k.slice(1)}</a>`).join('');
  const i=cap.inspection;let h='';
  if(!i)h='<span style="color:var(--muted)">Inspection not enabled</span>';
  else if(i.state==='running'||i.state==='queued')h=`<span style="color:var(--muted)">${i.state==='queued'?'Inspection queued…':'Running inspection…'}</span>`;
  else if(i.state==='failed')h=`<span style="color:var(--bad)">Inspection failed: ${i.error||''}</span>`;
  else{h=i.drawing?`Drawing <b>${i.drawing}</b>`+(i.score!=null?` · score ${f(i.score,2)}`:''):'No matching drawing · drawing-free analysis';
   if(i.status_counts)h+='<div class="counts">'+Object.entries(i.status_counts).map(([k,v])=>`<span class="pill">${v} ${k}</span>`).join('')+'</div>';
   h+='<div class="files">'+(i.image?`<a href="/files/${i.image}" target="_blank">Result image</a>`:'')+
    (i.report?`<a href="/files/${i.report}" target="_blank">Report</a>`:'')+`<a href="/files/${i.log}" target="_blank">Log</a></div>`;}
  $('insp').innerHTML=h;
 }else $('cap').classList.remove('show');
 lastCap=cap;}
async function poll(){if(busy)return;busy=true;
 try{const r=await fetch('/status',{cache:'no-store'});render(await r.json());}catch(e){$('camtxt').textContent='Server not reachable';}
 busy=false;}
setInterval(poll,120);poll();
$('retake').onclick=()=>fetch('/retake',{method:'POST'});
$('manual').onclick=()=>fetch('/capture',{method:'POST'});
$('feed').onerror=()=>setTimeout(()=>{$('feed').src='/stream.mjpg?'+Date.now()},1000);
</script></body></html>"""

PHONE_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<meta name="theme-color" content="#000"><title>Camera · Product Quality Check</title>
<style>
:root{--ok:#4cc38a;--warn:#f0a83a;--bad:#ef5f5f;--text:#f1f3f4;--muted:#b0b5ba}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{margin:0;height:100%;background:#000;color:var(--text);overflow:hidden;
font:15px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
#v,#o{position:fixed;inset:0;width:100%;height:100%;object-fit:contain}
#o{pointer-events:none}
.top{position:fixed;top:0;left:0;right:0;display:flex;justify-content:space-between;align-items:center;
padding:12px 16px;background:linear-gradient(#000a,#0000);font-size:13px;color:var(--muted)}
.top b{color:var(--text);font-weight:600}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#666;margin-right:6px}
.live .dot{background:var(--ok)}
.bottom{position:fixed;left:0;right:0;bottom:0;padding:16px 16px calc(18px + env(safe-area-inset-bottom));
background:linear-gradient(#0000,#000c 40%);display:flex;flex-direction:column;align-items:center;gap:10px}
.banner{display:flex;align-items:center;gap:10px;padding:11px 18px;border-radius:16px;background:#111d;
border:1px solid #fff2;font-weight:500;text-align:center;max-width:100%}
.banner i{width:10px;height:10px;border-radius:50%;background:#777;flex:none}
.banner.ok{border-color:#4cc38a99}.banner.ok i{background:var(--ok)}
.banner.warn i{background:var(--warn)}.banner.bad i{background:var(--bad)}
.hint{color:var(--muted);font-size:13px;min-height:18px;text-align:center}
.bar{width:220px;height:3px;border-radius:2px;background:#ffffff22;overflow:hidden}
.bar div{height:100%;width:0;background:var(--ok);transition:width .15s linear}
.cap{display:none;width:100%;max-width:420px;background:#151819ee;border:1px solid #ffffff1a;border-radius:16px;
padding:12px 14px}
.cap.show{display:block;animation:in .25s ease-out}@keyframes in{from{opacity:0;transform:translateY(8px)}}
.cap .row{display:flex;justify-content:space-between;font-size:13px;color:var(--muted);margin-top:4px}
.cap .row b{color:var(--text);font-weight:500;font-variant-numeric:tabular-nums}
.cap h4{margin:0;font-size:14px;display:flex;justify-content:space-between}
.btns{display:flex;gap:8px;margin-top:10px}
button{flex:1;font:inherit;font-weight:600;padding:11px;border-radius:12px;border:1px solid #ffffff2a;
background:#ffffff14;color:var(--text)}
button.primary{background:var(--text);color:#111;border:none}
.flash{position:fixed;inset:0;background:#fff;opacity:0;pointer-events:none;transition:opacity .4s ease-out}
.flash.on{opacity:.6;transition:none}
#start{position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;
padding:24px;text-align:center;background:#0e1012}
#start p{color:var(--muted);max-width:340px;margin:0}
#start button{flex:none;width:220px}
</style></head><body>
<video id="v" playsinline muted autoplay></video><canvas id="o"></canvas><div class="flash" id="flash"></div>
<div class="top" id="top"><span><span class="dot"></span><b>Product Quality Check</b></span><span id="info">Starting…</span></div>
<div class="bottom">
<div class="cap" id="cap"><h4><span>Captured</span><span id="c-time" style="color:var(--muted);font-weight:400"></span></h4>
<div class="row"><span>Scale</span><b id="c-scale"></b></div>
<div class="row"><span>Tilt · alignment · sharpness</span><b id="c-q"></b></div>
<div class="row"><span>Inspection</span><b id="c-insp"></b></div>
<div class="btns"><button class="primary" id="retake">Retake</button></div></div>
<div class="banner" id="banner"><i></i><span id="msg">Starting camera…</span></div>
<div class="bar"><div id="hold"></div></div><div class="hint" id="hint"></div></div>
<div id="start"><b style="font-size:18px">Product Quality Check</b>
<p>Point the camera straight down at the A4 reference frame with the part on it. The photo is taken automatically.</p>
<button class="primary" id="go">Start camera</button><p id="err" style="color:var(--bad)"></p></div>
<canvas id="c" style="display:none"></canvas>
<script>
const $=id=>document.getElementById(id),v=$('v'),o=$('o'),c=$('c'),oc=o.getContext('2d');
const tone={partial:'warn',adjust:'warn',steady:'warn',ready:'ok',capturing:'ok',captured:'ok',disconnected:'bad'};
let st=null,lastCap=null,run=false,n=0,t0=performance.now(),fps=0,lock=null;
const f=(x,d=1)=>x==null?'–':Number(x).toFixed(d);
async function start(){
  $('err').textContent='';
  if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia){
    $('err').textContent='This page needs a secure address: open it as http://localhost:8000/phone (USB) or https://… (Wi-Fi).';return;}
  try{
    const s=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:'environment'},
      width:{ideal:1920},height:{ideal:1080},frameRate:{ideal:30}}});
    v.srcObject=s;await v.play();
    const tr=s.getVideoTracks()[0];
    try{await tr.applyConstraints({advanced:[{focusMode:'continuous'}]});}catch(e){}
    try{lock=await navigator.wakeLock.request('screen');}catch(e){}
    $('start').style.display='none';run=true;loop();
  }catch(e){$('err').textContent='Camera not available: '+(e.message||e.name)+'. Allow camera access and try again.';}
}
async function loop(){
  while(run){
    if(v.readyState<2||!v.videoWidth){await new Promise(r=>setTimeout(r,50));continue;}
    const w=v.videoWidth,h=v.videoHeight;if(c.width!==w){c.width=w;c.height=h;}
    c.getContext('2d').drawImage(v,0,0,w,h);
    const blob=await new Promise(r=>c.toBlob(r,'image/jpeg',0.92));
    try{const r=await fetch('/phone/frame',{method:'POST',body:blob,headers:{'Content-Type':'image/jpeg'}});
      st=await r.json();render(w,h);}
    catch(e){$('msg').textContent='Connection to the computer lost. Retrying…';$('banner').className='banner bad';
      await new Promise(r=>setTimeout(r,800));}
    n++;const now=performance.now();if(now-t0>1000){fps=n*1000/(now-t0);n=0;t0=now;}
  }
}
function render(w,h){
  $('top').className='top live';
  $('info').textContent=`${w}×${h} · ${f(fps,0)} fps`;
  $('banner').className='banner '+(tone[st.state]||'');$('msg').textContent=st.message||'';
  $('hint').textContent=st.error||st.hint||'';
  let hold=['ready','capturing','captured'].includes(st.state)?1:st.state==='steady'?st.hold||0:0;
  $('hold').style.width=(hold*100)+'%';
  // overlay: the detected frame, mapped like object-fit:contain
  const W=o.clientWidth*devicePixelRatio,H=o.clientHeight*devicePixelRatio;if(o.width!==W){o.width=W;o.height=H;}
  oc.clearRect(0,0,W,H);
  if(st.quad){const k=Math.min(W/w,H/h),ox=(W-w*k)/2,oy=(H-h*k)/2,P=st.quad.map(p=>[ox+p[0]*k,oy+p[1]*k]);
    const col=['ready','capturing','captured'].includes(st.state)?'#4cc38a':['adjust','partial','steady'].includes(st.state)?'#f0a83a':'#eeeeee';
    oc.save();oc.fillStyle='rgba(0,0,0,.4)';oc.beginPath();oc.rect(0,0,W,H);oc.moveTo(...P[0]);
    for(const p of P.slice(1))oc.lineTo(...p);oc.closePath();oc.fill('evenodd');
    oc.strokeStyle=col;oc.lineWidth=2.5*devicePixelRatio;oc.beginPath();oc.moveTo(...P[0]);
    for(const p of P.slice(1))oc.lineTo(...p);oc.closePath();oc.stroke();
    oc.lineWidth=6*devicePixelRatio;oc.lineCap='round';
    for(let i=0;i<4;i++){const p=P[i];for(const q of[P[(i+3)%4],P[(i+1)%4]]){const dx=q[0]-p[0],dy=q[1]-p[1],
      L=Math.hypot(dx,dy),s=Math.min(28*devicePixelRatio,L*.15);oc.beginPath();oc.moveTo(...p);
      oc.lineTo(p[0]+dx/L*s,p[1]+dy/L*s);oc.stroke();}}
    oc.restore();}
  const cap=st.capture;
  if(cap){if(!lastCap||lastCap.id!==cap.id){const fl=$('flash');fl.classList.add('on');
      if(navigator.vibrate)navigator.vibrate(60);requestAnimationFrame(()=>requestAnimationFrame(()=>fl.classList.remove('on')));}
    $('cap').classList.add('show');$('c-time').textContent=new Date(cap.timestamp).toLocaleTimeString();
    $('c-scale').textContent=`${f(cap.mm_per_px,4)} mm/px`;
    $('c-q').textContent=`${f(cap.tilt_deg,1)}° · ${f(cap.alignment_score,2)} · ${f(cap.sharpness_score,2)}`;
    const i=cap.inspection;
    $('c-insp').textContent=!i?'off':i.state==='running'||i.state==='queued'?i.state+'…':i.state==='failed'?'failed':
      (i.drawing||'no matching drawing')+(i.status_counts?' · '+Object.entries(i.status_counts).map(([k,v])=>v+' '+k).join(', '):'');
  }else $('cap').classList.remove('show');
  lastCap=cap;
}
$('go').onclick=start;
start();  // starts at once where the browser allows it; otherwise the button stays
$('retake').onclick=()=>fetch('/retake',{method:'POST'});
document.addEventListener('visibilitychange',async()=>{if(document.visibilityState==='visible'&&run){
  try{lock=await navigator.wakeLock.request('screen');}catch(e){}}});
</script></body></html>"""


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if path == "/status":
                with app.cond:
                    st = app.status
                return self._send(200, json.dumps(st, default=str).encode(), "application/json")
            if path == "/stream.mjpg":
                return self._stream()
            if path.startswith("/files/"):
                return self._file(path[len("/files/"):])
            if path == "/phone":
                return self._send(200, PHONE_PAGE.encode(), "text/html; charset=utf-8")
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if path == "/phone/frame":
                return self._phone_frame(body)
            if path == "/retake":
                app.retake()
            elif path == "/capture":
                app.manual_capture()
            else:
                return self._send(404, b"not found", "text/plain")
            return self._send(200, b'{"ok":true}', "application/json")

        def _phone_frame(self, body):
            """A frame from the phone's browser. Answers with the status of the newest processed frame, so the
            phone shows the guidance with one round trip of delay (the phone waits for it before sending more)."""
            if getattr(app.cam, "kind", None) != "phone":
                return self._send(409, b'{"error":"start capture.py with --source phone"}', "application/json")
            try:
                app.cam.push(body)
            except ValueError as e:
                return self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")
            with app.cond:  # wait (briefly) until this frame has been processed
                seq = app.cam.seq
                app.cond.wait_for(lambda: app.processed_seq >= seq, timeout=1.0)
                st = app.status
            gates = st.get("gates") or {}
            hold = 0.0
            m = re.search(r"held (\d+)/(\d+)", (gates.get("stability") or {}).get("detail", ""))
            if m:
                hold = min(1.0, int(m.group(1)) / max(int(m.group(2)), 1))
            out = {k: st.get(k) for k in ("state", "message", "hint", "error", "quad", "capture")}
            out["hold"] = hold
            return self._send(200, json.dumps(out, default=str).encode(), "application/json")

        def _file(self, rel):
            from urllib.parse import unquote
            root = os.path.realpath(app.save_dir)
            p = os.path.realpath(os.path.join(root, unquote(rel)))
            if not p.startswith(root + os.sep) or not os.path.isfile(p):
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(p)[0] or "application/octet-stream"
            if p.endswith((".log", ".json")):
                ctype = "text/plain; charset=utf-8" if p.endswith(".log") else "application/json"
            with open(p, "rb") as f:
                return self._send(200, f.read(), ctype)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            seq = -1
            try:
                while True:
                    with app.cond:
                        if app.jseq == seq:
                            app.cond.wait(2.0)
                        jpg, seq = app.jpeg, app.jseq
                    if jpg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                     str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            self.close_connection = True

    return Handler


def build_parser():
    ap = argparse.ArgumentParser(
        description="Live camera capture with an A4 (210 x 297 mm) reference frame: detects the frame, checks the "
                    "camera alignment, waits until the image is steady and sharp, captures automatically, "
                    "rectifies the frame's plane, calibrates mm per pixel and runs the inspection on the result.",
        epilog="Open the printed URL in a browser. Captures go to --save-dir/<id>/. "
               "Phone as camera: an IP-camera app gives a stream URL, e.g. --source http://PHONE_IP:8080/video")
    ap.add_argument("--source", default="0",
                    help="'phone' (the phone's own camera through its web browser, no app: open /phone on the phone), "
                         "camera index (default 0), stream URL, video file, or a still image")
    ap.add_argument("--config", default=None, help="JSON overriding partvision/config.py (section 'camera' etc.)")
    ap.add_argument("--intrinsics", default=None, help="lens calibration JSON from calibrate_camera.py")
    ap.add_argument("--drawings", nargs="*", default=None,
                    help="drawings to match each capture against (default: dxf_files/ if it exists)")
    ap.add_argument("--drawing", default=None, help="check every capture against this one drawing")
    ap.add_argument("--no-inspect", action="store_true", help="only capture and calibrate; do not run the inspection")
    ap.add_argument("--tol", type=float, default=None, help="tolerance in mm (default: config)")
    ap.add_argument("--part-height-mm", type=float, default=None,
                    help="height of the measured face above the reference frame's plane (scale correction)")
    ap.add_argument("--save-dir", default=None, help="capture folder (default: config camera.save_dir)")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to open the UI from another device")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--https", action="store_true",
                    help="serve over HTTPS with a self-signed certificate: needed for the phone camera over Wi-Fi "
                         "(the browser allows the camera only on https:// or localhost)")
    ap.add_argument("--headless", action="store_true", help="no web UI: print status changes, exit after --max-captures")
    ap.add_argument("--max-captures", type=int, default=1, help="with --headless: stop after this many captures")
    ap.add_argument("--timeout", type=float, default=0, help="with --headless: give up after this many seconds")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
    C = config.load(args.config)
    cfg = dict(C["camera"])
    if args.part_height_mm is not None:
        cfg["part_height_mm"] = args.part_height_mm
    intr_path = args.intrinsics or cfg.get("intrinsics")
    intr = load_intrinsics(intr_path) if intr_path else None
    inspect = None
    if not args.no_inspect:
        drawings = args.drawings if args.drawings is not None else (["dxf_files"] if os.path.isdir("dxf_files") else [])
        inspect = dict(drawings=drawings, drawing=args.drawing, tol=args.tol, config_path=args.config)
    app = CameraApp(args.source, cfg, intr, args.save_dir or cfg["save_dir"], inspect)
    app.start()
    if args.headless:
        return _headless(app, args)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    srv.daemon_threads = True
    scheme = "http"
    if args.https:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(*_self_signed_cert())
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
        scheme = "https"
    host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"Camera UI: {scheme}://{host}:{args.port}/   (Ctrl+C to stop)", flush=True)
    if str(args.source) == "phone":
        print(f"Phone:     open {scheme}://{'localhost' if not args.https else _lan_ip()}:{args.port}/phone "
              "in the phone's browser" + ("" if args.https else "  (USB: run 'adb reverse tcp:%d tcp:%d' first)"
                                          % (args.port, args.port)), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
        srv.server_close()
    return 0


def _self_signed_cert():
    """A self-signed certificate for --https (created once with openssl, kept in ~/.cache/partvision)."""
    d = os.path.join(os.path.expanduser("~"), ".cache", "partvision")
    crt, key = os.path.join(d, "cert.pem"), os.path.join(d, "key.pem")
    if not (os.path.isfile(crt) and os.path.isfile(key)):
        os.makedirs(d, exist_ok=True)
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", crt,
                        "-days", "3650", "-subj", "/CN=partvision-camera"], check=True, capture_output=True)
    return crt, key


def _lan_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packet is sent; picks the outgoing interface
        return s.getsockname()[0]
    except OSError:
        return "THIS_COMPUTER_IP"
    finally:
        s.close()


def _headless(app, args):
    t0, last, done = time.monotonic(), None, set()
    try:
        while True:
            with app.cond:
                app.cond.wait(0.5)
                st = app.status
            key = (st.get("state"), st.get("message"), st.get("hint"))
            if key != last:
                print(f"[{time.monotonic() - t0:6.2f}s] {st.get('state'):<12} {st.get('message')}"
                      + (f"  ({st['hint']})" if st.get("hint") else ""), flush=True)
                last = key
            cap = app.last_capture
            if cap and cap["id"] not in done and (not cap.get("inspection") or cap["inspection"]["state"] != "running"):
                done.add(cap["id"])
                print(json.dumps(cap, indent=1, default=str), flush=True)
                if len(done) >= args.max_captures:
                    return 0
            if args.timeout and time.monotonic() - t0 > args.timeout:
                print("timeout", flush=True)
                return 1
    finally:
        app.stop()
