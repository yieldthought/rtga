"""A dependency-free, portable viewer for recorded RTGA decision traces."""

from __future__ import annotations

import html
import json
from pathlib import Path
import re


def write_viewer(trace: dict, output: str | Path) -> Path:
    """Write a standalone HTML replay for a schema-v1 trace and return its path.

    Frames represent the observed state at a decision, before its recorded action.
    Optional paths and diagnostics can be absent; missing values are never plotted
    as zero. The trace must contain only JSON-serializable, finite values.
    """
    if not isinstance(trace, dict) or not isinstance(trace.get("frames"), list):
        raise ValueError("A trace must be a dictionary containing a frames list")
    title = str((trace.get("metadata") or {}).get("title") or "RTGA trace")
    payload = json.dumps(trace, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # JSON lives in a script raw-text element, where HTML entities do not escape
    # a closing tag. Escape the characters themselves, retaining JSON round trips.
    for character, escaped in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"),
                               ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        payload = payload.replace(character, escaped)
    replacements = {"TITLE": html.escape(title, quote=True), "PAYLOAD": payload}
    document = re.sub(r"__TRACE_(TITLE|PAYLOAD)__", lambda match: replacements[match[1]], _HTML)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TRACE_TITLE__</title>
<style>
:root { color-scheme: dark; --bg:#11161c; --panel:#181f27; --line:#303b48; --ink:#e2e9ef; --muted:#a1b0bf; --blue:#89b9e8; --mint:#87cfb2; --pink:#e8a4b5; --gold:#e2c482; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:13px/1.45 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
main { max-width:1500px; margin:0 auto; padding:20px 24px 26px; }
h1 { font-size:22px; line-height:1.25; font-weight:620; margin:0 0 5px; overflow-wrap:anywhere; }
h2 { font-size:13px; font-weight:650; margin:0; }
p { margin:0; }
.muted, .note { color:var(--muted); }
.mono, output, .value { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-variant-numeric:tabular-nums; }
header { margin-bottom:16px; }
.meta { display:flex; flex-wrap:wrap; gap:5px 18px; margin-top:6px; }
.controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap; padding:11px 0; border-block:1px solid var(--line); }
button, select { border:1px solid #485767; border-radius:4px; background:#212b35; color:var(--ink); padding:6px 10px; font:inherit; }
button { cursor:pointer; min-width:38px; }
button:hover { border-color:var(--blue); }
button:disabled { opacity:.4; cursor:default; }
button:focus-visible, select:focus-visible, input:focus-visible, summary:focus-visible { outline:2px solid var(--blue); outline-offset:3px; }
#play { min-width:68px; }
.seek { flex:1; min-width:150px; display:flex; align-items:center; gap:8px; }
input[type=range] { width:100%; accent-color:var(--blue); cursor:pointer; }
#position { min-width:100px; text-align:right; }
.stats { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); border-bottom:1px solid var(--line); padding:12px 0 14px; gap:14px; }
.stat .label { color:var(--muted); font-size:11px; margin-bottom:3px; }
.value { font-size:17px; overflow-wrap:anywhere; }
.views { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:22px; padding-top:18px; }
.section-head { display:flex; justify-content:space-between; align-items:baseline; flex-wrap:wrap; gap:4px 12px; margin-bottom:7px; }
.canvas-wrap { background:var(--panel); border:1px solid var(--line); border-radius:4px; overflow:hidden; }
canvas { display:block; width:100%; }
#world, #population { height:405px; }
#timeline { height:236px; cursor:crosshair; }
.legend { display:flex; flex-wrap:wrap; gap:4px 14px; min-height:25px; margin:8px 0 5px; color:var(--muted); font-size:11px; }
.key { display:inline-block; height:8px; width:16px; border-radius:1px; margin-right:5px; vertical-align:1px; background:var(--key); }
.timeline-section { margin-top:18px; }
.note { font-size:11px; margin-top:6px; }
details { border-top:1px solid var(--line); margin-top:18px; padding-top:10px; }
summary { color:var(--muted); cursor:pointer; }
pre { white-space:pre-wrap; overflow-wrap:anywhere; color:var(--muted); max-height:260px; overflow:auto; font-size:11px; }
.empty { padding:40px 0; color:var(--muted); }
@media(max-width:760px) { main { padding:14px 12px 20px; } .views { grid-template-columns:1fr; gap:18px; } .stats { grid-template-columns:repeat(3,minmax(0,1fr)); } #world { height:350px; } #population { height:320px; } .seek { order:2; flex-basis:100%; } #position { margin-left:auto; } }
@media(prefers-reduced-motion:reduce) { * { scroll-behavior:auto; } }
</style>
</head>
<body><main>
<header><h1>__TRACE_TITLE__</h1><p class="muted">Recorded decisions · state before action · no live simulation</p><div class="meta mono" id="metadata"></div></header>
<div id="empty" class="empty" hidden>No frames were recorded in this trace.</div>
<div id="replay">
<div class="controls">
<button id="previous" aria-label="Previous frame" title="Previous frame (left arrow)">←</button>
<button id="play" aria-label="Play trace">Play</button>
<button id="next" aria-label="Next frame" title="Next frame (right arrow)">→</button>
<label for="speed">Speed</label><select id="speed"><option value="2">2 frames/s</option><option value="5">5 frames/s</option><option value="10" selected>10 frames/s</option><option value="20">20 frames/s</option><option value="40">40 frames/s</option></select>
<label class="seek" for="seek"><span class="muted">Frame</span><input id="seek" type="range" min="0" value="0" step="1" aria-label="Replay frame"></label>
<output id="position" for="seek"></output>
</div>
<div class="stats">
<div class="stat"><div class="label">Recorded step</div><div class="value" id="step">—</div></div>
<div class="stat"><div class="label">Action applied</div><div class="value" id="action">—</div></div>
<div class="stat"><div class="label">Plan objective</div><div class="value" id="objective">—</div></div>
<div class="stat"><div class="label">Prediction error</div><div class="value" id="error">—</div></div>
<div class="stat"><div class="label">Disagreement</div><div class="value" id="disagreement">—</div></div>
<div class="stat"><div class="label">Decision latency</div><div class="value" id="latency">—</div></div>
</div>
<div class="views">
<section><div class="section-head"><h2>World and predicted futures</h2><span id="world-label" class="muted mono"></span></div>
<div class="canvas-wrap"><canvas id="world" role="img" aria-label="Actual trajectory and recorded predicted paths in the environment"></canvas></div>
<div class="legend"><span><i class="key" style="--key:#dbe6ef"></i>Actual trail</span><span><i class="key" style="--key:#e2c482"></i>Selected future</span><span><i class="key" style="--key:#89b9e8"></i>Other futures</span><span><i class="key" style="--key:#87cfb2"></i>Switch</span><span><i class="key" style="--key:#e8a4b5"></i>Noise source</span></div>
<p class="note">Futures are predictions stored at this decision. The bright dot is the actual observed position.</p></section>
<section><div class="section-head"><h2>Population · future actions</h2><span id="population-label" class="muted mono"></span></div>
<div class="canvas-wrap"><canvas id="population" role="img" aria-label="Recorded action plans by row and future time offset by column"></canvas></div>
<div class="legend" id="action-legend"></div><p class="note">Each row is one recorded genome; the outlined column is its proposed next action. Row order follows the trace.</p></section>
</div>
<section class="timeline-section"><div class="section-head"><h2>Diagnostics over recorded steps</h2><span class="muted">Click or drag to seek</span></div>
<div class="canvas-wrap"><canvas id="timeline" role="img" aria-label="Prediction error, disagreement, and decision latency over recorded steps"></canvas></div>
<p class="note">Each diagnostic has its own vertical scale. Gaps mean unrecorded values; the cursor marks the current decision.</p></section>
</div>
<details><summary>Trace metadata, configuration and current frame metrics</summary><pre id="details"></pre></details>
</main>
<script type="application/json" id="trace-data">__TRACE_PAYLOAD__</script>
<script>
'use strict';
const trace = JSON.parse(document.getElementById('trace-data').textContent);
const frames = trace.frames || [], meta = trace.metadata || {}, env = trace.environment || {};
const byId = id => document.getElementById(id);
const finite = x => typeof x === 'number' && Number.isFinite(x);
const val = (x, digits=3) => finite(x) ? (Math.abs(x)>0 && Math.abs(x)<.001 ? x.toExponential(2) : x.toLocaleString('en-US',{maximumFractionDigits:digits})) : '—';
const actionColors = ['#637386','#89b9e8','#e8a4b5','#87cfb2','#e2c482','#bbabed','#e9ae82','#81cbd1'];
const defaultActions = ['no-op','left','right','up','down','interact'];
const names = meta.action_names || env.action_names || defaultActions;
const actionName = a => names[a] !== undefined ? String(names[a]) : String(a);
const color = a => actionColors[((Number(a)||0)%actionColors.length+actionColors.length)%actionColors.length];
let current=0, playing=false, previousTime=0, dragging=false;
const setText = (id, text) => { byId(id).textContent=text; };
const metaItems = [['method',meta.method],['seed',meta.seed],['variant',env.variant],['frames',frames.length]];
for (const [key,value] of metaItems) if (value !== undefined && value !== null) { const el=document.createElement('span'); el.textContent=key+' '+value; byId('metadata').append(el); }
const seenActions = new Set();
for (const f of frames) { if (finite(f.action)) seenActions.add(f.action); for (const p of f.plans || []) for (const a of p) if (finite(a)) seenActions.add(a); }
for (const a of [...seenActions].sort((a,b)=>a-b)) { const el=document.createElement('span'), key=document.createElement('i'); key.className='key'; key.style.setProperty('--key',color(a)); el.append(key,document.createTextNode(a+' '+actionName(a))); byId('action-legend').append(el); }
byId('seek').max=Math.max(0,frames.length-1);
if (!frames.length) { byId('empty').hidden=false; byId('replay').hidden=true; }

function context(id) {
  const canvas=byId(id), box=canvas.getBoundingClientRect(), dpr=Math.min(2,window.devicePixelRatio||1);
  const width=Math.max(1,box.width), height=Math.max(1,box.height);
  canvas.width=Math.round(width*dpr); canvas.height=Math.round(height*dpr);
  const c=canvas.getContext('2d'); c.setTransform(dpr,0,0,dpr,0,0); c.font='11px ui-monospace, monospace';
  return {c,w:width,h:height};
}
function line(c, points, xy, stroke, width, alpha=1) {
  c.beginPath(); let started=false;
  for (const p of points || []) { if (!Array.isArray(p) || !finite(p[0]) || !finite(p[1])) { started=false; continue; } const [x,y]=xy(p); if (started) c.lineTo(x,y); else c.moveTo(x,y); started=true; }
  c.strokeStyle=stroke; c.lineWidth=width; c.globalAlpha=alpha; c.stroke(); c.globalAlpha=1;
}
function world(f) {
  const {c,w,h}=context('world'), ew=finite(env.width)&&env.width>0?env.width:1, eh=finite(env.height)&&env.height>0?env.height:1;
  const scale=Math.min((w-68)/ew,(h-49)/eh), ww=scale*ew, hh=scale*eh, left=(w-ww)/2+8, top=(h-hh)/2-7;
  const xy=p=>[left+p[0]*scale,top+hh-p[1]*scale];
  c.strokeStyle='#2b3743'; c.lineWidth=1; c.fillStyle='#8f9fad';
  for (let n=0;n<=4;n++) { const x=left+ww*n/4,y=top+hh-hh*n/4; c.beginPath(); c.moveTo(x,top); c.lineTo(x,top+hh); c.moveTo(left,y); c.lineTo(left+ww,y); c.stroke(); c.textAlign='center'; c.fillText(val(ew*n/4,2),x,top+hh+17); c.textAlign='right'; c.fillText(val(eh*n/4,2),left-9,y+4); }
  c.textAlign='left'; c.fillText('y',left-23,top-5); c.textAlign='right'; c.fillText('x',left+ww+15,top+hh+17);
  c.save(); c.beginPath(); c.rect(left,top,ww,hh); c.clip();
  const paths=f.paths || [];
  paths.forEach((p,i)=>line(c,p,xy,actionColors[(i+1)%actionColors.length],1.2,.4));
  line(c,frames.slice(0,current+1).map(a=>a.state),xy,'#dbe6ef',1.8,.7);
  line(c,f.selected_path,xy,'#e2c482',2.5,.95);
  const state=f.state || [], door=!!state[4];
  if (finite(env.wall_x)) {
    const dy=Array.isArray(env.door_y)?env.door_y:[0,0];
    const wallWidth=finite(env.wall_half_width)?2*env.wall_half_width*scale:5;
    line(c,[[env.wall_x,0],[env.wall_x,dy[0]]],xy,'#a7b1bb',wallWidth);
    line(c,[[env.wall_x,dy[1]],[env.wall_x,eh]],xy,'#a7b1bb',wallWidth);
    c.setLineDash(door?[3,5]:[]); line(c,[[env.wall_x,dy[0]],[env.wall_x,dy[1]]],xy,door?'#87cfb2':'#e2c482',door?1.5:wallWidth); c.setLineDash([]);
  }
  function marker(p,stroke,label,square=false) { if (!Array.isArray(p)||!finite(p[0])||!finite(p[1])) return; const [x,y]=xy(p); c.strokeStyle=stroke; c.lineWidth=1.5; c.fillStyle='#181f27'; c.beginPath(); if(square)c.rect(x-5,y-5,10,10); else c.arc(x,y,6,0,2*Math.PI); c.fill(); c.stroke(); c.fillStyle=stroke; c.textAlign=x>w*.72?'right':'left'; c.fillText(label,x+(x>w*.72?-10:10),y-8); }
  marker(env.switch,'#87cfb2','switch',true); marker(env.noise_source,'#e8a4b5','noise');
  if(finite(state[0])&&finite(state[1])) { const [x,y]=xy(state); c.beginPath(); c.arc(x,y,finite(env.radius)?env.radius*scale:5,0,2*Math.PI); c.fillStyle='#f1f6f9'; c.fill(); c.strokeStyle='#11161c'; c.lineWidth=1.5; c.stroke(); }
  c.restore(); c.strokeStyle='#738292'; c.lineWidth=1.4; c.strokeRect(left,top,ww,hh);
  setText('world-label',finite(state[4])?'door '+(door?'open':'closed'):'');
}
function population(f) {
  const {c,w,h}=context('population'), plans=f.plans || [], cols=Math.max(1,...plans.map(p=>p.length));
  setText('population-label',plans.length+' plans · '+(plans.length?cols:0)+' steps');
  if (!plans.length) { c.fillStyle='#a1b0bf'; c.fillText('No population recorded at this decision.',18,30); return; }
  const left=40,top=30,right=12,bottom=28,cw=(w-left-right)/cols,rh=(h-top-bottom)/plans.length;
  plans.forEach((p,i)=>{ p.forEach((a,j)=>{ c.fillStyle=color(a); c.fillRect(left+j*cw+.4,top+i*rh+.6,Math.max(.5,cw-.8),Math.max(.5,rh-1.2)); }); if(plans.length<=20||i%2===0||i===plans.length-1){ c.textAlign='right'; c.fillStyle='#a1b0bf'; c.fillText(String(i+1),left-9,top+(i+.5)*rh+4); } });
  c.strokeStyle='#f1f6f9'; c.lineWidth=1.3; c.strokeRect(left-.5,top-.5,cw+1,rh*plans.length+1);
  c.fillStyle='#a1b0bf'; c.textAlign='left'; c.fillText('row',5,17); c.fillText('future offset',left,17);
  const stride=Math.max(1,Math.ceil(cols/Math.max(3,(w-65)/36)));
  c.textAlign='center'; for(let j=0;j<cols;j+=stride)c.fillText('+'+j,left+(j+.5)*cw,h-9);
}
function timeline() {
  const {c,w,h}=context('timeline'), left=w<520?93:130,right=16,top=18,bottom=25,row=(h-top-bottom)/3;
  const defs=[['prediction_error','Prediction error','#e8a4b5'],['disagreement','Disagreement','#87cfb2'],['latency_ms','Latency · ms','#89b9e8']];
  const steps=frames.map((f,i)=>finite(f.step)?f.step:i), start=Math.min(...steps),end=Math.max(...steps),span=Math.max(1,end-start);
  const xx=i=>left+(steps[i]-start)/span*(w-left-right);
  defs.forEach(([key,label,stroke],r)=>{
    const y=top+r*row,values=frames.map(f=>f[key]).filter(finite),maximum=values.length?Math.max(0,...values):0,minimum=values.length?Math.min(0,...values):0,range=maximum-minimum||1;
    const yy=v=>y+row-13-(v-minimum)/range*(row-22);
    c.fillStyle=stroke;c.textAlign='left';c.font='11px ui-sans-serif, system-ui';c.fillText(label,10,y+row/2);
    c.font='10px ui-monospace, monospace';c.fillStyle='#98a8b7';c.textAlign='right';c.fillText(values.length?val(maximum,2):'—',left-8,y+8);c.fillText(values.length?val(minimum,2):'—',left-8,y+row-11);
    c.strokeStyle='#303b48';c.lineWidth=1;c.beginPath();c.moveTo(left,y+row-13);c.lineTo(w-right,y+row-13);c.stroke();
    let previous=false;c.beginPath();frames.forEach((f,i)=>{if(!finite(f[key])){previous=false;return;}if(previous)c.lineTo(xx(i),yy(f[key]));else c.moveTo(xx(i),yy(f[key]));previous=true;});c.strokeStyle=stroke;c.lineWidth=1.5;c.stroke();
    frames.forEach((f,i)=>{if(finite(f[key])&&!finite(frames[i-1]?.[key])&&!finite(frames[i+1]?.[key])){c.beginPath();c.arc(xx(i),yy(f[key]),1.8,0,2*Math.PI);c.fillStyle=stroke;c.fill();}});
    if(finite(frames[current]?.[key])) {c.beginPath();c.arc(xx(current),yy(frames[current][key]),3,0,2*Math.PI);c.fillStyle=stroke;c.fill();}
  });
  c.strokeStyle='#dbe6ef';c.globalAlpha=.5;c.beginPath();c.moveTo(xx(current),8);c.lineTo(xx(current),h-bottom);c.stroke();c.globalAlpha=1;
  c.fillStyle='#a1b0bf';c.textAlign='center';for(let n=0;n<=4;n++) { const step=start+(end-start)*n/4; c.fillText(val(step,0),left+(w-left-right)*n/4,h-9); }
  c.textAlign='left';c.fillText('step',10,h-9);
}
function draw() {
  if(!frames.length){setText('details',JSON.stringify(meta,null,2));return;}
  const f=frames[current];
  setText('position',(current+1)+' / '+frames.length);byId('seek').value=current;
  setText('step',finite(f.step)?f.step:current);setText('action',finite(f.action)?f.action+' '+actionName(f.action):'—');
  setText('objective',val(f.objective));setText('error',val(f.prediction_error));setText('disagreement',val(f.disagreement));setText('latency',finite(f.latency_ms)?val(f.latency_ms,2)+' ms':'—');
  byId('previous').disabled=current===0;byId('next').disabled=current===frames.length-1;
  setText('details',JSON.stringify({metadata:meta,environment:env,current_frame_metrics:f.metrics||{}},null,2));
  world(f);population(f);timeline();
}
function pause(){playing=false;setText('play','Play');byId('play').setAttribute('aria-label','Play trace');}
function seek(index){current=Math.max(0,Math.min(frames.length-1,index));draw();}
byId('previous').onclick=()=>{pause();seek(current-1);};byId('next').onclick=()=>{pause();seek(current+1);};
byId('seek').oninput=e=>{pause();seek(Number(e.target.value));};
byId('play').onclick=()=>{if(!frames.length)return;if(playing){pause();return;}if(current===frames.length-1)seek(0);playing=true;previousTime=performance.now();setText('play','Pause');byId('play').setAttribute('aria-label','Pause trace');};
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON','SUMMARY'].includes(e.target.tagName))return;if(e.code==='Space'){e.preventDefault();byId('play').click();}if(e.code==='ArrowLeft'){e.preventDefault();pause();seek(current-1);}if(e.code==='ArrowRight'){e.preventDefault();pause();seek(current+1);}});
function seekTimeline(e){const box=byId('timeline').getBoundingClientRect(),left=box.width<520?93:130,ratio=Math.max(0,Math.min(1,(e.clientX-box.left-left)/(box.width-left-16))),steps=frames.map((f,i)=>finite(f.step)?f.step:i),target=Math.min(...steps)+ratio*(Math.max(...steps)-Math.min(...steps));let best=0;for(let i=1;i<steps.length;i++)if(Math.abs(steps[i]-target)<Math.abs(steps[best]-target))best=i;pause();seek(best);}
byId('timeline').addEventListener('pointerdown',e=>{dragging=true;byId('timeline').setPointerCapture(e.pointerId);seekTimeline(e);});
byId('timeline').addEventListener('pointermove',e=>{if(dragging)seekTimeline(e);});
byId('timeline').addEventListener('pointerup',()=>{dragging=false;});byId('timeline').addEventListener('pointercancel',()=>{dragging=false;});
function tick(now){if(playing&&now-previousTime>=1000/Number(byId('speed').value)){const advance=Math.floor((now-previousTime)*Number(byId('speed').value)/1000);previousTime=now;seek(current+advance);if(current===frames.length-1)pause();}requestAnimationFrame(tick);}
new ResizeObserver(()=>draw()).observe(byId('replay'));
draw();requestAnimationFrame(tick);
</script></body></html>
"""
