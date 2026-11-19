"""Live trace page: every model call, as it happens.

    python packages/agent-sdk-fast-vision/trace_page.py     # serves runs/fast-vision on http://127.0.0.1:8767
    python packages/agent-sdk-fast-vision/trace_page.py /path/to/run --port 8770

Reads what the fast vision loop writes (<run>/trace.jsonl and <run>/trace/<call>/),
so it runs beside the agent and survives agent restarts. Localhost only.

For each call the page shows the 6-frame image that was sent and when, the
exact request and response bodies, and what was applied to the game.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

from trace_cli import load

CALL_DIR = re.compile(r"^\d{6}$")
FILES = {"sheet.png": "image/png", "request.json": "application/json",
         "response.json": "application/json", "outcome.json": "application/json"}


def inflight(run: Path, finished: set, limit: int = 8) -> List[Dict[str, Any]]:
    """Calls whose folder exists but whose outcome has not been written yet."""
    root = run / "trace"
    if not root.is_dir():
        return []
    out = []
    for folder in sorted((p for p in root.iterdir() if p.is_dir() and CALL_DIR.match(p.name)),
                         reverse=True)[:limit]:
        call = int(folder.name)
        if call in finished or (folder / "outcome.json").exists():
            continue
        sent = (folder / "request.json").stat().st_mtime if (folder / "request.json").exists() else None
        started = sent or folder.stat().st_mtime
        # Longer than any model timeout: the agent stopped before this call finished.
        outcome = "in flight" if time.time() - started < 30 else "no outcome: agent stopped before this call finished"
        out.append({"call": call, "dir": f"trace/{folder.name}",
                    "sent": time.strftime("%H:%M:%S", time.localtime(sent)) + f".{int((sent % 1) * 1000):03d}"
                    if sent else "", "outcome": outcome})
    return out


def system_prompt(run: Path, rows: List[Dict[str, Any]]) -> str:
    for row in reversed(rows):
        path = run / row.get("dir", "") / "request.json"
        if path.exists():
            try:
                for message in json.loads(path.read_bytes()).get("messages", []):
                    if message.get("role") == "system" and isinstance(message.get("content"), str):
                        return message["content"]
            except ValueError:
                pass
    return ""


def make_handler(run: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(url.query, keep_blank_values=True)   # so a bare ?elide counts
            if url.path in ("/", "/index.html"):
                return self.send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if url.path == "/api/calls":
                rows = load(run)
                since = int((query.get("since") or ["0"])[0] or 0)
                limit = int((query.get("limit") or ["80"])[0] or 80)
                fresh = [r for r in rows if r.get("call", 0) > since][-limit:]
                payload = {
                    "run": str(run),
                    "total": len(rows),
                    "calls": fresh,
                    "inflight": inflight(run, {r.get("call") for r in rows}),
                    "prompt": system_prompt(run, rows) if since == 0 else None,
                    "now": time.time(),
                }
                return self.send(200, json.dumps(payload).encode("utf-8"), "application/json")
            match = re.match(r"^/trace/(\d{6})/([a-z.]+)$", url.path)
            if match and match.group(2) in FILES:
                path = run / "trace" / match.group(1) / match.group(2)
                if not path.is_file():
                    return self.send(404, b"pruned or not written yet", "text/plain")
                data = path.read_bytes()
                if match.group(2) == "request.json" and "elide" in query:
                    # Same body, image bytes shortened so it can be read in a browser tab.
                    data = re.sub(rb'("url":\s*"data:image/[a-z]+;base64,)([A-Za-z0-9+/=]{80})[A-Za-z0-9+/=]+',
                                  lambda m: m.group(1) + m.group(2) + b"...<" +
                                  str(len(m.group(0)) - len(m.group(1))).encode() + b" base64 chars>", data)
                return self.send(200, data, FILES[match.group(2)])
            return self.send(404, b"not found", "text/plain")

    return Handler


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent Call Trace</title>
<style>
:root{
  --bg:#f6f7f9;--panel:#fff;--ink:#14161a;--muted:#5d6675;--line:#e3e6eb;
  --ok:#13804b;--ok-bg:#e5f5ec;--drop:#6b7280;--drop-bg:#eef0f3;--bad:#c0362c;--bad-bg:#fbe9e7;
  --fly:#2f5bd3;--fly-bg:#e8eefc;--code:#f2f4f7;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0f1115;--panel:#171a20;--ink:#e8eaee;--muted:#9aa3b2;--line:#2a2f38;
  --ok:#4ccf8a;--ok-bg:#12301f;--drop:#9aa3b2;--drop-bg:#232831;--bad:#ff7b6e;--bad-bg:#3a1b18;
  --fly:#8aa8ff;--fly-bg:#1a2442;--code:#1f232b;}}
:root[data-theme="dark"]{
  --bg:#0f1115;--panel:#171a20;--ink:#e8eaee;--muted:#9aa3b2;--line:#2a2f38;
  --ok:#4ccf8a;--ok-bg:#12301f;--drop:#9aa3b2;--drop-bg:#232831;--bad:#ff7b6e;--bad-bg:#3a1b18;
  --fly:#8aa8ff;--fly-bg:#1a2442;--code:#1f232b;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px}
.bar{max-width:1280px;margin:0 auto;display:flex;flex-wrap:wrap;align-items:center;gap:10px 18px}
h1{font-size:17px;margin:0;font-weight:650}
.live{display:inline-flex;align-items:center;gap:6px;color:var(--muted);font-size:13px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--ok)}
.dot.off{background:var(--bad)}
.stats{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--muted)}
.stats b{color:var(--ink);font-variant-numeric:tabular-nums}
.spacer{flex:1}
button,select{font:inherit;font-size:13px;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer}
main{max-width:1280px;margin:0 auto;padding:16px}
details.prompt{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin-bottom:14px}
details.prompt summary{cursor:pointer;color:var(--muted)}
pre{margin:8px 0 0;white-space:pre-wrap;word-break:break-word;background:var(--code);border-radius:6px;padding:10px;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
.call{background:var(--panel);border:1px solid var(--line);border-radius:10px;margin-bottom:14px;display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);overflow:hidden}
.call.new{animation:flash 1.2s ease-out}
@keyframes flash{from{box-shadow:0 0 0 3px var(--fly)}to{box-shadow:0 0 0 0 transparent}}
.shot{background:#000;display:flex;align-items:center;justify-content:center;min-height:120px}
.shot img{width:100%;height:auto;display:block;cursor:zoom-in}
.body{padding:12px 14px;min-width:0}
.head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px}
.id{font-weight:650;font-variant-numeric:tabular-nums}
.badge{font-size:12px;font-weight:600;border-radius:999px;padding:2px 9px}
.applied{color:var(--ok);background:var(--ok-bg)}
.dropped{color:var(--drop);background:var(--drop-bg)}
.error{color:var(--bad);background:var(--bad-bg)}
.inflight{color:var(--fly);background:var(--fly-bg)}
.key{font:600 13px ui-monospace,Menlo,monospace;border:1px solid var(--line);border-radius:5px;padding:1px 6px}
.steps{display:grid;grid-template-columns:auto 1fr;gap:4px 10px;font-size:13px}
.steps .n{color:var(--muted);font-variant-numeric:tabular-nums}
.t{font-variant-numeric:tabular-nums;font-family:ui-monospace,Menlo,monospace;font-size:12.5px}
.muted{color:var(--muted)}
.frames{font-size:12px;color:var(--muted);margin:2px 0 4px;display:flex;flex-wrap:wrap;gap:2px 10px}
.frames span b{color:var(--ink);font-weight:600}
.out{font:12px/1.5 ui-monospace,Menlo,monospace;background:var(--code);border-radius:6px;padding:6px 8px;word-break:break-word;margin-top:2px}
a{color:var(--fly);text-decoration:none}
a:hover{text-decoration:underline}
.links{display:flex;gap:12px;font-size:12.5px;flex-wrap:wrap}
.empty{color:var(--muted);text-align:center;padding:60px 16px}
dialog{border:none;padding:0;background:transparent;max-width:96vw}
dialog img{max-width:96vw;max-height:92vh;display:block}
dialog::backdrop{background:rgba(0,0,0,.75)}
@media (max-width:820px){.call{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><div class="bar">
  <h1>Agent call trace</h1>
  <span class="live"><span class="dot" id="dot"></span><span id="liveText">connecting</span></span>
  <div class="stats" id="stats"></div>
  <span class="spacer"></span>
  <select id="filter" aria-label="Show">
    <option value="all">All calls</option><option value="applied">Applied</option>
    <option value="dropped">Dropped / no key</option><option value="error">Errors</option>
  </select>
  <button id="pause">Pause</button>
</div></header>
<main>
  <details class="prompt" id="promptBox" hidden><summary>System prompt sent with every call</summary><pre id="prompt"></pre></details>
  <div id="list"><div class="empty">Waiting for the first model call.<br>Start the runner with policy_fast_vision.py; calls appear here as they finish.</div></div>
</main>
<dialog id="zoom"><img alt="Sheet sent to the model"></dialog>
<script>
const $ = (s) => document.querySelector(s);
let since = 0, paused = false, rows = [], inflight = [], lastOk = 0, firstLoad = true;
const MAX = 150;
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const kind = (o) => o === "applied" ? "applied" : o === "in flight" ? "inflight" : o === "error" || (o||"").startsWith("send to game failed") ? "error" : "dropped";
const label = (o) => o === "applied" ? "applied" : o === "in flight" ? "in flight" : o === "error" ? "error" : (o||"").split(":")[0];

function card(r, fresh) {
  const k = kind(r.outcome), base = "/trace/" + r.dir.split("/").pop();
  const done = !!r.received;   // the model answered (or the HTTP call failed)
  const frames = (r.frames || []).map(f =>
    `<span>#${f.seq} <b class="t">${esc(f.captured)}</b>${f.key_after ? ` → <span class="key">${esc(f.key_after.replace("Key",""))}</span>` : ""}</span>`).join("");
  const p = r.parsed || {};
  const key = p.key && p.key !== "none" ? `<span class="key">${esc(p.key.replace("Key",""))}</span>` : `<span class="muted">none</span>`;
  let step4;
  const seen = p.seen && Object.keys(p.seen).length ? `<div class="muted">saw: <b style="color:var(--ink)">${esc(p.seen.chicken || "?")}</b> on ${esc(p.seen.on || "?")}${p.seen.danger ? " · " + esc(p.seen.danger) : ""}</div>` : "";
  const plan = r.plan ? `<div>plan <span class="t">${esc(r.plan)}</span>${r.slots_played ? ` <span class="muted">· played ${esc(r.slots_played)} slots</span>` : ""}${r.cut ? ` <span class="muted">· cut: ${esc(r.cut)}</span>` : ""}</div>` : "";
  if (r.outcome === "applied") step4 = `${plan}<div>${(r.sent_to_game||[]).map(esc).join("<br>")}</div><span class="muted">(chunk started ${r.wait_ms} ms after the answer)</span>`;
  else if (r.outcome === "in flight") step4 = `<span class="muted">waiting for the model</span>`;
  else step4 = `${plan}<span class="muted">nothing sent:</span> ${esc(r.outcome)}`;
  const output = !done ? `<span class="muted">—</span>` :
    `<span class="t">${esc(r.received)}</span> <span class="muted">${r.model_ms} ms · HTTP ${r.http || "—"}</span>
     ${seen}${r.output ? `<div class="out">${esc(r.output)}</div>` : ""}${r.error ? `<div class="out" style="color:var(--bad)">${esc(r.error)}</div>` : ""}`;
  return `<article class="call${fresh ? " new" : ""}" data-kind="${k}">
    <div class="shot"><img loading="lazy" src="${base}/sheet.png" alt="6-frame sheet sent in call ${r.call}"></div>
    <div class="body">
      <div class="head"><span class="id">Call ${r.call}</span><span class="badge ${k}">${esc(label(r.outcome))}</span>
        ${done ? `model said ${key} <span class="muted">${p.confidence ?? ""}</span>` : ""}</div>
      <div class="steps">
        <span class="n">1 · sent</span><div><span class="t">${esc(r.sent || "")}</span> <span class="muted">${(r.frames||[]).length || 6} frames in one image</span>
          <div class="frames">${frames}</div></div>
        <span class="n">2 · input</span><div class="links">
          <a href="${base}/request.json?elide" target="_blank">request.json (readable)</a>
          <a href="${base}/request.json" target="_blank">exact bytes</a>
          <a href="${base}/sheet.png" target="_blank">sheet.png</a>
          ${r.request_bytes ? `<span class="muted">${(r.request_bytes/1024).toFixed(0)} KB</span>` : ""}</div>
        <span class="n">3 · output</span><div>${output}
          ${done ? `<div class="links"><a href="${base}/response.json" target="_blank">response.json</a></div>` : ""}</div>
        <span class="n">4 · game</span><div>${step4}</div>
      </div>
    </div></article>`;
}

function stats() {
  const n = rows.length, by = {applied:0, dropped:0, error:0};
  rows.forEach(r => by[kind(r.outcome)]++);
  const ms = rows.filter(r => r.model_ms && r.outcome !== "error").map(r => r.model_ms).sort((a,b) => a-b);
  const med = ms.length ? ms[Math.floor(ms.length/2)] : "—";
  const keys = {}; rows.forEach(r => { const k = (r.parsed||{}).key || "none"; keys[k] = (keys[k]||0)+1; });
  const mix = Object.entries(keys).sort((a,b)=>b[1]-a[1]).map(([k,v]) => `${k.replace("Key","")} ${v}`).join(" · ");
  $("#stats").innerHTML = `<span>calls <b>${n}</b></span><span>applied <b>${by.applied}</b></span>
    <span>dropped <b>${by.dropped}</b></span><span>errors <b>${by.error}</b></span>
    <span>in flight <b>${inflight.filter(r => r.outcome === "in flight").length}</b></span><span>median <b>${med}</b> ms</span>
    <span>answers <b>${esc(mix) || "—"}</b></span>`;
}

// Keyed update: a card is rebuilt only when its content changed, so images
// that are already loaded are never re-fetched (no black flashes).
const cache = new Map();   // call -> {html, el}
function render(freshIds) {
  const f = $("#filter").value;
  const shown = [...inflight, ...rows.slice().reverse()]
    .filter(r => f === "all" || kind(r.outcome) === f);
  const list = $("#list");
  if (!shown.length) {
    cache.clear();
    list.innerHTML = `<div class="empty">${rows.length ? "No calls match this filter." : "Waiting for the first model call.<br>Start the runner with policy_fast_vision.py; calls appear here as they finish."}</div>`;
    stats(); return;
  }
  list.querySelector(".empty")?.remove();
  const keep = new Set();
  let prev = null;
  for (const r of shown) {
    keep.add(r.call);
    const html = card(r, false);
    let entry = cache.get(r.call);
    if (!entry || entry.html !== html) {
      const tmp = document.createElement("div"); tmp.innerHTML = html;
      const el = tmp.firstElementChild;
      if (entry) {
        // Same call, new state (in flight -> answered): keep the loaded image element.
        const img = entry.el.querySelector(".shot img");
        if (img) el.querySelector(".shot").replaceChildren(img);
        entry.el.replaceWith(el);
      }
      if (freshIds.has(r.call)) el.classList.add("new");
      entry = {html, el}; cache.set(r.call, entry);
    }
    const want = prev ? prev.nextSibling : list.firstChild;
    if (entry.el !== want) list.insertBefore(entry.el, want);
    prev = entry.el;
  }
  for (const [call, entry] of cache) if (!keep.has(call)) { entry.el.remove(); cache.delete(call); }
  stats();
}

async function poll() {
  if (!paused) {
    try {
      const res = await fetch(`/api/calls?since=${since}`, {cache: "no-store"});
      const data = await res.json();
      if (data.prompt) { $("#prompt").textContent = data.prompt; $("#promptBox").hidden = false; }
      const fresh = new Set(firstLoad ? [] : data.calls.map(r => r.call));
      if (data.calls.length) {
        rows = rows.concat(data.calls).slice(-MAX);
        since = data.calls[data.calls.length - 1].call;
      }
      const flyChanged = JSON.stringify(inflight.map(r=>r.call)) !== JSON.stringify(data.inflight.map(r=>r.call));
      inflight = data.inflight;
      if (data.calls.length || flyChanged || firstLoad) render(fresh);
      firstLoad = false; lastOk = Date.now();
      $("#dot").className = "dot"; $("#liveText").textContent = "live · " + data.run.split("/").slice(-2).join("/");
    } catch (e) {
      $("#dot").className = "dot off"; $("#liveText").textContent = "trace server unreachable";
    }
  }
  setTimeout(poll, 500);
}
$("#pause").onclick = () => { paused = !paused; $("#pause").textContent = paused ? "Resume" : "Pause";
  $("#liveText").textContent = paused ? "paused" : "live"; };
$("#filter").onchange = () => { cache.clear(); $("#list").innerHTML = ""; render(new Set()); };
$("#list").addEventListener("click", e => {
  if (e.target.tagName === "IMG") { $("#zoom img").src = e.target.src; $("#zoom").showModal(); } });
$("#zoom").addEventListener("click", () => $("#zoom").close());
poll();
</script>
</body>
</html>
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", default=os.environ.get("ANO_RUN_DIR", os.path.join(os.getcwd(), "runs", "fast-vision")))
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args(argv)
    run = Path(args.run).resolve()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(run))
    print(f"Trace page: http://127.0.0.1:{args.port}/  (reading {run})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
