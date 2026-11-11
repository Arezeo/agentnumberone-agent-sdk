"""GameScaling 9B/2B/0.8B GGUF policy for the Agent Number One SDK harness.

Drop-in for `scripts/remote-control/agent.py --policy`:

    python scripts/remote-control/agent.py --origin https://agentnumberone.io \
      --policy packages/agent-sdk-llamacpp/policy_llamacpp.py

Talks to a local llama.cpp server (`serve.sh`) and speaks the model's native
action grammar:

    <|action_start|>X Y Z ; g1 ; g2 ; g3 ; g4 ; g5 ; g6<|action_end|>

X,Y are RELATIVE pointer displacement in per-mille of screen size (X>0 right,
Y>0 down). Each group is a 33 ms key step. Nothing here raises: a bad action
must never kill the control channel.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import urllib.request

URL = os.environ.get("ANO_MODEL_URL", "http://127.0.0.1:8901/v1/chat/completions")
PROMPT_PATH = os.environ.get("ANO_SYSTEM_PROMPT", os.path.join(os.path.dirname(__file__), "gamescaling-system.txt"))
MAX_TOKENS = int(os.environ.get("ANO_MAX_TOKENS", "96"))
WARN_AFTER = int(os.environ.get("ANO_NOOP_WARN", "5"))  # consecutive empty actions before warning

try:
    SYSTEM = open(PROMPT_PATH).read()
except OSError:
    SYSTEM = "You are playing a game. Reply with the action block only."

KEYMAP = {
    "W": "KeyW", "A": "KeyA", "S": "KeyS", "D": "KeyD",
    "Q": "KeyQ", "E": "KeyE", "R": "KeyR", "F": "KeyF", "P": "KeyP",
    "SPACE": "Space", "SHIFT": "ShiftLeft", "CTRL": "ControlLeft",
    "TAB": "Tab", "ESC": "Escape", "ENTER": "Enter",
    "UP": "ArrowUp", "DOWN": "ArrowDown", "LEFT": "ArrowLeft", "RIGHT": "ArrowRight",
    "ONE": "Digit1", "TWO": "Digit2", "THREE": "Digit3",
}
BUTTONS = {"LB": "left", "RB": "right", "MB": "middle"}

STATE = {"ptr": [0.5, 0.5], "held": [], "last": 0.0, "noop": 0, "decisions": 0, "errors": 0}


def _b64(frame):
    img = frame.reformat(width=640, height=360, format="rgb24").to_image()
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=78)
    return base64.b64encode(buf.getvalue()).decode()


def _ask(b64):
    payload = {
        "model": "gamescaling",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "Current frame. Output the action only."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            ]},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def _parse(text):
    m = re.search(r"<\|action_start\|>(.*?)<\|action_end\|>", text, re.S)
    body = (m.group(1) if m else text).strip()
    parts = [p.strip() for p in body.split(";")]
    head = parts[0].split() if parts else []
    if len(head) < 2:
        return None
    try:
        dx, dy = float(head[0]), float(head[1])
    except ValueError:
        return None
    return dx, dy, [p.split() for p in parts[1:7]]


def _release(controls):
    for k in STATE["held"]:
        try:
            controls.keyUp(k)
        except Exception:
            pass
    STATE["held"] = []
    try:
        controls.releaseAll()
    except Exception:
        pass


def decide(frame, controls):
    _release(controls)
    t0 = time.time()
    try:
        text = _ask(_b64(frame))
        act = _parse(text)
    except Exception as exc:
        STATE["errors"] += 1
        print(f"policy error: {type(exc).__name__} {exc}", flush=True)
        return

    STATE["decisions"] += 1
    gap = (t0 - STATE["last"]) * 1000 if STATE["last"] else 0
    STATE["last"] = t0
    dt = (time.time() - t0) * 1000
    print(f"decide {dt:.0f}ms gap {gap:.0f}ms raw={text[:100]!r}", flush=True)

    if not act:
        STATE["noop"] += 1
        if STATE["noop"] == WARN_AFTER:
            print(f"WARNING: {WARN_AFTER} consecutive unparseable actions — "
                  f"check that the system prompt matches this game and that the "
                  f"model emits a non-empty action block", flush=True)
        return

    dx, dy, groups = act
    if dx == 0 and dy == 0 and not any(groups):
        STATE["noop"] += 1
        if STATE["noop"] == WARN_AFTER:
            print(f"WARNING: {WARN_AFTER} consecutive empty actions "
                  f"(0 0 0 ; ; ; ; ; ;) — the model is answering in-format but with "
                  f"no content. Usually the prompt describes a different game than "
                  f"the one on screen.", flush=True)
    else:
        STATE["noop"] = 0

    x = max(0.0, min(1.0, STATE["ptr"][0] + dx / 1000.0))
    y = max(0.0, min(1.0, STATE["ptr"][1] + dy / 1000.0))
    STATE["ptr"] = [x, y]
    try:
        controls.moveTo(x, y)
    except Exception as exc:
        print(f"moveTo rejected: {exc}", flush=True)

    pressed, buttons = [], []
    for tok in (groups[0] if groups else []):
        up = tok.strip().upper()
        if up in KEYMAP:
            try:
                controls.keyDown(KEYMAP[up])
                pressed.append(KEYMAP[up])
            except Exception:
                pass
        elif up in BUTTONS:
            buttons.append(BUTTONS[up])

    for b in buttons:
        try:
            controls.mouseDown(x, y, b)
            controls.mouseUp(x, y, b)
        except Exception:
            pass

    STATE["held"] = pressed


def summary():
    """Called by nobody automatically — handy from a REPL or a wrapper."""
    return dict(STATE)
