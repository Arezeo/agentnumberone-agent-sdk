"""Fast vision loop: frames in, one aggregated image out, an action chunk back.

    1. every tick: keep the latest N frames (default 6 = 5 history + current)
    2. tile them into ONE captioned contact sheet and send it to an
       OpenAI-compatible /chat/completions vision model (up to N calls in flight)
    3. the model reports what it sees, then plans the next ANO_CHUNK_MS as slots
       of ANO_CHUNK_SLOT_MS (one key or "none" per slot)
    4. the loop taps each slot on time; a newer plan replaces what is left of
       the old one; answers decided from older pictures are dropped

The model decides. The code records what it reports (trace/) and never
overrules its keys; the only filters are about time (stale or superseded
answers) and an optional confidence floor. ANO_CHUNK_MS=0 gives one key per call.
"""

from __future__ import annotations

import base64
import concurrent.futures
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sheet import Tile, build, encode_png, to_rgb


ALIASES: Dict[str, str] = {
    "forward": "KeyW", "fwd": "KeyW", "up": "KeyW", "KeyW": "KeyW", "ArrowUp": "KeyW",
    "backward": "KeyS", "back": "KeyS", "down": "KeyS", "KeyS": "KeyS", "ArrowDown": "KeyS",
    "left": "KeyA", "KeyA": "KeyA", "ArrowLeft": "KeyA",
    "right": "KeyD", "KeyD": "KeyD", "ArrowRight": "KeyD",
    "W": "KeyW", "S": "KeyS", "A": "KeyA", "D": "KeyD",
}


def canonical_direction(value: Any) -> Optional[str]:
    """Accept 'W', 'KeyW', 'forward', 'left', 'ArrowUp' ... -> a DOM code, or None."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if len(text) == 1:
        text = text.upper()
    return ALIASES.get(text)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return int(default)


SIMPLE_PROMPT = """\
You are playing CrossRoad: a chicken must cross lanes of traffic, steered with
W (forward / up the screen), S (back), A (left), D (right).

The image is a contact sheet of the last {frames} frames, oldest first,
left-to-right then top-to-bottom. The last tile is the current frame. Each
tile is captioned with its number, its age in seconds and the key pressed
after it.

Decide the ONE next key press.

Reply with exactly one JSON object and nothing else:
  {{"key": "W" | "A" | "S" | "D" | "none",
    "confidence": 0.0-1.0,
    "why": "<=100 chars, what you saw"}}

Rules:
- Trees and rocks block a hop in any direction.
- Prefer a key that moves the chicken toward empty road.
- Do not step into a lane that currently holds a car at the chicken's position.
- If every move is unsafe, answer "none" with confidence 0.
{instruction}"""


CHUNK_PROMPT = """\
You are playing CrossRoad: a chicken must cross lanes of traffic, steered with
W (forward / up the screen), S (back), A (left), D (right). One tap = one hop.

The image is a contact sheet of the last {frames} frames, oldest first,
left-to-right then top-to-bottom. The last tile (captioned NOW) is the current
frame. Each tile is captioned with its number, its age in seconds and the key
tapped after it.

First LOOK at the NOW tile, then PLAN the next {chunk_ms} ms as {slots} slots of
{slot_ms} ms. For each slot give the key to tap at the start of that slot, or
"none" to wait. A newer plan will replace whatever is left of this one.

Reply with exactly one JSON object and nothing else, fields in this order:
  {{"chicken": "standing" | "squashed_flat" | "not_visible",
    "on": "grass" | "road" | "water",
    "danger": "<=60 chars: nearest car in the chicken's lane or the next lane up, and which way it moves",
    "keys": [<{slots} entries, each "W" | "A" | "S" | "D" | "none">],
    "confidence": 0.0-1.0,
    "why": "<=80 chars"}}

Rules:
- If the chicken is squashed_flat or not_visible, every key is "none".
- Trees and rocks block a hop in any direction.
- Hop up (W) only into a lane where no car will reach the chicken's column
  during the hop; otherwise wait, or step sideways/back out of a car's path.
- Prefer progress up the screen when it is safe.
{instruction}"""


@dataclass
class SimpleConfig:
    frames: int = 6                 # 5 history + current
    action_gap_s: float = 0.5       # minimum spacing between applied actions
    request_gap_s: float = 0.0      # extra spacing between model requests
    min_confidence: float = 0.3
    max_answer_age_s: float = 1.0   # drop an answer older than this instead of pressing
    max_inflight: int = 1           # model calls allowed out at once (staggered by request_gap_s)
    chunk_ms: int = 0               # >0: the model plans this many ms of keys per call (action chunk)
    chunk_slot_ms: int = 250        # one key (or none) per slot
    say: bool = True                # send the model's "why" to the game's text channel
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    timeout_s: float = 8.0
    max_tokens: int = 120
    detail: str = "high"

    @classmethod
    def from_env(cls) -> Optional["SimpleConfig"]:
        base = os.environ.get("ANO_MODEL_BASE_URL", "").strip()
        if not base:
            return None
        max_inflight = max(1, env_int("ANO_MAX_INFLIGHT", cls.max_inflight))
        # With several calls out, launches must be staggered or they all carry
        # the same picture; default to 0.2 s (under one tick, so a request can
        # go out on every tick) unless the operator chose a gap.
        default_gap = 0.2 if max_inflight > 1 else cls.request_gap_s
        chunk_ms = max(0, env_int("ANO_CHUNK_MS", 1000))
        return cls(
            chunk_ms=chunk_ms,
            chunk_slot_ms=max(50, env_int("ANO_CHUNK_SLOT_MS", cls.chunk_slot_ms)),
            say=os.environ.get("ANO_SAY", "1") != "0",
            max_inflight=max_inflight,
            frames=env_int("ANO_FRAMES", cls.frames),
            # In chunk mode the slots set the pace; the gap only stops two taps
            # landing inside one hop.
            action_gap_s=env_float("ANO_ACTION_GAP_S", 0.2 if chunk_ms > 0 else cls.action_gap_s),
            request_gap_s=env_float("ANO_REQUEST_GAP_S", default_gap),
            min_confidence=env_float("ANO_MIN_CONFIDENCE", cls.min_confidence),
            max_answer_age_s=env_float("ANO_MAX_ANSWER_AGE_S", cls.max_answer_age_s),
            base_url=base.rstrip("/"),
            model=os.environ.get("ANO_MODEL", ""),
            api_key=os.environ.get("ANO_MODEL_API_KEY", ""),
            timeout_s=env_float("ANO_MODEL_TIMEOUT_MS", 8000) / 1000.0,
            # A chunk reply (what it saw + N keys + why) needs more room than one key.
            max_tokens=env_int("ANO_MODEL_MAX_TOKENS", 200 if chunk_ms > 0 else 120),
            detail=os.environ.get("ANO_MODEL_DETAIL", "high"),
        )


@dataclass
class Answer:
    key: str = ""
    confidence: float = 0.0
    why: str = ""
    error: str = ""
    ms: int = 0
    requested_at: float = 0.0
    finished_at: float = 0.0
    # trace: what was sent, what came back, when (wall clock)
    call_id: int = 0
    sent_wall: float = 0.0
    received_wall: float = 0.0
    http_status: int = 0
    raw_text: str = ""
    keys: List[str] = field(default_factory=list)   # chunk: one canonical key or "" per slot
    seen: Dict[str, Any] = field(default_factory=dict)   # what the model said it saw


def clock(wall: float) -> str:
    """Local wall-clock time with milliseconds, e.g. 15:35:11.002."""
    if not wall:
        return ""
    return time.strftime("%H:%M:%S", time.localtime(wall)) + f".{int((wall % 1) * 1000):03d}"


class CallTrace:
    """One folder per model call, so every decision can be read back exactly.

    <run>/trace/<call>/sheet.png      the one aggregated image that was sent
    <run>/trace/<call>/request.json   the exact HTTP body sent (the API key is a header, not in here)
    <run>/trace/<call>/response.json  the exact HTTP body received (or the error)
    <run>/trace/<call>/outcome.json   what happened to the answer: applied / noop / stale / ...
    <run>/trace.jsonl                 one line per call with every timestamp

    Off with ANO_TRACE=0; keeps the newest ANO_TRACE_KEEP folders (default 300).
    """

    def __init__(self, root: Optional[Path]) -> None:
        self.enabled = root is not None and os.environ.get("ANO_TRACE", "1") != "0"
        self.root = Path(root) if root is not None else None
        self.dir = self.root / "trace" if self.root is not None else None
        self.keep = max(10, int(os.environ.get("ANO_TRACE_KEEP", "300") or 300))
        self.lock = threading.Lock()
        self.calls: Dict[int, Dict[str, Any]] = {}

    def folder(self, call_id: int) -> Optional[Path]:
        if not self.enabled:
            return None
        path = self.dir / f"{call_id:06d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write(self, call_id: int, name: str, data: bytes) -> None:
        if not self.enabled:
            return
        try:
            (self.folder(call_id) / name).write_bytes(data)
        except Exception:
            pass

    def start(self, call_id: int, entry: Dict[str, Any], sheet_png: bytes) -> None:
        if not self.enabled:
            return
        with self.lock:
            self.calls[call_id] = entry
        self.write(call_id, "sheet.png", sheet_png)
        if call_id % 25 == 0:
            self.prune()

    def finish(self, answer: "Answer", outcome: str, **extra: Any) -> None:
        if not self.enabled:
            return
        with self.lock:
            entry = self.calls.pop(answer.call_id, {"call": answer.call_id})
        entry.update({
            "received": clock(answer.received_wall),
            "model_ms": answer.ms,
            "http": answer.http_status,
            "output": answer.raw_text,
            "parsed": {"key": answer.key or "none", "keys": answer.keys, "confidence": answer.confidence,
                       "why": answer.why, "seen": answer.seen},
            "error": answer.error,
            "outcome": outcome,
            **extra,
        })
        self.write(answer.call_id, "outcome.json", json.dumps(entry, indent=2).encode("utf-8"))
        try:
            with self.lock, open(self.root / "trace.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, separators=(",", ":")) + "\n")
        except Exception:
            pass

    def prune(self) -> None:
        try:
            folders = sorted(p for p in self.dir.iterdir() if p.is_dir())
            for old in folders[:-self.keep]:
                shutil.rmtree(old, ignore_errors=True)
        except Exception:
            pass


class SimplePolicy:
    """Frames in, one key press out, every ~0.5 s."""

    def __init__(self, config: SimpleConfig, *, board=None, runbook: str = "") -> None:
        self.config = config
        self.board = board
        self.prompt = self.prompt_for(runbook)
        self.window: List[Tuple[float, np.ndarray, str, float]] = []   # (arrival, rgb, action_after, wall)
        self.seq = 0
        self.inflight: List[concurrent.futures.Future] = []
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=config.max_inflight,
                                                          thread_name_prefix="ano-model")
        self.last_applied_request_at = 0.0   # answers decided from older pictures are dropped
        self.pending: Optional[Answer] = None
        self.last_action_at = 0.0
        self.last_request_at = 0.0
        self.last_key = ""
        self.counts = {"requests": 0, "answers": 0, "applied": 0, "noop": 0,
                       "errors": 0, "skipped_busy": 0, "skipped_gap": 0}
        self.last_error = ""
        self.trace = CallTrace(getattr(board, "root", None))
        self.call_id = 0
        self._composed: List[Dict[str, Any]] = []
        self.presses: List[Tuple[float, str]] = []   # (monotonic time sent, key), newest last
        # The chunk being played: answer, slot start times, keys, and per-slot result.
        self.chunk: Optional[Dict[str, Any]] = None

    @property
    def slots(self) -> int:
        if self.config.chunk_ms <= 0:
            return 1
        return max(1, self.config.chunk_ms // self.config.chunk_slot_ms)

    def prompt_for(self, instruction: str = "") -> str:
        extra = ("\nYou have also been told: " + instruction) if instruction else ""
        custom = os.environ.get("ANO_PROMPT_FILE", "").strip()
        if custom:
            # Your own game's prompt. Placeholders: {frames} {chunk_ms} {slots} {slot_ms} {instruction};
            # write literal braces as {{ }}.
            return Path(custom).read_text("utf-8").format(
                frames=self.config.frames, chunk_ms=self.slots * self.config.chunk_slot_ms,
                slots=self.slots, slot_ms=self.config.chunk_slot_ms, instruction=extra)
        if self.config.chunk_ms > 0:
            return CHUNK_PROMPT.format(frames=self.config.frames, chunk_ms=self.slots * self.config.chunk_slot_ms,
                                       slots=self.slots, slot_ms=self.config.chunk_slot_ms, instruction=extra)
        return SIMPLE_PROMPT.format(frames=self.config.frames, instruction=extra)

    # -- 1. keep the window ------------------------------------------------
    def observe(self, frame) -> None:
        self.seq += 1
        # The key "after" a frame is filled in at compose time from self.presses;
        # the third slot is kept for compatibility and no longer used for captions.
        self.window.append((time.monotonic(), to_rgb(frame), "", time.time()))
        if len(self.window) > self.config.frames:
            del self.window[: len(self.window) - self.config.frames]

    # -- 2. one image, one question ---------------------------------------
    def compose(self) -> Tuple[bytes, List[Dict[str, Any]]]:
        now = time.monotonic()
        wall_now = time.time()
        tiles = []
        self._composed = []
        frames = self.window[-self.config.frames:]
        for index, entry in enumerate(frames):
            arrival, rgb = entry[0], entry[1]
            # Keys actually sent between this frame and the next one -- not the
            # last key ever pressed, which made every tile read "then pressed W".
            until = frames[index + 1][0] if index + 1 < len(frames) else now
            action = "+".join(key for at, key in self.presses if arrival <= at < until) or entry[2]
            wall = entry[3] if len(entry) > 3 else wall_now - (now - arrival)
            newest = index == len(frames) - 1
            seq = self.seq - len(frames) + index + 1
            tiles.append((rgb, Tile(seq=seq, age_ms=int((now - arrival) * 1000),
                                    action="" if newest else action)))
            self._composed.append({"seq": seq, "captured": clock(wall),
                                   "age_ms": int((now - arrival) * 1000),
                                   "key_after": "" if newest else action})
        sheet, meta = build(tiles)
        self.last_sheet = sheet          # for controls.show_model_input
        return encode_png(sheet), meta

    def request(self, sheet_png: bytes) -> None:
        started = time.monotonic()
        self.last_request_at = started
        self.counts["requests"] += 1
        self.call_id += 1
        call_id = self.call_id
        self.trace.start(call_id, {"call": call_id, "frames": self._composed,
                                   "dir": f"trace/{call_id:06d}"}, sheet_png)
        self.inflight.append(self.pool.submit(self._call, sheet_png, started, call_id))

    def _call(self, sheet_png: bytes, started: float, call_id: int = 0) -> Answer:
        url = base64.b64encode(sheet_png).decode("ascii")
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Contact sheet follows, oldest first; last tile is now. Next key?"},
                    {"type": "image_url",
                     "image_url": {"url": "data:image/png;base64," + url,
                                   "detail": self.config.detail}},
                ]},
            ],
        }
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            self.config.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": "Bearer " + self.config.api_key}
                        if self.config.api_key else {})},
        )
        self.trace.write(call_id, "request.json", request.data)
        sent_wall = time.time()
        with self.trace.lock:
            if call_id in self.trace.calls:
                self.trace.calls[call_id]["sent"] = clock(sent_wall)
                self.trace.calls[call_id]["request_bytes"] = len(request.data)
        status, raw = 0, b""
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                status, raw = response.status, response.read()
            body = json.loads(raw.decode("utf-8"))
            text = body["choices"][0]["message"]["content"]
            answer = self._parse(text, started)
        except Exception as exc:
            if isinstance(exc, urllib.error.HTTPError):
                status = exc.code
                try:
                    raw = exc.read()
                except Exception:
                    pass
            answer = Answer(error=f"{type(exc).__name__}: {exc}"[:160], requested_at=started,
                            finished_at=time.monotonic(),
                            ms=int((time.monotonic() - started) * 1000))
            text = ""
        answer.call_id, answer.sent_wall, answer.received_wall = call_id, sent_wall, time.time()
        answer.http_status, answer.raw_text = status, text
        self.trace.write(call_id, "response.json", raw or answer.error.encode("utf-8"))
        return answer

    def _parse(self, text: str, started: float) -> Answer:
        answer = Answer(requested_at=started, finished_at=time.monotonic(),
                        ms=int((time.monotonic() - started) * 1000))
        body = (text or "").strip()
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end < start:
            answer.error = "no JSON in reply"
            self._log({"kind": "bad_reply", "raw": body[:300]})
            return answer
        try:
            data = json.loads(body[start:end + 1])
        except ValueError as exc:
            answer.error = f"bad JSON: {exc}"
            return answer
        raw = data.get("key") or data.get("action") or data.get("direction") or ""
        if isinstance(raw, str) and raw.strip().lower() in ("none", "noop", "wait", "stop", ""):
            answer.key = ""
        else:
            answer.key = canonical_direction(raw) or ""
            if not answer.key:
                answer.error = f"unusable key {raw!r}"
        keys = data.get("keys")
        if isinstance(keys, list) and keys:
            slots = []
            for item in keys[: self.slots]:
                text = str(item or "").strip()
                slots.append("" if text.lower() in ("none", "noop", "wait", "") else (canonical_direction(text) or ""))
            answer.keys = slots
            answer.key = next((k for k in slots if k), "")
            answer.error = ""        # "key" may be absent in chunk replies
        elif answer.key:
            answer.keys = [answer.key]
        # What the model says it saw is recorded for the trace; its keys are
        # played exactly as given -- the model decides, the code does not overrule it.
        answer.seen = {k: data.get(k) for k in ("chicken", "on", "danger") if k in data}
        answer.confidence = float(data.get("confidence", 0) or 0)
        answer.why = str(data.get("why") or "")[:120]
        return answer

    # -- 3. apply, spaced --------------------------------------------------
    def step(self, controls) -> str:
        """One loop iteration. Returns a short status string."""
        now = time.monotonic()

        # Collect a finished answer.
        finished = [f for f in self.inflight if f.done()]
        self.inflight = [f for f in self.inflight if not f.done()]
        for future in finished:
            answer = future.result()
            self.counts["answers"] += 1
            if answer.error:
                self.counts["errors"] += 1
                self.last_error = answer.error
                self.trace.finish(answer, "error")
                continue
            # Newest picture wins: with several calls out, answers can land
            # out of order, and an older one must never override a newer one.
            newest_held = max(self.last_applied_request_at,
                              self.pending.requested_at if self.pending is not None else 0.0)
            if answer.requested_at <= newest_held:
                self.counts["superseded"] = self.counts.get("superseded", 0) + 1
                self._log({"kind": "superseded", "key": answer.key, "model_ms": answer.ms})
                self.trace.finish(answer, "dropped: a newer answer was already held or applied")
                continue
            if self.pending is not None:
                self.counts["superseded"] = self.counts.get("superseded", 0) + 1
                self.trace.finish(self.pending, f"dropped: replaced by newer call {answer.call_id}")
            self.pending = answer

        # Apply the pending action if it has been long enough since the last one.
        status = "waiting"
        if self.pending is not None and now - self.pending.finished_at > self.config.max_answer_age_s:
            # The picture it was decided from is too old; pressing now is a guess.
            self.counts["stale"] = self.counts.get("stale", 0) + 1
            self._log({"kind": "stale", "key": self.pending.key,
                       "age_ms": int((now - self.pending.finished_at) * 1000)})
            self.trace.finish(self.pending, "dropped: stale (answer waited "
                              f"{int((now - self.pending.finished_at) * 1000)} ms)")
            self.last_applied_request_at = max(self.last_applied_request_at, self.pending.requested_at)
            self.pending = None
        if self.config.chunk_ms > 0:
            status = self._chunk_step(controls, now)
        elif self.pending is not None:
            if now - self.last_action_at < self.config.action_gap_s:
                self.counts["skipped_gap"] += 1
                status = "gap"
            else:
                answer = self.pending
                self.pending = None
                self.last_applied_request_at = max(self.last_applied_request_at, answer.requested_at)
                if answer.key and answer.confidence >= self.config.min_confidence:
                    try:
                        applied_wall = time.time()
                        controls.keyDown(answer.key)
                        controls.keyUp(answer.key)
                        self.presses.append((time.monotonic(), answer.key))
                        del self.presses[:-32]
                        self.trace.finish(answer, "applied", applied=clock(applied_wall),
                                          sent_to_game=[f"keyDown {answer.key}", f"keyUp {answer.key}"],
                                          wait_ms=int((now - answer.finished_at) * 1000))
                        self.counts["applied"] += 1
                        self.last_key = answer.key
                        self.last_action_at = now
                        status = "applied:" + answer.key
                        self._log({"kind": "action", "key": answer.key,
                                   "confidence": answer.confidence, "why": answer.why,
                                   "model_ms": answer.ms, "age_ms": int((now - answer.finished_at) * 1000)})
                    except Exception as exc:
                        self.counts["errors"] += 1
                        self.last_error = f"send failed: {exc}"
                        self.trace.finish(answer, f"send to game failed: {type(exc).__name__}: {exc}")
                        status = "send_error"
                else:
                    self.counts["noop"] += 1
                    self.trace.finish(answer, "no key: model said none" if not answer.key else
                                      f"no key: confidence {answer.confidence} < {self.config.min_confidence}")
                    status = "noop"
                    self._log({"kind": "noop", "confidence": answer.confidence,
                               "why": answer.why, "model_ms": answer.ms})

        # Drain anything the last batch left queued, so no key stays held.
        try:
            controls.releaseAll()
        except Exception:
            pass
        return status

    # -- 3b. action chunks ---------------------------------------------------
    def _chunk_step(self, controls, now: float) -> str:
        """Start the newest chunk (cutting the old one), then play slots that are due."""
        if self.pending is not None:
            answer, self.pending = self.pending, None
            self.last_applied_request_at = max(self.last_applied_request_at, answer.requested_at)
            if any(answer.keys) and answer.confidence >= self.config.min_confidence:
                self._cut_chunk(f"replaced by call {answer.call_id}")
                self.chunk = {"answer": answer, "start": now, "keys": list(answer.keys),
                              "next": 0, "sent": [], "skipped": []}
                self._say(controls, answer)
            else:
                # A fresh "wait" is also a plan: stop whatever is left of the old one.
                self._cut_chunk(f"stopped by call {answer.call_id} (planned no keys)")
                self.counts["noop"] += 1
                reason = ("no key: model planned none" if not any(answer.keys)
                          else f"no key: confidence {answer.confidence} < {self.config.min_confidence}")
                self.trace.finish(answer, reason, plan=self._plan_text(answer))
                self._say(controls, answer)
                self._log({"kind": "noop", "confidence": answer.confidence, "why": answer.why,
                           "seen": answer.seen, "model_ms": answer.ms})
        return self._play_chunk(controls, now)

    def _play_chunk(self, controls, now: float) -> str:
        chunk = self.chunk
        if chunk is None:
            return "waiting"
        slot_s = self.config.chunk_slot_ms / 1000.0
        status = "chunk"
        while chunk["next"] < len(chunk["keys"]):
            index = chunk["next"]
            due = chunk["start"] + index * slot_s
            if now < due:
                break
            key = chunk["keys"][index]
            if not key:
                chunk["next"] += 1
                continue
            if now - self.last_action_at < self.config.action_gap_s:
                # Previous hop still in progress; send late if the slot is still open.
                if now >= due + slot_s:
                    chunk["skipped"].append(index)
                    chunk["next"] += 1
                    continue
                break
            try:
                controls.keyDown(key)
                controls.keyUp(key)
            except Exception as exc:
                self.counts["errors"] += 1
                self.last_error = f"send failed: {exc}"
                chunk["skipped"].append(index)
                chunk["next"] += 1
                continue
            self.presses.append((time.monotonic(), key))
            del self.presses[:-32]
            self.counts["applied"] += 1
            self.last_key, self.last_action_at = key, now
            chunk["sent"].append({"slot": index, "key": key, "at": clock(time.time()),
                                  "late_ms": int((now - due) * 1000)})
            chunk["next"] += 1
            status = "applied:" + key
            self._log({"kind": "action", "key": key, "slot": index, "call": chunk["answer"].call_id,
                       "confidence": chunk["answer"].confidence, "why": chunk["answer"].why})
        if chunk["next"] >= len(chunk["keys"]):
            self._finish_chunk(chunk, cut="")
            self.chunk = None
        return status

    def _cut_chunk(self, reason: str) -> None:
        if self.chunk is not None:
            self._finish_chunk(self.chunk, cut=reason)
            self.chunk = None

    def _plan_text(self, answer: Answer) -> str:
        return " ".join((k.replace("Key", "") if k else "-") for k in answer.keys) or "-"

    def _finish_chunk(self, chunk: Dict[str, Any], cut: str) -> None:
        answer = chunk["answer"]
        sent = chunk["sent"]
        extra = {"plan": self._plan_text(answer),
                 "slots_played": f"{chunk['next']}/{len(chunk['keys'])}",
                 "sent_to_game": [f"{s['at']} tap {s['key']} (slot {s['slot']})" for s in sent],
                 "wait_ms": int((chunk["start"] - answer.finished_at) * 1000)}
        if chunk["skipped"]:
            extra["skipped_slots"] = chunk["skipped"]
        if cut:
            extra["cut"] = cut
        if sent:
            self.trace.finish(answer, "applied", applied=sent[0]["at"], **extra)
        else:
            self.trace.finish(answer, f"dropped: {cut or 'no slot was sent'}", **extra)

    def _say(self, controls, answer: Answer) -> None:
        """Return the model's own "why" to the game's text channel, as written."""
        if not self.config.say or not hasattr(controls, "say"):
            return
        text = (answer.why or "").strip()
        if not text or text == getattr(self, "_last_said", ""):
            return
        try:
            controls.say(text, final=True)
            self._last_said = text
        except Exception:
            pass

    def wants_request(self, now: float) -> bool:
        if len(self.inflight) >= self.config.max_inflight:
            self.counts["skipped_busy"] += 1
            return False
        if len(self.window) < 2:
            return False
        return now - self.last_request_at >= self.config.request_gap_s

    def _log(self, entry: Dict[str, Any]) -> None:
        if self.board is not None:
            try:
                self.board.log("results.jsonl", entry)
            except Exception:
                pass

    def stats(self) -> Dict[str, Any]:
        return {**self.counts, "inflight": len(self.inflight),
                "pending": self.pending is not None, "last_error": self.last_error,
                "window": len(self.window)}
