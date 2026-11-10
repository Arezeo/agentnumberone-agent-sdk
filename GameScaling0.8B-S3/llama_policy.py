"""GameScaling through a llama.cpp server -> Agent Number One controls.

A policy for `scripts/remote-control/agent.py --policy`. It was written for the
9B-S3 checkpoint as a 4-bit GGUF, where one decision takes longer than a second,
and it encodes what a slow native-action model needs on this runner:

1. Sampling: temperature 1.0, the publisher's live setting. At temperature 0
   the top choice is nearly always the empty action and the agent freezes.
2. Prompt: chosen per game, never fixed to one. `auto` uses the Pixel Duel
   prompt only on the Pixel Duel control surface, otherwise a general one.
   For your own game, point ANO_LLAMA_PROMPT_FILE at a prompt you wrote with
   `build_system_prompt`.
3. Context: the kit's user turn (commanded pointer, operator instruction) and
   earlier frames with their actions. History only grows and is then cleared,
   so the server's prompt cache stays valid and each decision evaluates one
   new image. Two empty actions in a row also clear it, because an all-empty
   history keeps the model answering empty. A change in the operator
   instruction does not clear it.
4. Execution: the runner sends a decision's actions only after `decide()`
   returns. The model therefore runs on its own thread; `decide()` returns at
   once on every video frame and plays the six 33 ms steps of the latest plan
   in time, moving the pointer progressively. As in the MLX runner, the last
   step stays held until the next plan, with a watchdog release.
5. Keys use the exact training vocabulary (`one` = Digit1, `One` = F1), mouse
   buttons are real holds, and scroll is sent.
6. Frames are converted with a private single-threaded converter. The live
   view converts the same frame on another thread, and a frame's own
   `reformat()` shares one cached scaler: two threads inside it can hang.

Never raises: a bad action must not close the control channel.
"""
import base64
import io
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream"))
from gamescaling import build_system_prompt, parse_action  # noqa: E402


def _data_dir():
    default = Path.home() / "Library/Application Support/AgentNumberOne/gamescaling"
    return Path(os.environ.get("ANO_GAMESCALING_DATA", str(default))).expanduser()


URL = os.environ.get("ANO_LLAMA_URL", "http://127.0.0.1:8901/v1/chat/completions")
GAME = os.environ.get("ANO_LLAMA_GAME", "auto")       # auto | general | voxelcraft | pixel-duel
PROMPT_FILE = os.environ.get("ANO_LLAMA_PROMPT_FILE", "")   # your own system prompt; overrides GAME
TEMPERATURE = float(os.environ.get("ANO_LLAMA_TEMPERATURE", "1.0"))
MAX_TOKENS = int(os.environ.get("ANO_LLAMA_MAX_TOKENS", "256"))
MAX_HISTORY = int(os.environ.get("ANO_LLAMA_HISTORY", "5"))   # frames per request before the history is cleared
NOOP_RESET = int(os.environ.get("ANO_LLAMA_NOOP_RESET", "2"))
IMAGE_SIZE = (int(os.environ.get("ANO_LLAMA_IMAGE_W", "640")), int(os.environ.get("ANO_LLAMA_IMAGE_H", "360")))
SLOT = int(os.environ.get("ANO_LLAMA_SLOT", "0"))
HOLD_WATCHDOG_S = float(os.environ.get("ANO_LLAMA_HOLD_WATCHDOG_S", "3"))
MAX_PLAN_AGE_S = float(os.environ.get("ANO_LLAMA_MAX_PLAN_AGE_S", "6"))
LOG_PATH = os.environ.get("ANO_LLAMA_LOG", str(_data_dir() / "results" / "llama-policy" / "decisions.jsonl"))

STEPS = 6
STEP_S = 0.2 / STEPS
CLOCK = time.monotonic      # replaced in tests
POINTER_NOTE = ("Mouse X/Y are relative screen displacements in per-mille units, applied to the "
                "current commanded pointer; they are not absolute coordinates. Your actions are "
                "executed for 200ms. Use observations to check progress. No hidden game state is "
                "available.")
MOUSE_LINE = "move the pointer or turn the camera relatively (X>0 right, Y>0 down); units are 1/1000 of screen width/height"


def _prompts():
    out = {}
    out["general"] = build_system_prompt(
        "an Agent Number One screen-controlled game or challenge",
        rules=("Follow the task, rules, player assignment and controls visible on the screen and "
               "the operator instructions. In first-person 3D games the mouse turns the camera, "
               "W A S D move, Space jumps, LB attacks or mines and RB uses or places. In top-down "
               "games W A S D move in screen directions and the mouse aims. When a dialog or menu "
               "is shown, move the pointer onto the wanted button and click LB. " + POINTER_NOTE),
        key_bindings={"W": "move forward / up", "A": "move left", "S": "move back / down",
                      "D": "move right", "Space": "jump, dash or task-specific space key",
                      "Shift": "hold shift", "Ctrl": "hold control", "Enter": "enter",
                      "Esc": "menu / close", "LB": "left mouse button: attack, mine or click",
                      "RB": "right mouse button: use or place", "Mouse X Y": MOUSE_LINE},
        mouse_aim=False)
    out["voxelcraft"] = build_system_prompt(
        "Voxelcraft",
        rules=("Voxelcraft is a first-person block-world survival game. Explore, mine blocks, "
               "gather resources and survive monsters, which come out at night. Moving the mouse "
               "turns the camera; the crosshair in the screen centre selects the block or monster "
               "you act on. Hold LB to mine a block; tap LB to hit a monster. Hearts at the bottom "
               "are health. When a dialog such as \"You died!\" is shown, move the pointer onto "
               "its button (for example Respawn) and click LB to continue. " + POINTER_NOTE),
        key_bindings={"W": "move forward", "A": "strafe left", "S": "move back",
                      "D": "strafe right", "Space": "jump / swim up", "Shift": "sneak",
                      "Ctrl": "sprint", "LB": "mine / attack / click a button",
                      "RB": "place / use / eat", "E": "open or close inventory",
                      "one two three four five six seven eight nine": "select hotbar slot",
                      "Esc": "pause menu", "Mouse X Y": MOUSE_LINE},
        mouse_aim=False)
    try:
        out["pixel-duel"] = (ROOT / "prompts" / "pixel-duel-system.txt").read_text().rstrip("\n")
    except OSError:
        out["pixel-duel"] = out["general"]
    if PROMPT_FILE:
        # A missing or empty file must stop the runner: playing with the wrong prompt is worse.
        out["custom"] = Path(PROMPT_FILE).expanduser().read_text().rstrip("\n")
        if not out["custom"].strip():
            raise ValueError("ANO_LLAMA_PROMPT_FILE is empty: " + PROMPT_FILE)
    return out


PROMPTS = _prompts()

# Training token -> physical DOM code. Case matters: one..nine are number-row
# keys, One..Twelve are F1..F12 (the runner rejects F-keys; they are skipped).
KEYS = {c: "Key" + c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
KEYS.update({w: "Digit%d" % i for i, w in enumerate(
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"))})
KEYS.update({w: "F%d" % (i + 1) for i, w in enumerate(
    ("One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve"))})
KEYS.update({"Shift": "ShiftLeft", "Ctrl": "ControlLeft", "Alt": "AltLeft", "Tab": "Tab",
             "Caps": "CapsLock", "Esc": "Escape", "Space": "Space", "Enter": "Enter",
             "Back": "Backspace", "Delete": "Delete", "Insert": "Insert", "Home": "Home",
             "End": "End", "Pause": "Pause", "Up": "ArrowUp", "Down": "ArrowDown",
             "Left": "ArrowLeft", "Right": "ArrowRight", "Equal": "Equal", "Minus": "Minus",
             "Period": "Period", "Slash": "Slash", "Quote": "Quote"})
BUTTONS = {"LB": "left", "RB": "right", "MB": "middle"}

_log_lock = threading.Lock()


def _log(**event):
    event["t"] = round(time.time(), 3)
    line = json.dumps(event, ensure_ascii=False)
    try:
        with _log_lock:
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            with open(LOG_PATH, "a") as handle:
                handle.write(line + "\n")
    except OSError:
        pass
    print(line[:300], flush=True)


def _profile(capabilities):
    if "custom" in PROMPTS:
        return "custom"
    if GAME != "auto":
        return GAME if GAME in PROMPTS else "general"
    # Pixel Duel's seat advertises only held controls; every other surface has click.
    return "general" if "click" in capabilities else "pixel-duel"


def _post(payload):
    request = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def _frame_image(frame):
    """The frame scaled to the model's input size, as a new rgb24 frame.

    Never call `frame.reformat()` or `frame.to_image()` on the runner's frame:
    they share one scaler cached on that frame, the live view uses it from
    another thread, and two threads inside it can block forever. That froze a
    live session until the runner's inference timeout closed it.
    """
    from av.video.reformatter import VideoReformatter
    return VideoReformatter().reformat(frame, width=IMAGE_SIZE[0], height=IMAGE_SIZE[1],
                                       format="rgb24", threads=1)


def _user(text, image):
    return [{"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image}}]


class Brain(threading.Thread):
    """Owns the model conversation. One request at a time; newest frame only."""

    def __init__(self):
        super().__init__(daemon=True, name="llama-policy-brain")
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.busy = False
        self.job = None
        self.result = None
        self.history = []          # (user_text, image_b64, action_text)
        self.profile = None
        self.reset = False
        self.count = 0

    def submit(self, job):
        with self.lock:
            if self.busy:
                return False
            self.busy, self.job = True, job
        if not self.is_alive():
            self.start()
        self.wake.set()
        return True

    def take(self):
        with self.lock:
            result, self.result = self.result, None
        return result

    def run(self):
        while True:
            self.wake.wait()
            self.wake.clear()
            job = self.job
            try:
                result = self.think(job)
            except Exception as exc:  # server down, timeout, bad JSON
                _log(kind="ask_failed", error=type(exc).__name__, detail=str(exc)[:200])
                self.history = []
                result = None
                time.sleep(0.5)
            with self.lock:
                self.result, self.busy = result, False

    def think(self, job):
        if self.reset or job["profile"] != self.profile:
            self.history, self.reset, self.profile = [], False, job["profile"]
        text = "current game screen\nCurrent commanded pointer (unverified): " + json.dumps(job["pointer"])
        if job["instruction"]:
            text += "\nOperator instruction: " + job["instruction"]
        messages = [{"role": "system", "content": PROMPTS[job["profile"]]}]
        for old_text, old_image, old_action in self.history:
            messages.append({"role": "user", "content": _user(old_text, old_image)})
            messages.append({"role": "assistant", "content": old_action})
        messages.append({"role": "user", "content": _user(text, job["image"])})
        started = time.monotonic()
        reply = _post({"model": "gamescaling", "messages": messages, "max_tokens": MAX_TOKENS,
                       "temperature": TEMPERATURE, "id_slot": SLOT, "cache_prompt": True})
        message = reply["choices"][0]["message"]
        content = message.get("content") or ""
        reasoning = (message.get("reasoning_content") or "").strip()
        action = parse_action(content, lenient=True)
        timings = reply.get("timings") or {}
        self.count += 1
        used = len(self.history)
        self.history.append((text, job["image"], action.history_text))
        recent = self.history[-NOOP_RESET:]
        if len(self.history) >= MAX_HISTORY:
            self.history = []      # growing-then-clearing keeps the prompt cache valid
        elif len(recent) == NOOP_RESET and all(parse_action(a).is_noop for _, _, a in recent):
            self.history = []      # an all-empty history keeps the model passive
        _log(kind="decision", n=self.count, profile=job["profile"],
             ms=round((time.monotonic() - started) * 1000), history=used,
             new_tokens=timings.get("prompt_n"), cached_tokens=timings.get("cache_n"),
             out_tokens=timings.get("predicted_n"),
             tok_s=round(timings.get("predicted_per_second") or 0, 1),
             parsed=action.parsed, noop=action.is_noop, raw=content[:160],
             reasoning=reasoning[:300], pointer=job["pointer"])
        return {"action": action, "reasoning": reasoning, "submitted": job["submitted"]}


BRAIN = Brain()

STATE = {"pointer": [0.5, 0.5], "keys": set(), "buttons": set(), "plan": None,
         "last_call": 0.0, "plan_started": 0.0, "instruction": ""}


def _emit(controls, name, *args, **kwargs):
    if name not in controls.capabilities:
        return False
    try:
        getattr(controls, name)(*args, **kwargs)
        return True
    except Exception as exc:
        _log(kind="rejected", action=name, error=str(exc)[:120])
        return False


def _release(controls, reason):
    if not STATE["keys"] and not STATE["buttons"]:
        return
    x, y = STATE["pointer"]
    for button in sorted(STATE["buttons"]):
        _emit(controls, "mouseUp", x, y, button)
    for key in sorted(STATE["keys"]):
        _emit(controls, "keyUp", key)
    _emit(controls, "releaseAll")
    STATE["keys"], STATE["buttons"] = set(), set()
    _log(kind="release", reason=reason)


def _apply_step(controls, plan, index):
    action = plan["action"]
    dx, dy, dz = action.mouse
    # Pointer: the 200 ms displacement is spread evenly over the six steps.
    x = min(1.0, max(0.0, plan["origin"][0] + dx / 1000.0 * (index + 1) / STEPS))
    y = min(1.0, max(0.0, plan["origin"][1] + dy / 1000.0 * (index + 1) / STEPS))
    if [x, y] != STATE["pointer"] and _emit(controls, "moveTo", x, y):
        STATE["pointer"] = [x, y]
    if index == 0 and dz:
        _emit(controls, "scroll", x, y, -100.0 * dz)   # Z>0 is scroll up
    tokens = action.steps[index] if index < len(action.steps) else []
    keys = {KEYS[t] for t in tokens if t in KEYS}
    buttons = {BUTTONS[t] for t in tokens if t in BUTTONS}
    for button in sorted(STATE["buttons"] - buttons):
        _emit(controls, "mouseUp", x, y, button)
    for key in sorted(STATE["keys"] - keys):
        _emit(controls, "keyUp", key)
    # First step re-sends every requested hold (as the MLX runner does), later steps only new ones.
    for key in sorted(keys if index == 0 else keys - STATE["keys"]):
        _emit(controls, "keyDown", key)
    for button in sorted(buttons if index == 0 else buttons - STATE["buttons"]):
        if "mouseDown" in controls.capabilities:
            _emit(controls, "mouseDown", x, y, button)
        else:
            _emit(controls, "click", x, y, button)
    STATE["keys"], STATE["buttons"] = keys, buttons


def _decide(frame, controls):
    now = CLOCK()
    if STATE["last_call"] and now - STATE["last_call"] > 2.0:
        # The task was paused or the page reconnected: the browser released everything.
        STATE["plan"] = None
        _release(controls, "gap")
        STATE["keys"], STATE["buttons"] = set(), set()
        BRAIN.reset = True
    STATE["last_call"] = now

    instruction = (controls.instruction or "").strip()
    if instruction != STATE["instruction"]:
        # A new or cleared operator message. The history is kept, so the message is plain text in the
        # current turn with no extra weight; see "Operator instructions" in docs/gamescaling-kit.md.
        STATE["instruction"] = instruction
        _log(kind="instruction", text=instruction[:200])

    picked = False
    result = BRAIN.take()
    if result and now - result["submitted"] <= MAX_PLAN_AGE_S:
        STATE["plan"] = {"action": result["action"], "origin": list(STATE["pointer"]),
                         "t0": now, "done": -1}
        STATE["plan_started"] = now
        picked = True
        if result["reasoning"]:
            try:
                controls.say(result["reasoning"], final=True)
            except Exception:
                pass

    plan = STATE["plan"]
    if plan:
        index = min(STEPS - 1, int((now - plan["t0"]) / STEP_S))
        for i in range(plan["done"] + 1, index + 1):   # never skip a step: taps still land
            _apply_step(controls, plan, i)
        plan["done"] = index
    if (STATE["keys"] or STATE["buttons"]) and now - STATE["plan_started"] > HOLD_WATCHDOG_S:
        _release(controls, "watchdog")

    # Ask for the next plan on a frame that arrived after the current plan began.
    if not picked and not BRAIN.busy:
        small = _frame_image(frame)
        buffer = io.BytesIO()
        small.to_image().save(buffer, format="JPEG", quality=85)   # already rgb24: no scaler involved
        try:
            controls.show_model_input(small)
        except Exception:
            pass
        BRAIN.submit({"image": base64.b64encode(buffer.getvalue()).decode(),
                      "pointer": [round(v, 3) for v in STATE["pointer"]],
                      "instruction": instruction,
                      "profile": _profile(controls.capabilities), "submitted": now})


def decide(frame, controls):
    try:
        _decide(frame, controls)
    except Exception as exc:
        _log(kind="policy_error", error=type(exc).__name__, detail=str(exc)[:200])
