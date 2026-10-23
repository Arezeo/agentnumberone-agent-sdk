"""Small PyAutoGUI-style action builder. Coordinates are normalized, never OS pixels."""
import math
import re
import time

VERSION = "arena.webrtc-control.v1"
MAX_BUFFER = 16384
# Optional text feature: operator instructions in, the agent's own text out.
TEXT_FEATURE = TEXT_CHANNEL = "text"
MAX_INSTRUCTION_CHARS = 500
MAX_AGENT_TEXT_CHARS = 1000
MAX_TEXT_MESSAGE_BYTES = 4096
TEXT_CONTROLS = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]")


def js_length(text):
    """Browser string length (UTF-16 code units), which the page's limits use."""
    return len(text.encode("utf-16-le")) // 2


def agent_text(text):
    """Make a text snapshot the page accepts: no control characters except newline, at most 1000 units."""
    if not isinstance(text, str):
        raise ValueError("Agent text must be a string.")
    text = TEXT_CONTROLS.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n"))
    while js_length(text) > MAX_AGENT_TEXT_CHARS:
        text = text[:-1]
    return text


def instruction_from(message, session):
    """Validate the page's operator instruction; empty text clears it."""
    if not isinstance(message, dict) or set(message) != {"v", "sessionId", "type", "id", "text"}:
        raise ValueError("Unexpected instruction fields.")
    if message["v"] != VERSION or message["sessionId"] != session or message["type"] != "instruction":
        raise ValueError("Wrong session or protocol.")
    if isinstance(message["id"], bool) or not isinstance(message["id"], int) or message["id"] < 1:
        raise ValueError("Invalid instruction id.")
    text = message["text"]
    if not isinstance(text, str) or js_length(text) > MAX_INSTRUCTION_CHARS or TEXT_CONTROLS.search(text):
        raise ValueError("Invalid instruction text.")
    return dict(id=message["id"], text=text)


def unit(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Coordinates must be finite numbers in [0,1].")
    return value


class Controls:
    """Build at most 32 actions during one decision; no networking on model threads.

    `instruction` is the operator's latest instruction ("" when none or the page
    has no text feature). `say` streams the agent's own text to the page.
    """
    def __init__(self, capabilities, instruction="", on_model_input=None):
        self.capabilities = frozenset(capabilities)
        self.instruction = instruction
        self.actions = []
        self.texts = []
        self._on_model_input = on_model_input

    def show_model_input(self, frame):
        """Optional live-view image after preprocessing, as an av.VideoFrame.

        Call immediately before inference. This reports pixels; it sends no input
        action and never supplies observations to the policy.
        """
        if self._on_model_input:
            self._on_model_input(frame)

    def say(self, text, final=True):
        """Send the whole utterance so far; repeat with final=False while it grows."""
        if not isinstance(final, bool):
            raise ValueError("final must be a boolean.")
        if len(self.texts) >= 32:
            raise ValueError("Decision exceeds 32 text snapshots.")
        self.texts.append((agent_text(text), final))

    def _add(self, kind, **fields):
        if kind not in self.capabilities:
            raise ValueError("Action is not supported by this surface: " + kind)
        if len(self.actions) >= 32:
            raise ValueError("Decision exceeds 32 actions.")
        self.actions.append(dict(type=kind, **fields))

    @staticmethod
    def _button(button):
        if button not in ("left", "middle", "right"):
            raise ValueError("Unsupported mouse button.")
        return button

    @staticmethod
    def _key(key):
        if not isinstance(key, str) or not re.fullmatch(r"Key[A-Z]|Digit[0-9]|Semicolon|Quote|Comma|Slash|Period|Arrow(Up|Down|Left|Right)|Space|Enter|Backspace|Delete|Tab|Escape|Home|End|PageUp|PageDown|Shift(Left|Right)|Control(Left|Right)|Alt(Left|Right)", key):
            raise ValueError("Use a supported physical DOM code, for example KeyW.")
        return key

    def moveTo(self, x, y):
        self._add("moveTo", x=unit(x), y=unit(y))

    def click(self, x, y, button="left"):
        self._add("click", x=unit(x), y=unit(y), button=self._button(button))

    def mouseDown(self, x, y, button="left"):
        self._add("mouseDown", x=unit(x), y=unit(y), button=self._button(button))

    def mouseUp(self, x, y, button="left"):
        self._add("mouseUp", x=unit(x), y=unit(y), button=self._button(button))

    def dragTo(self, x, y, toX, toY, duration=0.2, button="left"):
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 <= duration <= 1:
            raise ValueError("Drag duration must be between 0 and 1 seconds.")
        self._add("dragTo", x=unit(x), y=unit(y), toX=unit(toX), toY=unit(toY), durationMs=round(duration * 1000), button=self._button(button))

    def keyDown(self, key):
        self._add("keyDown", key=self._key(key))

    def keyUp(self, key):
        self._add("keyUp", key=self._key(key))

    def press(self, key):
        self._add("press", key=self._key(key))

    def write(self, text):
        if not isinstance(text, str) or len(text.encode("utf-16-le")) // 2 > 128 or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", text):
            raise ValueError("Text must contain at most 128 characters.")
        self._add("write", text=text)

    def scroll(self, x, y, deltaY, deltaX=0):
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 2000 for v in (deltaX, deltaY)):
            raise ValueError("Scroll delta must be finite and within ±2000.")
        self._add("scroll", x=unit(x), y=unit(y), deltaX=deltaX, deltaY=deltaY)

    def releaseAll(self):
        self._add("releaseAll")


class Context:
    """Translate the browser's monotonic clock without assuming synchronized clocks."""
    def __init__(self, message, received=None):
        self.message = dict(message)
        self.received = time.monotonic() if received is None else received

    @property
    def identity(self):
        return self.message.get("taskToken"), self.message.get("generation")

    def host_now(self, now=None):
        return self.message["nowMs"] + ((time.monotonic() if now is None else now) - self.received) * 1000

    def active(self, now=None):
        return bool(self.message.get("active")) and self.host_now(now) < self.message["expiresAtMs"]

    def envelope(self, session, sequence, action, now=None):
        host_now = self.host_now(now)
        if not self.active(now):
            raise ValueError("Task is inactive or has expired.")
        return dict(v=VERSION, sessionId=session, type="input", sequence=sequence,
                    taskToken=self.message["taskToken"], generation=self.message["generation"],
                    expiresAtMs=min(self.message["expiresAtMs"], host_now + 500), action=action)
