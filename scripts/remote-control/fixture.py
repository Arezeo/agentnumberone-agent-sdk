"""Deterministic pixel fixture for transport acceptance; this is not a general agent.

Legacy fixture for the original fixed, dark challenge. It does not solve the
current randomized challenge; use transport_smoke.py for the first-task transport
check, or provide a visual policy for a complete run. Recognizes only decoded RGB pixels.
No DOM, task identifiers, target messages or server state is read. Holds span
separate inference calls so video keeps draining during the hold.
"""
import time
import os
import json
import numpy  # Load the pixel conversion dependency before a peer connects.

_state = {"fixture": None, "started": 0, "step": 0}
_last_debug = 0


def classify(pixels):
    """Use region contrast so codec range/color rounding cannot hide dark UI."""
    height, width = pixels.shape[:2]

    def color(x, y):
        px, py = min(width - 1, int(x * width)), min(height - 1, int(y * height))
        patch = pixels[max(0, py - 2):min(height, py + 3), max(0, px - 2):min(width, px + 3)]
        return tuple(float(n) for n in numpy.median(patch, axis=(0, 1)))

    def green(x, y):
        r, g, b = color(x, y + 20 / 720)
        return r > 130 and g > 190 and g > r + 15 and b < 190

    fixture = None
    for name, x, y in (("move", .75, .46), ("click", .3, .57), ("mouse", .68, .52), ("drag", .25, .52)):
        if green(x, y):
            fixture = name
            break
    def contrast(inner, outer):
        return float(numpy.mean(inner) - numpy.mean(outer))

    probes = dict(key=color(560 / 1280, 300 / 720), keyBackground=color(520 / 1280, 300 / 720),
                  keyBottom=color(560 / 1280, 420 / 720), keyBottomBackground=color(520 / 1280, 420 / 720),
                  text=color(400 / 1280, 350 / 720), textBackground=color(300 / 1280, 350 / 720))
    if fixture is None:
        if contrast(probes["key"], probes["keyBackground"]) > 5 and contrast(probes["keyBottom"], probes["keyBottomBackground"]) > 5:
            fixture = "key"
        elif contrast(probes["text"], probes["textBackground"]) > 5:
            fixture = "type"
    return fixture, probes


def decide(frame, controls):
    global _last_debug
    pixels = frame.to_ndarray(format="rgb24")
    fixture, probes = classify(pixels)
    if os.environ.get("ARENA_FIXTURE_DEBUG") == "1" and time.monotonic() - _last_debug > 1:
        _last_debug = time.monotonic()
        print(json.dumps(dict(state="fixture_pixels", fixture=fixture, probes=probes)), flush=True)
    if fixture is None:
        return
    if fixture != _state["fixture"]:
        _state.update(fixture=fixture, started=time.monotonic(), step=0)
    elapsed = time.monotonic() - _state["started"]
    step = _state["step"]
    if fixture == "move" and step == 0:
        controls.moveTo(.75, .46)
    elif fixture == "click" and step == 0:
        controls.click(.3, .57)
    elif fixture == "mouse":
        if step == 0:
            controls.mouseDown(.68, .52)
        elif step == 1 and elapsed >= 1:
            controls.mouseUp(.68, .52)
        else:
            return
    elif fixture == "key":
        if step == 0:
            controls.keyDown("KeyW")
        elif step == 1 and elapsed >= .5:
            controls.keyUp("KeyW")
        else:
            return
    elif fixture == "type" and step == 0:
        controls.click(.5, .52)
        controls.write("hello arena")
        controls.press("Enter")
    elif fixture == "drag" and step == 0:
        controls.dragTo(.25, .52, .76, .52, duration=.3)
    else:
        return
    _state["step"] += 1
