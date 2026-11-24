"""decide(frame, controls) for the SDK reference runner, backed by a hosted vision model.

    ANO_MODEL_BASE_URL=https://<provider>/v1 ANO_MODEL=<vision model> ANO_MODEL_API_KEY=... \
    python scripts/remote-control/agent.py --origin https://agentnumberone.io --live-view \
        --decision-interval-ms 250 --inference-timeout-ms 3000 \
        --policy packages/agent-sdk-fast-vision/policy_fast_vision.py

Each tick is cheap: it stores the frame, plays any action-chunk slot that is
due, and -- when a model slot is free -- tiles the last frames into one image
and hands it to a worker thread. Model latency never blocks the tick.

See README.md for every setting and for how to read the trace.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fast_loop import SimpleConfig, SimplePolicy  # noqa: E402


class RunDir:
    """Where traces and the event log go (ANO_RUN_DIR, default ./runs/fast-vision)."""

    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def log(self, name: str, entry: dict) -> None:
        with open(self.root / name, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(entry, ts=time.time()), separators=(",", ":")) + "\n")


_loop = None


def _get() -> SimplePolicy:
    global _loop
    if _loop is None:
        config = SimpleConfig.from_env()
        if config is None or not config.model:
            raise RuntimeError("Set ANO_MODEL_BASE_URL (an OpenAI-compatible /v1 endpoint) and ANO_MODEL.")
        run = RunDir(os.environ.get("ANO_RUN_DIR", os.path.join(os.getcwd(), "runs", "fast-vision")))
        _loop = SimplePolicy(config, board=run)
        mode = (f"chunks of {_loop.slots} x {config.chunk_slot_ms} ms" if config.chunk_ms > 0 else "one key per call")
        print(f"[fast-vision] {config.model}: {config.frames} frames -> one image, {mode}, "
              f"{config.max_inflight} call(s) in flight; trace in {run.root}", flush=True)
    return _loop


def decide(frame, controls):
    loop = _get()
    now = time.monotonic()
    capabilities = frozenset(getattr(controls, "capabilities", ()) or ())
    # The prompt lists only the actions this game offers; the operator's
    # instruction travels with the image.
    loop.capabilities = capabilities
    loop.prompt = loop.prompt_for(getattr(controls, "instruction", "") or "")
    loop.observe(frame)
    loop.step(controls)
    if loop.wants_request(now) and capabilities & {"keyDown", "keyUp", "moveTo", "mouseDown", "mouseUp"}:
        sheet_png, _ = loop.compose()
        loop.request(sheet_png)
        _show(controls, getattr(loop, "last_sheet", None))


def _show(controls, sheet) -> None:
    """Report the exact image sent to the model in the SDK live view (/live)."""
    if sheet is None or not hasattr(controls, "show_model_input"):
        return
    try:
        from av import VideoFrame

        controls.show_model_input(VideoFrame.from_ndarray(sheet, format="rgb24"))
    except Exception:
        pass
