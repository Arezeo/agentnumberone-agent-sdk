"""Fast vision loop: pipelining, action chunks, pointer actions, captions, trace, sheet shape, decide()."""
import threading
import time
import unittest

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fast_loop import Answer, SimpleConfig, SimplePolicy, parse_slot, slot_text  # noqa: E402
from sheet import Tile, build  # noqa: E402


class FakeControls:
    def __init__(self):
        self.pressed = []

    def keyDown(self, key):
        self.pressed.append(key)

    def keyUp(self, key):
        pass

    def releaseAll(self):
        pass


class Scripted(SimplePolicy):
    """Replaces the HTTP call: each request pops (delay_s, key) from a script."""

    def __init__(self, config, script):
        super().__init__(config)
        self.script = list(script)
        self.lock = threading.Lock()

    def _call(self, sheet_png, started, call_id=0):
        with self.lock:
            delay, key = self.script.pop(0)
        time.sleep(delay)
        return Answer(key=key, confidence=0.9, requested_at=started,
                      finished_at=time.monotonic(), ms=int(delay * 1000), call_id=call_id,
                      raw_text='{"key":"%s"}' % key)


def cfg(**kw):
    base = dict(frames=3, action_gap_s=0.0, request_gap_s=0.0, max_inflight=1, model="m")
    base.update(kw)
    return SimpleConfig(**base)


def feed(policy):
    policy.window.append((time.monotonic(), np.zeros((4, 4, 3), np.uint8), ""))
    policy.window.append((time.monotonic(), np.zeros((4, 4, 3), np.uint8), ""))


def drain(policy, controls, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        policy.step(controls)
        time.sleep(0.01)


class MaxInflight(unittest.TestCase):
    def test_default_is_one_call_at_a_time(self):
        policy = Scripted(cfg(), [(0.2, "W"), (0.2, "W")])
        feed(policy)
        self.assertTrue(policy.wants_request(time.monotonic()))
        policy.request(b"x")
        self.assertFalse(policy.wants_request(time.monotonic()))

    def test_allows_n_calls_out(self):
        policy = Scripted(cfg(max_inflight=3), [(0.3, "W")] * 3)
        feed(policy)
        for _ in range(3):
            self.assertTrue(policy.wants_request(time.monotonic()))
            policy.request(b"x")
        self.assertFalse(policy.wants_request(time.monotonic()))
        self.assertEqual(policy.stats()["inflight"], 3)

    def test_request_gap_staggers_launches(self):
        policy = Scripted(cfg(max_inflight=3, request_gap_s=0.5), [(0.3, "W")] * 3)
        feed(policy)
        policy.request(b"x")
        self.assertFalse(policy.wants_request(time.monotonic()))

    def test_older_answer_never_overrides_newer(self):
        # First call is slow (A), second is fast (D): D finishes first and is
        # applied; A lands later from an older picture and must be dropped.
        policy = Scripted(cfg(max_inflight=2), [(0.5, "A"), (0.05, "D")])
        controls = FakeControls()
        feed(policy)
        policy.request(b"x")
        time.sleep(0.02)
        policy.request(b"x")
        drain(policy, controls, 0.8)
        self.assertEqual(controls.pressed, ["D"])
        self.assertEqual(policy.counts["superseded"], 1)

    def test_newer_replaces_waiting_pending(self):
        # Both answers land while the gap blocks pressing; only the newest is kept.
        policy = Scripted(cfg(max_inflight=2, action_gap_s=0.4), [(0.05, "A"), (0.1, "D")])
        controls = FakeControls()
        policy.last_action_at = time.monotonic()      # gap is closed for 0.4 s
        feed(policy)
        policy.request(b"x")
        time.sleep(0.01)
        policy.request(b"x")
        drain(policy, controls, 0.8)
        self.assertEqual(controls.pressed, ["D"])

    def test_stale_answer_is_dropped(self):
        policy = Scripted(cfg(max_answer_age_s=0.1, action_gap_s=5.0), [(0.0, "W")])
        controls = FakeControls()
        policy.last_action_at = time.monotonic()      # hold the answer in pending
        feed(policy)
        policy.request(b"x")
        drain(policy, controls, 0.4)
        self.assertEqual(controls.pressed, [])
        self.assertEqual(policy.counts["stale"], 1)

    def test_env_default_stagger(self):
        import os
        old = {k: os.environ.get(k) for k in ("ANO_MODEL_BASE_URL", "ANO_MAX_INFLIGHT", "ANO_REQUEST_GAP_S")}
        try:
            os.environ["ANO_MODEL_BASE_URL"] = "http://x"
            os.environ.pop("ANO_REQUEST_GAP_S", None)
            os.environ["ANO_MAX_INFLIGHT"] = "3"
            config = SimpleConfig.from_env()
            self.assertEqual((config.max_inflight, config.request_gap_s), (3, 0.2))
            os.environ["ANO_MAX_INFLIGHT"] = "1"
            self.assertEqual(SimpleConfig.from_env().request_gap_s, 0.0)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class Captions(unittest.TestCase):
    def test_key_caption_is_the_key_pressed_after_that_frame(self):
        policy = Scripted(cfg(frames=3), [])
        img = np.zeros((4, 4, 3), np.uint8)
        t = time.monotonic()
        policy.window = [(t - 0.6, img, "", 0.0), (t - 0.4, img, "", 0.0), (t - 0.2, img, "", 0.0)]
        policy.seq = 3
        policy.presses = [(t - 2.0, "KeyA"), (t - 0.5, "KeyW")]   # old press, then one after frame 1
        policy.compose()
        self.assertEqual([f["key_after"] for f in policy._composed], ["KeyW", "", ""])


class ChunkScripted(Scripted):
    def _call(self, sheet_png, started, call_id=0):
        with self.lock:
            delay, keys, chicken = self.script.pop(0)
        time.sleep(delay)
        ans = Answer(confidence=0.9, requested_at=started, finished_at=time.monotonic(),
                     ms=int(delay * 1000), call_id=call_id)
        ans.keys = list(keys)
        ans.key = next((k for k in keys if k), "")
        ans.seen = {"chicken": chicken}
        return ans


class Chunks(unittest.TestCase):
    def test_slots_play_on_time(self):
        policy = ChunkScripted(cfg(chunk_ms=400, chunk_slot_ms=100, action_gap_s=0.05),
                               [(0.0, ["KeyW", "", "KeyW", "KeyA"], "standing")])
        controls = FakeControls()
        feed(policy)
        policy.request(b"x")
        drain(policy, controls, 0.6)
        self.assertEqual(controls.pressed, ["KeyW", "KeyW", "KeyA"])
        times = [t for t, _ in policy.presses]
        self.assertGreater(times[1] - times[0], 0.15)   # slot 2 is 200 ms after slot 0

    def test_newer_chunk_cuts_the_old_one(self):
        policy = ChunkScripted(cfg(chunk_ms=800, chunk_slot_ms=200, action_gap_s=0.05, max_inflight=2),
                               [(0.0, ["KeyW", "KeyW", "KeyW", "KeyW"], "standing"),
                                (0.25, ["KeyS", "", "", ""], "standing")])
        controls = FakeControls()
        feed(policy)
        policy.request(b"x")
        time.sleep(0.01)
        policy.request(b"x")
        drain(policy, controls, 1.0)
        # Old chunk got slot 0 (and maybe slot 1) before the newer plan replaced it.
        self.assertEqual(controls.pressed[-1], "KeyS")
        self.assertLessEqual(controls.pressed.count("KeyW"), 2)

    def test_model_keys_are_played_as_given_whatever_it_reports(self):
        # The model's report is recorded, not enforced: it decides the keys.
        policy = ChunkScripted(cfg(chunk_ms=400, chunk_slot_ms=100, action_gap_s=0.05),
                               [(0.0, ["KeyW", "", "", ""], "squashed_flat")])
        controls = FakeControls()
        feed(policy)
        policy.request(b"x")
        drain(policy, controls, 0.6)
        self.assertEqual(controls.pressed, ["KeyW"])

    def test_a_none_plan_stops_the_old_plan(self):
        policy = ChunkScripted(cfg(chunk_ms=800, chunk_slot_ms=200, action_gap_s=0.05, max_inflight=2),
                               [(0.0, ["KeyW", "KeyW", "KeyW", "KeyW"], "standing"),
                                (0.1, ["", "", "", ""], "standing")])
        controls = FakeControls()
        feed(policy)
        policy.request(b"x")
        time.sleep(0.01)
        policy.request(b"x")
        drain(policy, controls, 1.0)
        self.assertEqual(controls.pressed, ["KeyW"])

    def test_parse_chunk_reply(self):
        policy = SimplePolicy(cfg(chunk_ms=1000, chunk_slot_ms=250))
        a = policy._parse('{"chicken":"standing","on":"grass","danger":"none","keys":["W","none","w","D"],'
                          '"confidence":0.7,"why":"gap"}', time.monotonic())
        self.assertEqual((a.error, a.keys, a.seen["on"]), ("", ["KeyW", "", "KeyW", "KeyD"], "grass"))
        dead = policy._parse('{"chicken":"squashed_flat","keys":["W","W","W","W"],"confidence":0.9}', time.monotonic())
        self.assertEqual(dead.keys, ["KeyW"] * 4)   # recorded as seen, not overruled
        self.assertIn('"keys"', policy.prompt)


class MouseControls(FakeControls):
    capabilities = ("keyDown", "keyUp", "moveTo", "mouseDown", "mouseUp", "releaseAll")

    def __init__(self):
        super().__init__()
        self.log = []

    def keyDown(self, key):
        super().keyDown(key)
        self.log.append(("keyDown", key))

    def moveTo(self, x, y):
        self.log.append(("moveTo", round(x, 3), round(y, 3)))

    def mouseDown(self, x, y, button="left"):
        self.log.append(("mouseDown", round(x, 3), round(y, 3), button))

    def mouseUp(self, x, y, button="left"):
        self.log.append(("mouseUp", round(x, 3), round(y, 3), button))


class PointerActions(unittest.TestCase):
    def test_slot_grammar(self):
        actions, bad = parse_slot("moveTo 0.52 0.61; mouseDown 0.52 0.61; mouseUp(0.52, 0.61)")
        self.assertEqual([a["type"] for a in actions], ["moveTo", "mouseDown", "mouseUp"])
        self.assertEqual(bad, [])
        self.assertEqual(slot_text(parse_slot("w")[0]), "KeyW")
        self.assertEqual(parse_slot("none"), ([], []))
        self.assertEqual(parse_slot("click 0.5 0.5")[1], ["click 0.5 0.5"])   # not guessed at
        self.assertEqual(parse_slot([{"type": "moveTo", "x": 2, "y": -1}])[0][0], {"type": "moveTo", "x": 1.0, "y": 0.0})

    def test_click_is_sent_as_explicit_sequence_in_one_slot(self):
        policy = SimplePolicy(cfg(chunk_ms=200, chunk_slot_ms=100))
        controls = MouseControls()
        policy.capabilities = frozenset(controls.capabilities)
        answer = policy._parse('{"keys":["moveTo 0.52 0.61; mouseDown 0.52 0.61; mouseUp 0.52 0.61","W"],'
                               '"confidence":0.9,"why":"press play again"}', time.monotonic())
        answer.call_id = 1
        policy.pending = answer
        drain(policy, controls, 0.35)
        self.assertEqual(controls.log, [("moveTo", 0.52, 0.61), ("mouseDown", 0.52, 0.61, "left"),
                                        ("mouseUp", 0.52, 0.61, "left"), ("keyDown", "KeyW")])

    def test_actions_the_game_does_not_offer_are_refused(self):
        policy = SimplePolicy(cfg(chunk_ms=100, chunk_slot_ms=100))
        controls = MouseControls()
        controls.capabilities = ("keyDown", "keyUp")        # keyboard-only game
        answer = policy._parse('{"keys":["moveTo 0.5 0.5; mouseDown 0.5 0.5; mouseUp 0.5 0.5"],"confidence":0.9}',
                               time.monotonic())
        policy.pending = answer
        drain(policy, controls, 0.2)
        self.assertEqual(controls.log, [])

    def test_prompt_lists_only_offered_actions(self):
        policy = SimplePolicy(cfg(chunk_ms=1000, chunk_slot_ms=250))
        policy.capabilities = frozenset({"keyDown", "keyUp"})
        self.assertNotIn("mouseDown x y", policy.prompt_for())
        policy.capabilities = frozenset({"keyDown", "keyUp", "moveTo", "mouseDown", "mouseUp"})
        text = policy.prompt_for("press play again")
        self.assertIn("mouseDown x y", text)
        self.assertNotIn("press play again", text)          # it goes next to the image instead
        content = policy._user_content(b"png", 0, policy.instruction)
        self.assertIn("Operator instruction: press play again", content[0]["text"])

    def test_now_frame_sent_full_size_when_pointer_actions_are_offered(self):
        policy = SimplePolicy(cfg(frames=3, frame_width=640))
        img = np.zeros((720, 1280, 3), np.uint8)
        policy.window = [(time.monotonic(), img, "", 0.0)]
        policy.capabilities = frozenset({"keyDown", "keyUp"})
        self.assertFalse(policy.wants_frame())
        policy.capabilities = frozenset({"keyDown", "keyUp", "moveTo", "mouseDown", "mouseUp"})
        self.assertTrue(policy.wants_frame())
        policy._now_frames[7] = img
        content = policy._user_content(b"png", 7, "")
        self.assertEqual([c["type"] for c in content], ["text", "image_url", "text", "image_url"])
        import base64
        now_png = base64.b64decode(content[3]["image_url"]["url"].split(",", 1)[1])
        self.assertEqual((int.from_bytes(now_png[16:20], "big"), int.from_bytes(now_png[20:24], "big")), (640, 360))


class Trace(unittest.TestCase):
    def test_numbering_continues_after_a_restart(self):
        import tempfile
        from pathlib import Path

        class Board:
            def __init__(self, root):
                self.root = Path(root)

            def log(self, *a):
                pass

        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "trace" / "000041").mkdir(parents=True)
            policy = SimplePolicy(cfg(), board=Board(root))
            self.assertEqual(policy.call_id, 41)

    def test_every_call_gets_an_outcome(self):
        import json
        import tempfile
        from pathlib import Path

        class Board:
            def __init__(self, root):
                self.root = Path(root)

            def log(self, *a):
                pass

        with tempfile.TemporaryDirectory() as root:
            policy = Scripted(cfg(max_inflight=2), [(0.3, "A"), (0.05, "D")])
            policy.board = Board(root)
            policy.trace = type(policy.trace)(Path(root))
            controls = FakeControls()
            feed(policy)
            policy.compose()
            policy.request(b"png1")
            time.sleep(0.02)
            policy.request(b"png2")
            drain(policy, controls, 0.6)
            rows = [json.loads(l) for l in open(Path(root) / "trace.jsonl")]
            outcomes = {r["call"]: r["outcome"] for r in rows}
            self.assertEqual(outcomes[2], "applied")
            self.assertTrue(outcomes[1].startswith("dropped"))
            self.assertEqual((Path(root) / "trace" / "000002" / "sheet.png").read_bytes(), b"png2")
            applied = [r for r in rows if r["outcome"] == "applied"][0]
            self.assertEqual(applied["sent_to_game"], ["keyDown D", "keyUp D"])
            self.assertTrue(applied["applied"])


class PromptFile(unittest.TestCase):
    def test_prompt_file_replaces_the_built_in_prompt(self):
        import os
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write('My game. {slots} slots of {slot_ms} ms. Reply {{"keys": []}}{instruction}')
        os.environ["ANO_PROMPT_FILE"] = handle.name
        try:
            policy = SimplePolicy(cfg(chunk_ms=1000, chunk_slot_ms=250))
            self.assertEqual(policy.prompt_for("go left"),
                             'My game. 4 slots of 250 ms. Reply {"keys": []}\nYou have also been told: go left')
        finally:
            os.environ.pop("ANO_PROMPT_FILE", None)
            os.unlink(handle.name)


class SheetShape(unittest.TestCase):
    def test_tiles_keep_the_frame_aspect(self):
        frame = np.zeros((768, 1024, 3), np.uint8)          # 4:3
        sheet, meta = build([(frame, Tile(seq=i, age_ms=0)) for i in range(6)])
        # 2 rows x 3 cols of 320x240 tiles (+ captions/gutters): not squashed into 16:9
        self.assertEqual(sheet.shape[1], 3 * 324 + 4)
        self.assertEqual(sheet.shape[0], 2 * (240 + 18 + 4) + 4)


class Decide(unittest.TestCase):
    def test_decide_requests_and_plays_a_chunk(self):
        import importlib
        import os
        import tempfile

        class Controls(FakeControls):
            capabilities = ("keyDown", "keyUp", "releaseAll")
            instruction = ""

        with tempfile.TemporaryDirectory() as run:
            env = {"ANO_MODEL_BASE_URL": "http://stub", "ANO_MODEL": "stub", "ANO_RUN_DIR": run,
                   "ANO_CHUNK_MS": "200", "ANO_CHUNK_SLOT_MS": "100", "ANO_ACTION_GAP_S": "0.05",
                   # one request only, so the whole chunk plays (a newer plan would cut it)
                   "ANO_REQUEST_GAP_S": "5"}
            old = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                module = importlib.import_module("policy_fast_vision")
                module._loop = None
                loop = module._get()
                loop._call = lambda png, started, call_id=0: loop._parse(
                    '{"chicken":"standing","on":"grass","danger":"none","keys":["W","D"],'
                    '"confidence":0.8,"why":"clear"}', started)
                controls = Controls()
                frame = np.zeros((48, 64, 3), np.uint8)
                end = time.monotonic() + 0.8
                while time.monotonic() < end:
                    module.decide(frame, controls)
                    time.sleep(0.02)
                self.assertEqual(controls.pressed, ["KeyW", "KeyD"])
                self.assertTrue((Path(run) / "trace").is_dir())
            finally:
                module._loop = None
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
