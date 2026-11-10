"""llama.cpp policy contracts: timed playback, holds, history and sampling. No server or weights."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/remote-control"))
sys.path.insert(0, str(ROOT / "GameScaling0.8B-S3"))
from sdk import Controls  # noqa: E402
from PIL import Image  # noqa: E402

CAPS = ["moveTo", "click", "mouseDown", "mouseUp", "keyDown", "keyUp", "scroll", "releaseAll"]
DUEL_CAPS = ["moveTo", "mouseDown", "mouseUp", "keyDown", "keyUp", "releaseAll"]


def load(**env):
    """A fresh policy module, so module state never leaks between tests."""
    scratch = tempfile.mkdtemp(prefix="llama-policy-test-")
    os.environ["ANO_LLAMA_LOG"] = str(Path(scratch) / "decisions.jsonl")
    for key in ("ANO_LLAMA_GAME", "ANO_LLAMA_TEMPERATURE", "ANO_LLAMA_HISTORY", "ANO_LLAMA_NOOP_RESET"):
        os.environ.pop(key, None)
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location("llama_policy_under_test", ROOT / "GameScaling0.8B-S3/llama_policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.print = lambda *args, **kwargs: None     # the decision log is not test output
    return module


class Frame:
    def reformat(self, width, height, format):
        self.size = (width, height)
        return self

    def to_image(self):
        return Image.new("RGB", self.size, (30, 60, 90))


def reply(content, reasoning=""):
    return {"choices": [{"message": {"content": content, "reasoning_content": reasoning}}],
            "timings": {"prompt_n": 1, "cache_n": 2, "predicted_n": 3, "predicted_per_second": 4.0}}


def job(policy, image="aGk=", instruction="", profile="general"):
    return {"image": image, "pointer": [0.5, 0.5], "instruction": instruction, "profile": profile, "submitted": 0.0}


class PlaybackTests(unittest.TestCase):
    def run_plan(self, text, times, capabilities=CAPS):
        policy = load()
        clock = [100.0]
        policy.CLOCK = lambda: clock[0]
        policy.BRAIN.busy = True            # never start the model thread in this test
        policy.BRAIN.result = {"action": policy.parse_action(text), "reasoning": "", "submitted": 100.0}
        sent = []
        for offset in times:
            clock[0] = 100.0 + offset
            controls = Controls(capabilities, "")
            policy.decide(Frame(), controls)
            sent.append([(a["type"], a.get("key") or a.get("button") or (round(a.get("x", 0), 4), round(a.get("y", 0), 4)))
                         for a in controls.actions])
        return policy, sent

    def test_six_steps_play_in_time_and_the_pointer_moves_progressively(self):
        text = "<|action_start|>60 -30 0 ; W ; W LB ; W LB ; W ; ; D<|action_end|>"
        policy, sent = self.run_plan(text, [0, .034, .067, .101, .134, .167])
        self.assertEqual(sent[0], [("moveTo", (0.51, 0.495)), ("keyDown", "KeyW")])
        self.assertEqual(sent[1], [("moveTo", (0.52, 0.49)), ("mouseDown", "left")])
        self.assertEqual(sent[2], [("moveTo", (0.53, 0.485))])
        self.assertEqual(sent[3], [("moveTo", (0.54, 0.48)), ("mouseUp", "left")])
        self.assertEqual(sent[4], [("moveTo", (0.55, 0.475)), ("keyUp", "KeyW")])
        self.assertEqual(sent[5], [("moveTo", (0.56, 0.47)), ("keyDown", "KeyD")])
        self.assertEqual(policy.STATE["keys"], {"KeyD"})

    def test_a_late_call_still_plays_every_step_so_taps_land(self):
        text = "<|action_start|>0 0 0 ; ; LB ; ; ; ; <|action_end|>"
        _, sent = self.run_plan(text, [0, .19])
        self.assertEqual([name for name, _ in sent[1]], ["mouseDown", "mouseUp"])

    def test_last_step_is_held_until_the_watchdog_releases_it(self):
        text = "<|action_start|>0 0 0 ; W ; W ; W ; W ; W ; W<|action_end|>"
        policy, sent = self.run_plan(text, [0, .2, 1.0, 1.9, 3.1])
        self.assertEqual(sent[1:4], [[], [], []])
        self.assertEqual([name for name, _ in sent[4]], ["keyUp", "releaseAll"])
        self.assertEqual(policy.STATE["keys"], set())

    def test_training_key_names_are_case_sensitive(self):
        policy = load()
        self.assertEqual((policy.KEYS["one"], policy.KEYS["One"]), ("Digit1", "F1"))
        self.assertEqual((policy.KEYS["Esc"], policy.KEYS["Back"], policy.KEYS["Up"]), ("Escape", "Backspace", "ArrowUp"))

    def test_an_unsupported_control_is_skipped_without_raising(self):
        text = "<|action_start|>0 0 2 ; One ; ; ; ; ; <|action_end|>"
        _, sent = self.run_plan(text, [0], capabilities=DUEL_CAPS)
        self.assertEqual(sent[0], [])


class ConversationTests(unittest.TestCase):
    def brain(self, answers, **env):
        policy = load(**env)
        payloads = []

        def post(payload):
            payloads.append(payload)
            return reply(answers[min(len(payloads), len(answers)) - 1])
        policy._post = post
        return policy, payloads

    def test_samples_at_temperature_one_and_carries_the_operator_instruction(self):
        policy, payloads = self.brain(["<|action_start|>1 0 0 ; W ; ; ; ; ; <|action_end|>"])
        policy.BRAIN.think(job(policy, instruction="Go left"))
        self.assertEqual(payloads[0]["temperature"], 1.0)
        text = payloads[0]["messages"][-1]["content"][0]["text"]
        self.assertIn("Current commanded pointer (unverified): [0.5, 0.5]", text)
        self.assertTrue(text.endswith("Operator instruction: Go left"))

    def test_history_only_grows_then_clears_so_the_prompt_prefix_is_reused(self):
        policy, payloads = self.brain(["<|action_start|>1 0 0 ; W ; ; ; ; ; <|action_end|>"], ANO_LLAMA_HISTORY="3")
        for index in range(5):
            policy.BRAIN.think(job(policy, image="img%d" % index))
        images = [[part["image_url"]["url"][-4:] for m in p["messages"] if m["role"] == "user"
                   for part in m["content"] if part["type"] == "image_url"] for p in payloads]
        self.assertEqual([len(i) for i in images], [1, 2, 3, 1, 2])
        self.assertEqual(images[2][:2], images[1])       # earlier turns are an unchanged prefix
        self.assertEqual(payloads[2]["messages"][2], {"role": "assistant", "content": "<|action_start|>1 0 0 ; W ; ; ; ; ; <|action_end|>"})

    def test_two_empty_actions_clear_the_history(self):
        policy, payloads = self.brain(["<|action_start|>0 0 0 ; ; ; ; ; ; <|action_end|>"])
        for _ in range(3):
            policy.BRAIN.think(job(policy))
        self.assertEqual([len(p["messages"]) for p in payloads], [2, 4, 2])

    def test_unparseable_reply_is_stored_as_an_explicit_empty_action(self):
        policy, payloads = self.brain(["I am not sure what to do."])
        result = policy.BRAIN.think(job(policy))
        self.assertFalse(result["action"].parsed)
        self.assertTrue(policy.parse_action(policy.BRAIN.history[0][2]).is_noop)

    def test_prompt_follows_the_game_not_one_fixed_title(self):
        policy = load()
        self.assertEqual(policy._profile(frozenset(DUEL_CAPS)), "pixel-duel")
        self.assertEqual(policy._profile(frozenset(CAPS)), "general")
        self.assertIn("Voxelcraft", policy.PROMPTS["voxelcraft"])
        self.assertNotIn("Pixel Duel", policy.PROMPTS["general"])
        self.assertEqual(load(ANO_LLAMA_GAME="voxelcraft")._profile(frozenset(DUEL_CAPS)), "voxelcraft")

    def test_a_changed_instruction_drops_the_old_frames(self):
        policy = load()
        policy.BRAIN.busy = True
        policy.decide(Frame(), Controls(CAPS, "Jump now"))
        self.assertTrue(policy.BRAIN.reset)


if __name__ == "__main__":
    unittest.main()
