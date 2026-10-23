"""Run with the researcher venv: python tests/webrtc-client.test.py.

The media test uses two actual aiortc peers on loopback. It proves decoded
moving frames and control exchange, not browser interoperability or WAN latency.
"""
import asyncio
import json
from pathlib import Path
import sys
import time
import threading
import unittest
import io
from types import SimpleNamespace
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "remote-control"))
from sdk import Controls, Context, VERSION, agent_text, instruction_from
from agent import AgentSession, make_app, origin
from live_view import LiveView, png
import av
from aiohttp.test_utils import TestClient, TestServer
from aiohttp import WSServerHandshakeError
from aiortc import RTCPeerConnection, RTCConfiguration, RTCSessionDescription, VideoStreamTrack
import numpy
import fixture
import transport_smoke


class SDKTests(unittest.TestCase):
    def test_model_preview_does_not_emit_controls(self):
        seen = []
        controls = Controls([], on_model_input=seen.append)
        frame = av.VideoFrame(16, 8, "rgb24")
        controls.show_model_input(frame)
        self.assertEqual(seen, [frame])
        self.assertEqual(controls.actions, [])

    def test_fixture_detects_dark_key_and_text_regions_after_codec_color_shift(self):
        for shift in (0, -10, 10):
            background = numpy.array([23, 23, 19], dtype=numpy.int16) + shift
            key = numpy.array([39, 39, 32], dtype=numpy.int16) + shift
            text = numpy.array([34, 34, 28], dtype=numpy.int16) + shift
            pixels = numpy.zeros((720, 1280, 3), dtype=numpy.uint8)
            pixels[:] = background
            self.assertIsNone(fixture.classify(pixels)[0])
            pixels[280:440, 544:736] = key
            self.assertEqual(fixture.classify(pixels)[0], "key")
            pixels[:] = background
            pixels[334:414, 330:950] = text
            self.assertEqual(fixture.classify(pixels)[0], "type")

    def test_transport_smoke_uses_visible_mint_targets_and_ignores_blank_frames(self):
        class Frame:
            def __init__(self, pixels): self.pixels = pixels
            def to_ndarray(self, format): return self.pixels
        pixels = numpy.zeros((720, 1280, 3), dtype=numpy.uint8)
        pixels[:] = [242, 245, 248]
        controls = Controls(["moveTo"])
        transport_smoke.decide(Frame(pixels), controls)
        self.assertEqual(controls.actions, [])
        pixels[300:340, 500:540] = [200, 232, 209]
        transport_smoke.decide(Frame(pixels), controls)
        self.assertEqual(len(controls.actions), 1)
        self.assertAlmostEqual(controls.actions[0]["x"], 518/1280, places=2)
        self.assertAlmostEqual(controls.actions[0]["y"], 318/720, places=2)

    def test_explicit_mouse_boundaries_and_capabilities(self):
        controls = Controls(["moveTo", "click", "mouseDown", "mouseUp", "dragTo", "scroll", "write"])
        controls.moveTo(.1, .2)
        controls.click(.3, .4)
        controls.mouseDown(.2, .2)
        controls.mouseUp(.8, .8)
        controls.dragTo(.2, .3, .8, .9, duration=.25)
        controls.scroll(.1, .1, 2)
        self.assertEqual(controls.actions[4]["durationMs"], 250)
        self.assertEqual(controls.actions[2]["x"], .2)
        for bad in (float("nan"), float("inf"), -1, 1.1, True):
            with self.assertRaises(ValueError): controls.click(bad, 0)
        with self.assertRaises(ValueError): controls.press("KeyW")
        with self.assertRaises(ValueError): controls.write("a" * 129)
        with self.assertRaises(ValueError): controls.scroll(0, 0, 2001)

    def test_task_clock_does_not_assume_clock_sync(self):
        context = Context(dict(taskToken="task", generation=2, nowMs=100, expiresAtMs=900, active=True), received=2000)
        self.assertAlmostEqual(context.host_now(2000.2), 300)
        envelope = context.envelope("session", 1, dict(type="releaseAll"), now=2000.2)
        self.assertAlmostEqual(envelope["expiresAtMs"], 800)
        self.assertFalse(context.active(2000.9))
        with self.assertRaises(ValueError): context.envelope("s", 2, {}, now=2001)

    def test_queue_is_bounded(self):
        controls = Controls(["releaseAll"])
        for _ in range(32): controls.releaseAll()
        with self.assertRaises(ValueError): controls.releaseAll()

    def test_text_snapshots_and_instructions_match_page_limits(self):
        self.assertEqual(agent_text("a\r\nb\tc\x07"), "a\nb c ")
        self.assertEqual(len(agent_text("x" * 1200)), 1000)
        self.assertEqual(len(agent_text("\U0001F600" * 600)), 500)  # two browser units each
        controls = Controls([], "Keep your distance")
        self.assertEqual(controls.instruction, "Keep your distance")
        controls.say("I", final=False)
        controls.say("I will")
        self.assertEqual(controls.texts, [("I", False), ("I will", True)])
        with self.assertRaises(ValueError): controls.say("x", final="yes")
        message = dict(v=VERSION, sessionId="s", type="instruction", id=1, text="Dodge\nmore")
        self.assertEqual(instruction_from(message, "s"), dict(id=1, text="Dodge\nmore"))
        for bad in (dict(message, sessionId="t"), dict(message, id=0), dict(message, id=True), dict(message, text="x" * 501),
                    dict(message, text="\x00"), dict(message, extra=1)):
            with self.assertRaises(ValueError): instruction_from(bad, "s")

    def test_secure_origins(self):
        self.assertEqual(origin("http://localhost:3016/"), "http://localhost:3016")
        self.assertEqual(origin("https://agentnumber.one"), "https://agentnumber.one")
        for bad in ("http://public.example", "https://user@site.example", "https://site.example/path", "null"):
            with self.assertRaises(ValueError): origin(bad)


class MovingVideo(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.counter = 0

    async def recv(self):
        frame = await super().recv()
        self.counter += 1
        # Change luminance every frame; preserve valid YUV chroma planes.
        frame.planes[0].update(bytes([32 + self.counter % 100]) * frame.planes[0].buffer_size)
        frame.planes[1].update(bytes([128]) * frame.planes[1].buffer_size)
        frame.planes[2].update(bytes([128]) * frame.planes[2].buffer_size)
        return frame


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_configured_age_drops_slow_result_without_disconnecting(self):
        sent = []

        def policy(frame, controls):
            time.sleep(.08)
            controls.click(.1, .2)

        class Channel:
            readyState = "open"
            bufferedAmount = 0
            def send(self, raw): sent.append(json.loads(raw))

        agent = AgentSession(policy, max_frame_age_ms=30)
        agent.session_id, agent.validated = "test", True
        agent.channels = {"motion": Channel(), "actions": Channel()}
        agent.context = Context(dict(taskToken="task", generation=1, active=True, nowMs=100, expiresAtMs=1000, capabilities=["click"]))
        agent.latest = (1, object(), time.monotonic(), agent.context.identity)
        try:
            await asyncio.sleep(.15)
            self.assertEqual(sent, [])
            self.assertEqual(agent.stale_decisions, 1)
            self.assertFalse(agent.closed)
        finally:
            await agent.close()

    async def test_only_explicit_inference_timeout_closes_peer(self):
        def policy(frame, controls):
            time.sleep(.08)
            controls.click(.1, .2)

        agent = AgentSession(policy, inference_timeout_ms=25)
        agent.session_id, agent.validated = "test", True
        agent.context = Context(dict(taskToken="task", generation=1, active=True, nowMs=100, expiresAtMs=1000, capabilities=["click"]))
        agent.latest = (1, object(), time.monotonic(), agent.context.identity)
        try:
            await asyncio.sleep(.12)
            self.assertTrue(agent.closed)
            self.assertEqual(agent.sequence, 0)
        finally:
            await agent.close()

    async def test_cadence_keeps_only_latest_frame_and_one_inflight_call(self):
        frames = []

        def policy(frame, controls):
            frames.append(frame)
            time.sleep(.03)

        agent = AgentSession(policy, decision_interval_ms=100)
        agent.session_id, agent.validated = "test", True
        agent.context = Context(dict(taskToken="task", generation=1, active=True, nowMs=100, expiresAtMs=1000, capabilities=[]))
        agent.latest = (1, "first", time.monotonic(), agent.context.identity)
        try:
            await asyncio.sleep(.05)
            agent.latest = (2, "superseded", time.monotonic(), agent.context.identity)
            await asyncio.sleep(.01)
            agent.latest = (3, "latest", time.monotonic(), agent.context.identity)
            await asyncio.sleep(.11)
            self.assertEqual(frames, ["first", "latest"])
            self.assertFalse(agent.closed)
        finally:
            await agent.close()

    async def test_slow_result_crossing_task_change_is_dropped(self):
        entered = threading.Event()
        sent = []

        def policy(frame, controls):
            entered.set()
            time.sleep(.08)
            controls.click(.1, .2)

        class Channel:
            readyState = "open"
            bufferedAmount = 0
            def send(self, raw): sent.append(json.loads(raw))

        agent = AgentSession(policy)
        agent.session_id, agent.validated = "test", True
        agent.channels = {"motion": Channel(), "actions": Channel()}
        agent.context = Context(dict(taskToken="old", generation=1, active=True, nowMs=100, expiresAtMs=1000, capabilities=["click"]))
        agent.latest = (1, object(), time.monotonic(), agent.context.identity)
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, .5))
            agent.context = Context(dict(taskToken="new", generation=1, active=True, nowMs=200, expiresAtMs=1000, capabilities=["click"]))
            await asyncio.sleep(.12)
            self.assertEqual(sent, [])
            self.assertEqual(agent.stale_decisions, 1)
        finally:
            await agent.close()

    async def test_input_sequence_is_global_across_channels(self):
        sent = []

        def policy(frame, controls):
            controls.moveTo(.1, .2)
            controls.click(.3, .4)
            controls.moveTo(.5, .6)

        class Channel:
            readyState = "open"
            bufferedAmount = 0
            def __init__(self, label): self.label = label
            def send(self, raw): sent.append((self.label, json.loads(raw)))

        agent = AgentSession(policy)
        agent.session_id, agent.validated = "test", True
        agent.channels = {label: Channel(label) for label in ("motion", "actions")}
        agent.context = Context(dict(taskToken="task", generation=1, active=True, nowMs=100, expiresAtMs=1000, capabilities=["click", "moveTo"]))
        agent.latest = (1, object(), time.monotonic(), agent.context.identity)
        try:
            await asyncio.sleep(.08)
            self.assertEqual([(label, value["sequence"]) for label, value in sent], [("motion", 1), ("actions", 2), ("motion", 3)])
        finally:
            await agent.close()

    async def test_websocket_requires_origin_token_and_single_peer(self):
        client = TestClient(TestServer(make_app("http://localhost:3016", "x" * 24, lambda f, c: None)))
        await client.start_server()
        try:
            for headers, path in (({}, "/control?token=" + "x" * 24), ({"Origin": "http://evil.example"}, "/control?token=" + "x" * 24), ({"Origin": "http://localhost:3016"}, "/control?token=bad")):
                with self.assertRaises(WSServerHandshakeError): await client.ws_connect(path, headers=headers)
            socket = await client.ws_connect("/control?token=" + "x" * 24, headers={"Origin": "http://localhost:3016"})
            with self.assertRaises(WSServerHandshakeError):
                await client.ws_connect("/control?token=" + "x" * 24, headers={"Origin": "http://localhost:3016"})
            # Old JPEG/action WS messages are rejected, not silently accepted.
            await socket.send_json(dict(type="frame", v=VERSION, sessionId="s"))
            await socket.receive(timeout=2)
            self.assertTrue(socket.closed or (await socket.receive(timeout=2)).type.name == "CLOSED")
        finally:
            await client.close()

    async def test_real_encoded_video_decodes_before_ready_and_policy_input(self):
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        def slow_policy(frame, controls):
            controls.show_model_input(frame.reformat(width=32, height=18, format="rgb24"))
            time.sleep(.6)
            controls.click(.3, .7)
        agent = AgentSession(slow_policy)
        motion = peer.createDataChannel("motion", ordered=False, maxRetransmits=0)
        actions = peer.createDataChannel("actions", ordered=True)
        peer.addTransceiver(MovingVideo(), direction="sendonly")
        ready = asyncio.Event()
        got_input = asyncio.Event()
        observed = []
        sid = "loopback-test"

        def send(kind, **fields):
            actions.send(json.dumps(dict(v=VERSION, sessionId=sid, type=kind, **fields)))

        @actions.on("message")
        def receive(raw):
            value = json.loads(raw)
            observed.append(value)
            if value["type"] == "ready":
                self.assertGreaterEqual(value["decodedFrames"], 2)
                self.assertGreater(value["videoTime"], 0)
                send("validated")
                ready.set()
            if value["type"] == "input": got_input.set()

        async def contexts():
            while True:
                if actions.readyState == "open":
                    now = time.monotonic() * 1000
                    send("context", nonce="nonce", taskToken="task-a", generation=1, active=True,
                         nowMs=now, expiresAtMs=now + 1000, capabilities=["click", "releaseAll"])
                await asyncio.sleep(.1)

        ticker = None
        try:
            await agent.signal(dict(v=VERSION, sessionId=sid, type="hello"))
            await peer.setLocalDescription(await peer.createOffer())
            answer = await agent.signal(dict(v=VERSION, sessionId=sid, type="offer", sdp=peer.localDescription.sdp))
            await peer.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type="answer"))
            ticker = asyncio.create_task(contexts())
            await asyncio.wait_for(ready.wait(), 10)
            await asyncio.wait_for(got_input.wait(), 5)
            command = next(v for v in observed if v["type"] == "input")
            self.assertEqual(command["action"], dict(type="click", x=.3, y=.7, button="left"))
            self.assertEqual(command["taskToken"], "task-a")
            self.assertTrue(agent.validated)
            # A model slower than 500 ms remains connected while the decoder
            # continuously drains moving frames; no old SDK deadline applies.
            self.assertGreaterEqual(agent.frames, 10)
            self.assertIsNotNone(agent.policy_input)
            self.assertEqual((agent.model_input[1].width, agent.model_input[1].height), (32, 18))
            self.assertGreaterEqual(agent.last_policy_ms, 500)
            observer = LiveView()
            observer.attach(agent)
            self.assertEqual(observer.state()["input_kind"], "model")
            self.assertGreater(observer.state()["received_fps"], 0)
            self.assertFalse(motion.ordered)
            print(json.dumps(dict(test="aiortc_loopback", decodedFrames=agent.frames,
                                  firstDecodedFrameMs=round((agent.first_frame_at - agent.started_at) * 1000, 1),
                                  action=command["action"]["type"])))
        finally:
            if ticker:
                ticker.cancel()
                await asyncio.gather(ticker, return_exceptions=True)
            await agent.close()
            await peer.close()

    async def test_text_channel_opens_only_when_advertised_and_carries_instruction_and_text(self):
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        seen = []

        def policy(frame, controls):
            seen.append(controls.instruction)
            if controls.instruction:
                controls.say("Moving " + controls.instruction.lower(), final=False)
                controls.say("Moving " + controls.instruction.lower())

        agent = AgentSession(policy)
        peer.createDataChannel("motion", ordered=False, maxRetransmits=0)
        actions = peer.createDataChannel("actions", ordered=True)
        peer.addTransceiver(MovingVideo(), direction="sendonly")
        sid = "text-loopback"
        opened, got_text = asyncio.Event(), asyncio.Event()
        texts, channels = [], []
        features = []

        def send(kind, **fields):
            actions.send(json.dumps(dict(v=VERSION, sessionId=sid, type=kind, **fields)))

        @actions.on("message")
        def receive(raw):
            if json.loads(raw)["type"] == "ready":
                send("validated")

        @peer.on("datachannel")
        def on_channel(channel):
            channels.append(channel)
            self.assertEqual((channel.label, channel.ordered, channel.maxRetransmits), ("text", True, None))

            @channel.on("message")
            def on_text(raw):
                value = json.loads(raw)
                texts.append(value)
                if value["final"]:
                    got_text.set()
            opened.set()

        async def contexts():
            while True:
                if actions.readyState == "open":
                    now = time.monotonic() * 1000
                    send("context", nonce="nonce", taskToken="task", generation=1, active=True, nowMs=now,
                         expiresAtMs=now + 1000, capabilities=["click"], **({"features": features} if features else {}))
                await asyncio.sleep(.05)

        ticker = None
        try:
            await agent.signal(dict(v=VERSION, sessionId=sid, type="hello"))
            await peer.setLocalDescription(await peer.createOffer())
            answer = await agent.signal(dict(v=VERSION, sessionId=sid, type="offer", sdp=peer.localDescription.sdp))
            await peer.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type="answer"))
            ticker = asyncio.create_task(contexts())
            await asyncio.sleep(.6)
            self.assertEqual(channels, [])  # old pages never see the channel
            features.append("text")
            await asyncio.wait_for(opened.wait(), 5)
            text = channels[0]
            while text.readyState != "open":
                await asyncio.sleep(.01)
            text.send(json.dumps(dict(v=VERSION, sessionId=sid, type="instruction", id=1, text="Left")))
            text.send(json.dumps(dict(v=VERSION, sessionId=sid, type="instruction", id=1, text="replayed id")))
            await asyncio.wait_for(got_text.wait(), 5)
            self.assertEqual(agent.instruction, dict(id=1, text="Left"))
            self.assertIn("Left", seen)
            self.assertEqual(texts[-1], dict(v=VERSION, sessionId=sid, type="text", utterance=1, text="Moving left", final=True))
            self.assertEqual(len(channels), 1)
            self.assertFalse(agent.closed)
        finally:
            if ticker:
                ticker.cancel()
                await asyncio.gather(ticker, return_exceptions=True)
            await agent.close()
            await peer.close()


class LiveViewTests(unittest.IsolatedAsyncioTestCase):
    async def test_observer_serves_distinct_frames_metrics_and_rejects_remote_access(self):
        from aiohttp import web
        now = time.monotonic()
        received = av.VideoFrame.from_ndarray(numpy.full((16, 32, 3), [220, 20, 20], dtype=numpy.uint8), format="rgb24")
        processed = av.VideoFrame.from_ndarray(numpy.full((8, 16, 3), [20, 20, 220], dtype=numpy.uint8), format="rgb24")
        observer = LiveView()
        session = SimpleNamespace(latest=(9, received, now, None), policy_input=(7, received, now-.2, None), model_input=None,
            closed=False, pc=SimpleNamespace(connectionState="connected"), validated=True,
            context=SimpleNamespace(active=lambda: True), frame_times=deque([now-.1, now]),
            decision_times=deque([now-1, now]), last_policy_ms=123, sequence=4, stale_decisions=2,
            input_selected_at=now-.2, session_id="test-session")
        observer.attach(session)
        app = web.Application()
        observer.install(app)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.get("/live")
            self.assertEqual(response.status, 200)
            self.assertIn("Policy input", await response.text())
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            state = await (await client.get("/live/state")).json()
            self.assertEqual((state["received_frame"], state["last_model_frame"], state["input_kind"]), (9, 7, "policy"))
            self.assertEqual((state["last_policy_ms"], state["commands_queued"]), (123, 4))
            session.model_input = (7, processed, now-.2, None)
            response = await client.get("/live/input.png")
            self.assertEqual(response.headers["X-Frame-Id"], "7")
            image = next(av.open(io.BytesIO(await response.read())).decode(video=0))
            self.assertEqual((image.width, image.height), (16, 8))
            pixels = image.to_ndarray(format="rgb24")
            self.assertTrue(numpy.all(pixels == [20, 20, 220]), "show the actual processed pixels losslessly")
            session.latest = (10, received, now, None)
            state = await (await client.get("/live/state")).json()
            self.assertEqual((state["received_frame"], state["last_model_frame"], state["input_kind"]), (10, 7, "model"))
            for path in ("/live", "/live/state", "/live/input.png", "/live/received.png"):
                self.assertEqual((await client.get(path, headers={"Host": "public.example"})).status, 403)
                self.assertEqual((await client.get(path, headers={"Origin": "https://evil.example"})).status, 403)
            with self.assertRaises(web.HTTPForbidden):
                observer.local_only(SimpleNamespace(host="localhost:8765", remote="10.0.0.2", headers={}))
            observer.attach(SimpleNamespace(**{**session.__dict__, "latest": None, "policy_input": None, "model_input": None}))
            self.assertEqual((await client.get("/live/input.png")).status, 404)
        finally:
            await client.close()

    async def test_reference_viewer_is_opt_in(self):
        for enabled, status in ((False, 404), (True, 200)):
            client = TestClient(TestServer(make_app("http://localhost:3016", "x"*24, lambda f,c: None, live_view=enabled)))
            await client.start_server()
            try:
                self.assertEqual((await client.get("/live")).status, status)
            finally:
                await client.close()


if __name__ == "__main__":
    unittest.main()
