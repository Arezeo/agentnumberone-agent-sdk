"""Researcher endpoint: WebSocket signaling, WebRTC video and input, optional text."""
import argparse
import asyncio
import importlib.util
import json
import math
import os
import secrets
import ssl
import time
from collections import deque
from urllib.parse import urlsplit

from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.sdp import candidate_from_sdp

from sdk import Controls, Context, VERSION, MAX_BUFFER, MAX_TEXT_MESSAGE_BYTES, TEXT_CHANNEL, TEXT_FEATURE, agent_text, instruction_from
from live_view import LiveView
from av import VideoFrame


def origin(value):
    parsed = urlsplit(value)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment
            or (parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"))):
        raise ValueError("Use an HTTPS page origin, or HTTP loopback origin.")
    return value.rstrip("/")


def load_policy(path):
    spec = importlib.util.spec_from_file_location("arena_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "decide", None)):
        raise ValueError("Policy must export decide(frame, controls).")
    return module.decide


class AgentSession:
    def __init__(self, policy, ice_servers=None, inference_timeout_ms=30000, decision_interval_ms=100, max_frame_age_ms=5000):
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers or []))
        self.policy = policy
        self.inference_timeout_ms = inference_timeout_ms
        self.decision_interval_ms = decision_interval_ms
        self.max_frame_age_ms = max_frame_age_ms
        self.session_id = None
        self.hello = None
        self.context = None
        self.channels = {}
        self.sequence = 0
        self.latest = None
        self.policy_input = None
        self.model_input = None
        self.input_selected_at = None
        self.last_policy_ms = None
        self.frame_times = deque(maxlen=600)
        self.decision_times = deque(maxlen=600)
        self.frames = 0
        self.stale_decisions = 0
        self.ready_sent = False
        self.validated = False
        self.closed = False
        self.on_closed = None
        self.tasks = set()
        self.first_frame_at = None
        self.last_video_time = 0
        self.started_at = time.monotonic()
        self.signal_count = 0
        self.text_channel = None
        self.instruction = None
        self.utterance = 0
        self.text_pending = {}
        self.pc.on("track", self.on_track)
        self.pc.on("datachannel", self.on_channel)
        self.pc.on("connectionstatechange", self.on_connection)
        self.spawn(self.inference())
        self.spawn(self.text_pump())

    def spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def on_connection(self):
        if self.pc.connectionState in ("closed", "failed", "disconnected"):
            await self.close()

    def on_track(self, track):
        if track.kind != "video":
            self.spawn(self.close())
        else:
            self.spawn(self.receive_video(track))

    async def receive_video(self, track):
        try:
            while not self.closed:
                frame = await track.recv()
                self.frames += 1
                self.latest = (self.frames, frame, time.monotonic(), self.context.identity if self.context else None)
                self.frame_times.append(self.latest[2])
                self.last_video_time = float(frame.time or 0)
                if self.first_frame_at is None:
                    self.first_frame_at = time.monotonic()
                self.maybe_ready()
        except asyncio.CancelledError:
            raise
        except Exception:
            await self.close()

    def on_channel(self, channel):
        valid = ((channel.label == "motion" and not channel.ordered and channel.maxRetransmits == 0 and channel.maxPacketLifeTime is None)
                 or (channel.label == "actions" and channel.ordered and channel.maxRetransmits is None and channel.maxPacketLifeTime is None))
        if not valid or channel.label in self.channels:
            channel.close()
            self.spawn(self.close())
            return
        self.channels[channel.label] = channel
        channel.on("message", lambda raw: self.on_message(channel.label, raw))
        channel.on("close", lambda: self.spawn(self.close()))
        channel.on("open", self.maybe_ready)

    def send(self, channel, message):
        target = self.channels.get(channel)
        if target is None or target.readyState != "open" or target.bufferedAmount > MAX_BUFFER:
            return False
        raw = json.dumps(dict(v=VERSION, sessionId=self.session_id, **message), separators=(",", ":"), allow_nan=False)
        if len(raw.encode()) > MAX_BUFFER:
            return False
        target.send(raw)
        return True

    def on_message(self, channel, raw):
        if channel != "actions" or not isinstance(raw, str) or len(raw.encode()) > MAX_BUFFER:
            return
        try:
            message = json.loads(raw)
            if message.get("v") != VERSION or message.get("sessionId") != self.session_id:
                return
            if message.get("type") == "context":
                if not all(isinstance(message.get(k), (int, float)) and not isinstance(message.get(k), bool) and math.isfinite(message[k]) for k in ("nowMs", "expiresAtMs", "generation")):
                    return
                if not isinstance(message.get("taskToken"), str) or not isinstance(message.get("capabilities"), list):
                    return
                self.context = Context(message)
                features = message.get("features")
                if isinstance(features, list) and TEXT_FEATURE in features:
                    self.open_text()
                self.maybe_ready()
            elif message.get("type") == "validated":
                self.validated = True
                print(json.dumps(dict(state="validated", decodedFrames=self.frames)), flush=True)
            elif message.get("type") == "error":
                # Errors are diagnostics; successful actions have no per-action ack.
                print(json.dumps(dict(state="control_error", code=message.get("code"))), flush=True)
        except (ValueError, TypeError, AttributeError):
            return

    def open_text(self):
        """Open the optional text channel once the page advertises it; old pages never see it."""
        if self.text_channel is not None or self.closed:
            return
        channel = self.text_channel = self.pc.createDataChannel(TEXT_CHANNEL, ordered=True)
        channel.on("message", self.on_text_message)

    def on_text_message(self, raw):
        if not isinstance(raw, str) or len(raw.encode()) > MAX_TEXT_MESSAGE_BYTES:
            return
        try:
            instruction = instruction_from(json.loads(raw), self.session_id)
        except (ValueError, TypeError):
            return
        if self.instruction and instruction["id"] <= self.instruction["id"]:
            return
        self.instruction = instruction
        self.on_instruction(instruction)

    def on_instruction(self, instruction):
        print(json.dumps(dict(state="instruction", id=instruction["id"], chars=len(instruction["text"]))), flush=True)

    @property
    def instruction_text(self):
        return self.instruction["text"] if self.instruction else ""

    def say(self, text, final=True):
        """Queue the latest snapshot of the current utterance; at most 10 messages per second are sent."""
        if self.text_channel is None:
            return None
        utterance = self.utterance + 1
        self.text_pending[utterance] = (agent_text(text), bool(final))
        if final:
            self.utterance = utterance
        while len(self.text_pending) > 4:
            self.text_pending.pop(min(self.text_pending))
        return utterance

    def send_text(self, utterance, text, final):
        channel = self.text_channel
        if channel is None or channel.readyState != "open" or channel.bufferedAmount > MAX_BUFFER:
            return False
        while True:
            raw = json.dumps(dict(v=VERSION, sessionId=self.session_id, type="text", utterance=utterance, text=text, final=final),
                             separators=(",", ":"), ensure_ascii=False)
            if len(raw.encode()) <= MAX_TEXT_MESSAGE_BYTES:
                break
            text = text[:-16]
        channel.send(raw)
        return True

    async def text_pump(self):
        try:
            while not self.closed:
                await asyncio.sleep(0.1)
                if self.text_pending:
                    utterance = min(self.text_pending)
                    if self.send_text(utterance, *self.text_pending[utterance]):
                        del self.text_pending[utterance]
        except asyncio.CancelledError:
            raise

    def maybe_ready(self):
        if (not self.ready_sent and self.context and self.frames >= 2 and self.last_video_time > 0
                and all(self.channels.get(k) and self.channels[k].readyState == "open" for k in ("motion", "actions"))):
            self.ready_sent = self.send("actions", dict(type="ready", nonce=self.context.message["nonce"], decodedFrames=self.frames, videoTime=self.last_video_time))

    async def inference(self):
        last = 0
        next_decision_at = 0
        try:
            while not self.closed:
                await asyncio.sleep(0.01)
                context, latest = self.context, self.latest
                if not self.validated or not context or not context.active() or not latest or latest[0] == last or latest[3] != context.identity:
                    continue
                if time.monotonic() < next_decision_at:
                    continue
                last = latest[0]
                next_decision_at = time.monotonic() + self.decision_interval_ms / 1000
                self.policy_input, self.model_input = latest, None
                self.input_selected_at = time.monotonic()
                def show_model_input(frame, observation=latest):
                    if not isinstance(frame, VideoFrame):
                        raise ValueError("show_model_input requires an av.VideoFrame.")
                    if not self.closed and self.policy_input is observation:
                        self.model_input = (observation[0], frame, observation[2], observation[3])
                controls = Controls(context.message["capabilities"], self.instruction_text, show_model_input)
                # Do not accumulate model jobs: a timeout closes this session and
                # the outstanding thread finishes in isolation without send access.
                try:
                    await asyncio.wait_for(asyncio.to_thread(self.policy, latest[1], controls), self.inference_timeout_ms / 1000)
                except asyncio.TimeoutError:
                    print(json.dumps(dict(state="policy_timeout")), flush=True)
                    await self.close()
                    return
                self.last_policy_ms = (time.monotonic() - self.input_selected_at) * 1000
                self.decision_times.append(time.monotonic())
                for text, final in controls.texts:
                    self.say(text, final)
                if not self.context or self.context.identity != context.identity or not self.context.active() or (time.monotonic() - latest[2]) * 1000 > self.max_frame_age_ms:
                    self.stale_decisions += 1
                    continue
                for action in controls.actions:
                    current = self.context
                    if current.identity != context.identity or not current.active():
                        break
                    channel = "motion" if action["type"] == "moveTo" else "actions"
                    sequence = self.sequence + 1
                    envelope = current.envelope(self.session_id, sequence, action)
                    if self.send(channel, {k: v for k, v in envelope.items() if k not in ("v", "sessionId")}):
                        self.sequence = sequence
        except asyncio.CancelledError:
            raise
        except Exception as error:
            print(json.dumps(dict(state="policy_failed", error=type(error).__name__)), flush=True)
            await self.close()

    async def signal(self, message):
        if not isinstance(message, dict) or message.get("v") != VERSION:
            raise ValueError("Unsupported signaling version.")
        if self.session_id is None:
            if message.get("type") != "hello" or not isinstance(message.get("sessionId"), str) or len(message["sessionId"]) > 128:
                raise ValueError("Expected hello.")
            self.session_id, self.hello = message["sessionId"], message
            return None
        if message.get("sessionId") != self.session_id:
            raise ValueError("Wrong session.")
        if message.get("type") == "offer" and self.pc.remoteDescription is None:
            await self.pc.setRemoteDescription(RTCSessionDescription(sdp=message["sdp"], type="offer"))
            await self.pc.setLocalDescription(await self.pc.createAnswer())
            return dict(v=VERSION, sessionId=self.session_id, type="answer", sdp=self.pc.localDescription.sdp)
        if message.get("type") == "ice":
            self.signal_count += 1
            if self.signal_count > 256:
                raise ValueError("Too many ICE candidates.")
            value = message.get("candidate")
            candidate = None
            if value is not None:
                candidate = candidate_from_sdp(value["candidate"].removeprefix("candidate:"))
                candidate.sdpMid, candidate.sdpMLineIndex = value.get("sdpMid"), value.get("sdpMLineIndex")
            await self.pc.addIceCandidate(candidate)
            return None
        raise ValueError("Signaling accepts hello, offer and ICE only.")

    async def close(self):
        if self.closed:
            return
        self.closed = True
        current = asyncio.current_task()
        for task in tuple(self.tasks):
            if task is not current:
                task.cancel()
        await self.pc.close()
        if self.on_closed:
            await self.on_closed()
        print(json.dumps(dict(state="closed", decodedFrames=self.frames, staleDecisions=self.stale_decisions,
                              firstDecodedFrameMs=None if self.first_frame_at is None else round((self.first_frame_at - self.started_at) * 1000, 1))), flush=True)


def make_app(allowed_origin, token, policy, ice_servers=None, inference_timeout_ms=30000, decision_interval_ms=100, max_frame_age_ms=5000, live_view=False):
    app = web.Application(client_max_size=65536)
    sessions = set()
    observer = LiveView() if live_view else None
    if observer:
        observer.install(app)

    async def connect(request):
        if request.headers.get("Origin") != allowed_origin or not secrets.compare_digest(request.query.get("token", ""), token):
            raise web.HTTPForbidden(text="Origin or pairing token rejected.")
        if sessions:
            raise web.HTTPConflict(text="One peer is already paired.")
        # Reserve the slot before the first await to prevent concurrent pairing.
        reservation = object()
        sessions.add(reservation)
        session = None
        socket = web.WebSocketResponse(max_msg_size=65536, compress=False, heartbeat=15)
        try:
            await socket.prepare(request)
            session = AgentSession(policy, ice_servers, inference_timeout_ms, decision_interval_ms, max_frame_age_ms)
            if observer:
                observer.attach(session)
            session.on_closed = socket.close
            async for message in socket:
                if message.type != WSMsgType.TEXT:
                    break
                try:
                    response = await session.signal(json.loads(message.data))
                    if response:
                        await socket.send_json(response)
                except (ValueError, KeyError, TypeError, AttributeError):
                    await socket.close(code=1008, message=b"Invalid signaling")
                    break
        finally:
            sessions.discard(reservation)
            if session:
                await session.close()
        return socket

    app.router.add_get("/control", connect)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True, type=origin)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--policy", default=os.path.join(os.path.dirname(__file__), "noop.py"))
    parser.add_argument("--live-view", action="store_true", help="Serve read-only received/policy/model images and metrics at /live (localhost only).")
    parser.add_argument("--decision-interval-ms", "--decision-ms", dest="decision_interval_ms", type=int, default=100,
                        help="Minimum interval between inference starts; does not limit model runtime (default: 100).")
    parser.add_argument("--inference-timeout-ms", type=int, default=30000,
                        help="Hard deadline for one model call; a timeout closes the peer (default: 30000).")
    parser.add_argument("--max-frame-age-ms", type=int, default=5000,
                        help="Drop model results older than this decoded-reception age, also requiring the same active task (default: 5000).")
    parser.add_argument("--cert")
    parser.add_argument("--key")
    parser.add_argument("--ice-servers", default="[]", help="JSON RTCIceServer list; prefer an environment-configured file for secrets.")
    args = parser.parse_args()
    if not 0 <= args.decision_interval_ms <= 60000:
        parser.error("--decision-interval-ms must be between 0 and 60000.")
    if not 1 <= args.inference_timeout_ms <= 120000:
        parser.error("--inference-timeout-ms must be between 1 and 120000.")
    if not 1 <= args.max_frame_age_ms <= 30000:
        parser.error("--max-frame-age-ms must be between 1 and 30000.")
    if bool(args.cert) != bool(args.key) or (args.host not in ("127.0.0.1", "::1", "localhost") and not args.cert):
        parser.error("Use both --cert and --key; non-loopback listeners require TLS.")
    tls = None
    if args.cert:
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(args.cert, args.key)
    token = os.environ.get("ARENA_CONTROL_TOKEN") or secrets.token_urlsafe(24)
    if len(token) < 24:
        parser.error("ARENA_CONTROL_TOKEN must contain at least 24 characters.")
    ice = [RTCIceServer(**value) for value in json.loads(args.ice_servers)]
    app = make_app(args.origin, token, load_policy(args.policy), ice, args.inference_timeout_ms,
                   args.decision_interval_ms, args.max_frame_age_ms, args.live_view)
    hostname = "[::1]" if args.host == "::1" else args.host
    print("Pair endpoint: %s://%s:%s/control?token=%s" % ("wss" if tls else "ws", hostname, args.port, token), flush=True)
    if args.live_view:
        print("Live view: %s://%s:%s/live (localhost only)" % ("https" if tls else "http", hostname if args.host in ("127.0.0.1", "::1", "localhost") else "127.0.0.1", args.port), flush=True)
    web.run_app(app, host=args.host, port=args.port, ssl_context=tls, access_log=None, print=None)


if __name__ == "__main__":
    main()
