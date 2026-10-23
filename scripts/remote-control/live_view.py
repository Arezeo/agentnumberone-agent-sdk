"""Optional receiver-side observer. Holds only the latest frames in memory."""
import asyncio
from pathlib import Path
import time
from urllib.parse import urlsplit

import av
from aiohttp import web


def png(frame):
    codec = av.CodecContext.create("png", "w")
    codec.width, codec.height, codec.pix_fmt = frame.width, frame.height, "rgb24"
    codec.thread_count = 1
    return b"".join(bytes(packet) for packet in codec.encode(frame.reformat(format="rgb24")) + codec.encode(None))


def rate(times, now):
    recent = [stamp for stamp in times if now - stamp <= 10]
    return (len(recent) - 1) / (now - recent[0]) if len(recent) > 1 and now > recent[0] else 0


class LiveView:
    def __init__(self):
        self.session = None
        self.cache = {}
        self.lock = asyncio.Lock()

    def attach(self, session):
        self.session = session
        self.cache.clear()

    def state(self):
        session, now = self.session, time.monotonic()
        latest = session.latest if session else None
        selected = (session.model_input or session.policy_input) if session else None
        connected = bool(session and not session.closed and session.pc.connectionState == "connected")
        return dict(model="Reference Python endpoint", peer_state="connected" if connected else "disconnected",
                    context_active=bool(connected and session.validated and session.context and session.context.active()),
                    received_fps=rate(session.frame_times, now) if connected else 0,
                    decision_hz=rate(session.decision_times, now) if connected else 0,
                    last_policy_ms=session.last_policy_ms if session else None,
                    commands_queued=session.sequence if session else 0,
                    stale_decisions=session.stale_decisions if session else 0,
                    received_frame=latest[0] if latest else None,
                    received_frame_age_ms=(now-latest[2])*1000 if latest else None,
                    last_model_frame=selected[0] if selected else None,
                    model_input_age_ms=(now-session.input_selected_at)*1000 if selected else None,
                    input_kind="model" if session and session.model_input else "policy",
                    session_id=session.session_id if session else None)

    @staticmethod
    def local_only(request):
        # A reverse proxy must not expose screenshots through its public hostname.
        host = urlsplit("//" + request.host).hostname
        if host not in ("127.0.0.1", "localhost", "::1") or request.remote not in ("127.0.0.1", "::1"):
            raise web.HTTPForbidden(text="Live view is available on localhost only.")
        if request.headers.get("Origin") not in (None, str(request.url.origin())):
            raise web.HTTPForbidden(text="Live view is same-origin only.")

    def install(self, app):
        headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}

        async def page(request):
            self.local_only(request)
            return web.Response(text=Path(__file__).with_name("live-view.html").read_text(), content_type="text/html",
                                headers={**headers, "Content-Security-Policy": "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'"})

        async def state(request):
            self.local_only(request)
            return web.json_response(self.state(), headers=headers)

        async def image(request):
            self.local_only(request)
            session = self.session
            kind = request.match_info["kind"]
            snapshot = (session.latest if kind == "received" else session.model_input or session.policy_input) if session else None
            if not snapshot:
                raise web.HTTPNotFound(text="Waiting for video.")
            key = (id(session), snapshot[0], id(snapshot[1]))
            async with self.lock:
                cached = self.cache.get(kind)
                if not cached or cached[0] != key:
                    cached = (key, await asyncio.to_thread(png, snapshot[1]))
                    if self.session is session:
                        self.cache[kind] = cached
            return web.Response(body=cached[1], content_type="image/png", headers={**headers, "X-Frame-Id": str(snapshot[0])})

        app.router.add_get("/live", page)
        app.router.add_get("/live/state", state)
        app.router.add_get(r"/live/{kind:received|input}.png", image)
