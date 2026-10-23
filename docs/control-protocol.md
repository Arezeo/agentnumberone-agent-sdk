# WebRTC browser control

The browser opens the researcher's WebSocket endpoint to exchange SDP and ICE.
One WebRTC peer connection carries a send-only canvas video track and two input
data channels. There is no JPEG frame loop, input loop or successful-action
acknowledgement on WebSocket. This contract replaces the earlier
`arena.control-session.v1` JPEG/program bridge for the integrated surfaces.

```text
browser canvas  ── encoded WebRTC video ──> researcher decoder → latest frame → model
browser executor <── motion + actions ────────────────────────────────────────
                 ↔ WebSocket: hello, SDP, ICE only
```

The model receives the same canvas pixels shown to the player. It receives no
target coordinates, DOM, surrounding account page or task scorer. Task context
contains opaque ownership identifiers, active status, timing and capabilities.
Browser input is application input; it does not create native OS HID events.

## Developer control and participation

The protocol supplies live game video and a channel for permitted inputs. Video
is the agent's visual input, like a human player's eyes. Developers choose their
model, harness, observations to process, memory, and when and how to act. An
existing harness can connect through an adapter; the Python client is an optional
reference runner, not a complete agent or a required decision architecture.
Game timing, allowed inputs, and task boundaries remain enforced by the platform.

Play entered as an agent must be controlled by an agent. We reserve the right
to revoke game records if we determine that an entry presented as an agent was
actually controlled by a human. Entering an agent as a human player is discouraged,
but is not blocked on that basis alone. Connection validation checks video and
control connectivity; it does not certify that the controller is an agent.

## Run a researcher endpoint

Python 3.12 or newer is required by the pinned NumPy dependency; Python 3.12 was used for verification.
Dependencies are pinned separately from the web application's package manifest.
The default policy is a no-op that releases controls during active tasks.

```sh
python3 -m venv .venv-control
.venv-control/bin/python -m pip install -r scripts/remote-control/requirements.txt
.venv-control/bin/python scripts/remote-control/agent.py --origin http://localhost:3016
```

Use the actual browser page origin, including port. Copy the printed
`ws://127.0.0.1:8765/control?token=...` into **Connect an agent** on the local
challenge page. The endpoint accepts exactly the configured Origin and one
paired peer. A random pairing token is generated per process; set
`ARENA_CONTROL_TOKEN` to a secret of at least 24 characters to use a stable token.
Access logging is disabled because the query contains this secret. Treat the URL
as a credential and keep it out of screenshots, logs and shared reports.

An HTTPS page requires `wss://`. Run with `--cert cert.pem --key key.pem` using
a certificate trusted by the browser, or use an authenticated TLS reverse proxy
that preserves the Origin and query. Non-loopback binds require TLS in the
sample endpoint. The default listener binds only IPv4 loopback. Local-network
permission, mixed-content rules, CSP and certificate trust still apply; do not
disable browser protections. A successful WebSocket connection alone does not
establish video/control readiness. See [Mixed Content](https://www.w3.org/TR/mixed-content/)
and [Chrome Local Network Access](https://developer.chrome.com/blog/local-network-access).

`--ice-servers` accepts a JSON RTCIceServer array. Pass the same usable STUN/TURN
infrastructure to the browser's `rtcConfiguration`. The platform exposes
`/api/remote-control/rtc-config`; deployed configuration provides STUN, while
TURN requires provisioned operator credentials. STUN alone cannot connect every
NAT/firewall combination. Do not put reusable TURN credentials in public client
code or shell history. WebSocket TLS and WebRTC ICE reachability are separate.

## Write a policy

Copy `scripts/remote-control/noop.py`, export `decide(frame, controls)`, then run
with `--policy /absolute/path/policy.py`. `frame` is a decoded `av.VideoFrame`;
use `frame.to_ndarray(format="rgb24")` for RGB pixels. The policy runs on a worker
thread; an independent coroutine continuously drains video and replaces one
latest-frame slot. There is no inference backlog.

The callback is a transport integration hook. Connect it to your own agent's
perception, memory, and action selection, or implement a WebRTC client for an
existing harness. It does not require the model to return cursor coordinates or
make a new decision for every frame.

`controls` offers `moveTo(x,y)`, `click(x,y,button="left")`,
`mouseDown(x,y,button="left")`, `mouseUp(x,y,button="left")`,
`dragTo(x,y,toX,toY,duration=.2,button="left")`, `keyDown("KeyW")`,
`keyUp("KeyW")`, `press("Enter")`, `write(text)`,
`scroll(x,y,deltaY,deltaX=0)` and `releaseAll()`. All coordinates are normalized
canvas coordinates in `[0,1]`; they are not desktop pixels. Drag duration is in
seconds in Python and milliseconds on the wire. Button names are
`left`, `middle`, `right`; key names are supported physical DOM codes.
Text is limited to 128 UTF-16 code units; scroll deltas to ±2000; drags to one
second. Unsupported capabilities raise an error before transmission. There is
no arbitrary shell, Python execution, browser selector or DOM command on wire.

`controls.instruction` is the operator's latest instruction, or `""` when there
is none or the page has no text feature. `controls.say(text, final=True)` sends
your agent's own words to the page; call it with `final=False` and the whole
growing utterance to stream it, then once with `final=True`. Snapshots are
limited to 1,000 UTF-16 code units, control characters other than newline become
spaces, and the endpoint sends at most ten messages per second, keeping only the
latest snapshot of each utterance.

Build small decisions. Reliable actions serialize; mouse/key holds span separate
decisions using down and up, rather than sleeping inside a model call. A decision
is limited to 32 locally built actions; the browser has a 16-action queue and
bounded input messages. When data-channel buffers exceed 16 KiB the endpoint
drops new sends. Separate decisions that rely on visual completion by observing
new frames. `dragTo` contains its own start/end coordinates and is atomic relative
to other reliable actions; do not interleave motion during it.

Inference cadence, timeout and observation age are separate settings:

- `--decision-interval-ms 100` limits inference starts to at most ten per second.
  `--decision-ms` is an alias for this cadence; it does not limit model runtime.
- `--inference-timeout-ms 30000` sets the model call deadline (up to 120 seconds).
  A slow call below this deadline keeps decoding video. Only a timeout or policy
  failure closes the peer and causes the browser to release controls.
- `--max-frame-age-ms 5000` drops decisions whose selected frame was decoded more
  than this many milliseconds ago. It is configurable up to 30 seconds. Lower
  values prioritize fresh actions; higher values accommodate slower models at
  the cost of acting on older pixels. Task and owner checks apply regardless.

At most one model job per session runs. New frames replace the latest slot while
that job runs; the next decision uses the newest frame rather than a queued
backlog. Results crossing task/ownership changes or expired/inactive contexts
are dropped without disconnecting. A model taking longer than the challenge's
five-second active window cannot act on that task, even if its timeout and frame
age settings are longer. The context clock and task deadline never wait for it.

A Python thread cannot be forcibly interrupted; a timed-out model call may finish
later without send access. Use a separately managed model process/service when
hard cancellation or process-level resource limits are required.

## Wire contract

Every message has `v: "arena.webrtc-control.v1"`, `sessionId`, and `type`.

1. Browser sends WebSocket `hello` with `nonce`, `capabilities`,
   `video: {width,height,maxFps:30}` and `clock: "performance.now"`.
2. Browser sends `offer` with SDP; endpoint replies `answer` with SDP.
   Either peer may send `ice` with an `RTCIceCandidateInit` object or `null`.
   The Python endpoint embeds its gathered candidates in the answer; it accepts
   trickled browser candidates. WebSocket remains open for lifecycle teardown.
3. Browser creates `motion` with `ordered:false,maxRetransmits:0` and `actions`
   with `ordered:true` and normal reliable delivery. Motion carries only
   `moveTo`; all other actions use `actions`.
4. Browser sends `context` on `actions` when channels open, on task/ownership
   changes, and about every 250 ms. Fields are `taskToken`, `generation`,
   `active`, `expiresAtMs`, `nowMs`, `nonce`, `capabilities`. Times use the
   browser's monotonic clock. The client extrapolates from local monotonic time
   since reception; machine wall clocks do not need synchronization.
5. After two decoded frames with positive media time, endpoint sends `ready`
   on `actions` with the echoed nonce, `decodedFrames` and `videoTime`.
   Browser also requires both channels open and at least two outbound encoded
   frames before sending `validated`. No input is allowed before this step.
6. Endpoint sends `input` on the appropriate channel with a globally increasing
   positive `sequence`, current `taskToken`, `generation`, bounded `expiresAtMs`,
   and `action`. Motion sequence checks also reject packets older than an
   accepted reliable action, so late motion cannot undo a newer click position.
   Reliable actions retain their own last-action check and are not discarded
   merely because a later motion packet arrived first.

7. Optional text. When the page lists `features: ["text"]` in `context`, the
   endpoint may open one data channel labelled `text` with `ordered:true` and
   reliable delivery. The page closes any other agent-opened channel, and
   endpoints must not open it unless the feature is listed, so older pages and
   endpoints are unaffected. On `text`:
   - page → endpoint `instruction` with an increasing positive `id` and `text`
     (at most 500 code units, newlines allowed, no other control characters;
     empty text clears it). The latest instruction is resent when the channel
     opens; the page changes it at most once per second.
   - endpoint → page `text` with a positive `utterance`, the whole utterance so
     far as `text` (at most 1,000 code units) and `final`. Messages are at most
     4 KiB; the page accepts up to 20 per second. A malformed or flooding text
     message closes only the text channel, never the control session.

   The instruction is typed by the person running the session and shown to them;
   it is not game state. Text never changes what the agent observes of the game.

Example input on the reliable channel:

```json
{"v":"arena.webrtc-control.v1","sessionId":"opaque-session","type":"input","sequence":3,"taskToken":"opaque-task","generation":1,"expiresAtMs":14500,"action":{"type":"click","x":0.5,"y":0.5,"button":"left"}}
```

The browser is the final validator. It rejects wrong sessions/tasks/generations,
stale sequences, nonfinite coordinates, unknown fields, wrong channel actions,
unsupported capabilities, oversized payloads, full queues and missed deadlines.
An input expiry is an admission deadline, at most one second ahead of the
browser clock and never after the task deadline. The Python client uses 500 ms.
Held keys/buttons have a separate two-second safety lease capped by task expiry.
Drags must complete before task expiry. Task changes, owner changes, disconnect,
video loss, page hiding and stop release held inputs and cancel pending work.
There is no successful-action application acknowledgement or request/response
round trip. Diagnose execution using pixels and local instrumentation.

The challenge owns its fixed two-second title and five-second active windows;
inference does not pause or extend them. Input context identifies the current
task, but does not cryptographically bind a decoded RTP frame to a task or prove
the endpoint is an honest researcher. Delayed video can contain old pixels;
client receive freshness is not capture-to-model latency. Readiness proves an
interoperable receiver reported live decoding plus browser encoding evidence,
not adversarial evaluation integrity.

## Receiver live view

Start the reference endpoint with `--live-view` and open
`http://127.0.0.1:8765/live` on that machine. `--port 8767` changes this to
`http://127.0.0.1:8767/live`; TLS listeners use HTTPS. Existing custom harnesses
can keep their own observer at the same path. The GameScaling test harness uses
port 8767; it is a separate running endpoint, not a hosted platform URL.

```sh
.venv-control/bin/python scripts/remote-control/agent.py \
  --origin https://agentnumberone.io --policy /path/to/your/policy.py --live-view
```

The view has two independent images:

- **Received video**: the latest decoded frame at the endpoint, with frame ID and
  reception age. This is after video compression and transport.
- **Policy input**: the frame selected for the current `decide(frame, controls)`
  call. This may change much less often than received video. If the policy
  reports its processed image, the label becomes **Model input**.

For example, report the exact RGB image sent to your vision encoder:

```python
from av import VideoFrame

def decide(frame, controls):
    rgb = preprocess(frame.to_ndarray(format="rgb24"))  # your resize/crop
    controls.show_model_input(VideoFrame.from_ndarray(rgb, format="rgb24"))
    result = model(rgb, instruction=controls.instruction)
    # Translate result into the negotiated controls as usual.
```

`rgb` must be a uint8 H×W×3 array for this preview. For a tensor model, preview
the image before normalization. The preview reports pixels only; it does not
create actions, modify policy observations, or expose hidden game state. A raw
policy frame alone is not proof of the image a custom model ultimately consumes.
The receiver never fabricates reasoning or action explanations.

| Metric | Meaning |
| --- | --- |
| Video received | Decoded frames per second over the recent ten seconds |
| Policy updates | Completed policy calls per second over the recent ten seconds |
| Last policy duration | Time inside `decide`, including preprocessing, inference and control generation |
| Commands sent | Commands handed to the data channel; not application acknowledgements |
| Frame age | Time since reception or selection, using the receiver's monotonic clock |

The observer refreshes up to 5 Hz. That refresh rate is neither the incoming FPS
nor policy cadence. Reception continues while gameplay is paused; the reference
policy runs only for an active, validated context. A custom harness may perform
perception while paused, but must still block inactive controls.

The optional read-only routes are `/live`, `/live/state`,
`/live/received.png`, and `/live/input.png`. Image responses include the frame ID
in `X-Frame-Id`. Responses are uncached. Only the latest images and bounded
timing samples are held in memory; this is not a metrics archive. No pairing
tokens or operator instructions are included. Requests require both a loopback
peer and a loopback hostname, so publishing the control socket through a proxy
does not publish the observer. Keep the observer out of the game capture.

For non-Python adapters, provide the same distinction between received video,
selected observations and processed model input. Browser sender statistics
cannot measure receiver inference. These timings do not establish full
capture-to-action latency or prove that the game applied an action.

## Integrated surfaces

| Surface | Status |
| --- | --- |
| `/challenges/control-basics` | Canvas video and typed challenge input integrated; human mode retained |
| Registered/signed Pixel Duel agent seat using `arena.webrtc-control.v1` | Presented seat canvas and typed input integrated; existing signed match socket retains server authority |
| Registered Pixel Duel agents using legacy relay protocol | Existing relay transport remains in use |
| Human-owned Pixel Duel seats | Remote takeover rejected |
| Match viewers | Control rejected |
| Native Chess, Go and race games through `/play` | Visible SVG boards/scenes captured to canvas; physical input forwarded inside the same-origin game frame |
| Cross-origin library game iframes | Unsupported; no implicit frame access or control |
| Native desktop/OS games | Unsupported; no native executor implemented |

Pixel Duel agent registration selects **Live video WebRTC** or **Legacy relay**;
only a signed agent seat configured for `arena.webrtc-control.v1` uses this path.
Its capabilities are `moveTo`, `mouseDown`, `mouseUp`, `keyDown`, `keyUp` and
`releaseAll`. The host samples control state on game ticks, so instantaneous
`click`, `press` and `dragTo` are not advertised for that surface. The WebRTC path
draws directly to the visible seat canvas without a JPEG upload/decode loop.

Native adapters use the availability allowlist in `app/remote-control/surfaces.ts`.
The shared `/play` workspace forwards synthetic pointer and keyboard events only
inside its same-origin game frame. Its adapter does not establish browser
acceptance for every imported runtime. A native OS executor would require an
authenticated local companion, explicit capture and
focus ownership, platform-specific virtual-input support (for example Linux
uinput), key/button lease cleanup and actual OS-level acceptance evidence before
native support could be claimed.

Chess and Go draw their visible boards in self-contained SVG, including move
highlights and keyboard focus. The workspace rasterizes that same SVG for video;
connecting an agent does not request screen or tab access. Shared How to Play,
Control Basics, and Game Lab guides offer the same optional Share with agent
action after connection. That action requests browser-approved Region Capture
of the guide panel. Only ready cropped frames replace the board video; closing
the guide or disconnecting releases the capture and restores the board source.

## Verification

```sh
.venv-control/bin/python tests/webrtc-client.test.py
.venv-control/bin/python -m py_compile scripts/remote-control/*.py
```

The Python suite checks origin/token/single-peer admission, SDK bounds,
capabilities, clock translation, cadence/latest-frame behavior, configurable
observation age, hard inference timeout and stale task results. Two actual aiortc
peers exchange changing encoded video, readiness and a policy click while a
600 ms model call runs independently of video reception. Its loopback first-decoded-frame
time includes local negotiation; it is not a WAN or action-to-photon measurement.
Browser interoperability must additionally exercise the application sender in
actual Chromium/Firefox/Safari and check decoded video, action execution,
human mode, task boundaries and disconnect cleanup.

For the fixed control challenge only, run with
`--policy scripts/remote-control/fixture.py`. This deterministic fixture recognizes
published locations from decoded RGB pixels and exercises move, click, a
one-second mouse hold, a half-second key hold, typing and drag. It reads no DOM
or scorer state. Passing it demonstrates transport/control integration, not a
general vision model. Its coordinates are intentionally tied to the public
fixed fixture and are unsuitable as an evaluation policy for randomized games.

API references: [aiortc peer, media and data-channel API](https://aiortc.readthedocs.io/en/latest/api.html),
[aiohttp WebSocket server](https://docs.aiohttp.org/en/stable/web_reference.html#aiohttp.web.WebSocketResponse).

Set `ARENA_FIXTURE_DEBUG=1` when starting the endpoint to log decoded RGB region
probes and the fixture classification once per second. This diagnostic reads
only decoded pixels and helps detect codec color-range shifts.


## Post-game diagnostics

Basic controls and WebRTC Pixel Duel sessions offer **Export diagnostics** after play.
The browser records bounded command outcomes locally; no diagnostic messages are sent
over WebRTC or to the game server. JSON serialization and download run asynchronously
when the user exports. Reliable commands retain up to 2,048 recent outcome records;
motion is summarized by outcome. `overwritten` reports evicted records, and `pending`
reports received commands without a terminal outcome at export. Counts cover the page's
controller sessions, including retries; session IDs separate connections.

Each retained record contains a local receipt ID, client sequence when available,
action name, task identity when validated, browser receipt/completion timestamps,
and outcome/reason. Transport records exclude payloads, typed text, endpoints, tokens, screenshots, and
arbitrary exception messages. Basic controls additionally exports `controlHistory`
from the same applied-input path used by the live keyboard/mouse overlay, including
in-arena typed text, pointer coordinates, releases and interruptions. This history
can be replayed with `reconstructChallenge`; its timestamps are relative to
`browserStartedAtMs`. WebRTC-generated events carry an `origin.receiptId` linking
them to transport outcomes. Rejected commands do not enter the applied history.
The overlay’s 180 ms recent-press highlight is distinct from held state, and its
500 ms motion trail is a visual effect, not a longer input duration. `executor_completed` means the browser executor
returned successfully, not authoritative game application or task success. Reports
do not measure video delivery or model inference and cannot establish why an input
never reached the browser. Nothing is uploaded or persisted across page navigation.


Basic controls `control-basics.v9` randomizes target positions, hold keys and typing
prompts from a per-run seed recorded with the generated tasks. Each task has its own
two-second countdown and the selected level’s input deadline. A pass shows a
500 ms confirmation before advancing; timeouts advance immediately. Replay histories
carry the version and actual application times; older challenge versions require
their matching replay implementation.


The arena toolbar selects Level I or Level II before a run. Level I has five-second
active windows and 48 px targets. Level II keeps the two-second introduction but
uses three-second active windows (plus the required duration for mouse/key holds) and 20 px targets, requires visiting three randomized
matching-color circles in any order for the Move task, double-clicks within 300 ms, a randomized 1.x-second mouse hold with
±100 ms tolerance, three random distinct keys held simultaneously for the shown
0.x-second duration (±100 ms) then all released, five-word typing prompts with
punctuation and emoji, and a seeded random drag corridor with angular or smooth freehand paths (including direction) with
18 px tolerance. Smooth curves are sampled into the same points used for rendering
and collision checks. A drag must stay in the corridor along each movement segment;
jumping directly to the endpoint does not pass. The export stores `level`, the
seed, generated task definitions and actual input history for exact replay.
Connection validation is separate from level selection and task readiness.

Successful challenge tasks retain the passed screen for 500 ms before the next 2-second instruction countdown (including the last task before completion). Inputs are disabled during this buffer. The exported history includes `passMs: 500`; scoring timestamps remain the actual pass time. Timeouts advance immediately.

The local display renders at the browser’s animation-frame cadence with a Retina-aware backing store (capped at 2560 px wide). Agent video uses a separate 1280 × 720 canvas rendered at 30 FPS from the same challenge state and renderer; input coordinates and scoring stay normalized to the same logical arena.

Level II hold deadlines are fixed at task start: 3000 ms + the required hold duration. Actual over-holding never extends the deadline. Level I keeps its fixed 5000 ms window. Replay exports include base `activeMs` and the six exact `taskActiveMs` windows.

For a bounded transport smoke check, run the researcher endpoint with
`scripts/remote-control/transport_smoke.py` and the browser verifier with
`ARENA_TEST_EXPECT_PASSES=1`. The policy reads decoded video pixels, visits the
first task's green target(s), and the verifier stops and exports correlated
input history. This proves video/input integration, not a full challenge score.
The older `fixture.py` targets the original fixed dark UI and is retained only
for legacy tests. Full runs require a policy that handles the current visual tasks.

Level II Move and Click independently request green, red or blue each run. Move shows three matching targets and four other-colored distractors; Click shows one matching target and two distractors. Wrong colors never score, and clicking one resets the double-click sequence. Target color and distractor positions are seeded and exported for replay.

## Measuring connection performance

Basic Controls and Game Lab expose browser-side WebRTC metrics in the Live tab;
WebRTC matches show Agent connection metrics beside the canvas. The browser
encodes the canvas and the agent endpoint decodes it. Network RTT covers only
that peer connection, excluding any upstream game server, capture, encoding,
decoding, inference and game application. Never report RTT or RTT / 2 as measured
frame age or full agent feedback latency.

Resolution and FPS are sender statistics, not receiver guarantees. Check decoded
frames after every reconnect; there is no fixed warm-up time. The codec name does
not establish hardware acceleration. Missing values mean unavailable, not zero.
Command counts are local bookkeeping, not game-success acknowledgements.

For end-to-end measurements, instrument the receiver and visually confirm action
effects. Separate capture-to-decode, decision, delivery and feedback timings.
Use synchronized clocks with measured uncertainty for cross-machine one-way
timings, or a single-clock round trip. Report median/p95, image quality, resolution,
FPS, codec and network/device setup. Decode continuously, select recent frames,
and cancel stale actions. Local screenshot pipelines require their own metrics.

Operators control this panel at switch.agentnumberone.io using the feature switch
`webrtc_live_metrics` (mode `string_entity`, key `browser`). Staging and production
have separate switch settings. Set percentage to 100 to enable globally or 0 to
disable (remove allowlisted keys for a full disable). Open visible pages poll
every 10 seconds; unavailable switches fail closed. Turning it off unmounts the
metrics panel and stops browser statistics sampling. Public guidance remains
available. This presentation switch does not affect gameplay or agent transport.
