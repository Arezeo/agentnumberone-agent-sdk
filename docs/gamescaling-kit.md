# GameScaling local FP16 kit

Connect [GameScaling](https://github.com/THUSI-Lab/GameScaling) to **[Agent Number One](https://agentnumberone.io)** using your own Apple Silicon Mac. The source ZIP includes setup, native text/action generation, optimized MLX execution, WebRTC transport, localhost live view, profiling, tests and an optional Doom reproduction. It contains no model weights or paid inference dependency.

## Requirements and download

Use an Apple Silicon Mac, macOS 26 or newer for the pinned MLX wheel, and **Python 3.12**. A 24 GB machine is our test host; other memory sizes and Mac generations are not certified. Run one model at a time. Weight download size is not total runtime memory: image tensors, language state, temporary conversions and macOS also need space.

Download the `gamescaling-fp16-kit.zip` asset and its `.sha256` from [GitHub Releases](https://github.com/Arezeo/agentnumberone-agent-sdk/releases). In the download directory:

```sh
shasum -a 256 -c gamescaling-fp16-kit.zip.sha256
unzip gamescaling-fp16-kit.zip
cd agentnumberone-gamescaling-kit
python3.12 gamescaling.py setup --model 0.8b-s3
python3.12 gamescaling.py doctor --model 0.8b-s3 --verify-weights
python3.12 gamescaling.py test
```

The ZIP runs without Git installed after extraction. Its `BUNDLE-MANIFEST.json` lists every source file's hash and the exact Git commit. Git users can clone `https://github.com/Arezeo/agentnumberone-agent-sdk.git` instead.

| Selection | Publisher checkpoint | Pinned revision | Source weight bytes |
| --- | --- | --- | --- |
| `0.8b-s3` | [THUSI-Lab/GameScaling-0.8B-S3](https://huggingface.co/THUSI-Lab/GameScaling-0.8B-S3) | `1b89640dcf2ef51fb53ad9338a0725e9622dd200` | 2,214,590,296 |
| `2b-s3` | [THUSI-Lab/GameScaling-2B-S3](https://huggingface.co/THUSI-Lab/GameScaling-2B-S3) | `0ef44e6569ebfeb310c5d5a5a973ae30cccba6ac` | 5,443,677,680 |

To install 2B, run the same `setup` command with `--model 2b-s3`. Use that selection on every subsequent command. Setup verifies the original weights against both publisher metadata and the pinned SHA-256 in `model_catalog.py`. It preserves the source BF16 files and casts floating parameters to **FP16 in memory**, without quantization or retraining. FP16 and BF16 are different numerical formats; this is not a claim of bit-identical publisher execution.

The Python environment lives in the extracted directory's `.venv-gamescaling`. Weights, run logs and the private pairing file default to `~/Library/Application Support/AgentNumberOne/gamescaling/`, outside Documents/iCloud. Choose another non-synced disk with `ANO_GAMESCALING_DATA`; point at an existing original checkpoint with `ANO_GAMESCALING_MODEL_PATH`. Paths with spaces are supported. Do not move a created virtual environment: extract first, then run setup.

## Native model check

Stop the live model before a GPU test. The launcher refuses a listener on port 8767 and holds a local lock against other kit inference commands. This does not detect unrelated GPU applications; close those yourself for uncontended measurements.

```sh
OUT="$HOME/Library/Application Support/AgentNumberOne/gamescaling/results"
python3.12 gamescaling.py smoke --model 0.8b-s3 --output "$OUT/smoke-08.json"
python3.12 gamescaling.py smoke --model 2b-s3 --output "$OUT/smoke-2.json"
```

This loads the real checkpoint, checks all floating parameters are FP16, supplies five image/action history turns plus a current image, generates native output and tests the upstream action parser. Without `--image /path/to/frame.png`, it uses a clearly labeled synthetic fixture. An empty reasoning section or no-op is recorded, not replaced by scripted text/actions. A successful smoke test establishes model execution and format compatibility, **not gameplay ability, conversation quality or sustained FPS**. Failed parsing exits nonzero and retains the raw output in the report.

To exercise the real model through encoded video and two actual WebRTC peers before configuring production TLS:

```sh
python3.12 gamescaling.py webrtc-smoke --model 0.8b-s3 \
  --output "$OUT/webrtc-08-run1"
```

Repeat for `2b-s3` with a new directory. This receives a moving synthetic fixture, sends an operator instruction, runs through six-image history, inspects the local viewer routes and verifies rejection of a nonlocal Host header. It records whether native text and input envelopes were received. Optional empty reasoning is valid; text-channel availability does not prove conversational ability. The test uses separate pairing credentials and does not change production pairing. It bypasses WebSocket signaling on the in-process loopback peers: token/Origin tests run in `test`, and production browser/WSS/ICE reachability still require the manual procedure below.

## Production WebRTC connection

The production website uses HTTPS, so its signaling connection requires trusted **`wss://`**. Start a trusted TLS reverse proxy to local port 8767. Preserve the browser's `Origin: https://agentnumberone.io` and forward only `/control`; keep `/live`, `/health`, images and diagnostics local. See [the signaling contract](control-protocol.md#run-a-researcher-endpoint).

One option, if the browser and endpoint already belong to your Tailscale network, is Tailscale Serve. Run `tailscale serve --help` and use its HTTPS reverse proxy to `http://127.0.0.1:8767`; use the hostname it reports. This requires your own Tailscale setup and is not performed by the kit. A loopback plain `ws://` URL or an untrusted self-signed certificate will not work from production. TURN credentials are optional and may be needed on restrictive networks; the kit does not provide a hosted relay.

Once your trusted endpoint exists:

```sh
python3.12 gamescaling.py run --model 0.8b-s3 \
  --profile pixel-duel --public-url wss://YOUR-HOST/control
```

For Control Basics or another game, use `--profile general`. For Crossroad, use `--profile crossroad`. Change `--model` to `2b-s3` to test 2B. The runner prints the **location** of `connection.txt`, not its token. Copy that file's complete paired URL into **Connect an agent → Connect through WebRTC** on the production game page. Keep it private. The endpoint checks the exact Origin and pairing token and admits one peer. That validates a connection, not a model's identity or safe behavior in its own local tools.

Open **[http://127.0.0.1:8767/live](http://127.0.0.1:8767/live)** on the endpoint machine. It shows received video, the selected current model image and timing/transport metrics. Native reasoning streams to the production chat and is saved in local event logs. The five history images remain in the model context. Viewer refresh is capped at 5 Hz; it is distinct from video reception FPS and model decision Hz. The viewer is read-only, local and uses no second model call. Viewing it does not make slow inference faster.

For DOM/WebGL games, approve sharing the current game tab when prompted in a desktop Chromium browser supporting Element Capture. Human Inputs/chat overlays are excluded by the production capture path; game menus and the visible agent cursor remain. If video is black, check the selected game tab, capture permission and received frame counter before evaluating the model. Do not point the capture at the viewer itself.

## End-to-end manual evaluation

For each model, use a fresh game session with the appropriate prompt profile. Record the exact commit, `doctor` output, Mac model, macOS, other CPU/GPU workloads, selected game, difficulty and timestamps.

1. **Connection and pixels.** Confirm pairing, video counters, readable nonblack received/model panels, source dimensions and active gameplay state. Six images appear after five accepted history turns; warm-up naturally has fewer. Menus/pause may allow perception and text while controls stay blocked.
2. **Control Basics Level I.** Record a session, run every task and save diagnostics. Note all six task outcomes separately. Verify the on-page cursor/key overlay against the endpoint's queued controls. A queued command does not prove the game applied it.
3. **Control Basics Level II.** Repeat with a fresh run; include chord, typing, dragging and held-input results. Unsupported capabilities or a parsed no-op are evidence to retain, not reasons to inject a solver.
4. **Pixel Duel.** Use [Game Lab](https://agentnumberone.io/lab), record the difficulty, fixed wall duration, final score, movements and shots. Let the game run during inference. Keep the model's own optional reasoning enabled.
5. **Conversation/instruction.** Ask who the opponent is and issue a visible task such as moving toward a location. Retain sent instruction, native reply, action and timestamps. Check both speech and behavior; a reply alone does not demonstrate control conditioning.
6. **Pause, resume and reconnect.** Confirm no actions apply during pause or after context invalidation/disconnect, held inputs release, and fresh frames are used on resume. Save the website's session export and the endpoint run files together.

Endpoint files are under `results/production-<model-key>/`: `events.jsonl` retains native output/reasoning, operator instruction and identity, parsed action, queued controls and cancellation/disposition, plus stage timing. `system-prompt.txt` is the selected prompt; `latest-model-input.png` is the latest selected frame. Keep logs and website exports private; this kit does not automatically upload results. Website export contents depend on the deployed platform version: keep endpoint logs to preserve the full model-side trace.

Report control passes and game scores alongside latency median/p95, decision Hz, video FPS, output token count/rate and rejected/stale results. Distinguish preprocessing, vision, language prefill/generation and action scheduling. `receive_to_end_ms` begins at local frame reception; it does **not** measure browser capture-to-action latency. There is no fixed 5 Hz performance promise. See the [complete metrics definitions](../GameScaling0.8B-S3/README.md#metrics-and-profiling).

## Cache correctness and performance experiment

Keep all six images and native live settings. Save at least eight consecutive 1280×720 game screenshots privately, stop the live model, and run:

```sh
python3.12 gamescaling.py profile --model 0.8b-s3 \
  --frames frame1.png frame2.png frame3.png frame4.png frame5.png frame6.png frame7.png frame8.png \
  --output "$HOME/Library/Application Support/AgentNumberOne/gamescaling/results/profile-08.json"
```

Repeat for `2b-s3`. This compares cold/cached last-token logits and repeated detached prefix restores, then times baseline, vision reuse, vision-plus-prefix reuse and all three caches on identical sliding six-image windows. The replay uses greedy sampling/48 tokens and discloses that difference; it is not a live score or the native sampling smoke test. No shortened history or forced empty reasoning is used in live operation.

Enabled optimizations are per-image preprocessing reuse, per-image vision-feature reuse, static system-prefix state reuse and overlap between inference and the preceding 200 ms WebRTC action sequence. The changing multimodal history still gets processed. CPU speculative preprocessing and GPU speculative vision remain disabled following measured scheduling regressions. [Previous experiments and limitations](../GameScaling0.8B-S3/experiments/scheduling.md) are bundled; they are not universal speedup guarantees. Change one cache/overlap switch at a time and repeat matched sessions if measuring a live gain.

## Optional original Doom scenario

```sh
python3.12 gamescaling.py setup --model 0.8b-s3 --with-doom
python3.12 gamescaling.py doom --model 0.8b-s3 --max-decisions 10 \
  --mode realtime --output "$HOME/Library/Application Support/AgentNumberOne/gamescaling/results/doom-08-run1"
```

Use `--model 2b-s3` for 2B and a new output directory for every run. The command fetches the pinned original DFP Battle-1 scenario plus MIT license, uses ViZDoom 1.3.2 and bundled Freedoom, and runs the exact vendored publisher Doom prompt. A local viewer appears at **http://127.0.0.1:8769/live** while the test runs. Inputs, native replies, timing, requested and engine-reported controls, configuration, summary and an engine `.lmp` replay remain in the output directory. The scenario has a 2100-tic episode limit and the runner has a 180-second wall cap (an in-flight GPU call must finish before shutdown).

`realtime` advances the 35 Hz engine while inference runs. `paused` stops it while inferring and is **diagnostic only**, not a real-time result. The six 30 Hz held states map over seven engine tics; mouse displacement uses a documented local 90° FOV mapping. The publisher did not release its game executor, sensitivity or a complete matched engine environment. This reproduces its prompt/scenario contract, **not exact paper gameplay or performance**. The model receives screenshots/history only; engine variables appear only in evaluator output. This minimal Doom runner executes inference/chunks sequentially; the WebRTC runner includes the overlap optimization.

## 9B-S3 as a 4-bit GGUF through llama.cpp

The MLX runner above covers 0.8B and 2B. The 9B checkpoint can be driven through a [llama.cpp](https://github.com/ggml-org/llama.cpp) server with [`llama_policy.py`](../GameScaling0.8B-S3/llama_policy.py), a policy for the reference runner. The kit does not download, convert or verify GGUF weights: bring your own conversion of the checkpoint and its `mmproj` file, under the publisher's license. Stop the MLX runner first; both use port 8767 and one model should own the GPU.

```sh
llama-server -m GameScaling-9B-S3.Q4_K_M.gguf --mmproj GameScaling-9B-S3.mmproj-f16.gguf \
  -c 4096 --host 127.0.0.1 --port 8901
.venv-gamescaling/bin/python GameScaling0.8B-S3/llama_run.py \
  --public-url wss://YOUR-HOST/control --game auto
```

`llama_run.py` checks the server, keeps the pairing token in the same private `connection.txt` as the MLX runner, prints that file's location and never the token, and starts `scripts/remote-control/agent.py` with the policy. The paired URL stays the same across restarts while `--public-url` is unchanged. `--game` selects the system prompt: `auto` (Pixel Duel prompt on the Pixel Duel seat, otherwise general), `general`, `voxelcraft` or `pixel-duel`. A 4,096-token context holds the system prompt (about 900 tokens), up to five 640 × 360 frames with their turns (about 260 to 290 tokens each) and the reply.

| Setting | Default | Meaning |
| --- | --- | --- |
| `ANO_LLAMA_TEMPERATURE` | `1.0` | Sampling temperature. Do not set 0; see below. |
| `ANO_LLAMA_HISTORY` | `5` | Frames a request may carry, the current one included, before the history is cleared. |
| `ANO_LLAMA_NOOP_RESET` | `2` | Consecutive empty actions that clear the history. |
| `ANO_LLAMA_IMAGE_W`, `ANO_LLAMA_IMAGE_H` | `640`, `360` | Image size sent to the model. |
| `ANO_LLAMA_MAX_TOKENS` | `256` | Reply cap, including optional reasoning. |
| `ANO_LLAMA_HOLD_WATCHDOG_S` | `3` | Releases inputs still held this long after a plan began. |
| `ANO_LLAMA_SLOT` | `0` | Server slot, pinned so the prompt cache is reused. |
| `ANO_LLAMA_LOG` | data folder, `results/llama-policy/decisions.jsonl` | One JSON line per decision: latency, tokens, raw action, reasoning. |

### Why the policy is built this way

These are the faults found in a first, simpler adapter, with the checks that showed them. Each check is a handful of samples on 2026-11-10, not a benchmark.

- **Sample at temperature 1.0.** With greedy decoding the top choice is nearly always the empty action `0 0 0 ; ; ; ; ; ;`. On one "You died" frame, four prompts at temperature 0 all returned it, including one whose reasoning said to click Respawn. At 1.0, three of four samples on the same frame clicked.
- **Name the game that is on screen.** The first adapter had a fixed Pixel Duel prompt while the page showed another game. Keep game text in the published prompt skeleton (`build_system_prompt`) and choose it per game.
- **Do not let an empty history build up.** With one empty action in the history, none of four samples acted on that frame. The policy clears the history after two consecutive empty actions.
- **Grow the history, then clear it.** A sliding window changes the prompt prefix every turn and forces the server to re-evaluate every image. Appending turns and clearing at five frames keeps the prefix cached: the median decision evaluated 257 new tokens.
- **Never block `decide()`.** The reference runner sends a decision's actions only after `decide()` returns, so a policy that waits for the model inside it cannot play six 33 ms steps. The model runs on its own thread; `decide()` returns on every frame and plays the steps that are due. The last step stays held until the next plan, as in the MLX runner, with a watchdog release.
- **Use the training vocabulary exactly.** `one` is the number-row key and `One` is F1; `LB`, `RB` and `MB` are held mouse buttons, not instant clicks.

### What was measured, and the limits

On an Apple M5 Pro with 24 GB, Homebrew llama.cpp build 11669, Q4_K_M weights and an f16 `mmproj`, one live Voxelcraft session on production made 847 decisions in about 50 minutes: median 1.67 s per decision (10th percentile 1.50 s, 90th 4.88 s, longest 8.07 s), 79 empty actions (9%) and none unparseable. Reasoning was written in 77 decisions, and those are among the slow ones. Generation ran at a median 33.7 tokens per second while the game rendered on the same GPU, and the median reply was 34 tokens. This is one session with one prompt; it says nothing about game skill.

- **It is not real-time.** A plan covers 200 ms and a decision takes well over a second, so most of the time the agent is holding the last step of an old plan.
- **Camera turning stops at the screen edge.** For pointer-locked games the page turns pointer movement into camera movement, and pointer positions are limited to the screen. In that session the commanded pointer sat on a screen edge for 139 of 847 decisions (16%), and there the agent could no longer turn that way.
- **Operator instructions are passed, not proven.** The instruction is added to the user turn and clears the history when it changes. In three samples each of two instructions, this checkpoint showed no sign of following them.
- **The GGUF conversion is not compared with the original weights**, and keys the runner rejects (function keys, for example) are skipped.

## What has been verified

The [release verification record](../GameScaling0.8B-S3/docs/portable-kit-validation.md) separates clean-source installation, CPU/WebRTC tests, actual model calls, cache numerical checks and manual production results. Test success does not erase previous failures on browser games or prove this checkpoint generalizes beyond its training distribution.

## Build a distributable ZIP

Maintainers can run `python3 tools/build_release.py --ref HEAD --output /outside/repo/gamescaling-fp16-kit.zip` from a committed revision. The builder reads committed Git blobs only, rejects model/runtime/private file paths and symlinks, embeds SHA-256 source hashes, and emits a ZIP checksum. It cannot detect every possible secret in source: review the diff before publishing. License/provenance are included; weights are fetched by each user under the publisher's license.
