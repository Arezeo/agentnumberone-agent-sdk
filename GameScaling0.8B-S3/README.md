# GameScaling 0.8B-S3 over WebRTC

Run the publisher's native visual/action model on Apple Silicon and connect it to **[Agent Number One](https://agentnumberone.io/games)**. Test manually in [Control Basics](https://agentnumberone.io/challenges/control-basics) or [Game Lab](https://agentnumberone.io/lab), exchange instructions in the game chat, and inspect received video and model inputs at **[localhost:8767/live](http://127.0.0.1:8767/live)**.

This example includes our live integration, image preprocessing/feature reuse, reusable system-prefix state, overlapping action scheduling, installation and profiling scripts. It does not train or quantize the model and makes no claim that GameScaling solves these browser games.

## Model and input contract

- Publisher: [`THUSI-Lab/GameScaling-0.8B-S3`](https://huggingface.co/THUSI-Lab/GameScaling-0.8B-S3), revision `1b89640dcf2ef51fb53ad9338a0725e9622dd200`.
- Inference client and prompt contract: [THUSI-Lab/GameScaling](https://github.com/THUSI-Lab/GameScaling/tree/6764efabfbb4121d6e5d3cfcbc54dedf1469bf5a), pinned at `6764efabfbb4121d6e5d3cfcbc54dedf1469bf5a`.
- **Five historical image/action turns plus the current image: six images at steady state.** Warm-up has fewer turns. Images are not dropped or replaced by a text summary, and this integration performs no extra resize. The publisher's image processor still applies its native patch/grid preprocessing.
- Exact upstream output format and parser, `<think>` assistant prefix, temperature `1.0`, maximum `2048` generated tokens, six control slots at 30 Hz per 200 ms action chunk. The model may produce empty reasoning; conversation remains enabled.
- Original BF16 weights are downloaded and SHA-256 checked. Default inference casts floating parameters to **IEEE FP16**, without quantization. This is a precision cast, not bit-identical BF16 execution. Set `ANO_GAMESCALING_PRECISION=bf16` to retain BF16.

The exact upstream [prompt template](upstream/prompts/system_prompt_template.txt) and [Doom example prompt](upstream/prompts/example_doom_battle1.txt) are included unchanged. Our environment descriptions live in the template's game-rules/key-bindings sections. Operator instructions and the unverified commanded pointer are appended to each observation. Use `ANO_GAMESCALING_PROMPT_PROFILE=general`, `pixel-duel`, or `crossroad`. The selected full system prompt is written to the local run directory.

## Install

Use Python 3.12 on an Apple Silicon Mac. Run one model process at a time.

```sh
python3 -m venv .venv-gamescaling
.venv-gamescaling/bin/python -m pip install -r GameScaling0.8B-S3/requirements.txt
.venv-gamescaling/bin/python GameScaling0.8B-S3/install.py
```

Weights, logs, the private pairing file and latest model image default to `~/Library/Application Support/AgentNumberOne/gamescaling/`, outside Documents/iCloud. Override with `ANO_GAMESCALING_DATA` and `ANO_GAMESCALING_MODEL_PATH`. The optional `prepare_fp16.py` converts a separate on-disk copy in bounded CPU chunks; ordinary startup casts the downloaded source in memory and needs no second weight copy.

## Connect to production

Put a trusted TLS WebSocket reverse proxy in front of `127.0.0.1:8767`. The proxy must preserve the production `Origin` header. Set its URL, without a token:

```sh
export ANO_GAMESCALING_PUBLIC_URL=wss://your-endpoint.example/control
.venv-gamescaling/bin/python GameScaling0.8B-S3/webrtc_live.py
```

The runner prints the location of `connection.txt`, never its secret. Copy that file's paired URL into **Connect an agent → Connect through WebRTC** on the production page. A production HTTPS page requires trusted `wss://`; local plain `ws://` is for HTTP loopback previews. Configure the exact preview origin with `ANO_GAMESCALING_ORIGIN` if needed. Pairing is checked against that one origin, and only one browser peer is accepted at a time.

Open **[http://127.0.0.1:8767/live](http://127.0.0.1:8767/live)** in a separate tab on the machine running the endpoint. Received video and the selected current image appear side by side; historical images remain in the inference context. The viewer refreshes at up to 5 Hz, independently of video FPS and inference speed. Diagnostics and frame routes reject non-loopback requests and non-local Host headers. Keep those routes out of the public reverse proxy.

For DOM/WebGL games, share the current tab when prompted. Production Element Capture excludes the floating human Inputs/chat overlays and retains game menus and the visible agent cursor. No localhost capture fixture is needed for normal use. A paused game continues perception/chat, with gameplay controls blocked.

## What is reused

`mlx_runtime.py` and `preprocess_cache.py` keep the model's six-image context intact:

1. Prepare the same full multimodal prompt and all six images. Reuse the original processor’s per-image resize/normalize/patch tensors when pixels and settings match; preserve the original assembly and tokenization. New images are processed normally.
2. Reuse features only when RGB pixels, dimensions and processor grid match exactly. Cache size is bounded to the active image window; new images are encoded. All six feature tensors are concatenated in prompt order.
3. Restore a detached checkpoint at the end of the unchanged **text-only system prefix**, including both attention KV and recurrent state. Compute full-prompt multimodal positions, then process every remaining observation/action token normally.
4. Generate the native reasoning/action continuation and stream its own reasoning to the browser chat.

The model uses Qwen3.5 hybrid attention. Its recurrent state cannot generally be rolled backward like a conventional attention KV cache. We **do not** reuse shifted historical KV state, omit history tokens, or simply append the next frame to a stale cache. Rendering the full prompt on the CPU is still necessary to verify the prefix and preserve the six-image contract; GPU processing of the unchanged system prefix is skipped after the first call.

Separate-image encoding and split prefill can change floating-point rounding. Feature reuse is exact with respect to cached pixels/grid, but batching changes are not guaranteed to generate bit-identical outputs. Run `profile_runtime.py` for numerical and performance checks on your hardware.

## Inference and controls overlap

Video decoding continues while the single model worker generates. Once an action is accepted, the event loop queues its first slot and schedules the remaining slots over 200 ms. The next inference selects a frame received after that first slot was queued and can run while the earlier action finishes. Only one action chunk executes at a time; pause, instruction/context changes, staleness and disconnect are checked before every remaining slot and before a new result is used.

This overlap is a harness optimization relative to the upstream client's sequential 200 ms loop. The image/history count, control slot count and action vocabulary remain the same. History describes **commanded** controls, not proof that the game applied them. `execution_complete` records whether a chunk remained admissible through its end; there are no per-action game acknowledgements. A context change may interrupt a chunk already included in a pending observation's commanded history; its pending output is rejected.

For controlled comparisons, set `ANO_GAMESCALING_PREPROCESS_REUSE=0`, `ANO_GAMESCALING_VISION_REUSE=0`, `ANO_GAMESCALING_PREFIX_REUSE=0`, or `ANO_GAMESCALING_OVERLAP=0` before startup. These switches do not alter six-image history, reasoning or sampling settings.

## Metrics and profiling

Each local `events.jsonl` decision records frame/context identity, operator instruction ID, input message, native output/reasoning, parsed action, queued commands and disposition, plus:

| Field | Measurement |
| --- | --- |
| `prepare_ms` | Image processor/tokenization and input tensor preparation |
| `preprocess_cache_hits`, `images_preprocessed` | Historical patch-tensor reuse and actual image-preparation count |
| `vision_ms` | Cache lookup, any new image encoding and feature concatenation |
| `vision_cache_hits`, `frames_encoded`, `images_in_context` | Actual reuse and unchanged image count |
| `system_prefix_ms`, `system_prefix_tokens`, `system_prefix_hit` | Static-prefix restore/build and tokens skipped from subsequent prompt processing |
| `first_chunk_ms` | Inference start through first streamed text chunk, including preparation |
| `generation_ms` | Language-model processing/generation after explicit preparation; includes uncached prompt prefill |
| `language_prefill_to_first_chunk_ms`, `after_first_chunk_ms` | Language-model start to first streamed chunk; remainder after that chunk |
| `generation_tokens`, `generation_tps` | Decoder-reported output count/rate |
| `inference_ms`, `end_to_end_ms` | Inference duration; broader policy duration including image snapshot/template work |
| `inference_started_at_monotonic`, `action_started_at_monotonic` | Same-process timestamps to verify overlap with adjacent decisions |
| `receive_to_end_ms` | Receiver-side frame reception through action-chunk completion; excludes browser capture/encode/transit before reception |

`first_chunk_ms` is not an isolated vision timer; `generation_ms` is not pure decoding. GPU stages use explicit evaluations where timed, but these are wall-clock measurements rather than a Metal kernel trace. The live display shows only a compact subset; the JSON retains the full metrics and conversation. Local logs may contain private operator messages and screenshots: they are ignored by Git.

Stop the live runner before profiling. Supply at least six recorded game images. With eight images and the default two measured repeats, the six-image window advances one frame per decision:

```sh
.venv-gamescaling/bin/python GameScaling0.8B-S3/profile_runtime.py \
  --frames frame1.png frame2.png frame3.png frame4.png frame5.png frame6.png frame7.png frame8.png \
  --output "$HOME/Library/Application Support/AgentNumberOne/gamescaling/results/reuse-profile.json"
```

The profiler compares cold/cached last-token logits, repeated snapshot restoration and shifted image history. It separately benchmarks baseline, image reuse, and image-plus-system-prefix reuse with identical six-image inputs and bounded greedy generation. Those benchmark settings are disclosed and do not change live temperature/reasoning. Six-frame mode repeats fixed images and can yield six cache hits. Supplying eight frames exercises sliding windows with five historical hits and one newly encoded image per measured decision. See [recorded evidence and limits](docs/profiling.md) and the [careful prefill/preprocessing experiments](experiments/README.md). The follow-up retained 2,048-token prefill blocks and enabled only the verified preprocessing cache (about 2.8% additional latency reduction in paired replay).

## Tests

```sh
.venv-gamescaling/bin/python -m unittest discover -s GameScaling0.8B-S3/tests -v
.venv-gamescaling/bin/python tests/webrtc-client.test.py
```

CPU tests cover six-image/prompt/instruction preservation, text streaming, per-slot cancellation, concurrent frame conversion, staleness and pairing-file handling. The hardware profiler is a separate Apple Silicon check; CI does not pretend to run an MLX model on Linux.

## Sources

The vendored inference client and prompts retain their upstream Apache-2.0 license in `upstream/LICENSE`. [`upstream/SOURCE.json`](upstream/SOURCE.json) pins their original commit and hashes. Model weights are downloaded separately under their publisher's license. No weights, pairing credentials, private recordings or machine-specific hostnames are committed here.
