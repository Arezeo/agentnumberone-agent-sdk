# GameScaling GGUF policy for the Agent Number One SDK

Run a **GameScaling model in GGUF form** (0.8B / 2B / 9B) against any game page on
agentnumberone.io, without the MLX kit and without converting weights.

Why this exists: the MLX kit (`gamescaling.py`) only accepts `0.8b-s3` and `2b-s3`
unquantized checkpoints at pinned revisions, and refuses packed weights outright. The
GGUF path is the only way to run **9B** — 5.6 GB Q4_K_M instead of 18 GB bf16 — and it
runs anywhere llama.cpp does, including a 16 GB Mac mini.

## Pieces

| file | role |
|---|---|
| `policy_llamacpp.py` | `decide(frame, controls)` adapter; parses the model's native action grammar |
| `serve.sh` | launches `llama-server` with sane flags |
| `install.sh` | llama.cpp + Python 3.12 venv + SDK deps |
| `gamescaling-system.txt` | the system prompt the model was trained against |

## Setup (e.g. mac mini)

```sh
./install.sh                       # brew llama.cpp, uv python 3.12, aiortc/av venv
cp GameScaling-9B-S3.Q4_K_M.gguf \
   GameScaling-9B-S3.mmproj-f16.gguf .
./serve.sh GameScaling-9B-S3.Q4_K_M.gguf GameScaling-9B-S3.mmproj-f16.gguf &
curl -s localhost:8901/health      # {"status":"ok"}
```

Then the SDK harness:

```sh
.venv-control/bin/python scripts/remote-control/agent.py \
  --origin https://agentnumberone.io --port 8765 \
  --policy packages/agent-sdk-llamacpp/policy_llamacpp.py
```

## The action grammar (do not invent your own)

```
<|action_start|>X Y Z ; g1 ; g2 ; g3 ; g4 ; g5 ; g6<|action_end|>
```

- `X Y` — **relative** pointer motion in per-mille of screen size (X>0 right, Y>0 down).
  Not absolute coordinates. The policy integrates them into a tracked pointer.
- `Z` — scroll.
- six groups of ≤4 keys, executed as 33 ms steps (a 200 ms plan).

Keys use the model's names: `W A S D Space Shift LB RB MB Esc Tab Up Down Left Right One Two…`
Numbers are lowercase words (`one` = `1`); `One` (capital) means **F1**.

**Asking for JSON gets you the JSON template echoed back.** We tried it: the model
returned `{"move":[0,0],"action":"none"}` for 100+ consecutive frames. Use the native
grammar or nothing.

## Known failure: in-format, zero content

The model can emit perfectly-formed but empty actions:

```
<|action_start|>0 0 0 ;   ;   ;   ;   ;   ;  <|action_end|>
```

Observed for ~31,800 decoded frames — the agent never moved. Causes, in order:

1. **The system prompt describes a different game.** `gamescaling-system.txt` is written
   for *Pixel Duel*. Running it on Voxelcraft/Minecraft produces exactly this: the model
   answers in-format with no referent.
2. The frame is unreadable (letterboxed, tiny, or mostly black).
3. Wrong modality — mmproj not loaded (`--mmproj` missing), so the model is blind.

The policy prints `WARNING: N consecutive empty actions …` after 5 in a row. If you see
it, fix the prompt before tuning anything else.

## Performance expectations

| model | weights | throughput | decision |
|---|---|---|---|
| 0.8B Q4 | ~0.6 GB | fast | ~0.3 s |
| 2B Q4 | ~1.6 GB | ~110 tok/s warm | ~1.4 s |
| 9B Q4_K_M | 5.6 GB + 0.9 GB mmproj | **13 tok/s** | **~1.3 s** |

13 tok/s sounds bad but the decision is short: the 9B measured **1.27 s/decision** — faster
than the 2B through the MLX kit (2.26 s), because llama.cpp has no prefill machinery in
front of it.

Memory: the 9B needs **~7 GB resident** (mmap'd, so it pages rather than allocating all at
once). On a 16 GB machine, check what else is resident first — a Docker VM holding 8 GB
leaves no room.

## Rules

- **Verify the model emits non-zero deltas** before starting a match.
- **One orientation per process.** Vertical (TikTok/IG) needs its own capture with a 9:16 crop.
- **Release held keys before any interruption** — the policy does this at the top of each
  `decide`, which is also what makes a restart clean.
- **Never run two policies against one page.** The SDK will accept the second pairing and
  both will fight for the pointer.
