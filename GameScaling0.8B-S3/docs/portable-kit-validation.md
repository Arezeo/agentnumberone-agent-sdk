# Portable FP16 kit verification

Checked locally on **2026-10-23**, Apple M5 Pro, 24 GiB unified memory, macOS 26.5.1, Python 3.12.14. A new virtual environment installed the declared dependencies successfully; `pip check` found no conflicts. Both installers verified the original checkpoints' pinned revisions/SHA-256 using existing local weight files. We did not redownload 7.7 GB merely to test installation.

Ordinary desktop apps, sync services and a local development VM remained running. The other model endpoint was stopped and GPU tests ran one model at a time. These are compatibility checks and small local measurements, not an idle-machine performance study or paper reproduction.

## CPU, transport and real model checks

**56 tests passed:** 35 example contracts/cache/scheduling tests, 18 SDK tests including actual encoded WebRTC video exchange, and 3 distribution/model-selection tests. CI runs these without model weights; native MLX checks below ran separately on the Mac.

| Check | 0.8B-S3 FP16 | 2B-S3 FP16 |
| --- | --- | --- |
| Original weight SHA-256 | Verified | Verified |
| Real six-image model call | Parsed native action | Parsed native action |
| Floating parameter types | `mlx.core.float16` only | `mlx.core.float16` only |
| Native model with two real WebRTC peers | Passed | Passed |
| Six-image context reached | Yes | Yes |
| Operator instruction delivered | Yes | Yes |
| Native text channel open | Yes | Yes |
| Native text snapshots received in loopback run | See JSON | See JSON |
| Local viewer image/state routes | Passed | Passed |
| Nonlocal viewer Host rejected | Yes | Yes |

The initial standalone smoke fixtures generated valid **no-op** packets for both models. Native reasoning is optional: a valid packet does not require nonempty prose. Loopback checks use moving synthetic pixels; they exercise real MLX, codec, peer connections, instructions, controls and viewer HTTP routes. They bypass in-process WebSocket signaling; separate SDK tests cover token/Origin authentication. No production browser, WSS reverse proxy or WAN reachability was certified by these loopback checks.

## Cache replay

Eight 1280×720 frames captured from the local Doom check, a general system prompt, five fixed historical action turns and a current frame were used. Each variant had one warm-up window and **two measured sliding windows**, retaining all six images. All sampled greedy continuations matched their corresponding baseline. Normalized last-token-logit L2 differences stayed below the existing **0.003** assertion for repeated prefix restores and shifted history; this is not bit-identical numerical execution or a universal quality guarantee.

| Variant, mean elapsed | 0.8B-S3 FP16 | 2B-S3 FP16 |
| --- | ---: | ---: |
| Baseline | 1,262 ms | 2,850 ms |
| Vision reuse | 813 ms | 1,726 ms |
| Vision + system prefix | 776 ms | 1,634 ms |
| Preprocessing + vision + system prefix | 745 ms | 1,566 ms |
| Process-wide peak MLX allocation across replay variants | 8.26 GiB | 10.79 GiB |

The replay uses temperature 0 and a 48-token cap for comparable continuations, with 2,048-token prefill blocks. Live/native checks retain temperature 1 and maximum 2,048 output tokens. Fixed ordering, two measured samples, ordinary background load and one frame corpus limit interpretation. These numbers do not establish sustained decision Hz, a statistical speedup confidence interval, total system memory requirements or game success. The eight source screenshots are not bundled; use your own recorded frames with the documented profiler.

## Bounded Doom checks

Both used the exact vendored publisher Doom prompt and pinned DFP Battle-1 scenario with the disclosed local ViZDoom/Freedoom/FOV adapter. The engine continued advancing during inference.

| Result | 0.8B-S3 FP16 | 2B-S3 FP16 |
| --- | ---: | ---: |
| Decisions / parsed packets | 8 / 8 | 6 / 6 |
| Decisions containing movement | 8 | 0 |
| Decisions containing fire | 0 | 0 |
| Kills | 0 | 0 |
| Final health | 100 | 100 |
| Engine controls matched requested vectors | Yes | Yes |

These short runs establish local execution, not Doom competence. In particular, **2B chose no-op packets in all six Doom decisions**. It should not be described as successfully playing Doom or solving Control Basics. Native FP16 and optional text/action compatibility do not guarantee a model follows instructions or generalizes to a new game.

## Fresh ZIP

The release ZIP is built from committed blobs with a source hash manifest, without weights, credentials, logs, recordings or a virtual environment. Fresh-extraction verification is recorded in the release PR. Follow the [full setup and manual production procedure](../../docs/gamescaling-kit.md) to evaluate your own browser/network environment.

The [sanitized JSON evidence](../benchmarks/portable-kit-validation.json) contains model revisions, source-independent measurements, numerical checks and loopback outcomes. Raw local traces include native output and evaluator events and remain outside the repository.
