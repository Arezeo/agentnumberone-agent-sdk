# Prefill and preprocessing experiments — 2026-10-23

**Keep 2,048-token prefill blocks. Enable exact processor-tensor reuse.** The additional measured gain is about **24 ms per decision (2.8%)**, not another large inference speedup. Six-image history, FP16 weights, `<think>` prefill, native action format, temperature 1 and the 2,048-token live output limit are preserved.

## Method

Only one model process ran at a time; the live endpoint was stopped during hardware tests and a clean copy of the previous revision was restored automatically afterward. No training, quantization, downscaling, frame dropping or forced empty reasoning was used.

Two replay corpora were used: ten unmodified model-input PNGs captured from the current production Crossroad session, and ten earlier Doom frames. Each corpus provided four successive six-image windows, each containing five historical frames plus the current frame. Each runtime variant received the same prompt, fixed historical actions and images for a given comparison. Every variant was warmed, and the measured order rotated by case and reversed on alternate rounds.

The timing runs used greedy sampling and a 48-token cap solely to make continuations comparable; they generated 31–33 tokens. Native live sampling was tested separately. An explicit 10 GiB MLX allocation limit bounded the experiments. Ordinary desktop apps remained running. Load averages and process-wide peak memory are in the JSON. The observed maximum MLX peak was 6.77 GiB in the prefill sweep and 3.28 GiB in the later cache run; these are process peaks, not isolated per-variant memory measurements.

Uncertainty bounds are exploratory repeated-window-cluster bootstrap estimates. Repeated timings of the same corpus/window are clustered; there are eight distinct windows, not 32 independent game sessions. These tests neither measure game scores nor establish universal quality equivalence. Compare configurations **within** each run; machine load changed between runs, so their absolute latencies cannot be combined into an additional speedup claim.

## Prefill block sweep

Each setting has 16 measured decisions across the two corpora and two timing rounds. Positive reduction means faster than the 2,048 baseline. All greedy outputs matched their paired baseline.

| Tokens per prefill block | Mean latency | Paired latency reduction | Exploratory 95% interval | Decision |
| --- | ---: | ---: | ---: | --- |
| 512 | 1,192 ms | −3.8% | −5.75% to −1.72% | Slower |
| 1,024 | 1,162 ms | −1.2% | −2.65% to +0.43% | Keep default |
| **2,048** | **1,148 ms** | Reference | — | **Retained** |
| 4,096 | 1,152 ms | −0.4% | −2.23% to +1.69% | Keep default |
| 8,192 | 1,132 ms | +1.5% | −0.42% to +3.14% | No reliable gain |
| No chunking | 1,446 ms | −25.9% | −30.79% to −21.08% | Slower |

[Measurements](../benchmarks/prefill-sweep-20261023.json). Different chunking can alter floating-point results slightly; the largest first-token log-probability relative L2 difference was about 0.000803. No setting changed a tested greedy continuation.

The first preprocessing wrapper was also included in that sweep, but it recorded **zero cache hits**: the custom MLX image processor bypassed its `preprocess()` method. That variant was a no-op and is explicitly excluded as evidence for preprocessing improvement. The corrected implementation hooks the actual per-image `_process_one()` operation, checks that it ran, and fails instead of silently reporting success when bypassed.

## Corrected preprocessing cache

Before timing, original and cached `input_ids`, masks, image grids and normalized patch tensors were compared across five windows per corpus. **Every value, shape and dtype matched exactly.** The processor still assembles all six images and expands the same image placeholders.

Four timing rounds produced 32 measured decisions per variant. Every moving window had five preprocessing-cache hits and one newly processed image.

| Configuration | Mean total | Mean first text chunk |
| --- | ---: | ---: |
| Existing vision/system-prefix reuse | 851.9 ms | 595.9 ms |
| Plus preprocessing-tensor reuse | 827.7 ms | 571.5 ms |

The mean paired latency reduction was **2.84%**, with a window-cluster bootstrap interval of approximately **2.68–3.02%**. All 32 greedy continuations and token sequences matched; first-token log-probabilities were identical. [Measurements](../benchmarks/preprocessing-cache-20261023.json).

The cache retains only the active input window. Keys include image shape, dtype and pixels, processor resize/normalization/patch settings, and call overrides. Context reset clears the cache, and the original processor method is restored even on exceptions. It is restricted to the pinned MLX processor and the single model-owner thread. Set `ANO_GAMESCALING_PREPROCESS_REUSE=0` to return to the prior preprocessing path.

## Native sampling and text

A separate test called the actual `NativePolicy.infer()` with **temperature 1, maximum 2,048 output tokens, five historical turns, six images, and the original `<think>` prefix**. Both variants received the same operator instruction and random seed. Order alternated between baseline-first and cache-first.

The initial four pairs all matched, but neither variant generated reasoning, so that run did not satisfy the intended text-coverage check. It is retained as an [initial coverage result](../benchmarks/native-preprocessing-initial-20261023.json), not presented as a successful conversational validation.

The expanded test used seeds 0–7 on each corpus with a changed-objective instruction. **All 16 pairs matched exactly**, including five pairs with native reasoning of 315–388 characters and a final text callback. All outputs parsed as native actions. This checks model output and text emission; it is not a test of production WebRTC delivery, general conversational quality or action application. [Output hashes and measurements](../benchmarks/native-preprocessing-20261023.json). Production screenshots and native descriptions of them remain local; the public report contains hashes and counts.

## Speculative decoding

No speculative decoder was enabled. The pinned checkpoint declares one MTP layer in its configuration but contains **zero MTP/draft tensors**. No compatible external drafter was supplied or validated. This is a feasibility finding, not a speculative-decoding performance benchmark. [Recorded check](../benchmarks/speculative-feasibility-20261023.json).

## Reproduce

Stop the live model first and use the environment from the [example README](../README.md). Store ten local production images in a manifest of the form `{"frames": [{"path": "/absolute/path/frame-00.png"}, ...]}`. Supply ten recorded Doom images as well. Screenshots and output files should remain outside the checkout and outside synced Documents folders.

```sh
.venv-gamescaling/bin/python GameScaling0.8B-S3/experiments/prefill_and_preprocessing.py \
  --production-corpus /path/to/corpus.json \
  --doom-frames /path/to/doom/input-*.png \
  --prefill-only --rounds 2 --output /path/to/prefill-results.json

.venv-gamescaling/bin/python GameScaling0.8B-S3/experiments/prefill_and_preprocessing.py \
  --production-corpus /path/to/corpus.json \
  --doom-frames /path/to/doom/input-*.png \
  --baseline-only-cache --rounds 4 --output /path/to/cache-results.json

.venv-gamescaling/bin/python GameScaling0.8B-S3/experiments/validate_preprocessing.py \
  --production-corpus /path/to/corpus.json \
  --doom-frames /path/to/doom/input-*.png \
  --seeds 0 1 2 3 4 5 6 7 \
  --instruction 'New objective: analyze the scene carefully and explain why your next movement is safe before taking action.' \
  --output /path/to/native-validation.json
```

Native validation fails its coverage check if all continuations omit reasoning; optional reasoning is never forced by injecting text into the assistant prefix. Results are hardware/workload dependent. The runner remains inspectable at [localhost:8767/live](http://127.0.0.1:8767/live) after restarting and reconnecting the production page.
