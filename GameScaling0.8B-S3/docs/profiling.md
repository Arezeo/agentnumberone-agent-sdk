# Profiling evidence

Measured on 2026-10-23 on an Apple M5 Pro MacBook with 24 GB unified memory. These are small, local experiments under ordinary desktop load, not paper-throughput reproduction or proof of gameplay skill. Only one model process ran during profiling.

## Current WebRTC runtime, FP16

[`fp16-sliding-window.json`](../benchmarks/fp16-sliding-window.json) uses eight recorded Doom images, a Crossroad system prompt, five fixed historical actions, and three consecutive six-image windows. The same window/action prompt is used for each runtime variant. The first window warms the runtime; the following two are measured. This is a computation replay, not a Crossroad game evaluation. All three variants emitted the same greedy text for each matched window.

| Runtime | Mean total | Mean first chunk | Historical features reused | New images encoded |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 1,724 ms | 1,373 ms | 0 | 6 |
| Vision-feature reuse | 1,172 ms | 822 ms | 5 | 1 |
| Vision + system-prefix reuse | 1,114 ms | 761 ms | 5 | 1 |

The combined variant reduced elapsed time by approximately **35%** in this replay. The unchanged system prefix contains **861 tokens**, restored from an actual prefix checkpoint; the full prompt still has **6,395 tokens**, including all six images. Image preparation averaged roughly 64 ms, the one new image plus feature lookup/assembly roughly 115 ms, and warm system-prefix handling roughly 9 ms. Uncached multimodal/language prompt processing and output generation remain significant.

Sampling for these paired measurements is greedy (`temperature=0`), with a 48-token cap and the original `<think>` prefix. The generated continuation ended at 33 tokens. **Live settings remain temperature 1, maximum 2,048 tokens and optional reasoning.** Do not use the replay latency as a promise for conversational live decisions.

Repeated restoration produced the same cached logits on both checks. Comparing full-batch cold inference with separately cached vision and split-prefix inference gave relative L2 logit differences of approximately **0.00260** for the initial window and **0.00239** for the shifted window. The highest-scoring next token matched. Floating-point results are not bit-identical, and a few successful checks do not establish unchanged gameplay quality across all inputs.

[`fp16-fixed-window.json`](../benchmarks/fp16-fixed-window.json) records an earlier all-six-features-cached replay: baseline 2,209 ms and combined reuse 1,275 ms. That workload and desktop load differ from the sliding-window run. Six cached images are an optimistic static-input case; the sliding-window result above is the more relevant reuse check.

## Historical Doom prototype

The earlier BF16 Doom prototype measured six full-history decisions per variant: approximately 1,202 ms without reuse and 838 ms with vision-feature reuse. It encoded six images versus one, respectively. See the [summary](../benchmarks/historical-doom-bf16-summary.json) and [vision feature comparison](../benchmarks/historical-doom-bf16-vision-parity.json).

That feature comparison found relative L2 difference `3.21755e-5` between batched and separately encoded BF16 features. Some greedy outputs differed in the earlier experiment; no bit-for-bit equivalence was claimed. The optimization originally existed in that separate Doom backend and was absent from the older WebRTC `NativePolicy`. This example ports it into the live WebRTC path and adds bounded static-prefix reuse, scheduling overlap, and stage metrics.

## What remains unmeasured

The replay does not measure production browser capture/encode latency, internet transit, actual game application time, score improvement, thermal steady state, or GPU kernel-level timing. The runner logs receiver-side timing and queued controls; it has no per-action application acknowledgement. The scheduler tests verify overlap and invalidation, but a browser must reconnect before collecting fresh production gameplay measurements.

## Follow-up: prefill and CPU preprocessing

The [2026-10-23 experiment report](../experiments/README.md) compared six prefill block settings and tested original processor-tensor reuse separately. No prefill setting reliably beat the existing 2,048-token default. The corrected preprocessing cache reduced paired replay latency by about 24 ms (2.8%), with bitwise input-tensor equality and all 32 greedy pairs matching. A separate 16-pair test at native live settings also matched every continuation, including five pairs with generated reasoning and final-text callbacks. This verifies computation and text emission, not a new production game score.
