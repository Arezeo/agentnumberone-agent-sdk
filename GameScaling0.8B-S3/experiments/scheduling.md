# Inference scheduling experiments

**Decision: keep the live runner’s scheduling unchanged.** GPU prefetch can reduce combined work in a conditional exact-hit case while delaying the current action; changed frames waste that work. CPU-only results did not establish a reliable action-latency improvement.

The production runner already receives video while inference runs and overlaps the previous 200 ms action chunk with the next inference. Inside `NativePolicy.infer()`, CPU input preparation, vision encoding, language prefill and token generation still run in order on one model-owner thread.

This experiment checks two separate questions:

1. Would precomputed image work actually be reusable under the existing frame-freshness rule?
2. Does overlapping that work reduce total completion time without delaying the current action?

## Admission constraint

The selected observation must arrive after the previous action starts. An older prefetch candidate can provide computation only if its pixels and processor identity exactly match that selected observation. It never replaces the selected observation's frame ID or timestamp. Context changes, pixel changes, future candidates, non-finite timestamps, mixed clock domains and processing changes reject reuse. This experimental gate covers computation reuse; it does not replace transport authorization, age limits or context expiry checks. Six-frame context, FP16 and native optional reasoning remain intact.

`prefetch_admission.py` encodes these rules and has CPU tests for each rejection case.

## Live trace

A read-only 12-second trace sampled the localhost received-JPEG observer at 5 Hz. In six observed action boundaries, the latest sampled candidate available at least 40 ms before the action differed from the first sampled eligible post-action frame. There were **zero potential matches** in this small trace.

JPEG equality could conceal raw-RGB differences and therefore would only indicate a potential hit. No raw-pixel hit is claimed. The trace is also sparse: it cannot establish the hit rate of candidates captured closer to the boundary or of other games. Images remain local.

An initial trace using system Python 3.9 was excluded because its process-relative monotonic clock did not match the Python 3.12 model's clock domain. The retained trace uses the same Python 3.12 interpreter as the model and checks clock consistency before comparing timestamps.

## Hardware protocol

The live endpoint is stopped during GPU experiments, with a clean snapshot of its previous revision restored automatically afterward. Only one FP16 checkpoint is loaded. A 10 GiB MLX memory cap bounds the experiment.

The measured work is **one full six-image decision A plus image preparation/encoding for a selected next observation B**. This is a conditional compute experiment, not a measured B decision, live action frequency or game-score evaluation. B's language inference is not included.

Compare:

- **Serial:** finish A, then prepare and encode B.
- **CPU prefetch:** after A's first token, prepare candidate C with a cloned, NumPy/PIL-only processor on one CPU worker; encode it after A only if admission succeeds.
- **GPU prefetch:** prepare C on that CPU worker, then submit its vision work on a separate MLX GPU stream from the single model-owner thread between decoded tokens. No second thread owns MLX model state.

Two explicit fixtures cover an exact-pixel match and changed pixels. The candidate precedes the action boundary; the selected observation follows it and retains its identity. An exact-hit fixture is a conditional opportunity, not a forecast that the live next image will match. On a miss, candidate work is discarded and B is prepared/encoded normally. Every pending GPU candidate is evaluated and synchronized before the end timestamp, including wasted work.

Both corpora use the same full six-image prompts and observations for every compared variant. Every A continuation and B feature tensor must match its serial counterpart. Greedy sampling and a 48-token cap make this an inspectable replay; the live runner stays at temperature 1 and a 2,048-token maximum. One warm-up round is excluded; variant order rotates and reverses across the remaining rounds. Report both total completion time and **A's action-ready time**, so a throughput benefit cannot conceal worse action latency. Separate-stream submission does not by itself prove that GPU kernels overlapped.

## Results

Each condition has eight measured runs (two corpora, four repeated rounds), plus excluded warm-ups. These are repeated timings of two replay cases, not eight independent games. Ordinary desktop applications remained active; a mid-run load average was about 12.8/10.3/8.6. Absolute timings must not be compared with earlier experiments conducted under different load.

| Condition | Variant | Mean A + B-feature work | Paired total reduction | Mean change to A action-ready time |
| --- | --- | ---: | ---: | ---: |
| Exact pixels | Serial | 1,729 ms | Reference | Reference |
| Exact pixels | CPU prefetch | 1,672 ms | +3.4% | −46 ms, inconclusive |
| Exact pixels | GPU prefetch | 1,640 ms | +5.5% | **+64 ms delay**, noisy |
| Changed pixels | Serial | 1,757 ms | Reference | Reference |
| Changed pixels | CPU prefetch | 1,691 ms | +3.2%, inconclusive | −67 ms, inconclusive |
| Changed pixels | GPU prefetch | 1,939 ms | **−10.7% (slower)** | **+175 ms delay** |

Positive total reduction means faster combined work; negative action-ready change means earlier A output. The GPU exact-hit total-reduction bootstrap interval was approximately +1.95% to +9.68%, but A's delay interval was −4 to +124 ms. The GPU miss penalty was clearer: total reduction −17.26% to −4.65%, and A delayed +88 to +256 ms. CPU-only action-ready intervals included no improvement. All intervals are exploratory small-sample estimates under variable desktop load.

Because candidate work starts only after A's first decoded chunk, differences before that point are unrelated runtime/load variation. The JSON includes secondary post-first-chunk diagnostics: in the GPU exact-hit fixture, that portion of A was about **119 ms longer**, while combined work after the first chunk was about **34 ms shorter**. In the GPU miss fixture, the post-first-chunk action portion was about **153 ms longer** and combined work about **160 ms longer**. These diagnostics do not replace full latency measurements or establish a production speedup.

All **48 measured A continuations and selected-B feature tensors** matched their paired serial results (including the serial reference rows). The CPU helper ran during A in each prefetch trial; GPU work was submitted before A output in each GPU-prefetch trial. This verifies scheduling and result preservation on the fixtures, not actual simultaneous GPU kernel execution. Peak MLX memory was approximately 3.0 GiB; only one checkpoint was loaded.

[Hardware measurements](../benchmarks/scheduling-prefetch-20261023.json) · [Sanitized live trace](../benchmarks/scheduling-live-trace-20261023.json). Screenshots and machine-specific paths remain local.

The live trace had no potential useful hits in six sampled boundaries. Combined with current-action delays and miss penalties, these results do not support replacing the live scheduler. CPU prefetch remains an experimental lead, with no demonstrated benefit to production decision latency. No native-temperature acceptance or production prefetch integration was attempted. The existing input/vision/system-prefix caches and action/inference overlap remain enabled.

## Reproduce

Use the installed Apple Silicon environment and a ten-image manifest as described in [the earlier experiment guide](README.md). Stop the live model first.

```sh
.venv-gamescaling/bin/python GameScaling0.8B-S3/experiments/scheduling_prefetch.py \
  --production-corpus /path/to/corpus.json \
  --doom-frames /path/to/doom/input-*.png \
  --rounds 4 --output /path/to/scheduling-results.json
```

Production integration requires measured useful hits, exact admission, bounded pending work, context/disconnect cancellation, unchanged native text/actions and a benefit in actual decision latency. This experiment does not change the production runner.
