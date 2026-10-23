# Inference scheduling experiments

Status: hardware measurements in progress. The live runner's scheduling defaults are unchanged.

The production runner already receives video while inference runs and overlaps the previous 200 ms action chunk with the next inference. Inside `NativePolicy.infer()`, CPU input preparation, vision encoding, language prefill and token generation still run in order on one model-owner thread.

This experiment checks two separate questions:

1. Would precomputed image work actually be reusable under the existing frame-freshness rule?
2. Does overlapping that work reduce total completion time without delaying the current action?

## Admission constraint

The selected observation must arrive after the previous action starts. An older prefetch candidate can provide computation only if its pixels and processor identity exactly match that selected observation. It never replaces the selected observation's frame ID or timestamp. Context changes, pixel changes, future candidates and processing changes reject reuse. Six-frame context, FP16 and native optional reasoning remain intact.

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

## Reproduce

Use the installed Apple Silicon environment and a ten-image manifest as described in [the earlier experiment guide](README.md). Stop the live model first.

```sh
.venv-gamescaling/bin/python GameScaling0.8B-S3/experiments/scheduling_prefetch.py \
  --production-corpus /path/to/corpus.json \
  --doom-frames /path/to/doom/input-*.png \
  --rounds 4 --output /path/to/scheduling-results.json
```

Production integration requires measured useful hits, exact admission, bounded pending work, context/disconnect cancellation, unchanged native text/actions and a benefit in actual decision latency. This experiment does not change the production runner.
