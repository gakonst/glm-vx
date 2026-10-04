# Implemented parity tooling and current evidence

Subsequent work: [implemented optimization progress](OPTIMIZATION-PROGRESS.md) and
[numerical-mode corrections/stage replay](../validation/STAGE_REPLAY.md).
The original measurements and failed receipts below remain historical evidence.

This change adds executable trace validation and CPU prefill/decode handoff.
**Automatic optimization promotion remains blocked.** Neither real-model strict
trace comparison passed, and the long-context/held-out release corpus is not yet
complete. No tolerances were loosened to accept these results.

## Implemented tools

- `validation/trace_gate.py`: separate trusted contract and pinned reference;
  exact teacher-forced inputs, tensor/case coverage, identity and cache metadata;
  per-operation numerical budgets; full-logit distribution/top-k checks; exact
  discrete checks; first-failure receipts. Missing, skipped and malformed evidence
  fails. [Format and trust boundary](../validation/TRACE_FORMAT.md).
- `validation/native/llama_trace.cpp`: test-only independent llama.cpp exporter,
  one prescribed token at a time, all 78 layer outputs plus complete logits.
  Build with `validation/build_reference.sh` against pinned upstream commit
  `11fe02151f79c41d0d4af7da708755d73b9c0da6`. Bounded to 256 tokens; no long-context
  claim. It is never linked into serving.
- `validation/export_vx_trace.py`: observer copies layer outputs, selected DSA
  positions, current-token cache entries and complete logits without exposing
  live arrays for mutation. Observer errors roll back the failed token's cache.
- `validation/expanded_reference.py` / `export_expanded_trace.py`: independent
  expanded-KV, batch-causal F32 equations from the pre-existing mathematical
  oracle. This does not call the production compressed model graph. It is **not
  an execution of the official Transformers model**. It uses the GGUF decoder,
  whose sampled outputs were checked independently below.
- `validation/ggml_oracle.py`: pinned native GGML C codec comparison against
  GGUFStore, using independently read raw file offsets. All eleven formats are
  required by the CLI. Missing optional assets can skip unit tests but cannot
  produce a passing CLI receipt.
- [CPU process handoff](DISAGGREGATION.md), including GGUF identity support.

## Actual trained-weight comparisons

Both runs used the complete verified 216.7 GB checkpoint and the same prescribed
input IDs `[9703,10056]`, including an incremental cached second position. This is
teacher forcing, not free generation. Each numerical comparison covers **158
arrays: 78 layer outputs and full-vocabulary logits at each of two positions**.
Other Vx-only cache/selection captures are retained locally but are not included
as independently validated stages. Native reference KV uses upstream defaults;
the precision differences are explicit rather than claimed to be identical.

| Independent reference | Strict gate result | Diagnostic finding |
|---|---|---|
| Native GGML quantized CPU | Fail at position 0, layer 0 | Layer normalized error 0.01498; second-position top token differs |
| Expanded-KV F32 equations | Fail at position 0, layer 9 | Layer normalized error 0.000001652; one element exceeds the frozen absolute/relative bound |

The expanded-F32 full-logit normalized errors are **2.24e-6 and 3.92e-6**, with
matching top-five IDs at both positions. Maximum absolute logit errors are
6.48e-5 and 7.03e-5. Softmax total-variation errors are 3.74e-6 and 4.46e-6.
These small normwise differences are useful diagnostics, **not a passing result**
under the frozen elementwise contract. The 4e-6 absolute / 4e-5 relative and
normalized budgets were taken from existing F32 test criteria before comparing
the candidate; they are not a calibrated trained-model release specification.

Native-GGML logit normalized errors are 0.00742 and 0.13097, with second-position
top tokens 748 versus 594. Different native quantized activation arithmetic and
cache precision are plausible contributors; this experiment does not isolate
all causes. Matching two earlier generated tokens did not establish numerical
compatibility. Preserve the native failed receipt when deciding whether the
serving target is dequantized-F32 semantics or a GGML-compatible quantized mode.

[Native failed receipt](parity-evidence/native-ggml-strict-failure.json),
[F32 failed receipt](parity-evidence/expanded-f32-strict-failure.json),
[complete per-array diagnostics](parity-evidence/trace-diagnostics.json).
Raw reference/candidate arrays stay in the local `build/parity` directory;
receipts contain hashes and aggregate metrics, not raw activations.

## Component checks

The real-weight codec run performed **1665 exact comparisons**, covering all
11 formats, 185 sampled rows, 740 block samples and 135 expert rows. Maximum
error was **zero**. This validates those samples and boundaries, not every weight
in the checkpoint. [Receipt](parity-evidence/ggml-codec-parity.json).

Real-dimension DSA component tests cover lengths 2047/2048/2049/4096, 32 index
heads and 128 dimensions, negative head weights, exact selection sets and stable
selection-boundary ties. All five passed. This is not a trained full-model
2049-token execution. The gate's 59 mutation tests and trace observer's two tests
also passed. [Focused results](parity-evidence/trace-tests.txt).

## What still blocks automatic promotion

1. Establish and freeze a justified F32 numerical contract across a wider
   independent corpus, including near-zero/cancellation behavior. Preserve the
   failed strict receipts; never have an optimizer edit its own tolerance.
2. Add intermediate router/indexer/attention stages to the independent trace
   mapping, calibrated layer replay fixtures, and official eager small-model
   execution. Current independent real traces cover layer outputs and logits.
3. Add real >2048-token pruning, long decoding, additional prompts and held-out
   cases. The current exporters deliberately have bounded short-sequence scope.
4. Integrate worker pools and durable handoff into serving; measure client latency,
   throughput, transfer bandwidth and memory on matched hardware.
5. Require actual GPU correctness and race/performance checks before GPU promotion.

The isolated O3 library remains a candidate; `kernels/build.sh` has not been
silently switched to it. Native-GGML comparison, expanded-F32 comparison and
preservation of an existing engine binary are separate acceptance questions.

## Final regression receipt

After sourcing the installed Vx/LLVM environment, the combined suite passed
**455 tests plus 37 subtests**, with **14 real-GPU tests skipped**. The default
compiled CPU library also passed 50 native kernel checks. An initial run without
that environment failed three compiler-discovery tests; it was rerun with the
required toolchain on PATH, without changing tests or numerical tolerances.
[Full suite](parity-evidence/full-tests.txt),
[native checks](parity-evidence/native-kernel-tests.json).

The final separate-process Vx smoke passed through a serialized handoff; the
receipt includes distinct worker PIDs, generated token IDs and per-stage timing.
This uses tiny random weights; GGUF process tests use synthetic checkpoint files.
Neither is presented as a full trained-model disaggregated speed measurement.
