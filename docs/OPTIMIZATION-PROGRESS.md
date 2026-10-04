# Correctness and optimization implementation, 2026-10-04

This implements a first substantial part of the [research roadmap](SERVING-RESEARCH.md).
It does not establish an optimal production GPU engine or full trained-model
parity. Experimental faster paths remain explicit options. The default compiler
level remains O0; no failed historical gate was turned into a pass.

## Correctness work

[Stage replay](../validation/STAGE_REPLAY.md) now captures normalization,
attention/indexer stages, routing weights, per-expert outputs and reductions.
It found two concrete issues: NumPy implicitly promoted some supposedly F32
reference operations to float64; expert accumulation order differed from the
pinned official eager implementation. Both production and the corrected F32
reference now sum routed experts by ascending expert ID and add shared experts
last. The old reference mode remains available for reproducing old failures.

Actual unmodified, hash-checked official Transformers eager code executes in an
isolated optional validation environment. Two small random models, each with
four layers and sparse/shared attention, pass layer/logit checks and causal
selection checks against official batch and incremental execution. This is
stronger independence than another backend executing our same graph, but is
still small-model evidence. Real layer-0/layer-9 replay identifies accumulated
rounding and the shared-expert projection as important divergence locations.

A fresh corrected-mode run traversed all 78 trained layers for two prescribed
tokens and captured **31,172 stage arrays**. The independent F32 reference
captured the same stages, and all **456 discrete arrays** matched. Both final
top-five token lists match; full-vocabulary normalized L2 errors are 2.31e-6 and
5.80e-6, with softmax total variation 4.08e-6 and 5.72e-6. Nevertheless, the
unchanged strict elementwise gate still fails at `p0.layer.9.output`, element
136 (layer max absolute error 1.30e-4, normalized L2 1.83e-6). No threshold was
relaxed. [The full corrected receipt](optimization-evidence/corrected-f32-trained-regression.json)
retains the failure and both run identities. Two short prefixes do not test
trained long-context sparse selection, quantization quality, or all prompts.

### Pinned official implementation on trained weights

The stronger comparison now also executes actual pinned Transformers eager
forward methods with streamed GGUF weight storage across all 78 trained layers.
The official trace captured 920 arrays; the final Vx trace captured 31,172 stage
arrays. Its archive is bitwise identical to the previous corrected Vx trace.
The official storage adapter itself matches ordinary whole-model forward
bitwise on 260 tiny-fixture tensors/logits. Imported code, decoder, native
libraries and the actual opened checkpoint shard set are fingerprinted.

The trained comparison completed with **306/306 exact discrete checks passing**
(causal attention membership and expert membership). **112/308 numerical checks
failed** the unchanged budgets. The first is `p0.layer.9.output`, element 110,
with layer max absolute error 1.3733e-4 and normalized L2 1.7564e-6. The second
position's logits also exceed the elementwise threshold. These failures remain
visible and automatic release promotion remains disabled.

The two final top-five lists agree, but that does not override a failed numerical
contract. See the [summary](optimization-evidence/official-trained-summary.json),
[complete comparison](optimization-evidence/official-trained-comparison.json),
and [reproduction instructions](../validation/OFFICIAL_GGUF_TRACE.md).
The 53 comparison mutation tests cover stale/partial/malformed traces and
provenance, route pairing, both actual official index dtypes, and missing data.

This compares decoded GGUF F32 numerical forwards. It does not establish parity
with original FP8 activation arithmetic, native GGML activation quantization,
long-context trained sparse pruning, sampling distributions or all prompts.
The two-token trace is an initial full-depth check, not the held-out corpus still
required for release. The official harness changes storage/orchestration and
shares the independently checked GGUF decoder/mapping; it is not an independent
weight-conversion implementation.

## Implemented CPU and serving changes

- Exact stable heap top-k replaces repeated scanning of earlier selections.
  It uses only the output workspace, handles ties/signed zero and selection
  boundaries, and is tested through one-million-element inputs.
- Batched prefill processes a bounded chunk layer by layer, batches projections,
  and groups tokens by expert. Eight-, four- and two-row tiles use independent dot accumulators to reuse weights;
  emitted O3 assembly uses SIMD multiply/add across independent rows, without
  reassociating each dot product. Decode and chunked expert reductions use the
  same official order. Nonfinal chunks omit the vocabulary projection.
- [Packed Vx GGUF dot products](../kernels/PACKED.md) cover every one of the eleven
  formats in the trained checkpoint. They retain F32 activations and ordered
  accumulation while avoiding expanded weight matrices. Singleton prefill projections and expert groups stay packed, including the final
  vocabulary projection. Multirow groups expand each needed weight once per
  chunk; packed GEMM is still missing.
- [Bounded prompt snapshots](PREFIX-SERVING.md) reuse compressed KV, indexer
  state and final prompt logits. Hits restore independent state; repeated and
  extended prompts, sampling, eviction and cancellation are tested. This is
  instance-local prefix reuse, not distributed/paged GPU caching.
- Fixed output-event bounds terminate slow consumers without blocking the model
  owner. Cancelled admissions are reclaimed at the next model boundary. Cache
  and phase timings are exposed in health/terminal completion metadata.

A chunk is indivisible: smaller `--prefill-chunk` and
`--decode-prefill-tokens` values improve cancellation and decode responsiveness
at the expense of batching opportunities. Trace callbacks currently require the
single-token executor; batch mode rejects them rather than emitting partial
validation evidence.

## Automatic CPU candidate search

[The compiler search](../validation/CPU_SEARCH.md) builds isolated O0/O1/O2/O3
libraries, binds the actual candidate in the numerical/model tests, checks
serving/cache/cancellation behavior, and rejects missing or skipped required
checks. Source, test, workload, codec and binary fingerprints are fixed across
the run. It then compares repeated synthetic matrix and prefill workloads,
requiring exact O0 output equality before ranking candidates.

The result names an experimental artifact. It never replaces the default
library, and `release_eligible` remains false while trained parity is incomplete.
This is compiler-level search on fixed CPU workloads; it does not yet generate
new kernels, tune GPU layouts, or prove performance under production traffic.

## GPU implementation

[GPU GEMM](../gpu/GEMM.md) now includes a Vx tiled tensor-core path with actual
TF32 MMA, shared staging, residual masks and explicit precision selection. LLVM
verification, NVIDIA ptxas assembly for SM80/SM90 and compiled Vx CPU SIMT tests
pass. GPU execution, race checking and performance remain unvalidated because
this machine has no CUDA device. This initial single-stage kernel still needs
larger tiles, asynchronous staging and measurements; it is not called optimal.

## Measurements and their scope

| Measurement | Result | Scope |
|---|---|---|
| Stable top-128 from 4096 scores | 2.958 ms to 0.041 ms | Isolated O3 CPU kernel/host call; 11 samples |
| Eight-row projections at GLM-related dimensions | 1.81–1.90x faster | Synthetic resident F32 matrices; 7 samples; exact scalar-baseline equality |
| Two-/four-/six-row projection tails | 1.55–1.83x faster | Resident F32 matrices, O3 before/after, 11 alternating samples, bitwise equality |
| Layer-wise prefill, 8/32/64 tokens | 1.26–1.42x faster | Synthetic four-layer model; 5 samples; exact sequential equality |
| IQ1_S 128x6144 packed projection | 6.070 ms to 1.100 ms | Bounded warm trained rows; dequantize-plus-dot comparison, not cached-F32 comparison |
| Previously primed prefix HTTP requests | 116.82 to 419.26 output tokens/s | Tiny random model; priming excluded; all outputs matched |
| Full trained packed regression, two prescribed tokens | All 626 arrays bitwise identical | Saved pre-correction Vx O3 baseline; not independent correctness |

The last run took 206.2 s versus a historical 461.9 s baseline. This is a
historical timing comparison with uncontrolled cache/load differences, not a
matched throughput benchmark. Both compared runs intentionally retain the old
expert reduction order to isolate the packed implementation. Subsequent
corrected-order runs must be assessed separately.

Receipts: [top-k](optimization-evidence/topk-benchmark.json),
[projection](optimization-evidence/linear-batch-benchmark.json),
[prefill](optimization-evidence/prefill-tiled-benchmark.json),
[small batch tails](optimization-evidence/linear-tails-o3.json),
[packed trained regression](optimization-evidence/packed-trained-regression.json),
[packed format tests/timings](../kernels/packed-evidence/),
[prefix HTTP](../benchmarks/evidence/prefix-http-vx-tiny-20261004.json).
These measurements have different scopes and cannot be multiplied into a
predicted whole-model speedup. The host was shared with other work.

## Full trained HTTP proof of concept

The actual six-shard, approximately 216.7 GB GGUF ran on CPU over HTTP/SSE.
A six-token chat prefix was submitted twice, greedily generating two tokens.
Both requests returned IDs `[10056,752]`, text **"Let me"**. The first request
reported zero cached input tokens; the repeated request reused all six tokens.

| Request | First token | Next-token interval | Total |
|---|---:|---:|---:|
| Initial prompt | 1334.156 s | 126.737 s | 1460.899 s |
| Same prompt, prefix already retained | 0.0194 s | 112.444 s | 112.464 s |

This is a **smoke test**, not a controlled serving benchmark. The run used
`--packed-weights --batched-prefill`, an explicit O3 library, one GGUF decode
thread and unset BLAS thread variables on a shared host. A separate official
reference ran concurrently during part of it. The fast repeated TTFT reuses
cached final-prompt logits as well as compressed KV/indexer state; it does not
measure newly computed prefill. Two generated tokens do not establish quality.

The library and production sources remained unchanged during the run. This
run precedes the final small-batch-tail/singleton-dispatch change; its exact
source/library hashes are retained. Retained prefix size was 1,798,090 bytes;
peak process RSS was approximately 28.94 GB. Both requests finished, and the
owned server exited successfully with its port closed. See the
[receipt](optimization-evidence/trained-http-prefix-smoke.json).

## Validation and candidate selection

The integrated default-library suite passed **772 tests and 37 subtests**.
Twenty-two real-GPU checks could not run without CUDA. One optional official
module was skipped by that environment; its **23 tests passed separately** in
the pinned CPU Torch environment. Compiler/PTX checks and the compiled Vx CPU
SIMT tests ran in the integrated suite. The singleton/tail focused suite also
passed **98 tests each at O0 and O3**. These are scoped test results, not
full-model release eligibility. [Integrated test log](optimization-evidence/final-suite.txt).

[CPU compiler search](../validation/CPU_SEARCH.md) builds isolated O0/O1/O2/O3
libraries, requires identical mandatory test coverage with zero skips, freezes
sources/reference/toolchain inputs, and checks exact finite outputs before
benchmarking. It never changes the default library or enables a serving mode.
Every selected candidate retains `release_eligible:false` until the independent
trained-model requirements are met. The trained numerical failure cannot be
bypassed by winning a benchmark. The final search passed **282 required checks, zero skips, for each of O0–O3**,
with the same test identities and finite exact benchmark outputs. Sources,
compiler/LLVM/linker inputs and the default library stayed unchanged during the
search. O2 ranked first at **1.399x O0** across three resident matrix cases and
two synthetic prefill workloads; O1 scored 1.228x and O3 1.201x. Seven samples
per case were used. The score is a workload-specific geometric mean, subject
to shared-host noise; it is not trained serving throughput or an optimality proof.

The selected O2 library SHA256 is
`d7338a6d70462322818f5e0db38a860ea4f4897f64fdb010281ff91ffd8d7110`.
It was not installed or enabled. [Full search receipt](optimization-evidence/cpu-compiler-search.json)
retains tests, samples, paths and source identities. A subsequent comparator-only
change admits the official int32 short-context index representation; its 53 tests
passed separately. No kernel, model, serving source or compiler gate changed.

## Run the explicit CPU candidate

```bash
source ../vx-toolchain/env.sh
VX_OPT_LEVEL=3 VX_BUILD_DIR=build-o3 kernels/build.sh
GLM_VX_LIBRARY="$PWD/kernels/build-o3/libglm_vx.so" OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m glm_vx.server \
  --gguf /path/to/glm-5.3-iq1s --backend vx --packed-weights \
  --batched-prefill --prefill-chunk 8 --decode-prefill-tokens 4 \
  --prefix-cache-mib 256 --max-pending-events 64
```

`--packed-weights` requires GGUF and the CPU Vx backend. The default path remains
available for comparison. Prefix snapshots require immutable weights while the
scheduler runs; changing weights in place requires advancing the documented
model cache revision. The cache budget covers retained snapshots, not total
process RSS or active request caches.

## Outstanding work

Full trained-model, long-context and held-out independent parity still blocks
release promotion. Original FP8 activation semantics and quantization quality
are separate from the dequantized-F32 GGUF contract. MTP, packed batched GEMM,
GPU-resident full-model execution, paged/tiered GPU KV, expert/tensor collectives,
network worker pools and measured P/D topology tuning remain to be implemented.
The current CPU process handoff remains a protocol proof of concept. None of
these missing capabilities is implied by the passing component tests above.
