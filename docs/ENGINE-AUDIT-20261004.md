# GLM-Vx engine audit, October 4–5, 2026

This update implements findings from the official GLM source, current vLLM,
SGLang, llama.cpp, KTransformers, provider reports and Vx's actual compiler.
It extends the [previous work](OPTIMIZATION-PROGRESS.md) and
[serving literature review](SERVING-RESEARCH.md). The inference engine remains
from scratch; external engines are references, not serving dependencies.

## Fixed correctness and validation bugs

1. **Omitted DSA schedule fields changed attention.** Validated loading used
   GLM-5.3-specific defaults 4/3 instead of the official omitted-field defaults
   1/2. A six-token fixture selected different keys and changed logits. Corrected
   ingestion now matches explicit official defaults exactly. Explicit metadata
   in the full GLM-5.3 checkpoint is unchanged.
2. **Cache identity omitted execution mode.** Prefill/decode handoff could accept
   a cache created with a different `packed_weights` mode. Prompt snapshots also
   survived changes to resolved epsilon, RoPE, indexer topology and related fields.
   Identities now bind these fields; mutation invalidates snapshots and mismatched
   handoff is rejected before ownership changes.
3. **GPU CPU-test artifacts could be stale.** A pre-existing `libsimt.so` was
   accepted solely because it existed. After adding an entry, this surfaced as
   missing-symbol errors; changes to existing entries could be tested against old
   code unnoticed. The build now binds all numerical sources, the C++ lane harness,
   build script and resulting binary by SHA256. Tests reject stale/missing manifests.

Reproductions and pinned upstream sources are in the
[numerical audit](CORRECTNESS-AUDIT-20261004B.md) and
[serving audit](SERVING-AUDIT-20261004B.md). No numerical tolerance was relaxed.

## Implemented Vx performance work

**Packed batched projections.** Supported quantized expert and attention linears
remain mmap-backed throughout multi-token prefill. A Vx kernel decodes each
weight once per token tile, reusing it across up to 64 independent accumulators.
The host packs only activation tiles; it never creates an expanded weight matrix.
Each dot retains its original left-to-right F32 multiply/add order. F32 weights
remain borrowed views, and older libraries explicitly retain packed matvec fallback.

We tested 8-, 16-, 32- and 64-token variants, then changed the activation layout
from a single transposed matrix to fixed-stride transposed tiles. This lets LLVM
vectorize small and irregular batches as well as the 64-token case. The generated
Vx source and generator are both checked in. ABI v2 makes the input-layout change
explicit; the host still understands the earlier experimental column-major ABI.
Actual object assembly contains vector multiply/add instructions, including
AVX-512, with no fused multiply-add in the checked IQ1_S entry.

**Expanded matvec.** Eight independent output rows form SIMD lanes, preserving
each row's reduction order. On bounded synthetic F32 matrices, the O3 version
was 1.62–2.08× faster for the reported large shapes than the prior scalar O3
kernel; the tiny 8×64 case regressed to 0.944×. All measurements and alternatives
are retained in the [row-SIMD evidence](../evidence/vx-row-tile/README.md).

**GPU index scores.** The [new Vx kernel](../gpu/INDEX-SCORES.md) computes all
weighted DSA head scores in one launch. The compatibility adapter changes H+1
launches / 2H+2 uploads / H+1 downloads into 1 / 3 / 1. A resident entry reuses
caller-owned GPU tensors/output and enqueues without host transfer or sync.
CPU SIMT checks the actual compiled arithmetic; ptxas assembles SM80/SM90 with
zero spills. No physical GPU execution or speedup has been established.

## Trained-weight CPU microbenchmarks

AMD EPYC 4484PX, CPU 0 affinity, Vx 0.0.2/LLVM 22.1.8, O3. Nine randomized,
interleaved timing samples per method; warmup excluded. Each case uses 128 rows
from one trained tensor/expert per GGUF format. All methods first match an
independent native GGML decode plus ordered F32 dot oracle bit-for-bit.
Activation packing, allocation and Python dispatch are inside timing.

The table is an equal-format geometric mean over the ten quantized formats.
F32 is measured separately in the receipt. Both baselines are **our engine**:
repeated packed Vx matvec, or GGUF decoding followed by expanded Vx batched GEMM.
These are not llama.cpp/vLLM comparisons and are not full-model token throughput.

| Token batch | Versus repeated packed matvec | Versus decode + expanded batch |
|---:|---:|---:|
| 2 | 1.77× | 2.45× |
| 4 | 3.66× | 3.03× |
| 8 | 7.01× | 3.47× |
| 16 | 13.12× | 4.66× |
| 17 | 7.92× | 2.73× |
| 32 | 20.13× | 5.40× |
| 64 | 24.16× | 5.42× |

[All 88 cases, raw samples, byte/input hashes and library fingerprints](audit-evidence-20261004b/packed-batch-final.json).
[Initial tile search](audit-evidence-20261004b/packed-tile-search.json) and
[64-token follow-up](audit-evidence-20261004b/packed-tile64-search.json) retain the
slower candidates. Timings on a shared host are not a universal optimum. Larger
serving chunks remain indivisible and can worsen decode latency; this kernel
result does not automatically increase the scheduler's chunk budget.

## Full trained regression and remaining parity boundary

The actual six-shard 216.7 GB mixed-bit GLM-5.3 checkpoint ran two teacher-forced
tokens `[9703, 10056]` through all 78 layers with packed batched prefill.
**1,898 checks were bit-for-bit equal to the previously saved scalar Vx trace**:
hidden inputs/norms, attention/MLP outputs, routed IDs/weights, selected indices,
compressed latents/RoPE/index keys and both complete 154,880-element logit vectors.
The observer only inspects helper results and never substitutes values.

[Receipt](audit-evidence-20261004b/trained-batched-regression.json),
[logits](audit-evidence-20261004b/trained-batched-logits.npz),
[reproducer](../validation/check_batched_regression.py).
The run took 158.25 seconds with 25.57 GiB peak RSS. It ran alongside validation
work and includes synchronous observation; this timing is diagnostic, not a
controlled end-to-end speed comparison. Reference archive, code, library,
configuration and actually opened shards are bound; shard digests are reused
from the declared verified-download manifest rather than rehashing 216.7 GB.

**Independent official parity remains incomplete.** The earlier official
comparison still has 112/308 numerical failures despite 306/306 discrete matches.
Preserving our scalar baseline does not repair that discrepancy. Original FP8
quality, long-context trained pruning, wider held-out inputs and real GPU execution
remain unproven. Release eligibility stays false; frozen budgets remain unchanged.

## Validation on the integrated tree

- [Default O0 full suite](audit-evidence-20261004b/final-tests.txt): **1,121 passed,
  37 subtests passed, 26 skipped**. Skips are 25 unavailable CUDA cases and the
  optional official-model module in the lightweight environment.
- [O3 numerical suite](audit-evidence-20261004b/numerics-o3.txt): **328 passed**.
- [Independent row-order checks](audit-evidence-20261004b/matvec-order.txt):
  **252 bit-exact cases at each O0/O3**, plus output canaries and invalid dimensions.
- [Existing native kernel checks](audit-evidence-20261004b/kernels-o3.json) pass O3.
- [Optional pinned official environment](audit-evidence-20261004b/official-tiny.txt):
  **23 passed** on tiny models. These validate the independent official trace
  machinery and do not resolve the trained numerical failures.

## Further findings from other servers

- **Physical cache layout must be one contract.** vLLM fixes show that missing FP8
  cache-mode metadata and inconsistent DCP block-table widths can break GLM.
  Allocation, admission, kernel metadata and handoff should share a descriptor
  before this engine adds paged/quantized/distributed KV. See the pinned commits
  in the [serving audit](SERVING-AUDIT-20261004B.md).
- **Persistent decode is a meaningful Vx direction.**
  [AMD/TileRT's September report](https://www.amd.com/en/developer/resources/technical-articles/2026/tilert-agentx-on-amd-instinct-gpus.html)
  describes a whole-forward persistent kernel, streaming producer/consumer
  synchronization, prefetch and peer writes followed by ordered local reduction.
  Its reported full GLM-5.3 result uses eight MI355X GPUs, original FP8 weights,
  BF16 KV and MTP K=3. That is a different hardware/precision regime from our CPU
  GGUF work. It supports investigating residency and launch gaps, not transplanting
  a performance number. The [vLLM X thread](https://x.com/vllm_project/status/2103297683188527487)
  points to the public InferenceX configuration.
- **P/D transfer must preserve more than KV.** The
  [vLLM/TileRT connector design](https://vllm.ai/blog/2026-07-14-vllm-tilert-pd)
  includes indexer and draft state, stages data before cache recycling, and overlaps
  transfer with the next prefill iteration. Its July benchmark is GLM-5.1, not 5.3.
  Our current CPU handoff validates owning copies; network transfer, MTP state and
  separate production worker pools remain future work.
- **Measure the complete stack after kernel search.**
  [ROCm Hyperloom](https://rocm.blogs.amd.com/software-tools-optimization/hyperloom-optimization/README.html)
  remeasures accepted kernels end to end and sweeps operating points. Its framework
  search can also change precision. Here such a change requires a separate quality
  contract; it cannot be promoted under the current bitwise/F32 regression checks.
- **Separate shipped Vx from proposals.** The
  [Vx capability audit](../evidence/vx-row-tile/README.md) executes typed f16/bf16
  conversion over all 65,536 encodings and inspects vector code generation. The
  upstream generic tensor fusion/zip/reduce document is a proposal. A fully
  resident typed model runtime, asynchronous tensor-core pipelines, graph capture,
  KV placement and collectives are not established by these CPU/SIMT results.

## Reproduce

```sh
source ../vx-toolchain/env.sh
VX_OPT_LEVEL=3 VX_BUILD_DIR=build-audit-o3 kernels/build.sh
gpu/testing/build_simt.sh
GLM_VX_LIBRARY=kernels/build-audit-o3/libglm_vx.so .venv/bin/python -m pytest -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0 \
  .venv/bin/python -m benchmarks.packed_batch \
  --library kernels/build-audit-o3/libglm_vx.so --rows 128 --repeats 9 \
  --checkpoint /path/to/glm-5.3-iq1s --reference /path/to/pinned/llama.cpp \
  --output build/my-packed-batch-benchmark.json
```

The benchmark requires the pinned native GGML oracle built as described in
[codec validation](../validation/ggml_oracle.py). Fresh trained regression output
must use a new directory; the observer rejects overwrite and mismatched references.
The default library build remains O0 and packed/batched serving remains explicit
via `--packed-weights --batched-prefill`. Compiler candidate selection is not a
release-parity approval.
