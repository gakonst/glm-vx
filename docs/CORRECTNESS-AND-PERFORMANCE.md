# Correctness gates before automatic optimization

Audit of engine revision `1f434489dbfb20fdef726d002f5f00b9911b2b91`.
This is a proposed acceptance contract plus a measured compiler experiment,
not a claim that the full parity gate is implemented or passed.

## What parity means

Keep two distinct targets:

1. **Execution parity on identical GGUF weights.** Pin all six checkpoint hashes,
   tokenizer and template, config, reference revision, numerical mode, and input
   token IDs. Validate the independent reference before using the current engine
   as an optimization baseline. The current two-token agreement is insufficient.
2. **Quality relative to original GLM-5.3 weights.** Quantization changes weights;
   exact original-FP8 parity is not a valid requirement for IQ1_S. Separately
   measure perplexity/task quality against the original model on fixed datasets.

Finite tests cannot prove correctness for every input. A strict, versioned
contract can make promotion automatic within a documented domain. Formal proofs
of individual kernels or memory contracts strengthen that domain but do not
prove the entire model, compiler, driver, tokenizer and server correct.

## Independent oracle and trace corpus

Use pinned llama.cpp on exactly the same GGUF as the independent integration
oracle, and independent mathematical/codec oracles for component tests. The
reference is test tooling, never the serving implementation. Its public
`llama_context_params.cb_eval` and named GLM-DSA graph tensors provide a route
for a test-only trace exporter. Synchronize backend copies before exporting.
Validate axis order and logical tensor meaning; do not compare identically named
buffers blindly, especially compressed versus expanded MLA. Existing NumPy
backend tests share the production model graph and cannot replace this oracle.

Capture at selected positions in **every layer**:

- embeddings, normalized input, Q/K/V or equivalent canonical MLA quantities;
- indexer scores, selected DSA positions and shared-selection reuse;
- router scores, selected experts and normalized routing weights;
- expert/shared MLP outputs, attention output, residual and layer output;
- logical cache contents/positions and the complete vocabulary logits.

Use teacher forcing: feed both engines the same prescribed token at each step,
regardless of their predictions. This locates the first divergent operation
without letting different generated contexts invalidate later comparisons.
Keep end-to-end free generation as a separate integration check. Trace storage
can contain selected layer/position tensors plus complete logits; hashes alone
cannot measure numerical errors. Keep raw model activations local by default
and publish summaries unless artifact distribution is explicitly intended.

At minimum include real-model prompts spanning 1, 2, 31, 32, 33, 127, 128, 129,
2047, 2048 and 2049 tokens; long decode; repeated/prefix-related inputs; Unicode,
empty user content within valid templates, and special tokens. Add substantial
fixed and held-out corpora before calling this a release gate. Small synthetic
models must exhaustively stress sparse/shared DSA with tiny top-k, dense/MoE
transitions, routing ties, cache eviction and RoPE offsets. Our six-token smoke
never reaches the real model's 2048-selection pruning boundary. CPU full-model
long-context tests will be expensive: build the corpus once on adequate hardware,
then use targeted layer fixtures for fast inner-loop checks.

Compare whole-prefill, incremental token decode and every supported chunk split.
Test serving interleavings, cancellation, disconnect, request isolation and seed
reproducibility independently of numerical tests. Concurrency must not change a
request's logical cache or output.

## Specific existing coverage and gaps

The expanded batch oracle in `tests/test_model.py` independently computes
per-head KV attention, while production uses compressed MLA. Short sparse/shared
DSA, MoE correction-bias behavior, high-position primitive RoPE, cache splitting
and request isolation already have tests. Preserve these rather than replacing
them with snapshots from the implementation.

Independent packed-byte expected values currently cover Q8_0 and selected IQ1_S
patterns. The real model uses **11 quantization formats**. Real row tests compare
Vx versus NumPy *after the same decoder*, so they cannot detect a shared decoder
bug. Add pinned GGML C decoder fixtures for all 11 formats, including block, row
and expert boundaries. Integrated GGUF model fixtures currently use F32 tensors;
add quantized, asymmetrically tagged K-B/V-B and expert fixtures so transposes and
slicing are checked together with actual decoding.

Add a second small-model oracle executing the pinned official eager GLM graph.
The existing mathematical oracle follows its equations but does not execute that
implementation. Test real-weight layer replays with frozen input activations and
cache contents before expensive full-model runs. This makes the optimization
inner loop practical without pretending tiny tests prove trained-model parity.

For RoPE, retain primitive coordinate checks and add independent end-to-end
indexer-prefix/query-suffix checks at positions 0, 1, 2048, 131071 and 1048575,
including unequal query/key positions. For DSA, compare selected sets when the
reference's selection order is unspecified, but never accept a different selected
set merely by sorting it. Selection-boundary ties need their own contract.

## Gate semantics

- Require exact shapes, positions, cache ownership, tensor mappings and token IDs.
- Freeze tie-breaking rules. Require exact discrete routing/DSA selections for
  strict regression. For cross-engine ties, inspect score margins and establish
  a documented rule before accepting a fixture; never silently waive mismatch.
- Reject NaNs and unexpected infinities; compare legitimate masked infinities
  explicitly. Require full-logit and intermediate comparisons, not only argmax.
- Use per-operation absolute/relative and normalized error budgets, plus logit
  distribution error and top-k margins. Calibrate and freeze budgets using the
  independent baseline and precision modes **before** tuning. A global `1e-3`
  tolerance is not a defensible substitute. Near-zero values need absolute bounds.
- Separate strict F32, fused/reassociated arithmetic, and reduced-precision
  contracts. New precision modes require a separately reviewed accuracy budget.
- Pin corpus/version/hash, weights, compiler, library/PTX, flags and hardware in
  every receipt. Missing references, skips, stale hashes, unsupported cases or
  incomplete runs mean **not eligible**, never a passing score.
- Mutation-test the gate: deliberately swap K/V axes, corrupt one quant block,
  shift a cache position, change a router weight and perturb a logit. Confirm that
  the expected test fails and identifies the first divergent stage.

## Automatic optimization loop

Keep an immutable champion and freeze the gate separately from candidate edits.
Generate a candidate in isolation; run codec/kernel properties and differential
fixtures first, then full model/cached decode/serving gates, then held-out checks.
Only passing candidates may be benchmarked for promotion. An optimizer must not
edit tolerances, expected outputs or evaluation data to make its candidate pass.

Measure warm compute, quant decoding, disk faults, allocation, host/device copies,
launches, prefill and decode separately. Compare matched hardware/thread budgets,
weights, prompts and output work. Alternate candidate/champion order, repeat runs,
retain samples, and require a predeclared improvement beyond measurement noise.
Use both tokens/s and client p50/p95 TTFT/inter-token latency under load, with a
memory limit. A throughput win that violates the latency budget is not a win.
Re-run the slower release suite on the exact binary before promoting; preserve
champion and receipts for rollback. GPU promotion requires actual hardware tests,
including race/memory checking where available. PTX assembly and CPU simulation
are necessary evidence but cannot substitute for GPU execution.

## Vx utilization audit

**We are not using Vx to its performance limits.**

| Area | Present | Gap |
|---|---|---|
| CPU compiler | `kernels/build.sh` invokes Vx without `-O`; installed compiler defaults to `-O0` | Explicit optimized build and parity gating |
| CPU numerical code | Scalar F32 loops; O3 disassembly still has scalar reduction operations | SIMD/blocking, parallel row/expert execution, fused packed-quant dot products |
| GGUF | Python/upstream NumPy codec expands selected weights into F32 | Native Vx quantized kernels, lower allocation and memory traffic |
| Prefill | Sequential token passes, matrix-vector projections | Batched/tiled matrix multiplication and fused prefill |
| GPU code | Coalesced matvec, shuffle reductions, shared memory/barriers, split online MLA, LLVM O3 | Tensor-core MMA, asynchronous copy pipelines and shape-specialized GEMM |
| GPU orchestration | Partial resident MLP chain; host model and compatibility transfers remain | Resident model/cache, fewer launches and transfers, batched scheduling |
| Vx placement | Raw-pointer C ABI and custom NVVM adapter | Native `Pinned`/`Tensor`, `spawn on`, `transfer`, machine admission and verified seams are not exercised |
| Hardware evidence | PTX assembled and CPU SIMT checked | No actual GPU correctness/performance measurement |

Vx's documented placement types, machine files and seam verification are useful
for proving placement/capacity/transfer obligations; they do not automatically
make scalar loops into peak-performance tensor-core kernels. Probe support with
small compile/run examples against the pinned compiler before migrating the
runtime. The current custom adapter specifically expects portable IR without a
host target: native topology integration is a compiler/runtime project, not an
extra flag on the existing script.

Primary language documentation: [heterogeneous programming](https://vxlang.org/docs/heterogeneous.html),
[machine files](https://vxlang.org/docs/machine-files.html). Installed compiler
help confirms default `-O0` and exposes `--verify-seams`, `--emit-seam-certs`,
`--machine` and `--host`; none are used in the current raw-pointer kernel build.

### Isolated O3 experiment

The validated default shared library was preserved. Built the same Vx source
with `-O3` and no fast-math option into a separate library. It passed **167 tests
plus 37 subtests** in the CPU/model suite and **50 native kernel checks**.
This does not repeat the previous 278-test GPU-inclusive suite or full trained
model generation. Four F32 matrix-vector cases were bit-identical between O0
and O3 on the tested inputs:

| Rows × columns | O0 median | O3 median | Kernel speedup |
|---|---:|---:|---:|
| 512 × 6144 | 7.718 ms | 1.983 ms | 3.89× |
| 2048 × 6144 | 30.909 ms | 8.264 ms | 3.74× |
| 6144 × 2048 | 30.668 ms | 7.755 ms | 3.95× |
| 6144 × 6144 | 92.424 ms | 25.545 ms | 3.62× |

Eleven alternating-order samples per variant/shape on one shared CPU. These are
resident synthetic F32 kernel measurements, **not full-model speedups**. Quant
unpacking and disk traffic are outside this benchmark. Do not infer bitwise
identity for untested inputs or all other operators.

[Samples and binary hashes](perf-audit-evidence/o3-benchmark.json),
[CPU tests](perf-audit-evidence/o3-tests.txt),
[kernel checks](perf-audit-evidence/o3-kernels.json).

Reproduce from project root after sourcing the Vx toolchain environment:

```sh
mkdir -p build/perf-audit
vxc kernels/kernels.vx -O3 --action emit-obj -o build/perf-audit/kernels-o3.o
cc -shared -o build/perf-audit/libglm_vx_o3.so build/perf-audit/kernels-o3.o -lm
GLM_VX_LIBRARY="$PWD/build/perf-audit/libglm_vx_o3.so" OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests kernels/test_backend.py -q
.venv/bin/python kernels/test_kernels.py build/perf-audit/libglm_vx_o3.so
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  .venv/bin/python docs/perf-audit-evidence/bench_o3.py
```

Priority order: independent trace oracle and fail-closed gate; promote measured
compiler improvements through that gate; profile the complete GGUF path; implement
packed quantized dot products and fused prefill; then GPU residency/tensor-core
work with real-device validation. Keep the larger serving-throughput objective
separate from isolated kernel wins.
