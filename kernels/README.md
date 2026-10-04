# GLM5.3 Vx numerical kernels

`kernels.vx` contains independent float32 Vx implementations. Only scalar `expf`
and `sqrtf` call libm. There is no NumPy/C/BLAS implementation behind these symbols.
`kernels.h` defines the raw-pointer C ABI. Caller owns all storage and must validate
contiguous buffers, sizes, finite inputs and int32-safe dimension products.

Implemented: RMSNorm, LayerNorm (indexer normalization), stable softmax, SwiGLU,
add/AXPY, row-major matvec, matrix multiplication with transposed right argument,
interleaved RoPE, corrected sigmoid top-k routing, arbitrary top-k, single-head
dense attention and attention over caller-selected tokens.

RoPE rotates the last `rope_dim` elements of each head. The caller computes
position-specific cos/sin coefficients (one per adjacent pair). For this model,
`rope_dim=64`, `theta=8000000`, RMS epsilon is `1e-5`, routing is 8 of 256 with
scale 2.5 and normalized *uncorrected* probabilities. Correction bias only selects
experts. `n_group=topk_group=1`, so all experts are eligible. Ties deterministically
choose the smaller index; this need not match PyTorch's unspecified tie order.
Routing adds `1e-20` to the selected probability sum, matching the inspected
`metadata/modeling_glm_moe_dsa.py` router. Attention callers supply causal eligible
tokens or selected IDs; these kernels do not manage caches or DSA indexer state.

Build with `VXC=/absolute/path/to/vxc kernels/build.sh`, then execute
`.venv/bin/python kernels/test_kernels.py`. The harness loads the real compiled
library and compares against independent NumPy float64 formulas, including full
model norm/RoPE sizes, underflow/large logits, correction-bias routing, in-place
operations, and dense/sparse attention. Missing compiler/library is a failure.

These are scalar CPU correctness kernels, not high-throughput GPU kernels.
Storage is float32 and matvec accumulation is float32. FP8/BF16 checkpoint decode
and FP8 block scaling belong to the checkpoint loader. No speed or full-model
quality claim follows from numerical microtests. Inputs containing NaN/+infinity,
all-masked softmax, overflowing norm sums and malformed pointers are unsupported.

## Verified build in this workspace

Compiled with the official Vx v0.0.2 Linux release, via
`/srv/nanocodex/workspace/vx-toolchain/bin/vxc`, using `--action emit-obj`.
`cc -shared ... -lm` produced `build/libglm_vx.so`; `build.log` records unmangled
exported symbols. All **50** checks passed in `test-results.json`; maximum absolute
error was `0.0001972442098576721` (long float32 matvec), within the test tolerance.

`glm_vx_rope_offset` additionally chooses the rotary region by element offset.
Use offset 0 for the indexer's first 64 elements; main attention can use the
last-region `glm_vx_rope` convenience function. `glm_vx_index_scores` implements
sum over heads of `weights[h] * relu(q[h] dot k[t] * qk_scale)`; supply weights
already multiplied by `heads**-0.5` and `qk_scale=dim**-0.5`.


## Python model interface

`from kernels.backend import VxBackend` supplies `matvec(w,x)`,
`rmsnorm(x,w,eps)`, `rope(x,position,theta)`, `softmax(x)`, `swiglu(gate,up)` and
`route(logits,bias,k,scale)`. It loads `build/libglm_vx.so`, or an explicit library
path / `GLM_VX_LIBRARY`. Missing libraries fail immediately; there is no fallback.
Arrays become contiguous float32 buffers, shapes/int32 indexing capacity are
checked, and each operation invokes Vx through ctypes. NumPy computes RoPE
cos/sin coefficients and manages memory only in this wrapper.

The model-facing `rope` uses `glm_vx_rope_glm`: GLM consumes interleaved input
pairs but returns packed `[all rotated-even, all rotated-odd]` output, as in
`metadata/modeling_glm_moe_dsa.py:156`. The lower-level `glm_vx_rope` and
`glm_vx_rope_offset` retain adjacent output. Applying the same output permutation
to q and k leaves their dot products unchanged, so both conventions can yield
identical logits when used consistently; the wrapper now matches the official
primitive output exactly. `glm_vx_rope_glm` does not permit in-place use.

`.venv/bin/python -m unittest kernels.test_backend -v` passed **7** integration
tests, including the actual Vx-backed four-layer tiny GLM prefill against the
independent expanded causal attention oracle in `tests/test_model.py`, shape
validation, noncontiguous inputs and failure when the library is unavailable.
See `backend-test-results.log`. The tiny random model verifies mechanics, not
trained checkpoint quality.


Optional native wrapper methods: `layernorm(x,w,bias,eps=1e-6)`,
`topk(scores,k)` returning int32 IDs, and
`index_scores(q,keys,weights,scale=None)` with the pre-scaled head-weight contract
above. They validate shapes and execute the already-tested Vx kernels.

The root's `scripts/benchmark.py --requests 2 --prompt 8 --tokens 8` also ran
successfully with `OPENBLAS_NUM_THREADS=1`; greedy outputs matched for both
backends. `model-benchmark.json` records this one tiny random-model CPU run.
It is an integration smoke benchmark, not full-model performance evidence.


RoPE coefficient generation now uses a process-wide LRU cache keyed by
`(dimension, position, theta)`, capped at128 entries. Only dimensions <=4096 are
cached, so coefficient array payload is bounded by2MiB; larger dimensions bypass
it. Cached arrays are read-only. A dedicated test verifies repeated-call reuse,
eviction after140 positions, unchanged numerical output, and large-dimension
bypass. Final validation also ran all11 tests in `tests.test_model`; see
`model-integration-results.log`.

Packed GGUF CPU matvec, exact codec coverage, the checkpoint integration hook,
opt-in optimized build, and bounded trained-row evidence are documented in
[PACKED.md](PACKED.md). The default build remains O0.
