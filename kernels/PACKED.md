# Packed GGUF CPU matvec

`GGUFCheckpoint.linear(name, x, backend)` is the optional model integration
hook. `x` must be one f32 vector. It resolves stacked expert offsets and calls
`backend.packed_matvec(raw_uint8, ggml_type, (rows, cols), x)` inside a scoped
`GGUFStore.packed_rows` borrow. Closing the store from another thread waits
for the borrow. Returned output owns memory; borrowed bytes must not escape.
Neither model.py nor glm_vx/backend.py is changed here. Batched prefill can
continue using the existing expanded matrix interface.

`VxBackend.supports_packed(kind)` advertises compiled symbols. All eleven
formats in the actual six-shard GLM-5.3-UD-IQ1_S checkpoint are covered: F32,
Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ2_XXS, IQ3_XXS, IQ4_XS.
F32 uses a direct mmap float view; all ten quantized codecs execute fused Vx
unpacking and multiplication. F16 is additionally covered by synthetic tests.
Other gguf-py-supported codecs, an older library without packed symbols, a
backend without packed support, and reconstructed KV-B use the explicit
existing tensor/dequantize + backend.matvec path. Packed calls bypass the
expanded-weight LRU entirely.

Numerics remain decoded-f32-weight times unquantized-f32-activation, with the
same ordered f32 accumulation as existing Vx matvec. This is **not** GGML's
Q8-activation dot mode. Codec diagnostic exports are only for validation;
serving never allocates an expanded matrix. Resident lookup storage is a
19,600-byte codebook plus a 256 KiB exact half-conversion table. Host checks
contiguity, dtype, shape/block alignment, byte counts, finite input/output,
and int32 index bounds before/after the native boundary.

## Build and validate

Tested with actual `vxc 0.0.2`, LLVM 22, on the existing Linux CPU host:

```bash
source ../vx-toolchain/env.sh
bash kernels/build.sh                       # unchanged default O0
VX_OPT_LEVEL=3 VX_BUILD_DIR=build-o3 bash kernels/build.sh
PYTHONPATH=$PWD .venv/bin/python -m pytest tests/test_packed_matvec.py
GLM_VX_LIBRARY=$PWD/kernels/build-o3/libglm_vx.so PYTHONPATH=$PWD \
  .venv/bin/python -m pytest tests/test_packed_matvec.py kernels/test_backend.py
PYTHONPATH=$PWD .venv/bin/python -m benchmarks.packed_matvec --rows 128 --repeats 7
```

This worktree used the shared `../glm-vx/.venv` instead of its own `.venv`.
The optimized library is an **opt-in candidate**. `VX_OPT_LEVEL` validates
0..3 and defaults to 0; no deployment/library environment was changed.

Recorded tests in `packed-evidence/`: O0 159 tests plus 4 subtests passed
(packed + GGUF reader/checkpoint + backend regression); O3 54 tests plus 4
subtests passed (packed + backend). Existing C ABI kernel suite passed 50
checks for each build. No tests were skipped in these recorded runs.

Synthetic payloads exercise raw random code/control bytes with finite scales,
including signed scales, zero and half subnormals. The oracle calls pinned
native GGML `dequantize_row_*` directly and never gguf-py. All 2,048 IQ1_S
codebook indices and all finite F16 bit patterns have additional coverage.
Native decoded bytes feed independent NumPy ordered f32 sums. Trained tests
cover all 11 formats and first/middle/last row and expert offsets, using
independent positional reads, and prohibit candidate Python dequantization.
Bounds tests cover malformed lengths/types/strides, invalid expert/row slices,
closed stores, int32 capacity, nonfinite values, and close/borrow concurrency.

## Bounded benchmark

The receipts fingerprint both candidate libraries and the pinned oracle.
Each sample is at most 128 contiguous middle rows of one middle tensor and
expert per format. Seven timings follow one warmup, with warm pages. The
benchmark also asserts exact native codec and ordered-dot equality before
recording timing. It never runs llama inference or reads all trained weights.

One pass sampled 6,281,216 packed bytes across all formats. The inventory totals 216,705,819,648 bytes.

| Format | Shape | Packed O0 ms | Packed O3 ms | Decode + F32 O3 ms | O3 speedup | Weight payload reduction |
|---|---|---:|---:|---:|---:|---:|
| F32 | 128 × 6144 | 1.989 | 0.508 | 0.851 | 1.67× | 1.00× |
| Q8_0 | 128 × 512 | 0.533 | 0.054 | 0.174 | 3.20× | 3.76× |
| Q2_K | 128 × 6144 | 5.979 | 1.114 | 2.248 | 2.02× | 12.19× |
| Q3_K | 128 × 2048 | 2.313 | 0.729 | 0.989 | 1.36× | 9.31× |
| Q4_K | 128 × 6144 | 6.666 | 1.436 | 2.273 | 1.58× | 7.11× |
| Q5_K | 128 × 16384 | 20.106 | 4.922 | 11.114 | 2.26× | 5.82× |
| Q6_K | 128 × 2048 | 1.789 | 0.742 | 0.816 | 1.10× | 4.88× |
| IQ2_XXS | 128 × 6144 | 8.082 | 1.197 | 9.156 | 7.65× | 15.52× |
| IQ3_XXS | 128 × 2048 | 2.978 | 0.639 | 3.279 | 5.13× | 10.45× |
| IQ1_S | 128 × 6144 | 7.719 | 1.100 | 6.070 | 5.52× | 20.48× |
| IQ4_XS | 128 × 2048 | 2.856 | 0.454 | 1.763 | 3.88× | 7.53× |

The largest storage formats are IQ3_XXS (87.55 GB), IQ1_S (66.69 GB), and
IQ2_XXS (36.54 GB); all three have fused paths. These are scalar CPU kernels.
O0 decoding overhead makes several formats slower than the old vectorized
Python decoder plus Vx F32 dot; reduced payload is not itself a speed claim.
Even O3 packed can be slower than a **previously cached expanded F32** dot;
the full receipts include that comparison. No single-expert/full-model token
throughput, cold-page streaming bandwidth, production load, batch fusion,
SIMD-specific assembly, or GPU packed performance is established here.
End-to-end parity remains a separate model-level gate.

## Source and license

Codebook constants and codec layouts derive from llama.cpp commit
`11fe02151f79c41d0d4af7da708755d73b9c0da6`, MIT licensed. `GGML-LICENSE`
retains the upstream copyright and license. `ggml_tables.json` pins the source
and binary hashes and offsets; `extract_ggml_tables.py` regenerates the data
only from the exact source hash. A test reproduces every byte and the notice.
The production path does not link to or execute GGML; only validation uses
the independent native reference library.
