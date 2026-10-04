# Block-scaled FP8 projection

`fp8.vx` implements a single-vector projection directly from row-major FP8
weights. Its numerical implementation is Vx, including the scalar decoder.
The compiler adapter supplies hardware intrinsics and emits PTX; there is no
CUDA C numerical replacement or CPU fallback for GPU execution.

## Entry and memory contract

```text
void glm_vx_gpu_fp8_matvec(
    float* out, const uint8_t* weight, const float* scales,
    const float* x, int32_t rows, int32_t cols)
```

For `N=rows`, `K=cols`:

- `weight`: contiguous `[N,K]` bytes in NVIDIA/ONNX-style **E4M3FN**, bias 7.
- `scales`: contiguous float32 `[ceil(N/128), ceil(K/128)]`, row-major.
- `x`: contiguous float32 `[K]`.
- `out`: contiguous float32 `[N]`, disjoint from all inputs.

The operation is
`out[r] = sum_c(decode(weight[r,c]) * scales[r/128,c/128] * x[c])`.
The scale values are **multipliers**. Some checkpoint formats name this tensor
`weight_scale_inv`; do not reciprocate an already supplied dequantization
multiplier. E4M3FN has signed zero, subnormals, a maximum finite magnitude of 448,
and two NaN encodings (`0x7f`, `0xff`); it has no infinity. E4M3FNUZ is different
and is not accepted as E4M3FN. Accumulation and scaling use float32; no bias,
batching, activation, or output quantization is fused.

Use a one-dimensional launch with `block.x` a multiple of 32, normally 128,
and `grid.x = ceil(N / (block.x / 32))`. Both y/z dimensions must be 1. Every
warp handles one row and every lane participates in the five full-mask shuffle
reductions. The last block may contain inactive **whole warps**. A partial
column tile keeps all lanes participating and contributes zero for columns
outside K. No dynamic shared memory is required. Legal block sizes are 32
through 1024, subject to the device/kernel resource limit. Invalid block sizes,
N<=0, or K<0 return without writing. K=0 writes zeros; callers can avoid launching
for N=0. Row/weight/scale offsets are computed in i64 so products of i32
row/column dimensions do not overflow i32.

## Work organization

Each warp traverses K in 128-column tiles. Adjacent lanes read adjacent bytes;
four groups of 32 byte loads cover a complete tile. Each lane loads the tile's
scale once for its four elements. Input-vector reads are also contiguous across
lanes. FP8 values are decoded in registers using integer masks/shifts and
float32 arithmetic. The kernel never materializes an expanded float32 weight
matrix and reads one byte of weight storage per element, versus four bytes for
float32 weights. This describes storage/traffic, **not a measured speedup**.
Scale/vector caching and achieved bandwidth are hardware-dependent, and decode
arithmetic can limit performance. This M=1 kernel uses neither tensor cores nor
shared-memory staging. Warp summation changes rounding relative to sequential
float32 summation.

## Build and validation

With Vx 0.0.2 and LLVM 22 tools configured as described in the toolchain docs:

```bash
python3 gpu/toolchain/build_ptx.py \
  --source gpu/fp8.vx --output gpu/build/fp8.ptx --arch sm_80 --keep-ir
.venv/bin/python -m pytest -q -rs gpu/test_fp8.py
```

`VXC` selects the Vx executable. The tests also discover the adjacent
`vx-toolchain/bin/vxc` installation. `VX_GPU_LLVM_BIN` can select the LLVM tool
directory for the builder. `VX_FP8_PTX` selects a nondefault test PTX path; its
same-stem JSON build manifest must match both the production source and PTX
SHA-256. The default PTX target requires SM80 or newer and a compatible CUDA
driver. No NVIDIA device is needed for offline compilation.

The test suite compiles the unchanged self-contained prefix of `fp8.vx` to a
native CPU object and links it for ctypes. It calls the **same scalar Vx decoder**
used by the GPU entry, comparing all 256 encodings against an independent
mathematical Python oracle, including signed zero and NaN. The Vx sequential
projection is checked against float32 NumPy arithmetic for empty dimensions,
warp tails, 127/128/129 boundaries in both axes, and multiple scale tiles.
Distinct block scales catch transposed indexing and incorrect floor division;
output sentinels detect extra writes.

When a real supported CUDA device is present, tests load the actual PTX through
`gpu.runtime`, allocate/copy device storage, launch, synchronize, and download
results. They cover all encodings and seven shape/launch combinations, compare
to the native Vx reference, and check output sentinels. These tests explicitly
skip with the driver's error when CUDA is unavailable. On a supported device,
missing/stale PTX, module errors, launch errors, and numerical mismatches fail.
The offline PTX structural test skips only when PTX has not yet been built.
CPU and offline tests cannot prove real GPU correctness or performance.

Validated in this workspace on 2026-10-04:

- **18 passed, 8 skipped** (`pytest -q -rs gpu/test_fp8.py`). All eight skips:
  `libcuda.so.1` absent, so real GPU execution was unavailable.
- Vx 0.0.2 → LLVM 22.1.8 → SM80 PTX 7.6 compilation succeeded.
- PTX has one `glm_vx_gpu_fp8_matvec` entry, hardware thread/block indices,
  four `ld.global.s8` instructions in the unrolled tile body, and five
  `shfl.sync.down.b32` reductions. No external/device calls, local spills, or
  shared-memory allocation remain. The CPU reference is dead-stripped.
- Source SHA-256: `74fd43b611fc22765483761d22331076b21fdd33a53f348f8d455c262945b1ea`.
- PTX SHA-256: `dbc37818dcf26aeabb44b8b9c7dfaadb85e3ba89f6803e3293ca5ddd03c745e2`.

No GPU throughput, tensor-core acceleration, full-model execution, or batched
projection claim follows from these checks.
