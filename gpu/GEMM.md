# Tiled tensor-core GEMM

`gemm.vx` contains two resident GPU GEMM entries for `C[M,N] = A[M,K] B[K,N]`.
The default F32 reference uses F32 multiply/add. The opt-in `tf32_rna` path uses
actual `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`, with six
`cvt.rna.tf32.f32` conversions per lane/instruction. It is a working offline-
compiled implementation, **not device-validated or performance-tuned**.

```python
from gpu.runtime import CUDAContext
from gpu.gemm import GEMMKernels

with CUDAContext(minimum_compute_capability=(8, 0)) as ctx:
    gemm = GEMMKernels(ctx.load_ptx("gpu/build/gemm.ptx"))
    # Populate a[M,K], b[K,N] as contiguous float32 DeviceTensors.
    # No CPU conversion or download occurs inside either operation.
    c_f32 = gemm.gemm(a, b)
    c_tf32 = gemm.gemm(a, b, precision="tf32_rna")
    ctx.synchronize()
```

The example's `a` and `b` must be allocated in `ctx`; callers own input data and
stream dependencies. Reuse `out=` for repeated launches. This API is separate
from the compatibility model backend and does not silently change its precision.

## Launch and storage contract

Both entries take `(out: f32*, a: const f32*, b: const f32*, m: i32, n: i32,
k: i32)` in that order. All matrices are contiguous row-major. Output cannot
overlap either input; inputs may alias. Only four-byte address alignment is
required. Scalar global loads handle arbitrary row strides induced by N/K.

TF32 uses exactly `(32,1,1)` threads and
`(ceil(M/16)*ceil(N/8),1,1)` blocks. Each CTA computes one 16x8 output tile,
iterating over K in chunks of eight. Every lane cooperatively stages A[16,8]
and B[8,8] into shared memory, synchronizes, loads its fragments, executes the
MMA and synchronizes before reusing the stage. Residual M/N/K loads become zero;
residual M/N output stores are masked. No padded allocations are needed.
There is one stage, not an asynchronous or double-buffered pipeline.

Shared memory contains 128 A floats, 64 B floats and 128 lane-private result
floats, totaling **320 floats / 1280 bytes**, statically allocated. The four
running accumulator values remain Vx scalars across K iterations; the adapter
stores each instruction's result into the lane's result region. Current LLVM
forwards those stores to the scalar accumulators but retains the shared stores.
This is a known optimization opportunity. F32 uses `(128,1,1)` threads and
`ceil(M*N/128)` blocks, one output element per thread, without shared memory.

M/N/K must fit nonnegative signed i32; byte lengths and pointer indexing use
signed i64. Host planning checks all three tensor lengths and the device grid
limit before allocation/launch. M=0 or N=0 returns an empty output without a
launch; K=0 writes zero to every output. Raw entries reject negative dimensions
and incorrect block.x uniformly; callers bypassing the wrapper must also enforce
one-dimensional launch geometry, pointer ranges, byte-length limits and exact
ABI. Overprovisioned grid blocks return uniformly before any barrier.

## Precision and compiler boundary

`f32` never invokes MMA or converts inputs to TF32. `tf32_rna` rounds each
multiplicand to TF32 using nearest rounding with halfway cases away from zero;
it accumulates into F32. It is not F32-equivalent. The finite-value contract
requires normal-or-zero inputs, finite rounded inputs and finite intermediates;
underflow/subnormal behavior is not certified. GPU MMA accumulation order and
rounding are not emulated exactly by the CPU test model.

Vx v0.0.2's portable C ABI cannot return the four-register LLVM MMA aggregate.
The isolated adapter implements the explicitly declared intrinsic:

```text
vx_gpu_mma_tf32_m16n8k8(out: f32*, offset: i32,
    a0, a1, a2, a3, b0, b1, c0, c1, c2, c3: f32) -> void
```

It rounds A/B, issues exactly one side-effecting convergent MMA and stores its
four results at `out[offset:offset+4]`. All 32 live lanes must execute the same
call; each lane must own a disjoint result region. It performs no implicit
barrier for surrounding shared-memory traffic. It replaces only that intrinsic,
not the tiling, load masks, loop, accumulation state or output mapping in Vx.
The adapter rejects ABI mismatches, less than 320 shared floats and pre-sm80
compilation, and requires MMA/conversions/shared loads/shared stores/barriers
in the tensor-core entry's emitted PTX. Unknown externs remain errors.

Fragment mapping follows NVIDIA's
[PTX m16n8k8 specification](https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688):
for lane `4*g+t`, A registers map to `(g,t)`, `(g+8,t)`, `(g,t+4)`,
`(g+8,t+4)`; B registers map to `(t,g)`, `(t+4,g)`; output registers map
to `(g,2*t)`, `(g,2*t+1)`, `(g+8,2*t)`, `(g+8,2*t+1)`.

## Reproduce validation

```bash
source ../vx-toolchain/env.sh
# Or configure Vx/LLVM22 via PATH and VX_GPU_LLVM_BIN.
PTXAS=/path/to/ptxas gpu/build.sh
python3 gpu/toolchain/build_ptx.py --source gpu/gemm.vx \
  --output gpu/build/gemm-sm90.ptx --arch sm_90 --shared-floats 320 \
  --ptxas /path/to/ptxas
bash gpu/testing/build_simt.sh
python -m pytest gpu -q
```

Checked evidence is in `evidence/gemm-sm{80,90}.json` and the adjacent assembly
reports. These manifests retain source/PTX/cubin hashes and explicitly set
`gpu_execution_validated: false`. LLVM 22.1.8 verifies the IR; NVIDIA ptxas
12.9.86 assembles both targets. On sm80, TF32 uses 56 registers, 1280 shared
bytes, zero stack and zero spills. These are static resource reports, not
throughput or occupancy measurements.

CPU tests compile the same `gemm.vx` bodies and run each lane using the existing
SIMT harness. A test-only MMA implementation reconstructs the documented
fragments, rounds inputs, and evaluates the dot products. Random rectangular
tails, multi-K tiles, K=0, precision modes, halfway values, invalid uniform
launches, extra CTAs and output guards are checked. Separate mock-driver tests
exercise ABI, stream, resource, grid, shape, overlap and alignment contracts.
`probe.vx` is now tracked so compiler tests also work in a clean checkout.

The real-device gate is `python -m pytest gpu/test_gemm_hardware.py -q`.
Without SM80+ CUDA it is explicitly skipped; on hardware, PTX/JIT/launch errors
fail. It checks both modes against independent FP64 equations, reduced-mode
error versus the original F32 inputs, tails, K=0 and offset output guards.
Run it under Compute Sanitizer memcheck/racecheck/synccheck as well before
production use. Hardware validation is still required for correctness, races,
numerical tolerances and performance. Next optimization work needs measured
multiwarp CTA tiling, larger K stages, accumulator ABI improvement, asynchronous
copies/double buffering, and shape-specific dispatch. No measured speedup or
optimality is claimed for this one-warp baseline.
