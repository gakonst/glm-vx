# Explicit NVIDIA GPU path

GPU kernels are written in Vx and compile to **real PTX** with hardware thread
indices, warp shuffle reductions, shared memory and block barriers. A small LLVM
adapter supplies hardware intrinsics; there is no CUDA C numerical replacement,
PyTorch/CuPy runtime, or implicit CPU fallback. Target: NVIDIA SM80+ by default.

**Not yet GPU-benchmarked or proven optimal.** This workspace has no CUDA driver
or GPU. Offline compilation, assembly (where recorded), and CPU SIMT arithmetic
validation are distinct from real-device execution. See GPU-REPORT.md for receipts.

## Build and run

Install Vx v0.0.2 and LLVM 22. Ensure `vxc`, `mlir-translate`, `opt`, `llc` are on
PATH, or set `VXC` and `VX_GPU_LLVM_BIN`. Then:

```sh
gpu/build.sh
# Optional: PTXAS=/path/to/ptxas gpu/build.sh validates native assembly too.
# On the original build machine, first: source ../vx-toolchain/env.sh
# Both files emit PTX plus source/hash/toolchain manifests in gpu/build/.
```

On a machine with an SM80+ NVIDIA GPU and compatible CUDA driver:

```sh
.venv/bin/python -m gpu.benchmark --kernel all --repeats 20 --output gpu-result.json
# Use the actual 6144 projection / 512-latent dimensions:
.venv/bin/python -m gpu.benchmark --model-shapes --output gpu-model-shapes.json
.venv/bin/python -m glm_vx.server --tiny --backend vx-gpu
```

The benchmark first checks numerical outputs, then uses CUDA events over resident
inputs. Timings exclude upload/download, startup and serving; they are not tokens
per second. Missing CUDA exits with status 2 and an explicit unavailable result.
Any numerical/JIT/launch failure is an error, not a CPU timing result.

`--backend vx-gpu` is an explicitly **hybrid compatibility adapter**: the existing
model still orchestrates on Python/NumPy host arrays. Every primitive uses PTX,
but intermediate activations transfer between calls. Read-only weights are held
in a bounded device LRU (default 512 MiB; `--gpu-weight-cache-mib` changes it).
Mutable arrays are re-uploaded; temporary buffers are released after synchronized
completion. The compatibility model uses fused compressed MLA across heads, but still gathers
and uploads the selected KV rows on each attention call. Direct FP8 projection
is currently a resident API, not integrated into checkpoint serving.

## Measured layout tuning

```sh
.venv/bin/python -m gpu.tune --rows 6144 --cols 6144 --output tuning.json
.venv/bin/python -m glm_vx.server --tiny --backend vx-gpu --gpu-tuning tuning.json
# Quantized projection candidate measurements (resident API only):
.venv/bin/python -m gpu.tune --fp8 --rows 6144 --cols 6144 --output fp8-tuning.json
```

The tuner checks outputs and compares 32/64/128/256-thread layouts using actual
CUDA events. A plan records the GPU, driver, PTX hash and exact matrix shape.
The compatibility backend rejects mismatched plans and uses a measured layout
only for its matching F32 shape. This selects the best *tested* layout; it does
not prove global optimality. No plan is emitted when CUDA is unavailable.

## Resident kernels

- Coalesced warp-per-row float32 matvec; tails need no padded weight allocation.
- Direct **FP8 E4M3FN + 128×128 block scales** matvec; no expanded float32 weight
  matrix or per-element scale tensor in GPU memory.
- Fused residual-add + RMSNorm with warp/block reductions; residual output is
  retained for the next residual connection.
- Grid-strided SwiGLU and GLM interleaved-input, half-split-output RoPE.
- Split compressed MLA decode: each CTA streams selected latent KV entries with
  online softmax; a second kernel stably merges partitions. This avoids expanded
  per-head K/V and a full attention score matrix. Scratch is O(heads×splits×rank).
- Standalone RMSNorm, stable softmax and corrected sigmoid routing for the
  compatibility backend.

Use `CUDAContext.tensor()` and `GLMKernels` from `gpu.runtime` to keep inputs and
outputs resident across launches. Launch partial MLA and merge on the same stream
(or provide an explicit event dependency). `gpu/kernels.h` defines the exact ABI,
shape bounds, alias rules and launch geometry. Scalar storage is float32 except
quantized FP8 weights, int32 selected indices and routing indices.

## Validation layers

```sh
# Mock CUDA API lifecycle/argument tests; no hardware claim:
.venv/bin/python -m pytest gpu/test_runtime.py -q
# Compile identical Vx source into a test-only CPU lane simulator:
source ../vx-toolchain/env.sh   # or supply your LLVM/Vx environment
gpu/testing/build_simt.sh
.venv/bin/python -m pytest gpu/test_simt.py gpu/test_backend.py -q
# Compiler IR/PTX contracts and exhaustive FP8 tests:
.venv/bin/python -m pytest gpu/toolchain/test_build_ptx.py gpu/test_fp8.py -q -rs
```

CPU simulation executes Vx kernels with host threads, shuffle emulation and CTA
barriers; it tests lane ownership, reductions, tails, mask behavior and merge
math. It does not model GPU scheduling, compiler-induced races, register pressure
or the GPU approximate/FTZ exponential. Real GPU cases explicitly skip when no
CUDA device exists and fail on missing PTX when a device does exist. The production
runtime cannot import or select the simulator as a fallback.

## Remaining performance work

The code removes several obvious data-movement/serialization costs inside the
kernels, but no optimality claim follows. High-throughput full GLM serving still
needs tensor-core FP8/BF16 batched GEMM, fully resident model orchestration,
quantized checkpoint placement/sharding, tensor/expert collectives, paged KV,
prefill fusion, and GPU tuning. Single-node GPU choice, interconnect, latency/SLO
and workload determine the next optimization. No GPU was rented or purchased.
