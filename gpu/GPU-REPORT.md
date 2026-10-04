# GPU implementation validation

This revision adds GPU-oriented Vx code, an explicit CUDA Driver API runtime,
a hybrid serving adapter, and a real-device layout tuner. **GPU execution and
performance have not been measured here. Optimality is not established.**

Subsequent serving improvements and current test evidence are recorded in
[the serving report](../benchmarks/README.md). The original GPU validation
counts below describe the initial GPU implementation.

## Delivered

- Eleven main Vx GPU entries: fused residual/RMSNorm, scalar/row RMSNorm and
  softmax, warp-cooperative corrected sigmoid routing, SwiGLU, GLM RoPE,
  coalesced matvec, split compressed MLA partial and stable merge.
- A separate direct FP8 E4M3FN block-scaled matvec entry. It reads packed weight
  bytes and block scales, avoiding full float32 weight expansion on the device.
- Vx → LLVM → NVPTX compilation with an explicit, checked hardware-intrinsic
  adapter. Unknown externs, invalid entry signatures, unresolved calls and
  assembly failures stop the build. No host fallback is treated as GPU execution.
- Context-owned device buffers, streams/events, launch validation, synchronized
  lifetime management and resident kernel wrappers using CUDA Driver API ctypes.
- `--backend vx-gpu` executes primitives and fused compressed MLA through PTX,
  with a bounded read-only weight cache. This is a **hybrid** adapter: host model
  orchestration and activation/KV transfers remain. FP8 direct projection is
  available through the resident API and benchmark, not the checkpoint forward.
- `python -m gpu.tune`: numerical checks followed by measured CUDA-event timing
  of 32/64/128/256-thread matvec layouts. Plans are device/driver/PTX/shape-bound.
  Only matching F32 plans can alter the compatibility backend's default layout.
- `.gitattributes` assigns the Rust lexer to `.vx` because GitHub Linguist does
  not currently provide a Vx grammar. Upstream metadata is marked vendored.

## Compilation and assembly

Vx v0.0.2, LLVM 22.1.8 and NVIDIA ptxas 12.9.86 compiled and assembled both modules
for SM80 (PTX 7.6) and SM90 (PTX 7.8). All four manifests match the final source
hashes and explicitly record `gpu_execution_validated:false`.

- Main source SHA256: `85463e38228f811ec2357d4eafb629eabf9ac74bfcdf081801d448979fd79e05`.
- FP8 source SHA256: `74fd43b611fc22765483761d22331076b21fdd33a53f348f8d455c262945b1ea`.
- All entries report zero stack frame, zero spill stores and zero spill loads.
- Main kernels use 22–40 registers on SM80 and 24–32 on SM90; block reductions
  use 1024 bytes of static shared memory. FP8 matvec uses 30/29 registers on
  SM80/SM90 respectively, with no shared storage.

These are assembler resource reports, not measured occupancy, bandwidth or
latency. Exact tool versions, hashes and resource reports are in [evidence](evidence/).
The optional NVIDIA assembler installation is local/ignored, not redistributed.

## Tests

Final command after rebuilding PTX and the CPU SIMT test library:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests kernels/test_backend.py gpu -q -rs
```

**133 passed, 13 skipped, 37 subtests passed.** All 13 skips are explicitly
real-device tests: eight FP8 hardware cases and five resident/model hardware
cases. The loader could not find `libcuda.so.1`; there is no GPU timing result.

Passing layers include:

- Existing CPU model, checkpoint, HTTP scheduler and native-kernel regressions.
- Actual Vx-to-PTX compiler tests, ABI rejection and failure-publication behavior.
- CPU simulation of the *same* GPU Vx kernels, using host threads, warp shuffle
  emulation, block barriers and shared memory. Numerical checks cover tails,
  multiwarp reductions, sparse masks, empty partitions and stable MLA merging.
- The complete small two-layer dense/MoE, full/shared-DSA model through the hybrid
  backend with an explicitly injected test context. Logits and request caches
  match the NumPy reference beyond the sparse top-k boundary, and the fused MLA
  path is required rather than silently using per-head softmax.
- Mock-driver context, ownership, buffer bounds, stream/event, same-stream MLA
  ordering, launch arguments, failures and no-fallback contracts.
- Exhaustive FP8 byte decoding, block scale boundaries, adapter immutable-weight
  LRU/pinning/cleanup, float32 scalar overflow rejection and tuning-plan validation.

The test-only SIMT context is never selected by production code. It does not
reproduce GPU scheduling, NVIDIA exponential approximation/FTZ behavior or
compiler-induced GPU memory races. CPU simulation therefore does not establish
GPU numerical correctness; [real-device tests](test_hardware.py) and the CUDA
benchmark remain the next required validation.

## Remaining work

No full trained GLM-5.3 checkpoint, GPU run, benchmark victory, tensor-core GEMM,
fully resident serving graph, model/expert sharding, paged GPU KV or fused batched
prefill is claimed. The current GPU kernels target decode and memory traffic;
production throughput still needs tensor-core batched FP8/BF16 projection and a
resident model pipeline. The tuner finds the fastest of its tested layouts for
a concrete shape/device, not a globally optimal serving engine.
