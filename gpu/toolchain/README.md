# Compile Vx GPU kernels to PTX on a CPU host

`build_ptx.py` compiles real Vx kernel bodies with vxc, translates the resulting
LLVM-dialect MLIR, maps the small [device ABI](ABI.md) to LLVM NVVM intrinsics,
optimizes, and emits NVIDIA PTX. It does not run the kernel, invoke Vx `spawn`,
call CUDA libraries, or fall back to a CPU implementation. All numerical
algorithms remain in the `.vx` source; Python adds only device ABI/compiler
plumbing. A CUDA Driver API runtime must load the PTX and launch its entries.

## Build

Requires Python 3.10+, the Vx compiler (tested v0.0.2), and an LLVM 22 build
containing NVPTX, `mlir-translate`, `opt`, and `llc`. These LLVM executables must
report version 22; mismatches fail. Put tools and their library dependencies on
PATH/LD_LIBRARY_PATH. `VXC` may select the Vx executable, and
`VX_GPU_LLVM_BIN` may select the directory of LLVM executables.

From the repository root, with the existing sibling installation:

```bash
source ../vx-toolchain/env.sh
python3 gpu/toolchain/build_ptx.py --source gpu/kernels.vx --output gpu/build/kernels.ptx --keep-ir
python3 gpu/toolchain/build_ptx.py --source gpu/fp8.vx --output gpu/build/fp8.ptx --keep-ir
python3 -m unittest discover -s gpu/toolchain -p 'test_*.py' -v
```

The script itself contains no machine-specific paths or dependency downloads.
An upstream portable Vx installation plus LLVM 22 works with ordinary PATH
configuration; sourcing this local environment is just one setup option.
`VX_DISABLE_CUDA=1` in that environment does not affect this offline path: the
release compiler emits portable pointer code, and LLVM's NVPTX backend emits
the device instructions. The runtime still requires an actual NVIDIA GPU.

Default target is `sm_80` / PTX 7.6. `--arch sm_90` selects PTX 7.8;
`--arch sm_90a` selects PTX 8.0. The supported target list is explicit, so an
unknown architecture cannot silently fall back to an LLVM default. Older
listed SM70+ targets use PTX 7.6 and require a driver supporting that PTX ISA.
PTX architecture describes the generated target; it does not detect hardware.
`--shared-floats` configures a static block-local allocation (default 256 f32).
Do not request dynamic launch shared memory for that array.

Each successful build writes PTX and a sibling JSON record with the source
hash, PTX hash, actual target/ISA, entry list, hardware feature evidence, and
tool versions. `--keep-ir` retains MLIR, original LLVM IR, adapted IR and
optimized IR. Compiler failures, unsupported externs, incompatible scalar ABI,
missing entries or thread indexing, and remaining device calls fail the build.
The output is published only after checks pass; existing output is left intact
on failure, and a nonzero exit must never be treated as a usable new build.

## Evidence and limits

`experiments/thread.vx` is the minimal real-source probe. Its PTX contains
`%tid.x`, `%ctaid.x`, `%ntid.x`, `ex2.approx.ftz.f32`, and `sqrt.rn.f32`.
`experiments/kernels.ptx` demonstrates eleven entries for the numerical kernels;
`experiments/fp8.ptx` demonstrates the FP8 projection entry. Corresponding JSON
files record the exact source/PTX hashes used. `kernels-sm90.ptx` and
`thread-sm90a.ptx` also demonstrate Hopper target compilation. The twelve compiler
regression tests include an actual Vx-to-PTX build and rejection tests for host
externs, ABI mismatches, descriptors, host target triples, and device calls.

The kernels have hardware thread indexing, synchronized warp shuffle,
block barriers and shared memory as applicable, with no unresolved runtime
functions. LLVM verifies the optimized IR. NVIDIA `ptxas` 12.9.86 was subsequently installed locally and successfully
assembled Ampere and Hopper cubins (see the optional assembly workflow below).
CUDA JIT loading, GPU execution, GPU numerical tolerances, race detection and
performance remain **not validated** on this CPU-only host.
There is no measured GPU speedup claim. Math uses approximate, FTZ exponential;
see ABI.md. CPU emulation cannot validate GPU scheduling or approximation error.

## Why an explicit adapter

Read-only inspection of the available upstream snapshot found:

- `src/driver.rs:1363` explains that `--emit-llvm` prints LLVM-dialect MLIR;
  it is not directly LLVM `.ll` output in the installed release.
- `src/dialect/VxLowering.cpp:1400` describes native GPU materialization and
  limitations around host helpers and unlowered linear algebra.
- `runtime/cuda_dispatch.cpp:813` explains why a serial outlined region launches
  one thread unless a proved/grid-strided launch payload is supplied.
- `runtime/cuda_dispatch.cpp:943` includes a host route for unrouted kernels.
- `scripts/tools/emit_gpu_kernel.sh` is an upstream MLIR-to-PTX spike; its input
  is handwritten MLIR rather than Vx kernel source.

Those are not a claim that current upstream cannot parallelize anything. This
adapter deliberately starts from explicitly per-thread Vx source and validates
the produced PTX. It does not rely on topology placement or a successful host
fallback as evidence of GPU work. The inspected upstream and installed v0.0.2
are separate snapshots; neither was modified.

## Optional NVIDIA assembly validation

An assembler can validate PTX and create native GPU cubins on a CPU host. For
this verification only, NVIDIA's pinned `nvidia-cuda-nvcc-cu12==12.9.86` wheel
was installed under ignored `gpu/toolchain/cuda-tools/`; no global packages were
changed. Reproduce with an available Python/pip installation:

```bash
python -m pip install --target gpu/toolchain/cuda-tools nvidia-cuda-nvcc-cu12==12.9.86
python3 gpu/toolchain/build_ptx.py --source gpu/kernels.vx --output gpu/build/kernels.ptx --keep-ir --ptxas gpu/toolchain/cuda-tools/nvidia/cuda_nvcc/bin/ptxas
python3 gpu/toolchain/build_ptx.py --source gpu/kernels.vx --output gpu/build/kernels-sm90.ptx --arch sm_90 --ptxas gpu/toolchain/cuda-tools/nvidia/cuda_nvcc/bin/ptxas
```

`--ptxas` is explicit: an unavailable or failed assembler fails the build.
Successful assembly publishes a sibling `.cubin` and `.ptxas.log`, records the
assembler version/cubin SHA256 and sets `assembly_validated:true` in JSON.
`gpu_execution_validated` remains false. Without this option the build only
checks LLVM/PTX and records `assembly_validated:false`. Assembler logs contain
register, static shared-memory and spill usage; these do not measure runtime
occupancy, correctness or speedup. Binary cubin artifacts and the downloaded
CUDA tools are local verification outputs, not committed dependencies.

Final eleven-entry source SHA256:
`85463e38228f811ec2357d4eafb629eabf9ac74bfcdf081801d448979fd79e05`.
All eleven main entries and the FP8 projection assembled with ptxas 12.9.86 for
both sm_80 and sm_90. Each reports zero stack frame and zero spill stores/loads.
The eleven main kernels use 22–40 registers on sm_80 and 24–32 on sm_90. Reduction
kernels use 1024 bytes static shared memory per CTA; kernels whose shared
storage is unused have it eliminated by optimization. Router's index shuffle
is present as `shfl.sync.idx` in the inspected PTX. Logs/manifests under
`experiments/` retain exact resource reports and hashes. Twelve regressions pass,
including compilation of all eleven current kernels and failed-assembler output
preservation. Re-run after source changes; these hashes describe this snapshot.

Published source-matched assembly manifests and resource reports are copied to
`gpu/evidence/`; `experiments/` is local scratch and is not committed.
