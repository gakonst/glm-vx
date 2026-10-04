# Ordered row SIMD in Vx v0.0.2 (2026-10-04)

Implemented `glm_vx_matvec` with eight independent output accumulators, followed
by two-row and one-row tails. Every row retains left-to-right f32 multiply then
add. The existing header already excludes output/input overlap. No tensor
repacking, allocation, activation quantization, FMA, or fast-math was introduced.
This is expanded-f32 matvec; it complements, but does not replace, packed-GGUF
multirow prefill work. No packed/backend/checkpoint files were edited here.

Actual compiler: `/srv/nanocodex/workspace/vx-toolchain/bin/vxc`, `vxc 0.0.2`.
Host: AMD EPYC 4484PX, Linux x86-64. The installed compiler emitted AVX-512
`vgatherdps`, `vmulps` with an eight-lane broadcast, and `vaddps` in the production
matvec (`kernels-o3.asm`). It emitted scalar `vmulss`/`vaddss` for the original
ordered baseline (`probe.asm`). There are no fused multiply-add instructions in
the production matvec. This is measured target-specific code, not a portable
object or a claim that every Vx target will vectorize the same way.

## Correctness and integration

- `correctness.txt`: 252 exact float32 cases per optimization level (O0 and O3),
  covering zero dimensions, all row-tail sizes, small/odd/long columns,
  cancellation, signed zeros, sentinel output boundaries and invalid dimensions.
  Oracle explicitly rounds each scalar product and each ordered addition to f32.
- `kernels-o0.txt`, `kernels-o3.txt`: existing 50 kernel checks pass at each level.
- `integration.txt`: 18 backend/model tests pass with the O3 production library,
  including the independent expanded attention oracle.
- `bench.py` additionally compares the production matvec bitwise with an
  unchanged scalar Vx baseline across 108 random shapes per optimization level.

## Bounded synthetic benchmark

One process pinned to CPU 0; seed 20261004; no BLAS calls for measured work.
Nine samples per candidate with randomized candidate order; each sample repeats
calls 3–200 times depending on matrix size. Table uses median wall time per call.
Inputs are warm/reused synthetic f32 row-major matrices. Other processes may
share the host; raw samples are retained in `results*.json`. Tiny calls can
regress because call overhead/noise dominates. Tile 2, 4 and 8 alternatives are
retained, and no universal optimum is claimed. Full trained model throughput,
packed decode throughput, cold mapped GGUF IO and other CPUs were not measured.

| Build | Rows × columns | Baseline ms | Production ms | Speedup |
|---|---:|---:|---:|---:|
| O0 | 32 × 256 | 0.020970 | 0.004852 | 4.322× |
| O0 | 1024 × 512 | 1.368625 | 0.212040 | 6.455× |
| O0 | 256 × 6144 | 4.112723 | 0.829411 | 4.959× |
| O3 | 8 × 64 | 0.000754 | 0.000798 | 0.944× |
| O3 | 32 × 256 | 0.005977 | 0.003193 | 1.872× |
| O3 | 256 × 256 | 0.036850 | 0.020751 | 1.776× |
| O3 | 1024 × 512 | 0.316757 | 0.184842 | 1.714× |
| O3 | 4096 × 512 | 1.239884 | 0.763264 | 1.624× |
| O3 | 256 × 6144 | 1.027222 | 0.600856 | 1.710× |
| O3 | 2048 × 6144 | 8.515435 | 4.100089 | 2.077× |
| O3 | 4096 × 6144 | 17.317388 | 8.985226 | 1.927× |
| O3 | 256 × 12288 | 2.025123 | 1.157279 | 1.750× |

## Capability audit and primary sources

- [Vx website](https://vxlang.org/) fetched 2026-10-04 (snapshot
  `vxlang-home.html`). Its placement/type-system examples are not evidence that
  an inference library or kernel is implemented in the installed release.
- [Release v0.0.2](https://github.com/vx-lang/Vx/releases/tag/v0.0.2), tag
  `24493283316e4f23b43e494faee02f45b8a953ae`. The release compiler was used unchanged.
- Upstream HEAD resolved on 2026-10-04 to
  `1c18982669a1c2c63a4212d82481f5a1a381ba04`; local source checkout is the older
  `48da4a1aea4cda37599648865531e5b62b82ae29`. Neither was built or modified.
- [Current tensor library plan](https://github.com/vx-lang/Vx/blob/1c18982669a1c2c63a4212d82481f5a1a381ba04/docs/implementation_plans/tensor_library.md)
  is explicitly a proposal. It describes current fixed-width handwritten vector
  loops and the requirement for explicit permission to reorder floating-point
  reductions. Generic arbitrary-rank zip/reduce APIs and tensor-value fusion /
  bufferization are planned work, not capabilities established for v0.0.2 here.
- [Current grammar](https://github.com/vx-lang/Vx/blob/1c18982669a1c2c63a4212d82481f5a1a381ba04/docs/lang/grammar.md)
  documents `<N x T>`. `types.vx` proves the installed release accepts explicit
  `<4 x f32>` addition and actually emits `vaddps` (`types.asm`), although this
  SIMD-by-value function was not invoked through ctypes (its vector ABI differs).
- Pointer loops over f16/bf16 compile, link, and execute. `test_types.py` checks
  all 65,536 encodings: every 63,488 finite f16 and 65,280 finite bf16 conversion
  matches bit-for-bit; NaN/Inf classes also match. Assembly has `vcvtph2ps` for f16
  and integer widening/shifts for bf16. This is concrete typed conversion support;
  it does not establish packed GGUF block decode from typed tensors. Byte-packed
  codec layouts still require their own exact unpacking and alignment handling.

Run `evidence/vx-row-tile/run.sh` from this worktree to reproduce local builds,
checks, assembly and measurements with the shared Python environment. It writes
only into this worktree. The release's `--action emit-llvm` returned LLVM-dialect
MLIR on stdout (`ir-build.txt`), not an LLVM IR file at `-o`; production object
disassembly is the evidence for final vectorization. No toolchain modifications,
GPU use, rentals or full trained model runs were performed.
