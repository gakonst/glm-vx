# Vx GPU compiler adapter ABI

Settled with kernel agents: ordinary C declarations in Vx:

```rust
extern "C" {
  fn vx_gpu_thread_idx_x() -> i32;
  fn vx_gpu_block_idx_x() -> i32;
  fn vx_gpu_block_dim_x() -> i32;
  fn vx_gpu_grid_dim_x() -> i32;
  fn vx_gpu_shuffle_down_f32(value: f32, delta: i32) -> f32;
  fn vx_gpu_shuffle_idx_f32(value: f32, src: i32) -> f32;
  fn vx_gpu_barrier() -> void;
  fn vx_gpu_shared_f32() -> *mut f32;
  fn expf(value: f32) -> f32;
  fn sqrtf(value: f32) -> f32;
}
```

Entry functions are named `glm_vx_gpu_*`, return `void`, and take raw pointers,
i32/i64/f32/f64 scalar arguments (no tensor descriptors or host values). Explicit
`return;` at the end of a Vx void function avoids a v0.0.2 implicit-return bug.
The compiler adapter adds the `ptx_kernel` calling convention to these functions.
Other defined functions are internal device helpers and are eligible for inlining.

Shuffle functions map to full-mask (`0xffffffff`) `shfl.sync.down`
and `shfl.sync.idx` operations with clamp 31 (32-lane width); every lane named by the mask
must execute the same shuffle. Return the caller's own value when a down shuffle
is outside the warp. Barrier maps to `bar.sync 0` and requires uniform block
participation. Shared pointer refers to one statically allocated block-local
array, default 256 f32 (1024 bytes), configurable with --shared-floats. Storage
is uninitialized and has normal CUDA shared-memory synchronization requirements.

expf uses `ex2.approx.ftz.f32(x * log2(e))`: approximate GPU exponential, with
subnormal results flushed to zero. sqrtf uses LLVM sqrt and NVPTX sqrt.rn.f32.
Neither math operation calls the host or requires libdevice. Numerical GPU
validation remains required; CPU emulation uses platform expf/sqrtf and cannot
validate the approximation's GPU error.

The optional TF32 MMA intrinsic and its exact warp, storage and precision
contract are documented in [GEMM.md](../GEMM.md). Compile `gemm.vx` with
`--shared-floats 320` and sm80 or newer. Its adapter uses LLVM inline PTX for
TF32 conversion and MMA, while staging/indexing/masking remain real Vx.
