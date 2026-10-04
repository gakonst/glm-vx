"""Resident GEMM launch contract. No conversion, readback, or CPU fallback.

Load gpu/build/gemm.ptx and construct GEMMKernels(module). ``gemm(a,b)`` uses
F32 arithmetic. ``precision="tf32_rna"`` explicitly rounds multiplicands to
TF32 (nearest, ties away from zero) and accumulates in F32 on tensor cores.
No GPU correctness/performance certification is implied by this wrapper.
"""
from dataclasses import dataclass

from .runtime import (CUDAUnavailable, DeviceTensor, Module, _integer, _i32,
                      _separate, _stream_handle)


@dataclass(frozen=True)
class GEMMPlan:
    entry: str
    grid: int
    block: int
    precision: str
    tile: tuple
    shared_bytes: int


def gemm_plan(m, n, k, *, precision="f32", max_grid=(1 << 31) - 1):
    """Plan arbitrary residual shapes; K=0 writes zeros, M/N=0 launch nothing.

    Dimensions fit signed i32; all tensor address calculations use signed i64.
    No padding or 16-byte alignment is required: global loads are scalar F32.
    ``max_grid`` is the device's x grid limit, checked before output allocation.
    """
    for value, name in ((m, "M"), (n, "N"), (k, "K")):
        _i32(value, name)
    for size in (m * k, k * n, m * n):
        _integer(size * 4, "GEMM tensor bytes")
    if precision == "f32":
        grid, block, tile, shared = (m * n + 127) // 128, 128, (1, 1, 1), 0
    elif precision == "tf32_rna":
        grid, block, tile, shared = ((m + 15) // 16) * ((n + 7) // 8), 32, (16, 8, 8), 1280
    else:
        raise ValueError("precision must be 'f32' or explicit 'tf32_rna'")
    _integer(grid, "GEMM grid", maximum=max_grid)
    return GEMMPlan("glm_vx_gpu_gemm_" + ("tf32" if precision == "tf32_rna" else "f32"),
                    grid, block, precision, tile, shared)


class GEMMKernels:
    """C = A @ B, contiguous A[M,K], B[K,N], C[M,N], all float32.

    sm80 or newer. A/B may alias; C must not overlap either input. Host checks
    storage/shape/layout only. Caller must ensure finite, normal-or-zero values,
    finite TF32-rounded inputs and finite intermediates. No alpha/beta, transpose,
    batching, or leading-stride variants. Work is enqueued on the chosen stream.
    """
    def __init__(self, module):
        if not isinstance(module, Module):
            raise TypeError("module must be a CUDA Module")
        module._ensure_open()
        self.module, self.context = module, module.context
        if self.context.compute_capability < (8, 0):
            raise CUDAUnavailable("GEMM module requires compute capability >= 8.0")
        self.kernels = {mode: module.kernel("glm_vx_gpu_gemm_" + mode) for mode in ("f32", "tf32")}
        if self.kernels["tf32"].static_shared_bytes < 1280:
            raise ValueError("TF32 GEMM module must provide at least 1280 bytes of static shared memory")

    def _tensor(self, tensor, name):
        if not isinstance(tensor, DeviceTensor):
            raise TypeError(name + " must be a DeviceTensor")
        tensor.allocation._ensure_open()
        if tensor.context is not self.context:
            raise ValueError(name + " belongs to a different CUDA context")
        if tensor.dtype != "float32" or len(tensor.shape) != 2:
            raise ValueError(name + " must be a float32 matrix")
        if tensor.pointer % 4:
            raise ValueError(name + " must be 4-byte aligned")
        return tensor

    def gemm(self, a, b, *, precision="f32", out=None, stream=None):
        self.module._ensure_open()
        a, b = self._tensor(a, "A"), self._tensor(b, "B")
        m, k = a.shape
        if b.shape[0] != k:
            raise ValueError("GEMM inner dimensions differ")
        n = b.shape[1]
        plan = gemm_plan(m, n, k, precision=precision, max_grid=self.context.max_grid[0])
        _stream_handle(self.context, stream)
        if out is not None:
            self._tensor(out, "C")
            if out.shape != (m, n):
                raise ValueError("GEMM output must have shape (M,N)")
            _separate(out, [a, b])
        else:
            out = self.context.tensor((m, n))
        if plan.grid:
            mode = "tf32" if precision == "tf32_rna" else "f32"
            self.kernels[mode].launch(plan.grid, plan.block, [out, a, b, _i32(m, "M"),
                                      _i32(n, "N"), _i32(k, "K")], stream=stream)
        return out
