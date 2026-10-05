"""Strict host boundary for fused Vx GGUF decoding and f32 matvec.

The only resident lookup data is 19,600 codec bytes and a 256 KiB exact half
conversion table. Weight bytes remain mmap-backed; no expanded matrix is made.
"""
import ctypes as C
from functools import lru_cache
import hashlib
from pathlib import Path
import sys
import numpy as np

# GGML enum -> (block elements, bytes, symbol suffix).
LAYOUTS = {1: (1, 2, 'f16'), 8: (32, 34, 'q8_0'),
           10: (256, 84, 'q2_k'), 11: (256, 110, 'q3_k'),
           12: (256, 144, 'q4_k'), 13: (256, 176, 'q5_k'),
           14: (256, 210, 'q6_k'), 16: (256, 66, 'iq2_xxs'),
           18: (256, 98, 'iq3_xxs'), 19: (256, 50, 'iq1_s'),
           23: (256, 136, 'iq4_xs')}
_F = C.POINTER(C.c_float)
_B = C.POINTER(C.c_uint8)
_I = C.c_int32
_MAX = np.iinfo(np.int32).max


@lru_cache(maxsize=1)
def lookup_tables():
    if sys.byteorder != 'little':
        raise RuntimeError('packed GGUF requires a little-endian host')
    raw = Path(__file__).with_name('ggml_tables.bin').read_bytes()
    if hashlib.sha256(raw).hexdigest() != '89c698e44735267dc93e98feb70b5a97d22351fb41440029450edba277cc11b9':
        raise RuntimeError('packed GGML lookup table fingerprint mismatch')
    table = np.frombuffer(raw, dtype=np.uint8)
    # Includes IEEE signed zero/subnormals/Inf/NaN without lossy arithmetic.
    half = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
    half.flags.writeable = False
    return half, table


def supports(lib, kind):
    kind = int(kind)
    return kind == 0 or (kind in LAYOUTS and hasattr(lib, 'glm_vx_packed_' + LAYOUTS[kind][2]))


def _buffers(raw, kind, shape):
    if int(kind) != 0 and int(kind) not in LAYOUTS:
        raise ValueError('unsupported packed GGML format')
    if len(shape) != 2 or any(type(n) is not int for n in shape):
        raise ValueError('packed weight shape must be two Python integer dimensions')
    rows, cols = shape
    block, size, suffix = (1, 4, 'f32') if int(kind) == 0 else LAYOUTS[int(kind)]
    if rows < 0 or cols <= 0 or cols % block:
        raise ValueError('packed matrix must have nonnegative rows and positive block-aligned cols')
    # Quantized kernels index bytes; the F32 route delegates to matvec and
    # indexes float elements. A valid >2 GiB F32 matrix must not be rejected
    # solely for its byte size (the full GLM vocabulary projection is 3.8 GB).
    if (rows > _MAX or cols > _MAX or rows*cols > _MAX or
            (int(kind) != 0 and rows*(cols//block)*size > _MAX)):
        raise ValueError('packed matrix exceeds Vx int32 indexing capacity')
    # Reject strided/wrongly typed buffers instead of allocating an unseen copy.
    if not isinstance(raw, np.ndarray) or raw.dtype != np.uint8 or not raw.flags.c_contiguous:
        raise ValueError('packed bytes must be a contiguous uint8 array')
    if raw.nbytes != rows*(cols//block)*size:
        raise ValueError('packed byte count does not match shape/format')
    if int(kind) == 0 and raw.ctypes.data % np.dtype(np.float32).alignment:
        raise ValueError('F32 packed bytes must be aligned for float loads')
    return rows, cols, suffix


def matvec(lib, raw, kind, shape, x):
    rows, cols, suffix = _buffers(raw, kind, shape)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1 or x.size != cols:
        raise ValueError('packed input must match weight columns')
    if not np.isfinite(x).all():
        raise ValueError('packed input must be finite')
    x = np.ascontiguousarray(x)
    out = np.empty(rows, dtype=np.float32)
    half, table = lookup_tables()
    fn = getattr(lib, 'glm_vx_packed_' + suffix)
    fn.argtypes, fn.restype = [_F, _B, _F, _F, _B, _I, _I], _I
    code = fn(out.ctypes.data_as(_F), raw.ctypes.data_as(_B), x.ctypes.data_as(_F),
              half.ctypes.data_as(_F), table.ctypes.data_as(_B), rows, cols)
    if code:
        raise ValueError(f'Vx packed {suffix} rejected arguments ({code})')
    if not np.isfinite(out).all():
        raise ValueError('nonfinite packed GGUF output')
    return out


def unpack_for_validation(lib, raw, kind, shape):
    """Diagnostic export only; never used by the serving matvec path."""
    rows, cols, suffix = _buffers(raw, kind, shape)
    half, table = lookup_tables()
    out = np.empty(shape, dtype=np.float32)
    fn = getattr(lib, 'glm_vx_unpack_' + suffix)
    fn.argtypes, fn.restype = [_F, _B, _F, _B, _I], _I
    code = fn(out.ctypes.data_as(_F), raw.ctypes.data_as(_B), half.ctypes.data_as(_F),
              table.ctypes.data_as(_B), rows*cols//LAYOUTS[int(kind)][0])
    if code:
        raise ValueError(f'Vx unpack {suffix} rejected arguments ({code})')
    return out


def batch_input(raw, kind, shape, x):
    """Validate bounded token batches before allocating/transposing buffers."""
    rows, cols, suffix = _buffers(raw, kind, shape)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != cols:
        raise ValueError('packed batch input must be [batch,weight columns]')
    batch = x.shape[0]
    if batch > 64:
        raise ValueError('packed batch supports at most 64 tokens')
    if batch * cols > _MAX or batch * rows > _MAX:
        raise ValueError('packed batch exceeds Vx int32 indexing capacity')
    if not np.isfinite(x).all():
        raise ValueError('packed input must be finite')
    return rows, cols, suffix, x


def matmul(lib, raw, kind, shape, x):
    """Decode each packed weight once per token tile; keep dot order unchanged.

    Input transpose and output allocation are included in the public operation.
    Older libraries explicitly use packed matvec per row, never expanded weights.
    """
    rows, cols, suffix, x = batch_input(raw, kind, shape, x)
    batch = len(x)
    if batch == 0 or rows == 0:
        return np.empty((batch, rows), dtype=np.float32)
    tiled = getattr(lib, 'glm_vx_packed_batch_v2_' + suffix, None)
    fn = tiled or getattr(lib, 'glm_vx_packed_batch_' + suffix, None)
    if fn is None or batch == 1:
        return np.stack([matvec(lib, raw, kind, shape, row) for row in x])
    if tiled is None:
        xt = np.ascontiguousarray(x.T)  # previous column-major kernel ABI
    else:
        # Fixed-stride token tiles let LLVM vectorize every tail width. Packing
        # touches activations only: O(batch*cols), never O(weight rows*cols).
        xt = np.empty(x.size, dtype=np.float32)
        token = 0
        for tile in (64, 32, 16, 8, 4, 2, 1):
            if batch-token >= tile:
                xt[token*cols:(token+tile)*cols].reshape(cols, tile)[:] = x[token:token+tile].T
                token += tile
    out = np.empty((batch, rows), dtype=np.float32)
    half, table = lookup_tables()
    fn.argtypes, fn.restype = [_F, _B, _F, _F, _B, _I, _I, _I], _I
    code = fn(out.ctypes.data_as(_F), raw.ctypes.data_as(_B), xt.ctypes.data_as(_F),
              half.ctypes.data_as(_F), table.ctypes.data_as(_B), rows, cols, batch)
    if code:
        raise ValueError(f'Vx packed batch {suffix} rejected arguments ({code})')
    if not np.isfinite(out).all():
        raise ValueError('nonfinite packed GGUF output')
    return out
