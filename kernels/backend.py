"""Model-facing ctypes interface to the compiled Vx kernels (no fallback).

NumPy handles contiguous storage and RoPE coefficient generation only. All six
model numerical operations execute the exported Vx functions in libglm_vx.so.
"""
import ctypes as C
from functools import lru_cache
import os
from pathlib import Path
import numpy as np

_F = C.POINTER(C.c_float)
_I = C.POINTER(C.c_int32)
_S = C.c_int32
_R = C.c_float
_MAX = np.iinfo(np.int32).max


def _array(value, name):
    result = np.asarray(value, dtype=np.float32)
    if result.ndim == 0:
        raise ValueError(f'{name} must have at least one dimension')
    if result.size > _MAX or any(d > _MAX for d in result.shape):
        raise ValueError(f'{name} exceeds Vx int32 indexing capacity')
    return np.ascontiguousarray(result)


def _pointer(value):
    return value.ctypes.data_as(_I if value.dtype == np.int32 else _F)


def _make_rope_coefficients(d, position, theta):
    # Match model reference precision. Cache entries are immutable host buffers.
    angles = position * np.power(theta, -np.arange(0, d, 2, dtype=np.float32) / d)
    cosine = np.ascontiguousarray(np.cos(angles), dtype=np.float32)
    sine = np.ascontiguousarray(np.sin(angles), dtype=np.float32)
    cosine.flags.writeable = False
    sine.flags.writeable = False
    return cosine, sine


@lru_cache(maxsize=128)
def _cached_rope_coefficients(d, position, theta):
    # Callers cache only d<=4096: <=2 MiB of coefficient payload, globally.
    return _make_rope_coefficients(d, position, theta)


class VxBackend:
    """Scalar CPU float32 Vx backend with explicit buffer/shape validation."""
    name = 'vx-cpu-f32'

    def __init__(self, library=None):
        self.library_path = Path(library or os.environ.get('GLM_VX_LIBRARY') or
                                 Path(__file__).parent / 'build' / 'libglm_vx.so').resolve()
        try:
            self.lib = C.CDLL(str(self.library_path))
        except OSError as exc:
            raise RuntimeError(f'Cannot load compiled Vx kernels at {self.library_path}; '
                               'run kernels/build.sh with VXC set to the Vx compiler. '
                               'No numerical fallback is enabled.') from exc
        signatures = {
            'layernorm': [_F, _F, _F, _F, _S, _R],
            'topk': [_I, _F, _S, _S],
            'index_scores': [_F, _F, _F, _F, _S, _S, _S, _R],
            'matvec': [_F, _F, _F, _S, _S],
            'matmul_nt': [_F, _F, _F, _S, _S, _S],
            'rmsnorm': [_F, _F, _F, _S, _R],
            'rope_glm': [_F, _F, _F, _F, _S, _S],
            'softmax': [_F, _F, _S],
            'swiglu': [_F, _F, _F, _S],
            'router': [_I, _F, _F, _F, _S, _S, _R],
        }
        for name, args in signatures.items():
            function = getattr(self.lib, 'glm_vx_' + name)
            function.argtypes, function.restype = args, _S

    def _call(self, name, *args):
        values = [_pointer(a) if isinstance(a, np.ndarray) else a for a in args]
        result = getattr(self.lib, 'glm_vx_' + name)(*values)
        if result:
            raise ValueError(f'Vx {name} rejected arguments (status {result})')

    def matvec(self, w, x):
        w, x = _array(w, 'weight'), _array(x, 'input')
        if w.ndim != 2 or x.ndim != 1 or w.shape[1] != x.size:
            raise ValueError('matvec expects weight[rows,cols] and input[cols]')
        out = np.empty(w.shape[0], dtype=np.float32)
        self._call('matvec', out, w, x, *w.shape)
        return out

    def linear_batch(self, w, x):
        """Native Vx X[batch,input] @ W[output,input]^T, scalar F32 reductions."""
        w, x = _array(w, 'weight'), _array(x, 'input')
        if w.ndim != 2 or x.ndim != 2 or w.shape[1] != x.shape[1]:
            raise ValueError('linear_batch expects weight[output,input] and input[batch,input]')
        if w.shape[0] * x.shape[0] > _MAX:
            raise ValueError('linear_batch output exceeds Vx int32 indexing capacity')
        out = np.empty((x.shape[0], w.shape[0]), dtype=np.float32)
        self._call('matmul_nt', out, x, w, x.shape[0], w.shape[0], w.shape[1])
        return out

    def rmsnorm(self, x, w, eps):
        x, w = _array(x, 'input'), _array(w, 'weight')
        if w.ndim != 1 or w.size != x.shape[-1] or not w.size:
            raise ValueError('rmsnorm weight must match the nonempty final input axis')
        eps = float(eps)
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError('rmsnorm epsilon must be finite and positive')
        out = np.empty_like(x)
        for source, target in zip(x.reshape(-1, w.size), out.reshape(-1, w.size)):
            self._call('rmsnorm', target, source, w, w.size, eps)
        return out

    def rope(self, x, position, theta):
        x = _array(x, 'input')
        d = x.shape[-1]
        if not d or d % 2:
            raise ValueError('RoPE final dimension must be positive and even')
        position, theta = float(position), float(theta)
        if not np.isfinite(position) or not np.isfinite(theta) or theta <= 0:
            raise ValueError('RoPE requires finite position and positive finite theta')
        # Repeated heads/layers reuse one position's coefficients. Large unusual
        # dimensions bypass the cache to preserve its <=2 MiB payload bound.
        coefficients = _cached_rope_coefficients if d <= 4096 else _make_rope_coefficients
        cosine, sine = coefficients(d, position, theta)
        out = np.empty_like(x)
        self._call('rope_glm', out, x, cosine, sine, x.size // d, d)
        return out

    def softmax(self, x):
        x = _array(x, 'input')
        n = x.shape[-1]
        if not n:
            raise ValueError('softmax final axis must be nonempty')
        rows = x.reshape(-1, n)
        if np.isnan(rows).any() or np.isposinf(rows).any() or not np.isfinite(rows).any(axis=-1).all():
            raise ValueError('softmax needs one finite score per row and no NaN/+inf')
        out = np.empty_like(x)
        for source, target in zip(rows, out.reshape(-1, n)):
            self._call('softmax', target, source, n)
        return out

    def swiglu(self, gate, up):
        gate, up = _array(gate, 'gate'), _array(up, 'up')
        if gate.shape != up.shape:
            raise ValueError('SwiGLU gate and up shapes must match')
        out = np.empty_like(gate)
        self._call('swiglu', out, gate, up, gate.size)
        return out

    def route(self, logits, bias, k, scale):
        logits, bias = _array(logits, 'logits'), _array(bias, 'bias')
        if logits.ndim != 1 or bias.shape != logits.shape:
            raise ValueError('route expects equal one-dimensional logits and bias')
        if not isinstance(k, (int, np.integer)) or not 0 < k <= logits.size:
            raise ValueError('route k must be an integer in [1, experts]')
        if not np.isfinite(logits).all() or not np.isfinite(bias).all() or not np.isfinite(scale):
            raise ValueError('route inputs and scale must be finite')
        ids, weights = np.empty(k, np.int32), np.empty(k, np.float32)
        self._call('router', ids, weights, logits, bias, logits.size, k, float(scale))
        return ids, weights


    def layernorm(self, x, w, bias, eps=1e-6):
        """Affine LayerNorm along the final axis, including indexer k_norm."""
        x, w, bias = _array(x, 'input'), _array(w, 'weight'), _array(bias, 'bias')
        if w.ndim != 1 or bias.shape != w.shape or w.size != x.shape[-1] or not w.size:
            raise ValueError('layernorm weight/bias must match the nonempty final input axis')
        eps = float(eps)
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError('layernorm epsilon must be finite and positive')
        out = np.empty_like(x)
        for source, target in zip(x.reshape(-1, w.size), out.reshape(-1, w.size)):
            self._call('layernorm', target, source, w, bias, w.size, eps)
        return out

    def topk(self, scores, k):
        """Descending finite scores, stable ties by smaller index; int32 IDs."""
        scores = _array(scores, 'scores')
        if scores.ndim != 1 or not np.isfinite(scores).all():
            raise ValueError('topk expects finite one-dimensional scores')
        if not isinstance(k, (int, np.integer)) or not 0 <= k <= scores.size:
            raise ValueError('topk k must be an integer in [0, scores.size]')
        ids = np.empty(k, np.int32)
        self._call('topk', ids, scores, scores.size, k)
        return ids

    def index_scores(self, q, keys, weights, scale=None):
        """DSA score per token: sum_h weights[h]*relu(dot(q[h],keys[t])*scale).

        q[heads,dim], keys[tokens,dim], weights[heads]. Caller pre-scales head
        weights by heads**-0.5. qk scale defaults to dim**-0.5. Keys must already
        contain only causal eligible tokens. No top-k selection is done here.
        """
        q, keys, weights = _array(q, 'query'), _array(keys, 'keys'), _array(weights, 'weights')
        if q.ndim != 2 or keys.ndim != 2 or weights.ndim != 1:
            raise ValueError('index_scores expects query[heads,dim], keys[tokens,dim], weights[heads]')
        heads, dim = q.shape
        if not heads or not dim or keys.shape[1] != dim or weights.size != heads:
            raise ValueError('index_scores dimensions do not match')
        scale = dim**-0.5 if scale is None else float(scale)
        if not np.isfinite(scale):
            raise ValueError('index_scores scale must be finite')
        out = np.empty(keys.shape[0], np.float32)
        self._call('index_scores', out, q, keys, weights, keys.shape[0], heads, dim, scale)
        return out
