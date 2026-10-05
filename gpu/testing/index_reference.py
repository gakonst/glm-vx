"""Independent DSA score references, test-only; never imported in production."""
import numpy as np


def inputs(heads=3, tokens=5, dim=33):
    rng = np.random.default_rng(817)
    q = rng.normal(0, .2, (heads, dim)).astype('f4')
    keys = rng.normal(0, .2, (tokens, dim)).astype('f4')
    weights = rng.normal(size=heads).astype('f4')
    weights[0] = -.75  # Already scaled signed head weight, not a probability.
    return q, keys, weights


def ordered_f32(q, keys, weights, scale):
    """Scalar dimension-order, then head-order f32; no BLAS or GPU reduction."""
    result = np.zeros(len(keys), dtype='f4')
    for t, key in enumerate(keys):
        for h, query in enumerate(q):
            dot = np.float32(0)
            for a, b in zip(query, key):
                dot = np.float32(dot + np.float32(a * b))
            activated = np.maximum(np.float32(dot * np.float32(scale)), np.float32(0))
            result[t] = np.float32(result[t] + np.float32(weights[h] * activated))
    return result


def warp_order_f32(q, keys, weights, scale):
    """Explicit rounded warp tree for exact CPU SIMT arithmetic verification."""
    result = np.zeros(len(keys), dtype='f4')
    for t, key in enumerate(keys):
        for h, query in enumerate(q):
            lanes = np.zeros(32, dtype='f4')
            for d in range(len(query)):
                lanes[d % 32] = np.float32(lanes[d % 32] + np.float32(query[d] * key[d]))
            for delta in (16, 8, 4, 2, 1):
                previous = lanes.copy()
                for lane in range(32):
                    lanes[lane] = np.float32(previous[lane] + previous[lane + delta if lane + delta < 32 else lane])
            activated = np.maximum(np.float32(lanes[0] * np.float32(scale)), np.float32(0))
            result[t] = np.float32(result[t] + np.float32(weights[h] * activated))
    return result


def reference_f64(q, keys, weights, scale):
    return np.maximum(q.astype('f8') @ keys.astype('f8').T * float(np.float32(scale)), 0).T @ weights.astype('f8')
