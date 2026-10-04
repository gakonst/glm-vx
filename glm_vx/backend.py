"""Independent NumPy oracle. Production selection never silently falls back here."""
import numpy as np

class NumpyBackend:
    name = "numpy-reference"
    def matvec(self, w, x):
        return np.asarray(w, dtype=np.float32) @ np.asarray(x, dtype=np.float32)
    def rmsnorm(self, x, w, eps):
        x = np.asarray(x, dtype=np.float32)
        return x * (1.0 / np.sqrt(np.mean(x*x) + eps)) * w
    def rope(self, x, position, theta):
        x = np.asarray(x, dtype=np.float32)
        d = x.shape[-1]
        if d % 2: raise ValueError("RoPE dimension must be even")
        angles = position * np.power(float(theta), -np.arange(0,d,2,dtype=np.float32)/d)
        c, s = np.cos(angles), np.sin(angles)
        # Official GLM interleaved input, half-split output layout.
        return np.concatenate((x[...,0::2]*c - x[...,1::2]*s,
                               x[...,1::2]*c + x[...,0::2]*s), axis=-1)
    def softmax(self, x):
        x = np.asarray(x, dtype=np.float32)
        exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)
    def swiglu(self, gate, up):
        gate = np.asarray(gate, dtype=np.float32)
        sigmoid = np.exp(-np.logaddexp(0, -gate))
        return gate * sigmoid * up
    def route(self, logits, bias, k, scale):
        scores = np.exp(-np.logaddexp(0, -np.asarray(logits,dtype=np.float32)))
        ids = np.argsort(-(scores + bias), kind="stable")[:k]
        weights = scores[ids]
        return ids, weights / (weights.sum() + 1e-20) * scale
