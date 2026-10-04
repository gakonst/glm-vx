"""Instance-scoped, byte-bounded CPU prompt snapshots.

Only internally generated pickle payloads are read; no external serialization
API is exposed. Each hit unpickles independent owning arrays/lists, including
DSA index keys/selections, absolute position and final prompt logits. This is
prompt reuse, not paged KV or fused prefill. Weights must be immutable while a
scheduler runs; increment model.prefix_cache_revision after any in-place edit.
The budget includes retained Python entry/key/blob storage, not live request
caches or temporary serialization/copy workspace.
"""
from dataclasses import dataclass
import json
import io
import pickle
import sys
import threading

import numpy as np
from .model import RequestCache


def _size(value):
    """Conservative accounting: repeated immutable references are charged again."""
    return sys.getsizeof(value) + (sum(_size(v) for v in value) if isinstance(value, tuple) else 0)


@dataclass(frozen=True)
class Hit:
    cache: RequestCache
    logits: np.ndarray
    tokens: int


class _SnapshotTooLarge(Exception):
    pass


class _BoundedWriter(io.BytesIO):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def write(self, data):
        if self.tell() + memoryview(data).nbytes > self.limit:
            raise _SnapshotTooLarge()
        return super().write(data)


class PrefixCache:
    def __init__(self, model, budget_bytes=0):
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError('prefix cache budget must be a nonnegative integer')
        self.model = model
        self.budget = budget_bytes
        self.entries = ()  # immutable LRU sequence, oldest first
        self.retained_bytes = 0
        self.lock = threading.Lock()
        self.hits = self.misses = self.evictions = self.rejected = 0
        self.namespace = self._namespace()

    def _namespace(self):
        # Same checkpoint name is insufficient. Bind the actual model, backend,
        # weights owner, complete config and explicitly managed weight revision.
        config = json.dumps(self.model.config, sort_keys=True, separators=(',', ':'))
        return (id(self.model), id(getattr(self.model, 'weights', None)),
                id(getattr(self.model, 'backend', None)), config,
                str(getattr(self.model, 'prefix_cache_revision', 0)))

    def _refresh(self):
        current = self._namespace()
        if current != self.namespace:
            self.entries = ()
            self.retained_bytes = 0
            self.namespace = current

    def lookup(self, tokens):
        if not self.budget: return None
        tokens = tuple(tokens)
        with self.lock:
            self._refresh()
            candidates = [(len(key), i) for i, (key, _) in enumerate(self.entries)
                          if len(key) <= len(tokens) and tokens[:len(key)] == key]
            if not candidates:
                self.misses += 1
                return None
            length, index = max(candidates)
            entry = self.entries[index]
            self.entries = self.entries[:index] + self.entries[index+1:] + (entry,)
            cache, logits = pickle.loads(entry[1])
            self.hits += 1
            return Hit(cache, logits, length)

    def store(self, tokens, cache, logits):
        if not self.budget: return False
        key = tuple(tokens)
        if not isinstance(cache, RequestCache) or cache.position != len(key):
            raise ValueError('prefix snapshot requires complete prompt RequestCache at exact position')
        logits = np.asarray(logits)
        if logits.shape != (self.model.config['vocab_size'],) or not np.all(np.isfinite(logits)):
            raise ValueError('prefix snapshot requires finite final prompt logits')
        # Reject obviously oversized snapshots before allocating serialization
        # workspace. Scalar/tuple overhead is included in the final exact check.
        raw_bytes = logits.nbytes + sum(
            np.asarray(row).nbytes for layer in cache.layers
            for name in ('latents', 'rope_keys', 'index_keys')
            for row in getattr(layer, name))
        if raw_bytes > self.budget:
            with self.lock: self.rejected += 1
            return False
        # Serialized storage owns all bytes; subsequent decode cannot mutate it.
        allowance = self.budget - _size(((key, b''),))
        try:
            with _BoundedWriter(allowance) as stream:
                pickle.Pickler(stream, protocol=5).dump((cache, logits))
                blob = stream.getvalue()
        except _SnapshotTooLarge:
            with self.lock: self.rejected += 1
            return False
        entry = (key, blob)
        with self.lock:
            self._refresh()
            if _size((entry,)) > self.budget:
                self.rejected += 1
                return False
            entries = tuple(old for old in self.entries if old[0] != key)
            while entries and _size(entries + (entry,)) > self.budget:
                entries = entries[1:]
                self.evictions += 1
            self.entries = entries + (entry,)
            self.retained_bytes = _size(self.entries)
            assert self.retained_bytes <= self.budget
            return True

    def clear(self):
        with self.lock:
            self.entries = ()
            self.retained_bytes = 0

    def status(self):
        with self.lock:
            return {'budget_bytes': self.budget, 'retained_bytes': self.retained_bytes,
                    'entries': len(self.entries), 'hits': self.hits, 'misses': self.misses,
                    'evictions': self.evictions, 'rejected': self.rejected}
