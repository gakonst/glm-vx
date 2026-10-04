"""Bounded CPU request handoff; no pickle, object arrays, or executable payloads.

The wire format is MAGIC | big-endian uint32 JSON length | canonical UTF-8 JSON
| fixed-layout little-endian arrays | SHA256(all preceding bytes). Shapes come
only from the locally verified model and bounded position, never array headers.
Array order: consumed token IDs, next-token logits, then each layer's MLA
latents, rotary keys, full-indexer keys (absent for shared layers), and selection.
The digest detects corruption; it is NOT authentication. Use an authenticated
transport with trusted producers. No scheduler, network service, or GPU path is
provided here.

In-memory float32 weight mappings, immutable GGUF checkpoints, and the
NumPy/Vx CPU backends are supported. Identity binds model/backend source, NumPy
version, Vx library bytes,
complete config/resolved architecture, and every named weight's shape and bytes
(or every byte of every GGUF shard, plus reader/dequantizer implementation).
GGUFIdentity hashes once on first model binding with 1 MiB chunks, then validates
file size/device/inode/mtime/ctime at transfer/claim. Shards must remain immutable;
stat checks detect ordinary edits/replacements, not malicious filesystem actors.
Weights/config/backend must stay immutable for a request's lifetime. Mapping
identity is rehashed at transfer/claim; GGUF bytes are never dequantized to hash.
No full production-checkpoint inference is asserted by the synthetic tests.

OwnedRequest disables the source after transfer. HandoffReceiver rejects replay
and duplicate request claims for its lifetime, with a bounded ledger. A single
receiver must own a destination ID. This is cooperative in-process ownership,
not durable cross-host exactly-once delivery: persist/coordinate the claim ledger
before using a transport with retries or restarts. Failed transport leaves the
source relinquished; recovering that transfer requires external coordination.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import hmac
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import struct
import threading
import uuid

import numpy as np

from .backend import NumpyBackend
from .model import GlmMoeDsaModel, LayerCache, RequestCache

MAGIC = b'GLMVXH1\n'
_ID = re.compile(r'[A-Za-z0-9_.:-]{1,128}\Z')
_GGUF_BIND_LOCK = threading.Lock()


class HandoffError(ValueError):
    """Invalid identity, wire payload, request state, or ownership transition."""


@dataclass(frozen=True)
class Limits:
    max_payload_bytes: int = 64 * 1024 * 1024
    max_header_bytes: int = 16 * 1024
    max_tokens: int = 8192
    max_layers: int = 256
    max_dimension: int = 65536
    max_vocab: int = 1_000_000
    max_weight_bytes: int = 64 * 1024 * 1024
    max_claims: int = 4096

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in vars(self).values()):
            raise HandoffError('limits must be positive integers')


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=True, allow_nan=False).encode('ascii')
    except (TypeError, ValueError, RecursionError) as exc:
        raise HandoffError('metadata must be finite canonical JSON') from exc


def _name(value):
    if type(value) is not str or not _ID.fullmatch(value):
        raise HandoffError('IDs must be 1..128 ASCII letters, digits, or _.:-')
    return value


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise HandoffError(f'invalid {label}')
    return value


def _dimensions(model, limits):
    if type(model) is not GlmMoeDsaModel:
        raise HandoffError('handoff requires the unmodified GlmMoeDsaModel class')
    _integer(model.n_layers, 1, limits.max_layers, 'layer count')
    for dim in (model.hidden, model.heads, model.rank, model.nope, model.rot,
                model.value_dim, model.index_heads, model.index_dim, model.index_topk):
        _integer(dim, 1, limits.max_dimension, 'model dimension')
    _integer(model.config['vocab_size'], 1, min(limits.max_vocab, 2**32 - 1), 'vocabulary')


def _file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_signature(path):
    try:
        value = path.stat()
    except OSError as exc:
        raise HandoffError('GGUF shard is missing or inaccessible') from exc
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


@dataclass(frozen=True)
class GGUFIdentity:
    """Immutable identity of an open GGUF checkpoint's actual mapped shard bytes.

    capture streams the read-only file mappings in 1 MiB chunks, without reading
    any tensor through the decoder. The paths/stats are local guards; only ordered
    shard sizes and content hashes enter the portable digest. Copying the exact
    checkpoint to another directory therefore preserves its identity. Capture
    costs a full sequential checkpoint read once per bound model/worker startup.
    """
    checkpoint: object
    paths: tuple
    stamps: tuple
    readers: tuple
    digest: str

    @classmethod
    def capture(cls, checkpoint):
        from .gguf_checkpoint import GGUFCheckpoint
        if type(checkpoint) is not GGUFCheckpoint or checkpoint.store.closed:
            raise HandoffError('GGUF identity requires an open GGUFCheckpoint')
        paths = tuple(Path(path).resolve() for path in checkpoint.store.paths)
        if not 1 <= len(paths) <= 1024 or len(paths) != len(checkpoint.store.readers):
            raise HandoffError('invalid GGUF shard list')
        stamps = tuple(_stat_signature(path) for path in paths)
        shards = []
        for reader, stamp in zip(checkpoint.store.readers, stamps):
            data = reader.data
            if data.flags.writeable or data.ndim != 1 or data.dtype != np.dtype('uint8') or data.nbytes != stamp[2]:
                raise HandoffError('GGUF shards must be immutable whole-file mappings')
            digest = hashlib.sha256()
            for offset in range(0, data.nbytes, 1024 * 1024):
                digest.update(data[offset:offset + 1024 * 1024].tobytes())
            shards.append([data.nbytes, digest.hexdigest()])
        if tuple(_stat_signature(path) for path in paths) != stamps:
            raise HandoffError('GGUF shard changed during identity capture')
        return cls(checkpoint, paths, stamps, tuple(id(r) for r in checkpoint.store.readers),
                   hashlib.sha256(_json({'kind': 'gguf-shards-v1', 'shards': shards})).hexdigest())

    def validate(self, checkpoint):
        if (checkpoint is not self.checkpoint or checkpoint.store.closed
                or tuple(Path(p).resolve() for p in checkpoint.store.paths) != self.paths
                or tuple(id(r) for r in checkpoint.store.readers) != self.readers
                or tuple(_stat_signature(path) for path in self.paths) != self.stamps):
            raise HandoffError('GGUF checkpoint changed after identity capture')


def _gguf_identity(model):
    from .gguf_checkpoint import GGUFCheckpoint
    if type(model.weights) is not GGUFCheckpoint:
        raise HandoffError('identity requires an in-memory weight mapping or GGUFCheckpoint')
    # A shared model can receive simultaneous requests. Only the first binder
    # scans checkpoint bytes; no partial identity is published to other callers.
    with _GGUF_BIND_LOCK:
        binding = getattr(model, '_handoff_gguf_identity', None)
        if binding is None:
            binding = GGUFIdentity.capture(model.weights)
            model._handoff_gguf_identity = binding
    binding.validate(model.weights)
    # Bind conversion/quantization semantics as well as raw checkpoint contents.
    modules = ('glm_vx.gguf_checkpoint', 'glm_vx.gguf_reader', 'glm_vx.config',
               'gguf.gguf_reader', 'gguf.quants', 'gguf.constants')
    decoder = {name: _file_digest(importlib.import_module(name).__file__) for name in modules}
    descriptor = dict(shards=binding.digest, decoder=decoder,
                      version=importlib.metadata.version('gguf'), config=model.weights.config)
    return hashlib.sha256(_json(descriptor)).hexdigest()


def _mapping_identity(weights, limits):
    if not weights or len(weights) > 100_000:
        raise HandoffError('invalid weight mapping size')
    total = 0
    for name, value in weights.items():
        if type(name) is not str or len(name) > 512 or not name:
            raise HandoffError('invalid weight name')
        if not isinstance(value, np.ndarray) or value.dtype.kind != 'f' or value.dtype.itemsize != 4:
            raise HandoffError('identity requires float32 weight arrays')
        total += value.nbytes
        if total > limits.max_weight_bytes:
            raise HandoffError('weight identity exceeds CPU proof-of-concept limit')
    digest = hashlib.sha256()
    for name in sorted(weights):
        value = weights[name]
        if not np.isfinite(value).all():
            raise HandoffError('weights must be finite')
        descriptor = _json([name, list(value.shape), '<f4'])
        digest.update(struct.pack('>I', len(descriptor)))
        digest.update(descriptor)
        digest.update(np.ascontiguousarray(value, dtype='<f4').tobytes())
    return digest.hexdigest()


def model_identity(model, limits=Limits()):
    """Bind supported CPU model semantics; GGUF bytes are scanned only once."""
    _dimensions(model, limits)
    from kernels.backend import VxBackend
    if type(model.backend) not in (NumpyBackend, VxBackend):
        raise HandoffError('only NumPy and Vx CPU backends have a wire contract')
    weights = (_mapping_identity(model.weights, limits) if isinstance(model.weights, Mapping)
               else _gguf_identity(model))
    model_module = importlib.import_module(GlmMoeDsaModel.__module__)
    backend_module = importlib.import_module(type(model.backend).__module__)
    backend = dict(name=type(model.backend).__name__,
                   source=_file_digest(backend_module.__file__), numpy=np.__version__)
    if type(model.backend) is VxBackend:
        backend['library'] = _file_digest(model.backend.library_path)
    resolved = {name: getattr(model, name) for name in (
        'n_layers', 'hidden', 'heads', 'rank', 'nope', 'rot', 'value_dim',
        'index_heads', 'index_dim', 'index_topk', 'eps', 'theta')}
    config = dict(config=model.config, indexer_types=model.indexer_types,
                  mlp_types=model.mlp_types, resolved=resolved)
    return dict(model=_file_digest(model_module.__file__),
                config=hashlib.sha256(_json(config)).hexdigest(),
                weights=weights, backend=hashlib.sha256(_json(backend)).hexdigest())


def _layout(model, position):
    yield 'tokens', '<u4', (position,)
    yield 'logits', '<f4', (model.config['vocab_size'],)
    for i, kind in enumerate(model.indexer_types):
        yield f'{i}.latents', '<f4', (position, model.rank)
        yield f'{i}.rope_keys', '<f4', (position, model.rot)
        yield f'{i}.index_keys', '<f4', (position if kind == 'full' else 0, model.index_dim)
        yield f'{i}.selected_indices', '<u4', (min(position, model.index_topk),)


def _body_size(model, position):
    return sum(math.prod(shape) * 4 for _, _, shape in _layout(model, position))


def _check_position(model, position, limits):
    _integer(position, 1, min(limits.max_tokens, int(model.config.get('max_position_embeddings', 2**31 - 1))),
             'cache position')
    if _body_size(model, position) + len(MAGIC) + 4 + 32 > limits.max_payload_bytes:
        raise HandoffError('cache exceeds payload byte limit')


def _check_arrays(model, arrays, position):
    if np.any(arrays['tokens'] >= model.config['vocab_size']):
        raise HandoffError('token outside vocabulary')
    previous = None
    for name, dtype, shape in _layout(model, position):
        value = arrays[name]
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise HandoffError(f'invalid array shape/dtype: {name}')
        if dtype == '<f4' and not np.isfinite(value).all():
            raise HandoffError(f'nonfinite array: {name}')
    for i, kind in enumerate(model.indexer_types):
        selected = arrays[f'{i}.selected_indices']
        if np.any(selected >= position) or len(np.unique(selected)) != len(selected):
            raise HandoffError('invalid sparse selection')
        if kind == 'shared' and not np.array_equal(selected, previous):
            raise HandoffError('shared DSA selection differs from preceding full layer')
        previous = selected


def _pack(request, destination, transfer_id):
    model, limits = request.model, request.limits
    position = request.cache.position
    _check_position(model, position, limits)
    if len(request.tokens) != position or len(request.cache.layers) != model.n_layers:
        raise HandoffError('inconsistent request cache/token lengths')
    if request.phase == 'prefill':
        if request.prompt_length != 0:
            raise HandoffError('unsealed prefill has a prompt length')
    elif request.phase == 'decode':
        _integer(request.prompt_length, 1, position, 'prompt length')
    else:
        raise HandoffError('invalid request phase')
    arrays = dict(tokens=request.tokens, logits=request.logits)
    for i, layer in enumerate(request.cache.layers):
        for field in ('latents', 'rope_keys', 'index_keys', 'selected_indices'):
            arrays[f'{i}.{field}'] = getattr(layer, field)
    encoded = {}
    for name, dtype, shape in _layout(model, position):
        raw = np.asarray(arrays[name])
        # Lists of no index keys are the sole shape-normalization exception.
        if name.endswith('.index_keys') and shape[0] == 0 and raw.shape == (0,):
            raw = np.empty(shape, dtype=dtype)
        if raw.shape != shape:
            raise HandoffError(f'invalid array shape: {name}')
        if dtype == '<u4':
            if raw.dtype.kind not in 'iu' or np.any(raw < 0) or np.any(raw > 2**32 - 1):
                raise HandoffError(f'invalid integer array: {name}')
        elif raw.dtype.kind != 'f' or raw.dtype.itemsize != 4:
            raise HandoffError(f'invalid float32 array: {name}')
        encoded[name] = np.ascontiguousarray(raw, dtype=dtype)
    _check_arrays(model, encoded, position)
    header = _json(dict(version=1, identity=request.identity, request_id=request.request_id,
                        source=request.owner_id, destination=destination, transfer_id=transfer_id,
                        position=position, phase=request.phase, prompt_length=request.prompt_length))
    if len(header) > limits.max_header_bytes:
        raise HandoffError('header exceeds byte limit')
    size = len(MAGIC) + 4 + len(header) + _body_size(model, position) + 32
    if size > limits.max_payload_bytes:
        raise HandoffError('payload exceeds byte limit')
    body = b''.join(encoded[name].tobytes() for name, _, _ in _layout(model, position))
    wire = MAGIC + struct.pack('>I', len(header)) + header + body
    return wire + hashlib.sha256(wire).digest()


def _unpack(payload, model, limits, expected_identity):
    if type(payload) is not bytes or not len(MAGIC) + 4 + 32 <= len(payload) <= limits.max_payload_bytes:
        raise HandoffError('invalid payload type or byte length')
    if not payload.startswith(MAGIC):
        raise HandoffError('unsupported handoff magic/version')
    length = struct.unpack_from('>I', payload, len(MAGIC))[0]
    offset = len(MAGIC) + 4
    if not 1 <= length <= limits.max_header_bytes or offset + length > len(payload) - 32:
        raise HandoffError('invalid header length')
    if not hmac.compare_digest(hashlib.sha256(memoryview(payload)[:-32]).digest(), payload[-32:]):
        raise HandoffError('payload digest mismatch')
    header_bytes = payload[offset:offset + length]
    try:
        header = json.loads(header_bytes)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise HandoffError('invalid JSON header') from exc
    fields = {'version', 'identity', 'request_id', 'source', 'destination', 'transfer_id',
              'position', 'phase', 'prompt_length'}
    if type(header) is not dict or set(header) != fields or _json(header) != header_bytes:
        raise HandoffError('noncanonical or unexpected header fields')
    if type(header['version']) is not int or header['version'] != 1:
        raise HandoffError('unsupported handoff version')
    if header['identity'] != expected_identity:
        raise HandoffError('model/config/weight/backend identity mismatch')
    for field in ('request_id', 'source', 'destination', 'transfer_id'):
        _name(header[field])
    if header['source'] == header['destination']:
        raise HandoffError('handoff source and destination must differ')
    position = header['position']
    _check_position(model, position, limits)
    if header['phase'] == 'prefill':
        _integer(header['prompt_length'], 0, 0, 'unsealed prompt length')
    elif header['phase'] == 'decode':
        _integer(header['prompt_length'], 1, position, 'prompt length')
    else:
        raise HandoffError('invalid request phase')
    offset += length
    if len(payload) != offset + _body_size(model, position) + 32:
        raise HandoffError('payload layout/length mismatch')
    arrays = {}
    for name, dtype, shape in _layout(model, position):
        count = math.prod(shape)
        # frombuffer creates only views until *all* shape/value checks pass.
        arrays[name] = np.frombuffer(payload, dtype=dtype, count=count, offset=offset).reshape(shape)
        offset += count * 4
    _check_arrays(model, arrays, position)
    cache = RequestCache([], position)
    for i in range(model.n_layers):
        cache.layers.append(LayerCache(
            latents=[row.copy() for row in arrays[f'{i}.latents']],
            rope_keys=[row.copy() for row in arrays[f'{i}.rope_keys']],
            index_keys=[row.copy() for row in arrays[f'{i}.index_keys']],
            selected_indices=arrays[f'{i}.selected_indices'].astype(np.int64)))
    return header, cache, arrays['logits'].copy(), arrays['tokens'].astype(np.int64).tolist()


class OwnedRequest:
    """Single-owner sequential prefill/decode state.

    append_prompt supports arbitrary nonempty chunks; only each chunk's final
    token computes logits. seal_prompt fixes the prompt boundary. greedy consumes
    the saved next-token logits, appends the sampled token exactly once, and keeps
    next logits. Thus the cache includes *all* returned generated tokens, including
    the last one. Each append/greedy call rolls back on inference failure.
    EOS/stop policies and stochastic sampler/RNG migration are out
    of scope. Methods serialize transitions; exposed model/cache are for inspection
    only and must not be mutated behind this ownership wrapper.
    """
    def __init__(self, model, request_id, owner_id, *, limits=Limits()):
        self.model, self.limits = model, limits
        self.request_id, self.owner_id = _name(request_id), _name(owner_id)
        self.identity = model_identity(model, limits)
        self.cache = model.new_cache()
        self.logits = None
        self.tokens = []
        self.prompt_length = 0
        self.phase = 'prefill'
        self._active = True
        self._lock = threading.RLock()

    @property
    def active(self):
        return self._active

    def _require_active(self):
        if not self._active:
            raise HandoffError('request ownership has been transferred')

    def append_prompt(self, tokens):
        with self._lock:
            self._require_active()
            if self.phase != 'prefill':
                raise HandoffError('prompt is already sealed')
            if not isinstance(tokens, (list, tuple)) or not tokens:
                raise HandoffError('prompt chunk must be a nonempty list or tuple')
            _check_position(self.model, self.cache.position + len(tokens), self.limits)
            for token in tokens:
                _integer(token, 0, self.model.config['vocab_size'] - 1, 'token')
            # Chunk atomicity: model rolls back its failing token; roll back earlier
            # successful tokens too, preserving the prior sampling logits.
            position, logits = self.cache.position, self.logits
            selections = [layer.selected_indices for layer in self.cache.layers]
            try:
                for index, token in enumerate(tokens):
                    if index == len(tokens) - 1:
                        self.logits = self.model.forward(token, self.cache)
                    else:
                        self.model.prefill_token(token, self.cache)
                    self.tokens.append(token)
            except Exception:
                for layer, selected in zip(self.cache.layers, selections):
                    del layer.latents[position:]
                    del layer.rope_keys[position:]
                    del layer.index_keys[position:]
                    layer.selected_indices = selected
                self.cache.position, self.logits = position, logits
                del self.tokens[position:]
                raise
            return self.logits.copy()

    def seal_prompt(self):
        with self._lock:
            self._require_active()
            if self.phase != 'prefill' or not self.tokens:
                raise HandoffError('seal requires a nonempty unsealed prompt')
            self.prompt_length, self.phase = self.cache.position, 'decode'

    def greedy(self, count=1):
        with self._lock:
            self._require_active()
            if self.phase != 'decode':
                raise HandoffError('seal prompt before decoding')
            _integer(count, 0, self.limits.max_tokens, 'decode count')
            _check_position(self.model, self.cache.position + count, self.limits)
            result = []
            position, previous_logits = self.cache.position, self.logits
            selections = [layer.selected_indices for layer in self.cache.layers]
            try:
                for _ in range(count):
                    token = int(np.argmax(self.logits))
                    logits = self.model.forward(token, self.cache)
                    self.tokens.append(token)
                    self.logits = logits
                    result.append(token)
            except Exception:
                # No caller-visible tokens were returned yet. Restore the whole
                # batch so an explicit caller retry cannot lose/duplicate samples.
                for layer, selected in zip(self.cache.layers, selections):
                    del layer.latents[position:]
                    del layer.rope_keys[position:]
                    del layer.index_keys[position:]
                    layer.selected_indices = selected
                self.cache.position, self.logits = position, previous_logits
                del self.tokens[position:]
                raise
            return result

    def transfer(self, destination, *, transfer_id=None):
        """Serialize then relinquish ownership; retain returned bytes for delivery."""
        with self._lock:
            self._require_active()
            destination = _name(destination)
            transfer_id = _name(transfer_id or uuid.uuid4().hex)
            if destination == self.owner_id:
                raise HandoffError('handoff source and destination must differ')
            if model_identity(self.model, self.limits) != self.identity:
                raise HandoffError('model identity changed during request')
            payload = _pack(self, destination, transfer_id)
            self._active = False
            return payload


class HandoffReceiver:
    """One destination's bounded, non-evicting, process-local claim ledger."""
    def __init__(self, model, owner_id, *, limits=Limits()):
        self.model, self.owner_id, self.limits = model, _name(owner_id), limits
        self.identity = model_identity(model, limits)
        self._requests, self._transfers = set(), set()
        self._lock = threading.Lock()

    def claim(self, payload, *, request_id, source):
        """Validate independently; invalid claims never consume ownership slots."""
        request_id, source = _name(request_id), _name(source)
        with self._lock:
            if model_identity(self.model, self.limits) != self.identity:
                raise HandoffError('receiver model identity changed')
            header, cache, logits, tokens = _unpack(payload, self.model, self.limits, self.identity)
            if (header['request_id'] != request_id or header['source'] != source
                    or header['destination'] != self.owner_id):
                raise HandoffError('request/source/destination ownership mismatch')
            if request_id in self._requests or header['transfer_id'] in self._transfers:
                raise HandoffError('duplicate request or replayed transfer')
            if len(self._requests) >= self.limits.max_claims:
                raise HandoffError('claim ledger is full')
            request = OwnedRequest(self.model, request_id, self.owner_id, limits=self.limits)
            request.cache, request.logits, request.tokens = cache, logits, tokens
            request.phase, request.prompt_length = header['phase'], header['prompt_length']
            self._requests.add(request_id)
            self._transfers.add(header['transfer_id'])
            return request
