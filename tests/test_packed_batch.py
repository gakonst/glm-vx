"""Ordered F32 parity for fused packed multi-token projections and real prefill."""
import struct
import numpy as np
import pytest
from kernels.backend import VxBackend
from kernels.packed import LAYOUTS, matmul
from test_packed_matvec import payload, ordered_dot, native, real_store
from test_gguf_checkpoint import fixture_data
from test_packed_model import write_mixed_fixture
from test_batched_prefill import compare_cache
from glm_vx.gguf_checkpoint import GGUFCheckpoint
from glm_vx.model import Model


@pytest.fixture(scope='module')
def backend():
    return VxBackend()


@pytest.mark.parametrize('kind', LAYOUTS)
@pytest.mark.parametrize('batch', list(range(1, 18)) + [31, 32, 63, 64])
def test_all_codec_tiles_tails_against_independent_ordered_oracle(backend, native, kind, batch, monkeypatch):
    import gguf
    monkeypatch.setattr(gguf.quants, 'dequantize', lambda *a: pytest.fail('candidate decoder called'))
    block, size, suffix = LAYOUTS[kind]
    raw = payload(kind)
    if kind == 1:
        weights = np.array([struct.unpack('<e', row.tobytes())[0] for row in raw], np.float32)
    else:
        weights = native.decode(raw.tobytes(), suffix.upper(), kind, 12*block)
    weights = weights.reshape(3, 4*block)
    x = np.random.default_rng(4400+kind+batch).normal(size=(batch, 4*block)).astype(np.float32)
    x[:, ::2] *= np.float32(2**12)
    expected = np.stack([ordered_dot(weights, row) for row in x])
    actual = backend.packed_linear_batch(raw, kind, weights.shape, x)
    assert actual.tobytes() == expected.tobytes()
    assert actual.flags.c_contiguous


@pytest.mark.parametrize('kind', [0, 8, 10, 11, 12, 13, 14, 16, 18, 19, 23])
def test_trained_tensor_batch_matches_native_codec(backend, native, real_store, kind):
    store = real_store
    candidates = [t for t in store.tensors.values() if int(t.tensor_type) == kind and len(store.shape(t.name)) >= 2]
    tensor = candidates[len(candidates)//2]
    shape = store.shape(tensor.name)
    expert = shape[0]//2 if len(shape) == 3 else None
    start = shape[-2]//2
    with store.packed_rows(tensor.name, expert=expert, rows=slice(start, start+3)) as (raw, dims, fmt):
        assert not raw.flags.owndata
        weights = native.decode(raw.tobytes(), tensor.tensor_type.name, kind, np.prod(dims)).reshape(dims)
        x = np.random.default_rng(981+kind).normal(size=(9, dims[1])).astype(np.float32)
        expected = np.stack([ordered_dot(weights, row) for row in x])
        assert backend.packed_linear_batch(raw, fmt, dims, x).tobytes() == expected.tobytes()


@pytest.mark.parametrize('batch', [0, 1, 3, 8, 17, 64])
def test_f32_borrowed_matrix_and_strided_activation_input(backend, batch):
    rng = np.random.default_rng(81)
    w = rng.normal(size=(7, 33)).astype(np.float32)
    x = rng.normal(size=(33, batch)).astype(np.float32).T
    expected = np.stack([ordered_dot(w, row) for row in x]) if batch else np.empty((0, 7), np.float32)
    actual = backend.packed_linear_batch(w.view(np.uint8), 0, w.shape, x)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('change, message', [
    ({'x': np.zeros((65, 256), np.float32)}, '64'),
    ({'x': np.zeros(256, np.float32)}, 'columns'),
    ({'x': np.zeros((2, 255), np.float32)}, 'columns'),
    ({'x': np.full((2, 256), np.nan, np.float32)}, 'finite'),
    ({'raw': np.zeros(49, np.uint8)}, 'byte count'),
    ({'raw': np.zeros(100, np.uint8)[::2]}, 'contiguous'),
    ({'kind': 999}, 'unsupported'),
    ({'shape': (2**31, 256)}, 'int32'),
])
def test_bounds(backend, change, message):
    args = dict(raw=np.zeros(50, np.uint8), kind=19, shape=(1, 256), x=np.ones((2, 256), np.float32))
    args.update(change)
    with pytest.raises(ValueError, match=message): backend.packed_linear_batch(**args)


def test_batch_indexing_capacity_without_giant_allocation():
    from kernels.packed import batch_input
    # rows*cols is legal but rows*batch is not. Zero-stride views avoid allocating
    # a giant fake payload; validation fails before a kernel or output allocation.
    raw = np.lib.stride_tricks.as_strided(np.zeros(1, np.uint8), shape=(0,), strides=(1,))
    import kernels.packed as packed
    from unittest.mock import patch
    with patch.object(packed, '_buffers', return_value=(2**30, 1, 'f16')):
        with pytest.raises(ValueError, match='int32'):
            batch_input(raw, 1, (2**30, 1), np.zeros((3, 1), np.float32))


def test_empty_rows_and_batch(backend):
    assert backend.packed_linear_batch(np.zeros(0, np.uint8), 19, (0, 256), np.zeros((7, 256), np.float32)).shape == (7, 0)
    assert backend.packed_linear_batch(np.zeros(50, np.uint8), 19, (1, 256), np.zeros((0, 256), np.float32)).shape == (0, 1)


def test_old_library_explicit_packed_fallback(backend):
    class MatvecOnly:
        def __getattr__(self, name):
            if name.startswith('glm_vx_packed_batch_'): raise AttributeError(name)
            return getattr(backend.lib, name)
    raw = payload(19, 3)
    x = np.random.default_rng(81).normal(size=(9, 256)).astype(np.float32)
    expected = np.stack([backend.packed_matvec(raw, 19, (3, 256), row) for row in x])
    assert matmul(MatvecOnly(), raw, 19, (3, 256), x).tobytes() == expected.tobytes()


def test_corrupt_scale_rejected(backend):
    raw = np.zeros(50, np.uint8)
    raw[:2] = np.frombuffer(struct.pack('<e', float('inf')), np.uint8)
    with pytest.raises(ValueError, match='nonfinite'):
        backend.packed_linear_batch(raw, 19, (1, 256), np.ones((8, 256), np.float32))


@pytest.mark.parametrize('batch', [2, 3, 8, 9, 17])
def test_prefill_never_expands_supported_projections_and_matches_decode(tmp_path, monkeypatch, batch):
    c, w = fixture_data(); path = tmp_path/'mixed.gguf'; write_mixed_fixture(path, c, w)
    plain, packed = GGUFCheckpoint(path, c), GGUFCheckpoint(path, c)
    a, b = Model(c, plain, VxBackend()), Model(c, packed, VxBackend(), packed_weights=True)
    ca, cb = a.new_cache(), b.new_cache()
    original = packed.tensor
    def no_expansion(name, *, rows=None):
        if rows is None and not name.endswith('kv_b_proj.weight'):
            source, expert = packed._resolve(name)
            if len(packed.store.shape(source)) >= 2:
                pytest.fail('expanded multirow packed projection: '+name)
        return original(name, rows=rows)
    monkeypatch.setattr(packed, 'tensor', no_expansion)
    try:
        tokens = [1+(i*7)%19 for i in range(batch)]
        expected = a.prefill(tokens, ca)
        from glm_vx.prefill import prefill
        actual = prefill(b, tokens, cb, output_all_logits=True)
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
        compare_cache(ca, cb)
        assert a.forward(11, ca).tobytes() == b.forward(11, cb).tobytes()
        compare_cache(ca, cb)
    finally:
        plain.close(); packed.close()


def test_checkpoint_old_backend_and_expanded_fallback(tmp_path, monkeypatch):
    from glm_vx.backend import NumpyBackend
    c, w = fixture_data(); path = tmp_path/'mixed.gguf'; write_mixed_fixture(path, c, w)
    cp = GGUFCheckpoint(path, c)
    class OldBackend(VxBackend): packed_linear_batch = None
    name = 'model.layers.1.mlp.experts.1.up_proj.weight'
    x = np.ones((3, w[name].shape[1]), np.float32)
    try:
        expected = np.stack([cp.linear(name, row, VxBackend()) for row in x])
        original = cp.tensor
        monkeypatch.setattr(cp, 'tensor', lambda *a: pytest.fail('old packed backend expanded weights'))
        assert cp.linear_batch(name, x, OldBackend()).tobytes() == expected.tobytes()
        monkeypatch.setattr(cp, 'tensor', original)
        np.testing.assert_allclose(cp.linear_batch(name, x, NumpyBackend()), expected, atol=2e-6)
        cp.close()
        with pytest.raises(RuntimeError, match='closed'): cp.linear_batch(name, x, VxBackend())
    finally: cp.close()


def test_generated_batch_source_reproducible():
    from pathlib import Path
    from kernels.generate_packed_batch import MARKER, generate
    path = Path(__file__).resolve().parents[1]/'kernels'/'packed.vx'
    assert MARKER+path.read_text().split(MARKER)[1] == generate()+'\n'
