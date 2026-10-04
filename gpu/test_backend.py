"""Hybrid adapter contract/numerics under explicitly injected CPU SIMT tests.

These tests do not establish GPU execution or performance. Production imports
never choose this test context; every fixture installs it with monkeypatch.
"""
import numpy as np
import pytest

import gpu.backend as adapter
from gpu.runtime import CUDAUnavailable
from gpu.testing.context import Context


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(adapter, 'CUDAContext', Context)
    value = adapter.GPUBackend(weight_cache_bytes=256)
    try:
        yield value
    finally:
        value.close()


def test_cuda_unavailable_propagates_without_cpu_fallback(monkeypatch):
    def absent(*args, **kwargs):
        raise CUDAUnavailable('test: no NVIDIA device')
    monkeypatch.setattr(adapter, 'CUDAContext', absent)
    with pytest.raises(CUDAUnavailable, match='no NVIDIA device'):
        adapter.GPUBackend()


def test_matvec_mutable_weights_never_cached(backend):
    weights = np.arange(12, dtype='f4').reshape(3, 4)
    x = np.arange(4, dtype='f4')
    np.testing.assert_array_equal(backend.matvec(weights, x), weights @ x)
    assert not backend.cache and backend.cache_bytes == 0
    weights += 2
    np.testing.assert_array_equal(backend.matvec(weights, x), weights @ x)
    assert all(t.closed for t in backend.ctx.tensors)


def test_readonly_weight_lru_is_bounded_and_reused(backend):
    weights = np.arange(16, dtype='f4').reshape(4, 4)
    weights.flags.writeable = False
    x = np.ones(4, dtype='f4')
    backend.matvec(weights, x)
    first_tensor = next(iter(backend.cache.values()))[1]
    backend.matvec(weights, x)
    assert next(iter(backend.cache.values()))[1] is first_tensor
    for i in range(8):
        other = np.full((4, 4), i + 1, dtype='f4')
        other.flags.writeable = False
        np.testing.assert_array_equal(backend.matvec(other, x), other @ x)
        assert backend.cache_bytes <= backend.cache_limit
        assert not backend.pinned
    assert first_tensor.closed
    assert backend.cache_bytes == sum(source.nbytes for source, tensor in backend.cache.values())
    assert all(not source.flags.writeable and not tensor.closed for source, tensor in backend.cache.values())


def test_pinned_weights_are_not_evicted_mid_operation(backend):
    backend.cache_limit = 64
    one = np.ones(16, dtype='f4');one.flags.writeable = False
    two = np.full(16, 2, dtype='f4');two.flags.writeable = False
    temps = []
    first = backend._upload(one, temps, True)
    second = backend._upload(two, temps, True)
    assert not first.closed
    assert backend.cache_bytes == 64
    assert second in temps
    backend._release(temps)
    assert not first.closed and second.closed and not backend.pinned


def test_temporary_cleanup_after_launch_failure(backend, monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError('injected launch failure')
    monkeypatch.setattr(backend, '_launch', failed)
    with pytest.raises(RuntimeError, match='injected launch failure'):
        backend.matvec(np.ones((3, 4), dtype='f4'), np.ones(4, dtype='f4'))
    assert all(t.closed for t in backend.ctx.tensors)
    assert not backend.pinned


@pytest.mark.parametrize('shape', [(67,), (3, 67)])
def test_rowwise_rmsnorm(backend, shape):
    rng = np.random.default_rng(88)
    x = rng.normal(size=shape).astype('f4')
    w = rng.normal(size=shape[-1]).astype('f4')
    expected = x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + 1e-5) * w
    np.testing.assert_allclose(backend.rmsnorm(x, w, 1e-5), expected, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize('shape', [(67,), (3, 67)])
def test_rowwise_softmax_including_masked_elements(backend, shape):
    rng = np.random.default_rng(22)
    x = rng.normal(size=shape).astype('f4')
    x[..., 1] = -np.inf
    z = np.exp(x - x.max(axis=-1, keepdims=True))
    expected = z / z.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(backend.softmax(x), expected, rtol=3e-6, atol=3e-6)


def test_router_selection_bias_and_stable_ties(backend):
    logits = np.array([0, 1, -1, 0], dtype='f4')
    bias = np.array([1, 0, 1, 1], dtype='f4')
    ids, weights = backend.route(logits, bias, 2, 2.5)
    np.testing.assert_array_equal(ids, [0, 3])
    np.testing.assert_allclose(weights, [1.25, 1.25], rtol=1e-6, atol=1e-6)


def test_rope_and_swiglu(backend):
    x = np.arange(24, dtype='f4').reshape(3, 8) / 10
    position, theta = 3, 10000
    angle = position * np.power(theta, -np.arange(0, 8, 2, dtype='f4') / 8)
    a, b = x[:, ::2], x[:, 1::2]
    expected = np.concatenate((a*np.cos(angle)-b*np.sin(angle), a*np.sin(angle)+b*np.cos(angle)), axis=-1)
    np.testing.assert_allclose(backend.rope(x, position, theta), expected, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(backend.swiglu(x, x), x*x/(1+np.exp(-x)), rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize('eps', [1e-50, 1e40])
def test_rmsnorm_rejects_epsilon_not_positive_finite_in_float32(backend, eps):
    # Positive finite Python floats can become zero/infinity at the kernel ABI.
    with pytest.raises(ValueError):
        backend.rmsnorm(np.ones(4, dtype='f4'), np.ones(4, dtype='f4'), eps)


def test_router_rejects_scale_that_overflows_float32(backend):
    with pytest.raises(ValueError):
        backend.route(np.zeros(4, dtype='f4'), np.zeros(4, dtype='f4'), 2, 1e40)


def test_tiny_glm_end_to_end_layout_and_cache(backend, monkeypatch):
    from glm_vx.backend import NumpyBackend
    from glm_vx.model import Model, tiny_weights
    from glm_vx.tiny import tiny_config
    config = tiny_config()
    config.update(vocab_size=16, hidden_size=8, num_attention_heads=1,
                  num_key_value_heads=1, kv_lora_rank=4, q_lora_rank=8,
                  qk_rope_head_dim=4, qk_nope_head_dim=4, qk_head_dim=8,
                  v_head_dim=4, index_head_dim=8, index_n_heads=1,
                  index_topk=2, intermediate_size=16, moe_intermediate_size=8,
                  num_hidden_layers=2, indexer_types=['full', 'shared'],
                  mlp_layer_types=['dense', 'sparse'], n_routed_experts=2,
                  num_experts_per_tok=2)
    weights = tiny_weights(config, seed=713)
    for weight in weights.values():
        weight.flags.writeable = False
    calls = []
    original_attention = backend.compressed_attention
    def traced_attention(q, qr, latent, rotary, scale):
        calls.append((q.shape, qr.shape, latent.shape, rotary.shape))
        return original_attention(q, qr, latent, rotary, scale)
    monkeypatch.setattr(backend, 'compressed_attention', traced_attention)
    def unexpected_softmax(*args, **kwargs):
        pytest.fail('fused model path must not invoke per-head backend.softmax')
    monkeypatch.setattr(backend, 'softmax', unexpected_softmax)
    actual_model = Model(config, weights, backend)
    oracle_model = Model(config, weights, NumpyBackend())
    actual_cache, oracle_cache = actual_model.new_cache(), oracle_model.new_cache()
    # Third token forces top-k selection over a prefix longer than index_topk;
    # two different nonzero positions exercise the rotary layout.
    for position, token in enumerate((1, 3, 7)):
        actual = actual_model.forward(token, actual_cache)
        expected = oracle_model.forward(token, oracle_cache)
        np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=3e-6)
        assert actual_cache.position == oracle_cache.position == position + 1
        for layer, (got, want) in enumerate(zip(actual_cache.layers, oracle_cache.layers)):
            np.testing.assert_array_equal(got.selected_indices, want.selected_indices)
            np.testing.assert_allclose(got.latents, want.latents, rtol=4e-5, atol=3e-6)
            np.testing.assert_allclose(got.rope_keys, want.rope_keys, rtol=4e-5, atol=3e-6)
            if layer == 0:
                np.testing.assert_allclose(got.index_keys, want.index_keys, rtol=4e-5, atol=3e-6)
            else:
                assert got.index_keys == []
        assert backend.cache_bytes <= backend.cache_limit
        assert not backend.pinned
    assert len(calls) == 6  # two layers for each of three tokens
    assert [shape[2][0] for shape in calls] == [1, 1, 2, 2, 2, 2]
    assert all(q == (1, 4) and qr == (1, 4) for q, qr, latent, rotary in calls)


def test_compressed_attention_multihead_split_boundary(backend):
    rng = np.random.default_rng(381)
    heads, rank, rdim, count = 2, 33, 8, 257
    q = rng.normal(0, .1, (heads, rank)).astype('f4')
    qr = rng.normal(0, .1, (heads, rdim)).astype('f4')
    latent = rng.normal(0, .1, (count, rank)).astype('f4')
    rotary = rng.normal(0, .1, (count, rdim)).astype('f4')
    scale = .125
    scores = (q.astype('f8') @ latent.astype('f8').T + qr.astype('f8') @ rotary.astype('f8').T) * scale
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = probabilities @ latent.astype('f8')
    actual = backend.compressed_attention(q, qr, latent, rotary, scale)
    assert actual.shape == (heads, rank)
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=3e-6)
    assert all(t.closed for t in backend.ctx.tensors)
