import copy
import numpy as np
import pytest
from glm_vx.model import GlmMoeDsaModel, tiny_weights
from glm_vx.tiny import tiny_config
from glm_vx.prefill import prefill
from kernels.backend import VxBackend


def models(seed=17, **overrides):
    c = tiny_config()
    c.update(overrides)
    w = tiny_weights(c, seed)
    return [GlmMoeDsaModel(c, w, VxBackend()) for _ in range(2)]


def compare_cache(a, b):
    assert a.position == b.position
    for x, y in zip(a.layers, b.layers):
        for name in ('latents', 'rope_keys', 'index_keys', 'selected_indices'):
            np.testing.assert_array_equal(getattr(x, name), getattr(y, name))


@pytest.mark.parametrize('chunks', [[12], [1]*12, [2, 3, 7], [4, 4, 4]])
@pytest.mark.parametrize('seed', [17, 81, 203])
def test_chunking_exact_logits_and_caches(chunks, seed):
    a, b = models(seed)
    ca, cb = a.new_cache(), b.new_cache()
    tokens = [1, 15, 2, 8, 5, 17, 21, 9, 4, 8, 3, 2]
    reference = a.prefill(tokens, ca)
    output, cursor = [], 0
    for count in chunks:
        output.extend(prefill(b, tokens[cursor:cursor+count], cb, output_all_logits=True))
        cursor += count
    np.testing.assert_array_equal(output, reference)
    compare_cache(ca, cb)
    for token in (7, 8, 13):
        np.testing.assert_array_equal(a.forward(token, ca), b.forward(token, cb))
        compare_cache(ca, cb)


def test_gemm_is_used_and_reduces_weight_reads():
    a, b = models()
    calls, batches = [], []
    original = b.backend.linear_batch
    def batch(w, x):
        batches.append(len(x))
        return original(w, x)
    b.backend.linear_batch = batch
    weights = b.weights
    def weight(name):
        calls.append(name)
        return weights[name]
    b.weights = weight
    tokens = list(range(1, 13))
    np.testing.assert_array_equal(prefill(b, tokens, b.new_cache()), a.prefill(tokens)[-1])
    assert 12 in batches and any(size > 1 for size in batches)
    # Dense projection loaded once for the chunk rather than once per token.
    assert calls.count('model.layers.0.mlp.gate_proj.weight') == 1
    assert calls.count('model.layers.0.self_attn.q_a_proj.weight') == 1


@pytest.mark.parametrize('extra', [dict(tie_word_embeddings=True), dict(attention_bias=True), dict(n_shared_experts=0)])
def test_supported_architecture_variants(extra):
    a, b = models(**extra)
    tokens = [11, 7, 2, 17, 1, 2, 3]
    np.testing.assert_array_equal(prefill(b, tokens, b.new_cache(), output_all_logits=True), a.prefill(tokens))


def test_failure_rolls_back_partial_layer_chunk():
    a, b = models()
    ca, cb = a.new_cache(), b.new_cache()
    a.forward(1, ca); b.forward(1, cb)
    old = b._weight
    def fail(name):
        if name == 'model.layers.2.mlp.gate.weight': raise RuntimeError('injected weight failure')
        return old(name)
    b._weight = fail
    with pytest.raises(RuntimeError, match='injected'):
        prefill(b, [7, 8, 9], cb)
    compare_cache(ca, cb)
    b._weight = old
    np.testing.assert_array_equal(prefill(b, [7, 8, 9], cb), a.prefill([7, 8, 9], ca)[-1])
    compare_cache(ca, cb)


@pytest.mark.parametrize('tokens', [[], [True], [-1], [256], [1.0], [1]*65])
def test_invalid_chunk_does_not_modify_state(tokens):
    a, b = models()
    ca, cb = a.new_cache(), b.new_cache()
    a.forward(1, ca); b.forward(1, cb)
    with pytest.raises(ValueError): prefill(b, tokens, cb)
    compare_cache(ca, cb)


def test_context_and_trace_rejected_before_mutation():
    a, b = models(max_position_embeddings=3)
    ca, cb = a.new_cache(), b.new_cache()
    a.forward(1, ca); b.forward(1, cb)
    with pytest.raises(ValueError, match='context'): prefill(b, [1, 2, 3], cb)
    compare_cache(ca, cb)
    b.trace = lambda *args: None
    with pytest.raises(ValueError, match='trace'): prefill(b, [1], cb)
    compare_cache(ca, cb)


@pytest.mark.parametrize('rows', [0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65])
def test_tiled_linear_batch_keeps_scalar_reduction_order(rows):
    backend = VxBackend()
    rng = np.random.default_rng(83)
    x = rng.normal(size=(rows, 129)).astype(np.float32)
    w = rng.normal(size=(33, 129)).astype(np.float32)
    # Mixed signs and magnitudes exercise cancellation in every independent sum.
    w[:, ::3] *= 8192
    result = backend.linear_batch(w, x)
    expected = np.stack([backend.matvec(w, row) for row in x]) if rows else np.empty((0, 33), np.float32)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_allclose(result, x.astype(np.float64) @ w.astype(np.float64).T, atol=.12, rtol=2e-5)


def test_nonfinal_chunk_skips_vocabulary_projection():
    a, b = models()
    cache = b.new_cache()
    original = b._weight
    reads = []
    def read(name):
        reads.append(name)
        return original(name)
    b._weight = read
    assert b.prefill_chunk([1, 7, 8], cache, output_logits=False) is None
    assert 'lm_head.weight' not in reads
    np.testing.assert_array_equal(b.prefill_chunk([3, 4], cache), a.prefill([1, 7, 8, 3, 4])[-1])


def test_batched_serving_prefix_reuse_and_seed_equivalence():
    from glm_vx.scheduler import Scheduler
    a, b = models()
    kwargs = dict(max_sequences=8, token_budget=4096, prefill_chunk=8)
    sa = Scheduler(a, **kwargs)
    sb = Scheduler(b, **kwargs, batched_prefill=True, prefix_cache_bytes=2**20)
    def collect(req):
        tokens = []
        while True:
            event = req.events.get(timeout=10)
            if event['type'] == 'token': tokens.append(event['token_id'])
            else:
                assert event['reason'] in ('length', 'stop'), event
                return tokens, event
    prompt = [1, 2, 5, 17, 19, 23, 2, 1, 7, 8, 11, 18]
    try:
        # Prime immutable prompt snapshot independently of sampling history.
        base = collect(sa.submit(prompt, 5, temperature=.8, seed=23))
        primed = collect(sb.submit(prompt, 5, temperature=.8, seed=23))
        assert base[0] == primed[0]
        requests = []
        for suffix, seed in [([], 0), ([19, 17, 1], 6), ([], 23), ([23], 29)]:
            tokens = prompt + suffix
            requests.append((sa.submit(tokens, 7, temperature=.8, seed=seed),
                             sb.submit(tokens, 7, temperature=.8, seed=seed)))
        for uncached, cached in requests:
            expected, actual = collect(uncached), collect(cached)
            assert actual[0] == expected[0]
            assert actual[1]['cached_prompt_tokens'] >= len(prompt)
        assert sb.status()['batched_prefill'] is True
        assert sb.status()['active_requests'] == 0
    finally:
        sa.close(); sb.close()
