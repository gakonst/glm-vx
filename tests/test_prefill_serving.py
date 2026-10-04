"""Serving prefill omits output heads, without changing causal state or logits."""
import numpy as np
import pytest
from glm_vx.model import Model, tiny_weights
from glm_vx.tiny import tiny_config
from glm_vx.backend import NumpyBackend

@pytest.mark.parametrize('tied', [False, True])
def test_prefill_without_logits_matches_decode_and_avoids_head(tied):
    cfg = tiny_config(); cfg['tie_word_embeddings'] = tied
    weights = tiny_weights(cfg, seed=19)
    reference = Model(cfg, weights, NumpyBackend())
    accessed = []
    def read(name):
        accessed.append(name)
        return weights[name]
    model = Model(cfg, read, NumpyBackend())
    expected, actual = reference.new_cache(), model.new_cache()
    for token in [1, 3, 7, 9, 2, 11]:
        reference.forward(token, expected)
        accessed.clear()
        assert model.prefill_token(token, actual) is None
        assert 'model.norm.weight' not in accessed
        assert 'lm_head.weight' not in accessed
        # Tied embeddings must be accessed once for lookup, never again as head.
        assert accessed.count('model.embed_tokens.weight') == 1
    for token in [5, 8]:
        np.testing.assert_array_equal(model.forward(token, actual), reference.forward(token, expected))
    assert actual.position == expected.position
    for a, e in zip(actual.layers, expected.layers):
        np.testing.assert_array_equal(a.latents, e.latents)
        np.testing.assert_array_equal(a.rope_keys, e.rope_keys)
        np.testing.assert_array_equal(a.index_keys, e.index_keys)
        np.testing.assert_array_equal(a.selected_indices, e.selected_indices)


def test_prefill_failure_rolls_back_cache_and_can_retry():
    cfg = tiny_config(); weights = tiny_weights(cfg, seed=19)
    model = Model(cfg, weights, NumpyBackend()); cache = model.new_cache()
    model.prefill_token(1, cache)
    name = 'model.layers.2.mlp.gate.weight'
    saved = weights.pop(name)
    with pytest.raises(KeyError): model.prefill_token(3, cache)
    assert cache.position == 1
    assert all(len(layer.latents) == 1 and len(layer.rope_keys) == 1 for layer in cache.layers)
    weights[name] = saved
    model.prefill_token(3, cache)
    reference = Model(cfg, weights, NumpyBackend()); other = reference.new_cache()
    reference.forward(1, other); reference.forward(3, other)
    np.testing.assert_array_equal(model.forward(4, cache), reference.forward(4, other))


def test_model_routes_dense_shared_and_selected_experts_through_resident_mlp():
    class Resident(NumpyBackend):
        calls = 0
        def mlp(self, gate, up, down, x):
            self.calls += 1
            return self.matvec(down, self.swiglu(self.matvec(gate, x), self.matvec(up, x)))
    cfg = tiny_config(); weights = tiny_weights(cfg, seed=19)
    backend = Resident(); model = Model(cfg, weights, backend)
    reference = Model(cfg, weights, NumpyBackend())
    a, e = model.new_cache(), reference.new_cache()
    for token in [1, 3, 7]:
        np.testing.assert_array_equal(model.forward(token, a), reference.forward(token, e))
    assert backend.calls == 3 * (1 + 3 * (2 + 1))


def test_scheduler_only_requests_logits_for_last_prompt_and_decode():
    from glm_vx.scheduler import Scheduler
    class Counting:
        config = {'vocab_size':16, 'max_position_embeddings':64, 'eos_token_id':[]}
        def __init__(self): self.calls = []
        def new_cache(self): return []
        def prefill_token(self, token, cache):
            self.calls.append(('prefill', token)); cache.append(token)
        def forward(self, token, cache):
            self.calls.append(('forward', token)); cache.append(token)
            return np.arange(16, dtype=np.float32)
    model = Counting(); scheduler = Scheduler(model, prefill_chunk=2)
    try:
        req = scheduler.submit([1, 2, 3, 4, 5], 3)
        while req.events.get(timeout=5)['type'] != 'done': pass
        assert model.calls == [('prefill', t) for t in [1,2,3,4]] + [('forward',t) for t in [5,15,15]]
    finally: scheduler.close()
