"""Small GLM architecture tests; NumPy here is an independent test oracle only."""
import unittest
from pathlib import Path

import numpy as np

from glm_vx.model import GlmMoeDsaModel, tiny_weights


class NumpyTestBackend:
    def matvec(self, w, x):
        return np.asarray(w, np.float32) @ np.asarray(x, np.float32)

    def rmsnorm(self, x, w, eps):
        return x / np.sqrt(np.mean(x * x) + eps) * w

    def rope(self, x, position, theta):
        angle = position * theta ** (-np.arange(0, len(x), 2, dtype=np.float32) / len(x))
        a, b = x[::2], x[1::2]
        return np.concatenate((a * np.cos(angle) - b * np.sin(angle), b * np.cos(angle) + a * np.sin(angle)))

    def softmax(self, x):
        p = np.exp(x - np.max(x))
        return p / p.sum()

    def swiglu(self, gate, up):
        return gate / (1 + np.exp(-gate)) * up

    def route(self, logits, bias, k, scale):
        p = 1 / (1 + np.exp(-logits))
        idx = np.argsort(-(p + bias), kind='stable')[:k]
        return idx, p[idx] / p[idx].sum() * scale


def tiny_fixture(seed=123):
    c = dict(hidden_size=12, vocab_size=23, num_hidden_layers=4,
             num_attention_heads=2, num_key_value_heads=2, q_lora_rank=7,
             kv_lora_rank=5, qk_nope_head_dim=3, qk_rope_head_dim=4,
             v_head_dim=4, index_n_heads=3, index_head_dim=6, index_topk=2,
             intermediate_size=17, moe_intermediate_size=8,
             n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
             routed_scaling_factor=2.5, rms_norm_eps=1e-5,
             indexer_types=['full', 'shared', 'full', 'shared'],
             mlp_layer_types=['dense', 'sparse', 'sparse', 'sparse'],
             rope_parameters={'rope_type': 'default', 'rope_theta': 10000},
             norm_topk_prob=True, n_group=1, topk_group=1,
             max_position_embeddings=64)
    rng = np.random.default_rng(seed)
    w = {}
    def matrix(name, shape):
        w[name + '.weight'] = rng.normal(0, .18, shape).astype(np.float32)
    def norm(name, dim, bias=False):
        w[name + '.weight'] = rng.uniform(.7, 1.3, dim).astype(np.float32)
        if bias:
            w[name + '.bias'] = rng.normal(0, .1, dim).astype(np.float32)
    def mlp(name, dim):
        matrix(name + '.gate_proj', (dim, 12))
        matrix(name + '.up_proj', (dim, 12))
        matrix(name + '.down_proj', (12, dim))
    matrix('model.embed_tokens', (23, 12))
    matrix('lm_head', (23, 12))
    norm('model.norm', 12)
    for i in range(4):
        p = f'model.layers.{i}'
        norm(p + '.input_layernorm', 12)
        norm(p + '.post_attention_layernorm', 12)
        a = p + '.self_attn'
        matrix(a + '.q_a_proj', (7, 12))
        norm(a + '.q_a_layernorm', 7)
        matrix(a + '.q_b_proj', (14, 7))
        matrix(a + '.kv_a_proj_with_mqa', (9, 12))
        norm(a + '.kv_a_layernorm', 5)
        matrix(a + '.kv_b_proj', (14, 5))
        matrix(a + '.o_proj', (12, 8))
        if c['indexer_types'][i] == 'full':
            matrix(a + '.indexer.wq_b', (18, 7))
            matrix(a + '.indexer.wk', (6, 12))
            norm(a + '.indexer.k_norm', 6, True)
            matrix(a + '.indexer.weights_proj', (3, 12))
        m = p + '.mlp'
        if i == 0:
            mlp(m, 17)
        else:
            matrix(m + '.gate', (4, 12))
            w[m + '.gate.e_score_correction_bias'] = rng.normal(0, .2, 4).astype(np.float32)
            mlp(m + '.shared_experts', 8)
            for e in range(4):
                mlp(m + f'.experts.{e}', 8)
    return c, w


def expanded_batch_oracle(c, w, tokens):
    """Independent eager equations with expanded per-head KV and causal masks.

    Derived from the official GLM-MoE-DSA forward definitions; this is not an
    execution of Transformers or a claim of official-checkpoint validation.
    """
    n = len(tokens)
    h, d, r, v = [c[k] for k in ('num_attention_heads', 'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim')]
    def linear(p, x):
        return x @ w[p + '.weight'].T
    def norm(p, x, eps):
        return x * (1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * w[p + '.weight']
    def rotate(x):
        # x is [sequence, heads, rope_dim]. Independent vectorized RoPE.
        a = np.arange(n, dtype=np.float32)[:, None, None] * c['rope_parameters']['rope_theta'] ** (-np.arange(0, r, 2, dtype=np.float32) / r)
        return np.concatenate((x[..., ::2] * np.cos(a) - x[..., 1::2] * np.sin(a),
                               x[..., 1::2] * np.cos(a) + x[..., ::2] * np.sin(a)), axis=-1)
    def mlp(p, x):
        gate = linear(p + '.gate_proj', x)
        return linear(p + '.down_proj', gate / (1 + np.exp(-gate)) * linear(p + '.up_proj', x))
    x = w['model.embed_tokens.weight'][tokens].copy()
    selections = None
    all_selections = []
    for layer in range(c['num_hidden_layers']):
        p = f'model.layers.{layer}'
        z = norm(p + '.input_layernorm', x, c['rms_norm_eps'])
        a = p + '.self_attn'
        qr = norm(a + '.q_a_layernorm', linear(a + '.q_a_proj', z), 1e-6)
        q = linear(a + '.q_b_proj', qr).reshape(n, h, d + r)
        q = np.concatenate((q[..., :d], rotate(q[..., d:])), axis=-1)
        ka = linear(a + '.kv_a_proj_with_mqa', z)
        latent = norm(a + '.kv_a_layernorm', ka[:, :c['kv_lora_rank']], 1e-6)
        expanded = linear(a + '.kv_b_proj', latent).reshape(n, h, d + v)
        kr = rotate(ka[:, None, c['kv_lora_rank']:])
        keys = np.concatenate((expanded[..., :d], np.broadcast_to(kr, (n, h, r))), axis=-1)
        values = expanded[..., d:]
        if c['indexer_types'][layer] == 'full':
            ip = a + '.indexer'
            iq = linear(ip + '.wq_b', qr).reshape(n, c['index_n_heads'], c['index_head_dim'])
            iq = np.concatenate((rotate(iq[..., :r]), iq[..., r:]), axis=-1)
            ik = linear(ip + '.wk', z)
            ik = ik - ik.mean(-1, keepdims=True)
            ik = norm(ip + '.k_norm', ik, 1e-6) + w[ip + '.k_norm.bias']
            ik = np.concatenate((rotate(ik[:, None, :r])[:, 0], ik[:, r:]), axis=-1)
            head_scores = np.maximum(np.einsum('shd,td->sht', iq, ik) / np.sqrt(c['index_head_dim']), 0)
            weights = linear(ip + '.weights_proj', z) / np.sqrt(c['index_n_heads'])
            scores = np.einsum('sh,sht->st', weights, head_scores)
            selections = [np.argsort(-scores[t, :t+1], kind='stable')[:min(t+1, c['index_topk'])] for t in range(n)]
        all_selections.append([s.copy() for s in selections])
        out = np.zeros((n, h, v), np.float32)
        for t in range(n):
            ids = selections[t]
            scores = np.einsum('hd,khd->hk', q[t], keys[ids]) / np.sqrt(d + r)
            prob = np.exp(scores - scores.max(-1, keepdims=True))
            prob /= prob.sum(-1, keepdims=True)
            out[t] = np.einsum('hk,khv->hv', prob, values[ids])
        x = x + linear(a + '.o_proj', out.reshape(n, h*v))
        z = norm(p + '.post_attention_layernorm', x, c['rms_norm_eps'])
        m = p + '.mlp'
        if c['mlp_layer_types'][layer] == 'dense':
            x = x + mlp(m, z)
        else:
            probabilities = 1 / (1 + np.exp(-linear(m + '.gate', z)))
            ids = np.argsort(-(probabilities + w[m + '.gate.e_score_correction_bias']), axis=-1, kind='stable')[:, :c['num_experts_per_tok']]
            y = mlp(m + '.shared_experts', z)
            for t in range(n):
                p_selected = probabilities[t, ids[t]]
                p_selected = p_selected / p_selected.sum() * c['routed_scaling_factor']
                for expert, weight in zip(ids[t], p_selected):
                    y[t] += mlp(m + f'.experts.{expert}', z[t]) * weight
            x += y
    logits = linear('lm_head', norm('model.norm', x, c['rms_norm_eps']))
    return logits, all_selections


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config, self.weights = tiny_fixture()
        self.model = GlmMoeDsaModel(self.config, self.weights, NumpyTestBackend())

    def test_absorbed_decode_matches_expanded_causal_batch(self):
        tokens = [1, 7, 2, 11, 4, 3, 16]
        expected, selected = expanded_batch_oracle(self.config, self.weights, tokens)
        cache = self.model.new_cache()
        got = self.model.prefill(tokens, cache)
        np.testing.assert_allclose(got, expected, rtol=3e-5, atol=3e-6)
        for i, state in enumerate(cache.layers):
            np.testing.assert_array_equal(state.selected_indices, selected[i][-1])
            self.assertEqual(np.shape(state.latents), (len(tokens), self.config['kv_lora_rank']))
            self.assertEqual(np.shape(state.rope_keys), (len(tokens), self.config['qk_rope_head_dim']))
            self.assertEqual(len(state.index_keys), len(tokens) if i in (0, 2) else 0)
        self.assertEqual(cache.position, len(tokens))

    def test_compiled_vx_model_matches_independent_expanded_oracle(self):
        library = Path(__file__).resolve().parents[1] / 'kernels' / 'build' / 'libglm_vx.so'
        if not library.exists():
            self.skipTest('Build kernels/build/libglm_vx.so to run native Vx integration')
        from kernels.backend import VxBackend
        called = set()
        class TracedVx(VxBackend):
            def _call(self, name, *args):
                called.add(name)
                return super()._call(name, *args)
        model = GlmMoeDsaModel(self.config, self.weights, TracedVx(library))
        tokens = [1, 7, 2, 11, 4, 3, 16]
        expected, selections = expanded_batch_oracle(self.config, self.weights, tokens)
        cache = model.new_cache()
        prefill = model.prefill(tokens[:4], cache)
        decode = np.stack([model.forward(token, cache) for token in tokens[4:]])
        np.testing.assert_allclose(np.concatenate((prefill, decode)), expected, rtol=3e-5, atol=3e-6)
        for layer, state in enumerate(cache.layers):
            np.testing.assert_array_equal(state.selected_indices, selections[layer][-1])
        self.assertTrue({'layernorm', 'index_scores', 'topk'}.issubset(called))

    def test_prefill_split_decode_and_independent_requests(self):
        tokens = [2, 8, 19, 4, 3, 12]
        expected, _ = expanded_batch_oracle(self.config, self.weights, tokens)
        cache = self.model.new_cache()
        first = self.model.prefill(tokens[:3], cache)
        other = self.model.new_cache()
        self.model.prefill([21, 20], other)
        tail = np.stack([self.model.forward(t, cache) for t in tokens[3:]])
        np.testing.assert_allclose(np.concatenate((first, tail)), expected, rtol=3e-5, atol=3e-6)
        self.assertEqual(other.position, 2)
        self.assertIsNot(other.layers[0].latents, cache.layers[0].latents)

    def test_sparse_selection_changes_attention_and_shares(self):
        tokens = [1, 7, 2, 11, 4, 3, 16]
        sparse = self.model.prefill(tokens)
        dense_config = dict(self.config, index_topk=100)
        dense = GlmMoeDsaModel(dense_config, self.weights, NumpyTestBackend()).prefill(tokens)
        np.testing.assert_allclose(sparse[:2], dense[:2], atol=2e-6)
        self.assertGreater(float(np.max(np.abs(sparse[2:] - dense[2:]))), .01)
        cache = self.model.new_cache()
        self.model.prefill(tokens, cache)
        for a, b in ((0, 1), (2, 3)):
            np.testing.assert_array_equal(cache.layers[a].selected_indices, cache.layers[b].selected_indices)
        self.assertEqual(len(cache.layers[0].selected_indices), 2)

    def test_failure_rolls_back_cache(self):
        cache = self.model.new_cache()
        self.model.forward(2, cache)
        name = 'model.layers.2.mlp.shared_experts.down_proj.weight'
        missing = self.weights.pop(name)
        with self.assertRaises(KeyError):
            self.model.forward(3, cache)
        self.assertEqual(cache.position, 1)
        self.assertTrue(all(len(layer.latents) == 1 for layer in cache.layers))
        self.weights[name] = missing
        actual = self.model.forward(3, cache)
        expected = self.model.prefill([2, 3])[-1]
        np.testing.assert_array_equal(actual, expected)

    def test_callable_weights_and_packed_experts(self):
        packed = dict(self.weights)
        for layer in (1, 2, 3):
            p = f'model.layers.{layer}.mlp.experts'
            gate_up, down = [], []
            for expert in range(4):
                gate_up.append(np.concatenate((packed.pop(f'{p}.{expert}.gate_proj.weight'), packed.pop(f'{p}.{expert}.up_proj.weight'))))
                down.append(packed.pop(f'{p}.{expert}.down_proj.weight'))
            packed[p + '.gate_up_proj'] = np.stack(gate_up)
            packed[p + '.down_proj'] = np.stack(down)
        m = GlmMoeDsaModel(self.config, packed.__getitem__, NumpyTestBackend())
        np.testing.assert_allclose(m.prefill([1, 2, 3]), self.model.prefill([1, 2, 3]), atol=2e-6)

    def test_checkpoint_embedding_reads_only_requested_row(self):
        weights = self.weights
        class RowReader:
            def __init__(self):
                self.reads = []
            def __getitem__(self, name):
                if name == 'model.embed_tokens.weight':
                    raise AssertionError('Full embedding table must not be decoded')
                return weights[name]
            def tensor(self, name, *, rows=None):
                self.reads.append((name, rows.start, rows.stop))
                return weights[name][rows]
        reader = RowReader()
        model = GlmMoeDsaModel(self.config, reader, NumpyTestBackend())
        tokens = [3, 7, 2]
        np.testing.assert_array_equal(model.prefill(tokens), self.model.prefill(tokens))
        self.assertEqual(reader.reads, [('model.embed_tokens.weight', t, t+1) for t in tokens])

    def test_router_bias_changes_selection_not_mixture_probabilities(self):
        p = 'model.layers.1.mlp'
        logits = np.array([-2.0, 2.0, 0.0, 1.0], dtype=np.float32)
        gate = np.zeros_like(self.weights[p + '.gate.weight'])
        gate[:, 0] = logits
        self.weights[p + '.gate.weight'] = gate
        self.weights[p + '.gate.e_score_correction_bias'] = np.array([3., -3., 0., 0.], np.float32)
        self.weights[p + '.shared_experts.down_proj.weight'].fill(0)
        # Give expert e the constant output e+1 for this input. The model must
        # select experts 0 and 3 despite uncorrected logits preferring 1 and 3.
        for expert in range(4):
            ep = p + f'.experts.{expert}'
            g = self.weights[ep + '.gate_proj.weight']
            u = self.weights[ep + '.up_proj.weight']
            d = self.weights[ep + '.down_proj.weight']
            g.fill(0); u.fill(0); d.fill(0)
            g[0, 0] = 2.0
            u[0, 0] = 1.0
            d[:, 0] = (expert + 1) / (2.0 / (1 + np.exp(-2.0)))
        x = np.zeros(self.config['hidden_size'], np.float32)
        x[0] = 1.0
        actual = self.model._feed_forward(1, x)
        probabilities = 1 / (1 + np.exp(-logits[[0, 3]]))
        expected = self.config['routed_scaling_factor'] * (probabilities[0] + 4 * probabilities[1]) / probabilities.sum()
        np.testing.assert_allclose(actual, expected, atol=2e-6)
        # An erroneous bias-adjusted probability blend differs substantially.
        biased = probabilities + np.array([3.0, 0.0])
        wrong = self.config['routed_scaling_factor'] * (biased[0] + 4 * biased[1]) / biased.sum()
        self.assertGreater(abs(float(actual[0]) - wrong), 1.0)

    def test_tiny_weights_determinism_and_architecture(self):
        c = dict(self.config, indexer_types=['full', 'full', 'full', 'shared'])
        weights = tiny_weights(c, seed=71)
        again = tiny_weights(c, seed=71)
        self.assertEqual(set(weights), set(again))
        for name in weights:
            np.testing.assert_array_equal(weights[name], again[name])
            self.assertEqual(weights[name].dtype, np.float32)
        self.assertNotIn('model.layers.3.self_attn.indexer.wk.weight', weights)
        self.assertIn('model.layers.1.self_attn.indexer.k_norm.bias', weights)
        self.assertIn('model.layers.1.mlp.experts.3.gate_proj.weight', weights)
        model = GlmMoeDsaModel(c, weights, NumpyTestBackend())
        expected, _ = expanded_batch_oracle(c, weights, [1, 2, 3, 4, 5])
        np.testing.assert_allclose(model.prefill([1, 2, 3, 4, 5]), expected, atol=2e-6, rtol=3e-5)
        with self.assertRaisesRegex(ValueError, 'limited to'):
            tiny_weights(dict(c, hidden_size=1_000_000))

    def test_tiny_weights_attention_bias_and_tied_embedding(self):
        c = dict(self.config, attention_bias=True, tie_word_embeddings=True)
        w = tiny_weights(c, seed=9)
        self.assertNotIn('lm_head.weight', w)
        self.assertEqual(w['model.layers.0.self_attn.q_a_proj.bias'].shape, (7,))
        logits = GlmMoeDsaModel(c, w, NumpyTestBackend()).prefill([1, 2, 3])
        self.assertEqual(logits.shape, (3, c['vocab_size']))
        self.assertTrue(np.isfinite(logits).all())

    def test_config_schedule_and_validation(self):
        c = dict(self.config, num_hidden_layers=7, indexer_types=None,
                 mlp_layer_types=None, first_k_dense_replace=3,
                 index_topk_freq=4, index_skip_topk_offset=3)
        m = GlmMoeDsaModel(c, self.weights, NumpyTestBackend())
        self.assertEqual(m.indexer_types, ['full', 'full', 'full', 'shared', 'shared', 'shared', 'full'])
        self.assertEqual(m.mlp_types, ['dense']*3 + ['sparse']*4)
        for update in ({'n_group': 2}, {'indexer_types': ['shared']*4}, {'rope_parameters': {'rope_type': 'yarn'}}):
            with self.assertRaises(ValueError):
                GlmMoeDsaModel(dict(self.config, **update), self.weights, NumpyTestBackend())
        with self.assertRaises(ValueError):
            self.model.forward(-1, self.model.new_cache())


if __name__ == '__main__':
    unittest.main()
