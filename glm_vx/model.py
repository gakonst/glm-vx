"""Single-request GLM-MoE-DSA inference, implemented without framework runtime.

The checkpoint's projections are [output, input]. All dot products, normalization,
RoPE, softmax, SwiGLU and expert routing run through the supplied backend. NumPy
is used for storage, indexing, scalar scaling and elementwise residual arithmetic.
The cache retains MLA latents, never expanded per-head keys or values. Decode
uses absorbed key projections and a weighted latent to avoid expanding old KV.
This float32/dequantized-weight path does not reproduce the official GPU FP8
dynamic-activation quantization or its exact rounding behavior.
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LayerCache:
    latents: list = field(default_factory=list)
    rope_keys: list = field(default_factory=list)
    index_keys: list = field(default_factory=list)
    selected_indices: Any = None


@dataclass
class RequestCache:
    layers: list
    position: int = 0

    @property
    def length(self):
        return self.position


class GlmMoeDsaModel:
    """A weights mapping (or name->tensor callable) and six-operation backend.

    Backend methods are matvec(w,x), rmsnorm(x,w,eps), rope(x,position,theta),
    softmax(x), swiglu(gate,up), route(logits,bias,k,scale). ``rope`` rotates
    adjacent input pairs; either interleaved or half-split output is valid,
    provided it uses the same layout for every call. Routing uses sigmoid,
    correction bias for selection only, and normalized selected probabilities.
    """

    def __init__(self, config, weights, backend):
        self.config = dict(config)
        self.weights, self.backend = weights, backend
        c = self.config
        self.n_layers = int(c['num_hidden_layers'])
        self.hidden = int(c['hidden_size'])
        self.heads = int(c['num_attention_heads'])
        self.rank = int(c['kv_lora_rank'])
        self.nope = int(c['qk_nope_head_dim'])
        self.rot = int(c['qk_rope_head_dim'])
        self.value_dim = int(c['v_head_dim'])
        self.index_heads = int(c['index_n_heads'])
        self.index_dim = int(c['index_head_dim'])
        self.index_topk = int(c['index_topk'])
        self.eps = float(c.get('rms_norm_eps', 1e-5))
        rope = c.get('rope_parameters') or {'rope_theta': c.get('rope_theta', 10000)}
        self.theta = float(rope.get('rope_theta', c.get('rope_theta', 10000)))
        if rope.get('rope_type', 'default') != 'default':
            raise ValueError('Only default RoPE is implemented; scaled/YaRN RoPE needs a matching backend')
        if not c.get('rope_interleave', True) or not c.get('indexer_rope_interleave', True):
            raise ValueError('GLM DSA requires interleaved-input RoPE')
        if c.get('n_group', 1) != 1 or c.get('topk_group', 1) != 1:
            raise ValueError('Backend route supports the official single-group router only')
        if not c.get('norm_topk_prob', True) or c.get('scoring_func', 'sigmoid') != 'sigmoid':
            raise ValueError('Backend route requires normalized sigmoid expert weights')
        if c.get('hidden_act', 'silu') != 'silu':
            raise ValueError('Only the checkpoint SwiGLU activation is supported')
        if self.rot <= 0 or self.rot % 2 or self.rot > self.index_dim:
            raise ValueError('RoPE dimension must be positive, even, and no larger than index_head_dim')
        if min(self.n_layers, self.hidden, self.heads, self.rank, self.index_topk) <= 0:
            raise ValueError('Model dimensions and index_topk must be positive')
        types = c.get('indexer_types')
        if types is None:
            pattern = c.get('index_topk_pattern')
            if pattern is not None:
                types = [{'F': 'full', 'S': 'shared'}[x] for x in pattern] if isinstance(pattern, str) else list(pattern)
            else:
                freq = max(int(c.get('index_topk_freq', 1)), 1)
                offset = int(c.get('index_skip_topk_offset', 2))
                types = ['full' if max(i - offset + 1, 0) % freq == 0 else 'shared' for i in range(self.n_layers)]
        self.indexer_types = list(types)
        dense = min(int(c.get('first_k_dense_replace', 1)), self.n_layers)
        self.mlp_types = list(c.get('mlp_layer_types') or (['dense'] * dense + ['sparse'] * (self.n_layers - dense)))
        if len(self.indexer_types) != self.n_layers or len(self.mlp_types) != self.n_layers:
            raise ValueError('Per-layer architecture arrays must match num_hidden_layers')
        if self.indexer_types[0] != 'full' or any(t not in ('full', 'shared') for t in self.indexer_types):
            raise ValueError('DSA layers must be full/shared and start with a full indexer')
        if any(t not in ('dense', 'sparse') for t in self.mlp_types):
            raise ValueError('MLP layer type must be dense or sparse')

    def _weight(self, name):
        return self.weights(name) if callable(self.weights) else self.weights[name]

    def _embedding(self, token_id):
        name = 'model.embed_tokens.weight'
        tensor_reader = getattr(self.weights, 'tensor', None)
        if callable(tensor_reader):
            # A full official embedding table is several GB; decode one row.
            row = tensor_reader(name, rows=slice(token_id, token_id + 1))[0]
        else:
            row = self._weight(name)[token_id]
        return np.asarray(row, dtype=np.float32).copy()

    def _linear(self, prefix, x, bias=False):
        y = self.backend.matvec(self._weight(prefix + '.weight'), x)
        if bias:
            y = y + self._weight(prefix + '.bias')
        return y

    def _norm(self, prefix, x, eps):
        return self.backend.rmsnorm(x, self._weight(prefix + '.weight'), eps)

    def new_cache(self):
        """Allocate independent state; model weights/backend can be shared by requests."""
        return RequestCache([LayerCache() for _ in range(self.n_layers)])

    def _index(self, prefix, x, q_residual, position, state):
        b = self.backend
        q = self._linear(prefix + '.wq_b', q_residual).reshape(self.index_heads, self.index_dim)
        key = self._linear(prefix + '.wk', x)
        # Prefer a fused native LayerNorm when exposed; the six-op backend
        # contract can express the same normalization as centered RMSNorm.
        layernorm = getattr(b, 'layernorm', None)
        if callable(layernorm):
            key = layernorm(key, self._weight(prefix + '.k_norm.weight'),
                            self._weight(prefix + '.k_norm.bias'), 1e-6)
        else:
            mean = b.matvec(np.full((1, self.index_dim), 1.0 / self.index_dim, dtype=np.float32), key)[0]
            key = self._norm(prefix + '.k_norm', key - mean, 1e-6) + self._weight(prefix + '.k_norm.bias')
        key = np.concatenate((b.rope(key[:self.rot], position, self.theta), key[self.rot:]))
        state.index_keys.append(np.asarray(key, dtype=np.float32).copy())
        keys = np.stack(state.index_keys)
        queries = np.stack([np.concatenate((b.rope(q[h, :self.rot], position, self.theta), q[h, self.rot:]))
                            for h in range(self.index_heads)])
        head_weights = self._linear(prefix + '.weights_proj', x) * self.index_heads ** -0.5
        index_scores = getattr(b, 'index_scores', None)
        if callable(index_scores):
            scores = index_scores(queries, keys, head_weights, self.index_dim ** -0.5)
        else:
            scores_by_head = [np.maximum(b.matvec(keys, query) * self.index_dim ** -0.5, 0.0)
                              for query in queries]
            scores = b.matvec(np.stack(scores_by_head).T, head_weights)
        count = min(self.index_topk, len(state.index_keys))
        # Stable tie breaking by position. Torch topk does not specify tie order;
        # untied scores agree, and all selected tokens remain causal.
        topk = getattr(b, 'topk', None)
        if callable(topk):
            return np.asarray(topk(scores, count), dtype=np.int64)
        return np.argsort(-scores, kind='stable')[:count].astype(np.int64)

    def _attention(self, layer, x, position, state, previous_selection):
        b, c = self.backend, self.config
        prefix = f'model.layers.{layer}.self_attn'
        has_bias = bool(c.get('attention_bias', False))
        q_residual = self._norm(prefix + '.q_a_layernorm', self._linear(prefix + '.q_a_proj', x, has_bias), 1e-6)
        query = self._linear(prefix + '.q_b_proj', q_residual).reshape(self.heads, self.nope + self.rot)
        compressed = self._linear(prefix + '.kv_a_proj_with_mqa', x, has_bias)
        latent = self._norm(prefix + '.kv_a_layernorm', compressed[:self.rank], 1e-6)
        rotary_key = b.rope(compressed[self.rank:], position, self.theta)
        state.latents.append(np.asarray(latent, dtype=np.float32).copy())
        state.rope_keys.append(np.asarray(rotary_key, dtype=np.float32).copy())
        if self.indexer_types[layer] == 'full':
            selected = self._index(prefix + '.indexer', x, q_residual, position, state)
        else:
            if previous_selection is None:
                raise ValueError('Shared DSA layer has no preceding full indexer selection')
            selected = previous_selection
        state.selected_indices = selected.copy()
        latents = np.stack([state.latents[int(i)] for i in selected])
        rotary_keys = np.stack([state.rope_keys[int(i)] for i in selected])
        kv_weight = self._weight(prefix + '.kv_b_proj.weight').reshape(self.heads, self.nope + self.value_dim, self.rank)
        outputs = []
        fused = getattr(b, 'compressed_attention', None)
        if callable(fused):
            latent_queries = np.stack([b.matvec(kv_weight[h, :self.nope].T, query[h, :self.nope])
                                       for h in range(self.heads)])
            rotary_queries = np.stack([b.rope(query[h, self.nope:], position, self.theta)
                                       for h in range(self.heads)])
            attended = fused(latent_queries, rotary_queries, latents, rotary_keys,
                             (self.nope + self.rot) ** -0.5)
            outputs = [b.matvec(kv_weight[h, self.nope:], attended[h]) for h in range(self.heads)]
            return self._linear(prefix + '.o_proj', np.concatenate(outputs), has_bias), selected
        for h in range(self.heads):
            # q_nope . (W_k c) == c . (W_k^T q_nope).
            latent_query = b.matvec(kv_weight[h, :self.nope].T, query[h, :self.nope])
            rotary_query = b.rope(query[h, self.nope:], position, self.theta)
            scores = b.matvec(latents, latent_query) + b.matvec(rotary_keys, rotary_query)
            probs = b.softmax(scores * (self.nope + self.rot) ** -0.5)
            attended_latent = b.matvec(latents.T, probs)
            outputs.append(b.matvec(kv_weight[h, self.nope:], attended_latent))
        return self._linear(prefix + '.o_proj', np.concatenate(outputs), has_bias), selected

    def _mlp(self, prefix, x):
        return self._linear(prefix + '.down_proj', self.backend.swiglu(
            self._linear(prefix + '.gate_proj', x), self._linear(prefix + '.up_proj', x)))

    def _expert(self, prefix, expert, x):
        # Native GLM checkpoint uses individual experts. Packed tensors support
        # the upstream runtime's in-memory representation too.
        try:
            gate = self._weight(f'{prefix}.experts.{expert}.gate_proj.weight')
        except KeyError:
            gate_up = self._weight(prefix + '.experts.gate_up_proj')[expert]
            projected = self.backend.matvec(gate_up, x)
            half = projected.shape[0] // 2
            return self.backend.matvec(self._weight(prefix + '.experts.down_proj')[expert],
                                       self.backend.swiglu(projected[:half], projected[half:]))
        up = self._weight(f'{prefix}.experts.{expert}.up_proj.weight')
        down = self._weight(f'{prefix}.experts.{expert}.down_proj.weight')
        return self.backend.matvec(down, self.backend.swiglu(self.backend.matvec(gate, x), self.backend.matvec(up, x)))

    def _feed_forward(self, layer, x):
        prefix = f'model.layers.{layer}.mlp'
        if self.mlp_types[layer] == 'dense':
            return self._mlp(prefix, x)
        c = self.config
        indices, weights = self.backend.route(self._linear(prefix + '.gate', x),
            self._weight(prefix + '.gate.e_score_correction_bias'),
            int(c['num_experts_per_tok']), float(c.get('routed_scaling_factor', 1.0)))
        result = np.zeros(self.hidden, dtype=np.float32)
        for expert, weight in zip(indices, weights):
            result += self._expert(prefix, int(expert), x) * float(weight)
        if int(c.get('n_shared_experts', 1)) > 0:
            result += self._mlp(prefix + '.shared_experts', x)
        return result

    def forward(self, token_id, cache):
        """Append one token at cache.position and return next-token logits."""
        if not isinstance(token_id, (int, np.integer)) or not 0 <= int(token_id) < int(self.config['vocab_size']):
            raise ValueError('token_id must be an integer in the vocabulary')
        if not isinstance(cache, RequestCache) or len(cache.layers) != self.n_layers:
            raise ValueError('Use a cache returned by this model.new_cache()')
        position = cache.position
        if position >= int(self.config.get('max_position_embeddings', 2**31 - 1)):
            raise ValueError('Request exceeds max_position_embeddings')
        for i, state in enumerate(cache.layers):
            expected_index = position if self.indexer_types[i] == 'full' else 0
            if len(state.latents) != position or len(state.rope_keys) != position or len(state.index_keys) != expected_index:
                raise ValueError('Inconsistent request cache lengths')
        old_selections = [state.selected_indices for state in cache.layers]
        try:
            x = self._embedding(int(token_id))
            selected = None
            for layer, state in enumerate(cache.layers):
                prefix = f'model.layers.{layer}'
                attended, selected = self._attention(layer, self._norm(prefix + '.input_layernorm', x, self.eps), position, state, selected)
                x = x + attended
                x = x + self._feed_forward(layer, self._norm(prefix + '.post_attention_layernorm', x, self.eps))
            x = self._norm('model.norm', x, self.eps)
            head = 'model.embed_tokens' if self.config.get('tie_word_embeddings', False) else 'lm_head'
            logits = self._linear(head, x)
        except Exception:
            # A missing shard/backend failure must not poison the request cache.
            for state, old_selection in zip(cache.layers, old_selections):
                del state.latents[position:]
                del state.rope_keys[position:]
                del state.index_keys[position:]
                state.selected_indices = old_selection
            raise
        cache.position += 1
        return logits

    def prefill(self, token_ids, cache=None):
        """Causal sequential prefill, returning logits for every input token.

        This favors bounded memory and exact decode semantics over a batched
        prefill kernel. Pass an explicit cache to continue decoding afterward.
        """
        if cache is None:
            cache = self.new_cache()
        results = [self.forward(token, cache) for token in token_ids]
        return np.stack(results) if results else np.empty((0, int(self.config['vocab_size'])), dtype=np.float32)


Model = GlmMoeDsaModel
GLMModel = GlmMoeDsaModel


def tiny_weights(config, seed=0):
    """Generate deterministic float32 test tensors with native checkpoint names.

    These are random, untrained weights for architecture/backend smoke tests.
    A ten-million-parameter cap prevents accidentally allocating the official
    hundreds-of-billions-parameter checkpoint through this demo helper.
    """
    model = GlmMoeDsaModel(config, {}, None)
    c = model.config
    shapes = {}
    norms = set()
    biases = set()

    def matrix(name, rows, cols, bias=False):
        shapes[name + '.weight'] = (int(rows), int(cols))
        if bias:
            shapes[name + '.bias'] = (int(rows),)
            biases.add(name + '.bias')

    def norm(name, width, bias=False):
        shapes[name + '.weight'] = (int(width),)
        norms.add(name + '.weight')
        if bias:
            shapes[name + '.bias'] = (int(width),)
            biases.add(name + '.bias')

    def mlp(name, width):
        matrix(name + '.gate_proj', width, model.hidden)
        matrix(name + '.up_proj', width, model.hidden)
        matrix(name + '.down_proj', model.hidden, width)

    matrix('model.embed_tokens', c['vocab_size'], model.hidden)
    if not c.get('tie_word_embeddings', False):
        matrix('lm_head', c['vocab_size'], model.hidden)
    norm('model.norm', model.hidden)
    for layer in range(model.n_layers):
        prefix = f'model.layers.{layer}'
        norm(prefix + '.input_layernorm', model.hidden)
        norm(prefix + '.post_attention_layernorm', model.hidden)
        a = prefix + '.self_attn'
        attention_bias = bool(c.get('attention_bias', False))
        matrix(a + '.q_a_proj', c['q_lora_rank'], model.hidden, attention_bias)
        norm(a + '.q_a_layernorm', c['q_lora_rank'])
        matrix(a + '.q_b_proj', model.heads * (model.nope + model.rot), c['q_lora_rank'])
        matrix(a + '.kv_a_proj_with_mqa', model.rank + model.rot, model.hidden, attention_bias)
        norm(a + '.kv_a_layernorm', model.rank)
        matrix(a + '.kv_b_proj', model.heads * (model.nope + model.value_dim), model.rank)
        matrix(a + '.o_proj', model.hidden, model.heads * model.value_dim, attention_bias)
        if model.indexer_types[layer] == 'full':
            matrix(a + '.indexer.wq_b', model.index_heads * model.index_dim, c['q_lora_rank'])
            matrix(a + '.indexer.wk', model.index_dim, model.hidden)
            norm(a + '.indexer.k_norm', model.index_dim, True)
            matrix(a + '.indexer.weights_proj', model.index_heads, model.hidden)
        m = prefix + '.mlp'
        if model.mlp_types[layer] == 'dense':
            mlp(m, c['intermediate_size'])
        else:
            experts = int(c['n_routed_experts'])
            matrix(m + '.gate', experts, model.hidden)
            name = m + '.gate.e_score_correction_bias'
            shapes[name] = (experts,)
            biases.add(name)
            for expert in range(experts):
                mlp(m + f'.experts.{expert}', c['moe_intermediate_size'])
            shared = int(c.get('n_shared_experts', 1))
            if shared:
                mlp(m + '.shared_experts', int(c['moe_intermediate_size']) * shared)
    parameters = sum(int(np.prod(shape)) for shape in shapes.values())
    if parameters > 10_000_000:
        raise ValueError(f'tiny_weights is limited to 10,000,000 parameters; config requests {parameters:,}')
    rng = np.random.default_rng(seed)
    std = float(c.get('initializer_range', 0.02))
    return {name: (np.ones(shape, dtype=np.float32) if name in norms else
                   np.zeros(shape, dtype=np.float32) if name in biases else
                   rng.normal(0.0, std, shape).astype(np.float32))
            for name, shape in shapes.items()}
