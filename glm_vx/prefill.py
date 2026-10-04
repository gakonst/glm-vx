"""Layer-wise CPU prefill with batched projections and grouped expert work.

This owns only one bounded chunk; the scheduler chooses chunk boundaries. KV is
still compressed, and each attention query sees only its causal prefix. Native
Vx linear_batch preserves the scalar reduction order; other backends may expose
the same contract or use explicit per-row matvec calls. This is not GPU batching.
"""
import numpy as np
from .model import RequestCache

MAX_PREFILL_CHUNK = 64


def _linear(model, prefix, x, bias=False):
    if model.packed_weights and len(x) == 1:
        # Retain packed matvec for singleton chunks and final vocabulary logits.
        return model._linear(prefix, x[0], bias)[None, :]
    w = model._weight(prefix + '.weight')
    batch = getattr(model.backend, 'linear_batch', None)
    y = batch(w, x) if callable(batch) else np.stack([model.backend.matvec(w, row) for row in x])
    if bias:
        y = y + model._weight(prefix + '.bias')
    return y


def _norm(model, prefix, x, eps):
    # Explicit row calls preserve compatibility with the six-operation backend.
    w = model._weight(prefix + '.weight')
    return np.stack([model.backend.rmsnorm(row, w, eps) for row in x])


def _mlp(model, prefix, x):
    gate = _linear(model, prefix + '.gate_proj', x)
    up = _linear(model, prefix + '.up_proj', x)
    activated = np.stack([model.backend.swiglu(g, u) for g, u in zip(gate, up)])
    return _linear(model, prefix + '.down_proj', activated)


def _feed_forward(model, layer, x):
    prefix = f'model.layers.{layer}.mlp'
    if model.mlp_types[layer] == 'dense':
        return _mlp(model, prefix, x)
    c, b = model.config, model.backend
    router_logits = _linear(model, prefix + '.gate', x)
    bias = model._weight(prefix + '.gate.e_score_correction_bias')
    count = int(c['num_experts_per_tok'])
    routed = [b.route(row, bias, count, float(c.get('routed_scaling_factor', 1.0))) for row in router_logits]
    ids = np.stack([r[0] for r in routed])
    weights = np.stack([r[1] for r in routed])
    # Group compute by expert and accumulate in ascending expert ID, matching
    # the pinned official eager reduction. Shared experts are added last.
    outputs = np.empty((len(x), count, model.hidden), dtype=np.float32)
    for expert in np.unique(ids):
        rows, slots = np.nonzero(ids == expert)
        expert_prefix = prefix + f'.experts.{int(expert)}'
        if model.packed_weights:
            # The GGUF manifest already validates named expert projections.
            # An existence probe through _weight would itself expand the gate.
            values = (model._expert(prefix, int(expert), x[rows[0]])[None, :]
                      if len(rows) == 1 else _mlp(model, expert_prefix, x[rows]))
        else:
            try:
                model._weight(expert_prefix + '.gate_proj.weight')
            except KeyError:
                # Preserve the alternate in-memory stacked expert representation.
                values = np.stack([model._expert(prefix, int(expert), x[row]) for row in rows])
            else:
                values = _mlp(model, expert_prefix, x[rows])
        outputs[rows, slots] = values
    result = np.zeros_like(x)
    for row in range(len(x)):
        for slot in np.argsort(ids[row], kind='stable'):
            result[row] += outputs[row, slot] * float(weights[row, slot])
    if int(c.get('n_shared_experts', 1)) > 0:
        result += _mlp(model, prefix + '.shared_experts', x)
    return result


def _attention(model, layer, x, start, state, previous):
    b, c = model.backend, model.config
    prefix = f'model.layers.{layer}.self_attn'
    bias = bool(c.get('attention_bias', False))
    qr = _norm(model, prefix + '.q_a_layernorm', _linear(model, prefix + '.q_a_proj', x, bias), 1e-6)
    queries = _linear(model, prefix + '.q_b_proj', qr, False).reshape(len(x), model.heads, model.nope + model.rot)
    compressed = _linear(model, prefix + '.kv_a_proj_with_mqa', x, bias)
    latent = _norm(model, prefix + '.kv_a_layernorm', compressed[:, :model.rank], 1e-6)
    kv = model._weight(prefix + '.kv_b_proj.weight').reshape(model.heads, model.nope + model.value_dim, model.rank)
    values, selections = [], []
    for offset in range(len(x)):
        pos = start + offset
        state.latents.append(np.asarray(latent[offset], dtype=np.float32).copy())
        state.rope_keys.append(np.asarray(b.rope(compressed[offset, model.rank:], pos, model.theta), dtype=np.float32).copy())
        if model.indexer_types[layer] == 'full':
            selected = model._index(prefix + '.indexer', x[offset], qr[offset], pos, state)
        else:
            selected = previous[offset]
        selections.append(selected.copy())
        state.selected_indices = selected.copy()
        latents = np.stack([state.latents[int(i)] for i in selected])
        rotary_keys = np.stack([state.rope_keys[int(i)] for i in selected])
        query = queries[offset]
        fused = getattr(b, 'compressed_attention', None)
        if callable(fused):
            latent_q = np.stack([b.matvec(kv[h, :model.nope].T, query[h, :model.nope]) for h in range(model.heads)])
            rotary_q = np.stack([b.rope(query[h, model.nope:], pos, model.theta) for h in range(model.heads)])
            attended = fused(latent_q, rotary_q, latents, rotary_keys, (model.nope + model.rot) ** -0.5)
            out = [b.matvec(kv[h, model.nope:], attended[h]) for h in range(model.heads)]
        else:
            out = []
            for h in range(model.heads):
                lq = b.matvec(kv[h, :model.nope].T, query[h, :model.nope])
                rq = b.rope(query[h, model.nope:], pos, model.theta)
                scores = b.matvec(latents, lq) + b.matvec(rotary_keys, rq)
                probabilities = b.softmax(scores * (model.nope + model.rot) ** -0.5)
                attended = b.matvec(latents.T, probabilities)
                out.append(b.matvec(kv[h, model.nope:], attended))
        values.append(np.concatenate(out))
    return _linear(model, prefix + '.o_proj', np.stack(values), bias), selections


def prefill(model, tokens, cache, *, output_all_logits=False, output_logits=True):
    """Append 1..64 tokens atomically and return final (or every) next logits.

    A failed chunk restores cache lengths and selections. Trace callbacks use
    the single-token executor until stage-complete batched tracing is supported.
    This prevents an apparently complete but incomplete trace from being emitted.
    """
    if type(output_logits) is not bool or type(output_all_logits) is not bool:
        raise ValueError('logit options must be boolean')
    if output_all_logits and not output_logits:
        raise ValueError('all logits requires output_logits')
    if model.trace is not None:
        raise ValueError('batched prefill does not yet support validation trace callbacks')
    if not isinstance(tokens, (list, tuple)) or not 0 < len(tokens) <= MAX_PREFILL_CHUNK:
        raise ValueError(f'prefill requires 1..{MAX_PREFILL_CHUNK} token IDs')
    if any(not isinstance(t, (int, np.integer)) or isinstance(t, (bool, np.bool_)) or
           not 0 <= int(t) < int(model.config['vocab_size']) for t in tokens):
        raise ValueError('invalid prefill token ID')
    if not isinstance(cache, RequestCache) or len(cache.layers) != model.n_layers:
        raise ValueError('Use a cache returned by model.new_cache()')
    start = cache.position
    if start + len(tokens) > int(model.config['max_position_embeddings']):
        raise ValueError('prefill exceeds context limit')
    for i, state in enumerate(cache.layers):
        expected_index = start if model.indexer_types[i] == 'full' else 0
        if len(state.latents) != start or len(state.rope_keys) != start or len(state.index_keys) != expected_index:
            raise ValueError('Inconsistent request cache lengths')
    old = [state.selected_indices for state in cache.layers]
    try:
        x = np.stack([model._embedding(int(token)) for token in tokens])
        selected = None
        for layer, state in enumerate(cache.layers):
            prefix = f'model.layers.{layer}'
            attention, selected = _attention(model, layer, _norm(model, prefix + '.input_layernorm', x, model.eps), start, state, selected)
            x = x + attention
            x = x + _feed_forward(model, layer, _norm(model, prefix + '.post_attention_layernorm', x, model.eps))
        if not output_logits:
            cache.position += len(tokens)
            return None
        final = x if output_all_logits else x[-1:]
        head = 'model.embed_tokens' if model.config.get('tie_word_embeddings', False) else 'lm_head'
        logits = _linear(model, head, _norm(model, 'model.norm', final, model.eps))
        cache.position += len(tokens)
        return logits if output_all_logits else logits[0]
    except BaseException:
        for i, state in enumerate(cache.layers):
            del state.latents[start:]
            del state.rope_keys[start:]
            del state.index_keys[start if model.indexer_types[i] == 'full' else 0:]
            state.selected_indices = old[i]
        cache.position = start
        raise
