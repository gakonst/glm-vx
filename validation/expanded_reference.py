"""Independent expanded-KV equations, not a Transformers execution.

f32-v2 uses explicit F32 scales and official eager expert-ID reduction order.
legacy-v1 preserves the original mixed F32/F64, shared-first oracle so existing
frozen traces can be diagnosed without silently replacing their numerical mode.
"""
import numpy as np


def expanded_batch_oracle(c, w, tokens, trace=None, *, numerical_mode='f32-v2',
                          start_layer=0, stop_layer=None, hidden_states=None,
                          initial_selections=None, output_logits=True):
    """Causal batch equations; optional captured layer inputs enable bounded replay.

    All positions 0..len(tokens)-1 must be present, including for layer replay.
    No compressed attention or production backend code is called here.
    """
    if numerical_mode not in ('f32-v2', 'legacy-v1'):
        raise ValueError('unknown reference numerical mode')
    legacy = numerical_mode == 'legacy-v1'
    if c.get('attention_bias', False):
        raise ValueError('expanded reference requires bias-free attention')
    n = len(tokens)
    h, d, r, v = [c[k] for k in ('num_attention_heads', 'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim')]

    def emit(pos, name, value):
        if trace is not None:
            trace(pos, name, np.array(value, copy=True))

    def batch(name, values):
        if trace is not None:
            for pos in range(n):
                emit(pos, name, values[pos])

    def linear(p, x):
        return x @ w[p + '.weight'].T

    def norm(p, x, eps):
        return x * (1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * w[p + '.weight']

    def scale(x, dim):
        # np.sqrt(int) is np.float64; NumPy 2 promotes the entire F32 array on
        # division by that scalar. This was a hidden precision change in v1.
        return x / np.sqrt(dim) if legacy else x * np.float32(dim ** -0.5)

    def rotate(x):
        a = np.arange(n, dtype=np.float32)[:, None, None] * c['rope_parameters']['rope_theta'] ** (-np.arange(0, r, 2, dtype=np.float32) / r)
        return np.concatenate((x[..., ::2] * np.cos(a) - x[..., 1::2] * np.sin(a),
                               x[..., 1::2] * np.cos(a) + x[..., ::2] * np.sin(a)), axis=-1)

    def mlp(p, x, pos=None):
        stage = p.replace('model.layers.', 'layer.').replace('.shared_experts', '.shared').replace('.experts.', '.expert.')
        gate = linear(p + '.gate_proj', x)
        up = linear(p + '.up_proj', x)
        activation = gate / (1 + np.exp(-gate)) * up
        for name, value in (('gate', gate), ('up', up), ('activation', activation)):
            if pos is None:
                batch(stage + '.' + name, value)
            else:
                emit(pos, stage + '.' + name, value)
        return linear(p + '.down_proj', activation)

    x = w['model.embed_tokens.weight'][tokens].copy() if hidden_states is None else np.array(hidden_states, dtype=np.float32, copy=True)
    if x.shape != (n, c['hidden_size']):
        raise ValueError('hidden_states must include every causal position')
    selections = initial_selections
    all_selections = []
    for layer in range(start_layer, c['num_hidden_layers'] if stop_layer is None else stop_layer):
        p, stage = f'model.layers.{layer}', f'layer.{layer}'
        batch(stage + '.input', x)
        z = norm(p + '.input_layernorm', x, c['rms_norm_eps'])
        batch(stage + '.input_norm', z)
        a, astage = p + '.self_attn', stage + '.attention'
        qa = linear(a + '.q_a_proj', z)
        batch(astage + '.q_a', qa)
        qr = norm(a + '.q_a_layernorm', qa, 1e-6)
        batch(astage + '.q_norm', qr)
        q = linear(a + '.q_b_proj', qr).reshape(n, h, d + r)
        batch(astage + '.query_projection', q)
        q = np.concatenate((q[..., :d], rotate(q[..., d:])), axis=-1)
        ka = linear(a + '.kv_a_proj_with_mqa', z)
        batch(astage + '.kv_a', ka)
        latent = norm(a + '.kv_a_layernorm', ka[:, :c['kv_lora_rank']], 1e-6)
        batch(stage + '.latent', latent)
        expanded = linear(a + '.kv_b_proj', latent).reshape(n, h, d + v)
        kr = rotate(ka[:, None, c['kv_lora_rank']:])
        batch(stage + '.rope_key', kr[:, 0])
        keys = np.concatenate((expanded[..., :d], np.broadcast_to(kr, (n, h, r))), axis=-1)
        values = expanded[..., d:]
        if c['indexer_types'][layer] == 'full':
            ip, istage = a + '.indexer', stage + '.indexer'
            iq = linear(ip + '.wq_b', qr).reshape(n, c['index_n_heads'], c['index_head_dim'])
            batch(istage + '.query_projection', iq)
            iq = np.concatenate((rotate(iq[..., :r]), iq[..., r:]), axis=-1)
            ik = linear(ip + '.wk', z)
            batch(istage + '.key_projection', ik)
            ik = ik - ik.mean(-1, keepdims=True)
            ik = norm(ip + '.k_norm', ik, 1e-6) + w[ip + '.k_norm.bias']
            batch(istage + '.key_norm', ik)
            ik = np.concatenate((rotate(ik[:, None, :r])[:, 0], ik[:, r:]), axis=-1)
            head_scores = np.maximum(scale(np.einsum('shd,td->sht', iq, ik), c['index_head_dim']), 0)
            weights = scale(linear(ip + '.weights_proj', z), c['index_n_heads'])
            batch(istage + '.query', iq)
            batch(istage + '.key', ik)
            batch(istage + '.head_weights', weights)
            scores = np.einsum('sh,sht->st', weights, head_scores)
            selections = []
            for t in range(n):
                emit(t, istage + '.scores', scores[t, :t+1])
                selections.append(np.argsort(-scores[t, :t+1], kind='stable')[:min(t+1, c['index_topk'])])
        if selections is None:
            raise ValueError('shared-layer replay requires initial_selections')
        all_selections.append([s.copy() for s in selections])
        out = np.zeros((n, h, v), np.float32)
        for t in range(n):
            ids = selections[t]
            emit(t, stage + '.selected', ids)
            scores = scale(np.einsum('hd,khd->hk', q[t], keys[ids]), d + r)
            prob = np.exp(scores - scores.max(-1, keepdims=True))
            prob /= prob.sum(-1, keepdims=True)
            for head in range(h):
                emit(t, astage + f'.head.{head}.scores', scores[head])
                emit(t, astage + f'.head.{head}.probabilities', prob[head])
            out[t] = np.einsum('hk,khv->hv', prob, values[ids])
        batch(astage + '.heads', out)
        attention_output = linear(a + '.o_proj', out.reshape(n, h*v))
        batch(astage + '.output', attention_output)
        x = x + attention_output
        batch(stage + '.attention_residual', x)
        z = norm(p + '.post_attention_layernorm', x, c['rms_norm_eps'])
        batch(stage + '.post_attention_norm', z)
        m, mstage = p + '.mlp', stage + '.mlp'
        if c['mlp_layer_types'][layer] == 'dense':
            y = mlp(m, z)
        else:
            router_logits = linear(m + '.gate', z)
            batch(mstage + '.router_logits', router_logits)
            probabilities = 1 / (1 + np.exp(-router_logits))
            ids = np.argsort(-(probabilities + w[m + '.gate.e_score_correction_bias']), axis=-1, kind='stable')[:, :c['num_experts_per_tok']]
            shared = mlp(m + '.shared_experts', z) if c.get('n_shared_experts', 1) > 0 else np.zeros_like(z)
            y = shared.copy() if legacy else np.zeros_like(z)
            for t in range(n):
                p_selected = probabilities[t, ids[t]]
                p_selected = p_selected / (p_selected.sum() + (0 if legacy else 1e-20)) * c['routed_scaling_factor']
                emit(t, mstage + '.route_ids', ids[t])
                emit(t, mstage + '.route_weights', p_selected)
                order = np.arange(len(ids[t])) if legacy else np.argsort(ids[t], kind='stable')
                emit(t, mstage + '.aggregation_ids', ids[t, order])
                for slot in order:
                    expert, weight = ids[t, slot], p_selected[slot]
                    output = mlp(m + f'.experts.{expert}', z[t], pos=t)
                    weighted = output * weight
                    emit(t, mstage + f'.expert.{expert}.output', output)
                    emit(t, mstage + f'.expert.{expert}.weighted', weighted)
                    y[t] += weighted
                    emit(t, mstage + f'.expert.{expert}.partial_sum', y[t])
                # Legacy sums already include shared; do not label them routed.
                emit(t, mstage + ('.shared_first_sum' if legacy else '.routed_sum'), y[t])
            batch(mstage + '.shared_output', shared)
            if not legacy:
                y += shared
        batch(mstage + '.output', y)
        x += y
        batch(stage + '.output', x)
    if not output_logits:
        return x, all_selections
    head = 'model.embed_tokens' if c.get('tie_word_embeddings', False) else 'lm_head'
    logits = linear(head, norm('model.norm', x, c['rms_norm_eps']))
    batch('logits', logits)
    return logits, all_selections
