"""Independent expanded-KV F32 equations, extracted from the existing test oracle.
Not the production compressed model graph; not an execution of Transformers.
Supported scope: default-RoPE, bias-free GLM, single-group routing, shared experts.
"""
import numpy as np

def expanded_batch_oracle(c, w, tokens, trace=None):
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
        if trace is not None:
            for pos in range(n): trace(pos, f'layer.{layer}.output', x[pos].copy())
    logits = linear('lm_head', norm('model.norm', x, c['rms_norm_eps']))
    return logits, all_selections
