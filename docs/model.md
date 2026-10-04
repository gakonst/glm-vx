# GLM-5.3 checkpoint and architecture evidence

This implementation reads model metadata and weight bytes without importing Transformers, PyTorch, Hugging Face remote code, or safetensors. Numerical equivalence to a full GLM-5.3 inference has **not** been established by these metadata and synthetic-byte tests. The checkpoint is about 755.6 GB; no full weight shard was downloaded during this work.

## Immutable sources

Fetched on 2026-10-04:

- Model: [`zai-org/GLM-5.3`, revision `aca966e4e02791568aa6a4ced368624b3d897f42`](https://huggingface.co/zai-org/GLM-5.3/tree/aca966e4e02791568aa6a4ced368624b3d897f42).
- [Official config](https://huggingface.co/zai-org/GLM-5.3/resolve/aca966e4e02791568aa6a4ced368624b3d897f42/config.json), SHA256 `3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69`.
- [Official weight index](https://huggingface.co/zai-org/GLM-5.3/resolve/aca966e4e02791568aa6a4ced368624b3d897f42/model.safetensors.index.json), SHA256 `e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf`.
- Transformers [`modeling_glm_moe_dsa.py`](https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py) and [`configuration_glm_moe_dsa.py`](https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py), pinned to commit `469230357aab0f2b303b0d638c1f8d06edb14184`. Local snapshots preserve the upstream Apache-2.0 copyright/license notices. The model config declares `transformers_version=5.15.0`; the reference inspected is the pinned commit, not a claim that the locally installed package is 5.15.0.

The pinned Transformers [`finegrained_fp8.py`](https://github.com/huggingface/transformers/blob/469230357aab0f2b303b0d638c1f8d06edb14184/src/transformers/integrations/finegrained_fp8.py) also confirms that dequantization multiplies FP32 weight values by `weight_scale_inv`. Its generic converter infers block shape from the scale grid; this loader instead uses the declared 128×128 block size and supports the actual 576-row partial-block projection.

Local copies and machine-readable SHA256 provenance are in `metadata/`. `sample-shard-header.json` records the first shard header (21,840 bytes, fetched with two HTTP Range requests totaling 21,848 bytes including the length prefix). This is header evidence only; its offsets refer to the remote full shard. The original index contains **118,629 tensor entries**, **141 shards**, and `metadata.total_size=755617140416` bytes. This total is serialized tensor data, not RAM required after dequantization.

## Base decoder

| Quantity | Official value |
|---|---:|
| Decoder layers | 78 (indices 0–77) |
| Hidden width | 6144 |
| Vocabulary | 154880 |
| Dense MLP layers | 0, 1, 2 |
| Dense intermediate width | 12288 |
| MoE layers | 3–77 |
| Routed experts / chosen per token | 256 / 8 |
| Shared experts | 1, always evaluated |
| Expert intermediate width | 2048 |
| MLA heads | 64 |
| Query low-rank width | 2048 |
| KV low-rank width | 512 |
| Query/key non-rotary + rotary width | 192 + 64 = 256 |
| Value width per head | 256 |
| Indexer heads / width | 32 / 128 |
| Index selection budget | 2048 tokens |
| Default RoPE theta | 8000000 |
| Maximum position metadata | 1048576 |

There are additional next-token-prediction tensors under `model.layers.78.*`. The official base model creates exactly 78 decoder layers and explicitly ignores unexpected `model.layers.78.*` keys. The base decoder does not execute this MTP layer. `GLMConfig.tensor_shapes()` describes only the base decoder; the manifest reports extras (including scales and MTP) rather than treating them as a 79th normal layer.

Each layer is pre-norm: RMSNorm → MLA attention → residual addition → RMSNorm → dense/MoE MLP → residual addition. Final model RMSNorm precedes a separate, untied `lm_head`. Layer/final RMSNorm epsilon is `1e-5`. **The Q and KV low-rank RMSNorms use `1e-6`**, because the upstream constructor uses its default epsilon. Indexer key normalization is **LayerNorm with weight and bias and epsilon `1e-6`**, not RMSNorm.

## MLA and RoPE

For attention input `x`, all checkpoint matrices use `[out_features, in_features]` layout:

1. `q_resid = RMSNorm(q_a_proj(x), eps=1e-6)`; `q_b_proj(q_resid)` reshapes into 64 heads of 256 dimensions. Split the **last 64 dimensions** for RoPE and the first 192 as unrotated query components.
2. `kv_a_proj_with_mqa(x)` emits 512 latent values plus 64 rotary key values. RMS-normalize only the 512 latent values with epsilon `1e-6`.
3. Cache the **normalized 512 latent values and rotated 64 key values** separately per layer. Shared DSA index selections do **not** share MLA KV state.
4. `kv_b_proj(latent)` emits 64 × (192+256); split each head into the 192 non-rotary key and 256 value. Concatenate the head-specific 192 key with the same 64 rotary key broadcast across heads.
5. Scale query-key logits by `1/sqrt(256)=1/16`, apply causal and selected-token masking, FP32 softmax, attention-value contraction, concatenate heads, and apply `o_proj`.

**Do not use serialized `head_dim=192` for rotary frequencies.** The upstream config's `__post_init__` resets effective `head_dim=qk_rope_head_dim=64`. Frequencies are `theta^(-2j/64)` for `j=0..31`. The checkpoint's default RoPE does not enable YaRN scaling.

Both MLA and indexer use the upstream interleaved-input rotary operation. Given input pairs `(x[0],x[1]), (x[2],x[3]), ...`, the result is concatenated as `[even*cos - odd*sin, odd*cos + even*sin]`. It is a **half-split output layout**, not alternating rotated values. Applying identical rotation to both query and key preserves their dot product, but persistent cache layout must be consistent. The indexer rotates its **first 64 dimensions**, unlike MLA's last 64.

## DSA token selection and reuse

`indexer_types` is authoritative when explicitly present. The official full-indexer layers are **0, 1, 2, 6, 10, ..., 74** (21 full indexers). Layers 3–5 reuse layer 2's selection; 7–9 reuse layer 6; and 75–77 reuse layer 74. A shared layer reuses the preceding full layer's **top-k indices for the same current queries**, not its queries, projected keys, output, or MLA cache. The model threads returned indices from each decoder layer to the next; a shared layer without previous indices must fail.

Without explicit types, the official schedule is `full` iff `max(layer - offset + 1, 0) % frequency == 0`, using offset 3 and frequency 4. `index_topk_pattern` overrides this schedule; `GLMConfig` supports strings of `F`/`S` or explicit full/shared lists. Every layer uses indexed attention, including the first three dense-MLP layers.

Each full indexer computes:

- `q = indexer.wq_b(q_resid)`, reshaped as 32 × 128.
- `k = LayerNorm(indexer.wk(x), eps=1e-6)` with learned bias; one shared 128-dimensional key per token, cached for this full-indexer layer.
- Rotate first 64 dimensions in both, preserving remaining 64.
- Per-head score `relu(dot(q_h,k)/sqrt(128))`, in FP32.
- Head coefficient `indexer.weights_proj(x)[h]/sqrt(32)`. Coefficients are not sigmoid or softmax probabilities and may be negative.
- Sum coefficient × per-head score, apply causal/padding mask, then choose `min(2048, total_cached_length)` top indices. Indexer causality is required before top-k; the main attention still applies causality after selection. Tie ordering is backend dependent.

The checkpoint names the coefficient matrix **`self_attn.indexer.weights_proj.weight`**, despite historical `indexers_proj` entries in the config's quantization exclusion list. Use actual index names. The header shows coefficient weights stored BF16; the reference marks `indexer.weights_proj` for FP32 retention and computes projected weights and scoring in float.

## MoE routing

The gate matrix is `[256,6144]`, applied to normalized MLP input in FP32. Compute sigmoid of gate logits. Add `mlp.gate.e_score_correction_bias` **only for expert selection**. Group scores are the sum of the two highest corrected scores per group. Select `topk_group` groups and then 8 corrected expert scores within selected groups. The official model has `n_group=topk_group=1`, so this does not restrict selection.

Gather mixing weights from the **original sigmoid scores**, without correction bias. Divide by their sum plus `1e-20` because `norm_topk_prob=true`, then multiply by `routed_scaling_factor=2.5`. An expert computes `down_proj(silu(gate_proj(x))*up_proj(x))`. Sum weighted routed expert outputs and add the always-active shared expert output **without the routing multiplier**. This is not a softmax router.

## Actual checkpoint tensor names

For `P=model.layers.{layer}`:

| Suffix | Shape |
|---|---|
| `P.input_layernorm.weight`, `P.post_attention_layernorm.weight` | `[6144]` |
| `P.self_attn.q_a_proj.weight` | `[2048,6144]` |
| `P.self_attn.q_a_layernorm.weight` | `[2048]` |
| `P.self_attn.q_b_proj.weight` | `[16384,2048]` |
| `P.self_attn.kv_a_proj_with_mqa.weight` | `[576,6144]` |
| `P.self_attn.kv_a_layernorm.weight` | `[512]` |
| `P.self_attn.kv_b_proj.weight` | `[28672,512]` |
| `P.self_attn.o_proj.weight` | `[6144,16384]` |
| `P.self_attn.indexer.wq_b.weight` | `[4096,2048]`, full indexers only |
| `P.self_attn.indexer.wk.weight` | `[128,6144]`, full indexers only |
| `P.self_attn.indexer.k_norm.weight`, `.bias` | `[128]`, full indexers only |
| `P.self_attn.indexer.weights_proj.weight` | `[32,6144]`, full indexers only |
| `P.mlp.gate.weight` | `[256,6144]`, MoE only |
| `P.mlp.gate.e_score_correction_bias` | `[256]`, MoE only |
| `P.mlp.experts.{expert}.gate_proj.weight`, `.up_proj.weight` | `[2048,6144]` |
| `P.mlp.experts.{expert}.down_proj.weight` | `[6144,2048]` |
| `P.mlp.shared_experts.gate_proj.weight`, `.up_proj.weight` | `[2048,6144]` |
| `P.mlp.shared_experts.down_proj.weight` | `[6144,2048]` |
| `P.mlp.gate_proj.weight`, `.up_proj.weight` | `[12288,6144]`, dense only |
| `P.mlp.down_proj.weight` | `[6144,12288]`, dense only |
| `model.embed_tokens.weight`, `lm_head.weight` | `[154880,6144]` |
| `model.norm.weight` | `[6144]` |

Transformers represents loaded experts internally as merged 3D `gate_up_proj` and `down_proj` parameters, but the **actual checkpoint stores separate per-expert gate/up/down matrices**. The loader intentionally uses the checkpoint names, not the internal merged representation.

## FP8 and loader contract

The real header identifies FP8 as safetensors dtype `F8_E4M3` (E4M3FN semantics). Exponent bias is 7, subnormal unit `2^-9`, largest finite value 448, and only absolute code `0x7f` is NaN; `0xff` is its negative-sign NaN counterpart. Exponent 15 is otherwise finite. `F8_E4M3FN` is also accepted as an explicit alias. No E4M3FNUZ, E5M2, integer quantization, or other dtype is silently reinterpreted.

Each FP8 2D `.weight` has a sibling `.weight_scale_inv`, stored **F32**, whose shape is `[ceil(rows/128),ceil(cols/128)]`. Dequantization **multiplies** decoded FP8 by this scale. Do not take its reciprocal. A 576×6144 projection therefore has 5×48 scale entries. Partial blocks and row-window starts inside blocks are supported. Dynamic activation quantization is checkpoint metadata; this decoder dequantizes weight values only, and does not claim to reproduce an FP8 activation GEMM backend.

```python
from glm_vx.config import GLMConfig
from glm_vx.checkpoint import SafetensorsCheckpoint

cfg = GLMConfig.from_file("/weights/config.json")
with SafetensorsCheckpoint("/weights", block_size=cfg.weight_block_size) as ckpt:
    # Metadata only: verifies names, not local presence or header validity.
    report = ckpt.validate_manifest(cfg.tensor_shapes())
    # Decoded, independently owned float32 row without full embedding allocation.
    embedding = ckpt.tensor("model.embed_tokens.weight", rows=slice(token_id, token_id + 1))
    matrix = ckpt.tensor("model.layers.0.self_attn.q_a_proj.weight")
```

`Checkpoint` aliases `SafetensorsCheckpoint`; `get`, `load`, and mapping-style `checkpoint[name]` alias `tensor`. `GLMConfig`, `GlmConfig`, and `ModelConfig` refer to the same validated dataclass. `cfg.to_dict()` provides normalized fields for the runtime. Unsupported multi-group routing and unnormalized mixing are rejected because the engine backend currently implements the official single-group normalized router. Accessors include `qk_head_dim`, effective rotary `head_dim`, `attention_scale`, `qk_norm_eps`, and `index_source(layer)`. A runtime implementing per-block GEMM can use `SafetensorsFile.tensor()` for an unscaled decoder and explicitly consume scales; `SafetensorsCheckpoint.tensor()` always returns scaled weights.

Mappings use OS read-only access and an LRU (8 open shards by default). Decoded arrays own memory and stay valid after eviction/close. Calls are intended for a serialized inference thread; instances are not thread-safe. Per-call decoded output defaults to an 8 GiB guard, configurable as `max_tensor_bytes`; this is not a total runtime memory budget. FP8 dequantization may temporarily retain both an unscaled and scaled float32 array. Row slicing is contiguous, exact-bounds, and nonnegative; it does not silently clamp requests. The loader is local-only and never downloads a shard.

Validation rejects duplicate JSON keys, bad header lengths, unsupported dtypes, invalid shapes, shape/byte-count mismatches, offset overflow/out-of-bounds, overlapping tensors, payload gaps/trailing bytes, non-string metadata values, absolute/traversal/symlink escape shard paths, missing index names, wrong FP8 scale shape/dtype, and nonpositive/nonfinite scales. A constructed index is not proof of local weights; `validate_manifest(..., inspect_shards=True)` opens headers and validates expected shapes, while an actual load validates scale values.

The synthetic tests cover all 256 FP8 codes, floating-point edge values and BF16 bit expansion, scale block edges and misaligned row windows, lazy shard loading and eviction, corruption/path rejection, and official metadata/name completeness. They do not test full-model output accuracy, accelerator kernels, or model throughput.
