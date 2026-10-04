# GLM MoE DSA GGUF converter audit

Completed 2026-10-04. Source-only investigation; no implementation or commits. Official llama.cpp master resolved through the GitHub commits API to **11fe02151f79c41d0d4af7da708755d73b9c0da6**. Sources below use that immutable pin. Only small Python sources/API metadata were fetched; no model weights.

## Conclusion and exact inverse

**No Q/K RoPE weight permutation, output projection permutation, or vocabulary/head permutation is applied by this converter to GlmMoeDsaForCausalLM.** The relevant nontrivial attention conversion is splitting HF `kv_b_proj.weight` per head and transposing K's final two axes. `GlmMoeDsaModel` inherits `DeepseekV2Model`, NOT the separate `DeepseekModel` containing a Q/K permutation.

First dequantize GGUF tensors and interpret their logical row-major NumPy/PyTorch shape as **reverse(descriptor.shape)**. The descriptor `data_shape` describes packed quantized bytes, not the floating-point matrix. Reversing the dimensions here is a storage shape convention, not an instruction to transpose a decoded ordinary 2D tensor.

For this descriptor set, H=64, K=192 non-RoPE coordinates, V=256, R=512 KV latent coordinates, Qrank=2048, Dmodel=6144, RoPE=64:

```python
# Kgg and Vgg are dequantized tensors with logical row-major shapes below.
Kgg = Kgg.reshape(64, 512, 192)  # blk.i.attn_k_b.weight
Vgg = Vgg.reshape(64, 256, 512)  # blk.i.attn_v_b.weight
W_kv_b = torch.cat((Kgg.transpose(1, 2), Vgg), dim=1).reshape(28672, 512)
# HF: model.layers.i.self_attn.kv_b_proj.weight
```

Concatenate on the **within-head output axis**, then flatten head and output. Do NOT concatenate flattened all-head K followed by flattened all-head V. Elementwise: `W_kv_b[h*448+j,r] = Kgg[h,r,j]` for j<192, and `W_kv_b[h*448+192+j,r] = Vgg[h,j,r]` for j<256. This is the exact layout inverse; lossy quantization prevents recovery of original prequantization values.

| GGUF name suffix | HF name suffix | Decoded shape / action |
|---|---|---|
| attn_q_b.weight | self_attn.q_b_proj.weight | [16384,2048], unchanged; heads [64,256], each [192 non-RoPE,64 RoPE] |
| attn_kv_a_mqa.weight | self_attn.kv_a_proj_with_mqa.weight | [576,6144], unchanged; rows [512 latent,64 RoPE] |
| indexer.attn_q_b.weight | self_attn.indexer.wq_b.weight | [4096,2048], unchanged; heads [32,128], each [64 RoPE,64 non-RoPE] |
| indexer.attn_k.weight | self_attn.indexer.wk.weight | [128,6144], unchanged; normalized key then [64 RoPE,64 non-RoPE] |
| attn_output.weight | self_attn.o_proj.weight | [6144,16384], unchanged; columns preserve [64 heads,256 value coordinates] |
| output.weight | lm_head.weight | [154880,6144], unchanged, no vocabulary-row permutation |
| token_embd.weight | model.embed_tokens.weight | [154880,6144], unchanged |

The ordinary layer prefixes are `blk.i.` versus `model.layers.i.`. Q-A, KV-A norm, Q-A norm, and indexer norm/projection are ordinary name mappings. The converter can omit a tied lm_head when `tie_word_embeddings=true`; this descriptor set explicitly contains output.weight. Expert tensors are separately stacked by ascending expert index; that is outside the attention-specific inverse above.

## RoPE evidence

Local HF `metadata/modeling_glm_moe_dsa.py:149-158` applies adjacent even/odd coordinate rotations, and does not demand a weight permutation. Main Q splits [non-RoPE,RoPE] at lines 391-402; KV-A splits [latent,RoPE] at 395-400. Indexer Q/K instead split [RoPE,non-RoPE] at 221-231. These different slice positions must be preserved. The upstream converter leaves all these weights untouched.

The local HF output path reshapes head-major attention output before o_proj at lines 457-458; its kv_b expansion at 365-375 matches the inverse above. Do not infer a head permutation from reversed GGUF descriptor dimensions.

## Immutable upstream citations

- [GlmMoeDsaModel registration, inheritance, and complete class, glm.py lines 316-382](https://github.com/ggml-org/llama.cpp/blob/11fe02151f79c41d0d4af7da708755d73b9c0da6/conversion/glm.py#L316-L382): GLM_DSA architecture; no modify_tensors override.
- [DeepseekV2Model declaration, deepseek.py line 236](https://github.com/ggml-org/llama.cpp/blob/11fe02151f79c41d0d4af7da708755d73b9c0da6/conversion/deepseek.py#L236): derives directly from TextModel.
- [Inherited complete tensor transform, deepseek.py lines 390-452](https://github.com/ggml-org/llama.cpp/blob/11fe02151f79c41d0d4af7da708755d73b9c0da6/conversion/deepseek.py#L390-L452): tied output omission, expert stacking, and kv_b split/transpose. Exact split/transpose at 433-449.
- [Base tensor name mapping and optional fusion, base.py line 685 onward](https://github.com/ggml-org/llama.cpp/blob/11fe02151f79c41d0d4af7da708755d73b9c0da6/conversion/base.py#L685): mapping layer; relevant ordinary projections retain values.
- [Main MLA aliases, tensor_mapping.py lines 1134-1160](https://github.com/ggml-org/llama.cpp/blob/11fe02151f79c41d0d4af7da708755d73b9c0da6/gguf-py/gguf/tensor_mapping.py#L1134-L1160).
- [Indexer aliases, tensor_mapping.py lines 1390-1400](https://github.com/ggml-org/llama.cpp/blob/11fe02151f79c41d0d4af7da708755d73b9c0da6/gguf-py/gguf/tensor_mapping.py#L1390-L1400).
- [Attention output alias, tensor_mapping.py line 330](https://github.com/ggml-org/llama.cpp/blob/11fe02151f79c41d0d4af7da708755d73b9c0da6/gguf-py/gguf/tensor_mapping.py#L330), [lm_head alias line 81](https://github.com/ggml-org/llama.cpp/blob/11fe02151f79c41d0d4af7da708755d73b9c0da6/gguf-py/gguf/tensor_mapping.py#L81).

## Local evidence and limits

- `build/gguf-tensor-descriptors.json` SHA256 `66b6e57a4bcb7410f538a0bbb86e7594c1807ebe6670498d2d5eb738052b4044`.
- `metadata/modeling_glm_moe_dsa.py` SHA256 `039993e90e650395e65ecfd6f803a7200799967b88535f5742c490fe2346f1d8`.
- Descriptor example: K GGUF [192,512,64], V [512,256,64], q_b [2048,16384], indexer q_b [2048,4096], KV-A [6144,576], output projection [16384,6144]. These directly match the converter layouts above.
- Audit proves current pinned official converter behavior plus descriptor shape compatibility. It does not establish which historical converter produced this particular GGUF, or numerical equivalence to original HF weights; neither provenance nor decoded weight comparison was provided/performed.
