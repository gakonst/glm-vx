"""Small random model for mechanical correctness, never a trained GLM checkpoint."""
def tiny_config():
    return dict(model_type="glm_moe_dsa", architectures=["GlmMoeDsaForCausalLM"],
        hidden_size=32, vocab_size=256, num_hidden_layers=4, num_attention_heads=2,
        num_key_value_heads=2, q_lora_rank=16, kv_lora_rank=8,
        qk_nope_head_dim=8, qk_rope_head_dim=4, qk_head_dim=12, v_head_dim=8,
        intermediate_size=64, moe_intermediate_size=16, n_routed_experts=4,
        n_shared_experts=1, num_experts_per_tok=2, first_k_dense_replace=1,
        mlp_layer_types=["dense","sparse","sparse","sparse"],
        indexer_types=["full","full","full","shared"], index_head_dim=8,
        index_n_heads=2,index_topk=4,indexer_rope_interleave=True,
        index_topk_freq=4,index_skip_topk_offset=3,
        n_group=1,topk_group=1,topk_method="noaux_tc",scoring_func="sigmoid",
        norm_topk_prob=True,routed_scaling_factor=2.5, rms_norm_eps=1e-5,
        rope_interleave=True,rope_parameters={"rope_theta":8000000,"rope_type":"default"},
        attention_bias=False,hidden_act="silu",tie_word_embeddings=False,
        max_position_embeddings=512,eos_token_id=[],pad_token_id=0)
