"""Independent HF-to-GGUF fixture for checkpoint mapping and model parity."""
from contextlib import contextmanager

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from glm_vx.checkpoint import CheckpointError
from glm_vx.config import GLMConfig
from glm_vx.gguf_checkpoint import GGUFCheckpoint
from glm_vx.model import Model, tiny_weights
from glm_vx.tiny import tiny_config
from test_model import NumpyTestBackend


def fixture_data():
    config = tiny_config()
    config.update(num_hidden_layers=2, indexer_types=["full", "shared"],
                  mlp_layer_types=["dense", "sparse"], index_topk=2)
    weights = tiny_weights(config, seed=931)
    # Nonuniform norm and bias values make every mapping observable.
    rng = np.random.default_rng(74)
    for name, value in weights.items():
        if value.ndim == 1:
            value[:] = rng.uniform(.7, 1.3, value.shape) if name.endswith("weight") else rng.normal(0, .1, value.shape)
    return config, weights


def packed_weights(config, weights):
    """Forward official layout, independent of production _resolve/_ALIASES."""
    tensors = {"token_embd.weight": weights["model.embed_tokens.weight"],
               "output.weight": weights["lm_head.weight"],
               "output_norm.weight": weights["model.norm.weight"]}
    pairs = [
        ("input_layernorm.weight", "attn_norm.weight"),
        ("post_attention_layernorm.weight", "ffn_norm.weight"),
        ("self_attn.q_a_proj.weight", "attn_q_a.weight"),
        ("self_attn.q_a_layernorm.weight", "attn_q_a_norm.weight"),
        ("self_attn.q_b_proj.weight", "attn_q_b.weight"),
        ("self_attn.kv_a_proj_with_mqa.weight", "attn_kv_a_mqa.weight"),
        ("self_attn.kv_a_layernorm.weight", "attn_kv_a_norm.weight"),
        ("self_attn.o_proj.weight", "attn_output.weight"),
        ("self_attn.indexer.wq_b.weight", "indexer.attn_q_b.weight"),
        ("self_attn.indexer.wk.weight", "indexer.attn_k.weight"),
        ("self_attn.indexer.k_norm.weight", "indexer.k_norm.weight"),
        ("self_attn.indexer.k_norm.bias", "indexer.k_norm.bias"),
        ("self_attn.indexer.weights_proj.weight", "indexer.proj.weight"),
        ("mlp.gate.weight", "ffn_gate_inp.weight"),
        ("mlp.gate.e_score_correction_bias", "exp_probs_b.bias"),
    ]
    for layer in range(config["num_hidden_layers"]):
        hf, gg = f"model.layers.{layer}.", f"blk.{layer}."
        for source, target in pairs:
            if hf + source in weights:
                tensors[gg + target] = weights[hf + source]
        kv = weights[hf + "self_attn.kv_b_proj.weight"].reshape(
            config["num_attention_heads"], config["qk_nope_head_dim"] + config["v_head_dim"], config["kv_lora_rank"])
        tensors[gg + "attn_k_b.weight"] = kv[:, :config["qk_nope_head_dim"]].transpose(0, 2, 1).copy()
        tensors[gg + "attn_v_b.weight"] = kv[:, config["qk_nope_head_dim"]:].copy()
        for projection in ("gate", "up", "down"):
            for source, target in (("mlp.", ""), ("mlp.shared_experts.", "_shexp")):
                name = hf + source + projection + "_proj.weight"
                if name in weights:
                    tensors[gg + "ffn_" + projection + target + ".weight"] = weights[name]
            if config["mlp_layer_types"][layer] == "sparse":
                tensors[gg + f"ffn_{projection}_exps.weight"] = np.stack([
                    weights[hf + f"mlp.experts.{expert}.{projection}_proj.weight"]
                    for expert in range(config["n_routed_experts"])])
    return tensors


def fixture_metadata(c):
    return {
        "context_length": c["max_position_embeddings"],
        "embedding_length": c["hidden_size"], "feed_forward_length": c["intermediate_size"],
        "block_count": c["num_hidden_layers"], "attention.head_count": c["num_attention_heads"],
        "attention.head_count_kv": 1, "attention.q_lora_rank": c["q_lora_rank"],
        "attention.kv_lora_rank": c["kv_lora_rank"], "rope.dimension_count": c["qk_rope_head_dim"],
        "attention.key_length": c["kv_lora_rank"] + c["qk_rope_head_dim"],
        "attention.value_length": c["kv_lora_rank"],
        "attention.key_length_mla": c["qk_nope_head_dim"] + c["qk_rope_head_dim"],
        "attention.value_length_mla": c["v_head_dim"], "vocab_size": c["vocab_size"],
        "expert_count": c["n_routed_experts"], "expert_used_count": c["num_experts_per_tok"],
        "expert_feed_forward_length": c["moe_intermediate_size"], "expert_shared_count": c["n_shared_experts"],
        "leading_dense_block_count": c["first_k_dense_replace"],
        "attention.indexer.head_count": c["index_n_heads"], "attention.indexer.key_length": c["index_head_dim"],
        "attention.indexer.top_k": c["index_topk"], "expert_group_count": c["n_group"],
        "expert_group_used_count": c["topk_group"], "expert_weights_norm": True, "expert_gating_func": 2,
        "rope.freq_base": float(c["rope_parameters"]["rope_theta"]),
        "expert_weights_scale": c["routed_scaling_factor"],
        "attention.layer_norm_rms_epsilon": c["rms_norm_eps"],
    }


def write_fixture(path, config, weights, *, metadata=None, tensors=None, architecture="glm-dsa"):
    values = fixture_metadata(config)
    values.update(metadata or {})
    writer = gguf.GGUFWriter(path, architecture)
    for key, value in values.items():
        if value is None:
            continue
        add = writer.add_bool if isinstance(value, bool) else writer.add_float32 if isinstance(value, float) else writer.add_uint32
        add("glm-dsa." + key, value)
    for name, value in (packed_weights(config, weights) if tensors is None else tensors).items():
        writer.add_tensor(name, np.ascontiguousarray(value, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


@contextmanager
def opened(path, config, **kwargs):
    checkpoint = GGUFCheckpoint(path, config, **kwargs)
    try:
        yield checkpoint
    finally:
        checkpoint.close()


def test_every_mapped_weight_is_exact_f32(tmp_path):
    config, weights = fixture_data()
    assert {name: value.shape for name, value in weights.items()} == GLMConfig.from_dict(config).tensor_shapes()
    path = write_fixture(tmp_path / "tiny.gguf", config, weights)
    with opened(path, config) as checkpoint:
        for name, expected in weights.items():
            actual = checkpoint.tensor(name)
            assert actual.dtype == np.float32
            assert not actual.flags.writeable
            np.testing.assert_array_equal(actual, expected, err_msg=name)
        assert not any(name.startswith("blk.1.indexer") for name in checkpoint.store.tensors)
        retained = checkpoint.tensor("model.layers.0.self_attn.kv_b_proj.weight")
    np.testing.assert_array_equal(retained, weights["model.layers.0.self_attn.kv_b_proj.weight"])


def test_embedding_reads_only_requested_rows_without_populating_cache(tmp_path, monkeypatch):
    config, weights = fixture_data()
    path = write_fixture(tmp_path / "rows.gguf", config, weights)
    with opened(path, config) as checkpoint:
        calls = []
        original = checkpoint.store.read

        def recording(name, **kwargs):
            calls.append((name, kwargs))
            return original(name, **kwargs)

        monkeypatch.setattr(checkpoint.store, "read", recording)
        model = Model(config, checkpoint, NumpyTestBackend())
        for token in (7, 0, config["vocab_size"] - 1):
            np.testing.assert_array_equal(model._embedding(token), weights["model.embed_tokens.weight"][token])
        assert calls == [("token_embd.weight", {"rows": slice(token, token + 1), "expert": None})
                         for token in (7, 0, config["vocab_size"] - 1)]
        assert checkpoint.cache_bytes == 0
        assert not checkpoint.cache
        selected = checkpoint.tensor("model.embed_tokens.weight", rows=slice(3, 8))
        np.testing.assert_array_equal(selected, weights["model.embed_tokens.weight"][3:8])
        expert = "model.layers.1.mlp.experts.2.down_proj.weight"
        np.testing.assert_array_equal(checkpoint.tensor(expert, rows=slice(1, 4)), weights[expert][1:4])
        with pytest.raises(CheckpointError, match="row slices"):
            checkpoint.tensor("model.layers.0.self_attn.kv_b_proj.weight", rows=slice(0, 1))


def test_decoded_lru_reuses_values_evicts_oldest_and_bypasses_large_weights(tmp_path, monkeypatch):
    config, weights = fixture_data()
    path = write_fixture(tmp_path / "lru.gguf", config, weights)
    a = "model.norm.weight"  # 128 bytes
    b = "model.layers.0.self_attn.q_a_layernorm.weight"  # 64 bytes
    c = "model.layers.0.self_attn.kv_a_layernorm.weight"  # 32 bytes
    with opened(path, config, cache_bytes=192) as checkpoint:
        calls = []
        original = checkpoint.store.read

        def recording(name, **kwargs):
            calls.append(name)
            return original(name, **kwargs)

        monkeypatch.setattr(checkpoint.store, "read", recording)
        first = checkpoint.tensor(a)
        checkpoint.tensor(b)
        assert checkpoint.cache_bytes == 192
        assert checkpoint.tensor(a) is first
        assert len(calls) == 2
        checkpoint.tensor(c)
        assert list(checkpoint.cache) == [a, c]
        assert checkpoint.cache_bytes == 160
        checkpoint.tensor(b)
        assert list(checkpoint.cache) == [c, b]
        assert checkpoint.cache_bytes == 96
        for _ in range(2):
            np.testing.assert_array_equal(checkpoint.tensor("lm_head.weight"), weights["lm_head.weight"])
        assert calls.count("output.weight") == 2
        assert list(checkpoint.cache) == [c, b]
        assert checkpoint.cache_bytes == sum(value.nbytes for value in checkpoint.cache.values())
    assert not checkpoint.cache and checkpoint.cache_bytes == 0
    np.testing.assert_array_equal(first, weights[a])
    with pytest.raises(RuntimeError, match="closed"):
        checkpoint.tensor(a)
    with opened(path, config, cache_bytes=0) as checkpoint:
        checkpoint.tensor(a)
        checkpoint.tensor(a)
        assert not checkpoint.cache and checkpoint.cache_bytes == 0


def assert_caches_equal(actual, expected):
    assert actual.position == expected.position
    for left, right in zip(actual.layers, expected.layers):
        for attribute in ("latents", "rope_keys", "index_keys"):
            values, reference = getattr(left, attribute), getattr(right, attribute)
            assert len(values) == len(reference)
            for value, expected_value in zip(values, reference):
                np.testing.assert_array_equal(value, expected_value)
        np.testing.assert_array_equal(left.selected_indices, right.selected_indices)


def test_prefill_decode_logits_and_request_caches_match_memory(tmp_path):
    config, weights = fixture_data()
    path = write_fixture(tmp_path / "model.gguf", config, weights)
    # Small cache forces repeated decoding and expert eviction during inference.
    with opened(path, config, cache_bytes=2048, decode_rows=2) as checkpoint:
        disk = Model(config, checkpoint, NumpyTestBackend())
        memory = Model(config, weights, NumpyTestBackend())
        dc, mc = disk.new_cache(), memory.new_cache()
        tokens = [1, 9, 3, 11, 8]  # beyond index_topk exercises sparse/shared DSA
        np.testing.assert_array_equal(disk.prefill(tokens, dc), memory.prefill(tokens, mc))
        assert_caches_equal(dc, mc)
        for token in (5, 21):
            np.testing.assert_array_equal(disk.forward(token, dc), memory.forward(token, mc))
            assert_caches_equal(dc, mc)
        assert len(dc.layers[0].index_keys) == 7
        assert dc.layers[1].index_keys == []
        np.testing.assert_array_equal(dc.layers[1].selected_indices, dc.layers[0].selected_indices)
        assert checkpoint.cache_bytes <= 2048
        assert "model.embed_tokens.weight" not in checkpoint.cache
        independent = disk.new_cache()
        np.testing.assert_array_equal(disk.forward(1, independent), memory.forward(1, memory.new_cache()))
        assert independent.position == 1 and dc.position == 7


@pytest.mark.parametrize("patch", [
    {"context_length": 1024}, {"embedding_length": 33}, {"vocab_size": 255}, {"block_count": 3},
    {"attention.head_count": 3}, {"attention.head_count_kv": 2},
    {"attention.key_length": 12 + 1}, {"attention.value_length": 9},
    {"attention.key_length_mla": 13}, {"attention.value_length_mla": 9},
    {"attention.q_lora_rank": 17}, {"attention.kv_lora_rank": 9},
    {"rope.dimension_count": 2}, {"attention.indexer.top_k": 3},
    {"expert_count": 5}, {"expert_weights_norm": False}, {"expert_gating_func": 1},
    {"expert_weights_scale": 1.0}, {"rope.freq_base": 10000.0},
    {"attention.layer_norm_rms_epsilon": .001}, {"embedding_length": None},
])
def test_metadata_mismatch_rejected_before_tensor_reads(tmp_path, monkeypatch, patch):
    config, weights = fixture_data()
    path = write_fixture(tmp_path / "bad-meta.gguf", config, weights, metadata=patch)

    def unexpected_read(*args, **kwargs):
        pytest.fail("invalid metadata must fail before decoding weights")

    from glm_vx.gguf_reader import GGUFStore
    monkeypatch.setattr(GGUFStore, "read", unexpected_read)
    with pytest.raises(CheckpointError, match="mismatch|block count"):
        GGUFCheckpoint(path, config)


def test_wrong_architecture_rejected(tmp_path):
    config, weights = fixture_data()
    path = write_fixture(tmp_path / "wrong-arch.gguf", config, weights, architecture="llama")
    with pytest.raises(CheckpointError, match="architecture"):
        GGUFCheckpoint(path, config)


@pytest.mark.parametrize("name,axis", [
    ("token_embd.weight", 0), ("blk.0.attn_q_b.weight", 0),
    ("blk.0.attn_kv_a_mqa.weight", 1), ("blk.0.attn_k_b.weight", 1),
    ("blk.0.attn_v_b.weight", 1), ("blk.1.ffn_gate_exps.weight", 0),
    ("blk.1.ffn_down_exps.weight", 1), ("blk.0.indexer.attn_q_b.weight", 0),
])
def test_tensor_shape_mismatch_rejected(tmp_path, name, axis):
    config, weights = fixture_data()
    tensors = packed_weights(config, weights)
    selection = [slice(None)] * tensors[name].ndim
    selection[axis] = slice(1, None)
    tensors[name] = tensors[name][tuple(selection)].copy()
    path = write_fixture(tmp_path / "bad-shape.gguf", config, weights, tensors=tensors)
    with pytest.raises(CheckpointError, match="shape mismatch|expert count"):
        GGUFCheckpoint(path, config)


def test_declared_mtp_extra_block_is_not_resolved_as_decoder_layer(tmp_path):
    config, weights = fixture_data()
    tensors = packed_weights(config, weights)
    tensors["blk.2.attn_norm.weight"] = np.ones(config["hidden_size"], np.float32)
    path = write_fixture(tmp_path / "mtp.gguf", config, weights, tensors=tensors,
                         metadata={"block_count": 3, "nextn_predict_layers": 1})
    with opened(path, config) as checkpoint:
        np.testing.assert_array_equal(checkpoint.tensor("model.norm.weight"), weights["model.norm.weight"])
        with pytest.raises(KeyError):
            checkpoint.tensor("model.layers.2.input_layernorm.weight")


@pytest.mark.parametrize("name", [
    "modelXlayersX0.input_layernorm.weight",
    "model.layers.1.mlpXexpertsX0.gate_proj.weight",
    "model.layers.0.mlp.gate_projXweight",
])
def test_malformed_tensor_names_do_not_alias_valid_weights(tmp_path, name):
    config, weights = fixture_data()
    path = write_fixture(tmp_path / "names.gguf", config, weights)
    with opened(path, config) as checkpoint:
        with pytest.raises(KeyError):
            checkpoint.tensor(name)
