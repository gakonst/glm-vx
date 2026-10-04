"""Validated GLM-MoE-DSA architecture metadata, independent of Transformers.

Dimensions may be reduced for correctness fixtures. Unsupported architectural
variants fail at load time rather than silently running a different model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any

MODEL_ID = "zai-org/GLM-5.3"
MODEL_REVISION = "aca966e4e02791568aa6a4ced368624b3d897f42"
TRANSFORMERS_REVISION = "469230357aab0f2b303b0d638c1f8d06edb14184"


class ConfigError(ValueError):
    """The configuration cannot be implemented by this engine."""


@dataclass(frozen=True)
class GLMConfig:
    vocab_size: int = 154880
    hidden_size: int = 6144
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    num_experts_per_tok: int = 8
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    index_head_dim: int = 128
    index_n_heads: int = 32
    index_topk: int = 2048
    first_k_dense_replace: int = 3
    index_topk_freq: int = 4
    index_skip_topk_offset: int = 3
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5
    rms_norm_eps: float = 1e-5
    rope_theta: float = 8_000_000.0
    max_position_embeddings: int = 1048576
    indexer_types: tuple[str, ...] = ()
    mlp_layer_types: tuple[str, ...] = ()
    weight_block_size: tuple[int, int] = (128, 128)
    eos_token_id: tuple[int, ...] = (154820, 154827, 154829)
    pad_token_id: int | None = 154820
    quant_method: str | None = "fp8"

    def __post_init__(self):
        positive = ("vocab_size hidden_size intermediate_size moe_intermediate_size "
                    "num_hidden_layers num_attention_heads num_key_value_heads n_shared_experts "
                    "n_routed_experts num_experts_per_tok q_lora_rank kv_lora_rank qk_nope_head_dim "
                    "qk_rope_head_dim v_head_dim index_head_dim index_n_heads index_topk "
                    "index_topk_freq n_group topk_group max_position_embeddings").split()
        for name in positive:
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be a positive integer")
        for name in ("first_k_dense_replace", "index_skip_topk_offset"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ConfigError(f"{name} must be a nonnegative integer")
        if self.num_key_value_heads != self.num_attention_heads:
            raise ConfigError("MLA requires num_key_value_heads == num_attention_heads")
        if self.qk_rope_head_dim % 2 or self.qk_rope_head_dim > self.index_head_dim:
            raise ConfigError("rotary width must be even and fit within index_head_dim")
        if (self.n_routed_experts % self.n_group or self.topk_group > self.n_group
                or self.n_routed_experts // self.n_group < 2
                or self.num_experts_per_tok > self.topk_group * (self.n_routed_experts // self.n_group)):
            raise ConfigError("invalid expert group/top-k dimensions")
        if self.n_group != 1 or self.topk_group != 1:
            raise ConfigError("engine routing currently supports only a single expert group")
        if self.norm_topk_prob is not True:
            raise ConfigError("engine routing requires norm_topk_prob=true")
        for name in ("routed_scaling_factor", "rms_norm_eps", "rope_theta"):
            v = getattr(self, name)
            if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
                raise ConfigError(f"{name} must be positive and finite")
        if self.quant_method not in (None, "fp8"):
            raise ConfigError("only unquantized or fp8 checkpoints are supported")
        if (not isinstance(self.weight_block_size, (tuple, list)) or len(self.weight_block_size) != 2
                or any(type(v) is not int or v <= 0 for v in self.weight_block_size)):
            raise ConfigError("weight_block_size must contain two positive integers")
        object.__setattr__(self, "weight_block_size", tuple(self.weight_block_size))
        types = self.indexer_types or tuple(
            "full" if max(i - self.index_skip_topk_offset + 1, 0) % self.index_topk_freq == 0
            else "shared" for i in range(self.num_hidden_layers))
        mlps = self.mlp_layer_types or tuple(
            "dense" if i < self.first_k_dense_replace else "sparse" for i in range(self.num_hidden_layers))
        for name, value, allowed in (("indexer_types", types, ("full", "shared")),
                                      ("mlp_layer_types", mlps, ("dense", "sparse"))):
            if not isinstance(value, (list, tuple)) or len(value) != self.num_hidden_layers or any(v not in allowed for v in value):
                raise ConfigError(f"{name} must specify every layer using {allowed}")
            object.__setattr__(self, name, tuple(value))
        if self.indexer_types[0] != "full":
            raise ConfigError("first layer must have a full indexer")
        if not isinstance(self.eos_token_id, (tuple, list)):
            raise ConfigError("eos_token_id must be a sequence")
        if any(type(v) is not int or not 0 <= v < self.vocab_size for v in self.eos_token_id):
            raise ConfigError("eos_token_id outside vocabulary")
        if self.pad_token_id is not None and (type(self.pad_token_id) is not int or not 0 <= self.pad_token_id < self.vocab_size):
            raise ConfigError("pad_token_id outside vocabulary")
        object.__setattr__(self, "eos_token_id", tuple(self.eos_token_id))

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def head_dim(self) -> int:
        """Effective HF rotary dimension, NOT the stale serialized head_dim."""
        return self.qk_rope_head_dim

    @property
    def attention_scale(self) -> float:
        return self.qk_head_dim ** -0.5

    @property
    def num_local_experts(self) -> int:
        return self.n_routed_experts

    @property
    def qk_norm_eps(self) -> float:
        # HF constructs the q_a/kv_a norms without config.rms_norm_eps.
        return 1e-6

    def index_source(self, layer: int) -> int:
        if type(layer) is not int or not 0 <= layer < self.num_hidden_layers:
            raise IndexError(layer)
        while self.indexer_types[layer] == "shared":
            layer -= 1
        return layer

    def to_dict(self) -> dict[str, Any]:
        """Return normalized architecture fields accepted by the model runtime."""
        result = asdict(self)
        result.update({"model_type": "glm_moe_dsa", "architectures": ["GlmMoeDsaForCausalLM"],
                       "qk_head_dim": self.qk_head_dim, "head_dim": self.head_dim,
                       "rope_parameters": {"rope_type": "default", "rope_theta": self.rope_theta}})
        result["indexer_types"] = list(self.indexer_types)
        result["mlp_layer_types"] = list(self.mlp_layer_types)
        result["eos_token_id"] = list(self.eos_token_id)
        result["weight_block_size"] = list(self.weight_block_size)
        if self.quant_method == "fp8":
            result["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3",
                                            "activation_scheme": "dynamic",
                                            "weight_block_size": list(self.weight_block_size)}
        return result

    @classmethod
    def from_file(cls, path: str | Path) -> "GLMConfig":
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GLMConfig":
        if not isinstance(data, dict):
            raise ConfigError("configuration must be an object")
        if data.get("model_type") != "glm_moe_dsa":
            raise ConfigError("model_type must be glm_moe_dsa")
        if data.get("architectures", ["GlmMoeDsaForCausalLM"]) != ["GlmMoeDsaForCausalLM"]:
            raise ConfigError("unsupported model architecture")
        supported = {
            "attention_bias": False, "mlp_bias": False, "attention_dropout": 0.0,
            "hidden_act": "silu", "scoring_func": "sigmoid", "topk_method": "noaux_tc",
            "rope_interleave": True, "indexer_rope_interleave": True,
            "tie_word_embeddings": False, "ep_size": 1, "pretraining_tp": 1,
            "moe_layer_freq": 1, "moe_router_dtype": "float32",
        }
        for name, value in supported.items():
            if name in data and (data[name] != value or (type(value) is bool and type(data[name]) is not bool)):
                raise ConfigError(f"unsupported {name}: {data[name]!r}")
        names = {f.name for f in fields(cls)}
        metadata_keys = {
            "model_type", "architectures", "dtype", "torch_dtype", "transformers_version",
            "initializer_range", "use_cache", "bos_token_id", "num_nextn_predict_layers",
            "index_share_for_mtp_iteration", "index_topk_pattern", "head_dim", "qk_head_dim",
            "rope_parameters", "quantization_config", "layer_types", "_name_or_path",
        }
        unknown = set(data) - names - metadata_keys - set(supported)
        if unknown:
            raise ConfigError(f"unknown configuration fields: {sorted(unknown)}")
        kwargs = {k: v for k, v in data.items() if k in names}
        rope = data.get("rope_parameters", {"rope_type": "default", "rope_theta": 8_000_000.0})
        if not isinstance(rope, dict) or rope.get("rope_type") != "default" or set(rope) - {"rope_type", "rope_theta"}:
            raise ConfigError("only default RoPE is supported")
        if "rope_theta" in data and data["rope_theta"] != rope.get("rope_theta", 8_000_000.0):
            raise ConfigError("conflicting rope_theta settings")
        kwargs["rope_theta"] = rope.get("rope_theta", 8_000_000.0)
        quant = data.get("quantization_config")
        if quant is None:
            kwargs["quant_method"] = None
        else:
            if not isinstance(quant, dict) or any(quant.get(k) != v for k, v in {
                "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic"}.items()):
                raise ConfigError("only dynamic e4m3 block FP8 quantization is supported")
            if set(quant) - {"quant_method", "fmt", "activation_scheme", "weight_block_size", "modules_to_not_convert"}:
                raise ConfigError("unknown quantization settings")
            kwargs["quant_method"] = "fp8"
            if "weight_block_size" in data and data["weight_block_size"] != quant.get("weight_block_size", [128, 128]):
                raise ConfigError("conflicting weight_block_size settings")
            kwargs["weight_block_size"] = quant.get("weight_block_size", (128, 128))
        if data.get("indexer_types") is None:
            kwargs.pop("indexer_types", None)
            pattern = data.get("index_topk_pattern")
            if pattern is not None:
                if isinstance(pattern, str):
                    try:
                        kwargs["indexer_types"] = tuple({"F": "full", "S": "shared"}[c] for c in pattern)
                    except KeyError as e:
                        raise ConfigError("invalid index_topk_pattern") from e
                elif isinstance(pattern, (list, tuple)):
                    kwargs["indexer_types"] = tuple(pattern)
                else:
                    raise ConfigError("invalid index_topk_pattern")
        for key in ("indexer_types", "mlp_layer_types"):
            if key in kwargs and kwargs[key] is None:
                kwargs.pop(key)
        eos = kwargs.get("eos_token_id", cls.eos_token_id)
        kwargs["eos_token_id"] = () if eos is None else (eos,) if type(eos) is int else eos
        config = cls(**kwargs)
        if data.get("qk_head_dim", config.qk_head_dim) != config.qk_head_dim:
            raise ConfigError("qk_head_dim disagrees with rope + nope dimensions")
        if data.get("layer_types") is not None and data["layer_types"] != ["indexed_attention"] * config.num_hidden_layers:
            raise ConfigError("all layers must use indexed_attention")
        return config

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        """Expected base-decoder checkpoint names; excludes FP8 scales and MTP."""
        h, nh, q, kv = self.hidden_size, self.num_attention_heads, self.q_lora_rank, self.kv_lora_rank
        out = {"model.embed_tokens.weight": (self.vocab_size, h), "model.norm.weight": (h,),
               "lm_head.weight": (self.vocab_size, h)}
        for i in range(self.num_hidden_layers):
            p = f"model.layers.{i}."
            out[p + "input_layernorm.weight"] = (h,)
            out[p + "post_attention_layernorm.weight"] = (h,)
            shapes = {"q_a_proj.weight": (q, h), "q_a_layernorm.weight": (q,),
                      "q_b_proj.weight": (nh * self.qk_head_dim, q),
                      "kv_a_proj_with_mqa.weight": (kv + self.qk_rope_head_dim, h),
                      "kv_a_layernorm.weight": (kv,),
                      "kv_b_proj.weight": (nh * (self.qk_nope_head_dim + self.v_head_dim), kv),
                      "o_proj.weight": (h, nh * self.v_head_dim)}
            if self.indexer_types[i] == "full":
                shapes.update({"indexer.wq_b.weight": (self.index_n_heads * self.index_head_dim, q),
                               "indexer.wk.weight": (self.index_head_dim, h),
                               "indexer.k_norm.weight": (self.index_head_dim,),
                               "indexer.k_norm.bias": (self.index_head_dim,),
                               "indexer.weights_proj.weight": (self.index_n_heads, h)})
            out.update({p + "self_attn." + k: v for k, v in shapes.items()})
            if self.mlp_layer_types[i] == "dense":
                mlps = [(p + "mlp.", self.intermediate_size)]
            else:
                out[p + "mlp.gate.weight"] = (self.n_routed_experts, h)
                out[p + "mlp.gate.e_score_correction_bias"] = (self.n_routed_experts,)
                mlps = [(p + "mlp.shared_experts.", self.n_shared_experts * self.moe_intermediate_size)]
                mlps += [(p + f"mlp.experts.{j}.", self.moe_intermediate_size) for j in range(self.n_routed_experts)]
            for prefix, width in mlps:
                out[prefix + "gate_proj.weight"] = (width, h)
                out[prefix + "up_proj.weight"] = (width, h)
                out[prefix + "down_proj.weight"] = (h, width)
        return out


ModelConfig = GLMConfig
GlmConfig = GLMConfig
