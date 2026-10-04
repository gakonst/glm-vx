"""GLM-DSA GGUF -> model tensor names, with bounded decoded-weight caching.

Inverse of llama.cpp conversion at 11fe02151f79c41d0d4af7da708755d73b9c0da6:
K-B is transposed per head; V-B is unchanged; experts are stacked by expert ID.
The MTP layer is excluded from ordinary autoregressive decoding.
"""
from collections import OrderedDict
import math
import re
import numpy as np
from .checkpoint import CheckpointError
from .config import GLMConfig
from .gguf_reader import GGUFStore

_ALIASES={
 'input_layernorm.weight':'attn_norm.weight',
 'post_attention_layernorm.weight':'ffn_norm.weight',
 'self_attn.q_a_proj.weight':'attn_q_a.weight',
 'self_attn.q_a_layernorm.weight':'attn_q_a_norm.weight',
 'self_attn.q_b_proj.weight':'attn_q_b.weight',
 'self_attn.kv_a_proj_with_mqa.weight':'attn_kv_a_mqa.weight',
 'self_attn.kv_a_layernorm.weight':'attn_kv_a_norm.weight',
 'self_attn.o_proj.weight':'attn_output.weight',
 'self_attn.indexer.wq_b.weight':'indexer.attn_q_b.weight',
 'self_attn.indexer.wk.weight':'indexer.attn_k.weight',
 'self_attn.indexer.k_norm.weight':'indexer.k_norm.weight',
 'self_attn.indexer.k_norm.bias':'indexer.k_norm.bias',
 'self_attn.indexer.weights_proj.weight':'indexer.proj.weight',
 'mlp.gate.weight':'ffn_gate_inp.weight',
 'mlp.gate.e_score_correction_bias':'exp_probs_b.bias',
}

class GGUFCheckpoint:
    def __init__(self,path,config,*,cache_bytes=512*1024**2,decode_rows=128,decode_threads=1):
        if type(cache_bytes) is not int or cache_bytes<0:raise ValueError('cache_bytes must be nonnegative')
        self.config=dict(config);self.validated=GLMConfig.from_dict(config)
        self.cache=OrderedDict();self.cache_bytes=0;self.limit=cache_bytes
        self.store=GGUFStore(path,decode_rows=decode_rows,decode_threads=decode_threads)
        try:
            self._validate_metadata()
            self.validate_manifest()
        except BaseException:self.close();raise

    def _validate_metadata(self):
        m=self.store.metadata;c=self.validated
        if m.get('general.architecture')!='glm-dsa':raise CheckpointError('GGUF architecture must be glm-dsa')
        checks={'context_length':c.max_position_embeddings,'embedding_length':c.hidden_size,'feed_forward_length':c.intermediate_size,
            'attention.head_count':c.num_attention_heads,'attention.q_lora_rank':c.q_lora_rank,
            'attention.kv_lora_rank':c.kv_lora_rank,'rope.dimension_count':c.qk_rope_head_dim,
            'attention.key_length_mla':c.qk_nope_head_dim+c.qk_rope_head_dim,
            'attention.value_length_mla':c.v_head_dim,'vocab_size':c.vocab_size,
            'expert_count':c.n_routed_experts,'expert_used_count':c.num_experts_per_tok,
            'expert_feed_forward_length':c.moe_intermediate_size,'expert_shared_count':c.n_shared_experts,
            'leading_dense_block_count':c.first_k_dense_replace,
            'attention.indexer.head_count':c.index_n_heads,'attention.indexer.key_length':c.index_head_dim,
            'attention.indexer.top_k':c.index_topk,'expert_group_count':c.n_group,
            'expert_group_used_count':c.topk_group,'expert_weights_norm':True,'expert_gating_func':2}
        for key,expected in checks.items():
            if m.get('glm-dsa.'+key)!=expected:raise CheckpointError('GGUF/config mismatch: '+key)
        mtp=m.get('glm-dsa.nextn_predict_layers',0)
        if type(mtp) is not int or mtp<0 or m.get('glm-dsa.block_count')!=c.num_hidden_layers+mtp:
            raise CheckpointError('GGUF block count does not match base decoder plus MTP')
        # These are MLA cache dimensions, not the expanded HF head dimensions.
        for key,expected in [('attention.head_count_kv',1),('attention.key_length',c.kv_lora_rank+c.qk_rope_head_dim),('attention.value_length',c.kv_lora_rank)]:
            if m.get('glm-dsa.'+key)!=expected:raise CheckpointError('GGUF MLA metadata mismatch: '+key)
        for key,expected in [('rope.freq_base',c.rope_theta),('expert_weights_scale',c.routed_scaling_factor),('attention.layer_norm_rms_epsilon',c.rms_norm_eps)]:
            if not math.isclose(float(m.get('glm-dsa.'+key,float('nan'))),expected,rel_tol=1e-6):raise CheckpointError('GGUF/config mismatch: '+key)

    def _resolve(self,name):
        plain={'model.embed_tokens.weight':'token_embd.weight','model.norm.weight':'output_norm.weight','lm_head.weight':'output.weight'}
        if name in plain:return plain[name],None
        match=re.fullmatch(r'model\.layers\.(\d+)\.(.+)',name)
        if not match:raise KeyError(name)
        layer=int(match[1]);suffix=match[2];prefix=f'blk.{layer}.'
        if not 0<=layer<self.validated.num_hidden_layers:raise KeyError(name)
        if suffix=='self_attn.kv_b_proj.weight':return prefix+'__kv_b__',None
        if suffix in _ALIASES:return prefix+_ALIASES[suffix],None
        expert=re.fullmatch(r'mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight',suffix)
        if expert:
            idx=int(expert[1])
            if idx>=self.validated.n_routed_experts:raise KeyError(name)
            return prefix+f'ffn_{expert[2]}_exps.weight',idx
        mlp=re.fullmatch(r'mlp\.(shared_experts\.)?(gate|up|down)_proj\.weight',suffix)
        if mlp:return prefix+f'ffn_{mlp[2]}'+('_shexp' if mlp[1] else '')+'.weight',None
        raise KeyError(name)

    def validate_manifest(self):
        c=self.validated
        for name,expected in c.tensor_shapes().items():
            source,expert=self._resolve(name)
            if source.endswith('__kv_b__'):
                prefix=source[:-8]
                if self.store.shape(prefix+'attn_k_b.weight')!=(c.num_attention_heads,c.kv_lora_rank,c.qk_nope_head_dim) or self.store.shape(prefix+'attn_v_b.weight')!=(c.num_attention_heads,c.v_head_dim,c.kv_lora_rank):raise CheckpointError('GGUF K/V-B shape mismatch')
                continue
            shape=self.store.shape(source)
            if expert is not None:
                if len(shape)!=3 or shape[0]!=c.n_routed_experts:raise CheckpointError('GGUF packed expert count mismatch')
                shape=shape[1:]
            if tuple(shape)!=tuple(expected):raise CheckpointError(f'GGUF tensor shape mismatch: {name}: {shape} != {expected}')

    def tensor(self,name,*,rows=None):
        if self.store.closed:raise RuntimeError('GGUF checkpoint is closed')
        if rows is None and name in self.cache:
            value=self.cache.pop(name);self.cache[name]=value;return value
        source,expert=self._resolve(name)
        if source.endswith('__kv_b__'):
            if rows is not None:raise CheckpointError('row slices of reconstructed KV-B unsupported')
            prefix=source[:-8];c=self.validated
            k=self.store.read(prefix+'attn_k_b.weight');v=self.store.read(prefix+'attn_v_b.weight')
            value=np.concatenate((k.transpose(0,2,1),v),axis=1).reshape(c.num_attention_heads*(c.qk_nope_head_dim+c.v_head_dim),c.kv_lora_rank)
            value.flags.writeable=False
        else:value=self.store.read(source,rows=rows,expert=expert)
        if rows is None and value.nbytes<=self.limit:
            while self.cache_bytes+value.nbytes>self.limit:
                _,old=self.cache.popitem(last=False);self.cache_bytes-=old.nbytes
            self.cache[name]=value;self.cache_bytes+=value.nbytes
        return value

    def linear(self,name,x,backend):
        """Optional model hook; fused packed weights with unchanged f32 input.

        Unsupported codecs/backends and reconstructed KV-B explicitly use the
        existing decoded tensor path. No quantized activation mode is enabled.
        """
        if self.store.closed:raise RuntimeError('GGUF checkpoint is closed')
        source,expert=self._resolve(name)
        supported=getattr(backend,'supports_packed',None)
        if source.endswith('__kv_b__') or supported is None:
            return backend.matvec(self.tensor(name),x)
        kind=self.store.tensors[source].tensor_type
        if not supported(kind):
            return backend.matvec(self.tensor(name),x)
        with self.store.packed_rows(source,expert=expert) as (raw,shape,kind):
            return backend.packed_matvec(raw,kind,shape,x)

    def __call__(self,name):return self.tensor(name)
    def close(self):
        self.cache.clear();self.cache_bytes=0;self.store.close()
