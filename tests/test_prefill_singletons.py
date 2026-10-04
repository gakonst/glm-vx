"""Singleton packed prefill must not materialize projection matrices."""
import numpy as np
import pytest
from glm_vx.model import Model,tiny_weights
from glm_vx.prefill import _linear
from glm_vx.gguf_checkpoint import GGUFCheckpoint
from glm_vx.tiny import tiny_config
from kernels.backend import VxBackend
from test_gguf_checkpoint import fixture_data
from test_packed_model import write_mixed_fixture
from test_batched_prefill import compare_cache


@pytest.mark.parametrize('singleton_chunk',[True,False])
def test_mixed_gguf_singletons_never_expand_and_match_logits_cache(tmp_path,monkeypatch,singleton_chunk):
    c,w=fixture_data();path=tmp_path/'mixed.gguf';write_mixed_fixture(path,c,w)
    plain=GGUFCheckpoint(path,c);packed=GGUFCheckpoint(path,c)
    class Routed(VxBackend):
        def __init__(self):super().__init__();self.calls=0
        def route(self,logits,bias,k,scale):
            # A two-token group shares expert0 while experts1/2 are singletons.
            ids=np.array([0,1+self.calls%2],np.int32);self.calls+=1
            return ids,np.array([1.5,1.0],np.float32)
    a=Model(c,plain,Routed());b=Model(c,packed,Routed(),packed_weights=True)
    ca,cb=a.new_cache(),b.new_cache();seen=[];original=packed.tensor;linear=packed.linear
    def no_expansion(name,*,rows=None):
        if rows is None and name.endswith('.weight') and not name.endswith('kv_b_proj.weight'):
            source,expert=packed._resolve(name)
            forbidden=(singleton_chunk and len(packed.store.shape(source))>=2) or '.experts.1.' in name or '.experts.2.' in name or name=='lm_head.weight'
            if forbidden:pytest.fail('expanded singleton projection: '+name)
        return original(name,rows=rows)
    def record(name,x,backend):seen.append(name);return linear(name,x,backend)
    monkeypatch.setattr(packed,'tensor',no_expansion);monkeypatch.setattr(packed,'linear',record)
    try:
        tokens=[7] if singleton_chunk else [7,11]
        np.testing.assert_array_equal(a.prefill(tokens,ca)[-1],b.prefill_chunk(tokens,cb))
        compare_cache(ca,cb)
        np.testing.assert_array_equal(a.forward(13,ca),b.forward(13,cb));compare_cache(ca,cb)
        assert 'lm_head.weight' in seen
        for projection in ('gate','up','down'):
            assert f'model.layers.1.mlp.experts.1.{projection}_proj.weight' in seen
            if not singleton_chunk:assert f'model.layers.1.mlp.experts.2.{projection}_proj.weight' in seen
    finally:plain.close();packed.close()


def test_singleton_projection_bias_and_shape_preserved():
    c=tiny_config();c['attention_bias']=True;weights=tiny_weights(c,39)
    class Checkpoint:
        def __call__(self,name):return weights[name]
        def linear(self,name,x,backend):return backend.matvec(weights[name],x)
    model=Model(c,Checkpoint(),VxBackend(),packed_weights=True)
    x=np.random.default_rng(48).normal(size=(1,c['hidden_size'])).astype(np.float32)
    prefix='model.layers.0.self_attn.q_a_proj'
    actual=_linear(model,prefix,x,bias=True)
    expected=model.backend.matvec(weights[prefix+'.weight'],x[0])+weights[prefix+'.bias']
    assert actual.shape==(1,len(expected));np.testing.assert_array_equal(actual[0],expected)


def test_nonpacked_alternate_stacked_experts_still_match_chunks():
    c=tiny_config();weights=tiny_weights(c,41);stacked=dict(weights)
    for layer,kind in enumerate(c['mlp_layer_types']):
        if kind!='sparse':continue
        prefix=f'model.layers.{layer}.mlp.experts'
        gu=[];down=[]
        for expert in range(c['n_routed_experts']):
            gu.append(np.concatenate([stacked.pop(f'{prefix}.{expert}.gate_proj.weight'),stacked.pop(f'{prefix}.{expert}.up_proj.weight')]))
            down.append(stacked.pop(f'{prefix}.{expert}.down_proj.weight'))
        stacked[prefix+'.gate_up_proj']=np.stack(gu);stacked[prefix+'.down_proj']=np.stack(down)
    a=Model(c,weights,VxBackend());b=Model(c,stacked,VxBackend());ca,cb=a.new_cache(),b.new_cache()
    for chunk in ([1],[7,11],[3,9,5]):
        np.testing.assert_array_equal(a.prefill(chunk,ca)[-1],b.prefill_chunk(chunk,cb));compare_cache(ca,cb)


@pytest.mark.parametrize('rows',range(1,18))
@pytest.mark.parametrize('cols',[129,2048])
def test_all_tiled_tails_preserve_order_under_large_cancellation(rows,cols):
    backend=VxBackend();rng=np.random.default_rng(971+rows)
    x=rng.normal(size=(rows,cols)).astype(np.float32)
    w=rng.normal(size=(33,cols)).astype(np.float32)
    w[:,::2]*=np.float32(2**20);w[:,1::4]*=np.float32(2**-10)
    expected=np.stack([backend.matvec(w,row) for row in x])
    actual=backend.linear_batch(w,x)
    assert actual.tobytes()==expected.tobytes()
