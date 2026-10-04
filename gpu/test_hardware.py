"""Real-device integration. Explicitly skipped when no supported CUDA GPU exists."""
from pathlib import Path
import numpy as np
import pytest
from gpu.runtime import CUDAContext,CUDAUnavailable
from gpu.benchmark import bench
from gpu.backend import GPUBackend
from glm_vx.tiny import tiny_config
from glm_vx.model import Model,tiny_weights
from glm_vx.backend import NumpyBackend

@pytest.fixture
def context():
    try:ctx=CUDAContext(minimum_compute_capability=(8,0))
    except CUDAUnavailable as e:pytest.skip('real GPU unavailable: '+str(e))
    try:yield ctx
    finally:ctx.close()

@pytest.mark.parametrize('kernel',['matvec','rmsnorm','mla','fp8'])
def test_resident_kernel_real_device(context,kernel):
    # PTX/JIT/arithmetic failures on an available device must fail, not skip.
    results=bench(context,Path(__file__).parent/'build',kernel,repeats=1)
    assert len(results)==1
    assert results[0]['median_kernel_ms']>=0

def test_small_model_real_device(context):
    c=tiny_config();c.update(hidden_size=8,vocab_size=16,num_hidden_layers=2,
        num_attention_heads=1,num_key_value_heads=1,q_lora_rank=8,kv_lora_rank=4,
        qk_nope_head_dim=4,qk_rope_head_dim=4,qk_head_dim=8,v_head_dim=4,
        intermediate_size=16,moe_intermediate_size=8,n_routed_experts=2,
        num_experts_per_tok=2,index_n_heads=1,index_head_dim=8,index_topk=2,
        mlp_layer_types=['dense','sparse'],indexer_types=['full','shared'])
    w=tiny_weights(c,seed=9)
    for a in w.values():a.flags.writeable=False
    b=GPUBackend(weight_cache_bytes=2*1024**2)
    try:
        actual=Model(c,w,b).prefill([1,2,3])
        expected=Model(c,w,NumpyBackend()).prefill([1,2,3])
        np.testing.assert_allclose(actual,expected,rtol=5e-4,atol=3e-5)
        np.testing.assert_array_equal(np.argmax(actual,axis=-1),np.argmax(expected,axis=-1))
    finally:b.close()
