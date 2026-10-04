import numpy as np
import pytest
from glm_vx.model import GlmMoeDsaModel
from glm_vx.backend import NumpyBackend
from tests.test_model import tiny_fixture


def test_trace_owns_arrays_and_preserves_logits():
    c,w=tiny_fixture();seen={}
    def capture(p,name,value):
        seen[p,name]=value.copy()
        value[...] = 0 # hostile observer must not mutate the live cache or logits
    observed=GlmMoeDsaModel(c,w,NumpyBackend(),trace=capture)
    baseline=GlmMoeDsaModel(c,w,NumpyBackend())
    a,b=observed.new_cache(),baseline.new_cache()
    for pos,tok in enumerate([1,7,2,11]):
        np.testing.assert_array_equal(observed.forward(tok,a),baseline.forward(tok,b))
        for layer in range(observed.n_layers):
            np.testing.assert_array_equal(seen[pos,f'layer.{layer}.selected'],b.layers[layer].selected_indices)
            np.testing.assert_array_equal(a.layers[layer].latents,b.layers[layer].latents)
        np.testing.assert_array_equal(seen[pos,'logits'],baseline.prefill([1,7,2,11][:pos+1])[-1])


def test_observer_failure_rolls_back_cache():
    c,w=tiny_fixture()
    def capture(p,name,value):
        if p==1 and name=='layer.1.output':raise RuntimeError('trace write failed')
    m=GlmMoeDsaModel(c,w,NumpyBackend(),trace=capture);cache=m.new_cache();m.forward(1,cache)
    old=[x.selected_indices.copy() for x in cache.layers]
    with pytest.raises(RuntimeError,match='trace write failed'):m.forward(2,cache)
    assert cache.position==1
    for state,indices in zip(cache.layers,old):
        assert len(state.latents)==len(state.rope_keys)==1
        np.testing.assert_array_equal(state.selected_indices,indices)
    m.trace=None
    np.testing.assert_array_equal(m.forward(2,cache),GlmMoeDsaModel(c,w,NumpyBackend()).prefill([1,2])[-1])
