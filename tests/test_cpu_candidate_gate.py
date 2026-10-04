"""Candidate-bound replacement for the model test that pins the default .so."""
import hashlib
import os
from pathlib import Path
import numpy as np
from glm_vx.model import GlmMoeDsaModel,tiny_weights
from glm_vx.tiny import tiny_config
from kernels.backend import VxBackend
from test_model import expanded_batch_oracle


def test_candidate_binding_and_expanded_model_oracle():
    backend=VxBackend()
    selected=os.environ.get('GLM_VX_LIBRARY')
    if selected:assert backend.library_path==Path(selected).resolve()
    expected_hash=os.environ.get('GLM_VX_CANDIDATE_SHA256')
    if expected_hash:assert hashlib.sha256(backend.library_path.read_bytes()).hexdigest()==expected_hash
    c=tiny_config();weights=tiny_weights(c,seed=918)
    model=GlmMoeDsaModel(c,weights,backend);tokens=[1,7,2,11,4,3,16]
    expected,selections=expanded_batch_oracle(c,weights,tokens)
    cache=model.new_cache()
    actual=np.concatenate((model.prefill(tokens[:4],cache),np.stack([model.forward(t,cache) for t in tokens[4:]])))
    np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=3e-6)
    for layer,state in enumerate(cache.layers):np.testing.assert_array_equal(state.selected_indices,selections[layer][-1])
    assert cache.position==len(tokens)
