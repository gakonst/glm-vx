"""Real GLM indexer dimensions at the first actual pruning boundary."""
import numpy as np
import pytest
from kernels.backend import VxBackend

@pytest.mark.parametrize('length',[2047,2048,2049,4096])
def test_real_shape_index_scores_and_selection(length):
    rng=np.random.default_rng(5300+length)
    queries=rng.normal(size=(32,128)).astype(np.float32)
    keys=rng.normal(size=(length,128)).astype(np.float32)
    weights=rng.normal(size=32).astype(np.float32) / np.sqrt(32)
    # Independent double-precision definition, including negative head weights.
    expected=(np.maximum(queries.astype(np.float64) @ keys.astype(np.float64).T / np.sqrt(128),0)*weights[:,None]).sum(axis=0)
    b=VxBackend();actual=b.index_scores(queries,keys,weights)
    np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=3e-5)
    count=min(2048,length)
    selected=b.topk(actual,count)
    oracle=np.argsort(-expected,kind='stable')[:count]
    # Top-k order is not used as a semantics claim; selected set must agree.
    np.testing.assert_array_equal(np.sort(selected),np.sort(oracle))
    assert len(np.unique(selected))==count
    assert np.all((selected>=0)&(selected<length))


def test_selection_boundary_ties_have_explicit_stable_policy():
    b=VxBackend();scores=np.zeros(2049,dtype=np.float32)
    np.testing.assert_array_equal(b.topk(scores,2048),np.arange(2048))
    scores[-1]=np.nextafter(np.float32(0),np.float32(1))
    selected=b.topk(scores,2048)
    assert selected[0]==2048
    np.testing.assert_array_equal(selected[1:],np.arange(2047))
