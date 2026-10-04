import numpy as np
from glm_vx.backend import NumpyBackend
from kernels.backend import VxBackend

def test_rope_actual_layout_and_routing_bias():
    a=NumpyBackend();b=VxBackend()
    x=np.array([1,2,3,4,5,6,7,8],dtype=np.float32)
    # Position zero still permutes to GLM's half-split output order.
    expected=np.array([1,3,5,7,2,4,6,8],dtype=np.float32)
    np.testing.assert_array_equal(a.rope(x,0,8000000),expected)
    np.testing.assert_array_equal(b.rope(x,0,8000000),expected)
    for pos in (1,131071,1048575):
        np.testing.assert_allclose(a.rope(x,pos,8000000),b.rope(x,pos,8000000),rtol=1e-6,atol=1e-6)
    logits=np.array([-1,3,.5,2],dtype=np.float32)
    bias=np.array([2,0,0,0],dtype=np.float32)
    ai,aw=a.route(logits,bias,2,2.5);bi,bw=b.route(logits,bias,2,2.5)
    np.testing.assert_array_equal(ai,bi);np.testing.assert_allclose(aw,bw,rtol=1e-6)
    assert ai.tolist()==[0,1]
    assert aw[0]<aw[1]  # bias selects expert 0; it must not increase its mixture weight.
