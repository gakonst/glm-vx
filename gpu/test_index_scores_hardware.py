"""Actual CUDA tests; unavailable hardware is a skip, not validation success."""
from pathlib import Path
import numpy as np
import pytest
from gpu.backend import GPUBackend
from gpu.runtime import CUDAContext, CUDAUnavailable, GLMKernels
from gpu.testing.index_reference import inputs, ordered_f32, reference_f64


@pytest.fixture
def context():
    try:
        ctx = CUDAContext(minimum_compute_capability=(8, 0))
    except CUDAUnavailable as error:
        pytest.skip('real GPU unavailable: ' + str(error))
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.mark.parametrize('heads,tokens,dim', [(3, 5, 33), (64, 7, 128)])
def test_resident_index_scores_real_device(context, heads, tokens, dim):
    kernels = GLMKernels(context.load_ptx(Path(__file__).parent / 'build/kernels.ptx'))
    args = inputs(heads, tokens, dim)
    tensors = [context.tensor(x.shape, data=x.tobytes()) for x in args]
    out = context.tensor((tokens,))
    stream = context.stream()
    for scale in (dim**-.5, -.25, 0.):
        assert kernels.index_scores(*tensors, scale, out=out, stream=stream) is out
        result = np.frombuffer(out.download(), dtype='f4')
        np.testing.assert_allclose(result, ordered_f32(*args, scale), rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(result, reference_f64(*args, scale), rtol=3e-5, atol=3e-6)


def test_hybrid_index_scores_real_device(context):
    backend = GPUBackend()
    try:
        args = inputs(3, 5, 33)
        result = backend.index_scores(*args)
        np.testing.assert_allclose(result, ordered_f32(*args, 33**-.5), rtol=3e-5, atol=3e-6)
        assert backend.index_scores(*inputs(tokens=0)).shape == (0,)
    finally:
        backend.close()
