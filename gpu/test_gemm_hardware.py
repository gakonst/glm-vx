"""Real GPU GEMM gate: skipped only when CUDA/SM80 hardware is unavailable."""
from pathlib import Path
import numpy as np
import pytest
from gpu.gemm import GEMMKernels
from gpu.runtime import CUDAContext, CUDAUnavailable
from gpu.test_gemm import tf32_rna


@pytest.fixture
def device():
    try:
        ctx=CUDAContext(minimum_compute_capability=(8,0))
    except CUDAUnavailable as error:
        pytest.skip('real GPU unavailable: '+str(error))
    try:
        # Missing PTX, JIT, arithmetic and launch failures are failures on a GPU.
        yield ctx,GEMMKernels(ctx.load_ptx(Path(__file__).parent/'build/gemm.ptx'))
    finally:
        ctx.close()


@pytest.mark.parametrize('shape',[(16,8,8),(17,9,13),(65,33,257),(3,9,0)])
@pytest.mark.parametrize('mode',['f32','tf32_rna'])
def test_gemm_real_device(device,shape,mode):
    ctx,gemm=device
    m,n,k=shape
    rng=np.random.default_rng(741)
    a=rng.normal(size=(m,k)).astype('f4');b=rng.normal(size=(k,n)).astype('f4')
    ad=ctx.tensor(a.shape,data=a);bd=ctx.tensor(b.shape,data=b)
    # Offset by one F32 to verify that only 4-byte alignment is required.
    backing=ctx.tensor((m*n+2,),data=np.full(m*n+2,123456.,'f4'))
    out=backing.view((m,n),offset=4)
    stream=ctx.stream()
    gemm.gemm(ad,bd,precision=mode,out=out,stream=stream)
    stream.synchronize()
    actual=np.frombuffer(out.download(),'f4').reshape(m,n)
    aa,bb=(tf32_rna(a),tf32_rna(b)) if mode=='tf32_rna' else (a,b)
    expected=aa.astype('f8')@bb.astype('f8')
    # Absolute allowance scales with dot-product magnitude and K. This checks
    # rounded-input GEMM separately from error relative to full-precision inputs.
    bound=2e-5*np.maximum(1.,np.abs(aa.astype('f8'))@np.abs(bb.astype('f8')))
    assert np.all(np.abs(actual-expected)<=bound)
    full=a.astype('f8')@b.astype('f8')
    if k:
        assert np.linalg.norm(actual-full)/max(np.linalg.norm(full),1e-30)<(0.002 if mode=='tf32_rna' else 2e-5)
    raw=np.frombuffer(backing.download(),'f4')
    assert raw[0]==123456. and raw[-1]==123456.
