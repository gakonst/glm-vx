"""CPU equations/SIMT/launch contracts; these never certify device execution."""
import ctypes as C
import numpy as np
import pytest

from gpu.gemm import GEMMKernels, gemm_plan
from gpu.runtime import CUDADriver, CUDAContext, CUDAUnavailable
from gpu.test_runtime import FakeLibrary, write, value
from gpu.test_simt import Simt


def tf32_rna(x):
    bits = np.asarray(x, dtype='f4').view('u4')
    return ((bits + np.uint32(0x1000)) & np.uint32(0xffffe000)).view('f4')


@pytest.mark.parametrize('shape', [(16,8,8), (1,1,1), (17,9,13), (31,17,25),
                                   (2,3,0), (3,11,65)])
@pytest.mark.parametrize('mode', ['f32', 'tf32_rna'])
def test_compiled_vx_simt_gemm(shape, mode):
    m,n,k = shape
    rng = np.random.default_rng(412)
    a = rng.normal(size=(m,k)).astype('f4')
    b = rng.normal(size=(k,n)).astype('f4')
    # Guard regions check masked output stores in partial M/N tiles.
    storage = np.full(m*n+16, 123456.0, dtype='f4')
    out = storage[8:-8].reshape(m,n)
    plan = gemm_plan(m,n,k,precision=mode)
    Simt().run(plan.entry,out,a,b,m,n,k,grid=plan.grid,block=plan.block)
    aa,bb = (tf32_rna(a),tf32_rna(b)) if mode == 'tf32_rna' else (a,b)
    expected = aa.astype('f8') @ bb.astype('f8')
    np.testing.assert_allclose(out,expected,rtol=2e-5,atol=1e-5)
    np.testing.assert_array_equal(storage[:8],123456.)
    np.testing.assert_array_equal(storage[-8:],123456.)


def test_tf32_rounding_and_lane_mapping_basis():
    # Both signs at an exact halfway boundary: TF32 RNA differs from ties-even.
    a = np.zeros((16,8), 'f4')
    a[0,0] = 1 + 2**-11
    a[8,4] = -1 - 2**-11
    b = np.zeros((8,8), 'f4')
    b[0,1] = 1
    b[4,6] = 1
    out = np.empty((16,8), 'f4')
    Simt().run('glm_vx_gpu_gemm_tf32',out,a,b,16,8,8,block=32)
    expected = np.zeros_like(out)
    expected[0,1],expected[8,6] = 1+2**-10,-1-2**-10
    np.testing.assert_array_equal(out,expected)


@pytest.mark.parametrize('m,n,k,block', [(16,8,8,64), (0,8,8,32),
                                        (16,0,8,32), (16,8,-1,32)])
def test_uniform_invalid_kernel_guards(m,n,k,block):
    a=np.zeros((16,8),'f4');b=np.zeros((8,8),'f4');out=np.full((16,8),77.,'f4')
    Simt().run('glm_vx_gpu_gemm_tf32',out,a,b,m,n,k,grid=2,block=block)
    np.testing.assert_array_equal(out,77.)


def test_extra_grid_cta_returns_uniformly():
    a=np.ones((1,1),'f4');b=np.ones((1,1),'f4');out=np.zeros((1,1),'f4')
    Simt().run('glm_vx_gpu_gemm_tf32',out,a,b,1,1,1,grid=3,block=32)
    np.testing.assert_array_equal(out,1.)


def test_plan_residual_and_empty_shapes():
    plan=gemm_plan(17,9,13,precision='tf32_rna')
    assert (plan.grid,plan.block,plan.tile,plan.shared_bytes)==(4,32,(16,8,8),1280)
    assert gemm_plan(17,9,13).entry.endswith('_f32')
    assert gemm_plan(0,9,13,precision='tf32_rna').grid==0
    assert gemm_plan(17,0,13).grid==0
    # 64-bit addresses are accepted without allocating huge tensors.
    assert gemm_plan(65536,65536,1,precision='tf32_rna').grid==33554432


@pytest.mark.parametrize('shape,kwargs', [((-1,8,8),{}), ((True,8,8),{}),
    ((2**31,1,1),{}), ((2**31-1,2**31-1,1),{}), ((17,9,13),{'precision':'auto'}),
    ((17,9,13),{'precision':'tf32_rna','max_grid':3})])
def test_plan_rejects_overflow_and_unknown_modes(shape,kwargs):
    with pytest.raises((ValueError,TypeError)):gemm_plan(*shape,**kwargs)


class GEMMLibrary(FakeLibrary):
    def dispatch(self,name,args):
        if name=='cuMemAlloc_v2':
            self.next_address=(self.next_address+255)//256*256
        if name=='cuFuncGetAttribute' and args[1]==1:
            write(args[0],C.c_int32,1280 if self.kernel_names[value(args[2])].endswith('_tf32') else 0)
            return 0
        return super().dispatch(name,args)


@pytest.fixture
def mock_gemm():
    library=GEMMLibrary()
    with CUDAContext(_driver=CUDADriver(_library=library)) as ctx:
        module=ctx.load_ptx_bytes('.version 7.6\n.target sm_80\n.address_size 64\n')
        yield ctx,GEMMKernels(module),library


def test_resident_launch_precision_stream_and_abi(mock_gemm):
    ctx,gemm,library=mock_gemm
    a=ctx.tensor((17,13));b=ctx.tensor((13,9));stream=ctx.stream()
    captured=[]
    def inspect(args):
        params=args[9]
        captured.append([C.cast(params[i],C.POINTER(C.c_int32))[0] for i in range(3,6)])
    library.on_launch=inspect
    out=gemm.gemm(a,b,precision='tf32_rna',stream=stream)
    assert out.shape==(17,9)
    assert library.launches[-1]==('glm_vx_gpu_gemm_tf32',(4,1,1,32,1,1),stream.handle.value)
    assert captured==[[17,9,13]]
    gemm.gemm(a,b,out=out)
    assert library.launches[-1][0]=='glm_vx_gpu_gemm_f32'
    assert library.launches[-1][1]==(2,1,1,128,1,1)
    assert 'cuMemcpyDtoH_v2' not in library.names()


def test_host_rejects_alias_shape_dtype_and_closed(mock_gemm):
    ctx,gemm,library=mock_gemm
    a=ctx.tensor((8,8));b=ctx.tensor((8,8))
    for kwargs in ({'out':a},{'out':b},{'out':ctx.tensor((7,8))},{'precision':'tf32'}):
        with pytest.raises(ValueError):gemm.gemm(a,b,**kwargs)
    with pytest.raises(ValueError):gemm.gemm(a,ctx.tensor((7,8)))
    with pytest.raises(ValueError):gemm.gemm(a,ctx.tensor((8,8),'float16'))
    assert not library.launches
    b.close()
    with pytest.raises(RuntimeError):gemm.gemm(a,b)


def test_empty_and_zero_k_host_contract(mock_gemm):
    ctx,gemm,library=mock_gemm
    out=gemm.gemm(ctx.tensor((0,3)),ctx.tensor((3,7)),precision='tf32_rna')
    assert out.shape==(0,7) and not library.launches
    gemm.gemm(ctx.tensor((5,0)),ctx.tensor((0,9)),precision='tf32_rna')
    assert library.launches[-1][1]==(2,1,1,32,1,1)


def test_rejects_insufficient_shared_module():
    with CUDAContext(_driver=CUDADriver(_library=FakeLibrary())) as ctx:
        module=ctx.load_ptx_bytes('mock')
        with pytest.raises(ValueError,match='1280'):GEMMKernels(module)


def test_rejects_pre_ampere():
    with CUDAContext(_driver=CUDADriver(_library=GEMMLibrary(capability=(7,5)))) as ctx:
        with pytest.raises(CUDAUnavailable):GEMMKernels(ctx.load_ptx_bytes('mock'))


def test_host_accepts_four_byte_not_sixteen_alignment(mock_gemm):
    ctx,gemm,library=mock_gemm
    a=ctx.tensor((65,)).view((8,8),offset=4)
    b=ctx.tensor((65,)).view((8,8),offset=4)
    out=ctx.tensor((65,)).view((8,8),offset=4)
    assert a.pointer%16==4
    assert gemm.gemm(a,b,out=out,precision='tf32_rna') is out


def test_host_rejects_other_context_and_grid_before_allocation(mock_gemm):
    ctx,gemm,library=mock_gemm
    a=ctx.tensor((17,13));b=ctx.tensor((13,9))
    with CUDAContext(_driver=CUDADriver(_library=GEMMLibrary())) as other:
        with pytest.raises(ValueError,match='different CUDA context'):
            gemm.gemm(a,other.tensor((13,9)))
        with pytest.raises(ValueError,match='different CUDA context'):
            gemm.gemm(a,b,stream=other.stream())
    count=len(library.allocations)
    ctx.max_grid=(3,65535,65535)
    with pytest.raises(ValueError,match='GEMM grid'):gemm.gemm(a,b,precision='tf32_rna')
    assert len(library.allocations)==count
