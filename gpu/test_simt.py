"""Same compiled Vx GPU kernels under test-only CPU lanes, NOT GPU validation."""
import ctypes as C
from pathlib import Path
import numpy as np
import pytest

LIB=Path(__file__).parent/'build/libsimt.so'

class Simt:
    def __init__(self):
        if not LIB.exists():raise RuntimeError('build gpu/testing/build_simt.sh before GPU arithmetic tests')
        self.lib=C.CDLL(str(LIB))
        self.entry=C.CFUNCTYPE(None,C.c_void_p)
        self.lib.simt_launch.argtypes=[self.entry,C.c_void_p,C.c_int32,C.c_int32]
        self.lib.simt_launch.restype=C.c_int32
    def run(self,name,*args,grid=1,block=32):
        values=[];types=[]
        for arg in args:
            if isinstance(arg,np.ndarray):
                assert arg.flags.c_contiguous
                values.append(C.c_void_p(arg.ctypes.data));types.append(C.c_void_p)
            elif isinstance(arg,float):values.append(C.c_float(arg));types.append(C.c_float)
            else:values.append(C.c_int32(arg));types.append(C.c_int32)
        f=getattr(self.lib,name);f.argtypes=types;f.restype=None
        errors=[]
        @self.entry
        def entry(_):
            try:f(*values)
            except BaseException as e:errors.append(e)
        assert self.lib.simt_launch(entry,None,grid,block)==0
        assert not errors

@pytest.fixture(scope='module')
def simt():return Simt()

@pytest.mark.parametrize('shape',[(7,131),(3,6144)])
def test_matvec_coalesced_tail(simt,shape):
    rng=np.random.default_rng(4);w=rng.normal(size=shape).astype('f4');x=rng.normal(size=shape[1]).astype('f4');out=np.zeros(shape[0],'f4')
    simt.run('glm_vx_gpu_matvec',out,w,x,*shape,grid=(shape[0]+1)//2,block=64)
    np.testing.assert_allclose(out,w.astype('f8')@x.astype('f8'),rtol=1e-5,atol=3e-5)

def test_residual_rmsnorm_multiwarp(simt):
    rng=np.random.default_rng(2);x=rng.normal(size=(2,67)).astype('f4');r=rng.normal(size=x.shape).astype('f4');w=rng.normal(size=67).astype('f4')
    out=np.empty_like(x);res=np.empty_like(x)
    simt.run('glm_vx_gpu_residual_rmsnorm',out,res,x,r,w,2,67,1e-5,grid=2,block=64)
    np.testing.assert_array_equal(res,x+r)
    expected=(x+r)/np.sqrt(np.mean((x+r)**2,axis=1,keepdims=True)+1e-5)*w
    np.testing.assert_allclose(out,expected,rtol=2e-6,atol=2e-6)

def test_rope_and_swiglu_tail(simt):
    rng=np.random.default_rng(23);x=rng.normal(size=(3,8)).astype('f4');angle=np.array([.2,.3,.5,.7],dtype='f4');co=np.cos(angle);si=np.sin(angle);out=np.empty_like(x)
    simt.run('glm_vx_gpu_rope_glm',out,x,co,si,3,8,block=32)
    expected=np.concatenate((x[:,::2]*co-x[:,1::2]*si,x[:,1::2]*co+x[:,::2]*si),axis=1)
    np.testing.assert_allclose(out,expected,rtol=1e-6,atol=1e-6)
    gate=rng.normal(size=103).astype('f4');up=rng.normal(size=103).astype('f4');out=np.empty_like(gate)
    simt.run('glm_vx_gpu_swiglu',out,gate,up,103,grid=2,block=32)
    np.testing.assert_allclose(out,gate/(1+np.exp(-gate))*up,rtol=2e-6,atol=1e-6)

def test_fp8_coalesced_scale_boundary(simt):
    from glm_vx.checkpoint import FP8_E4M3FN_TABLE
    rng=np.random.default_rng(7);rows,cols=3,257
    w=rng.integers(0,255,(rows,cols),dtype=np.uint8);w[w==127]=126;w[w==255]=254
    scales=np.array([[.01,.02,.03]],dtype='f4');x=rng.normal(size=cols).astype('f4');out=np.empty(rows,'f4')
    simt.run('glm_vx_gpu_fp8_matvec',out,w,scales,x,rows,cols,grid=2,block=64)
    decoded=FP8_E4M3FN_TABLE[w]*scales[0,np.arange(cols)//128]
    np.testing.assert_allclose(out,decoded.astype('f8')@x.astype('f8'),rtol=2e-5,atol=3e-5)

@pytest.mark.parametrize('count,splits',[(5,3),(0,3),(5,8)])
def test_split_online_mla_matches_dense_selected_oracle(simt,count,splits):
    rng=np.random.default_rng(23);heads,rank,rope,tokens=2,48,4,7
    q=rng.normal(size=(heads,rank)).astype('f4');qr=rng.normal(size=(heads,rope)).astype('f4')
    keys=rng.normal(size=(tokens,rank)).astype('f4');kr=rng.normal(size=(tokens,rope)).astype('f4')
    selected=np.array([0,-1,2,6,99],dtype='i4')[:count].copy()
    numerator=np.empty((heads*splits,rank),'f4');stats=np.empty((heads*splits,2),'f4');out=np.empty((heads,rank),'f4');scale=.25
    simt.run('glm_vx_gpu_mla_partial',numerator,stats,q,qr,keys,kr,selected,heads,rank,rope,tokens,count,0,splits,scale,grid=heads*splits,block=64)
    simt.run('glm_vx_gpu_mla_merge',out,numerator,stats,heads,rank,splits,grid=heads,block=64)
    ids=selected[(selected>=0)&(selected<tokens)]
    expected=np.zeros_like(out)
    if len(ids):
        scores=(q.astype('f8')@keys[ids].astype('f8').T+qr.astype('f8')@kr[ids].astype('f8').T)*scale
        probs=np.exp(scores-scores.max(axis=1,keepdims=True));probs/=probs.sum(axis=1,keepdims=True)
        expected=probs@keys[ids]
    np.testing.assert_allclose(out,expected,rtol=2e-5,atol=3e-6)
