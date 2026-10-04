#!/usr/bin/env python3
"""Numerical tests execute the compiled Vx shared library via its C ABI.

No Python/native fallback: missing library is a failure. NumPy is only the oracle.
Usage: python kernels/test_kernels.py [kernels/build/libglm_vx.so]
"""
import ctypes as C
import json
from pathlib import Path
import sys
import numpy as np

F = C.POINTER(C.c_float)
I = C.POINTER(C.c_int32)
S = C.c_int32
R = C.c_float
SIGNATURES = {
    'rmsnorm': [F, F, F, S, R],
    'layernorm': [F, F, F, F, S, R],
    'softmax': [F, F, S],
    'swiglu': [F, F, F, S],
    'add': [F, F, F, S],
    'axpy': [F, F, R, S],
    'matvec': [F, F, F, S, S],
    'matmul_nt': [F, F, F, S, S, S],
    'rope': [F, F, F, F, S, S, S],
    'rope_glm': [F, F, F, F, S, S],
    'rope_offset': [F, F, F, F, S, S, S, S],
    'index_scores': [F, F, F, F, S, S, S, R],
    'router': [I, F, F, F, S, S, R],
    'topk': [I, F, S, S],
    'attention': [F, F, F, F, F, S, S, S, R],
    'sparse_attention': [F, F, F, F, F, I, S, S, S, S, R],
}


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / 'build/libglm_vx.so').resolve()
    lib = C.CDLL(str(path))
    for name, sig in SIGNATURES.items():
        fn = getattr(lib, 'glm_vx_' + name)
        fn.argtypes, fn.restype = sig, S
    rng = np.random.default_rng(530)
    checks = []
    max_error = 0.0

    def call(name, *args, status=0):
        converted = [a.ctypes.data_as(I if a.dtype == np.int32 else F) if isinstance(a, np.ndarray) else a for a in args]
        actual = getattr(lib, 'glm_vx_' + name)(*converted)
        assert actual == status, (name, actual, status)

    def check(name, out, expected, atol=3e-5, rtol=3e-5):
        nonlocal max_error
        np.testing.assert_allclose(out, expected, atol=atol, rtol=rtol, err_msg=name)
        max_error = max(max_error, float(np.max(np.abs(out.astype(float) - expected))))
        checks.append(name)

    def rand(*shape):
        return rng.normal(size=shape).astype(np.float32)

    for n in [1, 7, 512, 2048, 6144]:
        x, w, out = rand(n), rand(n), np.empty(n, np.float32)
        ref = x.astype(float) / np.sqrt(np.mean(x.astype(float)**2) + 1e-5) * w
        call('rmsnorm', out, x, w, n, 1e-5)
        check(f'rmsnorm-{n}', out, ref)
        inplace = x.copy()
        call('rmsnorm', inplace, inplace, w, n, 1e-5)
        check(f'rmsnorm-inplace-{n}', inplace, ref)
    for n in [1, 128]:
        x, w, b, out = rand(n), rand(n), rand(n), np.empty(n, np.float32)
        ref = (x.astype(float)-x.astype(float).mean()) / np.sqrt(x.astype(float).var()+1e-5) * w + b
        call('layernorm', out, x, w, b, n, 1e-5)
        check(f'layernorm-{n}', out, ref)
    for x in [rand(1), rand(257), np.array([1000,1001,999,-1000],np.float32), np.array([-np.inf,0,-np.inf],np.float32)]:
        ref = np.exp(x.astype(float)-np.max(x)); ref /= ref.sum()
        out = np.empty_like(x)
        call('softmax',out,x,x.size); check(f'softmax-{len(checks)}',out,ref)
        call('softmax',x,x,x.size); check(f'softmax-inplace-{len(checks)}',x,ref)
    g = np.array([-1000,-100,-10,-1,0,1,10,100,1000],np.float32)
    u, out = rand(g.size), np.empty_like(g)
    call('swiglu',out,g,u,g.size)
    check('swiglu-extremes',out,g*np.exp(-np.logaddexp(0,-g.astype(float)))*u)
    a,b,out = rand(23),rand(23),np.empty(23,np.float32)
    call('add',out,a,b,23); check('add',out,a+b)
    ref=out.copy()+0.3*a
    call('axpy',out,a,0.3,23); check('axpy',out,ref)
    for rows,cols in [(1,1),(7,13),(256,192),(8,6144)]:
        w,x,out=rand(rows,cols),rand(cols),np.empty(rows,np.float32)
        call('matvec',out,w,x,rows,cols)
        check(f'matvec-{rows}x{cols}',out,w.astype(float)@x.astype(float),atol=8e-4)
    a,b,out=rand(3,17),rand(5,17),np.empty((3,5),np.float32)
    call('matmul_nt',out,a,b,3,5,17);check('matmul-nt',out,a.astype(float)@b.astype(float).T)
    for heads,dim,rd,pos in [(1,8,4,0),(64,256,64,10000),(32,128,64,1048575)]:
        x,out=rand(heads,dim),np.empty((heads,dim),np.float32)
        theta=8000000.0
        angles=pos*theta**(-np.arange(0,rd,2,dtype=float)/rd)
        c,s=np.cos(angles).astype(np.float32),np.sin(angles).astype(np.float32)
        ref=x.copy(); start=dim-rd
        ref[:,start::2]=x[:,start::2]*c-x[:,start+1::2]*s
        ref[:,start+1::2]=x[:,start+1::2]*c+x[:,start::2]*s
        call('rope',out,x,c,s,heads,dim,rd);check(f'rope-{heads}-{pos}',out,ref)
        call('rope',x,x,c,s,heads,dim,rd);check(f'rope-inplace-{heads}-{pos}',x,ref)
    x=rand(32,128); out=np.empty_like(x)
    c,s=rand(32),rand(32)
    ref=x.copy(); ref[:,:64:2]=x[:,:64:2]*c-x[:,1:64:2]*s; ref[:,1:64:2]=x[:,1:64:2]*c+x[:,:64:2]*s
    call('rope_offset',out,x,c,s,32,128,64,0);check('indexer-rope-prefix',out,ref)
    q,keys,weights=rand(32,128),rand(37,128),rand(32)/np.sqrt(np.float32(32))
    out=np.empty(37,np.float32)
    call('index_scores',out,q,keys,weights,37,32,128,128**-0.5)
    ref=weights.astype(float)@np.maximum((q.astype(float)@keys.astype(float).T)*128**-0.5,0)
    check('indexer-scores',out,ref)
    x=rand(32,64); out=np.empty_like(x); c,s=rand(32),rand(32)
    ref=np.concatenate((x[:,::2]*c-x[:,1::2]*s,x[:,1::2]*c+x[:,::2]*s),axis=-1)
    call('rope_glm',out,x,c,s,32,64);check('glm-rope-packed',out,ref)
    routing_cases=[(rand(256),rand(256),8),
        (np.array([4,-4,0,-2],np.float32),np.array([0,2,0,0],np.float32),2),
        (np.zeros(8,np.float32),np.zeros(8,np.float32),8),
        (np.full(8,-1000,np.float32),np.zeros(8,np.float32),3),
        (np.full(8,-46,np.float32),np.zeros(8,np.float32),3)]
    for case,(logits,bias,k) in enumerate(routing_cases):
        ids=np.empty(k,np.int32); weights=np.empty(k,np.float32)
        probs=np.exp(-np.logaddexp(0,-logits.astype(float)))
        expected_ids=np.argsort(-(probs+bias),kind='stable')[:k]
        call('router',ids,weights,logits,bias,logits.size,k,2.5)
        np.testing.assert_array_equal(ids,expected_ids)
        ref=probs[expected_ids]/(probs[expected_ids].sum()+1e-20)*2.5
        check(f'router-{case}',weights,ref)
    scores=np.array([-3,5,5,-9,2],np.float32); ids=np.empty(4,np.int32)
    call('topk',ids,scores,5,4);np.testing.assert_array_equal(ids,[1,2,4,0]);checks.append('topk-ties')
    for tokens,dim,vd in [(1,8,5),(17,256,256)]:
        q,keys,values=rand(dim),rand(tokens,dim),rand(tokens,vd)
        out,scratch=np.empty(vd,np.float32),np.empty(tokens,np.float32)
        scale=dim**-0.5
        scores=keys.astype(float)@q.astype(float)*scale
        probs=np.exp(scores-scores.max());probs/=probs.sum()
        call('attention',out,scratch,q,keys,values,tokens,dim,vd,scale)
        check(f'attention-{tokens}',out,probs@values.astype(float))
        selected=np.arange(0,tokens,2,dtype=np.int32)
        ss=scores[selected];pp=np.exp(ss-ss.max());pp/=pp.sum()
        call('sparse_attention',out,scratch,q,keys,values,selected,selected.size,tokens,dim,vd,scale)
        check(f'sparse-attention-{tokens}',out,pp@values[selected].astype(float))
        selected[0]=tokens
        before=out.copy()
        call('sparse_attention',out,scratch,q,keys,values,selected,selected.size,tokens,dim,vd,scale,status=-1)
        np.testing.assert_array_equal(out,before);checks.append(f'sparse-bounds-{tokens}')
    out=np.empty(1,np.float32)
    call('softmax',out,out,0,status=-1)
    call('rmsnorm',out,out,out,1,0.0,status=-1)
    call('rope',out,out,out,out,1,1,1,status=-1)
    checks.append('invalid-dimensions')
    print(json.dumps({'library':str(path),'passed':len(checks),'checks':checks,'max_absolute_error':max_error},indent=2))

if __name__=='__main__':main()
