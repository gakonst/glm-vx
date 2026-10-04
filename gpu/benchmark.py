"""Real-device correctness and CUDA event timing. Missing GPU is a hard failure.

python -m gpu.benchmark --ptx-dir gpu/build --kernel all --repeats 20
Inputs/weights stay on device during timing; reported kernel time excludes host
transfers and Python setup. This is NOT whole-model serving throughput.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np
from .runtime import CUDAContext, CUDAUnavailable

I=C.c_int32
F=C.c_float

def timed(ctx,stream,launch,repeats):
    for _ in range(3):launch()
    stream.synchronize()
    samples=[]
    with ctx.event() as begin, ctx.event() as end:
        for _ in range(repeats):
            begin.record(stream);launch();end.record(stream)
            samples.append(begin.elapsed_ms(end))
    return {'repeats':repeats,'median_kernel_ms':statistics.median(samples),'min_kernel_ms':min(samples),'samples_ms':samples}

def check(got,expected,rtol=2e-4,atol=2e-4):
    np.testing.assert_allclose(got,expected,rtol=rtol,atol=atol)
    return {'max_abs_error':float(np.max(np.abs(got-expected))),'rtol':rtol,'atol':atol}

def bench(ctx,path,which,repeats,model_shapes=False):
    rng=np.random.default_rng(2026)
    results=[]
    def upload(x):return ctx.tensor(x.shape,str(x.dtype),x.tobytes())
    def read(x):return np.frombuffer(x.download(),dtype=x.dtype).reshape(x.shape)
    if which != 'fp8':
        with ctx.load_ptx(path/'kernels.ptx') as mod, ctx.stream() as stream:
            if which in ('all','matvec'):
                rows,cols=(6144,6144) if model_shapes else (257,513)
                w=rng.normal(0,.02,(rows,cols)).astype('f4');x=rng.normal(size=cols).astype('f4')
                wd,xd=upload(w),upload(x);out=ctx.tensor((rows,),'float32')
                fn=mod.kernel('glm_vx_gpu_matvec')
                def launch():fn.launch(((rows+3)//4,),(128,),[out,wd,xd,I(rows),I(cols)],stream=stream)
                launch();proof=check(read(out),w.astype('f8')@x.astype('f8'))
                result={'kernel':'matvec','shape':[rows,cols],**proof,**timed(ctx,stream,launch,repeats)}
                result['weight_read_GB_s']=w.nbytes/(result['median_kernel_ms']*1e6)
                results.append(result)
                wd.close();xd.close();out.close()
            if which in ('all','rmsnorm'):
                rows,dim=4,6144
                x=rng.normal(size=(rows,dim)).astype('f4');r=rng.normal(size=x.shape).astype('f4');w=rng.normal(size=dim).astype('f4')
                xd,rd,wd=upload(x),upload(r),upload(w);out=ctx.tensor(x.shape);res=ctx.tensor(x.shape)
                fn=mod.kernel('glm_vx_gpu_residual_rmsnorm')
                def launch():fn.launch((rows,),(256,),[out,res,xd,rd,wd,I(rows),I(dim),F(1e-5)],stream=stream)
                launch();summed=x+r;expected=summed/np.sqrt(np.mean(summed*summed,axis=1,keepdims=True)+1e-5)*w
                proof=check(read(out),expected);np.testing.assert_array_equal(read(res),summed)
                results.append({'kernel':'residual_rmsnorm','shape':[rows,dim],**proof,**timed(ctx,stream,launch,repeats)})
                for t in (xd,rd,wd,out,res):t.close()
            if which in ('all','mla'):
                heads,rank,rdim,tokens,count,splits=(64,512,64,4096,2048,8) if model_shapes else (2,64,8,37,23,4)
                q=rng.normal(0,.1,(heads,rank)).astype('f4');qr=rng.normal(0,.1,(heads,rdim)).astype('f4')
                kv=rng.normal(0,.1,(tokens,rank)).astype('f4');kr=rng.normal(0,.1,(tokens,rdim)).astype('f4');ids=rng.choice(tokens,count,replace=False).astype('i4')
                qd,qrd,kvd,krd,idsd=map(upload,(q,qr,kv,kr,ids));num=ctx.tensor((heads,splits,rank));stats=ctx.tensor((heads,splits,2));out=ctx.tensor((heads,rank))
                partial,merge=mod.kernel('glm_vx_gpu_mla_partial'),mod.kernel('glm_vx_gpu_mla_merge');scale=.0625
                def launch():
                    partial.launch((heads*splits,),(rank,),[num,stats,qd,qrd,kvd,krd,idsd,I(heads),I(rank),I(rdim),I(tokens),I(count),I(0),I(splits),F(scale)],stream=stream)
                    merge.launch((heads,),(256,),[out,num,stats,I(heads),I(rank),I(splits)],stream=stream)
                launch();scores=(q.astype('f8')@kv[ids].astype('f8').T+qr.astype('f8')@kr[ids].astype('f8').T)*scale
                prob=np.exp(scores-scores.max(axis=1,keepdims=True));prob/=prob.sum(axis=1,keepdims=True)
                proof=check(read(out),prob@kv[ids],rtol=5e-4,atol=2e-5)
                results.append({'kernel':'split_compressed_mla','heads':heads,'rank':rank,'tokens':tokens,'selected_tokens':count,'splits':splits,**proof,**timed(ctx,stream,launch,repeats)})
                for t in (qd,qrd,kvd,krd,idsd,num,stats,out):t.close()
    if which in ('all','fp8'):
        from glm_vx.checkpoint import FP8_E4M3FN_TABLE
        with ctx.load_ptx(path/'fp8.ptx') as mod, ctx.stream() as stream:
            rows,cols=(6144,6144) if model_shapes else (129,257)
            w=rng.integers(0,256,(rows,cols),dtype='u1');w[(w&127)==127]=126
            scales=rng.uniform(.001,.003,((rows+127)//128,(cols+127)//128)).astype('f4');x=rng.normal(size=cols).astype('f4')
            wd,sd,xd=map(upload,(w,scales,x));out=ctx.tensor((rows,));fn=mod.kernel('glm_vx_gpu_fp8_matvec')
            def launch():fn.launch(((rows+3)//4,),(128,),[out,wd,sd,xd,I(rows),I(cols)],stream=stream)
            launch();decoded=FP8_E4M3FN_TABLE[w]*scales[np.arange(rows)[:,None]//128,np.arange(cols)[None,:]//128]
            proof=check(read(out),decoded.astype('f8')@x.astype('f8'),rtol=3e-4,atol=3e-4)
            result={'kernel':'fp8_matvec','shape':[rows,cols],**proof,**timed(ctx,stream,launch,repeats)}
            result['weight_read_GB_s']=w.nbytes/(result['median_kernel_ms']*1e6)
            results.append(result)
            for t in (wd,sd,xd,out):t.close()
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ptx-dir',type=Path,default=Path(__file__).parent/'build')
    p.add_argument('--kernel',choices=['all','matvec','rmsnorm','mla','fp8'],default='all')
    p.add_argument('--repeats',type=int,default=20);p.add_argument('--device',type=int,default=0)
    p.add_argument('--model-shapes',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args()
    if not 1<=a.repeats<=10000:p.error('repeats must be in 1..10000')
    try:
        with CUDAContext(a.device,minimum_compute_capability=(8,0)) as ctx:
            result={'status':'executed_on_gpu','device':ctx.name,'compute_capability':ctx.compute_capability,
                'driver_version':ctx.driver_version,'scope':'resident-kernel timing; excludes transfers, host setup and serving',
                'results':bench(ctx,a.ptx_dir,a.kernel,a.repeats,a.model_shapes)}
    except CUDAUnavailable as exc:
        print(json.dumps({'status':'unavailable','gpu_execution_verified':False,'reason':str(exc)}));return 2
    text=json.dumps(result,indent=2);print(text)
    if a.output:a.output.write_text(text+'\n')
    return 0
if __name__=='__main__':sys.exit(main())
