"""Measure warp/block layouts on a real GPU, then emit a shape-specific plan.

No suggested layout is labeled optimal without measurement. This explores only
32/64/128/256-thread matvec layouts, not the whole algorithm/design space.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from .runtime import CUDAContext,CUDAUnavailable
from .benchmark import timed,check

def tune(ctx,directory,rows,cols,fp8,repeats):
    rng=np.random.default_rng(42);name='fp8.ptx' if fp8 else 'kernels.ptx';path=directory/name
    with ctx.load_ptx(path) as module,ctx.stream() as stream:
        x=rng.normal(size=cols).astype('f4');out=ctx.tensor((rows,));xd=ctx.tensor(x.shape,'float32',x.tobytes())
        if fp8:
            from glm_vx.checkpoint import FP8_E4M3FN_TABLE
            w=rng.integers(0,256,(rows,cols),dtype='u1');w[(w&127)==127]=126
            scale=rng.uniform(.001,.003,((rows+127)//128,(cols+127)//128)).astype('f4')
            wd=ctx.tensor(w.shape,'uint8',w.tobytes());sd=ctx.tensor(scale.shape,'float32',scale.tobytes())
            expected=(FP8_E4M3FN_TABLE[w]*scale[np.arange(rows)[:,None]//128,np.arange(cols)[None,:]//128]).astype('f8')@x.astype('f8')
            args=[out,wd,sd,xd,C.c_int32(rows),C.c_int32(cols)];entry='glm_vx_gpu_fp8_matvec'
        else:
            w=rng.normal(0,.02,(rows,cols)).astype('f4');wd=ctx.tensor(w.shape,'float32',w.tobytes())
            expected=w.astype('f8')@x.astype('f8');args=[out,wd,xd,C.c_int32(rows),C.c_int32(cols)];entry='glm_vx_gpu_matvec'
        kernel=module.kernel(entry);candidates=[]
        for block in (32,64,128,256):
            if block>min(kernel.max_threads,ctx.max_threads_per_block):continue
            grid=(rows+block//32-1)//(block//32)
            def launch():kernel.launch((grid,),(block,),args,stream=stream)
            launch();proof=check(np.frombuffer(out.download(),dtype='f4'),expected,rtol=3e-4,atol=3e-4)
            candidates.append({'block':block,'grid':grid,**proof,**timed(ctx,stream,launch,repeats)})
        best=min(candidates,key=lambda x:x['median_kernel_ms'])
        return {'schema':'glm-vx-matvec-tuning-v1','gpu_execution_verified':True,
            'device':ctx.name,'compute_capability':list(ctx.compute_capability),'driver_version':ctx.driver_version,
            'dtype':'fp8-e4m3fn-block128' if fp8 else 'float32','shape':[rows,cols],
            'ptx_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'selected_block':best['block'],
            'selection_scope':'lowest median CUDA event time among tested block sizes for this shape; not global optimality',
            'candidates':candidates}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rows',type=int,default=6144);p.add_argument('--cols',type=int,default=6144)
    p.add_argument('--fp8',action='store_true');p.add_argument('--repeats',type=int,default=20);p.add_argument('--device',type=int,default=0)
    p.add_argument('--ptx-dir',type=Path,default=Path(__file__).parent/'build');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if min(a.rows,a.cols,a.repeats)<1 or a.rows*a.cols>2**31-1024 or a.repeats>10000:p.error('invalid dimensions/repeats')
    try:
        with CUDAContext(a.device,minimum_compute_capability=(8,0)) as ctx:r=tune(ctx,a.ptx_dir,a.rows,a.cols,a.fp8,a.repeats)
    except CUDAUnavailable as exc:
        print(json.dumps({'status':'unavailable','reason':str(exc),'gpu_execution_verified':False}));return 2
    text=json.dumps(r,indent=2);a.output.write_text(text+'\n');print(text);return 0
if __name__=='__main__':sys.exit(main())
