"""Compatibility backend executing numerical primitives in Vx PTX on CUDA.

This adapter is deliberately named hybrid: existing Python model orchestration
still reads activations on the host between primitive calls. Read-only weight
arrays can be cached on the device with a bounded LRU. For fully resident kernel
pipelines use gpu.runtime DeviceTensor/GLMKernels directly. This is not an
optimized end-to-end GPU serving graph and never falls back to CPU arithmetic.
"""
from collections import OrderedDict
import ctypes as C
import hashlib
import json
from pathlib import Path
import numpy as np
from .runtime import CUDAContext
from kernels.backend import _cached_rope_coefficients, _make_rope_coefficients

I=C.c_int32
F=C.c_float
_MAX=(1<<31)-1

def scalar_f32(value,name,positive=False):
    value=C.c_float(float(value)).value
    if not np.isfinite(value) or (value<=0 if positive else value<0):
        raise ValueError(name+' must remain finite and '+('positive' if positive else 'nonnegative')+' in float32')
    return value

def array(x,name='input'):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim<1 or x.size==0 or x.size>_MAX-1024:raise ValueError(name+' must be nonempty and int32-indexable')
    return np.ascontiguousarray(x)

class GPUBackend:
    name='vx-gpu-hybrid-f32'
    def __init__(self,ptx_dir=None,device=0,weight_cache_bytes=512*1024**2,tuning_file=None):
        if type(weight_cache_bytes) is not int or weight_cache_bytes<0:raise ValueError('weight cache size must be nonnegative')
        self.ctx=CUDAContext(device,minimum_compute_capability=(8,0))
        self.cache=OrderedDict();self.cache_bytes=0;self.cache_limit=weight_cache_bytes;self.pinned=set()
        self._closed=False
        try:
            ptx_path=Path(ptx_dir or Path(__file__).parent/'build')/'kernels.ptx'
            self.tuning=None
            if tuning_file is not None:
                with open(tuning_file) as source:plan=json.load(source)
                expected={'schema':'glm-vx-matvec-tuning-v1','gpu_execution_verified':True,
                    'device':self.ctx.name,'compute_capability':list(self.ctx.compute_capability),
                    'driver_version':self.ctx.driver_version,'dtype':'float32',
                    'ptx_sha256':hashlib.sha256(ptx_path.read_bytes()).hexdigest()}
                if any(plan.get(k)!=v for k,v in expected.items()):raise ValueError('tuning plan does not match this GPU, driver or PTX')
                shape=plan.get('shape')
                if not isinstance(shape,list) or len(shape)!=2 or any(type(d) is not int or d<1 for d in shape):raise ValueError('invalid tuning shape')
                if type(plan.get('selected_block')) is not int or plan['selected_block'] not in (32,64,128,256):raise ValueError('invalid tuned block size')
                self.tuning=plan
            self.module=self.ctx.load_ptx(ptx_path)
            self.stream=self.ctx.stream()
            names=('matvec','rmsnorm','softmax','swiglu','rope_glm','router','mla_partial','mla_merge')
            self.functions={name:self.module.kernel('glm_vx_gpu_'+name) for name in names}
        except BaseException:self.ctx.close();raise

    def close(self):
        if not self._closed:
            self.ctx.close();self.cache.clear();self.cache_bytes=0;self._closed=True

    def _upload(self,value,temps,cache=False):
        x=array(value)
        # Caching is only legal for caller-declared immutable storage. Hold the
        # source alive: raw addresses cannot be reused under an existing key.
        key=None
        if cache and not x.flags.writeable and x.nbytes<=self.cache_limit:
            key=(x.ctypes.data,x.shape,x.strides)
            if key in self.cache:
                source,tensor=self.cache.pop(key);self.cache[key]=(source,tensor)
                self.pinned.add(key)
                return tensor
        if key is not None:
            while self.cache_bytes+x.nbytes>self.cache_limit:
                candidates=[k for k in self.cache if k not in self.pinned]
                if not candidates:
                    key=None
                    break
                source,old=self.cache.pop(candidates[0]);self.cache_bytes-=source.nbytes;old.close()
        tensor=self.ctx.tensor(x.shape,'float32',x.tobytes())
        if key is None:temps.append(tensor)
        else:
            self.cache[key]=(x,tensor);self.cache_bytes+=x.nbytes;self.pinned.add(key)
        return tensor

    def _out(self,shape,temps,dtype='float32'):
        tensor=self.ctx.tensor(shape,dtype);temps.append(tensor);return tensor
    def _read(self,tensor):
        return np.frombuffer(tensor.download(),dtype=tensor.dtype).reshape(tensor.shape).copy()
    def _launch(self,name,grid,block,args):
        if self._closed:raise RuntimeError('GPU backend is closed')
        self.functions[name].launch((grid,),(block,),args,stream=self.stream)
    def _release(self,temps):
        # Explicit synchronization before releasing temporaries, including on
        # a failed launch. A sync error propagates and prevents unsafe frees.
        self.stream.synchronize()
        for tensor in reversed(temps):tensor.close()
        self.pinned.clear()

    def matvec(self,w,x):
        w,x=array(w,'weight'),array(x)
        if w.ndim!=2 or x.ndim!=1 or w.shape[1]!=x.size:raise ValueError('matvec shape mismatch')
        rows,cols=w.shape;temps=[]
        try:
            wd,xd=self._upload(w,temps,True),self._upload(x,temps);out=self._out((rows,),temps)
            block=self.tuning['selected_block'] if self.tuning and self.tuning['shape']==[rows,cols] else 128
            self._launch('matvec',(rows+block//32-1)//(block//32),block,[out,wd,xd,I(rows),I(cols)])
            return self._read(out)
        finally:self._release(temps)

    def mlp(self,gate,up,down,x):
        """Run down(SwiGLU(gate(x), up(x))) with resident intermediates.

        Upload all inputs before launching so blocking transfers cannot split
        the chain. Immutable weights share the bounded cache and stay pinned
        until all four launches finish; only the final projection is read back.
        """
        gate,up,down,x=array(gate,'gate weight'),array(up,'up weight'),array(down,'down weight'),array(x)
        if (gate.ndim!=2 or up.shape!=gate.shape or down.ndim!=2 or x.ndim!=1
                or gate.shape[1]!=x.size or down.shape[1]!=gate.shape[0]):
            raise ValueError('MLP shape mismatch')
        width=gate.shape[0];temps=[]
        try:
            gd,ud,dd=[self._upload(w,temps,True) for w in (gate,up,down)]
            xd=self._upload(x,temps)
            gated=self._out((width,),temps);upped=self._out((width,),temps)
            activated=self._out((width,),temps);out=self._out((down.shape[0],),temps)
            for weight,source,destination,shape in (
                    (gd,xd,gated,gate.shape),(ud,xd,upped,up.shape)):
                rows,cols=shape
                block=self.tuning['selected_block'] if self.tuning and self.tuning['shape']==[rows,cols] else 128
                self._launch('matvec',(rows+block//32-1)//(block//32),block,
                             [destination,weight,source,I(rows),I(cols)])
            self._launch('swiglu',(width+255)//256,256,[activated,gated,upped,I(width)])
            rows,cols=down.shape
            block=self.tuning['selected_block'] if self.tuning and self.tuning['shape']==[rows,cols] else 128
            self._launch('matvec',(rows+block//32-1)//(block//32),block,[out,dd,activated,I(rows),I(cols)])
            return self._read(out)
        finally:self._release(temps)

    def rmsnorm(self,x,w,eps):
        x,w=array(x),array(w,'weight');eps=scalar_f32(eps,'epsilon',positive=True)
        if w.ndim!=1 or w.size!=x.shape[-1] or not np.isfinite(eps) or eps<=0:raise ValueError('invalid RMSNorm shape/epsilon')
        dim=w.size;rows=x.size//dim;temps=[]
        try:
            xd,wd=self._upload(x,temps),self._upload(w,temps,True);out=self._out(x.shape,temps)
            for row in range(rows):
                offset=row*dim*4
                self._launch('rmsnorm',1,256,[out.view((dim,),offset=offset),xd.view((dim,),offset=offset),wd,I(dim),F(eps)])
            return self._read(out)
        finally:self._release(temps)

    def softmax(self,x):
        x=array(x);dim=x.shape[-1];rows=x.size//dim
        if np.isnan(x).any() or np.isposinf(x).any() or not np.isfinite(x).any(axis=-1).all():raise ValueError('invalid softmax values')
        temps=[]
        try:
            xd=self._upload(x,temps);out=self._out(x.shape,temps)
            for row in range(rows):
                offset=row*dim*4
                self._launch('softmax',1,256,[out.view((dim,),offset=offset),xd.view((dim,),offset=offset),I(dim)])
            return self._read(out)
        finally:self._release(temps)

    def swiglu(self,gate,up):
        gate,up=array(gate),array(up)
        if gate.shape!=up.shape:raise ValueError('SwiGLU shape mismatch')
        temps=[]
        try:
            gd,ud=self._upload(gate,temps),self._upload(up,temps);out=self._out(gate.shape,temps)
            self._launch('swiglu',(gate.size+255)//256,256,[out,gd,ud,I(gate.size)])
            return self._read(out)
        finally:self._release(temps)

    def rope(self,x,position,theta):
        x=array(x);dim=x.shape[-1]
        if dim%2 or not np.isfinite(position) or not np.isfinite(theta) or theta<=0:raise ValueError('invalid RoPE parameters')
        coefficients=_cached_rope_coefficients if dim<=4096 else _make_rope_coefficients
        cosine,sine=coefficients(dim,float(position),float(theta));temps=[]
        if not np.isfinite(cosine).all() or not np.isfinite(sine).all():raise ValueError('RoPE coefficients overflow float32')
        try:
            xd,cd,sd=self._upload(x,temps),self._upload(cosine,temps,True),self._upload(sine,temps,True);out=self._out(x.shape,temps)
            self._launch('rope_glm',(x.size//2+255)//256,256,[out,xd,cd,sd,I(x.size//dim),I(dim)])
            return self._read(out)
        finally:self._release(temps)

    def route(self,logits,bias,k,scale):
        logits,bias=array(logits),array(bias)
        scale=scalar_f32(scale,'router scale')
        if logits.ndim!=1 or bias.shape!=logits.shape or type(k) is not int or not 0<k<=min(8,logits.size) or logits.size>256:
            raise ValueError('GPU router requires <=256 experts, 1<=k<=8 and matching vector shapes')
        if not np.isfinite(logits).all() or not np.isfinite(bias).all() or not np.isfinite(scale) or scale<0:raise ValueError('invalid router values')
        temps=[]
        try:
            ld,bd=self._upload(logits,temps),self._upload(bias,temps,True)
            ids=self._out((k,),temps,'int32');weights=self._out((k,),temps)
            self._launch('router',1,32,[ids,weights,ld,bd,I(logits.size),I(k),F(scale)])
            return self._read(ids),self._read(weights)
        finally:self._release(temps)

    def compressed_attention(self,q_latent,q_rope,latents,rope_keys,scale):
        """Fused all-head attention over already-selected compressed KV rows.

        The model has applied causal DSA selection before this call. This hybrid
        entry uploads one gathered cache; resident callers can avoid that copy
        using GLMKernels.mla_decode directly.
        """
        q_latent,q_rope,latents,rope_keys=map(array,(q_latent,q_rope,latents,rope_keys))
        scale=scalar_f32(scale,'MLA scale',positive=True)
        if any(x.ndim!=2 for x in (q_latent,q_rope,latents,rope_keys)):raise ValueError('MLA expects matrices')
        heads,rank=q_latent.shape;count=latents.shape[0];rdim=q_rope.shape[1]
        if rank>1024 or q_rope.shape[0]!=heads or latents.shape[1]!=rank or rope_keys.shape!=(count,rdim):raise ValueError('MLA shape mismatch')
        if not np.isfinite(scale) or scale<=0:raise ValueError('MLA scale must be positive and finite')
        splits=min(16,max(1,(count+255)//256));block=((rank+31)//32)*32
        temps=[]
        try:
            qd,qdrot,kd,kdrot=[self._upload(x,temps) for x in (q_latent,q_rope,latents,rope_keys)]
            ids=self.ctx.tensor((count,),'int32',np.arange(count,dtype=np.int32).tobytes());temps.append(ids)
            numerator=self._out((heads,splits,rank),temps);stats=self._out((heads,splits,2),temps);out=self._out((heads,rank),temps)
            self._launch('mla_partial',heads*splits,block,[numerator,stats,qd,qdrot,kd,kdrot,ids,I(heads),I(rank),I(rdim),I(count),I(count),I(0),I(splits),F(scale)])
            self._launch('mla_merge',heads,256,[out,numerator,stats,I(heads),I(rank),I(splits)])
            return self._read(out)
        finally:self._release(temps)
