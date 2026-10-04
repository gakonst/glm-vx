"""Test-only runtime-shaped host wrapper for the same compiled GPU Vx kernels.

Never imported by the production runtime or serving code. Deliberately retains
'cpu-simt' in identity; this is not a mocked GPU performance result.
"""
import numpy as np
from gpu.test_simt import Simt
class Tensor:
    def __init__(self,shape,dtype='float32',data=None):
        self.shape=tuple(shape);self.dtype=dtype;self._closed=False;self.parent=None
        self.data=np.zeros(shape,dtype=dtype) if data is None else np.frombuffer(data,dtype=dtype).reshape(shape).copy()
    @property
    def closed(self):return self._closed or (self.parent is not None and self.parent.closed)
    @property
    def nbytes(self):return self.data.nbytes
    def view(self,shape,dtype=None,offset=0):
        assert not self.closed
        view=Tensor.__new__(Tensor);view.shape=tuple(shape);view.dtype=dtype or self.dtype;view._closed=False;view.parent=self
        size=int(np.prod(shape))*np.dtype(view.dtype).itemsize
        view.data=self.data.view(np.uint8).reshape(-1)[offset:offset+size].view(view.dtype).reshape(shape)
        return view
    def download(self):
        assert not self.closed
        return self.data.tobytes()
    def close(self):self._closed=True
class Stream:
    def synchronize(self):pass
class Kernel:
    def __init__(self,ctx,name):self.ctx,self.name=ctx,name
    def launch(self,grid,block,args,stream=None):
        unpacked=[]
        for a in args:
            if isinstance(a,Tensor):assert not a.closed;unpacked.append(a.data)
            else:unpacked.append(a.value)
        self.ctx.simt.run(self.name,*unpacked,grid=grid[0],block=block[0])
class Module:
    def __init__(self,ctx):self.ctx=ctx
    def kernel(self,name):
        getattr(self.ctx.simt.lib,name)
        return Kernel(self.ctx,name)
class Context:
    name='cpu-simt-test-only'
    def __init__(self,*args,**kwargs):self.simt=Simt();self.tensors=[]
    def load_ptx(self,path):return Module(self)
    def stream(self):return Stream()
    def tensor(self,shape,dtype='float32',data=None):
        t=Tensor(shape,dtype,data);self.tensors.append(t);return t
    def close(self):
        for t in self.tensors:t.close()
