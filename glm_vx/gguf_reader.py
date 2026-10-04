"""Read-only split GGUF storage. Quant unpacking uses gguf-py's reference decoder.

Only requested rows/experts are decoded; inference never runs through llama.cpp.
Returned arrays own their memory, so closing the store cannot invalidate them.
"""
from pathlib import Path
import math
import re
import struct
from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np
from .checkpoint import CheckpointError

class GGUFStore:
    def __init__(self,path,*,decode_rows=128,max_decode_bytes=8*1024**3,decode_threads=1):
        import gguf
        if type(decode_rows) is not int or decode_rows<1:raise ValueError('decode_rows must be positive')
        if type(max_decode_bytes) is not int or max_decode_bytes<4:raise ValueError('max_decode_bytes must be positive')
        if type(decode_threads) is not int or not 1<=decode_threads<=32:raise ValueError('decode_threads must be 1..32')
        self.pool=ThreadPoolExecutor(max_workers=decode_threads) if decode_threads>1 else None
        self.readers=[];self.tensors={};self.metadata={};self.closed=False
        self.decode_rows=decode_rows;self.max_decode_bytes=max_decode_bytes
        path=Path(path)
        if path.is_dir():
            first=sorted(path.glob('*-00001-of-*.gguf'))
            if not first:first=sorted(path.glob('*.gguf'))
            if len(first)!=1:raise CheckpointError('directory must identify one GGUF checkpoint')
            path=first[0]
        match=re.fullmatch(r'(.*)-(\d{5})-of-(\d{5})\.gguf',path.name)
        paths=[path]
        if match:
            count=int(match[3])
            if not 1<=count<=1024 or not 1<=int(match[2])<=count:raise CheckpointError('invalid split index/count')
            paths=[path.with_name(f'{match[1]}-{i:05}-of-{count:05}.gguf') for i in range(1,count+1)]
        self.paths=paths
        try:
            total=None
            for index,p in enumerate(paths):
                with p.open('rb') as f:
                    header=f.read(24)
                if len(header)!=24 or header[:4]!=b'GGUF':raise CheckpointError('invalid GGUF header')
                version,nt,nk=struct.unpack('<IQQ',header[4:])
                if version not in (2,3) or nt>1000000 or nk>100000:raise CheckpointError('unsupported GGUF header/counts/endian')
                reader=gguf.GGUFReader(p);self.readers.append(reader)
                if reader.endianess!=gguf.GGUFEndian.LITTLE:raise CheckpointError('only little-endian GGUF is supported')
                fields={k:v.contents() for k,v in reader.fields.items() if not k.startswith('GGUF.')}
                if index==0:self.metadata=fields
                if len(paths)>1:
                    if fields.get('split.no')!=index or fields.get('split.count')!=len(paths):raise CheckpointError('GGUF split index/count mismatch')
                    if total is None:total=fields.get('split.tensors.count')
                    if type(total) is not int or total<1:raise CheckpointError('missing/invalid split tensor manifest')
                    if fields.get('split.tensors.count')!=total:raise CheckpointError('GGUF split tensor count mismatch')
                elif fields.get('split.count',1)!=1:raise CheckpointError('missing GGUF split files')
                # Later shards can omit model metadata, but cannot contradict it.
                for key,value in fields.items():
                    if key.startswith(('general.architecture','glm-dsa.')) and key in self.metadata and self.metadata[key]!=value:
                        raise CheckpointError('inconsistent GGUF metadata: '+key)
                spans=[]
                for tensor in reader.tensors:
                    shape=tuple(int(n) for n in reversed(tensor.shape))
                    if not shape or min(shape)<=0:raise CheckpointError('invalid tensor shape: '+tensor.name)
                    block,blockbytes=gguf.GGML_QUANT_SIZES[tensor.tensor_type]
                    if shape[-1]%block:raise CheckpointError('quantization block does not divide row')
                    expected=math.prod(shape)//block*blockbytes
                    if tensor.n_bytes!=expected or tensor.data_offset<reader.data_offset or tensor.data_offset+expected>p.stat().st_size:
                        raise CheckpointError('invalid tensor span: '+tensor.name)
                    if tensor.name in self.tensors:raise CheckpointError('duplicate tensor: '+tensor.name)
                    if tensor.tensor_type not in (gguf.GGMLQuantizationType.F32,gguf.GGMLQuantizationType.F16) and tensor.tensor_type not in gguf.quants._type_traits:
                        raise CheckpointError('unsupported quantization: '+str(tensor.tensor_type))
                    spans.append((tensor.data_offset,tensor.data_offset+expected))
                    self.tensors[tensor.name]=tensor
                spans.sort()
                if any(a[1]>b[0] for a,b in zip(spans,spans[1:])):raise CheckpointError('overlapping tensor storage')
            if total is not None and total!=len(self.tensors):raise CheckpointError('incomplete tensor manifest')
        except BaseException:
            self.close();raise

    def shape(self,name):return tuple(int(n) for n in reversed(self.tensors[name].shape))

    def read(self,name,*,rows=None,expert=None):
        import gguf
        if self.closed:raise RuntimeError('GGUF store is closed')
        tensor=self.tensors[name];data=tensor.data;shape=self.shape(name)
        if expert is not None:
            if type(expert) is not int or len(shape)!=3 or not 0<=expert<shape[0]:raise CheckpointError('invalid expert index')
            data=data[expert];shape=shape[1:]
        if rows is not None:
            if len(shape)!=2 or not isinstance(rows,slice):raise CheckpointError('row slice requires a matrix')
            start,stop,step=rows.indices(shape[0])
            if step!=1:raise CheckpointError('only contiguous row slices supported')
            data=data[start:stop];shape=(max(0,stop-start),shape[1])
        if math.prod(shape)*4>self.max_decode_bytes:raise CheckpointError('decode exceeds byte bound; select one expert or rows')
        out=np.empty(shape,dtype=np.float32)
        if out.size:
            if tensor.tensor_type in (gguf.GGMLQuantizationType.F32,gguf.GGMLQuantizationType.F16):
                out[...] = data.reshape(shape)
            else:
                encoded=data.reshape(-1,data.shape[-1])
                decoded=out.reshape(-1,shape[-1])
                def chunk(start):
                    decoded[start:start+self.decode_rows]=gguf.quants.dequantize(np.asarray(encoded[start:start+self.decode_rows]),tensor.tensor_type)
                starts=range(0,len(encoded),self.decode_rows)
                if self.pool is None:
                    for start in starts:chunk(start)
                else:
                    futures=[self.pool.submit(chunk,start) for start in starts]
                    wait(futures)
                    for future in futures:future.result()
            if not np.isfinite(out).all():raise CheckpointError('nonfinite decoded GGUF tensor: '+name)
        out.flags.writeable=False
        return out

    def close(self):
        if self.closed:return
        self.closed=True
        if self.pool is not None:self.pool.shutdown(wait=True)
        self.tensors.clear()
        for reader in self.readers:
            reader.tensors.clear();reader.fields.clear();reader.data._mmap.close()
        self.readers.clear()
