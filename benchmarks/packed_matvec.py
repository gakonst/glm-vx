"""Bounded trained-row packed-vs-decoded CPU benchmark and native codec gate.

Reads at most --rows rows from ONE tensor/expert per format. Reports warm-page
microbenchmarks, not end-to-end token throughput or all-weight streaming speed.
No llama inference is run. Both packed and decoded matvec execute Vx.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import numpy as np
from glm_vx.gguf_reader import GGUFStore
from kernels.backend import VxBackend
from kernels.packed import unpack_for_validation
from validation.ggml_oracle import NativeGGML,DEFAULT_REFERENCE,DEFAULT_CHECKPOINT,fingerprint


def measure(fn,repeats):
    fn() # explicit warm-page/warm-kernel sample excluded
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter();fn();samples.append(time.perf_counter()-start)
    return {'median_seconds':statistics.median(samples),'samples_seconds':samples}


def run(args):
    if not 1<=args.rows<=512 or not 1<=args.repeats<=50:raise ValueError('rows 1..512; repeats 1..50')
    backend=VxBackend(args.library);native=NativeGGML(args.reference)
    store=GGUFStore(args.checkpoint)
    receipt={'schema':'glm-vx.packed-matvec.v1','scope':'bounded warm-page trained-row CPU microbenchmark; not token throughput',
             'numerics':'native GGML dequantized weights times unquantized f32 x; ordered f32 sum',
             'library':fingerprint(backend.library_path),'oracle':native.provenance(),
             'rows_limit':args.rows,'repeats':args.repeats,'samples':[], 'inventory':{}}
    try:
        paths={t.name:p for p,r in zip(store.paths,store.readers) for t in r.tensors}
        for t in store.tensors.values():
            d=receipt['inventory'].setdefault(t.tensor_type.name,{'tensors':0,'bytes':0})
            d['tensors']+=1;d['bytes']+=t.n_bytes
        kinds=sorted({int(t.tensor_type) for t in store.tensors.values()})
        for kind in kinds:
            candidates=[t for t in store.tensors.values() if int(t.tensor_type)==kind and len(store.shape(t.name))>=2]
            tensor=candidates[len(candidates)//2];shape=store.shape(tensor.name)
            expert=shape[0]//2 if len(shape)==3 else None
            rows=min(args.rows,shape[-2]);start=(shape[-2]-rows)//2
            selection={'expert':expert,'rows':slice(start,start+rows)}
            x=np.random.default_rng(991+kind).normal(size=shape[-1]).astype(np.float32)
            # Independent positional bytes; expected comes from pinned C codec.
            block,size=native.layout(tensor.tensor_type.name,kind)
            rowbytes=shape[-1]//block*size
            offset=int(tensor.data_offset)+((expert or 0)*shape[-2]+start)*rowbytes
            with paths[tensor.name].open('rb') as stream:
                raw=os.pread(stream.fileno(),rows*rowbytes,offset)
            expected=np.stack([native.decode(raw[i*rowbytes:(i+1)*rowbytes],tensor.tensor_type.name,kind,shape[-1]) for i in range(rows)])
            oracle=np.cumsum(expected*x,axis=1,dtype=np.float32)[:,-1]
            with store.packed_rows(tensor.name,**selection) as (view,dims,fmt):
                assert view.tobytes()==raw
                if kind:np.testing.assert_array_equal(unpack_for_validation(backend.lib,view,kind,dims),expected)
                actual=backend.packed_matvec(view,fmt,dims,x)
                np.testing.assert_array_equal(actual,oracle)
                packed=measure(lambda:backend.packed_matvec(view,fmt,dims,x),args.repeats)
            decoded=store.read(tensor.name,**selection)
            np.testing.assert_array_equal(decoded,expected)
            dense=measure(lambda:backend.matvec(decoded,x),args.repeats)
            full=measure(lambda:backend.matvec(store.read(tensor.name,**selection),x),args.repeats)
            receipt['samples'].append({'format':tensor.tensor_type.name,'tensor':tensor.name,'expert':expert,'start_row':start,
                'shape':[rows,shape[-1]],'byte_offset':offset,'raw_sha256':hashlib.sha256(raw).hexdigest(),
                'packed_bytes':len(raw),'expanded_bytes':expected.nbytes,
                'weight_payload_reduction':expected.nbytes/len(raw),'codec_exact':True,'dot_exact':True,
                'packed':packed,'cached_f32_matvec':dense,'decode_plus_f32_matvec':full,
                'speedup_vs_decode_plus_f32':full['median_seconds']/packed['median_seconds']})
            print(tensor.tensor_type.name, 'packed ms',round(packed['median_seconds']*1000,3),
                  'decode+dot ms',round(full['median_seconds']*1000,3),flush=True)
        receipt['status']='passed';return receipt
    finally:store.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=DEFAULT_REFERENCE)
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    p.add_argument('--library',type=Path)
    p.add_argument('--rows',type=int,default=128);p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--output',type=Path,default=Path('build/packed-benchmark.json'))
    args=p.parse_args();receipt=run(args)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(args.output)

if __name__=='__main__':main()
