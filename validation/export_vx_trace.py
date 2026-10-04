"""Bounded teacher-forced GGUF trace exporter. No reference substitution."""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from glm_vx.model import GlmMoeDsaModel
from glm_vx.gguf_checkpoint import GGUFCheckpoint
from kernels.backend import VxBackend
from validation.trace_identity import bind_model, bind_loaded, sha_file


def execution_pins():
    """Fingerprint imported implementations, independently of process CWD."""
    modules = ('glm_vx.model','glm_vx.config','glm_vx.checkpoint','glm_vx.gguf_checkpoint',
               'glm_vx.gguf_reader','kernels.backend','kernels.packed',
               'validation.trace_identity','validation.trace_gate')
    sources = {name.replace('.','/')+'.py':Path(importlib.import_module(name).__file__).resolve()
               for name in modules}
    sources['validation/export_vx_trace.py'] = Path(__file__).resolve()
    gguf = importlib.import_module('gguf.quants')
    numpy_native = importlib.import_module('numpy._core._multiarray_umath')
    root = Path(__file__).resolve().parents[1]
    evidence_path = root/'docs/parity-evidence/ggml-codec-parity.json'
    codec = json.loads(evidence_path.read_text())['candidate_decoder_source']['sha256']
    if sha_file(gguf.__file__) != codec:
        raise ValueError('installed GGUF codec differs from independently validated source')
    dependencies = sorted(Path(gguf.__file__).parent.glob('*.py'))
    dependencies += [Path(numpy_native.__file__), evidence_path]
    dependencies += sorted((Path(np.__file__).parent.parent/'numpy.libs').glob('*'))
    for name in ('numpy','gguf'):
        distribution = importlib.metadata.distribution(name)
        dependencies += [Path(distribution.locate_file(file)) for file in distribution.files or []
                         if Path(file).name in ('RECORD','direct_url.json','METADATA')]
    return {'source_paths':{k:str(v) for k,v in sources.items()},
            'source_sha256':{k:sha_file(v) for k,v in sources.items()},
            'dependencies':{'files_sha256':{str(p):sha_file(p) for p in dependencies if p.is_file()},
                            'versions':{n:importlib.metadata.version(n) for n in ('numpy','gguf')}}}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('gguf','output','library'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tokens',type=int,nargs='+',required=True)
    p.add_argument('--packed-weights',action='store_true')
    p.add_argument('--contract',type=Path,help='bind checkpoint identity for independent comparison')
    args=p.parse_args(argv)
    if args.output.exists():p.error('output exists; preserve previous traces')
    if not 1<=len(args.tokens)<=256:p.error('bounded exporter supports 1..256 tokens')
    identity=bind_model(args.gguf,args.contract) if args.contract else None
    config_hash=sha_file(args.gguf/'config.json')
    config=json.loads((args.gguf/'config.json').read_text())
    if any(t<0 or t>=config['vocab_size'] for t in args.tokens):p.error('token outside vocabulary')
    pins=execution_pins(); library_hash=sha_file(args.library)
    args.output.mkdir(parents=True);arrays={};start=time.monotonic()
    def capture(pos,name,value):
        key=f'p{pos}.{name}'
        if key in arrays:raise ValueError('duplicate trace tensor: '+key)
        arrays[key]=value
        if name.endswith('.output'):
            print(json.dumps({'position':pos,'tensor':name,'seconds':time.monotonic()-start}),flush=True)
    weights=GGUFCheckpoint(args.gguf,config,decode_threads=4)
    try:
        loaded=bind_loaded(identity,args.gguf,weights.store.paths) if identity else None
        model=GlmMoeDsaModel(config,weights,VxBackend(args.library),trace=capture,packed_weights=args.packed_weights)
        cache=model.new_cache()
        for token in args.tokens:model.forward(token,cache)
        np.savez(args.output/'arrays.npz',**arrays)
        arrays_hash=sha_file(args.output/'arrays.npz')
        if pins!=execution_pins() or library_hash!=sha_file(args.library):
            raise RuntimeError('source, dependency or library changed during trace')
        if config_hash!=sha_file(args.gguf/'config.json') or (identity is not None and
           (identity!=bind_model(args.gguf,args.contract) or loaded!=bind_loaded(identity,args.gguf,weights.store.paths))):
            raise RuntimeError('checkpoint identity changed during trace')
    finally:
        weights.close()
    receipt={'status':'complete','token_ids':args.tokens,'packed_weights':args.packed_weights,
             'model_identity':identity,'identity_bound':identity is not None,'loaded_checkpoint_paths':loaded,
             'config_sha256':config_hash,'arrays_sha256':arrays_hash,**pins,
             'numerical_mode':'f32-official-expert-order-v2','library_sha256':library_hash,
             'elapsed_seconds':time.monotonic()-start,'tensor_count':len(arrays),
             'scope':'Teacher-forced CPU GGUF trace; source/dependency and loaded-shard bindings, no producer attestation.'}
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
