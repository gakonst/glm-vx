"""Bounded teacher-forced GGUF trace exporter. No sampling or reference substitution."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from glm_vx.model import GlmMoeDsaModel
from glm_vx.gguf_checkpoint import GGUFCheckpoint
from kernels.backend import VxBackend

def main():
 p=argparse.ArgumentParser();p.add_argument('--gguf',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--tokens',type=int,nargs='+',required=True);p.add_argument('--packed-weights',action='store_true');args=p.parse_args()
 if args.output.exists():p.error('output exists; preserve previous traces')
 if len(args.tokens)>256:p.error('bounded exporter supports at most256 tokens')
 args.output.mkdir(parents=True);config=json.loads((args.gguf/'config.json').read_text());arrays={};start=time.monotonic()
 source_files=['glm_vx/model.py','glm_vx/gguf_checkpoint.py','glm_vx/gguf_reader.py','kernels/backend.py','kernels/packed.py','validation/export_vx_trace.py']
 source_hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in source_files}
 library_hash=hashlib.sha256(args.library.read_bytes()).hexdigest()
 def capture(pos,name,value):
  arrays[f'p{pos}.{name}']=value
  if name.endswith('.output'):print(json.dumps({'position':pos,'tensor':name,'seconds':time.monotonic()-start}),flush=True)
 weights=GGUFCheckpoint(args.gguf,config,decode_threads=4)
 try:
  model=GlmMoeDsaModel(config,weights,VxBackend(args.library),trace=capture,packed_weights=args.packed_weights);cache=model.new_cache()
  for token in args.tokens:model.forward(token,cache)
 finally:weights.close()
 if source_hashes!={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in source_files} or library_hash!=hashlib.sha256(args.library.read_bytes()).hexdigest():raise RuntimeError('source or library changed during trace')
 np.savez(args.output/'arrays.npz',**arrays)
 receipt={'status':'complete','token_ids':args.tokens,'packed_weights':args.packed_weights,'source_sha256':source_hashes,'numerical_mode':'f32-official-expert-order-v2','library_sha256':hashlib.sha256(args.library.read_bytes()).hexdigest(),'elapsed_seconds':time.monotonic()-start,'tensor_count':len(arrays),'scope':'Teacher-forced layer outputs, cache entries, selections, logits and per-operation CPU attention/indexer/MLP stages and expert reductions; fused backends expose only observable outputs.'}
 (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
