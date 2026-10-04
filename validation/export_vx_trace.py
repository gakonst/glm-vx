"""Bounded teacher-forced GGUF trace exporter. No sampling or reference substitution."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from glm_vx.model import GlmMoeDsaModel
from glm_vx.gguf_checkpoint import GGUFCheckpoint
from kernels.backend import VxBackend

def main():
 p=argparse.ArgumentParser();p.add_argument('--gguf',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--tokens',type=int,nargs='+',required=True);args=p.parse_args()
 if args.output.exists():p.error('output exists; preserve previous traces')
 if len(args.tokens)>256:p.error('bounded exporter supports at most256 tokens')
 args.output.mkdir(parents=True);config=json.loads((args.gguf/'config.json').read_text());arrays={};start=time.monotonic()
 def capture(pos,name,value):
  arrays[f'p{pos}.{name}']=value
  if name.endswith('.output'):print(json.dumps({'position':pos,'tensor':name,'seconds':time.monotonic()-start}),flush=True)
 weights=GGUFCheckpoint(args.gguf,config,decode_threads=4)
 try:
  model=GlmMoeDsaModel(config,weights,VxBackend(args.library),trace=capture);cache=model.new_cache()
  for token in args.tokens:model.forward(token,cache)
 finally:weights.close()
 np.savez(args.output/'arrays.npz',**arrays)
 receipt={'status':'complete','token_ids':args.tokens,'library_sha256':hashlib.sha256(args.library.read_bytes()).hexdigest(),'elapsed_seconds':time.monotonic()-start,'tensor_count':len(arrays),'scope':'Teacher-forced full layer outputs, current-token cache entries, selections and complete logits; other internal stages not yet exported.'}
 (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
