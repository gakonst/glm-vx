"""Real GGUF teacher-forced trace through independent expanded-KV F32 equations."""
import argparse,json,time
from pathlib import Path
import numpy as np
from glm_vx.gguf_checkpoint import GGUFCheckpoint
from validation.expanded_reference import expanded_batch_oracle

def main():
 p=argparse.ArgumentParser();p.add_argument('--gguf',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tokens',type=int,nargs='+',required=True);p.add_argument('--numerical-mode',choices=('f32-v2','legacy-v1'),default='f32-v2');a=p.parse_args()
 if a.output.exists():p.error('output exists')
 if len(a.tokens)>32:p.error('bounded expanded reference supports at most32 tokens')
 c=json.loads((a.gguf/'config.json').read_text())
 if c.get('attention_bias',False) or c.get('n_group',1)!=1 or c.get('topk_group',1)!=1 or c.get('n_shared_experts',1)<1 or c.get('rope_parameters',{}).get('rope_type','default')!='default':p.error('reference configuration unsupported')
 n=c['num_hidden_layers'];pattern=c.get('index_topk_pattern')
 if pattern is not None:c['indexer_types']=[{'F':'full','S':'shared'}[x] for x in pattern] if isinstance(pattern,str) else list(pattern)
 elif 'indexer_types' not in c:
  freq=c.get('index_topk_freq',1);offset=c.get('index_skip_topk_offset',2);c['indexer_types']=['full' if max(i-offset+1,0)%freq==0 else 'shared' for i in range(n)]
 c['mlp_layer_types']=c.get('mlp_layer_types') or ['dense' if i<c['first_k_dense_replace'] else 'sparse' for i in range(n)]
 a.output.mkdir(parents=True);arrays={};start=time.monotonic();w=GGUFCheckpoint(a.gguf,c,decode_threads=4)
 class Embedding:
  def __getitem__(self,tokens):return np.stack([w.tensor('model.embed_tokens.weight',rows=slice(t,t+1))[0] for t in tokens])
 class Weights:
  def __getitem__(self,name):return Embedding() if name=='model.embed_tokens.weight' else w(name)
 def capture(pos,name,value):
  arrays[f'p{pos}.{name}']=value
  if pos==len(a.tokens)-1:print(json.dumps({'tensor':name,'seconds':time.monotonic()-start}),flush=True)
 try:logits,selections=expanded_batch_oracle(c,Weights(),a.tokens,trace=capture,numerical_mode=a.numerical_mode)
 finally:w.close()
 for pos in range(len(a.tokens)):arrays[f'p{pos}.logits']=logits[pos]
 np.savez(a.output/'arrays.npz',**arrays)
 (a.output/'receipt.json').write_text(json.dumps({'status':'complete','numerical_mode':a.numerical_mode,'token_ids':a.tokens,'elapsed_seconds':time.monotonic()-start,'scope':'independent expanded-KV equations with explicit numerical_mode; shared GGUF decoder separately checked against native GGML; not official Transformers execution'},indent=2)+'\n')
if __name__=='__main__':main()
