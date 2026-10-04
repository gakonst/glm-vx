"""Bind expanded-F32 reference evidence to the same predeclared strict budgets.
The native-GGML failed contract/receipt is preserved; this is a different oracle.
"""
import argparse,json,sys
from pathlib import Path
from validation.prepare_real_trace_gate import sha,write,emit_manifest

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('output exists')
 receipt=json.loads((a.reference/'receipt.json').read_text())
 if receipt['status']!='complete' or receipt['token_ids']!=[9703,10056]:raise ValueError('reference incomplete')
 c=json.loads((a.base/'contract.json').read_text());c['contract_id']='real-2-position-expanded-f32-v1';c['scope']='Two teacher-forced real-weight positions; all78 layer outputs plus full logits versus independent expanded-KV F32 equations. No long-context/release/official-runtime parity claim.'
 c['pins']['reference_revision']=sha('validation/expanded_reference.py');c['pins']['numerical_mode']='dequantized F32: independent expanded-KV batch equations versus Vx compressed incremental decoder'
 c['producers']['reference']={'binary_sha256':sha(sys.executable),'compiler':sys.version,'libraries_sha256':{f:sha(f) for f in ['validation/expanded_reference.py','validation/export_expanded_trace.py','glm_vx/gguf_checkpoint.py','glm_vx/gguf_reader.py']},'flags':['OPENBLAS_NUM_THREADS=1','GGUF decode_threads=4','expanded causal batch equations'],'hardware':c['producers']['candidate']['hardware']}
 a.output.mkdir(parents=True)
 (a.output/'reference.npz').write_bytes((a.reference/'arrays.npz').read_bytes());(a.output/'candidate.npz').write_bytes((a.base/'candidate.npz').read_bytes())
 c['reference_trace_sha256']=sha(a.output/'reference.npz');write(a.output/'contract.json',c)
 for role in ('reference','candidate'):emit_manifest(a.output,c,role,a.output/f'{role}.npz')
if __name__=='__main__':main()
