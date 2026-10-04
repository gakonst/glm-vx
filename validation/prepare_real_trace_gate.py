"""Prepare a narrow, strict two-position reference contract before candidate comparison.

This development helper pins local artifacts. It does not certify their provenance
or calibrate acceptable mixed-precision error. Its strict existing F32 tolerances
are intentionally not relaxed if the independent quantized reference disagrees.
"""
import argparse,hashlib,json,platform
from pathlib import Path
import numpy as np
from validation.trace_gate import case_metadata

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def prepare(root,model,reference,reference_libs,candidate_lib):
 root.mkdir(parents=True,exist_ok=True)
 if (root/'contract.json').exists():raise ValueError('contract exists; do not overwrite trusted budgets')
 if (reference/'complete.txt').read_text().split()!=['2','154880']:raise ValueError('reference incomplete or wrong dimensions')
 tensors=[];arrays={}
 for pos in range(2):
  for layer in range(78):
   key=f'p{pos}.layer.{layer}.output';value=np.fromfile(reference/f'p{pos}-l_out-{layer}.f32',dtype='<f4')
   if value.shape!=(6144,):raise ValueError('bad reference layer shape')
   arrays[key]=value;tensors.append(dict(key=key,operation='layer_output',kind='float',shape=[6144],dtype='float32',position=pos,layer=layer,infinity='forbid'))
  key=f'p{pos}.logits';value=np.fromfile(reference/f'p{pos}-logits.f32',dtype='<f4')
  if value.shape!=(154880,):raise ValueError('bad logits shape')
  arrays[key]=value;tensors.append(dict(key=key,operation='logits',kind='logits',shape=[154880],dtype='float32',position=pos,layer=None,infinity='forbid'))
 np.savez(root/'reference.npz',**arrays)
 corpus={'token_ids':[9703,10056],'positions':[0,1],'case':'hello-let-teacher-forced-v1'}
 manifest=json.loads((model/'manifest.json').read_text())
 budget=dict(atol=4e-6,rtol=4e-5,normalized_l2=4e-5,normalization_floor=1e-6,softmax_tv=4e-5,top_k=5,top_k_margin_atol=4e-5)
 pins=dict(config_sha256=sha(model/'config.json'),weights_sha256={x['rfilename']:x['lfs']['sha256'] for x in manifest['files']},tokenizer_sha256=sha(model/'tokenizer.json'),template_sha256=sha(model/'chat_template.jinja'),corpus_sha256=hashlib.sha256(json.dumps(corpus,sort_keys=True).encode()).hexdigest(),corpus_version='hello-let-teacher-forced-v1',reference_revision='11fe02151f79c41d0d4af7da708755d73b9c0da6',numerical_mode='strict exploratory: GGML native quantized CPU versus Vx dequantized F32',tie_breaking='stable-lowest-index')
 producers={'reference':dict(binary_sha256=sha('build/parity/llama-trace'),compiler='GNU C++ 11.4 -O2 helper; upstream Release build',libraries_sha256={n:sha(reference_libs/n) for n in ['libllama.so','libggml.so','libggml-base.so','libggml-cpu.so']},flags=['cpu','threads=4','n_batch=1','n_ctx=256','teacher_forcing'],hardware=platform.machine()),'candidate':dict(binary_sha256=sha(candidate_lib),compiler='Vx v0.0.2 -O3, no fast-math flag',libraries_sha256={'model.py':sha('glm_vx/model.py'),'export_vx_trace.py':sha('validation/export_vx_trace.py')},flags=['cpu','GGUF decode_threads=4','teacher_forcing'],hardware=platform.machine())}
 case=dict(id=corpus['case'],token_ids=corpus['token_ids'],positions=[0,1],execution='decode',chunks=[1,1],cache_owner='isolated-single-request',layers=list(range(78)),vocab_size=154880,tensors=tensors)
 contract=dict(schema_version=1,contract_id='real-2-position-strict-v1',scope='Exploratory full layer outputs and complete logits on two teacher-forced real-weight positions; not a release gate or full internal-state parity.',pins=pins,producers=producers,reference_trace_sha256=sha(root/'reference.npz'),budgets={'layer_output':budget,'logits':budget},cases=[case]);write(root/'contract.json',contract)
 emit_manifest(root,contract,'reference',root/'reference.npz')

def emit_manifest(root,contract,role,trace):
 write(root/f'{role}-manifest.json',dict(schema_version=1,role=role,contract_sha256=sha(root/'contract.json'),pins=contract['pins'],producer=contract['producers'][role],status='complete',skipped=[],cases=[case_metadata(c) for c in contract['cases']],trace_sha256=sha(trace)))
def candidate(root,source):
 contract=json.loads((root/'contract.json').read_text());receipt=json.loads((source/'receipt.json').read_text())
 if receipt['status']!='complete' or receipt['token_ids']!=[9703,10056] or receipt['library_sha256']!=contract['producers']['candidate']['binary_sha256']:raise ValueError('candidate incomplete or mismatched')
 for n,h in contract['producers']['candidate']['libraries_sha256'].items():
  p={'model.py':'glm_vx/model.py','export_vx_trace.py':'validation/export_vx_trace.py'}[n]
  if sha(p)!=h:raise ValueError('candidate source changed')
 with np.load(source/'arrays.npz',allow_pickle=False) as z:arrays={s['key']:z[s['key']] for c in contract['cases'] for s in c['tensors']}
 np.savez(root/'candidate.npz',**arrays);emit_manifest(root,contract,'candidate',root/'candidate.npz')
def main():
 p=argparse.ArgumentParser();p.add_argument('operation',choices=['reference','candidate']);p.add_argument('--root',type=Path,required=True);p.add_argument('--model',type=Path);p.add_argument('--reference',type=Path);p.add_argument('--reference-libs',type=Path);p.add_argument('--candidate-library',type=Path);p.add_argument('--candidate-source',type=Path);a=p.parse_args()
 if a.operation=='reference':prepare(a.root,a.model,a.reference,a.reference_libs,a.candidate_library)
 else:candidate(a.root,a.candidate_source)
if __name__=='__main__':main()
