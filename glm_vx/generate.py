"""CPU GGUF proof of concept: trained weights, our GLM graph and Vx kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import importlib.metadata
import time
import numpy as np
from .model import Model
from .gguf_checkpoint import GGUFCheckpoint


def render_prompt(directory,prompt,raw=False,reasoning_effort="low"):
    if raw:return prompt
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    env=ImmutableSandboxedEnvironment(extensions=['jinja2.ext.loopcontrols'])
    def fail(message):raise ValueError(message)
    env.globals['raise_exception']=fail
    env.filters['tojson']=lambda value,**kwargs:json.dumps(value,ensure_ascii=False,**kwargs)
    template=env.from_string((directory/'chat_template.jinja').read_text())
    return template.render(messages=[{'role':'user','content':prompt}],tools=None,
                           add_generation_prompt=True,clear_thinking=True,reasoning_effort=reasoning_effort)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf',required=True)
    p.add_argument('--config')
    p.add_argument('--decode-threads',type=int,default=1)
    p.add_argument('--reasoning-effort',choices=['low','high','max'],default='low')
    p.add_argument('--prompt',default='Hello')
    p.add_argument('--raw',action='store_true',help='raw completion, without the official chat template')
    p.add_argument('--max-tokens',type=int,default=8)
    p.add_argument('--backend',choices=['vx','numpy-reference'],default='vx')
    p.add_argument('--decoded-cache-mib',type=int,default=512)
    p.add_argument('--packed-weights',action='store_true',help='opt in to Vx packed GGUF dot products with F32 activations')
    p.add_argument('--output',required=True,help='JSON receipt updated after every completed model token')
    a=p.parse_args()
    if a.packed_weights and a.backend != 'vx':p.error('packed weights requires --backend vx')
    if a.max_tokens<1 or a.decoded_cache_mib<0:p.error('positive max tokens and nonnegative cache required')
    directory=Path(a.gguf) if Path(a.gguf).is_dir() else Path(a.gguf).parent
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(directory/'tokenizer.json'))
    config=json.loads(Path(a.config or directory/'config.json').read_text())
    formatted=render_prompt(directory,a.prompt,a.raw,a.reasoning_effort)
    tokens=tokenizer.encode(formatted,add_special_tokens=False).ids
    if not tokens or len(tokens)+a.max_tokens>config['max_position_embeddings']:p.error('empty prompt or context overflow')
    if a.backend=='vx':
        from kernels.backend import VxBackend
        backend=VxBackend()
    else:
        from .backend import NumpyBackend
        backend=NumpyBackend()
    destination=Path(a.output);destination.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):commit=None
    report={'status':'loading','backend':backend.name,'checkpoint':str(directory),
            'engine_commit':commit,'packed_weights':a.packed_weights,'decode_threads':a.decode_threads,'gguf_version':importlib.metadata.version('gguf'),
            'prompt':a.prompt,'formatted_prompt':formatted,'prompt_token_ids':tokens,
            'generated_token_ids':[],'steps':[],'scope':'GGUF base decoder using the supplied checkpoint/config; no MTP/speculative decoding.',
            'dimensions':{k:config[k] for k in ('num_hidden_layers','hidden_size','vocab_size','n_routed_experts')},
            'tokenizer_sha256':hashlib.sha256((directory/'tokenizer.json').read_bytes()).hexdigest()}
    def save():
        report['elapsed_seconds']=time.monotonic()-start
        report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        temporary=destination.with_suffix(destination.suffix+'.tmp')
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temporary.replace(destination)
    class TracedModel(Model):
        def _feed_forward(self,layer,x):
            result=super()._feed_forward(layer,x)
            print(json.dumps({'event':'layer','layer':layer,'elapsed_seconds':time.monotonic()-start}),flush=True)
            return result
    weights=None
    try:
        save()
        weights=GGUFCheckpoint(a.gguf,config,cache_bytes=a.decoded_cache_mib*1024**2,decode_threads=a.decode_threads)
        report['gguf_metadata']={k:v for k,v in weights.store.metadata.items() if not k.startswith('tokenizer.')}
        manifest=directory/'manifest.json'
        if manifest.exists():report['source_manifest']=json.loads(manifest.read_text())
        model=TracedModel(config,weights,backend,packed_weights=a.packed_weights);cache=model.new_cache()
        report['status']='prefilling';save()
        for i,token in enumerate(tokens):
            began=time.monotonic()
            logits=model.forward(token,cache) if i==len(tokens)-1 else model.prefill_token(token,cache)
            report['steps'].append({'phase':'prefill','index':i,'seconds':time.monotonic()-began})
            save()
        report['status']='generating'
        eos=config.get('eos_token_id',[])
        if isinstance(eos,int):eos=[eos]
        for i in range(a.max_tokens):
            if not np.isfinite(logits).all():raise ValueError('nonfinite logits')
            token=int(np.argmax(logits));report['generated_token_ids'].append(token)
            if i==0:report['first_token_seconds_including_load']=time.monotonic()-start
            top=np.argsort(logits)[-5:][::-1]
            report['steps'].append({'phase':'sample','index':i,'token':token,'top5':[{'id':int(j),'logit':float(logits[j])} for j in top],
                                    'logits_sha256':hashlib.sha256(logits.tobytes()).hexdigest()})
            report['text']=tokenizer.decode(report['generated_token_ids'],skip_special_tokens=False)
            save();print(json.dumps({'event':'token','id':token,'text':report['text']},ensure_ascii=False),flush=True)
            if token in eos:report['finish_reason']='stop';break
            if i+1<a.max_tokens:
                began=time.monotonic();logits=model.forward(token,cache)
                report['steps'].append({'phase':'decode','index':i,'seconds':time.monotonic()-began})
        report['finish_reason']=report.get('finish_reason','length');report['status']='complete';save()
    except BaseException as exc:
        report['status']='failed';report['error']=type(exc).__name__+': '+str(exc);save();raise
    finally:
        if weights is not None:weights.close()
        if hasattr(backend,'close'):backend.close()

if __name__=='__main__':main()
