#!/usr/bin/env python3
"""Reproducible tiny-model CPU benchmark; NOT full GLM/GPU performance."""
import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
import numpy as np
from glm_vx.model import Model,tiny_weights
from glm_vx.tiny import tiny_config
from glm_vx.backend import NumpyBackend
from glm_vx.scheduler import Scheduler

def benchmark(backend,requests,prompt_len,new_tokens):
    config=tiny_config();model=Model(config,tiny_weights(config,seed=7),backend)
    # Warm library, weights and shapes outside the measurement.
    model.forward(1,model.new_cache())
    s=Scheduler(model,max_sequences=requests,token_budget=requests*(prompt_len+new_tokens),prefill_chunk=4)
    start=time.perf_counter()
    reqs=[s.submit([1+(i%100) for i in range(prompt_len)],new_tokens) for _ in range(requests)]
    outputs=[];ttfts=[]
    try:
        for req in reqs:
            out=[]
            while True:
                e=req.events.get(timeout=120)
                if e['type']=='done':
                    if e['reason']=='error':raise RuntimeError(e['error'])
                    break
                out.append(e['token_id'])
            outputs.append(out)
            ttfts.append(req.first_token_at-req.created)
        elapsed=time.perf_counter()-start
    finally:s.close()
    return {'backend':backend.name,'requests':requests,'prompt_tokens_each':prompt_len,
        'generated_tokens_each':new_tokens,'elapsed_seconds':elapsed,
        'output_tokens_per_second':sum(map(len,outputs))/elapsed,
        'ttft_median_seconds':statistics.median(ttfts),'ttft_max_seconds':max(ttfts),
        'outputs':outputs}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--requests',type=int,default=2)
    p.add_argument('--prompt',type=int,default=8);p.add_argument('--tokens',type=int,default=8)
    a=p.parse_args()
    from kernels.backend import VxBackend
    ref=benchmark(NumpyBackend(),a.requests,a.prompt,a.tokens)
    vx=benchmark(VxBackend(),a.requests,a.prompt,a.tokens)
    if vx['outputs']!=ref['outputs']:raise AssertionError('Vx/reference greedy output mismatch')
    report={'scope':'tiny random GLM-shaped CPU model; no GPU or trained full-model claim',
        'platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,
        'blas_threads':os.environ.get('OPENBLAS_NUM_THREADS','unset'),'results':[ref,vx],'greedy_outputs_match':True}
    text=json.dumps(report,indent=2);print(text)
    if a.output:Path(a.output).write_text(text+'\n')
if __name__=='__main__':main()
