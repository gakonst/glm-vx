#!/usr/bin/env python3
"""Tiny CPU concurrency/length matrix. Repeat measurements, verify every output."""
import json
import os
import platform
import statistics
from pathlib import Path
from benchmark import benchmark
from glm_vx.backend import NumpyBackend
from kernels.backend import VxBackend

results=[]
for requests in (1,2,4,8):
    for prompt in (8,32):
        # Alternate order to reduce systematic warmup/order bias.
        trials=[]
        for repeat in range(3):
            backends=[NumpyBackend(),VxBackend()]
            if repeat%2:backends.reverse()
            runs={b.name:benchmark(b,requests,prompt,16) for b in backends}
            assert runs['numpy-reference']['outputs']==runs['vx-cpu-f32']['outputs']
            for run in runs.values():run.pop('outputs')
            trials.append(runs)
        medians={}
        for name in ('numpy-reference','vx-cpu-f32'):
            medians[name]={key:statistics.median(t[name][key] for t in trials)
                for key in ('elapsed_seconds','output_tokens_per_second','ttft_median_seconds','ttft_max_seconds')}
        results.append({'requests':requests,'prompt_tokens_each':prompt,'output_tokens_each':16,'trials':trials,'medians':medians})
report={'scope':'tiny random GLM-shaped CPU model; no full-checkpoint or GPU execution',
    'platform':platform.platform(),'blas_threads':os.environ.get('OPENBLAS_NUM_THREADS','unset'),
    'repeats':3,'all_greedy_outputs_match':True,'results':results}
Path('benchmark-matrix.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps([{'requests':r['requests'],'prompt':r['prompt_tokens_each'],'medians':r['medians']} for r in results],indent=2))
