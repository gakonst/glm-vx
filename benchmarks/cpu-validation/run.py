"""Reproduce the CPU-only HTTP comparison from the repository root."""
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('http_bench', ROOT/'benchmarks/serve.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
runs=[]
for backend in ('vx','numpy-reference'):
    # Port zero avoids taking over another service. Server prints its bound URL.
    process=subprocess.Popen([sys.executable,'-m','glm_vx.server','--tiny','--backend',backend,'--port','0'],cwd=ROOT,env=env,stdout=subprocess.PIPE,text=True)
    try:
        ready=json.loads(process.stdout.readline());url=ready['listening']
        for concurrency in (1,2,4,8):
            for repeat in range(3):
                warmup=bench.completion(url,[1]*8,16)
                assert warmup['ok'],warmup
                def run(i):
                    prompt=[(i*17+j*3+1)%256 for j in range((8,128)[i%2])]
                    return bench.completion(url,prompt,16)
                start=time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    rows=list(pool.map(run,range(8)))
                elapsed=time.perf_counter()-start
                assert all(r['ok'] for r in rows),rows
                result={'backend':backend,'concurrency':concurrency,'repeat':repeat+1,'metrics':bench.summarize(rows,elapsed),'requests':rows}
                runs.append(result)
                (OUT/f'{backend}-c{concurrency}-r{repeat+1}.json').write_text(json.dumps(result,indent=2)+'\n')
            print(f'{backend} concurrency={concurrency}: 3 runs complete',flush=True)
    finally:
        process.send_signal(signal.SIGINT)
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=5)
expected=[r['output_sha256'] for r in runs[0]['requests']]
assert all([r['output_sha256'] for r in run['requests']]==expected for run in runs),'output mismatch'
summary=[]
for backend in ('vx','numpy-reference'):
    for concurrency in (1,2,4,8):
        group=[r['metrics'] for r in runs if r['backend']==backend and r['concurrency']==concurrency]
        summary.append({'backend':backend,'concurrency':concurrency,
            'output_tokens_per_second':statistics.median(r['output_tokens_per_second'] for r in group),
            'ttft_p50_ms':statistics.median(r['ttft_ms']['p50'] for r in group),
            'ttft_p95_ms':statistics.median(r['ttft_ms']['p95'] for r in group),
            'inter_token_p95_ms':statistics.median(r['inter_token_ms']['p95'] for r in group)})
report={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'cpu':subprocess.check_output(['lscpu'],text=True),'python':platform.python_version(),
    'scope':'CPU-only tiny random GLM-shaped model, seed7, 4 layers hidden32 vocab256; not full trained GLM-5.3',
    'workload':{'requests_per_run':8,'prompt_lengths':[8,128],'output_tokens':16,'repeats':3,'concurrency':[1,2,4,8], 'warmup_per_run':1,'blas_threads':1,'scheduler':'one model owner; concurrency is HTTP requests, not CPU worker threads'},
    'successful_requests':sum(r['metrics']['successful_requests'] for r in runs),'failed_requests':0,
    'all_output_hashes_match_across_backends_and_concurrency':True,
    'summary_median_of_runs':summary,'caveat':'Shared CPU, localhost HTTP, sequential backend runs; no confidence intervals or GPU claims.'}
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(summary,indent=2))
