"""Reproducible untrained tiny-model HTTP prefix reuse comparison.

Runs the same fixed mixed prompt trace against uncached, cold-cache and primed
cache schedulers. Each phase creates independent model/backend instances. Kernel
work remains sequential token prefill; concurrency is scheduler interleaving.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time

from benchmarks.serve import completion, summarize
from glm_vx.backend import NumpyBackend
from glm_vx.model import Model, tiny_weights
from glm_vx.scheduler import Scheduler
from glm_vx.server import Server
from glm_vx.tiny import tiny_config


def memory():
    status = Path('/proc/self/status').read_text().splitlines()
    return {line.split(':')[0]:line.split(':')[1].strip() for line in status
            if line.startswith(('VmRSS:', 'VmHWM:'))}


def phase(backend_name, budget, primed, prompts, requests, concurrency, output_tokens):
    if backend_name == 'vx':
        from kernels.backend import VxBackend
        backend = VxBackend()
    else: backend = NumpyBackend()
    cfg = tiny_config(); model = Model(cfg, tiny_weights(cfg, seed=19), backend)
    scheduler = Scheduler(model, prefix_cache_bytes=budget, prefill_chunk=8,
                          decode_prefill_tokens=2, max_sequences=concurrency)
    server = Server(('127.0.0.1', 0), scheduler)
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        warmup = [completion(url, p, 1) for p in prompts] if primed else []
        if any(not r['ok'] for r in warmup): raise RuntimeError('priming failed')
        before = scheduler.status()
        memory_before = memory()
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(pool.map(lambda i: completion(url, prompts[i % len(prompts)], output_tokens), range(requests)))
        elapsed = time.perf_counter() - started
        return {'metrics': summarize(rows, elapsed), 'requests': rows,
                'before': before, 'after': scheduler.status(),
                'process_memory_before':memory_before,'process_memory_after':memory(), 'priming_requests': len(warmup)}
    finally:
        server.shutdown(); server.server_close(); scheduler.close(); worker.join()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['numpy', 'vx'], default='vx')
    p.add_argument('--requests', type=int, default=16)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--output-tokens', type=int, default=12)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if min(args.requests,args.concurrency,args.output_tokens,args.repeats)<1: p.error('positive sizes required')
    prompts = [[(index*17+j*3+1)%256 for j in range(length)] for index,length in enumerate([16,64,16,64])]
    rounds = []
    for repeat in range(args.repeats):
        phases = {name: phase(args.backend,budget,prime,prompts,args.requests,args.concurrency,args.output_tokens)
                  for name,budget,prime in [('disabled',0,False),('cold',1024**2,False),('warm',1024**2,True)]}
        expected = [r.get('output_sha256') for r in phases['disabled']['requests']]
        equivalent = all(all(r['ok'] for r in value['requests']) and
                         [r.get('output_sha256') for r in value['requests']] == expected
                         for value in phases.values())
        rounds.append({'repeat':repeat,'outputs_equal':equivalent,'phases':phases})
    library = os.environ.get('GLM_VX_LIBRARY')
    result = {'schema':'glm-vx-prefix-http-v1','workload':vars(args),'rounds':rounds,
              'environment':{'python':platform.python_version(),
                             'cpu':next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),platform.machine()),
                             'source_sha256':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['glm_vx/scheduler.py','glm_vx/prefix.py','glm_vx/model.py','glm_vx/server.py','benchmarks/prefix_reuse.py','benchmarks/serve.py','kernels/backend.py']},
                             'build_note':'Prebuilt CPU library reused, not rebuilt; exact library digest recorded. No new compiler/flags claim.','platform':platform.platform(),
                             'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                             'library':library,'library_sha256':hashlib.sha256(Path(library).read_bytes()).hexdigest() if library else None,
                             'thread_env':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}},
              'measurement':'Client HTTP SSE timing, closed-loop mixed 16/64-token prompts; priming excluded. Untrained tiny random weights, seed 19. Repeated prompts are intentional. Prefix cache retains whole prompt snapshots; sequential prefill is not batched GEMM. Server stage timings include Python/backend dispatch; cached prompt tokens are excluded from computed prompt counts. No trained-checkpoint, GPU or distributed speed claim.'}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'outputs_equal':all(r['outputs_equal'] for r in rounds),
                      'rounds':[{name:value['metrics'] for name,value in r['phases'].items()} for r in rounds]},indent=2))
    return 0 if all(r['outputs_equal'] for r in rounds) else 1


if __name__ == '__main__': raise SystemExit(main())
