"""Exact selection benchmark. The quadratic-in-k scan is bounded deliberately."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
from kernels.backend import VxBackend


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=11)
    a = p.parse_args()
    if a.repeats < 3: p.error('at least three samples required')
    if a.output.exists(): p.error('refusing to overwrite evidence')
    variants = {'baseline': VxBackend(a.baseline), 'candidate': VxBackend(a.candidate)}
    cases = []
    rng = np.random.default_rng(5481)
    for n, k in [(512, 32), (4096, 128), (2049, 2048), (80000, 2048), (1048576, 2048)]:
        x = rng.normal(size=n).astype(np.float32)
        x[::5] = 0
        expected = np.lexsort((np.arange(n), -x))[:k]
        enabled = ['baseline', 'candidate'] if n*k*k <= 100_000_000 else ['candidate']
        samples = {name: [] for name in enabled}
        for name in enabled:
            np.testing.assert_array_equal(variants[name].topk(x, k), expected)
        for repeat in range(a.repeats):
            for name in enabled if repeat % 2 else list(reversed(enabled)):
                start = time.perf_counter_ns()
                result = variants[name].topk(x, k)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                np.testing.assert_array_equal(result, expected)
                samples[name].append(elapsed)
        case = {'n': n, 'k': k, 'milliseconds': samples,
                'median_ms': {name: statistics.median(v) for name, v in samples.items()},
                'scope': 'host ctypes call plus input validation and exact selection; resident F32 scores'}
        if len(enabled) == 2:
            case['speedup'] = case['median_ms']['baseline'] / case['median_ms']['candidate']
        else:
            case['baseline_not_measured'] = 'n*k*k exceeds declared 100,000,000 scan-work bound; no extrapolated speedup'
        cases.append(case)
    report = {'schema_version': 1, 'status': 'complete', 'seed': 5481,
              'scope': 'isolated CPU selection, not full-model speed or parity',
              'platform': platform.platform(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'libraries_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for name, path in [('baseline', a.baseline), ('candidate', a.candidate)]},
              'cases': cases}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps([{k: v for k, v in c.items() if k != 'milliseconds'} for c in cases], indent=2))


if __name__ == '__main__': main()
