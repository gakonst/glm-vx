"""Resident synthetic GLM-shaped CPU projections; not end-to-end throughput."""
import argparse,json,time,statistics,hashlib
from pathlib import Path
import numpy as np
from kernels.backend import VxBackend
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():p.error('output exists')
old=VxBackend(a.baseline);new=VxBackend(a.candidate)
rng=np.random.default_rng(735)
cases=[]
for m,n,k in [(8,512,6144),(8,2048,6144),(8,6144,6144),(16,2048,6144)]:
    x=rng.normal(size=(m,k)).astype(np.float32);w=rng.normal(size=(n,k)).astype(np.float32)
    expected=np.stack([old.matvec(w,row) for row in x])
    samples={'sequential_matvec':[],'tiled_batch':[]}
    for repeat in range(7):
        for name in list(samples) if repeat%2 else list(reversed(samples)):
            begin=time.perf_counter_ns()
            y=np.stack([old.matvec(w,row) for row in x]) if name=='sequential_matvec' else new.linear_batch(w,x)
            samples[name].append((time.perf_counter_ns()-begin)/1e6)
            np.testing.assert_array_equal(y,expected)
    med={name:statistics.median(v) for name,v in samples.items()}
    cases.append({'m':m,'n':n,'k':k,'median_ms':med,'speedup':med['sequential_matvec']/med['tiled_batch'],'samples_ms':samples})
result={'scope':'resident synthetic F32 projections at GLM-related shapes, exact baseline equality on these inputs, not trained-model parity', 'seed':735, 'baseline_sha256':hashlib.sha256(old.library_path.read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256(new.library_path.read_bytes()).hexdigest(),'cases':cases}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([{k:v for k,v in c.items() if k!='samples_ms'} for c in cases],indent=2))
