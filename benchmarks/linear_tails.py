"""Short, resident synthetic CPU benchmark for 4/2-row tails; no model IO."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
from kernels.backend import VxBackend


def run(baseline,candidate,repeats):
    if not 3<=repeats<=31:raise ValueError('repeats must be 3..31')
    backends={'baseline':VxBackend(baseline),'candidate':VxBackend(candidate)}
    result={'scope':'bounded resident synthetic f32 matrix tails; not trained/model/HTTP performance',
            'seed':659,'repeats':repeats,'release_eligible':False,'cases':[],
            'libraries':{key:{'path':str(b.library_path),'sha256':hashlib.sha256(b.library_path.read_bytes()).hexdigest()} for key,b in backends.items()}}
    rng=np.random.default_rng(659)
    for n,k in ((256,2048),(512,6144)):
        w=rng.normal(size=(n,k)).astype(np.float32)
        for rows in (2,4,6):
            x=rng.normal(size=(rows,k)).astype(np.float32)
            expected=np.stack([backends['baseline'].matvec(w,row) for row in x])
            samples={key:[] for key in backends}
            for b in backends.values():np.testing.assert_array_equal(b.linear_batch(w,x),expected)
            for repeat in range(repeats):
                for key in ('baseline','candidate') if repeat%2 else ('candidate','baseline'):
                    start=time.perf_counter_ns();actual=backends[key].linear_batch(w,x)
                    samples[key].append((time.perf_counter_ns()-start)/1e6)
                    assert actual.tobytes()==expected.tobytes()
            medians={key:statistics.median(values) for key,values in samples.items()}
            result['cases'].append({'rows':rows,'outputs':n,'inputs':k,'samples_ms':samples,'median_ms':medians,
                                    'speedup':medians['baseline']/medians['candidate'],'bitwise_equal':True})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=9);args=p.parse_args()
    if args.output.exists():p.error('refusing to overwrite benchmark evidence')
    result=run(args.baseline,args.candidate,args.repeats);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:v for k,v in c.items() if k!='samples_ms'} for c in result['cases']],indent=2))

if __name__=='__main__':main()
