import hashlib,json,time,statistics
from pathlib import Path
import numpy as np
from kernels.backend import VxBackend
rng=np.random.default_rng(5312026)
libs={'current_o0':Path('kernels/build/libglm_vx.so'),'candidate_o3':Path('build/perf-audit/libglm_vx_o3.so')}
b={k:VxBackend(v) for k,v in libs.items()};result=[]
for rows,cols in [(512,6144),(2048,6144),(6144,2048),(6144,6144)]:
 w=rng.standard_normal((rows,cols),dtype=np.float32);x=rng.standard_normal(cols,dtype=np.float32)
 y={k:v.matvec(w,x) for k,v in b.items()}
 np.testing.assert_array_equal(y['current_o0'],y['candidate_o3'])
 samples={k:[] for k in b}
 for i in range(11):
  for k in (list(b) if i%2==0 else list(reversed(b))):
   start=time.perf_counter();b[k].matvec(w,x);samples[k].append(time.perf_counter()-start)
 med={k:statistics.median(v) for k,v in samples.items()}
 result.append({'rows':rows,'cols':cols,'bitwise_equal':True,'median_seconds':med,'speedup':med['current_o0']/med['candidate_o3'],'samples_seconds':samples})
report={'scope':'Synthetic resident F32 matvec microbenchmark; same single-thread CPU process, alternating order; not full model or GGUF decoding. No fast-math flags.','libraries':{k:{'path':str(v),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in libs.items()},'cases':result}
Path('build/perf-audit/o3-benchmark.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(result,indent=2))
