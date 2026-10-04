import ctypes as C, json, time, statistics, os
from pathlib import Path
import numpy as np
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
opt=os.environ.get('BENCH_OPT', '3')
lib=C.CDLL(str(Path('evidence/vx-row-tile/probe'+('-o0' if opt=='0' else '')+'.so').resolve()))
production=C.CDLL(str(Path('kernels/'+('build' if opt=='0' else 'build-o3')+'/libglm_vx.so').resolve()))
p=C.POINTER(C.c_float)
funcs={n:getattr(lib,n) for n in ('baseline','tile2','tile4','tile8','integrated')}
funcs['integrated']=production.glm_vx_matvec
for f in funcs.values(): f.argtypes=[p,p,p,C.c_int,C.c_int];f.restype=C.c_int
rng=np.random.default_rng(20261004)
checks=0
for rows in (0,1,2,3,4,7,8,9,15,16,17,31):
 for cols in (0,1,3,7,8,17,63,256,513):
  w=rng.normal(size=(rows,cols)).astype('f4');x=rng.normal(size=cols).astype('f4')
  outs={n:np.zeros(rows,'f4') for n in funcs}
  for n,f in funcs.items():
   assert f(outs[n].ctypes.data_as(p),w.ctypes.data_as(p),x.ctypes.data_as(p),rows,cols)==0
   assert np.array_equal(outs[n].view('u4'),outs['baseline'].view('u4')), (n,rows,cols)
  checks+=1
for f in funcs.values():
 for dims in ((-1,3),(3,-1)):
  assert f(None,None,None,*dims)==-1
results=[]
for rows,cols in ((8,64),(32,256),(256,256),(1024,512),(4096,512),(256,6144),(2048,6144),(4096,6144),(256,12288)):
 if opt=='0' and (rows,cols) not in ((32,256),(1024,512),(256,6144)): continue
 w=rng.normal(size=(rows,cols)).astype('f4');x=rng.normal(size=cols).astype('f4');out=np.zeros(rows,'f4')
 args=(out.ctypes.data_as(p),w.ctypes.data_as(p),x.ctypes.data_as(p),rows,cols)
 batches={n:[] for n in funcs};reps=max(3,min(200,4000000//(rows*cols)))
 for f in funcs.values():f(*args)
 for iteration in range(9):
  for n in rng.permutation(list(funcs)):
   t=time.perf_counter_ns()
   for _ in range(reps): funcs[n](*args)
   batches[n].append((time.perf_counter_ns()-t)/reps/1e6)
 med={n:statistics.median(t) for n,t in batches.items()}
 results.append({'shape':[rows,cols],'repetitions':reps,'ms':med,'samples_ms':batches,'speedup':{n:med['baseline']/v for n,v in med.items()}})
 print(rows,cols,results[-1]['speedup'],flush=True)
Path('evidence/vx-row-tile/results'+('-o0' if opt=='0' else '')+'.json').write_text(json.dumps({'correctness_shapes':checks,'seed':20261004,'affinity':list(os.sched_getaffinity(0)),'results':results},indent=2)+'\n')
