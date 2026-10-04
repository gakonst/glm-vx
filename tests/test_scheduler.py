import time
import numpy as np
import pytest
from glm_vx.scheduler import Scheduler,BusyError
class Model:
    config={'vocab_size':8,'max_position_embeddings':32,'eos_token_id':[]}
    def new_cache(self):return []
    def forward(self,t,c):
        c.append(t); out=np.zeros(8,dtype=np.float32); out[(sum(c)+1)%8]=10; return out

def collect(req):
    tokens=[]
    while True:
        e=req.events.get(timeout=5)
        if e['type']=='done':return tokens,e
        tokens.append(e['token_id'])

def test_concurrent_isolation_and_release():
    s=Scheduler(Model(),prefill_chunk=1)
    try:
        a=s.submit([1,2,3],4);b=s.submit([3,2,1],5)
        aa,ea=collect(a);bb,eb=collect(b)
        assert aa==collect(s.submit([1,2,3],4))[0]
        assert bb==collect(s.submit([3,2,1],5))[0]
        assert ea['reason']==eb['reason']=='length'
        assert s.status()['reserved_tokens']==0
    finally:s.close()

def test_admission_validation():
    s=Scheduler(Model(),token_budget=5)
    try:
        with pytest.raises(BusyError):s.submit([1,2],4)
        for tokens,n in [([],1),([True],1),([8],1),([1],0),([1],True)]:
            with pytest.raises(ValueError):s.submit(tokens,n)
        for temp in [float('nan'),-1,float('inf')]:
            with pytest.raises(ValueError):s.submit([1],1,temp)
        assert s.status()['reserved_tokens']==0
    finally:s.close()

def test_cancel_error_releases():
    class Slow(Model):
        def forward(self,t,c):time.sleep(.01);return super().forward(t,c)
    s=Scheduler(Slow(),prefill_chunk=1)
    try:
        r=s.submit([1]*10,10);r.cancelled.set()
        assert collect(r)[1]['reason']=='cancelled'
        assert s.status()['reserved_tokens']==0
    finally:s.close()
    class Bad(Model):
        def forward(self,t,c):raise RuntimeError('test failure')
    s=Scheduler(Bad())
    try:
        assert collect(s.submit([1],1))[1]['reason']=='error'
        assert s.status()['reserved_tokens']==0
    finally:s.close()

def test_shutdown_does_not_claim_worker_stopped():
    import threading
    entered,release=threading.Event(),threading.Event()
    class Blocked(Model):
        def forward(self,t,c):entered.set();release.wait(2);return super().forward(t,c)
    s=Scheduler(Blocked());s.submit([1],3)
    try:
        assert entered.wait(1)
        with pytest.raises(TimeoutError,match='resources must remain open'):s.close(timeout=.01)
        assert s.thread.is_alive()
    finally:
        release.set();s.close()
    assert not s.thread.is_alive()
    assert s.status()['reserved_tokens']==0
