"""The real Vx library must be built: no skipped or fallback backend checks."""
import http.client
import json
import threading
import numpy as np
from glm_vx.model import Model,tiny_weights
from glm_vx.tiny import tiny_config
from glm_vx.backend import NumpyBackend
from glm_vx.scheduler import Scheduler
from glm_vx.server import Server
from kernels.backend import VxBackend

def test_vx_logits_sparse_history_reference():
    c=tiny_config();w=tiny_weights(c,seed=7)
    reference=Model(c,w,NumpyBackend());vx=Model(c,w,VxBackend())
    rc,vc=reference.new_cache(),vx.new_cache()
    # Longer than top-k=4; exercises DSA exclusion and full/shared selections.
    for token in [1,2,90,4,33,14,56,7,23,44,8,2]:
        a=reference.forward(token,rc);b=vx.forward(token,vc)
        np.testing.assert_allclose(b,a,rtol=2e-4,atol=2e-5)
        assert np.argmax(a)==np.argmax(b)
        for l in range(c['num_hidden_layers']):
            np.testing.assert_array_equal(rc.layers[l].selected_indices,vc.layers[l].selected_indices)
    assert vc.position==12

def test_real_vx_http_stream_generation():
    c=tiny_config();w=tiny_weights(c,seed=7)
    oracle=Model(c,w,NumpyBackend());cache=oracle.new_cache()
    for t in [1,2,3]:logits=oracle.forward(t,cache)
    expected=[]
    for _ in range(6):
        token=int(np.argmax(logits));expected.append(token);logits=oracle.forward(token,cache)
    scheduler=Scheduler(Model(c,w,VxBackend()))
    server=Server(('127.0.0.1',0),scheduler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=10)
        conn.request('POST','/v1/completions',json.dumps({'prompt':[1,2,3],'max_tokens':6,'stream':True}))
        r=conn.getresponse();assert r.status==200
        raw=r.read().decode();conn.close()
        events=[json.loads(x[6:]) for x in raw.splitlines() if x.startswith('data: ') and x!='data: [DONE]']
        assert [x['choices'][0]['token_ids'][0] for x in events[:-1]]==expected
        assert events[-1]['choices'][0]['finish_reason']=='length'
        assert events[-1]['usage']['completion_tokens']==6
        assert scheduler.status()['reserved_tokens']==0
    finally:server.shutdown();server.server_close();scheduler.close();thread.join()
