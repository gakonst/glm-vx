import http.client
import json
import threading
import pytest
from glm_vx.server import Server
from glm_vx.scheduler import Scheduler
from test_scheduler import Model

@pytest.fixture
def endpoint():
    class Backend: name='test'
    model=Model();model.backend=Backend()
    sched=Scheduler(model,token_budget=12)
    server=Server(('127.0.0.1',0),sched)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield server.server_port
    server.shutdown();server.server_close();sched.close();thread.join()

def request(port,body):
    c=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
    c.request('POST','/v1/completions',json.dumps(body),{'Content-Type':'application/json'})
    r=c.getresponse();status=r.status;raw=r.read();c.close();return status,raw

def test_json_and_stream_agree(endpoint):
    status,raw=request(endpoint,{'prompt':[1,2],'max_tokens':4})
    assert status==200
    expected=json.loads(raw)
    status,raw=request(endpoint,{'prompt':[1,2],'max_tokens':4,'stream':True})
    assert status==200
    events=[json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith('data: ') and line!='data: [DONE]']
    assert [e['choices'][0]['token_ids'][0] for e in events[:-1]]==expected['choices'][0]['token_ids']
    assert events[-1]['usage']==expected['usage']
    assert raw.endswith(b'data: [DONE]\n\n')

def test_rejects_bad_unknown_overcapacity(endpoint):
    for body in [{},[],{'prompt':[1],'stream':'yes'}, {'prompt':[1],'stop':'x'}, {'prompt':[1],'max_tokens':True}]:
        assert request(endpoint,body)[0]==400
    assert request(endpoint,{'prompt':[1,2],'max_tokens':11})[0]==429

def test_disconnected_json_client_cancels_between_forwards():
    import socket
    import time
    entered,release=threading.Event(),threading.Event()
    class Blocking(Model):
        calls=0
        def forward(self,t,c):
            self.calls+=1;entered.set();release.wait(3);return super().forward(t,c)
    model=Blocking();sched=Scheduler(model)
    server=Server(('127.0.0.1',0),sched)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    client=socket.create_connection(server.server_address)
    try:
        body=json.dumps({'prompt':[1],'max_tokens':20}).encode()
        client.sendall(b'POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: '+str(len(body)).encode()+b'\r\n\r\n'+body)
        assert entered.wait(1)
        client.shutdown(socket.SHUT_RDWR);client.close()
        time.sleep(.4);release.set()
        deadline=time.monotonic()+2
        while sched.status()['active_requests'] and time.monotonic()<deadline:time.sleep(.01)
        assert sched.status()['active_requests']==0
        assert model.calls==1
    finally:
        release.set();client.close();server.shutdown();server.server_close();sched.close();thread.join()
