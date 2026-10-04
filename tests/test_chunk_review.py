"""Independent stress checks against root's chunk executor; no root file edits."""
import threading
import numpy as np
import pytest
from glm_vx.model import Model,tiny_weights
from glm_vx.prefill import prefill
from glm_vx.tiny import tiny_config
from glm_vx.scheduler import Scheduler
from kernels.backend import VxBackend


def pair(**cfg):
    c=tiny_config(); c.update(cfg); w=tiny_weights(c,83)
    return Model(c,w,VxBackend()),Model(c,w,VxBackend())


def same(a,b):
    assert a.position==b.position
    for x,y in zip(a.layers,b.layers):
        for key in ('latents','rope_keys','index_keys','selected_indices'):
            np.testing.assert_array_equal(getattr(x,key),getattr(y,key))


@pytest.mark.parametrize('types',[['full','shared','shared','shared'],['full']*4,['full','shared','full','shared']])
@pytest.mark.parametrize('topk',[1,4,128])
def test_boundary_chunks_and_multiple_shared_layers(types,topk):
    a,b=pair(indexer_types=types,index_topk=topk,attention_bias=True,tie_word_embeddings=True)
    ca,cb=a.new_cache(),b.new_cache()
    tokens=[(i*29+1)%256 for i in range(71)]
    for start,end in [(0,3),(3,67),(67,71)]:
        expected=a.prefill(tokens[start:end],ca)
        actual=prefill(b,tokens[start:end],cb,output_all_logits=True)
        np.testing.assert_array_equal(actual,expected);same(ca,cb)
    np.testing.assert_array_equal(a.forward(27,ca),b.forward(27,cb));same(ca,cb)


def test_stacked_alternative_experts_exact():
    a,b=pair()
    w=dict(a.weights)
    for layer in (1,2,3):
        prefix=f'model.layers.{layer}.mlp.experts'
        gate=[]; down=[]
        for expert in range(4):
            gate.append(np.concatenate([w.pop(f'{prefix}.{expert}.gate_proj.weight'),w.pop(f'{prefix}.{expert}.up_proj.weight')]))
            down.append(w.pop(f'{prefix}.{expert}.down_proj.weight'))
        w[prefix+'.gate_up_proj']=np.stack(gate);w[prefix+'.down_proj']=np.stack(down)
    a.weights=b.weights=w
    ca,cb=a.new_cache(),b.new_cache()
    np.testing.assert_array_equal(a.prefill([1,3,7,11,15,19],ca),prefill(b,[1,3,7,11,15,19],cb,output_all_logits=True))
    same(ca,cb)


@pytest.mark.parametrize('fail_at',[1,7,23,37,71,97,127,173,229,231,232])
def test_injected_weight_read_failure_rolls_back_entire_partial_chunk(fail_at):
    a,b=pair(); ca,cb=a.new_cache(),b.new_cache()
    for token in (1,3,9):a.forward(token,ca);b.forward(token,cb)
    original=b._weight; reads=0
    def fail(name):
        nonlocal reads
        reads+=1
        if reads==fail_at:raise RuntimeError('audit failure')
        return original(name)
    b._weight=fail
    try:
        prefill(b,[7,11,15,19,21,23,27,29],cb)
    except RuntimeError:
        same(ca,cb)
        b._weight=original
        np.testing.assert_array_equal(a.prefill([7,11,15,19,21,23,27,29],ca)[-1],prefill(b,[7,11,15,19,21,23,27,29],cb))
        same(ca,cb)
    else: pytest.fail(f'failure point {fail_at} not reached (reads={reads})')


def collect(req):
    ids=[]
    while True:
        e=req.events.get(timeout=10)
        if e['type']=='done':return ids,e
        ids.append(e['token_id'])


@pytest.mark.parametrize('length',[3,9])
def test_cancel_during_final_or_partial_chunk_no_snapshot_no_emission(length):
    a,b=pair();entered=threading.Event();release=threading.Event()
    method=b.prefill_chunk
    def blocked(tokens,cache,**kw):
        entered.set();assert release.wait(3);return method(tokens,cache,**kw)
    b.prefill_chunk=blocked
    scheduler=Scheduler(b,batched_prefill=True,prefill_chunk=4,prefix_cache_bytes=2**20)
    try:
        req=scheduler.submit([1]*length,3)
        assert entered.wait(2);scheduler.cancel(req);release.set()
        ids,e=collect(req)
        assert ids==[] and e['reason']=='cancelled'
        assert scheduler.prefix_cache.status()['entries']==0
        assert scheduler.status()['reserved_tokens']==0
        b.prefill_chunk=method
        assert collect(scheduler.submit([1]*length,3))[1]['reason']=='length'
    finally:release.set();scheduler.close()
