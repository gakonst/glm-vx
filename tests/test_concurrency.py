import numpy as np
from glm_vx.model import Model,tiny_weights
from glm_vx.tiny import tiny_config
from kernels.backend import VxBackend
from glm_vx.scheduler import Scheduler
from test_scheduler import collect

def test_eight_vx_requests_match_serial_with_different_prompts():
    c=tiny_config();model=Model(c,tiny_weights(c,seed=44),VxBackend())
    prompts=[[i,2*i,3*i] for i in range(1,9)]
    s=Scheduler(model,max_sequences=8,token_budget=128,prefill_chunk=1)
    try:
        requests=[s.submit(p,6,temperature=.7,seed=100+i) for i,p in enumerate(prompts)]
        concurrent=[collect(r)[0] for r in requests]
        serial=[collect(s.submit(p,6,temperature=.7,seed=100+i))[0] for i,p in enumerate(prompts)]
        assert serial==concurrent
        assert s.status()['active_requests']==0
        assert s.status()['reserved_tokens']==0
    finally:s.close()
