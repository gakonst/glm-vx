"""Plan validation tests use explicit test context; no GPU measurements fabricated."""
import hashlib
import json
import pytest
import gpu.backend as adapter
from gpu.testing.context import Context

class PlanContext(Context):
    name='plan-validation-test-context'
    compute_capability=(8,0)
    driver_version=12090

def fixture_plan(tmp_path):
    ptx=tmp_path/'kernels.ptx';ptx.write_bytes(b'test-only-not-executable-PTX')
    return {'schema':'glm-vx-matvec-tuning-v1','gpu_execution_verified':True,
        'device':PlanContext.name,'compute_capability':[8,0],'driver_version':12090,
        'dtype':'float32','shape':[4,4],'ptx_sha256':hashlib.sha256(ptx.read_bytes()).hexdigest(),
        'selected_block':64}

@pytest.mark.parametrize('bad',[{'device':'other'},{'driver_version':12000},{'ptx_sha256':'bad'},
                               {'dtype':'fp8-e4m3fn-block128'},{'selected_block':33},{'selected_block':True},
                               {'shape':[True,4]},{'gpu_execution_verified':False}])
def test_reject_mismatched_plan(tmp_path,monkeypatch,bad):
    monkeypatch.setattr(adapter,'CUDAContext',PlanContext)
    plan=fixture_plan(tmp_path);plan.update(bad);path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):adapter.GPUBackend(tmp_path,tuning_file=path)

def test_matching_shape_uses_selected_layout(tmp_path,monkeypatch):
    import numpy as np
    monkeypatch.setattr(adapter,'CUDAContext',PlanContext)
    plan=fixture_plan(tmp_path);path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    b=adapter.GPUBackend(tmp_path,tuning_file=path);calls=[];original=b._launch
    def observe(name,grid,block,args):calls.append((name,grid,block));return original(name,grid,block,args)
    b._launch=observe
    try:
        b.matvec(np.ones((4,4),'f4'),np.ones(4,'f4'))
        b.matvec(np.ones((5,4),'f4'),np.ones(4,'f4'))
        assert calls==[('matvec',2,64),('matvec',2,128)]
    finally:b.close()
