"""Bounded audit evidence; no trained weights or whole official model execution."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
from glm_vx.config import GLMConfig
from glm_vx.model import GlmMoeDsaModel, tiny_weights
from glm_vx.backend import NumpyBackend
from glm_vx.tiny import tiny_config
from kernels.backend import VxBackend

root = Path('build/upstream-audit')
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
pins = json.loads(Path('metadata/sources.json').read_text())
for name in ('configuration_glm_moe_dsa.py', 'modeling_glm_moe_dsa.py'):
    assert sha(root/name) == pins['sources'][name]['sha256']

# Execute the actual, unmodified pinned class __post_init__ AST in isolation.
# Only its parent class is stubbed: we inspect fields computed before super().
# This verifies source semantics, not full Transformers/model execution.
parsed = ast.parse((root/'configuration_glm_moe_dsa.py').read_text())
cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'GlmMoeDsaConfig')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__post_init__')
minimal = ast.ClassDef(name='OfficialConfigMethod', bases=[ast.Name(id='Base', ctx=ast.Load())], keywords=[], body=[copy.deepcopy(method)], decorator_list=[])
class Base:
    def __post_init__(self, **kwargs):
        pass
namespace = {'Base': Base}
exec(compile(ast.fix_missing_locations(ast.Module(body=[minimal], type_ignores=[])), str(root/'configuration_glm_moe_dsa.py'), 'exec'), namespace)
OfficialConfigMethod = namespace['OfficialConfigMethod']
raw = tiny_config()
for key in ('indexer_types', 'index_topk_freq', 'index_skip_topk_offset'):
    raw.pop(key)
checks = []
for patch in ({}, {'index_topk_freq': 4}, {'index_skip_topk_offset': 3}, {'index_topk_freq': 4, 'index_skip_topk_offset': 3}, {'index_topk_pattern':'FSSF'}):
    config = dict(raw, **patch)
    official = OfficialConfigMethod()
    official.__dict__.update(config)
    official.indexer_types = None
    official.layer_types = None
    official.__post_init__(**patch)
    actual = GLMConfig.from_dict(config)
    assert list(actual.indexer_types) == official.indexer_types
    checks.append({'patch':patch, 'official_indexer_types':official.indexer_types, 'actual_indexer_types':list(actual.indexer_types)})

# Reproduce the parent revision without changing any main/worktree source file.
old = types.ModuleType('audit_parent_config')
sys.modules[old.__name__] = old
exec(compile(subprocess.check_output(['git','show','3ae29e5:glm_vx/config.py']), '3ae29e5:glm_vx/config.py', 'exec'), old.__dict__)
raw['index_topk'] = 2
weights = tiny_weights(raw, seed=31)
tokens = [1,7,2,11,4,9]
metrics = []
for backend in (NumpyBackend, VxBackend):
    for chunk in (False, True):
        models = [GlmMoeDsaModel(config, weights, backend()) for config in (
            old.GLMConfig.from_dict(raw).to_dict(),
            GLMConfig.from_dict(raw).to_dict(),
            dict(raw,indexer_types=['full']*4))]
        caches = [m.new_cache() for m in models]
        outputs = [(m.prefill_chunk(tokens,c) if chunk else m.prefill(tokens,c)) for m,c in zip(models,caches)]
        before, after, expected = outputs
        assert np.array_equal(after, expected)
        assert not np.array_equal(before, expected)
        metrics.append(dict(backend=backend.__name__, execution='chunk' if chunk else 'sequential',
            before_max_abs_logit_difference=float(np.max(np.abs(before-expected))),
            before_changed_logit_count=int(np.count_nonzero(before != expected)),
            total_logits=int(expected.size),
            before_index_key_counts=[len(s.index_keys) for s in caches[0].layers],
            after_index_key_counts=[len(s.index_keys) for s in caches[1].layers],
            before_last_layer_selected=caches[0].layers[-1].selected_indices.tolist(),
            after_last_layer_selected=caches[1].layers[-1].selected_indices.tolist(),
            after_exact_logit_equality=True))

sources=[]
for name in ('configuration_glm_moe_dsa.py','modeling_glm_moe_dsa.py'):
    sources.append(dict(file=name, **pins['sources'][name]))
for repo, pin, pairs in (
    ('vllm-project/vllm','138810056093301f4881050fcf2b1786939da387', (
        ('vllm_dsa_attention.py','vllm/models/deepseek_v32/attention.py'),
        ('vllm_dsa_model.py','vllm/models/deepseek_v32/nvidia/model.py'),
        ('vllm_deepseek_v2.py','vllm/model_executor/models/deepseek_v2.py'))),
    ('sgl-project/sglang','92d60351e2eb0dc2a947ff4a18c79bc54b6c77ce', (
        ('sglang_model_config.py','python/sglang/srt/configs/model_config.py'),
        ('sglang_deepseek_v2.py','python/sglang/srt/models/deepseek_v2.py'),
        ('sglang_forward_mla.py','python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py'),
        ('sglang_index_topk_share.py','python/sglang/srt/layers/attention/index_topk_share.py')))):
    for filename,path in pairs:
        sources.append(dict(file=filename,url=f'https://raw.githubusercontent.com/{repo}/{pin}/{path}',sha256=sha(root/filename)))
report=dict(base_revision='3ae29e5',finding='checkpoint DSA scheduling defaults silently select wrong layer topology',
            scope='tiny random F32 CPU models only; no trained model run or weights read',
            official_source_execution='unmodified pinned __post_init__ method AST with a no-op base class; not full Transformers execution',
            official_schedule_checks=checks,seed=31,tokens=tokens,index_topk=2,regressions=metrics,
            vx_library_sha256=sha('kernels/build/libglm_vx.so'),sources=sources)
Path('docs/audit-evidence-20261004b/config-defaults.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'checks':len(checks),'regressions':metrics},indent=2))
