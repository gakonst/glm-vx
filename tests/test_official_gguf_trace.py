"""Optional pinned-Torch tests: streamed storage vs the actual ordinary model."""
import gc
import weakref

import numpy as np
import pytest

torch = pytest.importorskip('torch', reason='run in isolated pinned official-venv')
from tests.test_model import tiny_fixture
from validation.official_gguf_trace import (LazyExpertBank, TraceArchive, WeightStorage,
    materialize_layer, official_modules, prepare_config, streamed_forward)


def ordinary_fixture(seed, tokens, *, all_sparse=False):
    _, official, configuration, _ = official_modules()
    c, w = tiny_fixture(seed)
    if all_sparse:
        # A second architecture variant uses existing sparse layer1 weights for
        # layer0 and forces expert routing at the first decoder layer too.
        c['mlp_layer_types'][0] = 'sparse'
        for name,value in list(w.items()):
            if name.startswith('model.layers.1.mlp.'):
                w[name.replace('layers.1.', 'layers.0.')] = value.copy()
    cfg = configuration.GlmMoeDsaConfig(**c, attention_dropout=0.0)
    cfg._attn_implementation = 'eager'
    model = official.GlmMoeDsaForCausalLM(cfg).float().eval()
    state = {}
    for name in model.state_dict():
        if name.endswith('.experts.gate_up_proj'):
            prefix = name.removesuffix('.experts.gate_up_proj')
            value = np.stack([np.concatenate([w[f'{prefix}.experts.{e}.{p}_proj.weight'] for p in ('gate','up')]) for e in range(c['n_routed_experts'])])
        elif name.endswith('.experts.down_proj'):
            prefix = name.removesuffix('.experts.down_proj')
            value = np.stack([w[f'{prefix}.experts.{e}.down_proj.weight'] for e in range(c['n_routed_experts'])])
        else:
            value = w[name]
        state[name] = torch.from_numpy(value.copy())
    model.load_state_dict(state, strict=True)
    expected = {}
    for index,layer in enumerate(model.model.layers):
        def capture_layer(module, inputs, outputs, index=index):
            hidden,selected = outputs
            for p in range(len(tokens)):
                expected[p,f'layer.{index}.output'] = hidden[0,p].detach().numpy().copy()
                expected[p,f'layer.{index}.official_selected'] = selected[0,p].detach().numpy().copy()
                ids = selected[0,p].detach().numpy()
                expected[p,f'layer.{index}.selected'] = np.sort(ids[ids <= p]).astype(np.int64)
        layer.register_forward_hook(capture_layer)
        if c['mlp_layer_types'][index] == 'sparse':
            def capture_route(module, inputs, outputs, index=index):
                logits,weights,ids = outputs
                for p in range(len(tokens)):
                    for name,value in (('router_logits',logits),('route_weights',weights),('route_ids',ids)):
                        expected[p,f'layer.{index}.mlp.{name}'] = value[p].detach().numpy().copy()
            layer.mlp.gate.register_forward_hook(capture_route)
    with torch.no_grad():
        logits = model(torch.tensor([tokens]),use_cache=False).logits[0].numpy().copy()
    for p in range(len(tokens)):
        expected[p,'logits'] = logits[p]
    return c,w,expected,logits


class RecordingReader:
    def __init__(self, weights):
        self.weights = weights
        self.requests = []
        self.references = []

    def __call__(self, name, *, rows=None):
        assert not name.endswith(('.experts.gate_up_proj','.experts.down_proj'))
        # At the next layer/head load every previous layer allocation must
        # already be released; final-only cleanup would not establish bounded RAM.
        if name.startswith('model.layers.'):
            current = int(name.split('.')[2])
            assert all(ref() is None for old,ref in self.references
                       if old.startswith('model.layers.') and int(old.split('.')[2]) < current)
        elif name in ('model.norm.weight','lm_head.weight'):
            assert all(ref() is None for old,ref in self.references if old.startswith('model.layers.'))
        self.requests.append((name,rows))
        value = self.weights[name]
        if rows is not None:
            value = value[rows]
        result = np.array(value, dtype=np.float32, copy=True, order='C')
        result.flags.writeable = False
        self.references.append((name,weakref.ref(result)))
        return result


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


@pytest.mark.parametrize('seed,tokens,all_sparse', [(123,[1,7,2,11],False), (456,[3,4,5,6,7,8],True), (789,[5],False)])
def test_streamed_matches_ordinary_official_bitwise_and_loads_only_selected(seed,tokens,all_sparse,monkeypatch):
    c,w,expected,logits = ordinary_fixture(seed,tokens,all_sparse=all_sparse)
    _,official,_,_ = official_modules()
    def forbidden(*args,**kwargs):
        raise AssertionError('streamed execution must not instantiate full model')
    monkeypatch.setattr(official.GlmMoeDsaForCausalLM,'__init__',forbidden)
    monkeypatch.setattr(official.GlmMoeDsaModel,'__init__',forbidden)
    reader,actual = RecordingReader(w),{}
    got,stats = streamed_forward(c,reader,tokens,lambda p,n,v: actual.__setitem__((p,n),v))
    np.testing.assert_array_equal(got,logits)
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_array_equal(actual[key],expected[key],err_msg=str(key))
    requested = {}
    for prefix,kind,expert in stats['lazy_expert_requests']:
        requested.setdefault((prefix,kind),[]).append(expert)
    for layer in range(c['num_hidden_layers']):
        if c['mlp_layer_types'][layer] != 'sparse':
            continue
        ids = sorted({int(expert) for p in range(len(tokens)) for expert in expected[p,f'layer.{layer}.mlp.route_ids']})
        for kind in ('gate_up','down'):
            assert requested[f'model.layers.{layer}.mlp',kind] == ids
    embedding_requests = [rows for name,rows in reader.requests if name=='model.embed_tokens.weight']
    assert embedding_requests == [slice(t,t+1) for t in tokens]
    # No layer/head tensors stay retained in the wrapper after execution.
    gc.collect()
    assert all(ref() is None for _,ref in reader.references)


def test_only_expert_storage_changes_official_forward_identity():
    torch,official,configuration,_ = official_modules()
    c,w = tiny_fixture()
    cfg = prepare_config(c,configuration)
    storage = WeightStorage(RecordingReader(w),torch,max_bytes=1024**2)
    layer = materialize_layer(cfg,1,official,storage)
    assert isinstance(layer.mlp.experts.gate_up_proj,LazyExpertBank)
    assert isinstance(layer.mlp.experts.down_proj,LazyExpertBank)
    assert layer.forward.__func__ is official.GlmMoeDsaDecoderLayer.forward
    assert layer.mlp.forward.__func__ is official.GlmMoeDsaMoE.forward
    assert layer.mlp.experts.forward.__func__ is official.GlmMoeDsaExperts.forward
    assert layer.self_attn.forward.__func__ is official.GlmMoeDsaAttention.forward
    assert all(t.device.type=='cpu' for t in layer.parameters())
    assert all(t.ndim <= 2 for t in layer.parameters())


def test_large_expert_count_never_materializes_a_bank(monkeypatch):
    torch,official,configuration,_ = official_modules()
    c,w=tiny_fixture()
    c['n_routed_experts']=256
    w['model.layers.1.mlp.gate.weight']=np.zeros((256,c['hidden_size']),np.float32)
    w['model.layers.1.mlp.gate.e_score_correction_bias']=np.zeros(256,np.float32)
    cfg=prepare_config(c,configuration)
    original_empty=torch.empty
    allocated_banks=[]
    def checked_empty(*args,**kwargs):
        result=original_empty(*args,**kwargs)
        if result.ndim==3 and result.shape[0]==256:
            allocated_banks.append(result.device.type)
            assert result.device.type=='meta'
        return result
    monkeypatch.setattr(torch,'empty',checked_empty)
    reader=RecordingReader(w)
    layer=materialize_layer(cfg,1,official,WeightStorage(reader,torch,max_bytes=1024**2))
    assert allocated_banks == ['meta','meta']
    assert not any('.experts.' in name for name,_ in reader.requests)
    with pytest.raises(ValueError,match='integer'):
        layer.mlp.experts.gate_up_proj[:]
    with pytest.raises(ValueError,match='integer'):
        layer.mlp.experts.down_proj[torch.tensor([0,1])]


@pytest.mark.parametrize('change', [{'attention_bias':True},{'mlp_bias':True},{'n_group':2},{'hidden_act':'relu'},
    {'rope_parameters':{'rope_type':'yarn','rope_theta':10000}},{'indexer_types':['shared']*4},
    {'layer_types':['full_attention']*4},{'n_shared_experts':0}])
def test_unsupported_configs_rejected_before_weight_loading(change):
    c,_=tiny_fixture();c.update(change)
    def forbidden(*args,**kwargs):
        raise AssertionError('unsupported config loaded a weight')
    with pytest.raises(ValueError):
        streamed_forward(c,forbidden,[1])


@pytest.mark.parametrize('tokens', [[],list(range(33)),[-1],[23],[True]])
def test_invalid_token_scope_rejected(tokens):
    c,_=tiny_fixture()
    with pytest.raises(ValueError):
        streamed_forward(c,lambda *a,**k: (_ for _ in ()).throw(AssertionError('read')),tokens)


def test_memory_budget_and_rank_guard_before_read():
    c,w=tiny_fixture();reader=RecordingReader(w)
    with pytest.raises(ValueError,match='budget'):
        streamed_forward(c,reader,[1],max_weight_bytes=8)
    assert not reader.requests
    storage=WeightStorage(reader,torch,max_bytes=1024)
    with pytest.raises(ValueError,match='bank'):
        storage.read('bank',(4,8,12))
    assert not reader.requests


def test_observer_owns_arrays_and_trace_archive_is_safe(tmp_path):
    c,w,_,expected=ordinary_fixture(123,[1,7])
    got,_=streamed_forward(c,RecordingReader(w),[1,7],lambda p,n,v:v.fill(0))
    np.testing.assert_array_equal(got,expected)
    archive=TraceArchive(tmp_path/'arrays.npz')
    archive.capture(0,'logits',got[0])
    with pytest.raises(ValueError,match='duplicate'):
        archive.capture(0,'logits',got[0])
    archive.close()
    with np.load(tmp_path/'arrays.npz',allow_pickle=False) as data:
        np.testing.assert_array_equal(data['p0.logits'],got[0])


@pytest.mark.parametrize('corrupt', [False,True])
def test_cli_receipt_complete_or_failed_without_trained_io(tmp_path,monkeypatch,corrupt):
    import json
    import glm_vx.gguf_checkpoint
    import validation.official_gguf_trace as tool
    c,w,_,expected=ordinary_fixture(123,[1,7])
    folder=tmp_path/'gguf';folder.mkdir()
    (folder/'config.json').write_text(json.dumps(c))
    (folder/'fixture.gguf').write_bytes(b'fixture file, checkpoint provider replaced only in this test')
    contract=tmp_path/'contract.json'
    contract.write_text(json.dumps({'pins':{'config_sha256':tool.sha_file(folder/'config.json'), 'weights_sha256':{'fixture.gguf':'a'*64}}}))
    if corrupt:
        w['model.layers.0.input_layernorm.weight']=w['model.layers.0.input_layernorm.weight'].astype(np.float64)
    closed=[]
    class FixtureCheckpoint:
        def __init__(self,path,config,**kwargs):
            assert kwargs['cache_bytes']==0
        def tensor(self,name,*,rows=None):
            value=w[name]
            return value if rows is None else value[rows].copy()
        def close(self):
            closed.append(True)
    monkeypatch.setattr(glm_vx.gguf_checkpoint,'GGUFCheckpoint',FixtureCheckpoint)
    monkeypatch.setattr(tool,'dependency_pins',lambda *args:{'fixture_only':True})
    output=tmp_path/'output'
    argv=['--gguf',str(folder),'--contract',str(contract),'--tokens','1','7','--output',str(output)]
    if corrupt:
        with pytest.raises(ValueError,match='exact F32'):
            tool.main(argv)
    else:
        tool.main(argv)
    receipt=json.loads((output/'receipt.json').read_text())
    assert receipt['status']==('failed' if corrupt else 'complete')
    assert receipt['eligible'] is False
    assert closed==[True]
    if not corrupt:
        with np.load(output/'arrays.npz') as archive:
            np.testing.assert_array_equal(archive['p1.logits'],expected[1])
        assert receipt['arrays_sha256']==tool.sha_file(output/'arrays.npz')
    with pytest.raises(SystemExit):
        tool.main(argv)  # Existing evidence cannot be overwritten.


def test_codec_and_official_pins_are_enforced(monkeypatch):
    import validation.official_gguf_trace as tool
    real_sha=tool.sha_file
    def changed_source(path):
        if str(path).endswith('modeling_glm_moe_dsa.py'):
            return '0'*64
        return real_sha(path)
    monkeypatch.setattr(tool,'sha_file',changed_source)
    with pytest.raises(ValueError,match='installed official source differs'):
        tool.official_modules()
