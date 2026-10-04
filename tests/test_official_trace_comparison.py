"""Independent mutation tests for bound official/Vx trace comparisons."""
import copy
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from validation.compare_official_trace import compare, main, REFERENCE_MODE, CANDIDATE_MODE
from validation.trace_identity import bind_model, bind_loaded, sha_file


@pytest.fixture
def evidence(tmp_path):
    model = tmp_path/'model'; model.mkdir()
    config = dict(num_hidden_layers=2, hidden_size=3, vocab_size=4,
                  first_k_dense_replace=1, index_topk=2,
                  num_experts_per_tok=2, n_routed_experts=3)
    config_path = model/'config.json'; config_path.write_text(json.dumps(config))
    (model/'shard.gguf').write_bytes(b'synthetic weight bytes')
    weight_hash = sha_file(model/'shard.gguf')
    (model/'manifest.json').write_text(json.dumps({'files':[{'rfilename':'quant/shard.gguf','lfs':{'sha256':weight_hash}}]}))
    refdir, canddir = tmp_path/'reference', tmp_path/'candidate'
    refdir.mkdir(); canddir.mkdir()
    lib = tmp_path/'candidate.so'; lib.write_bytes(b'fixture library')
    pin = 'a'*64
    producer = dict(binary_sha256=pin,compiler='synthetic',libraries_sha256={},flags=[],hardware='fixture')
    budget = dict(atol=1e-4,rtol=1e-5,normalized_l2=1e-4,normalization_floor=1e-6,
                  softmax_tv=1e-4,top_k=2,top_k_margin_atol=1e-4)
    specs, arrays = [], {}
    for pos in range(2):
        for layer in range(2):
            key=f'p{pos}.layer.{layer}.output'; arrays[key]=np.array([1,2,3],np.float32)
            specs.append(dict(key=key,operation='layer_output',kind='float',shape=[3],dtype='float32',position=pos,layer=layer,infinity='forbid'))
            arrays[f'p{pos}.layer.{layer}.selected']=np.arange(pos+1,dtype=np.int64)
        arrays[f'p{pos}.layer.1.mlp.route_ids']=np.array([0,2],np.int64)
        arrays[f'p{pos}.layer.1.mlp.route_weights']=np.array([.7,.3],np.float32)
        key=f'p{pos}.logits'; arrays[key]=np.array([1,2,3,4],np.float32)
        specs.append(dict(key=key,operation='logits',kind='logits',shape=[4],dtype='float32',position=pos,layer=None,infinity='forbid'))
    contract=dict(schema_version=1,contract_id='comparison-fixture-v1',scope='synthetic mutation tests',
                  pins=dict(config_sha256=sha_file(config_path),weights_sha256={'quant/shard.gguf':weight_hash},
                            tokenizer_sha256=pin,template_sha256=pin,corpus_sha256=pin,corpus_version='test',
                            reference_revision='a'*40,numerical_mode='f32',tie_breaking='stable-lowest-index'),
                  producers={'reference':producer,'candidate':copy.deepcopy(producer)},reference_trace_sha256=pin,
                  budgets={'layer_output':budget,'logits':copy.deepcopy(budget)},
                  cases=[dict(id='fixture',token_ids=[1,2],positions=[0,1],execution='decode',chunks=[1,1],
                              cache_owner='fixture',layers=[0,1],vocab_size=4,tensors=specs)])
    path=tmp_path/'contract.json'; path.write_text(json.dumps(contract))
    identity=bind_model(model,path)
    loaded=bind_loaded(identity,model,[model/'shard.gguf'])
    pins=json.loads((Path(__file__).resolve().parents[1]/'metadata/sources.json').read_text())
    ref=dict(status='complete',skipped=[],numerical_mode=REFERENCE_MODE,token_ids=[1,2],
             contract_sha256=sha_file(path),config_sha256=sha_file(config_path),checkpoint_files=identity['checkpoint_files'],
             transformers_revision=pins['transformers_revision'],
             dependencies={'files_sha256':{n:pins['sources'][n]['sha256'] for n in ('modeling_glm_moe_dsa.py','configuration_glm_moe_dsa.py')}})
    cand=dict(status='complete',skipped=[],numerical_mode=CANDIDATE_MODE,token_ids=[1,2],identity_bound=True,
              model_identity=identity,config_sha256=sha_file(config_path),library_sha256=sha_file(lib),
              source_sha256={'glm_vx/model.py':pin,'validation/export_vx_trace.py':pin})
    ref['loaded_checkpoint_paths']=loaded; cand['loaded_checkpoint_paths']=loaded
    codec=json.loads((Path(__file__).resolve().parents[1]/'docs/parity-evidence/ggml-codec-parity.json').read_text())
    cand['dependencies']={'files_sha256':{'/fake/gguf/quants.py':codec['candidate_decoder_source']['sha256'],'/fake/numpy/_multiarray_umath.so':pin},'versions':{'numpy':'fixture','gguf':'fixture'}}
    cand['source_sha256'].update({k:'a'*64 for k in ('glm_vx/config.py','glm_vx/checkpoint.py','glm_vx/gguf_checkpoint.py','glm_vx/gguf_reader.py','kernels/backend.py','kernels/packed.py','validation/trace_identity.py','validation/trace_gate.py')})
    cand['source_paths']={k:str((tmp_path/k).resolve()) for k in cand['source_sha256']}
    r=copy.deepcopy(arrays); a=copy.deepcopy(arrays)
    for pos in range(2):
        for layer in range(2):
            r[f'p{pos}.layer.{layer}.official_selected']=np.array([1,0],np.int64)
            a[f'p{pos}.layer.{layer}.selected']=a[f'p{pos}.layer.{layer}.selected'][::-1]
        for suffix in ('route_ids','route_weights'):
            k=f'p{pos}.layer.1.mlp.{suffix}'; a[k]=a[k][::-1]

    def write(ref_arrays=None,cand_arrays=None,ref_mutation=None,cand_mutation=None):
        for directory,data,receipt,mutation in ((refdir,r if ref_arrays is None else ref_arrays,ref,ref_mutation),
                                               (canddir,a if cand_arrays is None else cand_arrays,cand,cand_mutation)):
            np.savez(directory/'arrays.npz',**data)
            receipt=copy.deepcopy(receipt); receipt.update(arrays_sha256=sha_file(directory/'arrays.npz'),tensor_count=len(data))
            if mutation:mutation(receipt)
            (directory/'receipt.json').write_text(json.dumps(receipt))
        return compare(path,config_path,refdir,canddir,lib)
    return dict(model=model,contract=contract,path=path,config=config_path,reference=refdir,candidate=canddir,
                library=lib,r=r,a=a,write=write)


def test_paired_routes_and_causal_membership_pass_without_release(evidence):
    report=evidence['write']()
    assert report['status']=='pass',report
    assert len(report['numeric'])==8 and len(report['discrete'])==6
    assert report['release_eligible'] is False and report['producer_attestation'] is False
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize('mutation',[
    lambda r:r.update(status='partial'),lambda r:r.update(skipped=['layer']),
    lambda r:r.update(numerical_mode='unknown'),lambda r:r.update(token_ids=[2,1]),
    lambda r:r.update(loaded_checkpoint_paths={}),lambda r:r.update(source_paths={}),
    lambda r:r.update(dependencies={}),lambda r:r['dependencies']['files_sha256'].update({'/fake/gguf/quants.py':'b'*64}),
    lambda r:r.update(identity_bound=False),lambda r:r.update(config_sha256='b'*64),
    lambda r:r.update(library_sha256='b'*64),lambda r:r.update(arrays_sha256='b'*64),
    lambda r:r.update(tensor_count=1),lambda r:r.update(source_sha256={}),
    lambda r:r['model_identity'].update(contract_sha256='b'*64),
    lambda r:r['model_identity']['checkpoint_files']['quant/shard.gguf'].update(declared_sha256='b'*64),
    lambda r:r['model_identity']['checkpoint_files']['quant/shard.gguf'].update(bytes=999),
    lambda r:r['model_identity']['checkpoint_files']['quant/shard.gguf'].update(mtime_ns=999),
])
def test_candidate_receipt_mutations_fail_closed(evidence,mutation):
    r=evidence['write'](cand_mutation=mutation)
    assert r['status']=='fail' and r['errors'] and not r['release_eligible'],r


@pytest.mark.parametrize('mutation',[
    lambda r:r.update(transformers_revision='b'*40),
    lambda r:r['dependencies']['files_sha256'].update({'modeling_glm_moe_dsa.py':'b'*64}),
    lambda r:r.update(status='partial'),lambda r:r.update(contract_sha256='b'*64),
])
def test_reference_provenance_mutations_fail(evidence,mutation):
    r=evidence['write'](ref_mutation=mutation)
    assert r['status']=='fail' and r['errors'],r


@pytest.mark.parametrize('key,mutation',[
    ('p0.layer.0.output',lambda v:v+.1),
    ('p1.logits',lambda v:v+.1),
    ('p0.layer.0.output',lambda v:np.full_like(v,np.nan)),
    ('p0.layer.0.output',lambda v:v.astype(np.float64)),
    ('p0.layer.0.output',lambda v:v.reshape(1,-1)),
    ('p0.layer.1.mlp.route_weights',lambda v:v[::-1]),
    ('p0.layer.1.mlp.route_ids',lambda v:np.array([1,0],np.int64)),
    ('p0.layer.1.mlp.route_ids',lambda v:np.array([2,2],np.int64)),
    ('p1.layer.0.selected',lambda v:np.array([0,0],np.int64)),
    ('p0.layer.0.selected',lambda v:np.array([1],np.int64)),
    ('p1.layer.0.selected',lambda v:np.array([0],np.int64)),
    ('p1.layer.0.selected',lambda v:v.astype(np.int32)),
])
def test_bad_tensor_data_cannot_pass(evidence,key,mutation):
    a=copy.deepcopy(evidence['a']);a[key]=mutation(a[key])
    r=evidence['write'](cand_arrays=a)
    assert r['status']=='fail' and (r['first_failure'] or r['errors']),r


@pytest.mark.parametrize('key',['p0.layer.0.output','p1.logits','p0.layer.1.mlp.route_ids','p1.layer.0.selected'])
def test_missing_required_arrays_fail(evidence,key):
    a=copy.deepcopy(evidence['a']);del a[key]
    assert evidence['write'](cand_arrays=a)['errors']


def test_official_raw_slots_must_match_causal_selection(evidence):
    r=copy.deepcopy(evidence['r']);r['p0.layer.0.official_selected']=np.array([1,1],np.int64)
    result=evidence['write'](ref_arrays=r)
    assert result['status']=='fail' and result['errors']


def test_duplicate_zip_and_duplicate_json_rejected(evidence):
    evidence['write']();p=evidence['candidate']/'arrays.npz'
    with zipfile.ZipFile(p,'a') as z:
        with pytest.warns(UserWarning):z.writestr(z.namelist()[0],z.read(z.namelist()[0]))
    rp=evidence['candidate']/'receipt.json';r=json.loads(rp.read_text());r['arrays_sha256']=sha_file(p);rp.write_text(json.dumps(r))
    args=[evidence[k] for k in ('path','config','reference','candidate','library')]
    assert 'duplicate' in compare(*args)['errors'][0]
    evidence['write']();rp.write_text(rp.read_text()[:-1]+',"status":"complete"}')
    assert 'duplicate' in compare(*args)['errors'][0]


def test_archive_hash_detects_changed_bytes_and_library(evidence):
    evidence['write']();p=evidence['candidate']/'arrays.npz';p.write_bytes(p.read_bytes()+b'change')
    args=[evidence[k] for k in ('path','config','reference','candidate','library')]
    assert 'hash' in compare(*args)['errors'][0]
    evidence['write']();evidence['library'].write_bytes(b'other library')
    assert 'library' in compare(*args)['errors'][0]


def test_output_is_immutable(evidence,tmp_path):
    evidence['write']();out=tmp_path/'result.json';argv=[]
    for flag,key in [('contract','path'),('config','config'),('reference','reference'),('candidate','candidate'),('library','library')]:
        argv += ['--'+flag,str(evidence[key])]
    argv+=['--output',str(out)]
    assert main(argv)==0;original=out.read_bytes()
    with pytest.raises(SystemExit):main(argv)
    assert out.read_bytes()==original


@pytest.mark.parametrize('mutation',[
    lambda d:d['files'].append(copy.deepcopy(d['files'][0])),
    lambda d:d['files'][0]['lfs'].update(sha256='b'*64),
])
def test_bound_identity_rejects_manifest_mutation(evidence,mutation):
    p=evidence['model']/'manifest.json';d=json.loads(p.read_text());mutation(d);p.write_text(json.dumps(d))
    with pytest.raises(ValueError):bind_model(evidence['model'],evidence['path'])


def test_bound_identity_rejects_changed_config_missing_empty_and_duplicate_shards(evidence):
    p=evidence['config'];original=p.read_bytes();p.write_text('{}')
    with pytest.raises(ValueError,match='config'):bind_model(evidence['model'],evidence['path'])
    p.write_bytes(original);shard=evidence['model']/'shard.gguf';shard.write_bytes(b'')
    with pytest.raises(ValueError,match='empty'):bind_model(evidence['model'],evidence['path'])
    shard.unlink()
    with pytest.raises(FileNotFoundError):bind_model(evidence['model'],evidence['path'])
    shard.write_bytes(b'fixture')
    contract=evidence['contract'];weight_hash=contract['pins']['weights_sha256']['quant/shard.gguf']
    contract['pins']['weights_sha256']['other/shard.gguf']=weight_hash
    evidence['path'].write_text(json.dumps(contract))
    (evidence['model']/'manifest.json').write_text(json.dumps({'files':[{'rfilename':n,'lfs':{'sha256':v}} for n,v in contract['pins']['weights_sha256'].items()]}))
    with pytest.raises(ValueError,match='ambiguous'):bind_model(evidence['model'],evidence['path'])


def test_loaded_shards_must_be_the_bound_shards(evidence):
    identity=bind_model(evidence['model'],evidence['path'])
    path=evidence['model']/'shard.gguf'
    other=evidence['model']/'other-00001-of-00001.gguf';other.write_bytes(path.read_bytes())
    for paths in ([other],[path,other],[path,path],[]):
        with pytest.raises(ValueError,match='loaded checkpoint shards'):
            bind_loaded(identity,evidence['model'],paths)
    assert bind_loaded(identity,evidence['model'],[path])=={'quant/shard.gguf':str(path.resolve())}
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='metadata changed'):bind_loaded(identity,evidence['model'],[path])


def test_candidate_pins_bind_imported_sources_independent_of_cwd(tmp_path,monkeypatch):
    from validation.export_vx_trace import execution_pins
    before=execution_pins()
    monkeypatch.chdir(tmp_path)
    assert execution_pins()==before
    for key,path in before['source_paths'].items():
        assert Path(path).is_absolute() and sha_file(path)==before['source_sha256'][key]
    assert 'gguf' in before['dependencies']['versions']
