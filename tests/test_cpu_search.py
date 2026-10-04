"""Search control-plane tests: no compiler/trained weights required here."""
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import pytest
from validation import cpu_search as s


def xml(path,*,skip=False,empty=False,missing=False,error=False):
    root=ET.Element('testsuites');suite=ET.SubElement(root,'testsuite',errors='1' if error else '0')
    if not empty:
        for module in (*s.FILES,'tests/test_packed_matvec.py'):
            if missing and 'topk' in module:continue
            case=ET.SubElement(suite,'testcase',name='check',classname=Path(module).stem)
            if skip:ET.SubElement(case,'skipped')
    ET.ElementTree(root).write(path)


def report(lib='a'*64,base='b'*64,repeats=3,factor=1):
    return {'library_sha256':lib,'baseline_sha256':base,'cases':[
        {'id':name,'samples_seconds':[factor]*repeats,'median_seconds':factor,
         'exact_baseline':True,'output_sha256':'c'*64} for name in s.CASE_IDS]}


def test_gate_fixed_selectors_exclude_only_wrong_library_and_never_trained(tmp_path):
    argv=s.test_argv(Path('/venv/bin/python'),tmp_path/'gate.xml')
    assert argv[0]=='/venv/bin/python' and argv[1:3]==['-m','pytest']
    assert argv[argv.index('--deselect')+1]==s.DESELECT
    assert 'tests/test_cpu_candidate_gate.py' in argv
    assert all('trained' not in node for node in s.PACKED_NODES)
    assert all('gpu' not in node for node in argv)
    assert '--junitxml' in argv


@pytest.mark.parametrize('kwargs',[{'skip':True},{'empty':True},{'missing':True},{'error':True}])
def test_gate_rejects_skips_missing_modules_empty_or_session_error(tmp_path,kwargs):
    path=tmp_path/'gate.xml';xml(path,**kwargs)
    with pytest.raises(s.SearchError):s.inspect_junit(path)


def test_gate_success_requires_all_modules(tmp_path):
    path=tmp_path/'gate.xml';xml(path)
    assert s.inspect_junit(path)['skips']==0
    assert s.inspect_junit(path)['passed']==len(s.FILES)+1
    path.write_text('broken')
    with pytest.raises(s.SearchError):s.inspect_junit(path)


def test_source_freeze_includes_build_script_and_detects_add_remove_edit(tmp_path):
    (tmp_path/'kernels/build').mkdir(parents=True)
    (tmp_path/'pyproject.toml').write_text('a')
    (tmp_path/'kernels/build.sh').write_text('original')
    (tmp_path/'kernels/build/generated.py').write_text('ignored')
    before=s.snapshot(tmp_path)
    assert 'kernels/build.sh' in before
    assert 'kernels/build/generated.py' not in before
    (tmp_path/'kernels/build.sh').write_text('changed')
    (tmp_path/'kernels/new.vx').write_text('new')
    (tmp_path/'pyproject.toml').unlink()
    assert s.source_changes(before,s.snapshot(tmp_path))==['kernels/build.sh','kernels/new.vx','pyproject.toml']


@pytest.mark.parametrize('change', ['hash','cases','repeats','nan','zero','median','semantic','output'])
def test_benchmark_rejects_unbound_or_invalid_results(change):
    data=report()
    if change=='hash':data['library_sha256']='x'
    if change=='cases':data['cases'].pop()
    if change=='repeats':data['cases'][0]['samples_seconds'].pop()
    if change=='nan':data['cases'][0]['samples_seconds'][0]=float('nan')
    if change=='zero':data['cases'][0]['samples_seconds'][0]=0
    if change=='median':data['cases'][0]['median_seconds']=99
    if change=='semantic':data['cases'][0]['exact_baseline']=False
    if change=='output':data['cases'][0].pop('output_sha256')
    with pytest.raises(s.SearchError):s.validate_benchmark(data,'a'*64,'b'*64,3)


def test_selection_is_matched_excludes_failed_and_requires_o0():
    candidates=[{'optimization':i,'scoped_eligible':i!=3,'benchmark':report(factor=f)} for i,f in enumerate((4,2,1,.001))]
    best=s.rank_candidates(candidates)
    assert best['optimization']==2
    assert best['geomean_speedup_vs_o0']==pytest.approx(4)
    candidates[0]['scoped_eligible']=False
    with pytest.raises(s.SearchError,match='O0'):s.rank_candidates(candidates)


def test_selection_rejects_mismatched_semantic_outputs():
    candidates=[{'optimization':i,'scoped_eligible':True,'benchmark':report()} for i in (0,1)]
    candidates[1]['benchmark']['cases'][0]['output_sha256']='d'*64
    with pytest.raises(s.SearchError,match='outputs differ'):s.rank_candidates(candidates)


def test_argv_is_literal_and_timeout_ends_child(tmp_path):
    argv=[sys.executable,'-c','import sys;print(sys.argv[1])','$(touch NOT_EXECUTED); echo bad']
    receipt=s.command(argv,tmp_path,os.environ.copy(),tmp_path/'literal.txt',10)
    assert receipt['returncode']==0
    assert '$(touch NOT_EXECUTED)' in (tmp_path/'literal.txt').read_text()
    assert not (tmp_path/'NOT_EXECUTED').exists()
    receipt=s.command([sys.executable,'-c','import time;time.sleep(30)'],tmp_path,os.environ.copy(),tmp_path/'timeout.txt',.1)
    assert receipt['timed_out'] and receipt['returncode'] is None


def test_missing_required_test_writes_nonrelease_failure_without_commands(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir()
    monkeypatch.setattr(s,'command',lambda *a:pytest.fail('missing input must fail before executing'))
    receipt=s.run_search(tmp_path/'out',Path(sys.executable),Path('/absent'),tmp_path/'reference',root=root)
    assert receipt['status']=='failed' and receipt['selected'] is None
    assert receipt['release_eligible'] is False
    assert 'missing required' in receipt['error']
    assert json.loads((tmp_path/'out/receipt.json').read_text())==receipt
    with pytest.raises(s.SearchError,match='already exists'):
        s.run_search(tmp_path/'out',Path(sys.executable),Path('/absent'),tmp_path/'ref',root=root)


def test_output_cannot_overlap_source(tmp_path):
    with pytest.raises(s.SearchError,match='outside source'):
        s.run_search(tmp_path/'validation/out',Path(sys.executable),Path('/absent'),tmp_path/'ref',root=tmp_path)


@pytest.mark.parametrize('fault',[None,'source','library','skip','build','benchhash','collection','toolchain'])
def test_orchestration_binds_hashes_revokes_drift_never_promotes(tmp_path,monkeypatch,fault):
    from validation import ggml_oracle
    root=tmp_path/'repo';root.mkdir()
    for name in (*s.FILES,'tests/test_packed_matvec.py','kernels/build.sh','validation/cpu_search.py','pyproject.toml'):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# frozen\n')
    default=root/'kernels/build/libglm_vx.so';default.parent.mkdir();default.write_bytes(b'never replace')
    compiler=tmp_path/'vxc';compiler.write_bytes(b'compiler')
    toolchain=tmp_path/'compiler-environment';toolchain.write_bytes(b'pinned environment')
    reference=tmp_path/'ref';reference.mkdir();native=reference/'native.so';native.write_bytes(b'oracle')
    class Native:
        def __init__(self,*a):pass
        def provenance(self):return {'codec_source_sha256':{},'library':{'path':str(native),'sha256':s.digest(native)}}
    monkeypatch.setattr(ggml_oracle,'NativeGGML',Native)
    def fake_command(argv,cwd,env,log,timeout):
        log.write_text('controlled command')
        if argv[0]=='bash':
            lib=Path(env['GLM_VX_LIBRARY']);lib.parent.mkdir();lib.write_bytes(('lib'+env['VX_OPT_LEVEL']).encode())
            if fault=='source':(root/'kernels/build.sh').write_text('tampered')
            if fault=='toolchain':toolchain.write_bytes(b'changed environment')
            if fault=='build':return {'returncode':1}
        elif '--junitxml' in argv:
            assert env['PYTHONPATH']==str(root)
            assert env['GLM_VX_CANDIDATE_SHA256']==s.digest(env['GLM_VX_LIBRARY'])
            junit=Path(argv[argv.index('--junitxml')+1])
            xml(junit,skip=fault=='skip')
            if fault=='collection' and env['VX_OPT_LEVEL']=='1':
                tree=ET.parse(junit);tree.find('.//testcase').set('name','different_test');tree.write(junit)
            if fault=='library':Path(env['GLM_VX_LIBRARY']).write_bytes(b'tampered')
        elif '--worker-library' in argv:
            lib=Path(argv[argv.index('--worker-library')+1]);base=Path(argv[argv.index('--worker-baseline')+1])
            data=report(s.digest(lib),s.digest(base),factor=1/(int(lib.parent.parent.name[1:])+1))
            if fault=='benchhash':data['library_sha256']='bad'
            Path(argv[argv.index('--worker-output')+1]).write_text(json.dumps(data))
        return {'returncode':0,'argv':argv}
    monkeypatch.setattr(s,'command',fake_command)
    result=s.run_search(tmp_path/'out',Path(sys.executable),compiler,reference,repeats=3,root=root,toolchain_files=[toolchain])
    assert result['release_eligible'] is False
    assert default.read_bytes()==b'never replace'
    assert result['default_library']['unchanged']
    if fault=='collection':
        assert result['status']=='complete' and result['selected']['optimization']==3
        assert not result['candidates'][1]['scoped_eligible']
        assert 'collection differs' in result['candidates'][1]['error']
    elif fault is None:
        assert result['status']=='complete' and result['selected']['optimization']==3
        assert result['selected']['release_eligible'] is False
        assert all(c['tests']['skips']==0 and c['scoped_eligible'] for c in result['candidates'])
    else:
        assert result['status']=='failed' and result['selected'] is None
        assert not any(c['scoped_eligible'] for c in result['candidates'])


def test_root_pytest_configuration_is_frozen(tmp_path):
    for name in ('conftest.py', 'pytest.ini', 'tox.ini', 'setup.cfg'):
        (tmp_path/name).write_text('before')
    before = s.snapshot(tmp_path)
    assert set(before) == {'conftest.py', 'pytest.ini', 'tox.ini', 'setup.cfg'}
    (tmp_path/'pytest.ini').write_text('after')
    assert s.source_changes(before, s.snapshot(tmp_path)) == ['pytest.ini']


def test_duplicate_test_identity_is_rejected(tmp_path):
    path = tmp_path/'gate.xml'
    xml(path)
    tree = ET.parse(path)
    import copy
    tree.find('.//testsuite').append(copy.deepcopy(tree.find('.//testcase')))
    tree.write(path)
    with pytest.raises(s.SearchError, match='duplicate'):
        s.inspect_junit(path)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_benchmark_rejects_shared_nonfinite_outputs(bad):
    import numpy as np
    with pytest.raises(s.SearchError, match='finite'):
        s.assert_finite_exact(np.array([bad]), np.array([bad]))
    s.assert_finite_exact(np.array([1.0]), np.array([1.0]))


def test_gate_overrides_ambient_pytest_addopts(tmp_path):
    argv = s.test_argv(Path('/venv/bin/python'), tmp_path/'gate.xml', tmp_path)
    assert 'addopts=' in argv
    assert argv[argv.index('-c')+1] == str(tmp_path/'pyproject.toml')
