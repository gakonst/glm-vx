"""Explicit CPU O0/O1/O2/O3 candidate search; never a release/model-parity gate.

Source the Vx toolchain env before invoking. All children use structured argv,
fixed repository tests/workloads, and isolated candidate library paths.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
from importlib.metadata import version
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
LEVELS=(0,1,2,3)
FILES=('tests/test_model.py','tests/test_batched_prefill.py','tests/test_topk_heap.py',
       'tests/test_packed_model.py','tests/test_backend_oracle.py','kernels/test_backend.py',
       'tests/test_cpu_candidate_gate.py','tests/test_chunk_review.py',
       'tests/test_prefill_serving.py','tests/test_prefix_cache.py','tests/test_scheduler.py',
       'tests/test_scheduler_latency.py','tests/test_server.py','tests/test_model_trace.py',
       'tests/test_dsa_boundaries.py','tests/test_trace_gate.py')
PACKED_NODES=('test_synthetic_raw_bytes_codec_and_dot_exact','test_iq1_all_2048_grid_indices_and_scale_bits',
              'test_half_lookup_all_finite_bit_patterns','test_bounds_before_kernel',
              'test_store_rejects_bad_expert_or_rows','test_empty_rows_and_close_waits_for_borrow',
              'test_linear_uses_borrowed_bytes_and_explicit_fallback','test_corrupt_scale_fails_closed',
              'test_table_extraction_is_reproducible','test_backend_without_packed_symbols_has_explicit_fallback',
              'test_f32_capacity_counts_elements_and_rejects_unaligned_bytes')
DESELECT='tests/test_model.py::ModelTests::test_compiled_vx_model_matches_independent_expanded_oracle'
CASE_IDS=('matrix-8-256-6144','matrix-8-1024-6144','matrix-16-512-2048','prefill-8','prefill-32')


class SearchError(RuntimeError):pass


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def snapshot(root):
    """Freeze implementation, test and benchmark inputs, including uncommitted files."""
    paths=[root/name for name in ('pyproject.toml','conftest.py','pytest.ini','tox.ini','setup.cfg')
           if (root/name).is_file()]
    for directory in ('glm_vx','kernels','tests','validation','benchmarks'):
        paths.extend(p for p in (root/directory).rglob('*') if p.is_file()
                     and not any(part.startswith('build') or part in ('__pycache__','.pytest_cache','packed-evidence','evidence','cpu-validation') for part in p.relative_to(root).parts[:-1])
                     and p.suffix in ('.py','.vx','.h','.sh','.bin','.json','.toml'))
    return {str(p.relative_to(root)):digest(p) for p in sorted(set(paths))}


def source_changes(before,after):
    return sorted(k for k in before.keys()|after.keys() if before.get(k)!=after.get(k))


def test_argv(python,junit,root=ROOT):
    return [str(python),'-m','pytest','-q','--maxfail=1','--strict-markers','-o','xfail_strict=true','-o','addopts=','-c',str(root/'pyproject.toml'),
            '--junitxml',str(junit),'--deselect',DESELECT,*FILES,
            *['tests/test_packed_matvec.py::'+node for node in PACKED_NODES]]


def inspect_junit(path):
    """A successful process alone is insufficient: missing/empty/skipped gates fail."""
    try:root=ET.parse(path).getroot()
    except (OSError,ET.ParseError) as exc:raise SearchError('missing or malformed JUnit report') from exc
    cases=list(root.iter('testcase'))
    if not cases:raise SearchError('CPU gate collected no tests')
    counts={tag:sum(len(c.findall(tag)) for c in cases) for tag in ('failure','error','skipped')}
    # Also catch collection/session errors represented at testsuite level.
    for tag,attr in [('failure','failures'),('error','errors'),('skipped','skipped')]:
        counts[tag]=max(counts[tag],sum(int(s.get(attr,'0')) for s in root.iter('testsuite')))
    if any(counts.values()):raise SearchError('CPU gate requires zero failures/errors/skips: '+str(counts))
    modules={p.rsplit('/',1)[-1][:-3] for p in (*FILES,'tests/test_packed_matvec.py')}
    present={part for c in cases for part in c.get('classname','').split('.')}
    missing=modules-present
    if missing:raise SearchError('CPU gate missing required test modules: '+','.join(sorted(missing)))
    nodes=sorted(c.get('classname','')+'::'+c.get('name','') for c in cases)
    if len(nodes)!=len(set(nodes)):raise SearchError('CPU gate contains duplicate test identities')
    return {'passed':len(cases),'failures':0,'errors':0,'skips':0,'sha256':digest(path),'node_ids':nodes}


def command(argv,cwd,env,log,timeout):
    """No shell interpolation. Every invocation and result stays in the receipt."""
    start=time.monotonic()
    with log.open('wb') as stream:
        process=subprocess.Popen(argv,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=timeout);timed_out=False
        except subprocess.TimeoutExpired:
            timed_out=True;code=None
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
    return {'argv':[str(x) for x in argv],'cwd':str(cwd),'returncode':code,'timed_out':timed_out,
            'seconds':time.monotonic()-start,'log':str(log),'log_sha256':digest(log)}


def validate_benchmark(report,library_hash,baseline_hash,repeats):
    if report.get('library_sha256')!=library_hash or report.get('baseline_sha256')!=baseline_hash:
        raise SearchError('benchmark library fingerprint mismatch')
    cases=report.get('cases',[])
    if [c.get('id') for c in cases]!=list(CASE_IDS):raise SearchError('benchmark workload mismatch')
    for case in cases:
        values=case.get('samples_seconds',[])
        if len(values)!=repeats or any(type(v) not in (float,int) or not math.isfinite(v) or v<=0 for v in values):
            raise SearchError('benchmark requires repeated finite positive timings')
        if case.get('median_seconds')!=statistics.median(values) or case.get('exact_baseline') is not True:
            raise SearchError('benchmark semantic/median check failed')
        if len(case.get('output_sha256',''))!=64:raise SearchError('missing benchmark output hash')
    return report


def rank_candidates(candidates):
    baseline=next((c for c in candidates if c['optimization']==0 and c.get('scoped_eligible')),None)
    if baseline is None:raise SearchError('O0 must pass all gates to provide matched baseline')
    base=baseline['benchmark']['cases'];ranked=[]
    for candidate in candidates:
        if not candidate.get('scoped_eligible'):continue
        cases=candidate['benchmark']['cases']
        if [c['id'] for c in cases]!=[c['id'] for c in base]:raise SearchError('candidate workloads differ')
        if any(x['output_sha256']!=y['output_sha256'] for x,y in zip(cases,base)):
            raise SearchError('candidate benchmark outputs differ from O0')
        ratios=[a['median_seconds']/b['median_seconds'] for a,b in zip(base,cases)]
        candidate['geomean_speedup_vs_o0']=math.exp(sum(math.log(r) for r in ratios)/len(ratios))
        ranked.append(candidate)
    return sorted(ranked,key=lambda c:(-c['geomean_speedup_vs_o0'],c['optimization']))[0]


def run_search(output,python,vxc,reference,repeats=7,timeout=300,root=ROOT,toolchain_files=()):
    root=Path(root).resolve();output=Path(output).resolve()
    if not 3<=repeats<=31:raise SearchError('repeats must be 3..31')
    if not 1<=timeout<=1800:raise SearchError('timeout must be 1..1800 seconds per command')
    if output.exists():raise SearchError('output directory already exists; evidence is immutable')
    # Evidence must not enter source roots and invalidate its own snapshot.
    for directory in ('glm_vx','kernels','tests','validation','benchmarks'):
        if output==root/directory or root/directory in output.parents:
            raise SearchError('put search output outside source directories (e.g. build/cpu-search/run1)')
    output.mkdir(parents=True)
    receipt={'schema':'glm-vx.cpu-search.v1','status':'running','release_eligible':False,
             'scope':'synthetic resident CPU compiler candidates only; no trained-model parity or GPU claim',
             'release_blocker':'global trained strict numerical gate has not passed',
             'optimization_levels':list(LEVELS),'repeats':repeats,'test_files':list(FILES),
             'packed_test_nodes':list(PACKED_NODES),'deselected_default_library_test':DESELECT,
             'replacement':'tests/test_cpu_candidate_gate.py binds the expanded model oracle to the candidate',
             'candidates':[],'selected':None,'ranking':'equal-case geometric mean of O0/candidate median time'}
    before={};default=root/'kernels/build/libglm_vx.so';default_before=digest(default) if default.is_file() else None
    try:
        for name in (*FILES,'tests/test_packed_matvec.py','kernels/build.sh','validation/cpu_search.py','pyproject.toml'):
            if not (root/name).is_file():raise SearchError('missing required source/test: '+name)
        before=snapshot(root);receipt['sources_before']=before
        python=Path(python).absolute();vxc=Path(vxc).resolve()
        if not python.is_file() or not vxc.is_file():raise SearchError('Python/compiler executable missing')
        receipt['python']={'path':str(python),'sha256':digest(python)}
        receipt['compiler']={'path':str(vxc),'sha256':digest(vxc)}
        receipt['toolchain_files']={str(Path(p).resolve()):digest(p) for p in toolchain_files}
        from validation.ggml_oracle import NativeGGML
        native=NativeGGML(reference);receipt['native_codec_resource']=native.provenance()
        env=os.environ.copy()
        for name in ('PYTEST_ADDOPTS','PYTEST_PLUGINS'):env.pop(name,None)
        env.update(PYTHONPATH=str(root),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONHASHSEED='0',
                   OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                   GLM_VX_GGML_REFERENCE=str(Path(reference).resolve()),VXC=str(vxc))
        receipt['compiler_version']=command([str(vxc),'--version'],root,env,output/'compiler-version.txt',timeout)
        if receipt['compiler_version']['returncode']!=0:raise SearchError('compiler version query failed')
        for level in LEVELS:
            directory=output/f'O{level}';directory.mkdir();lib=directory/'build/libglm_vx.so'
            candidate={'optimization':level,'library':str(lib),'scoped_eligible':False,'release_eligible':False}
            receipt['candidates'].append(candidate)
            cenv=env|{'VX_OPT_LEVEL':str(level),'VX_BUILD_DIR':str(directory/'build'),'GLM_VX_LIBRARY':str(lib)}
            try:
                candidate['build']=command(['bash',str(root/'kernels/build.sh')],root,cenv,directory/'build.txt',timeout)
                if candidate['build']['returncode']!=0 or not lib.is_file():raise SearchError('candidate build failed')
                candidate['library_sha256']=digest(lib)
                cenv['GLM_VX_CANDIDATE_SHA256']=candidate['library_sha256']
                junit=directory/'tests.xml'
                candidate['test_command']=command(test_argv(python,junit,root),root,cenv,directory/'tests.txt',timeout)
                if candidate['test_command']['returncode']!=0:raise SearchError('candidate semantic tests failed')
                candidate['tests']=inspect_junit(junit)
                if digest(lib)!=candidate['library_sha256']:raise SearchError('candidate library changed during tests')
                candidate['semantic_passed']=True
            except (SearchError,OSError,ValueError) as exc:candidate['error']=str(exc)
            changes=source_changes(before,snapshot(root))
            if changes:raise SearchError('source/test/benchmark files changed: '+','.join(changes))
        baseline=receipt['candidates'][0]
        if not baseline.get('semantic_passed'):raise SearchError('O0 semantic gate failed; no benchmarking or selection')
        for candidate in receipt['candidates']:
            if not candidate.get('semantic_passed'):continue
            directory=Path(candidate['library']).parent.parent;result=directory/'benchmark.json'
            cenv=env|{'GLM_VX_LIBRARY':candidate['library']}
            argv=[str(python),'-m','validation.cpu_search','--worker-library',candidate['library'],
                  '--worker-baseline',baseline['library'],'--worker-output',str(result),'--repeats',str(repeats)]
            try:
                if candidate['tests']['node_ids']!=baseline['tests']['node_ids']:
                    raise SearchError('candidate test collection differs from O0')
                candidate['benchmark_command']=command(argv,root,cenv,directory/'benchmark.txt',timeout)
                if candidate['benchmark_command']['returncode']!=0:raise SearchError('candidate benchmark failed')
                candidate['benchmark']=validate_benchmark(json.loads(result.read_text()),candidate['library_sha256'],baseline['library_sha256'],repeats)
                if digest(candidate['library'])!=candidate['library_sha256']:raise SearchError('candidate library changed during benchmark')
                if digest(baseline['library'])!=baseline['library_sha256']:raise SearchError('O0 library changed during benchmark')
                candidate['scoped_eligible']=True
            except (SearchError,OSError,ValueError) as exc:candidate['error']=str(exc)
            if source_changes(before,snapshot(root)):raise SearchError('source/test/benchmark changed during benchmark')
        winner=rank_candidates(receipt['candidates'])
        receipt['selected']={k:winner[k] for k in ('optimization','library','library_sha256','geomean_speedup_vs_o0')}
        receipt['selected']['release_eligible']=False
        receipt['status']='complete'
    except Exception as exc:
        receipt['status']='failed';receipt['error']=f'{type(exc).__name__}: {exc}'
    finally:
        after=snapshot(root);receipt['sources_after']=after
        receipt['changed_sources']=source_changes(before,after) if before else []
        default_after=digest(default) if default.is_file() else None
        receipt['default_library']={'before_sha256':default_before,'after_sha256':default_after,'unchanged':default_before==default_after}
        drift=bool(receipt['changed_sources']) or default_before!=default_after
        for candidate in receipt['candidates']:
            if candidate.get('library_sha256'):
                lib=Path(candidate['library']);current=digest(lib) if lib.is_file() else None
                candidate['final_library_sha256']=current
                drift=drift or current!=candidate['library_sha256']
        if receipt.get('python'):
            interpreter=Path(receipt['python']['path'])
            drift=drift or not interpreter.is_file() or digest(interpreter)!=receipt['python']['sha256']
        if receipt.get('compiler'):
            compiler=Path(receipt['compiler']['path'])
            drift=drift or not compiler.is_file() or digest(compiler)!=receipt['compiler']['sha256']
        for name,expected in receipt.get('toolchain_files',{}).items():
            path=Path(name)
            drift=drift or not path.is_file() or digest(path)!=expected
        if receipt.get('native_codec_resource'):
            resource=receipt['native_codec_resource']
            for name,expected in resource['codec_source_sha256'].items():
                path=Path(reference)/name
                drift=drift or not path.is_file() or digest(path)!=expected
            path=Path(resource['library']['path'])
            drift=drift or not path.is_file() or digest(path)!=resource['library']['sha256']
        if drift:
            receipt['status']='failed';receipt['error']='source/compiler/library drift detected; eligibility revoked'
        if receipt['status']!='complete':
            receipt['selected']=None
            for candidate in receipt['candidates']:candidate['scoped_eligible']=False
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return receipt


def assert_finite_exact(actual,expected):
    import numpy as np
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise SearchError('benchmark outputs must be finite, including O0')
    np.testing.assert_array_equal(actual,expected)


def benchmark_worker(library,baseline,output,repeats):
    """Fixed synthetic workload, fresh process per compiler candidate; no model IO."""
    import numpy as np
    from kernels.backend import VxBackend
    from glm_vx.model import GlmMoeDsaModel,tiny_weights
    from glm_vx.prefill import prefill
    from glm_vx.tiny import tiny_config
    if output.exists():raise SearchError('worker output exists')
    current=VxBackend(library);base=VxBackend(baseline);rng=np.random.default_rng(735)
    report={'library_sha256':digest(library),'baseline_sha256':digest(baseline),'cases':[],
            'seed':735,'scope':'fixed synthetic resident matrix and prefill; warmup excluded',
            'environment':{'python':sys.version,'numpy':version('numpy'),'pytest':version('pytest'),
                           'gguf':version('gguf'),'tokenizers':version('tokenizers'),
                           'os':platform.system(),'os_release':platform.release(),
                           'machine':platform.machine(),'logical_cpus':os.cpu_count(),
                           'cpu_affinity_count':len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None}}
    def measure(name,fn,expected):
        assert_finite_exact(fn(),expected)
        samples=[]
        for _ in range(repeats):
            start=time.perf_counter();actual=fn();elapsed=time.perf_counter()-start
            assert_finite_exact(actual,expected);samples.append(elapsed)
        report['cases'].append({'id':name,'samples_seconds':samples,'median_seconds':statistics.median(samples),
                                'exact_baseline':True,'output_sha256':hashlib.sha256(actual.tobytes()).hexdigest()})
    for m,n,k in ((8,256,6144),(8,1024,6144),(16,512,2048)):
        x=rng.normal(size=(m,k)).astype(np.float32);w=rng.normal(size=(n,k)).astype(np.float32)
        measure(f'matrix-{m}-{n}-{k}',lambda:current.linear_batch(w,x),base.linear_batch(w,x))
    c=tiny_config();c.update(hidden_size=256,q_lora_rank=128,kv_lora_rank=64,num_attention_heads=4,
        qk_nope_head_dim=32,qk_rope_head_dim=16,v_head_dim=32,index_n_heads=4,index_head_dim=32,
        intermediate_size=512,moe_intermediate_size=128,index_topk=16)
    weights=tiny_weights(c,seed=487);a=GlmMoeDsaModel(c,weights,base);b=GlmMoeDsaModel(c,weights,current)
    for count in (8,32):
        tokens=rng.integers(0,c['vocab_size'],size=count).tolist()
        measure(f'prefill-{count}',lambda:prefill(b,tokens,b.new_cache()),prefill(a,tokens,a.new_cache()))
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--python',type=Path,default=Path(sys.executable))
    p.add_argument('--vxc',type=Path,default=Path(shutil.which('vxc') or 'vxc'))
    p.add_argument('--reference',type=Path,default=ROOT.parent/'glm-vx-reference')
    p.add_argument('--repeats',type=int,default=7);p.add_argument('--timeout',type=int,default=300)
    p.add_argument('--toolchain-file',type=Path,action='append',default=[],help='Additional LLVM/linker/environment file to pin and recheck')
    p.add_argument('--worker-library',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--worker-baseline',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--worker-output',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.worker_library:
        if not args.worker_baseline or not args.worker_output:p.error('worker requires baseline/output')
        benchmark_worker(args.worker_library,args.worker_baseline,args.worker_output,args.repeats);return 0
    if not args.output_dir:p.error('--output-dir is required')
    try:receipt=run_search(args.output_dir,args.python,args.vxc,args.reference,args.repeats,args.timeout,toolchain_files=args.toolchain_file)
    except SearchError as exc:print(str(exc),file=sys.stderr);return 2
    print(json.dumps({'status':receipt['status'],'selected':receipt['selected'],'release_eligible':False,
                      'receipt':str(args.output_dir/'receipt.json'),'error':receipt.get('error')}))
    return 0 if receipt['status']=='complete' else 1

if __name__=='__main__':raise SystemExit(main())
