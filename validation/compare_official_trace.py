"""Compare bound official/Vx trained traces without changing frozen budgets.

This is a finite development comparison, never automatic release eligibility.
Official attention uses membership masks; raw score/slot order is retained in
archives, while causal membership and ID-paired route weights are compared.
"""
import argparse
import json
from pathlib import Path
import zipfile
import numpy as np
from validation.trace_identity import sha_file
from validation.trace_gate import validate_contract, compare_tensor, GateError, parse_json, same_json

REFERENCE_MODE = 'official-eager-cpu-f32-streamed-storage-v1'
CANDIDATE_MODE = 'f32-official-expert-order-v2'


def require(value, message):
    if not value:
        raise ValueError(message)


def hashes(value):
    return (isinstance(value, dict) and bool(value) and
            all(isinstance(k, str) and isinstance(v, str) and len(v) == 64 and
                all(c in '0123456789abcdef' for c in v) for k, v in value.items()))


def archive(path, receipt):
    require(sha_file(path) == receipt['arrays_sha256'], 'trace archive hash differs')
    with zipfile.ZipFile(path) as zipped:
        names = zipped.namelist()
        require(len(names) == len(set(names)) and all(n.endswith('.npy') for n in names),
                'duplicate or non-NPY archive member')
    result = np.load(path, allow_pickle=False)
    if len(result.files) != receipt['tensor_count']:
        result.close()
        raise ValueError('trace tensor count differs')
    return result


def integer_ids(value, count, upper, label, *, dtypes=("int64",)):
    require(value.dtype in tuple(np.dtype(t) for t in dtypes) and value.shape == (count,), label+' must be integral IDs with required dtype/shape')
    require(len(np.unique(value)) == count and np.all(value >= 0) and np.all(value < upper), label+' has invalid/duplicate IDs')
    return np.argsort(value, kind='stable')


def compare(contract_path, config_path, reference_dir, candidate_dir, library):
    report = {'status': 'fail', 'release_eligible': False, 'numeric': [], 'discrete': [],
              'errors': [], 'scope': 'finite full-layer/logit and discrete-routing comparison; no release, long-context or original-FP8 quality proof',
              'canonicalization': 'causal DSA membership; expert IDs sorted together with their corresponding weights',
              'routing_budget': 'unchanged inherited layer_output budget, diagnostic, not independently calibrated',
              'producer_attestation': False,
              'contract_reuse': 'unchanged config/checkpoint pins, tensor coverage and budgets; current producer/archive hashes recorded here, historical producer/archive contract fields do not describe this comparison'}
    try:
        contract = parse_json(Path(contract_path).read_text()); validate_contract(contract)
        cfg = parse_json(Path(config_path).read_text())
        require(sha_file(config_path) == contract['pins']['config_sha256'], 'config differs from contract')
        require(len(contract['cases']) == 1, 'this comparison requires exactly one complete teacher-forced case')
        case = contract['cases'][0]; tokens = case['token_ids']; positions = case['positions']
        require(positions == list(range(len(tokens))), 'every teacher-forced position is required')
        require(case['layers'] == list(range(cfg['num_hidden_layers'])), 'every model layer is required')
        require(case['vocab_size'] == cfg['vocab_size'], 'vocabulary differs')
        expected_specs = {}
        for pos in positions:
            for layer in case['layers']:
                expected_specs[f'p{pos}.layer.{layer}.output'] = ([cfg['hidden_size']], 'float', 'layer_output', pos, layer)
            expected_specs[f'p{pos}.logits'] = ([cfg['vocab_size']], 'logits', 'logits', pos, None)
        require(set(expected_specs) == {s['key'] for s in case['tensors']}, 'contract must cover every full layer output and full logits')
        for spec in case['tensors']:
            require(same_json([spec[k] for k in ('shape','kind','operation','position','layer')], list(expected_specs[spec['key']])) and spec['dtype'] == 'float32' and spec['infinity'] == 'forbid', 'contract layer/logit schema differs')
        refdir, canddir = Path(reference_dir), Path(candidate_dir)
        ref = parse_json((refdir/'receipt.json').read_text()); cand = parse_json((canddir/'receipt.json').read_text())
        require(ref['status'] == cand['status'] == 'complete', 'both traces must be complete')
        require(not ref.get('skipped') and not cand.get('skipped'), 'skipped trace work is forbidden')
        require(ref['numerical_mode'] == REFERENCE_MODE and cand['numerical_mode'] == CANDIDATE_MODE, 'unsupported numerical mode')
        require(ref['token_ids'] == cand['token_ids'] == tokens, 'teacher-forced tokens differ')
        require(cand.get('identity_bound') is True, 'candidate has no bound checkpoint identity; export with --contract')
        identity = cand['model_identity']; contract_hash = sha_file(contract_path)
        require(ref['contract_sha256'] == identity['contract_sha256'] == contract_hash, 'frozen provenance contract differs')
        require(ref['config_sha256'] == cand['config_sha256'] == identity['config_sha256'] == contract['pins']['config_sha256'], 'model config identity differs')
        expected = contract['pins']['weights_sha256']
        for who, files in (('reference', ref['checkpoint_files']), ('candidate', identity['checkpoint_files'])):
            require({k:v['declared_sha256'] for k,v in files.items()} == expected, who+' declared weight identity differs')
            require(all(type(v['bytes']) is int and v['bytes'] > 0 and type(v['mtime_ns']) is int for v in files.values()), 'bad checkpoint file metadata')
        require(same_json(ref['checkpoint_files'],identity['checkpoint_files']), 'checkpoint file metadata changed between runs')
        for who, receipt in (('reference',ref),('candidate',cand)):
            loaded = receipt.get('loaded_checkpoint_paths',{})
            require(set(loaded) == set(expected) and len(set(loaded.values())) == len(expected) and all(Path(v).is_absolute() and Path(v).name == Path(k).name for k,v in loaded.items()), who+' loaded shard binding missing or inconsistent')
        pins = parse_json((Path(__file__).resolve().parents[1]/'metadata/sources.json').read_text())
        require(ref['transformers_revision'] == pins['transformers_revision'], 'official Transformers revision differs')
        for name in ('modeling_glm_moe_dsa.py', 'configuration_glm_moe_dsa.py'):
            matches = [v for k,v in ref['dependencies']['files_sha256'].items() if Path(k).name == name]
            require(matches == [pins['sources'][name]['sha256']], 'official source pin differs: '+name)
        require(hashes(ref['dependencies']['files_sha256']) and hashes(cand['source_sha256']), 'source/dependency pins missing or malformed')
        codec = parse_json((Path(__file__).resolve().parents[1]/'docs/parity-evidence/ggml-codec-parity.json').read_text())
        dependencies = cand.get('dependencies',{})
        require(hashes(dependencies.get('files_sha256')), 'candidate decoder dependencies missing')
        require([v for k,v in dependencies['files_sha256'].items() if Path(k).name == 'quants.py'] == [codec['candidate_decoder_source']['sha256']], 'candidate GGUF decoder differs from validated source')
        require(set(dependencies.get('versions',{})) >= {'numpy','gguf'} and any('_multiarray_umath' in k for k in dependencies['files_sha256']), 'candidate numeric dependency coverage incomplete')
        paths = cand.get('source_paths',{})
        require(set(paths) == set(cand['source_sha256']) and all(Path(p).is_absolute() for p in paths.values()), 'candidate executed source paths missing')
        required_sources = {'glm_vx/model.py','glm_vx/config.py','glm_vx/checkpoint.py','glm_vx/gguf_checkpoint.py',
                            'glm_vx/gguf_reader.py','kernels/backend.py','kernels/packed.py',
                            'validation/trace_identity.py','validation/trace_gate.py','validation/export_vx_trace.py'}
        require(required_sources <= set(cand['source_sha256']), 'candidate source coverage incomplete')
        require(sha_file(library) == cand['library_sha256'], 'candidate library differs from trace')
        report['identity'] = {'contract_sha256': contract_hash, 'reference_receipt_sha256': sha_file(refdir/'receipt.json'),
                              'candidate_receipt_sha256': sha_file(canddir/'receipt.json'),
                              'reference_arrays_sha256': ref['arrays_sha256'], 'candidate_arrays_sha256': cand['arrays_sha256'],
                              'candidate_library_sha256': cand['library_sha256'], 'checkpoint_bytes_rehashed': False}
        def numeric(key, r, a, spec, budget):
            row = {'key': key}
            try:
                row.update(compare_tensor(r, a, spec, budget, {'tensor': key})); row['pass'] = True
            except GateError as exc:
                row.update({'pass': False, 'failure': exc.detail})
            report['numeric'].append(row)
        with archive(refdir/'arrays.npz', ref) as r, archive(canddir/'arrays.npz', cand) as a:
            for spec in case['tensors']:
                key = spec['key']; require(key in r and key in a, 'missing required tensor: '+key)
                numeric(key, r[key], a[key], spec, contract['budgets'][spec['operation']])
            layer_types = cfg.get('mlp_layer_types') or ['dense' if i < cfg['first_k_dense_replace'] else 'sparse' for i in case['layers']]
            require(len(layer_types) == cfg['num_hidden_layers'] and set(layer_types) <= {'dense','sparse'}, 'unsupported MLP pattern')
            for pos in positions:
                for layer in case['layers']:
                    prefix = f'p{pos}.layer.{layer}'; key = prefix+'.selected'
                    require(key in r and key in a and prefix+'.official_selected' in r, 'missing selection: '+prefix)
                    ri, ai, raw = r[key], a[key], r[prefix+'.official_selected']
                    integer_ids(raw, min(cfg['index_topk'], len(tokens)), len(tokens), 'raw official selection', dtypes=('int32','int64'))
                    integer_ids(ri, len(ri), pos+1, key); integer_ids(ai, len(ai), pos+1, key)
                    require(np.array_equal(ri, np.sort(raw[raw <= pos])), 'official causal-membership trace contradicts raw slots')
                    require(len(ri) == len(ai) == min(cfg['index_topk'], pos+1), 'incomplete causal selection')
                    report['discrete'].append({'key': key, 'pass': bool(np.array_equal(np.sort(ri), np.sort(ai))), 'comparison': 'causal membership'})
                    if layer_types[layer] == 'sparse':
                        ids = prefix+'.mlp.route_ids'; weights = prefix+'.mlp.route_weights'
                        require(all(k in z for z in (r,a) for k in (ids,weights)), 'missing sparse routing: '+prefix)
                        count, experts = cfg['num_experts_per_tok'], cfg['n_routed_experts']
                        rp = integer_ids(r[ids], count, experts, ids); ap = integer_ids(a[ids], count, experts, ids)
                        same = bool(np.array_equal(r[ids][rp], a[ids][ap]))
                        report['discrete'].append({'key': ids, 'pass': same, 'comparison': 'expert membership, preserving weight pairing'})
                        spec = {'shape': [count], 'dtype': 'float32', 'kind': 'float', 'infinity': 'forbid'}
                        require(r[weights].shape == a[weights].shape == (count,), 'invalid route weight shape')
                        numeric(weights, r[weights][rp], a[weights][ap], spec, contract['budgets']['layer_output'])
        report['status'] = 'pass' if all(x['pass'] for x in report['numeric']+report['discrete']) else 'fail'
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    report['first_failure'] = next((x for x in report['numeric']+report['discrete'] if not x['pass']), None)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('contract','config','reference','candidate','library','output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists(): p.error('output exists; preserve prior evidence')
    result = compare(args.contract,args.config,args.reference,args.candidate,args.library)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('numeric','discrete')}))
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
