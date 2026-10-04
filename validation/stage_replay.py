"""Same-input, one-real-layer replay. Diagnostic only: never a promotion gate.

Reads prior layer outputs from an existing teacher-forced trace. Both paths use
exactly those inputs and causal positions; neither needs to run previous layers.
Budgets come unchanged from a frozen contract, provisionally applied to stages
for localization only (not a newly calibrated per-operation contract).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from glm_vx.model import GlmMoeDsaModel, LayerCache
from validation.expanded_reference import expanded_batch_oracle
from validation.trace_gate import compare_tensor, GateError


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_stages(reference, candidate, budget, *, logits_budget=None):
    """Retain candidate execution order; distinguish rounding from budget failure."""
    metrics = []
    for key, cand in candidate.items():
        if key not in reference:
            continue
        ref = reference[key]
        row = dict(key=key, reference_dtype=str(ref.dtype), candidate_dtype=str(cand.dtype),
                   exact=ref.dtype == cand.dtype and np.array_equal(ref, cand))
        spec = dict(shape=list(ref.shape), dtype=str(ref.dtype),
                    kind='indices' if ref.dtype.kind in 'iu' else ('logits' if logits_budget is not None and key.endswith('.logits') else 'float'), infinity='forbid')
        selected_budget = logits_budget if spec['kind'] == 'logits' else budget
        try:
            row.update(compare_tensor(ref, cand, spec, selected_budget, {'tensor': key}))
            row['within_inherited_budget'] = True
        except GateError as exc:
            row['within_inherited_budget'] = False
            row['failure'] = str(exc)
            row['failure_detail'] = exc.detail
        if ref.shape == cand.shape and ref.dtype.kind == 'f' and np.isfinite(ref).all() and np.isfinite(cand).all():
            delta = cand.astype(np.float64) - ref.astype(np.float64)
            row['max_abs'] = float(np.max(np.abs(delta), initial=0))
            row['normalized_l2'] = float(np.linalg.norm(delta) / max(np.linalg.norm(ref.astype(np.float64)), selected_budget['normalization_floor'] * np.sqrt(ref.size)))
            row['outside_elementwise_budget'] = int(np.count_nonzero(np.abs(delta) > selected_budget['atol'] + selected_budget['rtol'] * np.abs(ref)))
        metrics.append(row)
    return dict(scope='diagnostic, not gate eligibility; inherited layer-output budget is not a calibrated stage budget',
                first_nonidentical=next((m for m in metrics if not m['exact']), None),
                first_outside_inherited_budget=next((m for m in metrics if not m['within_inherited_budget']), None),
                missing_reference=[k for k in candidate if k not in reference],
                missing_candidate=[k for k in reference if k not in candidate], metrics=metrics)


def replay(c, weights, backend, layer, inputs, selections=None, numerical_mode='f32-v2'):
    reference, candidate = {}, {}
    def capture(target):
        def callback(pos, name, value):
            key = f'p{pos}.{name}'
            if key in target:
                raise ValueError('duplicate stage: ' + key)
            target[key] = value
        return callback
    model = GlmMoeDsaModel(c, weights, backend, trace=capture(candidate))
    normalized_config = dict(c, indexer_types=model.indexer_types, mlp_layer_types=model.mlp_types)
    state = LayerCache()
    for pos, x in enumerate(inputs):
        model.forward_layer(layer, x.copy(), pos, state, None if selections is None else selections[pos])
    class Mapping:
        def __getitem__(self, name):
            return weights(name) if callable(weights) else weights[name]
    expanded_batch_oracle(normalized_config, Mapping(), [0] * len(inputs), capture(reference),
                          numerical_mode=numerical_mode, start_layer=layer, stop_layer=layer+1,
                          hidden_states=inputs, initial_selections=selections, output_logits=False)
    return reference, candidate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--input-trace', type=Path, required=True)
    p.add_argument('--selection-trace', type=Path, help='defaults to input trace; needed for shared layers')
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--layer', type=int, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--numerical-mode', choices=('f32-v2', 'legacy-v1'), default='f32-v2')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('output exists; preserve prior evidence')
    c = json.loads((args.gguf / 'config.json').read_text())
    contract = json.loads(args.contract.read_text())
    if sha(args.gguf / 'config.json') != contract['pins']['config_sha256']:
        p.error('config does not match contract')
    if not 0 <= args.layer < c['num_hidden_layers']:
        p.error('invalid layer')
    tokens = contract['cases'][0]['token_ids']
    if len(tokens) > 32:
        p.error('replay is bounded to 32 causal positions')
    from glm_vx.gguf_checkpoint import GGUFCheckpoint
    from kernels.backend import VxBackend
    args.output.mkdir(parents=True)
    started = time.monotonic()
    weights = GGUFCheckpoint(args.gguf, c, decode_threads=4, cache_bytes=2*1024**3)
    try:
        with np.load(args.input_trace, allow_pickle=False) as archive:
            if args.layer:
                inputs = np.stack([archive[f'p{i}.layer.{args.layer-1}.output'] for i in range(len(tokens))])
            else:
                inputs = np.stack([weights.tensor('model.embed_tokens.weight', rows=slice(t,t+1))[0] for t in tokens])
        model = GlmMoeDsaModel(c, weights, None)
        selections = None
        if model.indexer_types[args.layer] == 'shared':
            with np.load(args.selection_trace or args.input_trace, allow_pickle=False) as archive:
                selections = [archive[f'p{i}.layer.{args.layer}.selected'] for i in range(len(tokens))]
        ref, cand = replay(c, weights, VxBackend(args.library), args.layer, inputs, selections, args.numerical_mode)
    finally:
        weights.close()
    np.savez(args.output / 'reference.npz', **ref)
    np.savez(args.output / 'candidate.npz', **cand)
    report = compare_stages(ref, cand, contract['budgets']['layer_output'])
    report.update(layer=args.layer, numerical_mode=args.numerical_mode, token_ids=tokens,
                  elapsed_seconds=time.monotonic()-started,
                  pins=dict(contract_sha256=sha(args.contract), input_trace_sha256=sha(args.input_trace),
                            selection_trace_sha256=sha(args.selection_trace or args.input_trace),
                            library_sha256=sha(args.library), config_sha256=sha(args.gguf/'config.json'),
                            source_sha256={f:sha(f) for f in ('glm_vx/model.py', 'validation/expanded_reference.py', 'validation/stage_replay.py')},
                            reference_sha256=sha(args.output/'reference.npz'), candidate_sha256=sha(args.output/'candidate.npz')),
                  checkpoint_identity='config verified; checkpoint bytes use existing frozen manifest, not rehashed by replay')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'metrics'}, indent=2))


if __name__ == '__main__':
    main()
