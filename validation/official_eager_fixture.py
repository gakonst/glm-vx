"""Execute hash-pinned, unmodified Transformers GLM eager CPU on tiny F32 weights.

Optional independent validation dependency; never imported by serving. No model
is downloaded. Strict source hashes come from metadata/sources.json. Numerical
thresholds are copied from the provided frozen contract, never chosen here.
"""
import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from glm_vx.model import GlmMoeDsaModel
from tests.test_model import tiny_fixture
from validation.expanded_reference import expanded_batch_oracle
from validation.stage_replay import compare_stages, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve evidence')
    import torch
    import transformers
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as official
    from transformers.models.glm_moe_dsa import configuration_glm_moe_dsa as configuration
    from kernels.backend import VxBackend
    sources = json.loads(Path('metadata/sources.json').read_text())
    for module in (official, configuration):
        path = Path(inspect.getfile(module))
        if sha(path) != sources['sources'][path.name]['sha256']:
            raise ValueError('official installed source hash differs: ' + path.name)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    contract = json.loads(args.contract.read_text())
    reports = []
    arrays = {}
    # A longer-than-topk fixture covers real sparse selection and shared layers.
    for seed, tokens in ((123, [1, 7, 2, 11]), (456, [3, 4, 5, 6, 7, 8])):
        c, w = tiny_fixture(seed)
        cfg = configuration.GlmMoeDsaConfig(**c, attention_dropout=0.0)
        cfg._attn_implementation = 'eager'
        model = official.GlmMoeDsaForCausalLM(cfg).float().eval()
        state = {}
        for name in model.state_dict():
            if name.endswith('.experts.gate_up_proj'):
                prefix = name.removesuffix('.experts.gate_up_proj')
                value = np.stack([np.concatenate([w[f'{prefix}.experts.{e}.{p}_proj.weight'] for p in ('gate', 'up')]) for e in range(c['n_routed_experts'])])
            elif name.endswith('.experts.down_proj'):
                prefix = name.removesuffix('.experts.down_proj')
                value = np.stack([w[f'{prefix}.experts.{e}.down_proj.weight'] for e in range(c['n_routed_experts'])])
            else:
                value = w[name]
            state[name] = torch.from_numpy(value.copy())
        model.load_state_dict(state, strict=True)
        reference, candidate, expanded = {}, {}, {}
        selected = {}
        offset = [0]
        def capture(target):
            return lambda p,n,v: target.__setitem__(f'p{p}.{n}', v.copy())
        handles = []
        for layer, module in enumerate(model.model.layers):
            def layer_hook(module, inputs, output, layer=layer):
                hidden, ids = output
                for t in range(hidden.shape[1]):
                    pos = offset[0] + t
                    reference[f'p{pos}.layer.{layer}.output'] = hidden[0,t].detach().numpy().copy()
                    # Eager batch can include masked future slots before topk is
                    # reached. Membership of causal slots is the actual mask.
                    indices = ids[0,t].detach().numpy()
                    selected[f'p{pos}.layer.{layer}.selected'] = np.sort(indices[indices <= pos]).astype(np.int64)
            handles.append(module.register_forward_hook(layer_hook))
        with torch.no_grad():
            result = model(torch.tensor([tokens]), use_cache=False)
        for pos in range(len(tokens)):
            reference[f'p{pos}.logits'] = result.logits[0,pos].numpy().copy()
        official_batch = {k:v.copy() for k,v in reference.items()}
        official_selected = {k:v.copy() for k,v in selected.items()}
        with torch.no_grad():
            cache = None
            for pos, token in enumerate(tokens):
                offset[0] = pos
                result = model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
                cache = result.past_key_values
                reference[f'p{pos}.logits'] = result.logits[0,0].numpy().copy()
        for handle in handles:
            handle.remove()
        vx = GlmMoeDsaModel(c, w, VxBackend(args.library), trace=capture(candidate))
        vx.prefill(tokens)
        expanded_batch_oracle(c, w, tokens, trace=capture(expanded))
        for execution, oracle in (('official_batch', official_batch), ('official_incremental', reference)):
            for implementation, actual in (('vx_compressed', candidate), ('expanded_f32', expanded)):
                # Full layer outputs/logits only: omitted internals are clearly
                # scoped here, while the other stage tooling compares internals.
                common = {k:actual[k] for k in oracle}
                comparison = compare_stages(oracle, common, contract['budgets']['layer_output'], logits_budget=contract['budgets']['logits'])
                selections = official_selected if execution == 'official_batch' else selected
                discrete = [{ 'key':k, 'official':v.tolist(), 'candidate':np.sort(actual[k]).tolist() }
                            for k,v in selections.items() if not np.array_equal(v, np.sort(actual[k]))]
                comparison.update(seed=seed, token_ids=tokens, execution=execution, implementation=implementation,
                                  causal_selection_mismatches=discrete)
                reports.append(comparison)
        for role, trace in (('official_batch',official_batch), ('official_incremental',reference), ('vx_compressed',candidate), ('expanded_f32',expanded)):
            arrays.update({f'seed{seed}.{role}.{k}':v for k,v in trace.items()})
    args.output.mkdir(parents=True)
    np.savez(args.output/'arrays.npz', **arrays)
    failed = any(r['first_outside_inherited_budget'] is not None or r['causal_selection_mismatches'] for r in reports)
    report = dict(status='fail' if failed else 'pass',
                  scope='actual pinned official eager CPU, two random tiny models, batch and cached decode; not trained-model promotion',
                  transformers_revision=sources['transformers_revision'], torch_version=torch.__version__,
                  transformers_version=transformers.__version__, numpy_version=np.__version__,
                  contract_sha256=sha(args.contract), library_sha256=sha(args.library),
                  source_sha256=sha(__file__), dependencies_sha256={f:sha(f) for f in ('glm_vx/model.py', 'validation/expanded_reference.py', 'validation/stage_replay.py', 'tests/test_model.py')}, arrays_sha256=sha(args.output/'arrays.npz'),
                  comparisons=reports)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='comparisons'},indent=2))
    for r in reports:
        print(json.dumps({k:v for k,v in r.items() if k!='metrics'}))
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
