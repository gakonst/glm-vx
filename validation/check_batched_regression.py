"""Observe full trained batched prefill against a saved scalar Vx trace.

This is regression evidence for identical GGUF/F32 arithmetic, NOT independent
model parity or release eligibility. Hooks only observe unmodified helper calls.
The reference archive/receipt and actual loaded checkpoint shards are bound.
"""
import argparse
import importlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from glm_vx.gguf_checkpoint import GGUFCheckpoint
from glm_vx.model import Model
from kernels.backend import VxBackend
from validation.export_vx_trace import execution_pins
from validation.trace_identity import bind_model, bind_loaded, sha_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('gguf', 'reference', 'contract', 'library', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('output exists; preserve old evidence')
    receipt_path, arrays_path = args.reference/'receipt.json', args.reference/'arrays.npz'
    reference = json.loads(receipt_path.read_text())
    identity = bind_model(args.gguf, args.contract)
    if (reference['status'] != 'complete' or not reference['packed_weights'] or
            reference['numerical_mode'] != 'f32-official-expert-order-v2' or
            reference['model_identity'] != identity or
            reference['arrays_sha256'] != sha_file(arrays_path)):
        raise ValueError('reference mode/checkpoint/archive binding mismatch')
    tokens = reference['token_ids']
    if not 1 <= len(tokens) <= 64: raise ValueError('one bounded chunk required')
    module = importlib.import_module('glm_vx.prefill')
    def pins():
        return dict(base=execution_pins(), prefill_sha256=sha_file(module.__file__),
                    observer_sha256=sha_file(__file__), library_sha256=sha_file(args.library))
    before = pins(); args.output.mkdir(parents=True)
    report = dict(status='running', scope=__doc__, eligible=False, tokens=tokens,
                  reference_receipt_sha256=sha_file(receipt_path),
                  reference_arrays_sha256=reference['arrays_sha256'], model_identity=identity,
                  pins=before, checks=[], failures=[])
    start = time.monotonic()
    with np.load(arrays_path, allow_pickle=False) as oracle:
        def check(key, actual):
            expected = oracle[key]
            actual = np.asarray(actual)
            if expected.dtype.kind in 'iu' and actual.dtype.kind in 'iu':
                # Trace exporter canonicalizes routing IDs to int64 explicitly.
                actual = actual.astype(expected.dtype)
            exact = (actual.shape == expected.shape and actual.dtype == expected.dtype and
                     actual.tobytes() == expected.tobytes())
            report['checks'].append(dict(key=key, shape=list(actual.shape), exact=exact))
            if not exact:
                report['failures'].append(key)
                raise AssertionError('bitwise regression: '+key)
        config = json.loads((args.gguf/'config.json').read_text())
        weights = GGUFCheckpoint(args.gguf, config, decode_threads=1)
        original = {name: getattr(module, name) for name in ('_norm', '_attention', '_feed_forward')}
        layer_now, row_now = None, 0
        class ObservedBackend(VxBackend):
            def route(self, logits, bias, k, scale):
                nonlocal row_now
                ids, values = super().route(logits, bias, k, scale)
                prefix = f'p{row_now}.layer.{layer_now}.mlp'
                for suffix, value in (('router_logits', logits), ('route_ids', ids), ('route_weights', values)):
                    check(prefix+'.'+suffix, value)
                row_now += 1
                return ids, values
        def norm(model, prefix, x, eps):
            result = original['_norm'](model, prefix, x, eps)
            if prefix.endswith('.input_layernorm'):
                layer = int(prefix.split('.')[2])
                for pos in range(len(tokens)):
                    check(f'p{pos}.layer.{layer}.input', x[pos])
                    check(f'p{pos}.layer.{layer}.input_norm', result[pos])
            elif prefix.endswith('.post_attention_layernorm'):
                layer = int(prefix.split('.')[2])
                for pos in range(len(tokens)):
                    check(f'p{pos}.layer.{layer}.attention_residual', x[pos])
                    check(f'p{pos}.layer.{layer}.post_attention_norm', result[pos])
            elif prefix == 'model.norm':
                for pos in range(len(tokens)): check(f'p{pos}.layer.{model.n_layers-1}.output', x[pos])
            return result
        def attention(model, layer, x, offset, state, previous):
            result, selected = original['_attention'](model, layer, x, offset, state, previous)
            for pos in range(len(tokens)):
                prefix = f'p{pos}.layer.{layer}'
                for suffix, value in (('attention.output', result[pos]), ('latent', state.latents[pos]),
                                      ('rope_key', state.rope_keys[pos]), ('selected', selected[pos])):
                    check(prefix+'.'+suffix, value)
                if model.indexer_types[layer] == 'full': check(prefix+'.indexer.key', state.index_keys[pos])
            return result, selected
        def feed_forward(model, layer, x):
            nonlocal layer_now, row_now
            layer_now, row_now = layer, 0
            result = original['_feed_forward'](model, layer, x)
            for pos in range(len(tokens)): check(f'p{pos}.layer.{layer}.mlp.output', result[pos])
            print(json.dumps(dict(layer=layer, seconds=time.monotonic()-start, exact_checks=len(report['checks']))), flush=True)
            return result
        try:
            loaded = bind_loaded(identity, args.gguf, weights.store.paths)
            module._norm, module._attention, module._feed_forward = norm, attention, feed_forward
            model = Model(config, weights, ObservedBackend(args.library), packed_weights=True)
            cache = model.new_cache()
            logits = module.prefill(model, tokens, cache, output_all_logits=True)
            for pos in range(len(tokens)): check(f'p{pos}.logits', logits[pos])
            if cache.position != len(tokens): raise AssertionError('cache position mismatch')
            if before != pins() or bind_model(args.gguf, args.contract) != identity or bind_loaded(identity, args.gguf, weights.store.paths) != loaded:
                raise RuntimeError('execution/checkpoint changed during run')
            np.savez(args.output/'logits.npz', logits=logits)
            report.update(status='passed', arrays_sha256=sha_file(args.output/'logits.npz'), loaded_checkpoint_paths=loaded)
        except BaseException as exc:
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            for name, fn in original.items(): setattr(module, name, fn)
            weights.close()
            report.update(elapsed_seconds=time.monotonic()-start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          timing_scope='diagnostic with synchronous observation, no throughput comparison')
            (args.output/'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('checks', 'pins', 'model_identity')}), flush=True)


if __name__ == '__main__': main()
