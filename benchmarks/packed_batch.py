"""Trained GGUF row microbenchmarks, with ordered correctness gates first.

Compares fused batch, repeated packed dots, resident expanded batch, and decode
plus expanded batch on exactly the same rows/inputs. Includes activation transpose
and Python dispatch; excludes disk-cold scans and HTTP. No whole-model speed claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time
import numpy as np
from glm_vx.gguf_reader import GGUFStore
from kernels.backend import VxBackend
from validation.ggml_oracle import NativeGGML, DEFAULT_REFERENCE, DEFAULT_CHECKPOINT, fingerprint


def interleaved(methods, repeats, seed):
    for fn in methods.values(): fn()
    samples = {key: [] for key in methods}
    rng = random.Random(seed)
    for _ in range(repeats):
        order = list(methods); rng.shuffle(order)
        for key in order:
            start = time.perf_counter_ns(); methods[key]()
            samples[key].append((time.perf_counter_ns()-start)/1e9)
    return {key: {'median_seconds': statistics.median(values), 'samples_seconds': values}
            for key, values in samples.items()}


def run(args):
    if not 1 <= args.rows <= 512 or not 3 <= args.repeats <= 50:
        raise ValueError('rows must be 1..512 and repeats 3..50')
    backend, native = VxBackend(args.library), NativeGGML(args.reference)
    candidates_by_name = {}
    for item in args.candidate_library:
        name, path = item.split('=', 1)
        if name in candidates_by_name or not name.isidentifier():
            raise ValueError('unique identifier required for each candidate')
        candidates_by_name[name] = VxBackend(Path(path))
    store = GGUFStore(args.checkpoint)
    receipt = {'schema': 'glm-vx.packed-batch.v1',
        'scope': 'bounded trained-row warm-page CPU microbenchmarks; not model throughput',
        'numerics': 'native GGML weight decode, unquantized F32 inputs, ordered F32 sum',
        'library': fingerprint(backend.library_path), 'oracle': native.provenance(),
        'rows_limit': args.rows, 'repeats': args.repeats, 'samples': [],
        'candidate_libraries': {name: fingerprint(b.library_path) for name, b in candidates_by_name.items()},
        'promotion': False, 'whole_model_parity': False}
    try:
        paths = {t.name: p for p, r in zip(store.paths, store.readers) for t in r.tensors}
        for kind in sorted({int(t.tensor_type) for t in store.tensors.values()}):
            candidates = [t for t in store.tensors.values() if int(t.tensor_type) == kind and len(store.shape(t.name)) >= 2]
            if not candidates: continue
            tensor = candidates[len(candidates)//2]; shape = store.shape(tensor.name)
            expert = shape[0]//2 if len(shape) == 3 else None
            rows = min(args.rows, shape[-2]); start = (shape[-2]-rows)//2
            selection = {'expert': expert, 'rows': slice(start, start+rows)}
            block, size = native.layout(tensor.tensor_type.name, kind)
            rowbytes = shape[-1]//block*size
            offset = int(tensor.data_offset)+((expert or 0)*shape[-2]+start)*rowbytes
            with paths[tensor.name].open('rb') as stream:
                raw = os.pread(stream.fileno(), rows*rowbytes, offset)
            oracle_weights = np.stack([native.decode(raw[i*rowbytes:(i+1)*rowbytes], tensor.tensor_type.name, kind, shape[-1]) for i in range(rows)])
            decoded = store.read(tensor.name, **selection)
            np.testing.assert_array_equal(decoded, oracle_weights)
            with store.packed_rows(tensor.name, **selection) as (view, dims, fmt):
                assert view.tobytes() == raw and not view.flags.owndata
                for batch in (1, 2, 4, 8, 16, 17, 32, 64):
                    x = np.random.default_rng(2030+kind+batch).normal(size=(batch, shape[-1])).astype(np.float32)
                    expected = np.stack([np.cumsum(oracle_weights*row, axis=1, dtype=np.float32)[:, -1] for row in x])
                    methods = {
                        'packed_batch': lambda: backend.packed_linear_batch(view, fmt, dims, x),
                        'repeated_packed': lambda: np.stack([backend.packed_matvec(view, fmt, dims, row) for row in x]),
                        'resident_expanded_batch': lambda: backend.linear_batch(decoded, x),
                        'decode_plus_expanded_batch': lambda: backend.linear_batch(store.read(tensor.name, **selection), x),
                    }
                    for name, candidate in candidates_by_name.items():
                        methods['candidate_'+name] = lambda candidate=candidate: candidate.packed_linear_batch(view, fmt, dims, x)
                    for name, fn in methods.items():
                        actual = fn()
                        if actual.tobytes() != expected.tobytes(): raise AssertionError(f'{tensor.name}/{batch}/{name}: ordered F32 mismatch')
                    timings = interleaved(methods, args.repeats, 99+kind+batch)
                    denom = timings['packed_batch']['median_seconds']
                    entry = {'format': tensor.tensor_type.name, 'tensor': tensor.name, 'expert': expert,
                        'start_row': start, 'shape': list(dims), 'batch': batch, 'byte_offset': offset,
                        'raw_sha256': hashlib.sha256(raw).hexdigest(), 'input_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
                        'packed_bytes': len(raw), 'expanded_bytes': oracle_weights.nbytes, 'exact': True,
                        'timings': timings,
                        'speedup_vs_repeated_packed': timings['repeated_packed']['median_seconds']/denom,
                        'speedup_vs_decode_plus_expanded': timings['decode_plus_expanded_batch']['median_seconds']/denom}
                    receipt['samples'].append(entry)
                    print(tensor.tensor_type.name, batch, 'packed batch ms', round(denom*1e3, 3),
                          'vs repeated', round(entry['speedup_vs_repeated_packed'], 3),
                          'vs decode+batch', round(entry['speedup_vs_decode_plus_expanded'], 3), flush=True)
        receipt['status'] = 'passed'
        return receipt
    finally: store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--library', type=Path)
    parser.add_argument('--candidate-library', action='append', default=[], help='NAME=PATH; measured interleaved on identical inputs')
    parser.add_argument('--rows', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--output', type=Path, default=Path('build/packed-batch-benchmark.json'))
    args = parser.parse_args(); result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(args.output)


if __name__ == '__main__': main()
