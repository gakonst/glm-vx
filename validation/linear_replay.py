"""Replay one captured linear operation with identical inputs and decoded weights.

F64 here is a diagnostic dot-product baseline, explicitly a different arithmetic
mode. It is never substituted for frozen reference evidence or production output.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from validation.stage_replay import compare_stages, sha


def linear_variants(weight, inputs, backend):
    return {'vx_f32': np.stack([backend.matvec(weight, x) for x in inputs]),
            'numpy_batch_f32': inputs @ weight.T,
            'numpy_row_f32': np.stack([weight @ x for x in inputs]),
            'f64_diagnostic_rounded_f32': (inputs.astype(np.float64) @ weight.astype(np.float64).T).astype(np.float32)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--keys', nargs='+', required=True)
    p.add_argument('--weight', required=True)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('output exists')
    from glm_vx.gguf_checkpoint import GGUFCheckpoint
    from kernels.backend import VxBackend
    c = json.loads((args.gguf/'config.json').read_text())
    contract = json.loads(args.contract.read_text())
    if sha(args.gguf/'config.json') != contract['pins']['config_sha256']:
        p.error('config does not match contract')
    with np.load(args.trace, allow_pickle=False) as trace:
        inputs = np.stack([trace[key] for key in args.keys])
    checkpoint = GGUFCheckpoint(args.gguf, c, decode_threads=4)
    try:
        weight = checkpoint(args.weight)
        variants = linear_variants(weight, inputs, VxBackend(args.library))
    finally:
        checkpoint.close()
    comparisons = {}
    for name, value in variants.items():
        comparisons[name] = compare_stages({'output':variants['numpy_batch_f32']}, {'output':value}, contract['budgets']['layer_output'])
    args.output.mkdir(parents=True)
    np.savez(args.output/'arrays.npz', inputs=inputs, **variants)
    report = dict(scope='identical-input projection diagnostic; no gate promotion',
                  weight=args.weight, keys=args.keys, shape=list(weight.shape),
                  input_trace_sha256=sha(args.trace), contract_sha256=sha(args.contract),
                  decoded_weight_sha256=hashlib.sha256(weight.tobytes()).hexdigest(),
                  library_sha256=sha(args.library), source_sha256=sha(__file__),
                  arrays_sha256=sha(args.output/'arrays.npz'), comparisons=comparisons)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
