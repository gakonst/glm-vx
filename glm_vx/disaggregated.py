"""Two real CPU processes, a bounded byte handoff, and tiny/GGUF greedy decoding.

Run: python -m glm_vx.disaggregated --backend numpy --tokens 1,7,2,11 \
         --partitions 1,3 --new-tokens 4
Use --backend vx for compiled CPU Vx; --library overrides GLM_VX_LIBRARY.
Use --gguf PATH [--config FILE] for a real checkpoint (token IDs are supplied
explicitly; no tokenizer/chat-template behavior is implied). Each worker hashes
all shards once at startup, so large-checkpoint identity has substantial I/O cost.
The default random tiny model is correctness plumbing, not trained generation or
a distributed performance claim.
The coordinator starts fresh prefill/decode Python interpreters. Request data is
canonical JSON on stdin; the only cache boundary is the validated binary format.
No application object is pickled. Each worker independently opens its weights.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .backend import NumpyBackend
from .handoff import HandoffError, HandoffReceiver, Limits, OwnedRequest, model_identity
from .model import GlmMoeDsaModel, tiny_weights
from .tiny import tiny_config


def _model(backend, seed, gguf=None, config_path=None, decoded_cache_mib=512):
    if backend == 'vx':
        from kernels.backend import VxBackend
        backend = VxBackend()
    elif backend == 'numpy':
        backend = NumpyBackend()
    else:
        raise HandoffError('backend must be numpy or vx')
    if gguf is None:
        config = tiny_config()
        return GlmMoeDsaModel(config, tiny_weights(config, seed), backend)
    from .gguf_checkpoint import GGUFCheckpoint
    path = Path(gguf)
    directory = path if path.is_dir() else path.parent
    config = json.loads(Path(config_path or directory / 'config.json').read_text())
    checkpoint = GGUFCheckpoint(path, config, cache_bytes=decoded_cache_mib * 1024**2)
    return GlmMoeDsaModel(config, checkpoint, backend)


def _worker(args):
    if args.library:
        os.environ['GLM_VX_LIBRARY'] = args.library
    started = time.perf_counter()
    model = _model(args.backend, args.seed, args.gguf, args.config, args.decoded_cache_mib)
    loaded = time.perf_counter()
    try:
        identity = model_identity(model)  # GGUF byte scan occurs once here.
        identified = time.perf_counter()
        limits = Limits()
        timings = dict(load_seconds=loaded - started, identity_seconds=identified - loaded)
        if args.worker == 'prefill':
            raw = sys.stdin.buffer.read(limits.max_header_bytes + 1)
            if len(raw) > limits.max_header_bytes:
                raise HandoffError('prefill input too large')
            data = json.loads(raw)
            tokens, partitions = data['tokens'], data['partitions']
            if (not isinstance(tokens, list) or not isinstance(partitions, list)
                    or not partitions or any(type(n) is not int or n <= 0 for n in partitions)
                    or sum(partitions) != len(tokens)):
                raise HandoffError('partitions must cover the prompt exactly')
            began = time.perf_counter()
            request = OwnedRequest(model, args.request_id, 'prefill')
            offset = 0
            for count in partitions:
                request.append_prompt(tokens[offset:offset + count])
                offset += count
            request.seal_prompt()
            prefilled = time.perf_counter()
            payload = request.transfer('decode')
            serialized = time.perf_counter()
            sys.stdout.buffer.write(payload)
            sys.stdout.buffer.flush()
            timings.update(prefill_seconds=prefilled - began,
                           serialize_seconds=serialized - prefilled,
                           pipe_write_seconds=time.perf_counter() - serialized)
            sys.stderr.write(json.dumps(dict(pid=os.getpid(), timings=timings)) + '\n')
        else:
            began = time.perf_counter()
            payload = sys.stdin.buffer.read(limits.max_payload_bytes + 1)
            received = time.perf_counter()
            request = HandoffReceiver(model, 'decode').claim(
                payload, request_id=args.request_id, source='prefill')
            claimed = time.perf_counter()
            received_position = request.cache.position
            generated = request.greedy(args.new_tokens)
            timings.update(pipe_read_seconds=received - began,
                           claim_seconds=claimed - received,
                           decode_seconds=time.perf_counter() - claimed)
            result = dict(request_id=request.request_id, backend=args.backend,
                          checkpoint_kind='gguf' if args.gguf else 'tiny', identity=identity,
                          decode_pid=os.getpid(), received_position=received_position,
                          prompt_length=request.prompt_length, final_position=request.cache.position,
                          generated_tokens=generated, consumed_tokens=request.tokens,
                          next_logits=request.logits.tolist(), decode_timings=timings)
            sys.stdout.write(json.dumps(result, sort_keys=True, allow_nan=False))
    finally:
        close = getattr(model.weights, 'close', None)
        if callable(close):
            close()


def run_demo(tokens, *, partitions=None, new_tokens=4, backend='numpy', seed=17, timeout=60,
             gguf=None, config=None, library=None, decoded_cache_mib=512):
    """Run prefill and decode in distinct processes and return JSON-compatible proof.

    Worker failures propagate as CalledProcessError; there is no implicit retry.
    The coordinator relays a bounded cache; GGUF is opened independently. Timings
    from this demonstration must not be treated as GPU throughput measurements.
    """
    request_id = uuid.uuid4().hex
    command = [sys.executable, '-m', 'glm_vx.disaggregated', '--backend', backend,
               '--seed', str(seed), '--new-tokens', str(new_tokens), '--request-id', request_id,
               '--decoded-cache-mib', str(decoded_cache_mib)]
    for flag, value in [('--gguf', gguf), ('--config', config), ('--library', library)]:
        if value is not None:
            command.extend([flag, str(value)])
    data = json.dumps(dict(tokens=tokens, partitions=partitions if partitions is not None else [len(tokens)]),
                      allow_nan=False).encode('utf-8')
    started = time.perf_counter()
    # Popen exposes each worker PID, letting tests prove distinct interpreters.
    with subprocess.Popen(command + ['--worker', 'prefill'], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        prefill_pid = process.pid
        try:
            payload, error = process.communicate(data, timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, stderr=error)
    if len(payload) > Limits().max_payload_bytes:
        raise HandoffError('worker emitted an oversized handoff')
    decoded = subprocess.run(command + ['--worker', 'decode'], input=payload,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             check=True, timeout=timeout)
    result = json.loads(decoded.stdout)
    prefill_receipt = json.loads(error)
    result.update(prefill_pid=prefill_pid, coordinator_pid=os.getpid(),
                  handoff_bytes=len(payload), identity_bound=True,
                  prefill_timings=prefill_receipt['timings'],
                  total_seconds=time.perf_counter() - started)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('numpy', 'vx'), default='numpy')
    parser.add_argument('--tokens', default='1,7,2,11')
    parser.add_argument('--partitions', default=None)
    parser.add_argument('--new-tokens', type=int, default=4)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--gguf', help='GGUF file or checkpoint directory; omitted uses tiny random weights')
    parser.add_argument('--config', help='GGUF config JSON; defaults to adjacent config.json')
    parser.add_argument('--library', help='set GLM_VX_LIBRARY for both Vx CPU workers')
    parser.add_argument('--decoded-cache-mib', type=int, default=512)
    parser.add_argument('--timeout', type=float, default=60, help='per-worker timeout; raise for large GGUF startup')
    parser.add_argument('--worker', choices=('prefill', 'decode'), help=argparse.SUPPRESS)
    parser.add_argument('--request-id', default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.decoded_cache_mib < 0:
        parser.error('--decoded-cache-mib must be nonnegative')
    if args.worker:
        _worker(args)
    else:
        result = run_demo([int(t) for t in args.tokens.split(',')],
                          partitions=None if args.partitions is None else [int(n) for n in args.partitions.split(',')],
                          new_tokens=args.new_tokens, backend=args.backend, seed=args.seed,
                          gguf=args.gguf, config=args.config, library=args.library,
                          decoded_cache_mib=args.decoded_cache_mib, timeout=args.timeout)
        print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
