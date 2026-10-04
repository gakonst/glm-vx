"""Client-observed HTTP SSE benchmark. No model/kernel timing substitution."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit


def completion(url, prompt, tokens, timeout=60):
    started = time.perf_counter(); times = []; ids = []; finished = False
    conn = None
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname:
            raise ValueError('URL must be HTTP(S)')
        cls = http.client.HTTPSConnection if parsed.scheme == 'https' else http.client.HTTPConnection
        conn = cls(parsed.hostname, parsed.port, timeout=timeout)
        body = json.dumps({'prompt':prompt, 'max_tokens':tokens, 'temperature':0, 'stream':True})
        conn.request('POST', parsed.path.rstrip('/') + '/v1/completions', body,
                     {'Content-Type':'application/json'})
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError('HTTP ' + str(response.status))
        if 'text/event-stream' not in response.getheader('Content-Type', ''):
            raise RuntimeError('expected SSE response')
        while True:
            remaining = timeout - (time.perf_counter() - started)
            if remaining <= 0: raise TimeoutError('request deadline exceeded')
            # HTTPConnection may relinquish its socket for Connection: close.
            sock = conn.sock or getattr(getattr(response.fp, 'raw', None), '_sock', None)
            if sock is not None: sock.settimeout(remaining)
            line = response.readline(1048577)
            if len(line) > 1048576: raise ValueError('oversize SSE line')
            if not line: raise RuntimeError('stream ended without [DONE]')
            if not line.startswith(b'data:'): continue
            data = line[5:].strip()
            if data == b'[DONE]':
                if not finished: raise RuntimeError('missing completion finish event')
                break
            event = json.loads(data)
            if 'error' in event: raise RuntimeError(str(event['error']))
            choice = event['choices'][0]
            if choice.get('finish_reason') is not None:
                if finished: raise RuntimeError('duplicate finish event')
                if choice['finish_reason'] not in ('length','stop'):
                    raise RuntimeError('unexpected finish reason')
                if choice.get('token_ids') != ids: raise RuntimeError('terminal token IDs disagree')
                finished = True
                continue
            if finished: raise RuntimeError('tokens after finish')
            emitted = choice.get('token_ids')
            if not isinstance(emitted, list) or not emitted or any(type(t) is not int for t in emitted):
                raise RuntimeError('expected streamed token IDs')
            if len(ids) + len(emitted) > tokens: raise RuntimeError('too many output tokens')
            tick = time.perf_counter()
            ids.extend(emitted); times.extend([tick] * len(emitted))
        if not ids: raise RuntimeError('no output tokens')
        return {'ok':True, 'prompt_tokens':len(prompt), 'output_tokens':len(ids),
                'ttft_ms':(times[0]-started)*1000,
                'inter_token_ms':[(b-a)*1000 for a,b in zip(times,times[1:])],
                'latency_ms':(time.perf_counter()-started)*1000,
                'output_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
    except Exception as exc:
        return {'ok':False, 'prompt_tokens':len(prompt), 'output_tokens':len(ids),
                'latency_ms':(time.perf_counter()-started)*1000,
                'error':type(exc).__name__ + ': ' + str(exc)}
    finally:
        if conn is not None: conn.close()


def percentiles(values):
    ordered = sorted(values)
    if not ordered: return {'p50':None, 'p95':None, 'max':None}
    def at(q):
        index = (len(ordered)-1)*q; lo = int(index); hi = math.ceil(index)
        return ordered[lo] + (ordered[hi]-ordered[lo])*(index-lo)
    return {'p50':at(.5), 'p95':at(.95), 'max':ordered[-1]}


def summarize(rows, elapsed):
    successful = [row for row in rows if row['ok']]
    return {'successful_requests':len(successful), 'failed_requests':len(rows)-len(successful),
            'elapsed_seconds':elapsed,
            'successful_output_tokens':sum(row['output_tokens'] for row in successful),
            'output_tokens_per_second':sum(row['output_tokens'] for row in successful)/elapsed,
            'ttft_ms':percentiles([row['ttft_ms'] for row in successful]),
            'inter_token_ms':percentiles([gap for row in successful for gap in row['inter_token_ms']]),
            'request_latency_ms':percentiles([row['latency_ms'] for row in successful])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8000')
    p.add_argument('--requests', type=int, default=16)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--prompt-lengths', default='8,128')
    p.add_argument('--tokens', type=int, default=32)
    p.add_argument('--vocab-size', type=int, default=256)
    p.add_argument('--timeout', type=float, default=60)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    try: lengths = [int(n) for n in args.prompt_lengths.split(',')]
    except ValueError: p.error('prompt lengths must be comma-separated integers')
    if min(args.requests,args.concurrency,args.tokens,args.vocab_size,*lengths) < 1 or args.warmup < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        p.error('positive workload sizes/timeout and nonnegative warmup required')
    def run(index):
        length = lengths[index % len(lengths)]
        prompt = [(index*17 + j*3 + 1) % args.vocab_size for j in range(length)]
        return completion(args.url, prompt, args.tokens, args.timeout)
    warmup = [run(i) for i in range(args.warmup)]
    if any(not row['ok'] for row in warmup):
        result = {'schema':'glm-vx-http-benchmark-v1', 'status':'warmup_failed', 'warmup':warmup}
    else:
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            rows = list(pool.map(run, range(args.requests)))
        elapsed = time.perf_counter()-start
        result = {'schema':'glm-vx-http-benchmark-v1', 'status':'ok' if all(r['ok'] for r in rows) else 'failed',
                  'workload':vars(args), 'prompt_lengths':lengths,
                  'metrics':summarize(rows,elapsed), 'requests':rows,
                  'by_prompt_length':{str(n):summarize([r for r in rows if r['prompt_tokens']==n],elapsed) for n in sorted(set(lengths))},
                  'measurement':'Client wall-clock HTTP streaming. Warmup excluded. Closed-loop bounded concurrency. No retries. Throughput counts successful requests only; inspect failures. Group throughput uses the whole run duration.'}
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('requests','workload')},allow_nan=False))
    return 0 if result['status']=='ok' else 2


if __name__ == '__main__': raise SystemExit(main())
