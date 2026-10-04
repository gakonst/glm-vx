# End-to-end serving performance

The target is client-observed time to first token (TTFT), inter-token latency,
request latency and aggregate output tokens/second under concurrent HTTP load.
Kernel timing alone does not establish serving performance.

Serving changes in this revision:

- Non-final prompt tokens update all transformer layers and causal caches but
  omit the unused final normalization and vocabulary projection. The final
  prompt token and every decode token still compute sampling logits.
- Ready decode requests run before a bounded amount of prefill work. With
  decoders active, at most `--decode-prefill-tokens` (default 4) prompt tokens run
  per scheduling round, across all
  prefilling requests. Prefill requests rotate so they continue making progress.
  An in-flight token remains indivisible; this is not GPU continuous batching.
- Dense, shared and individual-expert GPU MLPs keep gate, up and SwiGLU results
  on-device until the down projection finishes. Immutable weights use the
  existing bounded cache. Packed expert tensors retain the primitive path.
- HTTP SSE connections use TCP_NODELAY for small token writes.

The GPU path still has host orchestration, host KV/index selection, activation
transfers between other operators, per-call temporary allocations and scalar
matvec kernels. A resident model graph, tensor-core batched projections,
continuous GPU batching, GPU KV management and model/expert sharding remain
necessary for production throughput. No GPU serving speed is inferred from CPU
SIMT tests or offline assembly.

## Running the HTTP benchmark

Start the server and use a separate process:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m glm_vx.server --tiny --backend vx
.venv/bin/python benchmarks/serve.py --url http://127.0.0.1:8000 \
  --requests 16 --concurrency 4 --prompt-lengths 8,128 --tokens 32 \
  --output benchmark.json
```

For GPU measurements, start the server with `--backend vx-gpu` after building
PTX. Use the actual checkpoint/tokenizer and vocabulary size for trained-model
results; `--tiny` is an untrained mechanical workload. The benchmark uses token
IDs and deterministic greedy decoding, excludes explicit warmup, reports errors
without retrying them, checks terminal token IDs against incremental tokens, and
stores output hashes so matched runs can verify generated-token parity.

The closed-loop client keeps at most `--concurrency` HTTP requests in flight;
request latency starts when a worker sends its request. It excludes client-side
waiting for a worker, while total throughput includes the whole measured run.
Admission rejection is a failure, not a fast successful response. Inter-token
latency measures client receipt, including any transport buffering. Per-prompt
metrics expose long-prompt starvation that an aggregate median can conceal.

## Measured comparison

Baseline `1fe1f1f`, versus this revision with a four-token decode-time prefill
budget. Three runs each, 16 requests/run, concurrency 4, alternating 8/128-token
prompts, 32 output tokens, one warmup request, localhost HTTP, compiled Vx CPU
backend, tiny random model, OpenBLAS threads=1. The same machine ran the servers
and client; other machine activity was not controlled. These are medians of
per-run statistics, not pooled percentiles or confidence intervals.

| Client metric | Previous revision | This revision |
| --- | ---: | ---: |
| Output tokens/s | 164.7 | 166.7 |
| p95 inter-token gap | 42.5 ms | 16.4 ms |
| p50 request latency | 784.0 ms | 525.6 ms |
| p95 request latency | 940.2 ms | 1252.7 ms |
| p95 time to first token | 478.1 ms | 852.3 ms |

All 96 measured requests succeeded and every corresponding output hash matched.
Decode becomes smoother and median completion improves, but long prompts wait
longer for their first token and tail completion worsens. The small throughput
difference is within run-to-run variation; **no throughput speedup is established**.
Increase the prefill budget to favor prompt progress, or lower it to favor
ongoing streaming. The exploratory one-token budget run is retained and clearly
separate from the default-four-token comparison. No GPU or trained-model serving
performance was measured.

[Raw comparison and provenance](evidence/comparison.json), per-request traces and
[test output](evidence/tests.txt) are committed. Final validation: **170 passed,
14 explicitly skipped real-GPU tests, 37 subtests passed**. Tests cover causal
cache/output parity after skipped heads, deterministic scheduler work budgets,
admission/error/cancellation/shutdown, HTTP SSE metrics/error handling, and
source-identical CPU SIMT MLP numerics/cleanup. With immutable weights cached,
the resident MLP performs one activation upload and one download versus five
uploads and four downloads for separate primitive calls. Those counts are tested;
they are not GPU timing measurements.
