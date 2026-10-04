# Bounded prompt reuse and serving lifecycle

This change implements a CPU serving baseline from `SERVING-RESEARCH.md`.
Prefix reuse defaults off. Enable it explicitly with `--prefix-cache-mib 64`.
It reuses completed whole prompts which prefix a new request, choosing the
longest match. A shorter request than every stored prompt is a miss. There is
no radix tree, paged KV, true batched GEMM or distributed-cache claim.

## Semantics and ownership

The scheduler freezes input token IDs into a tuple. Before sampling the first
output, it snapshots the complete `RequestCache`: absolute position, every
layer's compressed MLA latents, RoPE keys, DSA index keys and selected indices,
and the logits following the final prompt token. Exact-prompt reuse samples
those saved logits without refeeding a token. An extension starts at the saved
absolute position and computes its remaining prompt. Generated tokens are never
published as prefix keys. Request RNG state remains private and starts from its
own seed, independent of cache hits or interleaving.

Snapshots use internal-only pickle byte blobs. They are created from live model
objects; there is no file, network or user-controlled deserialization entrypoint.
Every hit materializes independent lists and arrays. Decoder mutation, eviction
and another request cannot change a stored prefix or another decoder's cache.
This is deliberately a copy-based CPU baseline; long-prefix copying can erase
any computation savings and is measured separately.

A cache is scoped to the actual model instance, weights owner, backend instance,
complete configuration and `model.prefix_cache_revision` (default zero).
Identical model display names cannot share entries. Configuration, owner or
revision changes invalidate entries. Weights and backend numerical settings must
remain immutable during serving; callers that edit them in place must first
drain all requests and then increment `prefix_cache_revision` before subsequent
requests. In-flight model hot swapping is unsupported. A fresh scheduler is the
normal deployment path for a model revision. No cross-process/instance reuse is
exposed, so no checkpoint fingerprint is falsely treated as equivalent state.

## Capacity and lifecycle

The byte budget strictly bounds **retained snapshot storage**, including Python
key/tuple/blob sizes; repeated immutable references are conservatively charged
again. LRU eviction happens before publication. Oversized entries are rejected,
and a capped writer prevents building a serialized payload larger than its
single-entry allowance. Live request caches, fixed cache-manager metadata,
serialization interpreter workspace and deserialization into an admitted
request are separate allocations. This is not a process-RSS or GPU-memory
limit. The existing prompt-plus-generation token reservation remains a logical
admission bound, not a byte-accurate live-KV fit guarantee. `/health` exposes
budget, retained bytes, entries, hits, misses, evictions and rejected snapshots.
Closing a stopped scheduler clears its stored snapshots.

Prefill remains causal sequential token work. All ready decoders receive a turn
before one bounded total prefill slice, with round-robin prompt progress and
arrival checks at token boundaries. Cancellation now sweeps **all** queued
requests at the next model boundary instead of waiting for each prompt's turn.
An in-flight model call is indivisible. Cancellation after that call suppresses
new output and publication of a just-finished cancelled prompt.

`--max-pending-events` (default 64, minimum 2) bounds each unread output queue
independently of requested generation length. One slot is reserved for terminal
status. A full slow consumer terminates with `backpressure`, releases admission
and never blocks the model owner or other clients. JSON/SSE HTTP paths surface
that as an error, and socket disconnects cancel both streaming and ordinary
requests. HTTP admission failures include `Retry-After: 1`. Completion usage
includes `prompt_tokens_details.cached_tokens`; terminal responses also expose
prefill, decode and prefix lookup/snapshot seconds.

## Evidence, October 4, 2026

`tests/test_prefix_cache.py` covers exact first logits and every cache component,
independent mutation, greedy and seeded sampling, same/extended/missing prompts,
concurrent interleaving against an uncached independent model, frozen input,
output-prefix exclusion, EOS, cancellation, longest match, LRU eviction,
recomputation and instance/configuration/backend/revision invalidation. Both
NumPy and compiled Vx CPU backends exercise the interleaved output comparisons.
Scheduler and HTTP tests cover bounded slow consumers, cancellation sweeps,
JSON/SSE equivalence, real HTTP cache accounting and disconnect during cached
decode. These are untrained tiny-model and deterministic synthetic tests, not
trained full-model numerical validation.

Reproduce the focused suite and three-repeat HTTP benchmark from this worktree:

```sh
source ../vx-toolchain/env.sh
export PYTHONPATH="$PWD"
export GLM_VX_LIBRARY=/srv/nanocodex/workspace/glm-vx/kernels/build/libglm_vx.so
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/srv/nanocodex/workspace/glm-vx/.venv/bin/python -m pytest -q tests/test_prefix_cache.py tests/test_scheduler.py tests/test_scheduler_latency.py tests/test_server.py tests/test_serving_benchmark.py tests/test_prefill_serving.py
/srv/nanocodex/workspace/glm-vx/.venv/bin/python -m benchmarks.prefix_reuse --output benchmarks/evidence/prefix-http-vx-tiny-20261004.json
```

Measured on AMD EPYC 4484PX, prebuilt scalar Vx CPU library, random tiny weights
seed 19, four clients, sixteen requests per phase, twelve output tokens/request,
and four intentionally repeated 16/64-token prompt templates. All output hashes
agreed across disabled, cold and primed phases in all three repetitions. The
primed phase excludes four priming requests. Medians across the three runs:

| Metric | Disabled | Cold cache | Primed cache |
|---|---:|---:|---:|
| Client output tokens/s | 116.82 | 182.94 | 419.26 |
| Client TTFT p50 / p95, ms | 264.66 / 561.67 | 28.99 / 560.64 | 20.69 / 27.32 |
| Client inter-token p50 / p95, ms | 6.01 / 15.30 | 7.53 / 16.92 | 8.24 / 13.29 |
| Computed / cached prompt tokens | 640 / 0 | 288 / 352 | 0 / 640 |
| Total server prefill seconds | 1.25356 | 0.59917 | 0 |
| Total server decode seconds | 0.37909 | 0.41355 | 0.43959 |
| Total server prefix copy/store seconds | 0.00007 | 0.01134 | 0.00805 |

Warm output throughput ranged 397.98–451.92 tokens/s; disabled ranged
112.88–117.39. The warm median ratio is 3.59x for this highly repetitive tiny
workload, excluding priming; cold is 1.57x. Median inter-token latency **did not
improve**: cache hits change how many decoders overlap. Four snapshots retained
103,144 bytes under the explicit 1 MiB budget. Process RSS observations were
about 40 MiB; the receipt records before/after RSS and cumulative process HWM,
not isolated per-phase allocation peaks. Stage times include Python dispatch
and backend calls, not kernel-only timings. Source and library SHA-256 digests,
hardware, thread environment and per-request results are in the JSON receipt.

No trained-model run, GPU run or failed numerical gate was changed. True batched
prefill remains open. HTTP integration of `disaggregated.py` is intentionally
not implemented: its command launches fresh worker processes and hashes
checkpoint shards, rather than providing persistent cancellable worker pools.
Putting that command on the HTTP request path would introduce process startup,
model duplication and unbounded deployment cost. A proper integration needs
persistent ownership, bounded IPC, cancellation and failure recovery first.
