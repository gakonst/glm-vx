# CPU prefill/decode handoff proof of concept

`glm_vx.handoff` transfers a request between distinct owners, and
`glm_vx.disaggregated` runs prefill and decode in **separate CPU processes**.
This is a correctness foundation for disaggregated serving. The existing HTTP
scheduler is unchanged: there is no deployed worker pool, network transport,
GPU cache migration or measured serving-throughput improvement yet.

```sh
# Actual separate interpreters, compiled Vx, tiny random weights:
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m glm_vx.disaggregated \
  --backend vx --tokens 1,7,2,11,4 --partitions 2,1,2 --new-tokens 5

# GGUF-backed workers (IDs are raw prescribed tokens, no chat template applied):
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m glm_vx.disaggregated \
  --backend vx --gguf /path/to/glm-5.3-iq1s \
  --tokens 9703 --new-tokens 1 --timeout 3600
```

Both GGUF workers load the model independently and stream-hash every shard once
at model binding, without dequantizing it. For a 216.7 GB checkpoint this is a
substantial startup cost. A persistent worker pool should amortize it; this CLI
starts fresh processes to demonstrate the boundary. Tests cover GGUF-backed
workers using actual synthetic GGUF files and split shards. **The complete
216.7 GB trained checkpoint was not run through this split-process CLI.** Its
separate numerical trace runs are reported in PARITY-PROGRESS.md.

## Transferred state and invariants

The bounded binary format contains a canonical JSON header, little-endian numeric
arrays and a checksum. It never uses pickle or object arrays. Shapes derive from
the receiving model and bounded position, not arbitrary serialized dimensions.
It carries consumed token IDs, next-token logits, MLA latents, RoPE keys,
full-indexer keys, per-layer selections and prompt/cache positions.

Identity binds config, resolved architecture, exact weights, model/backend code,
NumPy version, and the Vx library bytes; GGUF identity additionally binds decoder
code/version. Source and receiver recheck identity at handoff. GGUF bytes are
hashed once, with local file-stat/mapping guards afterward; files must remain
immutable while the model is loaded. Content identity is portable across paths.

The source relinquishes ownership after creating the payload. A receiver rejects
wrong request/source/destination IDs, duplicate requests and replayed transfer
IDs. The final prefill logits travel with the cache, so decode samples from them
without repeating the last prompt token. Greedy continuation appends each sampled
token once and retains next logits; its cache includes the last returned token.
Prompt chunks and decode batches roll back on model/observer failure.

The current scope is greedy continuation. EOS/stop policy and stochastic RNG
migration are not implemented in this demo. `greedy(n)` always consumes all n
sampled tokens, even an EOS, so it is a cache-continuation check rather than the
HTTP server's complete generation policy. Default payload limit is 64 MiB;
oversized contexts fail explicitly. Change reviewed limits rather than silently
truncating cache state.

The checksum detects corruption, **not malicious producers**. Use authenticated
transport. Claim state is bounded and process-local, not durable distributed
exactly-once delivery. A crashed receiver, retransmission, lease/fencing or
cross-host recovery needs a coordinator and durable claim ledger. The demo
coordinates sequential workers; it does not overlap prefill and decode for
multiple requests. No speed benefit is inferred from separating processes.

## Measurements and next serving step

The receipt reports distinct PIDs, handoff bytes, model loading, identity hashing,
prefill, serialization, pipe writing/reading, claim validation, decode and total
time. [Recorded CPU smoke](parity-evidence/disaggregated-smoke.json) uses tiny
random weights and must not be quoted as full-model or GPU performance.

The next service architecture is persistent prefill workers and decode workers,
explicit cache capacity/admission, a bounded transfer queue and a durable ownership
protocol. Benchmark TTFT and inter-token p95 under mixed prompt lengths against
the existing scheduler. Disaggregation is beneficial only if those improvements
exceed transfer, memory duplication and queueing costs on the target hardware.
