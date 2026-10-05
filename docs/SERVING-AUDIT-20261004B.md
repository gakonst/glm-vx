# Serving contract and implementation audit, 2026-10-04

Base: `3ae29e5`. This supplements [SERVING-RESEARCH.md](SERVING-RESEARCH.md)
and accounts for [OPTIMIZATION-PROGRESS.md](OPTIMIZATION-PROGRESS.md).
Existing heap top-k, bounded prompt snapshots, admission/backpressure and batched
prefill are **not** presented as new work. Full GLM-5.3 is the target; Flash
architecture results are excluded. Upstream PR state and diffs were read from
the GitHub API. External performance numbers below are author reports, not
measurements reproduced here.

## Reproduced and fixed: execution identity omitted packed mode

**P0 numerical contract.** `GlmMoeDsaModel._linear` chooses packed GGUF arithmetic
using `model.packed_weights`. Before this patch, `handoff.model_identity` hashed
the same identity for models with different values of that flag, even with the
same legitimate GGUF checkpoint and Vx backend. A receiver could therefore
accept a cache produced through another execution path. Checkpoint equality is
insufficient to establish arithmetic equality.

The handoff config digest now includes `packed_weights`. A new GGUF/Vx test
constructs both modes, rejects the cross-mode claim, accepts a matching-mode
claim and continues decoding, and rejects a mode change after request creation
without relinquishing ownership. The fixture uses F32 GGUF tensors; it proves
the identity/ownership contract, not a trained quantized-output divergence.

**P1 snapshot identity.** `PrefixCache._namespace` also omitted packed mode and
resolved model fields. Changing `eps`, `theta`, `index_topk`, `indexer_types` or
`mlp_types` left a stale exact-prompt hit valid. These fields live outside the
config dictionary after construction. The namespace now snapshots resolved
architecture/numerical fields as canonical JSON, including packed mode. This
also detects later list-content changes rather than retaining mutable list
references in the namespace. Active requests still require immutable model
state; this patch does not authorize mutation during inference. In-place weight
edits continue to require the documented `prefix_cache_revision` increment.

Validation:

- Before the fix: all seven new cases failed (one handoff, six snapshot modes).
- After the fix: `../glm-vx/.venv/bin/python -m pytest -q tests/test_handoff.py
  tests/test_prefix_cache.py tests/test_scheduler.py tests/test_scheduler_latency.py
  tests/test_server.py` — **106 passed in 5.30 s**.
- After adding matching packed-mode continuation to the regression:
  the targeted handoff test passed again.
- No gates or tolerances changed. No full trained model or GPU execution ran.

The wire structure stays version 1; the identity digest changes, deliberately
rejecting payloads from workers using the older incomplete identity contract.
Private prefix snapshots are process-local and require no migration.

## New implementation-specific opportunities

### P1: cache geometry must have one physical-layout contract

vLLM [PR 48379](https://github.com/vllm-project/vllm/pull/48379), merged as
[`6472131298e7b9ce81651cac90cc2dcc28963b55`](https://github.com/vllm-project/vllm/commit/6472131298e7b9ce81651cac90cc2dcc28963b55),
sets `kv_quant_mode` on generic MLA cache specs. Its absence made allocation use
656 bytes/token for `fp8_ds_mla`, but reshape use 576 elements/token. The author
reports reproducing the GLM-5.2 failure and successful H200 validation after the
fix. This is a physical representation bug, not an attention tolerance issue.

vLLM [PR 50823](https://github.com/vllm-project/vllm/pull/50823), merged as
[`f0de1a604cad003379e5bb4dfc3cc5d2a1f25fa8`](https://github.com/vllm-project/vllm/commit/f0de1a604cad003379e5bb4dfc3cc5d2a1f25fa8),
delegates aggregate cache-group block-table width to the constituent layer
specifications. Previously the runner and indexer disagreed under DCP. The
added CPU test checks MLA DCP1/DCP2 and replicated Mamba layouts.

**Local gap:** `handoff._layout/_body_size` describes only contiguous F32 CPU
state; `gpu.runtime.GLMKernels.mla_partial` consumes caller-sized tensors;
`Scheduler.submit` reserves logical tokens without cache-byte accounting.
Before adding quantized/paged/DCP cache storage, introduce a descriptor shared
by allocation, admission, kernel metadata and serialization. Include storage
dtype, scale bytes, page size, full/shared indexer rows, shard/replica semantics
and layout version. Test allocation bytes and all metadata widths against that
one descriptor. Current F32 CPU code does not exhibit either upstream crash.

### P1 GPU: plan graph boundaries and padding together

SGLang [PR 34443](https://github.com/sgl-project/sglang/pull/34443), merged as
[`30cb848d4b455a288a37cfeb6a45aff34f5634a6`](https://github.com/sgl-project/sglang/commit/30cb848d4b455a288a37cfeb6a45aff34f5634a6),
fixes speculative DSA metadata padding under prefill context parallelism.
`cal_padded_tokens` applied CP alignment when the query path did not, producing
`num_splits` with the wrong row count. The patch makes both paths follow the
same CP-v2 condition. The submitter reports a GLM-5-FP8 serving reproduction;
this is not a local test or full-5.3 measured claim.

SGLang [PR 27053](https://github.com/sgl-project/sglang/pull/27053), merged as
[`d5e9176f6581e3c9274dd33a95315033aa194df5`](https://github.com/sgl-project/sglang/commit/d5e9176f6581e3c9274dd33a95315033aa194df5),
keeps the K-only indexer path eager, combines absorb-BMM plus attention into one
custom op, and adds breakable graph support. This avoids paying graph dispatch
for a single BMM. Its reported B200 1K/1K improvement is modest and workload
specific; increasing capture coverage from 2K to 8K also cost 2.71 GB/GPU.

**Gap at audit baseline:** `gpu/backend.py:90-100` downloads results and synchronizes/frees
scratch around primitive calls. `glm_vx/model.py:136-177` launches indexer work
through per-head primitives when the backend lacks `index_scores`; GPUBackend
has no fused indexer hook. `gpu/runtime.py:694` exposes resident asynchronous
primitives but no request graph planner. First prototype a resident indexer
and absorb-BMM/attention region with persistent scratch; choose boundaries from
launch/transfer traces before adding graph capture. A common batch descriptor
must define logical rows, physical padded rows and valid selections for all
stages. Test ragged short rows, mixed newly imported/continuing requests, and
future speculative verification modes. This is a design recommendation, not a
newly diagnosed CPU scheduler failure or an instruction to copy GPU tolerances.

**Implemented follow-up:** [fused Vx GPU index scoring](../gpu/INDEX-SCORES.md)
now supplies the missing backend hook and a resident runtime entry. The adapter
uses three uploads, one launch and one readback; the resident entry needs none
of those transfers. Compiled CPU SIMT and SM80/SM90 assembly are checked. The
broader projection/normalization/selection boundaries and actual GPU validation
remain open; this is not whole-model graph capture.

### P1 observability: separate readiness delay from compute

[Prime's October 2 full-5.3 engineering report](https://www.primeintellect.ai/blog/prime-inference)
describes loaded prefixes waiting another scheduler cycle. Reducing prompt
work per step cut queue delay in their workload. The useful extra measurement
is cache-ready-to-first-execution latency, distinct from cache transfer and
prefill computation.

**Local gap:** `Scheduler.Request` has creation/first-token timestamps and
prefill/decode/prefix durations, but `_finish` exports no admission delay,
first model-step time, longest indivisible chunk, or restored-cache-ready delay.
`_step` performs synchronous prefix lookup and a batch chunk of up to 64 tokens.
Record those boundaries before experimenting with an elapsed-time chunk target.
Measure cold arrivals beside exact hits and extended-prefix hits, preserving
per-request output and cancellation/backpressure checks. A lower token budget
is not automatically an improvement on this CPU host.

### P2: ownership must survive transfer and scratch growth

llama.cpp [PR 26500](https://github.com/ggml-org/llama.cpp/pull/26500), merged as
[`a7cc83bbae43e548df42c0af0df68f391315aa77`](https://github.com/ggml-org/llama.cpp/commit/a7cc83bbae43e548df42c0af0df68f391315aa77),
only serializes a remote buffer pointer when it belongs to the receiving RPC
socket, with a two-server regression. The reporter of
[issue 28047](https://github.com/ggml-org/llama.cpp/issues/28047) confirmed a
full GLM-5.3 multi-worker reproduction fixed at `1548a24`.

KTransformers commit
[`e80614fc2d2c3990bd1abe895d5014fc14733fde`](https://github.com/kvcache-ai/ktransformers/commit/e80614fc2d2c3990bd1abe895d5014fc14733fde)
adds [scratch-owner lifetime tests](https://github.com/kvcache-ai/ktransformers/blob/e80614fc2d2c3990bd1abe895d5014fc14733fde/kt-kernel/cpu_backend/test/test_shared_mem_buffer_lifetime.cpp):
growing a buffer must update all live request groups without replaying callbacks
for destroyed owners, including NUMA storage.

**Local status/gap:** the CPU handoff already copies arrays, disables the sender,
and bounds replay claims; GPU runtime already checks context ownership and
synchronizes before temporary release. Do not reimplement those. A future
zero-copy handoff or persistent scratch arena needs destination-scoped buffer
handles plus owner-generation tokens, tested across eviction, cancellation,
resize, failed enqueue and retransmission. These bugs are not present in the
current copying wire protocol. Durable receiver restart/retry coordination
remains intentionally absent from `HandoffReceiver`.

## Numerical and implementation evidence that must not be conflated

At reviewed llama.cpp head
[`0bb496dbd3af0add77ff82c406a915b41e839d56`](https://github.com/ggml-org/llama.cpp/blob/0bb496dbd3af0add77ff82c406a915b41e839d56/src/models/glm-dsa.cpp),
`glm-dsa.cpp` implements DSA selection, shared indexer layers and a fused
lightning-indexer branch. In contrast, older loader fix
[`796f41bedca8a786ab3eb5584cd97b7730b303d8`](https://github.com/ggml-org/llama.cpp/commit/796f41bedca8a786ab3eb5584cd97b7730b303d8)
only made absent indexer tensors optional while that graph used dense MLA.
Loading and generating at that older commit did not establish sparse-model
correctness. Pin the actual executing graph when using llama.cpp as evidence.

SGLang [index-share PR 29959](https://github.com/sgl-project/sglang/commit/92b800c531b33009ba7b412087740284eba8a943)
centralizes `should_run_indexer`: shared layers must not recompute using absent
weights if a pipeline drops the previous selection; NextN has a distinct
exception. Our `_attention` already raises when a shared layer has no selection.
Retain that check when adding pipeline or graph paths; no duplicate fix needed.

KTransformers' pinned
[AVX2 FP8 implementation](https://github.com/kvcache-ai/ktransformers/blob/a5d7ad90479c9e8bdee491c16bfd563fa9d70262/kt-kernel/operators/avx2/fp8-moe.hpp)
uses BF16 activation buffers, FP8 block-scaled weights, FP32 accumulation and
BF16 output conversion. This is concrete AMD-relevant implementation evidence,
but it is not interchangeable with this repo's F32-activation GGUF path. Its
[GLM-5.2 tutorial](https://github.com/kvcache-ai/ktransformers/blob/0e2ff783d86b7927de2966e0462eac20c2b083a8/doc/en/kt-kernel/GLM-5.2-Tutorial.md)
also describes a substantial TP8 hybrid deployment, not this CPU-only machine.

[Fireworks' numerical investigation](https://fireworks.ai/blog/reinforcement-learning-why-alignment-of-numerics-and-MoE-routing-matter)
supports treating precision and aggregation ordering as part of execution
identity. Its GLM experiment and Qwen aggregation diagnosis are separate
experiments. Neither validates our trained outputs. The existing failed trained
strict gates remain unchanged.

Additional inspected upstream heads: SGLang
`a99567ab7404902c3d1af03c0f05775262e0bbf4`; vLLM
`138810056093301f4881050fcf2b1786939da387`; KTransformers
`a5d7ad90479c9e8bdee491c16bfd563fa9d70262`.
