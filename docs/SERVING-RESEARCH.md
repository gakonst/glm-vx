# GLM serving research and implementation roadmap

Reviewed 2026-10-04 against GLM-Vx `4845ebf89b8b9a22bcd9b8811c77dfc5d6b77cf2`.
This is a broad primary-source review, not an exhaustive literature census or a
claim that the techniques below have been implemented or benchmarked in Vx.
Provider results describe their own hardware, workloads and numerical modes.

## Conclusions for this engine

Prefill/decode separation is one component. The missing work spans numerical
semantics, packed-weight computation, sparse indexing, speculative decoding,
cache management, batching, communication and API behavior. Our CPU process
handoff proves a transport contract on small fixtures; it does not establish a
fast disaggregated service. Neither current trained-model strict trace gate
passes; optimization promotion remains blocked.

Full GLM-5.3 and GLM-5.3-Flash must remain separate targets. The official
[release registry](https://github.com/zai-org/GLM-5) identifies full 5.3 with the
5.2 base, while Flash has a different hybrid architecture. Thus 5.2 serving work
is a strong starting point, subject to checkpoint configuration and numerical
validation. The older GLM-5 report is architectural background, not the complete
5.3 specification. Do not substitute Flash demonstrations for full-model evidence.

## GLM-specific provider engineering

### 1. Prime Inference — October 2, 2026

[Primary engineering post](https://www.primeintellect.ai/blog/prime-inference).
Full GLM-5.3 on GB200: separate phase topologies, cache-aware routing, native
compressed-KV attention, and block-major transfer layouts. Its study found
DEP8 useful for prefill and TP4 for decode; wider parallelism was not universally
better. The key lesson is to measure scheduling, cache capacity, kernel time and
transfer fragmentation together. Native low-bit KV needs its own numerical
contract. Their workload-specific results do not establish our optimal topology.

### 2. SGLang GLM-5.2 optimization — July 2026

[Engineering post](https://www.lmsys.org/blog/2026-07-13-glm52-optimization).
MTP with index sharing, overlapping CPU planning, graph-compatible draft paths,
indexer prologue fusion, and radix top-k are concrete missing opportunities.
Coarse low-precision histograms locate the selection threshold; exact FP32
refinement preserves selection semantics. The reported top-k kernel improvement
at one-million context is about 10x, not a whole-engine speedup. Our next GPU
profiling breakdown should separate index scoring, selection, page mapping and
attention, rather than treating DSA as one operation.

### 3. vLLM GLM-5.2 production SLAs — July 23, 2026

[Engineering post](https://vllm.ai/blog/2026-07-23-glm-5.2-nvfp4-b300-pd).
A particularly relevant handoff failure: newly arriving requests created mixed
decode batches that selected a slower execution path. Speculative padding
restored the efficient path; reported mean TPOT fell from roughly 40 to 22 ms.
Runner overhead, communication backend, graph mode, MTP and indexer caching also
matter. Test newly imported requests together with established decode requests,
not only isolated handoff correctness. Padding must preserve masks and outputs.

### 4. vLLM Hybrid HiSparse for full GLM-5.3 — September 8, 2026

[Engineering post and reproduction recipe](https://vllm.ai/blog/2026-09-08-glm53-part1-hybrid-sparse-offloading).
Keep KV resident while capacity permits; under pressure, release cold pages and
fetch only selected sparse rows into hot buffers. Resident and hot rows share a
pool; the resolver remains graph-compatible. Indexer state still grows with
context, and MTP increases simultaneous hot-row requirements. This targets KV
pressure and concurrency, not CPU expert-weight throughput. Model host-memory
budget, transfer traffic and hot-buffer occupancy explicitly.

### 5. SGLang HiSparse — April 10, 2026

[Engineering post](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse).
GLM-5.1 experiments demonstrate sparse KV offload using host storage plus a GPU
hot buffer. Higher concurrency can benefit substantially, while low concurrency
can pay transfer overhead. Use it as a baseline for residency policies, not a
reason to offload all caches eagerly. Distinguish cache offload from weights.

### 6. Baseten GLM-5.2

[Engineering post](https://www.baseten.co/blog/how-we-built-the-worlds-fastest-api-for-glm-52/).
Shared-DSA implementation, calibrated NVFP4 weights, MTP and Dynamo-backed
cache-aware P/D serving contribute together. This is useful evidence for
co-designing model execution and routing. IQ1_S GGUF and NVFP4 are different
numerical modes: neither their kernels nor quality results transfer automatically.

### 7. Fireworks GLM-5.2 Fast

[Provider post](https://fireworks.ai/blog/glm-5p2-fast).
Long agent contexts motivate tiered KV storage; attention and expert computation
benefit from different parallel layouts. Expert imbalance and collective cost
belong in the profiling plan alongside matrix throughput. Treat their speed
figures as provider measurements, not independently reproduced results here.

### 8. Fireworks numerical alignment — September 30, 2026

[Engineering investigation](https://fireworks.ai/blog/reinforcement-learning-why-alignment-of-numerics-and-MoE-routing-matter).
GLM-5.2 RL experiments show that probability disagreement can affect training.
A separate Qwen MoE investigation traces disagreement to precision and ordering
of expert aggregation. This does not diagnose our failure, but motivates tracing
routing weights, per-expert outputs and their weighted reduction. Matching expert
IDs or generated words is insufficient. The GLM experiment's zero-KL statement
is limited to its tested LoRA setup; it is not a universal parity guarantee.

## Papers and implementation references

Each entry identifies a mechanism to evaluate, not a dependency to silently
replace the from-scratch serving runtime.

| Source | Mechanism and applicability | Important boundary |
|---|---|---|
| [GLM-5 report](https://arxiv.org/html/2602.15763v1) | MLA, DSA and parameter-shared MTP explain the architectural starting point. | Pin 5.3 config; this is an older model report. Acceptance length is workload dependent. |
| [DeepSeek-V3.2 report](https://arxiv.org/html/2512.02556v1) | Lightning indexer scores context, selects positions, then sparse attention consumes selected MLA state. | Index scoring still scans history; sparse attention does not make the entire layer constant-time. |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | Specialized sparse prefill/decode and quantized paged-KV layouts. | Current September 30 release removed earlier model/Hopper support; README points legacy users to `ba89a3466e9470ad08ab39738d4e7bb66989e1e7`. Pin compatible code. |
| [FlashInfer](https://arxiv.org/abs/2501.01005) | Separate dynamic planning from graph-compatible execution; composable block-sparse KV and load-balanced attention. | Generic support is not proof of GLM-5.3 shape/layout compatibility. |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | Grouped expert GEMMs; masked graph-compatible decode; specialized DSA indexer logits. | SM90/SM100 packing and scaling differ. Include casting and layout conversion in timing. |
| [DeepEP](https://github.com/deepseek-ai/DeepEP) | Specialized expert dispatch/combine feeding grouped computation. | Current V2.5 differs from older NVSHMEM designs; communication consumes resources. CPU offload needs additional scheduling. |
| [DistServe](https://arxiv.org/abs/2401.09670) | Independently allocate prefill/decode pools and parallelism to maximize SLO-compliant goodput. | Network and KV/indexer transfer cost determine the break-even point. |
| [Splitwise](https://arxiv.org/abs/2311.18677) | Select different prompt/generation hardware by phase-specific performance, cost and power. | Decode is not always purely bandwidth-bound, particularly with batched MoE work. |
| [Mooncake](https://arxiv.org/abs/2407.00079) | Distributed KV storage, locality-aware scheduling and overload control for disaggregated serving. | Cache retrieval can cost more than recomputation; this is not expert-weight offload. |
| [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | Bounded prefill chunks avoid long decode stalls and improve iteration balance. | Establish this colocated baseline before claiming physical disaggregation helps. |
| [SGLang](https://arxiv.org/abs/2312.07104) | Radix prefix reuse and efficient structured decoding for branching programs. | Cache identity must include model, positions and all required sparse-indexer state. |
| [MegaScale-Infer](https://arxiv.org/html/2504.02263v4) | Attention/FFN pools aggregate expert work and overlap it with attention through microbatch pipelines. | Different from P/D separation; adds layer-level activation transfers and needs sufficient concurrency. |
| [FreeToken](https://arxiv.org/html/2608.16157v1) | Double-buffer prefill expert movement; cache decode experts; divide misses between GPU transfer and CPU execution using measured bandwidth. | Full GLM-5.2 demonstration uses substantial host RAM plus a GPU. Not CPU-only or full-5.3 evidence. |
| [KTransformers, SOSP 2025](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) | Packed CPU expert kernels, size-dependent AMX/AVX-512 paths and overlapped CPU/GPU scheduling. | Our AMD host does not inherit Intel AMX results. Expert Deferral changes computation and is excluded from strict parity. |

## Automatic optimization references

[MetaInfer v1](https://arxiv.org/abs/2607.12875v1) describes generating specialized
engines from runtime constraints and a contract knowledge base. The
[current v3](https://arxiv.org/abs/2607.12875v3) has a different title and emphasizes
model/hardware adaptation; its examples include GLM-5.3-Flash, not full 5.3.
Pin the revision when reproducing either approach.

[Baseten's October 2 experiment](https://www.baseten.co/blog/agentic-inference-optimization-faster-than-sota/)
applies the approach to Qwen on a B200 and evaluates deployed services. It reports
large improvements, including on speculation-friendly text, after substantial
compute expenditure. The useful transferable idea is a fixed evaluation contract
plus end-to-end profiling. It is not a GLM speedup forecast or evidence that tiny
fixtures suffice. Its accepted numerical deviations were an explicit choice;
our candidate must not make that choice by editing its own gate.

## X cross-checks

Publisher threads were useful discovery leads. Prefer their linked technical
writeups for implementation evidence; social metrics are not benchmark validation.

- [Prime launch thread](https://x.com/PrimeIntellect/status/2106146483003384253).
- [vLLM full GLM-5.3 support](https://x.com/vllm_project/status/2093354756244992383), with the [maintained recipe](https://recipes.vllm.ai/zai-org/GLM-5.3).
- [SGLang full GLM-5.3 support](https://x.com/sgl_project/status/2093355026223874067).
- [vLLM Hybrid HiSparse discussion](https://x.com/vllm_project/status/2097397769338282222).
- [vLLM/TileRT specialized decode announcement](https://x.com/vllm_project/status/2103297683188527487): a lead for heterogeneous serving-engine composition, not a reproduced result here.

## Ordered work for GLM-Vx

This ordering is our engineering judgment from the sources and current audit.

| Priority | Work | Acceptance evidence |
|---|---|---|
| P0 | Resolve reference numerical modes and first divergent operation; extend traces to MoE aggregation, indexer and cache state. | Independent teacher-forced traces on the same weights, calibrated frozen per-operation budgets, exact discrete contracts, held-out corpus. |
| P1 CPU | Gate O3; implement packed GGUF dot products in Vx; profile page faults, dequantization, allocations and expert execution. | All eleven checkpoint formats, boundary/tie fixtures, real-layer replay and full-model output; warm/cold timings with fixed threads. |
| P1 serving | True batched/chunked prefill, prefix reuse, persistent bounded cache storage and admission/backpressure. | Per-request outputs invariant to scheduling; client TTFT/ITL and memory under mixed arrivals. |
| P1 GPU | Resident execution, grouped/tiled tensor-core GEMMs, fused indexer preparation, exact top-k/page mapping, graph-compatible metadata. | Actual-device correctness, memory/race checks and per-stage profiles. PTX compilation alone is insufficient. |
| P1/P2 MTP | Load and validate the draft head; implement verification, rejection rollback and shared indexing according to checkpoint semantics. | Accepted output tokens/s including draft cost; greedy equivalence and, for stochastic decoding, correct acceptance/rejection distribution. |
| P2 cache | Paged sparse KV, pressure-driven host tier and cache-aware routing. | Eviction/reuse/handoff correctness, transfer volume and cache-hit distributions; quantify indexer growth. |
| P2 cluster | Compare colocated versus P/D pools with independently tuned parallelism and transfer layout. | Matched arrival traces; include mixed handoff batches, queueing, network cost and p95 latency. |
| P2/P3 | Attention/FFN disaggregation, hybrid expert placement, lower-precision KV. | Enough concurrency/hardware to justify transfer; separate numerical contracts for changed precision. |

CPU-only priorities remain relevant even if GPU work is deferred. On this host,
the checkpoint exceeds RAM; disk traffic must be measured independently from
resident arithmetic. A second process on the same CPU does not create more
memory bandwidth and can increase contention. Test disaggregation as a protocol
now; demonstrate performance benefits only on an appropriate measured deployment.

## Vx-specific implications

The [existing audit](CORRECTNESS-AND-PERFORMANCE.md) shows that we are far from
Vx's performance limits: scalar F32 CPU work, expanded GGUF rows, sequential
prefill, partial GPU residency and no validated tensor-core path. The isolated
O3 matvec improvement is real but not yet a full-engine promoted result.

[Vx's primary repository](https://github.com/vx-lang/Vx) describes typed residency
and heterogeneous placement. These can express cache ownership and transfer
obligations, but do not establish optimal numerical code generation. Build small
compiler capability probes for SIMD, tensor-core instructions, asynchronous
staging, layout and placement before selecting a kernel design. Inspect emitted
assembly/PTX and measure hardware counters. If an essential instruction is not
expressible, record a compiler/backend gap rather than calling scalar code optimal.

For each optimization retain a manifest binding source, compiler, flags,
hardware, checkpoint hashes, numerical mode, trace contract and workload.
The optimizer may change implementation and tuning parameters; it may not change
reference traces, thresholds, test coverage or benchmark semantics. Require
correctness first, then improvements beyond noise on held-out traffic, including
p50/p95 TTFT and inter-token latency, goodput, memory and accepted output tokens.
Do not count cached input tokens as newly computed throughput without labeling it.

## Review limitations

No GPU benchmark was run for this review. No complete original-weight quality
suite or long-context trained-model parity run was completed. Existing failed
receipts remain unchanged. Primary sources were reviewed for mechanisms and
applicability, not every linked commit or every published inference paper.
Future implementation should pin the relevant revisions and reproduce each
claimed benefit independently. A passing finite suite defines tested coverage;
it cannot prove correctness for every possible input.
