# GLM-Vx validation report

Built and executed on Linux x86-64 with official Vx v0.0.2 and LLVM 22.1.8. This is a CPU correctness prototype with Vx numerical kernels and Python serving/orchestration. It is not an optimal GPU engine and has not executed the trained full GLM-5.3 checkpoint.

## Verified

- 41 pytest tests passed, including 37 subtests; no skipped tests in the final run.
- 50 native Vx kernel numerical checks passed. Maximum observed absolute error across these checks: 0.00019724421 (float32 accumulation).
- Compiled Vx full/shared DSA, dense/MoE, compressed MLA paths match an independently written expanded-attention oracle on a scaled random-weight model.
- Vx versus NumPy logits, sparse selections, and greedy token outputs match within explicit tolerances on sequences longer than the tiny sparse top-k.
- Complete synthetic safetensors checkpoint loads and generates the same results as in-memory weights.
- HTTP JSON/SSE, invalid input, admission limits, cancellation, shutdown and eight simultaneous seeded requests tested.
- Official checkpoint manifest: 58,794 base-model tensors required, 118,629 indexed tensors across 141 shards. Metadata-only name validation is not evidence that full shards are locally present.

## Tiny-model CPU measurements

Three repeats per configuration, alternating backend order; one warmup token outside measurement. OPENBLAS_NUM_THREADS=1. Each request generates 16 tokens. Throughput includes prefill and scheduler execution; excludes startup/checkpoint load/HTTP transport. These are concurrent requests on one model worker, not CPU thread scaling or fused batching. All greedy outputs match between backends.

| Requests | Prompt tokens each | Vx output tok/s | NumPy output tok/s | Vx median TTFT (ms) |
|---:|---:|---:|---:|---:|
| 1 | 8 | 376.0 | 1035.9 | 14.0 |
| 1 | 32 | 199.6 | 567.7 | 51.2 |
| 2 | 8 | 415.9 | 1110.0 | 23.0 |
| 2 | 32 | 213.6 | 569.9 | 95.6 |
| 4 | 8 | 426.0 | 1063.2 | 45.0 |
| 4 | 32 | 216.5 | 588.8 | 189.5 |
| 8 | 8 | 437.5 | 1183.4 | 78.4 |
| 8 | 32 | 216.5 | 591.8 | 374.5 |

Vx is slower than NumPy on every measured tiny workload. The scalar kernels and frequent Python/ctypes calls remain bottlenecks. Native fused index scoring/LayerNorm/top-k, and bounded reuse of RoPE coefficients reduce overhead, but do not establish a competitive production engine. Initial and final raw measurements are retained; no statistical significance or full-model extrapolation is claimed.

## Boundaries and next requirements

- The full checkpoint declares 755,617,140,416 bytes. This machine initially had about 326 GiB free disk and 93 GiB RAM, and no detected NVIDIA GPU. Full-model execution was not attempted.
- FP8 blocks decode to float32 on CPU; dynamic FP8 activation quantization and tensor-core arithmetic are not reproduced. No trained-generation quality, full-checkpoint logit parity, or GPU throughput result exists.
- Current cache payload uses 190,464 float32 bytes per token at full model dimensions (including DSA index keys); 8,192 tokens require 1,560,281,088 payload bytes before objects/workspaces/weights. Logical scheduler token admission is not a GPU memory-fit proof.
- No tensor/expert parallel collectives, paged GPU cache, fused GPU prefill, prefix sharing, MTP speculative decoding, native GPU execution or complete chat/tool-calling API.
- Production optimization needs a specified GPU/node topology, local weights with enough storage, compiled accelerator kernels, full-model differential tests and workload-matched comparisons against established serving engines.

## Reproduce

See README.md, docs/toolchain.md, scripts/test.sh and scripts/benchmark_matrix.py. Source/model revisions and hashes are in metadata/sources.json, metadata/toolchain.json, and kernels/verification.json. The compiler installation is isolated at /srv/nanocodex/workspace/vx-toolchain; project is /srv/nanocodex/workspace/glm-vx.
