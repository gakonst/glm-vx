# GLM-Vx

An experimental, from-scratch GLM-MoE-DSA inference engine using **real compiled
Vx numerical kernels**. Python handles checkpoint IO, tensor storage, model
orchestration, request scheduling and HTTP. No vLLM, SGLang, Transformers or
PyTorch inference runtime is used. NumPy supplies storage and an explicit test
oracle; the `vx` backend never silently falls back to NumPy matrix operations.

**Current status: tested CPU prototype plus an explicit NVIDIA GPU kernel/backend path.**
The GPU path compiles to PTX; actual GPU execution and optimal performance remain
unverified. See [GPU implementation and run instructions](gpu/README.md).
The complete published GLM-5.3 checkpoint has not been run. The included tiny
model has deterministic random weights and is for testing mechanics only.

## Run

Prerequisites: Linux x86-64, Python 3.10+, C linker, working Vx v0.0.2 and LLVM 22.
See [Vx installation](https://vxlang.org/docs/getting-started.html).

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
VXC=/path/to/vxc kernels/build.sh
.venv/bin/python -m glm_vx.server --tiny --backend vx
```

On the implementation machine the isolated compiler is already available:

```sh
cd /srv/nanocodex/workspace/glm-vx
VXC=/srv/nanocodex/workspace/vx-toolchain/bin/vxc kernels/build.sh
.venv/bin/python -m glm_vx.server --tiny --backend vx
```

```sh
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":[1,2,3],"max_tokens":8,"stream":true}'
```

`/health` reports backend and capacity; `/v1/models` lists the local model.
`/v1/completions` accepts one nonempty token-ID prompt or a string when a local
`--tokenizer tokenizer.json` is provided. Greedy sampling is the default; finite
nonnegative temperature and an optional seed support reproducible sampling.
Unknown options are rejected. This is a **completion API subset**, not a claim
of complete OpenAI chat/tool-call compatibility. SSE emits exact token IDs;
decoded text is returned in the final event to avoid corrupting split UTF-8/BPE
pieces. Final `token_ids` contains the complete output, while earlier events
contain one delta ID each. Clients must not concatenate the final full list again.

The service binds loopback by default. It has no authentication or TLS; place a
suitable gateway in front before exposing it to a network.

## GPU implementation

[GPU code](gpu/README.md) adds coalesced F32/FP8 projection, fused residual/RMSNorm,
warp/block reductions, split online-softmax compressed MLA, and a CUDA Driver API
runtime. Use `--backend vx-gpu` for the explicitly hybrid serving adapter.
The [original CPU report](REPORT.md) remains a baseline, not a GPU benchmark.

## Implementation

- `kernels/kernels.vx`: float32 matvec, RMSNorm, LayerNorm, interleaved RoPE,
  stable softmax, SwiGLU, sigmoid router with correction bias, top-k and attention.
- `glm_vx/model.py`: dense and sparse blocks, selected routed experts plus shared
  experts, absorbed MLA decode, compressed request-local KV, full/shared DSA
  indexer selection, and causal sequential prefill.
- `glm_vx/checkpoint.py`: lazy read-only safetensors mmap with checked headers,
  BF16/F16/F32/FP8 E4M3FN decoding and block scales. No remote code execution.
- `glm_vx/scheduler.py`: fair request interleaving, chunked sequential prefill,
  prompt-plus-output token admission, cancellation, failure isolation and cleanup.
  Disconnect detection is cooperative between model forwards (socket polling
  every 250 ms); keep the TCP connection open until the reply. Shutdown raises
  an explicit timeout if a forward is still running, leaving weights open.
  This is not fused tensor batching or paged GPU KV allocation.
- `glm_vx/server.py`: bounded HTTP handlers, JSON completion and SSE transport.

## Validation and measurement

```sh
.venv/bin/python -m pytest tests kernels/test_backend.py -q
.venv/bin/python kernels/test_kernels.py
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python scripts/benchmark.py \
  --requests 2 --prompt 8 --tokens 8 --output benchmark.json
```

The benchmark verifies matching greedy outputs first, excludes one-token warmup,
then reports scheduler throughput and per-request time to first token on the
**tiny random model**. It does not measure full-model quality or GPU throughput.
See `REPORT.md` for the measured results and exact validation scope.

## Full checkpoint and remaining work

`--checkpoint DIRECTORY --tokenizer DIRECTORY/tokenizer.json` is an experimental
CPU path for a locally supplied checkpoint. It does not download weights.
Config/header validation and synthetic loader tests are not full-checkpoint
validation. Float32 dequantization is not numerically identical to a production
FP8 activation-quantized GEMM path. Metadata and source revisions are recorded in
`metadata/sources.json` and `docs/model.md`.

The available machine has no detected NVIDIA GPU, about 93 GiB RAM and roughly
326 GiB disk free at initial inspection. The checkpoint is about 756 GB: it cannot
be downloaded here in full. No GPU hardware was rented or provisioned.

To deliver an optimized production engine still requires: measured GPU tuning and tensor-core lowering, fused/parallel prefill, FP8 GEMM and quantized
activation parity, paged KV/prefix sharing, tensor/expert parallel collectives,
GPU topology-specific placement, trained-checkpoint logits/generation parity,
and throughput/latency benchmarks against established engines on matching
hardware and workloads. “Optimal” cannot be established from this CPU prototype.
