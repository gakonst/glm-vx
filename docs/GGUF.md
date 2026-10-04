# GLM-5.3 GGUF CPU proof of concept

This path runs the existing GLM model graph and Vx CPU arithmetic against the
full trained model's quantized GGUF tensors. The upstream `gguf==0.17.1` package
parses the container and unpacks quantization; it does not perform inference.
No llama.cpp, Transformers, vLLM or SGLang serving engine is invoked.

## Checkpoint and setup

Pinned model: `unsloth/GLM-5.3-GGUF`, revision
`346b3591c7f28d1a23716f97a065ecf12ec14771`, `UD-IQ1_S`, six files totaling
216,715,365,893 bytes. This is a mixed-bit quantization, not every weight at one
bit. It contains F32, IQ1_S, IQ2_XXS, IQ3_XXS, IQ4_XS, Q2_K, Q3_K, Q4_K, Q5_K,
Q6_K and Q8_0 tensors. The 78-layer base decoder is used; the extra MTP layer is
not required for ordinary autoregressive decoding and is not executed.

```sh
.venv/bin/pip install -e '.[gguf]'
# Build the Vx CPU library first, following the root README.
.venv/bin/python scripts/download_glm_gguf.py /path/to/glm-5.3-iq1s
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m glm_vx.generate \
  --gguf /path/to/glm-5.3-iq1s --prompt 'Hi' --max-tokens 8 --decode-threads 4 \
  --output completion.json
```

The downloader pins revisions, verifies every shard's SHA256, resumes `.part`
files, preserves a 10 GiB disk margin and verifies the matching official config,
tokenizer and chat template. It does not download a second copy to a Hub cache.
The default formats the user message with the pinned official chat template;
this selects low reasoning effort and adds the model's thinking prefix.
`--reasoning-effort` accepts low/high/max.
Generated output may be truncated before reasoning or the final answer finishes.

For HTTP serving:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m glm_vx.server \
  --gguf /path/to/glm-5.3-iq1s --backend vx --max-sequences 1
```

The existing `/v1/completions` endpoint accepts raw text or token IDs. It does not
apply a chat template automatically. This slow CPU path may need a much longer
client timeout; SSE sends heartbeats while the model works. The service defaults
to loopback and the same completion subset described in the root README.

## Memory and implementation

All six files are memory-mapped read-only. The loader validates split numbering,
complete tensor counts, duplicate names, bounds, layout and architecture metadata
before inference. It decodes only the selected expert or requested embedding
rows; it never expands an entire bank of 256 experts. Other matrices are decoded
as needed, in bounded row chunks. The decoded-weight LRU defaults to 512 MiB
(`--decoded-cache-mib`), but this is not a total-process memory cap: live matrices,
activations, decoder scratch and mapped file pages consume additional RAM. Optional
`--decode-threads 4` unpacks disjoint row chunks concurrently; this uses CPU
threads only and can increase temporary memory. A
full vocabulary projection can temporarily require several GiB. OS paging makes
this possible without fitting all weights in RAM; it does not make it fast.

The GGUF K-B transpose and per-head K/V split are explicitly inverted. Other
projections retain their coordinate ordering. [Converter audit](gguf-evidence/converter-audit.md)
links the pinned upstream transformation. Source quantization is lossy, so parity
with the original FP8 checkpoint is not implied.

Synthetic tests check real GGUF files, independent Q8_0/IQ1_S byte patterns,
row/expert slicing, mapping of every tensor, model/cache parity, cache eviction,
malformed files and CLI generation. [Header validation](gguf-evidence/header-validation.txt)
checked all 1,809 real checkpoint descriptors without substituting any weights.
The latest full regression run passed **278 tests plus 37 subtests**, with 14
actual GPU tests skipped; see [test output](gguf-evidence/parallel-focused-tests.txt).

## Actual trained-model execution

The complete 216.7 GB checkpoint was downloaded and SHA256-verified. A raw
`Hello` prompt traversed all 78 layers and selected EOS through Vx, the NumPy
arithmetic reference, and the real HTTP/SSE server. Vx and NumPy agreed on the
top-five token IDs, with maximum top-five logit difference below 0.000019.
This is a forward-pass and serving check, not a useful chat answer.

The Vx run took 430 seconds including loading, with 26.5 GB peak RSS. The HTTP
run took 423 seconds and emitted 42 SSE heartbeats. These runs overlapped on a
shared CPU and are **not a controlled performance benchmark**. The NumPy path
shares the model graph, so agreement does not independently validate the whole
architecture or establish parity with the original FP8 weights.
[Receipts and limitations](gguf-evidence/real-model-summary.json).


### Two-token chat and cached decode

The Vx engine completed the six-token abbreviated prefix
`[gMASK]<sop><|user|>Hi<|assistant|><think>` and generated **`Let me`**, token IDs
`[10056, 752]`. This exercises all 78 ordinary layers and the cached forward
for the second generated token. A separately built llama.cpp CPU reference
at revision `11fe02151f79c41d0d4af7da708755d73b9c0da6` generated the same text
from exactly the same input token IDs. The reference is validation only and
is never called by our engine.

The Vx run took **2032 seconds (33.9 minutes)** including load and sequential
prefill; first token arrived after 1768 seconds. Peak RSS was **32.2 GB** with
four quantization-decoder workers. Reference building/loading/inference
overlapped the run, so these are observed smoke-test timings, not a controlled
benchmark. The independent reference was substantially faster: its reported
prompt evaluation was 41.9 seconds and the cached decode was 8.7 seconds.

This abbreviated prefix omits the official reasoning-effort system turn.
`Let me` is only the start of reasoning, and generation stopped at the requested
two-token limit. This proves a limited real-model CPU generation path; it does
not establish answer quality, long-context correctness, full numerical parity,
or competitive serving speed.

[Summary](gguf-evidence/real-chat-summary.json),
[Vx receipt](gguf-evidence/real-vx-chat-completion.json),
[independent reference](gguf-evidence/llama-cpp-chat-reference.txt),
[tokenizer alignment](gguf-evidence/tokenizer-alignment.json). The 24 extra
GGUF vocabulary entries are padding slots; all 154856 actual tokenizer entries
match at the same IDs.

To reproduce the short smoke:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m glm_vx.generate \
  --gguf /path/to/glm-5.3-iq1s --raw \
  --prompt '[gMASK]<sop><|user|>Hi<|assistant|><think>' \
  --max-tokens 2 --decode-threads 4 --output completion.json
```
