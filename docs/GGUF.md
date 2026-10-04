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
  --gguf /path/to/glm-5.3-iq1s --raw --prompt 'Hello' --max-tokens 8 \
  --output completion.json
```

The downloader pins revisions, verifies every shard's SHA256, resumes `.part`
files, preserves a 10 GiB disk margin and verifies the matching official config,
tokenizer and chat template. It does not download a second copy to a Hub cache.
Remove `--raw` to format the user message with the pinned official chat template;
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
The full regression run passed 271 tests plus 37 subtests, with 14 actual GPU
tests skipped. A subsequent CLI suite passed three tests, including the newly
added real HTTP server process test (272 distinct passing tests in total).
Real trained-model execution evidence is recorded separately when available.
