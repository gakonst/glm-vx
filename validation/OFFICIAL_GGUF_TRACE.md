# Official eager GGUF trace with streamed weight storage

`validation/official_gguf_trace.py` executes the pinned official GLM eager CPU
**decoder layer, attention, router, expert, MLP, normalization, and rotary
forward methods**, using decoded GGUF F32 weights. It does not call the Vx model,
Vx numerical kernels, or the independent expanded NumPy equations. It does not
instantiate the official whole model or allocate a CPU expert bank.

This is an explicit storage adaptation, not a claim of untouched weight layout
or execution of `GlmMoeDsaForCausalLM.forward` itself. The wrapper orchestrates
unchanged official decoder layers in the same order, calls the official causal
mask and RoPE helpers, and uses the official RMSNorm and PyTorch Linear final
head. Tiny comparisons against the actual ordinary whole-model forward establish
bitwise agreement for the tested paths.

## Storage and numerical scope

- Instantiate one official `GlmMoeDsaDecoderLayer` under `torch.device('meta')`.
  Verify every parameter/buffer is meta before loading anything.
- Materialize that layer's attention/dense/shared/norm/router weights and buffers
  as CPU F32 views of decoded arrays. No reduced activation precision is enabled.
- Remove only `Experts.gate_up_proj` and `Experts.down_proj` meta parameters and
  replace their storage with `LazyExpertBank.__getitem__`. Unchanged official
  `Experts.forward` chooses the active experts. Each scalar lookup decodes that
  expert's gate/up (concatenated to the official `[2I,H]` layout) or down matrix.
  It cannot request a slice, vector of experts, or full `[E,2I,H]` bank.
- Keep checkpoint decoded caching disabled. Release a layer before loading the
  next one. Decode only requested embedding rows. The final full vocabulary head
  is materialized once and executed with ordinary `torch.nn.Linear.forward`.
  For this checkpoint it is approximately 3.8 GB decimal / 3.55 GiB.
- Run `eval()`/`torch.no_grad()` throughout. Read-only decoded arrays are wrapped
  without copying; official inference methods read, never update, those weights.
- Teacher forcing is limited to 1–32 contiguous positions beginning at zero.
  This tool performs one causal batch, not sampling or incremental-cache decode.
  Only the base decoder is included; no MTP head or draft verification runs.
- The 8 GiB default weight budget rejects oversized layer-plus-expert or individual
  weight materializations. It is **not** an OS-enforced peak-RSS cap: decoder
  temporaries, PyTorch/BLAS workspace, Python, allocator retention and page cache
  add overhead. The receipt records peak RSS and allocation estimates. Traces
  are written directly into an NPZ ZIP stream instead of retaining every stage.

The exact mode name is `official-eager-cpu-f32-streamed-storage-v1`. It differs
from original FP8/BF16 GPU execution and native GGML packed-dot/cache numerics.
The GGUF decoder and tensor-name/layout mapping remain shared with the engine;
this is independent numerical forward execution, not an independent checkpoint
conversion. The decoder's `gguf/quants.py` hash must match the earlier native
GGML codec evidence (`gguf==0.17.1`). Unsupported bias, non-SiLU, grouped routing,
non-normalized routing, nondefault RoPE, expert parallelism, non-indexed layer
patterns and invalid dimensions/tokens are rejected.

## Evidence and trace format

The installed model/config files must match `metadata/sources.json`, pinned to
Transformers `469230357aab0f2b303b0d638c1f8d06edb14184`. Receipts hash those files,
mask/RoPE/activation/MoE integration source, Torch linear/functional source, Torch
native libraries, NumPy's numerical extension, GGUF Python source, installed
package metadata/archive origin, and the local wrapper/checkpoint source.
Source/config/library fingerprints are checked again at completion.

`--contract` reuses **only** existing config/checkpoint identity. Its thresholds
are neither edited nor used to grant eligibility. Checkpoint SHA256 values are
explicitly recorded as **declared, not rehashed**; shard size/mtime are checked
before and after. The shard paths actually opened by the loader must match the
bound set; a neighboring unrelated checkpoint is rejected. Teacher-forced tokens are recorded separately. No 216 GB hash
scan is performed. A complete trace receipt still has `eligible:false`; an
independently approved comparison contract is a separate requirement. Exceptions
produce a `failed` receipt with a possibly partial NPZ, never a complete one.
Existing output directories are rejected.

For each position/layer, NPZ captures:

- `pN.layer.L.output`: official hidden output.
- `pN.layer.L.official_selected`: raw official top-k slots, including masked
  future slots that eager batch can return before enough causal tokens exist.
- `pN.layer.L.selected`: sorted **causal membership** used by the official mask.
  This ordering is not the engine's score-sorted selection ordering. Compare
  membership explicitly; never silently relabel raw ordered arrays as equal.
- Sparse MLP `router_logits`, `route_ids`, `route_weights`: raw official values,
  preserving the ID/weight association and unsorted-top-k order. Canonicalize
  pairs explicitly if comparing membership/weights with a score-sorted router.
- `pN.logits`: complete final vocabulary logits.

## Tiny validation and trained comparison

`tests/test_official_gguf_trace.py` runs in the isolated official environment:
**23 passed in 5.42 seconds**. No test opens the trained checkpoint. Tests include:

- Three ordinary official whole-model comparisons: seeds 123/456/789, lengths
  4/6/1, dense+sparse and all-sparse patterns, shared/full indexers, top-k 2.
  All **260** captured tensors and all logits are bitwise equal, including raw
  routing IDs/weights and raw/canonical selected positions.
- Meta-only allocation of the 256-expert banks; no full-model constructor called
  by streamed execution; loaded expert IDs exactly equal the official selected
  union, once per gate-up/down bank.
- Previous layer arrays released **before** the next layer loads, and no weight
  arrays retained after completion; scalar-only expert storage access.
- Unsupported-config, token-bound, memory/rank, changed official-source and
  observer-isolation checks; complete/failed CLI receipts and overwrite refusal
  using an explicitly injected tiny checkpoint provider.

`validation/evidence/official-streamed-tiny-20261004.json` binds the test result
and dependency fingerprints. The full trained run and its independent Vx comparison are recorded in the
[current integration report](../docs/OPTIMIZATION-PROGRESS.md). Original failed
gate receipts and all thresholds remain untouched.

## Reproduce a trace

The already-installed environment is in this parity worktree. It can run the
script from another checkout by setting that checkout's `PYTHONPATH`/cwd; the
receipt will fingerprint the actual loaded source. Use a fresh output directory.
No Vx library or toolchain environment is needed by this independent execution.

```sh
# From the checkout containing this implementation:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD" \
  /srv/nanocodex/workspace/glm-vx-parity/build/official-venv/bin/python \
  -m validation.official_gguf_trace \
  --gguf /srv/nanocodex/workspace/models/glm-5.3-iq1s \
  --contract /srv/nanocodex/workspace/glm-vx/build/parity/expanded-gate/contract.json \
  --tokens 9703 10056 --threads 1 --decode-threads 4 --max-weight-gib 8 \
  --output build/parity/official-trained-f32-streamed-v1
```

The ordinary installed runtime used for validation is CPU Torch `2.14.1+cpu`,
Transformers `5.19.0.dev0` at the pinned revision, NumPy `2.2.6`, GGUF `0.17.1`,
pytest `9.1.1`. The environment was aligned to the existing verified codec with:

```sh
/srv/nanocodex/workspace/glm-vx/.venv/bin/pip \
  --python build/official-venv/bin/python install pytest==9.1.1 gguf==0.17.1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH="$PWD" \
  build/official-venv/bin/python -m pytest tests/test_official_gguf_trace.py -q
```

## Compare to a bound Vx trace

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m validation.export_vx_trace \
  --gguf /path/to/glm-5.3-iq1s \
  --contract docs/optimization-evidence/frozen-trained-contract.json \
  --library kernels/build-o3/libglm_vx.so --packed-weights \
  --tokens 9703 10056 --output build/my-vx-trace
.venv/bin/python -m validation.compare_official_trace \
  --contract docs/optimization-evidence/frozen-trained-contract.json \
  --config /path/to/glm-5.3-iq1s/config.json \
  --reference build/my-official-trace --candidate build/my-vx-trace \
  --library kernels/build-o3/libglm_vx.so --output build/my-comparison.json
```

Use the same contract path/content and token IDs for both exporters, and a fresh
output path for every run. A manifest alongside the checkpoint must identify the
same declared shard digests. The candidate fingerprints actual imported local
modules, the independently validated GGUF decoder, NumPy and the selected native
library; all are checked again after serialization. The comparator verifies
receipt/archive hashes, loaded shard bindings, every layer and all vocabulary
logits for every required position. Missing, skipped or stale data fails.

DSA comparison is exact causal membership, retaining raw official selected slots
in the archive. Expert IDs compare exactly after canonical sorting; their weights
move with the corresponding IDs. Raw ordered arrays are never rewritten. Routing
weights use the inherited layer-output budget as an explicitly uncalibrated
additional diagnostic. Every numerical failure is retained; budgets remain the
original values. The old contract's producer/archive fields describe the historical
expanded reference, so this comparator records the current producer/archive pins
separately and explicitly reuses only geometry, weights/config and budgets.

Passing a finite comparison still sets `release_eligible:false`: it is not a
producer attestation, original-FP8 quality validation, held-out prompt suite,
long-context proof, or permission to alter tolerances. Mutation tests in
`tests/test_official_trace_comparison.py` verify rejection behavior independently
of either model implementation.
