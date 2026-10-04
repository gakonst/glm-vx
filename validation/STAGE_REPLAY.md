# Numerical modes and stage replay (2026-10-04)

The trained-model promotion gate is still failed. No thresholds, frozen traces,
contract, or historical receipts were edited. These tools localize disagreement;
they do not make a new mode eligible by borrowing an old mode's passing result.
The compact evidence is in `validation/evidence/stage-replay-20261004.json`.

## Findings and fixes

1. **The original expanded “F32” reference was mixed precision on NumPy 2.2.6.**
   `float32_array / np.sqrt(integer)` produces float64. This affected indexer
   head weights/scores and attention scores/softmax. `f32-v2` explicitly multiplies
   by F32 scales. `legacy-v1` preserves the original equations and reduction order.
   Legacy replay of real layer 9 exactly reproduces both saved historical layer
   outputs, bit for bit; a small-model test also checks historical equations.
2. **Expert reduction order did not match official eager.** Pinned Transformers
   `469230357aab0f2b303b0d638c1f8d06edb14184`,
   `metadata/modeling_glm_moe_dsa.py:520`, iterates active experts in ascending ID
   using `expert_hit.nonzero()`, then adds shared experts. Production formerly
   reduced in router-slot order; the old reference added shared first. Both
   corrected paths reduce by ID and add shared last. Router IDs/weights remain
   paired; selection bias is still used only for selection. A cancellation
   fixture produces 4 under official order, 1 under former production order,
   and 3 under shared-first order. Chunked/batched paths must adopt the same ID
   order; retaining slot order is not bitwise equivalent.
3. **First trained arithmetic divergence:** same-input layer-0 replay first
   differs at input RMSNorm (max absolute `4.7683716e-7`, normalized L2
   `9.0768771e-7`). This is already before compressed/expanded attention diverge
   structurally. Under the *inherited layer-output budget*, the first internal
   violation is indexer query projection (two elements, max `3.0517578e-5`).
   This is diagnostic; that budget has not been independently calibrated for
   every internal operation. Layer 0's output still passes its layer budget.
4. **Layer-9 localization:** replay from the historical expanded layer-8 outputs
   first differs at RMSNorm; the first inherited-budget failure is shared-expert
   down-projection output (41 elements, max `1.2207031e-4`, normalized L2
   `2.3041479e-6`). Routing IDs agree exactly; routed expert weighted reductions
   pass. Replaying the shared down projection with *identical reference
   activations* passes (Vx vs NumPy batch max `4.3869019e-5`, normalized L2
   `6.5587999e-7`). Upstream rounding amplified by this projection contributes
   to the failure; the evidence does not identify a standalone indexing bug.
5. **Actual official eager execution now exists.** The optional fixture executes
   unmodified, hash-checked Transformers model/config source, CPU Torch
   2.14.1+cpu, NumPy 2.2.6, eager F32. Two deterministic random four-layer models
   (4 and 6 positions, index top-k 2) pass for production and independent expanded
   paths against both official batch and cached decode. It checks layer outputs,
   full logits including frozen top-k/TV/margin rules, and exact causal selection
   membership. Official top-k may return masked future slots in early batch rows;
   only causal membership affects attention. This is small-model evidence, not
   trained-checkpoint or long-context validation.

The existing frozen expanded receipt still fails at `p0.layer.9.output`, element
136 (max `1.0681152e-4`, normalized L2 `1.6521142e-6`). The new F32 reference is a
new numerical mode and needs independently reviewed provenance and reference
traces. F32 reassociation, BLAS batch/row reductions, compressed MLA association,
and sigmoid/normalization implementation can differ without an algebraic bug.
A new numerical contract must explicitly specify those choices; the candidate
must not revise the old budget to pass. Native GGML packed-dot activation and
cache numerics remain a separate target, not a replacement F32 oracle. No full
78-layer rerun, native-GGML alignment fix, or optimization promotion is claimed.

## Trace contract

Both CPU paths emit owning copies at real execution boundaries:

- Layer inputs, input/post-attention norms, attention residual and layer outputs.
- Q-A/Q normalization/Q-B/KV-A, latent and rotary cache entries, selected positions.
- Indexer query/key projections, normalized/rotated keys and queries, head weights,
  causal scores; shared layers reuse the preceding selection explicitly.
- Per-head scaled attention scores and softmax probabilities, attended head
  outputs, output projection.
- Dense/shared/selected-expert gate, up and SwiGLU activation; routing logits,
  IDs, weights, aggregation IDs; each expert output, weighted output and partial
  sum; routed sum, shared output and final MLP output.

IDs in traces use int64. Floating dtypes are preserved so inadvertent promotion
is visible. Fused backends remain fused: only observable outputs are recorded,
with missing internal stages reported by the comparator. Tracing never silently
executes an unfused alternative. Observer exceptions retain normal request-cache
rollback. Low-level `forward_layer` replay leaves cache consistency/rollback to
its caller and is not a serving entry point.

`stage_replay` reports first bitwise difference separately from first numeric or
exact-discrete violation, preserves candidate execution order, and reports
missing stages. Internal budgets are inherited unchanged solely to localize
failure. It uses config/hash pins and previously verified checkpoint identity;
it does not rehash 200+ GB of checkpoint files or certify their bytes anew.

## Reproduction on this host

Run from the parity checkout. Use fresh output directories to preserve evidence.
The shared environment remains unchanged; official dependencies are isolated in
`build/official-venv` (1.1 GB). Trace/replay artifacts are about 25 MB; no model
weights were copied. Set a single BLAS thread and avoid concurrent trained IO.

```sh
source ../vx-toolchain/env.sh
export PYTHONPATH="$PWD"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export GLM_VX_LIBRARY=/srv/nanocodex/workspace/glm-vx/kernels/build/libglm_vx.so
parity_python=/srv/nanocodex/workspace/glm-vx/.venv/bin/python
parity_base=/srv/nanocodex/workspace/glm-vx/build/parity
parity_gguf=/srv/nanocodex/workspace/models/glm-5.3-iq1s
parity_library=/srv/nanocodex/workspace/glm-vx/build/perf-audit/libglm_vx_o3.so

# Execute separately for layer 0 and layer 9.
"$parity_python" -m validation.stage_replay \
  --gguf "$parity_gguf" --input-trace "$parity_base/expanded-hello-let/arrays.npz" \
  --selection-trace "$parity_base/vx-o3-hello-let-v2/arrays.npz" \
  --contract "$parity_base/expanded-gate/contract.json" --layer 9 \
  --library "$parity_library" --output build/parity/stage-layer9-pinned

"$parity_python" -m validation.linear_replay \
  --gguf "$parity_gguf" --trace build/parity/stage-layer9-pinned/reference.npz \
  --keys p0.layer.9.mlp.shared.activation p1.layer.9.mlp.shared.activation \
  --weight model.layers.9.mlp.shared_experts.down_proj.weight \
  --contract "$parity_base/expanded-gate/contract.json" --library "$parity_library" \
  --output build/parity/linear-layer9-pinned

build/official-venv/bin/python -m validation.official_eager_fixture \
  --contract "$parity_base/expanded-gate/contract.json" --library "$parity_library" \
  --output build/parity/official-eager-final

"$parity_python" -m pytest tests/test_stage_replay.py tests/test_model.py \
  tests/test_model_trace.py tests/test_dsa_boundaries.py tests/test_trace_gate.py \
  tests/test_backend_oracle.py tests/test_integration.py -q
```

The targeted suite returned 90 passed / 1 skipped (22.06 s); the skipped legacy
native-model test checks a hardcoded worktree library path and ignored the
configured environment. Linking that ignored build path to the existing main
library and running `pytest tests/test_model.py -k compiled_vx_model -q` returned
1 passed / 10 deselected. Thus all 91 targeted tests were exercised successfully.

Official optional environment setup used `python3 -m venv build/official-venv`;
Debian's missing ensurepip left a valid interpreter but no pip. Installation
succeeded with the existing environment's pip `--python` option, without sudo:

```sh
/srv/nanocodex/workspace/glm-vx/.venv/bin/pip \
  --python build/official-venv/bin/python install torch \
  --index-url https://download.pytorch.org/whl/cpu
/srv/nanocodex/workspace/glm-vx/.venv/bin/pip \
  --python build/official-venv/bin/python install 'numpy==2.2.6' \
  'https://github.com/huggingface/transformers/archive/469230357aab0f2b303b0d638c1f8d06edb14184.tar.gz'
```
