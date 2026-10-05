# Fused DSA index scores

`GPUBackend.index_scores(q, keys, weights, scale=None)` implements the existing
GLM model hook with one Vx kernel launch. The caller passes `q[heads,dim]`, causal
eligible `keys[tokens,dim]`, and signed `weights[heads]` already multiplied by
`heads**-0.5`. The result is `sum_h weights[h] * relu(dot(q[h], keys[t]) * scale)`
for each token. The default scale is `dim**-0.5`; explicit finite signed scales
and zero are accepted. There is no second head normalization, causal masking,
softmax, or top-k in this entry.

A warp owns one token and reduces each head's dot product across its 32 lanes.
Lane zero accumulates weighted ReLU outputs in head order. Four token warps per
128-thread block handle arbitrary token/dimension tails. Inactive whole warps
return before the full-mask shuffles. The kernel requires no scratch, shared
memory or atomics and never materializes the head-by-token score matrix.
Dimension products, rounded dimension loops and launch strides must fit int32.

The compatibility backend uploads each of the three inputs once, launches once,
and downloads only the final score vector. Mutable KV is never cached. For H
heads the former model fallback submitted H+1 matvec launches, 2H+2 uploads
(with mutable inputs) and H+1 downloads. Instrumented CPU SIMT wrapper tests
check both sequences, byte counts for the fused call, and cleanup after upload,
launch, readback and synchronization failures. These are transfer/launch-count
observations, **not GPU performance measurements**.

`GLMKernels.index_scores` also accepts resident DeviceTensors and an optional
reusable output. It enqueues one kernel without uploads, readback or synchronization.
The caller owns stream ordering and causal key selection. Empty token lists
produce an empty output without a launch. The compatibility wrapper additionally
avoids allocations/transfers for empty keys. Nonempty heads/dimensions and
matching shapes are required even for an empty key list. The host-array wrapper
rejects nonfinite inputs and scales that become nonfinite in float32. Resident
callers must enforce finite device inputs/intermediates themselves; the runtime
does not download device values to validate them. Finite inputs that overflow
intermediate f32 arithmetic are outside the kernel's supported value contract.

## Verification receipts (2026-10-04)

From the independent worktree based on `3ae29e5`, using the existing sibling
virtualenv and toolchain, with no global toolchain changes:

```sh
source ../vx-toolchain/env.sh
gpu/testing/build_simt.sh
../glm-vx/.venv/bin/python gpu/toolchain/build_ptx.py \
  --source gpu/kernels.vx --output gpu/build/kernels.ptx --keep-ir \
  --ptxas ../glm-vx/gpu/toolchain/cuda-tools/nvidia/cuda_nvcc/bin/ptxas
../glm-vx/.venv/bin/python gpu/toolchain/build_ptx.py \
  --source gpu/kernels.vx --output gpu/build/index-kernels-sm90.ptx --arch sm_90 \
  --ptxas ../glm-vx/gpu/toolchain/cuda-tools/nvidia/cuda_nvcc/bin/ptxas
../glm-vx/.venv/bin/python -m pytest gpu/test_index_scores.py \
  gpu/test_index_scores_hardware.py gpu/test_simt.py gpu/test_runtime.py \
  gpu/test_backend.py gpu/test_resident_mlp.py gpu/toolchain/test_build_ptx.py -q -rs
```

- [SM80 manifest](evidence/index-scores-sm80.json) and
  [assembler report](evidence/index-scores-sm80-assembly.txt).
- [SM90 manifest](evidence/index-scores-sm90.json) and
  [assembler report](evidence/index-scores-sm90-assembly.txt).
- [Focused test output](evidence/index-scores-tests.txt).

The installed compiler reports Vx 0.0.2; LLVM is 22.1.8 and ptxas is 12.9.86.
Manifests include the exact source/PTX/cubin hashes and keep
`gpu_execution_validated:false`. Both targets assemble all twelve main entries. The new scoring entry uses 27
registers on SM80 and 32 on SM90, with zero stack/spills and no shared memory
or barriers. The focused suite reports **135 passed, 4 skipped**: three new
index-score CUDA cases and the existing MLP CUDA case skipped because
`libcuda.so.1` is unavailable.
CPU SIMT executes the actual compiled Vx source and checks exact ordered warp
f32 arithmetic against an independent test reference, as well as scalar ordered
f32 and f64 references with fixed `rtol=3e-5, atol=3e-6`. Cases include signed
weights, explicit negative/zero scales, 64 heads, dimension/warp tails, guarded
outputs, zero tokens, nonfinite rejection and dimension/overflow validation.
Existing tiny-model tests verify that the hook is used for causal prefixes and
preserve selected indices against the NumPy model reference. Fake CUDA tests
verify resident ABI, output reuse, no host transfers, stream choice and overlap
validation; these tests do not execute PTX.

The real CUDA tests skip only when a supported device/driver is unavailable;
missing PTX, JIT, launch and numerical errors on a supported device fail. No GPU
execution, scheduling/race validation, full-model trained parity, speedup, or
end-to-end resident serving is established here. This is the fused scoring phase
only: indexer projections, key normalization/RoPE/cache gathering, top-k and model
orchestration still have existing host boundaries. It is not full graph capture
or a replacement for SGLang's absorb-BMM/attention work. No hardware was rented,
no benchmarks were run, and root CPU/model files were not changed.

The integrated repository also fingerprints all CPU-SIMT numerical sources, the
C++ lane harness, its build script and the linked binary. Re-run
`gpu/testing/build_simt.sh` after source changes; a stale or missing build manifest
fails before loading the library. This prevents a previous compiled kernel from
masquerading as current-source validation. See the [integrated audit](../docs/ENGINE-AUDIT-20261004.md)
for the full suite and trained CPU regression results.
