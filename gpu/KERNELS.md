# Parallel Vx GPU kernels

`kernels.vx` contains the numerical implementation; `kernels.h` defines the device
entrypoint argument order, buffer layouts, alias rules, and launch requirements.
The source targets portable NVIDIA sm80+ float32 through the adapter in
[`toolchain/`](toolchain/ABI.md). There is no CUDA C numerical replacement.
The adapter supplies thread indices, warp shuffles, block synchronization,
shared storage, and scalar math operations.

This is a parallel kernel prototype. Compilation and CPU emulation have been
checked; NVIDIA execution, race checking, latency, throughput, full-checkpoint
parity, and integration into the model's device execution path remain unverified.
No claim of optimal GPU performance is made.

## Entry points and launch geometry

All launches are one-dimensional (`grid.y=z=block.y=z=1`). These names identify
PTX entries loaded through a GPU driver; the C prototypes are an ABI description,
not a CPU-callable library interface. All pointers refer to device memory.

| Entry suffix (`glm_vx_gpu_`) | Work assignment | Suggested block.x | grid.x |
| --- | --- | --- | --- |
| `residual_rmsnorm` | One block per row, columns striped across threads | 256 | rows |
| `swiglu` | One element per thread per grid-stride iteration | 256 | ceil(n/256), or a smaller positive grid |
| `rope_glm` | One adjacent input pair per thread | 256 | ceil(heads*dim/2/256), or a smaller positive grid |
| `matvec` | One warp per output row, columns striped across lanes | 128 | ceil(rows/4) |
| `mla_partial` | One block per head and selected-token partition | 32*ceil(rank/32) | heads*splits |
| `mla_merge` | One block per head, output coordinates across threads | 256 | heads |
| `rmsnorm` | One vector, stable single-vector ABI | 256 | 1 |
| `rmsnorm_rows` | One row per block, columns striped across threads | 256 | rows |
| `softmax` | One vector, stable single-vector ABI | 256 | 1 |
| `softmax_rows` | One row per block, cooperative maximum and sum | 256 | rows |
| `router` | One warp selects up to 8 experts from at most 256 | **32 required** | 1 |

For empty elementwise work, skip the launch or use a positive grid: a zero-sized
CUDA launch is not valid. Reduction blocks must contain whole warps and have
32–1024 threads. `mla_partial` additionally requires `rank <= block.x`, limiting
this version to rank 1024. Launch products, buffer offsets, loop rounding and
strides must fit signed int32. The header specifies positive/nonnegative scalar
requirements. Invalid dimensions generally return without writing output and do
not report an error; the caller must validate them before launch.

The block-reduction helper uses 33 shared floats: up to 32 warp sums and a
separate broadcast slot. Lane zero writes each warp sum, a block barrier makes
those sums visible, the first warp reduces them, and a second barrier broadcasts
the result. Every lane participates in its warp shuffle. Masked dimensions feed
zero into the reduction. The adapter provides **static** shared storage (default 256 floats / 1024 bytes,
of which these kernels need 33 floats / 132 bytes); do not add a dynamic
shared-memory launch allocation for it.

## Numerical behavior

**Residual + RMSNorm.** A block writes `r = x + residual`, reduces `sum(r*r)`,
and writes `r * weight / sqrt(mean(r*r) + eps)`. Adjacent threads access adjacent
columns; each thread processes approximately `dim/block.x` columns. The residual
result is retained for the next residual connection. The two outputs must be
distinct; exact input/output aliases allowed by the header are safe.

**Standalone RMSNorm and softmax.** These batched-row forms support existing
model operators and exact in-place output. RMSNorm uses the same cooperative
sum-of-squares helper. Softmax first reduces the maximum, then computes
exponentials and reduces their sum. It accepts negative-infinity masks if at
least one score is finite. NaN, positive infinity, and entirely masked vectors
are outside its contract. Explicit `rmsnorm_rows` and `softmax_rows` entries take
`rows,dim` parameters and use one block per row. RMSNorm weights are shared across
rows. The original `rmsnorm` and `softmax` symbols retain their single-vector
`n` argument ABI and use the same internal helpers with `rows=1,dim=n`.

**Router.** A single full warp selects up to eight experts from at most 256.
Each lane scans at most eight expert logits; repeated warp argmax reductions
select descending `sigmoid(logit)+bias`, resolving ties by smaller expert ID.
Selected original sigmoid values are normalized by their sum plus `1e-20`, then
multiplied by the caller's scale. Selection bias does not change output weights.
Indices are exchanged as float32 shuffle values, which represent IDs up to 255
exactly. A block barrier after each selection makes its ID visible to all lanes
before excluding it from the next argmax. Only the selected IDs and weights need
to return to the host in a compatibility integration. E>256/K>8 require another
implementation and must be rejected by the caller.

**SwiGLU.** Computes `gate * sigmoid(gate) * up`. The sign-dependent sigmoid avoids
exponential overflow for large finite negative gates. Exact in-place output is
supported. Input and output loads/stores are contiguous across neighboring
threads.

**GLM RoPE.** Reads adjacent input pairs `(a,b)` and writes two contiguous halves:
`[a*cos-b*sin, b*cos+a*sin]` for each head. This matches the existing CPU
`glm_vx_rope_glm` convention, including its permutation. Coefficients are supplied
for the query position; frequency/position generation remains the caller's job.
Output must not alias input. This kernel operates on the rotary component; it
does not copy a separate nonrotary component. It is not the interleaved-output
RoPE convention used by some other models.

**Matvec.** Each warp cooperates on one contiguous row of `[rows,cols]` weights.
Lane `l` processes columns `l, l+32, ...`, then five shuffle-down operations
produce a lane-zero sum. No thread independently scans the full row. This is a
float32 SIMT decode matvec, without tensor cores or a GEMM claim.

**Sparse absorbed MLA.** The query's nonrotary key projection must already be
absorbed into `q_latent[heads,rank]`. Cache storage remains compressed:
`cache_latent[tokens,rank]` and `cache_rope[tokens,rope_dim]`. For selected token
`t`, the block reduces

```
score[t] = scale * (dot(q_latent[h], cache_latent[t])
                 + dot(q_rope[h], cache_rope[t]))
value[t] = cache_latent[t]
```

One thread owns one latent output coordinate. The rotary dot is striped across
threads and supports dimensions larger than the block. Each partition processes
selected-list positions `split, split+splits, ...`. Query dimensions are reduced
cooperatively; separate partitions expose parallelism across selected tokens.
Within a partition, online softmax maintains the maximum `m`, denominator `l`,
and vector numerator `a`:

```
m_next = max(m, score)
alpha = exp(m - m_next)
beta = exp(score - m_next)
a_next = alpha*a + beta*value
l_next = alpha*l + beta
```

The first valid token initializes `(m,l,a)=(score,1,value)`. An empty partition
stores zero numerator and `l=0`; its stored `m=0` is ignored during the merge.
The merge chooses the maximum among nonempty partitions, rescales their
numerators and denominators, and normalizes once. Scratch comprises
`heads*splits*rank` numerator floats and `heads*splits*2` statistics floats, with
no score/probability array or expanded key/value cache. The resulting compressed
context still requires the model's absorbed value/output projection.

`selection_stride=0` shares one selected list across heads; otherwise it must be
at least `count`, allowing per-head lists. Negative and out-of-range token IDs
are masked. Repeated IDs are counted repeatedly. An empty or fully masked list
returns zeros. The caller supplies causal eligibility: the kernels do not infer
positions or filter future valid token IDs. Use the same GPU stream for partial
and merge launches, or an explicit event dependency.

## Precision and limitations

Input values and resulting scores/sums must remain finite, except for the
explicit negative-infinity softmax masks. Float32 reductions
change accumulation order compared with serial CPU code. The PTX adapter lowers
`expf` to `ex2.approx.ftz.f32(x*log2(e))`; it is approximate and flushes exponential
subnormal results to zero. CPU libm emulation does not validate this GPU math
path. Small rounding differences and underflow behavior require device-side
numerical tolerances rather than bit equality.

Launch sizes above are initial choices, not benchmark results. MLA currently
uses two block barriers per valid selected token and computes scalar softmax
updates redundantly across threads. Splitting reduces the serial token chain at
the cost of scratch and a merge launch. Selection gathers can be irregular even
though each selected cache row is read contiguously. There is no tiling of tokens,
async staging, tensor-core path, quantization, expert dispatch, batched prefill,
paged cache mapping, or autotuning in these kernels.

## Validation performed

On 2026-10-04, the adapter compiled all six entries to sm80 PTX using Vx v0.0.2
and LLVM 22. Agent 1 inspected emitted hardware thread registers, `shfl.sync`,
`bar.sync`, shared storage, exponential and square-root instructions, and
confirmed no unresolved external calls. Build artifacts and compiler provenance
are produced by the toolchain's build command; see its documentation.

An additional **CPU-only** check compiled this exact Vx file with `vxc --action
emit-obj` and linked a hardware-emulation shim plus libm. The shim supplied
thread-local indices, real pthread barriers per warp/block, a 33-float shared
array, and shuffle exchange storage. It contained no replacement numerical
kernels. `LD_PRELOAD=shim.so` was necessary while compiling because this Vx
compiler resolves external symbols during object emission. A Python launcher
invoked each logical thread through ctypes. NumPy served only as the oracle.
Temporary emulator artifacts were removed; this was an ad-hoc validation, not a
committed reusable test suite.

All 25 comparisons passed at `rtol=2e-5, atol=2e-5` with RNG seed 918:

| Cases | Maximum observed absolute error |
| --- | ---: |
| SwiGLU lengths 0, 1, 257, 1007; gate standard deviation 15 | 3.81470e-6 |
| Packed GLM RoPE, 3 heads, dimensions 2, 64, 66 | 0 |
| Residual outputs, 2 rows, dimensions 1, 33, 257, 6144 | 0 |
| RMSNorm on those same shapes, epsilon 1e-5 | 9.53674e-7 |
| Matvec, 7 rows, columns 0, 1, 31, 33, 521 | 3.81470e-6 |
| Five sparse MLA configurations | 3.57628e-7 |

MLA configurations covered ranks 1/33/63, rotary dimensions 0/7/65, selected
counts 0/9, splits 1/3/12, shared/per-head selection, duplicate IDs, negative and
out-of-range masked IDs, and scales 0/0.5/25. More partitions than selected tokens
exercised empty-partition merging; count zero exercised all-empty output.
The tests used 64-thread blocks except 128-thread matvec blocks, including a
partially used final matvec block. GPU scheduling/races, 512/1024-rank MLA,
full model behavior, and hardware performance were not tested.

### Compatibility additions

The standalone `rmsnorm`, `softmax`, and `router` additions passed another 29
CPU-only pthread-emulation comparisons with RNG seed 919 and the same
`rtol=2e-5, atol=2e-5`. Together with the original checks, 54 numerical comparisons
passed. The second batch covered:

| Cases | Maximum observed absolute error |
| --- | ---: |
| RMSNorm lengths 1/33/257/6144, separate and exact in-place output | 9.53674e-7 |
| Softmax lengths 1/33/257/4097, negative-infinity masks, separate and in-place output | 1.19210e-7 |
| Router E=1/7/32/33/64/256, K=min(E,8), random and tied logits | 5.96047e-8 |
| Router E=256/K=8, all sigmoid probabilities underflowed to zero | 0 |

Router IDs matched the stable NumPy ordering exactly in all 13 configurations.
CPU libm and GPU approximate exponential may rank extremely close unequal scores
differently; the exact-tie rule applies to the scores computed by each path.

### Explicit batched-row forms

The optional `_rows` entrypoints preserve the original single-vector symbols'
argument counts. The batched helpers passed 16 further CPU-only comparisons:
3 rows, dimensions 1/33/257/6144, RMSNorm and masked softmax, separate and exact
in-place outputs. A fourth excess block verified row bounds guards. RNG seed was
920 and tolerances were again 2e-5; maximum absolute error was 9.53674e-7 for
RMSNorm and 5.96047e-8 for softmax. This brings the ad-hoc development comparisons
to 70, including checks made before extraction into shared internal helpers.

### Final compiler and assembler verification

The final 11-entry source SHA-256 is
`85463e38228f811ec2357d4eafb629eabf9ac74bfcdf081801d448979fd79e05`.
It was compiled through the Vx/LLVM adapter and NVIDIA CUDA 12.9.86 `ptxas`
for both sm80 and sm90. All 11 entries assembled successfully with zero register
spills and zero stack use. Reported register counts were 22–40 on sm80 and
24–32 on sm90. The two explicit `_rows` entries each used 30 registers on sm80,
32 on sm90, and 1024 bytes of static shared storage under the adapter's default
allocation. Twelve compiler regression tests passed, covering all 11 entry
names. Final provenance and assembler reports are under [evidence](evidence/).

These are compiler/assembler resource reports, not measured GPU performance or
proof of device-side numerical correctness. Runtime wrappers were independently
inspected to confirm original vector argument counts, separate `_rows` symbols,
and a 32-thread router launch. Mock-runtime tests validate launch contracts;
they do not execute on a GPU.
