#ifndef GLM_VX_GPU_KERNELS_H
#define GLM_VX_GPU_KERNELS_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* NVIDIA device entrypoint ABI, not callable host library functions.
 * Pass arguments in this order to a CUDA Driver API launch of the PTX entry.
 * All arrays are contiguous row-major DEVICE allocations; arithmetic is f32.
 * Index products, rounded loop bounds, and launch strides must fit int32.
 * 1D grid/block only: y=z=1. Positive block/grid dimensions are mandatory.
 * Caller validates allocation sizes, finite inputs/intermediates, scalar ranges,
 * stream dependencies, and overlap. Invalid scalar shapes generally no-op;
 * kernels return no status. There are no device allocations or atomics.
 * Block reductions need block.x % 32 == 0, 32 <= block.x <= 1024, and
 * 33 shared f32 slots (132 bytes) supplied by the compiler's shared intrinsic.
 * Do not also allocate this static shared memory as dynamic launch memory.
 */

/* grid.x >= rows; recommended block.x=256. x/residual/residual_out/out:[rows,dim],
 * weight:[dim], rows>0, dim>0, eps>0. residual_out=x+residual, then
 * out=residual_out*rsqrt(mean(residual_out^2)+eps)*weight.
 * Exact alias residual_out==x or residual is allowed; out may alias x/residual
 * but must be distinct from residual_out to retain the residual result.
 * No partial overlap or overlap with weight. */
void glm_vx_gpu_residual_rmsnorm(float *out, float *residual_out,
    const float *x, const float *residual, const float *weight,
    int32_t rows, int32_t dim, float eps);

/* n>=0; any positive 1D launch. Grid-strided, suggested block.x=256.
 * Exact out==gate or out==up allowed; no partial overlap. */
void glm_vx_gpu_swiglu(float *out, const float *gate, const float *up, int32_t n);

/* x/out:[heads,dim], cos/sin:[dim/2], heads>0, even dim>0.
 * GLM adjacent-pair input becomes packed even-half/odd-half output.
 * NO output/input overlap: this operation permutes elements.
 * Any positive 1D launch, suggested block.x=256, ceil(heads*dim/2/256) CTAs. */
void glm_vx_gpu_rope_glm(float *out, const float *x, const float *cosine,
    const float *sine, int32_t heads, int32_t dim);

/* out:[rows], weight:[rows,cols], x:[cols]; rows>0, cols>=0.
 * block.x multiple32, grid.x>=ceil(rows/(block.x/32)); suggested block.x=128.
 * Each warp reduces one row. cols=0 writes zeros. No output/input overlap. */
void glm_vx_gpu_matvec(float *out, const float *weight, const float *x,
    int32_t rows, int32_t cols);

/* Absorbed compressed MLA, one decode query per head:
 * q_latent:[heads,rank], q_rope:[heads,rope_dim],
 * cache_latent:[tokens,rank], cache_rope:[tokens,rope_dim].
 * score[t]=scale*(dot(q_latent[h],cache_latent[t])+dot(q_rope[h],cache_rope[t])).
 * The compressed value is cache_latent[t]; no expanded K/V is materialized.
 * selected[h*selection_stride+j] gives eligible token IDs. selection_stride=0
 * shares a list between heads; otherwise selection_stride>=count. Caller must
 * enforce causality. Negative or >=tokens IDs are masked; duplicates retain
 * repeated probability mass. count=0/tokens=0/all masked produce zero output.
 * numerator:[heads,splits,rank]; stats:[heads,splits,2] holds (max,sum_exp).
 * grid.x>=heads*splits, block.x multiple32 and >=rank, rank<=1024.
 * Recommended block.x=32*ceil(rank/32). heads/rank/splits>0, count/rope_dim/
 * tokens>=0. One CTA scans indices s,s+splits,... for its head/split.
 * Every output/scratch allocation must be distinct from all other buffers.
 * Execute merge afterward on the SAME stream, or use an explicit dependency.
 */
void glm_vx_gpu_mla_partial(float *numerator, float *stats,
    const float *q_latent, const float *q_rope, const float *cache_latent,
    const float *cache_rope, const int32_t *selected,
    int32_t heads, int32_t rank, int32_t rope_dim, int32_t tokens,
    int32_t count, int32_t selection_stride, int32_t splits, float scale);

/* out:[heads,rank]. grid.x>=heads; any positive block.x, suggested 256.
 * Stable merge of partial states; empty/all-masked heads yield zero.
 * No output/input overlap. heads/rank/splits must match partial launch. */
void glm_vx_gpu_mla_merge(float *out, const float *numerator, const float *stats,
    int32_t heads, int32_t rank, int32_t splits);
/* Standalone row RMSNorm: out/x:[rows,dim], weight:[dim]; rows/dim>0, eps>0.
 * grid.x>=rows, block.x multiple32 in [32,1024], suggested256.
 * Exact out==x allowed; no partial overlap or output/weight overlap. */
void glm_vx_gpu_rmsnorm_rows(float *out, const float *x, const float *weight,
    int32_t rows, int32_t dim, float eps);

/* Stable row softmax: out/x:[rows,dim], rows/dim>0; grid=rows/block256 suggested.
 * Block.x multiple32 in [32,1024]. Exact out==x allowed.
 * -infinity masks accepted when at least one element is finite;
 * NaN, +infinity and all-masked inputs are invalid. */
void glm_vx_gpu_softmax_rows(float *out, const float *x, int32_t rows, int32_t dim);

/* One-warp stable GLM router: grid.x=1, block.x=32 REQUIRED.
 * logits/bias:[experts], indices/weights:[k]; 1<=experts<=256,
 * 1<=k<=min(experts,8). Descending sigmoid(logits)+bias selects experts;
 * ties select the smaller expert ID. Output weights use original sigmoid
 * normalized by (sum_selected_sigmoid + 1e-20), multiplied by scale.
 * Finite logits/bias/scores/scale required. No output/input overlap.
 * Only selected IDs/weights are outputs; logits remain device-resident. */
void glm_vx_gpu_router(int32_t *indices, float *weights,
    const float *logits, const float *bias, int32_t experts, int32_t k,
    float scale);
/* Stable original single-vector ABIs. Equivalent to the corresponding _rows
 * kernel with rows=1 and dim=n; grid1/block256 recommended. */
void glm_vx_gpu_rmsnorm(float *out, const float *x, const float *weight,
    int32_t n, float eps);
void glm_vx_gpu_softmax(float *out, const float *x, int32_t n);
#ifdef __cplusplus
}
#endif
#endif
