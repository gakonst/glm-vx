#ifndef GLM_VX_KERNELS_H
#define GLM_VX_KERNELS_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* All return 0 on success, -1 on invalid dimensions; callers validate buffers.
 * f32 pointers: contiguous row-major host storage. No device support implied.
 * n/dim products and indexing must fit signed int32. No null pointer checks.
 * Finite inputs required; softmax accepts -inf masked entries if one is finite.
 * In-place allowed: rmsnorm, layernorm, softmax, swiglu, add, axpy, rope.
 * Other output/scratch buffers must not overlap any inputs. */
int32_t glm_vx_rmsnorm(float *out, const float *x, const float *weight, int32_t n, float eps);
int32_t glm_vx_layernorm(float *out, const float *x, const float *weight, const float *bias, int32_t n, float eps);
int32_t glm_vx_softmax(float *out, const float *x, int32_t n);
int32_t glm_vx_swiglu(float *out, const float *gate, const float *up, int32_t n);
int32_t glm_vx_add(float *out, const float *a, const float *b, int32_t n);
int32_t glm_vx_axpy(float *out, const float *x, float alpha, int32_t n);
int32_t glm_vx_matvec(float *out, const float *weight, const float *x, int32_t rows, int32_t cols);
int32_t glm_vx_matmul_nt(float *out, const float *a, const float *b, int32_t m, int32_t n, int32_t k);
int32_t glm_vx_rope(float *out, const float *x, const float *cosine, const float *sine, int32_t heads, int32_t head_dim, int32_t rope_dim);
int32_t glm_vx_router(int32_t *indices, float *weights, const float *logits, const float *bias, int32_t experts, int32_t k, float scale);
int32_t glm_vx_topk(int32_t *indices, const float *scores, int32_t n, int32_t k);
int32_t glm_vx_attention(float *out, float *scratch, const float *q, const float *keys, const float *values, int32_t tokens, int32_t dim, int32_t value_dim, float scale);
int32_t glm_vx_sparse_attention(float *out, float *scratch, const float *q, const float *keys, const float *values, const int32_t *selected, int32_t count, int32_t tokens, int32_t dim, int32_t value_dim, float scale);
int32_t glm_vx_rope_offset(float *out, const float *x, const float *cosine, const float *sine, int32_t heads, int32_t head_dim, int32_t rope_dim, int32_t offset);
int32_t glm_vx_index_scores(float *out, const float *q, const float *keys, const float *weights, int32_t tokens, int32_t heads, int32_t dim, float qk_scale);
/* GLM packed output, out must not alias x. */
int32_t glm_vx_rope_glm(float *out, const float *x, const float *cosine, const float *sine, int32_t heads, int32_t dim);
#ifdef __cplusplus
}
#endif
#endif
