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
/* Packed weights: immutable little-endian GGUF bytes; x stays f32.
 * half is the exact 65536-entry IEEE half->f32 table; table is the 19600-byte
 * pinned ggml_tables.bin. Caller checks buffer byte counts, block alignment,
 * finite data and signed-int32 index bounds. No weight matrix allocation.
 * Batch v2 kernels pack x as transposed [cols,tile] tiles (64..1), write
 * out[batch,rows], and
 * require batch in 0..64 plus batch*cols and batch*rows within int32 capacity.
 * Unpack exports are diagnostics only, not used by serving. */
#define GLM_VX_PACKED_DECL(kind) \
int32_t glm_vx_packed_##kind(float *out, const uint8_t *w, const float *x, const float *half, const uint8_t *table, int32_t rows, int32_t cols); \
int32_t glm_vx_packed_batch_v2_##kind(float *out, const uint8_t *w, const float *x, const float *half, const uint8_t *table, int32_t rows, int32_t cols, int32_t batch); \
int32_t glm_vx_unpack_##kind(float *out, const uint8_t *w, const float *half, const uint8_t *table, int32_t blocks);
GLM_VX_PACKED_DECL(f16)
GLM_VX_PACKED_DECL(q8_0)
GLM_VX_PACKED_DECL(q2_k)
GLM_VX_PACKED_DECL(q3_k)
GLM_VX_PACKED_DECL(q4_k)
GLM_VX_PACKED_DECL(q5_k)
GLM_VX_PACKED_DECL(q6_k)
GLM_VX_PACKED_DECL(iq1_s)
GLM_VX_PACKED_DECL(iq2_xxs)
GLM_VX_PACKED_DECL(iq3_xxs)
GLM_VX_PACKED_DECL(iq4_xs)
#undef GLM_VX_PACKED_DECL
#ifdef __cplusplus
}
#endif
#endif
