#pragma once

#include "ggml.h"

#if defined(__CUDACC__)
static __host__ __device__ inline int ggml_paged_attn_kv_head(int q_head, int n_heads, int n_heads_kv) {
    return q_head / (n_heads / n_heads_kv);
}
#else
static inline int ggml_paged_attn_kv_head(int q_head, int n_heads, int n_heads_kv) {
    return q_head / (n_heads / n_heads_kv);
}
#endif

static inline bool ggml_paged_attn_tiled_prefill_supported(
        int       head_dim,
        enum ggml_type q_type,
        bool      q_contiguous,
        bool      q_aligned,
        int       n_q_tokens,
        int       n_sequences,
        int       n_q_tiles,
        bool      tensor_cores_available,
        enum ggml_type k_type,
        enum ggml_type v_type) {
    const bool k_native = k_type == GGML_TYPE_Q8_0 || k_type == GGML_TYPE_TURBO3_0 || k_type == GGML_TYPE_TURBO4_0;
    const bool v_native = v_type == GGML_TYPE_Q8_0 || v_type == GGML_TYPE_TURBO3_0 || v_type == GGML_TYPE_TURBO4_0;
    const bool supported_head_dim = head_dim == 128 || head_dim == 256;
    return n_q_tokens > n_sequences && tensor_cores_available && supported_head_dim && q_type == GGML_TYPE_F32 &&
        q_contiguous && q_aligned && n_q_tiles > 0 && n_q_tiles <= 65535 && k_native && v_native;
}

#if defined(GGML_USE_CUDA)
#ifdef __cplusplus
extern "C" {
#endif
unsigned long long ggml_paged_attn_tiled_prefill_launch_count(void);
void ggml_paged_attn_tiled_prefill_launch_count_reset(void);
#ifdef __cplusplus
}
#endif
#endif
