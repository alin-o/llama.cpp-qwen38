#pragma once

#include "ggml.h"

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
    return n_q_tokens > n_sequences && tensor_cores_available && head_dim == 128 && q_type == GGML_TYPE_F32 &&
        q_contiguous && q_aligned && n_q_tiles > 0 && n_q_tiles <= 65535 && k_native && v_native;
}
