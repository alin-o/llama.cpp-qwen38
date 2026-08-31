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

enum ggml_paged_attn_context_bucket {
    GGML_PAGED_ATTN_CONTEXT_4K = 0,
    GGML_PAGED_ATTN_CONTEXT_16K,
    GGML_PAGED_ATTN_CONTEXT_32K,
    GGML_PAGED_ATTN_CONTEXT_64K,
    GGML_PAGED_ATTN_CONTEXT_128K,
    GGML_PAGED_ATTN_CONTEXT_LONG,
};

struct ggml_paged_attn_cuda_variant {
    int32_t n_warps;
    int32_t n_partitions;
    int32_t n_q_heads;
};

struct ggml_paged_attn_cuda_device_caps {
    int32_t cc;
    int32_t n_sms;
    int32_t warp_size;
    int32_t max_threads_per_block;
    int64_t shared_mem_per_block;
    bool    sm89_compiled;
    bool    graph_capture_safe;
};

static inline bool ggml_paged_attn_select_cuda_prefill_row_cache(
        int context_bucket, int head_dim, int n_heads, int n_heads_kv, int n_sequences, int n_q_tokens,
        enum ggml_type k_type, enum ggml_type v_type,
        struct ggml_paged_attn_cuda_device_caps caps) {
    // The shared row-address cache is benchmarked only for this native Ada (SM 8.9) prefill bucket.
    const bool supported_device = caps.cc == 890 && caps.n_sms > 0 && caps.warp_size == 32 &&
        caps.max_threads_per_block >= 1024 && caps.shared_mem_per_block >= 44 * 1024 &&
        caps.sm89_compiled && caps.graph_capture_safe;
    if (!supported_device || context_bucket != GGML_PAGED_ATTN_CONTEXT_4K || head_dim != 256 ||
            n_heads <= 0 || n_heads_kv <= 0 || n_heads % n_heads_kv != 0 ||
            n_heads / n_heads_kv != 8 || n_sequences != 1 || n_q_tokens != 1023) {
        return false;
    }

    return (k_type == GGML_TYPE_Q8_0 && v_type == GGML_TYPE_Q8_0) ||
        (k_type == GGML_TYPE_TURBO3_0 && v_type == GGML_TYPE_TURBO3_0) ||
        (k_type == GGML_TYPE_TURBO4_0 && v_type == GGML_TYPE_TURBO4_0);
}

static inline int ggml_paged_attn_context_bucket(int max_context_len) {
    if (max_context_len <= 4096) {
        return GGML_PAGED_ATTN_CONTEXT_4K;
    }
    if (max_context_len <= 16384) {
        return GGML_PAGED_ATTN_CONTEXT_16K;
    }
    if (max_context_len <= 32768) {
        return GGML_PAGED_ATTN_CONTEXT_32K;
    }
    if (max_context_len <= 65536) {
        return GGML_PAGED_ATTN_CONTEXT_64K;
    }
    if (max_context_len <= 131072) {
        return GGML_PAGED_ATTN_CONTEXT_128K;
    }
    return GGML_PAGED_ATTN_CONTEXT_LONG;
}

static inline int ggml_paged_attn_bucket_upper_bound(int bucket) {
    static const int bounds[] = { 4096, 16384, 32768, 65536, 131072, INT32_MAX };
    return bounds[bucket];
}

#if defined(__CUDACC__)
static __host__ __device__ inline int ggml_paged_attn_tail_partition(int context_len, int n_partitions) {
#else
static inline int ggml_paged_attn_tail_partition(int context_len, int n_partitions) {
#endif
    int tokens_per_partition{};
    for (int covered = 0; covered < context_len; covered += n_partitions) {
        ++tokens_per_partition;
    }
    return context_len > 0 ? (context_len - 1) / tokens_per_partition : 0;
}

#if defined(__CUDACC__)
static __host__ __device__ inline bool ggml_paged_attn_current_row_owner(
#else
static inline bool ggml_paged_attn_current_row_owner(
#endif
        int q_head, int n_heads, int n_heads_kv, int partition, int n_partitions, int context_len) {
    const int gqa_ratio = n_heads / n_heads_kv;
    return q_head % gqa_ratio == 0 && partition == ggml_paged_attn_tail_partition(context_len, n_partitions);
}

static inline struct ggml_paged_attn_cuda_variant ggml_paged_attn_select_cuda_variant(
        int context_bucket, int head_dim, int n_heads, int n_heads_kv, int n_sequences, int n_q_tokens,
        enum ggml_type k_type, enum ggml_type v_type,
        struct ggml_paged_attn_cuda_device_caps caps) {
    struct ggml_paged_attn_cuda_variant result = { 32, 1, 1 };
    // The bucket policy below is benchmarked only for native Ada (SM 8.9). Keep the legacy kernel elsewhere.
    const bool supported_device = caps.cc == 890 && caps.n_sms > 0 && caps.warp_size == 32 &&
        caps.max_threads_per_block >= 1024 && caps.shared_mem_per_block >= 44 * 1024 &&
        caps.sm89_compiled && caps.graph_capture_safe;
    if (!supported_device || context_bucket < GGML_PAGED_ATTN_CONTEXT_4K ||
            context_bucket > GGML_PAGED_ATTN_CONTEXT_LONG || head_dim != 256 ||
            n_heads <= 0 || n_heads_kv <= 0 || n_heads % n_heads_kv != 0 ||
            n_sequences <= 0 || n_q_tokens != n_sequences) {
        return result;
    }

    const int gqa_ratio = n_heads / n_heads_kv;
    if (k_type == GGML_TYPE_Q8_0 && v_type == GGML_TYPE_Q8_0) {
        if (context_bucket == GGML_PAGED_ATTN_CONTEXT_4K || (n_sequences >= 4 && gqa_ratio != 6)) {
            return result;
        }

        result.n_warps = 8;
        result.n_partitions = 8;
        if (n_sequences >= 4 && (n_heads / 2) * n_sequences * result.n_partitions >= caps.n_sms) {
            result.n_q_heads = 2;
        }
        return result;
    }

    const bool measured_turbo_shape = context_bucket <= GGML_PAGED_ATTN_CONTEXT_16K &&
        (gqa_ratio == 6 || gqa_ratio == 8) &&
        (n_sequences == 1 || n_sequences == 4 || n_sequences == 8);
    if (measured_turbo_shape && k_type == GGML_TYPE_TURBO3_0 && v_type == GGML_TYPE_TURBO3_0) {
        result.n_warps = 16;
    } else if (measured_turbo_shape && k_type == GGML_TYPE_TURBO4_0 && v_type == GGML_TYPE_TURBO4_0) {
        result.n_warps = 8;
    }
    return result;
}

#if defined(GGML_USE_CUDA)
#ifdef __cplusplus
extern "C" {
#endif
unsigned long long ggml_paged_attn_tiled_prefill_launch_count(void);
void ggml_paged_attn_tiled_prefill_launch_count_reset(void);
unsigned long long ggml_paged_attn_q8_decode_launch_count(void);
void ggml_paged_attn_q8_decode_launch_count_reset(void);
unsigned long long ggml_paged_attn_turbo_decode_launch_count(void);
void ggml_paged_attn_turbo_decode_launch_count_reset(void);
unsigned long long ggml_paged_attn_q8_combined_write_launch_count(void);
void ggml_paged_attn_q8_combined_write_launch_count_reset(void);
unsigned long long ggml_paged_attn_q8_fused_write_launch_count(void);
void ggml_paged_attn_q8_fused_write_launch_count_reset(void);
#ifdef __cplusplus
}
#endif
#endif

#if defined(GGML_USE_CUDA)
bool ggml_paged_attn_cuda_runtime_test(void);
#endif
