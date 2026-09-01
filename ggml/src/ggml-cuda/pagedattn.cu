#include "pagedattn.cuh"

#include "cpy-utils.cuh"
#include "cp-async.cuh"
#include "fattn-common.cuh"
#include "ggml-paged-attn.h"
#include "set-rows.cuh"
#include "turbo-quant.cuh"
#include "turbo-wht.cuh"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static std::atomic<unsigned long long> g_paged_prefill_launch_count{ 0 };
static std::atomic<unsigned long long> g_paged_q8_decode_launch_count{ 0 };
static std::atomic<unsigned long long> g_paged_turbo_decode_launch_count{ 0 };
static std::atomic<unsigned long long> g_paged_turbo_token_vec_launch_count{ 0 };
static std::atomic<unsigned long long> g_paged_turbo_combined_write_launch_count{ 0 };
static std::atomic<unsigned long long> g_paged_q8_combined_write_launch_count{ 0 };
static std::atomic<unsigned long long> g_paged_q8_fused_write_launch_count{ 0 };

extern "C" unsigned long long ggml_paged_attn_tiled_prefill_launch_count(void) {
    return g_paged_prefill_launch_count.load(std::memory_order_relaxed);
}

extern "C" void ggml_paged_attn_tiled_prefill_launch_count_reset(void) {
    g_paged_prefill_launch_count.store(0, std::memory_order_relaxed);
}

extern "C" unsigned long long ggml_paged_attn_q8_decode_launch_count(void) {
    return g_paged_q8_decode_launch_count.load(std::memory_order_relaxed);
}

extern "C" void ggml_paged_attn_q8_decode_launch_count_reset(void) {
    g_paged_q8_decode_launch_count.store(0, std::memory_order_relaxed);
}

extern "C" unsigned long long ggml_paged_attn_turbo_decode_launch_count(void) {
    return g_paged_turbo_decode_launch_count.load(std::memory_order_relaxed);
}

extern "C" void ggml_paged_attn_turbo_decode_launch_count_reset(void) {
    g_paged_turbo_decode_launch_count.store(0, std::memory_order_relaxed);
}

extern "C" unsigned long long ggml_paged_attn_turbo_token_vec_launch_count(void) {
    return g_paged_turbo_token_vec_launch_count.load(std::memory_order_relaxed);
}

extern "C" void ggml_paged_attn_turbo_token_vec_launch_count_reset(void) {
    g_paged_turbo_token_vec_launch_count.store(0, std::memory_order_relaxed);
}

extern "C" unsigned long long ggml_paged_attn_turbo_combined_write_launch_count(void) {
    return g_paged_turbo_combined_write_launch_count.load(std::memory_order_relaxed);
}

extern "C" void ggml_paged_attn_turbo_combined_write_launch_count_reset(void) {
    g_paged_turbo_combined_write_launch_count.store(0, std::memory_order_relaxed);
}

extern "C" unsigned long long ggml_paged_attn_q8_combined_write_launch_count(void) {
    return g_paged_q8_combined_write_launch_count.load(std::memory_order_relaxed);
}

extern "C" void ggml_paged_attn_q8_combined_write_launch_count_reset(void) {
    g_paged_q8_combined_write_launch_count.store(0, std::memory_order_relaxed);
}

extern "C" unsigned long long ggml_paged_attn_q8_fused_write_launch_count(void) {
    return g_paged_q8_fused_write_launch_count.load(std::memory_order_relaxed);
}

extern "C" void ggml_paged_attn_q8_fused_write_launch_count_reset(void) {
    g_paged_q8_fused_write_launch_count.store(0, std::memory_order_relaxed);
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#if defined(TURING_MMA_AVAILABLE)
#include <mma.h>
namespace wmma = nvcuda::wmma;
#endif
#endif

static __device__ __forceinline__ float paged_cache_value(const char * row, int index, int type) {
    switch (type) {
        case GGML_TYPE_Q8_0: {
            const block_q8_0 * block = (const block_q8_0 *) row + index / QK8_0;
            return __half2float(block->d) * block->qs[index % QK8_0];
        }
        case GGML_TYPE_TURBO3_0: {
            const block_turbo3_0 * block = (const block_turbo3_0 *) row + index / QK_TURBO3;
            return turbo3_dequant(block, index % QK_TURBO3, __half2float(block->norm));
        }
        case GGML_TYPE_TURBO4_0: {
            const block_turbo4_0 * block = (const block_turbo4_0 *) row + index / QK_TURBO4;
            return turbo4_dequant(block, index % QK_TURBO4, __half2float(block->norm));
        }
        default:
            return 0.0f;
    }
}
static __device__ __forceinline__ float4 paged_cache_value4(const char * row, int index, int type) {
    switch (type) {
        case GGML_TYPE_Q8_0: {
            const block_q8_0 * block = (const block_q8_0 *) row + index / QK8_0;
            const int offset = index % QK8_0;
            const float d = __half2float(block->d);
            return make_float4(d * block->qs[offset + 0], d * block->qs[offset + 1],
                               d * block->qs[offset + 2], d * block->qs[offset + 3]);
        }
        case GGML_TYPE_TURBO3_0: {
            const block_turbo3_0 * block = (const block_turbo3_0 *) row + index / QK_TURBO3;
            const int offset = index % QK_TURBO3;
            const uint8_t qs = block->qs[offset / 4];
            const uint8_t signs = block->signs[offset / 8];
            const float norm = __half2float(block->norm);
            return make_float4(
                norm * TURBO3_CENTROIDS[((qs >> 0) & 3) | (((signs >> ((offset + 0) % 8)) & 1) << 2)],
                norm * TURBO3_CENTROIDS[((qs >> 2) & 3) | (((signs >> ((offset + 1) % 8)) & 1) << 2)],
                norm * TURBO3_CENTROIDS[((qs >> 4) & 3) | (((signs >> ((offset + 2) % 8)) & 1) << 2)],
                norm * TURBO3_CENTROIDS[((qs >> 6) & 3) | (((signs >> ((offset + 3) % 8)) & 1) << 2)]);
        }
        case GGML_TYPE_TURBO4_0: {
            const block_turbo4_0 * block = (const block_turbo4_0 *) row + index / QK_TURBO4;
            const uint16_t qs = *(const uint16_t *) (block->qs + index % QK_TURBO4 / 2);
            const float norm = __half2float(block->norm);
            return make_float4(
                norm * TURBO4_CENTROIDS[(qs >>  0) & 0xf], norm * TURBO4_CENTROIDS[(qs >>  4) & 0xf],
                norm * TURBO4_CENTROIDS[(qs >>  8) & 0xf], norm * TURBO4_CENTROIDS[(qs >> 12) & 0xf]);
        }
        default:
            return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

__global__ void paged_q8_cache_write_kernel(
        const float * k_new, const float * v_new, const int32_t * write_rows,
        block_q8_0 * k_cache, block_q8_0 * v_cache, int head_dim, int n_rows,
        size_t k_row_stride, size_t v_row_stride) {
    const int blocks_per_row = head_dim / QK8_0;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows * blocks_per_row) {
        return;
    }

    const int src_row = i / blocks_per_row;
    const int block = i % blocks_per_row;
    const int dst_row = write_rows[src_row];
    const float * k_src = k_new + (size_t) src_row * head_dim + block * QK8_0;
    const float * v_src = v_new + (size_t) src_row * head_dim + block * QK8_0;
    block_q8_0 * k_dst = (block_q8_0 *) ((char *) k_cache + (size_t) dst_row * k_row_stride) + block;
    block_q8_0 * v_dst = (block_q8_0 *) ((char *) v_cache + (size_t) dst_row * v_row_stride) + block;
    quantize_f32_q8_0_block(k_src, k_dst);
    quantize_f32_q8_0_block(v_src, v_dst);
}

static void paged_launch_q8_cache_write(
        cudaStream_t stream, const float * k_new, const float * v_new, const int32_t * write_rows,
        block_q8_0 * k_cache, block_q8_0 * v_cache, int head_dim, int64_t n_rows,
        size_t k_row_stride, size_t v_row_stride) {
    const int64_t n_blocks = n_rows * head_dim / QK8_0;
    constexpr int block_size_write = 256;
    paged_q8_cache_write_kernel<<<
        (n_blocks + block_size_write - 1) / block_size_write, block_size_write, 0, stream>>>(
        k_new, v_new, write_rows, k_cache, v_cache, head_dim, n_rows, k_row_stride, v_row_stride);
    CUDA_CHECK(cudaGetLastError());
}

static __device__ __forceinline__ void paged_store_half4(half * dst, float4 value) {
    *(half2 *) (dst + 0) = __floats2half2_rn(value.x, value.y);
    *(half2 *) (dst + 2) = __floats2half2_rn(value.z, value.w);
}

static __device__ __forceinline__ void paged_store_transposed_half4(half (* dst)[16], int col, int row, float4 value) {
    dst[col + 0][row] = __float2half_rn(value.x);
    dst[col + 1][row] = __float2half_rn(value.y);
    dst[col + 2][row] = __float2half_rn(value.z);
    dst[col + 3][row] = __float2half_rn(value.w);
}

static __device__ __forceinline__ float4 paged_load_float4(const float * src) {
    return *(const float4 *) src;
}

static __device__ __forceinline__ void paged_store_float4(float * dst, float4 value) {
    *(float4 *) dst = value;
}

static __device__ __forceinline__ float block_reduce_sum_full(float value, float * smem, int tid, int head_dim) {
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int n_warps = (head_dim + 31) >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) {
        smem[warp_id] = value;
    }
    __syncthreads();

    value = tid < n_warps ? smem[tid] : 0.0f;
    if (warp_id == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0) {
            smem[0] = value;
        }
    }
    __syncthreads();
    return smem[0];
}

__global__ void paged_attention_decode_kernel(
        const float * q,
        const char * k_cache,
        const char * v_cache,
        const int * block_table,
        const int * context_lens,
        const int * batch_offsets,
        const int * batch_lens,
        size_t k_stride_token,
        size_t k_stride_head,
        size_t k_stride_block,
        size_t v_stride_token,
        size_t v_stride_head,
        size_t v_stride_block,
        int n_heads_kv,
        int block_size,
        int max_blocks,
        int k_type,
        int v_type,
        float scale,
        bool skip_multi_token,
        float * out) {
    extern __shared__ float smem[];
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int n_heads = gridDim.x;
    const int head_dim = blockDim.x;
    const int kv_head_idx = ggml_paged_attn_kv_head(head_idx, n_heads, n_heads_kv);
    const int seq_start = batch_offsets[seq_idx];
    const int num_new_tokens = batch_lens[seq_idx];
    if (skip_multi_token && num_new_tokens > 1) {
        return;
    }

    for (int i = 0; i < num_new_tokens; ++i) {
        const int token_batch_idx = seq_start + i;
        const float q_value = q[(size_t) token_batch_idx * n_heads * head_dim + (size_t) head_idx * head_dim + tid] * scale;
        const int q_pos = context_lens[seq_idx] - num_new_tokens + i;
        float qk_max = -FLT_MAX;
        float exp_sum = 0.0f;
        float acc = 0.0f;

        for (int bid = 0; bid <= q_pos / block_size; ++bid) {
            const int physical_block = block_table[seq_idx * max_blocks + bid];
            const int start_token = bid * block_size;
            const int end_token = min(start_token + block_size, q_pos + 1);
            for (int token = start_token; token < end_token; ++token) {
                const int token_in_block = token % block_size;
                const char * k_row = k_cache + (size_t) physical_block * k_stride_block + (size_t) kv_head_idx * k_stride_head + (size_t) token_in_block * k_stride_token;
                const char * v_row = v_cache + (size_t) physical_block * v_stride_block + (size_t) kv_head_idx * v_stride_head + (size_t) token_in_block * v_stride_token;
                const float qk = block_reduce_sum_full(q_value * paged_cache_value(k_row, tid, k_type), smem, tid, head_dim);
                const float qk_max_new = fmaxf(qk_max, qk);
                const float exp_old = __expf(qk_max - qk_max_new);
                const float exp_new = __expf(qk - qk_max_new);
                exp_sum = exp_sum * exp_old + exp_new;
                acc = acc * exp_old + exp_new * paged_cache_value(v_row, tid, v_type);
                qk_max = qk_max_new;
            }
        }
        out[(size_t) token_batch_idx * n_heads * head_dim + (size_t) head_idx * head_dim + tid] = acc / (exp_sum + 1e-6f);
    }
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
template <int HEAD_DIM, int TILE_TOKENS>
__global__ void paged_attention_decode_gqa_kernel(
        const float * q, const char * k_cache, const char * v_cache,
        const int * block_table, const int * context_lens, const int * batch_offsets,
        size_t k_stride_token, size_t k_stride_head, size_t k_stride_block,
        size_t v_stride_token, size_t v_stride_head, size_t v_stride_block,
        int n_heads, int n_heads_kv, int block_size, int max_blocks, int k_type, int v_type,
        float scale, float * out) {
    extern __shared__ float tile[];
    float * k_tile = tile;
    float * v_tile = k_tile + TILE_TOKENS * HEAD_DIM;

    const int tid = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int kv_head = blockIdx.x;
    const int seq = blockIdx.y;
    const int heads_per_kv = n_heads / n_heads_kv;
    const bool active = warp < heads_per_kv;
    const int batch_start = batch_offsets[seq];
    const int q_pos = context_lens[seq] - 1;
    const int head = kv_head * heads_per_kv + warp;
    const size_t q_base = (size_t) batch_start * n_heads * HEAD_DIM + (size_t) head * HEAD_DIM;
    float qv[HEAD_DIM / WARP_SIZE] = {};
    float acc[HEAD_DIM / WARP_SIZE] = {};
#pragma unroll
    for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
        if (active) {
            qv[i] = q[q_base + lane + i * WARP_SIZE];
        }
    }

    float qk_max = -FLT_MAX;
    float exp_sum = 0.0f;
    const int limit = q_pos + 1;

    for (int token_start = 0; token_start < limit; token_start += TILE_TOKENS) {
        if (k_type == GGML_TYPE_Q8_0 && v_type == GGML_TYPE_Q8_0) {
            for (int vec = tid; vec < TILE_TOKENS * HEAD_DIM / 4; vec += blockDim.x) {
                const int key = vec / (HEAD_DIM / 4);
                const int dim = (vec % (HEAD_DIM / 4)) * 4;
                const int token = token_start + key;
                float4 k_values = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 v_values = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (token < limit) {
                    const int physical_block = block_table[seq * max_blocks + token / block_size];
                    const int token_in_block = token % block_size;
                    const char * k_row = k_cache + (size_t) physical_block * k_stride_block +
                        (size_t) kv_head * k_stride_head + (size_t) token_in_block * k_stride_token;
                    const char * v_row = v_cache + (size_t) physical_block * v_stride_block +
                        (size_t) kv_head * v_stride_head + (size_t) token_in_block * v_stride_token;
                    k_values = paged_cache_value4(k_row, dim, k_type);
                    v_values = paged_cache_value4(v_row, dim, v_type);
                }
                const int base = key * HEAD_DIM + dim;
                k_tile[base + 0] = k_values.x;
                k_tile[base + 1] = k_values.y;
                k_tile[base + 2] = k_values.z;
                k_tile[base + 3] = k_values.w;
                v_tile[base + 0] = v_values.x;
                v_tile[base + 1] = v_values.y;
                v_tile[base + 2] = v_values.z;
                v_tile[base + 3] = v_values.w;
            }
        } else {
            for (int i = tid; i < TILE_TOKENS * HEAD_DIM; i += blockDim.x) {
                const int token = token_start + i / HEAD_DIM;
                const int dim = i % HEAD_DIM;
                float k_value = 0.0f;
                float v_value = 0.0f;
                if (token < limit) {
                    const int physical_block = block_table[seq * max_blocks + token / block_size];
                    const int token_in_block = token % block_size;
                    const char * k_row = k_cache + (size_t) physical_block * k_stride_block +
                        (size_t) kv_head * k_stride_head + (size_t) token_in_block * k_stride_token;
                    const char * v_row = v_cache + (size_t) physical_block * v_stride_block +
                        (size_t) kv_head * v_stride_head + (size_t) token_in_block * v_stride_token;
                    k_value = paged_cache_value(k_row, dim, k_type);
                    v_value = paged_cache_value(v_row, dim, v_type);
                }
                k_tile[i] = k_value;
                v_tile[i] = v_value;
            }
        }
        __syncthreads();

        for (int key = 0; key < TILE_TOKENS && token_start + key < limit; ++key) {
            float qk = 0.0f;
#pragma unroll
            for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
                qk += qv[i] * k_tile[key * HEAD_DIM + lane + i * WARP_SIZE];
            }
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                qk += __shfl_down_sync(0xffffffffu, qk, offset);
            }
            qk = __shfl_sync(0xffffffffu, qk, 0) * scale;
            const float new_max = fmaxf(qk_max, qk);
            const float old_scale = __expf(qk_max - new_max);
            const float weight = __expf(qk - new_max);
            exp_sum = exp_sum * old_scale + weight;
#pragma unroll
            for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
                acc[i] = acc[i] * old_scale + weight * v_tile[key * HEAD_DIM + lane + i * WARP_SIZE];
            }
            qk_max = new_max;
        }
        __syncthreads();
    }

    if (active) {
        const float inv_sum = 1.0f / (exp_sum + 1e-6f);
#pragma unroll
        for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
            out[q_base + lane + i * WARP_SIZE] = acc[i] * inv_sum;
        }
    }
}

static __device__ __forceinline__ void paged_online_softmax_scales(
        float value, float current_max, float & new_max, float & old_scale, float & value_scale) {
    const bool value_is_new_max = value > current_max;
    new_max = value_is_new_max ? value : current_max;
    old_scale = value_is_new_max ? __expf(current_max - value) : 1.0f;
    value_scale = value >= current_max ? 1.0f : __expf(value - current_max);
}

template <int HEAD_DIM>
static __device__ __forceinline__ float paged_q8_k_dot(
        const char * k_row, const int * shared_q_i32, const float2 * shared_q_ds) {
    static_assert(HEAD_DIM % (sizeof(int) * WARP_SIZE) == 0, "head dimension must map evenly to q8 lane fragments");
    int q_i32[HEAD_DIM / (sizeof(int) * WARP_SIZE)];
    float2 q_ds[HEAD_DIM / (sizeof(int) * WARP_SIZE)];
#pragma unroll
    for (int i0 = 0; i0 < HEAD_DIM / (int) sizeof(int); i0 += WARP_SIZE) {
        const int i = i0 + threadIdx.x;
        q_i32[i0 / WARP_SIZE] = shared_q_i32[i];
        q_ds[i0 / WARP_SIZE] = shared_q_ds[i / QI8_1];
    }
    return vec_dot_fattn_vec_KQ_q8_0<HEAD_DIM, WARP_SIZE>(k_row, nullptr, q_i32, q_ds);
}

static __device__ __forceinline__ void paged_q8_v_load4(
        const char * v_row, int dim, float * values) {
    dequantize_V_q8_0<float, 4>(v_row, values, dim);
}

template <int HEAD_DIM>
static __device__ __forceinline__ void paged_q8_quantize_current_rows(
        const float * k_new, const float * v_new, int src_row,
        block_q8_0 * current_k, block_q8_0 * current_v, int tid, int n_threads) {
    constexpr int ROW_BLOCKS = HEAD_DIM / QK8_0;
    for (int block = tid; block < ROW_BLOCKS; block += n_threads) {
        const size_t offset = (size_t) src_row * HEAD_DIM + block * QK8_0;
        quantize_f32_q8_0_block(k_new + offset, current_k + block);
        quantize_f32_q8_0_block(v_new + offset, current_v + block);
    }
}

template <int HEAD_DIM>
static __device__ __forceinline__ void paged_q8_persist_current_rows(
        const block_q8_0 * current_k, const block_q8_0 * current_v, const int * write_rows,
        int src_row, char * k_cache, char * v_cache, size_t k_row_stride, size_t v_row_stride,
        int tid, int n_threads) {
    constexpr int ROW_BLOCKS = HEAD_DIM / QK8_0;
    const int dst_row = write_rows[src_row];
    block_q8_0 * k_dst = (block_q8_0 *) (k_cache + (size_t) dst_row * k_row_stride);
    block_q8_0 * v_dst = (block_q8_0 *) (v_cache + (size_t) dst_row * v_row_stride);
    for (int block = tid; block < ROW_BLOCKS; block += n_threads) {
        k_dst[block] = current_k[block];
        v_dst[block] = current_v[block];
    }
}

template <int HEAD_DIM, bool ENABLED>
struct paged_q8_current_rows {
};

template <int HEAD_DIM>
struct paged_q8_current_rows<HEAD_DIM, true> {
    block_q8_0 k[HEAD_DIM / QK8_0];
    block_q8_0 v[HEAD_DIM / QK8_0];
};

// Warps process disjoint context positions, then warp 0 merges their online-softmax partials.
template <int HEAD_DIM, int N_WARPS, bool PACKED_QV, bool FUSE_CURRENT>
__global__ __launch_bounds__(WARP_SIZE * N_WARPS, 1) void paged_attention_decode_q8_parallel_kernel(
        const float * q, const float * k_new, const float * v_new, char * k_cache, char * v_cache,
        const int * block_table, const int * write_rows, const int * context_lens, const int * batch_offsets,
        size_t k_stride_token, size_t k_stride_head, size_t k_stride_block,
        size_t v_stride_token, size_t v_stride_head, size_t v_stride_block,
        int n_heads, int n_heads_kv, int block_size, int max_blocks, float scale, float * out) {
    __shared__ float partial_max[N_WARPS];
    __shared__ float partial_sum[N_WARPS];
    __shared__ float partial_acc[N_WARPS][HEAD_DIM];
    __shared__ int q_i32[HEAD_DIM / sizeof(int)];
    __shared__ float2 q_ds[HEAD_DIM / QK8_1];
    __shared__ paged_q8_current_rows<HEAD_DIM, FUSE_CURRENT> current;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int head = blockIdx.x;
    const int seq = blockIdx.y;
    const int kv_head = ggml_paged_attn_kv_head(head, n_heads, n_heads_kv);
    const int batch_start = batch_offsets[seq];
    const int limit = context_lens[seq];
    const int current_src_row = batch_start * n_heads_kv + kv_head;
    const size_t q_base = (size_t) batch_start * n_heads * HEAD_DIM + (size_t) head * HEAD_DIM;
    float qv[HEAD_DIM / WARP_SIZE];
    float acc[HEAD_DIM / WARP_SIZE] = {};
    if constexpr (PACKED_QV) {
        if (warp == 0) {
#pragma unroll
            for (int i0 = 0; i0 < HEAD_DIM / (int) sizeof(int); i0 += WARP_SIZE) {
                quantize_q8_1_to_shared<float2, WARP_SIZE>(
                    q + q_base + i0 * sizeof(int), scale, q_i32 + i0, q_ds + i0 / QI8_1);
            }
        }
        __syncthreads();
    } else {
#pragma unroll
        for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
            qv[i] = q[q_base + lane + i * WARP_SIZE];
        }
    }

    if constexpr (FUSE_CURRENT) {
        const int tid = warp * WARP_SIZE + lane;
        paged_q8_quantize_current_rows<HEAD_DIM>(
            k_new, v_new, current_src_row, current.k, current.v, tid, N_WARPS * WARP_SIZE);
        __syncthreads();
        if (ggml_paged_attn_current_row_owner(head, n_heads, n_heads_kv, 0, 1, limit)) {
            paged_q8_persist_current_rows<HEAD_DIM>(
                current.k, current.v, write_rows, current_src_row, k_cache, v_cache,
                k_stride_token, v_stride_token, tid, N_WARPS * WARP_SIZE);
        }
        __syncthreads();
    }

    float qk_max = -FLT_MAX;
    float exp_sum = 0.0f;
    const int cache_limit = FUSE_CURRENT ? limit - 1 : limit;
    for (int token = warp; token < cache_limit; token += N_WARPS) {
        const int physical_block = block_table[seq * max_blocks + token / block_size];
        const int token_in_block = token % block_size;
        const char * k_row = k_cache + (size_t) physical_block * k_stride_block +
            (size_t) kv_head * k_stride_head + (size_t) token_in_block * k_stride_token;
        const char * v_row = v_cache + (size_t) physical_block * v_stride_block +
            (size_t) kv_head * v_stride_head + (size_t) token_in_block * v_stride_token;
        float qk = 0.0f;
        if constexpr (PACKED_QV) {
            qk = paged_q8_k_dot<HEAD_DIM>(k_row, q_i32, q_ds);
        } else {
#pragma unroll
            for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
                qk += qv[i] * paged_cache_value(k_row, lane + i * WARP_SIZE, GGML_TYPE_Q8_0);
            }
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            qk += __shfl_down_sync(0xffffffffu, qk, offset);
        }
        qk = __shfl_sync(0xffffffffu, qk, 0) * (PACKED_QV ? 1.0f : scale);
        float new_max = qk_max;
        float old_scale = 1.0f;
        float weight = 0.0f;
        paged_online_softmax_scales(qk, qk_max, new_max, old_scale, weight);
        if constexpr (PACKED_QV) {
#pragma unroll
            for (int group = 0; group < HEAD_DIM / (4 * WARP_SIZE); ++group) {
                float values[4];
                const int dim = lane * 4 + group * 4 * WARP_SIZE;
                paged_q8_v_load4(v_row, dim, values);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int ai = group * 4 + i;
                    acc[ai] = acc[ai] * old_scale + weight * values[i];
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
                acc[i] = acc[i] * old_scale +
                    weight * paged_cache_value(v_row, lane + i * WARP_SIZE, GGML_TYPE_Q8_0);
            }
        }
        exp_sum = exp_sum * old_scale + weight;
        qk_max = new_max;
    }

    if constexpr (FUSE_CURRENT) {
        if (warp == (limit - 1) % N_WARPS) {
            const char * k_row = (const char *) current.k;
            const char * v_row = (const char *) current.v;
            float qk = 0.0f;
            if constexpr (PACKED_QV) {
                qk = paged_q8_k_dot<HEAD_DIM>(k_row, q_i32, q_ds);
            } else {
#pragma unroll
                for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
                    qk += qv[i] * paged_cache_value(k_row, lane + i * WARP_SIZE, GGML_TYPE_Q8_0);
                }
            }
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                qk += __shfl_down_sync(0xffffffffu, qk, offset);
            }
            qk = __shfl_sync(0xffffffffu, qk, 0) * (PACKED_QV ? 1.0f : scale);
            float new_max;
            float old_scale;
            float weight;
            paged_online_softmax_scales(qk, qk_max, new_max, old_scale, weight);
            if constexpr (PACKED_QV) {
#pragma unroll
                for (int group = 0; group < HEAD_DIM / (4 * WARP_SIZE); ++group) {
                    float values[4];
                    paged_q8_v_load4(v_row, lane * 4 + group * 4 * WARP_SIZE, values);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        acc[group * 4 + i] = acc[group * 4 + i] * old_scale + weight * values[i];
                    }
                }
            } else {
#pragma unroll
                for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
                    acc[i] = acc[i] * old_scale +
                        weight * paged_cache_value(v_row, lane + i * WARP_SIZE, GGML_TYPE_Q8_0);
                }
            }
            exp_sum = exp_sum * old_scale + weight;
            qk_max = new_max;
        }
    }

    if (lane == 0) {
        partial_max[warp] = qk_max;
        partial_sum[warp] = exp_sum;
    }
#pragma unroll
    for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
        const int dim = PACKED_QV ? lane * 4 + (i / 4) * 4 * WARP_SIZE + i % 4 : lane + i * WARP_SIZE;
        partial_acc[warp][dim] = acc[i];
    }
    __syncthreads();

    if (warp == 0) {
        float final_max = -FLT_MAX;
        float final_sum = 0.0f;
        float final_acc[HEAD_DIM / WARP_SIZE] = {};
#pragma unroll
        for (int w = 0; w < N_WARPS; ++w) {
            final_max = fmaxf(final_max, partial_max[w]);
        }
#pragma unroll
        for (int w = 0; w < N_WARPS; ++w) {
            const float partial_scale = partial_max[w] == final_max ?
                1.0f : __expf(partial_max[w] - final_max);
            final_sum += partial_sum[w] * partial_scale;
#pragma unroll
            for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
                const int dim = PACKED_QV ? lane * 4 + (i / 4) * 4 * WARP_SIZE + i % 4 : lane + i * WARP_SIZE;
                final_acc[i] += partial_acc[w][dim] * partial_scale;
            }
        }
        const float inv_sum = 1.0f / (final_sum + 1e-6f);
#pragma unroll
        for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
            const int dim = PACKED_QV ? lane * 4 + (i / 4) * 4 * WARP_SIZE + i % 4 : lane + i * WARP_SIZE;
            out[q_base + dim] = final_acc[i] * inv_sum;
        }
    }
}

template <int HEAD_DIM>
static __device__ __forceinline__ size_t paged_partial_offset(
        int seq, int head, int partition, int n_heads, int n_partitions) {
    return (((size_t) seq * n_heads + head) * n_partitions + partition) * (HEAD_DIM + 2);
}

template <int HEAD_DIM, int N_WARPS, int N_PARTITIONS, bool PACKED_QV, bool FUSE_CURRENT>
__global__ __launch_bounds__(WARP_SIZE * N_WARPS, 1) void paged_attention_decode_q8_partial_kernel(
        const float * q, const float * k_new, const float * v_new, char * k_cache, char * v_cache,
        const int * block_table, const int * write_rows, const int * context_lens, const int * batch_offsets,
        size_t k_stride_token, size_t k_stride_head, size_t k_stride_block,
        size_t v_stride_token, size_t v_stride_head, size_t v_stride_block,
        int n_heads, int n_heads_kv, int block_size, int max_blocks, float scale, float * partials) {
    __shared__ float warp_max[N_WARPS];
    __shared__ float warp_sum[N_WARPS];
    __shared__ float warp_acc[N_WARPS][HEAD_DIM];
    __shared__ int q_i32[HEAD_DIM / sizeof(int)];
    __shared__ float2 q_ds[HEAD_DIM / QK8_1];
    __shared__ paged_q8_current_rows<HEAD_DIM, FUSE_CURRENT> current;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int head = blockIdx.x;
    const int seq = blockIdx.y;
    const int partition = blockIdx.z;
    const int kv_head = ggml_paged_attn_kv_head(head, n_heads, n_heads_kv);
    const int limit = context_lens[seq];
    const int partition_size = (limit + N_PARTITIONS - 1) / N_PARTITIONS;
    const int token_begin = partition * partition_size;
    const int token_limit = token_begin + partition_size < limit ? token_begin + partition_size : limit;
    const bool is_tail_partition = partition == ggml_paged_attn_tail_partition(limit, N_PARTITIONS);
    const int current_src_row = batch_offsets[seq] * n_heads_kv + kv_head;
    const size_t q_base = (size_t) batch_offsets[seq] * n_heads * HEAD_DIM + (size_t) head * HEAD_DIM;
    float qv[HEAD_DIM / WARP_SIZE];
    float acc[HEAD_DIM / WARP_SIZE] = {};

    if constexpr (PACKED_QV) {
        if (warp == 0) {
#pragma unroll
            for (int i0 = 0; i0 < HEAD_DIM / (int) sizeof(int); i0 += WARP_SIZE) {
                quantize_q8_1_to_shared<float2, WARP_SIZE>(
                    q + q_base + i0 * sizeof(int), scale, q_i32 + i0, q_ds + i0 / QI8_1);
            }
        }
        __syncthreads();
    } else {
#pragma unroll
        for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
            qv[i] = q[q_base + lane + i * WARP_SIZE];
        }
    }

    if constexpr (FUSE_CURRENT) {
        if (is_tail_partition) {
            const int tid = warp * WARP_SIZE + lane;
            paged_q8_quantize_current_rows<HEAD_DIM>(
                k_new, v_new, current_src_row, current.k, current.v, tid, N_WARPS * WARP_SIZE);
            __syncthreads();
            if (ggml_paged_attn_current_row_owner(
                    head, n_heads, n_heads_kv, partition, N_PARTITIONS, limit)) {
                paged_q8_persist_current_rows<HEAD_DIM>(
                    current.k, current.v, write_rows, current_src_row, k_cache, v_cache,
                    k_stride_token, v_stride_token, tid, N_WARPS * WARP_SIZE);
            }
            __syncthreads();
        }
    }

    float qk_max = -FLT_MAX;
    float exp_sum = 0.0f;
    const int cache_token_limit = FUSE_CURRENT && is_tail_partition ? token_limit - 1 : token_limit;
    for (int token = token_begin + warp; token < cache_token_limit; token += N_WARPS) {
        const int physical_block = block_table[seq * max_blocks + token / block_size];
        const int token_in_block = token % block_size;
        const char * k_row = k_cache + (size_t) physical_block * k_stride_block +
            (size_t) kv_head * k_stride_head + (size_t) token_in_block * k_stride_token;
        const char * v_row = v_cache + (size_t) physical_block * v_stride_block +
            (size_t) kv_head * v_stride_head + (size_t) token_in_block * v_stride_token;
        float qk = 0.0f;
        if constexpr (PACKED_QV) {
            qk = paged_q8_k_dot<HEAD_DIM>(k_row, q_i32, q_ds);
        } else {
#pragma unroll
            for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
                qk += qv[i] * paged_cache_value(k_row, lane + i * WARP_SIZE, GGML_TYPE_Q8_0);
            }
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            qk += __shfl_down_sync(0xffffffffu, qk, offset);
        }
        qk = __shfl_sync(0xffffffffu, qk, 0) * (PACKED_QV ? 1.0f : scale);
        float new_max;
        float old_scale;
        float weight;
        paged_online_softmax_scales(qk, qk_max, new_max, old_scale, weight);
        if constexpr (PACKED_QV) {
#pragma unroll
            for (int group = 0; group < HEAD_DIM / (4 * WARP_SIZE); ++group) {
                float values[4];
                paged_q8_v_load4(v_row, lane * 4 + group * 4 * WARP_SIZE, values);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    acc[group * 4 + i] = acc[group * 4 + i] * old_scale + weight * values[i];
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
                acc[i] = acc[i] * old_scale +
                    weight * paged_cache_value(v_row, lane + i * WARP_SIZE, GGML_TYPE_Q8_0);
            }
        }
        exp_sum = exp_sum * old_scale + weight;
        qk_max = new_max;
    }

    if constexpr (FUSE_CURRENT) {
        const int current_warp = (limit - 1 - token_begin) % N_WARPS;
        if (is_tail_partition && warp == current_warp) {
            const char * k_row = (const char *) current.k;
            const char * v_row = (const char *) current.v;
            float qk = 0.0f;
            if constexpr (PACKED_QV) {
                qk = paged_q8_k_dot<HEAD_DIM>(k_row, q_i32, q_ds);
            } else {
#pragma unroll
                for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
                    qk += qv[i] * paged_cache_value(k_row, lane + i * WARP_SIZE, GGML_TYPE_Q8_0);
                }
            }
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                qk += __shfl_down_sync(0xffffffffu, qk, offset);
            }
            qk = __shfl_sync(0xffffffffu, qk, 0) * (PACKED_QV ? 1.0f : scale);
            float new_max;
            float old_scale;
            float weight;
            paged_online_softmax_scales(qk, qk_max, new_max, old_scale, weight);
            if constexpr (PACKED_QV) {
#pragma unroll
                for (int group = 0; group < HEAD_DIM / (4 * WARP_SIZE); ++group) {
                    float values[4];
                    paged_q8_v_load4(v_row, lane * 4 + group * 4 * WARP_SIZE, values);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        acc[group * 4 + i] = acc[group * 4 + i] * old_scale + weight * values[i];
                    }
                }
            } else {
#pragma unroll
                for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
                    acc[i] = acc[i] * old_scale +
                        weight * paged_cache_value(v_row, lane + i * WARP_SIZE, GGML_TYPE_Q8_0);
                }
            }
            exp_sum = exp_sum * old_scale + weight;
            qk_max = new_max;
        }
    }

    if (lane == 0) {
        warp_max[warp] = qk_max;
        warp_sum[warp] = exp_sum;
    }
#pragma unroll
    for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
        const int dim = PACKED_QV ? lane * 4 + (i / 4) * 4 * WARP_SIZE + i % 4 : lane + i * WARP_SIZE;
        warp_acc[warp][dim] = acc[i];
    }
    __syncthreads();

    if (warp == 0) {
        float final_max = -FLT_MAX;
        float final_sum = 0.0f;
        float final_acc[HEAD_DIM / WARP_SIZE] = {};
#pragma unroll
        for (int w = 0; w < N_WARPS; ++w) {
            final_max = fmaxf(final_max, warp_max[w]);
        }
#pragma unroll
        for (int w = 0; w < N_WARPS; ++w) {
            const float partial_scale = warp_max[w] == final_max ? 1.0f : __expf(warp_max[w] - final_max);
            final_sum += warp_sum[w] * partial_scale;
#pragma unroll
            for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
                const int dim = PACKED_QV ? lane * 4 + (i / 4) * 4 * WARP_SIZE + i % 4 : lane + i * WARP_SIZE;
                final_acc[i] += warp_acc[w][dim] * partial_scale;
            }
        }
        float * partial = partials + paged_partial_offset<HEAD_DIM>(seq, head, partition, n_heads, N_PARTITIONS);
        if (lane == 0) {
            partial[0] = final_max;
            partial[1] = final_sum;
        }
#pragma unroll
        for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
            const int dim = PACKED_QV ? lane * 4 + (i / 4) * 4 * WARP_SIZE + i % 4 : lane + i * WARP_SIZE;
            partial[2 + dim] = final_acc[i];
        }
    }
}

template <int HEAD_DIM, int N_WARPS, int N_PARTITIONS, int N_Q_HEADS, int TILE_TOKENS,
          bool USE_CP_ASYNC, bool FUSE_CURRENT>
__global__ __launch_bounds__(WARP_SIZE * N_WARPS, 1) void paged_attention_decode_q8_grouped_partial_kernel(
        const float * q, const float * k_new, const float * v_new, char * k_cache, char * v_cache,
        const int * block_table, const int * write_rows, const int * context_lens, const int * batch_offsets,
        size_t k_stride_token, size_t k_stride_head, size_t k_stride_block,
        size_t v_stride_token, size_t v_stride_head, size_t v_stride_block,
        int n_heads, int n_heads_kv, int block_size, int max_blocks, float scale, float * partials) {
    constexpr int ROW_BLOCKS = HEAD_DIM / QK8_0;
    constexpr int ROW_WORDS = sizeof(block_q8_0) * ROW_BLOCKS / sizeof(int);
    constexpr int ROW_BYTES = ROW_WORDS * sizeof(int);
    constexpr int ROW_CHUNKS = ROW_BYTES / 16;
    constexpr int N_BUFFERS = USE_CP_ASYNC ? 2 : 1;
    constexpr int CONTEXT_WARPS = N_WARPS / N_Q_HEADS;
    static_assert(N_WARPS % N_Q_HEADS == 0, "warps must divide evenly between grouped heads");
    static_assert(TILE_TOKENS % CONTEXT_WARPS == 0, "tile tokens must divide evenly between context warps");
    static_assert(ROW_BYTES % 16 == 0, "q8 rows must divide into 16-byte asynchronous copies");

    __shared__ int q_i32[N_Q_HEADS][HEAD_DIM / sizeof(int)];
    __shared__ float2 q_ds[N_Q_HEADS][HEAD_DIM / QK8_1];
    __shared__ __align__(16) int k_tile[N_BUFFERS][TILE_TOKENS][ROW_WORDS];
    __shared__ __align__(16) int v_tile[N_BUFFERS][TILE_TOKENS][ROW_WORDS];
    __shared__ int physical_blocks[N_BUFFERS][TILE_TOKENS];
    __shared__ int tokens_in_block[N_BUFFERS][TILE_TOKENS];
    __shared__ float warp_max[N_Q_HEADS][CONTEXT_WARPS];
    __shared__ float warp_sum[N_Q_HEADS][CONTEXT_WARPS];
    __shared__ float warp_acc[N_Q_HEADS][CONTEXT_WARPS][HEAD_DIM];
    __shared__ paged_q8_current_rows<HEAD_DIM, FUSE_CURRENT> current;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * WARP_SIZE + lane;
    const int head_in_group = warp / CONTEXT_WARPS;
    const int context_warp = warp % CONTEXT_WARPS;
    const int head = blockIdx.x * N_Q_HEADS + head_in_group;
    const int group_head = blockIdx.x * N_Q_HEADS;
    const int seq = blockIdx.y;
    const int partition = blockIdx.z;
    const int kv_head = ggml_paged_attn_kv_head(head, n_heads, n_heads_kv);
    const int limit = context_lens[seq];
    const int partition_size = (limit + N_PARTITIONS - 1) / N_PARTITIONS;
    const int token_begin = partition * partition_size;
    const int token_limit = token_begin + partition_size < limit ? token_begin + partition_size : limit;
    const bool is_tail_partition = partition == ggml_paged_attn_tail_partition(limit, N_PARTITIONS);
    const int current_src_row = batch_offsets[seq] * n_heads_kv + kv_head;
    const int cache_token_limit = FUSE_CURRENT && is_tail_partition ? token_limit - 1 : token_limit;
    const size_t q_base = (size_t) batch_offsets[seq] * n_heads * HEAD_DIM + (size_t) head * HEAD_DIM;
    float acc[HEAD_DIM / WARP_SIZE] = {};

    if (context_warp == 0) {
#pragma unroll
        for (int i0 = 0; i0 < HEAD_DIM / (int) sizeof(int); i0 += WARP_SIZE) {
            quantize_q8_1_to_shared<float2, WARP_SIZE>(
                q + q_base + i0 * sizeof(int), scale,
                q_i32[head_in_group] + i0, q_ds[head_in_group] + i0 / QI8_1);
        }
    }
    __syncthreads();

    if constexpr (FUSE_CURRENT) {
        if (is_tail_partition) {
            paged_q8_quantize_current_rows<HEAD_DIM>(
                k_new, v_new, current_src_row, current.k, current.v, tid, N_WARPS * WARP_SIZE);
            __syncthreads();
            if (ggml_paged_attn_current_row_owner(
                    group_head, n_heads, n_heads_kv, partition, N_PARTITIONS, limit)) {
                paged_q8_persist_current_rows<HEAD_DIM>(
                    current.k, current.v, write_rows, current_src_row, k_cache, v_cache,
                    k_stride_token, v_stride_token, tid, N_WARPS * WARP_SIZE);
            }
            __syncthreads();
        }
    }

    float qk_max = -FLT_MAX;
    float exp_sum = 0.0f;
    if constexpr (USE_CP_ASYNC) {
        if (tid < TILE_TOKENS) {
            const int token = token_begin + tid;
            if (token < cache_token_limit) {
                physical_blocks[0][tid] = block_table[seq * max_blocks + token / block_size];
                tokens_in_block[0][tid] = token % block_size;
            }
        }
        __syncthreads();
        for (int item = tid; item < 2 * TILE_TOKENS * ROW_CHUNKS; item += N_WARPS * WARP_SIZE) {
            const int tile_token = (item / ROW_CHUNKS) % TILE_TOKENS;
            const int chunk = item % ROW_CHUNKS;
            const bool load_v = item >= TILE_TOKENS * ROW_CHUNKS;
            if (token_begin + tile_token < cache_token_limit) {
                const char * cache = load_v ? v_cache : k_cache;
                const size_t stride_block = load_v ? v_stride_block : k_stride_block;
                const size_t stride_head = load_v ? v_stride_head : k_stride_head;
                const size_t stride_token = load_v ? v_stride_token : k_stride_token;
                const char * src = cache + (size_t) physical_blocks[0][tile_token] * stride_block +
                    (size_t) kv_head * stride_head + (size_t) tokens_in_block[0][tile_token] * stride_token + chunk * 16;
                char * dst = (char *) (load_v ? v_tile[0][tile_token] : k_tile[0][tile_token]) + chunk * 16;
                cp_async_cg_16<256>(ggml_cuda_cvta_generic_to_shared(dst), src);
            }
        }
        cp_async_wait_all();
        __syncthreads();
    }
    for (int token_start = token_begin; token_start < cache_token_limit; token_start += TILE_TOKENS) {
        const int tile_index = (token_start - token_begin) / TILE_TOKENS;
        const int buffer = tile_index % N_BUFFERS;
        if constexpr (!USE_CP_ASYNC) {
            if (tid < TILE_TOKENS) {
                const int token = token_start + tid;
                if (token < cache_token_limit) {
                    physical_blocks[buffer][tid] = block_table[seq * max_blocks + token / block_size];
                    tokens_in_block[buffer][tid] = token % block_size;
                }
            }
            __syncthreads();

            for (int item = tid; item < TILE_TOKENS * ROW_WORDS; item += N_WARPS * WARP_SIZE) {
                const int tile_token = item / ROW_WORDS;
                const int word = item % ROW_WORDS;
                if (token_start + tile_token < cache_token_limit) {
                    const char * k_row = k_cache + (size_t) physical_blocks[buffer][tile_token] * k_stride_block +
                        (size_t) kv_head * k_stride_head + (size_t) tokens_in_block[buffer][tile_token] * k_stride_token;
                    const char * v_row = v_cache + (size_t) physical_blocks[buffer][tile_token] * v_stride_block +
                        (size_t) kv_head * v_stride_head + (size_t) tokens_in_block[buffer][tile_token] * v_stride_token;
                    k_tile[buffer][tile_token][word] = ((const int *) k_row)[word];
                    v_tile[buffer][tile_token][word] = ((const int *) v_row)[word];
                }
            }
            __syncthreads();
        } else if (token_start + TILE_TOKENS < cache_token_limit) {
            const int next_buffer = (tile_index + 1) % N_BUFFERS;
            const int next_start = token_start + TILE_TOKENS;
            if (tid < TILE_TOKENS) {
                const int token = next_start + tid;
                if (token < cache_token_limit) {
                    physical_blocks[next_buffer][tid] = block_table[seq * max_blocks + token / block_size];
                    tokens_in_block[next_buffer][tid] = token % block_size;
                }
            }
            __syncthreads();
            for (int item = tid; item < 2 * TILE_TOKENS * ROW_CHUNKS; item += N_WARPS * WARP_SIZE) {
                const int tile_token = (item / ROW_CHUNKS) % TILE_TOKENS;
                const int chunk = item % ROW_CHUNKS;
                const bool load_v = item >= TILE_TOKENS * ROW_CHUNKS;
                if (next_start + tile_token < cache_token_limit) {
                    const char * cache = load_v ? v_cache : k_cache;
                    const size_t stride_block = load_v ? v_stride_block : k_stride_block;
                    const size_t stride_head = load_v ? v_stride_head : k_stride_head;
                    const size_t stride_token = load_v ? v_stride_token : k_stride_token;
                    const char * src = cache + (size_t) physical_blocks[next_buffer][tile_token] * stride_block +
                        (size_t) kv_head * stride_head +
                        (size_t) tokens_in_block[next_buffer][tile_token] * stride_token + chunk * 16;
                    char * dst = (char *) (load_v ? v_tile[next_buffer][tile_token] :
                        k_tile[next_buffer][tile_token]) + chunk * 16;
                    cp_async_cg_16<256>(ggml_cuda_cvta_generic_to_shared(dst), src);
                }
            }
        }

#pragma unroll
        for (int i = 0; i < TILE_TOKENS / CONTEXT_WARPS; ++i) {
            const int tile_token = context_warp + i * CONTEXT_WARPS;
            if (token_start + tile_token < cache_token_limit) {
                const char * k_row = (const char *) k_tile[buffer][tile_token];
                const char * v_row = (const char *) v_tile[buffer][tile_token];
                float qk = paged_q8_k_dot<HEAD_DIM>(k_row, q_i32[head_in_group], q_ds[head_in_group]);
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                    qk += __shfl_down_sync(0xffffffffu, qk, offset);
                }
                qk = __shfl_sync(0xffffffffu, qk, 0);
                float new_max;
                float old_scale;
                float weight;
                paged_online_softmax_scales(qk, qk_max, new_max, old_scale, weight);
#pragma unroll
                for (int group = 0; group < HEAD_DIM / (4 * WARP_SIZE); ++group) {
                    float values[4];
                    paged_q8_v_load4(v_row, lane * 4 + group * 4 * WARP_SIZE, values);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        acc[group * 4 + j] = acc[group * 4 + j] * old_scale + weight * values[j];
                    }
                }
                exp_sum = exp_sum * old_scale + weight;
                qk_max = new_max;
            }
        }
        if constexpr (USE_CP_ASYNC) {
            cp_async_wait_all();
        }
        __syncthreads();
    }

    if constexpr (FUSE_CURRENT) {
        if (is_tail_partition && context_warp == 0) {
            const char * k_row = (const char *) current.k;
            const char * v_row = (const char *) current.v;
            float qk = paged_q8_k_dot<HEAD_DIM>(k_row, q_i32[head_in_group], q_ds[head_in_group]);
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                qk += __shfl_down_sync(0xffffffffu, qk, offset);
            }
            qk = __shfl_sync(0xffffffffu, qk, 0);
            float new_max;
            float old_scale;
            float weight;
            paged_online_softmax_scales(qk, qk_max, new_max, old_scale, weight);
#pragma unroll
            for (int group = 0; group < HEAD_DIM / (4 * WARP_SIZE); ++group) {
                float values[4];
                paged_q8_v_load4(v_row, lane * 4 + group * 4 * WARP_SIZE, values);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[group * 4 + j] = acc[group * 4 + j] * old_scale + weight * values[j];
                }
            }
            exp_sum = exp_sum * old_scale + weight;
            qk_max = new_max;
        }
    }

    if (lane == 0) {
        warp_max[head_in_group][context_warp] = qk_max;
        warp_sum[head_in_group][context_warp] = exp_sum;
    }
#pragma unroll
    for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
        const int dim = lane * 4 + (i / 4) * 4 * WARP_SIZE + i % 4;
        warp_acc[head_in_group][context_warp][dim] = acc[i];
    }
    __syncthreads();

    if (context_warp == 0) {
        float final_max = -FLT_MAX;
        float final_sum = 0.0f;
        float final_acc[HEAD_DIM / WARP_SIZE] = {};
#pragma unroll
        for (int w = 0; w < CONTEXT_WARPS; ++w) {
            final_max = fmaxf(final_max, warp_max[head_in_group][w]);
        }
#pragma unroll
        for (int w = 0; w < CONTEXT_WARPS; ++w) {
            const float partial_scale = warp_max[head_in_group][w] == final_max ?
                1.0f : __expf(warp_max[head_in_group][w] - final_max);
            final_sum += warp_sum[head_in_group][w] * partial_scale;
#pragma unroll
            for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
                const int dim = lane * 4 + (i / 4) * 4 * WARP_SIZE + i % 4;
                final_acc[i] += warp_acc[head_in_group][w][dim] * partial_scale;
            }
        }
        float * partial = partials + paged_partial_offset<HEAD_DIM>(seq, head, partition, n_heads, N_PARTITIONS);
        if (lane == 0) {
            partial[0] = final_max;
            partial[1] = final_sum;
        }
#pragma unroll
        for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
            const int dim = lane * 4 + (i / 4) * 4 * WARP_SIZE + i % 4;
            partial[2 + dim] = final_acc[i];
        }
    }
}

template <int DIRECTION, int HEAD_DIM>
static __device__ __forceinline__ void paged_turbo_wht_256(
        const float * src, float * values, int tid) {
    static_assert(HEAD_DIM == 256, "fused Turbo WHT supports head dimension 256");
    static_assert(DIRECTION == 0 || DIRECTION == 1, "invalid Turbo WHT direction");
    if (tid < QK_TURBO3_GROUP) {
        const float sign = DIRECTION == 0 ? TURBO3_WHT_SIGNS_1[tid] : TURBO3_WHT_SIGNS_2[tid];
        values[tid] = src[tid] * sign;
        values[QK_TURBO3_GROUP + tid] = src[QK_TURBO3_GROUP + tid] * sign;
    }
    __syncthreads();

#define PAGED_TURBO_WHT_STAGE(h) \
    if (tid < QK_TURBO3_GROUP && tid % (2*(h)) < (h)) { \
        const float a0 = values[tid]; \
        const float b0 = values[tid + (h)]; \
        const float a1 = values[QK_TURBO3_GROUP + tid]; \
        const float b1 = values[QK_TURBO3_GROUP + tid + (h)]; \
        values[tid] = a0 + b0; \
        values[tid + (h)] = a0 - b0; \
        values[QK_TURBO3_GROUP + tid] = a1 + b1; \
        values[QK_TURBO3_GROUP + tid + (h)] = a1 - b1; \
    } \
    __syncthreads();

    PAGED_TURBO_WHT_STAGE(1)
    PAGED_TURBO_WHT_STAGE(2)
    PAGED_TURBO_WHT_STAGE(4)
    PAGED_TURBO_WHT_STAGE(8)
    PAGED_TURBO_WHT_STAGE(16)
    PAGED_TURBO_WHT_STAGE(32)
    PAGED_TURBO_WHT_STAGE(64)
#undef PAGED_TURBO_WHT_STAGE

    if (tid < QK_TURBO3_GROUP) {
        const float sign = DIRECTION == 0 ? TURBO3_WHT_SIGNS_2[tid] : TURBO3_WHT_SIGNS_1[tid];
        values[tid] *= 0.08838834764831845f * sign;
        values[QK_TURBO3_GROUP + tid] *= 0.08838834764831845f * sign;
    }
    __syncthreads();
}

template <int HEAD_DIM, int N_PARTITIONS, bool FUSE_TURBO_WHT = false>
__global__ void paged_attention_decode_reduce_kernel(
        const float * partials, const int * batch_offsets, int n_heads, float * out) {
    __shared__ float values[HEAD_DIM];
    const int head = blockIdx.x;
    const int seq = blockIdx.y;
    const int dim = threadIdx.x;
    float final_max = -FLT_MAX;
#pragma unroll
    for (int partition = 0; partition < N_PARTITIONS; ++partition) {
        const float * partial = partials + paged_partial_offset<HEAD_DIM>(seq, head, partition, n_heads, N_PARTITIONS);
        final_max = fmaxf(final_max, partial[0]);
    }

    float final_sum = 0.0f;
    float final_acc = 0.0f;
#pragma unroll
    for (int partition = 0; partition < N_PARTITIONS; ++partition) {
        const float * partial = partials + paged_partial_offset<HEAD_DIM>(seq, head, partition, n_heads, N_PARTITIONS);
        const float partial_scale = partial[0] == final_max ? 1.0f : __expf(partial[0] - final_max);
        final_sum += partial[1] * partial_scale;
        final_acc += partial[2 + dim] * partial_scale;
    }
    const size_t out_base = ((size_t) batch_offsets[seq] * n_heads + head) * HEAD_DIM;
    if constexpr (FUSE_TURBO_WHT) {
        values[dim] = final_acc / (final_sum + 1e-6f);
        __syncthreads();
        paged_turbo_wht_256<1, HEAD_DIM>(values, values, dim);
        if (dim < QK_TURBO3_GROUP) {
            out[out_base + dim] = values[dim];
            out[out_base + QK_TURBO3_GROUP + dim] = values[QK_TURBO3_GROUP + dim];
        }
    } else {
        out[out_base + dim] = final_acc / (final_sum + 1e-6f);
    }
}

// Address setup is paged; attention control flow is shared with contiguous Turbo decode.
template <int HEAD_DIM, int N_WARPS, ggml_type TYPE_K, ggml_type TYPE_V>
__global__ __launch_bounds__(WARP_SIZE * N_WARPS, 1) void paged_attention_decode_turbo_vec_kernel(
        const float * q, const char * k_cache, const char * v_cache,
        const int * block_table, const int * context_lens, const int * batch_offsets,
        size_t k_stride_token, size_t k_stride_head, size_t k_stride_block,
        size_t v_stride_token, size_t v_stride_head, size_t v_stride_block,
        int n_heads, int n_heads_kv, int block_size, int max_blocks, float scale, float * out) {
    __shared__ fattn_decode_vec_shared<HEAD_DIM, N_WARPS> shared;
    const int head = blockIdx.x;
    const int seq = blockIdx.y;
    const int kv_head = ggml_paged_attn_kv_head(head, n_heads, n_heads_kv);
    const int limit = context_lens[seq];
    const size_t q_base = (size_t) batch_offsets[seq] * n_heads * HEAD_DIM + (size_t) head * HEAD_DIM;
    const fattn_paged_kv_address kv_address = {
        k_cache + (size_t) kv_head * k_stride_head,
        v_cache + (size_t) kv_head * v_stride_head,
        block_table + (size_t) seq * max_blocks,
        k_stride_token, k_stride_block, v_stride_token, v_stride_block, block_size,
    };
    fattn_decode_vec_core<HEAD_DIM, N_WARPS, TYPE_K, TYPE_V>(
        q + q_base, kv_address, fattn_decode_identity_score{}, nullptr, limit, scale, out + q_base, shared);
}

template <int HEAD_DIM, int N_PARTITIONS, ggml_type TYPE_K, ggml_type TYPE_V, bool FUSE_TURBO_WHT>
__global__ __launch_bounds__(128, 1) void paged_attention_decode_turbo_token_vec_kernel(
        const float * q, const char * k_cache, const char * v_cache,
        const int * block_table, const int * context_lens, const int * batch_offsets,
        size_t k_stride_token, size_t k_stride_head, size_t k_stride_block,
        size_t v_stride_token, size_t v_stride_head, size_t v_stride_block,
        int n_heads, int n_heads_kv, int block_size, int max_blocks, float scale,
        float * partials) {
    static_assert(HEAD_DIM == 256, "paged Turbo token vec supports head dimension 256");
    static_assert(TYPE_K == TYPE_V, "paged Turbo token vec requires matching K/V formats");

    constexpr int N_WARPS = 4;
    constexpr int TILE_TOKENS = N_WARPS * WARP_SIZE;
    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<TYPE_K, HEAD_DIM, WARP_SIZE>();
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<TYPE_V, float, 4>();
    constexpr int Q_FRAGMENTS = HEAD_DIM / (2 * WARP_SIZE);
    constexpr int V_GROUPS = HEAD_DIM / (4 * WARP_SIZE);

    __shared__ fattn_kv_rows row_cache[TILE_TOKENS];
    __shared__ float weights[TILE_TOKENS];
    __shared__ fattn_decode_vec_shared<HEAD_DIM, N_WARPS> combined;
    __shared__ float q_values[HEAD_DIM];

    const int head = blockIdx.x;
    const int seq = blockIdx.y;
    const int partition = blockIdx.z;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid = warp * WARP_SIZE + lane;
    const int kv_head = ggml_paged_attn_kv_head(head, n_heads, n_heads_kv);
    const int limit = context_lens[seq];
    const size_t q_base = (size_t) batch_offsets[seq] * n_heads * HEAD_DIM + (size_t) head * HEAD_DIM;
    const fattn_paged_kv_address kv_address = {
        k_cache + (size_t) kv_head * k_stride_head,
        v_cache + (size_t) kv_head * v_stride_head,
        block_table + (size_t) seq * max_blocks,
        k_stride_token, k_stride_block, v_stride_token, v_stride_block, block_size,
    };

    const float * q_row_f32 = q + q_base;
    if constexpr (FUSE_TURBO_WHT) {
        paged_turbo_wht_256<0, HEAD_DIM>(q_row_f32, q_values, tid);
        q_row_f32 = q_values;
    }
    const float2 * q_row = (const float2 *) q_row_f32;
    float2 q_reg[Q_FRAGMENTS];
    float acc[HEAD_DIM / WARP_SIZE] = {};
#pragma unroll
    for (int i = 0; i < Q_FRAGMENTS; ++i) {
        q_reg[i] = q_row[lane * Q_FRAGMENTS + i];
        q_reg[i].x *= scale;
        q_reg[i].y *= scale;
    }

    float qk_max = -FLT_MAX;
    float qk_sum = 0.0f;
    for (int tile = partition * TILE_TOKENS; tile < limit; tile += N_PARTITIONS * TILE_TOKENS) {
        const int token = tile + tid;
        float tile_max = -FLT_MAX;
#pragma unroll
        for (int i = 0; i < WARP_SIZE; ++i) {
            const int row = warp * WARP_SIZE + i;
            const bool valid = tile + row < limit;
            fattn_kv_rows rows = { nullptr, nullptr };
            float qk = -FLT_MAX / 2.0f;
            if (valid) {
                rows = kv_address.rows(tile + row);
                qk = vec_dot_KQ(rows.k, q_reg, nullptr, nullptr);
                qk = warp_reduce_sum(qk);
            }
            tile_max = fmaxf(tile_max, qk);
            if (lane == i) {
                row_cache[row] = rows;
                weights[row] = qk;
            }
        }

        const float new_max = fmaxf(qk_max, tile_max);
        const float old_scale = fattn_softmax_rescale(qk_max, new_max);
#pragma unroll
        for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
            acc[i] *= old_scale;
        }
        qk_sum *= old_scale;
        qk_max = new_max;

        const float weight = token < limit ? __expf(weights[tid] - qk_max) : 0.0f;
        weights[tid] = weight;
        qk_sum += warp_reduce_sum(weight);
        __syncthreads();

#pragma unroll
        for (int i = 0; i < WARP_SIZE; ++i) {
            const int row = warp * WARP_SIZE + i;
            const float row_weight = weights[row];
            if (tile + row < limit) {
#pragma unroll
                for (int group = 0; group < V_GROUPS; ++group) {
                    float values[4];
                    dequantize_V(row_cache[row].v, values, lane * 4 + group * 4 * WARP_SIZE);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        acc[group * 4 + j] += row_weight * values[j];
                    }
                }
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
        combined.partial_max[warp] = qk_max;
        combined.partial_sum[warp] = qk_sum;
    }
#pragma unroll
    for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
        const int dim = lane * 4 + (i / 4) * 4 * WARP_SIZE + i % 4;
        combined.partial_acc[warp][dim] = acc[i];
    }
    __syncthreads();

    if (warp == 0) {
        float final_max = -FLT_MAX;
        float final_sum = 0.0f;
        float final_acc[HEAD_DIM / WARP_SIZE] = {};
#pragma unroll
        for (int w = 0; w < N_WARPS; ++w) {
            final_max = fmaxf(final_max, combined.partial_max[w]);
        }
#pragma unroll
        for (int w = 0; w < N_WARPS; ++w) {
            const float partial_scale = fattn_softmax_rescale(combined.partial_max[w], final_max);
            final_sum += combined.partial_sum[w] * partial_scale;
#pragma unroll
            for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
                const int dim = lane * 4 + (i / 4) * 4 * WARP_SIZE + i % 4;
                final_acc[i] += combined.partial_acc[w][dim] * partial_scale;
            }
        }

        float * partial = partials + paged_partial_offset<HEAD_DIM>(
            seq, head, partition, n_heads, N_PARTITIONS);
        if (lane == 0) {
            partial[0] = final_max;
            partial[1] = final_sum;
        }
#pragma unroll
        for (int i = 0; i < HEAD_DIM / WARP_SIZE; ++i) {
            const int dim = lane * 4 + (i / 4) * 4 * WARP_SIZE + i % 4;
            partial[2 + dim] = final_acc[i];
        }
    }
}

#endif

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#if defined(TURING_MMA_AVAILABLE)
template <int HEAD_DIM, bool CACHE_ROW_OFFSETS>
__global__ void paged_attention_prefill_mma_kernel(
        const float * q,
        const char * k_cache,
        const char * v_cache,
        const int * block_table,
        const int * context_lens,
        const int * batch_offsets,
        const int * batch_lens,
        size_t k_stride_token,
        size_t k_stride_head,
        size_t k_stride_block,
        size_t v_stride_token,
        size_t v_stride_head,
        size_t v_stride_block,
        int n_heads_kv,
        int block_size,
        int max_blocks,
        int k_type,
        int v_type,
        float scale,
        float * out) {
    constexpr int Q_TILE = 16;
    constexpr int K_TILE = 16;

    extern __shared__ __align__(32) char shared_storage[];
    char * shared = shared_storage;
    constexpr size_t Q_BYTES = Q_TILE * HEAD_DIM * sizeof(half);
    constexpr size_t K_BYTES = K_TILE * HEAD_DIM * sizeof(half);
    constexpr size_t V_BYTES = HEAD_DIM * K_TILE * sizeof(half);
    constexpr size_t SCORES_BYTES = Q_TILE * K_TILE * sizeof(float);
    constexpr size_t WEIGHTS_BYTES = Q_TILE * K_TILE * sizeof(half);
    constexpr size_t VALUES_BYTES = Q_TILE * HEAD_DIM * sizeof(float);
    half (*q_shared)[HEAD_DIM] = (half (*)[HEAD_DIM]) shared;
    half (*k_shared)[HEAD_DIM] = (half (*)[HEAD_DIM]) (shared + Q_BYTES);
    half (*v_shared)[K_TILE] = (half (*)[K_TILE]) (shared + Q_BYTES + K_BYTES);
    float (*scores_shared)[K_TILE] = (float (*)[K_TILE]) (shared + Q_BYTES + K_BYTES + V_BYTES);
    half (*weights_shared)[K_TILE] = (half (*)[K_TILE]) (shared + Q_BYTES + K_BYTES + V_BYTES + SCORES_BYTES);
    float (*values_shared)[HEAD_DIM] = (float (*)[HEAD_DIM]) (shared + Q_BYTES + K_BYTES + V_BYTES + SCORES_BYTES + WEIGHTS_BYTES);
    float (*value_acc_shared)[HEAD_DIM] = (float (*)[HEAD_DIM]) (shared + Q_BYTES + K_BYTES + V_BYTES + SCORES_BYTES + WEIGHTS_BYTES + VALUES_BYTES);
    float * max_shared = (float *) (shared + Q_BYTES + K_BYTES + V_BYTES + SCORES_BYTES + WEIGHTS_BYTES + 2 * VALUES_BYTES);
    float * sum_shared = max_shared + Q_TILE;
    float * rescale_shared = sum_shared + Q_TILE;
    __shared__ size_t k_row_offsets[K_TILE];
    __shared__ size_t v_row_offsets[K_TILE];

    const int tid = threadIdx.x;
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int q_tile_start = blockIdx.z * Q_TILE;
    const int n_heads = gridDim.x;
    const int num_new_tokens = batch_lens[seq_idx];
    if (num_new_tokens <= 1 || q_tile_start >= num_new_tokens) {
        return;
    }

    const int seq_start = batch_offsets[seq_idx];
    const int context_len = context_lens[seq_idx];
    const int kv_head_idx = ggml_paged_attn_kv_head(head_idx, n_heads, n_heads_kv);

    for (int vec = tid; vec < Q_TILE * HEAD_DIM / 4; vec += blockDim.x) {
        const int q_row = vec / (HEAD_DIM / 4);
        const int q_dim = (vec % (HEAD_DIM / 4)) * 4;
        const int q_token = q_tile_start + q_row;
        if (q_tile_start + q_row < num_new_tokens) {
            const size_t q_offset = (size_t) (seq_start + q_token) * n_heads * HEAD_DIM + (size_t) head_idx * HEAD_DIM + q_dim;
            paged_store_half4(&q_shared[q_row][q_dim], *(const float4 *) (q + q_offset));
        } else {
            paged_store_half4(&q_shared[q_row][q_dim], make_float4(0.0f, 0.0f, 0.0f, 0.0f));
        }
    }

    if (tid < Q_TILE) {
        max_shared[tid] = -FLT_MAX;
        sum_shared[tid] = 0.0f;
    }
    __syncthreads();

    for (int vec = tid; vec < Q_TILE * HEAD_DIM / 4; vec += blockDim.x) {
        paged_store_float4(&value_acc_shared[0][0] + vec * 4, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
    }
    __syncthreads();

    for (int token_start = 0; token_start < context_len; token_start += K_TILE) {
        if constexpr (CACHE_ROW_OFFSETS) {
            if (tid < K_TILE) {
                const int token = token_start + tid;
                if (token < context_len) {
                    const int physical_block = block_table[seq_idx * max_blocks + token / block_size];
                    const int token_in_block = token % block_size;
                    k_row_offsets[tid] = (size_t) physical_block * k_stride_block + (size_t) token_in_block * k_stride_token;
                    v_row_offsets[tid] = (size_t) physical_block * v_stride_block + (size_t) token_in_block * v_stride_token;
                }
            }
            __syncthreads();
        }
        for (int vec = tid; vec < K_TILE * HEAD_DIM / 4; vec += blockDim.x) {
            const int key = vec / (HEAD_DIM / 4);
            const int dim = (vec % (HEAD_DIM / 4)) * 4;
            const int token = token_start + key;
            if (token < context_len) {
                size_t k_row_offset;
                size_t v_row_offset;
                if constexpr (CACHE_ROW_OFFSETS) {
                    k_row_offset = k_row_offsets[key];
                    v_row_offset = v_row_offsets[key];
                } else {
                    const int physical_block = block_table[seq_idx * max_blocks + token / block_size];
                    const int token_in_block = token % block_size;
                    k_row_offset = (size_t) physical_block * k_stride_block + (size_t) token_in_block * k_stride_token;
                    v_row_offset = (size_t) physical_block * v_stride_block + (size_t) token_in_block * v_stride_token;
                }
                const char * k_row = k_cache + k_row_offset + (size_t) kv_head_idx * k_stride_head;
                const char * v_row = v_cache + v_row_offset + (size_t) kv_head_idx * v_stride_head;
                paged_store_half4(&k_shared[key][dim], paged_cache_value4(k_row, dim, k_type));
                paged_store_transposed_half4(v_shared, dim, key, paged_cache_value4(v_row, dim, v_type));
            } else {
                paged_store_half4(&k_shared[key][dim], make_float4(0.0f, 0.0f, 0.0f, 0.0f));
                paged_store_transposed_half4(v_shared, dim, key, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
            }
        }
        __syncthreads();

        if (tid < WARP_SIZE) {
            wmma::fragment<wmma::accumulator, Q_TILE, K_TILE, 16, float> scores;
            wmma::fill_fragment(scores, 0.0f);
#pragma unroll
            for (int dim = 0; dim < HEAD_DIM; dim += 16) {
                wmma::fragment<wmma::matrix_a, Q_TILE, K_TILE, 16, half, wmma::row_major> q_fragment;
                wmma::fragment<wmma::matrix_b, Q_TILE, K_TILE, 16, half, wmma::col_major> k_fragment;
                wmma::load_matrix_sync(q_fragment, &q_shared[0][dim], HEAD_DIM);
                wmma::load_matrix_sync(k_fragment, &k_shared[0][dim], HEAD_DIM);
                wmma::mma_sync(scores, q_fragment, k_fragment, scores);
            }
            wmma::store_matrix_sync(&scores_shared[0][0], scores, K_TILE, wmma::mem_row_major);
        }
        __syncthreads();

        if (tid < Q_TILE) {
            const int q_row = tid;
            const int q_token = q_tile_start + q_row;
            if (q_token < num_new_tokens) {
                const int q_pos = context_len - num_new_tokens + q_token;
                float tile_max = -FLT_MAX;
#pragma unroll
                for (int key = 0; key < K_TILE; ++key) {
                    const int token = token_start + key;
                    if (token <= q_pos && token < context_len) {
                        tile_max = fmaxf(tile_max, scores_shared[q_row][key] * scale);
                    }
                }

                const float qk_max = fmaxf(max_shared[q_row], tile_max);
                const float old_scale = __expf(max_shared[q_row] - qk_max);
                float tile_sum = 0.0f;
#pragma unroll
                for (int key = 0; key < K_TILE; ++key) {
                    const int token = token_start + key;
                    const float weight = token <= q_pos && token < context_len ? __expf(scores_shared[q_row][key] * scale - qk_max) : 0.0f;
                    weights_shared[q_row][key] = __float2half_rn(weight);
                    tile_sum += weight;
                }
                max_shared[q_row] = qk_max;
                sum_shared[q_row] = sum_shared[q_row] * old_scale + tile_sum;
                rescale_shared[q_row] = old_scale;
            }
        }
        __syncthreads();

        const int warp = tid / WARP_SIZE;
        if (warp < HEAD_DIM / K_TILE) {
            const int value_tile = warp * K_TILE;
            wmma::fragment<wmma::matrix_a, Q_TILE, K_TILE, K_TILE, half, wmma::row_major> weights;
            wmma::fragment<wmma::matrix_b, Q_TILE, K_TILE, K_TILE, half, wmma::col_major> values;
            wmma::fragment<wmma::accumulator, Q_TILE, K_TILE, K_TILE, float> value_acc;
            wmma::load_matrix_sync(weights, &weights_shared[0][0], K_TILE);
            wmma::load_matrix_sync(values, &v_shared[value_tile][0], K_TILE);
            wmma::fill_fragment(value_acc, 0.0f);
            wmma::mma_sync(value_acc, weights, values, value_acc);
            wmma::store_matrix_sync(&values_shared[0][value_tile], value_acc, HEAD_DIM, wmma::mem_row_major);
        }
        __syncthreads();

        for (int vec = tid; vec < Q_TILE * HEAD_DIM / 4; vec += blockDim.x) {
            const int q_row = vec / (HEAD_DIM / 4);
            const int dim = (vec % (HEAD_DIM / 4)) * 4;
            if (q_tile_start + q_row < num_new_tokens) {
                const float old_scale = rescale_shared[q_row];
                const float4 value = paged_load_float4(&values_shared[q_row][dim]);
                const float4 acc = paged_load_float4(&value_acc_shared[q_row][dim]);
                paged_store_float4(&value_acc_shared[q_row][dim], make_float4(
                    acc.x * old_scale + value.x, acc.y * old_scale + value.y,
                    acc.z * old_scale + value.z, acc.w * old_scale + value.w));
            }
        }
        __syncthreads();
    }

    for (int vec = tid; vec < Q_TILE * HEAD_DIM / 4; vec += blockDim.x) {
        const int q_row = vec / (HEAD_DIM / 4);
        const int dim = (vec % (HEAD_DIM / 4)) * 4;
        const int q_token = q_tile_start + q_row;
        if (q_tile_start + q_row < num_new_tokens) {
            const size_t out_offset = (size_t) (seq_start + q_token) * n_heads * HEAD_DIM + (size_t) head_idx * HEAD_DIM + dim;
            const float4 acc = paged_load_float4(&value_acc_shared[q_row][dim]);
            const float inv_sum = 1.0f / (sum_shared[q_row] + 1e-6f);
            paged_store_float4(out + out_offset, make_float4(
                acc.x * inv_sum, acc.y * inv_sum, acc.z * inv_sum, acc.w * inv_sum));
        }
    }
}
#else
template <int HEAD_DIM, bool CACHE_ROW_OFFSETS>
__global__ void paged_attention_prefill_mma_kernel(
        const float * q,
        const char * k_cache,
        const char * v_cache,
        const int * block_table,
        const int * context_lens,
        const int * batch_offsets,
        const int * batch_lens,
        size_t k_stride_token,
        size_t k_stride_head,
        size_t k_stride_block,
        size_t v_stride_token,
        size_t v_stride_head,
        size_t v_stride_block,
        int n_heads_kv,
        int block_size,
        int max_blocks,
        int k_type,
        int v_type,
        float scale,
        float * out) {
    GGML_UNUSED_VARS(q, k_cache, v_cache, block_table, context_lens, batch_offsets, batch_lens,
        k_stride_token, k_stride_head, k_stride_block, v_stride_token, v_stride_head, v_stride_block,
        n_heads_kv, block_size, max_blocks, k_type, v_type, scale, out, CACHE_ROW_OFFSETS);
    NO_DEVICE_CODE;
}
#endif
#endif

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
template <int HEAD_DIM>
static bool paged_attn_runtime_prefill_test_case() {
    constexpr int n_heads = 4;
    constexpr int n_heads_kv = 2;
    constexpr int block_size = 16;
    constexpr int n_tokens = 32;
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM);
    const size_t cache_bytes = 2 * n_heads_kv * block_size * row_bytes;
    const size_t q_bytes = (size_t) n_tokens * n_heads * HEAD_DIM * sizeof(float);
    const size_t out_bytes = q_bytes;
    const size_t smem_bytes =
        (size_t) (3 * 16 * HEAD_DIM) * sizeof(half) + (size_t) (16 * 16) * (sizeof(float) + sizeof(half)) +
        (size_t) (2 * 16 * HEAD_DIM + 3 * 16) * sizeof(float);
    float * q = nullptr; char * k = nullptr; char * v = nullptr; int * table = nullptr;
    int * lens = nullptr; int * offsets = nullptr; int * batch_lens = nullptr; float * out = nullptr;
    bool ok = cudaMalloc(&q, q_bytes) == cudaSuccess && cudaMalloc(&k, cache_bytes) == cudaSuccess &&
        cudaMalloc(&v, cache_bytes) == cudaSuccess && cudaMalloc(&table, 2 * sizeof(int)) == cudaSuccess &&
        cudaMalloc(&lens, sizeof(int)) == cudaSuccess && cudaMalloc(&offsets, sizeof(int)) == cudaSuccess &&
        cudaMalloc(&batch_lens, sizeof(int)) == cudaSuccess && cudaMalloc(&out, out_bytes) == cudaSuccess;
    if (!ok) { return false; }
    cudaMemset(q, 0, q_bytes);
    std::vector<unsigned char> host_cache(cache_bytes, 0);
    for (int block = 0; block < 2; ++block) {
        for (int head = 0; head < n_heads_kv; ++head) {
            for (int token = 0; token < block_size; ++token) {
                unsigned char * row = host_cache.data() + ((size_t) block * n_heads_kv * block_size + head * block_size + token) * row_bytes;
                const uint16_t scale = 0x3c00;
                memcpy(row, &scale, sizeof(scale));
                memset(row + sizeof(scale), head + 1, row_bytes - sizeof(scale));
            }
        }
    }
    cudaMemcpy(k, host_cache.data(), cache_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(v, host_cache.data(), cache_bytes, cudaMemcpyHostToDevice);
    const int host_table[] = { 0, 1 }; const int host_len[] = { n_tokens }; const int host_zero[] = { 0 };
    cudaMemcpy(table, host_table, sizeof(host_table), cudaMemcpyHostToDevice);
    cudaMemcpy(lens, host_len, sizeof(host_len), cudaMemcpyHostToDevice);
    cudaMemcpy(offsets, host_zero, sizeof(host_zero), cudaMemcpyHostToDevice);
    cudaMemcpy(batch_lens, host_len, sizeof(host_len), cudaMemcpyHostToDevice);
    if constexpr (HEAD_DIM == 256) {
        ok = cudaFuncSetAttribute(paged_attention_prefill_mma_kernel<256, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes) == cudaSuccess;
    }
    paged_attention_prefill_mma_kernel<HEAD_DIM, false><<<dim3(n_heads, 1, 2), dim3(HEAD_DIM == 256 ? 512 : 256), smem_bytes>>>(
        q, k, v, table, lens, offsets, batch_lens, row_bytes, row_bytes * block_size,
        row_bytes * block_size * n_heads_kv, row_bytes, row_bytes * block_size,
        row_bytes * block_size * n_heads_kv, n_heads_kv, block_size, 2, GGML_TYPE_Q8_0,
        GGML_TYPE_Q8_0, 1.0f, out);
    std::vector<float> host_out((size_t) n_tokens * n_heads * HEAD_DIM);
    ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
        cudaMemcpy(host_out.data(), out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    for (float value : host_out) {
        ok = ok && std::isfinite(value);
    }
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(table); cudaFree(lens); cudaFree(offsets); cudaFree(batch_lens); cudaFree(out);
    return ok;
}
static bool paged_attn_runtime_fallback_test_case() {
    constexpr int head_dim = 64;
    constexpr int n_heads = 4;
    constexpr int n_heads_kv = 2;
    constexpr int block_size = 16;
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, head_dim);
    const size_t q_bytes = 2 * n_heads * head_dim * sizeof(float);
    const size_t cache_bytes = n_heads_kv * block_size * row_bytes;
    float * q = nullptr; char * k = nullptr; char * v = nullptr; int * table = nullptr;
    int * lens = nullptr; int * offsets = nullptr; int * batch_lens = nullptr; float * out = nullptr;
    bool ok = cudaMalloc(&q, q_bytes) == cudaSuccess && cudaMalloc(&k, cache_bytes) == cudaSuccess &&
        cudaMalloc(&v, cache_bytes) == cudaSuccess && cudaMalloc(&table, sizeof(int)) == cudaSuccess &&
        cudaMalloc(&lens, sizeof(int)) == cudaSuccess && cudaMalloc(&offsets, sizeof(int)) == cudaSuccess &&
        cudaMalloc(&batch_lens, sizeof(int)) == cudaSuccess && cudaMalloc(&out, q_bytes) == cudaSuccess;
    if (!ok) { return false; }
    cudaMemset(q, 0, q_bytes); cudaMemset(k, 0, cache_bytes); cudaMemset(v, 0, cache_bytes);
    const int zero = 0; const int length = 2;
    cudaMemcpy(table, &zero, sizeof(zero), cudaMemcpyHostToDevice);
    cudaMemcpy(lens, &length, sizeof(length), cudaMemcpyHostToDevice);
    cudaMemcpy(offsets, &zero, sizeof(zero), cudaMemcpyHostToDevice);
    cudaMemcpy(batch_lens, &length, sizeof(length), cudaMemcpyHostToDevice);
    ggml_paged_attn_tiled_prefill_launch_count_reset();
    const size_t smem_bytes = ((size_t) (head_dim + 31) / 32) * sizeof(float);
    paged_attention_decode_kernel<<<dim3(n_heads, 1), dim3(head_dim), smem_bytes>>>(
        q, k, v, table, lens, offsets, batch_lens, row_bytes, row_bytes * block_size,
        row_bytes * block_size * n_heads_kv, row_bytes, row_bytes * block_size,
        row_bytes * block_size * n_heads_kv, n_heads_kv, block_size, 1, GGML_TYPE_Q8_0,
        GGML_TYPE_Q8_0, 1.0f, false, out);
    std::vector<float> host_out(2 * n_heads * head_dim);
    ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
        cudaMemcpy(host_out.data(), out, q_bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
        ggml_paged_attn_tiled_prefill_launch_count() == 0;
    for (float value : host_out) {
        ok = ok && std::isfinite(value) && fabsf(value) < 1e-6f;
    }
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(table); cudaFree(lens); cudaFree(offsets); cudaFree(batch_lens); cudaFree(out);
    return ok;
}

#endif

bool ggml_paged_attn_cuda_runtime_test(void) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const bool covered_128 = paged_attn_runtime_prefill_test_case<128>();
    const bool covered_256 = paged_attn_runtime_prefill_test_case<256>();
    const bool covered_fallback = paged_attn_runtime_fallback_test_case();
    fprintf(stderr, "paged runtime coverage: 128=%d 256=%d fallback=%d\n", covered_128, covered_256, covered_fallback);
    return covered_128 && covered_256 && covered_fallback;
#else
    return false;
#endif
}
static bool paged_kv_type_supported(ggml_type type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static ggml_paged_attn_cuda_device_caps paged_attn_query_cuda_device_caps(
        cudaStream_t stream, cudaStreamCaptureStatus * capture_status_out = nullptr) {
    const auto & device_info = ggml_cuda_info().devices[ggml_cuda_get_device()];
    const bool sm89_compiled = ggml_cuda_has_arch(GGML_CUDA_CC_ADA_LOVELACE);
    bool graph_capture_safe = false;
    if (capture_status_out != nullptr || (device_info.cc == GGML_CUDA_CC_ADA_LOVELACE && sm89_compiled)) {
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        const cudaError_t capture_error = cudaStreamIsCapturing(stream, &capture_status);
        graph_capture_safe = capture_error == cudaSuccess && capture_status != cudaStreamCaptureStatusInvalidated;
        if (capture_status_out != nullptr) {
            *capture_status_out = capture_status;
        }
    }
    return {
        device_info.cc,
        device_info.nsm,
        device_info.warp_size,
        device_info.max_threads_per_block,
        (int64_t) device_info.smpb,
        sm89_compiled,
        graph_capture_safe,
    };
}
#endif

void ggml_cuda_op_paged_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k_new = dst->src[1];
    const ggml_tensor * v_new = dst->src[2];
    const ggml_tensor * k_cache = dst->src[3];
    const ggml_tensor * v_cache = dst->src[4];
    const ggml_tensor * block_table = dst->src[5];
    const ggml_tensor * write_rows = dst->src[6];
    const ggml_tensor * context_lens = dst->src[7];
    const ggml_tensor * batch_offsets = dst->src[8];
    const ggml_tensor * batch_lens = dst->src[9];
    const float * op_params_f = (const float *) dst->op_params;
    const int block_size = ((const int32_t *) (op_params_f + 1))[0];
    const int max_blocks = ((const int32_t *) (op_params_f + 2))[0];
    const int context_bucket = ((const int32_t *) (op_params_f + 1))[2];
    const bool fuse_turbo_wht = ((const int32_t *) (op_params_f + 1))[3];
    const int head_dim = q->ne[0];
    const int n_heads = q->ne[1];
    const int n_heads_kv = k_new->ne[1];

    GGML_ASSERT(n_heads != 0 && n_heads_kv != 0 && head_dim <= 256);
    GGML_ASSERT(n_heads % n_heads_kv == 0);
    GGML_ASSERT(paged_kv_type_supported(k_cache->type) && paged_kv_type_supported(v_cache->type));
    const int64_t n_write_rows = k_new->ne[1] * k_new->ne[2] * k_new->ne[3];
    const ggml_tensor * k_write_source = k_cache;
    while ((k_write_source->op == GGML_OP_RESHAPE || k_write_source->op == GGML_OP_VIEW) &&
           k_write_source->src[0] != nullptr) {
        k_write_source = k_write_source->src[0];
    }
    const bool separate_write_scheduled = k_write_source->op == GGML_OP_SET_ROWS;
    const bool combined_q8_write = !separate_write_scheduled &&
        k_cache->type == GGML_TYPE_Q8_0 && v_cache->type == GGML_TYPE_Q8_0 &&
        k_new->type == GGML_TYPE_F32 && v_new->type == GGML_TYPE_F32 && ggml_is_contiguous(k_new) &&
        ggml_is_contiguous(v_new) && ggml_are_same_shape(k_new, v_new) && ggml_are_same_shape(k_cache, v_cache) &&
        k_new->ne[0] == head_dim && k_cache->ne[0] == head_dim && ggml_is_contiguous(k_cache) &&
        ggml_is_contiguous(v_cache) && head_dim % QK8_0 == 0 &&
        write_rows->type == GGML_TYPE_I32 && ggml_is_contiguous(write_rows) &&
        ggml_nelements(write_rows) == n_write_rows &&
        k_cache->nb[1] == ggml_row_size(GGML_TYPE_Q8_0, head_dim) &&
        v_cache->nb[1] == ggml_row_size(GGML_TYPE_Q8_0, head_dim);
    bool decode_q8 = false;
    bool decode_turbo = false;
    bool fuse_current_q8 = false;
    bool combined_turbo_write = false;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    decode_q8 = k_cache->type == GGML_TYPE_Q8_0 && v_cache->type == GGML_TYPE_Q8_0 &&
        head_dim == 256 && q->type == GGML_TYPE_F32 && ggml_is_contiguous(q) &&
        ggml_cuda_is_aligned(q, sizeof(float4)) && q->ne[2] == batch_lens->ne[0] && q->ne[3] == 1;
    const char * turbo_vec_env = getenv("GGML_CUDA_PAGED_TURBO_VEC");
    decode_turbo = (turbo_vec_env == nullptr || atoi(turbo_vec_env) != 0) &&
        (k_cache->type == GGML_TYPE_TURBO3_0 || k_cache->type == GGML_TYPE_TURBO4_0) &&
        (v_cache->type == GGML_TYPE_TURBO3_0 || v_cache->type == GGML_TYPE_TURBO4_0) &&
        head_dim == 256 && q->type == GGML_TYPE_F32 && ggml_is_contiguous(q) &&
        ggml_cuda_is_aligned(q, sizeof(float2)) && q->ne[2] == batch_lens->ne[0] && q->ne[3] == 1;
    combined_turbo_write = fuse_turbo_wht && !separate_write_scheduled &&
        ((k_cache->type == GGML_TYPE_TURBO3_0 && v_cache->type == GGML_TYPE_TURBO3_0) ||
         (k_cache->type == GGML_TYPE_TURBO4_0 && v_cache->type == GGML_TYPE_TURBO4_0)) &&
        k_new->type == GGML_TYPE_F32 && v_new->type == GGML_TYPE_F32 &&
        ggml_are_same_shape(k_new, v_new) && ggml_are_same_shape(k_cache, v_cache) &&
        ggml_is_contiguous(k_new) && ggml_is_contiguous(v_new) &&
        ggml_is_contiguous(k_cache) && ggml_is_contiguous(v_cache) &&
        k_new->ne[0] == head_dim && k_new->ne[2] == batch_lens->ne[0] && k_new->ne[3] == 1 &&
        write_rows->type == GGML_TYPE_I32 && ggml_is_contiguous(write_rows) &&
        ggml_nelements(write_rows) == n_write_rows &&
        k_cache->nb[1] == ggml_row_size(k_cache->type, head_dim) &&
        v_cache->nb[1] == ggml_row_size(v_cache->type, head_dim);
    // The candidate remains opt-in because matched 27B decode did not beat the combined-write fallback.
    const char * fused_write_env = getenv("GGML_CUDA_PAGED_Q8_FUSED_WRITE");
    fuse_current_q8 = combined_q8_write && decode_q8 && k_new->ne[2] == batch_lens->ne[0] && k_new->ne[3] == 1 &&
        fused_write_env != nullptr && atoi(fused_write_env) != 0;
#endif
    if (combined_turbo_write) {
        const char * combined_write_env = getenv("GGML_CUDA_PAGED_TURBO_COMBINED_WRITE");
        const bool combine_launches = combined_write_env == nullptr || atoi(combined_write_env) != 0;
        if (combine_launches) {
            g_paged_turbo_combined_write_launch_count.fetch_add(1, std::memory_order_relaxed);
        }
        ggml_cuda_set_rows_turbo_pair(
            ctx, k_new, v_new, write_rows, k_cache, v_cache, combine_launches);
    }
    if (combined_q8_write && !fuse_current_q8) {
        g_paged_q8_combined_write_launch_count.fetch_add(1, std::memory_order_relaxed);
        paged_launch_q8_cache_write(
            ctx.stream(),
            (const float *) k_new->data, (const float *) v_new->data, (const int32_t *) write_rows->data,
            (block_q8_0 *) k_cache->data, (block_q8_0 *) v_cache->data, head_dim, n_write_rows,
            k_cache->nb[1], v_cache->nb[1]);
    }

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (decode_q8) {
        g_paged_q8_decode_launch_count.fetch_add(1, std::memory_order_relaxed);
        const auto & device_info = ggml_cuda_info().devices[ggml_cuda_get_device()];
        const int cc = device_info.cc;
        const bool force_float_q = getenv("GGML_CUDA_PAGED_Q8_FLOAT_Q") != nullptr;
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        const ggml_paged_attn_cuda_device_caps caps =
            paged_attn_query_cuda_device_caps(ctx.stream(), &capture_status);
        ggml_paged_attn_cuda_variant variant = ggml_paged_attn_select_cuda_variant(
            context_bucket, head_dim, n_heads, n_heads_kv, batch_lens->ne[0], q->ne[2],
            k_cache->type, v_cache->type, caps);
        const char * force_warps = getenv("GGML_CUDA_PAGED_Q8_WARPS");
        const char * force_partitions = getenv("GGML_CUDA_PAGED_Q8_PARTITIONS");
        const char * force_q_heads = getenv("GGML_CUDA_PAGED_Q8_HEADS");
        if (force_warps) {
            variant.n_warps = atoi(force_warps);
        }
        if (force_partitions) {
            variant.n_partitions = atoi(force_partitions);
        }
        if (force_q_heads) {
            variant.n_q_heads = atoi(force_q_heads);
        }
        const int gqa_ratio = n_heads / n_heads_kv;
        const bool valid_warps = variant.n_warps == 4 || variant.n_warps == 8 ||
            variant.n_warps == 16 || variant.n_warps == 32;
        const bool valid_partitions = variant.n_partitions == 1 || variant.n_partitions == 2 ||
            variant.n_partitions == 4 || variant.n_partitions == 8;
        const bool valid_q_heads = variant.n_q_heads == 1 || variant.n_q_heads == 2 || variant.n_q_heads == 4;
        const bool valid_grouping = gqa_ratio % variant.n_q_heads == 0 && n_heads % variant.n_q_heads == 0;
        if (!valid_warps || !valid_partitions || !valid_q_heads || !valid_grouping ||
            (variant.n_q_heads > 1 && variant.n_warps != 4 * variant.n_q_heads)) {
            GGML_LOG_WARN("%s: invalid forced q8 decode variant warps=%d partitions=%d heads=%d; using legacy\n",
                __func__, variant.n_warps, variant.n_partitions, variant.n_q_heads);
            variant = { 32, 1, 1 };
        }
        if (fuse_current_q8 && variant.n_q_heads == 1) {
            fuse_current_q8 = false;
            g_paged_q8_combined_write_launch_count.fetch_add(1, std::memory_order_relaxed);
            paged_launch_q8_cache_write(
                ctx.stream(),
                (const float *) k_new->data, (const float *) v_new->data, (const int32_t *) write_rows->data,
                (block_q8_0 *) k_cache->data, (block_q8_0 *) v_cache->data, head_dim, n_write_rows,
                k_cache->nb[1], v_cache->nb[1]);
        }
        if (fuse_current_q8) {
            g_paged_q8_fused_write_launch_count.fetch_add(1, std::memory_order_relaxed);
        }

        bool l2_window_active = false;
        const char * l2_window_env = getenv("GGML_CUDA_PAGED_Q8_L2_WINDOW");
        if (l2_window_env != nullptr && atoi(l2_window_env) != 0 &&
                capture_status == cudaStreamCaptureStatusNone) {
            int max_window = 0;
            int max_persisting = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&max_window, cudaDevAttrMaxAccessPolicyWindowSize, ggml_cuda_get_device()));
            CUDA_CHECK(cudaDeviceGetAttribute(&max_persisting, cudaDevAttrMaxPersistingL2CacheSize, ggml_cuda_get_device()));
            const size_t table_bytes = ggml_nbytes(block_table);
            const size_t window_bytes = std::min(table_bytes, (size_t) max_window);
            if (window_bytes > 0 && max_persisting > 0) {
                CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                    std::min((size_t) max_persisting, window_bytes)));
                cudaStreamAttrValue attribute = {};
                attribute.accessPolicyWindow.base_ptr = block_table->data;
                attribute.accessPolicyWindow.num_bytes = window_bytes;
                attribute.accessPolicyWindow.hitRatio = 1.0f;
                attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
                attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
                CUDA_CHECK(cudaStreamSetAttribute(ctx.stream(), cudaStreamAttributeAccessPolicyWindow, &attribute));
                l2_window_active = true;
            }
        }

#define LAUNCH_Q8_LEGACY(N_WARPS, PACKED_QV, FUSE_CURRENT) \
        paged_attention_decode_q8_parallel_kernel<256, N_WARPS, PACKED_QV, FUSE_CURRENT> \
            <<<dim3(n_heads, batch_lens->ne[0]), dim3(WARP_SIZE, N_WARPS), 0, ctx.stream()>>>( \
            (const float *) q->data, (const float *) k_new->data, (const float *) v_new->data, \
            (char *) k_cache->data, (char *) v_cache->data, (const int *) block_table->data, \
            (const int *) write_rows->data, (const int *) context_lens->data, (const int *) batch_offsets->data, \
            k_cache->nb[1], k_cache->nb[2], k_cache->nb[3], v_cache->nb[1], v_cache->nb[2], v_cache->nb[3], \
            n_heads, n_heads_kv, block_size, max_blocks, op_params_f[0], (float *) dst->data)

        if (variant.n_partitions == 1 && variant.n_q_heads == 1) {
#define DISPATCH_Q8_LEGACY(PACKED_QV, FUSE_CURRENT) \
            switch (variant.n_warps) { \
                case 4:  LAUNCH_Q8_LEGACY(4,  PACKED_QV, FUSE_CURRENT); break; \
                case 8:  LAUNCH_Q8_LEGACY(8,  PACKED_QV, FUSE_CURRENT); break; \
                case 16: LAUNCH_Q8_LEGACY(16, PACKED_QV, FUSE_CURRENT); break; \
                default: LAUNCH_Q8_LEGACY(32, PACKED_QV, FUSE_CURRENT); break; \
            }
            if (!force_float_q && cc >= GGML_CUDA_CC_DP4A) {
                if (fuse_current_q8) {
                    DISPATCH_Q8_LEGACY(true, true);
                } else {
                    DISPATCH_Q8_LEGACY(true, false);
                }
            } else {
                if (fuse_current_q8) {
                    DISPATCH_Q8_LEGACY(false, true);
                } else {
                    DISPATCH_Q8_LEGACY(false, false);
                }
            }
#undef DISPATCH_Q8_LEGACY
        } else {
            const size_t partial_count = (size_t) batch_lens->ne[0] * n_heads *
                variant.n_partitions * (256 + 2);
            ggml_cuda_pool_alloc<float> partials(ctx.pool(), partial_count);
#define LAUNCH_Q8_PARTIAL(N_WARPS, N_PARTITIONS, PACKED_QV, FUSE_CURRENT) \
            paged_attention_decode_q8_partial_kernel<256, N_WARPS, N_PARTITIONS, PACKED_QV, FUSE_CURRENT> \
                <<<dim3(n_heads, batch_lens->ne[0], N_PARTITIONS), dim3(WARP_SIZE, N_WARPS), 0, ctx.stream()>>>( \
                (const float *) q->data, (const float *) k_new->data, (const float *) v_new->data, \
                (char *) k_cache->data, (char *) v_cache->data, (const int *) block_table->data, \
                (const int *) write_rows->data, (const int *) context_lens->data, \
                (const int *) batch_offsets->data, k_cache->nb[1], k_cache->nb[2], k_cache->nb[3], \
                v_cache->nb[1], v_cache->nb[2], v_cache->nb[3], n_heads, n_heads_kv, block_size, max_blocks, \
                op_params_f[0], partials.ptr)
#define CONFIGURE_Q8_GROUPED(N_WARPS, N_PARTITIONS, N_Q_HEADS, USE_CP_ASYNC, FUSE_CURRENT) \
            do { \
                const char * carveout_kib_env = getenv("GGML_CUDA_PAGED_Q8_CARVEOUT_KIB"); \
                if (carveout_kib_env != nullptr && atoi(carveout_kib_env) > 0 && \
                        capture_status == cudaStreamCaptureStatusNone) { \
                    int shared_mem_per_sm = 0; \
                    CUDA_CHECK(cudaDeviceGetAttribute(&shared_mem_per_sm, \
                        cudaDevAttrMaxSharedMemoryPerMultiprocessor, ggml_cuda_get_device())); \
                    const int carveout_kib = atoi(carveout_kib_env); \
                    const int carveout_percent = std::max(0, std::min(100, \
                        (int) (((int64_t) carveout_kib * 1024 * 100 + shared_mem_per_sm - 1) / shared_mem_per_sm))); \
                    CUDA_CHECK(cudaFuncSetAttribute( \
                        paged_attention_decode_q8_grouped_partial_kernel< \
                            256, N_WARPS, N_PARTITIONS, N_Q_HEADS, 32, USE_CP_ASYNC, FUSE_CURRENT>, \
                        cudaFuncAttributePreferredSharedMemoryCarveout, carveout_percent)); \
                } \
            } while (0)
#define LAUNCH_Q8_GROUPED(N_WARPS, N_PARTITIONS, N_Q_HEADS, USE_CP_ASYNC, FUSE_CURRENT) \
            CONFIGURE_Q8_GROUPED(N_WARPS, N_PARTITIONS, N_Q_HEADS, USE_CP_ASYNC, FUSE_CURRENT); \
            paged_attention_decode_q8_grouped_partial_kernel< \
                256, N_WARPS, N_PARTITIONS, N_Q_HEADS, 32, USE_CP_ASYNC, FUSE_CURRENT> \
                <<<dim3(n_heads / N_Q_HEADS, batch_lens->ne[0], N_PARTITIONS), \
                    dim3(WARP_SIZE, N_WARPS), 0, ctx.stream()>>>( \
                (const float *) q->data, (const float *) k_new->data, (const float *) v_new->data, \
                (char *) k_cache->data, (char *) v_cache->data, (const int *) block_table->data, \
                (const int *) write_rows->data, (const int *) context_lens->data, \
                (const int *) batch_offsets->data, k_cache->nb[1], k_cache->nb[2], k_cache->nb[3], \
                v_cache->nb[1], v_cache->nb[2], v_cache->nb[3], n_heads, n_heads_kv, block_size, max_blocks, \
                op_params_f[0], partials.ptr)
#define LAUNCH_Q8_REDUCE(N_PARTITIONS) \
            paged_attention_decode_reduce_kernel<256, N_PARTITIONS> \
                <<<dim3(n_heads, batch_lens->ne[0]), dim3(256), 0, ctx.stream()>>>( \
                partials.ptr, (const int *) batch_offsets->data, n_heads, (float *) dst->data)
#define DISPATCH_Q8_PARTITIONS(LAUNCH_BODY, FUSE_CURRENT) \
            switch (variant.n_partitions) { \
                case 1: { LAUNCH_BODY(1, FUSE_CURRENT); LAUNCH_Q8_REDUCE(1); } break; \
                case 2: { LAUNCH_BODY(2, FUSE_CURRENT); LAUNCH_Q8_REDUCE(2); } break; \
                case 4: { LAUNCH_BODY(4, FUSE_CURRENT); LAUNCH_Q8_REDUCE(4); } break; \
                default: { LAUNCH_BODY(8, FUSE_CURRENT); LAUNCH_Q8_REDUCE(8); } break; \
            }

            if (variant.n_q_heads == 2) {
                const char * cp_async_env = getenv("GGML_CUDA_PAGED_Q8_CP_ASYNC");
                const bool use_cp_async = cp_async_available(cc) &&
                    (cp_async_env == nullptr || atoi(cp_async_env) != 0);
                if (use_cp_async) {
#define LAUNCH_GROUP_2_ASYNC(N_PARTITIONS, FUSE_CURRENT) \
                    LAUNCH_Q8_GROUPED(8, N_PARTITIONS, 2, true, FUSE_CURRENT)
                    if (fuse_current_q8) {
                        DISPATCH_Q8_PARTITIONS(LAUNCH_GROUP_2_ASYNC, true);
                    } else {
                        DISPATCH_Q8_PARTITIONS(LAUNCH_GROUP_2_ASYNC, false);
                    }
#undef LAUNCH_GROUP_2_ASYNC
                } else {
#define LAUNCH_GROUP_2(N_PARTITIONS, FUSE_CURRENT) \
                    LAUNCH_Q8_GROUPED(8, N_PARTITIONS, 2, false, FUSE_CURRENT)
                    if (fuse_current_q8) {
                        DISPATCH_Q8_PARTITIONS(LAUNCH_GROUP_2, true);
                    } else {
                        DISPATCH_Q8_PARTITIONS(LAUNCH_GROUP_2, false);
                    }
#undef LAUNCH_GROUP_2
                }
            } else if (variant.n_q_heads == 4) {
#define LAUNCH_GROUP_4(N_PARTITIONS, FUSE_CURRENT) \
                LAUNCH_Q8_GROUPED(16, N_PARTITIONS, 4, false, FUSE_CURRENT)
                if (fuse_current_q8) {
                    DISPATCH_Q8_PARTITIONS(LAUNCH_GROUP_4, true);
                } else {
                    DISPATCH_Q8_PARTITIONS(LAUNCH_GROUP_4, false);
                }
#undef LAUNCH_GROUP_4
            } else if (variant.n_warps == 4) {
#define LAUNCH_SINGLE_4(N_PARTITIONS, FUSE_CURRENT) \
                LAUNCH_Q8_PARTIAL(4, N_PARTITIONS, true, FUSE_CURRENT)
                if (fuse_current_q8) {
                    DISPATCH_Q8_PARTITIONS(LAUNCH_SINGLE_4, true);
                } else {
                    DISPATCH_Q8_PARTITIONS(LAUNCH_SINGLE_4, false);
                }
#undef LAUNCH_SINGLE_4
            } else if (variant.n_warps == 16) {
#define LAUNCH_SINGLE_16(N_PARTITIONS, FUSE_CURRENT) \
                LAUNCH_Q8_PARTIAL(16, N_PARTITIONS, true, FUSE_CURRENT)
                if (fuse_current_q8) {
                    DISPATCH_Q8_PARTITIONS(LAUNCH_SINGLE_16, true);
                } else {
                    DISPATCH_Q8_PARTITIONS(LAUNCH_SINGLE_16, false);
                }
#undef LAUNCH_SINGLE_16
            } else {
#define LAUNCH_SINGLE_8(N_PARTITIONS, FUSE_CURRENT) \
                LAUNCH_Q8_PARTIAL(8, N_PARTITIONS, true, FUSE_CURRENT)
                if (fuse_current_q8) {
                    DISPATCH_Q8_PARTITIONS(LAUNCH_SINGLE_8, true);
                } else {
                    DISPATCH_Q8_PARTITIONS(LAUNCH_SINGLE_8, false);
                }
#undef LAUNCH_SINGLE_8
            }
#undef DISPATCH_Q8_PARTITIONS
#undef LAUNCH_Q8_REDUCE
#undef LAUNCH_Q8_GROUPED
#undef CONFIGURE_Q8_GROUPED
#undef LAUNCH_Q8_PARTIAL
        }
#undef LAUNCH_Q8_LEGACY
        if (l2_window_active) {
            cudaStreamAttrValue attribute = {};
            attribute.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
            attribute.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
            CUDA_CHECK(cudaStreamSetAttribute(ctx.stream(), cudaStreamAttributeAccessPolicyWindow, &attribute));
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#endif

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (decode_turbo) {
        g_paged_turbo_decode_launch_count.fetch_add(1, std::memory_order_relaxed);
        GGML_ASSERT(!fuse_turbo_wht ||
            ((k_cache->type == GGML_TYPE_TURBO3_0 && v_cache->type == GGML_TYPE_TURBO3_0) ||
             (k_cache->type == GGML_TYPE_TURBO4_0 && v_cache->type == GGML_TYPE_TURBO4_0)));
        const ggml_paged_attn_cuda_device_caps caps = paged_attn_query_cuda_device_caps(ctx.stream());
        const char * token_vec_env = getenv("GGML_CUDA_PAGED_TURBO_TOKEN_VEC");
        const bool use_token_vec = (token_vec_env == nullptr || atoi(token_vec_env) != 0) &&
            ggml_paged_attn_cuda_turbo_token_vec_supported(
                context_bucket, head_dim, n_heads, n_heads_kv, batch_lens->ne[0], q->ne[2],
                k_cache->type, v_cache->type, caps);
        if (use_token_vec) {
            int n_partitions = 8;
            const char * partitions_env = getenv("GGML_CUDA_PAGED_TURBO_TOKEN_PARTITIONS");
            if (partitions_env != nullptr) {
                n_partitions = atoi(partitions_env);
            }
            if (n_partitions != 1 && n_partitions != 2 && n_partitions != 4 && n_partitions != 8) {
                GGML_LOG_WARN("%s: invalid Turbo token vec partition count %d; using 8\n",
                    __func__, n_partitions);
                n_partitions = 8;
            }
            g_paged_turbo_token_vec_launch_count.fetch_add(1, std::memory_order_relaxed);
            ggml_cuda_pool_alloc<float> partials(ctx.pool());
            partials.alloc((size_t) batch_lens->ne[0] * n_heads * n_partitions * (256 + 2));
#define LAUNCH_TURBO_TOKEN_VEC(N_PARTITIONS, TYPE, FUSE_TURBO_WHT) \
            paged_attention_decode_turbo_token_vec_kernel<256, N_PARTITIONS, TYPE, TYPE, FUSE_TURBO_WHT> \
                <<<dim3(n_heads, batch_lens->ne[0], N_PARTITIONS), dim3(WARP_SIZE, 4), 0, ctx.stream()>>>( \
                (const float *) q->data, (const char *) k_cache->data, (const char *) v_cache->data, \
                (const int *) block_table->data, (const int *) context_lens->data, \
                (const int *) batch_offsets->data, k_cache->nb[1], k_cache->nb[2], k_cache->nb[3], \
                v_cache->nb[1], v_cache->nb[2], v_cache->nb[3], n_heads, n_heads_kv, block_size, max_blocks, \
                op_params_f[0], partials.ptr); \
            paged_attention_decode_reduce_kernel<256, N_PARTITIONS, FUSE_TURBO_WHT> \
                <<<dim3(n_heads, batch_lens->ne[0]), dim3(256), 0, ctx.stream()>>>( \
                partials.ptr, (const int *) batch_offsets->data, n_heads, (float *) dst->data)
#define DISPATCH_TURBO_TOKEN_PARTITIONS(TYPE, FUSE_TURBO_WHT) \
            switch (n_partitions) { \
                case 1: LAUNCH_TURBO_TOKEN_VEC(1, TYPE, FUSE_TURBO_WHT); break; \
                case 2: LAUNCH_TURBO_TOKEN_VEC(2, TYPE, FUSE_TURBO_WHT); break; \
                case 8: LAUNCH_TURBO_TOKEN_VEC(8, TYPE, FUSE_TURBO_WHT); break; \
                default: LAUNCH_TURBO_TOKEN_VEC(4, TYPE, FUSE_TURBO_WHT); break; \
            }
            if (k_cache->type == GGML_TYPE_TURBO3_0) {
                if (fuse_turbo_wht) {
                    DISPATCH_TURBO_TOKEN_PARTITIONS(GGML_TYPE_TURBO3_0, true);
                } else {
                    DISPATCH_TURBO_TOKEN_PARTITIONS(GGML_TYPE_TURBO3_0, false);
                }
            } else {
                if (fuse_turbo_wht) {
                    DISPATCH_TURBO_TOKEN_PARTITIONS(GGML_TYPE_TURBO4_0, true);
                } else {
                    DISPATCH_TURBO_TOKEN_PARTITIONS(GGML_TYPE_TURBO4_0, false);
                }
            }
#undef DISPATCH_TURBO_TOKEN_PARTITIONS
#undef LAUNCH_TURBO_TOKEN_VEC
            CUDA_CHECK(cudaGetLastError());
            return;
        }

        const ggml_paged_attn_cuda_variant variant = ggml_paged_attn_select_cuda_variant(
            context_bucket, head_dim, n_heads, n_heads_kv, batch_lens->ne[0], q->ne[2],
            k_cache->type, v_cache->type, caps);
        int turbo_warps = variant.n_warps;
        const char * turbo_warps_env = getenv("GGML_CUDA_PAGED_TURBO_WARPS");
        if (turbo_warps_env != nullptr) {
            turbo_warps = atoi(turbo_warps_env);
        }
        if (turbo_warps != 4 && turbo_warps != 8 && turbo_warps != 16 && turbo_warps != 32) {
            GGML_LOG_WARN("%s: invalid Turbo decode warp count %d; using 32\n", __func__, turbo_warps);
            turbo_warps = 32;
        }
        ggml_cuda_pool_alloc<float> transformed_q(ctx.pool());
        ggml_cuda_pool_alloc<float> transformed_out(ctx.pool());
        const float * q_data = (const float *) q->data;
        float * out_data = (float *) dst->data;
        if (fuse_turbo_wht) {
            transformed_q.alloc(ggml_nelements(q));
            transformed_out.alloc(ggml_nelements(dst));
            ggml_cuda_turbo_wht_f32(
                ctx.stream(), q_data, transformed_q.ptr, ggml_nelements(q), 0);
            q_data = transformed_q.ptr;
            out_data = transformed_out.ptr;
        }
#define LAUNCH_TURBO_VEC(N_WARPS, TYPE_K, TYPE_V) \
        paged_attention_decode_turbo_vec_kernel<256, N_WARPS, TYPE_K, TYPE_V> \
            <<<dim3(n_heads, batch_lens->ne[0]), dim3(WARP_SIZE, N_WARPS), 0, ctx.stream()>>>( \
            q_data, (const char *) k_cache->data, (const char *) v_cache->data, \
            (const int *) block_table->data, (const int *) context_lens->data, \
            (const int *) batch_offsets->data, k_cache->nb[1], k_cache->nb[2], k_cache->nb[3], \
            v_cache->nb[1], v_cache->nb[2], v_cache->nb[3], n_heads, n_heads_kv, block_size, max_blocks, \
            op_params_f[0], out_data)
#define DISPATCH_TURBO_TYPES(N_WARPS) \
        if (k_cache->type == GGML_TYPE_TURBO3_0) { \
            if (v_cache->type == GGML_TYPE_TURBO3_0) { \
                LAUNCH_TURBO_VEC(N_WARPS, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0); \
            } else { \
                LAUNCH_TURBO_VEC(N_WARPS, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0); \
            } \
        } else if (v_cache->type == GGML_TYPE_TURBO3_0) { \
            LAUNCH_TURBO_VEC(N_WARPS, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0); \
        } else { \
            LAUNCH_TURBO_VEC(N_WARPS, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0); \
        }
        switch (turbo_warps) {
            case 4:
                DISPATCH_TURBO_TYPES(4);
                break;
            case 8:
                DISPATCH_TURBO_TYPES(8);
                break;
            case 16:
                DISPATCH_TURBO_TYPES(16);
                break;
            default:
                DISPATCH_TURBO_TYPES(32);
                break;
        }
#undef DISPATCH_TURBO_TYPES
#undef LAUNCH_TURBO_VEC
        if (fuse_turbo_wht) {
            ggml_cuda_turbo_wht_f32(
                ctx.stream(), out_data, (float *) dst->data, ggml_nelements(dst), 1);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#endif


    const int n_q_tiles = (q->ne[2] + 15) / 16;
    bool use_tiled_prefill = false;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    use_tiled_prefill = ggml_paged_attn_tiled_prefill_supported(
        head_dim, q->type, ggml_is_contiguous(q), ggml_cuda_is_aligned(q, sizeof(float4)), q->ne[2], batch_lens->ne[0],
        n_q_tiles, turing_mma_available(cc), k_cache->type, v_cache->type);
    GGML_LOG_DEBUG("%s: tiled prefill dispatch=%d\n", __func__, use_tiled_prefill);

    if (use_tiled_prefill) {
        const ggml_paged_attn_cuda_device_caps caps = paged_attn_query_cuda_device_caps(ctx.stream());
        bool cache_row_offsets = ggml_paged_attn_select_cuda_prefill_row_cache(
            context_bucket, head_dim, n_heads, n_heads_kv, batch_lens->ne[0], q->ne[2],
            k_cache->type, v_cache->type, caps);
        if (const char * env = getenv("GGML_CUDA_PAGED_PREFILL_ROW_CACHE")) {
            cache_row_offsets = atoi(env) != 0;
        }
        const size_t tiled_smem_bytes =
            (size_t) (3 * 16 * head_dim) * sizeof(half) + (size_t) (16 * 16) * (sizeof(float) + sizeof(half)) +
            (size_t) (2 * 16 * head_dim + 3 * 16) * sizeof(float);
        switch (head_dim) {
            case 128:
                paged_attention_prefill_mma_kernel<128, false><<<dim3(n_heads, batch_lens->ne[0], n_q_tiles), dim3(256), tiled_smem_bytes, ctx.stream()>>>(
                    (const float *) q->data, (const char *) k_cache->data, (const char *) v_cache->data,
                    (const int *) block_table->data, (const int *) context_lens->data,
                    (const int *) batch_offsets->data, (const int *) batch_lens->data,
                    k_cache->nb[1], k_cache->nb[2], k_cache->nb[3],
                    v_cache->nb[1], v_cache->nb[2], v_cache->nb[3],
                    n_heads_kv, block_size, max_blocks, k_cache->type, v_cache->type,
                    op_params_f[0], (float *) dst->data);
                break;
            case 256:
                if (cache_row_offsets) {
                    CUDA_CHECK(cudaFuncSetAttribute(paged_attention_prefill_mma_kernel<256, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, tiled_smem_bytes));
                    paged_attention_prefill_mma_kernel<256, true><<<dim3(n_heads, batch_lens->ne[0], n_q_tiles), dim3(512), tiled_smem_bytes, ctx.stream()>>>(
                        (const float *) q->data, (const char *) k_cache->data, (const char *) v_cache->data,
                        (const int *) block_table->data, (const int *) context_lens->data,
                        (const int *) batch_offsets->data, (const int *) batch_lens->data,
                        k_cache->nb[1], k_cache->nb[2], k_cache->nb[3],
                        v_cache->nb[1], v_cache->nb[2], v_cache->nb[3],
                        n_heads_kv, block_size, max_blocks, k_cache->type, v_cache->type,
                        op_params_f[0], (float *) dst->data);
                } else {
                    CUDA_CHECK(cudaFuncSetAttribute(paged_attention_prefill_mma_kernel<256, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, tiled_smem_bytes));
                    paged_attention_prefill_mma_kernel<256, false><<<dim3(n_heads, batch_lens->ne[0], n_q_tiles), dim3(512), tiled_smem_bytes, ctx.stream()>>>(
                        (const float *) q->data, (const char *) k_cache->data, (const char *) v_cache->data,
                        (const int *) block_table->data, (const int *) context_lens->data,
                        (const int *) batch_offsets->data, (const int *) batch_lens->data,
                        k_cache->nb[1], k_cache->nb[2], k_cache->nb[3],
                        v_cache->nb[1], v_cache->nb[2], v_cache->nb[3],
                        n_heads_kv, block_size, max_blocks, k_cache->type, v_cache->type,
                        op_params_f[0], (float *) dst->data);
                }
                break;
            default:
                GGML_ABORT("Invalid tiled paged attention head size");
        }
        CUDA_CHECK(cudaGetLastError());
        g_paged_prefill_launch_count.fetch_add(1, std::memory_order_relaxed);
        GGML_LOG_DEBUG("%s: launched paged_attention_prefill_mma_kernel head_dim=%d\n", __func__, head_dim);
    }
#endif

    const size_t smem_bytes = ((size_t) head_dim + 31) / 32 * sizeof(float);
    paged_attention_decode_kernel<<<dim3(n_heads, batch_lens->ne[0]), dim3(head_dim), smem_bytes, ctx.stream()>>>(
        (const float *) q->data,
        (const char *) k_cache->data,
        (const char *) v_cache->data,
        (const int *) block_table->data,
        (const int *) context_lens->data,
        (const int *) batch_offsets->data,
        (const int *) batch_lens->data,
        k_cache->nb[1], k_cache->nb[2], k_cache->nb[3],
        v_cache->nb[1], v_cache->nb[2], v_cache->nb[3],
        n_heads_kv, block_size, max_blocks, k_cache->type, v_cache->type,
        op_params_f[0], use_tiled_prefill, (float *) dst->data);
    CUDA_CHECK(cudaGetLastError());
}
