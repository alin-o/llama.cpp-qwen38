#include "pagedattn.cuh"

#include "ggml-paged-attn.h"
#include "turbo-quant.cuh"
#include <atomic>
#include <cmath>
#include <cstdio>

static std::atomic<unsigned long long> g_paged_prefill_launch_count{ 0 };

extern "C" unsigned long long ggml_paged_attn_tiled_prefill_launch_count(void) {
    return g_paged_prefill_launch_count.load(std::memory_order_relaxed);
}

extern "C" void ggml_paged_attn_tiled_prefill_launch_count_reset(void) {
    g_paged_prefill_launch_count.store(0, std::memory_order_relaxed);
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#if defined(TURING_MMA_AVAILABLE)
#include <mma.h>
namespace wmma = nvcuda::wmma;
#endif
#endif

static __device__ __forceinline__ float paged_cache_value(const char * row, int index, int type) {
    switch (type) {
        case GGML_TYPE_F16:
            return __half2float(((const half *) row)[index]);
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
#if defined(TURING_MMA_AVAILABLE)
template <int HEAD_DIM>
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

    const int tid = threadIdx.x;
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int q_tile_start = blockIdx.z * Q_TILE;
    const int n_heads = gridDim.x;
    const int num_new_tokens = batch_lens[seq_idx];
    if (num_new_tokens <= 1 || q_tile_start >= num_new_tokens) {
        return;
    }

    const int context_len = context_lens[seq_idx];
    const int kv_head_idx = ggml_paged_attn_kv_head(head_idx, n_heads, n_heads_kv);

    for (int vec = tid; vec < Q_TILE * HEAD_DIM / 4; vec += blockDim.x) {
        const int q_row = vec / (HEAD_DIM / 4);
        const int q_dim = (vec % (HEAD_DIM / 4)) * 4;
        const int q_token = q_tile_start + q_row;
        if (q_tile_start + q_row < num_new_tokens) {
            const size_t q_offset = (size_t) q_token * n_heads * HEAD_DIM + (size_t) head_idx * HEAD_DIM + q_dim;
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
        for (int vec = tid; vec < K_TILE * HEAD_DIM / 4; vec += blockDim.x) {
            const int key = vec / (HEAD_DIM / 4);
            const int dim = (vec % (HEAD_DIM / 4)) * 4;
            const int token = token_start + key;
            if (token < context_len) {
                const int physical_block = block_table[seq_idx * max_blocks + token / block_size];
                const int token_in_block = token % block_size;
                const char * k_row = k_cache + (size_t) physical_block * k_stride_block + (size_t) kv_head_idx * k_stride_head + (size_t) token_in_block * k_stride_token;
                const char * v_row = v_cache + (size_t) physical_block * v_stride_block + (size_t) kv_head_idx * v_stride_head + (size_t) token_in_block * v_stride_token;
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
            const size_t out_offset = (size_t) q_token * n_heads * HEAD_DIM + (size_t) head_idx * HEAD_DIM + dim;
            const float4 acc = paged_load_float4(&value_acc_shared[q_row][dim]);
            const float inv_sum = 1.0f / (sum_shared[q_row] + 1e-6f);
            paged_store_float4(out + out_offset, make_float4(
                acc.x * inv_sum, acc.y * inv_sum, acc.z * inv_sum, acc.w * inv_sum));
        }
    }
}
#else
template <int HEAD_DIM>
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
        n_heads_kv, block_size, max_blocks, k_type, v_type, scale, out);
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
        ok = cudaFuncSetAttribute(paged_attention_prefill_mma_kernel<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes) == cudaSuccess;
    }
    paged_attention_prefill_mma_kernel<HEAD_DIM><<<dim3(n_heads, 1, 2), dim3(HEAD_DIM == 256 ? 512 : 256), smem_bytes>>>(
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
    return type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0;
}

void ggml_cuda_op_paged_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k_new = dst->src[1];
    const ggml_tensor * k_cache = dst->src[3];
    const ggml_tensor * v_cache = dst->src[4];
    const ggml_tensor * block_table = dst->src[5];
    const ggml_tensor * context_lens = dst->src[7];
    const ggml_tensor * batch_offsets = dst->src[8];
    const ggml_tensor * batch_lens = dst->src[9];
    const float * op_params_f = (const float *) dst->op_params;
    const int block_size = ((const int32_t *) (op_params_f + 1))[0];
    const int max_blocks = ((const int32_t *) (op_params_f + 2))[0];
    const int head_dim = q->ne[0];
    const int n_heads = q->ne[1];
    const int n_heads_kv = k_new->ne[1];

    GGML_ASSERT(n_heads != 0 && n_heads_kv != 0 && head_dim <= 256);
    GGML_ASSERT(n_heads % n_heads_kv == 0);
    GGML_ASSERT(paged_kv_type_supported(k_cache->type) && paged_kv_type_supported(v_cache->type));

    const int n_q_tiles = (q->ne[2] + 15) / 16;
    bool use_tiled_prefill = false;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    use_tiled_prefill = ggml_paged_attn_tiled_prefill_supported(
        head_dim, q->type, ggml_is_contiguous(q), ggml_cuda_is_aligned(q, sizeof(float4)), q->ne[2], batch_lens->ne[0],
        n_q_tiles, turing_mma_available(cc), k_cache->type, v_cache->type);
    GGML_LOG_DEBUG("%s: tiled prefill dispatch=%d\n", __func__, use_tiled_prefill);

    if (use_tiled_prefill) {
        const size_t tiled_smem_bytes =
            (size_t) (3 * 16 * head_dim) * sizeof(half) + (size_t) (16 * 16) * (sizeof(float) + sizeof(half)) +
            (size_t) (2 * 16 * head_dim + 3 * 16) * sizeof(float);
        switch (head_dim) {
            case 128:
                paged_attention_prefill_mma_kernel<128><<<dim3(n_heads, batch_lens->ne[0], n_q_tiles), dim3(256), tiled_smem_bytes, ctx.stream()>>>(
                    (const float *) q->data, (const char *) k_cache->data, (const char *) v_cache->data,
                    (const int *) block_table->data, (const int *) context_lens->data,
                    (const int *) batch_offsets->data, (const int *) batch_lens->data,
                    k_cache->nb[1], k_cache->nb[2], k_cache->nb[3],
                    v_cache->nb[1], v_cache->nb[2], v_cache->nb[3],
                    n_heads_kv, block_size, max_blocks, k_cache->type, v_cache->type,
                    op_params_f[0], (float *) dst->data);
                break;
            case 256:
                CUDA_CHECK(cudaFuncSetAttribute(paged_attention_prefill_mma_kernel<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, tiled_smem_bytes));
                paged_attention_prefill_mma_kernel<256><<<dim3(n_heads, batch_lens->ne[0], n_q_tiles), dim3(512), tiled_smem_bytes, ctx.stream()>>>(
                    (const float *) q->data, (const char *) k_cache->data, (const char *) v_cache->data,
                    (const int *) block_table->data, (const int *) context_lens->data,
                    (const int *) batch_offsets->data, (const int *) batch_lens->data,
                    k_cache->nb[1], k_cache->nb[2], k_cache->nb[3],
                    v_cache->nb[1], v_cache->nb[2], v_cache->nb[3],
                    n_heads_kv, block_size, max_blocks, k_cache->type, v_cache->type,
                    op_params_f[0], (float *) dst->data);
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
