#include "pagedattn.cuh"

#include "ggml-paged-attn.h"
#include "turbo-quant.cuh"

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
            const char4 qs = *(const char4 *) (block->qs + index % QK8_0);
            const float d = __half2float(block->d);
            return make_float4(d * qs.x, d * qs.y, d * qs.z, d * qs.w);
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
    const int kv_head_idx = head_idx / (n_heads / n_heads_kv);
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

    __shared__ half q_shared[Q_TILE][HEAD_DIM];
    __shared__ half k_shared[K_TILE][HEAD_DIM];
    __shared__ half v_shared[HEAD_DIM][K_TILE];
    __shared__ float scores_shared[Q_TILE][K_TILE];
    __shared__ half weights_shared[Q_TILE][K_TILE];
    __shared__ float values_shared[Q_TILE][HEAD_DIM];
    __shared__ float value_acc_shared[Q_TILE][HEAD_DIM];
    __shared__ float max_shared[Q_TILE];
    __shared__ float sum_shared[Q_TILE];
    __shared__ float rescale_shared[Q_TILE];

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
    const int kv_head_idx = head_idx / (n_heads / n_heads_kv);

    for (int vec = tid; vec < Q_TILE * HEAD_DIM / 4; vec += blockDim.x) {
        const int q_row = vec / (HEAD_DIM / 4);
        const int q_dim = (vec % (HEAD_DIM / 4)) * 4;
        const int q_token = context_len - num_new_tokens + q_tile_start + q_row;
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
        const int q_token = context_len - num_new_tokens + q_tile_start + q_row;
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
        paged_attention_prefill_mma_kernel<128><<<dim3(n_heads, batch_lens->ne[0], n_q_tiles), dim3(256), 0, ctx.stream()>>>(
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
            op_params_f[0], (float *) dst->data);
        CUDA_CHECK(cudaGetLastError());
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
