#include "pagedattn.cuh"

#include "turbo-quant.cuh"

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
        op_params_f[0], (float *) dst->data);
}
