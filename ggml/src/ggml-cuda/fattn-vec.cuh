#include "common.cuh"
#include "fattn-common.cuh"

static int ggml_cuda_fattn_vec_get_nthreads_host(const int cc) {
    return 128;
    GGML_UNUSED(cc);
}

static constexpr __device__ int ggml_cuda_fattn_vec_get_nthreads_device() {
    return 128;
}

// Currently llvm with the amdgcn target does not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
extern "C" void ggml_cuda_fattn_shared_turbo_decode_launch_count_add(void);

template<int D, int nwarps, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
__global__ __launch_bounds__(WARP_SIZE * nwarps, 1) void flash_attn_ext_decode_vec_shared(
        const char * Q, const char * K, const char * V, const char * mask, const char * sinks, float * dst,
        float scale, float max_bias, float m0, float m1, uint32_t n_head_log2, float logit_softcap,
        int n_heads, int n_heads_kv, int n_sequences, int n_tokens,
        int32_t nb02, int32_t nb03, int32_t nb11, int32_t nb12, int64_t nb13,
        int32_t nb21, int32_t nb22, int64_t nb23, int32_t ne33, int64_t nb33) {
    __shared__ fattn_decode_vec_shared<D, nwarps> shared;
    const int head = blockIdx.x;
    const int sequence = blockIdx.y;
    const int gqa_ratio = n_heads / n_heads_kv;
    const int kv_head = head / gqa_ratio;
    const fattn_contiguous_kv_address kv_address = {
        K + (size_t) sequence * nb13 + (size_t) kv_head * nb12,
        V + (size_t) sequence * nb23 + (size_t) kv_head * nb22,
        (size_t) nb11,
        (size_t) nb21,
    };
    const half * mask_row = mask == nullptr ? nullptr :
        (const half *) (mask + (size_t) (sequence % ne33) * nb33);
    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);
    const fattn_decode_contiguous_score<use_logit_softcap> score = { mask_row, slope, logit_softcap };
    const float * sink = sinks == nullptr ? nullptr : (const float *) sinks + head;
    const float * q = (const float *) (Q + (size_t) sequence * nb03 + (size_t) head * nb02);
    float * out = dst + ((size_t) sequence * n_heads + head) * D;
    fattn_decode_vec_core<D, nwarps, type_K, type_V>(
        q, kv_address, score, sink, n_tokens, scale, out, shared);

    GGML_UNUSED(n_sequences);
}

template<int D, int nwarps, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
__global__ __launch_bounds__(WARP_SIZE * nwarps, 1) void flash_attn_ext_decode_vec_shared_batch(
        const char * Q, const char * K, const char * V, const char * mask, const char * sinks, float * dst,
        float scale, float max_bias, float m0, float m1, uint32_t n_head_log2, float logit_softcap,
        int n_heads, int n_heads_kv, int n_sequences, int n_tokens,
        int32_t nb01, int32_t nb02, int32_t nb03, int32_t nb11, int32_t nb12, int64_t nb13,
        int32_t nb21, int32_t nb22, int64_t nb23, int32_t nb31, int32_t ne33, int64_t nb33,
        int max_shared_tokens) {
    __shared__ fattn_decode_vec_shared<D, nwarps> shared;
    const int head = blockIdx.x;
    const int sequence = blockIdx.y;
    const int query = blockIdx.z;
    const int gqa_ratio = n_heads / n_heads_kv;
    const int kv_head = head / gqa_ratio;
    const fattn_contiguous_kv_address kv_address = {
        K + (size_t) sequence * nb13 + (size_t) kv_head * nb12,
        V + (size_t) sequence * nb23 + (size_t) kv_head * nb22,
        (size_t) nb11,
        (size_t) nb21,
    };
    const half * mask_row = mask == nullptr ? nullptr :
        (const half *) (mask + (size_t) (sequence % ne33) * nb33 + (size_t) query * nb31);
    int visible_tokens = n_tokens;
    if (gridDim.z > 1) {
        while (mask_row != nullptr && visible_tokens > 0 && isinf(__half2float(mask_row[visible_tokens - 1]))) {
            --visible_tokens;
        }
    }
    if (visible_tokens > max_shared_tokens) {
        return;
    }
    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);
    const fattn_decode_contiguous_score<use_logit_softcap> score = { mask_row, slope, logit_softcap };
    const float * sink = sinks == nullptr ? nullptr : (const float *) sinks + head;
    const float * q = (const float *) (Q + (size_t) sequence * nb03 + (size_t) head * nb02 + (size_t) query * nb01);
    float * out = dst + (((size_t) sequence * gridDim.z + query) * n_heads + head) * D;
    fattn_decode_vec_core<D, nwarps, type_K, type_V>(
        q, kv_address, score, sink, visible_tokens, scale, out, shared);

    GGML_UNUSED(n_sequences);
}

template<int D, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
static void launch_flash_attn_ext_decode_vec_shared(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, int max_shared_tokens) {
    constexpr int nwarps = 32;
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if constexpr (use_logit_softcap) {
        scale /= logit_softcap;
    }

    const uint32_t n_head = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));
    const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    ggml_cuda_fattn_shared_turbo_decode_launch_count_add();
    if (Q->ne[1] == 1) {
        flash_attn_ext_decode_vec_shared<D, nwarps, type_K, type_V, use_logit_softcap>
            <<<dim3(Q->ne[2], Q->ne[3]), dim3(WARP_SIZE, nwarps), 0, ctx.stream()>>>(
            (const char *) Q->data, (const char *) K->data, (const char *) V->data,
            mask == nullptr ? nullptr : (const char *) mask->data,
            sinks == nullptr ? nullptr : (const char *) sinks->data, (float *) dst->data,
            scale, max_bias, m0, m1, n_head_log2, logit_softcap,
            Q->ne[2], K->ne[2], Q->ne[3], K->ne[1], Q->nb[2], Q->nb[3],
            K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3],
            mask == nullptr ? 0 : mask->ne[3], mask == nullptr ? 0 : mask->nb[3]);
    } else {
        flash_attn_ext_decode_vec_shared_batch<D, nwarps, type_K, type_V, use_logit_softcap>
            <<<dim3(Q->ne[2], Q->ne[3], Q->ne[1]), dim3(WARP_SIZE, nwarps), 0, ctx.stream()>>>(
            (const char *) Q->data, (const char *) K->data, (const char *) V->data,
            mask == nullptr ? nullptr : (const char *) mask->data,
            sinks == nullptr ? nullptr : (const char *) sinks->data, (float *) dst->data,
            scale, max_bias, m0, m1, n_head_log2, logit_softcap,
            Q->ne[2], K->ne[2], Q->ne[3], K->ne[1], Q->nb[1], Q->nb[2], Q->nb[3],
            K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3],
            mask == nullptr ? 0 : mask->nb[1], mask == nullptr ? 0 : mask->ne[3],
            mask == nullptr ? 0 : mask->nb[3], max_shared_tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}
#endif

template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // D == head size
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), 1)
static __global__ void flash_attn_ext_vec(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_lc();
#ifdef FLASH_ATTN_AVAILABLE
    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

#ifdef GGML_USE_HIP
#ifdef RDNA
    constexpr int nthreads_KQ_q = 2;
#else
    constexpr int nthreads_KQ_q = 4;
#endif // RDNA
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#else
    constexpr int nthreads_KQ_q = (D/4 < 32 ? D/4 : 32);
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#endif // GGML_USE_HIP

    constexpr int nthreads    = ggml_cuda_fattn_vec_get_nthreads_device();
    constexpr bool K_uses_float_q = type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_BF16 || type_K == GGML_TYPE_TURBO3_0 || type_K == GGML_TYPE_TURBO4_0;
    constexpr int nthreads_KQ = K_uses_float_q ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr int nthreads_V  = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 128 / cpy_nb : nthreads_V_q;

    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");

    constexpr int V_rows_per_thread = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 2*cpy_ne : 4;
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ>();
    constexpr bool Q_q8_1 = !K_uses_float_q;
#ifdef V_DOT2_F32_F16_AVAILABLE
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, half,  V_rows_per_thread>();
#else
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, float, V_rows_per_thread>();
#endif // V_DOT2_F32_F16_AVAILABLE

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const half * maskh  = (const half  *) (mask + nb33*(sequence % ne33) + nb31*ic0);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = nthreads / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    constexpr int ne_KQ      = ncols*D;
    constexpr int ne_combine = nwarps*V_cols_per_iter*D;
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2            VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ half   KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#else
    float2           VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ float  KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#endif // V_DOT2_F32_F16_AVAILABLE

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2  Q_reg[ncols][(D/2)/nthreads_KQ]; // Will be initialized completely.
#else
    __align__(16) float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // V_DOT2_F32_F16_AVAILABLE
    int    Q_i32[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    float2  Q_ds[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];

    ggml_cuda_pdl_sync();
    if constexpr (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 1 && ic0 + j >= int(ne01.z)) {
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    if (i0 + WARP_SIZE <= int(D/sizeof(int)) || i < int(D/sizeof(int))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
            } else {
                const float * Q_f = (const float *) (Q + j*nb01);
                constexpr int nthreads_quantize = D/sizeof(int) < WARP_SIZE ? D/sizeof(int) : WARP_SIZE;
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<float2, nthreads_quantize>
                        (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#ifdef V_DOT2_F32_F16_AVAILABLE
        const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;

                __align__(16) float2 tmp[cpy_ne] = {{0.0f, 0.0f}};
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0/nthreads_KQ + i1] = make_half2(tmp[i1].x, tmp[i1].y);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#else
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x *= scale;
                Q_reg[j][k].y *= scale;
            }
        }
#endif // V_DOT2_F32_F16_AVAILABLE
    }

    int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    if (ncols == 1 && int(ne01.z) > 1 && mask) {
        while (k_VKQ_max > 0 && isinf(__half2float(maskh[k_VKQ_max - 1]))) {
            --k_VKQ_max;
        }
    }
    // Match the reduction geometry of a single causal query when verification rows cross a KV tile boundary.
    // Blocks beyond this query's visible range contribute a neutral partial result to the shared combine pass.
    const int parallel_blocks = ncols == 1 && int(ne01.z) > 1 ?
        min((int) gridDim.y, max(1, (k_VKQ_max + D - 1) / D)) : gridDim.y;
    K     += blockIdx.y*nthreads * nb11;
    V     += blockIdx.y*nthreads * nb21;
    maskh += blockIdx.y*nthreads;
    for (int k_VKQ_0 = blockIdx.y*nthreads; blockIdx.y < parallel_blocks && k_VKQ_0 < k_VKQ_max; k_VKQ_0 += parallel_blocks*nthreads,
             // Increment pointers after each loop:
             K += parallel_blocks*nthreads*nb11, V += parallel_blocks*nthreads*nb21, maskh += parallel_blocks*nthreads) {

        const fattn_contiguous_kv_address kv_address = { K, V, (size_t) nb11, (size_t) nb21 };

        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[ncols]; // KQ in registers.

        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ-1))) + i_KQ_0;

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                const fattn_kv_rows rows = kv_address.rows(i_KQ);
                float sum = vec_dot_KQ(rows.k, Q_reg[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                if (mask && (ncols == 1 || ic0 + j < int(ne01.z))) {
                    sum += slope*__half2float(maskh[j*ne11 + i_KQ]);
                }

                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);

                if ((nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ) == uint32_t(i_KQ_0)) {
                    KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = fattn_softmax_rescale(KQ_max[j], KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }

#ifndef GGML_USE_HIP
        __syncwarp();
#endif // GGML_USE_HIP

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);

#ifdef V_DOT2_F32_F16_AVAILABLE
            half2 KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = __half2half2(KQ[j*nthreads + k]);
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                half2 tmp[V_rows_per_thread/2];
                if constexpr (type_V == GGML_TYPE_BF16) {
                    float2 tmp_f[V_rows_per_thread/2];
                    dequantize_V(kv_address.rows(k).v, tmp_f,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                    for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
                        tmp[i_VKQ_1] = __float22half2_rn(tmp_f[i_VKQ_1]);
                    }
                } else {
                    dequantize_V(kv_address.rows(k).v, tmp,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
                }
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#else
            float KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                float2 tmp[V_rows_per_thread/2];
                dequantize_V(kv_address.rows(k).v, tmp,
                    2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y*KQ_k[j];
                    }
                }
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    if (sinks && blockIdx.y == 0) {
        const float sink = ((const float *) sinks)[head];

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            const float kqmax_new_j = fmaxf(sink, KQ_max[j]);
            const float KQ_max_scale = fattn_softmax_rescale(KQ_max[j], kqmax_new_j);
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + (threadIdx.x == 0 ? expf(sink - KQ_max[j]) : 0.0f);

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 1 && ic0 + j_VKQ >= int(ne01.z)) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = fattn_softmax_rescale(KQ_max[j_VKQ], kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

#ifdef V_DOT2_F32_F16_AVAILABLE
        half2 * VKQ_tmp = (half2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

        const half2 kqmax_scale_h2 = make_half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread*sizeof(half)>(VKQ_tmp + i_VKQ, &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
        }
#else
        float2 * VKQ_tmp = (float2 *) KQ + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // V_DOT2_F32_F16_AVAILABLE

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }

        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                if (gridDim.y == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence*int(ne01.z) + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;
            }
        }

        if (j_VKQ < ncols-1) {
            __syncthreads();
        }

    }

    if (gridDim.y != 1 && tid < ncols && (ncols == 1 || ic0 + tid < int(ne01.z))) {
        dst_meta[((sequence*int(ne01.z) + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(cc);
    const int nwarps   = nthreads / WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap>;
    const bool need_f16_K = type_K == GGML_TYPE_F16;
    const bool need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<D, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];
    const bool token_sequential = ggml_get_op_params_i32(KQV, 4) != 0;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (D == 256 &&
            (type_K == GGML_TYPE_TURBO3_0 || type_K == GGML_TYPE_TURBO4_0) &&
            (type_V == GGML_TYPE_TURBO3_0 || type_V == GGML_TYPE_TURBO4_0)) {
        const char * shared_decode_env = getenv("GGML_CUDA_FATTN_TURBO_SHARED");
        const bool force_shared_decode = shared_decode_env != nullptr && atoi(shared_decode_env) != 0;
        const bool use_shared_decode = shared_decode_env != nullptr ?
            force_shared_decode : dst->src[1]->ne[1] <= 256;
        if (Q->ne[1] == 1 && use_shared_decode) {
            if (logit_softcap == 0.0f) {
                launch_flash_attn_ext_decode_vec_shared<D, type_K, type_V, false>(ctx, dst, INT_MAX);
            } else {
                launch_flash_attn_ext_decode_vec_shared<D, type_K, type_V, true>(ctx, dst, INT_MAX);
            }
            return;
        }

        const bool use_shared_batch = shared_decode_env == nullptr || force_shared_decode;
        if (token_sequential && Q->ne[1] > 1 && Q->ne[1] <= 4 && use_shared_batch &&
                (force_shared_decode || dst->src[1]->ne[1] <= 256)) {
            if (logit_softcap == 0.0f) {
                launch_flash_attn_ext_decode_vec_shared<D, type_K, type_V, false>(
                    ctx, dst, force_shared_decode ? INT_MAX : 256);
            } else {
                launch_flash_attn_ext_decode_vec_shared<D, type_K, type_V, true>(
                    ctx, dst, force_shared_decode ? INT_MAX : 256);
            }
            return;
        }
    }
#endif

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (!token_sequential || Q->ne[1] > 4) {
        constexpr int cols_per_block = 2;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 1;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (D == 256 &&
            (type_K == GGML_TYPE_TURBO3_0 || type_K == GGML_TYPE_TURBO4_0) &&
            (type_V == GGML_TYPE_TURBO3_0 || type_V == GGML_TYPE_TURBO4_0)) {
        const char * shared_decode_env = getenv("GGML_CUDA_FATTN_TURBO_SHARED");
        const bool force_shared_decode = shared_decode_env != nullptr && atoi(shared_decode_env) != 0;
        const bool use_shared_decode = shared_decode_env == nullptr || force_shared_decode;
        if (use_shared_decode) {
            if (logit_softcap == 0.0f) {
                launch_flash_attn_ext_decode_vec_shared<D, type_K, type_V, false>(
                    ctx, dst, force_shared_decode ? INT_MAX : 256);
            } else {
                launch_flash_attn_ext_decode_vec_shared<D, type_K, type_V, true>(
                    ctx, dst, force_shared_decode ? INT_MAX : 256);
            }
        }
    }
#endif
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template void ggml_cuda_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_BF16); \

EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_BF16)

#define EXTERN_DECL_FATTN_VEC_TURBO3(D, type_K, type_V) \
    extern DECL_FATTN_VEC_CASE(D, type_K, type_V);

EXTERN_DECL_FATTN_VEC_TURBO3( 64, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)
EXTERN_DECL_FATTN_VEC_TURBO3(128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)
EXTERN_DECL_FATTN_VEC_TURBO3(256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)
EXTERN_DECL_FATTN_VEC_TURBO3( 64, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_TURBO3(128, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_TURBO3(256, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_TURBO3( 64, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0)
EXTERN_DECL_FATTN_VEC_TURBO3(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0)
EXTERN_DECL_FATTN_VEC_TURBO3(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0)

#define EXTERN_DECL_FATTN_VEC_TURBO4(D, type_K, type_V) \
    extern DECL_FATTN_VEC_CASE(D, type_K, type_V);

EXTERN_DECL_FATTN_VEC_TURBO4( 64, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)
EXTERN_DECL_FATTN_VEC_TURBO4(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)
EXTERN_DECL_FATTN_VEC_TURBO4(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)
EXTERN_DECL_FATTN_VEC_TURBO4( 64, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_TURBO4(128, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_TURBO4(256, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_TURBO4( 64, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0)
EXTERN_DECL_FATTN_VEC_TURBO4(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0)
EXTERN_DECL_FATTN_VEC_TURBO4(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0)
