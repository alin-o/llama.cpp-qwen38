#include "turbo-quant.cuh"
#include "turbo-wht.cuh"

template <int direction>
static __global__ void turbo3_wht_f32(
        const float * __restrict__ src,
        float * __restrict__ dst,
        int64_t n_groups) {
    const int64_t group = blockIdx.x;
    if (group >= n_groups) {
        return;
    }

    const int lane = threadIdx.x;
    __shared__ float values[QK_TURBO3_GROUP];
    values[lane] = src[group*QK_TURBO3_GROUP + lane]
            *((direction == 0) ? TURBO3_WHT_SIGNS_1[lane] : TURBO3_WHT_SIGNS_2[lane]);
    __syncthreads();

#define TURBO3_WHT_STAGE(h) \
    if (lane % (2*(h)) < (h)) { \
        const float a = values[lane]; \
        const float b = values[lane + (h)]; \
        values[lane] = a + b; \
        values[lane + (h)] = a - b; \
    } \
    __syncthreads();

    TURBO3_WHT_STAGE(1)
    TURBO3_WHT_STAGE(2)
    TURBO3_WHT_STAGE(4)
    TURBO3_WHT_STAGE(8)
    TURBO3_WHT_STAGE(16)
    TURBO3_WHT_STAGE(32)
    TURBO3_WHT_STAGE(64)
#undef TURBO3_WHT_STAGE

    const float sign = (direction == 0) ? TURBO3_WHT_SIGNS_2[lane] : TURBO3_WHT_SIGNS_1[lane];
    dst[group*QK_TURBO3_GROUP + lane] = values[lane]*0.08838834764831845f*sign;
}

void ggml_cuda_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src->ne[0] % QK_TURBO3_GROUP == 0);

    int direction = 0;
    memcpy(&direction, dst->op_params, sizeof(direction));
    const int64_t n_groups = ggml_nelements(src)/QK_TURBO3_GROUP;
    const dim3 blocks(n_groups);
    const dim3 threads(QK_TURBO3_GROUP);
    if (direction == 0) {
        turbo3_wht_f32<0><<<blocks, threads, 0, ctx.stream()>>>(
                (const float *) src->data, (float *) dst->data, n_groups);
    } else {
        turbo3_wht_f32<1><<<blocks, threads, 0, ctx.stream()>>>(
                (const float *) src->data, (float *) dst->data, n_groups);
    }
}
