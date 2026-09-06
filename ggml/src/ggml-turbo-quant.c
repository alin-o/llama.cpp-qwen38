#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <assert.h>
#include <math.h>
#include <string.h>

static const float turbo3_centroids[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f,
};

static const float turbo3_signs_1[QK_TURBO3_GROUP] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,
};

static const float turbo3_signs_2[QK_TURBO3_GROUP] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1,
};

static int turbo3_nearest(float x) {
    if (x < -0.154259f) return 0;
    if (x < -0.091775f) return 1;
    if (x < -0.043589f) return 2;
    if (x <  0.000000f) return 3;
    if (x <  0.043589f) return 4;
    if (x <  0.091775f) return 5;
    if (x <  0.154259f) return 6;
    return 7;
}

static void turbo3_wht(float * x) {
    for (int i = 0; i < QK_TURBO3_GROUP; ++i) {
        x[i] *= turbo3_signs_1[i];
    }
    for (int h = 1; h < QK_TURBO3_GROUP; h *= 2) {
        for (int i = 0; i < QK_TURBO3_GROUP; i += 2*h) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j];
                const float b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < QK_TURBO3_GROUP; ++i) {
        x[i] *= 0.08838834764831845f * turbo3_signs_2[i];
    }
}

void quantize_row_turbo3_0_ref(
        const float * GGML_RESTRICT x,
        block_turbo3_0 * GGML_RESTRICT y,
        int64_t k) {
    assert(k % QK_TURBO3_GROUP == 0);

    for (int64_t g = 0; g < k/QK_TURBO3_GROUP; ++g) {
        float values[QK_TURBO3_GROUP];
        float norm2 = 0.0f;
        for (int j = 0; j < QK_TURBO3_GROUP; ++j) {
            values[j] = x[g*QK_TURBO3_GROUP + j];
            norm2 += values[j]*values[j];
        }

        const float norm = sqrtf(norm2);
        const float inv_norm = norm > 1e-10f ? 1.0f/norm : 0.0f;
        for (int j = 0; j < QK_TURBO3_GROUP; ++j) {
            values[j] *= inv_norm;
        }
        turbo3_wht(values);

        block_turbo3_0 * block = &y[g];
        memset(block->qs, 0, sizeof(block->qs));
        memset(block->signs, 0, sizeof(block->signs));

        float recon2 = 0.0f;
        for (int j = 0; j < QK_TURBO3; ++j) {
            const int index = turbo3_nearest(values[j]);
            block->qs[j/4] |= (index & 3) << (2*(j % 4));
            block->signs[j/8] |= ((index >> 2) & 1) << (j % 8);
            recon2 += turbo3_centroids[index]*turbo3_centroids[index];
        }

        const float recon = sqrtf(recon2);
        block->norm = GGML_FP32_TO_FP16(recon > 1e-10f ? norm/recon : norm);
    }
}

void dequantize_row_turbo3_0(
        const block_turbo3_0 * GGML_RESTRICT x,
        float * GGML_RESTRICT y,
        int64_t k) {
    assert(k % QK_TURBO3 == 0);

    for (int64_t b = 0; b < k/QK_TURBO3; ++b) {
        const float norm = GGML_FP16_TO_FP32(x[b].norm);
        for (int j = 0; j < QK_TURBO3; ++j) {
            const int low = (x[b].qs[j/4] >> (2*(j % 4))) & 3;
            const int high = (x[b].signs[j/8] >> (j % 8)) & 1;
            y[b*QK_TURBO3 + j] = norm*turbo3_centroids[low | (high << 2)];
        }
    }
}

size_t quantize_turbo3_0(
        const float * GGML_RESTRICT src,
        void * GGML_RESTRICT dst,
        int64_t nrows,
        int64_t n_per_row,
        const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3 == 0);

    const size_t row_size = n_per_row/QK_TURBO3*sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_turbo3_0_ref(
                src + row*n_per_row,
                (block_turbo3_0 *) ((char *) dst + row*row_size),
                n_per_row);
    }
    return nrows*row_size;
}

static const float turbo4_centroids[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f,
};

static int turbo4_nearest(float x) {
    if (x < -0.145561f) return 0;
    if (x < -0.103361f) return 1;
    if (x < -0.079142f) return 2;
    if (x < -0.060009f) return 3;
    if (x < -0.043430f) return 4;
    if (x < -0.028293f) return 5;
    if (x < -0.013964f) return 6;
    if (x <  0.000000f) return 7;
    if (x <  0.013964f) return 8;
    if (x <  0.028293f) return 9;
    if (x <  0.043430f) return 10;
    if (x <  0.060009f) return 11;
    if (x <  0.079142f) return 12;
    if (x <  0.103361f) return 13;
    if (x <  0.145561f) return 14;
    return 15;
}

void quantize_row_turbo4_0_ref(
        const float * GGML_RESTRICT x,
        block_turbo4_0 * GGML_RESTRICT y,
        int64_t k) {
    assert(k % QK_TURBO4_GROUP == 0);

    for (int64_t g = 0; g < k/QK_TURBO4_GROUP; ++g) {
        float values[QK_TURBO4_GROUP];
        float norm2 = 0.0f;
        for (int j = 0; j < QK_TURBO4_GROUP; ++j) {
            values[j] = x[g*QK_TURBO4_GROUP + j];
            norm2 += values[j]*values[j];
        }

        const float norm = sqrtf(norm2);
        const float inv_norm = norm > 1e-10f ? 1.0f/norm : 0.0f;
        for (int j = 0; j < QK_TURBO4_GROUP; ++j) {
            values[j] *= inv_norm;
        }
        turbo3_wht(values);

        block_turbo4_0 * block = &y[g];
        memset(block->qs, 0, sizeof(block->qs));

        float recon2 = 0.0f;
        for (int j = 0; j < QK_TURBO4; ++j) {
            const int index = turbo4_nearest(values[j]);
            block->qs[j/2] |= (index & 0xF) << (4*(j % 2));
            recon2 += turbo4_centroids[index]*turbo4_centroids[index];
        }

        const float recon = sqrtf(recon2);
        block->norm  = GGML_FP32_TO_FP16(recon > 1e-10f ? norm/recon : norm);
        block->rnorm = GGML_FP32_TO_FP16(0.0f);
    }
}

void dequantize_row_turbo4_0(
        const block_turbo4_0 * GGML_RESTRICT x,
        float * GGML_RESTRICT y,
        int64_t k) {
    assert(k % QK_TURBO4 == 0);

    for (int64_t b = 0; b < k/QK_TURBO4; ++b) {
        const float norm = GGML_FP16_TO_FP32(x[b].norm);
        for (int j = 0; j < QK_TURBO4; ++j) {
            const int index = (x[b].qs[j/2] >> (4*(j % 2))) & 0xF;
            y[b*QK_TURBO4 + j] = norm*turbo4_centroids[index];
        }
    }
}

size_t quantize_turbo4_0(
        const float * GGML_RESTRICT src,
        void * GGML_RESTRICT dst,
        int64_t nrows,
        int64_t n_per_row,
        const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4 == 0);

    const size_t row_size = n_per_row/QK_TURBO4*sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_turbo4_0_ref(
                src + row*n_per_row,
                (block_turbo4_0 *) ((char *) dst + row*row_size),
                n_per_row);
    }
    return nrows*row_size;
}
