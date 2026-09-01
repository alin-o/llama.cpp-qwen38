#pragma once

#include "common.cuh"

void ggml_cuda_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_turbo_wht_f32(
        cudaStream_t stream, const float * src, float * dst, int64_t n_elements, int direction);
