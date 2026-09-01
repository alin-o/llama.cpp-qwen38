#pragma once

#include "common.cuh"

#define CUDA_SET_ROWS_BLOCK_SIZE 256

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_set_rows_turbo_pair(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src_k, const ggml_tensor * src_v, const ggml_tensor * rows,
        const ggml_tensor * dst_k, const ggml_tensor * dst_v, bool combine_launches);
