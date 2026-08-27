#include "common.cuh"

void ggml_cuda_op_paged_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

extern "C" unsigned long long ggml_paged_attn_tiled_prefill_launch_count(void);
extern "C" void ggml_paged_attn_tiled_prefill_launch_count_reset(void);
