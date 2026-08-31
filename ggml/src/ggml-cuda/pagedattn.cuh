#include "common.cuh"

void ggml_cuda_op_paged_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

extern "C" unsigned long long ggml_paged_attn_tiled_prefill_launch_count(void);
extern "C" void ggml_paged_attn_tiled_prefill_launch_count_reset(void);
extern "C" unsigned long long ggml_paged_attn_q8_decode_launch_count(void);
extern "C" void ggml_paged_attn_q8_decode_launch_count_reset(void);
extern "C" unsigned long long ggml_paged_attn_q8_combined_write_launch_count(void);
extern "C" void ggml_paged_attn_q8_combined_write_launch_count_reset(void);
extern "C" unsigned long long ggml_paged_attn_q8_fused_write_launch_count(void);
extern "C" void ggml_paged_attn_q8_fused_write_launch_count_reset(void);
bool ggml_paged_attn_cuda_runtime_test(void);
