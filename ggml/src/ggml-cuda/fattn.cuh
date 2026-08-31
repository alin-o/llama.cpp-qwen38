#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

extern "C" unsigned long long ggml_cuda_fattn_shared_turbo_decode_launch_count(void);
extern "C" void ggml_cuda_fattn_shared_turbo_decode_launch_count_reset(void);
extern "C" void ggml_cuda_fattn_shared_turbo_decode_launch_count_add(void);
