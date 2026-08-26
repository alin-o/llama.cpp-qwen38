#include "common.cuh"

__global__ void paged_attention_write_kernel(const float * k_new,
                                             const float * v_new,
                                             half * k_cache,
                                             half * v_cache,
                                             const int * write_slots,
                                             const int * batch_offsets,
                                             const int * batch_lens,
                                             const size_t stride_token,
                                             const size_t stride_head,
                                             const size_t stride_block,
                                             const int n_heads_kv,
                                             const int block_size);

__global__ void paged_attention_decode_kernel(const float * __restrict__ q,
                                              const half * __restrict__ k_cache,
                                              const half * __restrict__ v_cache,
                                              const int * __restrict__ block_table,
                                              const int * __restrict__ context_lens,
                                              const int * __restrict__ batch_offsets,
                                              const int * __restrict__ batch_lens,
                                              const size_t stride_token,
                                              const size_t stride_head,
                                              const size_t stride_block,
                                              const int n_heads_kv,
                                              const int block_size,
                                              const int max_blocks,
                                              const float scale,
                                              float * __restrict__ out);

void ggml_cuda_op_paged_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
