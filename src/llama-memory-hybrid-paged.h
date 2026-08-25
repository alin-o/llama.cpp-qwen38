#pragma once

#include "llama-batch.h"
#include "llama-kv-cache-paged.h"
#include "llama-memory-recurrent.h"

#include <memory>
#include <vector>

class llama_memory_hybrid_paged : public llama_memory_i {
public:
    using layer_filter_cb = llama_memory_i::layer_filter_cb;

    llama_memory_hybrid_paged(
        const llama_model & model,
        uint32_t head_dim,
        uint32_t n_heads_kv,
        uint32_t block_size,
        uint32_t n_layers,
        uint32_t n_ubatch,
        uint32_t n_seq_max,
        ggml_backend_t backend_gpu,
        ggml_backend_t backend_cpu,
        ggml_type paged_type,
        uint32_t n_gpu_blocks,
        uint32_t n_cpu_blocks,
        float watermark,
        ggml_type type_r,
        ggml_type type_s,
        uint32_t rs_size,
        uint32_t n_rs_seq,
        bool offload,
        const layer_filter_cb & filter_recr = nullptr);

    llama_memory_context_ptr init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) override;
    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;
    bool get_can_shift() const override { return false; }
    void clear(bool data) override;
    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;
    void seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override;
    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;
    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read(llama_io_read_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    llama_kv_cache_paged * get_mem_attn() const { return mem_attn.get(); }
    llama_memory_recurrent * get_mem_recr() const { return mem_recr.get(); }

private:
    const std::unique_ptr<llama_kv_cache_paged> mem_attn;
    const std::unique_ptr<llama_memory_recurrent> mem_recr;
};

class llama_memory_hybrid_paged_context : public llama_memory_context_i {
public:
    explicit llama_memory_hybrid_paged_context(llama_memory_status status);
    explicit llama_memory_hybrid_paged_context(llama_memory_hybrid_paged * mem);
    llama_memory_hybrid_paged_context(llama_memory_hybrid_paged * mem, llama_context * lctx, bool optimize);
    llama_memory_hybrid_paged_context(llama_memory_hybrid_paged * mem,
                                      llama_memory_context_ptr ctx_attn,
                                      std::vector<llama_ubatch> ubatches);

    bool next() override;
    bool apply() override;
    llama_memory_status get_status() const override { return status; }
    const llama_ubatch & get_ubatch() const override;

    const llama_kv_cache_paged_context * get_attn() const;
    const llama_memory_recurrent_context * get_recr() const;

private:
    size_t i_next = 0;
    std::vector<llama_ubatch> ubatches;
    const llama_memory_context_ptr ctx_attn;
    const llama_memory_context_ptr ctx_recr;
    const llama_memory_status status;
};
