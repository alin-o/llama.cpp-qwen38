#include "llama-memory-hybrid-paged.h"

#include "llama-context.h"
#include "llama-impl.h"
#include "llama-model.h"

llama_memory_hybrid_paged::llama_memory_hybrid_paged(
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
        const layer_filter_cb & filter_recr) :
    mem_attn(new llama_kv_cache_paged(head_dim, n_heads_kv, block_size, n_layers, n_ubatch, n_seq_max)),
    mem_recr(new llama_memory_recurrent(
        model, type_r, type_s, offload, rs_size, n_seq_max, n_rs_seq,
        filter_recr == nullptr ? [&](int32_t il) { return model.hparams.is_recr(il); } : filter_recr)) {
    mem_attn->init(backend_gpu, backend_cpu, paged_type, n_gpu_blocks, n_cpu_blocks, watermark);
}

llama_memory_context_ptr llama_memory_hybrid_paged::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    balloc.split_reset();
    std::vector<llama_ubatch> ubatches;
    while (true) {
        llama_ubatch ubatch;
        if (embd_all) {
            ubatch = balloc.split_seq(n_ubatch);
        } else {
            ubatch = balloc.split_equal(n_ubatch, false, mem_recr->n_rs_seq > 0 ? mem_recr->n_rs_seq + 1 : 0);
        }
        if (ubatch.n_tokens == 0) {
            break;
        }
        ubatches.push_back(std::move(ubatch));
    }
    if (balloc.get_n_used() < balloc.get_n_tokens() || !mem_recr->prepare(ubatches)) {
        return std::make_unique<llama_memory_hybrid_paged_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }
    auto ctx_attn = mem_attn->prepare(ubatches);
    return std::make_unique<llama_memory_hybrid_paged_context>(this, std::move(ctx_attn), std::move(ubatches));
}

llama_memory_context_ptr llama_memory_hybrid_paged::init_full() {
    return std::make_unique<llama_memory_hybrid_paged_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_paged::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_paged_context>(this, lctx, optimize);
}

void llama_memory_hybrid_paged::clear(bool data) {
    mem_attn->clear(data);
    mem_recr->clear(data);
}

bool llama_memory_hybrid_paged::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return mem_attn->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_paged::seq_cp(llama_seq_id s, llama_seq_id d, llama_pos p0, llama_pos p1) {
    mem_attn->seq_cp(s, d, p0, p1);
    mem_recr->seq_cp(s, d, p0, p1);
}

void llama_memory_hybrid_paged::seq_keep(llama_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);
}

void llama_memory_hybrid_paged::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid_paged::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    mem_attn->seq_div(seq_id, p0, p1, d);
    mem_recr->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_hybrid_paged::seq_pos_min(llama_seq_id seq_id) const {
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid_paged::seq_pos_max(llama_seq_id seq_id) const {
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_paged::memory_breakdown() const {
    auto result = mem_attn->memory_breakdown();
    for (const auto & item : mem_recr->memory_breakdown()) {
        result[item.first] += item.second;
    }
    return result;
}

void llama_memory_hybrid_paged::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    mem_recr->state_write(io, seq_id, flags);
}

void llama_memory_hybrid_paged::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    mem_recr->state_read(io, seq_id, flags);
}

llama_memory_hybrid_paged_context::llama_memory_hybrid_paged_context(llama_memory_status status) : status(status) {}

llama_memory_hybrid_paged_context::llama_memory_hybrid_paged_context(llama_memory_hybrid_paged * mem) :
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {}

llama_memory_hybrid_paged_context::llama_memory_hybrid_paged_context(llama_memory_hybrid_paged * mem, llama_context * lctx, bool optimize) :
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {}

llama_memory_hybrid_paged_context::llama_memory_hybrid_paged_context(
        llama_memory_hybrid_paged * mem, llama_memory_context_ptr ctx_attn_in, std::vector<llama_ubatch> ubatches_in) :
    ubatches(std::move(ubatches_in)),
    ctx_attn(std::move(ctx_attn_in)),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), ubatches)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {}

bool llama_memory_hybrid_paged_context::next() {
    GGML_ASSERT(status == LLAMA_MEMORY_STATUS_SUCCESS);
    ctx_attn->next();
    ctx_recr->next();
    return ++i_next < ubatches.size();
}

bool llama_memory_hybrid_paged_context::apply() {
    GGML_ASSERT(!llama_memory_status_is_fail(status));
    return ctx_attn->apply() && ctx_recr->apply();
}

const llama_ubatch & llama_memory_hybrid_paged_context::get_ubatch() const {
    GGML_ASSERT(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_paged_context * llama_memory_hybrid_paged_context::get_attn() const {
    return static_cast<const llama_kv_cache_paged_context *>(ctx_attn.get());
}

const llama_memory_recurrent_context * llama_memory_hybrid_paged_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}
