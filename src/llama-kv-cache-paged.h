#pragma once

#include "llama-batch.h"
#include "llama-block-manager.h"
#include "llama-graph.h"
#include "llama-memory.h"
#include "llama-sequence-group.h"

#include <cmath>

//
// llama_kv_cache_paged
//

class llama_kv_cache_paged : public llama_memory_i {
  public:
    llama_kv_cache_paged(uint32_t head_dim,
                         uint32_t n_head_kv,
                         uint32_t block_size,
                         uint32_t n_layers,
                         uint32_t n_ubatch,
                         uint32_t n_seq_max);

    llama_kv_cache_paged(uint32_t head_dim,
                         uint32_t n_head_kv,
                         uint32_t block_size,
                         uint32_t n_layers,
                         uint32_t n_ubatch,
                         uint32_t n_seq_max,
                         std::vector<uint32_t> attention_layers);

    void init(ggml_backend_t backend_gpu,
              ggml_backend_t backend_cpu,
              enum ggml_type type_k,
              enum ggml_type type_v,
              uint32_t       n_gpu_blocks,
              uint32_t       n_cpu_blocks,
              float          watermark);  // percentage

    bool allocate(int32_t num_tokens, llama_sequence_group & group);
    void free_blocks(llama_sequence_group & group);
    bool release_seq_tail(llama_seq_id seq_id, uint32_t keep_tokens);
    bool retain_block_ids(const llama_block_ids & block_ids);
    void release_retained_block_ids(const llama_block_ids & block_ids);
    bool checkpoint_blocks(llama_seq_id seq_id, uint32_t n_tokens, llama_block_ids & block_ids) const;
    bool attach_checkpoint(llama_sequence_group & group, const llama_block_ids & block_ids,
                           uint32_t n_tokens, bool private_writer, bool * did_cow = nullptr);
    bool cow_partial_tail(llama_sequence_group & group, uint32_t n_tokens,
                          uint32_t * replaced_block = nullptr);
    bool rollback_partial_tail(llama_sequence_group & group, uint32_t n_tokens,
                               uint32_t replaced_block);
    uint32_t get_block_ref_count(uint32_t block_id) const;
    bool swap_in(llama_sequence_group & group, int32_t num_tokens = 1);
    bool swap_out(llama_sequence_group & group);

    void     set_paged_batch_info(const llama_paged_batch_info * info);
    bool register_group(llama_sequence_group & group);
    uint32_t get_num_gpu_blocks() const;
    uint32_t get_num_free_gpu_blocks() const;
    uint32_t get_num_cpu_blocks() const;
    uint32_t get_num_free_cpu_blocks() const;

    //
    // llama_memory_i
    //
    llama_memory_context_ptr init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) override;

    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;
    llama_memory_context_ptr prepare(const std::vector<llama_ubatch> & ubatches);

    struct ggml_tensor * get_k_tensor(int layer_idx) const;
    struct ggml_tensor * get_v_tensor(int layer_idx) const;
    int32_t get_physical_layer(int layer_idx) const;
    uint32_t get_n_attention_layers() const;
    size_t get_bytes_per_block() const;

    bool get_can_shift() const override { return false; }

    void clear(bool data) override;

    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;

    void seq_cp(llama_seq_id /*seq_id_src*/,
                llama_seq_id /*seq_id_dst*/,
                llama_pos /*p0*/,
                llama_pos /*p1*/) override { /* implement later CoW mechanism */
    }

    void seq_keep(llama_seq_id /*seq_id*/) override {}

    void seq_add(llama_seq_id /*seq_id*/, llama_pos /*p0*/, llama_pos /*p1*/, llama_pos /*shift*/) override {}

    void seq_div(llama_seq_id /*seq_id*/, llama_pos /*p0*/, llama_pos /*p1*/, int /*d*/) override {}

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io,
                     llama_seq_id seq_id = -1,
                     llama_state_seq_flags flags = 0) const override;

    void state_read(llama_io_read_i & io,
                    llama_seq_id seq_id = -1,
                    llama_state_seq_flags flags = 0) override;

    //
    // Helpers to llama_memory_i
    //
    void set_seq_min_pos(llama_seq_id seq_id, llama_pos new_min);
    void set_seq_max_pos(llama_seq_id seq_id, llama_pos new_max);

  private:
    friend class llama_kv_cache_paged_context;

    void concat_block_ids(llama_block_ids & to_block_table, const llama_block_ids & from_block_table);
    void do_block_copy(const llama_block_ids & src_ids, const llama_block_ids & new_ids, bool to_gpu);
    void do_gpu_block_copy(uint32_t src_id, uint32_t dst_id);
    void release_block_ids(const llama_block_ids & block_ids);

    std::vector<struct ggml_tensor *> k_gpu_layers;
    std::vector<struct ggml_tensor *> v_gpu_layers;
    std::vector<struct ggml_tensor *> k_cpu_layers;
    std::vector<struct ggml_tensor *> v_cpu_layers;

    enum ggml_type kv_type_k;
    enum ggml_type kv_type_v;
    llama_block_manager block_manager;

    // Non-owning pointer to the batch currently being processed.
    // Lifetime: set by the scheduler at the end of step(), cleared at the
    // start of the next step() (before the batch's paged_* arrays are freed).
    // The ordering in llama_paged_scheduler_impl::clear_batch is load-bearing;
    // do not reorder without updating init_batch's contract.
    const llama_paged_batch_info * last_paged_info = nullptr;

    const uint32_t head_dim;
    const uint32_t n_heads_kv;
    const uint32_t block_size;
    const uint32_t n_layers;
    const uint32_t n_ubatch;
    const uint32_t n_seq_max;
    const std::vector<int32_t>  layer_to_physical;
    const std::vector<uint32_t> physical_to_layer;
    uint32_t       num_gpu_blocks;
    uint32_t       num_cpu_blocks;
    uint32_t       block_bytes_k;
    uint32_t       block_bytes_v;
    bool initialized = false;

    ggml_backend_t gpu_backend;
    ggml_backend_t cpu_backend;

    struct seq_range {
        llama_pos min = -1;
        llama_pos max = -1;
    };
    struct restored_group_state {
        llama_sequence_group_status status;
        int64_t                     t_arrival_time;
        int64_t                     t_first_token_us;
        uint32_t                    n_prompt;
        uint32_t                    n_decoded;
        uint32_t                    n_past;
        std::vector<llama_token>    logical_seq;
    };

    std::unordered_map<llama_seq_id, restored_group_state> restored_groups;

    std::unordered_map<llama_seq_id, seq_range> sequence_positions;

    std::map<llama_seq_id, llama_block_ids> sequence_blocks;
    std::unordered_map<llama_seq_id, llama_sequence_group *> sequence_groups;
};

class llama_kv_cache_paged_context : public llama_memory_context_i {
  public:
    llama_kv_cache_paged_context(llama_kv_cache_paged * parent, const std::vector<llama_ubatch> & in_ubatch) :
        manager(parent),
        ubatches(in_ubatch) {
        i_cur = 0;
    }

    llama_kv_cache_paged_context(llama_memory_status status) : status(status) {}

    void    set_batch_data(const llama_paged_batch_info & info);
    int32_t get_n_tokens() const;
    int32_t tokens() const { return n_tokens; }
    int32_t get_batch_size() const;
    int32_t get_max_blocks() const;
    int32_t get_max_context_len() const;

    const int32_t * get_write_slots() const;
    const int32_t * get_block_table() const;
    const int32_t * get_write_rows() const;

    const int32_t * get_context_lens() const;
    const int32_t * get_batch_offsets() const;
    const int32_t * get_batch_lens() const;

    void set_n_tokens(int32_t new_n_tokens);
    void set_batch_size(int32_t new_batch_size);
    void set_max_blocks(int32_t new_max_blocks);

    struct ggml_tensor * get_k(int layer_idx) const;
    struct ggml_tensor * get_v(int layer_idx) const;

    //
    // llama_memory_context_i
    //
    bool                 next() override;
    bool                 apply() override;
    const llama_ubatch & get_ubatch() const override;

    llama_memory_status get_status() const override { return status; }

  private:
    const llama_kv_cache_paged * manager;

    //
    // batch processing context
    //
    std::vector<llama_ubatch> ubatches;
    size_t                    i_cur = 0;      // index of ubatch to process

    void select_ubatch();

    const llama_paged_batch_info * full_info = nullptr;
    std::vector<int32_t> paged_write_slots;
    std::vector<int32_t> paged_block_table;
    std::vector<int32_t> paged_context_lens;
    std::vector<int32_t> paged_batch_offsets;
    std::vector<int32_t> paged_batch_lens;
    std::vector<int32_t> paged_write_rows;


    int32_t n_tokens   = 0;
    int32_t batch_size = 0;
    int32_t max_blocks = 0;

    llama_memory_status status = LLAMA_MEMORY_STATUS_SUCCESS;
};
