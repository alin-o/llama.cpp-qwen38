#pragma once

#include "llama-kv-cache-paged.h"

#include <clocale>
#include <array>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

class llama_memory_recurrent;

enum class llama_scheduler_status {
    OK,
    DEADLOCK,  // cannot make progress
};

using llama_checkpoint_key = std::array<uint8_t, 32>;

struct llama_checkpoint_payload {
    std::vector<uint8_t> recurrent;
    std::vector<uint8_t> draft;
    std::vector<uint8_t> speculative;
    bool recurrent_complete  = true;
    bool draft_complete      = true;
    bool speculative_complete = true;
};

struct llama_checkpoint_view {
    const uint8_t * recurrent   = nullptr;
    size_t recurrent_size       = 0;
    const uint8_t * draft       = nullptr;
    size_t draft_size           = 0;
    const uint8_t * speculative = nullptr;
    size_t speculative_size     = 0;
    uint32_t n_tokens            = 0;
};

enum class llama_checkpoint_publish_fault {
    NONE,
    TARGET,
    RECURRENT,
    DRAFT,
    SPECULATIVE,
    TOKENS,
    COMPATIBILITY,
    INTEGRITY,
};

struct llama_checkpoint_metrics {
    uint64_t lookups             = 0;
    uint64_t hits                = 0;
    uint64_t equality_mismatches = 0;
    uint64_t hit_tokens          = 0;
    uint64_t suffix_tokens       = 0;
    uint64_t build_winners       = 0;
    uint64_t build_waiters       = 0;
    uint64_t builds_coalesced    = 0;
    uint64_t wait_timeouts       = 0;
    uint64_t publications        = 0;
    uint64_t publication_failures = 0;
    uint64_t evictions           = 0;
    uint64_t cow_copies          = 0;
    uint64_t rollbacks           = 0;
    uint64_t fallbacks           = 0;
    uint64_t resident_pages      = 0;
    uint64_t logical_page_refs   = 0;
    uint64_t pins                = 0;
    uint64_t graph_reuses        = 0;
    uint64_t graph_rebuilds      = 0;
    uint64_t resident_host_bytes = 0;
    uint64_t host_quota_bytes    = 0;
    uint64_t admission_rejections = 0;
    uint64_t page_reclamations    = 0;
    uint64_t cow_failures         = 0;
    uint64_t restore_successes    = 0;
    uint64_t restore_failures     = 0;
    uint64_t records             = 0;
};

// Speculative decode API design: a future spec step reserves 1 + spec_n target
// tokens per decode sequence. Its update accepts per-sequence counts, commits
// only accepted rows, and calls release_seq_tail(seq_id, keep_tokens) to free
// rejected trailing blocks. Spec proposals remain resident while pressure
// scheduling promotes the oldest request.
class llama_paged_scheduler_impl {
  public:
    llama_paged_scheduler_impl(
            uint32_t                 n_ctx,
            uint32_t                 block_sz,
            int32_t                  n_batch,
            llama_kv_cache_paged *   kv_manager,
            llama_memory_recurrent * recurrent_manager = nullptr);
    ~llama_paged_scheduler_impl();

    llama_scheduler_status step(llama_batch & batch, int32_t spec_n = 0);
    bool                   queue_request(llama_sequence_group group, uint32_t n_past = 0);
    bool                   retain_request(int32_t request_id, bool checkpoint_before_last);
    bool                   is_retained(int32_t request_id) const;
    void                   update(const llama_batch & batch, const std::vector<llama_token> & new_tokens,
                                   const std::vector<uint32_t> & accepted, const int8_t * stop_flags);
    void                   update(const llama_batch & batch, const std::vector<llama_token> & new_tokens,
                                   const int8_t * stop_flags) {
        update(batch, new_tokens, {}, stop_flags);
    }
    void                   remove_request(int32_t request_id);
    void                   set_on_finish(llama_paged_on_finish_cb cb, void * user_data);
    void                   set_on_recompute(llama_paged_on_recompute_cb cb, void * user_data);
    void                   set_batch_policy(llama_paged_batch_token_limit_cb token_limit_cb,
                                            llama_paged_batch_compatible_cb  compatible_cb,
                                            void *                           user_data);
    llama_sequence_group *         get_group_from_id(int32_t request_id) const;
    const llama_paged_batch_info * get_curr_batch_info() const;
    llama_paged_cache_stats        get_cache_stats() const;
    llama_checkpoint_metrics       get_checkpoint_metrics() const;

    bool publish_checkpoint(int32_t request_id, uint32_t n_tokens,
                            const std::string & fingerprint,
                            const llama_checkpoint_payload & payload,
                            llama_checkpoint_key * key_out = nullptr,
                            llama_checkpoint_publish_fault fault = llama_checkpoint_publish_fault::NONE);
    bool queue_request_cached(llama_sequence_group group,
                              const std::string & fingerprint,
                              llama_checkpoint_view * view,
                              uint32_t * n_prefix_used);
    void evict_unpinned_checkpoints(uint32_t pages_needed = 0, uint64_t host_bytes_needed = 0);
    void force_checkpoint_digest_for_test(bool enabled);
    void set_checkpoint_quotas_for_test(uint32_t page_quota, uint64_t host_quota);
    void set_checkpoint_build_gate_for_test(bool closed);
    bool wait_for_checkpoint_build_gate_for_test(uint32_t timeout_ms);
    bool wait_for_checkpoint_metrics_for_test(uint64_t build_waiters, uint64_t wait_timeouts,
                                              uint32_t timeout_ms);
    void set_request_paused(int32_t request_id, bool paused);
    uint32_t checkpoint_pin_depth(int32_t request_id) const;
    llama_block_ids request_block_ids(int32_t request_id) const;

  private:
    void insert_sorted_by_arrival_time(llama_sequence_group_ptr new_group, llama_sequence_group_list & list);

    bool check_deadlock(uint32_t n_candidates, uint32_t n_swapped, uint32_t n_waiting) const;
    bool check_livelock(uint32_t n_candidates, uint32_t n_swapped, uint32_t prev_n_swapped);

    void set_running(llama_sequence_group_ptr group);
    void set_swapped(llama_sequence_group_ptr group);
    void set_waiting(llama_sequence_group_ptr group);

    bool finish(llama_sequence_group & group);
    void complete_request(int32_t request_id);
    void evict_retained_requests();
    void release_checkpoint_pin(int32_t request_id);

    int32_t get_curr_decode_tokens() const;
    int32_t get_scheduled_tokens(const llama_sequence_group & group) const;

    void activate_priority_request(llama_sequence_group_raw_list & candidates);
    void process_running_list(llama_sequence_group_raw_list & candidates);
    void process_swapped_list(llama_sequence_group_raw_list & candidates);
    void process_waiting_list(llama_sequence_group_raw_list & candidates, int32_t remaining_token_bugdet);

    void swap_out_or_recompute(llama_sequence_group_ptr group_ptr);

    int32_t calculate_global_slot_index(int32_t token_pos, std::vector<uint32_t> & block_table);

    void clear_batch(llama_batch & batch);
    void populate_batch_from(const llama_sequence_group_raw_list & candidates, llama_batch & batch);

    llama_sequence_group_list running;
    llama_sequence_group_list swapped;
    llama_sequence_group_list waiting;
    std::unordered_map<int32_t, llama_sequence_group_ptr> retained;
    std::unordered_set<int32_t> retain_on_finish;
    std::unordered_set<int32_t> checkpoint_before_last;

    // Used for fast lookups
    std::unordered_map<int32_t, llama_sequence_group *> id_to_group;

    const uint32_t         n_seq_max_ctx;
    const uint32_t         block_size;
    const int32_t          n_batch;
    llama_kv_cache_paged * kv_cache_manager = nullptr;
    llama_memory_recurrent * recurrent_manager = nullptr;
    llama_paged_batch_info curr_info;
    int32_t spec_n = 0;

    enum class checkpoint_state { BUILDING, READY, FAILED };
    struct checkpoint_record {
        llama_checkpoint_key key = {};
        llama_checkpoint_key predecessor = {};
        bool has_predecessor = false;
        checkpoint_state state = checkpoint_state::BUILDING;
        std::vector<llama_token> tokens;
        std::string fingerprint;
        llama_block_ids blocks;
        uint32_t terminal_extent = 0;
        llama_checkpoint_payload payload;
        llama_checkpoint_key integrity = {};
        uint64_t last_access = 0;
        uint32_t depth = 0;
        uint32_t pins = 0;
        bool resident = false;
        uint64_t host_bytes = 0;
        std::condition_variable ready_cv;
    };

    using checkpoint_record_ptr = std::shared_ptr<checkpoint_record>;
    mutable std::mutex checkpoint_mutex;
    std::unordered_map<std::string, checkpoint_record_ptr> checkpoints;
    std::unordered_map<int32_t, checkpoint_record_ptr> request_checkpoint_pins;
    std::unordered_set<int32_t> paused_requests;
    std::unordered_map<uint32_t, uint32_t> checkpoint_page_refs;
    llama_checkpoint_metrics checkpoint_metrics;
    uint32_t checkpoint_page_quota = 0;
    uint64_t checkpoint_host_quota = 2ULL * 1024 * 1024 * 1024;
    uint64_t checkpoint_resident_host_bytes = 0;
    uint64_t checkpoint_clock = 0;
    bool force_digest_collision = false;
    bool checkpoint_build_gate_closed = false;
    uint32_t checkpoint_builders_at_gate = 0;
    std::condition_variable checkpoint_test_cv;
    std::array<int32_t, 3> graph_signature = { -1, -1, -1 };


    int32_t priority_request_id = -1;

    uint32_t n_livelock_steps   = 0;
    uint32_t prev_n_swapped     = 0;
    uint32_t max_livelock_steps = 20;

    // Callback for output tracking
    llama_paged_on_finish_cb    on_finish_cb          = nullptr;
    void *                      on_finish_user_data   = nullptr;
    llama_paged_on_recompute_cb on_recompute_cb        = nullptr;
    void *                      on_recompute_user_data = nullptr;
    llama_paged_batch_token_limit_cb token_limit_cb      = nullptr;
    llama_paged_batch_compatible_cb  compatible_cb       = nullptr;
    void *                           batch_policy_data    = nullptr;
};
