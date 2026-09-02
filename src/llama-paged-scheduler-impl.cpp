#include "llama-paged-scheduler-impl.h"

#include <algorithm>
#include <stdexcept>
#include "llama-impl.h"
#include "llama-memory-recurrent.h"

llama_paged_scheduler_impl::llama_paged_scheduler_impl(
        uint32_t                 n_ctx,
        uint32_t                 block_sz,
        int32_t                  n_batch,
        llama_kv_cache_paged *   kv_manager,
        llama_memory_recurrent * recurrent_manager) :
    n_seq_max_ctx(n_ctx),
    block_size(block_sz),
    n_batch(n_batch),
    kv_cache_manager(kv_manager),
    recurrent_manager(recurrent_manager),
    curr_info{} {}

llama_paged_scheduler_impl::~llama_paged_scheduler_impl() {
    kv_cache_manager->set_paged_batch_info(nullptr);
    delete[] curr_info.write_slots;
    delete[] curr_info.block_table;
    delete[] curr_info.context_lens;
    delete[] curr_info.batch_offsets;
    delete[] curr_info.batch_lens;

    for (const auto & item : id_to_group) {
        kv_cache_manager->seq_rm(item.first, -1, -1);
        if (recurrent_manager) {
            recurrent_manager->seq_rm(item.first, -1, -1);
        }
    }
}

bool llama_paged_scheduler_impl::check_deadlock(uint32_t n_candidates, uint32_t n_swapped, uint32_t n_waiting) const {
    if (n_candidates == 0 && (n_swapped > 0 || n_waiting > 0)) {
        LLAMA_LOG_ERROR(
            "%s: Scheduler deadlock detected. "
            "%d sequence(s) are swapped out and %d are waiting, "
            "but there are not enough free GPU blocks to make progress. "
            "Hint: increase n_gpu_blocks (currently %d) or reduce n_sequences.\n",
            __func__, n_swapped, n_waiting, kv_cache_manager->get_num_gpu_blocks());
        return true;
    }
    return false;
}

bool llama_paged_scheduler_impl::check_livelock(
        uint32_t n_candidates, uint32_t n_swapped, uint32_t prev_n_swapped) {
    if (n_candidates > 0) {
        n_livelock_steps = 0;
        return false;
    }

    // swapped count is non-zero and not decreasing
    if (n_swapped > 0 && n_swapped >= prev_n_swapped) {
        n_livelock_steps++;
        if (n_livelock_steps >= max_livelock_steps) {
            LLAMA_LOG_ERROR(
                "%s: Livelock detected. Swapped count has been "
                "non-decreasing for %d steps (currently %d swapped). "
                "Increase n_gpu_blocks (currently %d) or reduce "
                "n_sequences.\n",
                __func__, n_livelock_steps, n_swapped, kv_cache_manager->get_num_gpu_blocks());
            return true;
        }
    } else {
        // Swapped count decreased - swap-ins are happening, reset counter
        n_livelock_steps = 0;
    }
    return false;
}

int32_t llama_paged_scheduler_impl::get_curr_decode_tokens() const {
    int32_t result = 0;
    for (const auto & group : running) {
        result += get_scheduled_tokens(*group);
    }
    return result;
}

int32_t llama_paged_scheduler_impl::get_scheduled_tokens(const llama_sequence_group & group) const {
    if (group.n_past < group.n_prompt) {
        int32_t remaining = group.n_prompt - group.n_past;
        if (remaining > 1 && checkpoint_before_last.count(group.request_id)) {
            --remaining;
        }
        return std::min<int32_t>(remaining, n_batch);
    }
    return 1 + spec_n;
}

llama_scheduler_status llama_paged_scheduler_impl::step(llama_batch & batch, int32_t spec_n) {
    this->spec_n = std::max(0, spec_n);
    // Free previous inference batches
    clear_batch(batch);

    llama_sequence_group_raw_list candidates;
    process_running_list(candidates);
    process_swapped_list(candidates);

    const int32_t remaining = n_batch - get_curr_decode_tokens();
    process_waiting_list(candidates, remaining);

    while (!candidates.empty()) {
        int32_t total_tokens = 0;
        for (const auto * group : candidates) {
            total_tokens += get_scheduled_tokens(*group);
        }
        if (total_tokens <= n_batch) {
            break;
        }
        candidates.pop_back();
    }
    const uint32_t n_running = running.size();
    const uint32_t n_swapped    = swapped.size();
    const uint32_t n_waiting    = waiting.size();
    const uint32_t n_candidates = candidates.size();

    LLAMA_LOG_INFO("%s: Scheduler status: running=%d, swapped=%d, waiting=%d, candidates=%d\n", __func__, n_running,
                   n_swapped, n_waiting, n_candidates);

    const bool deadlock = check_deadlock(n_candidates, n_swapped, n_waiting);
    const bool livelock = check_livelock(n_candidates, n_swapped, prev_n_swapped);  // updates n_livelock_steps
    if (deadlock || livelock) {
        return llama_scheduler_status::DEADLOCK;
    }

    prev_n_swapped = n_swapped;
    populate_batch_from(candidates, batch);
    kv_cache_manager->set_paged_batch_info(&curr_info);
    return llama_scheduler_status::OK;
}

bool llama_paged_scheduler_impl::queue_request(llama_sequence_group group, uint32_t n_past) {
    // Rejecting any requests that exceeds max context for a seq
    if (group.n_prompt >= n_seq_max_ctx) {
        LLAMA_LOG_ERROR("%s: request %d exceeds max context (%d > %d).\n", __func__, group.request_id, group.n_prompt,
                        n_seq_max_ctx);
        return false;
    }

    if (n_past > group.n_prompt) {
        LLAMA_LOG_ERROR("%s: request %d has invalid cached prefix (%u > %u).\n", __func__, group.request_id,
                        n_past, group.n_prompt);
        return false;
    }

    if (const auto cached = retained.find(group.request_id); cached != retained.end()) {
        llama_sequence_group * old = cached->second.get();
        const size_t required_blocks = (n_past + block_size - 1) / block_size;
        const bool prefix_matches = n_past > 0 && old->logical_seq.size() >= n_past &&
            group.logical_seq.size() >= n_past &&
            std::equal(group.logical_seq.begin(), group.logical_seq.begin() + n_past, old->logical_seq.begin());
        const bool blocks_match = old->n_past >= n_past && old->block_table.size() >= required_blocks;
        const bool positions_match = kv_cache_manager->seq_pos_min(group.request_id) == 0 &&
            kv_cache_manager->seq_pos_max(group.request_id) >= (llama_pos) n_past - 1;
        const bool recurrent_matches = recurrent_manager == nullptr ||
            recurrent_manager->seq_pos_max(group.request_id) == (llama_pos) n_past - 1;

        if (prefix_matches && blocks_match && positions_match && recurrent_matches &&
            kv_cache_manager->release_seq_tail(group.request_id, n_past)) {
            kv_cache_manager->set_seq_max_pos(group.request_id, n_past - 1);
            llama_sequence_group_ptr group_ptr = std::move(cached->second);
            retained.erase(cached);
            group_ptr->status          = llama_sequence_group_status::PENDING;
            group_ptr->t_arrival_time  = group.t_arrival_time;
            group_ptr->t_first_token_us = 0;
            group_ptr->n_prompt        = group.n_prompt;
            group_ptr->n_decoded       = 0;
            group_ptr->n_past          = n_past;
            group_ptr->logical_seq     = std::move(group.logical_seq);
            set_waiting(std::move(group_ptr));
            return true;
        }

        remove_request(group.request_id);
        n_past = 0;
    }

    if (id_to_group.count(group.request_id)) {
        LLAMA_LOG_ERROR("%s: request %d is already queued.\n", __func__, group.request_id);
        return false;
    }


    auto group_ptr = std::make_unique<llama_sequence_group>(std::move(group));

    const bool restored = kv_cache_manager->register_group(*group_ptr);
    if (!restored && !group_ptr->block_table.empty()) {
        LLAMA_LOG_ERROR("%s: request %d has restored blocks without scheduler state.\n", __func__, group_ptr->request_id);
        kv_cache_manager->seq_rm(group_ptr->request_id, -1, -1);
        if (recurrent_manager) {
            recurrent_manager->seq_rm(group_ptr->request_id, -1, -1);
        }
        return false;
    }

    const llama_sequence_group_status restored_status = group_ptr->status;
    if (restored && restored_status == llama_sequence_group_status::FINISHED) {
        LLAMA_LOG_ERROR("%s: request %d has finished scheduler state.\n", __func__, group_ptr->request_id);
        kv_cache_manager->seq_rm(group_ptr->request_id, -1, -1);
        if (recurrent_manager) {
            recurrent_manager->seq_rm(group_ptr->request_id, -1, -1);
        }
        return false;
    }

    id_to_group[group_ptr->request_id] = group_ptr.get();
    group_ptr->status = llama_sequence_group_status::PENDING;
    if (restored && restored_status == llama_sequence_group_status::RUNNING) {
        set_running(std::move(group_ptr));
    } else if (restored && restored_status == llama_sequence_group_status::SWAPPED) {
        set_swapped(std::move(group_ptr));
    } else {
        set_waiting(std::move(group_ptr));
    }
    return true;
}

void llama_paged_scheduler_impl::insert_sorted_by_arrival_time(llama_sequence_group_ptr    new_group_ptr,
                                                               llama_sequence_group_list & list) {
    GGML_ASSERT(new_group_ptr && "New group cannot be sorted because it's nullptr.");
    // linear scan: list is std::list (no random access) and small (one entry per sequence)
    auto it = std::find_if(list.begin(), list.end(),
                           [&](const llama_sequence_group_ptr & group) {
                               GGML_ASSERT(group && "group cannot be checked for arrival time because it's nullptr.");
                               return group->t_arrival_time > new_group_ptr->t_arrival_time;
                           });
    list.insert(it, std::move(new_group_ptr));
}

void llama_paged_scheduler_impl::set_running(llama_sequence_group_ptr group_ptr) {
    GGML_ASSERT(group_ptr && group_ptr->status != llama_sequence_group_status::RUNNING &&
                "Request is already running.");
    group_ptr->status = llama_sequence_group_status::RUNNING;
    insert_sorted_by_arrival_time(std::move(group_ptr), running);
}

void llama_paged_scheduler_impl::set_swapped(llama_sequence_group_ptr group_ptr) {
    GGML_ASSERT(group_ptr && group_ptr->status != llama_sequence_group_status::SWAPPED &&
                "Request is already swapped.");
    group_ptr->status = llama_sequence_group_status::SWAPPED;
    insert_sorted_by_arrival_time(std::move(group_ptr), swapped);
}

void llama_paged_scheduler_impl::set_waiting(llama_sequence_group_ptr group_ptr) {
    GGML_ASSERT(group_ptr && group_ptr->status != llama_sequence_group_status::WAITING &&
                "Request is already waiting.");
    group_ptr->status = llama_sequence_group_status::WAITING;
    insert_sorted_by_arrival_time(std::move(group_ptr), waiting);
}

bool llama_paged_scheduler_impl::finish(llama_sequence_group & group) {
    GGML_ASSERT(kv_cache_manager && "kv_cache_manager is nullptr.");
    GGML_ASSERT(group.status == llama_sequence_group_status::FINISHED && "Request was not marked as finished.");
    // We prioritize user CB, otherwise we log by default
    if (on_finish_cb) {
        // TODO perhaps just have the callback take sequence_group and user_data
        on_finish_cb(group.request_id, group.logical_seq.data(), (int32_t) group.logical_seq.size(),
                     on_finish_user_data);
    } else {
        LLAMA_LOG_DEBUG("%s: Request: %d generated %d tokens.\n", __func__, group.request_id, group.n_decoded);
    }
    group.status = llama_sequence_group_status::FINISHED;
    checkpoint_before_last.erase(group.request_id);
    if (retain_on_finish.erase(group.request_id) > 0) {
        if (kv_cache_manager->release_seq_tail(group.request_id, group.n_past)) {
            return true;
        }
        LLAMA_LOG_WARN("%s: request %d could not retain its paged state; discarding it.\n",
                       __func__, group.request_id);
    }
    kv_cache_manager->free_blocks(group);
    if (recurrent_manager) {
        recurrent_manager->seq_rm(group.request_id, -1, -1);
    }
    id_to_group.erase(group.request_id);
    return false;
}

void llama_paged_scheduler_impl::complete_request(int32_t request_id) {
    auto it = id_to_group.find(request_id);
    if (it == id_to_group.end()) {
        return;
    }

    llama_sequence_group * group = it->second;
    group->status = llama_sequence_group_status::FINISHED;

    auto finish_group = [&](llama_sequence_group_list & list) {
        for (auto list_it = list.begin(); list_it != list.end(); ++list_it) {
            if (list_it->get() == group) {
                const bool keep = finish(*group);
                if (keep) {
                    retained[request_id] = std::move(*list_it);
                }
                list.erase(list_it);
                return true;
            }
        }
        return false;
    };

    if (finish_group(running) || finish_group(swapped) || finish_group(waiting)) {
        if (priority_request_id == request_id) {
            priority_request_id = -1;
        }
    }
}

void llama_paged_scheduler_impl::remove_request(int32_t request_id) {
    retain_on_finish.erase(request_id);
    checkpoint_before_last.erase(request_id);
    if (const auto it = retained.find(request_id); it != retained.end()) {
        kv_cache_manager->free_blocks(*it->second);
        if (recurrent_manager) {
            recurrent_manager->seq_rm(request_id, -1, -1);
        }
        id_to_group.erase(request_id);
        retained.erase(it);
        return;
    }
    complete_request(request_id);
}

bool llama_paged_scheduler_impl::retain_request(int32_t request_id, bool needs_checkpoint) {
    if (!id_to_group.count(request_id) || retained.count(request_id)) {
        return false;
    }
    retain_on_finish.insert(request_id);
    if (needs_checkpoint) {
        checkpoint_before_last.insert(request_id);
    }
    return true;
}

bool llama_paged_scheduler_impl::is_retained(int32_t request_id) const {
    return retained.count(request_id) != 0;
}

void llama_paged_scheduler_impl::evict_retained_requests() {
    while (!retained.empty()) {
        remove_request(retained.begin()->first);
    }
}


// Try to swap a running sequence out to CPU.
// if the CPU pool is full, fall back to recomputation by resetting the sequence's decode state
// and sending it back to the waiting queue.
//
// Takes ownership of group_ptr.
// On return, the sequence is either in the swapped list (CPU pool had room)
// or the waiting list (recomputed).
void llama_paged_scheduler_impl::swap_out_or_recompute(llama_sequence_group_ptr group_ptr) {
    GGML_ASSERT(group_ptr && "group_ptr is nullptr");
    GGML_ASSERT(kv_cache_manager && "kv_cache_manager is nullptr");

    const int32_t rid = group_ptr->request_id;

    const bool swap_ok = kv_cache_manager->swap_out(*group_ptr);
    if (swap_ok) {
        LLAMA_LOG_DEBUG("%s: (swapped_out) request_id=%d was swapped out to make room.\n", __func__, rid);
        set_swapped(std::move(group_ptr));
        return;
    }

    // There was not enough CPU memory to swap the request (recomputation)
    kv_cache_manager->free_blocks(*group_ptr);
    kv_cache_manager->seq_rm(rid, -1, -1);
    if (recurrent_manager) {
        recurrent_manager->seq_rm(rid, -1, -1);
    }
    group_ptr->n_past    = 0;
    group_ptr->n_decoded = 0;
    group_ptr->logical_seq.resize(group_ptr->n_prompt);

    LLAMA_LOG_DEBUG("%s: (recomputation) request_id=%d was sent for recomputation.\n", __func__, rid);
    set_waiting(std::move(group_ptr));
}

void llama_paged_scheduler_impl::activate_priority_request(llama_sequence_group_raw_list & candidates) {
    if (priority_request_id != -1 || running.size() <= 1) {
        return;
    }

    priority_request_id = running.front()->request_id;
    LLAMA_LOG_DEBUG("%s: prioritizing oldest request_id=%d under GPU pressure.\n", __func__, priority_request_id);

    auto it = std::next(running.begin());
    while (it != running.end()) {
        llama_sequence_group_ptr group_ptr = std::move(*it);
        it = running.erase(it);
        swap_out_or_recompute(std::move(group_ptr));
    }

    llama_sequence_group * priority_group = running.front().get();
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(), [&](llama_sequence_group * group) {
                         return group != priority_group;
                     }),
                     candidates.end());
}


void llama_paged_scheduler_impl::process_running_list(llama_sequence_group_raw_list & candidates) {
    GGML_ASSERT(kv_cache_manager && "kv_cache_manager is nullptr.");

    llama_sequence_group_list::iterator it = running.begin();
    while (it != running.end()) {
        llama_sequence_group * group = it->get();
        GGML_ASSERT(group && "group is nullptr.");

        if (group->status == llama_sequence_group_status::FINISHED) {
            const bool was_priority = group->request_id == priority_request_id;
            const int32_t request_id = group->request_id;
            const bool keep = finish(*group);
            if (keep) {
                retained[request_id] = std::move(*it);
            }
            it = running.erase(it);
            if (was_priority) {
                priority_request_id = -1;
                LLAMA_LOG_DEBUG("%s: priority request completed; resuming queued requests.\n", __func__);
            }
            continue;
        }
        const int32_t scheduled_tokens = get_scheduled_tokens(*group);
        const int32_t allocation_tokens = scheduled_tokens +
            (group->n_past < group->n_prompt && group->n_past + scheduled_tokens == group->n_prompt ? 1 : 0);
        uint32_t current_capacity  = group->block_table.size() * block_size;
        uint32_t required_capacity = group->n_past + allocation_tokens;
        LLAMA_LOG_DEBUG(
            "%s: (running) request_id=%d: current_capacity (tokens)=%d, required capacity (tokens) = %d toks\n",
            __func__, group->request_id, current_capacity, required_capacity);
        if (required_capacity > current_capacity) {
            LLAMA_LOG_DEBUG("%s: (running_pending) request_id=%d: requires a new block to decode.\n", __func__,
                            group->request_id);
            bool success = kv_cache_manager->allocate(allocation_tokens, *group);
            if (!success) {
                evict_retained_requests();
                success = kv_cache_manager->allocate(allocation_tokens, *group);
            }
            if (!success) {
                if (running.size() > 1) {
                    activate_priority_request(candidates);

                    if (group->request_id != priority_request_id) {
                        it = running.end();
                        continue;
                    }

                    // Try allocating again after swapping younger requests.
                    success = kv_cache_manager->allocate(allocation_tokens, *group);
                }

                if (!success) {
                    // If allocate failed, it means we must evict the current request
                    llama_sequence_group_ptr self = std::move(*it);
                    it                            = running.erase(it);

                    swap_out_or_recompute(std::move(self));
                    continue;
                }
            }
            LLAMA_LOG_DEBUG("%s: (running_restored) request_id=%d: found a new block to continue decoding.\n", __func__,
                            group->request_id);
        }

        // A request is a candidate if we there is still room for generation without adding blocks
        // or if there was enough GPU memory to allocate another physical block.
        candidates.push_back(group);
        ++it;
    }
}

void llama_paged_scheduler_impl::process_swapped_list(llama_sequence_group_raw_list & candidates) {
    GGML_ASSERT(kv_cache_manager && "kv_cache_manager is nullptr.");
    if (priority_request_id != -1) {
        return;
    }
    llama_sequence_group_list::iterator it = swapped.begin();
    while (it != swapped.end()) {
        llama_sequence_group * group = it->get();

        GGML_ASSERT(group && "the group to swap is nullptr.");
        const int32_t scheduled_tokens = get_scheduled_tokens(*group);
        const int32_t allocation_tokens = scheduled_tokens +
            (group->n_past < group->n_prompt && group->n_past + scheduled_tokens == group->n_prompt ? 1 : 0);
        bool success = kv_cache_manager->swap_in(*group, allocation_tokens);
        if (!success && !retained.empty()) {
            evict_retained_requests();
            success = kv_cache_manager->swap_in(*group, allocation_tokens);
        }
        if (!success) {
            // We respect FCFS, so we stop here to prevent a younger swapped request from jumping ahead.
            break;
        }
        candidates.push_back(group);
        llama_sequence_group_ptr group_ptr = std::move(*it);
        LLAMA_LOG_DEBUG("%s: (swapped_in) request_id=%d back in for processing.\n", __func__, group_ptr->request_id);
        set_running(std::move(group_ptr));
        it = swapped.erase(it);
    }
}

void llama_paged_scheduler_impl::process_waiting_list(llama_sequence_group_raw_list & candidates,
                                                      int32_t                         remaining_token_budget) {
    GGML_ASSERT(kv_cache_manager && "kv_cache_manager is nullptr.");
    if (priority_request_id != -1) {
        return;
    }
    llama_sequence_group_list::iterator it    = waiting.begin();
    size_t                              count = 0;
    while (it != waiting.end()) {
        llama_sequence_group * group = it->get();
        GGML_ASSERT(group && "the waiting group is nullptr.");

        if (remaining_token_budget <= 0) {
            break;
        }

        const int32_t batch_tokens = group->n_past < group->n_prompt
            ? std::min(get_scheduled_tokens(*group), remaining_token_budget)
            : get_scheduled_tokens(*group);
        if (batch_tokens > remaining_token_budget) {
            break;
        }
        const int32_t allocation_tokens = batch_tokens +
            (group->n_past < group->n_prompt && group->n_past + batch_tokens == group->n_prompt ? 1 : 0);

        ++count;
        // Reserve room for speculative decode without charging it to prefill.
        bool success = kv_cache_manager->allocate(allocation_tokens, *group);
        if (!success && !retained.empty()) {
            evict_retained_requests();
            success = kv_cache_manager->allocate(allocation_tokens, *group);
        }
        if (!success) {
            // We respect FCFS, so we stop here to prevent a younger waiting request from jumping ahead.
            break;
        }
        candidates.push_back(group);
        remaining_token_budget -= batch_tokens;
        llama_sequence_group_ptr group_ptr = std::move(*it);
        LLAMA_LOG_DEBUG("%s: (start) request_id=%d sent for processing.\n", __func__, group_ptr->request_id);
        set_running(std::move(group_ptr));
        it = waiting.erase(it);
    }
    if (count > 0) {
        LLAMA_LOG_DEBUG("%s: Started %ld waiting requests\n", __func__, count);
    }
}

int32_t llama_paged_scheduler_impl::calculate_global_slot_index(int32_t                 token_pos,
                                                                std::vector<uint32_t> & block_table) {
    GGML_ASSERT(block_size && "block_size needs to be greater than 0");
    const int32_t block_table_id = token_pos / block_size;
    const int32_t offset         = token_pos % block_size;

    const size_t block_table_size = block_table.size();
    if ((size_t) block_table_id >= block_table_size) {
        LLAMA_LOG_ERROR("%s: block_table_id=%d is OOB for pos=%d. Block table size=%ld.\n", __func__, block_table_id,
                        token_pos, block_table_size);
        LLAMA_LOG_ERROR("%s: block_table_contents: [ ", __func__);
        for (size_t id = 0; id < block_table_size; ++id) {
            LLAMA_LOG_ERROR("%d ", block_table[id]);
            if (id == block_table_size - 1) {
                LLAMA_LOG_ERROR("]\n");
            }
        }
        GGML_ASSERT(false && "block_table_id OOB");
    }
    const int32_t block_id = block_table.at(block_table_id);

    return (block_id * block_size) + offset;
}

void llama_paged_scheduler_impl::clear_batch(llama_batch & batch) {
    LLAMA_LOG_DEBUG("%s: clearing batch.", __func__);
    // Invalidate last scheduled batch info before freeing the arrays
    // (MUST be called before the delete[]).
    kv_cache_manager->set_paged_batch_info(nullptr);

    delete[] curr_info.write_slots;
    delete[] curr_info.block_table;
    delete[] curr_info.context_lens;
    delete[] curr_info.batch_offsets;
    delete[] curr_info.batch_lens;
    curr_info = {};  // reset to defaults

    batch.n_tokens = 0;
}

void llama_paged_scheduler_impl::populate_batch_from(const llama_sequence_group_raw_list & candidates,
                                                     llama_batch &                         batch) {
    if (candidates.empty()) {
        LLAMA_LOG_DEBUG("%s: No candidates for this step.\n", __func__);
        batch.n_tokens = 0;
        return;
    }
    int32_t total_tokens = 0;
    int32_t batch_size   = candidates.size();
    int32_t max_blocks   = 0;

    LLAMA_LOG_DEBUG("%s: Creating batch from candidates (%d requests). n_batch=%d\n", __func__, batch_size, n_batch);

    // Calculating required sizes
    for (const auto & group : candidates) {
        GGML_ASSERT(group && "candidate request is nullptr.");
        total_tokens += get_scheduled_tokens(*group);
        max_blocks = std::max(max_blocks, (int32_t) group->block_table.size());
    }

    GGML_ASSERT(total_tokens <= (int32_t) n_batch && "total_tokens exceeds n_batch - token budget logic is broken");

    // Allocate for the scheduler's full token budget once so graph input pointers stay stable across steps.
    if (batch.token == nullptr) {
        batch = llama_batch_init(n_batch, 0, 1);
    }
    GGML_ASSERT(batch.token != nullptr && "llama_batch_init failed to allocate tokens.");

    batch.n_tokens = total_tokens;

    curr_info.n_seq            = batch_size;
    curr_info.n_tokens         = total_tokens;
    curr_info.n_blocks_per_seq = max_blocks;

    curr_info.write_slots   = new int32_t[total_tokens];
    curr_info.block_table   = new int32_t[batch_size * max_blocks];
    curr_info.context_lens  = new int32_t[batch_size];
    curr_info.batch_offsets = new int32_t[batch_size];
    curr_info.batch_lens    = new int32_t[batch_size];
    LLAMA_LOG_DEBUG("%s: created llama_batch: n_seq=%d, n_tokens=%d, n_blocks_per_seq=%d\n", __func__, curr_info.n_seq,
                    batch.n_tokens, curr_info.n_blocks_per_seq);

    int32_t token_offset = 0;
    for (int seq_id = 0; seq_id < batch_size; ++seq_id) {
        llama_sequence_group * group = candidates[seq_id];
        GGML_ASSERT(group && "Make sure the candidates are not nullptr.");

        const bool    is_prefill = group->n_past < group->n_prompt;
        const int32_t new_tokens = get_scheduled_tokens(*group);

        if (is_prefill) {
            GGML_ASSERT(group->logical_seq.size() >= group->n_past + (size_t) new_tokens &&
                        "logical_seq too small for prefill");
        } else {
            GGML_ASSERT(!group->logical_seq.empty() && "logical_seq empty during decode");
        }

        for (int token_idx = 0; token_idx < new_tokens; ++token_idx) {
            int32_t batch_start_id = token_offset + token_idx;

            batch.token[batch_start_id] = is_prefill ? group->logical_seq[group->n_past + token_idx] : group->logical_seq.back();
            batch.pos[batch_start_id]   = group->n_past + token_idx;  // n_past starts at 0

            batch.n_seq_id[batch_start_id]  = 1;
            batch.seq_id[batch_start_id][0] = group->request_id;

            batch.logits[batch_start_id] = !is_prefill ||
                (group->n_past + token_idx + 1 == group->n_prompt);

            int32_t token_pos                     = group->n_past + token_idx;
            if (!is_prefill) {
                batch.logits[batch_start_id] = true;
            }
            curr_info.write_slots[batch_start_id] = calculate_global_slot_index(token_pos, group->block_table);
            LLAMA_LOG_DEBUG("%s: llama_batch seq_id: %d (req_id %d) token %d: pos: %d, global_slot_idx=%d\n", __func__,
                            seq_id, group->request_id, token_idx, token_pos, curr_info.write_slots[batch_start_id]);
        }

        // Populate block table (1D): [batch_size * max_blocks]
        const int32_t curr_block_table_size = group->block_table.size();
        for (int block = 0; block < max_blocks; ++block) {
            int  flattened_id                   = (seq_id * max_blocks) + block;  // row-major
            bool need_padding                   = block >= curr_block_table_size;
            curr_info.block_table[flattened_id] = need_padding ? -1 : group->block_table[block];
        }

        curr_info.context_lens[seq_id]  = group->n_past + new_tokens;
        curr_info.batch_offsets[seq_id] = token_offset;
        curr_info.batch_lens[seq_id]    = new_tokens;
        token_offset += new_tokens;
    }
}

// new_tokens contain 1 token per sequence in the batch
// accepted contains one count per sequence; empty uses the full batch lengths.
void llama_paged_scheduler_impl::update(const llama_batch &              batch,
                                        const std::vector<llama_token> & new_tokens,
                                        const std::vector<uint32_t> &    accepted,
                                        const int8_t *                   stop_flags) {
    GGML_ASSERT((int32_t) new_tokens.size() >= curr_info.n_seq && "new_tokens size does not match with batch size.");
    GGML_ASSERT(accepted.empty() || (int32_t) accepted.size() >= curr_info.n_seq);
    GGML_ASSERT(stop_flags != nullptr && "stop_flags can't be null");

    for (int i = 0; i < curr_info.n_seq; ++i) {
        const int32_t token_offset = curr_info.batch_offsets[i];
        const int32_t request_id = batch.seq_id[token_offset][0];
        auto it = id_to_group.find(request_id);
        if (it == id_to_group.end()) {
            LLAMA_LOG_WARN("%s: request_id %d not found in scheduler, skipping\n", __func__, request_id);
            continue;
        }

        llama_sequence_group * group = it->second;
        GGML_ASSERT(group != nullptr);
        const uint32_t proposal = curr_info.batch_lens[i];
        const uint32_t commit = accepted.empty() ? proposal : std::min(accepted[i], proposal);
        GGML_ASSERT(commit > 0 && commit <= proposal);
        const bool was_prefill = group->n_past < group->n_prompt;

        const llama_pos range_min = kv_cache_manager->seq_pos_min(request_id);
        if (range_min < 0) {
            kv_cache_manager->set_seq_min_pos(request_id, batch.pos[token_offset]);
        }
        kv_cache_manager->set_seq_max_pos(request_id, batch.pos[token_offset + commit - 1]);

        group->n_past += commit;
        group->n_decoded += commit;
        if (was_prefill) {
            GGML_ASSERT(group->n_past <= group->n_prompt);
            if (group->n_past == group->n_prompt) {
                group->t_first_token_us = ggml_time_us();
                group->logical_seq.push_back(new_tokens[i]);
            }
        } else {
            for (uint32_t j = 1; j < commit; ++j) {
                group->logical_seq.push_back(batch.token[token_offset + j]);
            }
            group->logical_seq.push_back(new_tokens[i]);
        }

        if (commit < proposal) {
            if (recurrent_manager && !recurrent_manager->seq_rm(request_id, group->n_past, -1)) {
                throw std::runtime_error("failed to roll back paged recurrent state");
            }
            kv_cache_manager->release_seq_tail(request_id, group->n_past);
        }

        if (stop_flags[i] || group->n_past >= n_seq_max_ctx) {
            group->status = llama_sequence_group_status::FINISHED;
            complete_request(request_id);
        }
    }
}

void llama_paged_scheduler_impl::set_on_finish(llama_paged_on_finish_cb cb, void * user_data) {
    on_finish_cb        = cb;
    on_finish_user_data = user_data;
}

llama_sequence_group * llama_paged_scheduler_impl::get_group_from_id(int32_t request_id) const {
    return id_to_group.count(request_id) ? id_to_group.at(request_id) : nullptr;
}

const llama_paged_batch_info * llama_paged_scheduler_impl::get_curr_batch_info() const {
    return &curr_info;
}
