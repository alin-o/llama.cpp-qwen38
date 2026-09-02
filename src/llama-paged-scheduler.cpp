#include "ggml.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-memory-hybrid-paged.h"
#include "llama-paged-scheduler-impl.h"

struct llama_paged_scheduler {
    llama_paged_scheduler_impl impl;

    llama_paged_scheduler(
            uint32_t                  n_ctx,
            uint32_t                  block_sz,
            uint32_t                  n_batch,
            llama_kv_cache_paged *    kv_manager,
            llama_memory_recurrent *  recurrent_manager) :
        impl(n_ctx, block_sz, n_batch, kv_manager, recurrent_manager) {}
};

LLAMA_API struct llama_paged_scheduler * llama_paged_scheduler_init(struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    auto * memory = ctx->get_memory();
    auto * paged_kv = dynamic_cast<llama_kv_cache_paged *>(memory);
    llama_memory_recurrent * recurrent = nullptr;
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid_paged *>(memory)) {
        paged_kv = hybrid->get_mem_attn();
        recurrent = hybrid->get_mem_recr();
    }
    if (!paged_kv) {
        LLAMA_LOG_ERROR(
            "%s: context does not have a paged KV cache. "
            "Make sure to pass --kv-paged (-kvp) and use a "
            "supported architecture. SWA architectures (gemma3, llama4, etc.) "
            "are not yet supported.\n",
            __func__);
        return nullptr;
    }

    // Extract params
    const uint32_t n_ctx    = ctx->n_ctx();
    const uint32_t block_sz = ctx->block_size();
    const uint32_t n_batch  = ctx->n_batch();
    GGML_ASSERT(n_batch == ctx->n_ubatch() && "kv_paged requires n_batch == n_ubatch.");

    try {
        return new llama_paged_scheduler(n_ctx, block_sz, n_batch, paged_kv, recurrent);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: Error when creating llama_paged_scheduler: %s\n", __func__, e.what());
        return nullptr;
    }
}

LLAMA_API void llama_paged_scheduler_free(struct llama_paged_scheduler * sched) {
    if (sched) {
        delete sched;
    }
}

LLAMA_API bool llama_paged_scheduler_prepare_batch(struct llama_paged_scheduler * sched, struct llama_batch * batch) {
    if (!sched || !batch) {
        return false;
    }

    llama_scheduler_status status;
    try {
        status = sched->impl.step(*batch);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return false;
    }

    if (status == llama_scheduler_status::DEADLOCK) {
        LLAMA_LOG_ERROR("%s: Deadlock detected.\n", __func__);
        return false;
    }

    return true;
}

LLAMA_API bool llama_paged_scheduler_add_request(struct llama_paged_scheduler * sched,
                                                 const llama_token *            tokens,
                                                 int32_t                        n_tokens,
                                                 int32_t                        request_id) {
    if (!sched || !tokens) {
        return false;
    }

    llama_sequence_group group;
    group.request_id = request_id;
    group.n_prompt   = n_tokens;
    group.n_decoded  = 0;
    group.n_past     = 0;
    for (int i = 0; i < n_tokens; ++i) {
        group.logical_seq.push_back(tokens[i]);
    }
    group.t_arrival_time = ggml_time_us();  // int64_t milliseconds

    try {
        return sched->impl.queue_request(group);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return false;
    }
}

LLAMA_API bool llama_paged_scheduler_add_request_with_prefix(struct llama_paged_scheduler * sched,
                                                              const llama_token *            tokens,
                                                              int32_t                        n_tokens,
                                                              int32_t                        request_id,
                                                              int32_t                        n_prefix,
                                                              int32_t *                      n_prefix_used) {
    if (n_prefix_used) {
        *n_prefix_used = 0;
    }
    if (!sched || !tokens || n_tokens <= 0 || n_prefix < 0 || n_prefix >= n_tokens) {
        if (sched) {
            sched->impl.remove_request(request_id);
        }
        return false;
    }

    llama_sequence_group group;
    group.request_id = request_id;
    group.n_prompt   = n_tokens;
    group.logical_seq.assign(tokens, tokens + n_tokens);
    group.t_arrival_time = ggml_time_us();

    try {
        if (sched->impl.queue_request(std::move(group), n_prefix)) {
            if (n_prefix_used) {
                const llama_sequence_group * queued = sched->impl.get_group_from_id(request_id);
                *n_prefix_used = queued && queued->n_past == (uint32_t) n_prefix ? n_prefix : 0;
            }
            return true;
        }
        llama_sequence_group cold;
        cold.request_id = request_id;
        cold.n_prompt   = n_tokens;
        cold.logical_seq.assign(tokens, tokens + n_tokens);
        cold.t_arrival_time = ggml_time_us();
        const bool queued = sched->impl.queue_request(std::move(cold));
        if (!queued) {
            sched->impl.remove_request(request_id);
        }
        return queued;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        sched->impl.remove_request(request_id);
        try {
            llama_sequence_group cold;
            cold.request_id = request_id;
            cold.n_prompt   = n_tokens;
            cold.logical_seq.assign(tokens, tokens + n_tokens);
            cold.t_arrival_time = ggml_time_us();
            const bool queued = sched->impl.queue_request(std::move(cold));
            if (!queued) {
                sched->impl.remove_request(request_id);
            }
            return queued;
        } catch (const std::exception & retry_error) {
            LLAMA_LOG_ERROR("%s: cold retry failed: %s\n", __func__, retry_error.what());
            sched->impl.remove_request(request_id);
            return false;
        }
    }
}

LLAMA_API bool llama_paged_scheduler_retain_request(
        struct llama_paged_scheduler * sched, int32_t request_id, bool checkpoint_before_last) {
    return sched && sched->impl.retain_request(request_id, checkpoint_before_last);
}

LLAMA_API bool llama_paged_scheduler_is_retained(
        const struct llama_paged_scheduler * sched, int32_t request_id) {
    return sched && sched->impl.is_retained(request_id);
}

LLAMA_API void llama_paged_scheduler_set_batch_policy(
        struct llama_paged_scheduler *      sched,
        llama_paged_batch_token_limit_cb    token_limit_cb,
        llama_paged_batch_compatible_cb     compatible_cb,
        void *                              user_data) {
    if (sched) {
        sched->impl.set_batch_policy(token_limit_cb, compatible_cb, user_data);
    }
}

LLAMA_API void llama_paged_scheduler_update(struct llama_paged_scheduler * sched,
                                            struct llama_batch *           batch,
                                            const llama_token *            tokens,
                                            const int8_t *                 stop_flags) {
    if (!sched || !batch || !tokens || !stop_flags) {
        return;
    }

    const auto * info = sched->impl.get_curr_batch_info();
    GGML_ASSERT(info != nullptr && "no batch info was set.");
    std::vector<llama_token> tokens_vec(tokens, tokens + info->n_seq);
    sched->impl.update(*batch, tokens_vec, stop_flags);
}
LLAMA_API void llama_paged_scheduler_remove_request(struct llama_paged_scheduler * sched, int32_t request_id) {
    if (sched) {
        sched->impl.remove_request(request_id);
    }
}


LLAMA_API void llama_paged_scheduler_set_on_finish(struct llama_paged_scheduler * sched,
                                                   llama_paged_on_finish_cb       cb,
                                                   void *                         user_data) {
    sched->impl.set_on_finish(cb, user_data);
}

LLAMA_API void llama_paged_scheduler_set_on_recompute(struct llama_paged_scheduler * sched,
                                                      llama_paged_on_recompute_cb    cb,
                                                      void *                         user_data) {
    if (sched) {
        sched->impl.set_on_recompute(cb, user_data);
    }
}

LLAMA_API bool llama_paged_scheduler_get_seq_state(struct llama_paged_scheduler * sched,
                                                   int32_t                        request_id,
                                                   struct llama_paged_seq_state * out_state) {
    if (!sched || !out_state) {
        return false;
    }

    llama_sequence_group * group = sched->impl.get_group_from_id(request_id);
    if (group == nullptr) {
        LLAMA_LOG_ERROR("%s: request_id=%d does not exist.", __func__, request_id);
        return false;
    }

    out_state->request_id       = group->request_id;
    out_state->n_prompt         = group->n_prompt;
    out_state->n_decoded        = group->n_decoded;
    out_state->n_past           = group->n_past;
    out_state->t_arrival_us     = group->t_arrival_time;
    out_state->t_first_token_us = group->t_first_token_us;
    return true;
}

LLAMA_API bool llama_paged_scheduler_get_cache_stats(const struct llama_paged_scheduler * sched,
                                                      struct llama_paged_cache_stats *     out_stats) {
    if (!sched || !out_stats) {
        return false;
    }
    *out_stats = sched->impl.get_cache_stats();
    return true;
}

LLAMA_API const struct llama_paged_batch_info * llama_paged_scheduler_get_batch_info(
    const struct llama_paged_scheduler * sched) {
    if (!sched) {
        return nullptr;
    }
    return sched->impl.get_curr_batch_info();
}
LLAMA_API bool llama_paged_scheduler_prepare_batch_ex(struct llama_paged_scheduler * sched, struct llama_batch * batch, int32_t spec_n) {
    if (!sched || !batch) {
        return false;
    }
    try {
        return sched->impl.step(*batch, spec_n) != llama_scheduler_status::DEADLOCK;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return false;
    }
}

LLAMA_API void llama_paged_scheduler_update_ex(struct llama_paged_scheduler * sched,
                                               struct llama_batch * batch,
                                               const llama_token * tokens,
                                               const uint32_t * accepted,
                                               const int8_t * stop_flags) {
    if (!sched || !batch || !tokens || !stop_flags) {
        return;
    }
    const auto * info = sched->impl.get_curr_batch_info();
    GGML_ASSERT(info != nullptr);
    std::vector<llama_token> token_vec(tokens, tokens + info->n_seq);
    std::vector<uint32_t> accepted_vec;
    if (accepted) {
        accepted_vec.assign(accepted, accepted + info->n_seq);
    }
    sched->impl.update(*batch, token_vec, accepted_vec, stop_flags);
}
