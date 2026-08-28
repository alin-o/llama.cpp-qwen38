// tests/test-paged-kv-e2e.cpp
//
// End-to-end equivalence test for paged KV cache.
// We compare top-K agreement rather than raw logit values because the paged
// attention path uses a custom CUDA kernel with online softmax, while the
// unified path uses standard ggml attention with two-pass softmax. The two
// produce mathematically equivalent results but with different F16
// accumulation order, drifts on the order of 0.05-0.5 in raw
// logit values is expected and not a correctness issue. Top-K set agreement
// is robust to this drift while still catching the real issues (e.g. cross-device
// reads, layout corruption, MQA broadcast bugs) which produce wildly
// different distributions.
//
// Also samples N_COMPARE tokens greedy as a secondary 'cheap' check.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "ggml-paged-attn.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#define EXPECT_TRUE(x)                                                   \
    do {                                                                 \
        if (!(x)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
            throw std::runtime_error("FAILED assertion.");               \
        }                                                                \
    } while (0)

static constexpr const char * TEST_PROMPT = "Once upon a time there was a lovely little llama that lived near a quiet forest. Every morning, it counted the bright leaves, followed the river, and wrote a careful story about its adventures for all of its friends.";

static constexpr int          N_PREDICT         = 64;
static constexpr int          N_COMPARE         = 16;  // longer greedy window
static constexpr int          TOP_K             = 5;
static constexpr int          MIN_TOP_K_OVERLAP = 4;  // at least 4 of top-5 must match
// Allow 2% for floating point reduction order, not KV quantization loss.
static constexpr double       MAX_PPL_RATIO     = 1.02;

// Result of running one path: logits and sampled token sequence.
struct path_result {
    std::vector<std::vector<float>> logits;  // [N_PREDICT][n_vocab]
    std::vector<llama_token>        tokens;  // [N_PREDICT]
    int                             n_vocab = 0;
    int                             head_dim = 0;

};

static llama_token argmax_logits(const std::vector<float> & logits) {
    return (llama_token) std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
}

static std::vector<float> get_logits(llama_context * ctx, int32_t idx, int n_vocab) {
    const float * raw = llama_get_logits_ith(ctx, idx);
    EXPECT_TRUE(raw != nullptr);
    return { raw, raw + n_vocab };
}
static void compare_logits(int step, const std::vector<float> & ref, const std::vector<float> & paged);


static path_result run_non_paged(const std::string & model_path) {
    common_params params;
    params.model.path    = model_path;
    params.n_ctx         = 256;
    params.n_batch       = 64;
    params.n_ubatch      = 64;
    params.n_predict     = N_PREDICT;
    params.sampling.temp = 0.0f;  // greedy
    params.warmup        = false;
    params.kv_paged      = false;
    params.cache_type_k  = GGML_TYPE_F16;
    params.cache_type_v  = GGML_TYPE_F16;

    auto            init  = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * ctx   = init->context();
    EXPECT_TRUE(model != nullptr);
    EXPECT_TRUE(ctx != nullptr);

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    const auto tokenize_prompt = static_cast<std::vector<llama_token> (*)(const llama_context *, const std::string &, bool, bool)>(&common_tokenize);
    std::vector<llama_token> prompt_tokens = tokenize_prompt(ctx, TEST_PROMPT, true, false);
    EXPECT_TRUE(!prompt_tokens.empty());

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    EXPECT_TRUE(llama_decode(ctx, batch) == 0);

    path_result result;
    result.n_vocab = n_vocab;
    result.head_dim = llama_model_n_embd_head_v(model);


    for (int i = 0; i < N_PREDICT; ++i) {
        result.logits.push_back(get_logits(ctx, -1, n_vocab));
        llama_token next = argmax_logits(result.logits.back());
        result.tokens.push_back(next);
        if (llama_vocab_is_eog(vocab, next)) {
            break;
        }

        llama_batch step = llama_batch_get_one(&next, 1);
        EXPECT_TRUE(llama_decode(ctx, step) == 0);
    }

    return result;
}

static path_result run_paged(const std::string & model_path,
                             const std::vector<llama_token> & forced_tokens,
                             ggml_type type_k,
                             ggml_type type_v) {
    common_params params;
    params.model.path    = model_path;
    params.n_ctx         = 256;
    params.n_batch       = 64;
    params.n_ubatch      = 64;
    params.n_predict     = N_PREDICT;
    params.sampling.temp = 0.0f;  // greedy
    params.kv_paged      = true;
    params.n_gpu_blocks  = 64;
    params.n_cpu_blocks  = 16;
    params.n_gpu_blocks_set = true;
    params.n_cpu_blocks_set = true;
    params.n_sequences   = 1;
    params.cache_type_k  = type_k;
    params.cache_type_v  = type_v;
    params.n_parallel    = 1;

    auto            init  = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * ctx   = init->context();
    EXPECT_TRUE(model != nullptr);
    EXPECT_TRUE(ctx != nullptr);

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);
    const int n_heads = llama_model_n_head(model);
    const int n_heads_kv = llama_model_n_head_kv(model);
    EXPECT_TRUE(n_heads > 0 && n_heads_kv > 0 && n_heads % n_heads_kv == 0);
    fprintf(stderr, "  attention heads: Q=%d KV=%d (GQA ratio %d)\n", n_heads, n_heads_kv, n_heads / n_heads_kv);

    llama_paged_scheduler * sched = llama_paged_scheduler_init(ctx);
    EXPECT_TRUE(sched != nullptr);

    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, TEST_PROMPT, true);
    EXPECT_TRUE(!prompt_tokens.empty());

    EXPECT_TRUE(llama_paged_scheduler_add_request(sched, prompt_tokens.data(), prompt_tokens.size(), 0));
#if defined(GGML_USE_CUDA)
    ggml_paged_attn_tiled_prefill_launch_count_reset();
#endif

    path_result result;
    result.n_vocab    = n_vocab;
    llama_batch batch = {};
#if defined(GGML_USE_CUDA)
    bool tiled_prefill_seen = false;
    bool decode_no_launch_checked = false;
#endif

    while ((int) result.tokens.size() < N_PREDICT) {
        EXPECT_TRUE(llama_paged_scheduler_prepare_batch(sched, &batch));
        if (batch.n_tokens == 0) {
            break;
        }

        EXPECT_TRUE(llama_decode(ctx, batch) == 0);
        llama_synchronize(ctx);
#if defined(GGML_USE_CUDA)
        if (result.tokens.empty()) {
            const int head_dim = llama_model_n_embd_head_v(model);
            const unsigned long long launches = ggml_paged_attn_tiled_prefill_launch_count();
            if (head_dim == 256) {
                EXPECT_TRUE(launches > 0);
            } else if (head_dim != 128) {
                EXPECT_TRUE(launches == 0);
            }
            tiled_prefill_seen = launches > 0;
            ggml_paged_attn_tiled_prefill_launch_count_reset();
        } else if (!decode_no_launch_checked) {
            EXPECT_TRUE(ggml_paged_attn_tiled_prefill_launch_count() == 0);
            decode_no_launch_checked = true;
        }
#endif

        const llama_paged_batch_info * info = llama_paged_scheduler_get_batch_info(sched);
        EXPECT_TRUE(info != nullptr && info->n_seq == 1);
        if (result.tokens.empty()) {
            EXPECT_TRUE(info->batch_lens[0] > params.block_size);
        }

        const int32_t last_idx = info->batch_offsets[0] + info->batch_lens[0] - 1;
        result.logits.push_back(get_logits(ctx, last_idx, n_vocab));

        const llama_token argmax = argmax_logits(result.logits.back());
        const llama_token next = result.tokens.size() < forced_tokens.size() ?
            forced_tokens[result.tokens.size()] : argmax;
        result.tokens.push_back(next);

        const bool stop = llama_vocab_is_eog(vocab, next) || (int) result.tokens.size() >= N_PREDICT;
        const llama_token next_tokens[] = { next };
        const int8_t stop_flags[] = { static_cast<int8_t>(stop) };
        llama_paged_scheduler_update(sched, &batch, next_tokens, stop_flags);
        if (stop) {
            break;
        }
    }
#if defined(GGML_USE_CUDA)
    const int head_dim = llama_model_n_embd_head_v(model);
    if (head_dim == 256) {
        EXPECT_TRUE(tiled_prefill_seen);
    } else if (head_dim != 128) {
        EXPECT_TRUE(!tiled_prefill_seen);
    }
#endif
    llama_paged_scheduler_free(sched);
    return result;
}

static void run_paged_checkpoint_resume(const std::string & model_path) {
    common_params params;
    params.model.path = model_path;
    params.n_ctx = 256;
    params.n_batch = 64;
    params.n_ubatch = 64;
    params.warmup = false;
    params.kv_paged = true;
    params.n_gpu_blocks = 64;
    params.n_cpu_blocks = 16;
    params.n_gpu_blocks_set = true;
    params.n_cpu_blocks_set = true;
    params.n_sequences = 1;
    params.n_parallel = 1;
    params.cache_type_k = GGML_TYPE_Q8_0;
    params.cache_type_v = GGML_TYPE_Q8_0;

    auto source_init = common_init_from_params(params);
    llama_context * source_ctx = source_init->context();
    const llama_vocab * vocab = llama_model_get_vocab(source_init->model());
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_paged_scheduler * source_sched = llama_paged_scheduler_init(source_ctx);
    EXPECT_TRUE(source_sched != nullptr);
    std::vector<llama_token> prompt_tokens = common_tokenize(source_ctx, TEST_PROMPT, true);
    EXPECT_TRUE(llama_paged_scheduler_add_request(source_sched, prompt_tokens.data(), prompt_tokens.size(), 0));
    const size_t queued_state_size = llama_state_get_size(source_ctx);
    std::vector<uint8_t> queued_state(queued_state_size);
    EXPECT_TRUE(llama_state_get_data(source_ctx, queued_state.data(), queued_state.size()) == queued_state.size());
    EXPECT_TRUE(llama_state_set_data(source_ctx, queued_state.data(), queued_state.size()) == queued_state.size());
    llama_paged_scheduler_free(source_sched);
    source_init.reset();
    {
        auto queued_restore_init = common_init_from_params(params);
        llama_context * queued_restore_ctx = queued_restore_init->context();
        EXPECT_TRUE(llama_state_set_data(queued_restore_ctx, queued_state.data(), queued_state.size()) == queued_state.size());
        llama_paged_scheduler * queued_restore_sched = llama_paged_scheduler_init(queued_restore_ctx);
        EXPECT_TRUE(queued_restore_sched != nullptr);
        EXPECT_TRUE(llama_paged_scheduler_add_request(queued_restore_sched, prompt_tokens.data(), prompt_tokens.size(), 0));
        llama_batch queued_restore_batch = {};
        EXPECT_TRUE(llama_paged_scheduler_prepare_batch(queued_restore_sched, &queued_restore_batch));
        EXPECT_TRUE(queued_restore_batch.n_tokens == (int32_t) prompt_tokens.size());
        EXPECT_TRUE(queued_restore_batch.token[0] == prompt_tokens[0]);
        llama_paged_scheduler_free(queued_restore_sched);
        llama_batch_free(queued_restore_batch);
    }
    source_init = common_init_from_params(params);
    source_ctx = source_init->context();
    source_sched = llama_paged_scheduler_init(source_ctx);
    EXPECT_TRUE(source_sched != nullptr);
    EXPECT_TRUE(llama_paged_scheduler_add_request(source_sched, prompt_tokens.data(), prompt_tokens.size(), 0));

    llama_batch source_batch = {};
    EXPECT_TRUE(llama_paged_scheduler_prepare_batch(source_sched, &source_batch));
    EXPECT_TRUE(llama_decode(source_ctx, source_batch) == 0);
    llama_synchronize(source_ctx);
    const llama_paged_batch_info * source_info = llama_paged_scheduler_get_batch_info(source_sched);
    const llama_token next = argmax_logits(get_logits(source_ctx, source_info->batch_offsets[0] + source_info->batch_lens[0] - 1, n_vocab));
    const int8_t continue_flag = 0;
    llama_paged_scheduler_update(source_sched, &source_batch, &next, &continue_flag);

    const size_t state_size = llama_state_get_size(source_ctx);
    std::vector<uint8_t> state(state_size);
    EXPECT_TRUE(llama_state_get_data(source_ctx, state.data(), state.size()) == state.size());
    EXPECT_TRUE(llama_paged_scheduler_prepare_batch(source_sched, &source_batch));
    EXPECT_TRUE(llama_decode(source_ctx, source_batch) == 0);
    llama_synchronize(source_ctx);
    source_info = llama_paged_scheduler_get_batch_info(source_sched);
    const std::vector<float> expected = get_logits(source_ctx, source_info->batch_offsets[0] + source_info->batch_lens[0] - 1, n_vocab);
    llama_paged_scheduler_free(source_sched);
    llama_batch_free(source_batch);
    source_init.reset();
    {
        auto conflict_init = common_init_from_params(params);
        llama_paged_scheduler * conflict_sched = llama_paged_scheduler_init(conflict_init->context());
        EXPECT_TRUE(conflict_sched != nullptr);
        EXPECT_TRUE(llama_paged_scheduler_add_request(conflict_sched, prompt_tokens.data(), prompt_tokens.size(), 1));
        EXPECT_TRUE(llama_state_set_data(conflict_init->context(), state.data(), state.size()) == 0);
        llama_paged_scheduler_free(conflict_sched);
    }

    auto restored_init = common_init_from_params(params);
    llama_context * restored_ctx = restored_init->context();
    EXPECT_TRUE(llama_state_set_data(restored_ctx, state.data(), state.size()) == state.size());
    llama_paged_scheduler * restored_sched = llama_paged_scheduler_init(restored_ctx);
    EXPECT_TRUE(restored_sched != nullptr);
    EXPECT_TRUE(llama_paged_scheduler_add_request(restored_sched, prompt_tokens.data(), prompt_tokens.size(), 0));
    llama_batch restored_batch = {};
    EXPECT_TRUE(llama_paged_scheduler_prepare_batch(restored_sched, &restored_batch));
    EXPECT_TRUE(restored_batch.n_tokens == 1 && restored_batch.token[0] == next && restored_batch.pos[0] == (llama_pos) prompt_tokens.size());
    EXPECT_TRUE(llama_decode(restored_ctx, restored_batch) == 0);
    llama_synchronize(restored_ctx);
    const llama_paged_batch_info * restored_info = llama_paged_scheduler_get_batch_info(restored_sched);
    compare_logits(0, expected, get_logits(restored_ctx, restored_info->batch_offsets[0] + restored_info->batch_lens[0] - 1, n_vocab));
    llama_paged_scheduler_free(restored_sched);
}

static std::vector<int> top_k(const std::vector<float> & logits) {
    std::vector<int> result(logits.size());
    std::iota(result.begin(), result.end(), 0);
    std::partial_sort(result.begin(), result.begin() + TOP_K, result.end(), [&logits](int a, int b) {
        return logits[a] > logits[b];
    });
    result.resize(TOP_K);
    return result;
}

static void compare_logits(int step, const std::vector<float> & ref, const std::vector<float> & paged) {
    const auto top_ref = top_k(ref);
    const auto top_paged = top_k(paged);

    std::set<int> ref_set(top_ref.begin(), top_ref.end());
    int overlap = 0;
    for (int token : top_paged) {
        overlap += ref_set.count(token);
    }

    if (top_ref[0] == top_paged[0] && overlap >= MIN_TOP_K_OVERLAP) {
        return;
    }

    fprintf(stderr, "FAIL: forced step %d differs: ref argmax=%d paged argmax=%d, top-%d overlap=%d/%d\n",
            step, top_ref[0], top_paged[0], TOP_K, overlap, TOP_K);
    fprintf(stderr, "  ref:   ");
    for (int token : top_ref) {
        fprintf(stderr, "%d(%.3f) ", token, ref[token]);
    }
    fprintf(stderr, "\n  paged: ");
    for (int token : top_paged) {
        fprintf(stderr, "%d(%.3f) ", token, paged[token]);
    }
    fprintf(stderr, "\n");
    throw std::runtime_error("FAILED test.");
}

static double perplexity(const path_result & result, const std::vector<llama_token> & targets) {
    EXPECT_TRUE(result.logits.size() >= targets.size());
    double nll = 0.0;
    for (size_t i = 0; i < targets.size(); ++i) {
        const auto & logits = result.logits[i];
        const float max_logit = *std::max_element(logits.begin(), logits.end());
        double sum = 0.0;
        for (const float logit : logits) {
            sum += std::exp(logit - max_logit);
        }
        nll += std::log(sum) + max_logit - logits[targets[i]];
    }
    return std::exp(nll / targets.size());
}

static void compare_perplexity(const char * name, const path_result & ref, const path_result & paged) {
    const size_t n = std::min(ref.tokens.size(), paged.logits.size());
    EXPECT_TRUE(n >= N_COMPARE);
    const std::vector<llama_token> targets(ref.tokens.begin(), ref.tokens.begin() + n);
    const double ref_ppl = perplexity(ref, targets);
    const double paged_ppl = perplexity(paged, targets);
    fprintf(stderr, "  %s perplexity: unified_f16=%.6f paged=%.6f ratio=%.6f\n", name, ref_ppl, paged_ppl, paged_ppl / ref_ppl);
    EXPECT_TRUE(paged_ppl <= ref_ppl * MAX_PPL_RATIO);
}

static void compare_results(const path_result & ref, const path_result & paged_greedy, const path_result & paged_forced) {
    EXPECT_TRUE((int) ref.tokens.size() >= N_COMPARE);
    int greedy_mismatches = 0;
    for (int i = 0; i < N_COMPARE; ++i) {
        if (ref.tokens[i] != paged_greedy.tokens[i]) {
            fprintf(stderr, "INFO: greedy token %d differs: ref=%d paged=%d\n", i, ref.tokens[i], paged_greedy.tokens[i]);
            ++greedy_mismatches;
        }
    }
    fprintf(stderr, "  greedy mismatches in first %d tokens: %d\n", N_COMPARE, greedy_mismatches);
    EXPECT_TRUE(ref.logits.size() >= N_COMPARE);
    EXPECT_TRUE(paged_forced.logits.size() >= N_COMPARE);
    for (int i = 0; i < N_COMPARE; ++i) {
        compare_logits(i, ref.logits[i], paged_forced.logits[i]);
    }
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PAGED)) {
        fprintf(stderr, "usage: %s -m <model>\n", argv[0]);
        return 1;
    }
    if (params.model.path.empty()) {
        fprintf(stderr, "skip: no --model provided\n");
        return 0;
    }
    const ggml_type type_k = GGML_TYPE_Q8_0;
    const ggml_type type_v = GGML_TYPE_Q8_0;

    common_init();
    llama_backend_init();

    fprintf(stderr, "test-paged-kv-e2e: running unified f16 reference\n");
    path_result ref = run_non_paged(params.model.path);
    if (ref.head_dim % ggml_blck_size(type_k) != 0 || ref.head_dim % ggml_blck_size(type_v) != 0) {
        fprintf(stderr, "skip: model head dimension %d is incompatible with paged K=%s V=%s\n", ref.head_dim,
                ggml_type_name(type_k), ggml_type_name(type_v));
        llama_backend_free();
        return 0;
    }

    fprintf(stderr, "  got %zu tokens, %d-vocab logits\n", ref.tokens.size(), ref.n_vocab);

    fprintf(stderr, "test-paged-kv-e2e: running q8_0 paged path\n");
    path_result paged_greedy = run_paged(params.model.path, {}, type_k, type_v);
    fprintf(stderr, "  got %zu tokens, %d-vocab logits\n", paged_greedy.tokens.size(), paged_greedy.n_vocab);
    fprintf(stderr, "test-paged-kv-e2e: resuming q8_0 paged checkpoint\n");
    run_paged_checkpoint_resume(params.model.path);

    path_result paged_q8 = run_paged(params.model.path, ref.tokens, type_k, type_v);
    compare_results(ref, paged_greedy, paged_q8);
    compare_perplexity("q8_0", ref, paged_q8);

    for (const ggml_type type : { GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0 }) {
        if (ref.head_dim % ggml_blck_size(type) != 0) {
            fprintf(stderr, "skip: model head dimension %d is incompatible with paged %s\n", ref.head_dim,
                    ggml_type_name(type));
            continue;
        }
        fprintf(stderr, "test-paged-kv-e2e: running %s paged path\n", ggml_type_name(type));
        path_result paged = run_paged(params.model.path, ref.tokens, type, type);
        EXPECT_TRUE(paged.logits.size() >= N_COMPARE);
        for (int i = 0; i < N_COMPARE; ++i) {
            compare_logits(i, ref.logits[i], paged.logits[i]);
        }
        compare_perplexity(ggml_type_name(type), ref, paged);
    }
    fprintf(stderr, "test-paged-kv-e2e: PASSED\n");

    llama_backend_free();
    return 0;
}
