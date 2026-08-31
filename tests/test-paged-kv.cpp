#include "ggml-backend.h"
#include "ggml-paged-attn.h"
#include "llama-block-manager.h"
#include "llama-kv-cache-paged.h"
#include "llama-io.h"
#include "llama-paged-scheduler-impl.h"
#include "llama-hparams.h"

#include <stdexcept>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define TEST(name) static void name()
#define RUN(name)                                   \
    do {                                            \
        fprintf(stderr, "  running %-40s ", #name); \
        fflush(stderr);                             \
        name();                                     \
        fprintf(stderr, "OK\n");                    \
    } while (0)

#define EXPECT_EQ(a, b)                                                                                          \
    do {                                                                                                         \
        auto _a = (a);                                                                                           \
        auto _b = (b);                                                                                           \
        if (_a != _b) {                                                                                          \
            fprintf(stderr, "\n  FAIL %s:%d: expected %s == %s, got %lld vs %lld\n", __FILE__, __LINE__, #a, #b, \
                    (long long) _a, (long long) _b);                                                             \
        }                                                                                                        \
    } while (0)

#define EXPECT_TRUE(x)                                                       \
    do {                                                                     \
        if (!(x)) {                                                          \
            fprintf(stderr, "\n  FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
            std::abort();                                                    \
        }                                                                    \
    } while (0)
#define EXPECT_FALSE(x) EXPECT_TRUE(!(x))

class memory_io final : public llama_io_write_i, public llama_io_read_i {
public:
    void write(const void * src, size_t size) override {
        const auto * bytes = static_cast<const uint8_t *>(src);
        data.insert(data.end(), bytes, bytes + size);
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        const size_t begin = data.size();
        data.resize(begin + size);
        ggml_backend_tensor_get(tensor, data.data() + begin, offset, size);
    }

    void read(void * dst, size_t size) override {
        if (offset + size > data.size()) {
            throw std::runtime_error("paged test input is truncated");
        }
        std::memcpy(dst, data.data() + offset, size);
        offset += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t tensor_offset, size_t size) override {
        if (offset + size > data.size()) {
            throw std::runtime_error("paged test input is truncated");
        }
        ggml_backend_tensor_set(tensor, data.data() + offset, tensor_offset, size);
        offset += size;
    }

    size_t n_bytes() override {
        return offset;
    }

    std::vector<uint8_t> data;
    size_t offset = 0;
};

TEST(test_paged_graph_tensor_compatibility) {
    ggml_init_params params = {
        /*.mem_size   =*/ 2*ggml_tensor_overhead() + 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);
    EXPECT_TRUE(ctx != nullptr);
    ggml_tensor * captured = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    ggml_tensor * current = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    EXPECT_TRUE(captured != nullptr && current != nullptr);
    int buffer_tag = 0;
    int storage[8] = {};
    captured->buffer = reinterpret_cast<ggml_backend_buffer_t>(&buffer_tag);
    current->buffer = captured->buffer;
    captured->data = storage;
    current->data = storage;
    EXPECT_TRUE(llm_graph_can_reuse_paged_tensor(captured, current));
    current->data = storage + 1;
    EXPECT_FALSE(llm_graph_can_reuse_paged_tensor(captured, current));
    current->data = storage;
    current->ne[0] = 2;
    EXPECT_FALSE(llm_graph_can_reuse_paged_tensor(captured, current));
    current->ne[0] = captured->ne[0];
    current->type = GGML_TYPE_F16;
    EXPECT_FALSE(llm_graph_can_reuse_paged_tensor(captured, current));
    current->nb[1] += 4;
    EXPECT_FALSE(llm_graph_can_reuse_paged_tensor(captured, current));
    ggml_free(ctx);
}

TEST(test_paged_graph_reuse_accepts_mapping_remap) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(backend != nullptr);

    constexpr uint32_t n_layers = 1;
    constexpr uint32_t n_heads_kv = 2;
    constexpr uint32_t head_dim = 32;
    constexpr uint32_t block_size = 16;
    constexpr uint32_t n_ubatch = 4;
    constexpr uint32_t n_seq_max = 1;

    llama_kv_cache_paged cache(head_dim, n_heads_kv, block_size, n_layers, n_ubatch, n_seq_max);
    cache.init(backend, backend, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 4, 2, 0.0f);
    llama_ubatch ubatch = {};
    ubatch.n_tokens = 1;
    std::vector<llama_ubatch> ubatches = { ubatch };
    llama_kv_cache_paged_context captured_ctx(&cache, ubatches);
    llama_kv_cache_paged_context current_ctx(&cache, ubatches);
    int32_t write_slots_a[] = { 0 };
    int32_t block_table_a[] = { 0, 1, 2, 3 };
    int32_t context_lens_a[] = { 16 };
    int32_t batch_offsets_a[] = { 0 };
    int32_t batch_lens_a[] = { 1 };
    int32_t write_slots_b[] = { 16 };
    int32_t block_table_b[] = { 3, 2, 1, 0 };
    int32_t context_lens_b[] = { 17 };
    int32_t batch_offsets_b[] = { 0 };
    int32_t batch_lens_b[] = { 1 };
    llama_paged_batch_info info_a = { 4, 1, 1, write_slots_a, block_table_a, context_lens_a, batch_offsets_a, batch_lens_a };
    llama_paged_batch_info info_b = { 4, 1, 1, write_slots_b, block_table_b, context_lens_b, batch_offsets_b, batch_lens_b };
    captured_ctx.set_batch_data(info_a);
    current_ctx.set_batch_data(info_b);

    llama_hparams hparams = {};
    hparams.n_layer_all = n_layers;
    hparams.n_head_kv_arr[0] = n_heads_kv;
    hparams.n_embd_head_k_full = head_dim;
    hparams.n_embd_head_v_full = head_dim;
    llama_cparams cparams = {};
    cparams.block_size = block_size;
    llm_graph_input_attn_kv_paged input(hparams, cparams, &captured_ctx);
    ggml_init_params ggml_params = { 16 * ggml_tensor_overhead() + 4096, nullptr, false };
    ggml_context * ggml_ctx = ggml_init(ggml_params);
    EXPECT_TRUE(ggml_ctx != nullptr);
    input.paged_write_rows = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I32, n_heads_kv);
    input.paged_block_table = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_I32, 4, 1);
    input.paged_context_lens = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I32, 1);
    input.paged_batch_offsets = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I32, 1);
    input.paged_batch_lens = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I32, 1);
    llm_graph_params graph_params = {};
    graph_params.hparams = hparams;
    graph_params.cparams = cparams;
    graph_params.ubatch = ubatch;
    graph_params.mctx = &current_ctx;
    EXPECT_TRUE(input.can_reuse(graph_params));
    current_ctx.set_batch_data(info_a);
    EXPECT_TRUE(input.can_reuse(graph_params));
    int32_t context_lens_long[] = { 65536 };
    llama_paged_batch_info info_long = {
        4, 1, 1, write_slots_b, block_table_b, context_lens_long, batch_offsets_b, batch_lens_b
    };
    current_ctx.set_batch_data(info_long);
    EXPECT_FALSE(input.can_reuse(graph_params));
    ggml_free(ggml_ctx);
    ggml_backend_free(backend);
}

TEST(test_paged_graph_reuse_rejects_pool_change) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(backend != nullptr);

    constexpr uint32_t n_layers = 1;
    constexpr uint32_t n_heads_kv = 2;
    constexpr uint32_t head_dim = 32;
    constexpr uint32_t block_size = 16;
    constexpr uint32_t n_ubatch = 4;
    constexpr uint32_t n_seq_max = 1;

    llama_kv_cache_paged captured_cache(head_dim, n_heads_kv, block_size, n_layers, n_ubatch, n_seq_max);
    llama_kv_cache_paged current_cache(head_dim, n_heads_kv, block_size, n_layers, n_ubatch, n_seq_max);
    captured_cache.init(backend, backend, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 4, 2, 0.0f);
    current_cache.init(backend, backend, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 4, 2, 0.0f);

    llama_ubatch ubatch = {};
    ubatch.n_tokens = 1;
    std::vector<llama_ubatch> ubatches = { ubatch };
    llama_kv_cache_paged_context captured_ctx(&captured_cache, ubatches);
    llama_kv_cache_paged_context current_ctx(&current_cache, ubatches);
    captured_ctx.set_n_tokens(1);
    current_ctx.set_n_tokens(1);
    captured_ctx.set_batch_size(1);
    current_ctx.set_batch_size(1);
    captured_ctx.set_max_blocks(4);
    current_ctx.set_max_blocks(4);

    llama_hparams hparams = {};
    hparams.n_layer_all = n_layers;
    hparams.n_head_kv_arr[0] = n_heads_kv;
    hparams.n_embd_head_k_full = head_dim;
    hparams.n_embd_head_v_full = head_dim;

    llama_cparams cparams = {};
    cparams.block_size = block_size;

    llm_graph_input_attn_kv_paged input(hparams, cparams, &captured_ctx);
    ggml_init_params ggml_params = {
        /*.mem_size   =*/ 16 * ggml_tensor_overhead() + 4096,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ggml_ctx = ggml_init(ggml_params);
    EXPECT_TRUE(ggml_ctx != nullptr);
    input.paged_write_rows = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I32, n_heads_kv);
    input.paged_block_table = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_I32, 4, 1);
    input.paged_context_lens = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I32, 1);
    input.paged_batch_offsets = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I32, 1);
    input.paged_batch_lens = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I32, 1);

    llm_graph_params graph_params = {};
    graph_params.hparams = hparams;
    graph_params.cparams = cparams;
    graph_params.ubatch = ubatch;
    graph_params.mctx = &current_ctx;
    EXPECT_FALSE(input.can_reuse(graph_params));
    EXPECT_FALSE(input.paged_k[0] == current_cache.get_k_tensor(0));

    ggml_free(ggml_ctx);
    ggml_backend_free(backend);
}


// Testing block_manager main functionality

TEST(test_block_manager_leak_simple) {
    const uint32_t n_gpu_blocks = 16;
    const uint32_t n_cpu_blocks = 8;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    EXPECT_EQ(block_manager.n_free_gpu_blocks(), n_gpu_blocks);
    EXPECT_EQ(block_manager.n_free_cpu_blocks(), n_cpu_blocks);

    auto gpu_ids = block_manager.checkout_gpu_blocks(10);
    EXPECT_EQ(gpu_ids.size(), 10u);
    EXPECT_EQ(block_manager.n_free_gpu_blocks(), (n_gpu_blocks - 10u));

    auto cpu_ids = block_manager.checkout_cpu_blocks(5);
    EXPECT_EQ(cpu_ids.size(), 5u);
    EXPECT_EQ(block_manager.n_free_cpu_blocks(), (n_cpu_blocks - 5u));

    block_manager.release_gpu_blocks(gpu_ids);
    block_manager.release_cpu_blocks(cpu_ids);

    EXPECT_EQ(block_manager.n_free_gpu_blocks(), n_gpu_blocks);
    EXPECT_EQ(block_manager.n_free_cpu_blocks(), n_cpu_blocks);
}

// Testing that we always return to full (stress test)
TEST(test_block_manager_leak_repeated) {
    const uint32_t block_size   = 16;
    const uint32_t n_gpu_blocks = 64;
    const uint32_t n_cpu_blocks = 32;
    const float    watermark    = 0.0f;
    const int      n_iter       = 1000;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    for (int iter = 0; iter < n_iter; ++iter) {
        const uint32_t n   = (iter % block_size) + 1;
        auto           ids = block_manager.checkout_gpu_blocks(n);
        EXPECT_EQ(ids.size(), (size_t) n);
        block_manager.release_gpu_blocks(ids);
        EXPECT_EQ(block_manager.n_free_gpu_blocks(), n_gpu_blocks);
    }
}

// Attempting to check-out more blocks than available. It should return empty.
TEST(test_block_manager_checkout_too_many) {
    const uint32_t n_gpu_blocks = 8;
    const uint32_t n_cpu_blocks = 4;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    auto ids = block_manager.checkout_gpu_blocks(9);
    EXPECT_EQ(ids.size(), 0u);
    EXPECT_EQ(block_manager.n_free_gpu_blocks(), n_gpu_blocks);

    auto ids2 = block_manager.checkout_cpu_blocks(5);
    EXPECT_EQ(ids2.size(), 0u);
    EXPECT_EQ(block_manager.n_free_cpu_blocks(), n_cpu_blocks);
}

TEST(test_block_manager_watermark) {
    // watermark=0.2 means 2 blocks reserved as safety (always consider max gpu blocks)
    const uint32_t n_gpu_blocks = 10;
    const uint32_t n_cpu_blocks = 10;
    const float    watermark    = 0.2f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    // 10 free, safety=2
    EXPECT_TRUE(block_manager.has_free_gpu_blocks(8));
    EXPECT_FALSE(block_manager.has_free_gpu_blocks(9));
    EXPECT_FALSE(block_manager.has_free_gpu_blocks(10));

    auto ids = block_manager.checkout_gpu_blocks(5);
    EXPECT_EQ(ids.size(), 5u);
    // 5 free, safety=2
    EXPECT_TRUE(block_manager.has_free_gpu_blocks(3));
    EXPECT_FALSE(block_manager.has_free_gpu_blocks(4));

    block_manager.release_gpu_blocks(ids);
    EXPECT_TRUE(block_manager.has_free_gpu_blocks(8));
}

TEST(test_block_manager_watermark_zero) {
    // watermark=0 means the entire pool is requestable.
    const uint32_t n_gpu_blocks = 10;
    const uint32_t n_cpu_blocks = 10;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);
    EXPECT_TRUE(block_manager.has_free_gpu_blocks(10));
    EXPECT_FALSE(block_manager.has_free_gpu_blocks(11));
}

TEST(test_block_manager_gpu_cpu_disjoint) {
    // Just making sure CPU and GPU blocks are disjoint
    const uint32_t n_gpu_blocks = 8;
    const uint32_t n_cpu_blocks = 4;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    auto gpu_ids = block_manager.checkout_gpu_blocks(8);
    auto cpu_ids = block_manager.checkout_cpu_blocks(4);

    for (auto id : gpu_ids) {
        EXPECT_TRUE(block_manager.is_gpu(id));
    }
    for (auto id : cpu_ids) {
        EXPECT_FALSE(block_manager.is_gpu(id));
    }

    block_manager.release_gpu_blocks(gpu_ids);
    block_manager.release_cpu_blocks(cpu_ids);
}

// Testing llama_kv_cache_paged book-keeping

// kv_cache_paged with arbitrary shape (we only care about the bookkeeping)
static llama_kv_cache_paged make_kv() {
    return llama_kv_cache_paged(
        /*head_dim=*/64,
        /*n_heads_kv=*/4,
        /*block_size=*/16,
        /*n_layers=*/2,
        /*n_ubatch=*/32,
        /*n_seq_max=*/8);
}

TEST(test_seq_pos_default_unknown) {
    auto kv = make_kv();
    EXPECT_EQ(kv.seq_pos_min(0), -1);
    EXPECT_EQ(kv.seq_pos_max(0), -1);
    EXPECT_EQ(kv.seq_pos_min(42), -1);
}

TEST(test_seq_pos_set_and_get) {
    auto kv = make_kv();
    kv.set_seq_min_pos(0, 5);
    kv.set_seq_max_pos(0, 17);
    EXPECT_EQ(kv.seq_pos_min(0), 5);
    EXPECT_EQ(kv.seq_pos_max(0), 17);

    // Independent per seq_id.
    kv.set_seq_min_pos(1, 100);
    EXPECT_EQ(kv.seq_pos_min(1), 100);
    EXPECT_EQ(kv.seq_pos_min(0), 5);  // unchanged
}

TEST(test_seq_pos_overwrite) {
    auto kv = make_kv();
    kv.set_seq_min_pos(0, 5);
    kv.set_seq_min_pos(0, 9);
    EXPECT_EQ(kv.seq_pos_min(0), 9);
}

TEST(test_seq_pos_seq_rm_removes) {
    auto kv = make_kv();
    kv.set_seq_min_pos(0, 5);
    kv.set_seq_max_pos(0, 17);
    EXPECT_EQ(kv.seq_pos_min(0), 5);

    EXPECT_FALSE(kv.seq_rm(0, 0, 0));
    EXPECT_EQ(kv.seq_pos_min(0), 5);
    EXPECT_TRUE(kv.seq_rm(0, -1, -1));
    EXPECT_EQ(kv.seq_pos_min(0), -1);
    EXPECT_EQ(kv.seq_pos_max(0), -1);
}

TEST(test_seq_pos_clear_removes_all) {
    auto kv = make_kv();
    kv.set_seq_min_pos(0, 5);
    kv.set_seq_min_pos(1, 7);
    kv.set_seq_min_pos(2, 9);

    kv.clear(/*data=*/false);
    EXPECT_EQ(kv.seq_pos_min(0), -1);
    EXPECT_EQ(kv.seq_pos_min(1), -1);
    EXPECT_EQ(kv.seq_pos_min(2), -1);
}

TEST(test_free_blocks_releases_to_pool) {
    // Initialize a KV cache with a real CPU backend (used as both "GPU" and CPU).
    // This is fine for testing
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(backend != nullptr);

    const uint32_t n_gpu_blocks = 16;
    const uint32_t n_cpu_blocks = 8;
    const float    watermark    = 0.0f;

    auto kv = make_kv();
    kv.init(/*backend_gpu=*/backend,
            /*backend_cpu=*/backend, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, n_gpu_blocks, n_cpu_blocks, watermark);

    // allocate() pulls from the GPU block pool. After releasing, the count
    // must return to the initial value.
    const uint32_t n_gpu_initial = kv.get_num_gpu_blocks();
    EXPECT_EQ(n_gpu_initial, n_gpu_blocks);

    llama_sequence_group group;
    group.request_id = 0;
    group.n_prompt   = 32;  // 2 blocks
    group.n_decoded  = 0;

    bool success = kv.allocate(/*num_tokens=*/0, group);
    EXPECT_TRUE(success);
    EXPECT_EQ(group.block_table.size(), 2u);

    // Allocating a second sequence further reduces the pool.
    llama_sequence_group group2;
    group2.request_id = 1;
    group2.n_prompt   = 48;  // 3 blocks
    group2.n_decoded  = 0;

    success = kv.allocate(0, group2);
    EXPECT_TRUE(success);
    EXPECT_EQ(group2.block_table.size(), 3u);

    // Free the first sequence: we release 2 blocks
    kv.free_blocks(group);
    EXPECT_EQ(group.block_table.size(), 0u);

    llama_sequence_group group3;
    group3.request_id = 2;
    group3.n_prompt   = 32;  // 2 blocks
    group3.n_decoded  = 0;
    success           = kv.allocate(0, group3);
    EXPECT_TRUE(success);
    EXPECT_EQ(group3.block_table.size(), 2u);

    // All blocks released
    kv.free_blocks(group2);
    kv.free_blocks(group3);

    llama_sequence_group group_full;
    group_full.request_id = 99;
    group_full.n_prompt   = n_gpu_blocks * 16;  // exactly n_gpu_blocks worth
    group_full.n_decoded  = 0;
    success               = kv.allocate(0, group_full);
    EXPECT_TRUE(success);
    EXPECT_EQ(group_full.block_table.size(), 16u);
    kv.free_blocks(group_full);

    ggml_backend_free(backend);
}
TEST(test_release_seq_tail_releases_trailing_blocks) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(backend != nullptr);

    auto kv = make_kv();
    kv.init(backend, backend, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 4, 1, 0.0f);
    llama_sequence_group group;
    group.request_id = 7;
    group.n_prompt = 48;
    EXPECT_TRUE(kv.allocate(0, group));
    EXPECT_EQ(group.block_table.size(), 3u);
    EXPECT_TRUE(kv.release_seq_tail(group.request_id, 17));
    EXPECT_EQ(group.block_table.size(), 2u);
    EXPECT_TRUE(kv.release_seq_tail(group.request_id, 32));
    EXPECT_EQ(group.block_table.size(), 2u);
    kv.free_blocks(group);
    ggml_backend_free(backend);
}


TEST(test_clear_and_seq_rm_release_blocks) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(backend != nullptr);

    auto kv = make_kv();
    kv.init(backend, backend, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 2, 1, 0.0f);
    llama_sequence_group first;
    first.request_id = 1;
    first.n_prompt = 16;
    EXPECT_TRUE(kv.allocate(0, first));
    EXPECT_TRUE(kv.seq_rm(first.request_id, -1, -1));
    EXPECT_TRUE(first.block_table.empty());

    llama_sequence_group second;
    second.request_id = 2;
    second.n_prompt = 32;
    EXPECT_TRUE(kv.allocate(0, second));
    kv.clear(false);
    EXPECT_TRUE(second.block_table.empty());

    llama_sequence_group third;
    third.request_id = 3;
    third.n_prompt = 32;
    EXPECT_TRUE(kv.allocate(0, third));
    ggml_backend_free(backend);
}

TEST(test_paged_storage_types_are_native) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(backend != nullptr);

    for (const ggml_type type : { GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0 }) {
        llama_kv_cache_paged kv(/*head_dim=*/128, /*n_heads_kv=*/4, /*block_size=*/16,
                                /*n_layers=*/2, /*n_ubatch=*/32, /*n_seq_max=*/8);
        kv.init(backend, backend, type, type, /*n_gpu_blocks=*/2, /*n_cpu_blocks=*/1, /*watermark=*/0.0f);
        const ggml_type expected = type == GGML_TYPE_F16 ? GGML_TYPE_Q8_0 : type;
        EXPECT_EQ(kv.get_k_tensor(0)->type, expected);
        EXPECT_EQ(kv.get_v_tensor(0)->type, expected);
    }

    ggml_backend_free(backend);
}

TEST(test_paged_state_round_trip) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(backend != nullptr);

    auto kv = make_kv();
    kv.init(backend, backend, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 4, 2, 0.0f);
    llama_sequence_group group;
    group.request_id = 3;
    group.n_prompt   = 32;
    EXPECT_TRUE(kv.allocate(0, group));
    const llama_block_ids block_ids = group.block_table;
    kv.set_seq_min_pos(group.request_id, 0);
    kv.set_seq_max_pos(group.request_id, 31);

    auto * k = kv.get_k_tensor(0);
    std::vector<uint8_t> expected(ggml_nbytes(k), 0x5a);
    std::vector<uint8_t> cleared(expected.size());
    ggml_backend_tensor_set(k, expected.data(), 0, expected.size());

    memory_io io;
    kv.state_write(io);
    EXPECT_TRUE(!io.data.empty());
    ggml_backend_tensor_set(k, cleared.data(), 0, cleared.size());
    kv.clear(false);
    io.offset = 0;
    kv.state_read(io);

    const size_t block_bytes = ggml_nbytes(k) / 4;
    std::vector<uint8_t> restored(block_bytes);
    for (uint32_t block_id : block_ids) {
        ggml_backend_tensor_get(k, restored.data(), block_id * block_bytes, block_bytes);
        EXPECT_TRUE(restored == std::vector<uint8_t>(block_bytes, 0x5a));
    }
    EXPECT_EQ(kv.seq_pos_min(group.request_id), 0);
    EXPECT_EQ(kv.seq_pos_max(group.request_id), 31);
    EXPECT_EQ(io.offset, io.data.size());
    ggml_backend_free(backend);
}

TEST(test_paged_state_read_failure_preserves_live_cache) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(backend != nullptr);

    auto kv = make_kv();
    kv.init(backend, backend, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 4, 2, 0.0f);
    llama_sequence_group group;
    group.request_id = 3;
    group.n_prompt = 16;
    EXPECT_TRUE(kv.allocate(0, group));
    const llama_block_ids expected = group.block_table;
    kv.set_seq_min_pos(group.request_id, 0);
    kv.set_seq_max_pos(group.request_id, 15);

    memory_io io;
    kv.state_write(io);
    io.data.pop_back();
    io.offset = 0;
    bool failed = false;
    try {
        kv.state_read(io);
    } catch (const std::runtime_error &) {
        failed = true;
    }
    EXPECT_TRUE(failed);
    EXPECT_TRUE(group.block_table == expected);
    EXPECT_EQ(kv.seq_pos_min(group.request_id), 0);
    EXPECT_EQ(kv.seq_pos_max(group.request_id), 15);
    ggml_backend_free(backend);
}

TEST(test_paged_sequence_state_read_failure_preserves_live_cache) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(backend != nullptr);

    auto kv = make_kv();
    kv.init(backend, backend, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 4, 2, 0.0f);
    llama_sequence_group group;
    group.request_id = 3;
    group.n_prompt = 16;
    EXPECT_TRUE(kv.allocate(0, group));
    const llama_block_ids expected = group.block_table;
    kv.set_seq_min_pos(group.request_id, 0);
    kv.set_seq_max_pos(group.request_id, 15);

    memory_io io;
    kv.state_write(io, group.request_id);
    io.data.pop_back();
    io.offset = 0;
    bool failed = false;
    try {
        kv.state_read(io, group.request_id);
    } catch (const std::runtime_error &) {
        failed = true;
    }
    EXPECT_TRUE(failed);
    EXPECT_TRUE(group.block_table == expected);
    EXPECT_EQ(kv.seq_pos_min(group.request_id), 0);
    EXPECT_EQ(kv.seq_pos_max(group.request_id), 15);
    ggml_backend_free(backend);
}

TEST(test_paged_sequence_state_preserves_other_sequences) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(backend != nullptr);

    auto kv = make_kv();
    kv.init(backend, backend, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 4, 2, 0.0f);
    llama_sequence_group first;
    first.request_id = 1;
    first.n_prompt = 16;
    llama_sequence_group second;
    second.request_id = 2;
    second.n_prompt = 16;
    EXPECT_TRUE(kv.allocate(0, first));
    EXPECT_TRUE(kv.allocate(0, second));
    kv.set_seq_min_pos(first.request_id, 0);
    kv.set_seq_max_pos(first.request_id, 15);
    kv.set_seq_min_pos(second.request_id, 0);
    kv.set_seq_max_pos(second.request_id, 15);

    auto * k = kv.get_k_tensor(0);
    const size_t block_bytes = ggml_nbytes(k) / 4;
    std::vector<uint8_t> zeros(ggml_nbytes(k));
    std::vector<uint8_t> zero_block(block_bytes);
    std::vector<uint8_t> first_data(block_bytes, 0x5a);
    std::vector<uint8_t> second_data(block_bytes, 0x7b);
    ggml_backend_tensor_set(k, zeros.data(), 0, zeros.size());
    ggml_backend_tensor_set(k, first_data.data(), first.block_table[0] * block_bytes, block_bytes);
    ggml_backend_tensor_set(k, second_data.data(), second.block_table[0] * block_bytes, block_bytes);

    memory_io io;
    kv.state_write(io, first.request_id);
    ggml_backend_tensor_set(k, zeros.data(), 0, zeros.size());
    io.offset = 0;
    kv.state_read(io, first.request_id);

    std::vector<uint8_t> restored(block_bytes);
    ggml_backend_tensor_get(k, restored.data(), first.block_table[0] * block_bytes, block_bytes);
    EXPECT_TRUE(restored == first_data);
    ggml_backend_tensor_get(k, restored.data(), second.block_table[0] * block_bytes, block_bytes);
    EXPECT_TRUE(restored == zero_block);

    llama_sequence_group third;
    third.request_id = 3;
    third.n_prompt = 32;
    EXPECT_TRUE(kv.allocate(0, third));
    EXPECT_EQ(third.block_table.size(), 2u);
    ggml_backend_free(backend);
}

TEST(test_paged_state_round_trip_after_swap) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(backend != nullptr);

    auto kv = make_kv();
    kv.init(backend, backend, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 4, 2, 0.0f);
    llama_sequence_group group;
    group.request_id = 3;
    group.n_prompt = 16;
    EXPECT_TRUE(kv.allocate(0, group));
    kv.set_seq_min_pos(group.request_id, 0);
    kv.set_seq_max_pos(group.request_id, 15);
    EXPECT_TRUE(kv.swap_out(group));

    memory_io io;
    kv.state_write(io, group.request_id);
    group.block_table.clear();
    io.offset = 0;
    kv.state_read(io, group.request_id);
    EXPECT_EQ(group.block_table.size(), 1u);
    EXPECT_EQ(kv.seq_pos_min(group.request_id), 0);
    EXPECT_EQ(kv.seq_pos_max(group.request_id), 15);
    EXPECT_TRUE(kv.swap_in(group));
    ggml_backend_free(backend);
}

// Testing scheduler

// For easy testing and clean-up.
// KV cache paged + scheduler (must free CPU backend)
struct paged_test_fixture {
    ggml_backend_t                              backend = nullptr;
    std::unique_ptr<llama_kv_cache_paged>       kv;
    std::unique_ptr<llama_paged_scheduler_impl> sched;

    ~paged_test_fixture() {
        if (backend) {
            ggml_backend_free(backend);
        }
    }

    // rule of 5
    paged_test_fixture()                                       = default;
    paged_test_fixture(paged_test_fixture &&)                  = default;
    paged_test_fixture & operator=(paged_test_fixture &&)      = default;
    paged_test_fixture(const paged_test_fixture &)             = delete;
    paged_test_fixture & operator=(const paged_test_fixture &) = delete;
};

static paged_test_fixture make_fixture(uint32_t n_ctx        = 128,
                                       uint32_t block_size   = 16,
                                       uint32_t n_batch      = 64,
                                       uint32_t n_gpu_blocks = 4,
                                       uint32_t n_cpu_blocks = 2) {
    paged_test_fixture fixture;
    fixture.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(fixture.backend != nullptr);

    fixture.kv = std::unique_ptr<llama_kv_cache_paged>(new llama_kv_cache_paged(
        /*head_dim=*/64u,
        /*n_heads_kv=*/4u,
        /*block_size=*/block_size,
        /*n_layers=*/2u,
        /*n_ubatch=*/n_batch,
        /*n_seq_max=*/8u));
    fixture.kv->init(fixture.backend, fixture.backend, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, n_gpu_blocks, n_cpu_blocks, /*watermark=*/0.0f);

    fixture.sched = std::unique_ptr<llama_paged_scheduler_impl>(
        new llama_paged_scheduler_impl(n_ctx, block_size, n_batch, fixture.kv.get()));
    return fixture;
}

static llama_sequence_group make_group(int32_t request_id, uint32_t n_prompt) {
    llama_sequence_group group;
    group.request_id     = request_id;
    group.n_prompt       = n_prompt;
    group.n_decoded      = 0;
    group.n_past         = 0;
    group.t_arrival_time = request_id;  // control ordering based on request_id
    group.logical_seq.assign(n_prompt, /*dummy token=*/1);
    return group;
}

TEST(test_scheduler_state_restores_block_ownership) {
    auto fixture = make_fixture();
    EXPECT_TRUE(fixture.sched->queue_request(make_group(3, 16)));

    llama_batch batch = {};
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    auto * group = fixture.sched->get_group_from_id(3);
    EXPECT_TRUE(group != nullptr);
    const llama_block_ids expected = group->block_table;
    fixture.kv->set_seq_min_pos(3, 0);
    fixture.kv->set_seq_max_pos(3, 15);

    memory_io io;
    fixture.kv->state_write(io);
    group->block_table.clear();
    memory_io seq_io;
    fixture.kv->state_write(seq_io, 3);
    group->block_table.clear();
    seq_io.offset = 0;
    fixture.kv->state_read(seq_io, 3);
    EXPECT_TRUE(group->block_table == expected);
    io.offset = 0;
    fixture.kv->state_read(io);
    EXPECT_TRUE(group->block_table == expected);

    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    llama_batch_free(batch);
}

TEST(test_scheduler_resumes_fresh_checkpoint) {
    auto source = make_fixture();

    EXPECT_TRUE(source.sched->queue_request(make_group(3, 16)));

    llama_batch source_batch = {};
    EXPECT_TRUE(source.sched->step(source_batch) == llama_scheduler_status::OK);
    const int8_t continue_flag[] = { 0 };
    source.sched->update(source_batch, { 42 }, continue_flag);
    auto * source_group = source.sched->get_group_from_id(3);
    EXPECT_TRUE(source_group != nullptr);
    EXPECT_EQ(source_group->n_past, 16u);
    EXPECT_EQ(source_group->n_decoded, 16u);

    memory_io io;
    source.kv->state_write(io);

    auto restored = make_fixture();
    io.offset = 0;
    restored.kv->state_read(io, 4);
    EXPECT_TRUE(restored.sched->queue_request(make_group(4, 16)));
    auto * restored_group = restored.sched->get_group_from_id(4);
    EXPECT_TRUE(restored_group != nullptr);
    EXPECT_EQ(restored_group->n_past, 16u);
    EXPECT_EQ(restored_group->n_decoded, 16u);
    EXPECT_EQ(restored_group->logical_seq.back(), 42);

    llama_batch restored_batch = {};
    EXPECT_TRUE(restored.sched->step(restored_batch) == llama_scheduler_status::OK);
    EXPECT_EQ(restored_batch.n_tokens, 1);
    EXPECT_EQ(restored_batch.token[0], 42);
    EXPECT_EQ(restored_batch.pos[0], 16);
    restored.sched->update(restored_batch, { 43 }, continue_flag);
    EXPECT_EQ(restored_group->n_past, 17u);
    EXPECT_EQ(restored_group->n_decoded, 17u);
    EXPECT_EQ(restored_group->logical_seq.back(), 43);
    llama_batch_free(source_batch);
    llama_batch_free(restored_batch);
}

TEST(test_scheduler_teardown_unregisters_groups) {
    auto fixture = make_fixture();
    EXPECT_TRUE(fixture.sched->queue_request(make_group(3, 16)));
    fixture.sched.reset();
    fixture.sched = std::unique_ptr<llama_paged_scheduler_impl>(
        new llama_paged_scheduler_impl(128, 16, 64, fixture.kv.get()));
    EXPECT_TRUE(fixture.sched->queue_request(make_group(3, 16)));
}

TEST(test_scheduler_no_deadlock_on_empty) {
    // No requests queued means no deadlock
    auto                   fixture = make_fixture();
    llama_batch            batch   = {};
    llama_scheduler_status status  = fixture.sched->step(batch);
    EXPECT_TRUE(status == llama_scheduler_status::OK);
    EXPECT_EQ(batch.n_tokens, 0);
}

TEST(test_scheduler_deadlock_oversize_waiting_request) {
    // There are 2 blocks, the waiting request needs 3.
    // Attempt to process prefill for this requets will fail and the request will remain in waiting (deadlock).
    auto fixture = make_fixture(128, 16, 64, /*n_gpu_blocks=*/2, /*n_cpu_blocks=*/1);
    auto group   = make_group(0, /*n_prompts=*/32);

    bool queued = fixture.sched->queue_request(group);
    EXPECT_TRUE(queued);

    llama_batch            batch  = {};
    llama_scheduler_status status = fixture.sched->step(batch);
    EXPECT_TRUE(status == llama_scheduler_status::DEADLOCK);

    // Next steps will continue remained deadlocked
    status = fixture.sched->step(batch);
    EXPECT_TRUE(status == llama_scheduler_status::DEADLOCK);
}

TEST(test_scheduler_prioritizes_oldest_request_under_pressure) {
    auto fixture = make_fixture(/*n_ctx=*/128, /*block_size=*/16, /*n_batch=*/200,
                                /*n_gpu_blocks=*/13, /*n_cpu_blocks=*/10);
    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/79)));
    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/1, /*n_prompt=*/62)));
    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/2, /*n_prompt=*/47)));

    llama_batch batch = {};
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    EXPECT_TRUE(batch.n_tokens == 188);
    const int8_t continue_flags[] = { 0, 0, 0 };
    fixture.sched->update(batch, { 10, 11, 12 }, continue_flags);

    for (int i = 0; i < 16; ++i) {
        EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
        fixture.sched->update(batch, { 13 + i, 14 + i, 15 + i }, continue_flags);
    }

    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    const auto * info = fixture.sched->get_curr_batch_info();
    EXPECT_TRUE(info->n_seq == 1);
    EXPECT_TRUE(batch.seq_id[0][0] == 0);
    EXPECT_TRUE(fixture.sched->get_group_from_id(1)->block_table.front() >= 13);
    EXPECT_TRUE(fixture.sched->get_group_from_id(2)->block_table.front() >= 13);

    const int8_t continue_flag[] = { 0 };
    fixture.sched->update(batch, { 30 }, continue_flag);
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    EXPECT_TRUE(batch.seq_id[0][0] == 0);
    const int8_t stop_flag[] = { 1 };
    fixture.sched->update(batch, { 31 }, stop_flag);
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    EXPECT_TRUE(fixture.sched->get_curr_batch_info()->n_seq == 2);
    EXPECT_TRUE(batch.seq_id[0][0] == 1);
    EXPECT_TRUE(batch.seq_id[1][0] == 2);
    llama_batch_free(batch);
}

TEST(test_scheduler_admits_full_token_budget_prefill) {
    auto fixture = make_fixture(/*n_ctx=*/128, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/5, /*n_cpu_blocks=*/1);
    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/64)));

    llama_batch batch = {};
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    EXPECT_TRUE(batch.n_tokens == 64);
    const auto * info = fixture.sched->get_curr_batch_info();
    EXPECT_TRUE(info->n_seq == 1);
    EXPECT_TRUE(info->batch_lens[0] == 64);
    EXPECT_TRUE(info->n_blocks_per_seq == 5);
    EXPECT_TRUE(info->write_slots[15] == info->write_slots[0] + 15);
    EXPECT_TRUE(info->write_slots[16] == info->write_slots[0] + 16);

    const int8_t continue_flag[] = { 0 };
    fixture.sched->update(batch, { 1 }, continue_flag);
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    EXPECT_TRUE(batch.n_tokens == 1);
    EXPECT_TRUE(batch.pos[0] == 64);
    llama_batch_free(batch);
}

TEST(test_scheduler_batches_two_cross_block_prefills) {
    auto fixture = make_fixture(/*n_ctx=*/128, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/4, /*n_cpu_blocks=*/1);
    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/20)));
    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/1, /*n_prompt=*/24)));

    llama_batch batch = {};
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    const auto * info = fixture.sched->get_curr_batch_info();
    EXPECT_TRUE(info->n_seq == 2);
    EXPECT_TRUE(batch.n_tokens == 44);
    EXPECT_TRUE(info->batch_offsets[0] == 0);
    EXPECT_TRUE(info->batch_lens[0] == 20);
    EXPECT_TRUE(info->batch_offsets[1] == 20);
    EXPECT_TRUE(info->batch_lens[1] == 24);
    EXPECT_TRUE(info->n_blocks_per_seq == 2);
    EXPECT_TRUE(info->write_slots[15] == info->write_slots[0] + 15);
    EXPECT_TRUE(info->write_slots[16] == info->write_slots[0] + 16);
    EXPECT_TRUE(batch.pos[43] == 23);
    llama_batch_free(batch);
}

TEST(test_paged_attention_head_mapping_and_dispatch_selection) {
    for (int q_head = 0; q_head < 8; ++q_head) {
        EXPECT_TRUE(ggml_paged_attn_kv_head(q_head, 8, 1) == 0);
    }
    for (int q_head = 0; q_head < 8; ++q_head) {
        EXPECT_TRUE(ggml_paged_attn_kv_head(q_head, 8, 2) == q_head / 4);
    }
    for (int q_head = 0; q_head < 24; ++q_head) {
        EXPECT_TRUE(ggml_paged_attn_kv_head(q_head, 24, 4) == q_head / 6);
    }

    EXPECT_TRUE(ggml_paged_attn_tiled_prefill_supported(
        128, GGML_TYPE_F32, true, true, 32, 1, 2, true, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    EXPECT_TRUE(ggml_paged_attn_tiled_prefill_supported(
        256, GGML_TYPE_F32, true, true, 32, 1, 2, true, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    EXPECT_FALSE(ggml_paged_attn_tiled_prefill_supported(
        64, GGML_TYPE_F32, true, true, 32, 1, 2, true, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    EXPECT_FALSE(ggml_paged_attn_tiled_prefill_supported(
        192, GGML_TYPE_F32, true, true, 32, 1, 2, true, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    EXPECT_FALSE(ggml_paged_attn_tiled_prefill_supported(
        256, GGML_TYPE_F32, true, true, 32, 1, 2, true, GGML_TYPE_F16, GGML_TYPE_Q8_0));
    EXPECT_FALSE(ggml_paged_attn_tiled_prefill_supported(
        256, GGML_TYPE_F32, true, true, 2, 2, 1, true, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));

    EXPECT_TRUE(ggml_paged_attn_context_bucket(4096) == GGML_PAGED_ATTN_CONTEXT_4K);
    EXPECT_TRUE(ggml_paged_attn_context_bucket(4097) == GGML_PAGED_ATTN_CONTEXT_16K);
    EXPECT_TRUE(ggml_paged_attn_context_bucket(65536) == GGML_PAGED_ATTN_CONTEXT_64K);
    EXPECT_TRUE(ggml_paged_attn_context_bucket(65537) == GGML_PAGED_ATTN_CONTEXT_128K);
    for (int n_partitions : { 1, 2, 4, 8 }) {
        for (int context_len : { 1, 15, 16, 17, 31, 32, 33, 127 }) {
            const int tail = ggml_paged_attn_tail_partition(context_len, n_partitions);
            EXPECT_TRUE(tail >= 0 && tail < n_partitions);
            int owners = 0;
            for (int partition = 0; partition < n_partitions; ++partition) {
                owners += partition == tail;
            }
            EXPECT_TRUE(owners == 1);
        }
    }
    for (const auto [n_heads, n_heads_kv] : { std::pair<int, int>{ 24, 4 }, { 32, 4 } }) {
        for (int n_partitions : { 1, 2, 4, 8 }) {
            for (int n_q_heads : { 1, 2, 4 }) {
                if ((n_heads / n_heads_kv) % n_q_heads != 0) {
                    continue;
                }
                std::vector<int> owners(n_heads_kv, 0);
                for (int partition = 0; partition < n_partitions; ++partition) {
                    for (int q_head = 0; q_head < n_heads; q_head += n_q_heads) {
                        if (ggml_paged_attn_current_row_owner(
                                q_head, n_heads, n_heads_kv, partition, n_partitions, 65)) {
                            ++owners[ggml_paged_attn_kv_head(q_head, n_heads, n_heads_kv)];
                        }
                    }
                }
                for (int count : owners) {
                    EXPECT_TRUE(count == 1);
                }
                const int tail = ggml_paged_attn_tail_partition(65, n_partitions);
                for (int partition = 0; partition < n_partitions; ++partition) {
                    if (partition != tail) {
                        EXPECT_FALSE(ggml_paged_attn_current_row_owner(
                            0, n_heads, n_heads_kv, partition, n_partitions, 65));
                    }
                }
            }
        }
    }

    const ggml_paged_attn_cuda_device_caps ada = { 890, 128, 32, 1024, 48 * 1024, true };
    const ggml_paged_attn_cuda_variant shallow = ggml_paged_attn_select_cuda_variant(
        GGML_PAGED_ATTN_CONTEXT_4K, 32, 4, 1, ada);
    EXPECT_TRUE(shallow.n_warps == 32 && shallow.n_partitions == 1 && shallow.n_q_heads == 1);
    const ggml_paged_attn_cuda_variant long_single = ggml_paged_attn_select_cuda_variant(
        GGML_PAGED_ATTN_CONTEXT_64K, 32, 4, 1, ada);
    EXPECT_TRUE(long_single.n_warps == 8 && long_single.n_partitions == 8 && long_single.n_q_heads == 1);
    const ggml_paged_attn_cuda_variant long_batch = ggml_paged_attn_select_cuda_variant(
        GGML_PAGED_ATTN_CONTEXT_64K, 32, 4, 2, ada);
    EXPECT_TRUE(long_batch.n_warps == 8 && long_batch.n_partitions == 8 && long_batch.n_q_heads == 1);
    const ggml_paged_attn_cuda_variant long_batch4 = ggml_paged_attn_select_cuda_variant(
        GGML_PAGED_ATTN_CONTEXT_16K, 32, 4, 4, ada);
    EXPECT_TRUE(long_batch4.n_warps == 32 && long_batch4.n_partitions == 1 && long_batch4.n_q_heads == 1);
    const ggml_paged_attn_cuda_variant long_batch8 = ggml_paged_attn_select_cuda_variant(
        GGML_PAGED_ATTN_CONTEXT_16K, 32, 4, 8, ada);
    EXPECT_TRUE(long_batch8.n_warps == 32 && long_batch8.n_partitions == 1 && long_batch8.n_q_heads == 1);
    const ggml_paged_attn_cuda_variant long_batch6 = ggml_paged_attn_select_cuda_variant(
        GGML_PAGED_ATTN_CONTEXT_16K, 24, 4, 4, ada);
    EXPECT_TRUE(long_batch6.n_warps == 8 && long_batch6.n_partitions == 8 && long_batch6.n_q_heads == 2);

    for (const ggml_paged_attn_cuda_device_caps unsupported : {
            ggml_paged_attn_cuda_device_caps{ 860, 128, 32, 1024, 48 * 1024, true },
            ggml_paged_attn_cuda_device_caps{ 890, 128, 64, 1024, 48 * 1024, true },
            ggml_paged_attn_cuda_device_caps{ 890, 128, 32, 512, 48 * 1024, true },
            ggml_paged_attn_cuda_device_caps{ 890, 128, 32, 1024, 32 * 1024, true },
            ggml_paged_attn_cuda_device_caps{ 890, 128, 32, 1024, 48 * 1024, false },
        }) {
        const ggml_paged_attn_cuda_variant fallback = ggml_paged_attn_select_cuda_variant(
            GGML_PAGED_ATTN_CONTEXT_64K, 32, 4, 1, unsupported);
        EXPECT_TRUE(fallback.n_warps == 32 && fallback.n_partitions == 1 && fallback.n_q_heads == 1);
    }
}

TEST(test_decode_only_batch_skips_tiled_prefill) {
    EXPECT_FALSE(ggml_paged_attn_tiled_prefill_supported(
        128, GGML_TYPE_F32, true, true, 2, 2, 1, true, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    EXPECT_FALSE(ggml_paged_attn_tiled_prefill_supported(
        256, GGML_TYPE_F32, true, true, 2, 2, 1, true, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
}

#if defined(GGML_USE_CUDA)
template <int tokens_per_sequence>
static std::vector<float> run_paged_attention_production_case(
        ggml_backend_t backend, int head_dim, int n_heads, int n_heads_kv) {
    constexpr int block_size = 16;
    constexpr int max_blocks = 2;
    constexpr int n_sequences = 2;
    constexpr int context_length = 24;
    constexpr int n_cache_blocks = n_sequences * max_blocks;
    constexpr int n_tokens = n_sequences * tokens_per_sequence;

    const size_t graph_size = 16;
    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    EXPECT_TRUE(ctx != nullptr);

    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, n_tokens);
    ggml_tensor * k_new = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads_kv, n_tokens);
    ggml_tensor * v_new = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads_kv, n_tokens);
    ggml_tensor * k_cache = ggml_new_tensor_4d(
        ctx, GGML_TYPE_Q8_0, head_dim, block_size, n_heads_kv, n_cache_blocks);
    ggml_tensor * v_cache = ggml_new_tensor_4d(
        ctx, GGML_TYPE_Q8_0, head_dim, block_size, n_heads_kv, n_cache_blocks);
    ggml_tensor * block_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, max_blocks, n_sequences);
    ggml_tensor * write_slots = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_tensor * context_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_sequences);
    ggml_tensor * batch_offsets = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_sequences);
    ggml_tensor * batch_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_sequences);
    ggml_tensor * out = ggml_paged_attn(
        ctx, q, k_new, v_new, k_cache, v_cache, block_table, write_slots,
        context_lens, batch_offsets, batch_lens, 1.0f / std::sqrt((float) head_dim), block_size, max_blocks,
        ggml_paged_attn_context_bucket(context_length));

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    EXPECT_TRUE(buffer != nullptr);

    std::vector<float> q_data(ggml_nelements(q));
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = 0.025f * std::sin((float) (i % 97)) + 0.01f * (float) ((i / head_dim) % n_heads);
    }

    const size_t n_cache_rows = (size_t) n_cache_blocks * n_heads_kv * block_size;
    std::vector<float> k_data(n_cache_rows * head_dim);
    std::vector<float> v_data(n_cache_rows * head_dim);
    for (size_t row = 0; row < n_cache_rows; ++row) {
        const int kv_head = (row / block_size) % n_heads_kv;
        const int physical_block = row / (block_size * n_heads_kv);
        const int token = row % block_size;
        for (int dim = 0; dim < head_dim; ++dim) {
            const size_t index = row * head_dim + dim;
            k_data[index] = 0.015f * (float) ((dim % 11) - 5) + 0.09f * kv_head +
                0.025f * physical_block + 0.004f * token;
            v_data[index] = 0.02f * (float) ((dim % 13) - 6) + 0.13f * kv_head +
                0.04f * physical_block + 0.006f * token;
        }
    }
    std::vector<uint8_t> k_quantized(ggml_nbytes(k_cache));
    std::vector<uint8_t> v_quantized(ggml_nbytes(v_cache));
    EXPECT_TRUE(ggml_quantize_chunk(
        GGML_TYPE_Q8_0, k_data.data(), k_quantized.data(), 0, n_cache_rows, head_dim, nullptr) == k_quantized.size());
    EXPECT_TRUE(ggml_quantize_chunk(
        GGML_TYPE_Q8_0, v_data.data(), v_quantized.data(), 0, n_cache_rows, head_dim, nullptr) == v_quantized.size());

    const std::vector<float> new_data((size_t) n_tokens * n_heads_kv * head_dim, 0.0f);
    const int32_t block_table_data[] = { 1, 0, 3, 2 };
    const std::vector<int32_t> write_slots_data(n_tokens, 0);
    const int32_t context_lens_data[] = { context_length, context_length };
    const int32_t batch_offsets_data[] = { 0, tokens_per_sequence };
    const int32_t batch_lens_data[] = { tokens_per_sequence, tokens_per_sequence };

    ggml_backend_tensor_set(q, q_data.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k_new, new_data.data(), 0, ggml_nbytes(k_new));
    ggml_backend_tensor_set(v_new, new_data.data(), 0, ggml_nbytes(v_new));
    ggml_backend_tensor_set(k_cache, k_quantized.data(), 0, k_quantized.size());
    ggml_backend_tensor_set(v_cache, v_quantized.data(), 0, v_quantized.size());
    ggml_backend_tensor_set(block_table, block_table_data, 0, sizeof(block_table_data));
    ggml_backend_tensor_set(write_slots, write_slots_data.data(), 0, ggml_nbytes(write_slots));
    ggml_backend_tensor_set(context_lens, context_lens_data, 0, sizeof(context_lens_data));
    ggml_backend_tensor_set(batch_offsets, batch_offsets_data, 0, sizeof(batch_offsets_data));
    ggml_backend_tensor_set(batch_lens, batch_lens_data, 0, sizeof(batch_lens_data));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(graph, out);
    EXPECT_TRUE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

template <int tokens_per_sequence>
static void check_paged_attention_production_case(
        ggml_backend_t cpu_backend, ggml_backend_t cuda_backend,
        int head_dim, int n_heads, int n_heads_kv, bool expect_tiled) {
    const std::vector<float> reference = run_paged_attention_production_case<tokens_per_sequence>(
        cpu_backend, head_dim, n_heads, n_heads_kv);
    ggml_paged_attn_tiled_prefill_launch_count_reset();
    const std::vector<float> actual = run_paged_attention_production_case<tokens_per_sequence>(
        cuda_backend, head_dim, n_heads, n_heads_kv);
    const unsigned long long launches = ggml_paged_attn_tiled_prefill_launch_count();
    EXPECT_TRUE(expect_tiled ? launches > 0 : launches == 0);
    EXPECT_TRUE(actual.size() == reference.size());

    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_TRUE(std::isfinite(actual[i]));
        squared_error += (double) (actual[i] - reference[i]) * (actual[i] - reference[i]);
        squared_reference += (double) reference[i] * reference[i];
    }
    EXPECT_TRUE(squared_reference > 0.0);
    EXPECT_TRUE(squared_error / squared_reference < 5e-3);
}

TEST(test_paged_attention_cuda_production_correctness) {
    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t cuda_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    EXPECT_TRUE(cpu_backend != nullptr);
    EXPECT_TRUE(cuda_backend != nullptr);

    check_paged_attention_production_case<20>(cpu_backend, cuda_backend, 128, 4, 1, true);
    check_paged_attention_production_case<20>(cpu_backend, cuda_backend, 256, 4, 2, true);
    check_paged_attention_production_case<20>(cpu_backend, cuda_backend, 64, 4, 1, false);
    check_paged_attention_production_case<1>(cpu_backend, cuda_backend, 256, 8, 1, false);

    ggml_backend_free(cuda_backend);
    ggml_backend_free(cpu_backend);
}

struct paged_decode_result {
    std::vector<float> output;
    std::vector<uint8_t> k_cache;
    std::vector<uint8_t> v_cache;
};

struct paged_decode_resources {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
};

static paged_decode_result run_paged_attention_decode_case(
        ggml_backend_t backend, int n_heads, int n_heads_kv,
        int context_length, int n_sequences, int block_size,
        const std::vector<int32_t> * initial_context_lens = nullptr,
        const std::vector<int32_t> * replay_context_lens = nullptr,
        int n_graph_computes = 1, bool remap_before_last_compute = false,
        paged_decode_resources * retained_resources = nullptr) {
    constexpr int head_dim = 256;
    int max_context_length = context_length;
    for (const std::vector<int32_t> * lengths : { initial_context_lens, replay_context_lens }) {
        if (lengths != nullptr) {
            EXPECT_TRUE((int) lengths->size() == n_sequences);
            max_context_length = std::max(max_context_length, *std::max_element(lengths->begin(), lengths->end()));
        }
    }
    const int max_blocks = (max_context_length + block_size - 1) / block_size;
    const int n_cache_blocks = n_sequences * max_blocks;
    const int n_tokens = n_sequences;
    const int n_write_rows = n_tokens * n_heads_kv;

    const size_t graph_size = 16;
    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    EXPECT_TRUE(ctx != nullptr);

    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, n_tokens);
    ggml_tensor * k_new = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads_kv, n_tokens);
    ggml_tensor * v_new = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads_kv, n_tokens);
    ggml_tensor * k_cache = ggml_new_tensor_4d(
        ctx, GGML_TYPE_Q8_0, head_dim, block_size, n_heads_kv, n_cache_blocks);
    ggml_tensor * v_cache = ggml_new_tensor_4d(
        ctx, GGML_TYPE_Q8_0, head_dim, block_size, n_heads_kv, n_cache_blocks);
    ggml_tensor * block_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, max_blocks, n_sequences);
    ggml_tensor * write_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_write_rows);
    ggml_tensor * context_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_sequences);
    ggml_tensor * batch_offsets = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_sequences);
    ggml_tensor * batch_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_sequences);
    ggml_tensor * out = ggml_paged_attn(
        ctx, q, k_new, v_new, k_cache, v_cache, block_table, write_rows,
        context_lens, batch_offsets, batch_lens, 1.0f / std::sqrt((float) head_dim), block_size, max_blocks,
        ggml_paged_attn_context_bucket(context_length));

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    EXPECT_TRUE(buffer != nullptr);

    std::vector<float> q_data(ggml_nelements(q));
    std::vector<float> k_new_data(ggml_nelements(k_new));
    std::vector<float> v_new_data(ggml_nelements(v_new));
    for (size_t i = 0; i < q_data.size(); ++i) {
        const int dim = i % head_dim;
        const int head = (i / head_dim) % n_heads;
        const int seq = i / ((size_t) head_dim * n_heads);
        q_data[i] = 0.55f * std::sin(0.071f * (dim + 1) * (1 + head % 3)) +
            0.35f * std::cos(0.113f * (dim + 3) + 0.29f * head + 0.41f * seq);
    }
    for (size_t i = 0; i < k_new_data.size(); ++i) {
        const int dim = i % head_dim;
        const int head = (i / head_dim) % n_heads_kv;
        const int seq = i / ((size_t) head_dim * n_heads_kv);
        k_new_data[i] = 0.60f * std::sin(0.071f * (dim + 1) * (1 + (head + seq) % 3)) +
            0.30f * std::cos(0.113f * (dim + 3) + 0.37f * head - 0.43f * seq);
        v_new_data[i] = 0.45f * std::cos(0.097f * (dim + 5) + 0.31f * head + 0.53f * seq);
    }

    const size_t n_cache_rows = (size_t) n_cache_blocks * n_heads_kv * block_size;
    std::vector<float> k_data(n_cache_rows * head_dim);
    std::vector<float> v_data(n_cache_rows * head_dim);
    for (size_t row = 0; row < n_cache_rows; ++row) {
        for (int dim = 0; dim < head_dim; ++dim) {
            const size_t i = row * head_dim + dim;
            const float direction = row % 2 == 0 ? 1.0f : -1.0f;
            k_data[i] = direction * 0.65f * std::sin(0.071f * (dim + 1) * (1 + row % 3)) +
                0.35f * std::cos(0.113f * (dim + 3) + 0.37f * row);
            v_data[i] = 0.50f * std::cos(0.097f * (dim + 5) + 0.23f * row) +
                0.15f * std::sin(0.047f * (dim + 1) * (1 + row % 5));
        }
    }
    std::vector<uint8_t> k_quantized(ggml_nbytes(k_cache));
    std::vector<uint8_t> v_quantized(ggml_nbytes(v_cache));
    EXPECT_TRUE(ggml_quantize_chunk(
        GGML_TYPE_Q8_0, k_data.data(), k_quantized.data(), 0, n_cache_rows, head_dim, nullptr) == k_quantized.size());
    EXPECT_TRUE(ggml_quantize_chunk(
        GGML_TYPE_Q8_0, v_data.data(), v_quantized.data(), 0, n_cache_rows, head_dim, nullptr) == v_quantized.size());

    std::vector<int32_t> block_table_data((size_t) n_sequences * max_blocks);
    std::vector<int32_t> write_rows_data(n_write_rows);
    std::vector<int32_t> context_lens_data = initial_context_lens != nullptr ?
        *initial_context_lens : std::vector<int32_t>(n_sequences, context_length);
    std::vector<int32_t> batch_offsets_data(n_sequences);
    std::vector<int32_t> batch_lens_data(n_sequences, 1);
    for (int seq = 0; seq < n_sequences; ++seq) {
        batch_offsets_data[seq] = seq;
        for (int logical = 0; logical < max_blocks; ++logical) {
            block_table_data[(size_t) seq * max_blocks + logical] =
                seq * max_blocks + max_blocks - logical - 1;
        }
        const int logical_pos = context_lens_data[seq] - 1;
        const int physical_block = block_table_data[(size_t) seq * max_blocks + logical_pos / block_size];
        for (int head = 0; head < n_heads_kv; ++head) {
            write_rows_data[seq * n_heads_kv + head] = physical_block * n_heads_kv * block_size +
                head * block_size + logical_pos % block_size;
        }
    }

    ggml_backend_tensor_set(q, q_data.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k_new, k_new_data.data(), 0, ggml_nbytes(k_new));
    ggml_backend_tensor_set(v_new, v_new_data.data(), 0, ggml_nbytes(v_new));
    ggml_backend_tensor_set(k_cache, k_quantized.data(), 0, k_quantized.size());
    ggml_backend_tensor_set(v_cache, v_quantized.data(), 0, v_quantized.size());
    ggml_backend_tensor_set(block_table, block_table_data.data(), 0, ggml_nbytes(block_table));
    ggml_backend_tensor_set(write_rows, write_rows_data.data(), 0, ggml_nbytes(write_rows));
    ggml_backend_tensor_set(context_lens, context_lens_data.data(), 0, ggml_nbytes(context_lens));
    ggml_backend_tensor_set(batch_offsets, batch_offsets_data.data(), 0, ggml_nbytes(batch_offsets));
    ggml_backend_tensor_set(batch_lens, batch_lens_data.data(), 0, ggml_nbytes(batch_lens));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(graph, out);
    EXPECT_TRUE(n_graph_computes > 0);
    for (int compute = 0; compute < n_graph_computes; ++compute) {
        if (compute == n_graph_computes - 1 &&
                (remap_before_last_compute || replay_context_lens != nullptr)) {
            if (replay_context_lens != nullptr) {
                context_lens_data = *replay_context_lens;
            }
            for (int seq = 0; seq < n_sequences; ++seq) {
                if (remap_before_last_compute && max_blocks > 1) {
                    std::swap(block_table_data[(size_t) seq * max_blocks],
                        block_table_data[(size_t) seq * max_blocks + 1]);
                }
                const int logical_pos = context_lens_data[seq] - 1;
                const int physical_block = block_table_data[(size_t) seq * max_blocks + logical_pos / block_size];
                for (int head = 0; head < n_heads_kv; ++head) {
                    write_rows_data[seq * n_heads_kv + head] = physical_block * n_heads_kv * block_size +
                        head * block_size + logical_pos % block_size;
                }
            }
            ggml_backend_tensor_set(block_table, block_table_data.data(), 0, ggml_nbytes(block_table));
            ggml_backend_tensor_set(write_rows, write_rows_data.data(), 0, ggml_nbytes(write_rows));
            ggml_backend_tensor_set(context_lens, context_lens_data.data(), 0, ggml_nbytes(context_lens));
        }
        EXPECT_TRUE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    }

    paged_decode_result result;
    result.output.resize(ggml_nelements(out));
    result.k_cache.resize(ggml_nbytes(k_cache));
    result.v_cache.resize(ggml_nbytes(v_cache));
    ggml_backend_tensor_get(out, result.output.data(), 0, ggml_nbytes(out));
    ggml_backend_tensor_get(k_cache, result.k_cache.data(), 0, ggml_nbytes(k_cache));
    ggml_backend_tensor_get(v_cache, result.v_cache.data(), 0, ggml_nbytes(v_cache));
    if (retained_resources != nullptr) {
        retained_resources->ctx = ctx;
        retained_resources->buffer = buffer;
        retained_resources->graph = graph;
    } else {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    return result;
}

static int run_paged_attention_decode_benchmark(int argc, char ** argv) {
    if (argc != 8) {
        fprintf(stderr, "usage: %s --paged-decode-bench DEPTH N_HEADS N_HEADS_KV N_SEQUENCES BLOCK_SIZE REPS\n", argv[0]);
        return 1;
    }
    const int context_length = atoi(argv[2]);
    const int n_heads = atoi(argv[3]);
    const int n_heads_kv = atoi(argv[4]);
    const int n_sequences = atoi(argv[5]);
    const int block_size = atoi(argv[6]);
    const int repetitions = atoi(argv[7]);
    constexpr int head_dim = 256;
    const int max_blocks = (context_length + block_size - 1) / block_size;
    const int n_cache_blocks = n_sequences * max_blocks;
    const int n_write_rows = n_sequences * n_heads_kv;
    const size_t graph_size = 16;
    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    EXPECT_TRUE(ctx != nullptr);
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    EXPECT_TRUE(backend != nullptr);

    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, n_sequences);
    ggml_tensor * k_new = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads_kv, n_sequences);
    ggml_tensor * v_new = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads_kv, n_sequences);
    ggml_tensor * k_cache = ggml_new_tensor_4d(
        ctx, GGML_TYPE_Q8_0, head_dim, block_size, n_heads_kv, n_cache_blocks);
    ggml_tensor * v_cache = ggml_new_tensor_4d(
        ctx, GGML_TYPE_Q8_0, head_dim, block_size, n_heads_kv, n_cache_blocks);
    ggml_tensor * block_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, max_blocks, n_sequences);
    ggml_tensor * write_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_write_rows);
    ggml_tensor * context_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_sequences);
    ggml_tensor * batch_offsets = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_sequences);
    ggml_tensor * batch_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_sequences);
    ggml_tensor * out = ggml_paged_attn(
        ctx, q, k_new, v_new, k_cache, v_cache, block_table, write_rows,
        context_lens, batch_offsets, batch_lens, 1.0f / std::sqrt((float) head_dim), block_size, max_blocks,
        ggml_paged_attn_context_bucket(context_length));
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    EXPECT_TRUE(buffer != nullptr);

    std::vector<float> q_data(ggml_nelements(q));
    std::vector<float> new_data(ggml_nelements(k_new));
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = 0.25f * std::sin(0.017f * (float) (i + 1));
    }
    for (size_t i = 0; i < new_data.size(); ++i) {
        new_data[i] = 0.20f * std::cos(0.013f * (float) (i + 3));
    }

    std::vector<uint8_t> cache_data(ggml_nbytes(k_cache));
    std::vector<float> cache_row(head_dim);
    const size_t cache_row_size = ggml_row_size(GGML_TYPE_Q8_0, head_dim);
    std::vector<uint8_t> quantized_row(cache_row_size);
    for (int i = 0; i < head_dim; ++i) {
        cache_row[i] = 0.45f * std::sin(0.031f * (float) (i + 1));
    }
    EXPECT_TRUE(ggml_quantize_chunk(
        GGML_TYPE_Q8_0, cache_row.data(), quantized_row.data(), 0, 1, head_dim, nullptr) == cache_row_size);
    for (size_t offset = 0; offset < cache_data.size(); offset += cache_row_size) {
        memcpy(cache_data.data() + offset, quantized_row.data(), cache_row_size);
    }

    std::vector<int32_t> block_table_data((size_t) n_sequences * max_blocks);
    std::vector<int32_t> write_rows_data(n_write_rows);
    std::vector<int32_t> context_lens_data(n_sequences, context_length);
    std::vector<int32_t> batch_offsets_data(n_sequences);
    std::vector<int32_t> batch_lens_data(n_sequences, 1);
    for (int seq = 0; seq < n_sequences; ++seq) {
        batch_offsets_data[seq] = seq;
        for (int logical = 0; logical < max_blocks; ++logical) {
            block_table_data[(size_t) seq * max_blocks + logical] = seq * max_blocks + max_blocks - logical - 1;
        }
        const int logical_pos = context_length - 1;
        const int physical_block = block_table_data[(size_t) seq * max_blocks + logical_pos / block_size];
        for (int head = 0; head < n_heads_kv; ++head) {
            write_rows_data[seq * n_heads_kv + head] = physical_block * n_heads_kv * block_size +
                head * block_size + logical_pos % block_size;
        }
    }

    ggml_backend_tensor_set(q, q_data.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k_new, new_data.data(), 0, ggml_nbytes(k_new));
    ggml_backend_tensor_set(v_new, new_data.data(), 0, ggml_nbytes(v_new));
    ggml_backend_tensor_set(k_cache, cache_data.data(), 0, cache_data.size());
    ggml_backend_tensor_set(v_cache, cache_data.data(), 0, cache_data.size());
    ggml_backend_tensor_set(block_table, block_table_data.data(), 0, ggml_nbytes(block_table));
    ggml_backend_tensor_set(write_rows, write_rows_data.data(), 0, ggml_nbytes(write_rows));
    ggml_backend_tensor_set(context_lens, context_lens_data.data(), 0, ggml_nbytes(context_lens));
    ggml_backend_tensor_set(batch_offsets, batch_offsets_data.data(), 0, ggml_nbytes(batch_offsets));
    ggml_backend_tensor_set(batch_lens, batch_lens_data.data(), 0, ggml_nbytes(batch_lens));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(graph, out);
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    }
    ggml_backend_synchronize(backend);
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repetitions; ++i) {
        EXPECT_TRUE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    }
    ggml_backend_synchronize(backend);
    const auto elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    fprintf(stdout,
        "paged_decode_bench depth=%d heads=%d kv_heads=%d sequences=%d block=%d reps=%d us_per_op=%.3f\n",
        context_length, n_heads, n_heads_kv, n_sequences, block_size, repetitions, elapsed / repetitions);

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}

static double paged_decode_relative_squared_error(
        const std::vector<float> & actual, const std::vector<float> & reference) {
    EXPECT_TRUE(actual.size() == reference.size());
    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_TRUE(std::isfinite(actual[i]));
        squared_error += (double) (actual[i] - reference[i]) * (actual[i] - reference[i]);
        squared_reference += (double) reference[i] * reference[i];
    }
    EXPECT_TRUE(squared_reference > 0.0);
    return squared_error / squared_reference;
}

static void check_paged_attention_decode_case(
        ggml_backend_t cpu_backend, ggml_backend_t cuda_backend,
        int n_heads, int n_heads_kv, int context_length, int n_sequences, int block_size) {
    const paged_decode_result reference = run_paged_attention_decode_case(
        cpu_backend, n_heads, n_heads_kv, context_length, n_sequences, block_size);

    setenv("GGML_CUDA_PAGED_Q8_FUSED_WRITE", "0", 1);
    ggml_paged_attn_q8_combined_write_launch_count_reset();
    ggml_paged_attn_q8_fused_write_launch_count_reset();
    const paged_decode_result combined = run_paged_attention_decode_case(
        cuda_backend, n_heads, n_heads_kv, context_length, n_sequences, block_size);
    const unsigned long long combined_write_launches = ggml_paged_attn_q8_combined_write_launch_count();
    EXPECT_TRUE(combined_write_launches == 1);
    EXPECT_TRUE(ggml_paged_attn_q8_fused_write_launch_count() == 0);
    unsetenv("GGML_CUDA_PAGED_Q8_FUSED_WRITE");

    setenv("GGML_CUDA_PAGED_Q8_WARPS", "8", 1);
    setenv("GGML_CUDA_PAGED_Q8_PARTITIONS", "1", 1);
    setenv("GGML_CUDA_PAGED_Q8_HEADS", "2", 1);
    setenv("GGML_CUDA_PAGED_Q8_FUSED_WRITE", "1", 1);
    ggml_paged_attn_q8_combined_write_launch_count_reset();
    ggml_paged_attn_q8_fused_write_launch_count_reset();
    const paged_decode_result fused = run_paged_attention_decode_case(
        cuda_backend, n_heads, n_heads_kv, context_length, n_sequences, block_size);
    const unsigned long long fused_combined_write_launches = ggml_paged_attn_q8_combined_write_launch_count();
    const unsigned long long fused_write_launches = ggml_paged_attn_q8_fused_write_launch_count();
    EXPECT_TRUE(fused_combined_write_launches == 0);
    EXPECT_TRUE(fused_write_launches == 1);
    unsetenv("GGML_CUDA_PAGED_Q8_WARPS");
    unsetenv("GGML_CUDA_PAGED_Q8_PARTITIONS");
    unsetenv("GGML_CUDA_PAGED_Q8_HEADS");
    unsetenv("GGML_CUDA_PAGED_Q8_FUSED_WRITE");

    setenv("GGML_CUDA_PAGED_Q8_FLOAT_Q", "1", 1);
    ggml_paged_attn_q8_decode_launch_count_reset();
    ggml_paged_attn_q8_combined_write_launch_count_reset();
    ggml_paged_attn_q8_fused_write_launch_count_reset();
    const paged_decode_result float_q = run_paged_attention_decode_case(
        cuda_backend, n_heads, n_heads_kv, context_length, n_sequences, block_size);
    EXPECT_TRUE(ggml_paged_attn_q8_decode_launch_count() > 0);
    EXPECT_TRUE(ggml_paged_attn_q8_combined_write_launch_count() > 0);
    EXPECT_TRUE(ggml_paged_attn_q8_fused_write_launch_count() == 0);
    unsetenv("GGML_CUDA_PAGED_Q8_FLOAT_Q");

    ggml_paged_attn_q8_decode_launch_count_reset();
    const paged_decode_result packed_q = run_paged_attention_decode_case(
        cuda_backend, n_heads, n_heads_kv, context_length, n_sequences, block_size);
    EXPECT_TRUE(ggml_paged_attn_q8_decode_launch_count() > 0);
    EXPECT_TRUE(float_q.k_cache == reference.k_cache);
    EXPECT_TRUE(float_q.v_cache == reference.v_cache);
    EXPECT_TRUE(packed_q.k_cache == reference.k_cache);
    EXPECT_TRUE(packed_q.v_cache == reference.v_cache);
    EXPECT_TRUE(combined.k_cache == reference.k_cache);
    EXPECT_TRUE(combined.v_cache == reference.v_cache);
    EXPECT_TRUE(fused.k_cache == combined.k_cache);
    EXPECT_TRUE(fused.v_cache == combined.v_cache);

    const double combined_error = paged_decode_relative_squared_error(combined.output, reference.output);
    const double fused_error = paged_decode_relative_squared_error(fused.output, combined.output);
    const double float_error = paged_decode_relative_squared_error(float_q.output, reference.output);
    const double packed_error = paged_decode_relative_squared_error(packed_q.output, reference.output);
    const double added_error = fmax(0.0, packed_error - float_error);
    fprintf(stderr,
        " q8_1 error: combined=%g fused-vs-combined=%g float=%g packed=%g added=%g "
        "write-launches:fallback=%llu/fused=%llu/fused-combined=%llu ",
        combined_error, fused_error, float_error, packed_error, added_error,
        combined_write_launches, fused_write_launches, fused_combined_write_launches);
    EXPECT_TRUE(combined_error < 5e-3);
    EXPECT_TRUE(fused_error < 5e-3);
    EXPECT_TRUE(float_error < 5e-3);
    EXPECT_TRUE(packed_error < 5e-3);
    EXPECT_TRUE(added_error < 1e-4);
}

static void check_paged_attention_decode_variant(
        ggml_backend_t cpu_backend, ggml_backend_t cuda_backend,
        int n_heads, int n_heads_kv, int context_length, int n_sequences, int block_size,
        int n_warps, int n_partitions, int n_q_heads) {
    const paged_decode_result reference = run_paged_attention_decode_case(
        cpu_backend, n_heads, n_heads_kv, context_length, n_sequences, block_size);
    const std::string warps = std::to_string(n_warps);
    const std::string partitions = std::to_string(n_partitions);
    const std::string q_heads = std::to_string(n_q_heads);
    setenv("GGML_CUDA_PAGED_Q8_WARPS", warps.c_str(), 1);
    setenv("GGML_CUDA_PAGED_Q8_PARTITIONS", partitions.c_str(), 1);
    setenv("GGML_CUDA_PAGED_Q8_HEADS", q_heads.c_str(), 1);
    setenv("GGML_CUDA_PAGED_Q8_FUSED_WRITE", "1", 1);
    ggml_paged_attn_q8_combined_write_launch_count_reset();
    ggml_paged_attn_q8_fused_write_launch_count_reset();
    const paged_decode_result actual = run_paged_attention_decode_case(
        cuda_backend, n_heads, n_heads_kv, context_length, n_sequences, block_size);
    unsetenv("GGML_CUDA_PAGED_Q8_WARPS");
    unsetenv("GGML_CUDA_PAGED_Q8_PARTITIONS");
    unsetenv("GGML_CUDA_PAGED_Q8_HEADS");
    unsetenv("GGML_CUDA_PAGED_Q8_FUSED_WRITE");
    EXPECT_TRUE((ggml_paged_attn_q8_fused_write_launch_count() > 0) == (n_q_heads > 1));
    EXPECT_TRUE((ggml_paged_attn_q8_combined_write_launch_count() > 0) == (n_q_heads == 1));
    EXPECT_TRUE(actual.k_cache == reference.k_cache);
    EXPECT_TRUE(actual.v_cache == reference.v_cache);
    EXPECT_TRUE(paged_decode_relative_squared_error(actual.output, reference.output) < 5e-3);
}

TEST(test_paged_attention_q8_decode_boundaries) {
    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t cuda_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    EXPECT_TRUE(cpu_backend != nullptr);
    EXPECT_TRUE(cuda_backend != nullptr);

    for (int context_length : { 1, 15, 16, 17, 31, 32, 33 }) {
        check_paged_attention_decode_case(cpu_backend, cuda_backend, 24, 4, context_length, 1, 16);
    }
    check_paged_attention_decode_case(cpu_backend, cuda_backend, 24, 4, 33, 3, 16);
    check_paged_attention_decode_case(cpu_backend, cuda_backend, 32, 4, 33, 2, 32);

    ggml_backend_free(cuda_backend);
    ggml_backend_free(cpu_backend);
}

TEST(test_paged_attention_q8_partition_and_group_candidates) {
    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t cuda_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    EXPECT_TRUE(cpu_backend != nullptr);
    EXPECT_TRUE(cuda_backend != nullptr);

    for (int n_partitions : { 1, 2, 4, 8 }) {
        check_paged_attention_decode_variant(
            cpu_backend, cuda_backend, 24, 4, 65, 1, 16, 8, n_partitions, 1);
    }
    for (int n_partitions : { 1, 2, 4, 8 }) {
        check_paged_attention_decode_variant(
            cpu_backend, cuda_backend, 24, 4, 65, 3, 16, 8, n_partitions, 2);
        check_paged_attention_decode_variant(
            cpu_backend, cuda_backend, 32, 4, 97, 2, 32, 8, n_partitions, 2);
    }
    for (int n_partitions : { 1, 2, 4, 8 }) {
        check_paged_attention_decode_variant(
            cpu_backend, cuda_backend, 32, 4, 97, 2, 32, 16, n_partitions, 4);
    }
    setenv("GGML_CUDA_PAGED_Q8_CP_ASYNC", "1", 1);
    check_paged_attention_decode_variant(
        cpu_backend, cuda_backend, 24, 4, 97, 4, 16, 8, 8, 2);
    unsetenv("GGML_CUDA_PAGED_Q8_CP_ASYNC");

    ggml_backend_free(cuda_backend);
    ggml_backend_free(cpu_backend);
}

TEST(test_paged_attention_q8_graph_replay_restore_and_mixed_depth) {
    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t cuda_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    EXPECT_TRUE(cpu_backend != nullptr);
    EXPECT_TRUE(cuda_backend != nullptr);

    const std::vector<int32_t> initial_lens = { 4097, 1025, 128, 65 };
    const std::vector<int32_t> restored_lens = { 8192, 2049, 256, 33 };
    const paged_decode_result reference = run_paged_attention_decode_case(
        cpu_backend, 24, 4, 8192, 4, 16, &initial_lens, &restored_lens, 2, true);

    setenv("GGML_CUDA_PAGED_Q8_WARPS", "8", 1);
    setenv("GGML_CUDA_PAGED_Q8_HEADS", "2", 1);
    setenv("GGML_CUDA_PAGED_Q8_FUSED_WRITE", "1", 1);
    ggml_paged_attn_q8_decode_launch_count_reset();
    ggml_paged_attn_q8_combined_write_launch_count_reset();
    ggml_paged_attn_q8_fused_write_launch_count_reset();
    const paged_decode_result replayed = run_paged_attention_decode_case(
        cuda_backend, 24, 4, 8192, 4, 16, &initial_lens, &restored_lens, 4, true);
    const unsigned long long launches = ggml_paged_attn_q8_decode_launch_count();
    const unsigned long long fused_launches = ggml_paged_attn_q8_fused_write_launch_count();
    const unsigned long long combined_launches = ggml_paged_attn_q8_combined_write_launch_count();
    fprintf(stderr, " mixed-depth same-bucket graph: host=%llu fused=%llu combined=%llu replayed=2 remap=1 ",
        launches, fused_launches, combined_launches);
    EXPECT_TRUE(launches == 2);
    EXPECT_TRUE(fused_launches == launches);
    EXPECT_TRUE(combined_launches == 0);
    EXPECT_TRUE(replayed.k_cache == reference.k_cache);
    EXPECT_TRUE(replayed.v_cache == reference.v_cache);
    EXPECT_TRUE(paged_decode_relative_squared_error(replayed.output, reference.output) < 5e-3);

    ggml_backend_free(cuda_backend);
    ggml_backend_free(cpu_backend);

    ggml_backend_t transition_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    EXPECT_TRUE(transition_backend != nullptr);
    ggml_paged_attn_q8_decode_launch_count_reset();
    ggml_paged_attn_q8_combined_write_launch_count_reset();
    ggml_paged_attn_q8_fused_write_launch_count_reset();
    paged_decode_resources transition_resources[2];
    int transition_index = 0;
    unsigned long long previous_launches = 0;
    unsigned long long previous_fused_launches = 0;
    for (int context_length : { 4096, 65536 }) {
        const paged_decode_result transitioned = run_paged_attention_decode_case(
            transition_backend, 24, 4, context_length, 1, 16, nullptr, nullptr, 3, false,
            &transition_resources[transition_index++]);
        EXPECT_TRUE(!transitioned.output.empty());
        const unsigned long long transition_launches = ggml_paged_attn_q8_decode_launch_count();
        const unsigned long long transition_fused_launches = ggml_paged_attn_q8_fused_write_launch_count();
        fprintf(stderr, " bucket-transition depth=%d host=%llu fused=%llu combined=%llu replayed=1 ",
            context_length, transition_launches - previous_launches,
            transition_fused_launches - previous_fused_launches,
            ggml_paged_attn_q8_combined_write_launch_count());
        EXPECT_TRUE(transition_launches - previous_launches == 2);
        EXPECT_TRUE(transition_fused_launches - previous_fused_launches == 2);
        EXPECT_TRUE(ggml_paged_attn_q8_combined_write_launch_count() == 0);
        previous_launches = transition_launches;
        previous_fused_launches = transition_fused_launches;
    }
    EXPECT_TRUE(ggml_backend_graph_compute(transition_backend, transition_resources[0].graph) == GGML_STATUS_SUCCESS);
    EXPECT_TRUE(ggml_backend_graph_compute(transition_backend, transition_resources[1].graph) == GGML_STATUS_SUCCESS);
    EXPECT_TRUE(ggml_paged_attn_q8_decode_launch_count() == previous_launches);
    EXPECT_TRUE(ggml_paged_attn_q8_fused_write_launch_count() == previous_fused_launches);
    EXPECT_TRUE(ggml_paged_attn_q8_combined_write_launch_count() == 0);
    fprintf(stderr, " bucket-transition cached variants replayed=2 ");
    for (paged_decode_resources & resources : transition_resources) {
        ggml_backend_buffer_free(resources.buffer);
        ggml_free(resources.ctx);
    }
    ggml_backend_free(transition_backend);
    unsetenv("GGML_CUDA_PAGED_Q8_WARPS");
    unsetenv("GGML_CUDA_PAGED_Q8_HEADS");
    unsetenv("GGML_CUDA_PAGED_Q8_FUSED_WRITE");
}

TEST(test_paged_attention_cuda_runtime_coverage) {
    EXPECT_TRUE(ggml_paged_attn_cuda_runtime_test());
}
#endif


TEST(test_scheduler_rejects_oversized_prompt) {
    auto fixture = make_fixture(/*n_ctx=*/64, /*block_size=*/16, /*n_batch=*/128,
                                /*n_gpu_blocks=*/32, /*n_cpu_blocks=*/8);

    bool queued = fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/64));
    EXPECT_FALSE(queued);

    queued = fixture.sched->queue_request(make_group(/*id=*/1, /*n_prompt=*/128));
    EXPECT_FALSE(queued);

    // A request just under the limit is accepted.
    queued = fixture.sched->queue_request(make_group(/*id=*/2, /*n_prompt=*/63));
    EXPECT_TRUE(queued);
}

TEST(test_scheduler_swaps_and_resumes_request) {
    auto fixture = make_fixture(/*n_ctx=*/128, /*block_size=*/32, /*n_batch=*/64,
                                /*n_gpu_blocks=*/3, /*n_cpu_blocks=*/2);
    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/30)));
    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/1, /*n_prompt=*/30)));

    llama_batch batch = {};
    const int8_t continue_flags[] = { 0, 0 };
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    fixture.sched->update(batch, { 1, 1 }, continue_flags);
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    fixture.sched->update(batch, { 2, 2 }, continue_flags);

    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    fixture.sched->update(batch, { 3, 3 }, continue_flags);
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    auto * first  = fixture.sched->get_group_from_id(0);
    auto * second = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(first != nullptr);
    EXPECT_TRUE(second != nullptr);
    EXPECT_TRUE(first->status == llama_sequence_group_status::SWAPPED ||
                second->status == llama_sequence_group_status::SWAPPED);

    for (int i = 0; i < 21; ++i) {
        fixture.sched->update(batch, { 4 + i }, continue_flags);
        EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    }

    const int8_t finish_flags[] = { 1 };
    fixture.sched->update(batch, { 25 }, finish_flags);
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    first  = fixture.sched->get_group_from_id(0);
    second = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(first == nullptr);
    EXPECT_TRUE(second != nullptr);
    EXPECT_TRUE(second->status == llama_sequence_group_status::RUNNING);
    EXPECT_TRUE(second->n_past >= 32);
    llama_batch_free(batch);
}

TEST(test_scheduler_reserves_spec_batch_after_swap) {
    auto fixture = make_fixture(/*n_ctx=*/128, /*block_size=*/32, /*n_batch=*/64,
                                /*n_gpu_blocks=*/2, /*n_cpu_blocks=*/1);
    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/30)));
    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/1, /*n_prompt=*/30)));

    llama_batch batch = {};
    const int8_t continue_flags[] = { 0, 0 };
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    fixture.sched->update(batch, { 1, 2 }, continue_flags);

    EXPECT_TRUE(fixture.sched->step(batch, /*spec_n=*/3) == llama_scheduler_status::OK);
    EXPECT_EQ(batch.n_tokens, 4);
    const llama_paged_batch_info * info = fixture.sched->get_curr_batch_info();
    EXPECT_EQ(info->n_seq, 1);
    EXPECT_EQ(info->batch_lens[0], 4);

    auto * first  = fixture.sched->get_group_from_id(0);
    auto * second = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(first != nullptr);
    EXPECT_TRUE(second != nullptr);
    EXPECT_EQ(first->block_table.size(), 2u);
    EXPECT_TRUE(second->status == llama_sequence_group_status::SWAPPED);

    const int8_t finish_flags[] = { 1 };
    fixture.sched->update(batch, { 3 }, { 4 }, finish_flags);

    EXPECT_TRUE(fixture.sched->step(batch, /*spec_n=*/3) == llama_scheduler_status::OK);
    info = fixture.sched->get_curr_batch_info();
    EXPECT_EQ(info->n_seq, 1);
    EXPECT_EQ(info->batch_lens[0], 4);
    EXPECT_EQ(batch.n_tokens, 4);
    EXPECT_EQ(batch.seq_id[0][0], 1);
    second = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(second != nullptr);
    EXPECT_TRUE(second->status == llama_sequence_group_status::RUNNING);
    EXPECT_EQ(second->block_table.size(), 2u);
    llama_batch_free(batch);
}

TEST(test_scheduler_caps_constrained_spec_batch) {
    auto fixture = make_fixture(/*n_ctx=*/128, /*block_size=*/16, /*n_batch=*/4,
                                /*n_gpu_blocks=*/4, /*n_cpu_blocks=*/2);
    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/1)));
    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/1, /*n_prompt=*/1)));

    llama_batch batch = {};
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    const int8_t continue_flags[] = { 0, 0 };
    fixture.sched->update(batch, { 2, 3 }, continue_flags);

    EXPECT_TRUE(fixture.sched->step(batch, /*spec_n=*/3) == llama_scheduler_status::OK);
    const auto * info = fixture.sched->get_curr_batch_info();
    EXPECT_EQ(info->n_seq, 1);
    EXPECT_EQ(info->batch_lens[0], 4);
    EXPECT_EQ(batch.n_tokens, 4);
    const int8_t stop_flag[] = { 1 };
    fixture.sched->update(batch, { 4 }, { 2 }, stop_flag);
    llama_batch_free(batch);
}

int main(int argc, char ** argv) {
#if defined(GGML_USE_CUDA)
    if (argc > 1 && strcmp(argv[1], "--paged-graph-test") == 0) {
        unsetenv("GGML_CUDA_DISABLE_GRAPHS");
        RUN(test_paged_attention_q8_graph_replay_restore_and_mixed_depth);
        return 0;
    }
    setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
    if (argc > 1 && strcmp(argv[1], "--paged-decode-bench") == 0) {
        return run_paged_attention_decode_benchmark(argc, argv);
    }
#else
    GGML_UNUSED(argc);
    GGML_UNUSED(argv);
#endif
    fprintf(stderr, "test-paged-kv: block_manager\n");
    RUN(test_paged_graph_tensor_compatibility);
    RUN(test_paged_graph_reuse_accepts_mapping_remap);
    RUN(test_paged_graph_reuse_rejects_pool_change);
    RUN(test_block_manager_leak_simple);
    RUN(test_block_manager_leak_repeated);
    RUN(test_block_manager_checkout_too_many);
    RUN(test_block_manager_watermark);
    RUN(test_block_manager_watermark_zero);
    RUN(test_block_manager_gpu_cpu_disjoint);

    fprintf(stderr, "test-paged-kv: llama_kv_cache_paged seq_pos\n");
    RUN(test_seq_pos_default_unknown);
    RUN(test_seq_pos_set_and_get);
    RUN(test_seq_pos_overwrite);
    RUN(test_seq_pos_seq_rm_removes);
    RUN(test_seq_pos_clear_removes_all);

    fprintf(stderr, "test-paged-kv: llama_kv_cache_paged free_blocks\n");
    RUN(test_free_blocks_releases_to_pool);
    RUN(test_scheduler_caps_constrained_spec_batch);
    RUN(test_release_seq_tail_releases_trailing_blocks);
    RUN(test_clear_and_seq_rm_release_blocks);
    RUN(test_paged_state_round_trip);
    RUN(test_paged_sequence_state_preserves_other_sequences);
    RUN(test_paged_state_round_trip_after_swap);
    RUN(test_paged_storage_types_are_native);

    RUN(test_scheduler_state_restores_block_ownership);
    RUN(test_scheduler_resumes_fresh_checkpoint);
    RUN(test_paged_sequence_state_read_failure_preserves_live_cache);
    RUN(test_paged_state_read_failure_preserves_live_cache);
    RUN(test_scheduler_teardown_unregisters_groups);
    fprintf(stderr, "test-paged-kv: llama_kv_cache_paged scheduler\n");
    RUN(test_scheduler_no_deadlock_on_empty);
    RUN(test_scheduler_deadlock_oversize_waiting_request);
    RUN(test_scheduler_rejects_oversized_prompt);
    RUN(test_scheduler_prioritizes_oldest_request_under_pressure);
    RUN(test_scheduler_admits_full_token_budget_prefill);
    RUN(test_scheduler_batches_two_cross_block_prefills);
    RUN(test_paged_attention_head_mapping_and_dispatch_selection);
    RUN(test_scheduler_swaps_and_resumes_request);
    RUN(test_scheduler_reserves_spec_batch_after_swap);

#if defined(GGML_USE_CUDA)
    RUN(test_paged_attention_cuda_production_correctness);
    RUN(test_paged_attention_q8_decode_boundaries);
    RUN(test_paged_attention_q8_partition_and_group_candidates);
    RUN(test_paged_attention_cuda_runtime_coverage);
#endif
    RUN(test_decode_only_batch_skips_tiled_prefill);
    fprintf(stderr, "test-paged-kv: ALL PASSED\n");
    return 0;
}
