#include "ggml-backend.h"
#include "llama-block-manager.h"
#include "llama-kv-cache-paged.h"
#include "llama-io.h"
#include "llama-paged-scheduler-impl.h"
#include <stdexcept>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

    for (const ggml_type type : { GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0 }) {
        llama_kv_cache_paged kv(/*head_dim=*/128, /*n_heads_kv=*/4, /*block_size=*/16,
                                /*n_layers=*/2, /*n_ubatch=*/32, /*n_seq_max=*/8);
        kv.init(backend, backend, type, type, /*n_gpu_blocks=*/2, /*n_cpu_blocks=*/1, /*watermark=*/0.0f);
        const ggml_type expected = type == GGML_TYPE_Q8_0 ? type : GGML_TYPE_Q8_0;
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
    auto fixture = make_fixture(/*n_ctx=*/128, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/2, /*n_cpu_blocks=*/2);
    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/14)));
    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/1, /*n_prompt=*/14)));

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

    const int8_t finish_flags[] = { 1 };
    fixture.sched->update(batch, { 4 }, finish_flags);
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    first  = fixture.sched->get_group_from_id(0);
    second = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(first == nullptr);
    EXPECT_TRUE(second != nullptr);
    EXPECT_TRUE(second->status == llama_sequence_group_status::RUNNING);
    EXPECT_TRUE(second->n_past >= 16);
    llama_batch_free(batch);
}

int main(int /*argc*/, char ** /*argv*/) {
    fprintf(stderr, "test-paged-kv: block_manager\n");
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
    RUN(test_scheduler_swaps_and_resumes_request);

    fprintf(stderr, "test-paged-kv: ALL PASSED\n");
    return 0;
}
