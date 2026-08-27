#include "llama-kv-cache-paged.h"

#include "llama-impl.h"
#include "llama-io.h"

#include <algorithm>
#include <stdexcept>

static ggml_type paged_storage_type(ggml_type type) {
    return type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0 ? GGML_TYPE_Q8_0 : type;
}


// llama_kv_cache_paged
//

llama_kv_cache_paged::llama_kv_cache_paged(uint32_t head_dim,
                                           uint32_t n_heads_kv,
                                           uint32_t block_size,
                                           uint32_t n_layers,
                                           uint32_t n_ubatch,
                                           uint32_t n_seq_max) :
    kv_type_k(GGML_TYPE_F16),
    kv_type_v(GGML_TYPE_F16),
    head_dim(head_dim),
    n_heads_kv(n_heads_kv),
    block_size(block_size),
    n_layers(n_layers),
    n_ubatch(n_ubatch),
    n_seq_max(n_seq_max),
    num_gpu_blocks(0),
    num_cpu_blocks(0),
    gpu_backend(nullptr),
    cpu_backend(nullptr) {}

void llama_kv_cache_paged::init(ggml_backend_t backend_gpu,
                                ggml_backend_t backend_cpu,
                                enum ggml_type type_k,
                                enum ggml_type type_v,
                                uint32_t       n_gpu_blocks,
                                uint32_t       n_cpu_blocks,
                                float          watermark) {
    const auto supported_type = [](ggml_type type) {
        return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0;
    };
    if (!supported_type(type_k) || !supported_type(type_v)) {
        throw std::runtime_error(format("paged KV supports q8_0, turbo3_0, and turbo4_0 storage, got K=%s V=%s",
                                        ggml_type_name(type_k), ggml_type_name(type_v)));
    }
    const ggml_type storage_type_k = paged_storage_type(type_k);
    const ggml_type storage_type_v = paged_storage_type(type_v);
    if (storage_type_k != type_k || storage_type_v != type_v) {
        LLAMA_LOG_WARN("%s: paged TurboQuant quality gate failed; selecting q8_0 storage\n", __func__);
        type_k = storage_type_k;
        type_v = storage_type_v;
    }

    if (head_dim > 256 || head_dim % ggml_blck_size(type_k) != 0 || head_dim % ggml_blck_size(type_v) != 0) {
        throw std::runtime_error(format("paged KV head dimension %u is incompatible with K=%s V=%s", head_dim,
                                        ggml_type_name(type_k), ggml_type_name(type_v)));
    }

    GGML_ASSERT(backend_cpu && "backend_cpu is nullptr");
    GGML_ASSERT(backend_gpu && "backend_gpu is nullptr");
    const ggml_backend_dev_t dev = ggml_backend_get_device(backend_gpu);
    if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        LLAMA_LOG_WARN("%s: no GPU device found, allocating KV block pool on CPU. This is valid for testing but it will be slow.\n", __func__);
    }

    GGML_ASSERT(n_gpu_blocks && "n_gpu_blocks need to be greater than 0.");
    GGML_ASSERT(n_cpu_blocks && "n_cpu_blocks need to be greater than 0.");


    LLAMA_LOG_INFO("%s: initializing paged KV cache. n_gpu_blocks=%d, n_cpu_blocks=%d, block_size=%d, watermark=%0.2f\n", __func__,
                   n_gpu_blocks, n_cpu_blocks, block_size, watermark);
    num_gpu_blocks = n_gpu_blocks;
    num_cpu_blocks = n_cpu_blocks;
    kv_type_k      = type_k;
    kv_type_v      = type_v;
    gpu_backend    = backend_gpu;
    cpu_backend    = backend_cpu;
    block_bytes_k  = block_size * n_heads_kv * ggml_row_size(kv_type_k, head_dim);
    block_bytes_v  = block_size * n_heads_kv * ggml_row_size(kv_type_v, head_dim);

    ggml_init_params gpu_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 2 * n_layers,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_gpu = ggml_init(gpu_params);
    for (uint32_t il = 0; il < n_layers; ++il) {
        k_gpu_layers.push_back(ggml_new_tensor_4d(ctx_gpu, type_k, head_dim, block_size, n_heads_kv, n_gpu_blocks));
        v_gpu_layers.push_back(ggml_new_tensor_4d(ctx_gpu, type_v, head_dim, block_size, n_heads_kv, n_gpu_blocks));
    }
    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(ctx_gpu, backend_gpu);
    GGML_ASSERT(buf_gpu && "Failed to allocate GPU KV cache buffer");
    ggml_backend_buffer_clear(buf_gpu, 0);

    ggml_init_params cpu_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 2 * n_layers,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_cpu = ggml_init(cpu_params);
    for (uint32_t il = 0; il < n_layers; ++il) {
        k_cpu_layers.push_back(ggml_new_tensor_4d(ctx_cpu, type_k, head_dim, block_size, n_heads_kv, n_cpu_blocks));
        v_cpu_layers.push_back(ggml_new_tensor_4d(ctx_cpu, type_v, head_dim, block_size, n_heads_kv, n_cpu_blocks));
    }
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    GGML_ASSERT(buf_cpu && "Failed to allocate CPU KV cache buffer");
    ggml_backend_buffer_clear(buf_cpu, 0);

    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);
    initialized = true;
}

bool llama_kv_cache_paged::register_group(llama_sequence_group & group) {
    const auto registered = sequence_groups.find(group.request_id);
    if (registered != sequence_groups.end() && registered->second != &group) {
        throw std::runtime_error("paged KV request id is already registered");
    }

    const auto restored = restored_groups.find(group.request_id);
    if (restored != restored_groups.end()) {
        if (group.n_prompt != restored->second.n_prompt || group.logical_seq.size() < group.n_prompt ||
            !std::equal(group.logical_seq.begin(), group.logical_seq.begin() + group.n_prompt,
                        restored->second.logical_seq.begin())) {
            throw std::runtime_error("paged KV checkpoint prompt does not match restored sequence");
        }
    }

    sequence_groups[group.request_id] = &group;
    if (const auto blocks = sequence_blocks.find(group.request_id); blocks != sequence_blocks.end()) {
        group.block_table = blocks->second;
    }
    if (restored == restored_groups.end()) {
        return false;
    }

    group.status          = restored->second.status;
    group.t_arrival_time  = restored->second.t_arrival_time;
    group.t_first_token_us = restored->second.t_first_token_us;
    group.n_prompt        = restored->second.n_prompt;
    group.n_decoded       = restored->second.n_decoded;
    group.n_past          = restored->second.n_past;
    group.logical_seq     = std::move(restored->second.logical_seq);
    restored_groups.erase(restored);
    return true;
}

bool llama_kv_cache_paged::allocate(int32_t num_tokens, llama_sequence_group & group) {
    register_group(group);
    const uint32_t curr_block_count = group.block_table.size();
    const uint32_t total_num_tokens = group.n_decoded == 0 ? std::max(group.n_prompt, (uint32_t) num_tokens)
                                                           : group.n_decoded + num_tokens;
    uint32_t num_requested_blocks = std::ceil((float) total_num_tokens / block_size) - curr_block_count;
    LLAMA_LOG_DEBUG("%s: curr_block_count=%d, total_num_tokens=%d, num_requested_blocks=%d\n", __func__,
                    curr_block_count, total_num_tokens, num_requested_blocks);

    if (num_requested_blocks == 0) {
        return true;
    }

    if (!block_manager.has_free_gpu_blocks(num_requested_blocks)) {
        LLAMA_LOG_DEBUG("%s: insufficient GPU blocks. Requested: %d.\n", __func__, num_requested_blocks);
        return false;
    }

    llama_block_ids new_ids = block_manager.checkout_gpu_blocks(num_requested_blocks);
    concat_block_ids(group.block_table, new_ids);
    sequence_blocks[group.request_id] = group.block_table;
    LLAMA_LOG_DEBUG("%s: successfully allocated %d.\n", __func__, num_requested_blocks);
    return true;
}

void llama_kv_cache_paged::release_block_ids(const llama_block_ids & block_ids) {
    llama_block_ids blocks_to_free_gpu;
    llama_block_ids blocks_to_free_cpu;

    for (uint32_t block_id : block_ids) {
        if (block_manager.is_gpu(block_id)) {
            blocks_to_free_gpu.push_back(block_id);
        } else {
            blocks_to_free_cpu.push_back(block_id);
        }
    }

    if (!blocks_to_free_gpu.empty()) {
        block_manager.release_gpu_blocks(blocks_to_free_gpu);
    }
    if (!blocks_to_free_cpu.empty()) {
        block_manager.release_cpu_blocks(blocks_to_free_cpu);
    }
}

void llama_kv_cache_paged::free_blocks(llama_sequence_group & group) {
    if (sequence_blocks.count(group.request_id)) {
        seq_rm(group.request_id, -1, -1);
    } else {
        release_block_ids(group.block_table);
        group.block_table.clear();
    }
}

void llama_kv_cache_paged::do_block_copy(const llama_block_ids & src_ids,
                                         const llama_block_ids & new_ids,
                                         bool                    to_gpu) {
    const uint32_t num_blocks = src_ids.size();
    LLAMA_LOG_DEBUG("%s: num_blocks_size=%d, new_ids_size=%ld\n", __func__, num_blocks, new_ids.size());
    GGML_ASSERT(num_blocks == new_ids.size() && "src_ids and new_ids do not have the same size.");

    const auto & src_k_layers = to_gpu ? k_cpu_layers : k_gpu_layers;
    const auto & src_v_layers = to_gpu ? v_cpu_layers : v_gpu_layers;
    const auto & dst_k_layers = to_gpu ? k_gpu_layers : k_cpu_layers;
    const auto & dst_v_layers = to_gpu ? v_gpu_layers : v_cpu_layers;

    GGML_ASSERT(src_k_layers.size() == n_layers && src_v_layers.size() == n_layers && "src layer count mismatch.");
    GGML_ASSERT(dst_k_layers.size() == n_layers && dst_v_layers.size() == n_layers && "dst layer count mismatch.");

    std::vector<uint8_t> staging(std::max(block_bytes_k, block_bytes_v));

    for (uint32_t il = 0; il < n_layers; ++il) {
        for (uint32_t i = 0; i < num_blocks; ++i) {
            const uint32_t src_global = src_ids[i];
            const uint32_t dst_global = new_ids[i];
            const uint32_t src_local  = to_gpu ? src_global - num_gpu_blocks : src_global;
            const uint32_t dst_local  = to_gpu ? dst_global : dst_global - num_gpu_blocks;

            ggml_backend_tensor_get(src_k_layers[il], staging.data(), (size_t) src_local * block_bytes_k, block_bytes_k);
            ggml_backend_tensor_set(dst_k_layers[il], staging.data(), (size_t) dst_local * block_bytes_k, block_bytes_k);
            ggml_backend_tensor_get(src_v_layers[il], staging.data(), (size_t) src_local * block_bytes_v, block_bytes_v);
            ggml_backend_tensor_set(dst_v_layers[il], staging.data(), (size_t) dst_local * block_bytes_v, block_bytes_v);
        }
    }
}

bool llama_kv_cache_paged::swap_in(llama_sequence_group & group) {
    const uint32_t num_blocks = group.block_table.size();
    if (num_blocks == 0) {
        return true;
    }

    const uint32_t required_blocks = std::max(
        num_blocks,
        (uint32_t) std::ceil((float) (group.n_past + 1) / block_size));
    if (!block_manager.has_free_gpu_blocks(required_blocks)) {
        return false;
    }

    llama_block_ids new_ids = block_manager.checkout_gpu_blocks(required_blocks);
    llama_block_ids restored_ids(new_ids.begin(), new_ids.begin() + num_blocks);
    do_block_copy(group.block_table, restored_ids, /*to_gpu=*/true);

    release_block_ids(group.block_table);
    group.block_table = new_ids;
    sequence_blocks[group.request_id] = group.block_table;
    return true;
}

bool llama_kv_cache_paged::swap_out(llama_sequence_group & group) {
    const uint32_t num_blocks = group.block_table.size();
    if (num_blocks == 0) {
        return true;
    }

    if (!block_manager.has_free_cpu_blocks(num_blocks)) {
        return false;
    }

    llama_block_ids new_ids = block_manager.checkout_cpu_blocks(num_blocks);
    do_block_copy(group.block_table, new_ids, /*to_gpu=*/false);

    release_block_ids(group.block_table);
    group.block_table = new_ids;
    sequence_blocks[group.request_id] = group.block_table;
    return true;
}

void llama_kv_cache_paged::set_paged_batch_info(const llama_paged_batch_info * info) {
    last_paged_info = info;
}

uint32_t llama_kv_cache_paged::get_num_gpu_blocks() const {
    return num_gpu_blocks;
}

void llama_kv_cache_paged::concat_block_ids(llama_block_ids &       to_block_table,
                                            const llama_block_ids & from_block_table) {
    to_block_table.insert(to_block_table.end(), from_block_table.begin(), from_block_table.end());
}

// llama_memory_i

llama_memory_context_ptr llama_kv_cache_paged::init_batch(llama_batch_allocr & balloc,
                                                          uint32_t             n_ubatch,
                                                          bool /*embd_all*/) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);
            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch));
        }

        // Failed to find a suitable split
        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        auto ctx = std::make_unique<llama_kv_cache_paged_context>(this, std::move(ubatches));

        // Do not use balloc's internal batch. It does not carry any paged metadata.
        GGML_ASSERT(last_paged_info && "no paged batch info set before init_batch was called.");
        ctx->set_batch_data(*last_paged_info);
        return ctx;
    } while (false);

    return std::make_unique<llama_kv_cache_paged_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache_paged::prepare(const std::vector<llama_ubatch> & ubatches) {
    auto ctx = std::make_unique<llama_kv_cache_paged_context>(this, ubatches);
    GGML_ASSERT(last_paged_info && "no paged batch info set before prepare was called.");
    ctx->set_batch_data(*last_paged_info);
    return ctx;
}

// Used by llama_context scheduler to dry-run
llama_memory_context_ptr llama_kv_cache_paged::init_full() {
    LLAMA_LOG_DEBUG("%s: reserving graph for n_ubatch=%d, n_seq_max=%d, num_gpu_blocks=%d\n", __func__, n_ubatch,
                    n_seq_max, num_gpu_blocks);

    // Create a "dummy" ubatch that represents the maximum capacity
    // of the system to let the scheduler reserve enough space for metadata.
    llama_ubatch ubatch = {};
    ubatch.n_tokens     = n_ubatch;   // maximum tokens
    ubatch.n_seqs       = n_seq_max;  // maximum sequences
    ubatch.n_pos        = 1;

    std::vector<llama_ubatch> ubatches = { ubatch };

    auto ctx = std::make_unique<llama_kv_cache_paged_context>(this, ubatches);

    ctx->set_batch_size(n_seq_max);       // maximum possible sequences
    ctx->set_n_tokens(n_ubatch);          // representative token count
    ctx->set_max_blocks(num_gpu_blocks);  // every block could theoretically belong to one seq

    return ctx;
}

llama_memory_context_ptr llama_kv_cache_paged::init_update(llama_context * /*lctx*/, bool /*optimize*/) {
    std::vector<llama_ubatch> dummy_ubatch = {};
    auto                      ctx          = std::make_unique<llama_kv_cache_paged_context>(this, dummy_ubatch);
    // TODO maybe confirm block counts or clean up stale pointers
    return ctx;
}

struct ggml_tensor * llama_kv_cache_paged::get_k_tensor(int layer_idx) const {
    return k_gpu_layers[layer_idx];
}

struct ggml_tensor * llama_kv_cache_paged::get_v_tensor(int layer_idx) const {
    return v_gpu_layers[layer_idx];
}

void llama_kv_cache_paged::clear(bool data) {
    for (const auto & item : sequence_groups) {
        item.second->block_table.clear();
    }
    if (initialized) {
        block_manager.restore({});
        if (data) {
            for (ggml_tensor * tensor : { k_gpu_layers.front(), v_gpu_layers.front(), k_cpu_layers.front(), v_cpu_layers.front() }) {
                ggml_backend_buffer_clear(tensor->buffer, 0);
            }
        }
    }
    sequence_groups.clear();
    sequence_positions.clear();
    sequence_blocks.clear();
    restored_groups.clear();
}

bool llama_kv_cache_paged::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (p0 >= 0 || p1 >= 0) {
        return false;
    }
    if (seq_id < 0) {
        clear(false);
        return true;
    }
    if (const auto blocks = sequence_blocks.find(seq_id); blocks != sequence_blocks.end()) {
        release_block_ids(blocks->second);
        sequence_blocks.erase(blocks);
    }
    if (const auto group = sequence_groups.find(seq_id); group != sequence_groups.end()) {
        group->second->block_table.clear();
        sequence_groups.erase(group);
    }
    sequence_positions.erase(seq_id);
    restored_groups.erase(seq_id);
    return true;
}

llama_pos llama_kv_cache_paged::seq_pos_min(llama_seq_id seq_id) const {
    auto it = sequence_positions.find(seq_id);
    return (it != sequence_positions.end()) ? it->second.min : -1;
}

llama_pos llama_kv_cache_paged::seq_pos_max(llama_seq_id seq_id) const {
    auto it = sequence_positions.find(seq_id);
    return (it != sequence_positions.end()) ? it->second.max : -1;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_paged::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> breakdown;
    for (uint32_t il = 0; il < n_layers; ++il) {
        for (ggml_tensor * tensor : { k_gpu_layers[il], v_gpu_layers[il], k_cpu_layers[il], v_cpu_layers[il] }) {
            breakdown[ggml_backend_buffer_get_type(tensor->buffer)] += ggml_nbytes(tensor);
        }
    }
    return breakdown;
}

void llama_kv_cache_paged::set_seq_min_pos(llama_seq_id seq_id, llama_pos new_min) {
    sequence_positions[seq_id].min = new_min;
}

void llama_kv_cache_paged::set_seq_max_pos(llama_seq_id seq_id, llama_pos new_max) {
    sequence_positions[seq_id].max = new_max;
}

void llama_kv_cache_paged::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);

    const uint32_t magic = 0x504b5633;
    io.write(&magic, sizeof(magic));
    io.write(&head_dim, sizeof(head_dim));
    io.write(&n_heads_kv, sizeof(n_heads_kv));
    io.write(&block_size, sizeof(block_size));
    io.write(&n_layers, sizeof(n_layers));
    io.write(&num_gpu_blocks, sizeof(num_gpu_blocks));
    io.write(&num_cpu_blocks, sizeof(num_cpu_blocks));
    const int32_t type_k = kv_type_k;
    const int32_t type_v = kv_type_v;
    io.write(&type_k, sizeof(type_k));
    io.write(&type_v, sizeof(type_v));

    std::vector<llama_seq_id> seq_ids;
    if (seq_id == -1) {
        seq_ids.reserve(sequence_positions.size());
        for (const auto & item : sequence_positions) {
            seq_ids.push_back(item.first);
        }
        std::sort(seq_ids.begin(), seq_ids.end());
    } else if (sequence_positions.count(seq_id)) {
        seq_ids.push_back(seq_id);
    } else {
        throw std::runtime_error("paged KV sequence state is empty");
    }
    const uint32_t n_sequences = seq_ids.size();
    io.write(&n_sequences, sizeof(n_sequences));
    std::vector<uint32_t> block_ids;
    for (const llama_seq_id id : seq_ids) {
        const auto & range = sequence_positions.at(id);
        const auto blocks = sequence_blocks.find(id);
        const uint32_t n_blocks = blocks == sequence_blocks.end() ? 0 : blocks->second.size();
        io.write(&id, sizeof(id));
        io.write(&range, sizeof(range));
        io.write(&n_blocks, sizeof(n_blocks));
        if (n_blocks) {
            io.write(blocks->second.data(), n_blocks * sizeof(uint32_t));
            block_ids.insert(block_ids.end(), blocks->second.begin(), blocks->second.end());
        }
        const auto group = sequence_groups.find(id);
        const uint8_t has_group = group != sequence_groups.end() &&
                                  group->second->logical_seq.size() >= group->second->n_prompt;
        io.write(&has_group, sizeof(has_group));
        if (has_group) {
            const auto & saved = *group->second;
            const uint32_t status = (uint32_t) saved.status;
            const uint32_t logical_seq_size = saved.logical_seq.size();
            io.write(&status, sizeof(status));
            io.write(&saved.t_arrival_time, sizeof(saved.t_arrival_time));
            io.write(&saved.t_first_token_us, sizeof(saved.t_first_token_us));
            io.write(&saved.n_prompt, sizeof(saved.n_prompt));
            io.write(&saved.n_decoded, sizeof(saved.n_decoded));
            io.write(&saved.n_past, sizeof(saved.n_past));
            io.write(&logical_seq_size, sizeof(logical_seq_size));
            io.write(saved.logical_seq.data(), logical_seq_size * sizeof(llama_token));
        }
    }

    const uint32_t n_blocks = block_ids.size();
    io.write(&n_blocks, sizeof(n_blocks));
    for (uint32_t block_id : block_ids) {
        io.write(&block_id, sizeof(block_id));
        const bool is_gpu = block_manager.is_gpu(block_id);
        const uint32_t local_id = is_gpu ? block_id : block_id - num_gpu_blocks;
        for (uint32_t il = 0; il < n_layers; ++il) {
            ggml_tensor * k = is_gpu ? k_gpu_layers[il] : k_cpu_layers[il];
            ggml_tensor * v = is_gpu ? v_gpu_layers[il] : v_cpu_layers[il];
            io.write_tensor(k, (size_t) local_id * block_bytes_k, block_bytes_k);
            io.write_tensor(v, (size_t) local_id * block_bytes_v, block_bytes_v);
        }
    }
}

void llama_kv_cache_paged::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    uint32_t magic;
    uint32_t head_dim_ref;
    uint32_t n_heads_kv_ref;
    uint32_t block_size_ref;
    uint32_t n_layers_ref;
    uint32_t n_gpu_blocks_ref;
    uint32_t n_cpu_blocks_ref;
    int32_t type_k_ref;
    int32_t type_v_ref;
    io.read(&magic, sizeof(magic));
    io.read(&head_dim_ref, sizeof(head_dim_ref));
    io.read(&n_heads_kv_ref, sizeof(n_heads_kv_ref));
    io.read(&block_size_ref, sizeof(block_size_ref));
    io.read(&n_layers_ref, sizeof(n_layers_ref));
    io.read(&n_gpu_blocks_ref, sizeof(n_gpu_blocks_ref));
    io.read(&n_cpu_blocks_ref, sizeof(n_cpu_blocks_ref));
    io.read(&type_k_ref, sizeof(type_k_ref));
    io.read(&type_v_ref, sizeof(type_v_ref));
    if (magic != 0x504b5633 || head_dim_ref != head_dim || n_heads_kv_ref != n_heads_kv || block_size_ref != block_size ||
        n_layers_ref != n_layers || n_gpu_blocks_ref != num_gpu_blocks || n_cpu_blocks_ref != num_cpu_blocks ||
        type_k_ref != kv_type_k || type_v_ref != kv_type_v) {
        throw std::runtime_error("incompatible paged KV state");
    }

    uint32_t n_sequences;
    io.read(&n_sequences, sizeof(n_sequences));
    if (seq_id != -1 && n_sequences != 1) {
        throw std::runtime_error("paged KV sequence state contains multiple sequences");
    }
    if (seq_id != -1 && (seq_id < 0 || (uint32_t) seq_id >= n_seq_max)) {
        throw std::runtime_error("invalid paged KV destination sequence id");
    }

    if (seq_id == -1) {
        for (const auto & item : sequence_groups) {
            item.second->block_table.clear();
        }
        sequence_positions.clear();
        sequence_blocks.clear();
        restored_groups.clear();
    }
    std::vector<uint32_t> serialized_blocks;
    llama_block_ids loaded_blocks;
    std::unordered_map<uint32_t, uint32_t> block_remap;
    for (uint32_t i = 0; i < n_sequences; ++i) {
        llama_seq_id id;
        seq_range range;
        uint32_t n_blocks;
        io.read(&id, sizeof(id));
        io.read(&range, sizeof(range));
        io.read(&n_blocks, sizeof(n_blocks));
        if ((seq_id == -1 && (id < 0 || (uint32_t) id >= n_seq_max || sequence_blocks.count(id))) ||
            (seq_id != -1 && id < 0)) {
            throw std::runtime_error("invalid paged KV sequence id");
        }
        const llama_seq_id target_id = seq_id == -1 ? id : seq_id;
        llama_block_ids blocks(n_blocks);
        if (n_blocks) {
            io.read(blocks.data(), n_blocks * sizeof(uint32_t));
            serialized_blocks.insert(serialized_blocks.end(), blocks.begin(), blocks.end());
        }
        if (seq_id == -1) {
            sequence_blocks[target_id] = std::move(blocks);
        } else {
            loaded_blocks = std::move(blocks);
        }
        uint8_t has_group;
        io.read(&has_group, sizeof(has_group));
        if (has_group) {
            uint32_t status;
            uint32_t logical_seq_size;
            restored_group_state restored;
            io.read(&status, sizeof(status));
            io.read(&restored.t_arrival_time, sizeof(restored.t_arrival_time));
            io.read(&restored.t_first_token_us, sizeof(restored.t_first_token_us));
            io.read(&restored.n_prompt, sizeof(restored.n_prompt));
            io.read(&restored.n_decoded, sizeof(restored.n_decoded));
            io.read(&restored.n_past, sizeof(restored.n_past));
            io.read(&logical_seq_size, sizeof(logical_seq_size));
            if (status > (uint32_t) llama_sequence_group_status::FINISHED ||
                restored.n_past > n_blocks * block_size || logical_seq_size < restored.n_prompt) {
                throw std::runtime_error("invalid paged scheduler state");
            }
            restored.status = (llama_sequence_group_status) status;
            restored.logical_seq.resize(logical_seq_size);
            io.read(restored.logical_seq.data(), logical_seq_size * sizeof(llama_token));
            restored_groups[target_id] = std::move(restored);
        }
        sequence_positions[target_id] = range;
    }

    if (seq_id != -1) {
        const auto group = sequence_groups.find(seq_id);
        const auto restored = restored_groups.find(seq_id);
        if (group != sequence_groups.end() && restored != restored_groups.end() &&
            group->second->status != restored->second.status) {
            throw std::runtime_error("paged KV checkpoint status conflicts with active sequence");
        }
    }

    if (seq_id == -1) {
        std::vector<uint32_t> allocated_blocks;
        for (const auto & item : sequence_blocks) {
            allocated_blocks.insert(allocated_blocks.end(), item.second.begin(), item.second.end());
        }
        if (!block_manager.restore(allocated_blocks)) {
            throw std::runtime_error("invalid paged KV block table");
        }
        for (const auto & item : sequence_groups) {
            register_group(*item.second);
        }
    } else {
        std::vector<uint32_t> source_blocks = loaded_blocks;
        std::sort(source_blocks.begin(), source_blocks.end());
        if (std::adjacent_find(source_blocks.begin(), source_blocks.end()) != source_blocks.end() ||
            (source_blocks.empty() ? false : source_blocks.back() >= num_gpu_blocks + num_cpu_blocks)) {
            throw std::runtime_error("invalid paged KV block table");
        }

        if (const auto old = sequence_blocks.find(seq_id); old != sequence_blocks.end()) {
            release_block_ids(old->second);
            sequence_blocks.erase(old);
        }
        if (const auto group = sequence_groups.find(seq_id); group != sequence_groups.end()) {
            group->second->block_table.clear();
        }

        const uint32_t n_gpu = std::count_if(loaded_blocks.begin(), loaded_blocks.end(),
                                             [&](uint32_t id) { return id < num_gpu_blocks; });
        const uint32_t n_cpu = loaded_blocks.size() - n_gpu;
        llama_block_ids gpu_blocks = block_manager.checkout_gpu_blocks(n_gpu);
        llama_block_ids cpu_blocks = block_manager.checkout_cpu_blocks(n_cpu);
        if (gpu_blocks.size() != n_gpu || cpu_blocks.size() != n_cpu) {
            release_block_ids(gpu_blocks);
            release_block_ids(cpu_blocks);
            throw std::runtime_error("insufficient paged KV blocks to restore sequence");
        }
        size_t gpu_index = 0;
        size_t cpu_index = 0;
        llama_block_ids destination_blocks;
        destination_blocks.reserve(loaded_blocks.size());
        for (uint32_t source : loaded_blocks) {
            const uint32_t destination = source < num_gpu_blocks ? gpu_blocks[gpu_index++] : cpu_blocks[cpu_index++];
            block_remap[source] = destination;
            destination_blocks.push_back(destination);
        }
        sequence_blocks[seq_id] = std::move(destination_blocks);
        if (const auto group = sequence_groups.find(seq_id); group != sequence_groups.end()) {
            register_group(*group->second);
        }
    }

    uint32_t n_saved_blocks;
    io.read(&n_saved_blocks, sizeof(n_saved_blocks));
    if (n_saved_blocks != serialized_blocks.size()) {
        throw std::runtime_error("invalid paged KV block data count");
    }
    std::vector<uint32_t> expected_data_blocks = serialized_blocks;
    for (uint32_t i = 0; i < n_saved_blocks; ++i) {
        uint32_t block_id;
        io.read(&block_id, sizeof(block_id));
        const auto expected = std::find(expected_data_blocks.begin(), expected_data_blocks.end(), block_id);
        if (expected == expected_data_blocks.end()) {
            throw std::runtime_error("invalid paged KV block data id");
        }
        expected_data_blocks.erase(expected);
        const uint32_t destination_id = seq_id == -1 ? block_id : block_remap.at(block_id);
        const bool is_gpu = block_manager.is_gpu(destination_id);
        const uint32_t local_id = is_gpu ? destination_id : destination_id - num_gpu_blocks;
        for (uint32_t il = 0; il < n_layers; ++il) {
            ggml_tensor * k = is_gpu ? k_gpu_layers[il] : k_cpu_layers[il];
            ggml_tensor * v = is_gpu ? v_gpu_layers[il] : v_cpu_layers[il];
            io.read_tensor(k, (size_t) local_id * block_bytes_k, block_bytes_k);
            io.read_tensor(v, (size_t) local_id * block_bytes_v, block_bytes_v);
        }
    }

}

// llama_kv_cache_paged_context

void llama_kv_cache_paged_context::set_batch_data(const llama_paged_batch_info & info) {
    paged_write_slots   = info.write_slots;
    paged_block_table   = info.block_table;
    paged_context_lens  = info.context_lens;
    paged_batch_offsets = info.batch_offsets;
    paged_batch_lens    = info.batch_lens;
    n_tokens = info.n_tokens;
    paged_write_rows.resize((size_t) n_tokens * manager->n_heads_kv);
    for (int32_t token = 0; token < n_tokens; ++token) {
        const int32_t slot = paged_write_slots[token];
        const int32_t block = slot / (int32_t) manager->block_size;
        const int32_t token_in_block = slot % (int32_t) manager->block_size;
        for (uint32_t head = 0; head < manager->n_heads_kv; ++head) {
            paged_write_rows[(size_t) token * manager->n_heads_kv + head] =
                block * (int32_t) (manager->block_size * manager->n_heads_kv) +
                (int32_t) head * manager->block_size + token_in_block;
        }
    }

    max_blocks          = info.n_blocks_per_seq;
    batch_size          = info.n_seq;
}

bool llama_kv_cache_paged_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    if (++i_cur >= ubatches.size()) {
        return false;
    }
    return true;
}

bool llama_kv_cache_paged_context::apply() {
    // Nothing to do for paged KV cache, return true to allow for execution
    return true;
}

const llama_ubatch & llama_kv_cache_paged_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_cur];
}

struct ggml_tensor * llama_kv_cache_paged_context::get_k(int layer_idx) const {
    GGML_ASSERT(manager && "manager has not been initialized.");
    return manager->get_k_tensor(layer_idx);
}

struct ggml_tensor * llama_kv_cache_paged_context::get_v(int layer_idx) const {
    GGML_ASSERT(manager && "manager has not been initialized.");
    return manager->get_v_tensor(layer_idx);
}

int32_t llama_kv_cache_paged_context::get_n_tokens() const {
    return n_tokens;
}

int32_t llama_kv_cache_paged_context::get_batch_size() const {
    return batch_size;
}

int32_t llama_kv_cache_paged_context::get_max_blocks() const {
    return max_blocks;
}

int32_t * llama_kv_cache_paged_context::get_write_slots() const {
    return paged_write_slots;
}

const int32_t * llama_kv_cache_paged_context::get_write_rows() const {
    return paged_write_rows.data();
}


int32_t * llama_kv_cache_paged_context::get_block_table() const {
    return paged_block_table;
}

int32_t * llama_kv_cache_paged_context::get_context_lens() const {
    return paged_context_lens;
}

int32_t * llama_kv_cache_paged_context::get_batch_offsets() const {
    return paged_batch_offsets;
}

int32_t * llama_kv_cache_paged_context::get_batch_lens() const {
    return paged_batch_lens;
}

void llama_kv_cache_paged_context::set_n_tokens(int32_t new_n_tokens) {
    n_tokens = new_n_tokens;
}

void llama_kv_cache_paged_context::set_batch_size(int32_t new_batch_size) {
    batch_size = new_batch_size;
}

void llama_kv_cache_paged_context::set_max_blocks(int32_t new_max_blocks) {
    max_blocks = new_max_blocks;
}
