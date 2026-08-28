#include "llama-kv-cache-paged.h"

#include "llama-impl.h"
#include "llama-io.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

static ggml_type paged_storage_type(ggml_type type) {
    return type == GGML_TYPE_F16 ? GGML_TYPE_Q8_0 : type;
}


// llama_kv_cache_paged
//

llama_kv_cache_paged::llama_kv_cache_paged(uint32_t head_dim,
                                           uint32_t n_heads_kv,
                                           uint32_t block_size,
                                           uint32_t n_layers,
                                           uint32_t n_ubatch,
                                           uint32_t n_seq_max) :
    kv_type_k(GGML_TYPE_Q8_0),
    kv_type_v(GGML_TYPE_Q8_0),
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
        return type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0;
    };
    if (!supported_type(type_k) || !supported_type(type_v)) {
        throw std::runtime_error(format("paged KV supports q8_0, turbo3_0, and turbo4_0 storage, got K=%s V=%s",
                                        ggml_type_name(type_k), ggml_type_name(type_v)));
    }
    const ggml_type storage_type_k = paged_storage_type(type_k);
    const ggml_type storage_type_v = paged_storage_type(type_v);
    if (storage_type_k != type_k || storage_type_v != type_v) {
        LLAMA_LOG_WARN("%s: paged KV type %s/%s selects q8_0 production storage\n", __func__,
                       ggml_type_name(type_k), ggml_type_name(type_v));
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


    LLAMA_LOG_INFO("%s: initializing paged KV cache. n_gpu_blocks=%d, n_cpu_blocks=%d, block_size=%d, K=%s, V=%s, watermark=%0.2f\n", __func__,
                   n_gpu_blocks, n_cpu_blocks, block_size, ggml_type_name(type_k), ggml_type_name(type_v), watermark);
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
        seq_ids.reserve(sequence_positions.size() + sequence_blocks.size() + sequence_groups.size() + restored_groups.size());
        const auto append_ids = [&](const auto & sequences) {
            for (const auto & item : sequences) {
                seq_ids.push_back(item.first);
            }
        };
        append_ids(sequence_positions);
        append_ids(sequence_blocks);
        append_ids(sequence_groups);
        append_ids(restored_groups);
        std::sort(seq_ids.begin(), seq_ids.end());
        seq_ids.erase(std::unique(seq_ids.begin(), seq_ids.end()), seq_ids.end());
    } else if (sequence_positions.count(seq_id) || sequence_blocks.count(seq_id) || sequence_groups.count(seq_id) || restored_groups.count(seq_id)) {
        seq_ids.push_back(seq_id);
    } else {
        throw std::runtime_error("paged KV sequence state is empty");
    }
    const uint32_t n_sequences = seq_ids.size();
    io.write(&n_sequences, sizeof(n_sequences));
    std::vector<uint32_t> block_ids;
    for (const llama_seq_id id : seq_ids) {
        const auto position = sequence_positions.find(id);
        const seq_range range = position == sequence_positions.end() ? seq_range{} : position->second;
        const auto blocks = sequence_blocks.find(id);
        const uint32_t n_blocks = blocks == sequence_blocks.end() ? 0 : blocks->second.size();
        io.write(&id, sizeof(id));
        io.write(&range, sizeof(range));
        io.write(&n_blocks, sizeof(n_blocks));
        if (n_blocks) {
            io.write(blocks->second.data(), n_blocks * sizeof(uint32_t));
            block_ids.insert(block_ids.end(), blocks->second.begin(), blocks->second.end());
        }
        const auto active_group = sequence_groups.find(id);
        const auto restored_group = restored_groups.find(id);
        const bool has_active_group = active_group != sequence_groups.end() &&
                                      active_group->second->logical_seq.size() >= active_group->second->n_prompt;
        const bool has_restored_group = restored_group != restored_groups.end() &&
                                        restored_group->second.logical_seq.size() >= restored_group->second.n_prompt;
        const uint8_t has_group = has_active_group || has_restored_group;
        io.write(&has_group, sizeof(has_group));
        if (has_group) {
            const auto status = has_active_group ? (uint32_t) active_group->second->status : (uint32_t) restored_group->second.status;
            const auto t_arrival_time = has_active_group ? active_group->second->t_arrival_time : restored_group->second.t_arrival_time;
            const auto t_first_token_us = has_active_group ? active_group->second->t_first_token_us : restored_group->second.t_first_token_us;
            const auto n_prompt = has_active_group ? active_group->second->n_prompt : restored_group->second.n_prompt;
            const auto n_decoded = has_active_group ? active_group->second->n_decoded : restored_group->second.n_decoded;
            const auto n_past = has_active_group ? active_group->second->n_past : restored_group->second.n_past;
            const auto & logical_seq = has_active_group ? active_group->second->logical_seq : restored_group->second.logical_seq;
            const uint32_t logical_seq_size = logical_seq.size();
            io.write(&status, sizeof(status));
            io.write(&t_arrival_time, sizeof(t_arrival_time));
            io.write(&t_first_token_us, sizeof(t_first_token_us));
            io.write(&n_prompt, sizeof(n_prompt));
            io.write(&n_decoded, sizeof(n_decoded));
            io.write(&n_past, sizeof(n_past));
            io.write(&logical_seq_size, sizeof(logical_seq_size));
            io.write(logical_seq.data(), logical_seq_size * sizeof(llama_token));
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

    struct saved_sequence {
        llama_seq_id          id;
        seq_range             range;
        llama_block_ids       blocks;
        bool                  has_group = false;
        restored_group_state  group;
    };
    struct saved_block {
        uint32_t                              id;
        std::vector<std::vector<uint8_t>>     k;
        std::vector<std::vector<uint8_t>>     v;
    };

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
    if (n_sequences > n_seq_max || (seq_id != -1 && n_sequences != 1)) {
        throw std::runtime_error("invalid paged KV sequence count");
    }
    if (seq_id != -1 && (seq_id < 0 || (uint32_t) seq_id >= n_seq_max)) {
        throw std::runtime_error("invalid paged KV destination sequence id");
    }

    const uint32_t n_total_blocks = num_gpu_blocks + num_cpu_blocks;
    std::vector<saved_sequence> sequences;
    std::vector<uint32_t> serialized_blocks;
    std::unordered_set<llama_seq_id> saved_sequence_ids;
    sequences.reserve(n_sequences);
    for (uint32_t i = 0; i < n_sequences; ++i) {
        saved_sequence saved;
        uint32_t n_blocks;
        io.read(&saved.id, sizeof(saved.id));
        io.read(&saved.range, sizeof(saved.range));
        io.read(&n_blocks, sizeof(n_blocks));
        if (saved.id < 0 || (seq_id == -1 && (uint32_t) saved.id >= n_seq_max) || n_blocks > n_total_blocks) {
            throw std::runtime_error("invalid paged KV sequence id or block count");
        }
        if (!saved_sequence_ids.insert(saved.id).second) {
            throw std::runtime_error("duplicate paged KV sequence id");
        }
        saved.blocks.resize(n_blocks);
        if (n_blocks) {
            io.read(saved.blocks.data(), n_blocks * sizeof(uint32_t));
            serialized_blocks.insert(serialized_blocks.end(), saved.blocks.begin(), saved.blocks.end());
        }
        uint8_t has_group;
        io.read(&has_group, sizeof(has_group));
        if (has_group > 1) {
            throw std::runtime_error("invalid paged scheduler state");
        }
        saved.has_group = has_group;
        if (saved.has_group) {
            uint32_t status;
            uint32_t logical_seq_size;
            io.read(&status, sizeof(status));
            io.read(&saved.group.t_arrival_time, sizeof(saved.group.t_arrival_time));
            io.read(&saved.group.t_first_token_us, sizeof(saved.group.t_first_token_us));
            io.read(&saved.group.n_prompt, sizeof(saved.group.n_prompt));
            io.read(&saved.group.n_decoded, sizeof(saved.group.n_decoded));
            io.read(&saved.group.n_past, sizeof(saved.group.n_past));
            io.read(&logical_seq_size, sizeof(logical_seq_size));
            if (status > (uint32_t) llama_sequence_group_status::FINISHED || saved.group.n_past > n_blocks * block_size ||
                logical_seq_size < saved.group.n_prompt ||
                (n_blocks != 0 && logical_seq_size > n_blocks * block_size + 1)) {
                throw std::runtime_error("invalid paged scheduler state");
            }
            saved.group.status = (llama_sequence_group_status) status;
            saved.group.logical_seq.resize(logical_seq_size);
            io.read(saved.group.logical_seq.data(), logical_seq_size * sizeof(llama_token));
        }
        sequences.push_back(std::move(saved));
    }

    std::sort(serialized_blocks.begin(), serialized_blocks.end());
    if (serialized_blocks.size() > n_total_blocks || (!serialized_blocks.empty() && serialized_blocks.back() >= n_total_blocks) ||
        std::adjacent_find(serialized_blocks.begin(), serialized_blocks.end()) != serialized_blocks.end()) {
        throw std::runtime_error("invalid paged KV block table");
    }

    uint32_t n_saved_blocks;
    io.read(&n_saved_blocks, sizeof(n_saved_blocks));
    if (n_saved_blocks != serialized_blocks.size()) {
        throw std::runtime_error("invalid paged KV block data count");
    }
    std::vector<saved_block> data;
    data.reserve(n_saved_blocks);
    std::vector<uint32_t> expected_blocks = serialized_blocks;
    for (uint32_t i = 0; i < n_saved_blocks; ++i) {
        saved_block saved;
        io.read(&saved.id, sizeof(saved.id));
        const auto expected = std::lower_bound(expected_blocks.begin(), expected_blocks.end(), saved.id);
        if (expected == expected_blocks.end() || *expected != saved.id) {
            throw std::runtime_error("invalid paged KV block data id");
        }
        expected_blocks.erase(expected);
        saved.k.resize(n_layers);
        saved.v.resize(n_layers);
        for (uint32_t il = 0; il < n_layers; ++il) {
            saved.k[il].resize(block_bytes_k);
            saved.v[il].resize(block_bytes_v);
            io.read(saved.k[il].data(), block_bytes_k);
            io.read(saved.v[il].data(), block_bytes_v);
        }
        data.push_back(std::move(saved));
    }

    const auto validate_group = [&](llama_seq_id id, const saved_sequence & saved) {
        if (!saved.has_group) {
            return;
        }
        const auto group = sequence_groups.find(id);
        if (group != sequence_groups.end() &&
            (group->second->status != saved.group.status || group->second->n_prompt != saved.group.n_prompt ||
             group->second->logical_seq.size() < saved.group.n_prompt ||
             !std::equal(group->second->logical_seq.begin(), group->second->logical_seq.begin() + saved.group.n_prompt,
                         saved.group.logical_seq.begin()))) {
            throw std::runtime_error("paged KV checkpoint conflicts with active sequence");
        }
    };
    for (const auto & saved : sequences) {
        validate_group(seq_id == -1 ? saved.id : seq_id, saved);
    }
    if (seq_id == -1) {
        for (const auto & item : sequence_groups) {
            const auto saved = std::find_if(sequences.begin(), sequences.end(), [&](const saved_sequence & sequence) {
                return sequence.id == item.first && sequence.has_group;
            });
            if (saved == sequences.end()) {
                throw std::runtime_error("paged KV checkpoint omits active sequence");
            }
        }
    }

    std::unordered_map<uint32_t, uint32_t> block_remap;
    if (seq_id == -1) {
        if (!block_manager.restore(serialized_blocks)) {
            throw std::runtime_error("invalid paged KV block table");
        }
        for (const auto & item : sequence_groups) {
            item.second->block_table.clear();
        }
        sequence_positions.clear();
        sequence_blocks.clear();
        restored_groups.clear();
        for (const auto & saved : sequences) {
            sequence_positions[saved.id] = saved.range;
            sequence_blocks[saved.id] = saved.blocks;
            if (saved.has_group) {
                restored_groups[saved.id] = saved.group;
            }
        }
        for (const auto & item : sequence_groups) {
            register_group(*item.second);
        }
    } else {
        const saved_sequence & saved = sequences.front();
        const auto old = sequence_blocks.find(seq_id);
        const llama_block_ids old_blocks = old == sequence_blocks.end() ? llama_block_ids{} : old->second;
        const uint32_t old_gpu = std::count_if(old_blocks.begin(), old_blocks.end(), [&](uint32_t id) { return id < num_gpu_blocks; });
        const uint32_t old_cpu = old_blocks.size() - old_gpu;
        const uint32_t new_gpu = std::count_if(saved.blocks.begin(), saved.blocks.end(), [&](uint32_t id) { return id < num_gpu_blocks; });
        const uint32_t new_cpu = saved.blocks.size() - new_gpu;
        if (block_manager.n_free_gpu_blocks() + old_gpu < new_gpu || block_manager.n_free_cpu_blocks() + old_cpu < new_cpu) {
            throw std::runtime_error("insufficient paged KV blocks to restore sequence");
        }
        release_block_ids(old_blocks);
        if (old != sequence_blocks.end()) {
            sequence_blocks.erase(old);
        }
        const llama_block_ids gpu_blocks = block_manager.checkout_gpu_blocks(new_gpu);
        const llama_block_ids cpu_blocks = block_manager.checkout_cpu_blocks(new_cpu);
        size_t gpu_index = 0;
        size_t cpu_index = 0;
        llama_block_ids destination_blocks;
        destination_blocks.reserve(saved.blocks.size());
        for (uint32_t source : saved.blocks) {
            const uint32_t destination = source < num_gpu_blocks ? gpu_blocks[gpu_index++] : cpu_blocks[cpu_index++];
            block_remap[source] = destination;
            destination_blocks.push_back(destination);
        }
        sequence_blocks[seq_id] = std::move(destination_blocks);
        sequence_positions[seq_id] = saved.range;
        restored_groups.erase(seq_id);
        if (saved.has_group) {
            restored_groups[seq_id] = saved.group;
        }
        if (const auto group = sequence_groups.find(seq_id); group != sequence_groups.end()) {
            group->second->block_table.clear();
            register_group(*group->second);
        }
    }

    for (const auto & saved : data) {
        const uint32_t destination_id = seq_id == -1 ? saved.id : block_remap.at(saved.id);
        const bool is_gpu = block_manager.is_gpu(destination_id);
        const uint32_t local_id = is_gpu ? destination_id : destination_id - num_gpu_blocks;
        for (uint32_t il = 0; il < n_layers; ++il) {
            ggml_tensor * k = is_gpu ? k_gpu_layers[il] : k_cpu_layers[il];
            ggml_tensor * v = is_gpu ? v_gpu_layers[il] : v_cpu_layers[il];
            ggml_backend_tensor_set(k, saved.k[il].data(), (size_t) local_id * block_bytes_k, block_bytes_k);
            ggml_backend_tensor_set(v, saved.v[il].data(), (size_t) local_id * block_bytes_v, block_bytes_v);
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
