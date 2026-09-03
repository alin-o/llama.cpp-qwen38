#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

struct llama_checkpoint_payload {
    std::vector<uint8_t> recurrent;
    std::vector<uint8_t> draft;
    std::vector<uint8_t> speculative;
    bool recurrent_complete   = true;
    bool draft_complete       = true;
    bool speculative_complete = true;
};

bool llama_paged_scheduler_publish_checkpoint_move(
        llama_paged_scheduler * sched,
        int32_t request_id,
        uint32_t n_tokens,
        const std::string & fingerprint,
        llama_checkpoint_payload payload);
