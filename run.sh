#!/usr/bin/env bash
set -euo pipefail

# DSpark test: draft-dspark, block size 7 (matches RadixArk/Qwen3.8-27B-DSpark)
# A/B: DRAFT_GGUF selects the drafter conversion (erlidev = script default)
# if OOM: lower CTX (e.g. 65536) or CACHE_PROFILE=q8turbo3 (smaller KV)
#DRAFT_GGUF=/models/DSpark/Qwen3.8-27B-DSpark-Q8_0-magnitude.gguf \
CACHE_PROFILE=turbo3 SPEC=mtp QWEN38_UPSTREAM_PORT=8881 NP=1 DRAFT_MAX=3 CTX=128000 \
SLOT_PROMPT_CACHE_THRESHOLD=0.75 \
docker compose -f docker-compose.qwen38.yml \
  --profile upstream up -d --build upstream-candidate

# previous MTP baseline (for A/B):
# CACHE_PROFILE=turbo4 SPEC=mtp QWEN38_UPSTREAM_PORT=8881 NP=1 DRAFT_MAX=3 CTX=128000 \
# SLOT_PROMPT_CACHE_THRESHOLD=0.75 \
# docker compose -f docker-compose.qwen38.yml \
#   --profile upstream up -d --build upstream-candidate

docker logs -f qwen38-evaluation-upstream-candidate-1