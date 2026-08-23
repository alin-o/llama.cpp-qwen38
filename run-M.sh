#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Q4_K_M + separate MTP head loader (alternative to the Q4_K_XL run above)
#
# The ggml-org Qwen3.8-27B-Q4_K_M.gguf trunk has NO embedded MTP head (0
# blk.64.* tensors), so run-upstream-server.sh auto-loads the separate head via
# --spec-draft-model (it picks /qwen38-cache/mtp-Qwen3.8-27B-Q4_0.gguf).
# Same upstream-candidate service: stop the Q4_K_XL config first, then run this.
# CTX=128000 OOMs: the trunk is ~1.4 GB bigger than Q4_K_XL and the MTP compute
# buffer (~1.3 GiB) no longer fits in 24 G. CTX=65536 frees ~2 GiB of KV.
# If still tight: CACHE_PROFILE=q8turbo3 (smallest KV) or CTX=49152.
# ============================================================================
 CACHE_PROFILE=turbo4 SPEC=mtp QWEN38_UPSTREAM_PORT=8881 NP=1 DRAFT_MAX=3 CTX=65536 \
 MODEL_GGUF=/qwen38-cache/Qwen3.8-27B-Q4_K_M.gguf \
 SLOT_PROMPT_CACHE_THRESHOLD=0.75 \
 docker compose -f docker-compose.qwen38.yml \
   --profile upstream up -d --build upstream-candidate

docker logs -f qwen38-evaluation-upstream-candidate-1
