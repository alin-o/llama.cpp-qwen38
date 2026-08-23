#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Q4_K_S + embedded MTP head loader
#
# The unsloth Qwen3.8-27B-UD-Q4_K_S.gguf EMBEDS its MTP head (866 tensors
# incl. blk.64.*), so run-upstream-server.sh does NOT pass --spec-draft-model;
# draft-mtp runs against the head inside the trunk (same path as Q4_K_XL).
# Same upstream-candidate service: stop the other config first, then run this.
# Q4_K_S is ~2 GB smaller than Q4_K_XL, so CTX=128000 has comfortable headroom.
# ============================================================================
 CACHE_PROFILE=turbo4 SPEC=mtp QWEN38_UPSTREAM_PORT=8881 NP=1 DRAFT_MAX=3 CTX=128000 \
 MODEL_GGUF=/qwen38-cache/Qwen3.8-27B-UD-Q4_K_S.gguf \
 SLOT_PROMPT_CACHE_THRESHOLD=0.75 \
 docker compose -f docker-compose.qwen38.yml \
   --profile upstream up -d --build upstream-candidate

docker logs -f qwen38-evaluation-upstream-candidate-1
