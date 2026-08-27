#!/usr/bin/env bash
set -u

root="$(cd "$(dirname "$0")/../.." && pwd)"
out="$root/benchmark-results/paged-prefill-head256-20260827-v2"
raw="$out/raw/unified"
mkdir -p "$raw"

for model_name in qwen tiel; do
    model_path="/models/qwen38/Qwen3.8-27B-UD-Q4_K_S.gguf"
    if [[ "$model_name" == tiel ]]; then
        model_path="/models/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S.gguf"
    fi

    for ubatch in 256 512 1024 2048 4096; do
        log="$raw/${model_name}-ub${ubatch}.json"
        err="$raw/${model_name}-ub${ubatch}.stderr.log"
        printf 'unified: %s ubatch=%d\n' "$model_name" "$ubatch"
        TMPDIR="$root/build-verify-cuda/tmp" timeout 600 "$root/build-verify-cuda/bin/llama-bench" \
            -m "$model_path" -dev cuda0 -ngl 99 -sm none -mg 0 \
            -p 64,256,1024,4096 -n 8 -b "$ubatch" -ub "$ubatch" \
            -ctk q8_0 -ctv q8_0 -r 3 -o json > "$log" 2> "$err"
        printf 'exit=%d\n' "$?" >> "$err"
    done
done

