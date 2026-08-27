#!/usr/bin/env bash
set -u

root="$(cd "$(dirname "$0")/../.." && pwd)"
out="$root/benchmark-results/paged-prefill-head256-20260827-v2"
raw="$out/raw/matrix"
mkdir -p "$raw"

qwen_model="/models/qwen38/Qwen3.8-27B-UD-Q4_K_S.gguf"
tiel_model="/models/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S.gguf"
prompt_file="$root/reports/paged-4096-prompt.txt"
binary="$root/build-verify-cuda/bin/llama-paged"

printf '%s\n' 'model,prompt_tokens,block_size,ubatch_size,requests,repetition,warmup,pp_tok_s,tg_tok_s,status,raw_log' > "$out/paged-raw.csv"
printf '%s\n' 'model,prompt_tokens,block_size,ubatch_size,requests,measured_repetitions,median_pp_tok_s,median_tg_tok_s,status' > "$out/paged-medians.csv"
: > "$out/commands.txt"

run_one() {
    local model_name="$1"
    local model_path="$2"
    local prompt_tokens="$3"
    local block_size="$4"
    local ubatch="$5"
    local requests="$6"
    local per_request_tokens=$((prompt_tokens / requests))
    local prompt_bytes=$((per_request_tokens * 2))
    local context=512
    local gpu_blocks=$(((prompt_tokens + block_size - 1) / block_size + 16))
    local prompt
    local tag
    local log
    local rc
    local i
    local status
    local pp
    local tg
    local median_pp
    local median_tg
    local -a pp_values=()
    local -a tg_values=()

    if ((prompt_tokens > 256)); then
        context=2048
    fi
    if ((prompt_tokens > 1024)); then
        context=8192
    fi

    prompt="$(head -c "$prompt_bytes" "$prompt_file")"
    tag="${model_name}-p${prompt_tokens}-bs${block_size}-ub${ubatch}-r${requests}"
    log="$raw/$tag.log"

    local -a cmd=(
        "$binary"
        -m "$model_path"
        -dev cuda0
        -ngl 99
        -sm none
        -mg 0
        -c "$context"
        -b "$ubatch"
        -ub "$ubatch"
        -n 8
        -p "$prompt"
        -kvp
        -ctk q8_0
        -ctv q8_0
        -ngpub "$gpu_blocks"
        -ncpub 16
        -kvbls "$block_size"
        --kv-paged-watermark 0
        -ns "$requests"
        -np "$requests"
        --timing
        --perf
        --temp 0
        -s 1234
    )

    printf 'LLAMA_PAGED_REPETITIONS=4 TMPDIR=%q ' "$root/build-verify-cuda/tmp" >> "$out/commands.txt"
    printf '%q ' "${cmd[@]}" >> "$out/commands.txt"
    printf '> %q 2>&1\n' "$log" >> "$out/commands.txt"

    printf 'matrix: %s\n' "$tag"
    LLAMA_PAGED_REPETITIONS=4 TMPDIR="$root/build-verify-cuda/tmp" timeout 300 "${cmd[@]}" > "$log" 2>&1
    rc=$?
    status="ok"
    if ((rc != 0)); then
        status="exit_$rc"
    fi

    mapfile -t pp_values < <(awk '/pp t\/s/{print $NF}' "$log")
    mapfile -t tg_values < <(awk '/tg t\/s/{print $NF}' "$log")
    if ((${#pp_values[@]} != 4 || ${#tg_values[@]} != 4)); then
        status="${status}_incomplete_${#pp_values[@]}pp_${#tg_values[@]}tg"
    fi

    for i in 0 1 2 3; do
        pp="${pp_values[$i]:-}"
        tg="${tg_values[$i]:-}"
        printf '%s,%d,%d,%d,%d,%d,%s,%s,%s,%s,%s\n' \
            "$model_name" "$prompt_tokens" "$block_size" "$ubatch" "$requests" "$((i + 1))" \
            "$([[ $i == 0 ]] && printf true || printf false)" "$pp" "$tg" "$status" "raw/matrix/$tag.log" \
            >> "$out/paged-raw.csv"
    done

    median_pp=""
    median_tg=""
    if ((${#pp_values[@]} == 4)); then
        median_pp="$(printf '%s\n' "${pp_values[@]:1:3}" | sort -n | sed -n '2p')"
    fi
    if ((${#tg_values[@]} == 4)); then
        median_tg="$(printf '%s\n' "${tg_values[@]:1:3}" | sort -n | sed -n '2p')"
    fi
    printf '%s,%d,%d,%d,%d,3,%s,%s,%s\n' \
        "$model_name" "$prompt_tokens" "$block_size" "$ubatch" "$requests" \
        "$median_pp" "$median_tg" "$status" >> "$out/paged-medians.csv"
}

for model_name in qwen tiel; do
    model_path="$qwen_model"
    if [[ "$model_name" == tiel ]]; then
        model_path="$tiel_model"
    fi

    for prompt_tokens in 64 256 1024 4096; do
        ubatches=(256)
        if ((prompt_tokens >= 1024)); then
            ubatches+=(512 1024)
        fi
        if ((prompt_tokens >= 4096)); then
            ubatches+=(2048 4096)
        fi
        for block_size in 16 32 64; do
            for ubatch in "${ubatches[@]}"; do
                for requests in 1 2; do
                    run_one "$model_name" "$model_path" "$prompt_tokens" "$block_size" "$ubatch" "$requests"
                done
            done
        done
    done
done

