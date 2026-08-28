#!/usr/bin/env bash
set -u

root="$(cd "$(dirname "$0")/../.." && pwd)"
out="$root/artifacts/native-kv-comparison-20260828"
raw="$out/raw"
prompt_file="$root/reports/paged-4096-prompt.txt"
bench="$root/build-verify-cuda/bin/llama-bench"
paged="$root/build-verify-cuda/bin/llama-paged"

mkdir -p "$raw"
printf '%s\n' 'model,kv_type,path,fa,prompt_tokens,decode_tokens,pp_tok_s,tg_tok_s,status,raw_log' > "$out/results.csv"

median_last_three() {
    tail -n 3 | sort -n | sed -n '2p'
}

run_unified() {
    local model_name="$1"
    local model_path="$2"
    local kv_type="$3"
    local pp_log="$raw/${model_name}-${kv_type}-kvu-fa-on-pp.json"
    local pp_err="$raw/${model_name}-${kv_type}-kvu-fa-on-pp.stderr.log"
    local tg_log="$raw/${model_name}-${kv_type}-kvu-fa-on-tg.json"
    local tg_err="$raw/${model_name}-${kv_type}-kvu-fa-on-tg.stderr.log"
    local rc
    local pp
    local tg

    echo "benchmark: $model_name $kv_type kvu FA=on"
    TMPDIR="$root/build-verify-cuda/tmp" timeout 600 "$bench" \
        -m "$model_path" -dev cuda0 -ngl 99 -sm none -mg 0 \
        -p 1024 -n 0 -b 1024 -ub 1024 \
        -ctk "$kv_type" -ctv "$kv_type" -fa on -r 12 -o json > "$pp_log" 2> "$pp_err"
    rc=$?
    if ((rc != 0)); then
        printf '%s,%s,kvu,on,1024,8,,,exit_%d,raw/%s\n' \
            "$model_name" "$kv_type" "$rc" "$(basename "$pp_err")" >> "$out/results.csv"
        return
    fi

    TMPDIR="$root/build-verify-cuda/tmp" timeout 600 "$bench" \
        -m "$model_path" -dev cuda0 -ngl 99 -sm none -mg 0 \
        -p 0 -n 8 -d 1024 -b 1024 -ub 1024 \
        -ctk "$kv_type" -ctv "$kv_type" -fa on -r 4 -o json > "$tg_log" 2> "$tg_err"
    rc=$?
    if ((rc != 0)); then
        printf '%s,%s,kvu,on,1024,8,,,exit_%d,raw/%s\n' \
            "$model_name" "$kv_type" "$rc" "$(basename "$tg_err")" >> "$out/results.csv"
        return
    fi

    pp="$(jq -r '.[] | select(.n_prompt == 1024 and .n_gen == 0 and .n_depth == 0) | .samples_ts[9:] | sort | .[1]' "$pp_log")"
    tg="$(jq -r '.[] | select(.n_prompt == 0 and .n_gen == 8 and .n_depth == 1024) | .samples_ts[1:] | sort | .[1]' "$tg_log")"
    printf '%s,%s,kvu,on,1024,8,%s,%s,ok,raw/%s\n' \
        "$model_name" "$kv_type" "$pp" "$tg" "$(basename "$pp_log")" >> "$out/results.csv"
}

run_paged() {
    local model_name="$1"
    local model_path="$2"
    local kv_type="$3"
    local log="$raw/${model_name}-${kv_type}-kvp-fa-on.log"
    local rc
    local prompt
    local pp
    local tg

    prompt="$(head -c 2048 "$prompt_file")"
    echo "benchmark: $model_name $kv_type kvp FA=on"
    LLAMA_PAGED_REPETITIONS=4 TMPDIR="$root/build-verify-cuda/tmp" timeout 600 "$paged" \
        -m "$model_path" -dev cuda0 -ngl 99 -sm none -mg 0 \
        -c 2048 -b 1024 -ub 1024 -n 8 -p "$prompt" \
        -kvp -ctk "$kv_type" -ctv "$kv_type" -fa on \
        -ngpub 48 -ncpub 16 -kvbls 32 --kv-paged-watermark 0 \
        -ns 1 -np 1 --timing --perf --temp 0 -s 1234 > "$log" 2>&1
    rc=$?
    if ((rc != 0)); then
        printf '%s,%s,kvp,on,1024,8,,,exit_%d,raw/%s\n' \
            "$model_name" "$kv_type" "$rc" "$(basename "$log")" >> "$out/results.csv"
        return
    fi

    if [[ "$(rg -c 'pp t/s' "$log")" != 4 || "$(rg -c 'tg t/s' "$log")" != 4 ]]; then
        printf '%s,%s,kvp,on,1024,8,,,incomplete_repetitions,raw/%s\n' \
            "$model_name" "$kv_type" "$(basename "$log")" >> "$out/results.csv"
        return
    fi
    pp="$(awk '/pp t\/s/{print $NF}' "$log" | median_last_three)"
    tg="$(awk '/tg t\/s/{print $NF}' "$log" | median_last_three)"
    printf '%s,%s,kvp,on,1024,8,%s,%s,ok,raw/%s\n' \
        "$model_name" "$kv_type" "$pp" "$tg" "$(basename "$log")" >> "$out/results.csv"
}

qwen_model="/models/qwen38/Qwen3.8-27B-UD-Q4_K_S.gguf"
tiel_model="/models/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S.gguf"

for model_name in qwen38-27b tiel-35b-a3b; do
    model_path="$qwen_model"
    if [[ "$model_name" == tiel-35b-a3b ]]; then
        model_path="$tiel_model"
    fi
    for kv_type in q8_0 turbo3 turbo4; do
        run_unified "$model_name" "$model_path" "$kv_type"
        run_paged "$model_name" "$model_path" "$kv_type"
        printf '%s,%s,kvu,off,1024,8,,,unsupported_quantized_v_requires_fa,\n' "$model_name" "$kv_type" >> "$out/results.csv"
        printf '%s,%s,kvp,off,1024,8,,,unsupported_quantized_v_requires_fa,\n' "$model_name" "$kv_type" >> "$out/results.csv"
    done
done
