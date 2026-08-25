#!/usr/bin/env bash
# speed-bench comparison of drafters: off vs mtp vs dspark, all on turbo4 KV.
# Restarts the upstream-candidate container per config (port 8881) and runs the
# speed-bench client against it. Results go to docker/results/<ts>-bench-drafter/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PY="$ROOT/docker/.venv-speed-bench/bin/python"
export HF_HOME="$ROOT/docker/.hf-cache"
SB="$ROOT/tools/server/bench/speed-bench/speed_bench.py"
# /models must be the unsloth store for these runs (compose default is the
# cache root); this bench references /models/<file> paths directly.
export QWEN38_MODEL_DIR="${QWEN38_MODEL_DIR:-/mnt/S/.cache/llama.cpp/qwen38/unsloth-Qwen3.8-27B-GGUF-27af057e}"
COMPOSE_FILE="$ROOT/docker-compose.yml"
PORT="${QWEN38_UPSTREAM_PORT:-8881}"
RESULTS_DIR="$ROOT/docker/results/$(date -u +%Y%m%dT%H%M%SZ)-bench-drafter"
mkdir -p "$RESULTS_DIR"

# fixed bench params (identical for every config/category)
CATEGORIES=(coding math reasoning)
LIMIT=5
OSL=512
LOG_FILE="$RESULTS_DIR/run.log"
echo "results: $RESULTS_DIR"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }

wait_health() {
    local i
    for i in $(seq 1 300); do
        if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            log "server healthy after ${i}s"
            return 0
        fi
        sleep 1
    done
    log "server did not become healthy"
    return 1
}

start_server() { # $1=PROFILE $2=SPEC, rest: extra env
    local profile="$1" spec="$2"; shift 2
    log "starting upstream-candidate (PROFILE=$profile SPEC=$spec, env: $*)"
    env "$@" CACHE_PROFILE="$profile" SPEC="$spec" QWEN38_UPSTREAM_PORT="$PORT" NP=1 \
        CTX=65536 SLOT_PROMPT_CACHE_THRESHOLD=0.75 \
        docker compose -f "$COMPOSE_FILE" --profile upstream \
        up -d --build upstream-candidate 2>&1 | tee -a "$LOG_FILE"
    wait_health
}

# sampling presets for --extra-inputs (speed-bench default is greedy temp 0)
EXTRA_DEFAULT='{"temperature":0}'
EXTRA_REC='{"temperature":1.0,"top_p":0.95,"top_k":20}'

run_bench() { # $1=config $2=category $3=extra-inputs (default greedy)
    local extra="${3:-$EXTRA_DEFAULT}"
    log "speed-bench: $1 / $2 (extra: $extra)"
    "$VENV_PY" "$SB" --url "127.0.0.1:$PORT" --bench qualitative \
        --category "$2" --limit "$LIMIT" --osl "$OSL" \
        --concurrency 1 --timeout 600 --extra-inputs "$extra" \
        --output "$RESULTS_DIR/$1-$2.json" 2>&1 | tee -a "$LOG_FILE"
}

bench_config() { # $1=config $2=extra-inputs (default greedy)
    local cfg="$1"
    local extra="${2:-$EXTRA_DEFAULT}"
    local cat
    for cat in "${CATEGORIES[@]}"; do
        run_bench "$cfg" "$cat" "$extra"
    done
}

stop_server() {
    log "stopping upstream-candidate"
    docker compose -f "$COMPOSE_FILE" --profile upstream stop upstream-candidate >>"$LOG_FILE" 2>&1
    sleep 15
}

# 1. baseline: no drafters, turbo4 KV (replaces the current 8881 container)
#stop_server
#start_server turbo3 off
#bench_config off

# 2. no drafters, q8 K / turbo4 V KV (KV quant comparison)
#stop_server
#start_server q8turbo4 off
#bench_config q8turbo4

#stop_server
#start_server q8q5 off
#bench_config q8q5

#stop_server
#start_server iq4nl off
#bench_config iq4nl

# 3. mtp drafter, turbo3 KV, q8_0 draft cache (existing MTP baseline)
#stop_server
#start_server turbo3 mtp DRAFT_MAX=3
#bench_config mtp-t3

# 4. mtp drafter, turbo4 KV, turbo4 draft cache (draft KV quant comparison)
#stop_server
#start_server turbo4 mtp DRAFT_MAX=3 DRAFT_CACHE_K=turbo4 DRAFT_CACHE_V=turbo4
#bench_config mtp-t4

# 5. dspark drafter, turbo4 KV (magnitude conversion, DRAFT_MAX=4)
#stop_server
#start_server turbo4 dspark \
#    DRAFT_MAX=4 DRAFT_GGUF=/models/DSpark/Qwen3.8-27B-DSpark-Q8_0-magnitude.gguf
#bench_config dspark

# 6. baseline, turbo4 KV, recommended sampling (temp 1.0 / top-p 0.95 / top-k 20)
stop_server
start_server turbo4 dspark DRAFT_MAX=4 DRAFT_GGUF=/models/DSpark/Qwen3.8-27B-DSpark-Q8_0-magnitude.gguf
bench_config dspark-rec4 "$EXTRA_REC"

# 7. mtp drafter, turbo4 KV, recommended sampling (same workload, real production sampling)
stop_server
start_server turbo4 mtp DRAFT_MAX=4
bench_config mtp-rec-4 "$EXTRA_REC"

python3 - "$RESULTS_DIR" "${CATEGORIES[@]}" <<'EOF'
import json, os, sys, statistics
d, cats = sys.argv[1], sys.argv[2:]
cfgs = ("off", "q8turbo4", "q8q5", "iq4nl", "mtp", "mtp-t4", "dspark",
        "off-rec", "mtp-rec", "dspark-rec4", "mtp-rec-4")
print(f"\n=== drafter bench summary ({d}) ===")
print(f"{'config':10} {'pred t/s':>9} {'prompt t/s':>10} {'accept':>7} {'speedup':>8}")
rows = {}
for cfg in cfgs:
    preds, prompts, draft_n, accepted, present = [], [], 0, 0, False
    for cat in cats:
        path = f"{d}/{cfg}-{cat}.json"
        if not os.path.exists(path):
            continue
        present = True
        summary = json.load(open(path))["summary"]
        s = next((e for e in summary if e.get("category") == "overall"), summary[0])
        if s.get("avg_pred_t_s"): preds.append(s["avg_pred_t_s"])
        if s.get("avg_prompt_t_s"): prompts.append(s["avg_prompt_t_s"])
        draft_n += s.get("draft_n") or 0
        accepted += s.get("accepted") or 0
    if not present:
        continue
    rows[cfg] = statistics.mean(preds)
    acc = accepted / draft_n if draft_n else None
    acc_s = f"{acc:.3f}" if acc is not None else "-"
    print(f"{cfg:10} {rows[cfg]:9.2f} {statistics.mean(prompts):10.1f} {acc_s:>7} {'':>8}")
base = rows.get("off") or rows.get("off-rec")
base_name = "off" if rows.get("off") else "off-rec"
if base:
    for cfg in cfgs:
        if cfg != base_name and cfg in rows:
            print(f"{cfg:10} {'':>9} {'':>10} {'':>7} {rows[cfg] / base:8.3f}x vs {base_name}")
EOF

stop_server
log "done."
