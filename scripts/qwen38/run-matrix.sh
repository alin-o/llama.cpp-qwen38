#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# /models must be the unsloth store for these runs (compose default is the
# cache root); both entrypoints resolve trunk/MTP/mmproj under /models.
export QWEN38_MODEL_DIR="${QWEN38_MODEL_DIR:-/mnt/S/.cache/llama.cpp/qwen38/unsloth-Qwen3.8-27B-GGUF-27af057e}"
COMPOSE=(docker compose -f "${ROOT}/docker-compose.yml")
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/docker/results/$(date -u +%Y%m%dT%H%M%SZ)-matrix}"
LONG_CONTEXT="${LONG_CONTEXT:-0}"
PROFILE_SET="${PROFILE_SET:-all}"
mkdir -p "$RESULTS_DIR"

cleanup() {
    "${COMPOSE[@]}" stop upstream-candidate >/dev/null 2>&1 || true
}
trap cleanup EXIT

run_one() {
    local engine="$1"
    local service="$2"
    local port="$3"
    local profile="$4"
    local cache_profile="$5"
    local spec="$6"
    local long_arg=()
    if [[ "$LONG_CONTEXT" == 1 ]]; then
        long_arg=(--long-context)
    fi

    echo "matrix: starting ${profile}" >&2
    CACHE_PROFILE="$cache_profile" SPEC="$spec" "${COMPOSE[@]}" --profile "$engine" up -d --no-build "$service"

    nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits -l 1 > "${RESULTS_DIR}/${profile}.gpu.csv" &
    local monitor_pid=$!
    set +e
    python3 "${ROOT}/scripts/qwen38/acceptance.py" \
        --base-url "http://127.0.0.1:${port}" \
        --profile "$profile" \
        --output "${RESULTS_DIR}/${profile}.json" \
        "${long_arg[@]}"
    local status=$?
    set -e
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" >/dev/null 2>&1 || true
    "${COMPOSE[@]}" logs --no-color "$service" > "${RESULTS_DIR}/${profile}.server.log" 2>&1
    "${COMPOSE[@]}" stop "$service"
    "${COMPOSE[@]}" rm -f "$service"
    if [[ $status -ne 0 ]]; then
        echo "matrix: ${profile} failed; continuing to collect the full matrix" >&2
    fi
}

selected() {
    [[ "$PROFILE_SET" == all || ",${PROFILE_SET}," == *",$1,"* ]]
}

selected upstream-q8q5-base && run_one upstream upstream-candidate "${QWEN38_UPSTREAM_PORT:-8882}" upstream-q8q5-base q8q5 off
selected upstream-q8q5-mtp && run_one upstream upstream-candidate "${QWEN38_UPSTREAM_PORT:-8882}" upstream-q8q5-mtp q8q5 mtp
selected upstream-iq4nl-base && run_one upstream upstream-candidate "${QWEN38_UPSTREAM_PORT:-8882}" upstream-iq4nl-base iq4nl off
selected upstream-iq4nl-mtp && run_one upstream upstream-candidate "${QWEN38_UPSTREAM_PORT:-8882}" upstream-iq4nl-mtp iq4nl mtp

python3 "${ROOT}/scripts/qwen38/compare-results.py" --results-dir "$RESULTS_DIR" --output "${RESULTS_DIR}/decision.json"
echo "matrix: results are in ${RESULTS_DIR}" >&2
