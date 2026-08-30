#!/usr/bin/env bash
#
# Single launcher for the qwen38 upstream-candidate llama-server stack.
# Replaces the old run.sh / run-S.sh / run-M.sh: per-quant settings live in
# model-configs/<model>.env (KEY=VALUE, sourced into the compose environment).
#
#   ./run.sh                # start the default model (qwen38-xl), follow logs
#   ./run.sh qwen38-s       # start another model, follow logs
#   ./run.sh ornith1.5      # a manually downloaded model (generic entrypoint)
#   ./run.sh list           # list available models
#   ./run.sh status         # live container, its model, health
#   ./run.sh logs [tail]    # follow the container logs (default 200 lines)
#   ./run.sh stop           # stop and remove the container
#
# NO_BUILD=1 ./run.sh <model>  reuse the image instead of rebuilding it
#
# The config file is the base; a non-empty caller env var overrides it for a
# one-off run (e.g. CTX=49152 ./run.sh qwen38-m). Any other stray export of
# a managed var is cleared, so it cannot change the run silently.
#
# One model at a time: starting a model recreates the same upstream-candidate
# container, so switching models is just running this again.
# If OOM: lower CTX (e.g. 65536) or CACHE_PROFILE=q8turbo3 (smallest KV).

set -euo pipefail
cd "$(dirname "$0")"

COMPOSE_FILE="docker-compose.yml"
SERVICE="upstream-candidate"
CONFIG_DIR="model-configs"
DEFAULT_MODEL="qwen38-xl"
CONTAINER_NAME="qwen38-evaluation-${SERVICE}-1"
# Vars the compose service interpolates; the config file sets these.
MANAGED_VARS="MODEL_GGUF MTP_GGUF MMPROJ_GGUF DRAFT_GGUF MODEL_ENTRYPOINT CACHE_ROOT CACHE_PROFILE \
SPEC CTX DRAFT_MAX DRAFT_CACHE_K DRAFT_CACHE_V NP CACHE_REUSE \
SLOT_PROMPT_CACHE_THRESHOLD QWEN38_UPSTREAM_PORT QWEN38_MODEL_DIR QWEN38_CACHE_ROOT \
KV_PAGED KV_BLOCK_SIZE N_GPU_BLOCKS N_CPU_BLOCKS KV_PAGED_WATERMARK"

die() { echo "error: $*" >&2; exit 1; }

container_id() {
    docker ps -aq --filter "name=^/${CONTAINER_NAME}$" | head -n1
}

usage() {
    awk 'NR > 1 { if ($0 ~ /^#/) { sub(/^# ?/, ""); print; next } exit }' "$0"
}

require_docker() {
    command -v docker >/dev/null 2>&1 || die "docker not found in PATH"
}

cmd_list() {
    local f model gguf entry note
    echo "available models (in $CONFIG_DIR/):"
    for f in "$CONFIG_DIR"/*.env; do
        [ -f "$f" ] || continue
        model="$(basename "$f" .env)"
        gguf="$(grep -E '^MODEL_GGUF=' "$f" | head -n1 | cut -d= -f2- || true)"
        entry="$(grep -E '^MODEL_ENTRYPOINT=' "$f" | head -n1 | cut -d= -f2- || true)"
        note=""; [ -n "$entry" ] && note="  (generic entrypoint)"
        printf '  %-14s %s%s\n' "$model" "${gguf:-<compose default trunk>}" "$note"
    done
}

cmd_status() {
    require_docker
    local cid state model port
    cid="$(container_id)"
    if [ -z "$cid" ]; then
        echo "no ${SERVICE} container found (stopped or never started)"
        return 0
    fi
    state="$(docker inspect -f '{{.State.Status}}' "$cid")"
    model="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$cid" \
        | grep '^MODEL_GGUF=' | head -n1 | cut -d= -f2- || true)"
    port="$(docker inspect -f '{{(index (index .NetworkSettings.Ports "8080/tcp") 0).HostPort}}' "$cid" 2>/dev/null || true)"
    echo "container: $CONTAINER_NAME  state: $state"
    echo "model:     ${model:-<compose default>}"
    if [ -n "$port" ]; then
        echo "endpoint:  http://localhost:$port"
        if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            echo "health:    ok"
        else
            echo "health:    not answering (still loading, or down)"
        fi
    fi
}

cmd_logs() {
    require_docker
    local tail="${1:-200}" cid
    cid="$(container_id)"
    [ -n "$cid" ] || die "no ${SERVICE} container found"
    docker logs --tail "$tail" -f "$cid"
}

cmd_stop() {
    require_docker
    docker compose -f "$COMPOSE_FILE" --profile upstream down --timeout 30 "$SERVICE"
    echo "stopped."
}

wait_healthy() {
    local port="$1" deadline now next_report start cid state
    start="$(date +%s)"
    deadline=$(( start + 20 * 60 ))
    next_report=30
    echo "waiting for http://127.0.0.1:$port/health (model load can take several minutes) ..."
    while :; do
        if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            echo "server is up on :$port"
            return 0
        fi
        # the entrypoint can die before serving (bad config, missing file,
        # OOM): a dead container will never become healthy, so bail with logs
        cid="$(container_id)"
        if [ -z "$cid" ]; then
            echo "error: container vanished before the server came up" >&2
            exit 1
        fi
        state="$(docker inspect -f '{{.State.Status}}' "$cid" 2>/dev/null || true)"
        if [ "$state" != "running" ]; then
            echo "error: container stopped before the server came up (state: ${state:-gone}); recent logs:" >&2
            docker logs --tail 40 "$cid" >&2
            exit 1
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "error: server never came up on :$port; recent logs:" >&2
            docker logs --tail 40 "$(container_id)" >&2 || true
            exit 1
        fi
        now=$(( $(date +%s) - start ))
        if [ "$now" -ge "$next_report" ]; then
            echo "  still loading ... ${now}s"
            next_report=$(( next_report + 30 ))
        fi
        sleep 5
    done
}

cmd_start() {
    local model="${1:-$DEFAULT_MODEL}" cfg build_args=()
    [[ "$model" =~ ^[A-Za-z0-9._-]+$ ]] || die "bad model name: $model"
    cfg="$CONFIG_DIR/$model.env"
    if [ ! -f "$cfg" ]; then
        echo "error: no config for model '$model' (expected $cfg)" >&2
        cmd_list >&2
        exit 1
    fi
    require_docker

    # Caller env wins for a one-off override: remember name+value, clear all
    # managed vars, apply the config, then put the overrides back on top.
    local v
    declare -A caller_overrides=()
    for v in $MANAGED_VARS; do
        if [[ -n "${!v:-}" ]]; then
            caller_overrides["$v"]="${!v}"
        fi
        unset "$v"
    done
    set -a
    # shellcheck disable=SC1090
    source "$cfg"
    set +a
    for v in "${!caller_overrides[@]}"; do
        local cfgval="${!v:-<unset>}"
        export "$v=${caller_overrides[$v]}"
        echo "note: env override $v=${caller_overrides[$v]} (config: $cfgval)"
    done

    if [ "${NO_BUILD:-0}" != "1" ]; then
        build_args+=(--build)
    fi
    docker compose -f "$COMPOSE_FILE" --profile upstream up -d "${build_args[@]}" "$SERVICE"

    local port="${QWEN38_UPSTREAM_PORT:-8882}"
    wait_healthy "$port"

    echo
    # the qwen38 entrypoint defaults SPEC to mtp, the generic one to off
    local spec_default="mtp"
    [ -z "${MODEL_ENTRYPOINT:-}" ] || spec_default="off"
    echo "============================================================"
    echo " model     : $model  ($cfg)"
    if [ -n "${MODEL_GGUF:-}" ]; then
        echo " weights   : $MODEL_GGUF"
    fi
    if [ -n "${MODEL_ENTRYPOINT:-}" ]; then
        echo " entrypoint: $MODEL_ENTRYPOINT"
    fi
    echo " spec      : ${SPEC:-$spec_default}  ctx=${CTX:-102400}  np=${NP:-1}  cache=${CACHE_PROFILE:-turbo4}"
    echo " endpoint  : http://localhost:$port"
    echo " stop      : ./run.sh stop"
    echo "============================================================"
    echo
    docker logs --tail 200 -f "$(container_id)"
}

case "${1:-}" in
    "")      cmd_start "$DEFAULT_MODEL" ;;
    stop)    cmd_stop ;;
    status)  cmd_status ;;
    list)    cmd_list ;;
    logs)    cmd_logs "${2:-200}" ;;
    help|-h|--help) usage ;;
    *)       cmd_start "$1" ;;
esac
