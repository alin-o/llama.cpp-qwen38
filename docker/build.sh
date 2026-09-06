#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
CARAPA_STACKS_DIR="${CARAPA_STACKS_DIR:-$SCRIPT_DIR/../../carapa/my-stacks}"
CARAPA_BUILDX_SCRIPT="$CARAPA_STACKS_DIR/scripts/carapa-buildx.sh"

if [ ! -r "$CARAPA_BUILDX_SCRIPT" ]; then
    echo "Missing Carapa BuildKit helper: $CARAPA_BUILDX_SCRIPT" >&2
    exit 1
fi

# qwen38 shares Carapa's builder and local registries. Recreate the builder
# after host cleanup instead of failing with "no builder carapa found".
# shellcheck disable=SC1090
. "$CARAPA_BUILDX_SCRIPT"
CARAPA_BUILDKIT_CONFIG="$CARAPA_STACKS_DIR/scripts/buildkitd.toml"
carapa_buildx_init

CARAPA_IMAGE="carapa-llama-cpp:latest"

if ! docker image inspect "$CARAPA_IMAGE" >/dev/null 2>&1; then
    echo "Missing local image $CARAPA_IMAGE." >&2
    echo "Build it first: cd /home/alin/ai-workspace/carapa/my-stacks/stacks/llama-cpp && ./build.sh" >&2
    exit 1
fi

# Keep every CMake/compiler line visible. The default TTY progress UI collapses
# parallel build failures before they can be read.
BUILDKIT_PROGRESS=plain \
BUILDX_BUILDER="${CARAPA_BUILDER:-carapa}" \
    docker compose --profile upstream build llama-server
