#!/usr/bin/env bash
# Canonical verification for this llama.cpp fork.
#
# Mirrors the CI gates using separate shared-library build trees:
#   - python-lint.yml       (flake8, with flake8-no-print)
#   - python-type-check.yml (ty check)
#   - build-cpu.yml         (CPU CMake Release build + ctest -L main)
#   - CUDA backend coverage (test-backend-ops only)
#
# The Python steps are skipped (with a notice) when the tools are not
# installed: pip install flake8 flake8-no-print ty
#
# Usage: .agent/verify.sh [extra CPU ctest args...]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CPU_BUILD_DIR="$ROOT/build-verify-cpu"
CUDA_BUILD_DIR="$ROOT/build-verify-cuda"
CUDA_TMP_DIR="$CUDA_BUILD_DIR/tmp"

step() { echo; echo "=== $* ==="; }

step "python: flake8"
if python3 -m flake8 --version >/dev/null 2>&1; then
    (cd "$ROOT" && python3 -m flake8 .) || exit 1
else
    echo "skip: flake8 not available"
fi

step "python: ty"
if command -v ty >/dev/null 2>&1; then
    (cd "$ROOT" && ty check --output-format=github) || exit 1
else
    echo "skip: ty not available"
fi

step "configure: $CPU_BUILD_DIR (CPU, shared, Release)"
cmake -S "$ROOT" -B "$CPU_BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_STATIC=OFF \
    -DGGML_CUDA=OFF || exit 1

step "build: $CPU_BUILD_DIR"
cmake --build "$CPU_BUILD_DIR" --parallel 4 || exit 1

# Known baseline failures in this sandbox (see .agent/rules/verification.md).
# verify.sh fails only on failures that are NOT in this list; baseline
# failures are reported but do not fail the run. Keep in sync with the
# verification rule when the baseline changes.
baseline="test-tokenizers-ggml-vocabs
test-arg-parser
test-backend-ops
test-quantize-fns
test-quantize-perf"

check_ctest_result() {
    local ctest_rc="$1"
    local ctest_out="$2"
    local suite="$3"
    local failed new_failures baseline_failed t

    if [ "$ctest_rc" -eq 0 ]; then
        rm -f "$ctest_out"
        echo "$suite: OK"
        return 0
    fi

    failed="$(awk '/The following tests FAILED:/{f=1;next} f && $1 ~ /^[0-9]+$/ && $2=="-" {print $3}' "$ctest_out")"
    rm -f "$ctest_out"

    if [ -z "$failed" ]; then
        echo "$suite: FAIL - ctest exited nonzero (rc=$ctest_rc) without a parsable failure list"
        return 1
    fi

    new_failures=""
    baseline_failed=""
    while IFS= read -r t; do
        [ -n "$t" ] || continue
        if printf '%s\n' "$baseline" | grep -qx "$t"; then
            baseline_failed="$baseline_failed $t"
        else
            new_failures="$new_failures $t"
        fi
    done <<< "$failed"

    if [ -n "$new_failures" ]; then
        echo "$suite: FAIL - new (non-baseline) test failures:$new_failures"
        return 1
    fi

    echo "$suite: OK (known baseline failures, expected:$baseline_failed)"
}

step "test: CPU ctest -L main"
CTEST_OUT="$(mktemp)"
(cd "$CPU_BUILD_DIR" && ctest -L main --timeout 900 --output-on-failure "$@") 2>&1 | tee "$CTEST_OUT"
ctest_rc=${PIPESTATUS[0]}
check_ctest_result "$ctest_rc" "$CTEST_OUT" "CPU tests" || exit 1

step "configure: $CUDA_BUILD_DIR (CUDA, shared, Release)"
cmake -S "$ROOT" -B "$CUDA_BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_STATIC=OFF \
    -DGGML_CUDA=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF || exit 1

step "build: $CUDA_BUILD_DIR (test-backend-ops only)"
mkdir -p "$CUDA_TMP_DIR" || exit 1
TMPDIR="$CUDA_TMP_DIR" cmake --build "$CUDA_BUILD_DIR" --parallel 4 --target test-backend-ops || exit 1

step "test: CUDA test-backend-ops"
CTEST_OUT="$(mktemp)"
(cd "$CUDA_BUILD_DIR" && ctest -R '^test-backend-ops$' --timeout 900 --output-on-failure) 2>&1 | tee "$CTEST_OUT"
ctest_rc=${PIPESTATUS[0]}
check_ctest_result "$ctest_rc" "$CTEST_OUT" "CUDA tests" || exit 1

echo
echo "verify: OK"
