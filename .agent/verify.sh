#!/usr/bin/env bash
# Canonical verification for this llama.cpp fork.
#
# Mirrors the CI gates against the local CUDA tree (build-verify):
#   - python-lint.yml       (flake8, with flake8-no-print)
#   - python-type-check.yml (ty check)
#   - build-cpu.yml         (CMake Release build + ctest -L main)
#
# The Python steps are skipped (with a notice) when the tools are not
# installed: pip install flake8 flake8-no-print ty
#
# Usage: .agent/verify.sh [extra ctest args...]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT/build-verify"

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

step "build: $BUILD_DIR (CUDA, Release)"
if [ ! -f "$BUILD_DIR/build.ninja" ]; then
    # fresh checkout/worktree: full configure + build (slow, tens of minutes)
    cmake -B "$BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON || exit 1
fi
cmake --build "$BUILD_DIR" --parallel 4 || exit 1

# Known baseline failures in this sandbox (see .agent/rules/verification.md).
# verify.sh fails only on failures that are NOT in this list; baseline
# failures are reported but do not fail the run. Keep in sync with the
# verification rule when the baseline changes.
baseline="test-tokenizers-ggml-vocabs
test-arg-parser
test-backend-ops
test-quantize-fns
test-quantize-perf"

step "test: ctest -L main"
CTEST_OUT="$(mktemp)"
(cd "$BUILD_DIR" && ctest -L main --timeout 900 --output-on-failure "$@") 2>&1 | tee "$CTEST_OUT"
ctest_rc=${PIPESTATUS[0]}

if [ "$ctest_rc" -eq 0 ]; then
    rm -f "$CTEST_OUT"
    echo
    echo "verify: OK"
    exit 0
fi

failed="$(awk '/The following tests FAILED:/{f=1;next} f && $1 ~ /^[0-9]+$/ && $2=="-" {print $3}' "$CTEST_OUT")"
rm -f "$CTEST_OUT"

if [ -z "$failed" ]; then
    echo
    echo "verify: FAIL - ctest exited nonzero (rc=$ctest_rc) without a parsable failure list"
    exit 1
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
    echo
    echo "verify: FAIL - new (non-baseline) test failures:$new_failures"
    exit 1
fi

echo
echo "verify: OK (known baseline failures, expected:$baseline_failed)"
