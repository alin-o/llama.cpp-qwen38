---
trigger: model_decision
description: Use when verifying a change, running tests, building, or triaging a test failure in this llama.cpp fork (canonical verify command, build dir, known baseline failures).
---

# Verification

Canonical command, run from the repo root:

    .agent/verify.sh [extra ctest args...]

Steps:
1. `python3 -m flake8 .` (skipped with a notice when flake8 is not installed)
2. `ty check --output-format=github` (skipped when `ty` is not installed)
3. Incremental CPU/shared build of `build-verify-cpu` (CMake/Ninja, Release, `GGML_CUDA=OFF`, capped at `--parallel 4`)
4. `ctest -L main --timeout 900 --output-on-failure` inside `build-verify-cpu`
5. Incremental CUDA/shared build of only `test-backend-ops` in `build-verify-cuda`, capped at `--parallel 4`
6. Run `test-backend-ops` inside `build-verify-cuda`

Both trees are configured on every invocation with `BUILD_SHARED_LIBS=ON` and `GGML_STATIC=OFF`. This prevents stale CMake cache values from switching verification back to large statically linked executables. CUDA compiler temporary files are routed into `build-verify-cuda/tmp` because the container's `/tmp` tmpfs is too small for four parallel nvcc jobs. Extra arguments passed to `verify.sh` apply to the full CPU ctest suite only.

Exit status: nonzero only for lint/type failures, build failures, or test failures that are NOT in the baseline list below. Known baseline failures are printed as a summary and the command still exits 0. The baseline list is embedded in `verify.sh` - keep the two in sync when the baseline changes.

Prerequisites: cmake, ninja, a C++ toolchain, and the CUDA toolkit. Optional Python gates: `pip install flake8 flake8-no-print ty`.
Do not use `build/`, `build-cuda/`, or the obsolete `build-verify/`; `build-verify-cpu` and `build-verify-cuda` are the canonical build dirs.
All tasks share these warm trees (no per-task worktrees): never create another build dir; every verification rebuild must go through this script.

Runtime: incremental builds take a few minutes; a fresh CPU configure plus full build takes tens of minutes on this box. The CUDA tree builds only the backend correctness test and its dependencies.

Known baseline failures in the agent sandbox (not regressions; do not chase them or weaken checks to force a pass). Two classes:

Environmental (sandbox constraints; may pass elsewhere or when the host GPU is idle):
- `test-tokenizers-ggml-vocabs`: needs git-lfs fixtures; the files here are LFS pointers and git-lfs is not installed
- `test-arg-parser`: the "test good URL" step does a live HTTP GET to http://ggml.ai; the sandbox has no network egress

Fork code bugs (pre-existing; need dedicated fix tasks - do not fix them silently as a side effect of other work):
- `test-backend-ops`: SEGFAULT in CUDA gated delta net coverage; exercised only by the focused CUDA tree
- `test-quantize-fns`: SEGFAULT, CPU-only quantization functions; predates sandbox GPU enablement
- `test-quantize-perf`: SEGFAULT in the q2_0 `quantize_row_q_reference` benchmark; CPU-only; predates sandbox GPU enablement

A failure is a regression only when it is not in the list above, or when a previously passing test now fails. The tiny test model `tinyllamas/stories15M-q4_0.gguf` (fetched by the `test-download-model` fixture into `build-verify-cpu/tinyllamas/`) and the 27B working set can be run in the sandbox when relevant (force `-dev cuda0` and use a single turn with `-st` for focused tiny-model checks).
