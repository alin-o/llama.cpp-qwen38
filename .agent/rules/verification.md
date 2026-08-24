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
3. Incremental build of `build-verify` (CMake/Ninja, Release, `GGML_CUDA=ON`); configures it automatically on a fresh checkout
4. `ctest -L main --timeout 900 --output-on-failure` inside `build-verify`

Exit status: nonzero only for lint/type failures, build failures, or test failures that are NOT in the baseline list below. Known baseline failures are printed as a summary and the command still exits 0. The baseline list is embedded in `verify.sh` - keep the two in sync when the baseline changes.

Prerequisites: cmake, ninja, a C++ toolchain. Optional Python gates: `pip install flake8 flake8-no-print ty`.
Do not use `build/` or `build-cuda/`; `build-verify` is the canonical build dir.
All tasks share this warm tree (no per-task worktrees): never create another build dir and never run a fresh CMake configure; every rebuild must be incremental through this script.

Runtime: incremental builds take a few minutes; a fresh configure plus full build takes tens of minutes on this 8-core box.

Known baseline failures in the agent sandbox (not regressions; do not chase them or weaken checks to force a pass). Two classes:

Environmental (sandbox constraints; may pass elsewhere or when the host GPU is idle):
- `test-tokenizers-ggml-vocabs`: needs git-lfs fixtures; the files here are LFS pointers and git-lfs is not installed
- `test-arg-parser`: the "test good URL" step does a live HTTP GET to http://ggml.ai; the sandbox has no network egress
- `test-backend-ops`: the full op matrix exhausts the sandbox's free VRAM (the host normally uses most of the card; only ~2 GB is free) and segfaults late in the run. Isolated op runs (`-o <op>`) pass; rerun the full matrix when the host GPU is idle

Fork code bugs (pre-existing; need dedicated fix tasks - do not fix them silently as a side effect of other work):
- `test-paged-kv-e2e`: SEGFAULT - context init attempts a (size_t)-1 byte allocation; reproduces with `-ngl 0` (CPU) and with CUDA; tied to the paged-KV work (commits `420a3ec1b` / `28ef673fa`)
- `test-quantize-fns`: SEGFAULT, CPU-only quantization functions; predates sandbox GPU enablement
- `test-quantize-perf`: SEGFAULT in the q2_0 `quantize_row_q_reference` benchmark; CPU-only; predates sandbox GPU enablement

A failure is a regression only when it is not in the list above, or when a previously passing test now fails. The tiny test model `tinyllamas/stories15M-q4_0.gguf` (fetched by the `test-download-model` fixture into `build-verify/tinyllamas/`) can be run in the sandbox on the limited GPU (force `-dev cuda0`, single turn `-st`); the 27B working set and servers must stay on the host.
