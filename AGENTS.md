# Instructions for llama.cpp (private fork)

This is a private fork of ggml-org/llama.cpp for a single-user setup. Agents do all coding, testing, and review; the user runs real-model inference on the host. Nothing here goes upstream.

## Setup

- Hardware target: one RTX 4090, 24 GB VRAM, CUDA backend. Pick context length, batch size, and quantization to fit that; do not assume more VRAM or more than one GPU.
- Remotes: `origin` is this fork, `upstream` is ggml-org/llama.cpp. Sync from upstream only when the user asks, on the branch they name. On conflicts in fork-specific files (this file, `model-configs/`, `scripts/qwen38/`, `docker-compose.qwen38.yml`, `run.sh`), keep the fork version unless the user says otherwise.
- Real-model inference and model servers (docker-compose, qwen38 scripts) run on the host by the user. Agents build in the sandbox and may run the small ctest/tiny test models there on the limited 2 GB GPU slice, but never run the 27B working set or start servers (see `.agent/rules/sandbox-and-execution.md`).

## Verification

- Canonical command: `.agent/verify.sh [extra ctest args...]` - python lint (flake8, ty when installed), Release CUDA build in `build-verify`, `ctest -L main`.
- Sandbox baseline failures (LFS fixtures, no network egress, no CUDA init) are documented in `.agent/rules/verification.md`. Treat anything else as a regression; do not weaken checks to force a pass.

## Code standards

- Prefer the simplest thing that works. A simpler change that does 90% of the job is often preferable to a complex one that does 100%.
- ASCII only in code and comments: no emdash, unicode arrow, multiplication sign, or ellipsis; use `-`, `->`, `x`, `...` instead.
- Comments: concise (usually 1-2 lines), only where the code is not self-explanatory, no hard-wrapping to a fixed column, simple wordings.
- Reuse existing infrastructure. No new subsystems, dependencies, or files under `tests/*` without user confirmation; do not add tests for trivial changes.
- The Jinja engine lives in `common/jinja`; llama.cpp does NOT use Minja.
- Read all relevant files before editing; changes must blend in with the surrounding code. Large or new-pattern changes: ask the user first.

## Submissions

- Agents commit, push, and open PRs on the user's behalf.
- Commit messages: concise one-line subject, `Assisted-by: <assistant name>` trailer, never `Co-authored-by:`.
- Agents must understand the code they write and be able to defend it in review.

## Useful resources

- Skills: reusable task workflows in the [skills/](skills/) directory - check there before starting a task.
- [How to add a new model](docs/development/HOWTO-add-model.md)
- [Build docs](docs/build.md), [server usage docs](tools/server/README.md), [server dev docs](tools/server/README-dev.md)
- [PEG parser](docs/development/parsing.md), [auto parser](docs/autoparser.md), [Jinja engine](common/jinja/README.md)
