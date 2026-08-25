# Instructions for llama.cpp (private fork)

This is a private fork of ggml-org/llama.cpp for a single-user setup. Agents do all coding, testing, review, and real-model inference validation. Nothing here goes upstream.

## Setup

- Hardware target: one RTX 4090, 24 GB VRAM, CUDA backend. Pick context length, batch size, and quantization to fit that; do not assume more VRAM or more than one GPU.
- Remotes: `origin` is this fork, `upstream` is ggml-org/llama.cpp. Sync from upstream only when the user asks, on the branch they name. On conflicts in fork-specific files (this file, `model-configs/`, `scripts/qwen38/`, `docker-compose.yml`, `run.sh`), keep the fork version unless the user says otherwise.
- The agent sandbox has direct access to the single RTX 4090 and the `/models` cache. Agents must run relevant 27B inference, benchmarks, and bounded server validation themselves with source-matched binaries from `build-verify`.
- Docker and Docker Compose are not available in the sandbox. Do not hand inference testing to the user for that reason: translate the applicable `model-configs/` and container entrypoint settings into direct `build-verify/bin/llama-*` arguments. Compose deployment itself is not an agent-side validation surface.

## Verification

- Canonical command: `.agent/verify.sh [extra ctest args...]` - python lint (flake8, ty when installed), Release CUDA build in `build-verify`, `ctest -L main`.
- Sandbox baseline failures (LFS fixtures and restricted network egress) are documented in `.agent/rules/verification.md`. Treat anything else as a regression; do not weaken checks to force a pass.

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
