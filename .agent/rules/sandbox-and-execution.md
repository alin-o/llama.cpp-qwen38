---
trigger: model_decision
description: Use when building, running, or testing llama.cpp binaries or models in this sandbox; covers the 2 GB GPU-slice sandbox, the /models cache mount, and the run.sh + model-configs/ execution flow.
---

# Sandbox and execution environment

- Agents run in a docker sandbox with a 2 GB VRAM slice of the host GPU. Small test models (the ctest fixture set, a few MB to tens of MB each) fit, so CUDA test paths can run here. The 27B working set does not fit and runs on the host.
- `/models` is the host cache root (`/mnt/S/.cache/llama.cpp/`), writable. It holds the `qwen38/` working set (trunks, MTP/DSpark drafters, wikitext) plus other store entries. Top-level HF/Ollama entries whose real weights live in the unmounted `blobs/` dir are dead symlinks - repair by re-download (see hf-model-downloads.md).
- Path trap: inside the compose run container, `/models` means only the `qwen38/unsloth-...-27af057e` model dir and `/qwen38-cache` the cache root - same path names as in the sandbox, different trees.
- To run a model, use `./run.sh [model]` (verbs: list, status, logs, stop). Per-model settings live in `model-configs/<model>.env` (KEY=VALUE, sourced into the compose environment; non-empty caller env vars override with a printed note). It wraps `docker compose -f docker-compose.yml --profile upstream up -d --build llama-server`, waits for /health, then tails the logs.
- The compose service defaults to `/docker/run-model-server.sh` for every model; `MODEL_ENTRYPOINT` remains available for an explicit override. Each config must set `MODEL_GGUF` and explicitly set `SPEC=off`, `SPEC=mtp`, or `SPEC=dspark`. Optional `MMPROJ_GGUF`, `MTP_GGUF`, `DRAFT_GGUF`, `CHAT_TEMPLATE_FILE`, and `REASONING_FORMAT` select additional model capabilities; an embedded MTP head is detected automatically.
- run.sh waits for /health but bails immediately (with the last 40 log lines) if the container dies before serving, so a bad config surfaces as an error, not a 20-minute "still loading".
- Adding a model = one new `model-configs/<name>.env`. In-container, `/qwen38-cache` is the host dir named by `QWEN38_CACHE_ROOT` (default: the `qwen38/` subdir of the cache root) and `/models` only the qwen38 unsloth dir. Weights downloaded to the cache ROOT (manually pulled files) need `QWEN38_CACHE_ROOT=/mnt/S/.cache/llama.cpp` in the config so `/qwen38-cache/<file>.gguf` resolves; never reference `/models/...` for them.
- Exec space: the repo dir is writable and executable; `/`, `/tmp`, and `/home/ubuntu` are read-only or noexec, so helper scripts must live in the repo dir to run.

- From the sandbox the running stack is reachable at http://llama-proxy:8881, NOT localhost:8881. The sandbox is in its own network namespace, so localhost never reaches the container. curl http://llama-proxy:8881/health (expect {"status":"ok"}) and smoke-test via the /v1/chat/completions endpoint.

- Ornith is reasoning-first: its tokens land in reasoning_content, so visible content is empty until the reasoning block closes - test with a generous max_tokens to see real text.

- When a downloaded GGUF fails to load, read its general.architecture from the GGUF header (first KV string) and grep src/llama-arch.cpp for that name to confirm the build supports it, rather than assuming the image is too old. Ornith's header reports qwen35moe, which this tree supports (src/models/qwen35moe.cpp) - no rebuild needed.
- Agents verify code changes with `.agent/verify.sh` (build + `ctest -L main`), which may load the small test models on the 2 GB GPU. For real-model validation (the 27B qwen38 working set), state exactly in the task evidence what the user should run on the host.
