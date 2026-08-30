---
trigger: model_decision
description: Use when building, running, or testing llama.cpp binaries or models in this sandbox; covers full-GPU execution, the /models cache mount, and the run.sh + model-configs/ execution flow.
---

# Sandbox and execution environment

- Agents run in a docker sandbox with access to the host GPU's full available memory. CUDA test paths and the 27B working set can run in the sandbox, subject to normal model and workload requirements.
- `/models` is the host cache root (`/mnt/S/.cache/llama.cpp/`), writable. It holds the `qwen38/` working set (trunks, MTP/DSpark drafters, wikitext) plus other store entries. Top-level HF/Ollama entries whose real weights live in the unmounted `blobs/` dir are dead symlinks - repair by re-download (see hf-model-downloads.md).
- Path trap: inside the compose run container, `/models` is the model dir named by `QWEN38_MODEL_DIR` (compose default: the cache root; the qwen38 configs point it at the `qwen38/unsloth-...` dir) and `/qwen38-cache` the cache root - same path names as in the sandbox, different trees.
- Docker and Docker Compose are not available in the agent sandbox and are not expected to become available. `./run.sh` documents the host deployment flow but cannot be used by agents. Do not hand inference validation to the user because of this limitation.
- For source-matched model validation, first build with `.agent/verify.sh`, then run the appropriate direct binary from `build-verify/bin/` (`llama-server`, `llama-cli`, `llama-bench`, `llama-paged`, or another existing tool). Translate the applicable `model-configs/<model>.env`, `run-upstream-server.sh`, and `run-model-server.sh` settings into direct arguments. Run relevant 27B inference, benchmarks, and bounded server checks yourself against `/models` on the full RTX 4090.
- Two entrypoints behind the same compose service: `run-upstream-server.sh` (default; qwen38 stack - chat template, mmproj, MTP embedded-head magic) and `run-model-server.sh` (generic: bare llama-server, needs MODEL_GGUF, optional MMPROJ_GGUF / SPEC=mtp+MTP_GGUF). A config picks the generic one by setting `MODEL_ENTRYPOINT=/docker/run-model-server.sh`.
- Trap: the compose file defaults `SPEC=mtp` (qwen38 stack), and run.sh only exports what the config sets - so a generic config without SPEC gets mtp injected and the generic entrypoint bails asking for MTP_GGUF. Generic configs must set `SPEC=off` explicitly.
- In the host deployment flow, run.sh waits for /health but bails immediately (with the last 40 log lines) if the container dies before serving, so a bad config surfaces as an error, not a 20-minute "still loading". This is deployment documentation, not the agent validation path.
- Adding a model = one new `model-configs/<name>.env`. In-container, `/qwen38-cache` is the host dir named by `QWEN38_CACHE_ROOT` (default: the `qwen38/` subdir of the cache root) and `/models` the dir named by `QWEN38_MODEL_DIR` (default: the cache root). Weights downloaded to the cache ROOT (manually pulled files) need `QWEN38_CACHE_ROOT=/mnt/S/.cache/llama.cpp` in the config so `/qwen38-cache/<file>.gguf` resolves; never reference `/models/...` for them.
- Exec space: the repo dir is writable and executable; `/`, `/tmp`, and `/home/ubuntu` are read-only or noexec, so helper scripts must live in the repo dir to run.

- From the sandbox the running stack is reachable at http://llama-proxy:8881, NOT localhost:8881. The sandbox is in its own network namespace, so localhost never reaches the container. curl http://llama-proxy:8881/health (expect {"status":"ok"}) and smoke-test via the /v1/chat/completions endpoint.

- Ornith is reasoning-first: its tokens land in reasoning_content, so visible content is empty until the reasoning block closes - test with a generous max_tokens to see real text.

- When a downloaded GGUF fails to load, read its general.architecture from the GGUF header (first KV string) and grep src/llama-arch.cpp for that name to confirm the build supports it, rather than assuming the image is too old. Ornith's header reports qwen35moe, which this tree supports (src/models/qwen35moe.cpp) - no rebuild needed.
- Agents verify code changes with `.agent/verify.sh` (build + `ctest -L main`), which may load test models on the full GPU. Real-model validation, including the 27B qwen38 working set, can run in the sandbox when relevant.

## Benchmark configuration preflight

- Before loading a large model, inspect the executable's `--help` and the relevant source validations and scheduler conditions. Record hard relationships such as mutually exclusive flags, required equality between batch and microbatch sizes, and whether a new prompt must fit entirely within one batch.
- Build a small feasibility table that separates workload minimums (prompt length, generated tokens, and scheduler headroom), hard harness constraints, tunable memory allocations (context, batch, and cache blocks), and optional model components that are actually loaded.
- Reduce memory allocations from workload requirements first. Do not reduce batch size when the harness requires the complete prompt to fit in one batch.
- Validate a candidate incrementally: confirm that the process reaches health, one request completes, and timing fields are usable before running warmups and repeated samples.
- Change one independent variable at a time. After a failed invariant, stop trying nearby values until the emitting validation or scheduler condition is understood.
- Check load logs or GGUF metadata before attributing memory use to mmproj, MTP, or another optional component. Distinguish embedded but unused tensors from components that were allocated.
- Treat error messages as hypotheses. If observed behavior conflicts with a message, inspect the source condition that emits it before attempting more configurations.
- Keep rejected probes cheap and record why they were invalid. Reserve repeated sampling for a configuration whose harness and instrumentation have passed preflight.
