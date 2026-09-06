# Qwen3.8 27B upstream migration

This checkout is pinned by `qwen38/upstream.lock`. The existing TurboQuant repository is a read-only comparison target and remains the rollback deployment.

## Artifacts

The evaluation uses revision `27af057ecb382ddfea5d12837360a8980560e3ed` of `unsloth/Qwen3.8-27B-GGUF`:

- `Qwen3.8-27B-UD-Q4_K_XL.gguf`
- `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`
- `mmproj-BF16.gguf`

The target GGUF already contains its NextN tensors. The primary matrix uses that embedded head in both engines: upstream creates its MTP context from the target, while the fork points `--model-draft` back to the same file and uses its shared-model NextN path. The standalone Q4_0 MTP GGUF is downloaded and validated as a fallback, but is not mixed into the primary engine comparison.

Download and verify them:

```bash
./scripts/qwen38/download-artifacts.sh
python3 scripts/qwen38/verify-gguf.py \
  --target /mnt/S/.cache/llama.cpp/qwen38/unsloth-Qwen3.8-27B-GGUF-27af057e/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --mtp /mnt/S/.cache/llama.cpp/qwen38/unsloth-Qwen3.8-27B-GGUF-27af057e/MTP/mtp-Qwen3.8-27B-Q4_0.gguf \
  --mmproj /mnt/S/.cache/llama.cpp/qwen38/unsloth-Qwen3.8-27B-GGUF-27af057e/mmproj-BF16.gguf \
  --json qwen38/results/artifact-preflight.json
```

The SHA-256 values are recorded in `qwen38/artifacts.sha256`. Runtime commands use explicit local paths and never use `-hf`.

## Build and short matrix

Both images use CUDA 13.1, compute capability 8.9, static libraries, flash attention, and the same compiler image.

```bash
docker compose -f docker/docker-compose.yml --profile upstream build llama-server
./scripts/qwen38/run-matrix.sh
```

The matrix runs these sequentially so only one model occupies VRAM:

- Fork TurboQuant3, baseline and NextN.
- Upstream `q8_0` K plus `q5_0` V, baseline and MTP.
- Upstream `iq4_nl` K/V, baseline and MTP.

Each profile receives the same target, MTP head, projector, 102400-token context, one slot, full GPU offload, and draft maximum 4. The harness tests non-streaming and streaming chat, all four reasoning-effort values, preserved reasoning, structured XML tool calls, vision, slot save/erase/restore, 20 deterministic prompts, timings, MTP acceptance, and free VRAM.

To add all 12 fresh (prompt-cache-disabled) long-context retrieval cases and
the explicit 100000-token prefill plus 2048-token generation case to each
profile:

```bash
LONG_CONTEXT=1 ./scripts/qwen38/run-matrix.sh
```

After the short matrix identifies which profiles still have a path to pass,
the expensive cases can be restricted without changing their settings:

```bash
LONG_CONTEXT=1 \
PROFILE_SET=fork-turbo3-base,fork-turbo3-nextn,upstream-iq4nl-base,upstream-iq4nl-mtp \
./scripts/qwen38/run-matrix.sh
```

The fork's custom multimodal server returns HTTP 501 for slot serialization.
The harness records that historical limitation but continues collecting its
performance baseline. Upstream profiles still fail the API gate if slot
save/restore does not work.

`decision.json` requires at least 1 GiB free VRAM, no API failures, exact baseline/MTP greedy output parity, no more than a 10 percent speed regression against the fork, at least a 3 percent upstream MTP gain, and 11 of 12 long-context retrieval successes. The preferred profile is `q8q5`; `iq4nl` is the capacity fallback.

## Quality and final gates

Download WikiText and run identical perplexity jobs:

```bash
./scripts/qwen38/download-wikitext.sh
./scripts/qwen38/run-perplexity.sh
```

The quality gate allows at most a 1 percent PPL increase from TurboQuant3.

After a profile passes the short, memory, long-context, and quality gates, keep its MTP service on port 8882 and run:

```bash
python3 scripts/qwen38/soak.py \
  --base-url http://127.0.0.1:8882 \
  --hours 24 \
  --minimum-requests 100 \
  --output qwen38/results/soak.json

BASE_URL=http://127.0.0.1:8882/v1 ./scripts/qwen38/codex-smoke.sh
```

The Codex test uses a disposable repository and the server's Responses API. It must implement a small function, execute its tests, and leave the supplied test unchanged. The provider is passed with per-command configuration and does not modify the user's Codex configuration.

Generate the single decision report after every gate has finished:

```bash
python3 scripts/qwen38/finalize-report.py \
  --matrix-dir qwen38/results/MATRIX_DIR \
  --quality qwen38/results/PPL_DIR/quality.json \
  --soak qwen38/results/soak.json \
  --codex-status qwen38/results/codex-smoke/status.txt \
  --output qwen38/results/comparison-report.md
```

Before finalizing, refresh the environment record used by the report:

```bash
python3 scripts/qwen38/record-environment.py \
  --output qwen38/results/environment.json
```

## Rollout and rollback

Do not bind upstream to port 8881 or add a restart-enabled production service until `comparison-report.md` says `PASS - switch to upstream`. Then copy the selected cache and MTP settings into a production compose override, stop the old service, and bind upstream to 8881. Keep the old image, compose file, cached Qwen3.6 artifacts, and slot state intact until the upstream service completes another 24 hours on port 8881.

If upstream fails only KV capacity or quality, use this release as the base for a CUDA-only TurboQuant3 KV port. Do not port the fork's custom Qwen NextN, multimodal, server, or weight-quantization code. If a core Qwen, MTP, or vision gate fails in both engines, leave Qwen3.6 in production.
