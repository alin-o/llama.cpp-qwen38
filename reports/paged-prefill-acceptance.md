# Paged head-dim-256 prefill acceptance evidence

Date: 2026-08-27

Source commit: `d7b476213`

GPU: NVIDIA GeForce RTX 4090, compute capability 8.9, 24082 MiB VRAM.

The benchmark binary included an opt-in `LLAMA_PAGED_REPETITIONS` evidence-only harness change. The harness destroys the scheduler and clears llama memory between repetitions so each workload starts with empty KV state while the model and CUDA runtime remain warm. It is not retained in production source; the exact patch is preserved as `repetition-harness.patch` with the raw evidence.

Raw artifacts are under `benchmark-results/paged-prefill-head256-20260827-v2/`:

- `paged-raw.csv`: all four paged observations per configuration; repetition 1 is the discarded warmup.
- `paged-medians.csv`: median of paged repetitions 2-4 and explicit capacity status.
- `unified-raw.csv`: llama-bench raw samples after its built-in warmup.
- `unified-medians.csv`: median unified q8_0 pp/tg rates.
- `commands.txt`: exact command for every paged configuration.
- `repetition-harness.patch`: exact temporary repeated-run harness applied to source commit `d7b476213`.
- `SHA256SUMS`: checksums for every evidence artifact.
- `raw/`: unabridged application, oracle, capacity-failure, unified JSON, and profiler output.

## Exact models and prompt

- Qwen: `/models/qwen38/Qwen3.8-27B-UD-Q4_K_S.gguf`, 15,358,213,024 bytes, Q=24, KV=4, GQA ratio 6, head dimension 256.
- Tiel: `/models/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S.gguf`, 21,792,045,600 bytes, Q=16, KV=2, GQA ratio 8, head dimension 256.
- Prompt source: `reports/paged-4096-prompt.txt`, repeated ASCII ` x` tokens. Each matrix command reads exactly `2 * per_request_tokens` bytes. Application logs confirm the requested aggregate prompt token count for every executed row.

## Unified-f16 quality oracle versus paged q8_0

Exact commands:

```sh
TMPDIR="$PWD/build-verify-cuda/tmp" build-verify-cuda/bin/test-paged-kv-e2e -m /models/qwen38/Qwen3.8-27B-UD-Q4_K_S.gguf -ngl 99 -ctk q8_0 -ctv q8_0
TMPDIR="$PWD/build-verify-cuda/tmp" build-verify-cuda/bin/test-paged-kv-e2e -m /models/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S.gguf -ngl 99 -ctk q8_0 -ctv q8_0
```

`run_non_paged()` constructs the unified reference without overriding the default f16 K/V cache. `run_paged()` explicitly uses the parsed q8_0 K/V types. The test forces the unified token sequence through paged q8_0, compares logits/top-k and greedy tokens, and computes perplexity with a maximum allowed ratio of 1.10.

| Model | Unified f16 PPL | Paged q8_0 PPL | Ratio | Result |
| --- | ---: | ---: | ---: | --- |
| Qwen3.8-27B | 2.648921 | 2.651827 | 1.001097 | pass |
| Tiel-Coder-35B-A3B-MTP | 2.272060 | 2.408728 | 1.060152 | pass |

Raw logs: `raw/oracle-qwen.log` and `raw/oracle-tiel.log`.

The same runs reset the CUDA tiled-prefill launch counter, assert a head-dim-256 tiled launch after prefill, reset it again, execute a scheduler-produced decode-only batch, and assert that the count remains zero. Both exact-model runs end with `test-paged-kv-e2e: PASSED`.

## Warmed benchmark matrix

Paged matrix dimensions:

- Models: Qwen and Tiel.
- Aggregate prompt tokens: 64, 256, 1024, 4096.
- Paged block sizes: 16, 32, 64.
- Ubatch sizes: every power of two from 256 through the aggregate prompt size.
- Request shapes: `1 x P` and equal-total-token `2 x (P/2)`.
- Cache: q8_0 K/V.
- Decode: 8 tokens per request.
- Repetitions: one warmup plus three measured runs in a single source-matched process; reported value is the measured median.

This is 120 paged configurations and 480 attempted repetition slots. The outcome is:

| Status | Configurations | Meaning |
| --- | ---: | --- |
| executed | 54 | Four complete pp/tg observations; median is reported. |
| scheduler rejection | 60 | Ubatch is smaller than the per-request prompt; no prompt/decode tokens ran and no throughput is reported. |
| capacity failure | 6 | Tiel with ubatch 4096 cannot allocate the 3976.11 MiB CUDA compute buffer. |

The 4096-token Tiel surface is not omitted: all lower ubatches are recorded, and `2 x 2048` executes for every block size at ubatch 2048. The single-request ubatch-4096 and two-request ubatch-4096 rows preserve the exact CUDA OOM output rather than presenting zeros as performance.

Unified q8_0 comparison uses `llama-bench` on the same source, GPU, model, prompt lengths, ubatches, and 8-token decode. llama-bench performs its built-in warmup and emits three raw `samples_ts` values. There are 40 unified prompt/ubatch medians. Unified has no paged block-size dimension or equivalent simultaneous multi-request mode, so paired ratios below use the single-request rows where the surfaces match.

Representative single-request medians:

| Model | Prompt | Block | Ubatch | Paged pp tok/s | Unified pp tok/s | Paged/unified | Paged tg tok/s | Unified tg tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen | 64 | 32 | 256 | 1653.06 | 1846.75 | 0.895 | 45.83 | 50.30 |
| Qwen | 256 | 32 | 256 | 2793.84 | 2945.80 | 0.948 | 41.62 | 50.30 |
| Qwen | 1024 | 32 | 1024 | 2705.42 | 2959.73 | 0.914 | 29.39 | 49.58 |
| Qwen | 4096 | 32 | 4096 | 1828.26 | 2664.88 | 0.686 | 13.54 | 50.02 |
| Tiel | 64 | 32 | 256 | 3219.96 | 1081.60 | 2.977 | 134.88 | 168.62 |
| Tiel | 256 | 32 | 256 | 6910.33 | 1650.62 | 4.187 | 110.10 | 168.62 |
| Tiel | 1024 | 32 | 1024 | 9145.39 | 10005.10 | 0.914 | 65.52 | 163.10 |

For comparison with the pre-tiled scalar-fallback report, Qwen 4096 pp rises from 54.22 to 1828.26 tok/s (33.7x), and Tiel 1024 pp rises from 341.76 to 9145.39 tok/s (26.8x). These results are tied to the tiled-kernel profiler name below; scalar-fallback rows are not included.

## Raw profiler attribution

Representative valid workload for both models: 1024 prompt tokens, block size 32, ubatch 1024, one request, q8_0 K/V, 8 decode tokens.

Profile command form:

```sh
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none -o OUTPUT build-verify-cuda/bin/llama-paged -m MODEL -dev cuda0 -ngl 99 -sm none -mg 0 -c 2048 -b 1024 -ub 1024 -n 8 -p PROMPT -kvp -ctk q8_0 -ctv q8_0 -ngpub 48 -ncpub 16 -kvbls 32 --kv-paged-watermark 0 -ns 1 -np 1 --timing --perf --temp 0 -s 1234
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv --output - OUTPUT.nsys-rep
```

| Model | Whole prefill wall ms | Tiled prefill kernel | Launches | Kernel total ms | Kernel / whole prefill | Whole decode wall ms | Decode kernel total ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen | 411.638 | `paged_attention_prefill_mma_kernel<256>` | 16 | 56.652 | 13.76% | 284.112 | 13.398 |
| Tiel | 277.853 | `paged_attention_prefill_mma_kernel<256>` | 10 | 23.316 | 8.39% | 137.205 | 8.248 |

The Nsight kernel sums report the tiled prefill kernel as 14.1% of all captured Qwen GPU-kernel time and 19.1% of all captured Tiel GPU-kernel time. The table uses the stricter attribution requested here: kernel time divided by the application's whole prefill wall time, leaving the rest attributable to other model kernels, transfers, graph work, and synchronization.

Raw profiler artifacts:

- `raw/profiler/qwen-p1024-bs32-ub1024.nsys-rep`
- `raw/profiler/qwen-cuda-gpu-kern-sum.csv`
- `raw/profiler/qwen-app.log`
- `raw/profiler/tiel-p1024-bs32-ub1024.nsys-rep`
- `raw/profiler/tiel-cuda-gpu-kern-sum.csv`
- `raw/profiler/tiel-app.log`
