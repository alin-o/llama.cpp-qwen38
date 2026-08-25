# Tiel-Coder standalone KV benchmark (2026-08-25)

These are prototype standalone measurements. They are not `llama-server` E2E or HTTP throughput measurements.

## Boundary and workload

- GPU: NVIDIA GeForce RTX 4090, 24564 MiB total; 24083 MiB free immediately before the matrix
- Model: `/models/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S.gguf`
- Unified driver: `build-verify/bin/llama-continuous-batch`
- Paged driver: `build-verify/bin/llama-paged`
- Direct timing boundary: the drivers' inference loops, excluding model load
- Workload: 8 fixed built-in prompts, 8 parallel sequences, 128 generated tokens per sequence, `--ignore-eos`
- Sampling: seed 1234, temperature 0
- Compute: CUDA device 0, full offload, split mode none, flash attention on
- Batch: `-b 512 -ub 512`
- Context values are allocated shared KV pool sizes, not prompt lengths. The fixed prompts total 76 tokens.
- GPU telemetry was sampled every 200 ms with `nvidia-smi`.

Unified command template:

```bash
build-verify/bin/llama-continuous-batch \
  -m /models/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S.gguf \
  -kvu -no-kvp -ngl 999 -sm none -mg 0 \
  -ns 8 -np 8 -n 128 -c CONTEXT -b 512 -ub 512 \
  -ctk KV_TYPE -ctv KV_TYPE -fa on -s 1234 --temp 0 --ignore-eos
```

Paged command template:

```bash
build-verify/bin/llama-paged \
  -m /models/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S.gguf \
  -kvp -no-kvu -ngl 999 -sm none -mg 0 \
  -ns 8 -np 8 -n 128 -c CONTEXT -b 512 -ub 512 \
  -ctk f16 -ctv f16 -fa on -s 1234 --temp 0 --ignore-eos
```

## Unified results

Each successful arm generated exactly 1024 tokens. Prompt throughput is an approximation computed as 76 prompt tokens divided by average TTFT; it includes first-token sampling and is not a dedicated long-prompt prefill benchmark.

| Context | KV | Prompt wall (avg TTFT, ms) | Approx prompt tok/s | Decode aggregate tok/s | TPOT ms/token | Peak VRAM MiB | Peak power W |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4096 | f16 | 147.5 | 515.3 | 515.76 | 14.5 | 20953 | 307.55 |
| 4096 | q8_0 | 148.7 | 511.1 | 508.25 | 14.7 | 20919 | 304.86 |
| 4096 | turbo3 | 160.6 | 473.2 | 507.77 | 14.6 | 20891 | 308.05 |
| 4096 | turbo4 | 151.3 | 502.3 | 504.94 | 14.8 | 20897 | 307.01 |
| 32768 | f16 | 145.6 | 522.0 | 512.68 | 14.6 | 21509 | 310.87 |
| 32768 | q8_0 | 156.0 | 487.2 | 502.20 | 14.8 | 21243 | 306.03 |
| 32768 | turbo3 | 155.9 | 487.5 | 510.71 | 14.6 | 21001 | 313.53 |
| 32768 | turbo4 | 147.4 | 515.6 | 508.47 | 14.7 | 21045 | 310.42 |
| 128000 | f16 | 142.1 | 534.8 | 513.24 | 14.6 | 23369 | 310.33 |
| 128000 | q8_0 | 160.7 | 472.9 | 498.85 | 14.9 | 22201 | 307.56 |
| 128000 | turbo3 | 147.0 | 517.0 | 500.71 | 14.9 | 21365 | 305.27 |
| 128000 | turbo4 | 144.9 | 524.5 | 504.11 | 14.8 | 21541 | 307.04 |

The output was deterministic across context sizes for each KV type. Output differed between KV types, so these measurements do not imply lossless quantized KV behavior.

## Paged result: unsupported for this architecture

All three f16 paged attempts (4096, 32768, and 128000 context) exited before the scheduler loop with status 1:

```text
common_fit_paged_kv_blocks: free_vram=4288.6 MiB, bytes_per_block=655360, n_gpu_blocks=5223, n_cpu_blocks=1305
llama_paged_scheduler_init: context does not have a paged KV cache.
main: Failed to initialize scheduler.
```

Tiel-Coder reports architecture `qwen35moe`. This fork classifies `LLM_ARCH_QWEN35MOE` as hybrid, and `llama_model::create_memory` routes hybrid models to `llama_memory_hybrid` before reaching the standard-attention `kv_paged` branch. Consequently, the context memory is not a `llama_kv_cache_paged`, and `llama_paged_scheduler_init` rejects it. No `prepare_batch`, `llama_decode`, scheduler `update`, swap, or recompute occurred, so there is no paged throughput number to report.

## Profiling availability

Nsight Systems and Nsight Compute launchers are present, but their target executables are not executable in this environment (`Permission denied`). No kernel-counter conclusions are claimed.

Raw `.log` files and matching `.gpu.csv` telemetry files are in this directory.
