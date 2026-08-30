# Qwen3.8 27B GSQ-RCO IQ3_XXS KV benchmark

Date: 2026-08-30 UTC

## Result

The IQ3_XXS weights load and run successfully with the unified (KVU) and paged (KVP) cache paths for q8_0, Turbo3, and Turbo4 caches.

| KV type | FA | KVU prefill tok/s | KVP prefill tok/s | KVP/KVU | KVU decode tok/s | KVP decode tok/s | KVP/KVU | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| q8_0 | on | 2814.10 | 2559.82 | 91.0% | 70.08 | 36.62 | 52.3% | measured |
| Turbo3 | on | 2800.37 | 2483.80 | 88.7% | 67.89 | 33.29 | 49.0% | measured |
| Turbo4 | on | 2792.05 | 2435.39 | 87.2% | 67.30 | 32.90 | 48.9% | measured |
| q8_0 | off | - | - | - | - | - | - | unsupported |
| Turbo3 | off | - | - | - | - | - | - | unsupported |
| Turbo4 | off | - | - | - | - | - | - | unsupported |

FA-off remains unsupported because K and V use the same quantized format. This tree rejects quantized V caches without Flash Attention; no substitute V format was used.

Compared with the Qwen3.8 27B Q4_K_S weights in the 2026-08-28 run, IQ3_XXS achieved 91.6-94.3% of unified prefill and 93.9-94.5% of paged prefill, depending on cache type. Decode was faster: 131.9-132.1% of Q4_K_S for unified and 119.1-122.8% for paged.

## Method

- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB, driver 595.84.
- Benchmark binary: build 10664, commit `43752b821`, matching the prior report's binary.
- Model: `/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf`.
- Download source: `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`, file `Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf`.
- File size: 10,094,357,632 bytes; SHA-256 `fdfcb6a29b11188956dfbfd904223588a6c1b77eb250c3e8a36e1bd269df91f7`.
- GGUF-reported model type: `qwen35 27B IQ3_S - 3.4375 bpw`.
- Both paths: full CUDA0 offload, split mode none, FA on, matching K/V cache types, batch and ubatch 1,024, one sequence, 1,024 prompt tokens, and 8 decoded tokens.
- Unified prefill: 12 in-context repetitions; median of the final three after nine warmups.
- Unified decode: depth 1,024, four repetitions; median of the final three after one warmup.
- Paged: block size 32, 48 GPU blocks, 16 CPU blocks, zero watermark, four in-context repetitions; median of the final three after one warmup.

Paged allocation reported the same native cache block sizes as the prior Q4_K_S Qwen run: 4,456,448 bytes for q8_0, 1,638,400 for Turbo3, and 2,228,224 for Turbo4.

## Evidence

- `results.csv`: machine-readable results and support status.
- `run.sh`: exact reproducible benchmark commands.
- `raw/*`: unified JSON samples, stderr logs, and paged logs.
- `pull.log`: completed Hugging Face download record.

