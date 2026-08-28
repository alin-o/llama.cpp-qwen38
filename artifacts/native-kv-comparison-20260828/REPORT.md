# Native KV cache comparison: Qwen3.8 27B and Tiel 35B-A3B

Date: 2026-08-28 UTC

## Result

Turbo3 and Turbo4 are now allocated and executed as their native KV types. They are no longer mapped to q8_0. The performance comparison below is type-matched: unified q8_0 versus paged q8_0, unified Turbo3 versus paged Turbo3, and unified Turbo4 versus paged Turbo4.

FA-off rows are not benchmarkable with K and V both set to any of these three formats. All three V formats are quantized, and this source tree rejects quantized V when Flash Attention is disabled. These cells are marked unsupported instead of silently changing the cache format.

| Model | KV type | FA | KVU pp tok/s | KVP pp tok/s | KVP/KVU pp | KVU tg tok/s | KVP tg tok/s | KVP/KVU tg | Result |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.8 27B | q8_0 | on | 3070.76 | 2713.54 | 88.4% | 53.12 | 29.81 | 56.1% | measured |
| Qwen3.8 27B | Turbo3 | on | 2970.02 | 2645.76 | 89.1% | 51.46 | 27.95 | 54.3% | measured |
| Qwen3.8 27B | Turbo4 | on | 3001.50 | 2576.96 | 85.9% | 50.95 | 27.23 | 53.4% | measured |
| Qwen3.8 27B | q8_0 | off | - | - | - | - | - | - | unsupported: quantized V requires FA |
| Qwen3.8 27B | Turbo3 | off | - | - | - | - | - | - | unsupported: quantized V requires FA |
| Qwen3.8 27B | Turbo4 | off | - | - | - | - | - | - | unsupported: quantized V requires FA |
| Tiel 35B-A3B | q8_0 | on | 8989.47 | 9177.27 | 102.1% | 188.95 | 67.45 | 35.7% | measured |
| Tiel 35B-A3B | Turbo3 | on | 9812.10 | 8776.74 | 89.4% | 177.43 | 59.63 | 33.6% | measured |
| Tiel 35B-A3B | Turbo4 | on | 9789.25 | 8554.93 | 87.4% | 174.35 | 59.22 | 34.0% | measured |
| Tiel 35B-A3B | q8_0 | off | - | - | - | - | - | - | unsupported: quantized V requires FA |
| Tiel 35B-A3B | Turbo3 | off | - | - | - | - | - | - | unsupported: quantized V requires FA |
| Tiel 35B-A3B | Turbo4 | off | - | - | - | - | - | - | unsupported: quantized V requires FA |

The paged path is close to unified prefill at this 1,024-token workload: 85.9-89.1% for Qwen, 87.4-89.4% for Tiel Turbo3/4, and 102.1% for Tiel q8_0. Decode remains substantially slower in paged mode: 53.4-56.1% of unified for Qwen and 33.6-35.7% for Tiel.

## Apples-to-apples method

- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB, driver 595.84.
- Source: commit `43752b821e46b0404dee86a1fe080e8def425d17` plus the working-tree native-type and comparison changes described below.
- Models:
  - `/models/qwen38/Qwen3.8-27B-UD-Q4_K_S.gguf`
  - `/models/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S.gguf`
- Both paths: full CUDA0 offload, split mode none, FA on, K and V set to the same named format, batch 1,024, ubatch 1,024, one sequence, 1,024 prompt tokens, and 8 decoded tokens.
- Unified (`kvu`) prefill: `llama-bench` pp1024. Twelve in-context repetitions were collected because Tiel q8_0 ramps slowly after model load; the first nine are warmup and the median of the final three is reported.
- Unified (`kvu`) decode: `llama-bench` with `n_depth=1024` and `n_gen=8`. The first repetition is warmup and the median of the final three is reported. This matters: a standalone tg8 test at depth zero is not comparable to paged decode after a 1,024-token prompt.
- Paged (`kvp`): `llama-paged`, block size 32, 48 GPU blocks, 16 CPU blocks, zero watermark. Four in-context repetitions were collected; the first is warmup and the median of the final three is reported.
- The paged prompt log confirms exactly 1,024 prompt tokens for both tokenizers.

Machine-readable results are in `results.csv`. The exact commands are reproducible through `run.sh`; raw JSON and logs are under `raw/`.

## Proof that Turbo is native

The two former aliases were removed from `common/common.cpp` and `src/llama-kv-cache-paged.cpp`. Only the existing f16-to-q8_0 paged default remains. `tests/test-paged-kv.cpp` now asserts that Turbo3 and Turbo4 tensors retain their requested type.

The paged block sizes in the final logs independently prove that the formats are not q8_0 aliases:

| Model | q8_0 bytes/block | Turbo3 bytes/block | Turbo4 bytes/block |
|---|---:|---:|---:|
| Qwen3.8 27B | 4,456,448 | 1,638,400 | 2,228,224 |
| Tiel 35B-A3B | 1,392,640 | 512,000 | 696,320 |

If Turbo3/4 were still mapped to q8_0, all three values within a model would be identical. Unified JSON also records `type_k` and `type_v` as `turbo3` or `turbo4` in the corresponding rows.

## FA off

`src/llama-context.cpp` rejects a disabled FA setting whenever `ggml_is_quantized(type_v)` is true. Representative q8_0 `kvu` and `kvp` attempts were run for both models and exited 1; paged logs contain the exact message `quantized V cache requires flash_attn to be enabled`. The rule applies identically to Turbo3 and Turbo4 because they are quantized types.

Consequently, a numeric FA-on versus FA-off comparison for q8_0/Turbo3/Turbo4 K+V is not a benchmark gap; it is an unsupported configuration. Producing FA-off numbers requires either:

1. benchmarking a different configuration such as quantized K with f16 V, which is not the same requested K/V format; or
2. implementing a non-FA attention path that supports these quantized V formats.

No fallback or substitute result is used in the table.

## Same-format quality check

The corrected E2E driver compares each paged format to the same unified format. It no longer compares unified f16 to every paged quantization and attributes the combined delta to paging.

| Model | KV type | Same-format result |
|---|---|---|
| Qwen3.8 27B | q8_0 | pass; PPL 1.908666 KVU vs 1.909536 KVP, ratio 1.000456; 0/16 greedy mismatches |
| Qwen3.8 27B | Turbo3 | pass; PPL 2.041686 KVU vs 2.024161 KVP, ratio 0.991416; 0/16 greedy mismatches |
| Qwen3.8 27B | Turbo4 | strict top-k diagnostic failure at forced step 1: same argmax, top-5 overlap 3/5, 0/16 greedy mismatches; PPL not reached |
| Tiel 35B-A3B | q8_0 | exact-output mismatch at forced step 7: different argmax on a 0.021-logit near-tie, top-5 overlap 4/5; subsequent greedy output diverges |
| Tiel 35B-A3B | Turbo3/4 | not reached because the driver is fail-fast after the q8_0 failure |

This resolves the original attribution error but does not make every native format numerically identical. Qwen q8_0 and Turbo3 pass the current strict gate. Qwen Turbo4 keeps the same argmax and all 16 greedy tokens, but the test terminates because two low-ranked top-5 candidates differ at one step. That is a strict diagnostic failure, not a demonstrated generation failure.

Tiel q8_0 has a reproducible exact-output difference, but the available evidence does not establish a cache implementation bug. At the first mismatch, unified's two leading logits differ by only 0.021 (16.289 versus 16.268), smaller than the raw-logit drift expected from the different attention reduction orders. The paged result swaps those candidates while retaining 4/5 top-5 overlap. The later greedy mismatches cascade from that first changed token. Because the fail-fast driver did not calculate perplexity, the quality verdict is inconclusive pending a longer forced-token PPL/KL evaluation.

## Verification

- Canonical `.agent/verify.sh`: passed. CPU E2E and paged storage tests pass; only the four documented CPU baselines failed. The CUDA backend test hit its documented gated-delta-net/paged-attention baseline assertion and was accepted by the verifier.
- Native q8_0, Turbo3, and Turbo4 performance runs completed for both models and both `kvu`/`kvp` paths with FA on.
- FA-off representative runs failed as required by the global quantized-V validation.
- Corrected real-model E2E logs are retained for both models.

## Files

- `REPORT.md`: this report.
- `results.csv`: full 24-row performance/support matrix.
- `run.sh`: reproducible benchmark driver.
- `raw/*-kvu-fa-on-pp.json`: unified prefill samples.
- `raw/*-kvu-fa-on-tg.json`: unified depth-1,024 decode samples.
- `raw/*-kvp-fa-on.log`: paged warmup and measured samples.
- `raw/*-fa-off.log`: representative unsupported-configuration evidence.
- `raw/*-quality-e2e.log`: corrected same-format quality evidence.
