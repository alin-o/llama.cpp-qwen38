# KVU versus KVP: Qwen3.8 27B and Tiel MoE

Date: 2026-08-28 UTC; updated 2026-08-30 UTC

## Performance results

The table compares native, type-matched caches: KVU q8_0 versus KVP q8_0, KVU Turbo3 versus KVP Turbo3, and KVU Turbo4 versus KVP Turbo4. Turbo3 and Turbo4 are not mapped to q8_0. The GSQ-RCO IQ3_XXS weight quantization was added on 2026-08-30 using the same benchmark binary and method.

Workload: RTX 4090, full CUDA offload, one sequence, 1,024 prompt tokens, 8 decode tokens at cache depth 1,024, batch and ubatch 1,024. Results are steady-state medians of three measured samples after warmup.

| Model | KV type | FA | KVU prefill tok/s | KVP prefill tok/s | KVP/KVU | KVU decode tok/s | KVP decode tok/s | KVP/KVU | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3.8 27B Q4_K_S | q8_0 | on | 3070.76 | 2713.54 | 88.4% | 53.12 | 29.81 | 56.1% | measured |
| Qwen3.8 27B Q4_K_S | Turbo3 | on | 2970.02 | 2645.76 | 89.1% | 51.46 | 27.95 | 54.3% | measured |
| Qwen3.8 27B Q4_K_S | Turbo4 | on | 3001.50 | 2576.96 | 85.9% | 50.95 | 27.23 | 53.4% | measured |
| Qwen3.8 27B Q4_K_S | q8_0 | off | - | - | - | - | - | - | unsupported |
| Qwen3.8 27B Q4_K_S | Turbo3 | off | - | - | - | - | - | - | unsupported |
| Qwen3.8 27B Q4_K_S | Turbo4 | off | - | - | - | - | - | - | unsupported |
| Qwen3.8 27B GSQ-RCO IQ3_XXS | q8_0 | on | 2814.10 | 2559.82 | 91.0% | 70.08 | 36.62 | 52.3% | measured |
| Qwen3.8 27B GSQ-RCO IQ3_XXS | Turbo3 | on | 2800.37 | 2483.80 | 88.7% | 67.89 | 33.29 | 49.0% | measured |
| Qwen3.8 27B GSQ-RCO IQ3_XXS | Turbo4 | on | 2792.05 | 2435.39 | 87.2% | 67.30 | 32.90 | 48.9% | measured |
| Qwen3.8 27B GSQ-RCO IQ3_XXS | q8_0 | off | - | - | - | - | - | - | unsupported |
| Qwen3.8 27B GSQ-RCO IQ3_XXS | Turbo3 | off | - | - | - | - | - | - | unsupported |
| Qwen3.8 27B GSQ-RCO IQ3_XXS | Turbo4 | off | - | - | - | - | - | - | unsupported |
| Tiel 35B-A3B MoE | q8_0 | on | 8989.47 | 9177.27 | 102.1% | 188.95 | 67.45 | 35.7% | measured |
| Tiel 35B-A3B MoE | Turbo3 | on | 9812.10 | 8776.74 | 89.4% | 177.43 | 59.63 | 33.6% | measured |
| Tiel 35B-A3B MoE | Turbo4 | on | 9789.25 | 8554.93 | 87.4% | 174.35 | 59.22 | 34.0% | measured |
| Tiel 35B-A3B MoE | q8_0 | off | - | - | - | - | - | - | unsupported |
| Tiel 35B-A3B MoE | Turbo3 | off | - | - | - | - | - | - | unsupported |
| Tiel 35B-A3B MoE | Turbo4 | off | - | - | - | - | - | - | unsupported |

FA-off is unsupported for all rows because K and V use the same quantized format. This tree rejects every quantized V cache unless Flash Attention is enabled. No substitute V format was used.

### IQ3_XXS weight-quantization comparison

Against the prior Q4_K_S Qwen weights, GSQ-RCO IQ3_XXS reached 91.6-94.3% of KVU prefill speed and 93.9-94.5% of KVP prefill speed. Decode improved to 131.9-132.1% of Q4_K_S for KVU and 119.1-122.8% for KVP. This comparison changes the model weight quantization only; cache types and benchmark settings remain type-matched and identical.

## Quality results and interpretation

The quality results in this section apply to the original Q4_K_S Qwen run. The 2026-08-30 IQ3_XXS addition is a performance benchmark only; no IQ3_XXS quality-equivalence result is claimed.

The E2E check compares each KVP format with KVU using the same format. For the first 16 forced-token steps, its strict per-step rule requires both:

- the same top-1 token; and
- at least four common tokens between the two top-5 sets.

It also compares 16 independently greedy tokens and, if the strict per-step rule does not stop the test first, requires KVP perplexity to be no more than 2% worse than KVU.

| Model | KV type | Observation | Interpretation |
|---|---|---|---|
| Qwen3.8 27B | q8_0 | 0/16 greedy mismatches; PPL 1.908666 KVU vs 1.909536 KVP, ratio 1.000456 | pass |
| Qwen3.8 27B | Turbo3 | 0/16 greedy mismatches; PPL 2.041686 KVU vs 2.024161 KVP, ratio 0.991416 | pass |
| Qwen3.8 27B | Turbo4 | 0/16 greedy mismatches; at forced step 1 both choose token 3833, but top-5 overlap is 3/5 | strict diagnostic failure; no demonstrated generation failure |
| Tiel 35B-A3B MoE | q8_0 | first top-1 mismatch at forced step 7; 9/16 greedy tokens differ after that point | exact-output mismatch, but bug status is inconclusive |
| Tiel 35B-A3B MoE | Turbo3/4 | not run because the E2E driver stops at the q8_0 failure | no quality result |

### What the Qwen Turbo4 result means

"Diagnostic gate" refers only to the test rule above. At forced step 1, KVU and KVP have the same top three tokens in the same order and the same top-1 token. Their fourth and fifth candidates differ, so overlap is 3/5 instead of the required 4/5. All 16 greedy tokens match. The test stops before calculating perplexity.

This is a strict test failure, but the available result does not show a user-visible generation error. It should be recorded as "inconclusive under the strict top-5 gate," not as a Turbo4 correctness bug.

### Is the Tiel q8_0 result a bug?

Not proven. It is a real reproducible difference between the two execution paths, but the first mismatch is a near-tie:

- KVU: token 13293 = 16.289, token 2197 = 16.268; margin 0.021.
- KVP: token 2197 = 16.195, token 13293 = 16.090.
- Four of the top five candidates still overlap.

The paged CUDA attention path and unified Flash Attention use different reduction orders. The E2E test itself documents expected raw-logit drift of roughly 0.05-0.5; that is larger than the KVU top-1 margin at this step. A small numerical change can therefore swap the winner. Once greedy decoding selects the other token at step 7, the subsequent 9/16 mismatch count is cascading sequence divergence, not nine independent cache errors.

The lower-level paged-attention CUDA correctness test also covers head dimension 128 (the Tiel path) against the CPU reference and passes its relative-error bound. Taken together, current evidence does not establish layout corruption, wrong GQA head mapping, or a q8_0 storage bug. It does establish that KVP is not bit-for-bit/top-1 equivalent for this prompt.

To decide quality acceptance, the next useful check is a non-fail-fast evaluation over a larger corpus using forced tokens, reporting perplexity/KL divergence and top-k agreement. Until then, classify Tiel q8_0 as **quality equivalence inconclusive**, not **confirmed bug**.

## Reproducibility and evidence

- Machine-readable matrix: `native-kv-comparison-20260828/results.csv`
- Rerun script: `native-kv-comparison-20260828/run.sh`
- Raw benchmark, FA-off, and E2E logs: `native-kv-comparison-20260828/raw/`
- Detailed implementation and verification notes: `native-kv-comparison-20260828/REPORT.md`
- IQ3_XXS machine-readable results: `gsq-rco-iq3-xxs-benchmark-20260830/results.csv`
- IQ3_XXS rerun script and raw logs: `gsq-rco-iq3-xxs-benchmark-20260830/run.sh`, `gsq-rco-iq3-xxs-benchmark-20260830/raw/`
- IQ3_XXS detailed benchmark report: `gsq-rco-iq3-xxs-benchmark-20260830/REPORT.md`

The paged allocation logs show different bytes per block for q8_0, Turbo3, and Turbo4, independently confirming native storage:

| Model | q8_0 bytes/block | Turbo3 bytes/block | Turbo4 bytes/block |
|---|---:|---:|---:|
| Qwen3.8 27B | 4,456,448 | 1,638,400 | 2,228,224 |
| Tiel 35B-A3B MoE | 1,392,640 | 512,000 | 696,320 |
