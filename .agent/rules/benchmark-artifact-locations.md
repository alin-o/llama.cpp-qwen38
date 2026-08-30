---
trigger: model_decision
description: Use when running benchmarks, storing benchmark outputs, or writing benchmark and performance reports in this llama.cpp project.
---
# Benchmark artifact locations

- Store benchmark inputs, raw logs, profiles, CSV/JSON results, reproduction scripts, and other benchmark evidence under `/workspace/llama-cpp/benchmark-results`.
- Keep `/workspace/llama-cpp/benchmark-results` ignored by Git. Benchmark results and supporting evidence must remain local artifacts and must never be added to the repository index.
- Store user-requested benchmark, performance, and investigation reports under `/workspace/group/artifacts/reports`.
- Do not create benchmark data under `/workspace/llama-cpp/artifacts` or directly under `/workspace/group/artifacts`.
- Keep reports separate from their benchmark-data directories. Give each report a descriptive, collision-resistant filename, normally including the benchmark date.
- When moving or creating artifacts, update reproduction scripts and cross-references to use the canonical paths.
