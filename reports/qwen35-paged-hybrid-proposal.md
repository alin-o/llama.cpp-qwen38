## Proposal: Qwen35 hybrid-memory integration with the paged scheduler

**What**: Add a paged attention-cache path for Qwen35 hybrid recurrent/attention memory so the existing paged scheduler can run Qwen3.8 27B while preserving the recurrent-state component.

**Why**: The exact-Qwen benchmark matrix showed that `-kvp` never constructs a paged cache for Qwen35, so scheduler initialization fails before any paged-attention kernel can run.

**Where**: Primarily `src/llama-model.cpp`, `src/llama-memory-hybrid.*`, `src/llama-kv-cache-paged.*`, `src/llama-paged-scheduler*`, the associated graph inputs, and existing paged tests. Follow-up validation uses the existing Qwen benchmark artifacts and commands.

**Evidence**: `benchmark-results/qwen38-27b-standalone-20260825/report.md` records exact-Qwen failures at 4k and 32k with `context does not have a paged KV cache`, plus an 8,208 MiB hybrid-KV allocation OOM at 128k. Its code audit shows Qwen35 constructs `llama_memory_hybrid` instead of `llama_kv_cache_paged`; the report concludes this integration is a prerequisite for the existing paged CUDA kernel task.

### Builder: PASS

- This is a measured functional gap: paged throughput is currently zero for the stated Qwen3.8 target. The task enables an already-planned kernel optimization and reuses the existing hybrid memory, paged cache, scheduler, and benchmark infrastructure rather than creating a parallel subsystem.

### Critic: PASS

- Hybrid recurrent state and paged attention state have different lifetime and rollback rules, so a careless integration could corrupt sequences or increase VRAM. The task will explicitly preserve the recurrent component, extend existing tests, keep the unified path unchanged, and require fail-fast behavior for unsupported combinations until they are proven safe.

### Operator: PASS

- The repository already has a warm CUDA build, paged tests, exact-Qwen weights, RTX 4090 access, and reproducible benchmark artifacts. Verification can use `.agent/verify.sh`, targeted existing tests, and the recorded Qwen matrix; rollback is removal/disablement of the new paged-hybrid selection while leaving the current unified hybrid path intact.

### Verdict: CREATE

The measured blocker, existing infrastructure, and direct dependency from the queued CUDA kernel work justify a dedicated prerequisite task.
