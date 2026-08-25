# Shared-Prefix KV Caching for llama-server

> Design phase. Server-authoritative hashing; client is a pure pointer store.
> Target: a single-GPU `llama.cpp` server.

## 1. Problem
Parallel requests to `llama-server` share large **stable** prompt segments
(harness tools, skills index, project prompts, per-stage prompts). Today every
request is self-contained, so the server re-prefills (recomputes KV) for the
shared prefix on **every** request — wasted latency and compute. The moving
part (`task_description`, `subagent_*`, `current_progress`) is the only thing
that actually changes per request.

## 2. Goals / non-goals
**Goals**
- Reuse precomputed KV for stable segments across concurrent requests.
- Recompute only when a segment's content changes.
- Lazily persist KV to disk for cross-restart recovery.
- Keep the harness request lightweight (content-less warm path).

**Non-goals**
- Change GGUF model loading / weights.
- Multi-GPU (single-GPU setup).

## 3. Roles
- **Server** — sole authority for hashing and KV. Owns/loads canonical segment
  content, computes the authoritative hash, binds KV to that hash.
- **Client (harness)** — pointer store of `{id -> hash}`. Sends
  `[id, hash] + tail`. Performs **no** tokenization, hashing, or content send
  on the warm path.

## 4. Version ids
- **Segment id** — stable, namespace-scoped, e.g.
  `project:<proj>:system_prompt`, `project:<proj>:skills_index`,
  `harness:<variant>:tools`. A logical segment differs per project/harness.
- **Version id** — **server-authored, opaque to the client.** The client never
  tokenizes or hashes; it only stores `{segment id -> version id}` and echoes it
  back. Integrity comes from construction: the server binds
  `content -> version -> KV`, and only the server touches those three. A matching
  version ⇒ reuse this segment's KV; a different version ⇒ recompute. No client
  input can serve stale KV.
- **Content-derived.** The server derives the version from the segment's
  (normalized) content, so it is recoverable on restart (recompute from the
  persisted content) instead of needing a separate persisted counter.
- **Unique per distinct KV state.** That is the only real constraint: states with
  identical tokens share a version (the whole point); states with different tokens
  get different versions. **Numerical ids must avoid collision** — use a 64-bit
  value (or a persisted monotonic counter); a short 16/32-bit number risks a
  repeat after many edits.
- **Cross-API sharing is a normalization choice.** If the server normalizes the
  same logical content identically across `chat/completions`, `responses`, and
  `messages`, one version id can cover all three. If formatting genuinely differs,
  the server just assigns different versions; the scheme does not care, because it
  is keyed on distinct KV states, not on token-hash equality.

## 5. Protocol
### 5.1 Segment registration (cold)
```
POST /segment  { "id": "<id>", "content": "<raw bytes>" }
-> 200 { "id": "<id>", "hash": "0x…" }
```
Server tokenizes, computes the authoritative hash, builds KV, binds it, lazily
persists to disk, and returns the hash. Client persists `id -> hash`.

### 5.2 Completion (warm — content-less)
```
POST /completions
{ "segments": [ { "id": "…", "hash": "0x…" }, … ], "tail": "<…>" }
```
For each segment the server compares the supplied hash to its stored hash:
- **match** → load that segment's KV pages into the request (RAM hit).
- **mismatch / unknown** → `miss` for that id; client re-registers (5.3).

Server assembles `segments + tail` KV and runs.

### 5.3 Miss / refresh
Server returns `miss` for stale/unknown ids. Client re-registers those segments
via 5.1 (sends `content` once), gets a new hash, retries 5.2.

## 6. KV assembly & execution
Server concatenates each segment's KV pages, then the tail's KV, and runs one
forward pass. Only the tail's tokens are computed; segment KV is reused.

## 7. Invalidation (content versioning)
When a segment's content changes (skill edited, tool description updated), the
server recomputes its hash → advances the version → old KV is dropped and
rebuilt once on next need. The client's stale hash produces a `miss` → refresh.
The client never serves stale content because it only replays server-issued
hashes, and the server re-checks them.

## 8. Disk persistence (lazy)
- On start the disk store is **empty**.
- Each identifier is paged from SSD → RAM on first need, at most once per start.
- SSD is durability/cold-storage, not a per-request hot cache. The one-shot
  reload latency per cold identifier is accepted.

## 9. Client registry
- Durable `{id -> hash}` map, persisted across client restarts.
- Namespaced by project/harness (composite ids).
- Refreshed on server `miss`; it is the only client state.

## 10. Safety & invariants
- KV is reused **only** when the client-supplied hash equals the server-computed
  hash for that id.
- Strong hash + token-id hashing ⇒ no practical collision ⇒ matching hash means
  same tokens ⇒ no stale KV.
- The server is the sole truth source; the client hash is a claim, not trust.

## 11. Concurrency
- Shared KV pages are **read-only** ⇒ safe for many concurrent requests.
- Page store is ref-counted; a page is evicted only when no in-flight request
  needs it (thread-safe).

## 12. Where it hooks in (llama.cpp)
- Stock `llama-server` has **no cross-request prefix KV reuse**; its
  `--prompt-cache` is single-context. This adds a content-addressed segment KV
  pool on top of the existing paged KV cache / ggml backend, plus a `/segment`
  endpoint and a `segments[]` field in the completion request (HTTP + WS).

## 13. Metrics
- Prefill latency & token cost for shared segments (target: near-zero repeat).
- Prefix cache hit rate.
- Shared-store memory footprint.
- Added per-request latency (target ~0).

## 14. Open questions
- Exact llama-server request API surface (HTTP + WS) and KV-seed hook point.
- Whether the server owns canonical content (self-serve cold) or the client must
  re-send `content` on cold — here: client sends `content` once on cold.
- Disk store layout for the lazy page cache.

## 15. Phases / tasks
1. Measure & lock design (baseline + decision sign-off).
2. Shared KV segment store in llama.cpp (content-addressed, ref-counted).
3. Server request field (`/segment` + `segments[]`); correctness tests.
4. Invalidation by content version.
5. Lazy disk persistence.
6. Harness wiring + benchmark (content-less warm path).
