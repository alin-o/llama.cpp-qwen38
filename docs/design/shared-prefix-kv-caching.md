# Cumulative continuation checkpoints for llama-server

Status: Phase 1 design contract, 2026-09-03. This document is normative for
Phases 2-6. The measured deployment is one NVIDIA RTX 4090, one CUDA device,
one `llama-server`, Qwen3.8 27B Q4_K_S, paged Turbo4 target KV, embedded MTP
with at most three draft tokens, and continuous batching. Multi-GPU, tensor
split, remote devices, and CPU model/KV offload are outside the contract.

## 1. Required invariant

A reusable object is the complete continuation state after an exact cumulative
rendered-token prefix:

```text
checkpoint(A)
checkpoint(A -> B)
checkpoint(A -> B -> C)
```

The store must never compute `KV(B)` in isolation and attach it after an
arbitrary `A`. A page may be physically shared by descendant records, but its
semantic identity always includes its complete ancestry. Similar source JSON,
similar text, a marker match, or a digest match alone is not proof of reuse.

The accepted 63-prompt CarapaBoard trace contains only a three-token universal
prefix, but its useful branches have substantial exact overlap. At 64-token
pages, 665,216 of 899,152 prompt tokens were avoidable against the best prior
request. Separately retained snapshots used 14,019 complete pages, while the
cumulative prefix trie had 3,625 unique pages. The exact 19,322-token Codex
prefix occurred in 13 captures and occupies 301 pages. The store therefore
supports multiple roots and branches; it is not a single global prompt cache.

## 2. Identity

There are two deliberately different identifiers:

- `version_id` is an opaque, random, server-issued 128-bit base64url value for
  one immutable logical registration. It is a client pointer, not a content
  hash and not correctness evidence.
- `checkpoint_key` is an internal SHA-256 digest. It is never accepted from a
  client as proof that state is reusable.

The only namespace in this single-user deployment is the process-local
`default` namespace. The namespace remains explicit in protocol and identity
so accidental cross-instance mixing is impossible; there is no tenant,
organization, or ACL product in this feature.

Registrations are immutable. Creating a new logical name requires that no
current version exists. Replacing it uses compare-and-swap with the exact
current `expected_version_id`. A successful replacement creates a new
`version_id`, retires the old version for new lookup, and leaves existing pins
valid. Concurrent writers with the same expected version have exactly one
winner; the rest receive `409 prefix_cache_conflict`.

The checkpoint key is SHA-256 over a length-delimited canonical binary encoding
of:

1. checkpoint schema version and namespace;
2. the predecessor checkpoint key, or 32 zero bytes for a root;
3. the complete execution fingerprint below; and
4. the cumulative rendered token count and every signed 32-bit token ID in
   little-endian order.

The ordered `(segment_id, version_id)` lineage maps logical registrations to
this record and is retained as record metadata, but random public versions are
not hashed into semantic state identity. Two registrations may therefore
deduplicate to one physical record only when predecessor, exact cumulative
tokens, namespace, and execution fingerprint are all identical.

After digest lookup, the server compares the candidate's retained token count
and every exact token ID with the request prefix. A mismatch is a closed miss,
increments `prefix_checkpoint_equality_mismatch_total`, and never selects the
record. This second check is mandatory even though SHA-256 collision is not an
operational expectation.

The execution fingerprint binds every input that can alter continuation or
storage compatibility:

- model file identity and weights digest, architecture, vocabulary/tokenizer,
  chat-template source digest, and template arguments;
- active adapters, aLoRA configuration and activation boundary, and other
  model modifiers;
- RoPE, position, attention, flash-attention, causal, context, batch, and
  microbatch settings that affect results or graph shape;
- target and draft KV types, page geometry, physical backend/device, and
  hybrid/recurrent layout;
- embedded-MTP implementation, draft limit, sampler/speculative state schema,
  and draft model identity;
- llama.cpp serialization schema, ABI-relevant build identity, and endianness.

Performance-only attention dispatch buckets are recorded for graph selection,
but are not part of semantic identity. The dispatch bucket is selected from the
total populated depth after attachment, never from suffix length alone.

## 3. Immutable record and complete state

Each record contains or references:

- its key, predecessor, ordered logical lineage, lifecycle, pins, last access,
  and persistence locator;
- the exact cumulative token IDs used for post-digest equality verification;
- immutable target paged-KV blocks, including the populated extent of the
  terminal block;
- Qwen3.8 terminal hybrid/recurrent state and its layout/copy plan;
- embedded-MTP draft KV through the same continuation position and all
  speculative implementation state needed to resume deterministically;
- the execution fingerprint, integrity checks, build/schema metadata, and
  graph-dispatch metadata.

For the measured Qwen3.8 profile, target paged KV is 17,408 bytes per token and
the embedded MTP draft KV is 2,176 bytes per token (K and V combined). Runtime
recurrent storage is 598.5 MiB per active slot at draft limit three: one 149.625
MiB terminal row and three rollback rows. A checkpoint stores one terminal
149.625 MiB recurrent state, not all per-slot rollback workspaces. The request
restores that terminal state into its own runtime row before speculative
rollback rows are initialized.

The draft KV and recurrent snapshot may be held in bounded host memory while
the hot target pages occupy the paged GPU pool. They are still required parts
of a ready record. A record cannot claim a hit if either payload is missing or
incompatible. Phase 2 must make embedded-MTP draft KV physically allocatable
and restorable independently of the logical context ceiling; a statically
partitioned dense draft arena must not reintroduce `logical_context / NP`.

Exact-verification metadata is four bytes per token, plus eight bytes per
64-token page reference and a bounded record header. Implementations may
compress persisted token arrays, but the in-memory equality check must remain
exact and must not require rerendering or trusting a digest.

## 4. Partial pages and private suffixes

Shared pages are immutable. Complete pages may be referenced by any number of
records and slots. A checkpoint ending in a partial page retains that page and
its exact populated length, but no request may append into it directly.

Before a private suffix write, the scheduler allocates a private page, copies
the populated prefix of the shared terminal page, atomically switches only that
request's block table, and then permits writes. Allocation or copy failure rolls
back the private page and falls back to an earlier complete-page checkpoint or
uncached execution. The shared page never becomes transiently writable.

## 5. Public registration and warm-use schema

The three existing generation APIs gain the same optional top-level extension:

```json
{
  "prefix_cache": {
    "namespace": "default",
    "segments": [
      {
        "segment_id": "harness-tools",
        "version_id": "opaque-existing-version"
      },
      {
        "segment_id": "project-rules",
        "register": {
          "expected_version_id": "opaque-old-version-or-null",
          "content": {}
        }
      }
    ]
  }
}
```

Exactly one of `version_id` and `register` is present per segment. On first
creation `expected_version_id` is JSON null; replacement supplies the current
value. Ordered segment IDs must be unique within one request. The namespace is
optional and defaults only to `default`; any other value is rejected.

`register.content` is API-native stable input:

- Chat Completions: ordered `messages` fragments and optional ordered `tools`.
- Responses: ordered `input` items and optional ordered `tools`.
- Anthropic Messages: ordered `system` blocks, `messages` fragments, and
  optional ordered `tools`.

Concrete cold requests use these shapes (payload content abbreviated):

```http
POST /v1/chat/completions
{
  "model": "qwen38",
  "prefix_cache": {"segments": [{
    "segment_id": "codex-system-tools",
    "register": {"expected_version_id": null,
                 "content": {"messages": [{"role": "system", "content": "..."}],
                             "tools": [{"type": "function", "function": {"name": "..."}}]}}
  }]},
  "messages": [{"role": "user", "content": "private tail"}]
}
```

```http
POST /v1/responses
{
  "model": "qwen38",
  "prefix_cache": {"segments": [{
    "segment_id": "codex-system-tools",
    "register": {"expected_version_id": null,
                 "content": {"input": [{"role": "system", "content": "..."}],
                             "tools": [{"type": "function", "name": "..."}]}}
  }]},
  "input": [{"role": "user", "content": "private tail"}]
}
```

```http
POST /v1/messages
{
  "model": "qwen38",
  "prefix_cache": {"segments": [{
    "segment_id": "codex-system-tools",
    "register": {"expected_version_id": null,
                 "content": {"system": [{"type": "text", "text": "..."}],
                             "messages": [], "tools": [{"name": "..."}]}}
  }]},
  "messages": [{"role": "user", "content": "private tail"}],
  "max_tokens": 16
}
```

The corresponding warm request changes only the segment entry to
`{"segment_id":"codex-system-tools","version_id":"opaque-value"}` and omits
the registered content. A cold registration is also a normal generation: after
CAS acceptance, the server renders the registered prefix with the supplied
tail, generates normally, and publishes eligible cumulative boundaries.

The normal API body contains only the private tail when warm references are
used. Array fragments are concatenated in segment order and then followed by
the normal body arrays. Render-affecting scalar options remain on the normal
request and are forbidden inside a segment. A tools fragment may be registered
in more than one logical segment, but final ordering and duplicates are
preserved exactly; the server does not semantically merge tools.

Adapters attach non-rendered boundary tags to the normalized common chat
messages/tools while converting API-native fragments. `common_chat_templates_apply()`
is extended to return output provenance at those tags. It emits the complete
prompt once. The final rendered string is tokenized once, and a logical boundary
is promotable only if its provenance byte offset maps exactly between two final
tokens. A boundary inside a token is reported as `not_token_aligned` and is not
cached; it is never rounded. Marker strings are optional promotion hints only.

Warm content-less use succeeds only while every `(segment_id, version_id)` can
be resolved to its stored normalized fragment. Unknown, retired, or
post-restart process-local versions return HTTP 409 with
`prefix_cache_stale`; the server cannot safely reconstruct omitted content.
If content is present, cache allocation/build failure is fail-open and the
normal completion continues uncached.

Non-streaming responses add:

```json
{
  "prefix_cache": {
    "status": "hit|partial_hit|built|miss|fallback",
    "hit_tokens": 19322,
    "suffix_tokens": 256,
    "fallback_reason": null,
    "registrations": [
      {"segment_id": "project-rules", "version_id": "opaque-new-version"}
    ]
  }
}
```

Streaming responses put the same top-level extension on the first ordinary
stream frame before any generated token. Cache decisions and stale/conflict
validation finish before response headers. A late resource failure becomes a
labelled internal fallback and does not corrupt the stream. An invalid
registration returns the API's normal error envelope before streaming starts.

`Idempotency-Key` makes a registration retry idempotent for 24 hours or the
remaining process lifetime, whichever is shorter. Reusing a key with different
canonical input is a conflict. Retrying a generation without such a key may
generate again, matching current API semantics.

## 6. Lookup, build, and publication

After final tokenization the server searches registered cumulative boundaries
from longest to shortest. It pins the first ready, exactly equal, compatible
record. A nonresident persisted record enters the Phase 5 restore path. If no
record can be attached, execution starts from the nearest earlier ready record
or token zero.

There is one single-flight entry per checkpoint key, not one global cache lock.
The winner owns `building`; followers never build or publish that key. A
follower waits at most 50 ms if its deadline permits. After 50 ms it immediately
falls back to the nearest ready predecessor or uncached execution. It also
bypasses immediately when fewer than 1,024 tokens would be avoided, its request
deadline has less than 50 ms remaining, or the cache waiter count already
equals the configured maximum slots. This bound is below 2.5% of the measured
4,096-token no-cache prefill and prevents cache work from causing material
head-of-line blocking.

Publication is transactional:

1. allocate/copy every target, recurrent, draft, speculative, token, and
   compatibility payload into builder-owned state;
2. validate exact tokens, terminal positions, integrity, and predecessor;
3. publish all pointers with one `building -> ready` transition; and
4. wake waiters.

Cancellation, timeout, allocation failure, decode failure, or shutdown changes
the entry to failed and releases all builder-owned resources. No partial record
is selectable. Cancellation of a follower affects only that follower. If the
builder request is cancelled after the boundary was fully evaluated, a
detached store-owned publication may finish; otherwise it rolls back. Another
slot's pin or waiter is never invalidated by request cancellation.

## 7. Ownership, eviction, and quota

Pins are per request. One immutable record may be pinned by all slots. Retired
versions remain valid to existing pins but cannot acquire new pins. A target
page is reclaimable only after all checkpoint references and slot pins are
gone. Draft/recurrent host payloads follow the same record lifetime.

Admission uses two explicit quotas:

- GPU checkpoint target pages: 512 of the configured 2,048 pages by default.
  This holds the measured 301-page Codex root and leaves 1,536 pages (98,304
  tokens) for private active state. For a deliberately reduced 1,024-page
  NP=10 profile the default is 384 pages, sufficient for that root and leaving
  640 pages for private state.
- Host checkpoint payload: 2 GiB by default, including recurrent snapshots,
  draft KV, speculative state, exact tokens, and record metadata. The measured
  host had 57 GiB RAM and about 20 GiB available; the default is deliberately
  bounded below that headroom.

The general default is `min(512, floor(gpu_blocks * 3 / 8))`, with an automatic
minimum of 301 pages only when the pool has at least 512 pages. The scheduler
may evict below that soft quota to preserve `max(64, 8 * max_slots)` free target
pages for active progress. Quota is charged once per unique physical page and
once per unique record payload, not once per pin or logical reference.

Admission rejects a candidate that cannot fit after evicting ready unpinned
records. Eviction is LRU within the namespace, deepest descendant first on an
equal timestamp. `building`, pinned, and COW-in-progress records are ineligible.
Descendant metadata can remain only while its predecessor identity remains
known. Reclamation is asynchronous and never holds the scheduler lock while
performing host I/O or GPU copies.

At quota exhaustion the completion must continue from an earlier checkpoint
or uncached. It must not wait indefinitely, evict a pin, overcommit VRAM, fail a
valid generation, or serialize unrelated requests.

## 8. Dynamic logical context contract

The current non-unified context derives `n_ctx_seq = n_ctx / n_seq_max`.
Consequently `--ctx-size 128000 --parallel 8` exposes 16,128 tokens per slot
after padding, and `--parallel 10` exposes 12,800. This is the measured legacy
baseline, not the feature contract.

Phase 2 adds `--ctx-size-per-request N` with default zero:

- zero preserves every legacy `--ctx-size` and `--parallel` behavior;
- nonzero is accepted only for causal paged KV and is mutually exclusive with
  an explicitly supplied nonzero `--ctx-size`;
- `N` is the logical prompt plus generation ceiling for every request,
  independent of `--parallel`;
- `--parallel` is only the maximum number of active request slots; and
- `--n-gpu-blocks`, `--n-cpu-blocks`, block size, recurrent workspaces, and the
  draft physical arena define residency capacity. They do not redefine the
  logical ceiling.

For the production profile the new launch spelling is therefore
`--ctx-size-per-request 128000 --parallel 8` (or 10), not an emulated
`--ctx-size 1024000` or `1280000`. Prompt admission requires final prompt
tokens plus requested generation headroom to be below 128,000. The three draft
tokens are separately included in physical scheduling headroom.

Implementation must update these currently coupled sites as one change:

- `common_params`, argument parsing, `common_context_params_to_llama()`, and
  paged-KV fit sizing currently divide effective `n_ctx` by `n_parallel`;
- `llama_context` derives and pads `n_ctx_seq`, while model graph/RoPE/attention
  planning reads it;
- target and embedded-MTP context creation inherit the same static split;
- recurrent runtime allocation scales with `n_seq_max` and the speculative
  rollback row count;
- `server_context` copies `llama_n_ctx_seq()` into `slot.n_ctx`, then uses it
  for prompt rejection, truncation, generation stop, restore validation, and
  error reporting;
- paged scheduler construction currently passes `ctx->n_ctx()` as
  `n_seq_max_ctx`; queue/update enforce that value even though the earlier
  server admission usually enforces the smaller slot value;
- fit logic estimates blocks per sequence as `effective_n_ctx / n_parallel`;
  this must instead use the explicit per-request logical value while fitting
  only the configured physical pools;
- `/props`, `/slots`, model metadata, defaults, logs, and error responses expose
  the divided slot value today.

The new public metadata exposes `logical_context_per_request`,
`max_parallel_slots`, `continuous_batching`, and a `paged_kv` object containing
block size, GPU/CPU block counts, and physical token capacities. Legacy `n_ctx`
fields remain for compatibility and equal the logical per-request limit when
the new mode is active. Logs print both logical and physical values.

The scheduler receives the logical per-request limit explicitly. Its block
manager admits, swaps, evicts, or recomputes based on physical pools. Target and
draft graph inputs use dynamic block tables and total populated depth. No dense
allocation may multiply 128k by the slot count merely to express the logical
limit.

## 9. API and runtime hook boundary

Routes are registered in `tools/server/server.cpp`. Chat Completions enters
`post_chat_completions`; Responses first uses
`server_chat_convert_responses_to_chatcmpl()`; Anthropic Messages first uses
`server_chat_convert_anthropic_to_oai()`. All three then call
`oaicompat_chat_params_parse()` in `tools/server/server-context.cpp`.

`oaicompat_chat_params_parse()` in `tools/server/server-common.cpp` parses the
common messages/tools and calls `common_chat_templates_apply()`, placing the
single rendered prompt in `llama_params["prompt"]`. The Phase 3 adapters parse
`prefix_cache` before conversion and carry boundary tags beside, never inside,
the normalized chat objects. The common renderer returns the final prompt plus
boundary provenance.

`handle_completions_impl()` tokenizes the rendered prompt and creates the
server task. Slot start and prompt handling in `server-context.cpp` are the
last safe lookup point: final token IDs, exact boundary offsets, slot, model
contexts, and paged scheduler all exist, but no uncached suffix has executed.
After attachment it calls `llama_paged_scheduler_add_request_with_prefix()`
with the shared target block table and restores recurrent, draft-KV, and
speculative state before graph inputs are populated.

The existing `common_prompt_checkpoint` and slot-local hooks demonstrate the
required state surface: they capture/load target state, draft state, and
`common_speculative_*_state`. `llama_memory_hybrid_paged::state_write/read`
compose paged attention and recurrent state. `llama_kv_cache_paged` and
`llama_memory_recurrent` provide their respective serialization paths. These
are reference mechanisms, not the cross-request store: current checkpoints are
owned by one slot and current retained paged prefixes keep the same request ID.

## 10. Graph refresh and equivalence

Attachment completes before graph inputs are refreshed. Page IDs, block table,
context lengths, write slots, terminal recurrent row, draft mapping, and
speculative state are mutable inputs only when tensor shapes/types/strides,
captured addresses, topology, backend placement, and attention launch bucket
remain compatible. In that case refresh without rebuild is required.

A different batch/ubatch shape, logical-context graph shape, storage address
assumption, recurrent plan, MTP plan, backend/model configuration, or attention
dispatch bucket selects an existing compatible graph variant or causes one
rebuild/update/recapture before decode. Never run a reused graph with stale page
IDs, suffix-only context length, or stale recurrent/draft state. Every decision
emits a reason-labelled reuse or rebuild metric.

Correctness means greedy output tokens, logits within the existing backend test
tolerance, target/draft terminal positions, recurrent state bytes, and
speculative accept/rollback behavior match a full uncached prefill at the same
source commit and fingerprint.

## 11. Restart and fail-open behavior

Before Phase 5 all registrations, versions, records, and residency are process
local. Restart empties them; a content-less old version returns stale. Phase 5
persists registrations and immutable records atomically. Restore accepts only
matching schema/build/model/tokenizer/template/layout/backend/MTP fingerprints,
valid lengths and checksums, and a complete predecessor chain. Otherwise it
quarantines the record and follows the normal stale/miss or uncached path.

Resource exhaustion, no admissible victim, COW failure, corrupt persistence,
restore timeout, publication failure, or an unsupported execution fingerprint
is fail-open to the nearest earlier ready checkpoint or normal uncached
completion. Malformed schema, wrong namespace, stale content-less version, and
CAS conflict are genuine protocol errors and remain explicit.

## 12. Required observability

Counters/histograms, labelled only by bounded enums rather than IDs or prompt
content, must cover:

- lookup result and depth, exact-equality rejection, suffix tokens computed,
  avoided prefill tokens, and estimated/observed avoided prefill time;
- build winner, waiter, coalesced completion, wait timeout, cancellation,
  publication success/failure, and fallback reason;
- unique physical pages, logical references, pins, COW copies/failures, GPU and
  host quota use, admission rejection, eviction, reclamation, swap, and
  recompute;
- restore winner/waiter/result/integrity rejection;
- graph reuse/rebuild/update reason and final populated-depth bucket; and
- per-slot logical limit, active overlap, prompt/decode tokens, TTFT, makespan,
  and busy slots per decode.

No metric or log may contain raw prompt text, token arrays, segment content, or
unbounded client-controlled identifiers.

## 13. Phase ownership

- Phase 2 owns dynamic context semantics, the in-memory cumulative record/page
  store, exact identity/equality, MTP-complete capture/restore, COW,
  single-flight, quotas, pins, publication, graph refresh, and fail-open paths.
- Phase 3 owns the three API schemas, boundary provenance through conversion,
  streaming/non-streaming responses, and cross-API exact-token tests.
- Phase 4 owns CAS replacement, retirement, in-flight old readers, idempotency,
  reclamation, and lifecycle fault tests.
- Phase 5 owns atomic persistence, compatibility/integrity validation, lazy
  restore, and restore single-flight.
- Phase 6 owns CarapaBoard wiring and the predeclared RTX 4090 NP=4/8/10
  performance, capacity, fairness, and protocol gates.

The dated Phase 1 report contains the complete fixture/oracle/metric/pass-fail
matrix. A phase must pass its owned cases when introduced and all applicable
earlier cases as regressions. No later phase may reopen the identity,
registration, wait, COW, quota, context, restart, or graph decisions above
without retaining the original gate and documenting new measured evidence.
