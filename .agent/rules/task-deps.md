---
trigger: model_decision
description: Use when creating or updating kanban tasks on this project's board, when completing or advancing a task stage, or when deciding which backlog task to start next.
---

# Task and stage management

- Board semantics: `in_progress` means QUEUED. The board itself never starts a queued task whose `depends_on_ids` are not all completed. Put dep-blocked work in `in_progress` so it is queued and auto-starts when unblocked; do not park it in `backlog` "to keep it safe" and do not move it back to `backlog` while waiting on dependencies.
- Stage advancement is automatic: when a stage run finishes, do not call `update_task` with a status pointing at the next stage. Finish the stage's work and end the run. If a later stage must be omitted, set `skip_stages` instead.
- A dependency-blocked or execution-slot-waiting `in_progress` task is not stalled. Inspect recent run state and errors before requeuing anything.
- `backlog` may contain intentionally deferred work. Move a backlog task to `in_progress` only when the user or task says it is ready, or evidence shows a failed checkpoint parked it there.
- For a question handoff, add a one-shot comment to the current executable stage, move the card to `question`, and stop. When an authoritative answer is available, comment on that executable stage and return the card to `in_progress`.
- Use one-shot stage comments for ordinary handoffs. Use persistent comments only for stable guidance after the same root failure recurs in the same stage.
- Inspect run and repository evidence after checkpoint failures. If the requested work already exists, hand off verification instead of asking another agent to reimplement it.
- Before creating a task, list the existing board tasks and check for related work in the same area (same feature, same files, same tests).
- If the new task is a prerequisite for existing tasks (it fixes a bug they build on, or they consume its output), add the new task to their `depends_on_ids` so the board cannot start them before it completes.
- If the new task builds on existing backlog work, set its own `depends_on_ids` accordingly instead of relying on priority order.
- If an existing task already covers the requested work, update that task instead of creating a duplicate.
- Keep priorities consistent with the dependency order: a prerequisite must have a lower (more urgent) priority number than its dependents.
