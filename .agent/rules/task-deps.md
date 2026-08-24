---
trigger: model_decision
description: Use when creating or updating kanban tasks on this project's board, or when deciding which backlog task to start next.
---

# Task dependencies

- Board semantics: `in_progress` means QUEUED. The board itself never starts a queued task whose `depends_on_ids` are not all completed. Put dep-blocked work in `in_progress` so it is queued and auto-starts when unblocked; do not park it in `backlog` "to keep it safe" and do not move it back to `backlog` while waiting on dependencies.
- Before creating a task, list the existing board tasks and check for related work in the same area (same feature, same files, same tests).
- If the new task is a prerequisite for existing tasks (it fixes a bug they build on, or they consume its output), add the new task to their `depends_on_ids` so the board cannot start them before it completes.
- If the new task builds on existing backlog work, set its own `depends_on_ids` accordingly instead of relying on priority order.
- If an existing task already covers the requested work, update that task instead of creating a duplicate.
- Keep priorities consistent with the dependency order: a prerequisite must have a lower (more urgent) priority number than its dependents.
- Never start a task whose `depends_on_ids` are not all completed.
