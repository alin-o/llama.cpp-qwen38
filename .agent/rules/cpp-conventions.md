---
trigger: model_decision
description: Use when writing or reviewing C++ or CMake code in this llama.cpp fork (ASCII only, comments, tests, jinja engine, reuse of existing infrastructure).
---

# C++ and repository conventions

- ASCII only in code and comments: no emdash, no unicode arrow, no multiplication sign; use `-`, `->`, `x` instead.
- Comments: concise (usually 1-2 lines), only where the code is not self-explanatory; never hard-wrap to a fixed column; plain simple wording.
- Reuse existing infrastructure. Do not add new subsystems, new dependencies, or new files under `tests/*` without explicit user approval; reuse existing test infrastructure and do not add tests for trivial changes.
- The Jinja engine lives in `common/jinja`. llama.cpp does NOT use Minja.
- Read all relevant files before editing; changes must blend in with the surrounding code. Large or new-pattern changes require user confirmation first (use the question handoff).
- Commit messages: concise, with an `Assisted-by: <assistant name>` trailer; never `Co-authored-by:`.
