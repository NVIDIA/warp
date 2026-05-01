# CHANGELOG Language Review Examples

Loaded during Phase 5a of the skill. Use these examples to calibrate the language-review pass (which renders as the **CHANGELOG Review Notes** appendix in the report): flagging entries whose language would confuse end users.

## What to flag

### 🗣️ Internal / implementation language

Flag entries that reference:

- **Internal module paths**: `warp._src.foo`, `warp._src.codegen`, `warp._src.context`
- **C++/CUDA internal types**: `launch_bounds_t`, `tile_register_t`, `exec_mode_t`
- **Private identifiers** with leading underscores: `_foo`, `Module._compile`
- **Implementation details** that users shouldn't care about: "Refactor internal dispatch path", "Reorganize private helpers"

Good examples (user-facing):

> Add `wp.tile_scatter_add()` for per-thread cooperative adds into shared-memory tiles.

> Switch CPU JIT linker from RTDyld to JITLink, fixing sporadic access violations when the CUDA driver fragments the virtual address space.

Flag these:

> Update warp._src.codegen to emit launch_bounds_t<N> template.
> *Reason:* references internal module + private C++ template. User-facing rewrite: "Reduce register pressure for lower-dimensional kernel launches."

> Refactor tile_register_t to unify storage path.
> *Reason:* implementation detail, no user-observable effect. Candidate for deletion from CHANGELOG, not rewording.

**Carry GH refs through rewrites.** When suggesting a user-facing rewrite, preserve any `[GH-NNNN](https://github.com/NVIDIA/warp/issues/NNNN)` link from the original entry. The audit row's "Why" cell holds the rewrite suggestion verbatim, so a reader can paste it back into CHANGELOG.md; a missing GH ref forces them to dig back through history to attach one.

### 📝 Too terse

Flag entries that are too short to convey meaning or that omit enough context for a user to act on:

- Under ~10 words AND no linked issue for context.
- Entries like "Fix bug in tile_load" without specifying which bug or what the fix does.

Flag:

> Fix bug.
> *Reason:* insufficient — user can't tell what was fixed.

> Improve performance.
> *Reason:* no specifics; which API, how much, under what conditions?

Don't flag:

> Fix `wp.tile_map()` with `wp.tile_store()` failing for custom vector and matrix types created via `wp.types.vector()` or `wp.types.matrix()` ([GH-1311](...)).

This is long-ish but the specificity is load-bearing.

### 🔗 Suspected wrong GH reference

Flag when the CHANGELOG entry's topic doesn't match the commits that cite that GH number.

**Tier-1 heuristic (always on, fully local):**

For entry "Add wp.tile_dot for fused dot product (GH-1364)", fetch commits tagged GH-1364 and look at their subjects and file paths. If every commit only touches `.gitlab-ci.yml` or `.github/**`, the GH ref is likely wrong — the entry describes a kernel API change, but nothing in those commits modifies kernel code. Flag.

Don't flag if:
- Commits touch `warp/_src/builtins.py` for a tile entry → topic matches.
- Commits touch `docs/**` and the entry is in the Documentation section → topic matches.

**Tier-2 heuristic (only if `gh` CLI is installed + authenticated):**

For each GH ref, run `gh issue view <num> --json title,body`. Compare the issue's title/topic to the entry's description. If clearly unrelated (e.g., issue is "Improve sparse matmul", entry is "Add tile BVH query"), flag.

Skip tier-2 silently if `gh` is absent or auth fails. Don't emit warnings to the user.

## Judgment philosophy

**Err on "mention, don't block"**: flagging should raise a question for human review, not gate the report. The CHANGELOG Review Notes appendix shows flagged entries and a one-line reason; a human decides.

**Don't auto-rewrite**: Claude flags the entry, never modifies it. The release manager updates CHANGELOG.md manually.

**Prefer false positives over false negatives**: a flag that turns out to be fine costs a 5-second eyeball. A missed wrong-ref or jargon-leak ships to users.

## Row format (CHANGELOG Review Notes appendix)

| Entry (excerpt) | Flag | Why |
|---|---|---|
| "Refactor launch_bounds_t template..." | 🗣️ Internal language | References `launch_bounds_t` C++ template in user-facing prose |
| "Fix crash" | 📝 Too terse | 2 words, no context link |
| "Add wp.tile_dot (GH-1364)" | 🔗 Wrong ref? | Commits tagged GH-1364 touch only CI files |

Entry excerpt should be ~60-80 chars so the table stays scannable.
