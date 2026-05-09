# Release notes style rules

These are the hard constraints on what the rendered release notes look like. They are
informed by Warp's published release notes from v1.10 through v1.12.1, plus the user
preferences captured in this skill. Apply them strictly during Phase 6.

## GitHub-issue / PR references

**Use plain `#NNNN`, never `[GH-NNNN](https://github.com/NVIDIA/warp/issues/NNNN)`.**

The release notes are posted on the same GitHub repository. GitHub auto-resolves `#NNNN`
to a clickable link with the issue / PR title shown on hover. Bracketed `[GH-NNNN](...)`
links are correct in `CHANGELOG.md` (which is meant to be consumable outside of GitHub),
but they look heavyweight in release notes that always render on github.com.

✅ Good: `Fix kernel cache serving stale binaries when compilation settings change (#903).`

❌ Bad: `Fix kernel cache serving stale binaries when compilation settings change ([GH-903](https://github.com/NVIDIA/warp/issues/903)).`

When a single bullet covers multiple issues, list them comma-separated, each with `#`:

✅ Good: `Tile programming gains new primitives (#1287, #1298, #1342, #1364).`

The `#NNNN` form means GitHub renders **either** the linked issue's title **or** the PR
title — both are useful. Do not strip the `#` prefix.

## Admonitions: use the GFM forms only

When calling out information that needs visual separation, use GitHub Flavored
Markdown alert blocks. The five canonical forms are:

```markdown
> [!NOTE]      — useful background even when skimming
> [!TIP]       — helpful advice for doing things better or more easily
> [!IMPORTANT] — key information users need to achieve their goal
> [!WARNING]   — urgent info needing immediate attention to avoid problems
> [!CAUTION]   — risks or negative outcomes of certain actions
```

Each renders with a coloured icon and label on github.com. Do **not** use
ad-hoc forms — they look out of place on the rendered release page:

❌ Bad: `> **Note:** Useful background...`

❌ Bad: `**Note:** Useful background...`

❌ Bad: `> _Note:_ Useful background...`

❌ Bad: `> ⚠️ Warning: ...` (use `> [!WARNING]` instead)

✅ Good:
```markdown
> [!NOTE]
> Useful background...
```

Pick the alert type by intent, not by emoji preference: `[!IMPORTANT]` is for
information the user needs to succeed (experimental status, breaking-change
migration paths); `[!NOTE]` is for incidental context; `[!CAUTION]` is for
warnings about risks the reader can avoid; `[!WARNING]` is for things actively
going wrong if ignored. Limit to one or two per article and never stack two
admonitions consecutively.

## Experimental sections

When a `### …` block describes a feature whose CHANGELOG entry carries the bold
`**Experimental**` marker, place a `> [!IMPORTANT]` admonition immediately under the
heading. Use this exact shape:

```markdown
### Graph capture serialization and CPU replay

> [!IMPORTANT]
> This is an experimental feature. The API may change without a formal deprecation cycle.

Warp 1.13 introduces ...
```

The user prefers `[!IMPORTANT]` over `[!WARNING]` or `[!NOTE]` for experimental status:
the feature is real and ready to try, but the stability bar is lower than a stable API.

If the wider release also has some non-experimental features that depend on the same
subsystem, do **not** place the admonition at the section group level (e.g., not above
`## Graph capture`). Place it on each specific `###` block whose feature is experimental.
A reader scanning a section heading should be able to tell at a glance which subfeatures
have a stability caveat.

For inline mentions of experimental features in tables of contents or intros, append
`(experimental)` in parentheses:

✅ Good: `experimental graph capture serialization`

❌ Bad: `**EXPERIMENTAL** graph capture serialization`

## Lead with the unlock, not API names

The first sentence of every feature description should answer: *what is now possible
that wasn't before?* API names belong in the supporting sentence, not the lead.

❌ Bad: `Warp 1.13 adds capture_save() and capture_load() functions for graph
serialization, plus a new wp.handle scalar type.`

✅ Good: `Warp 1.13 introduces a portable serialized-graph format. Graphs captured
in Python can now be saved to a .wrp file with capture_save() and replayed from either
Python or standalone C++ via capture_load(), enabling cross-process and cross-language
graph reuse. A new wp.handle scalar type carries mesh handles across load.`

The lead paragraph and the lead feature's `### …` block both follow this rule. API
names are inevitable and welcome — just put them after the unlock.

## Group similar features

Tile primitives go together. JAX features go together. `warp.fem` features go together.
Even if the CHANGELOG lists them as separate entries with different GH refs, the release
notes group them. The grouping reflects how a user thinks about features ("I'm doing tile
work"), not how the codebase organizes them.

✅ Good — single themed section:

```markdown
## New tile primitives

This release adds several tile primitives that shorten common kernel patterns:

- `wp.tile_dot()` (#1364) replaces the longer
  `wp.tile_sum(wp.tile_map(wp.tensordot, a, b))` form for ...
- `wp.tile_axpy(alpha, src, dest)` (#1363) fuses the in-place update ...
- `wp.tile_scatter_add()` (#1342) and `wp.tile_scatter_masked()` (#1298) ...
```

❌ Bad — three separate `## ...` sections each describing one tile primitive.

## Order by impact

Within each release, sections appear roughly in this order:

1. Lead feature(s) — the one or two items that warrant a dedicated `## …` block at the
   top of `## New features`.
2. Other new features and themed groups (tile, JAX, fem, etc.).
3. Performance improvements (if substantial).
4. Language enhancements / compilation tooling.
5. New examples (if any are notable).
6. Bug fixes (only if any are worth highlighting; the changelog already has them all).
7. Announcements (deprecations, platform support).
8. Acknowledgments.

The lead feature is whatever the user identified during scope confirmation, or
whatever the skill author judged most exciting from the CHANGELOG. For 1.13 this is
APIC graph capture serialization with CPU graph replay.

## Headlines and section titles

Section headings (`## …`, `### …`) are short, descriptive, and active. They are not
sentences — no trailing periods, no full prose.

✅ Good: `### Graph capture serialization and CPU replay`

❌ Bad: `### Adding graph capture serialization with CPU replay support.`

Sub-headings inside a feature block (e.g., "Key capabilities") are bold paragraph
labels followed by a colon, not their own `####` headings.

✅ Good:

```markdown
**Key capabilities:**

- 1D / 2D / 3D ...
```

❌ Bad:

```markdown
#### Key capabilities

- 1D / 2D / 3D ...
```

## Code examples

**Code examples are the default, not the exception.** Every `### …` block describing
a new public API or a behavioral change ships with a runnable Python code block.
The example shows the simplest meaningful usage:

- Imports up top (`import warp as wp` etc.).
- **No `wp.init()` call.** Warp initializes implicitly on first use; calling
  `wp.init()` in examples is a stale pattern. Drop it from any snippet you
  draft, even when it appears in older release notes.
- A short kernel or call sequence.
- Show the program's output (see "Rendering output" below).

Examples are working code, not pseudo-code. Aim for 5-15 lines.

### Rendering output

Program output is rendered in one of two shapes; pick by length and structure:

- **Single short value → trailing `#` comment** on the `print(...)` line.
  Use this for a one-line scalar / array printout that fits comfortably:
  ```python
  print(out.numpy()[:8])  # [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]
  ```

- **Multi-line or structured output → separate fenced ```text block**
  immediately after the Python block, introduced by a one-line caption.
  Use this whenever the output is a table, multi-line dump, named-section
  report, or anything the eye needs to scan vertically. Burying multi-line
  output as `# comments` inside the kernel listing buries the actual feature
  value. Example:

  ```python
  with wp.ScopedMemoryTracker("training step"):
      train_step(model, batch)
  ```

  Output:

  ```text
  === Memory tracking summary: training step ===
    GPU peak:    1.42 GiB
    GPU current: 612 MiB
    Top allocations:
      [model.parameters]      512 MiB  (12 calls)
      (native:bvh)            256 MiB  (3 calls)
      [batch.activations]     128 MiB  (8 calls)
  ```

  Always introduce the second block with a short caption ("Output:", "Prints:",
  "Result:") on its own line, separated from the Python block by a blank line,
  so the rendered page makes the relationship obvious.

The threshold: if the output is one short line, use the inline comment. If the
output spans 2+ lines OR has internal structure (rows, tables, key:value pairs,
nested indentation), use the separate `text` block. The reader should never
have to mentally reformat a `# comment` block to understand what was printed.

### Verify example output by running it

**Every runnable Python snippet in the release notes must be executed end-to-end
before the draft is sent for review.** This is not "required when output IS the
feature"; it is required by default. The cost of running each snippet is ~1
minute; the cost of shipping a broken snippet in published release notes is
reputational.

Procedure for each snippet:

1. Save the snippet verbatim to `/tmp/release_notes_example_<short-name>.py`.
2. Run with `uv run /tmp/release_notes_example_<short-name>.py` from the repo
   root. Capture stdout exactly.
3. Embed the captured output in the release notes (single-line → trailing
   `# comment`; multi-line/structured → separate ```text block, per "Rendering
   output" above).
4. Delete the temp file when done.

If the snippet fails, fix the snippet (it has a bug) and re-run. Do not embed
hand-typed expected output, do not paraphrase what the output "should" look
like, do not fabricate plausible numbers. Categories of bugs that have been
caught only by running (illustrative, not exhaustive): missing required
keyword arguments that older docs didn't surface; device-mismatch from
arrays created outside the intended `wp.ScopedDevice` scope; `print(...)`
expressions whose rendered form leaks an internal `warp._src.*` path;
hand-typed expected output that doesn't match the actual rendering.

Output rendering is non-negotiable when the output IS the feature being
demonstrated:
- **`wp.ScopedMemoryTracker` / `wp.config.track_memory`** — the allocation
  summary printed at scope exit IS the headline value of the feature; show the
  actual table in a separate block.
- **`wp.print_diagnostics()`** — the formatted environment dump is the entire
  point; render the dump in a `text` block.
- **bfloat16 / fp64 promotion** — short dtype confirmations may use the inline
  trailing-comment form.
- **Approximate-math intrinsics** — a side-by-side numeric delta is short
  enough to inline; a multi-row precision table belongs in a `text` block.

If running the snippet is impractical, prefer making it runnable (supply
minimal inputs) over annotating around the gap. When that's not feasible:
mark API-shape-only snippets with a header comment so readers don't try to
copy-paste-and-run (e.g. `# API-shape example; requires a CUDA context`); for
C++ / cross-language consumer snippets, see `feature-investigation.md`
"Cross-language requirements" for the elision and signature-verification
protocol. If a snippet you authored needs to run but cannot (build not
current, missing dependency, GPU-only test environment), surface the failure
to the user rather than fabricating output.

The only entries that may legitimately be prose-only are single-parameter tidy-ups,
single-flag toggles, or pure perf optimizations with no API change. When in doubt,
include the snippet. Past reviews consistently flagged "this section is all bullets
with no code" as the dominant complaint; defaulting to a snippet is the cure.

For sections that group multiple related new APIs (e.g., "New tile primitives"), each
named API gets its own short snippet rather than one combined dump. Snippets can
share a single setup if obvious, but each new symbol's call site is shown literally.

### Cross-language and cross-runtime features

Any feature whose claim crosses a language or runtime boundary ships with a snippet
**on each side**, not just the Python authoring side. Specifically:

- **Python → standalone C++** (e.g. graph capture serialization): a Python `…_save`
  snippet plus a minimal C++ snippet showing the consumer side (headers, init order,
  load + replay calls). If a working in-tree example exists under
  `warp/examples/cpp/`, link it as a relative URL on the release tag.
- **Python → JAX**: a Python kernel definition plus a `jax_kernel(...)` binding
  showing the consumer-side import and call pattern.
- **Python → PyTorch / DLPack**: a snippet showing the round-trip (`wp.from_torch`
  / `__dlpack__`) on the consumer side.

Naming the consumer-side header file or import path is not optional. A reader who
only sees the producer side cannot evaluate whether the feature solves their
problem.

### Breaking changes use before/after snippets

Any breaking change in `## Bug fixes`, `## Changes to existing APIs`, or any section
that announces a behavioral shift ships a **before / after** code block, not prose.
Comment headers (`# v1.X behavior:` / `# v1.Y migration:`) make the intent explicit.

Example shape:

```python
# v1.12 behavior: out.grad still holds the seed after backward()
out = wp.zeros(N, dtype=float, requires_grad=True)
with wp.Tape() as tape:
    wp.launch(forward_kernel, dim=N, inputs=[x], outputs=[out])
out.grad = wp.ones_like(out)
tape.backward()
print(out.grad)   # v1.12: ones, v1.13: zeros

# v1.13 migration: keep output grads, pass fresh seeds each call
out = wp.array(shape=N, dtype=float, requires_grad=True, retain_grad=True)
with wp.Tape() as tape:
    wp.launch(forward_kernel, dim=N, inputs=[x], outputs=[out])
tape.backward(grads={out: wp.ones_like(out)})
```

Migration prose alone, no matter how clear, leaves users guessing at the literal
call shape. The before/after snippet removes the guess.

**Pick the right fenced language for the change shape:**

- Use **two ```python blocks** with `# v1.X behavior:` / `# v1.Y migration:`
  comment headers when the migration spans multiple lines, requires new setup,
  or shows a behavioral shift (e.g., when `.grad` reads now zero, when a default
  changes, when an output now requires `retain_grad=True`). The two-block form
  carries imports, scope, and full call context that a `diff` view would
  fragment.

- Use a **single ```diff block** when the change is a tight signature shift,
  parameter rename, or single-line call-site update. `-` lines mark the old
  shape, `+` lines mark the new. GitHub renders these with red/green highlights
  that scan instantly:

  ```diff
  - m = wp.matrix_from_rows(wp.vec2(1, 2), wp.vec2(3, 4))
  + m = wp.mat22(wp.vec2(1, 2), wp.vec2(3, 4))
  ```

  ```diff
  - wp.tile_load(arr, shape=N, offset=i * N)
  + wp.tile_load(arr, shape=N, offset=i * N, aligned=True)  # new opt-in flag
  ```

  Reserve `diff` blocks for changes a reader can absorb in 1-3 lines on each
  side. Anything bigger goes in two `python` blocks instead, since `diff`
  syntax highlighting drops Python coloring (only the leading `+`/`-` is
  colored) and long migration code reads worse without it.

### Linking in-tree examples

When a feature has a working example shipped under `warp/examples/**`, link it
explicitly. Use the release tag in the URL so the link won't drift:
`[example_name](https://github.com/NVIDIA/warp/blob/v<version>/warp/examples/<path>)`.
Never link by SHA; never link by `main`. If the link would point to a file that
moved or no longer exists at the tag, drop it.

### Artifacts produced by a feature

When a feature writes a new on-disk artifact (file format, serialized graph,
checkpoint), describe what gets written beyond the obvious file. Sidecar
directories, version-pinned binaries, and architecture-pinned CUBINs are common
footguns: a user shipping `name.wrp` without the `name_modules/` directory will
hit a silent load failure. Use a literal directory-tree code block when the
artifact is more than one file:

```
scene.wrp                 # operation byte stream + region snapshots + metadata
scene_modules/
    abc123.cubin / .meta  # one per kernel module, arch-pinned
```

## Proper-noun capitalization

Always capitalize: NumPy, Warp, NVIDIA, CUDA, JAX, PyTorch, Python, OpenGL, Linux,
Windows, macOS, Apple Silicon, ARM64, x86_64. Never lowercase these in prose; the only
exception is when the literal symbol is being quoted in code (`numpy.float32` is fine
inside backticks).

## Em dashes

**Do not use em dashes (`—`) in prose.** Use a comma, period, or colon instead. This is
a per-user preference for the Warp release notes style.

❌ Bad: `Warp v1.13 introduces graph capture serialization — letting captured graphs roundtrip ...`

✅ Good: `Warp v1.13 introduces graph capture serialization, letting captured graphs roundtrip ...`

✅ Good: `Warp v1.13 introduces graph capture serialization. Captured graphs can roundtrip ...`

This applies to the rendered output. Em dashes inside backtick-quoted code or in
verbatim quotes from external sources are exempt.

## Stylistic conventions

### Punctuation in prose

- Avoid em dashes (covered above).
- Avoid prose semicolons. They are typographically heavier than the separation
  usually warrants. Convert to periods unless the two clauses are strongly
  parallel and gain meaning from being joined. Code-block semicolons (C++
  statement terminators) are unaffected.

### Hyphens in compound modifiers

When two or more words modify a noun together, hyphenate them: "ray-heavy
workloads", "shared-memory tile", "host-CPU specialization", "Warp-based
renderer", "header-only library", "power-of-2 tile sizes", "16-byte alignment",
"multi-byte elements".

### Subscript-style type annotations in snippets

Use the subscript-style form for tile/array parameter annotations:

- `x: wp.array[float]` (not `wp.array(dtype=float)`)
- `A: wp.array2d[float]` (not `wp.array2d(dtype=float)`)
- `t: wp.tile[float]` (not `wp.tile(dtype=float)`)

The legacy call-syntax form was deprecated in user code in v1.12 (#1216)
because static type checkers like Pyright flag it as a constructor call.
Release-notes snippets should use the modern form.

### Match in-tree code conventions for non-Python snippets

For C++/CUDA snippets, match the commenting and structural conventions of the
in-tree example for the feature.

### Warp vocabulary

Use Warp's own terminology, not borrowed framework terms:

- ✅ tile, array, kernel, struct, builtin (Warp).
- ❌ tensor (PyTorch), ufunc (NumPy), op (general ML).
- "graph" is overloaded: be explicit about CUDA graph vs. autograd graph vs.
  capture graph.

## Audience: Python users, not native developers

The Warp release-notes audience is "someone using Warp from Python". They see
`@wp.kernel`, `wp.tile_load`, `wp.array`, the `storage=` parameter. They do
**not** read `warp/native/*.h`, do not know CUDA's hardware model, do not see
the difference between RTDyld and JITLink, do not write PTX.

If a sentence in the release notes would only make sense to someone who reads
`warp/native/*.h`, rewrite it.

GPU/CUDA internals that must be translated or dropped:

- `float4` vectorization, coalesced access, warp memory coalescing, bank
  conflicts, scattered access patterns, stride-1 access.
- PTX intrinsics, `fma.rn.bf16`, fence instructions.
- RTDyld, JITLink, CPU JIT linker internals.

When the headline value of a change is a perf win, lead with the user-visible
impact ("X% faster on Y workload" or "kernel compiles in 1s instead of 18s")
and elide the mechanism. When the mechanism IS user-visible, describe it in
user-comprehensible terms ("reads by rows instead of columns", "skips the slow
LTO compile step", "stays in register memory rather than round-tripping through
global"). Don't expose CUDA primitives the user can't act on.

Other internal terminology that does not belong in release notes:

- Skill-internal names ("Phase 5b", "headline counts", "{{PLACEHOLDER}}").
- Internal Python module paths (`warp._src.*`). Use the public surface
  (`@wp.kernel`, `wp.tile_load`).
- C++ / CUDA internal type names (`launch_bounds_t`, `tile_register_t`,
  `wp_array_t`).
- Commit SHAs, branch names, internal jira IDs, audit-tool flags.

If the CHANGELOG or a docstring uses any of these terms, paraphrase before
lifting into the release notes (see "CHANGELOG paraphrasing" below).

## CHANGELOG paraphrasing

The CHANGELOG is written for code reviewers who know the internals. Release
notes are for Python users who interact only with the public `warp` API.
**Lifting a CHANGELOG bullet verbatim is almost always wrong.** Treat each
bullet as a starting point and:

1. Identify any technical jargon ("bank conflicts", "vectorized path",
   "coalesced byte-copy", "LTO", "fp32 round-trip", "rank cap", "register
   spills") and rewrite for the Python-user audience using the rules above.
2. Fetch the linked GitHub issue or MR (`gh issue view <NNNN>` /
   `gh pr view <NNNN>`) and look at the motivation, benchmark numbers, and
   scenario text. The user-visible framing ("this scene was X% slower at
   power-of-2 tile sizes", "this workflow required a workaround") usually
   lives there, not in the CHANGELOG bullet which has been compressed for
   engineers.
3. Cross-reference any implementation claim with the source. If the CHANGELOG
   says "uses native bf16 intrinsics", verify the path is actually taken on
   the relevant arch (see `feature-investigation.md` "Verify symbol
   references"). If the CHANGELOG says "now supports N-D for N >= 2", check
   what the previous form already supported. "Supports N-D" can mean either a
   wholly new capability or just lifting a rank cap; the user-facing framing
   differs.
4. Translate "improved memory access patterns" / "high-rank tile traffic" /
   "follows correct semantics" into a concrete user-visible win, OR drop the
   claim. See "No hand-wavy claims" below.

## No hand-wavy claims

Sentences that claim behavior is "correct", "as expected", "compatible", "works
properly", or "follows the right pattern" without naming the scenarios are
hand-wavy and don't earn their place. Either:

- **Be specific.** Name the scenarios where it matters ("works inside
  `wp.capture_begin()` graph capture", "matches RMM's `CudaAsyncMemoryResource`
  semantics", "integrates with PyTorch's RMM-backed allocator").
- **Or drop the sentence.**

Test: if a reader can replace the sentence with `<reasonable thing happens>`
without losing information, the sentence is hand-wavy and should be cut.

Every technical claim should pass through one of three filters:

1. **Concrete user scenario** with a named API or workflow.
2. **Measured impact** with units ("1.2-1.7x faster", "compiles in 1s instead
   of 18s", "drops simulation time from 1.41s to 0.98s").
3. **Linkable issue / MR / doc** for the full detail.

## Forward-looking claims

Release notes are public commitments. Any forward-looking claim ("will land in
vX.Y", "scheduled for", "slated for", "next release", "soon") must be backed
by a primary source that explicitly commits to that schedule:

- A CHANGELOG `### Deprecated` bullet that names the removal version.
- The runtime `DeprecationWarning` emit string text in the source.
- The docstring of the deprecated symbol or feature.
- A design doc or tracker issue with explicit version commitment.
- A maintainer's explicit statement on the issue/MR.

The deprecation policy ("at least 4 months / 4 feature releases") is a
**floor**, not a schedule. Picking the earliest legal version and stating it
as a release plan is a tacit promise the project would have to keep.

Without a primary source, use neutral framings:

- "...will be removed in a future feature release per the standard deprecation
  timeline." (link to `compatibility.html` if appropriate)
- "...tracked in #NNNN" (when there's an open tracker)
- "...designed-but-deferred" (for tracker items in "no immediate action"
  sections)

When citing a tracker issue, **read the issue structure**. Many issues
distinguish "Closes known gaps" / "Diagnostics" / "Architectural"
(actively-tracked) from "Designed but not implemented anywhere, no immediate
action required" (deferred). Don't promote items from the deferred section to
"tracked" status.

For release-cadence and process claims, use aspirational language ("Warp aims
to publish a feature release every month") rather than declarative ("Warp now
publishes..."). Cadence claims that read as permanent commitments lock the
team in.

## Announcements section style

Announcements are written in the second person ("Update to Python 3.10 or newer
to continue receiving Warp updates") and lead with the user-facing impact.
Bullet shape:

```markdown
- **Python 3.9 is no longer supported.** Python 3.10 is now the minimum
  required version. Update to Python 3.10 or newer to continue receiving Warp
  updates.
```

Bold the leading clause; the rest is normal prose. Each bullet should answer *what
changes, when, and what to do about it*.

For deprecations that already shipped in this release (i.e., the deprecation warning
fires now, the removal lands later), state the current state and refer to the
deprecation policy without inventing a version: `Deprecated in 1.13; will be
removed in a future feature release per the standard deprecation timeline.`
Only name a specific removal version when a primary source commits to it
(see "Forward-looking claims" above).

### Tone for removals and deprecations

Removal and deprecation bullets must:

- State the change as fact (lead clause, bolded).
- Give the actionable migration step in neutral language tied to the user's
  outcome.
- Avoid scolding or "told-you-so" framing about prior deprecation cycles.

❌ Bad (snarky, blames the user for not migrating):

```markdown
- **Python 3.9 is no longer supported.** Python 3.10 is now required. If you
  saw the deprecation warning in v1.12, the time to upgrade is now.
```

✅ Good (neutral, action-tied):

```markdown
- **Python 3.9 is no longer supported.** Python 3.10 is now the minimum
  required version. Update to Python 3.10 or newer to continue receiving Warp
  updates.
```

Mention prior-release context only when it's load-bearing, e.g. "finalizing
the deprecation announced in v1.11", not as a callout to users who didn't
migrate.

## Acknowledgments style

Open with the standard line:

```markdown
We also thank the following contributors from outside the core Warp development team:
```

Each bullet:

```markdown
- @username for [specific contribution] (#NNNN).
```

Be specific about the contribution. "for fixing the inverted `verbose` flag in
`wp.capture_debug_dot_print()`" beats "for a bugfix". The reader is the contributor's
future employer or peer; the line should reflect what they actually did.

When a contributor has multiple PRs, either combine into one bullet with multiple `#NNNN`
refs, or list separate bullets if the contributions are unrelated. Use judgment.

## Footer

End the document with a horizontal rule and a one-line pointer to the full changelog:

```markdown
---

For a complete list of changes, see the [full changelog](https://github.com/NVIDIA/warp/blob/v<version>/CHANGELOG.md).
```

For bugfix releases the changelog pointer can replace much of the body — the user has
already said "see the full changelog for details" works as the dominant story for a
bugfix release. Do not duplicate every fix bullet from the changelog.
