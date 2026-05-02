# Warp {{VERSION_STRING}} {{REPORT_KIND}} Report
Generated: {{REPORT_DATE}}

<!-- {{REPORT_KIND}} is either "Pre-Release" or "Release Candidate", chosen in
     Phase 1 from the version string and head ref. {{VERSION_STRING}} is the
     raw version (e.g. "1.13.0dev0" for pre-release, "1.13.0rc1" for RC). -->

- Mode: {{MODE_DESCRIPTION}}
- Head: {{HEAD_REF}} @ `{{HEAD_SHA_SHORT}}`
- Base: {{BASE_REF}} @ `{{BASE_SHA_SHORT}}`
- Commits in range: {{N_COMMITS}}

<!-- {{MODE_DESCRIPTION}} is a one-liner:
     Pre-release: "Pre-release audit of unreleased work on main"
     RC: "Release candidate readiness review (release branch cut)" -->


**Headline counts**

- {{N_NEW_API}} new public APIs (Python: {{N_NEW_PY}}, kernel: {{N_NEW_KERNEL}})
- {{N_BREAKING}} breaking changes
- {{N_CHANGED}} changes to existing API
- {{N_BEHAVIORAL}} behavioral / support changes
- {{N_FIXED}} fixes

**Bake distribution**

| Bucket | Commits |
|---|---:|
| 🟢 > 14 days in main | {{N_BAKE_GREEN}} |
| 🟡 7 to 14 days | {{N_BAKE_YELLOW}} |
| 🟠 < 7 days | {{N_BAKE_ORANGE}} |

{{ANOMALY_BANNER_IF_ANY}}

<!-- Anomaly banner fires ONLY when at least one commit has
     main_match_state == "missing" (subject not present on main_ref). Format:
     > ⚠️ **N commits in the release have no equivalent on main. Investigate:
     > these shipped without nightly/main-branch bake.**

     Commits with main_match_state == "ambiguous" (subject appears more than
     once on main_ref — typical for `Revert "..."`, `Bump version`, or
     replayed cherry-picks) are NOT a bake-gap signal: the commit IS on main,
     the script just could not pick a single canonical occurrence. Surface
     them as a separate row in the bake distribution table ("⚪ ambiguous
     main match: K commits") rather than firing the banner.

     In pre-release mode (resolved.head.sha == resolved.main_ref.sha) the
     banner does not fire and the bake table is replaced with an "Age
     distribution" table from days_since_merge: there is no meaningful
     bake-gap condition when head IS main.

     If resolved.empty_main_index == true, replace the banner with a
     prominent header note (e.g., "main_ref had no commits in <base>..<main_ref>;
     main bake unverifiable") instead of firing the routine bake-gap banner. -->


---

## Release highlights

{{HEADLINE_SUMMARY}}

<!-- Claude's qualitative synthesis of what would land in the official release
     notes. Drafted in Phase 6a. NOT release notes: a reviewer's preview so the
     release manager can see at a glance whether the real release notes will
     match expectations and spot items that need a keep/defer decision.

     Shape: one short intro paragraph (2-3 sentences) followed by 4-8 bulleted
     highlight items. Each bullet starts with a bold 2-6 word headline, then a
     colon and a one-sentence rationale (what it is and why it matters).

     Include risk flags inline when they apply:
       - 🟠 `N days` bake (if the headline item's minimum bake is < 7 days)
       - ⚠️ Experimental (if the CHANGELOG entry carries an "Experimental" marker)
       - ⚠️ Breaking (if the headline item is a breaking change)

     Do NOT include counts ("4 new APIs were added"): that's already in the
     headline counts block above. Highlights are qualitative.

     Example shape (do NOT copy verbatim; pick items that are actually headline
     material in THIS release):

     Warp 1.13 lands a pluggable allocator interface, a new bfloat16 scalar type,
     and a batch of tile primitives. Two breaking changes (kernel thread-ID
     flattening and backward-pass gradient zeroing) need migration notes in the
     release post.

     - **Pluggable allocator interface** ([GH-781](...)): `wp.Allocator`, `wp.RmmAllocator`,
       and `wp.ScopedAllocator` let downstream stacks route Warp allocations through
       RMM or custom pools. 🟠 3 days bake.
     - **bfloat16 scalar type** ([GH-1332](...)): first-class `wp.bfloat16` with autodiff,
       DLPack, and PyTorch interop. Headline for ML users. 🟠 3 days bake.
     - **tile_dot / tile_axpy / tile_scatter_add** ([GH-1287](...), [GH-1298](...),
       [GH-1342](...), [GH-1364](...)): new tile primitives that shorten common kernel
       patterns. Every GH ref is an individual hyperlink; never collapse to
       "(multiple GHs)" or a plain-text comma list.
     - **⚠️ Breaking: `wp.tid()` flattens excess dimensions** ([GH-1270](...)): kernels
       that unpack fewer values than launch dims now get a flat index; needs a migration
       note.
     - **⚠️ Breaking: backward pass zeros output gradients** ([GH-1062](...)): inspection
       of `.grad` after `tape.backward()` now requires `retain_grad=True`.
     - **⚠️ Experimental: cuBQL BVH backend for `wp.Mesh`** ([GH-1286](...)): ray queries
       only; selectable via `bvh_constructor="cubql"`.
     - **Python 3.9 support removed**: release notes should lead with the minimum-version
       bump.
-->

---

## Contents

{{CONTENTS_BULLETS}}

<!--
Expand the TOC to include every `###` heading rendered in the body, not
just the `##` top-level sections. List each per-symbol / per-topic heading
as a sub-bullet under its parent section. Example shape (replace with the
real symbols / topics present in this specific report):

- [New API](#new-api)
  - [Python scope](#python-scope)
    - [`wp.<symbol>`](#wpsymbol)
    - ...
  - [Kernel scope](#kernel-scope)
    - [`wp.<builtin>`](#wpbuiltin)
    - ...
- [Breaking Changes](#breaking-changes)
  - per-entry heading as sub-bullet
- [Changes to Existing API](#changes-to-existing-api)
  - per-entry heading as sub-bullet
- [Behavioral & Support Changes](#behavioral--support-changes)
  - per-topic heading as sub-bullet
- [Fixed](#fixed)
- [CHANGELOG Review Notes](#changelog-review-notes) (only if the conditional
  appendix renders content)

The sample symbol names above are illustrative placeholders; do NOT ship them
as-is. GitHub auto-renders a floating outline panel, but an explicit TOC
still helps raw-text readers.
-->

---

## New API

### Python scope

{{NEW_PYTHON_TABLES_BY_KIND}}

<!-- Render one summary table per Kind. Columns: Symbol | Description | GH | Bake.
     Example groupings: "Functions", "Classes / context managers", "Scalar types",
     "Decorators", "Enums / flags". A scope with only one Kind gets one table.

     The Symbol cell uses a short-form call shape (no type annotations; defaults
     included). Examples:
     - `wp.print_memory_report(file=None, sort="size", max_items=10)`
     - `wp.tile_scatter_add(dest, indices, values, *, atomic=True)`
     - `wp.ScopedMemoryTracker(label, devices=None)`
     - `wp.TextureResourceFlags`  (no parens for enums, scalars, decorators)
     - `wp.bfloat16` -->

{{NEW_PYTHON_DETAIL_BLOCKS}}

<!-- Per-symbol block template. Headings use the symbol name alone (no colons/em dashes).
     The single fenced code block underneath holds signature-shaped content and
     the docstring.

Simple context-manager / data class (constructor-only rendering):

### `wp.ScopedMemoryTracker`

Links: [GH-1269](...), commits: [26bc6ea4](...), [5613042d](...)
Source: `warp/_src/utils.py`
Bake: 🟢 47 days in main

```python
class ScopedMemoryTracker:
    """Context manager for tracking memory allocations within a scope.

    Captures per-category totals for host, pinned host, and GPU memory and
    returns them via .summary() after __exit__.
    """

    def __init__(self, label: str, devices: Optional[Sequence[DeviceLike]] = None)
```

Class with multiple public methods (interface-style): list every public method
with its signature and docstring, not just __init__.

### `wp.Allocator`

Links: [GH-NNN](...), commit: [sha](...)
Source: `warp/_src/context.py`
Bake: 🟢 30 days in main

```python
class Allocator:
    """Base class for custom GPU memory allocators.

    Subclass and implement alloc and free to plug into Warp's memory system
    via wp.set_cuda_allocator.
    """

    def __init__(self, device: DeviceLike)

    def alloc(self, size: int) -> int
    """Allocate size bytes on the device; return a raw device pointer."""

    def free(self, ptr: int, size: int) -> None
    """Release a previously allocated block."""

    def empty_cache(self) -> None
    """Release cached but unused blocks back to the driver."""
```

Enum / IntFlag: list every member with its value and per-member doc.

### `wp.TextureResourceFlags`

Links: [GH-1238](...), commit: [sha](...)
Source: `warp/_src/texture.py`
Bake: 🟡 12 days in main

```python
class TextureResourceFlags(IntFlag):
    """Flags controlling how a texture resource is bound and accessed."""

    NONE = 0
    """No special flags (default)."""

    COLOR_ATTACHMENT = 1 << 0
    """Texture may be used as a color render target."""

    DEPTH_ATTACHMENT = 1 << 1
    """Texture may be used as a depth render target."""

    STORAGE = 1 << 2
    """Texture may be bound for generic read/write in kernels."""
```

Kernel builtins: synthesized pythonic form (NOT the add_builtin() call).

### `wp.tile_scatter_add`

Links: [GH-1342](...), commit: [3b7fa644](...)
Source: `warp/_src/builtins.py`
Bake: 🟢 21 days in main

```python
tile_scatter_add(
    dest: tile,
    indices: array(dtype=int),
    values: tile,
    *,
    atomic: bool = True,
) -> None
"""Per-thread cooperative adds into a shared-memory tile.

Set atomic=False when indices are guaranteed unique across threads for ~4x
faster shared-memory writes.
"""
```
-->

### Kernel scope

{{NEW_KERNEL_TABLES_BY_KIND}}

<!-- Same Kind grouping as Python scope. Kernel kinds typically include:
     "Tile operations", "Queries", "Types", "Primitives". -->

{{NEW_KERNEL_DETAIL_BLOCKS}}

---

## ⚠️ Breaking Changes

{{BREAKING_ENTRIES}}

<!-- Flat list. Do NOT group under "Author-labeled" / "Unlabeled" / "Semantic"
     subheadings. Every entry is simply a confirmed breaking change, regardless
     of how it was identified. Claude collects entries from four sources:
     (1) CHANGELOG entries carrying **Breaking:**,
     (2) CHANGELOG Removed entries (implicitly breaking),
     (3) public-surface AST diffs between base and HEAD that weren't labeled,
     (4) codegen/native commits that Claude VERIFIED with a real test (Phase 4f);
     ambiguous candidates that were not verified do NOT appear here.

     Per-entry render format:

     ### <heading: symbol name or short descriptive title>

     Links: [GH-NNN](...), commit(s): [sha](...). 🟢 N days baked in main.

     [If signature diff applies, a fenced diff block.]
     [If behavior shift, a before/after code snippet. For verified semantic
      breaks, the snippet IS the verification test Claude ran, plus the
      captured output of the two builds showing the observable difference.]

     [1-3 sentences of explanatory prose in plain user-facing language.]

     [Full CHANGELOG text blockquoted, if the entry came from CHANGELOG.]

     Illustrative example (do not copy verbatim):

     ### Short-circuit `and`/`or` evaluation in kernels

     Links: [GH-1329](https://github.com/NVIDIA/warp/issues/1329),
     commit: [a72c39d4](https://github.com/NVIDIA/warp/commit/a72c39d4). 🟢 22 days baked in main.

     Previously all operands were evaluated eagerly. Now matches Python: the
     RHS is only evaluated if needed. Kernels that relied on side effects in
     the RHS of an `and`/`or` will behave differently.

     Before:
     ```python
     @wp.kernel
     def k(arr: wp.array(dtype=int)):
         i = wp.tid()
         # Crashed on empty arrays because arr[i] always ran.
         if arr and arr[i] == 0:
             ...
     ```

     After:
     ```python
     @wp.kernel
     def k(arr: wp.array(dtype=int)):
         i = wp.tid()
         # arr[i] only runs when arr is non-empty.
         if arr and arr[i] == 0:
             ...
     ```
-->

---

## Changes to Existing API

<!-- Covers CHANGELOG Changed, Removed, Deprecated, plus capability extensions
     routed here from the new-API classification pass (e.g., "Add support for X
     in existing Y"). -->

{{CHANGED_SUMMARY_TABLE}}

<!-- Columns: API | Kind | Breaking | Description | GH | Commits | Bake
     Kind values: signature change, new parameter, capability extension, removed,
     deprecated, semantic change. Description is a short phrase (≤ 10 words). -->

{{CHANGED_DETAIL_BLOCKS}}

<!-- Per-entry template (signature change). Use a colon in the heading, not an em dash.

### `wp.tile_fft`: new parameter

Breaking: **No** (additive keyword-only default)
Links: [GH-1317](...), commit: [5c5f67e9](...)
Bake: 🟢 38 days in main

```diff
- def tile_fft(a: Tile) -> Tile
+ def tile_fft(a: Tile, *, axis: int = -1) -> Tile
```

**From CHANGELOG**
> Allow `wp.tile_fft()` and `wp.tile_ifft()` to operate on N-D tiles (N >= 2),
> computing the FFT along the last dimension with all leading dimensions treated
> as independent batches ([GH-1317](...)).

Capability extension (existing API gains a new option/behavior):

### `@wp.kernel`: `module_options` parameter

Links: [GH-1250](...), commit: [abcdef12](...)
Bake: 🟢 35 days in main

Adds a `module_options` dict parameter to `@wp.kernel` for inline module-level
compilation options on `"unique"` modules.

```diff
- def kernel(f=None, *, enable_backward=None)
+ def kernel(f=None, *, enable_backward=None, module_options=None)
```

**From CHANGELOG**
> <full CHANGELOG text blockquoted>

Removed symbol:

### `wp.old_name`: removed

Links: [GH-NNN](...), commit: [sha](...)
Deprecation window: Deprecated in 1.11.0; removed here.

```diff
- def old_name(arg: int) -> int
```

**From CHANGELOG (prior deprecation in 1.11.0)**
> Deprecate `wp.old_name`. Use `wp.new_name` instead. Will be removed in 1.13.

**From CHANGELOG (removal)**
> <full CHANGELOG text blockquoted>

Experimental-softened entry (symbol was shipped with **Experimental** marker in a prior release; change here is technically breaking but the stability bar was advertised up front):

### `Texture` class constants and subclass defaults: adjusted

Breaking: **Experimental** (since 1.12.0)
Links: [GH-1238](...), commit: [sha](...)
Bake: 🟢 40 days in main

The texture classes landed as experimental in 1.12.0; this release removes the `Texture.ADDRESS_*` / `Texture.FILTER_*` class constants and changes `Texture2D` / `Texture3D` default `dtype` and `num_channels` values. Callers that relied on the old defaults need to pass them explicitly.

```diff
- class Texture2D:
-     ADDRESS_CLAMP = ...
-     FILTER_LINEAR = ...
-     def __init__(self, ..., dtype=<old>, num_channels=<old>)
+ class Texture2D:
+     def __init__(self, ..., dtype=<new>, num_channels=<new>)
```

**From CHANGELOG (1.12.0 introduction, still experimental)**
> **Experimental**: Add `wp.Texture1D`, `wp.Texture2D`, and `wp.Texture3D` classes ... ([GH-1122](...)).

**From CHANGELOG (this release)**
> <full CHANGELOG text blockquoted>
-->

---

## Behavioral & Support Changes

<!-- Group by topic with short descriptive section headings synthesized from
     the entry content. Use colons in headings if separation is needed.
     Related topics should live together.

     Example headings (illustrative, choose based on actual content):
     - "Python 3.9 support dropped"
     - "Build requirements on Linux"
     - "CPU JIT linker: RTDyld to JITLink"
     - "CPU default target: host-detected"
     - "Anisotropic voxel spacing"
     - "Kernel cache versioning"

     Each topic: a short paragraph summary, links, commits, bake. -->

{{BEHAVIORAL_SECTIONS}}

---

## Fixed

{{FIXED_TABLE}}

<!-- Columns: Fix | GH | Commits | Bake
     Keep the full CHANGELOG text in the Fix column; no truncation.
     Do NOT mention fixes from previously-shipped patch releases. The commit-list
     tool scopes to <base>..<head> so those are already excluded. -->

{{OPTIONAL_APPENDIX}}

<!-- Render conditionally based on content and wrap each non-empty section in
     a GFM <details> block so it collapses by default (the umbrella content is
     reference material, not headline reading).

     Three cases:

     1. Both CHANGELOG-orphan list AND language-review flags are empty:
        Render nothing. No appendix heading, no trailing section.

     2. Exactly ONE is non-empty: Render that one as a top-level section
        (no "Audit Appendix" umbrella) with the table inside <details>.

        ## CHANGELOG Entries Without Matching Commits
        <details>
        <summary>N entries (click to expand)</summary>

        | Entry | GH refs | Suspected reason |
        |---|---|---|
        | full entry text | ... | ... |

        </details>

     3. Both are non-empty: Render an umbrella section; each subsection gets
        its own <details>.

        ## Audit Appendix

        <details>
        <summary>N CHANGELOG entries without matching commits (click to expand)</summary>

        | Entry | GH refs | Suspected reason |
        |---|---|---|
        ...

        </details>

        <details>
        <summary>N CHANGELOG entries flagged for review (click to expand)</summary>

        | Entry | Flag | Why |
        |---|---|---|
        ...

        </details>

     Column rules for BOTH tables: full entry text (no truncation).
     Flag glyphs: 🔗 (suspected wrong GH ref), 🗣️ (internal language),
                  📝 (too terse or missing context).
     An entry with multiple flags appears once per flag. -->

<!-- Report ends here. Do NOT append "end of report", a closing quote, a thanks
     note, or any terminal marker. -->
