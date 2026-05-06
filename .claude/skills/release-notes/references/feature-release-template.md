<!--
Feature release template (X.Y.0). Fill in every {{PLACEHOLDER}}; drop sections
whose placeholders resolve to "no content" rather than leaving an empty heading.

The shape mirrors recent feature releases (v1.10.0, v1.11.0, v1.12.0). It is
NOT a rigid form — drop or rename sections as the release content warrants.

Sections appear in roughly impact order:
  1. Lead feature(s) under `## New features` with the most exciting first.
  2. Themed groups (tile, JAX, fem, compilation, etc.).
  3. Performance / language enhancements.
  4. New examples (if notable).
  5. Bug fixes (only if any are worth highlighting; the changelog has them all).
  6. Announcements (deprecations, platform support).
  7. Acknowledgments.
-->

# Warp v{{VERSION_STRING}}

{{INTRO_PARAGRAPH}}

<!--
INTRO_PARAGRAPH: 2-3 sentences. Lead with the unlock from the lead feature.
Then 1-2 supporting themes. Closer about non-headline improvements.

Example shape (DO NOT copy):
> Warp v1.13 introduces experimental graph capture serialization with CPU
> replay, letting captured graphs roundtrip through a portable .wrp file
> and load from standalone C++. This release also adds the wp.bfloat16
> scalar type, a pluggable allocator interface, and a batch of new tile
> primitives.
-->

{{IMPORTANT_HEADER_NOTE_IF_ANY}}

<!--
IMPORTANT_HEADER_NOTE_IF_ANY: Used only when this release contains a high-impact
removal or breaking change that the user must see before reading the rest of the
notes. Render as bold-paragraph or > [!IMPORTANT] admonition. Drop this
placeholder entirely if there is nothing to call out at the top.

Example:
> **Important**: This release removes the `warp.sim` module (deprecated since
> v1.8), which has been superseded by the [Newton physics engine](...).
-->

For a complete list of changes, see the [full changelog](https://github.com/NVIDIA/warp/blob/v{{VERSION_STRING}}/CHANGELOG.md).

## New features

### {{LEAD_FEATURE_TITLE}}

{{LEAD_FEATURE_EXPERIMENTAL_ADMONITION}}

<!--
LEAD_FEATURE_EXPERIMENTAL_ADMONITION (only when the lead feature is experimental):
> [!IMPORTANT]
> This is an experimental feature. The API may change without a formal deprecation cycle.
-->

{{LEAD_FEATURE_PROSE}}

<!--
LEAD_FEATURE_PROSE: 2-3 sentences explaining what is now possible. Lead with the
unlock, then name the supporting symbols. Reference issues/PRs with `(#NNNN)`.
-->

```python
{{LEAD_FEATURE_CODE}}
```

<!--
LEAD_FEATURE_CODE: a complete, runnable, 8-15 line Python example showing the
simplest meaningful usage. Imports + kernel/call + result comment.
-->

{{LEAD_FEATURE_CONSUMER_CODE_IF_CROSS_LANGUAGE}}

<!--
LEAD_FEATURE_CONSUMER_CODE_IF_CROSS_LANGUAGE: when the lead feature crosses a
language or runtime boundary (Python -> C++, Python -> JAX, Python -> standalone
runtime), include a SECOND code block on the consumer side. Render as:

  Loading and replaying from standalone C++ (no Python interpreter required):

  ```cpp
  // headers, init order, load + replay calls
  ```

  See [`<example-name>`](https://github.com/NVIDIA/warp/blob/v<version>/warp/examples/cpp/<path>)
  for the full demo.

Naming the consumer-side header file or import path is mandatory. If a working
example exists in `warp/examples/`, link it on the release tag URL. Drop this
placeholder entirely when the lead feature is Python-only.
-->

{{LEAD_FEATURE_ARTIFACT_BLOCK_IF_ANY}}

<!--
LEAD_FEATURE_ARTIFACT_BLOCK_IF_ANY: when the lead feature writes a new on-disk
artifact (file format, serialized graph), describe what gets written beyond the
obvious file. Sidecar directories, arch-pinned binaries, version metadata are
common footguns. Render as a literal directory tree:

  **What gets written:**

  ```
  scene.wrp                 # operation byte stream + region snapshots + metadata
  scene_modules/
      abc123.cubin / .meta  # one per kernel module, arch-pinned
  ```

Drop entirely when the lead feature has no on-disk artifact.
-->

**Key capabilities:**

{{LEAD_FEATURE_BULLETS}}

<!--
LEAD_FEATURE_BULLETS: 3-6 bullets naming the supporting APIs / parameters /
behaviors and what each unlocks. Bold the leading clause.

When the lead feature is experimental, also include a "Known limitations" tail
bullet enumerating what does NOT yet work, sourced via the protocol in
references/feature-investigation.md (search the implementation for "not yet
supported" / "currently only" patterns). Examples:
  - **Known limitations:** Volume and BVH serialization are not yet supported
    (only `wp.Mesh` handles remap via `wp.handle`); CPU `.wrp` load currently
    requires a CUDA-built Warp library; saved CUBINs are pinned to one compute
    capability.

Skipping the limitations bullet for an experimental feature is a review-fail
pattern — readers discover the limit only after their code breaks.
-->

{{ADDITIONAL_LEAD_FEATURE_BLOCKS}}

<!--
ADDITIONAL_LEAD_FEATURE_BLOCKS: if the release has 2 lead features (rare), repeat
the lead-feature block shape for each. Otherwise drop this placeholder entirely.
-->

{{REMAINING_NEW_FEATURE_SUBSECTIONS}}

<!--
REMAINING_NEW_FEATURE_SUBSECTIONS: per-feature `### …` blocks for other notable
additions in `### Added`. Each block carries:

  ### Subject

  > [!IMPORTANT] (only if experimental)
  > This is an experimental feature. ...

  Prose paragraph (1-3 sentences with `(#NNNN)`) explaining what is now possible.

  ```python
  # 5-15 line runnable snippet showing the simplest meaningful usage
  ```

A code block is the DEFAULT for every `### …` block, not the exception. The only
entries that may legitimately be prose-only are single-parameter tidy-ups,
single-flag toggles, or pure perf optimizations with no API change. When in
doubt, include the snippet.

Group similar items (e.g., quaternion helpers, tile primitives, jax features) into
their own `## …` sections rather than burying related items as `###` peers under
a single `## New features`.
-->

## {{THEME_1_TITLE}}

{{THEME_1_BODY}}

<!--
THEME_1_TITLE / THEME_1_BODY: Repeat for each themed group. Common shapes:
  - "## Tile programming enhancements" (and "## Tile performance improvements"
    when there are notable perf changes worth a separate block)
  - "## JAX integration"
  - "## warp.fem enhancements"
  - "## Compilation and tooling"
  - "## Performance improvements"
  - "## Language enhancements"

Each theme section contains 2-6 `###` subtopics. Each named API / capability
gets its OWN short Python snippet. Multiple related APIs (e.g., a family of
new tile primitives) can share a single setup section, but each new symbol's
call site is shown literally.

When a theme has only 2-3 single-line entries with no behavioral nuance, a
flat bullet list is acceptable in place of `###` subtopics:
  ## <Theme>
  - Bullet 1 with `wp.foo()` call shape (#NNNN).
  - Bullet 2 with `wp.bar()` call shape (#NNNN).
But the bar for "no snippet needed" is high; prefer `###` + snippet.
-->

{{ADDITIONAL_THEMES}}

<!--
ADDITIONAL_THEMES: more `## <Theme>` sections. Order by user-impact roughly,
not by changelog ordering.
-->

{{NEW_EXAMPLES_SECTION_IF_ANY}}

<!--
NEW_EXAMPLES_SECTION_IF_ANY:

  ## New examples

  The new [`example_name`](https://github.com/NVIDIA/warp/blob/v<version>/warp/examples/<path>)
  example demonstrates ... It uses ... showcasing ...

Always link example paths on the release tag URL, never on `main` or by SHA.
Drop the section if no new examples warrant top-level mention.
-->

{{BREAKING_CHANGES_SECTION_IF_ANY}}

<!--
BREAKING_CHANGES_SECTION_IF_ANY: a TOP-LEVEL `## Breaking changes` section is
warranted when a release has 1+ user-observable breaking change (entries flagged
`**Breaking:**` in the CHANGELOG, plus removals that affect call shape). Each
breaking-change subsection carries a before/after code snippet, NOT prose. See
references/style-rules.md "Breaking changes use before/after snippets" for the
shape. Example:

  ## Breaking changes

  ### Backward-pass zeros output gradients (#1062)

  ```python
  # v1.12 behavior: out.grad still holds the seed after backward()
  out = wp.zeros(N, dtype=float, requires_grad=True)
  ...
  tape.backward()
  print(out.grad)   # v1.12: ones, v1.13: zeros

  # v1.13 migration: ...
  ```

  One-paragraph rationale: why the old behavior was incorrect, what the
  performance implications of the new behavior are.

If the only breaking changes are uninteresting (single-typed-renames), they may
stay in the CHANGELOG and be omitted here. Use judgment — readers cannot afford
to miss numerics or call-shape shifts.
-->

{{BUG_FIXES_SECTION_IF_ANY}}

<!--
BUG_FIXES_SECTION_IF_ANY: optional. Only include if there are 2+ fixes worth
highlighting (the changelog already lists everything). Group by category:

  ## Bug fixes

  ### <Category>

  Short paragraph summarizing related fixes with trailing `(#NNNN)` refs.
  Include a before/after snippet for any fix where the old behavior was a
  silent no-op or a silent semantic shift (e.g., GH-1336-style mutating-builtin
  bugs at Python scope).

Bugfix releases use the bugfix template instead. This section in a feature release
is for fixes that are noteworthy on their own merits.
-->

## Announcements

{{ANNOUNCEMENTS_BODY}}

<!--
ANNOUNCEMENTS_BODY: zero or more sub-sections. Common shapes:

  ### Upcoming removals

  The following deprecations will be finalized in **Warp <next-version>**:

  - **<API> will be removed.** <Migration guidance>.

  ### Platform support

  - **Python 3.X**: <support change>.
  - **CUDA <toolkit>**: <support change>.
  - **<OS / arch>**: <support change>.

  ### <other announcement>

  <Prose>

If there is nothing to announce, drop this whole `## Announcements` section.
-->

## Acknowledgments

{{ACKNOWLEDGMENTS_PREAMBLE}}

<!--
ACKNOWLEDGMENTS_PREAMBLE:
  We also thank the following contributors from outside the core Warp development team:
-->

{{ACKNOWLEDGMENTS_BULLETS}}

<!--
ACKNOWLEDGMENTS_BULLETS: one bullet per external contributor. Format:
  - @username for <specific contribution> (#NNNN).

If there are zero external contributors, replace the bullet list with:
  No external contributions in this release.
-->

---

For a complete list of changes, see the [full changelog](https://github.com/NVIDIA/warp/blob/v{{VERSION_STRING}}/CHANGELOG.md).
