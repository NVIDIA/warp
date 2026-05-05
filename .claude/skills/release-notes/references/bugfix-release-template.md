<!--
Bugfix release template (X.Y.Z, Z > 0). Fill in every {{PLACEHOLDER}}; drop
sections whose placeholders resolve to "no content" rather than leaving an empty
heading.

Bugfix releases are slimmed down compared to feature releases. The dominant
content is `## Highlights` (a categorized digest of the fixes), with optional
Announcements and a brief Acknowledgments section. There is no `## New features`
section.

The shape mirrors v1.12.1 and v1.11.1.
-->

# Warp v{{VERSION_STRING}}

Warp v{{VERSION_STRING}} is a bugfix release following v{{PREV_VERSION_STRING}}. For a complete list of changes, see the [changelog](https://github.com/NVIDIA/warp/blob/v{{VERSION_STRING}}/CHANGELOG.md).

## Highlights

{{HIGHLIGHTS_BULLETS}}

<!--
HIGHLIGHTS_BULLETS: 3-6 bold-prefixed bullets that group related fixes by
category. Each bullet is 1-3 sentences.

Categories are descriptive labels, not section names from the changelog. Pick what
fits the release:
  - **Tile Correctness**
  - **Silent Correctness Bugs**
  - **Type System and Tooling**
  - **CUDA Graphs**
  - **Code Generation**
  - **Developer Experience**
  - **Documentation**
  - **Build and Packaging**
  - **<release-specific theme>**

Shape:
- **<Category>**: <1-3 sentence summary of the fixes in this category, with
  `(#NNNN)` refs trailing each specific claim>.

Example (from v1.12.1):
> - **Tile Correctness**: Fixed several tile kernel issues, including kernel
>   dispatch using incorrect `block_dim` across devices (#1254), `wp.tile_matmul()`
>   and `wp.tile_fft()` ignoring module-level `enable_backward` (#1320), and
>   `@wp.func` with tile parameters failing to compile with shared-memory tiles
>   (#1313). Tile parameters in `@wp.func` are now passed by reference for both
>   register and shared storage, matching Python's semantics for mutable objects.

Group similar fixes — do NOT enumerate every changelog bullet as its own
highlight. The full changelog already does that.
-->

{{NOTABLE_NON_FIX_CHANGES_IF_ANY}}

<!--
NOTABLE_NON_FIX_CHANGES_IF_ANY: occasionally a bugfix release also ships a
removal or notable behavior change (e.g., v1.12.1 removed Kit extensions). If
the changelog has `### Removed` / `### Changed` content for this release,
include a bullet for it under Highlights with a category like "Kit Extensions
Removed" or "Behavior Change". Drop this placeholder if the release is
fixes-only.
-->

{{NEW_EXAMPLES_BULLET_IF_ANY}}

<!--
NEW_EXAMPLES_BULLET_IF_ANY: bugfix releases sometimes ship new examples (v1.12.1
shipped 4). Render as a final highlight bullet:

- **New Examples**: Added a differentiable 2-D Navier-Stokes optimization
  example and three `warp.fem` examples for Taylor-Green vortex,
  Kelvin-Helmholtz instability, and shallow water equations.

Drop entirely if no new examples landed.
-->

{{ANNOUNCEMENTS_SECTION_IF_ANY}}

<!--
ANNOUNCEMENTS_SECTION_IF_ANY: only include if the bugfix carries a deprecation
finalization or platform support change worth restating. Bugfix releases often
restate deprecation timelines from the parent feature release so users continue
to see them. Shape:

  ## Announcements

  ### Upcoming removals

  The following deprecations will be finalized in **Warp <next-version>**:

  - **<API> will be removed.** <Migration guidance>.

  ### Platform support

  ...

If there is nothing to announce, drop this whole `## Announcements` section.
-->

## Acknowledgments

{{ACKNOWLEDGMENTS_PREAMBLE}}

<!--
ACKNOWLEDGMENTS_PREAMBLE: usually:
  Thanks to <user> for <contribution> (#NNNN).
or, for multiple contributors:
  We thank the following contributors:

If there is one external contributor, render as a single sentence rather than a
bullet list (matches v1.12.1 style):
  Thanks to @username for <contribution> (#NNNN).
-->

{{ACKNOWLEDGMENTS_BULLETS}}

<!--
ACKNOWLEDGMENTS_BULLETS: bullet list when there are 2+ external contributors:
  - @username for <specific contribution> (#NNNN).

Drop the bullet list when the preamble was rendered as a single sentence.

If there are zero external contributors, replace the section body
with: "No external contributions in this release." (preserve the section heading).
-->
