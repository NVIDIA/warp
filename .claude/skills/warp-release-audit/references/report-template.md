<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

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
     once on main_ref, as with reverts or replayed commits) are NOT a bake-gap
     signal: the commit IS on main,
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
       - Experimental (only when the experimental capability is headline-worthy)
       - ⚠️ Breaking (if the headline item is a breaking change)

     Do NOT include counts ("4 new APIs were added"): that's already in the
     headline counts block above. Highlights are qualitative.

     Generic bullet shape:

     - **<capability headline>** ([GH-NNN](...)): <what users can now do and why
       it matters>. <optional bake hint>
     - **⚠️ Breaking: <behavior headline>** ([GH-NNN](...)): <affected users and
       concrete migration>.
     - **Experimental: <feature headline>** ([GH-NNN](...)): <change and
       advertised stability level>.
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
- [Changelog Review Notes](#changelog-review-notes) (only if the conditional
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

     The Symbol cell uses a short-form call shape with parameter names and
     defaults but no type annotations. Do not add parentheses to enums, scalar
     types, or decorators. -->

{{NEW_PYTHON_DETAIL_BLOCKS}}

<!-- Per-symbol block contract:

### `wp.<symbol>`

Links: [GH-NNN](...), commit(s): [sha](...)
Source: `<public source path>`
Bake: <bucket and days>

```python
<signature-shaped declaration>
"""<verbatim docstring, when present>"""
```

For classes, include the class docstring, constructor, and every additional
public method. For enums and flags, include every member and its documentation.
For kernel builtins, synthesize the public Python-style signature rather than
showing the registration call. -->

### Kernel scope

{{NEW_KERNEL_TABLES_BY_KIND}}

<!-- Same Kind grouping as Python scope. Kernel kinds typically include:
     "Tile operations", "Queries", "Types", "Primitives". -->

{{NEW_KERNEL_DETAIL_BLOCKS}}

---

## ⚠️ Breaking Changes

{{BREAKING_ENTRIES}}

<!-- Flat list. Do NOT group under identification-method subheadings. Include
     only alarming changes to supported, reachable surfaces. Planned removals
     with satisfied deprecation windows and experimental changes belong in
     Changes to Existing API with their softer labels, not in this count.

     Per-entry render format:

     ### <heading: symbol name or short descriptive title>

     Links: [GH-NNN](...), commit(s): [sha](...). 🟢 N days baked in main.

     [If signature diff applies, a fenced diff block.]
     [If behavior shifts, include a self-contained before/after example. Prefer
      existing tests/docs; otherwise use the verified reproduction and outputs.]

     [1-3 sentences of explanatory prose in plain user-facing language.]

     [Full CHANGELOG text blockquoted, if the entry came from CHANGELOG.]

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

<!-- Per-entry contract. Use a colon in the heading, not an em dash.

### `wp.<symbol>`: <change kind>

Breaking: **<Yes | No | Planned removal | Experimental>** (<short reason>)
Links: [GH-NNN](...), commit(s): [sha](...)
Bake: <bucket and days>
Deprecation window: <when applicable>

```diff
- <base signature when applicable>
+ <HEAD signature when applicable>
```

<plain-language effect and self-contained migration guidance when needed>

**From CHANGELOG**
> <full current entry>

For removals, also quote the matching prior deprecation entry. For experimental
changes, state the release that introduced the experimental marker. -->

---

## Behavioral & Support Changes

<!-- Group by topic with short descriptive section headings synthesized from
     the entry content. Use colons in headings if separation is needed.
     Related topics should live together.

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

     1. Both unmatched-fragment list AND language-review flags are empty:
        Render nothing. No appendix heading, no trailing section.

     2. Exactly ONE is non-empty: Render that one as a top-level section
        (no "Audit Appendix" umbrella) with the table inside <details>.

        ## Changelog Fragments Without Matching Commits
        <details>
        <summary>N entries (click to expand)</summary>

        | Source fragments | Entry | GH refs | Suspected reason |
        |---|---|---|---|
        | `changelog/...` | full entry text | ... | ... |

        </details>

     3. Both are non-empty: Render an umbrella section; each subsection gets
        its own <details>.

        ## Audit Appendix

        <details>
        <summary>N changelog fragments without matching commits (click to expand)</summary>

        | Source fragments | Entry | GH refs | Suspected reason |
        |---|---|---|---|
        ...

        </details>

        <details>
        <summary>N changelog entries flagged for review (click to expand)</summary>

        | Source fragments | Entry | Flag | Why |
        |---|---|---|---|
        ...

        </details>

     Column rules for BOTH tables: source fragment paths and full entry text
     (no truncation).
     Flag glyphs: 🔗 (suspected wrong GH ref), 🗣️ (internal language),
                  📝 (too terse or missing context).
     An entry with multiple flags appears once per flag. -->

<!-- Report ends here. Do NOT append "end of report", a closing quote, a thanks
     note, or any terminal marker. -->
