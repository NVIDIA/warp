---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: warp-release-audit
description: "Use when generating a Warp pre-release or release-candidate audit report from Towncrier fragments and release history."
license: Apache-2.0
---

# Release Audit

Generates a markdown audit of the upcoming Warp release for keep/defer decisions. Runs in two modes (auto-detected): a **pre-release** spot-check while work is still landing on main, or a **release-candidate** readiness review after the release branch is cut.

**Output:** a single markdown report, filed according to the destination chosen in Phase 1:
- **Secret gist** (default when `gh` is available and authenticated): stable filename `warp-<version-string>-<prerelease|rc>-report.md`, stable description `Warp <version-string> <Pre-Release|Release Candidate> Report`. Later runs against the same version revise the same gist in place; prior versions are preserved in the gist's git history.
- **Local markdown file** (fallback when `gh` unavailable, or opt-in when `gh` available): dated path at `$(git rev-parse --show-toplevel)/warp-<version-string>-<prerelease|rc>-report-<YYYY-MM-DD>.md`. Not auto-committed; user moves, shares, or deletes as desired.

**Inputs inferred from repo state:** target version from `VERSION.md`, checked against `warp/config.py`; base from the latest previous-minor tag; head from `upstream/release-<target>` when present, otherwise `upstream/main`.

**References to load on demand:** use `references/destination-rules.md` in Phase
1; `references/classification-rules.md` in Phases 3-5;
`references/language-review-examples.md` in Phase 5a; and
`references/report-template.md` plus `references/render-rules.md` in Phase 6b.

## Phase 1 — Align on scope

1. Read the version string from `VERSION.md`, Warp's canonical version source, and verify that `warp/config.py` declares the same value. If either value is missing or unparsable, or if they differ, show both raw values and wait for explicit user confirmation instead of selecting one. (`warp/__init__.py` aliases `config.version` and is not an independent version source.) Parse the confirmed version to extract the target minor (e.g., `1.13.0dev0` → target `1.13`) and determine the report mode:
   - If the version string contains `"rc"` (e.g., `1.13.0rc1`) → **RC mode**: this is a release-candidate readiness report.
   - If the version string contains `"dev"` (e.g., `1.13.0dev0`) → **Pre-release mode**: this is an early-stage audit of unreleased work.
   - Otherwise → pre-release mode (default), but record the raw version string so the header can show it as-is.

2. Parse the target version into integers `(major, minor)` (e.g., `1.13.0dev0` → `(1, 13)`). Enumerate previous-minor tags:
   ```bash
   git tag --list 'v<major>.<minor - 1>.*' --sort=-v:refname
   ```
   Take the first result as the base candidate. **Major-boundary fallback**: if `minor == 0` (e.g., target `(2, 0)`), there is no `v2.-1.*` line. In that case, enumerate tags from the previous major instead: `git tag --list 'v<major - 1>.*' --sort=-v:refname`, and take the highest as the base candidate. Always use integer math on the parsed `(major, minor)` tuple; never treat the dotted version string as a float (`1.13 - 0.1` is not `1.12`).

3. Probe for the head:
   ```
   git rev-parse --verify upstream/release-<target>
   ```
   If this succeeds, head = `upstream/release-<target>` and this is also a strong signal for **RC mode** (the branch-cut has happened). Otherwise try `origin/release-<target>`; otherwise head = `upstream/main` (falling back to `origin/main`, then `main`) which is **pre-release mode**. Record whichever fallback was used for the report header.

4. **Reconcile mode** from version string and head:
   - Version says RC AND head is a release branch → **RC report** (strong match).
   - Version says RC but head is main (branch not cut yet) → **RC report** (version is authoritative; note the mismatch in the header).
   - Version says dev AND head is a release branch → **RC report** (branch cut implies we're past the dev window).
   - Version says dev AND head is main → **Pre-release report**.

5. Read `references/destination-rules.md`, probe `gh`, and look up matching
   gists as specified there.

6. Present the resolved refs and destination options, then **wait for explicit
   user confirmation**. Record the selected destination token for Phase 6c and
   do not continue until the user confirms.

## Phase 2 — Gather ground truth

1. Resolve `<skill-dir>` to the directory containing the currently loaded `SKILL.md` for this skill (for example, `<repo>/.claude/skills/warp-release-audit` or `<repo>/.codex/skills/warp-release-audit`). Do not hardcode `.claude` or `.codex`; the skill content must work from either tree. Run the commit-list tool via `uv run` as a single command:
   ```bash
   uv run "<skill-dir>/scripts/list_commits.py" \
     --base <base-ref> \
     --head <head-ref> \
     --report-date "$(date +%F)" \
     --main-ref <resolved-main-ref>
   ```
   Use `$(date +%F)` literally so the shell supplies today's date; do NOT substitute a date you generated yourself, since training-time bias can produce stale `days_since_merge` / `days_in_main` values. Capture stdout as the `commit_list_json`.

2. Read `changelog/README.md` and enumerate fragment paths from `<head-ref>`, not from the current working tree:

   ```bash
   git show <head-ref>:changelog/README.md
   git ls-tree -r --name-only <head-ref> -- changelog
   ```

   Exclude `changelog/README.md`. Record path, identifier, category, optional counter, and full content. Numeric IDs are GitHub issues; `+slug` IDs are
   orphans with no generated issue link.

3. Render those exact fragments with pinned Towncrier in a temporary detached
   worktree, keeping the selected ref's configuration and fragment set together:

   ```bash
   audit_worktree=$(mktemp -d)
   git worktree add --detach "$audit_worktree" <head-ref>
   trap 'git worktree remove --force "$audit_worktree" 2>/dev/null || true' EXIT
   draft_status=0
   (cd "$audit_worktree" && uvx --from towncrier==25.8.0 towncrier build \
     --draft --version <target-X.Y.Z> --date "$(date +%F)") || draft_status=$?
   git worktree remove --force "$audit_worktree"
   trap - EXIT
   test "$draft_status" -eq 0
   ```

   Capture `## [<target-X.Y.Z>]` as the pending release view. Always remove the temporary worktree, including after failure. A draft never modifies the ref.

4. Parse the six rendered subsections. For every bullet, extract:
   - **Raw text (FULL — never truncate)**: the full bullet content (may span multiple lines).
   - **Section**: one of the six names above.
   - **GH refs**: regex `GH-(\d+)` over the bullet text. Dedup.
   - **Breaking flag**: presence of the literal string `**Breaking:**` in the bullet.
   - **Source fragments**: numeric paths matching generated GH refs, or orphan
     paths whose content produced the bullet. Identical numeric fragments may
     map to one Towncrier bullet with several links.

5. Read `CHANGELOG.md` at `<head-ref>` separately as historical release data.
   Do not infer pending content from its newest released section.

6. Read `docs/user_guide/compatibility.rst` and `design/deprecations.md` at
   `<head-ref>`. Treat them as the repository's canonical definitions of the
   documented stability boundary and planned deprecation schedule. If either
   file is absent, record that the policy source was unavailable and avoid
   inventing stability promises from file names, symbol visibility, or presumed
   downstream use.

## Phase 3 — Cross-reference

1. **Build the commit ↔ pending-entry join** on GH-ref overlap and fragment
   provenance:
   - For each rendered entry with at least one GH ref, find commits (from
     `commit_list_json`) whose `gh_refs` intersect.
   - For every entry with zero matches, inspect the introducing and modifying
     commits for each source fragment:
     ```
     git log --reverse --follow --format='%H|%s|%cs' \
       <base-ref>..<head-ref> -- <fragment-path>
     ```
     The fragment-introducing commit is the primary backing commit unless it is
     a fragment-only follow-up; in that case inspect the issue ID, symbols, and
     nearby history to find the code commit. Orphan fragments use the same path
     history and do not require a GH ref.
   - After these passes, surface entries with no backing commit in the
     **Changelog Fragments Without Matching Commits** appendix. Include every
     source path and the full rendered text; never truncate.

2. **Do NOT build an audit trace of unmatched commits.** The old "commits
   without changelog entries" appendix adds noise without value. Commits that
   do not map to a pending entry are not surfaced in the report.

## Phase 4 — Analyze API surface

Before labeling any compatibility change, record four facts:

- **Surface:** the documented API, behavior, format, or platform affected.
- **Reachability:** how a user reaches it through a supported entry point.
- **Deprecation:** stable, experimental, deprecated with the scheduled removal
  release, or outside the documented stability boundary.
- **Action:** dismiss, document normally, label as a planned removal, or raise
  an alarming breaking-change finding.

A source or ABI difference is not automatically release risk. Raise an
alarming finding only when the change affects a reachable surface covered by
the compatibility policy, or when base-versus-HEAD execution confirms a
supported user-visible behavior change. Direct use of internal native headers
is in scope only if the current policy explicitly promises that interface.

### 4a — Determine what is genuinely NEW API

For each pending `Added` entry, extract the named symbol(s) (text in backticks matching `wp.*`, `@wp.*`, or patterns like `ClassName`).

For EACH named symbol, check if it existed at base:

- For top-level `wp.X`: look at `git show <base>:warp/__init__.py` — does it re-export `X`?
- For `wp.submodule.X`: read the submodule source at base.
- For `@wp.kernel`, `@wp.struct`, `@wp.func` decorator-style: check `warp/__init__.py` at base.
- For kernel-scope builtins (found in `warp/_src/builtins.py`): check if the builtin was registered at base via `git show <base>:warp/_src/builtins.py | grep '"<name>"'`.

Classification:
- **Genuinely new** (symbol was not present at base) → **New API** section.
- **Existed at base** (entry is adding a new parameter, option, or capability to something that already existed) → **Changes to Existing API** section with a "capability extension" or "new parameter" kind. Cite the backing commit's signature diff.
- **Does not name a single symbol** and describes a cross-cutting capability →
  **Behavioral & Support Changes** unless one mentioned symbol is genuinely new.

**If an entry mixes new and pre-existing symbols**, split it: genuinely new
symbols go to New API and extensions go to Changes to Existing API.

### 4b — Resolve New API signatures + docstrings

For each Python-scope symbol confirmed as new:

1. Find its real source module via `warp/__init__.py` re-exports at HEAD.
2. `ast.parse` the source module; find the `FunctionDef` / `ClassDef`.
3. Extract the signature by re-stringifying the args (preserve type annotations).
4. Extract the docstring verbatim via `ast.get_docstring(node)`.
5. Render as shown in the template.

For each kernel-scope builtin confirmed as new:

1. Read `warp/_src/builtins.py` at HEAD. Find the `add_builtin("<name>", ...)` call.
2. **Do NOT show the `add_builtin()` registration call itself.** Instead, SYNTHESIZE a Python-function-style signature from its arguments:
   - The builtin name becomes the function name.
   - Each entry in `input_types=` or `inputs=` (or positional args) becomes a function parameter with type annotation.
   - The `value_type` or `output` arg (or return value) becomes the return annotation.
   - Rewrite Warp internal type classes (e.g., `tile(dtype=...)`, `vector(length=N, dtype=...)`) as they appear — this is the user-facing form.
3. Example of the required output shape:
   ```python
   builtin_name(inout: tile(dtype=Float, shape=tuple[int, ...])) -> None
   ```
   NOT:
   ```python
   add_builtin("builtin_name", input_types={"inout": tile(...)}, ...)
   ```
4. The docstring comes from the `doc=` parameter of `add_builtin()`. Render verbatim as a blockquote.

### 4c — Symbol resolution fallbacks

- **Symbol in backticks doesn't resolve at HEAD**: render the entry with a ⚠️ note: "Couldn't resolve `<symbol>` in source — verify entry names a real public symbol." Do not fabricate a header like `wp.*(no symbol)*`. Use the entry's natural subject as the section title.
- **Entry describes a topic, not a symbol**: use a short descriptive title
  summarized from the entry, not a synthetic `wp.*` name.
- **Header naming rule**: use the actual fully-qualified symbol or a short
  descriptive title extracted from the entry. Never use `wp.*`,
  `wp.(something)`, or another stub pattern.

### 4d — Changed / Removed / Deprecated signature diffs

For each pending entry in Changed / Removed / Deprecated (plus any "capability extension" entries routed here from 4a):

- Compute signatures at base and HEAD (same resolution as 4b).
- For signature-shape changes: render a fenced `diff` block showing `-` and `+` lines.
- For semantic-only changes (no signature shift) where the entry has `**Breaking:**`: skip the diff block; include the backing commit's URL and the full CHANGELOG text.
- For Removed entries: show the old signature on a `-` line; omit `+`.

**Deprecation-window lookup for Removed entries.** For every Removed entry (and every Changed entry whose prose describes a removal), search CHANGELOG.md for the matching prior Deprecated entry:

1. Extract distinctive tokens from the Removed entry: the named symbol(s) in backticks and, if the entry carries a GH ref, that ref number.
2. Scan all released-version sections of `CHANGELOG.md` beneath the Towncrier
   insertion marker, top-down, for a `### Deprecated` bullet that names the
   same symbol(s) OR the same GH ref. The FIRST such entry is the latest match
   because `CHANGELOG.md` is reverse chronological.
3. Record: (a) the release version heading that contains the Deprecated entry (e.g., `1.11.0`), (b) the full Deprecated entry text.
4. In the rendered Removed entry, include a one-line deprecation window: `Deprecated in X.Y.Z; removed here.` Do NOT fabricate a version if no prior entry is found.

**Removal risk.** Apply the stability and reachability gate before assigning an
alarm level:

- If a supported stable surface has no prior deprecation, surface
  `🚨 Policy: removed without prior deprecation` in Breaking Changes. A
  simultaneous deprecate-and-remove is the same policy violation.
- If the documented deprecation window is satisfied, label it **Planned removal**,
  provide migration guidance, and do not count it as an alarming breaking change.
  Keep it in Changes to Existing API so users can find the removal and replacement.
- If the surface is experimental, use the experimental treatment in Phase 4h.
- If the surface is outside the documented stability boundary or was never
  reachable through a supported entry point, do not manufacture a compatibility
  finding. Preserve the CHANGELOG entry in its normal category when one exists.

Include the deprecation window in every rendered removal detail and in the
Changes-to-Existing-API row. For an unplanned removal that also appears in
Breaking Changes, include it there as well.

### 4e — Signature-AST diff for unlabeled breaking changes

Independently of CHANGELOG content, compare the committed public API at base and
HEAD. Read `references/classification-rules.md` for module selection, public-stub
coverage, signature compatibility, and manual fallbacks. Always include `warp`
and add public submodules named by pending entries:

```bash
uv run "<skill-dir>/scripts/diff_public_api.py" \
  --base <base-ref> \
  --head <head-ref> \
  --module warp \
  --module <public-submodule>
```
Repeat `--module` per resolved submodule and omit the placeholder when there are
none. Capture stdout as `api_diff_json`.

For each `api_diff_json.changes[]` item:
- Classify the fact through the stability and reachability gate before deciding
  whether it belongs in Breaking Changes.
- Merge supported signature changes and public-stub removals into Changes to
  Existing API; add alarming entries only for unplanned stable breaks.
- Surface helper warnings concisely and manually inspect the affected path.
- Render supported breaking signature shifts with a `diff` and a concrete
  before/after call.

**Removed symbols require risk classification, not a marker check.** Do not
flag a Removed entry merely for lacking `**Breaking:**`. Classify it using the
stability, reachability, and deprecation-window rules above. Only unplanned
removals of supported stable surfaces enter the alarming Breaking Changes
count; planned and experimental removals use their respective labels.

### 4f — Deprecated compatibility path exceptions

Independently inspect deprecated compatibility paths for newly introduced exceptions. This catches breaks in old import paths or wrappers that still exist only to warn and forward callers to a replacement API.

1. Identify candidate paths from commits and CHANGELOG entries, then discard
   paths outside the documented stability boundary:
   - Files or packages whose path contains `deprecated`, `deprecation`, `compat`, `compatibility`, `backcompat`, or a deprecated namespace name.
   - Public modules named in `Deprecated`, `Removed`, or migration-style CHANGELOG entries.
   - Modules that issue deprecation warnings, forward imports from an old
     namespace to a replacement, or provide compatibility aliases.
2. Compare candidate files at base and HEAD. Look specifically for new `raise` statements, changed exception types or messages, stricter guard conditions before forwarding/delegation, removed `try`/`except` fallbacks, and import-time errors added to a path that previously warned and delegated.
3. If the old deprecated path can now raise a new or stricter exception before reaching the replacement API, add a Breaking Changes entry and a Changes-to-Existing-API row with kind `semantic change` and description `new exception in deprecated compatibility path`.
4. Include a short before/after snippet showing the deprecated call or import form, the new exception behavior, and the replacement path users should call instead. If the behavior is ambiguous from the diff, verify with a minimal Python script at base and HEAD before reporting it.
5. Do not suppress this merely because the path is deprecated. A documented,
   still-supported compatibility path carries a migration contract until its
   scheduled removal. An internal or unsupported compatibility helper does not.

### 4g — Semantic-breaking verification (the assistant performs the verification)

Inspect every pending entry and backing commit whose diff plausibly changes
supported, user-observable behavior. Use changes under code generation and
native implementation paths as high-risk hints, not an exhaustive scope; Python
runtime, serialization, interop, and platform-support changes may also require
verification. Conversely, touching a high-risk path is not evidence that an
internal symbol is public.

For each candidate:

1. Skip if the commit is already mapped to a pending entry carrying `**Breaking:**` (it's already in Breaking Changes).
2. Read the commit's diff: `git show --stat <sha>` then `git show <sha>` for small diffs, or read specific hunks for large ones.
3. **Triage** into one of three buckets:
   - **Clearly not breaking** → drop. Examples: renaming internal symbols, comment/format changes, pure internal refactors with no emitted-code difference, performance optimizations that preserve semantics, test-only changes, build-system changes, bug fixes where the pre-fix behavior was itself a bug.
   - **Clearly breaking** with an obvious user-observable shift visible from the diff alone → include directly (proceed to step 5).
   - **Ambiguous** — the diff suggests the change could affect emitted code or runtime behavior, but the assistant cannot tell from reading alone whether a user would observe a difference → **verify by running code** (step 4).

4. **Verification by running code.** For ambiguous candidates, the assistant must actually run Warp at both base and HEAD and compare observable output:
   - Build Warp at HEAD using `uv run build_lib.py --quick` (~2-4 min) if not already built. If a build is already current, skip.
   - Check out the base tag in a separate worktree or save the current HEAD state, build Warp at base, capture the built library. `git worktree add` is useful here to avoid disturbing HEAD. Alternatively, git-stash + checkout + build + stash-pop.
   - Write a minimal Python test script that exercises the hypothesized behavior. The script should live under `/tmp/` (never commit). Example for a numerical-algorithm change: a small kernel that applies the changed op to a fixed input and prints the result. Example for a codegen change: a kernel whose emitted code should differ; compare via `wp.get_module().save_kernel_source(...)` or equivalent introspection.
   - Run the test against the base build and the HEAD build; capture outputs.
   - Compare: if outputs agree → change is NOT user-observably breaking → drop.
   - If outputs differ → confirmed breaking. Proceed to step 5, using the actual test script and its before/after outputs as the evidence in the report.
   - Restore the worktree/HEAD state so the session continues cleanly.

   **Never punt with "please verify".** If a candidate is ambiguous, either verify it by running code or drop it. Unverified flags do not land in the report.

5. For each confirmed breaking change, add an entry to the Breaking Changes section with:
   - A short descriptive heading (no em dashes; use a colon or just the name).
   - A 1-2 sentence summary of what changed and why it affects users.
   - **A before/after code snippet** illustrating the change. For author-labeled or signature-diff cases, synthesize from the diff. For verified semantic breaks, use the actual test script + outputs captured in step 4.
   - Commit link(s).
   - GH ref link (if any) in the entry text.

   Make every migration example self-contained: define the affected kernel,
   struct, argument, or setup needed to understand the call. Prefer a complete
   example from existing tests or documentation. If neither is suitable, write
   a minimal reproduction and verify it at base and HEAD. Do not synthesize an
   abstract call that omits the declaration responsible for the behavior.

**Never produce an unexamined list of candidates.** Every Breaking Changes entry either has explicit CHANGELOG backing, a signature-diff detected shape change, a public stub removal, a new exception in a deprecated compatibility path, or assistant-verified behavioral evidence.

### 4h — Experimental-marker cross-reference

Some symbols are shipped with an explicit `**Experimental**` marker in the CHANGELOG entry that introduced them. Changes to those symbols do NOT carry the same stability contract as changes to stable APIs: the whole point of the marker is to reserve the right to break them. The report must reflect that so the release manager does not over-weight the concern.

For each entry in Breaking Changes, Changes to Existing API, and Removed (as collected through 4a–4g), determine whether the affected symbol or feature area is currently experimental:

1. Collect candidate symbols and feature-area phrases from the entry:
   backticked identifiers, class names, and distinctive descriptive nouns.
2. Search all released-version sections of `CHANGELOG.md` beneath the Towncrier
   insertion marker for bullets that both carry `**Experimental**` (bold, with
   or without trailing colon) AND name one of the candidates from step 1. Also
   match via GH ref if the current entry and a prior experimental entry share a
   GH number.
3. If a match exists AND there is no subsequent CHANGELOG bullet in a later released version explicitly promoting the symbol to stable (e.g., "Promote `wp.Foo` out of experimental", "Stabilize `wp.Bar`"), the symbol is still experimental. Record: (a) the release version that introduced the symbol as experimental, (b) the full text of that introduction bullet.
4. Also check the module source at HEAD for an in-code `.. experimental` / `Experimental:` / `experimental_api` / `@experimental` annotation on the symbol's declaration. If present, treat as experimental regardless of CHANGELOG signal.

Tag every matched entry internally as `experimental=True`. Do not alter the CHANGELOG text itself.

**How the tag changes rendering:**
- Do not include an experimental-only compatibility change in Breaking Changes
  or the alarming breaking count. Keep it in Changes to Existing API.
- In the Changes-to-Existing-API table, show `Experimental` rather than `Yes`,
  even when the change is technically source-breaking.
- Include it in Release Highlights only when the capability itself is
  headline-worthy. Label it `Experimental:` without a warning glyph and state
  the advertised stability level rather than leading with migration urgency.

**Never drop the entry.** Experimental treatment changes placement and tone,
not visibility. A removed or signature-changed experimental symbol still
appears in Changes to Existing API.

## Phase 5 — Review CHANGELOG language and bake

### 5a — Language review (renders as the **Changelog Review Notes** appendix)

Read `references/language-review-examples.md`. For EACH rendered pending entry,
apply LLM judgment and identify its source fragment path(s):

- **🔗 Wrong ref (tier-1)**: for every GH ref in the entry, fetch the mapped commits' subjects and paths. If the entry topic doesn't match the commits' actual scope, flag.
- **🔗 Wrong ref (tier-2)**: if `gh --version` and `gh auth status` both succeed, run `gh issue view <num> --json title,body` per ref and compare issue title to entry topic. Skip silently if `gh` unavailable.
- **🗣️ Internal language**: internal module paths (`warp._src.*`), C++/CUDA type names (`launch_bounds_t`, `tile_register_t`), private identifiers.
- **📝 Too terse**: under ~10 words with no context.

Record flagged entries. Keep the FULL entry text in the audit table — do not truncate.

### 5b — Bake aggregation

**Pre-release mode (`resolved.head.sha == resolved.main_ref.sha`).** Every commit's main equivalent is itself, so `days_in_main == days_since_merge` and the bake distribution would just restate the age histogram. Render an "Age distribution" table from `days_since_merge` (same 🟢/🟡/🟠 thresholds), label the column "Days since merge", and skip both the "Bake distribution" table and the anomaly banner. There is no meaningful "didn't bake on main" condition when head IS main.

**RC mode (`resolved.head.sha != resolved.main_ref.sha`).** Partition commits by `main_match_state`:

- `state == "unique"`: bucket by `days_in_main` into **🟢 (>14 days)**, **🟡 (7–14 days)**, **🟠 (<7 days)**.
- `state == "missing"`: subject not present on main_ref. Inspect the commit before
  calling it a bake gap. Exclude expected release bookkeeping when the diff only
  synchronizes release metadata or release-specific documentation and does not
  change user-facing product or distribution behavior. Use repository policy,
  release history, and the semantic content of the diff; subject wording and a
  fixed path allowlist are hints, never the deciding rule. Count the remaining
  missing commits separately and, if non-zero, fire the ⚠️ banner in the report
  header. Optionally summarize excluded bookkeeping in a quiet footnote.
- `state == "ambiguous"`: subject appears more than once on main_ref, as with
  reverts or replayed commits. The commit IS on main; the script just could not
  pick a single canonical occurrence. Render a separate row labeled
  "⚪ ambiguous main match: K commits" and do not fire the banner.

If `resolved.empty_main_index == true`, the bake table is meaningless: every commit will resolve as `missing`. Render only `days_since_merge` stats and surface the empty-main-index condition prominently in the report header (e.g., "main_ref `<ref>` had no commits in `<base>..<main_ref>` — main bake unverifiable") instead of firing the routine bake-gap banner.

Never compare a `days_in_main` of `null` (emitted for both `missing` and `ambiguous`) to the numeric thresholds.

## Phase 6 — Write report to the chosen destination

### 6a — Draft the release highlights

Before filling the template, synthesize the `{{HEADLINE_SUMMARY}}` section. This is the only part of the report that requires qualitative judgment rather than mechanical rendering. Everything else flows from the cross-reference and classification work in Phases 3-5; this step picks what a reader should know *first*.

**What the summary is (and isn't):**
- IS: a reviewer's preview of what the official release notes will likely call out, written so the release manager can sanity-check the upcoming release post at a glance.
- IS NOT: the actual release notes. Do not write copy the marketing team would ship.
- IS NOT: a restatement of the headline counts. The counts block right above it already carries the quantitative summary; the highlights carry the qualitative one.

**How to pick items.** Select 4 to 8 bullets from the material already analyzed (New API, Breaking Changes, Changes to Existing API, Behavioral & Support, Removed). Use LLM judgment. An item belongs in the highlights if at least one of these is true:
- It changes a user's mental model of Warp (a new scalar type, a new public protocol, a platform dropped).
- It is a breaking change that needs a migration note in the release post.
- It is a headline-worthy experimental capability whose stability bar readers
  need to understand. The marker alone does not force inclusion.
- It unlocks a workflow that was previously impossible or awkward.
- Multiple smaller entries form a coherent theme worth a single combined bullet.

An item does NOT belong in the highlights if any of these is true (drop even if the pending entry is present):
- It is a pure bug fix whose symptom description fits in one line and has no surprising semantics (goes under Fixed, not highlights).
- It is a build-system, CI, or infrastructure change with no runtime user effect.
- It is an internal refactor already scoped away from user-visible surface.
- It is a capability extension to an existing parameter that a typical user would not notice (e.g. a defaults tidy-up).

Aim for 4-8 bullets total. Fewer than 4 almost always means you missed a theme; more than 8 means you listed changes instead of highlights.

**How to write each bullet.** Each bullet leads with a bold 2-6 word headline,
then a colon, then one sentence of rationale that explains what it is and why
it matters. Prepend `⚠️ Breaking:` only for alarming breaking changes, use
`Experimental:` for headline-worthy experimental work, and append a bake hint
(`🟠 N days bake.`) when the headline item's minimum bake is under 7 days.
Example:

> - **⚠️ Breaking: <affected behavior>** ([GH-NNN](...)): <who is affected and the concrete migration>.

**Lead with the unlock, not the mechanism.** State the new user capability and
why it matters before listing supporting API names or implementation details.
For a new artifact, format, protocol, or cross-language boundary, name it and
state the workflow it enables.

**GH refs MUST be hyperlinks, always.** Every `GH-NNNN` in a highlight bullet is a markdown link to `https://github.com/NVIDIA/warp/issues/NNNN`. This applies even when a single bullet combines multiple GH refs. Do NOT use shortcuts like `(multiple GHs)`, `(GH-1287, GH-1298, ...)` in plain text, or `(see CHANGELOG)`. If the bullet covers six issues, render all six as individual links, either inline (`([GH-1287](...), [GH-1298](...), [GH-1335](...))`) or in a trailing parenthesis at the end of the headline. There is no upper limit on link count; a reader can scan links but cannot resolve plain numbers.

**Experimental softening.** If Phase 4h tagged an entry as experimental and it
is otherwise highlight-worthy, use `Experimental:` rather than a warning or
breaking prefix. Lead with what changed and its stability bar, not migration
urgency.

Open the summary with a 2-3 sentence intro paragraph that names the shape of the release in plain language. This sets the tone for everything below it. Do not stuff the intro with numbers or repeat the bake distribution.

**Output style rules apply here too.** No em dashes. No skill-internal terminology ("Phase 4f"). No "end of summary" markers. The summary reads as release-note input, not as an audit artifact.

### 6b — Fill template

Read `references/report-template.md`. Fill in every `{{PLACEHOLDER}}` marker, including the `{{HEADLINE_SUMMARY}}` produced in 6a.

Read `references/render-rules.md` and apply every rule there: URL shapes, signature + docstring code-block forms, table column specs, audit-appendix conditional, and the output-style hard constraints (no em dashes, no skill-internal terminology, no terminal markers, every GH ref hyperlinked, no Phase names).

### 6c — Write output to chosen destination

Read the filing instructions in `references/destination-rules.md` and act on the
destination token confirmed in Phase 1. Return the report location, headline
counts, and revision-history note when applicable.

## Regexes and parsing rules (inline reference)

- GH ref: `\bGH-(\d+)` — word boundary prevents matching inside other identifiers.
- Breaking flag: literal substring `**Breaking:**` (with the colon).
- Fragment path:
  `^changelog/(?:\+[A-Za-z0-9][A-Za-z0-9-]*|\d+)\.(added|removed|deprecated|changed|fixed|documentation)(?:\.\d+)?\.md$`.
- Rendered target header: `## [<target-X.Y.Z>]` with optional trailing date.
- CHANGELOG subsection headers: `### Added`, `### Removed`, `### Deprecated`, `### Changed`, `### Fixed`, `### Documentation`.
- Symbol extraction from entry text: backtick-quoted `wp.X`, `wp.X.Y`, `@wp.X`, or bare `ClassName` (capitalized identifier). The FIRST backtick-quoted symbol in the bullet is usually the primary subject.

## Failure modes

- **Rendered entry has no backing commit after fragment-history lookup:** surface it in **Changelog Fragments Without Matching Commits** with source paths and reason "no associated commit found, verify".
- **`Added` entry names a symbol not resolvable at HEAD**: render with a ⚠️ note; do NOT emit synthetic `wp.*` stub names.
- **`upstream/` remote missing**: substitute `origin/`. Note the substitution in the report header.
- **Release branch exists but contains no new commits past main**: treat as head==main effectively; skip cherry-pick detection.
- **No pending fragments:** render an empty Towncrier draft and warn "No pending changelog fragments found at `<head-ref>`." Do not substitute the newest historical release section.
- **Towncrier draft fails:** surface the command output and stop; the audit cannot use a hand-built approximation of the release section.
- **`gh` installed but not authenticated**: treat as `gh` unavailable; skip gist matching and gist prompt; add one-line chat note.
