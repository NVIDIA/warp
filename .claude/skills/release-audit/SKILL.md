---
name: release-audit
description: "Generate a Warp release audit report (pre-release or RC, auto-detected from version string and head ref). Manually invoke as a slash command."
disable-model-invocation: true
argument-hint: "[target-version]"
allowed-tools: Bash(git log *) Bash(git show *) Bash(git tag *) Bash(git rev-parse *) Bash(git cherry *) Bash(git diff *) Bash(git diff-tree *) Bash(git worktree *) Bash(git stash *) Bash(git checkout *) Bash(python3 *list_commits.py*) Bash(python3 /tmp/*) Bash(rm /tmp/warp-*) Bash(uv run build_lib.py *) Bash(uv run python3 *) Bash(gh --version) Bash(gh auth status) Bash(gh gist create *) Bash(gh gist list *) Bash(gh gist edit *) Bash(gh issue view *) Read Write Grep Glob
---

# Release Audit

Generates a markdown audit of the upcoming Warp release for keep/defer decisions. Runs in two modes (auto-detected): a **pre-release** spot-check while work is still landing on main, or a **release-candidate** readiness review after the release branch is cut.

**Output:** a single markdown report, filed according to the destination chosen in Phase 1:
- **Secret gist** (default when `gh` is available and authenticated): stable filename `warp-<version-string>-<prerelease|rc>-report.md`, stable description `Warp <version-string> <Pre-Release|Release Candidate> Report`. Later runs against the same version revise the same gist in place; prior versions are preserved in the gist's git history.
- **Local markdown file** (fallback when `gh` unavailable, or opt-in when `gh` available): dated path at `$(git rev-parse --show-toplevel)/warp-<version-string>-<prerelease|rc>-report-<YYYY-MM-DD>.md`. Not auto-committed; user moves, shares, or deletes as desired.

**Inputs inferred from repo state:**
- Target version from `warp/config.py` / `warp/__init__.py` / `VERSION.md`.
- Base = latest tag matching previous minor's line (`vX.Y-prev.*`, latest patch).
- Head = `upstream/release-<target>` if it exists, else `upstream/main`.

**Reference documents to load on demand** (via `Read`):
- `references/report-template.md` — fill this in during Phase 6b (the `{{HEADLINE_SUMMARY}}` placeholder is drafted in Phase 6a).
- `references/render-rules.md` — URL shapes, signature/docstring code-block forms, table column specs, audit-appendix conditional, and output-style hard constraints. Loaded in Phase 6b.
- `references/classification-rules.md` — path/symbol rules used in Phases 3-5.
- `references/language-review-examples.md` — Phase 5a calibration for the CHANGELOG Review Notes appendix.

## Phase 1 — Align on scope

1. Read the version string. Try `warp/config.py` (`version = "..."`), then `warp/__init__.py` (`__version__ = ...`), then `VERSION.md`. Parse to extract the target minor (e.g., `1.13.0dev0` → target `1.13`) and determine the report mode:
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

5. **Probe `gh` availability and look up any existing matching gist.**

   First run:
   ```
   gh --version && gh auth status
   ```
   If either fails → `gh` unavailable; destination will be a local markdown file only; skip the rest of this step.

   Both succeeded → compute the stable gist title for this report:
   ```
   Warp <version-string> <Pre-Release|Release Candidate> Report
   ```
   (e.g., `Warp 1.13.0rc1 Release Candidate Report`, `Warp 1.13.0dev0 Pre-Release Report`). No date. The gist filename and description are stable so later runs can find the same gist and revise it; gist git history preserves prior versions automatically.

   Run `gh gist list --limit 1000` and filter rows whose description exactly matches that stable title. Capture the matching gist IDs; display URLs are `https://gist.github.com/<id>`. Record the match count (0, 1, or N ≥ 2) for step 6.

6. Present proposal in chat and **wait for explicit user confirmation** of refs AND output destination. Mandatory pause.

   Lead with the mode-specific intro line:
   - Pre-release: `Generating **pre-release report** for Warp **<version>**. Base **<base-ref>** → Head **<head-ref>**. **<N>** commits in range.`
   - RC: `Generating **release-candidate report** for Warp **<version>**. Base **<base-ref>** → Head **<head-ref>** (release branch cut). **<N>** commits in range.`

   Append the output block for the current `gh` / match state:

   **`gh` unavailable:**
   > Output: markdown file at repo root (`gh` not available). Confirm, or specify different refs?

   **`gh` available, 0 matches:**
   > Output: (a) new secret gist [default], (b) local markdown file at repo root. Confirm refs + pick.

   **`gh` available, 1 match:**
   > Output: (a) revise existing gist `<url>` [default], (b) new secret gist, (c) local markdown file at repo root. Confirm refs + pick.

   **`gh` available, N matches (N ≥ 2):**
   > Multiple existing gists share the stable title:
   > 1. `<url-1>` — updated `<time-1>`
   > 2. `<url-2>` — updated `<time-2>`
   > ...
   >
   > Output: (a) revise gist by number, (b) new secret gist, (c) local markdown file at repo root. Confirm refs + pick.

   Do not run any further phase until the user confirms refs and (when `gh` is available) chooses destination. Translate the user's reply into exactly one of the destination tokens `local`, `new-gist`, `revise-gist:<id>` and record it for Phase 6c. Letters `(a)`, `(b)`, `(c)` are positional within the current branch's prompt, not global: resolve them against the option list you just showed the user. For the N-match branch, the user picks a gist by the number you listed (e.g., "revise 2" → `revise-gist:<id-of-listed-row-2>`), or says new / local.

## Phase 2 — Gather ground truth

1. Run the commit-list tool as a single command (so it stays inside the `Bash(python3 *list_commits.py*)` allow rule rather than splitting into a separate `cd`):
   ```bash
   python3 "$(git rev-parse --show-toplevel)/.claude/skills/release-audit/scripts/list_commits.py" \
     --base <base-ref> \
     --head <head-ref> \
     --report-date "$(date +%F)" \
     --main-ref <resolved-main-ref>
   ```
   The `$(git rev-parse --show-toplevel)` segment expands to an absolute path that ends in `/list_commits.py`, which the wildcard rule matches. Use `$(date +%F)` literally so the shell supplies today's date; do NOT substitute a date you generated yourself, since training-time bias can produce stale `days_since_merge` / `days_in_main` values. Capture stdout as the `commit_list_json`.

2. Read `CHANGELOG.md` at HEAD using the Read tool. Locate the `## [Unreleased]` header. Collect all content from that header up to (but not including) the next `## [X.Y.Z]` header.

3. Parse subsections. Each starts with `### Added`, `### Removed`, `### Deprecated`, `### Changed`, `### Fixed`, or `### Documentation`. For every bullet under each subsection, extract:
   - **Raw text (FULL — never truncate)**: the full bullet content (may span multiple lines).
   - **Section**: one of the six names above.
   - **GH refs**: regex `GH-(\d+)` over the bullet text. Dedup.
   - **Breaking flag**: presence of the literal string `**Breaking:**` in the bullet.

## Phase 3 — Cross-reference

1. **Build the commit ↔ CHANGELOG join** on GH-ref overlap:
   - For each CHANGELOG entry with at least one GH ref, find commits (from `commit_list_json`) whose `gh_refs` intersect.
   - For each CHANGELOG entry with ZERO matching commits on GH ref, attempt a secondary lookup — find the commit(s) on HEAD that introduced or modified this exact entry text in `CHANGELOG.md`:
     ```
     git log --reverse -S'<distinctive substring from the entry>' --format='%H|%s|%cs' -- CHANGELOG.md
     ```
     The first commit whose subject isn't a CHANGELOG-only edit (e.g., not "Clean up changelog", not a version bump) usually IS the code change associated with the entry. Record that as the backing commit.
   - After these two passes, any CHANGELOG entry still with no matching commit is a genuine orphan (deferred or dropped from the release). Surface in the **CHANGELOG Entries Without Matching Commits** appendix. Keep full entry text; do not truncate in the report.

2. **Do NOT build an audit trace of unmatched commits.** The old "commits without CHANGELOG entries" appendix adds noise without value. Commits that don't map to a CHANGELOG entry are not surfaced in the report.

## Phase 4 — Analyze API surface

### 4a — Determine what is genuinely NEW API

For each CHANGELOG `Added` entry, extract the named symbol(s) (text in backticks matching `wp.*`, `@wp.*`, or patterns like `ClassName`).

For EACH named symbol, check if it existed at base:

- For top-level `wp.X`: look at `git show <base>:warp/__init__.py` — does it re-export `X`?
- For `wp.submodule.X` (e.g., `wp.config.track_memory`): read the submodule source at base.
- For `@wp.kernel`, `@wp.struct`, `@wp.func` decorator-style: check `warp/__init__.py` at base.
- For kernel-scope builtins (found in `warp/_src/builtins.py`): check if the builtin was registered at base via `git show <base>:warp/_src/builtins.py | grep '"<name>"'`.

Classification:
- **Genuinely new** (symbol was not present at base) → **New API** section.
- **Existed at base** (entry is adding a new parameter, option, or capability to something that already existed) → **Changes to Existing API** section with a "capability extension" or "new parameter" kind. Cite the backing commit's signature diff.
- **Does not name a single symbol** (e.g., "Add external texture interoperability for CUDA and OpenGL") and describes a cross-cutting capability → **Behavioral & Support Changes** section unless one of the mentioned symbols is genuinely new (in which case split: put the new symbol in New API, the capability description in Behavioral).

**If an entry mentions multiple symbols where some are new and some pre-existed** (e.g., "Add `wp.ScopedMemoryTracker` and `wp.config.track_memory`"), split: the genuinely new symbols each get a New API entry; the extensions to existing symbols each get a Changes entry.

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
   tile_fft(inout: tile(dtype=vector(length=2, dtype=Float), shape=tuple[int, ...])) -> None
   ```
   NOT:
   ```python
   add_builtin("tile_fft", input_types={"inout": tile(...)}, ...)
   ```
4. The docstring comes from the `doc=` parameter of `add_builtin()`. Render verbatim as a blockquote.

### 4c — Symbol resolution fallbacks

- **Symbol in backticks doesn't resolve at HEAD**: render the entry with a ⚠️ note: "Couldn't resolve `<symbol>` in source — verify entry names a real public symbol." Do not fabricate a header like `wp.*(no symbol)*`. Use the entry's natural subject as the section title.
- **Entry describes a topic, not a symbol** (e.g., "Add external texture interoperability for CUDA and OpenGL"): use a short descriptive title summarized from the entry (e.g., "External texture interop"), not a synthetic `wp.*` name.
- **Header naming rule**: section headers in the report should be either real fully-qualified symbols (`wp.tile_scatter_add`) OR short descriptive titles extracted from the entry. Never `wp.*`, `wp.(something)`, or similar stub patterns.

### 4d — Changed / Removed / Deprecated signature diffs

For each CHANGELOG entry in Changed / Removed / Deprecated (plus any "capability extension" entries routed here from 4a):

- Compute signatures at base and HEAD (same resolution as 4b).
- For signature-shape changes: render a fenced `diff` block showing `-` and `+` lines.
- For semantic-only changes (no signature shift) where the entry has `**Breaking:**`: skip the diff block; include the backing commit's URL and the full CHANGELOG text.
- For Removed entries: show the old signature on a `-` line; omit `+`.

**Deprecation-window lookup for Removed entries.** For every Removed entry (and every Changed entry whose prose describes a removal), search CHANGELOG.md for the matching prior Deprecated entry:

1. Extract distinctive tokens from the Removed entry: the named symbol(s) in backticks and, if the entry carries a GH ref, that ref number.
2. Scan the released-version sections of CHANGELOG.md (everything below `## [Unreleased]`) top-down for a `### Deprecated` bullet that names the same symbol(s) OR the same GH ref. The FIRST such entry (highest version, since CHANGELOG is reverse-chronological) is the deprecation introduction.
3. Record: (a) the release version heading that contains the Deprecated entry (e.g., `1.11.0`), (b) the full Deprecated entry text.
4. In the rendered Removed entry, include a one-line deprecation window: `Deprecated in X.Y.Z; removed here.` Do NOT fabricate a version if no prior entry is found.

**Missing-deprecation flag.** If a Removed entry has no prior Deprecated entry in any released CHANGELOG section, surface this prominently in the Breaking Changes section as `🚨 Policy: removed without prior deprecation`, not a quiet "verify please" line. The release manager needs this to block the release or add migration tooling. If the current CHANGELOG's own `### Deprecated` section also includes the same symbol, note that this is a simultaneous deprecate-and-remove, which is itself a policy violation (the deprecation window is meant to ship in an earlier release than the removal).

The deprecation window belongs in BOTH the Breaking Changes entry for the removal AND the Changes-to-Existing-API row (in the Description cell or as an appended sentence in the detail block). A reader should never have to ask "was this deprecated first, and for how long?"

### 4e — Signature-AST diff for unlabeled breaking changes

Independently of CHANGELOG content, compute the public API surface at base vs. HEAD:
- Base: `git show <base-ref>:warp/__init__.py` → parse with `ast`. Resolve each re-export's real signature at base. Also apply to `warp/_src/builtins.py` for kernel builtins. If `git show` fails (file absent at the base ref, e.g. older releases predating the `warp/_src/` layout) or `ast.parse` raises, skip the signature-diff check for that file and emit: "Cannot perform signature-diff check: base is too old or lacks public API exports."
- HEAD: same, against the resolved head ref's committed state (`git show <head-ref>:warp/__init__.py` and `git show <head-ref>:warp/_src/builtins.py`), consistent with Phase 2's "at HEAD" reads. Not the uncommitted working tree. Apply the same skip-and-emit fallback if either file is missing or unparsable at the head ref.
- For each symbol whose signature shape changed AND whose matching CHANGELOG entry (if any) doesn't carry `**Breaking:**` → add to Breaking Changes section as "unlabeled signature change — please verify and consider flagging."

**Exception: Removed symbols are breaking by definition.** A symbol that appears in CHANGELOG's `Removed` section does NOT need a `**Breaking:**` marker to be valid. Do NOT flag Removed entries as "unlabeled breaking" — the section name itself communicates the breakage. Removed entries surface in the Breaking Changes callout and in the Changes-to-Existing-API section (as "removed" kind), but the report must not whinge about missing `**Breaking:**` labels on them.

### 4f — Semantic-breaking verification (Claude does the work)

For each commit in `commit_list_json` that touches `warp/_src/codegen.py` OR any path under `warp/native/**`:

1. Skip if the commit is already mapped to a CHANGELOG entry carrying `**Breaking:**` (it's already in Breaking Changes).
2. Read the commit's diff: `git show --stat <sha>` then `git show <sha>` for small diffs, or read specific hunks for large ones.
3. **Triage** into one of three buckets:
   - **Clearly not breaking** → drop. Examples: renaming internal symbols, comment/format changes, pure internal refactors with no emitted-code difference, performance optimizations that preserve semantics, test-only changes, build-system changes, bug fixes where the pre-fix behavior was itself a bug.
   - **Clearly breaking** with an obvious user-observable shift visible from the diff alone → include directly (proceed to step 5).
   - **Ambiguous** — the diff suggests the change could affect emitted code or runtime behavior, but Claude cannot tell from reading alone whether a user would observe a difference → **verify by running code** (step 4).

4. **Verification by running code.** For ambiguous candidates, Claude must actually run Warp at both base and HEAD and compare observable output:
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

**Never produce an unexamined list of candidates.** Every Breaking Changes entry either has explicit CHANGELOG backing, a signature-diff detected shape change, or Claude-verified behavioral evidence.

### 4g — Experimental-marker cross-reference

Some symbols are shipped with an explicit `**Experimental**` marker in the CHANGELOG entry that introduced them. Changes to those symbols do NOT carry the same stability contract as changes to stable APIs: the whole point of the marker is to reserve the right to break them. The report must reflect that so the release manager does not over-weight the concern.

For each entry in Breaking Changes, Changes to Existing API, and Removed (as collected through 4a–4f), determine whether the affected symbol or feature area is currently experimental:

1. Collect candidate symbols / feature-area phrases from the entry: backticked identifiers, class names, and (for topic-style entries like "External texture interop") the most distinctive descriptive noun phrase.
2. Search CHANGELOG.md in released-version sections (everything below `## [Unreleased]`) for bullets that both carry `**Experimental**` (bold, with or without trailing colon) AND name one of the candidates from step 1. Also match via GH ref if the current entry and a prior experimental entry share a GH number.
3. If a match exists AND there is no subsequent CHANGELOG bullet in a later released version explicitly promoting the symbol to stable (e.g., "Promote `wp.Foo` out of experimental", "Stabilize `wp.Bar`"), the symbol is still experimental. Record: (a) the release version that introduced the symbol as experimental, (b) the full text of that introduction bullet.
4. Also check the module source at HEAD for an in-code `.. experimental` / `Experimental:` / `experimental_api` / `@experimental` annotation on the symbol's declaration. If present, treat as experimental regardless of CHANGELOG signal.

Tag every matched entry internally as `experimental=True`. Do not alter the CHANGELOG text itself.

**How the tag changes rendering:**
- Breaking Changes heading for the entry: prefix with `⚠️ Experimental:` rather than the implicit "Breaking" framing. Include a short sentence reminding readers this symbol is opt-in and documented as experimental, with the release where the experimental marker was introduced.
- Changes-to-Existing-API table: the Breaking column shows `Experimental` rather than `Yes`, even when the change is technically source-breaking.
- Release Highlights bullet (Phase 6a): use `⚠️ Experimental:` as the risk prefix, not `⚠️ Breaking:`. The rationale sentence should mention the stability bar, not lead with migration urgency.

**Never drop the entry.** Experimental softening is about alarm level, not suppression. The reader still needs to see what changed. A removed or signature-changed experimental symbol still appears in Changes to Existing API; the softening only adjusts tone and the risk-prefix glyph.

## Phase 5 — Review CHANGELOG language and bake

### 5a — Language review (renders as the **CHANGELOG Review Notes** appendix)

Read `references/language-review-examples.md`. For EACH CHANGELOG entry, apply LLM judgment:

- **🔗 Wrong ref (tier-1)**: for every GH ref in the entry, fetch the mapped commits' subjects and paths. If the entry topic doesn't match the commits' actual scope, flag.
- **🔗 Wrong ref (tier-2)**: if `gh --version` and `gh auth status` both succeed, run `gh issue view <num> --json title,body` per ref and compare issue title to entry topic. Skip silently if `gh` unavailable.
- **🗣️ Internal language**: internal module paths (`warp._src.*`), C++/CUDA type names (`launch_bounds_t`, `tile_register_t`), private identifiers.
- **📝 Too terse**: under ~10 words with no context.

Record flagged entries. Keep the FULL entry text in the audit table — do not truncate.

### 5b — Bake aggregation

**Pre-release mode (`resolved.head.sha == resolved.main_ref.sha`).** Every commit's main equivalent is itself, so `days_in_main == days_since_merge` and the bake distribution would just restate the age histogram. Render an "Age distribution" table from `days_since_merge` (same 🟢/🟡/🟠 thresholds), label the column "Days since merge", and skip both the "Bake distribution" table and the anomaly banner. There is no meaningful "didn't bake on main" condition when head IS main.

**RC mode (`resolved.head.sha != resolved.main_ref.sha`).** Partition commits by `main_match_state`:

- `state == "unique"`: bucket by `days_in_main` into **🟢 (>14 days)**, **🟡 (7–14 days)**, **🟠 (<7 days)**.
- `state == "missing"`: subject not present on main_ref. These shipped to the release branch without a main-side bake. Before reporting them, **filter out routine release-branch artifacts that the release manager already expects**: commits whose subject matches `Set version to <X>`, `Bump version to <X>`, or `Release <X>` AND whose file changes are confined to version-bump paths (`VERSION.md`, `warp/config.py`, `warp/native/version.h`, `docs/conf.py`, `pyproject.toml`, `asv/tag_commits.txt`, `tools/pre-commit-hooks/check_version_consistency.py`). These are expected to live only on the release branch; surfacing them as a banner is noise, not signal. Count remaining missing commits (post-filter) separately and, if non-zero, fire the ⚠️ banner in the report header listing them. If the filter removed any commits, optionally note them as a quiet footnote under the bake table (e.g., `K version-bump commits not present on main — expected for a release branch`); do not call them out at the same alarm level as a real bake gap.
- `state == "ambiguous"`: subject appears more than once on main_ref (typical for `Revert "..."`, `Bump version`, or replayed cherry-picks). The commit IS on main; the script just could not pick a single canonical occurrence. Render as a separate row in the bake table labeled "⚪ ambiguous main match: K commits" and DO NOT fire the banner. These are not a bake-gap signal.

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
- It is user-facing and carries an explicit `**Experimental**` marker (readers need to know the stability bar).
- It unlocks a workflow that was previously impossible or awkward (e.g. external texture interop, pluggable allocators).
- Multiple smaller entries form a coherent theme worth a single combined bullet (e.g. "new tile primitives: tile_dot, tile_axpy, tile_scatter_add").

An item does NOT belong in the highlights if any of these is true (drop even if the CHANGELOG entry is present):
- It is a pure bug fix whose symptom description fits in one line and has no surprising semantics (goes under Fixed, not highlights).
- It is a build-system, CI, or infrastructure change with no runtime user effect.
- It is an internal refactor already scoped away from user-visible surface.
- It is a capability extension to an existing parameter that a typical user would not notice (e.g. a defaults tidy-up).

Aim for 4-8 bullets total. Fewer than 4 almost always means you missed a theme; more than 8 means you listed changes instead of highlights.

**How to write each bullet.** Each bullet leads with a bold 2-6 word headline, then a colon, then one sentence of rationale that explains what it is and why it matters. Attach risk flags inline when they apply: prepend `⚠️ Breaking:` or `⚠️ Experimental:` to the headline where relevant, and append a bake hint (`🟠 N days bake.`) when the headline item's minimum bake is under 7 days. Example:

> - **⚠️ Breaking: `wp.tid()` flattens excess dimensions** ([GH-1270](...)): kernels that unpack fewer values than launch dimensions now receive a flat index; release post needs a migration snippet.

**Lead with the unlock, not the mechanism.** The rationale sentence should answer "what is newly possible, and why would a user care?" — not "what API names were added." API names are a detail; novel capability is the story. If a feature introduces a new artifact or format (a file format, wire protocol, cross-language boundary, new kind of object), NAME that artifact and state what it unlocks. Do not reduce the feature to the function that produces it.

Negative example (what NOT to do) — based on a real miss from an earlier report on the graph-capture feature:

> - **Graph capture serialization + CPU replay** ([GH-1349](...)): preliminary `capture_save` / `capture_load` to `.wrp` files with new `wp.handle` scalar for mesh-handle remapping; CPU graph capture gains real replay.

This bullet mentions the `.wrp` extension in passing but misses the actual story: `.wrp` is a new portable serialized-graph format, and the same file can be loaded from **standalone C++** (not just Python), which is the unprecedented capability. A reader skimming for headline material sees "save and load" and moves on.

Better shape:

> - **Graph capture serialization + CPU replay** ([GH-1349](...)): introduces `.wrp`, a portable serialized-graph format that can be captured in Python (`capture_save`) and replayed from either Python or standalone C++ (`capture_load`), plus a new `wp.handle` scalar for mesh-handle remapping across load. CPU graph capture gains a real replay path in the same pass.

Same work, but a reader learns that there's a new interchange format and a Python-to-C++ bridge. When picking the leading noun for your rationale, ask: if this feature were being pitched in a 30-second release-notes overview, what is the single thing a user would want to hear? Write THAT, and let the API names ride along as supporting detail.

**GH refs MUST be hyperlinks, always.** Every `GH-NNNN` in a highlight bullet is a markdown link to `https://github.com/NVIDIA/warp/issues/NNNN`. This applies even when a single bullet combines multiple GH refs. Do NOT use shortcuts like `(multiple GHs)`, `(GH-1287, GH-1298, ...)` in plain text, or `(see CHANGELOG)`. If the bullet covers six issues, render all six as individual links, either inline (`([GH-1287](...), [GH-1298](...), [GH-1335](...))`) or in a trailing parenthesis at the end of the headline. There is no upper limit on link count; a reader can scan links but cannot resolve plain numbers.

**Experimental softening.** If Phase 4g tagged an entry as experimental, the highlight bullet uses `⚠️ Experimental:` as its risk prefix rather than `⚠️ Breaking:`, even if the change is technically source-breaking. The rationale sentence should lead with what changed and its stability bar, not with migration urgency; readers already know experimental APIs can shift.

Open the summary with a 2-3 sentence intro paragraph that names the shape of the release in plain language. This sets the tone for everything below it. Do not stuff the intro with numbers or repeat the bake distribution.

**Output style rules apply here too.** No em dashes. No skill-internal terminology ("Phase 4f"). No "end of summary" markers. The summary reads as release-note input, not as an audit artifact.

### 6b — Fill template

Read `references/report-template.md`. Fill in every `{{PLACEHOLDER}}` marker, including the `{{HEADLINE_SUMMARY}}` produced in 6a.

Read `references/render-rules.md` and apply every rule there: URL shapes, signature + docstring code-block forms, table column specs, audit-appendix conditional, and the output-style hard constraints (no em dashes, no skill-internal terminology, no terminal markers, every GH ref hyperlinked, no Phase names).

### 6c — Write output to chosen destination

The destination was decided in Phase 1: one of `local`, `new-gist`, or `revise-gist:<id>`. Act on that choice.

**Filename conventions:**
- Local file (at repo root): `warp-<version-string>-<prerelease|rc>-report-<today>.md` — dated, user-facing.
- Gist file (inside the gist): `warp-<version-string>-<prerelease|rc>-report.md` — no date. Stable name so later runs can revise the same gist in place.

**Stable gist description** (used when creating a new gist; also the matching key for Phase 1):
```
Warp <version-string> <Pre-Release|Release Candidate> Report
```

**If destination is `local`:**
1. Write to `$(git rev-parse --show-toplevel)/<local-filename>` using the Write tool.
2. Print a one-line chat summary:
   - Local path.
   - Headline counts (N new APIs, K breaking, M changed, L behavioral, F fixes).

**If destination is `new-gist`:**
1. Write to `/tmp/<gist-filename>` (stable name, no date) using the Write tool.
2. Create the gist:
   ```
   gh gist create --desc "<stable-desc>" /tmp/<gist-filename>
   ```
   Capture the gist URL from stdout.
3. Delete `/tmp/<gist-filename>` so no local artifact remains.
4. Print a one-line chat summary:
   - Gist URL.
   - Headline counts.

**If destination is `revise-gist:<id>`:**
1. Write to `/tmp/<gist-filename>` (same stable name the existing gist already uses) using the Write tool.
2. Revise the gist:
   ```
   gh gist edit <id> --filename <gist-filename> /tmp/<gist-filename>
   ```
   The `--filename` flag selects which file *inside the gist* to replace; the trailing local path supplies the new content. Without `--filename`, `gh` treats the local path as a gist-side filename selector, fails to find it, and falls through to its interactive editor. Do NOT pass `--desc`: keeping the description stable is what lets the next run match this gist again. Prior versions of the file are preserved automatically in the gist's git history.
3. Delete `/tmp/<gist-filename>`.
4. Print a one-line chat summary:
   - Gist URL (`https://gist.github.com/<id>`).
   - Note: "revised in place; prior versions kept in gist git history".
   - Headline counts.

Never pass `--public`. Never file a destination the user did not choose.

## Regexes and parsing rules (inline reference)

- GH ref: `\bGH-(\d+)` — word boundary prevents matching inside other identifiers.
- Breaking flag: literal substring `**Breaking:**` (with the colon).
- CHANGELOG Unreleased section header: `## [Unreleased]` — may have trailing date text; match prefix only.
- CHANGELOG subsection headers: `### Added`, `### Removed`, `### Deprecated`, `### Changed`, `### Fixed`, `### Documentation`.
- Symbol extraction from entry text: backtick-quoted `wp.X`, `wp.X.Y`, `@wp.X`, or bare `ClassName` (capitalized identifier). The FIRST backtick-quoted symbol in the bullet is usually the primary subject.

## Failure modes

- **CHANGELOG entry with zero GH refs AND no backing commit found via `git log -S`**: surface in the **CHANGELOG Entries Without Matching Commits** appendix with reason "no associated commit found, verify".
- **`Added` entry names a symbol not resolvable at HEAD**: render with a ⚠️ note; do NOT emit synthetic `wp.*` stub names.
- **`upstream/` remote missing**: substitute `origin/`. Note the substitution in the report header.
- **Release branch exists but contains no new commits past main**: treat as head==main effectively; skip cherry-pick detection.
- **CHANGELOG `[Unreleased]` missing or empty**: header warns: "No `[Unreleased]` entries found in CHANGELOG.md."
- **`gh` installed but not authenticated**: treat as `gh` unavailable; skip gist matching and gist prompt; add one-line chat note.
