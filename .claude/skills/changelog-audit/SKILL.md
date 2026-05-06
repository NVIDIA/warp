---
name: changelog-audit
description: "Audit Warp CHANGELOG.md before a release: recover lost entries, sort by user impact, refine entry language, line-wrap, and (release-branch mode) bump compare refs. Manually invoke as a slash command."
disable-model-invocation: true
argument-hint: "[ref]"
allowed-tools: Task Bash(git log *) Bash(git show *) Bash(git tag *) Bash(git rev-parse *) Bash(git diff *) Bash(git diff-tree *) Bash(git worktree *) Bash(git branch *) Bash(git switch *) Bash(git status) Bash(git status --porcelain) Bash(git config user.email) Bash(git config user.name) Bash(date *) Bash(diff *) Bash(gh --version) Bash(gh auth status) Bash(gh issue view *) Bash(gh issue list *) Bash(gh search *) Bash(uv run build_lib.py *) Bash(uv run /tmp/*) Bash(uv run python3 /tmp/*) Bash(rm /tmp/changelog-*) Bash(rm /tmp/warp-changelog-*) Bash(nvidia-smi) Bash(which nvcc) Read Edit Write Grep Glob
---

# Changelog Audit

Audits `CHANGELOG.md` before a Warp release. Operates in two modes auto-detected from the resolved ref name:

- **Release-branch mode** (ref name matches `^release-`): the upcoming release section is being finalized. Run all cleanup passes, promote `[Unreleased]` → `[X.Y.Z]`, and bump the compare-link reference block at the bottom.
- **Main mode** (anything else, including bare `main`): tidy the live `[Unreleased]` section in place. If main's CHANGELOG is out of sync with one or more released tags, also back-port the tag's section, dedupe redundant `[Unreleased]` entries, and rotate the compare-link refs. No rename of `[Unreleased]`.

Cleanup passes (some mode-conditional, run in this order):

- **Phase 1.5 (main only, conditional) — Post-release sync.** When a stable tag exists whose `[X.Y.Z]` section is missing from main's CHANGELOG, back-port the section from the tag, dedupe `[Unreleased]` entries that already shipped, and rotate compare-link refs. Runs unconditionally when triggered (not skippable).
- **Phase 2 — Lost-entry recovery.** Find entries that landed inside an already-released section due to merge=union and propose moves up.
- **Phase 3 — Verify, consolidate, link metadata, and check section placement.** Confirm non-trivial entries are accurate against the actual code (run code if needed); merge "Add X / Fix X / Change X" sequences for never-shipped features into a single accurate entry; retro-search GitHub for missing GH refs; and re-classify entries that landed in the wrong subsection (e.g., a behavior change wrongly under Fixed) or are missing a `**Breaking:**` marker.
- **Phase 4 — Impact sort.** Within each subsection, most user-impactful first; soft preference for keeping similar entries adjacent.
- **Phase 5 — Language pass.** Drop entries with no user-facing impact, rewrite jargon-heavy entries, fan out to user-perspective subagents on ambiguous cases, then run the editorial conventions sweep (imperative mood, hyphenation, GH-link position, symbol formatting consistency).
- **Phase 6 — Line-wrap and consolidated diff.** 120-char hard limit, prefer fewer lines, preserve semantic line breaks where they help raw-text readers.
- **Phase 7 (release-branch only) — Promote + ref bump.** Rename header and update the link-reference block.

Edits land in `CHANGELOG.md` directly. **All passes stage their changes to an in-memory buffer**; nothing is written until Phase 6 surfaces the consolidated diff and the user confirms. Phase 1.5 (post-release sync, on dedupe decisions), Pass 2 (lost-entry recovery), Pass 3 (consolidations and retro-GH ref insertions), and Pass 5 (rewrites and deletions) prompt before staging certain decisions so the user can decide on a per-candidate basis. Phase 7 (release-branch only) confirms its rename + ref-bump diff separately and writes after that confirmation.

**Inputs:**

- `[ref]` (positional argument, optional). Any git ref. If omitted, defaults to `HEAD`. The skill never assumes the current working tree's branch matches what the user intends to audit; the ref is the source of truth for content reads.

**Reference files** (loaded on demand via Read):

- `references/sorting-rubric.md` — Phase 4 impact-ordering rules with worked examples.
- `references/language-conventions.md` — Phase 5 conventions: what belongs in CHANGELOG, what doesn't, internal-jargon flag list, user-perspective subagent prompt template.

## Phase 1 — Resolve scope

1. Parse the ref argument. If empty, treat as `HEAD`.

2. Determine **mode** from the ref. Resolve the ref to a name, then strip any remote prefix:

   ```bash
   resolved=$(git rev-parse --verify --abbrev-ref --symbolic-full-name <ref> 2>/dev/null \
              || git rev-parse --verify --abbrev-ref --symbolic-full-name origin/<ref> 2>/dev/null \
              || git rev-parse --verify --abbrev-ref --symbolic-full-name upstream/<ref> 2>/dev/null)
   short=${resolved#origin/}; short=${short#upstream/}
   ```

   If `$short` matches `^release-` (e.g., `release-1.13`, `release-1.13.4`) → **release-branch mode**. Otherwise → **main mode**.

   Edge case: ref is a tag like `v1.13.0rc1`. Treat as release-branch mode if the tag's commit is reachable from a `release-*` branch (`git branch -r --contains <tag>`); else main mode. When in doubt, ask.

3. Determine **target version**:
   - Read `VERSION.md` and `warp/config.py` *at the resolved ref* via `git show <ref>:VERSION.md` and `git show <ref>:warp/config.py`. Take the first that parses to a `MAJOR.MINOR.PATCH` (strip `dev0`, `rc1`, `.dev0`, etc.). VERSION.md wins on conflict.
   - In release-branch mode the version *must* parse to a clean `X.Y.Z`. If it doesn't, surface the raw string and ask the user.
   - In main mode the version drives nothing (no rename), but record it for the report header.

4. Determine **base release tag** for the lost-entry diff. Use the parsed `(major, minor)` from step 3 to restrict the tag pattern; do not just take the highest tag merged into the ref (an `rc` of the same minor would land first).

   ```bash
   # Release-branch mode: previous minor (X.Y-1).*. Use integer math on the parsed version.
   git tag --merged <ref> --list 'v<major>.<minor-1>.*' --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1

   # Main mode (or fallback): highest clean stable tag overall.
   git tag --list 'v*' --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1
   ```

   Filter excludes anything with `rc`, `dev`, `beta`, or other suffixes. Take the first result.

   **Major-boundary fallback**: if `minor == 0` (e.g., target `(2, 0)`), there is no `v2.-1.*` line. Enumerate from the previous major instead: `git tag --merged <ref> --list 'v<major-1>.*'` and take the highest. For example, target `(2, 0)` → pattern `v1.*` → take the highest stable `v1.x.y` tag from the result. Always use integer math on the parsed `(major, minor)` tuple; never treat the dotted version string as a float.

   - In release-branch mode, the result is typically the last release of the previous minor (e.g., `release-1.13` → `v1.12.1`).
   - In main mode, the result is the last stable release overall.
   - If no tag exists, skip Phase 2 and note in the report.

5. Locate the **working directory** for edits:
   - If `git -C <cwd> rev-parse HEAD` resolves to the same commit as `git rev-parse <ref>`, edit in place.
   - Otherwise run `git worktree list` and check whether `<ref>` is checked out in a sibling worktree. If so, tell the user the path and ask whether to `cd` there or create a fresh worktree.
   - If neither, propose `git worktree add ../warp-changelog-audit-<ref-slug> <ref>` and wait for confirmation. Compute `<ref-slug>` by replacing every `/` and any non-`[A-Za-z0-9._-]` character in the ref with `-` (e.g., `origin/release-1.13` → `origin-release-1.13`).

6. **Detect post-release sync need (main mode only).** Get the list of stable tags merged into `<ref>`:

   ```bash
   git tag --merged <ref> --list 'v*' --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$'
   ```

   For each tag, check whether `^## \[X.Y.Z\]( - .*)?$` appears in main's `CHANGELOG.md` (read at `<ref>`). The set of tags whose header is **absent** is the **missing-tags set**. If non-empty, Phase 1.5 will run; surface the set in the confirmation below.

   In release-branch mode, skip this check entirely.

7. Present scope and **wait for explicit user confirmation** before any mutation:

   ```
   Auditing CHANGELOG.md at <ref> in <working-dir>.
   Mode: <release-branch|main>.
   Target version: <X.Y.Z> (source: <VERSION.md|warp/config.py|user>).
   Base for lost-entry diff: <vX.Y-prev.Z> (or: no tag, skipping Phase 2).
   Post-release sync needed: <vX.Y.Z, vX.Y.Z+1, ...>   (main mode only; omit line if missing-tags set is empty)
   Release date placeholder: <YYYY-?? | YYYY-MM-DD>   (release-branch mode only; default `YYYY-??`)
   Confirm to proceed, or override release date.
   ```

   Mandatory pause. The release-date line is omitted in main mode (no rename happens). The post-release-sync line is omitted in release-branch mode and in main mode when the missing-tags set is empty.

## Phase 1.5 — Post-release sync (main mode only, conditional)

**Skip in release-branch mode. Skip in main mode when the missing-tags set from Phase 1.6 is empty.**

When a release branch is cut and tagged (`vX.Y.Z`), the release-branch audit promotes `[Unreleased]` → `[X.Y.Z]` *on the release branch*. Main's CHANGELOG keeps the same bullets under `[Unreleased]` until someone back-ports the section. This phase performs that back-port, removes redundant `[Unreleased]` entries that already shipped, and rotates the compare-link refs.

The sync runs **unconditionally** when missing tags are detected — the user does not get to skip it. Out-of-sync CHANGELOGs on main are a bug, not a preference. The user is consulted only on (a) the feature-branch name and (b) per-entry duplicate-removal decisions.

Process the missing-tags set **oldest-first** (sort ascending). Each newly inserted section goes immediately below `[Unreleased]`, so the final file ends up with the most recent release first — same convention as before.

### 1.5.1 — Branch enforcement (must run before any mutation)

Determine the current branch in the working directory:

```bash
git -C <wd> rev-parse --abbrev-ref HEAD
```

- **`main`, `master`, or `HEAD` (detached)**: the skill MUST switch to a feature branch before mutating CHANGELOG.md. This is required by the user's "never commit to main" guidance (see auto-memory).
- **Anything else**: the user is already on a feature branch. Re-use it; do not create another.

When a feature branch must be created, propose `<user>/sync-changelog-v<newest-missing-tag>`:

- `<user>` defaults to the local-part of `git config user.email` (everything before the first `@`). Example: `ershi@nvidia.com` → `ershi`.
- `<newest-missing-tag>` is the highest version in the missing-tags set (e.g., if 1.13.0 *and* 1.13.1 are missing, use `1.13.1`). The branch name reflects how far up main is being synced.

Surface the proposed name and confirm. The user may override; sanity-check overrides against `^[A-Za-z0-9._/-]+$` and reject anything else. If the user rejects without an alternative, **abort the entire audit** — Phase 1.5 cannot proceed without a writable branch.

Before switching, check working-tree cleanliness:

```bash
git -C <wd> status --porcelain
```

If non-empty, surface the dirty state and ask whether to commit, stash by hand, or abort. **Do not auto-stash.**

Once clean and the name is confirmed:

```bash
git -C <wd> switch -c <branch>
```

If the branch already exists, surface and ask: switch to it (`git switch <branch>`), pick a different name, or abort.

### 1.5.2 — Insert missing sections (oldest-first)

For each missing tag, in ascending version order:

1. **Read the tag's CHANGELOG** and extract the `[X.Y.Z]` section. The section runs from its header line up to (but not including) the next `## [` header line OR the start of the link-reference block (line matching `^\[[^\]]+\]: `), whichever comes first.

   ```bash
   git show v<X.Y.Z>:CHANGELOG.md > /tmp/changelog-v<X.Y.Z>.md
   ```

2. **Find the insertion point in main's CHANGELOG.** Insert immediately after the `[Unreleased]` section ends (next `## [` header or link-reference block) and before existing released-version sections begin. Because we process oldest-first and always insert at the same anchor, newer synced tags end up *above* older synced tags by the time the loop finishes — preserving most-recent-first order.

3. **Insert verbatim.** The tag's `CHANGELOG.md` is the authoritative source — text from the tag was already audited by the release-branch run, possibly rewritten. Do NOT re-edit on insert; the release-branch audit's wording wins.

4. **Edge case — tag has no `[X.Y.Z]` section in its own CHANGELOG** (release was tagged without the rename). Drop that tag from the missing-tags set, surface a warning in the report, continue.

5. **Edge case — `[X.Y.Z]` section already exists in main's CHANGELOG with diverging bullets** (partial manual back-port already done). Surface a side-by-side diff of (main's existing `[X.Y.Z]` section) vs (tag's `[X.Y.Z]` section) and ask: replace with the tag's version, merge bullet-by-bullet, or skip this tag's sync. **Do not silently overwrite.**

### 1.5.3 — Dedupe `[Unreleased]` entries against newly inserted sections

After every missing tag's section is inserted, walk every bullet in `[Unreleased]` and compare against bullets in the freshly inserted sections. Three-tier match:

**Token Jaccard** (used in both tiers below) is `|tokens_A ∩ tokens_B| / |tokens_A ∪ tokens_B|`, where `tokens_X` is the **set** (deduplicated) of words from bullet `X` after: lowercasing, stripping surrounding punctuation and backticks, and splitting on whitespace and punctuation. Sets, not multisets — repeated words count once. Use this same definition consistently so the ≥ 0.5 and ≥ 0.7 thresholds reproduce.

- **Tier 1 — GH-ref intersection.** Extract every `GH-(\d+)` from the `[Unreleased]` bullet. If any of those refs also appears in a freshly-inserted bullet AND the prose is plausibly the same change (shared lead verb after `- `, shared inline-code symbol, OR token Jaccard ≥ 0.5), this is a **high-confidence duplicate**.
- **Tier 2 — Prose similarity (no shared GH ref).** If no GH refs match but token Jaccard ≥ 0.7 against any freshly-inserted bullet, this is a **medium-confidence duplicate**.
- **Tier 3 — No match.** Leave alone.

For each Tier 1 / Tier 2 hit, **prompt the user with a side-by-side**:

```
Candidate duplicate (tier <1|2>):
  [Unreleased]:  <on-main bullet text>
  [X.Y.Z]:       <tag's bullet text>
  Match signal:  <GH-NNNN> | text similarity <score>
Remove from [Unreleased]?  (y/n)
```

The tag's version is authoritative (the release-branch audit may have rewritten it). On confirm, remove the `[Unreleased]` entry — the tag's version is already present in the new section, so no information is lost. On reject, leave the `[Unreleased]` entry alone — the user is signaling these are different changes that happen to share an issue number or wording.

**Important caveat: a GH-ref match alone is NOT sufficient to confirm a duplicate.** Realistic scenarios where the issue number is shared but the entry is genuinely new:

- A bug fix in `[Unreleased]` that addresses a regression introduced by a feature shipped under the same issue number, where no follow-up issue was filed because the user-visible topic is unchanged.
- A follow-up entry for a feature shipped in `[X.Y.Z]` whose extension was tracked under the original issue (e.g., "Add wp.foo() (GH-123)" shipped, then "Allow wp.foo() to accept dtype= (GH-123)" follows).

The Tier 1 prose-similarity gate filters most of these, but the per-entry confirmation is the real safety net. The user sees both texts and decides.

### 1.5.4 — Rotate compare-link refs

For each synced tag, add a line to the link-reference block at the bottom of the file:

```
[X.Y.Z]: https://github.com/NVIDIA/warp/releases/tag/vX.Y.Z
```

Insert in version order (most recent first), interleaved with any existing link-reference lines.

Update the `[Unreleased]` link to point at the **newest** synced tag:

```
[Unreleased]: https://github.com/NVIDIA/warp/compare/v<newest-synced-tag>...HEAD
```

If the file has no `[Unreleased]:` link line, leave the file alone for this update — do not invent a link the file did not previously carry. (Same behavior as Phase 7.3 on the release branch.)

### 1.5.5 — Stage to the buffer

All Phase 1.5 mutations stage to the same in-memory buffer that Phases 2-6 use. Phase 6's consolidated diff surfaces the sync alongside the rest of the audit. The user confirms the final diff once before any write.

## Phase 2 — Lost-entry recovery

The CHANGELOG uses `merge=union` (see `.gitattributes`). When two branches both add bullets under the same section header, the union strategy concatenates both sets without conflict. If branch A added under `[Unreleased]` and branch B (older base) added under what is *now* a released section header, the merged file ends up with B's bullets stranded inside an already-released section.

1. **Read both files and diff them.** Pull each version of CHANGELOG.md to a temp file:

   ```bash
   git show <base-tag>:CHANGELOG.md > /tmp/changelog-base.md
   git show <ref>:CHANGELOG.md > /tmp/changelog-head.md
   diff -u /tmp/changelog-base.md /tmp/changelog-head.md
   ```

2. **Identify candidate lost entries** by walking the HEAD file, not the diff. For every bullet line (regex: `^- `) that appears under a released-version header (`## [X.Y.Z]`) at HEAD, check whether the same bullet text appears anywhere in the base file (any section). Three cases:
   - Bullet exists in base under the *same* released-version header → expected, not a candidate.
   - Bullet exists in base under `[Unreleased]` and the matching section at HEAD is the version that was promoted from `[Unreleased]` since base was tagged → expected, not a candidate.
   - Bullet does not appear in base at all (or appears only under a different released-version header) → **candidate lost entry**.

   Bullets under `[Unreleased]` at HEAD are *never* candidates — the staging area is allowed to grow. We only flag content that ended up frozen inside a released section without being there at base.

3. **Triage each candidate.** For every candidate bullet:
   - Extract a distinctive substring (~30-50 chars; pick a phrase unique enough that `git log -S` returns at most a few hits). Avoid backticks, `$`, `\`, and unbalanced quotes inside the substring — pick a plain-text phrase or escape the entire arg with single quotes (and if the phrase itself contains a single quote, prefer a different substring; mixing quote types inside `git log -S` is fragile).
   - Find the inserting commit:
     ```bash
     git log <base-tag>..<ref> -S'<substring>' --format='%H|%s|%cs|%an' -- CHANGELOG.md
     ```
     Usually one commit. If multiple, take the earliest (the original insertion; later commits may have only moved the line).
   - Inspect that commit's CHANGELOG diff (`git show <sha> -- CHANGELOG.md`) AND its non-CHANGELOG footprint (`git show --stat <sha>`).
   - Classify:
     - **Lost** — the commit's other changes describe a feature whose merge landed *after* the base tag was cut. The bullet was authored against `[Unreleased]` upstream but landed inside an older section due to merge=union. **Propose move to upcoming-release section** (i.e., the still-`[Unreleased]` section at HEAD; Phase 7 promotes that header later in release-branch mode).
     - **Retroactive** — the commit explicitly edits a released section (e.g., fixing a broken issue link, correcting a typo, the user's own retroactive cleanup). **Skip; not a lost entry**.
     - **Ambiguous** — surface the commit URL/subject and ask the user.

4. **For each move, prompt the user before staging.** Show: original location (section header + bullet), proposed destination subsection, inserting commit subject + short SHA. **Stage** on yes (mutate the in-memory buffer; do not write the file); skip on no. One prompt per candidate.

5. **Subsection placement.** When moving up a lost entry, place it under the same subsection name (Added / Removed / etc.) it was in below. If the target section doesn't yet have that subsection, create one in Warp's canonical order: **Added → Removed → Deprecated → Changed → Fixed → Documentation** (Security is reserved by Keep-a-Changelog convention but Warp's CHANGELOG does not currently use it; only insert a `### Security` subsection if a lost entry was originally in one). Append to the bottom of the destination subsection — Phase 4 handles ordering.

## Phase 3 — Verify, consolidate, and link metadata

This phase establishes content correctness *before* the prose pass: confirm the upcoming-release section says true things, merge sequences that pretend to be three independent entries when they describe one feature's pre-release iteration, and chase down missing GH refs. Heaviest pass in the skill — budget time for it.

### 3a — Verify accuracy on non-trivial / high-profile entries

For each entry under the upcoming-release section, decide whether it warrants verification:

- **Verify**: Added entries that introduce new public API surface; entries flagged `**Breaking:**`; entries that quantify behavior (e.g., "~4x faster", "now requires CUDA 12.4+", "no longer triggers in <X> condition"); entries marked `**Experimental**` (the experimental marker doesn't lower the bar; the entry must still be accurate); any entry the user has called out as a headline feature for the upcoming release.
- **Skip**: trivial bug fixes, doc-only tweaks that survived earlier filters, internal refactors with token user-facing prose, entries that don't make a verifiable claim.

For each entry to verify:

1. **Find the introducing commit(s).** Search outside CHANGELOG.md so you locate the *code* change, not the bullet itself:
   ```bash
   git log <base-tag>..<ref> -S'<distinctive substring>' --format='%H|%s|%cs' \
     -- '*.py' '*.cpp' '*.cu' '*.h'
   ```
   For decorator-style or builtin-style entries, also try path-based scoping (e.g., `-- warp/_src/builtins.py warp/_src/codegen.py`).

2. **Read the diff.** `git show <sha>` for each candidate commit. Cross-check the entry against the code on these axes:
   - Symbol names match the actual code (no typo `wp.tile_dot` vs `wp.tile_dotproduct`).
   - Parameter names match the function signature.
   - Behavior described matches the code (e.g., if the entry says "atomic=True default", the code's default must be `atomic=True`).
   - GH ref topic matches the entry (`gh issue view <num> --json title,body` if `gh` is authenticated; skip silently if not).

3. **Run code if needed.** For entries that quantify behavior or describe non-obvious user-observable effects (e.g., "supports N-D tiles", "FFT now operates along the last dim", "default opt level is now -O2"), write a small standalone Python script under `/tmp/` and run it to confirm. **Never use `python -c "..."` for kernel code** — Warp's codegen calls `inspect.getsourcelines()` which fails for code not in a file (per `AGENTS.md`). Always `uv run /tmp/<name>.py`.

4. **Build Warp if needed.** If `warp/bin/` is empty or stale, run `uv run build_lib.py` (~5 min) or `uv run build_lib.py --quick` (~2-4 min) before running test scripts. **Do not skip the run "because Warp isn't built"** — build first. (Quick build is only safe if the CUDA driver version ≥ Toolkit version: check `nvidia-smi` driver vs the Toolkit set via `WARP_CUDA_PATH` / `CUDA_HOME` / `which nvcc`.)

5. **Record findings.** For each verified entry, capture: verified-as-accurate, verified-with-revision (and what the revision should be), or could-not-verify (and why). Stage rewrites where the entry is misstated. Could-not-verify cases land in the final report so a human can re-check.

### 3b — Consolidate Add+Fix+Change sequences for never-shipped features

Sometimes a feature is added under `[Unreleased]`, then a follow-up commit fixes or changes its behavior — also under `[Unreleased]`, before any tag has shipped the feature. Three entries appear:

```
- Add `wp.foo()`
- Fix `wp.foo()` crash on empty input
- Change `wp.foo()` default behavior to match `wp.bar()`
```

A user only ever sees the post-iteration version; the intermediate states never shipped to anyone. The CHANGELOG should reflect that:

```
- Add `wp.foo()` ... (final, accurate description after the follow-ups)
```

**Consolidate when**:

- All entries in the group are in the *upcoming-release section* (no entry has crossed a release boundary).
- The entries name the same symbol, system, or feature area.
- The follow-up entries describe iterations on the new feature, not separate user-facing fixes to behavior that already shipped.

**Do NOT consolidate when**:

- The Add entry sits in a previously-released section and the Fix/Change is in `[Unreleased]`. Users on the old release saw the bug; the Fix entry is real news for them. Keep both.
- The entries describe genuinely independent changes (e.g., adding a feature *and* fixing an unrelated bug whose fix happens to touch the same module).

**Process**:

1. Group candidate entries by symbol / feature area within the upcoming-release section.
2. For each group with 2+ entries, read each entry's introducing commit (Phase 3a's lookup gives this for free).
3. If the follow-ups are pre-shipping iterations on the same feature, draft a single revised entry that describes the final state. **Preserve every GH ref from every consolidated entry** — the consolidated bullet may reference multiple issues.
4. **Confirm with the user before staging** — show the original entries side by side with the proposed consolidation so the user can see what's being merged. One prompt per consolidation.
5. Stage: replace the Add entry's text with the consolidated version; mark the Fix/Change entries for deletion.
6. Record in the action log: original entries (verbatim), consolidated text, reason.

### 3c — Retro GH-ref search for entries without a link

For each entry in the upcoming-release section that does not contain a `[GH-NNNN](...)` link:

1. **Decide whether it warrants a GH issue** (judgment, not mechanical):
   - **Likely warrants** — new public API; breaking change; behavior shift; removed or deprecated symbol; performance change with a quantified claim. These should have an issue or PR for traceability.
   - **Acceptable without** — trivial bug fixes (one-line corrections, narrow corner-case patches), small doc additions, formatting cleanups.

2. **For "likely warrants" entries, search GitHub** (skip silently if `gh --version` and `gh auth status` don't both succeed):
   ```bash
   gh issue list --search '<key terms from entry>' --state all \
     --json number,title,url,state --limit 10
   gh search prs '<key terms from entry>' --json number,title,url,state --limit 10
   ```
   Look for an issue or PR whose title and body match the entry's topic. If a clear match is found, propose adding `([GH-NNNN](https://github.com/NVIDIA/warp/issues/NNNN))` to the entry. **Confirm with the user before staging** — you are filling in a missing connection, not inventing one.

3. **Flag retroactive-issue candidates.** If the entry warrants an issue but no existing issue or PR matches, flag it in the final report under "Recommend filing a retro GH issue before release." **Do not file the issue yourself** — the release manager decides.

### 3d — Section mischaracterization

Each entry sits under a subsection header (Added / Removed / Deprecated / Changed / Fixed / Documentation). The header is a contract — `### Added` should describe a brand-new public API; `### Fixed` should describe a bug fix from prior behavior; `### Changed` should describe a deliberate behavior or API shift to an already-shipped surface; `### Removed` describes a removal; `### Deprecated` introduces a sunset. Sometimes entries land in the wrong section.

Common mischaracterizations to catch:

- **Bug fix wrongly under `### Changed`.** The entry describes restoring intended behavior rather than changing it. Move to `### Fixed`.
- **Behavior change wrongly under `### Fixed`.** The entry describes a deliberate API/behavior shift, not a fix from a buggy state. Move to `### Changed`.
- **Lifted documented limitation wrongly under `### Fixed`.** The entry describes implementing a previously-unsupported feature whose absence was a *documented limitation*, not a bug. Common signals: the entry uses words like "now supports", "remove the restriction on", "lift the limitation"; or the topic matches an item under `docs/user_guide/limitations.rst` (e.g., "Strings cannot be passed into kernels", "Arrays can have a maximum of four dimensions", "Structs cannot have generic members"). A documented limitation is a known constraint the project chose to live with — lifting it is a new capability, not a fix. Move to `### Added` (if a new symbol or feature surface) or `### Changed` (if extending existing API). Also flag in the final report so the release manager can remove or update the now-lifted entry in `limitations.rst`.
- **Public-symbol rename wrongly under `### Added`.** The entry says "Add `wp.X`" but `wp.X` is the new name of an existing `wp.Y`. This is a `### Changed` (rename), and likely also wants a `### Deprecated` for `wp.Y`.
- **Removal wrongly under `### Changed`.** The entry says "Change `wp.foo()` to ..." but the change is actually a removal. Move to `### Removed` and check Phase 7-style deprecation-window expectations (was there a prior `### Deprecated` introduction in a released section?).
- **Missing `**Breaking:**` marker.** Every entry under `### Changed` or `### Removed` whose code change is source-incompatible should carry `**Breaking:**`. Same for `### Added` entries that announce a new requirement (e.g., minimum CUDA Toolkit bump).

For each entry in the upcoming-release section:

1. **Read the entry's prose in light of its section header.** Does the verb and substance match the contract? "Add" / "Allow" / "Support" belongs in Added; "Fix" / "Correct" / "Resolve" belongs in Fixed; "Change" / "Update" / "Switch" / "Rename" belongs in Changed; "Remove" / "Drop" belongs in Removed; "Deprecate" / "Mark" belongs in Deprecated.
2. **For ambiguous cases, look at the introducing commit** (Phase 3a's lookup gives this for free) and decide based on the diff. A genuine new symbol vs. a rename is usually unambiguous from the diff: a rename adds the new name in one place and the old name disappears (or is shimmed). A new symbol has no prior counterpart.
3. **Stage proposed moves** with **per-entry user confirmation**. Show the entry, its current section, the proposed section, and a one-line reason. Apply on yes; skip on no.
4. **For Changed/Removed/Added entries that are source-breaking but lack `**Breaking:**`**, propose adding the marker. Confirm per entry. The marker is a public stability signal — never add it without confirming.
5. Track every reclassification and `**Breaking:**` insertion in the action log; surface counts in the final report.

## Phase 4 — Impact sort

Operate on the upcoming release section (`[Unreleased]` at this stage, in both modes — Phase 7 renames it later in release-branch mode).

1. Read `references/sorting-rubric.md`.

2. For each subsection present (in Warp's canonical order: Added → Removed → Deprecated → Changed → Fixed → Documentation; Security only if present):
   - Score each entry on user-impact (high / mid / low) per the rubric.
   - Sort high → mid → low.
   - Within the same impact tier, prefer to keep entries on the same topic adjacent (e.g., several `wp.tile_*` entries clustered) but never demote a higher-impact entry to enforce grouping.

3. Stage the reordering in memory; do **NOT** write the file yet. Phases 2-6 share a single staged buffer; Phase 6 surfaces the consolidated diff once.

## Phase 5 — Language pass

Read `references/language-conventions.md`.

1. **Mechanical filters first.** For each entry in the upcoming release section:
   - **Test-only changes** (entry text mentions only test files, test runners, or CI configuration with no user-observable effect) → propose deletion.
   - **Trivial doc tweaks** (typo fixes, single-sentence rewords with no new content) → propose deletion.
   - **Whole-section additions** (a new user guide page, a new example, a new doctest-driven reference) → keep.
   - **Internal-jargon entries** (entry text references internal module paths `warp._src.*`, C++ template names like `launch_bounds_t`, private identifiers, "refactor internal X" framings) → flag for rewrite (if there is real user-observable effect) or deletion (if not).
   - **RST double-backticks** (entry uses `` ``foo`` `` from RST/docstring convention where markdown wants `` `foo` ``) → fix in place across the upcoming-release section. Replace every `` ``X`` `` with `` `X` `` whenever `X` contains no backtick; leave `` `` `…` `` `` alone (legitimate markdown for code containing a literal backtick), and never touch fenced ``` ``` ``` blocks. Mechanical; no per-entry confirmation; surface the count in the final report. See `references/language-conventions.md` "Markdown backticks, not RST double-backticks".
   - **Informal experimental hedging** (entry uses `preliminary`, `early` / `early access`, `alpha` / `beta` qualifying the feature, `tentative`, `WIP` / `work-in-progress`, `draft`, `provisional`, or unbolded `experimental` / trailing `(experimental)` in place of the canonical `**Experimental**:` prefix) → flag for rewrite to use the canonical marker. See `references/language-conventions.md` "Experimental-feature flag convention" for the full list, exemplar (cuBQL BVH backend), and rewrite pattern. Confirm with the user per entry before staging — adding `**Experimental**:` is a public stability signal.

2. **User-perspective subagent fan-out for ambiguous entries.** An entry is ambiguous if it survived the mechanical filters but you are not sure (a) whether a typical Warp user would understand the change or (b) whether the change has user-visible impact at all.

   For each ambiguous entry, dispatch a fresh subagent. The canonical prompt template lives in `references/language-conventions.md` — load it and use it verbatim. **Spawn all subagents for a single audit in one parallel batch** (one message with multiple Agent tool calls). Do not serialize.

   Use the responses to decide: keep as-is, rewrite, or delete. The subagent **does not draft the rewrite**; you draft, the subagent only reacts as a user.

3. **When rewriting**, preserve every `[GH-NNNN](https://github.com/NVIDIA/warp/issues/NNNN)` link from the original, verbatim. Lead the rewrite with what the user can now do (or what stopped working before the fix); push implementation detail to the back, or drop it.

4. **Track every action** (kept / rewritten / deleted) with a one-line reason. The action log is the audit trail surfaced in the final chat report.

5. **Editorial conventions sweep.** Read these sections of `references/language-conventions.md` and apply each rule to *every* entry in the upcoming-release section (not just entries the prior steps touched):

   - **Imperative mood** — entry starts with an imperative verb (`Add`, `Fix`, `Update`, `Switch`, `Reduce`, etc.). Past tense (`Added`), present indicative (`Adds`), and declarative passives (`X is now ...`, `X now supports Y`) get rewritten.
   - **Hyphenation** — compound modifiers before nouns get hyphens (`user-facing API`, `16-byte alignment`, `per-thread cooperative add`); same words in predicate position do not (`the API is user facing`).
   - **GH link position** — every `[GH-NNNN](...)` link sits at the *end* of the entry, in parentheses, immediately before the closing period. Mid-sentence GH refs get moved to the tail.
   - **Symbol formatting consistency** — every Python symbol referenced in an entry follows the rules in "Symbol formatting conventions" (module-qualified, parens on functions, `@` on decorators). Inconsistencies within a single entry get fixed; differences across entries get fixed too — the upcoming-release section should read as one consistent voice.

   Most of these are mechanical; apply without per-entry confirmation. Confirm with the user only when the rewrite changes the entry's *meaning* (e.g., an imperative-mood rewrite that changes which verb is used, or a hyphenation that disambiguates a Warp-specific term).

   Track corrections as counts (per-entry detail is too noisy):
   - Imperative-mood corrections: <I>
   - Hyphenation corrections: <H>
   - GH-link repositions: <L>
   - Symbol-formatting fixes: <S>

   Surface counts in the final report.

## Phase 6 — Line-wrap and consolidated diff

1. **Wrap every entry in the upcoming release section to ≤ 120 columns per line.** This applies to *all* bullets in the section, not only entries the previous phases touched — pre-existing entries that exceed 120 chars get reflowed too. The 120-char limit is **hard**. Never produce a 121-char line.

2. **Prefer fewer lines** subject to the 120 limit. A bullet that fits on one line should be one line; do not insert a break for aesthetics.

3. **Use semantic line breaks** when a bullet must span multiple lines:
   - Break at clause boundaries: after a comma that introduces a new clause; after a period inside a multi-sentence bullet; before an opening parenthesis introducing a GH ref; before a `with` / `for` / `when` subordinator.
   - Never break inside an inline code span, inside a markdown link, or in the middle of a noun phrase.
   - Continuation lines align with the bullet's content column (typically two spaces beyond `- `).
   - Markdown viewers ignore these breaks; they only help raw-text readers, so prefer breaks that read well in plaintext but never break correctness.

4. **Show the consolidated diff** of all staged changes from Phases 1.5-6 (after post-release sync if main mode triggered it, after lost-entry moves, after sorting, after language edits, after wrap) and **wait for user confirmation** before writing.

5. On confirmation, write the file. In main mode, this is the final mutation.

## Phase 7 — Promote + ref bump (release-branch mode only)

Skip in main mode.

1. **Rename the header.** `## [Unreleased] - YYYY-??` becomes `## [X.Y.Z] - <release-date>`, where `<release-date>` is the placeholder the user confirmed in Phase 1.7 (default `YYYY-??`, override only if explicitly supplied). Release dates often slip; a placeholder beats a wrong fact in the file.

2. **Add the new compare-link** in the link-reference block at the bottom of the file. Insert in version order (most recent first):

   ```
   [X.Y.Z]: https://github.com/NVIDIA/warp/releases/tag/vX.Y.Z
   ```

   The `vX.Y.Z` tag may not exist yet — that's fine. The URL becomes valid the moment the tag is created. Adding the ref now means release-prep does not need a follow-up commit just for the link.

3. **Update or remove the `[Unreleased]` link** based on file state after Phase 7.1. First grep for `^\[Unreleased\]:` in the link-reference block:
   - **Link present, no `[Unreleased]` header above** (we just renamed it to `[X.Y.Z]` and there is no fresh `[Unreleased]` above it) → drop the `[Unreleased]:` line; nothing on this branch back-references it.
   - **Link present, `[Unreleased]` header still above `[X.Y.Z]`** (a fresh `[Unreleased]` was added above the promoted section, e.g., during patch-release prep) → keep `[Unreleased]:` and update its compare base to `vX.Y.Z`:
     ```
     [Unreleased]: https://github.com/NVIDIA/warp/compare/vX.Y.Z...HEAD
     ```
   - **No `[Unreleased]:` link in the file** → leave the file alone for this step. Do not invent a link the file did not previously carry.

4. **Show the diff** of the rename + ref-block changes and **confirm before writing**.

## Final report

Print a single chat message summarizing every change applied:

```
Changelog audit complete for <ref> (<mode>). Target version <X.Y.Z>.

Post-release sync (main mode only; omit block if Phase 1.5 did not run):
- Branch: switched to <branch-name> (created from <prior-branch>) before mutating CHANGELOG.md.
- Synced sections for: <vX.Y.Z, vX.Y.Z+1, ...>.
- Skipped (tag has no [X.Y.Z] section in its own CHANGELOG): <list, if any>.
- Partial-backport conflicts: <list, with the user's resolution: replace | merge | skip>.
- Deduped <D> entries from [Unreleased]; <K> kept after per-entry review.
  - DEDUPED: "<unreleased excerpt>" → already in [<X.Y.Z>] as "<authoritative excerpt>" (matched on <GH-NNNN | text similarity <score>>)
  - KEPT (user override): "<unreleased excerpt>" — appeared similar to [<X.Y.Z>] entry "<...>" but user signaled distinct change.
- Compare-link refs added: <list of [X.Y.Z]: lines>; [Unreleased]: rotated to compare base v<newest-synced-tag> (or: file has no [Unreleased]: link, left as-is).

Lost-entry recovery: <N> moved, <M> skipped.
- <bullet excerpt>: moved from [<old-version>] to [<target>] (commit <sha-short>: "<subject>")
- ...

Verify, consolidate, link metadata:
- Verified accurate: <V> entries.
- Verified with revision: <R'> entries (each with original → revised excerpt + reason).
- Could-not-verify: <U'> entries (flag for human review; list each with the obstacle: "Warp build failed", "no commit found", "ambiguous behavior").
- Consolidated: <C> Add+Fix+Change groups merged into single entries (each with the original entries' excerpts and the consolidated excerpt).
- Retro GH refs added: <G> entries got a previously-missing `[GH-NNNN]` link from existing GitHub issues/PRs (each with the entry excerpt and the issue URL).
- Recommend filing retro GH issues before release: <F> entries (list excerpts so the release manager can decide which need a retro filing).
- Section reclassifications: <RC> entries moved between subsections (each line: "<excerpt>: <src-section> → <dst-section> — <reason>").
- `**Breaking:**` markers added: <B> entries (list excerpts so the user can confirm the breakage call).

Impact sort: <K> entries reordered across <S> subsections.

Language pass: <D> deleted, <R> rewritten, <U> kept verbatim.
- DELETED: "<original excerpt>" — <reason>
- REWRITTEN: "<original excerpt>" → "<new excerpt>" — <reason>
- ...

Editorial conventions sweep: <I> imperative-mood corrections, <H> hyphenation corrections, <L> GH-link repositions, <S> symbol-formatting fixes.
- Backtick fix: <B> RST-style ``X`` occurrences across <BE> entries rewritten to markdown `X`.

Line-wrap: <W> entries reflowed to fit 120-char limit.

Ref bump (release-branch mode): renamed [Unreleased] → [X.Y.Z]; added compare-link for vX.Y.Z; <kept|dropped|absent — left as-is> [Unreleased]: ref. (Or in main mode: skipped.)
```

Every rewritten / deleted entry **must** include a one-line reason ("references internal `warp._src.codegen` API", "no user-observable effect", "subagent feedback: typical user couldn't tell what changed without reading the source"). The report is the audit trail.

## Failure modes

- **Ref doesn't exist.** Surface the `git rev-parse` error and stop.
- **CHANGELOG `[Unreleased]` missing or empty.**
  - Main mode: nothing to do, exit cleanly.
  - Release-branch mode: surface; the user may have already renamed the header (so the upcoming-release section is `[X.Y.Z]` already). If `[X.Y.Z]` is present, treat *that* as the target for Phases 2-6 and skip the rename in Phase 7.
- **`[X.Y.Z]` already exists in the file with the version we'd be promoting to.** Release-branch mode: rename was already done. Skip the rename but still verify and bump refs if missing.
- **Lost-entry diff produces no candidates.** Print "No lost entries detected." and proceed to Phase 4.
- **Subagent fan-out returns nothing for an entry** (timeout, refusal). Treat as "keep" by default but flag in the final report so a human can re-check.
- **No release tag exists yet** (e.g., brand-new repo or `release-1.0` for the very first stable). Skip Phase 2, note in the report, proceed.
- **Worktree creation fails** (disk full, target path exists). Surface the error; do not fall back to dirtying the user's current tree.
- **Warp build needed for Phase 3a verification but `build_lib.py` fails** (CUDA mismatch, compiler error, missing dep). Surface the error verbatim, mark all entries that would have required a runtime check as could-not-verify, and continue with the remaining phases. Do not silently skip verification; do not paper over the build failure with a guess about what the entry probably means.
- **`gh` unavailable or unauthenticated for Phase 3c retro search.** Skip the GitHub search silently for entries lacking GH refs, but still flag entries you would have searched for so the release manager can run the search by hand.
- **Phase 1.5 working tree dirty when branch creation needed.** Surface the `git status --porcelain` output and ask the user to commit or stash by hand. Do NOT auto-stash; do NOT proceed against a dirty tree. Aborting and asking the user to clean up is the right move.
- **Phase 1.5 user rejects all proposed branch names.** Phase 1.5 cannot proceed without a writable feature branch. Abort the entire audit; surface the missing-tags set so the user can come back to it.
- **Phase 1.5 `git switch -c` fails because the branch already exists.** Surface the existing branch's HEAD commit (`git rev-parse <branch>`) and ask: switch to it (re-use), pick a different name, or abort. Do not force-update an existing branch.
- **Phase 1.5 tag has no `[X.Y.Z]` section in its own CHANGELOG** (release was tagged without the rename). Drop that tag from the missing-tags set, log it under the report's "Skipped (tag has no [X.Y.Z] section in its own CHANGELOG)" line, continue with the remaining tags.
- **Phase 1.5 partial-backport conflict** (the section header for the version being synced already exists in main's CHANGELOG with diverging bullets). Surface a side-by-side diff of (main's existing section) vs (tag's section) and ask the user to choose: replace, merge bullet-by-bullet, or skip this tag. Record the resolution in the final report.
- **Phase 1.5 missing-tags set requires a tag that is not present locally.** `git show v<X.Y.Z>:CHANGELOG.md` fails. Surface the failure and suggest `git fetch --tags`. If the user re-runs after fetching, the audit can resume; if not, drop that tag from the set with a warning.

## Regexes and parsing rules (inline reference)

- `[Unreleased]` header: `^## \[Unreleased\]( - .*)?$`. May carry a trailing date or `??` placeholder.
- Released-version header: `^## \[(\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?)\]( - .*)?$`.
- Subsection header: `^### (Added|Changed|Deprecated|Removed|Fixed|Security|Documentation)$`.
- GH ref: `\bGH-(\d+)` (word boundary prevents matching inside identifiers).
- Breaking flag: literal substring `**Breaking:**`.
- Experimental flag: literal substring `**Experimental**` (with or without trailing colon, may be `**Experimental:**`).
- Link-reference line: `^\[([^\]]+)\]: (https?://.*)$`.
- Bullet: `^- ` at column 0; continuation lines indented by two or more spaces.
