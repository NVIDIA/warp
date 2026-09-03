---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: warp-changelog-audit
description: "Use when auditing and recovering Warp changelog fragments, finalizing a release changelog, or synchronizing a tagged release back to main."
license: Apache-2.0
---

# Changelog Audit

Audit the pending files under `changelog/`, preview their combined Towncrier
render, and optionally finalize or synchronize a release. `CHANGELOG.md` is the
historical release record, not the source of pending entries.

Read `changelog/README.md` from the target ref before acting. Its fragment
naming and content rules are authoritative.

## Modes

- **Main:** audit and edit pending fragments. Never build a release section.
- **Release branch:** audit fragments, build the release section, then sort the
  generated section by user impact and update comparison links.
- **Post-release sync:** on main, bring back the exact release build commit so
  the tagged section's final wording and ordering win while only shipped
  fragments are removed.

## Hard rules

- Run Towncrier through the pinned command from `changelog/README.md`.
- Use a draft build for every preview. A draft must not modify the repository.
- Edit fragment files before release finalization, not `CHANGELOG.md`.
- After the final build, edit only the new generated release section and the
  comparison-link block. This release-only normalization is where impact
  sorting and line wrapping happen.
- Treat the tagged release section as authoritative during post-release sync.
  Never reconstruct its order from fragments or Towncrier's default ordering.
- Delete shipped fragments only when the release build commit proves which
  paths Towncrier consumed. Never delete pending fragments by prose similarity
  or shared issue number.
- Keep all proposed edits in a temporary staging area until the user reviews a
  consolidated diff. Release finalization and post-release cherry-picks each
  require a separate explicit confirmation.
- Use a unique `WARP_CACHE_PATH` for every Warp command. Build Warp when a
  verification needs it; do not skip verification because binaries are stale.

## Inputs and references

The optional positional argument is any Git ref and defaults to `HEAD`.

Read these references when their phases begin:

- `references/language-conventions.md` for fragment inclusion and language.
- `references/sorting-rubric.md` for final generated-section ordering.

## Phase 1: Resolve scope

1. Resolve the ref, symbolic name, SHA, and working directory. Strip
   `origin/` or `upstream/` before classifying the branch. A name beginning
   with `release-` selects release-branch mode; everything else selects main
   mode.
2. Read `VERSION.md` and `warp/config.py` at the ref. Parse the leading
   `MAJOR.MINOR.PATCH`; `VERSION.md` wins on conflict.
3. Resolve the previous stable tag using integer version components. For a
   feature release, use the highest stable tag from the previous minor (or
   previous major at an `X.0` boundary). For a patch release, use the highest
   lower patch tag in the same minor. Exclude prerelease tags.
4. Enumerate recognized fragment paths at the ref. In the target worktree,
   also include uncommitted and untracked fragment files:

   ```bash
   git ls-tree -r --name-only <ref> -- changelog
   rg --files changelog
   ```

   Exclude `changelog/README.md`. A fragment path must match:

   ```text
   changelog/(<digits>|+<readable-slug>).(added|removed|deprecated|changed|fixed|documentation)(.<counter>).md
   ```

5. Run the pinned draft render in the worktree that contains the exact
   fragment state being audited:

   ```bash
   uvx --from towncrier==25.8.0 towncrier build --draft \
     --version <X.Y.Z> --date <YYYY-MM-DD>
   ```

   For an immutable ref not checked out anywhere, use a temporary detached
   worktree and remove it after rendering. Do not use a detached worktree when
   the audit must include uncommitted fragment edits.
6. In main mode, parse the newest stable version section already present in
   `CHANGELOG.md`. Among stable tags reachable from the ref, consider only
   versions newer than that section; tags for older historical releases may
   intentionally have no section. Newer tags whose sections are absent form
   the post-release sync set. Process them oldest first before auditing
   remaining fragments.
7. If edits are needed, require a writable feature or release branch. Never
   mutate `main`, `master`, or detached `HEAD`; propose a feature branch and
   wait. Surface a dirty worktree and let the user decide how to handle it.
   Never auto-stash.
8. Present mode, ref/SHA, working directory, target version, previous tag,
   fragment count, draft result, and missing release tags. Wait for explicit
   confirmation.

## Phase 2: Synchronize released tags on main

Skip unless main is missing a stable tag's release section.

For each missing tag, find the dedicated release build commit between the
previous stable tag and that tag:

```bash
git log --reverse --format='%H' <previous-tag>..<tag> -- CHANGELOG.md changelog/
git show --stat --summary <candidate>
git show --format=fuller <candidate> -- CHANGELOG.md changelog/
```

The correct commit adds `## [X.Y.Z]`, updates comparison links, and deletes the
fragment paths consumed by that release. Verify the tag contains the candidate
and that its generated section exactly matches `git show <tag>:CHANGELOG.md`.

Preferred sync:

1. Show the build commit SHA, tagged section, and deleted fragment paths.
2. Confirm the cherry-pick with the user.
3. Cherry-pick that exact commit onto the writable main-sync branch.
4. Verify the inserted section byte-for-byte against the tag and verify that
   fragment paths absent from the build commit remain untouched.

If the build commit cannot be cherry-picked cleanly, use a manual fallback
only when its diff still proves the consumed fragment paths. Insert the tagged
section verbatim, preserving its final order; delete exactly those proven
paths; and apply the build commit's comparison-link changes. Show the complete
fallback diff and wait for confirmation. If consumed paths cannot be proven,
stop rather than guessing.

Do not compare released entries with remaining fragments for deduplication.
The build commit is the source of truth for what shipped.

## Phase 3: Build the fragment model and recover omissions

For every pending fragment, record:

- path, identifier, category, optional counter, and full content;
- numeric GitHub issue ID or orphan status;
- introducing and modifying commits from
  `git log --follow <previous-tag>..<ref> -- <path>`;
- rendered Towncrier bullet and generated issue links;
- `**Breaking:**` and `**Experimental**` markers;
- named public symbols and feature area.

Validate names and content against `changelog/README.md`. In particular:

- Content represents one eventual bullet and does not begin with `-`.
- Numeric identifiers refer to GitHub issues, not pull requests or merge
  requests.
- Orphan identifiers begin with `+` and use a readable slug.
- Fragment content does not contain the GitHub issue link that Towncrier adds.
- Several issue IDs for one change use identical fragment contents so
  Towncrier combines them into one bullet.

Surface malformed or duplicate identities before any prose review. Re-run the
draft after every rename, category move, consolidation, or deletion.

If a fragment already exists at `<previous-tag>`, mark it as carried over and
inspect at most the nearest 20 older path commits with
`git log --follow --max-count=20 <previous-tag> -- <path>`. Do not replace the
selected ref or release range with an unbounded history search.

### Missing-fragment recovery

Translate the old lost-entry safeguard into fragment terms. Enumerate commits
in the release range with the same cherry-pick filtering used by release notes:

```bash
git log --no-merges --reverse --cherry-pick --right-only \
  <previous-tag>...<ref> --format='%H%x1f%s%x1f%b' --name-status
```

Map commits to fragments using, in order:

1. fragment paths introduced or modified by that commit;
2. GH refs in the subject/body matching numeric fragment identifiers;
3. issue topic, named symbols, touched public paths, and fragment path history.

Inspect every remaining commit against `changelog/README.md`. Drop test-only,
CI-only, release automation, formatting, and internal refactors with no user
effect. A commit affecting public API, runtime behavior, performance,
compatibility, diagnostics, packaging, supported workflows, or substantive
documentation is a missing-fragment candidate.

For each candidate, show the commit, user impact, proposed category, identifier,
path, and content. Prefer the supplied GitHub issue number; otherwise propose a
readable `+slug`. Combine several commits for the same not-yet-released feature
into one final-state entry when appropriate. Confirm each proposal before
staging a new fragment, then re-render the draft. Never infer coverage merely
because unrelated fragment prose sounds similar.

If no previous tag exists, report that omission recovery was skipped; do not
invent an unbounded history range.

## Phase 4: Verify and consolidate

### Accuracy

Verify new public APIs, breaking changes, quantified claims, experimental
features, and requested headline items against the implementation and docs.
Find candidate commits through fragment history, issue IDs, symbols, and the
release range; never search pending prose in `CHANGELOG.md`.

Read diffs and confirm symbol names, signatures, defaults, behavior, and issue
topic. For non-obvious runtime claims, write a temporary script and run it with
`uv run`. Kernel scripts must be real files, never `python -c`. If native code
changed or binaries are stale, rebuild first. Record accurate, revised, and
unverifiable results.

### Pre-release iteration consolidation

Group fragments that describe Add/Fix/Change iterations on the same feature
when none of those states shipped. Draft one entry describing the final state.
If several issues must remain linked, keep one identical fragment per issue.
Show original paths/text and proposed paths/text, then confirm before staging.

Do not consolidate a change to already-released behavior with its earlier
introduction.

### Identifier and category review

- If an orphan fragment has a clearly matching GitHub issue, propose renaming
  it to that issue ID. Never put the generated issue link in its text.
- If a category is wrong, propose a suffix rename such as
  `.fixed.md` to `.changed.md`.
- Propose `**Breaking:**` for source-incompatible Changed, Removed, or new
  requirement entries. Confirm because it is a public stability signal.
- Flag lifted documented limitations for the matching documentation update.

Use per-entry confirmation for identifier changes, category moves,
consolidations, and stability markers.

## Phase 5: Language pass on fragments

Read `references/language-conventions.md` and apply it to every pending
fragment.

1. Propose deletion of internal-only, test-only, CI-only, and trivial prose
   entries. Keep user-observable API, behavior, performance, compatibility,
   diagnostic, packaging, and substantive documentation changes.
2. Rewrite internal jargon around user impact. Preserve the fragment's issue
   identity in its filename, not as a manually written issue link.
3. Normalize Markdown backticks, imperative mood, hyphenation, stability
   markers, and public-symbol formatting.
4. Use the reference's user-perspective subagent prompt for genuinely
   ambiguous entries. The subagent reacts; it does not write the replacement.
5. Wrap fragment contents to at most 120 columns without breaking Markdown
   links or inline code.

Track every deletion and meaning-bearing rewrite with a one-line reason.
Mechanical corrections may be summarized by count.

Stage proposed files under a temporary directory, render Towncrier there, and
show one consolidated source diff plus the resulting draft. Wait for explicit
confirmation before writing fragment changes to the worktree. Re-run the draft
after writing and require a clean render.

In main mode, stop here after reporting the audit and any post-release sync.

## Phase 6: Finalize on a release branch

Require an explicit release date. Do not substitute a guessed date or
`YYYY-??` in a final build.

1. Copy `pyproject.toml`, `CHANGELOG.md`, and the current `changelog/` tree to
   a temporary directory. Run the final Towncrier build there:

   ```bash
   uvx --from towncrier==25.8.0 towncrier build --yes \
     --version <X.Y.Z> --date <YYYY-MM-DD>
   ```

2. Read `references/sorting-rubric.md`. In the generated `## [X.Y.Z]`
   section, preserve canonical subsection order and sort entries high to low
   impact within each subsection. Use topic clustering only as a tie-breaker.
3. Wrap the generated section to at most 120 columns. Towncrier's generated
   issue links stay at each entry's end.
4. Add `[X.Y.Z]` to the comparison-link block and rotate `[Unreleased]` to
   compare `vX.Y.Z...HEAD`. Do not edit older release sections.
5. Show the exact simulated diff: generated section, sorted order, consumed
   fragment deletions, and comparison links. Wait for explicit confirmation.
6. Run the same build in the actual release worktree and apply the reviewed
   ordering, wrapping, and link updates. Compare the actual diff with the
   simulation; stop if they differ unexpectedly.
7. Re-run a draft. It must report no pending fragments that were part of this
   release. Fragments intentionally excluded from the release must remain.

Keep the build, fragment deletions, final ordering, and link changes together
in one dedicated release changelog commit when the user has authorized a
commit. Record its full SHA; post-release synchronization depends on it. Do not
silently create a commit when the user requested only an audit.

## Final report

Report:

- mode, ref/SHA, target version, previous tag, and fragment counts;
- post-release build commits synced and exact fragment paths removed;
- invalid names or identities and their resolutions;
- missing-fragment candidates created, skipped, or rejected, with commit SHAs;
- verification results, consolidations, identifier/category changes, and
  stability-marker changes;
- deleted, rewritten, and unchanged fragments with reasons;
- final draft status;
- release finalization details, impact reorder count, comparison-link changes,
  and dedicated build commit SHA or `pending commit`;
- warnings and unverifiable claims.

## Failure modes

- **No pending fragments:** report a clean empty draft. In release mode, ask
  whether the release was already built; never infer pending content from the
  newest historical section.
- **No previous tag:** skip missing-fragment recovery and report the omission.
- **Draft render fails:** surface Towncrier output and stop prose editing until
  naming or configuration is fixed.
- **Target ref is not writable:** use or propose a worktree/branch and wait.
- **Dirty worktree blocks branch or sync work:** surface it; never auto-stash.
- **Build commit is ambiguous or missing:** require the recorded SHA or prove
  the unique commit from tag history. Never guess consumed fragments.
- **Old stable tag has no historical section:** ignore it when it is not newer
  than the newest section already recorded in `CHANGELOG.md`.
- **Tagged and main sections diverge:** the tag wins. Show the diff before
  replacing main's section.
- **Final build differs from simulation:** stop and show both diffs.
- **Warp verification build fails:** report the error verbatim and mark only
  affected claims unverifiable; continue with non-runtime review.
- **GitHub CLI is unavailable:** skip live issue lookup, retain filename-based
  identity, and report lookup omissions.

## Parsing rules

- Fragment path:
  `^changelog/(?:\+[A-Za-z0-9][A-Za-z0-9-]*|\d+)\.(added|removed|deprecated|changed|fixed|documentation)(?:\.\d+)?\.md$`
- Released header: `^## \[(\d+\.\d+\.\d+(?:-[A-Za-z0-9.]+)?)\]( - .*)?$`
- Subsection: `^### (Added|Removed|Deprecated|Changed|Fixed|Documentation)$`
- Generated GH ref: `\bGH-(\d+)`
- Breaking marker: literal `**Breaking:**`.
- Experimental marker: literal `**Experimental**` with an optional colon
  inside or immediately after the bold span.
