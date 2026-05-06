---
name: release-notes
description: "Generate a starting-point GitHub release notes draft for an upcoming Warp release (feature or bugfix, auto-detected from the version). Manually invoke as a slash command from a release branch, or pass a branch / git ref as the argument."
disable-model-invocation: true
argument-hint: "[branch-or-ref]"
allowed-tools: Bash(git log *) Bash(git show *) Bash(git tag *) Bash(git rev-parse *) Bash(git diff *) Bash(git diff-tree *) Bash(git cat-file *) Bash(git ls-tree *) Bash(git merge-base *) Bash(uv run *list_contributors.py*) Bash(gh --version) Bash(gh auth status) Bash(gh release view *) Bash(gh api *) Bash(gh pr view *) Bash(gh gist create *) Bash(gh gist list *) Bash(gh gist edit *) Bash(uv run /tmp/release_notes_example_*) Bash(rm /tmp/release_notes_example_*) Bash(rm /tmp/warp-*) Read Write Grep Glob
---

# Release Notes Draft

Generates a markdown draft of the impending Warp release's GitHub release notes, written
in GitHub Flavored Markdown. Auto-detects feature vs bugfix release from the version,
picks the correct prior tag for diffing, and produces a digestible user-facing summary
that the release manager edits and posts to the release page.

**Output:** a single markdown report, filed according to the destination chosen in Phase 1:
- **Secret gist** (default when `gh` is available and authenticated): stable filename `warp-<version-string>-release-notes.md`, stable description `Warp <version-string> Release Notes Draft`. Later runs against the same version revise the same gist in place; prior versions are preserved in the gist's git history.
- **Local markdown file** (fallback when `gh` unavailable, or opt-in when `gh` available): dated path at `$(pwd)/warp-<version-string>-release-notes-<YYYY-MM-DD>.md`. Not auto-committed; the user reviews, edits, and copy-pastes into the GitHub release form.

**Inputs:** the positional argument is a branch name (resolved as `upstream/<name>` → `origin/<name>` → local), a tag (e.g. `v1.13.0`), or omitted (defaults to `HEAD`). Version string comes from `VERSION.md` at the resolved head, or parsed from the tag name.

## Phase 1 — Resolve target and previous release

1. **Resolve the head ref.** With no argument, head = `HEAD`. With an argument, try in order: `upstream/<arg>`, `origin/<arg>`, `<arg>` (local branch or tag). Use the first that resolves via `git rev-parse --verify <candidate>`. Record both the symbolic ref and the resolved SHA. If nothing resolves, abort with the candidate list shown.

2. **Read the version string.** If the argument was a tag matching `v<X>.<Y>.<Z>...`, parse the version directly from the tag (strip the leading `v`). Otherwise read `VERSION.md` at the head:
   ```bash
   git show <head-ref>:VERSION.md
   ```
   Strip whitespace; this is the raw version string (e.g. `1.13.0rc3`, `1.13.0`, `1.13.1`).

3. **Parse `(major, minor, patch)`** from the leading numeric portion; discard pre-release suffixes (`rc1`, `dev0`, `.post1`). Release notes are written for the eventual release, so `1.13.0rc3` becomes target version `1.13.0`.

4. **Classify release type.** `patch == 0` → **feature release** (full notes); `patch > 0` → **bugfix release** (slim notes).

5. **Determine the previous release tag** (the diff base) using integer math on the parsed tuple:
   - **Feature release (`X.Y.0`):** highest `vX.<Y-1>.*` tag. If `minor == 0`, fall back to highest `v<X-1>.*`.
     ```bash
     git tag --list 'v<major>.<minor-1>.*' --sort=-v:refname
     ```
   - **Bugfix release (`X.Y.Z`, `Z > 0`):** highest `vX.Y.<Z'>` where `Z' < Z`. Enumerate and filter:
     ```bash
     git tag --list 'v<major>.<minor>.*' --sort=-v:refname
     ```

6. **Probe the head for an existing tag.** If `git tag --points-at <head-ref>` returns a tag matching the target version (e.g. `v1.13.0`), the tag exists but the GitHub release isn't published yet. Note this in the confirmation message; CHANGELOG URLs can use the tag.

7. **Count commits in the range** using the **same cherry-pick-filtered range the contributor helper uses** (so the scope shown to the user matches the work that will be analyzed):
   ```bash
   git log --no-merges --oneline --cherry-pick --right-only <prev-tag>...<head-ref> | wc -l
   ```
   Note the `...` (symmetric difference) and `--cherry-pick --right-only` flags: these drop commits whose patch-id has an equivalent on the prior release. This matters when the previous tag is on a bugfix branch rather than an ancestor of the head — without filtering, the proposal can overstate scope by hundreds of commits, and the user confirms a misleading count. If zero after filtering, abort.

8. **Probe `gh` and look up matching gists.** Run `gh --version && gh auth status`. If either fails, force destination to `local` and skip the rest. Otherwise run `gh gist list --limit 1000` and filter rows whose description exactly matches `Warp <version-string> Release Notes Draft`. Record the matching gist IDs and count (0, 1, N ≥ 2) for Phase 2. Display URLs are `https://gist.github.com/<id>`.

## Phase 2 — Confirm scope and destination with user

Print a single proposal block and **wait for explicit user confirmation**. Mandatory pause; do not proceed to Phase 3 until the user replies.

Lead with the scope line:

> Drafting **<feature|bugfix>** release notes for Warp **<version-string>**.
> - Head: `<head-ref>` @ `<short-sha>` <(tag `<tag>` exists at this SHA)>
> - Previous release: `<prev-tag>` @ `<short-sha>`
> - Commits in range: `<N>`

Append the output block matching the current `gh` / match state:

**`gh` unavailable:**
> Output: local markdown file at `<cwd>/warp-<version-string>-release-notes-<today>.md` (`gh` not available). Confirm, or specify different refs?

**`gh` available, 0 matches:**
> Output: (a) new secret gist [default], (b) local markdown file in cwd. Confirm refs + pick.

**`gh` available, 1 match:**
> Output: (a) revise existing gist `<url>` [default], (b) new secret gist, (c) local markdown file in cwd. Confirm refs + pick.

**`gh` available, N matches (N ≥ 2):**
> Multiple existing gists share the stable title:
> 1. `<url-1>` — updated `<time-1>`
> 2. `<url-2>` — updated `<time-2>`
> ...
>
> Output: (a) revise gist by number, (b) new secret gist, (c) local markdown file in cwd. Confirm refs + pick.

Translate the reply into one destination token (`local`, `new-gist`, `revise-gist:<id>`). Letters `(a)`/`(b)`/`(c)` are positional within the prompt shown; for the N-match branch a numeric reply (e.g. "revise 2") selects from the listed gists. New refs in the reply re-run Phase 1.

## Phase 3 — Gather inputs

### 3a — CHANGELOG section

Read `CHANGELOG.md` at the head ref:
```bash
git show <head-ref>:CHANGELOG.md
```

Locate the section for this release. Try `## [<version-string>]` first (e.g. `## [1.13.0]`), then fall back to `## [Unreleased]` if the release branch hasn't promoted yet.

Collect bullets per subsection (`### Added`, `### Removed`, `### Deprecated`, `### Changed`, `### Fixed`, `### Documentation`). For each bullet, capture: full raw text, section name, GH refs (regex `\bGH-(\d+)`, deduped), experimental flag (`**Experimental**` present), breaking flag (`**Breaking:**` present).

### 3b — Commit list and contributor attribution

Run the contributor enumeration script via `uv run` so it picks up the repo's pinned Python (the script uses 3.10+ syntax and the repo policy in `AGENTS.md` requires `uv run` over the system interpreter). Stays inside `Bash(uv run *list_contributors.py*)`. The temp changelog uses the `warp-` prefix so the existing cleanup glob catches it:
```bash
git show <head-ref>:CHANGELOG.md > /tmp/warp-<version-string>-changelog.md
uv run "$(git rev-parse --show-toplevel)/.claude/skills/release-notes/scripts/list_contributors.py" \
  --base <prev-tag> \
  --head <head-ref> \
  --changelog /tmp/warp-<version-string>-changelog.md \
  --target-version <target-version-string>
```
Delete `/tmp/warp-<version-string>-changelog.md` after the script returns.

The script emits JSON with `commits[]` (sha, subject, author name+email, GH refs from subject+body+CHANGELOG diff, files touched) and `contributors[]` aggregated per `(name, email)`. Each contributor record carries: `commit_count`, `gh_login` (best-effort: noreply email pattern, then `gh api`), `classification` (`nvidia` | `external` — see `references/contributor-attribution.md`), `pr_summary` (PR number + subject pairs from squash commits), `changelog_sections` (CHANGELOG section labels their GH refs land in), and `already_shipped` (true when all attributed work falls in a *prior* CHANGELOG section, meaning they were already thanked).

Two robustness filters run by default: a cherry-pick filter (drops commits whose patch-id appears in `<prev-tag>`, common when fixes land on bugfix branches before main) and CHANGELOG-section attribution (sets `already_shipped`). See `references/contributor-attribution.md` for the rationale.

Capture the JSON; Phase 5d uses `contributors[]` filtered by `classification == "external"` AND `already_shipped == False`.

### 3c — Past-release calibration (optional)

For tone calibration, fetch one recent release body of matching type via `gh release view <tag> --repo NVIDIA/warp --json body --jq .body`. Mirror structure and register; do not copy content.

## Phase 4 — Plan grouping and inspect implementations (feature releases only)

Bugfix releases skip Phase 4 and proceed to Phase 5 (include a one-sentence intro in Phase 5a). Read `references/style-rules.md`, `references/feature-investigation.md`, and `references/feature-release-template.md` before writing.

1. **Identify the lead feature** — the single most exciting capability in the release. Gets its own `### …` block at the top of `## New features` with a code example. Prefer items that unlock previously-impossible workflows (new artifact, file format, cross-language boundary) over new parameters. Honor the lead the user named at invocation.
2. **Plan theme groups.** Bucket remaining `### Added` / `### Changed` entries into 3–6 themes that reflect what users *do* with the features. Common shapes: "Tile programming enhancements", "JAX integration", "`warp.fem` enhancements", "Compilation and tooling", "Performance improvements", "Language enhancements", "New examples".
3. **Order by impact.** Lead feature first, then themes by user-facing impact. `## Bug fixes` near the end, omitted if no fix warrants a highlight.
4. **Tag experimental items.** Any bullet flagged `**Experimental**` in 3a gets a `> [!IMPORTANT]` admonition under its `### …` heading.
5. **Inspect implementations for caveats, artifacts, and in-tree examples.** Apply the protocol in `references/feature-investigation.md` to the lead feature and to every `### …` block you plan to write. The protocol surfaces (a) "not yet supported" / TODO limits buried in the implementation, (b) on-disk or wire artifacts beyond the obvious file (sidecar directories, compute-arch pinning, version metadata), (c) existing in-tree examples worth linking (`warp/examples/**`, `warp/tests/**`), and (d) cross-language requirements (any feature crossing Python ↔ C++ ↔ JAX needs a snippet on each side, not just Python). Skipping this step is the difference between "looks like a release announcement" and "lets me start using the feature."

## Phase 5 — Draft sections

### 5a — Intro paragraph

Open the body with a 2-3 sentence paragraph that names the shape of the release. Lead with the unlock, not API names. Feature-release intros name the lead feature first, then 1-2 supporting themes. Bugfix-release intros are one sentence: `Warp v<version> is a bugfix release following v<prev>.` plus a changelog pointer.

### 5b — Sections

Fill in `references/feature-release-template.md` (feature release) or `references/bugfix-release-template.md` (bugfix release). Both templates carry per-section instructions inline. **Code examples are the default**: every `### …` block describing a new public API or behavioral change ships with a runnable Python snippet unless the change is a single-parameter tidy-up. Snippets must NOT call `wp.init()` (Warp initializes implicitly). Breaking changes in `## Bug fixes` or `## Changes` ship with **before/after** snippets, not prose. When a snippet has a `print(...)`, run it via `uv run /tmp/release_notes_example_<name>.py` and embed the **real** captured output. Render single-line output as a trailing `# comment`; render multi-line or structured output as a separate fenced ```text block following the Python block (see `references/style-rules.md` "Rendering output"). Required when the output IS the feature (memory tracker, diagnostics, dtype promotion). Use `> [!NOTE]` / `> [!IMPORTANT]` / `> [!CAUTION]` admonitions, never ad-hoc `**Note:**` or quoted-bold forms. See `references/style-rules.md`.

### 5c — Announcements

This section appears in both feature and bugfix release notes (deprecations carry forward across releases until they actually land). Pull from:

1. **CHANGELOG `### Removed` and `### Deprecated`** at the head ref.
2. **CHANGELOG `### Deprecated` from previous releases** that name a future removal version matching this release or the next. Search prior sections for the literal phrases `will be removed in`, `scheduled for removal in`, `removed in <X.Y>` (replacing with the actual next-version number). Each match is a candidate forward-looking entry under `### Upcoming removals` (next release).
3. **Platform support ranges** read directly from the source. Always confirm and surface:
   - **Python**: read `pyproject.toml` (`requires-python` and the `Programming Language :: Python :: 3.X` classifiers) and `.python-version` at the head ref.
   - **CUDA Toolkit**: read `warp/_src/build_dll.py` for `MIN_CTK_VERSION` (or equivalent) and the `CTK_*` constants in `build_lib.py`.
   - **OS / arch**: read the `cibuildwheel` config in `pyproject.toml` and any `manylinux_*` / `macosx_*` constraints in build scripts.
   State the range explicitly even when unchanged from the prior release ("Python 3.10–3.14 supported, 3.10 minimum"). Do not infer; quote what the build actually checks.
4. **Out-of-CHANGELOG facts the user is responsible for** (allocator policy changes, security advisories, planned support changes the build doesn't yet enforce). Surface candidates in the chat summary so the user can confirm or drop.

Sub-headings are typically `### Upcoming removals` and `### Platform support`. Drop sub-headings (or the whole `## Announcements` section) when empty.

### 5d — Acknowledgments

Use `contributors[]` from Phase 3b filtered by `already_shipped == False` (those marked True were thanked in a prior release; surface in the chat summary instead).

For each remaining `external` contributor:
- If `gh_login` is non-null, render `- @<gh-login> for <one-line summary> (#<NNNN>).`.
- If `gh_login` is null (no `gh` available, or no GitHub user could be associated with the commit email), do NOT render `@None` / `@null`. Render `- <name> for <one-line summary> (#<NNNN>).` and surface the entry in the chat summary so the release manager can resolve the handle or drop the line before posting. The trailing `(#<NNNN>)` is omitted when no CHANGELOG bullet matches the contributor's commit (per `references/contributor-attribution.md`).

If the list is empty after filtering, render `No external contributions in this release.` under the section heading.

## Phase 6 — Render and write to chosen destination

Fill in every `{{PLACEHOLDER}}` in the selected template, applying `references/style-rules.md` (chiefly: `#NNNN` not `[GH-NNNN](...)`, experimental → `> [!IMPORTANT]`, no em dashes, no skill-internal terminology, proper-noun capitalization). The destination was chosen in Phase 2: one of `local`, `new-gist`, or `revise-gist:<id>`.

**Filename conventions:**
- Local file (in `$(pwd)`): `warp-<version-string>-release-notes-<today>.md` — dated, user-facing.
- Gist file (inside the gist): `warp-<version-string>-release-notes.md` — no date. Stable name so later runs revise the same gist in place.

**Stable gist description** (used when creating a new gist; also the matching key for Phase 1.8):
```
Warp <version-string> Release Notes Draft
```

**If destination is `local`:** write the rendered markdown to `$(pwd)/<local-filename>` using the Write tool.

**If destination is `new-gist`:**
1. Write to `/tmp/<gist-filename>` using the Write tool.
2. `gh gist create --desc "<stable-desc>" /tmp/<gist-filename>` — capture the URL from stdout.
3. Delete `/tmp/<gist-filename>`.

**If destination is `revise-gist:<id>`:**
1. Write to `/tmp/<gist-filename>` (same stable name the existing gist already uses).
2. `gh gist edit <id> --filename <gist-filename> /tmp/<gist-filename>`. The `--filename` flag selects which file *inside the gist* to replace; the trailing local path supplies the new content. Without `--filename`, `gh` treats the local path as a gist-side filename selector, fails to find it, and falls through to its interactive editor. Do NOT pass `--desc`: keeping the description stable is what lets the next run match this gist again. Prior versions are preserved automatically in the gist's git history.
3. Delete `/tmp/<gist-filename>`.

Never pass `--public`. Never file a destination the user did not choose.

Print a one-line chat summary with output path/URL and headline counts (features, announcements, external contributors). Append as warranted: "revised in place; prior versions kept in gist git history" for `revise-gist`; "Already-shipped contributors skipped: @user1 (sections=[1.12.1]), ..." for `already_shipped == True` entries the user can spot-check; and "Unresolved GitHub handles (rendered with name only): <name> (<email>), ..." for any external contributors whose `gh_login` is null in the rendered acks, so the release manager can resolve or drop those lines before posting.

## Failure modes

- **Argument doesn't resolve.** Print the candidate list (`upstream/<arg>`, `origin/<arg>`, `<arg>`) and abort. Do not silently fall back to HEAD.
- **`VERSION.md` missing or unparsable at the head.** Abort with the raw file contents shown.
- **No previous tag found.** Feature release: no prior minor exists (only at `v1.0.0`-equivalents). Bugfix release: no prior `vX.Y.<Z'>` with `Z' < Z`. Abort and ask the user to specify the base manually.
- **CHANGELOG section not found.** Neither `## [<version-string>]` nor `## [Unreleased]` matches. Abort; the user likely passed the wrong head ref.
- **`gh` unavailable.** `list_contributors.py` falls back to email-only classification (only `@nvidia.com` registers as internal); private-membership NVIDIA staff committing from noreply / personal addresses will be misclassified as `external`. Note in the chat summary.

## Regexes and parsing rules (inline reference)

- GH ref: `\bGH-(\d+)`.
- Tag pattern: `^v(\d+)\.(\d+)\.(\d+)([a-z0-9.+-]*)$`. The trailing group captures `rc1`, `dev0`, `.post1`, etc.
- CHANGELOG section header: `^## \[([^\]]+)\]` — the bracketed contents are the version or `Unreleased`.
- CHANGELOG subsection header: `^### (Added|Removed|Deprecated|Changed|Fixed|Documentation)`.
- Experimental marker: `\*\*Experimental\*\*:?` (bold, optional trailing colon).
- Breaking marker: literal `**Breaking:**`.
- noreply.github.com email login: `^\d+\+([^@]+)@users\.noreply\.github\.com$` → group 1 is the GitHub username.
