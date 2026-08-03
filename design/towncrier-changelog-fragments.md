# Towncrier changelog fragments

**Status**: Implemented

## Motivation

Warp currently asks contributors to edit the `Unreleased` section of
`CHANGELOG.md` in every pull request or merge request with a user-facing change.
That works, but it also makes the changelog a frequent source of merge conflicts.
It is particularly awkward once a release branch has split from `main`: both
branches continue editing the same section, and the release notes must be
reconciled by hand.

Towncrier moves those edits into small files under `changelog/`. Contributors
write the entry next to the change that needs it. A maintainer reviews the
combined draft and builds the final section on the release branch.

This design keeps the policy deliberately light. Human reviewers decide whether
a change needs a fragment and whether the wording is useful. Automation only
checks that the fragments people add can be rendered.

## Requirements

| ID | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| R1 | Record user-facing changes in individual fragments | Must | Internal-only changes may omit a fragment |
| R2 | Preserve Warp's existing Markdown headings and category order | Must | Added, Removed, Deprecated, Changed, Fixed, Documentation |
| R3 | Link entries to GitHub issues when an issue exists | Must | Pull request and merge request numbers are not issue identifiers |
| R4 | Support readable entries when there is no GitHub issue | Must | Use Towncrier's native orphan identifiers |
| R5 | Validate added fragments in both GitHub and GitLab CI | Must | Run only when changelog-related paths change |
| R6 | Preserve fragments merged to `main` after a release branch is cut | Must | Synchronize the release build commit back to `main` |
| R7 | Adopt the process after the 1.16 changelog is synchronized to `main` and migrate retained 1.17 entries | Must | Avoid a mixed transition |

Non-goals:

- Requiring every pull request or merge request to add a fragment through CI.
- Adding a custom policy script, `.skip` files, or `towncrier check`.
- Adding a pre-commit hook or a permanent test suite for the changelog setup.
- Adding Towncrier to Warp's project dependencies or lockfile.
- Converting historical changelog entries.
- Automating the release build or changelog synchronization.
- Updating the repository's `.claude` and `.codex` skills in the initial change.

## Design

### Towncrier configuration

Towncrier configuration lives in `pyproject.toml`. The project uses
`changelog/` for fragments and writes releases into `CHANGELOG.md` beneath a
Markdown insertion marker. The title format remains
`## [{version}] - {project_date}`.

The configuration defines these types in the same order as the current
changelog:

1. `added`
2. `removed`
3. `deprecated`
4. `changed`
5. `fixed`
6. `documentation`

Numeric identifiers use an `issue_pattern` that accepts digits only. The
`issue_format` points to `https://github.com/NVIDIA/warp/issues/{issue}`.
Towncrier's `ignore` list is explicitly empty so a draft build reports malformed
fragment names instead of silently skipping them. `wrap` is disabled to avoid
rewrapping contributor prose.

The built-in Markdown rendering is enough for Warp's changelog format, so there
is no custom Jinja template. CI and maintainer commands use Towncrier 25.8.0
through `uvx`; Towncrier does not become a project dependency.

### Fragment names and content

When work has a GitHub issue, its fragments use the issue number:

```text
1708.fixed.md
1708.documentation.md
```

The number refers to the GitHub issue, not the GitHub pull request or GitLab
merge request. Towncrier appends the issue link when it renders the entry, so
contributors do not put that link in the fragment text. Links needed by the
content itself, such as a migration guide, are still allowed.

When there is no GitHub issue, contributors use a readable orphan identifier:

```text
+fix-cuda-graph-capture.fixed.md
```

The leading `+` is Towncrier's native signal that the entry has no issue link.
The slug should describe the change rather than use an opaque hash. A normalized
branch description is often a useful starting point, but it is not required.
When a branch name is used, drop any username prefix and replace runs of
non-alphanumeric characters with `-`.

A fragment contains one eventual changelog bullet and does not begin with `-`.
It uses imperative present tense and describes the effect on users. Entries in
the `changed`, `deprecated`, and `removed` categories include migration guidance
when users need to update their code.

Several entries of the same type use numeric counters:

```text
1708.fixed.md
1708.fixed.1.md
```

When one change resolves several GitHub issues, contributors may put the same
text in a fragment for each issue. Towncrier combines identical entries into one
bullet with multiple issue links.

### Human review decides whether a fragment is needed

An internal-only change does not need a fragment. The pull request and merge
request templates ask the author to confirm that a fragment was added when the
change affects users. Reviewers handle missing fragments and weak prose as part
of normal review.

There is no placeholder file for changes that do not need release notes. This
avoids building a second policy system around Towncrier and keeps routine
maintenance changes quiet.

### Draft validation

GitHub gets a dedicated `.github/workflows/changelog.yml` workflow for mirrored
pull request branches. GitLab gets a small job in `.gitlab-ci.yml`. The two jobs
run the same command:

```console
uvx --from towncrier==25.8.0 towncrier build --draft \
  --version 0.0.0 --date 1970-01-01
```

They run only when one of these paths changes:

- `CHANGELOG.md`
- `changelog/**`
- `pyproject.toml`

The draft catches invalid filenames, unknown types, invalid numeric issue
identifiers, duplicate fragment identities, and rendering failures. It does not
require a fragment, choose between an issue and orphan identifier, judge the
prose, or prohibit a direct changelog edit. Those remain review decisions.

This validation is separate from Warp's full pull request workflow. A fragment
edit therefore does not restart unrelated documentation, lockfile, or platform
test jobs. There is no pre-commit hook because GitLab runs pre-commit against all
files, which would turn a path-scoped check back into an every-merge-request
check.

### Contributor and maintainer documentation

`changelog/README.md` is the source of truth for fragment names, categories,
writing guidance, local preview commands, and the release procedure.
`CONTRIBUTING.md` and `AGENTS.md` link to that guide instead of telling people to
edit `CHANGELOG.md` directly.

The GitHub pull request and GitLab merge request templates replace their current
changelog checkbox with wording based on user impact:

> Added a changelog fragment if this change affects users.

`docs/project/changelog.md` remains unchanged because it already exposes the
top-level changelog. Warp has no CODEOWNERS rule to adjust for the new guide.

### Transition and release flow

The initial Towncrier change lands after the 1.16 changelog has been synchronized
to `main`. Because user-facing 1.17 work already exists at that point, the change
migrates the five retained `Unreleased` entries into fragments before replacing
that section's contents with Towncrier's insertion marker. It leaves all released
history alone.

For each release, the maintainer works on `release-X.Y`:

1. Confirm that the branch has the fragments for every user-facing change that
   will ship.
2. Run a draft build with the real version and date.
3. Edit fragment wording as needed and repeat the draft until the release notes
   read well.
4. Run `uvx --from towncrier==25.8.0 towncrier build --yes --version X.Y.Z --date YYYY-MM-DD`.
5. Update the comparison links at the bottom of `CHANGELOG.md`.
6. Commit the generated release section, consumed fragment deletions, and link
   updates together in a dedicated changelog build commit.

After the release, create a branch from the current `main` and cherry-pick that
exact build commit. The cherry-pick adds the released section and removes only
the fragments that shipped. Fragments merged to `main` after `release-X.Y` was
cut remain in place for the next release. The synchronization goes through a
small changelog-only pull request or merge request.

## Alternatives considered

### Custom policy around Towncrier

A policy script could require every change to add either a fragment or an
explicit skip file. It could also enforce identifier grouping and release-branch
rules. That provides stronger automation, but it creates a repository-specific
system that needs its own tests and maintenance. Warp's reviewers already decide
whether a change is user-facing, so the extra machinery does not earn its cost.

### `towncrier check`

`towncrier check` verifies that a branch adds at least one recognized fragment.
Using it would also require an exemption mechanism for internal changes. It does
not replace the release draft review, and Warp does not need CI to enforce
fragment presence.

### Fragments with no CI validation

Review alone could manage fragments, but malformed names or configuration errors
would then appear during a release. A path-scoped draft build catches those
mistakes without adding policy.

## Testing strategy

This change does not add tests under `warp/tests` or a separate meta-test suite.
The implementation is verified directly with temporary fragments:

- Render a numeric issue fragment and confirm the category and GitHub issue link.
- Render a readable orphan and confirm that no issue link appears.
- Confirm malformed names and unknown categories fail the draft build.
- Render all six categories and confirm their order.
- Give two issue fragments identical text and confirm Towncrier produces one
  entry with both links.
- Run a final build in a disposable copy and confirm it inserts the release,
  removes consumed fragments, and leaves historical entries intact.
- Check the GitHub and GitLab path filters and run pre-commit on the files changed
  by the implementation.

## Follow-up

After the Towncrier process lands, update the mirrored skills under `.claude`
and `.codex` so they stop assuming direct edits to `CHANGELOG.md`. At minimum,
audit `changelog-audit`, `release-audit`, `release-notes`, and
`warp-closing-issue` in both skill trees. Keep that work in a separate change so
the initial rollout stays focused and the skill instructions can be tested
against the process that actually landed.
