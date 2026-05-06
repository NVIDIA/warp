# Contributor attribution rules

The Acknowledgments section thanks **non-NVIDIA contributors** who have made
non-trivial contributions to the release. The script
`scripts/list_contributors.py` classifies each contributor as `nvidia` (no
acknowledgment) or `external` (acknowledged), using two positive-internal
signals: NVIDIA email domains and NVIDIA GitHub org membership.

## Classification rules (applied in order, first match wins)

1. **`nvidia` (definitively internal)** — any one of:
   - Author email matches `*@nvidia.com` or `*@exchange.nvidia.com`. Free,
     deterministic, no API call.
   - Author's resolved GitHub login is a member of the NVIDIA GitHub
     organization. The script calls `gh api /orgs/NVIDIA/members/<login>` and
     treats a 204 No Content response as a positive match. This catches staff
     who commit from personal or `users.noreply.github.com` addresses, which
     is common because GitHub defaults org membership to private and most
     staff don't toggle it public.

   These contributors are NOT listed in the Acknowledgments section.

2. **`external`** — anyone who matches neither check above.

   These contributors ARE listed in the Acknowledgments section, with the
   contribution drawn from the matching CHANGELOG bullet.

## Caveat: caller must be an NVIDIA org member for the membership check

The org-membership endpoint `/orgs/<org>/members/<login>` returns 204 only
when the authenticated caller is also an org member. Outside callers see 404
for everyone, including actual private-membership members. The skill is for
the release manager (an NVIDIA org member by definition), so the
caller-is-member assumption holds. If someone outside the org runs the skill,
private-membership NVIDIA staff committing from non-NVIDIA addresses will
fall through to `external` and surface in the rendered acknowledgments. The
release manager should review the acks list before publishing in any case.

The endpoint returns 204 for both public and private members, so toggling
membership public/private does not affect classification when the caller is
in the org.

## Trivial-contribution filter

Not every external commit warrants an Acknowledgments line. Filter out:

- **Pure typo fixes** in docs or comments (one-line changes to text).
- **Single-character formatting fixes** that the original author pre-approved
  (`black` reformatting from a CI bot, etc.).
- **Trivial CI tweaks** with no user-visible effect.

Use the per-contributor `commit_count` and the file paths touched (in `commits[]`
records) to make this judgment. A contributor with `commit_count == 1` whose only
commit modified a single file matching `*.md` and changed `<3` lines is a typo fix.
A contributor with `commit_count >= 2`, OR any commit touching `warp/_src/` or
`warp/native/`, is non-trivial.

The script does NOT auto-filter. Phase 5d of the skill applies the trivial filter
during render, using the script's per-contributor data.

## GitHub login resolution

The script tries two strategies in order:

1. **Email pattern** — for emails matching
   `^\d+\+([^@]+)@users\.noreply\.github\.com$`, extract group 1 as the
   GitHub login. Free, deterministic, no API call.
2. **`gh api` per sample commit** — for unresolved authors, run
   `gh api /repos/NVIDIA/warp/commits/<sha> --jq .author.login` once per
   unique author (using the author's first commit in the range as the
   sample). Network-bound; cached per author within the script run.

If both strategies fail (no `gh`, no auth, no GitHub user record for the
commit), the contributor record carries `gh_login: null`. The org-membership
check is skipped (it requires a login), and the contributor is classified
as `external` based on email alone. Such records still surface in the
rendered acknowledgments but with no `@`-handle to mention; the release
manager should spot-check or drop them by hand.

## Cherry-pick filtering and CHANGELOG-section attribution

Two robustness filters are applied automatically by the script. Both are
defensive against the same failure mode: a contributor's work being shipped
in a *prior* release (typically a bugfix release branch) and re-applied onto
the current release branch with a different SHA, which would otherwise look
like a fresh contribution.

1. **Cherry-pick filter (default on).** The script enumerates commits with
   `git log --no-merges --reverse --cherry-pick --right-only <base>...<head>`
   so commits whose patch-id has an equivalent in `<base>` are dropped. This
   catches the common case where the same fix landed in the previous bugfix
   release via cherry-pick (different SHA, same patch). Pass
   `--include-cherry-picked` to the script to disable this filter when you
   need to see all raw commits in the range.

2. **CHANGELOG-section attribution (on when `--changelog` is passed).** The
   script parses the CHANGELOG file at the head ref into versioned sections
   (`## [Unreleased]`, `## [1.13.0]`, `## [1.12.1]`, ...) and maps every
   `GH-NNNN` reference to the section(s) it appears in. For each contributor,
   the script sums the GH refs from:
   - the commit subject and body, and
   - any `[GH-NNNN]` refs the commit *added* to `CHANGELOG.md` (so we still
     get attribution when the commit message itself omits the issue ref).

   When **all** of those refs land only in non-target sections (i.e., none
   appear under the target version or `Unreleased`), the contributor is tagged
   `already_shipped: true`. A contributor with refs split across the target
   section and a prior section stays `already_shipped: false` (they have at
   least some current-release work to thank). Phase 5d skips already-shipped
   contributors from the rendered acknowledgments and surfaces them in the
   chat summary.

The `--target-version` flag is required when `--changelog` is used; together
with the implicit `Unreleased` label, it defines what counts as "this
release's CHANGELOG section."

## Commit summarization for ack lines

For each external contributor, the script provides `pr_summary`: a list
of `{pr, subject, sha}` records. **Note**: Warp's primary commit flow is GitLab
merge requests, which do *not* append `(#NNNN)` PR trailers to squash subjects.
Expect `pr` to be `None` for most commits. The trailer is extracted opportunistically
because some commits (cherry-picks from GitHub PRs, occasional direct GitHub merges)
do carry it.

To find the right GitHub issue / PR ref for a contribution, cross-reference the
`subject` and the file paths in `commits[].files` against `CHANGELOG.md` bullets:

1. Take a distinctive substring from the contributor's commit subject.
2. Search the CHANGELOG `[Unreleased]` (or `[X.Y.Z]`) section for a bullet that
   matches the subject's topic.
3. The `GH-NNNN` ref in the matched bullet is the issue / PR you want for the ack
   line. Render it as `#NNNN` in the release notes (per `style-rules.md`).

If no CHANGELOG bullet matches, the contribution may be unlisted in the changelog
(rare; usually means a small fix that the release manager folded into someone
else's bullet). In that case render the ack with the commit subject and no `#NNNN`,
e.g. `- @username for fixing deprecation warnings breaking pickle.`

Examples (from past release notes — these used the standard GitHub `(#NNNN)` form):
- `- @lenroe for reducing CPU kernel launch overhead by caching internal dispatch
  structures and optimizing type comparisons (#1160).`
- `- @StafaH for contributing to the hardware-accelerated texture feature through
  review and helpful contributions (#1169).`

The script does not draft these sentences — that requires reading the commit /
PR context. The skill consumes the raw `pr_summary` records and writes the prose
during Phase 5d.
