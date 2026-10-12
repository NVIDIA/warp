---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: warp-closing-issue
description: Use when the user provides Warp commit SHA(s) and GitHub issue number(s) to assess, draft issue comments, post progress updates, or recommend whether issue threads should stay open or close.
license: Apache-2.0
---

# Warp Closing Issue

## Overview

Assess user-supplied Warp commits against user-supplied GitHub issues. Produce the
assessment and exact comment draft before any public write.

## Core rules

- Stay within the supplied commits and issues. Never search for replacement inputs.
- Search NVIDIA/warp issues for every confirmed potentially related problem before
  recommending tracking. Candidate issues are deduplication evidence, not replacement
  inputs or authorized targets.
- Apply the scope gate per intended commit/issue mapping. A clearly unrelated mapping
  gets `no public update`; all plausible mappings continue.
- Every plausible mapping gets a substantive draft comment. Existing comments,
  including terse or automated closure comments, never replace it.
- Do not post or close anything before the user sees the exact target, assessment,
  and draft and explicitly confirms the write.
- Use full 40-character SHAs as plain text in public comments.
- Keep commands, cache paths, temporary paths, and other local execution details out
  of public comments. Describe committed test coverage separately from local checks.
- Passing committed tests does not replace a reproduction based on the report when
  one is feasible.
- Every behavioral fix requires a bounded source search and executable checks for
  plausible repetitions of the failure mechanism.
- Put each public prose paragraph and list item on one physical line. Use line breaks
  only between Markdown blocks.
- Keep latent bugs outside the original issue private unless the user explicitly asks
  to file or mention them.

## Conditional references

- If a commit touches `changelog/*.md` or generated changelog content, read
  [changelog-evidence.md](references/changelog-evidence.md).
- If public API behavior matters, read
  [public-api-comments.md](references/public-api-comments.md).
- Read [commands.md](references/commands.md) when Git, GitHub, or local verification
  syntax would help. Reading it is mandatory before any GitHub write because it
  contains the safe write and read-back forms.

## Workflow

1. **Resolve scope.** Record the supplied commits, issues, and requested outcome. Use
   the newest supplied commit that contains the others as the assessed head. Inspect
   disjoint histories separately. Honor explicit commit/issue mappings; otherwise
   treat the complete commit set as candidate evidence for each issue.

2. **Gate each mapping.** Read the issue body and enough comments to identify the
   report, then inspect the associated commit messages, diffs, paths, and tests. A
   clearly unrelated mapping stops here with no public draft:

   ```markdown
   Assessment: no public update

   Scope mismatch:
   - Issue <#>: <reported behavior>
   - Supplied commit(s): <full SHA(s) and unrelated behavior>
   - Evidence: <why paths, behavior, tests, and intent do not overlap>
   - Likely input error: <probable SHA/issue mismatch>
   - Recommendation: verify the supplied SHA and issue number
   ```

   Continue every uncertain, partial, or plausible mapping. Stop the whole request at
   this gate only if every mapping is clearly unrelated.

3. **Gather and classify evidence.** For each plausible mapping, read the full issue
   discussion and inspect the commit diff, tests, docs, and relevant implementation.
   Treat commit messages as orientation, not proof. Classify each commit:

   | Type | Meaning |
   | --- | --- |
   | Behavioral fix | Changes the code path behind the report. |
   | Test-only | Adds confidence but cannot close the issue alone. |
   | Docs/changelog-only | Supports intent or release metadata but is not behavioral evidence. |
   | Follow-up | Completes or corrects earlier issue-linked work. |
   | Beyond scope | Changes related behavior outside the reported requirements. |

4. **Map requirements, API behavior, and tests.** Mark every requirement `addressed`,
   `partial`, or `missing`, citing commit, test, and reproduction evidence. For 10 or
   fewer new or modified named tests, describe coverage by file and test name. For
   more than 10, name each test file and summarize the behaviors, boundaries, error
   cases, and relevant platforms or devices covered without enumerating individual
   test names. State when no tests changed. If public API behavior is relevant,
   distinguish Python APIs from kernel-only builtins and check code, docs, tests, and
   changelog intent.

5. **Reproduce the report.** When the issue provides a reproducer, expected behavior,
   or clear boundary, run a temporary check on the assessed commit. Recreate the
   report directly, vary only relevant dimensions, and assert observable results.
   Prefer a standalone script for process, import, runtime, cache, environment, or
   packaging bugs. Use `uv run` and a unique `WARP_CACHE_PATH`; rebuild if native
   changes require it. Keep temporary artifacts out of the repository.

   Record each result privately as `passes`, `fails in scope`, `inconclusive`, or
   `not run` with a reason. An in-scope failure prevents closure. Do not substitute a
   committed test run for a feasible reproduction or expand into unrelated fuzzing.

6. **Audit potentially related paths.** For every behavioral fix, derive the failure
   mechanism and inspect direct siblings, callers, shared validators or registries,
   and obvious uses of the same faulty pattern. Run targeted temporary reproductions
   for the most plausible analogous paths. Source inspection alone is insufficient
   when an executable analogue exists. Stop after direct analogues and the
   highest-signal same-pattern paths; do not perform broad fuzzing.

   For every confirmed problem, read [commands.md](references/commands.md) and search
   open and closed NVIDIA/warp issues before recommending new tracking. Search from
   both the affected API names and the observable symptom, diagnostic, or failure
   mechanism; split searches as needed so every affected API is covered. Inspect the
   body and relevant comments of each plausible candidate rather than relying on its
   title. Judge coverage against the confirmed problem, not against every topic in the
   candidate: additional scope does not turn an otherwise complete match into a partial
   one. Classify existing issue coverage:
   - `Exact`: the candidate covers the complete confirmed problem. Record its number,
     title, and current state. An exact open match is already tracked; never recommend
     filing a duplicate.
   - `Partial`: the candidate omits some affected APIs, conditions, or behavior from
     the confirmed problem. Record the covered and uncovered scope before recommending
     how to track the gap.
   - `None found`: the searches completed and inspection found no plausible match.
   - `Inconclusive`: issue search or candidate inspection could not be completed. State
     why and never present this as `None found`.

   An exact closed match requires checking whether the confirmed behavior is a
   regression, incomplete coverage, or outside that issue's resolved scope before
   recommending follow-up. Searching and citing candidate issues does not authorize
   commenting on them, changing their state, or including them in a public draft.

   Add a private `Potentially Related Issues` section:
   - `Confirmed`: affected path or API, shared mechanism, executable result, why it is
     outside the original issue, existing issue coverage classification and evidence,
     and tracking recommendation.
   - `Checked, not affected`: analogous paths exercised successfully.
   - `Inconclusive`: remaining uncertainty and why execution did not resolve it.
   - `None confirmed`: bounded source search and executable checks performed, or why
     no analogous executable check was plausible.

   Out-of-scope findings do not block closure. Keep each finding and proposed remedy
   in this section, not `Spotted Improvements`. Do not file or publicize it without a
   separate explicit request. An original requirement that still fails belongs under
   behavioral probes as `fails in scope`.

7. **Choose status and action independently.**
   - Resolution status is `addressed` only when all reported behavior is fixed, test
     coverage is adequate or reasonably absent, and reproductions pass or could not
     run for a defensible reason. Otherwise it is `progress`.
   - Action `close` means post the comment, then close an open, addressed issue when
     issue-reporter verification permits. Action `comment-only` means post without a
     state change; use it for incomplete fixes, open issues left for verification,
     and all already-closed issues. Never reopen or re-close merely to attach the comment.

   Identify the issue reporter from `user.login` in the issue API (or
   `author.login` in `gh issue view`) and the author of each primary behavioral-fix
   commit from `author.login` in the commit API. Use
   [commands.md](references/commands.md) for the read forms. Do not use the
   authenticated GitHub user as a proxy for the issue reporter; `committer.login`
   may be a merge bot.
   Explicit user guidance or issue discussion can establish authorship when the
   commit API does not link an author account. Otherwise record identity as unknown.

   For an open, addressed issue, recommend closure when the issue reporter authored
   a primary behavioral-fix commit, unless the discussion explicitly requests
   further verification. Otherwise prefer leaving the issue open for reporter
   verification. Explain the identity evidence and its effect on the action.

8. **Draft before writing.** Repeat this block for every plausible mapping:

   ```markdown
   Assessment: <close | comment-only>

   Issue <#>: <title>
   - Resolution status: <addressed | progress>
   - GitHub state action: <close after comment | leave open | no state change because already closed>
   - Requested outcome: <...>
   - Commits: <primary full SHA(s)>; supporting: <full SHA(s) or none>
   - Reporter and fix author: <issue reporter login> / <primary behavioral-fix commit author login(s) or unknown>; <identity evidence and match status>
   - What changed: <behavior summary>
   - Public API behavior: <none | Python scope summary | kernel scope summary | unsupported cases>
   - Test coverage:
     - <10 or fewer named tests: files, test names, and behavior; more than 10: test files and grouped behavior summary; or no-tests detail>
   - Behavioral probes:
     - <required private probe summary: passes/fails in scope/inconclusive/not run, behavior checked, and issue relevance>
   - Beyond issue scope: <extra changes or none>
   - Reporter verification: <close after comment | leave open for reporter verification and why | no state change because already closed>
   - Recommendation: <action and why>

   Potentially Related Issues:
   - Confirmed: <affected path or API, shared mechanism, executable result, and why it is outside the original issue>
     - Existing issue coverage: <Exact | Partial | None found | Inconclusive> — <issue number, title, state, and overlap evidence; or completed search scope / reason search was incomplete>
     - Tracking recommendation: <already tracked; no duplicate | update existing coverage | consider a new issue | resolve search uncertainty first>
   - <Checked, not affected / Inconclusive / None confirmed, with the bounded source search, executable evidence, or why no executable analogue was plausible>

   Spotted Improvements:
   - <other actionable follow-up or none; do not repeat potentially related issues>

   Draft comment:
   <opening resolution or progress statement and issue-specific explanation; put each prose paragraph on one physical source line>

   ### Public API behavior

   <include only when relevant>

   ### Test coverage

   - <10 or fewer named tests: files, test names, and behavior; more than 10: test files and grouped behavior summary; or no-tests detail; put each list item on one physical source line>

   ### Behavioral verification

   <include only when public probe results materially clarify the outcome, risk, or requester-facing behavior>

   ### Follow-up to consider

   - <include only for actionable, issue-related future work; state when it does not block closure>

   Confirm whether to post this comment to <#>. If the GitHub state action is
   `close after comment`, also confirm whether to close <#> as completed. Do not ask
   for closure confirmation when the issue is already closed.
   ```

## Public comment contract

Start an addressed resolution, including for an already-closed issue, with:

```markdown
This is addressed by <full-sha>.
```

Start a progress update with:

```markdown
Progress update: <full-sha> landed <summary>.
```

Then explain the change in terms of the report and state whether the issue remains
open. Mention supporting metadata only when useful. Write in a factual maintainer
voice and do not restate the commit message.

Public sections use level-three headings and one blank line before their content:
- `### Public API behavior`: only when relevant and before test coverage.
- `### Test coverage`: always. For 10 or fewer new or modified named tests, use
  bullets naming files, tests, and changed behavior. For more than 10, name each test
  file and summarize the kinds of behavior, boundaries, error cases, and relevant
  platforms or devices covered without listing individual test names.
- `### Behavioral verification`: only when executed checks clarify the result, risk,
  or how the requester can verify it.
- `### Follow-up to consider`: only for actionable work within the original issue;
  say when it does not block closure.

Describe local verification naturally, for example, "a reproduction similar to the
reported issue" or "a targeted reproduction of the reported behavior." Never use
internal shorthand such as "issue-shaped." Do not expose commands or local paths.

## After confirmation

Read [commands.md](references/commands.md), post the reviewed body, and fetch the
comment by ID to verify an exact match before any state change. Close only an open
issue with action `close after comment` and explicit closure confirmation. Report the
comment ID, URL, final state, state reason, and close time when applicable. Stop on a
write failure; never guess a different target.

## Red flags

- A plausible mapping gets no draft because the issue is closed or already commented.
- A mismatched mapping continues, a valid mapping stops with another mismatch, or the
  assessment searches for replacement inputs.
- Resolution status, reporter verification, and GitHub state action are conflated.
- The authenticated GitHub user is mistaken for the issue reporter, the Git
  committer is mistaken for the fix author, or matching reporter and fix-author
  logins are overlooked when recommending verification.
- Evidence relies on the commit message, committed tests, or generated changelog
  alone when direct evidence is available.
- A feasible reproduction is skipped, or native changes are tested without considering
  a rebuild.
- A behavioral fix omits the bounded related-path audit, executable analogous checks,
  or `Potentially Related Issues`.
- A confirmed related problem lacks open-and-closed NVIDIA/warp issue searches, an
  existing-coverage classification, or a tracking recommendation; a title match or one
  empty query is treated as proof that no issue exists; or an exact open match is
  recommended for duplicate filing.
- A related finding appears in `Spotted Improvements` or the public draft without an
  explicit request.
- The public draft claims resolution for progress, exposes local execution details,
  presents a kernel-only API as Python, uses internal shorthand, or violates the
  required Markdown shape.
- A GitHub write occurs before confirmation or is not read back before state changes.

## Maintenance

When editing the Codex-side project skill, sync the mirrored Claude copy before
committing:

```bash
uv run tools/pre-commit-hooks/sync_skills.py --from codex
```
