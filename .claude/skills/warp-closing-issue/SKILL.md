---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: warp-closing-issue
description: Use when the user provides Warp commit SHA(s) and GitHub issue number(s) to assess, draft issue comments, post progress updates, or recommend whether issue threads should stay open or close.
license: Apache-2.0
---

# Warp Closing Issue

## Overview

Assess user-supplied Warp commits against user-supplied GitHub issues. Produce a
scoped assessment and draft comment first; public GitHub writes require explicit
confirmation after the user sees the exact target issues and comment body.

## Hard Rules

- Stay within the user-supplied commit SHA(s) and issue number(s). Do not search for
  extra commits or issues; ask the user for more SHAs/issues if scope is incomplete.
- Read the issue body and comments, not just the title.
- Treat commit messages as orientation, not proof. Inspect diffs and changed files.
- Commenting is not always closure. Supported actions are `close`, `comment-only`,
  `keep open`, and `no public update`.
- Never post comments or close issues before showing the assessment and exact draft.
- Public comments use full 40-character SHAs as plain text for GitHub auto-linking.
- Keep local execution noise out of public comments: no `WARP_CACHE_PATH`,
  `/tmp/...`, local worktree paths, or shell setup.
- Do not use a local test command as the public test note. Public comments describe
  test coverage changes in the commit set: new tests, modified tests, or no tests.
- Do not treat passing committed tests as sufficient behavioral verification when
  the issue includes a runnable repro or clear expected behavior. When feasible,
  write and run small temporary probes inspired by the issue.
- Require behavioral probe summaries in the private assessment. Public comments may
  mention probes only when they clarify the recommendation, remaining risk, or
  requester-facing behavior.

Optional command snippets live in [commands.md](references/commands.md). Prefer the
GitHub app/MCP connector when it fits; `gh` is installed and authenticated here and is
fine for gaps or simple issue operations.

## Checklist

1. **Resolve scope.** Record supplied commits, issues, and requested outcome, if any:
   closure assessment, progress update, comment only, or unspecified.

2. **Gather evidence.** For each issue, extract author, author association, reported
   symptoms, reproducers, expected behavior, follow-up comments, maintainer asks, and
   current state. For each commit, inspect message, diff, tests, docs/CHANGELOG
   changes, and touched areas such as `warp/native/`.

3. **Classify commits.**

   | Type | Meaning |
   | --- | --- |
   | Behavioral fix | Changes the code path behind the issue. |
   | Test-only | Adds confidence, but cannot close by itself. |
   | Docs/CHANGELOG-only | Supporting/progress metadata, not fix evidence. |
   | Follow-up | Completes or corrects earlier issue-linked work. |
   | Beyond scope | Related cleanup or broader behavior worth surfacing. |

4. **Map requirements.** For each issue requirement, state commit evidence, test
   coverage evidence, behavioral probe evidence if available, and status:
   addressed, partial, or missing.

5. **Review tests.** Inspect the supplied commits for test changes. Use unordered
   bullets in the assessment/comment:
   - New tests: file path, test function/class names, and what each case checks.
   - Modified tests: file path, test names, and what behavior or expectation changed.
   - No tests: state that no test changes were included and recommend whether that was
     reasonable or a potential review oversight.

   You may still run committed tests when useful, but prefer probes that add issue
   specific signal beyond "the merged tests pass." Follow Warp policy locally:
   unique `WARP_CACHE_PATH`, `uv run`, and rebuild native
   libraries when `warp/native/` changes require it. Do not put local verification
   commands in the public issue comment.

6. **Probe issue-shaped behavior.** When the issue has a repro, expected behavior,
   or clear boundary conditions, create one or more temporary scripts that exercise
   the reported behavior on the supplied commit/worktree. These are transient
   working artifacts; do not add them to the repo unless the user explicitly asks.

   Prefer probes that:
   - Recreate the original repro as directly as possible.
   - Vary only issue-relevant dimensions likely to expose blind spots.
   - Assert observable behavior, not just absence of a crash.
   - Run outside the test suite when the issue is about script, import, process,
     runtime, environment, cache, or packaging context.
   - Use `uv run` and a unique `WARP_CACHE_PATH` for Warp commands.

   Avoid probes that:
   - Merely rerun a committed test without adding issue-specific signal.
   - Expand into broad fuzzing or unrelated API compatibility.
   - Depend on timing or local environment details unless the issue is
     environment-specific.

   Classify probe results in the private assessment:
   - `passes`: supports closure or progress assessment.
   - `fails in scope`: blocks closure or changes recommendation to `comment-only`
     / `keep open`.
   - `inconclusive`: mention as residual risk, but do not overstate it.
   - `not run`: explain why, such as unavailable hardware, excessive cost, or
     insufficient repro detail.

7. **Decide action.**
   - `close`: every reported symptom and expected behavior is addressed, relevant
     comments are covered, test coverage is adequate or the lack of tests is
     reasonable for the change, and behavioral probes pass or were not feasible for
     a defensible reason.
   - `comment-only`: supplied commits are relevant progress, but the issue should remain
     open.
   - `keep open`: gaps remain and a public comment would not add value.
   - `no public update`: commits are peripheral, speculative, or already covered.

   Include a requester-verification recommendation. If the issue author appears
   external to NVIDIA, prefer a resolution/progress comment that leaves the issue open
   so they can verify. If the issue author matches the current requesting user, closure
   is appropriate once the requirements are addressed; verify that identity from local
   user guidance, GitHub authenticated user data, or explicit user input rather than
   hardcoding a username.

8. **Draft before writing.** Output:

   ```markdown
   Assessment: <close | comment-only | keep open | no public update>

   Issue <#>: <title>
   - Requested outcome: <...>
   - Commits: <primary full SHA(s)>; supporting: <full SHA(s) or none>
   - What changed: <behavior summary>
   - Test coverage:
     - <new/modified/no tests detail>
   - Behavioral probes:
     - <required private probe summary: passes/fails in scope/inconclusive/not run, behavior checked, and issue relevance>
   - Beyond issue scope: <extra changes or none>
   - Requester verification: <close now | leave open for requester verification and why>
   - Recommendation: <action and why>

   Spotted Improvements:
   - <actionable follow-up or none>

   Draft comment:
   <exact public comment; include behavioral probes only when useful for public clarity>

   Confirm whether to post this comment to <#>. If closure is recommended, also
   confirm whether to close <#> as completed.
   ```

## Comment Shape

For closure, start with:

```markdown
This is addressed by <full-sha>.
```

For progress/comment-only updates, start with:

```markdown
Progress update: <full-sha> landed <summary>.
```

Then explain what changed in issue terms, mention supporting commits if useful, and
state whether the issue should remain open. Mention docs/CHANGELOG-only commits only
as supporting metadata. Include `Spotted Improvements` only for actionable follow-up
work. If leaving an externally filed issue open for requester verification, say that
directly. Drop empty sections.

Include test coverage as unordered bullets. Name changed test files and test functions.
If no tests changed, say whether that is reasonable for the commit type or a potential
oversight.

Behavioral probe summaries are required in the private assessment. In the public
comment, mention probes only when they clarify the outcome, explain residual risk, or
help the requester verify the fix. When included publicly, summarize checked behavior
and result without local commands, cache paths, temp paths, or worktree paths.

Write in a factual maintainer voice, usually third person: "The change updates...",
"Coverage was added...". Keep wording direct and precise, adding detail when it
clarifies impact, scope, test coverage, remaining gaps, beyond-scope work, or follow-up.
Do not restate the commit message mechanically; use the comment to augment the commit
with issue-specific context.

## Write Actions

After explicit confirmation, post the issue-specific comment. Close only issues that
were both recommended for closure and explicitly confirmed for closure. Verify final
GitHub state and report comment IDs, URLs, state, state reason, and close time.

If any write fails, stop and report the exact failure. Do not retry against a different
issue by guess.

## Red Flags

- About to call a GitHub write API before showing the draft.
- The draft implies completion for a progress update.
- The issue has multiple requirements and the commits cover only one.
- The commit touches `warp/native/` and rebuild state was not considered.
- The only evidence is the commit message.
- The issue has a runnable repro but the assessment only reruns committed tests.
- The private assessment omits behavioral probe results or a reason probes were not run.
- The draft contains `/tmp/`, `WARP_CACHE_PATH`, or local paths.

## Maintenance

When editing the Codex-side project skill, sync the mirrored Claude copy before
committing:

```bash
uv run tools/pre-commit-hooks/sync_skills.py --from codex
```
