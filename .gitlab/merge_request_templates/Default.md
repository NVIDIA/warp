<!-- Bringing in a GitHub pull request? Choose the github-pr template instead. -->
<!-- Labels: apply every existing project label whose description fits this MR. Do not create new labels. -->
<!-- If there is a related GitHub issue, end the MR title with [GH-123]. -->

## Context

<!-- Name the concrete problem this change solves (what broke, was missing, or
was slow) and why it matters. Link related GitHub issues, NVBugs, or
downstream work. -->

## What does this change?

<!-- Explain how the change works and call out any non-obvious choices. Do not
retell the diff. Include a short usage example for a new feature. -->

## Validation

### Bug reproducer

<!-- For a bug fix, include a minimal reproducer and show the failure before
the fix and success after it. Otherwise write "N/A: not a bug fix". -->

### Related-code audit

<!-- For a bug fix, check for the same problem in other dtypes, devices, and
backends, and in other functions that share the faulty code. For a feature,
check related APIs and backends for consistency. List problems you find but
do not fix here under Deferred work. Write "N/A: <reason>" if there is nothing
similar to check. -->

- Audit scope:
- Findings:

### Other validation

<!-- List the checks you ran and the hardware used. For performance work, give
before/after results and say whether you added ASV coverage or used a focused
benchmark. Note anything you did not test and why. Do not paste raw logs. -->

## Deferred work

<!-- List work you are intentionally leaving out of this MR, such as review
suggestions you agreed to defer or problems found by the related-code audit.
Link the issue for any item that is already tracked. Update this list during
review whenever you defer something or pick it back up. Write "None" if
nothing is deferred. -->

## Checklist

<!-- Use [x] for completed items. Use [~] for items that do not apply, and end
the item with "N/A: <reason>". Leave [ ] for incomplete items and explain them
under Validation. -->

- [ ] Every change in this MR is part of addressing the problem described under Context.
- [ ] I verified that new or existing tests cover the behavior changed by this MR, including error paths and edge cases.
- [ ] I considered how any changes to code generation, native code, or kernel launches could affect runtime, occupancy, and JIT compile time, and measured any likely effects, including on GPU architectures other than the one I developed on.
- [ ] I updated any affected documentation or design documents.
- [ ] I added a [changelog fragment](https://github.com/NVIDIA/warp/blob/main/changelog/README.md) if this change affects users, and used a `+` prefix only for changes that do not require a GitHub issue.
