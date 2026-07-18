## Description

<!-- What does this MR change and why? Be specific:
     - Summarize the problem or motivation
     - Describe the approach taken and key design decisions
     - Note any alternatives considered and why they were rejected
     Reference GitHub issues with GH-<NUM>. Use "closes GH-<NUM>" if applicable. -->

## Changes

<!-- List the key changes, grouped logically. Example:

- **warp/_src/codegen.py**: Refactored kernel compilation to support X
- **warp/tests/test_codegen.py**: Added tests for new compilation path
- **docs/**: Updated API reference for new parameter

This section helps reviewers navigate the diff efficiently. -->

## Checklist

<!-- See the Contributing Guidelines for general guidance:
     https://nvidia.github.io/warp/latest/project/contribution_guide.html -->

- [ ] New or existing tests cover these changes.
- [ ] The documentation is up to date with these changes.
- [ ] [CHANGELOG.md](CHANGELOG.md) is updated for any user-facing changes under the `Unreleased` section.

## Validation summary

<!--
Explain what was verified and why it is sufficient for review. Write a short
step-by-step validation narrative, not a command dump. Prefer test names plus
behavior summaries.

For test-driven changes, include red/green evidence when applicable, e.g.:
- Verified the new test fails on the target branch without this change.
- Verified the new test passes on this branch with the fix.

If testing was not run, say so and explain the risk or blocker. Include
commands only when they help reproduce the validation.
-->

## Bug fix

<!-- If this is a bug fix, provide a minimal code example that reproduces the
     issue WITHOUT this MR applied. Delete this section if not applicable. -->

```python
import warp as wp
# Code that demonstrates the bug
```

## New feature / enhancement

<!-- If this is a new feature or enhancement, provide a code example showing
     what this MR enables. Delete this section if not applicable. -->

```python
import warp as wp
# Code that demonstrates the new capability
```
