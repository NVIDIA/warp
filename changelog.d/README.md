<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Changelog fragments

Warp uses [Towncrier](https://towncrier.readthedocs.io/) to collect release
notes. Pull requests and merge requests add small files here instead of editing
`CHANGELOG.md` directly.

Add a fragment when a change affects Warp users. Internal maintenance does not
need one. Reviewers decide whether a fragment is needed and whether it explains
the user impact clearly.

## Choose an identifier

Use the GitHub issue number when the work has an issue:

```text
1708.fixed.md
1708.documentation.md
```

The number identifies a GitHub issue, not a GitHub pull request or GitLab merge
request. Towncrier adds the GitHub issue link when it renders the fragment, so do
not put that link in the fragment text. Other useful links, such as migration
guides, are allowed.

When there is no GitHub issue, use a readable orphan identifier:

```text
+fix-cuda-graph-capture.fixed.md
```

The leading `+` tells Towncrier that the entry has no issue link. Prefer a short
description over an opaque hash. A normalized branch description is often a
good starting point, but it is not required. If you use one, drop the username
prefix and replace runs of non-alphanumeric characters with `-`.

## Choose a category

Use one of these suffixes. They render in this order:

1. `added`
2. `removed`
3. `deprecated`
4. `changed`
5. `fixed`
6. `documentation`

Each fragment becomes one bullet. Do not start its content with `-`. Write in
imperative present tense and describe what changes for users. Include migration
guidance in `changed`, `deprecated`, and `removed` entries when users need to
update their code.

Use a numeric counter for several entries with the same identifier and category:

```text
1708.fixed.md
1708.fixed.1.md
```

If a change resolves several GitHub issues, create one fragment for each issue
with identical text. Towncrier combines them into one bullet with multiple issue
links.

## Preview pending entries

Render the pending fragments without modifying the changelog:

```console
uvx --from towncrier==25.8.0 towncrier build --draft \
  --version X.Y.Z --date YYYY-MM-DD
```

## Build a release

Build release notes on `release-X.Y` after the release audit has selected every
change that will ship. Preview with the real version and date, edit the fragments
until the result reads well, then build:

```console
uvx --from towncrier==25.8.0 towncrier build --draft \
  --version X.Y.Z --date YYYY-MM-DD
uvx --from towncrier==25.8.0 towncrier build --yes \
  --version X.Y.Z --date YYYY-MM-DD
```

Update the comparison links at the bottom of `CHANGELOG.md`. Commit the generated
release section, consumed fragment deletions, and link updates together in one
dedicated changelog build commit.

After the release, create a branch from the current `main` and cherry-pick that
exact build commit. Open a small changelog-only pull request or merge request.
The cherry-pick removes the fragments that shipped while leaving newer fragments
on `main` pending for the next release.
