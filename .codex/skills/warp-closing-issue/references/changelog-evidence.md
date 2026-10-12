<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Changelog evidence

Read this reference only when a supplied commit touches `changelog/*.md`, generated
`CHANGELOG.md`, or related release metadata.

- Treat `changelog/*.md` fragments as the editable source of pending intent.
- Treat generated `CHANGELOG.md` sections as release history or build metadata, never
  as behavioral fix evidence by themselves.
- Read `changelog/README.md` and validate each touched fragment's identifier,
  category, optional counter, and content.
- A numeric fragment filename identifies a GitHub issue, and Towncrier generates the
  link. The fragment text should not repeat the issue reference.
- Compare numeric fragment identifiers with the supplied issue. Inspect sibling
  fragments with the same identifier at the assessed head so complementary,
  duplicate, and counter-based entries are not mistaken for separate fixes.
- If Added, Fixed, and Changed fragments describe iterations of one unreleased
  feature, flag them for changelog audit. Do not consolidate them in this workflow.
- When combined rendering matters, run the pinned Towncrier draft in the assessed
  worktree and map the rendered bullet back to its source fragments. The draft is
  read-only; never edit generated `CHANGELOG.md`.
