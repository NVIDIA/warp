<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Report Destination Rules

Load this reference during Phase 1 before proposing an output destination.

## Discover existing gists

Run `gh --version && gh auth status`. If either fails, use a local markdown
file. Otherwise, compute the stable description:

```text
Warp <version-string> <Pre-Release|Release Candidate> Report
```

Run `gh gist list --limit 1000` and collect rows whose description exactly
matches. The filename and description contain no date so later audits revise
the same gist while its git history preserves prior versions.

## Ask for confirmation

Lead with one of these lines:

- Pre-release: `Generating **pre-release report** for Warp **<version>**. Base **<base-ref>** → Head **<head-ref>**. **<N>** commits in range.`
- RC: `Generating **release-candidate report** for Warp **<version>**. Base **<base-ref>** → Head **<head-ref>** (release branch cut). **<N>** commits in range.`

Append the applicable options:

- `gh` unavailable: local markdown file; ask the user to confirm the refs.
- No match: new secret gist (default) or local file.
- One match: revise that gist (default), create a new secret gist, or use a
  local file.
- Multiple matches: list each URL and update time, then offer revision by list
  number, a new secret gist, or a local file.

Wait for explicit confirmation of the refs and destination. Do not begin Phase
2 before confirmation. Record exactly one destination token: `local`,
`new-gist`, or `revise-gist:<id>`. Resolve letter replies against the options
shown in the current prompt; letters have no global meaning.

## File the report

Use these stable names:

- Local: `warp-<version-string>-<prerelease|rc>-report-<today>.md` at the repo
  root.
- Gist: `warp-<version-string>-<prerelease|rc>-report.md`.
- Gist description: `Warp <version-string> <Pre-Release|Release Candidate>
  Report`.

For `local`, write the report at the repo root. For `new-gist`, write a temporary
file and run:

```bash
gh gist create --desc "<stable-desc>" /tmp/<gist-filename>
```

For `revise-gist:<id>`, write a temporary file and run:

```bash
gh gist edit <id> --filename <gist-filename> /tmp/<gist-filename>
```

The `--filename` argument selects the file inside the gist. Do not pass
`--desc`, because the stable description is the matching key for later runs.
Delete temporary files after either gist operation. Never pass `--public` and
never file to a destination the user did not choose.

Return the local path or gist URL plus headline counts. For a revised gist,
also say that the edit was in place and prior versions remain in gist history.
