<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Optional Commands

Prefer the GitHub app/MCP connector when it provides the needed issue/comment data.
Use `gh` where it is simpler or connector coverage is missing; this machine has `gh`
installed and authenticated.

## GitHub Reads

```bash
gh issue view <issue> --repo NVIDIA/warp --comments \
  --json number,title,state,stateReason,body,comments,labels,createdAt,updatedAt,closedAt,url
gh api repos/NVIDIA/warp/commits/<sha> --jq '{sha: .sha, html_url: .html_url}'
```

## Local Commit Reads

```bash
git merge-base --is-ancestor <sha> upstream/main
git show --stat --find-renames <sha>
git show --format=fuller --no-patch <sha>
git show --find-renames <sha>
```

## Local Warp Verification

Use a unique cache path locally. Public issue comments should describe test coverage
changes from the commits, not local verification commands.

```bash
WARP_CACHE_PATH=/tmp/<unique-warp-cache> \
uv run <focused-test-command>
```

If `warp/native/` changed and the built library is stale or missing, rebuild first:

```bash
uv run build_lib.py
```

Use `uv run build_lib.py --quick` only when the CUDA driver/toolkit check supports it.

## GitHub Writes

Post comments through the GitHub app/MCP connector when available, or `gh` if needed.
For closure:

```bash
gh issue close <issue> --repo NVIDIA/warp --reason completed
gh issue view <issue> --repo NVIDIA/warp \
  --json number,state,stateReason,closedAt,url,comments
```
