<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GitHub commands

Prefer the GitHub app/MCP connector when it provides the needed issue/comment data.
Use `gh` where it is simpler or connector coverage is missing; this machine has `gh`
installed and authenticated.

## GitHub Reads

For JSON reads, include the `comments` field to retrieve the discussion. Do not
combine `--comments` with `--json`; `gh issue view` rejects that combination.

```bash
gh issue view <issue> --repo NVIDIA/warp \
  --json number,title,state,stateReason,author,body,comments,labels,createdAt,updatedAt,closedAt,url
gh api repos/NVIDIA/warp/commits/<sha> \
  --jq '{sha: .sha, author_login: .author.login, html_url: .html_url}'
```

For each confirmed potentially related problem, search both open and closed issues
from independent query angles, including affected API names and the observed symptom,
diagnostic, or failure mechanism. Split API searches when necessary so every affected
path is covered:

```bash
gh search issues "<affected API terms>" --repo NVIDIA/warp --state open \
  --limit 100 --json number,title,state,url,body,updatedAt
gh search issues "<symptom, diagnostic, or mechanism>" --repo NVIDIA/warp --state open \
  --limit 100 --json number,title,state,url,body,updatedAt
gh search issues "<affected API terms>" --repo NVIDIA/warp --state closed \
  --limit 100 --json number,title,state,url,body,updatedAt
gh search issues "<symptom, diagnostic, or mechanism>" --repo NVIDIA/warp --state closed \
  --limit 100 --json number,title,state,url,body,updatedAt
```

Inspect every plausible candidate's body and discussion before classifying its
coverage. Search results and titles alone are not sufficient:

```bash
gh issue view <candidate> --repo NVIDIA/warp \
  --json number,title,state,stateReason,body,comments,labels,createdAt,updatedAt,closedAt,url
```

These are deduplication reads only. A candidate issue is not an assessed target, and
the search does not authorize a comment, state change, or new issue filing.

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

Use `gh issue comment --body-file` for new issue comments with multiline bodies:

```bash
set -euo pipefail

body_file=$(mktemp)
cat > "$body_file" <<'EOF'
<exact reviewed comment body>
EOF

comment_url=$(gh issue comment <issue> --repo NVIDIA/warp --body-file "$body_file")
comment_id=${comment_url##*issuecomment-}

case "$comment_id" in
  ""|*[!0-9]*)
    printf 'Could not resolve comment ID from %s\n' "$comment_url" >&2
    exit 1
    ;;
esac
```

Use `gh api --input` with JSON for editing existing comments or other endpoints
that do not support `--body-file`:

```bash
set -euo pipefail

body_file=$(mktemp)
json_file=$(mktemp)
comment_id=<comment-id>

cat > "$body_file" <<'EOF'
<exact reviewed comment body>
EOF

jq -n --rawfile body "$body_file" '{body:$body}' > "$json_file"

gh api -X PATCH repos/NVIDIA/warp/issues/comments/"$comment_id" \
  --input "$json_file"
```

Compare the public comment body exactly with the reviewed file before closing:

```bash
set -euo pipefail

if ! gh api repos/NVIDIA/warp/issues/comments/"$comment_id" \
  | jq --exit-status --rawfile expected "$body_file" \
      'select(.body == $expected) | {id,html_url,body}'; then
  printf 'Comment body mismatch or read-back failed\n' >&2
  exit 1
fi
```

Never use `gh api -f body=@file` or `gh api --raw-field body=@file` for
multiline comment bodies; those forms can send `@file` literally.

Before closure, fetch the issue again. Close only if it is still open and closure was
explicitly confirmed:

```bash
set -euo pipefail

current_state=$(gh issue view <issue> --repo NVIDIA/warp --json state --jq .state)
if [ "$current_state" != "OPEN" ]; then
  printf 'Issue is no longer open; skipping close\n' >&2
  exit 1
fi
if ! gh issue close <issue> --repo NVIDIA/warp --reason completed; then
  printf 'Issue close failed\n' >&2
  exit 1
fi
gh issue view <issue> --repo NVIDIA/warp \
  --json number,state,stateReason,closedAt,url,comments
```
