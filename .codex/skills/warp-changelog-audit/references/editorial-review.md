<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Editorial Review

Use for fragment proposals and continued review of a newly generated release.
Keep the requested artifact, release target, date, and current review stage in
view. Carry accepted decisions forward across turns.

## Proposal format

Each review reply contains the following, in order:

| Part | Content |
| --- | --- |
| Current | Verbatim entries for a rewrite or merge; current positions for ordering. |
| Proposed | Exact replacement text or destination. |
| Reason | User benefit and why the change preserves the relevant conditions. |
| Context | A clickable full-file preview and a diff, with their status stated explicitly. |

Show only the candidate entries or moves in the reply. Put the surrounding
changelog in the preview file. When an existing preview shows only current text,
label it as current. If temporary preview writes are authorized, create and link
a separate proposed full file and diff. If the review permits only a response,
link the current file, show the exact candidate replacement, and state that the
proposal has not been staged. The read-only reply still contains only the
candidate, its reason, and the sorting approach; the surrounding section stays
in the linked file. Verify linked files exist and describe what they contain.

## Review sequence

First settle wording, categories, and useful consolidation. Recheck the full
section for overlaps that became clearer during rewriting. Preserve distinct
failures when merging would create a list of unrelated changes.

When the user requests incremental review, present one merge at a time. For a
long subsection, settle the opening high-impact entries, review the middle in
small workflow groups, then inspect the tail and compare whole-group placement.
Show a few consequential moves per batch and retain reasonable relative order.
Use the sorting rubric to keep importance ahead of topic clustering.

## Applying decisions

Track each proposal as pending, approved, or applied, with its exact text or
move. Apply the approved proposal before presenting the next one. Record the
result and refresh the preview; keep unrelated pending proposals separate.
Mechanical corrections can be batched and summarized by count.

Before regenerating a preview, compare it with the last generated snapshot.
Preserve and incorporate user edits. Before applying a proposal, verify that
the real file still matches the reviewed baseline. Check issue identities,
canonical subsection order, line width, stability markers, and older-release
integrity. Derive entry counts from the current file so approved merges can
change the count without breaking later checks.

## Decision scope

Honor explicit exclusions and wording choices throughout the review. Record
release-specific decisions as such; they do not establish permanent rules about
an integration, feature area, or ordering between subsystems. Use previous
releases as style evidence, and explain decisions through the behavior changed.

After fragment consumption, stage edits to the new release section directly.
Preserve the build commit and its consumed-path evidence. An empty Towncrier
draft confirms fragment consumption; it does not authorize rebuilding the
section over reviewed prose. Keep temporary review files outside commits.
