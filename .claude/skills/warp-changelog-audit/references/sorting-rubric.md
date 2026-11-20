<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Impact Sorting Rubric

Use when sorting the generated release section. Preserve canonical subsection
order, then rank entries by practical user impact within each subsection.

## Assess the consequence

For each entry, identify:

1. **Workflow and reach:** what users can now do, or which existing work resumes.
2. **Severity:** migration work, incorrect results, crashes, blocked execution,
   a performance regression, or a smaller convenience.
3. **Likelihood:** ordinary use or a narrow input, configuration, or backend path.
4. **Benefit and limits:** expected effect on real work, including restrictions
   that keep a newly available feature from delivering its full potential.

A new public name, dtype protocol, parameter, or exception class does not assign
an impact tier. An optional NumPy dtype convenience or diagnostic exception can
remain near the end even though its API is new. A new compute capability,
platform, or reusable workflow can deserve higher placement because it removes
a practical limitation.

`**Experimental**` describes stability, not importance. Rank an experimental
feature by what it enables and its current limits. Treat a breaking change's
actual migration requirement as significant; the word "exception" alone says
nothing about the importance of an Added exception type.

## Working impact bands

| Band | Typical user impact |
| --- | --- |
| High | Required migration; broadly encountered wrong results, crashes, or execution failures; significant regressions; substantial new workflows; platform support that enables previously blocked users. |
| Middle | Correctness and performance fixes in specialized workflows; useful extensions of existing operations; substantive examples and guides. |
| Low | Optional helper and diagnostic APIs; small declaration/setup conveniences; narrow corner cases; minor diagnostic wording and license notices. |

These are consequence-based bands, not permanent rankings of subsystems.
An example's computation fix belongs in Fixed but can sit below library fixes
with broader reach. A new example or guide belongs in Documentation and is
ranked there by its usefulness.

Benchmark numbers support an impact assessment; their presence does not set
priority. A hot-path regression can outrank a measured but minor setup saving.
Distinguish execution-time benefits from Python declaration or first-call costs,
and keep application-specific results scoped to the measured workload.

## Ordering comparable entries

Within an impact band, use topic clustering, then clarity and specificity as
tie-breakers. Preserve an existing reasonable order when no consequential
difference is apparent. Keep a higher-impact entry visible even when it breaks
a topic cluster.

For a long subsection, follow `editorial-review.md`: settle the opening, review
small workflow groups and the tail, then compare group placement. Review a few
consequential moves per batch rather than requiring a total ranking of dozens
of entries at once. Retain the user's accepted ordering and explicit exclusions.
