<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Authorization checkpoint

Use this mandatory checkpoint after stage 1 finds at least one candidate pattern
and before any measured or experimental work. Its purpose is to make an
automatically offered Warp assessment opt-in once the likely scope and cost are
visible.

## What is allowed before the checkpoint

Read source, documentation, configuration and existing logs. Identify candidate
areas, derive the proposed contract and check the cheap disqualifiers. Do not
run profilers or workloads, install or build dependencies, create environments
or harnesses, acquire a GPU lock, prototype, benchmark, or create the report
directory.

## What to report

Send one concise early-findings message containing:

1. **Framing:** Warp is only a candidate and no adoption recommendation has
   been made.
2. **Observed facts:** candidate areas with `path:line`, the pattern each
   matches, existing device/backend facts, and which cheap gates were checked.
3. **Hypotheses and unknowns:** inferred objective and execution regime,
   unmeasured bottleneck hypotheses, unanswered deployment/workload/ownership
   questions, and anything that could end the assessment early.
4. **Proposed next steps:** the entry points and representative workload to
   profile; time and memory measurements; semantic contract and oracle; improved
   incumbent and non-Warp baselines; the minimum Warp prototype only if those
   survive; correctness audit; complete-boundary benchmark; and the final
   report directory with diffs, scripts and raw results.
5. **Resource preview:** name required datasets, environment/dependency work,
   expected CPU/GPU use, number and breadth of sweeps, and likely wall-clock or
   storage and analysis/token cost when known. Use `unknown until <probe>`
   rather than inventing an estimate.

Then ask:

> Do you want me to continue with this scoped measured assessment? A yes
> authorizes the profiling, minimum prototypes and benchmarks described above;
> I will stop and ask again if the scope or resource cost grows materially.

The user may approve, decline, narrow the scope, or answer the open questions.
Only an explicit approval after this preview starts stage 2. The original
performance request, an explicit skill invocation, or silence does not count.

## Outcomes

- **Approved:** record the approved scope and the authorizing response in the
  eventual report, then begin stage 2.
- **Narrowed:** restate the revised scope and obtain explicit approval for it.
- **Declined or no response:** stop. Produce no verdict and no report directory.
  This is not `ABORT`; Warp was plausible but the assessment never ran.
- **Scope or cost grows later:** stop at the discovery point, explain the
  change, and obtain fresh approval before continuing.
