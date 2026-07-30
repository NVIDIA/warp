<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Skill card — warp-adopt

## Identity

| Field | Value |
|---|---|
| `name` | `warp-adopt` |
| Directory | `warp-adopt/` (matches `name`) |
| Category | Engineering assessment / migration decision support |
| Subject product | NVIDIA Warp (`warp-lang`) |

## Purpose

Assesses whether porting a **narrow part** of an existing codebase to NVIDIA
Warp could add real value, and what evidence would justify it. Produces a
reviewable `warp-adoption-report/` directory with the report, per-solution
diffs, benchmark scripts and raw results, plus per-seam GO / CONDITIONAL GO /
INCONCLUSIVE / NO-GO verdicts — or, when a cheap gate disqualifies Warp before
anything is measured, `ABORT`: a one-sentence hand-back with no directory.
Optimizes for correct engineering judgment rather than adoption: recommending
a non-Warp route, or no change at all, is a first-class outcome.
When stage 1 finds plausible patterns, the skill reports them with a
scope/resource preview and waits for explicit user authorization before any
profiling or experiment.

## Trigger boundary

Warp is **offered as a candidate solution, not requested by name**. The user
normally will not say "Warp"; they describe a slow, memory-bound or
capacity-limited hot path whose shape Warp could address.

**Triggers on:** a request to make an existing codebase faster, smaller in
memory, or larger in capacity, where the hot path is Warp-shaped — irregular or
branch-heavy per-element work; spatial queries (neighbour/radius search,
closest point, ray casting, mesh queries, BVHs, hash grids); particle,
collision, contact, cloth or geometry simulation; Python loops around per-item
device work; many small kernels; or large materialized intermediates — with
NVIDIA GPUs in play.

**Does not trigger when the request itself rules Warp out:** CPU-only
deployment or no NVIDIA GPU; required AMD/Apple/TPU portability; dense tensor
algebra or neural-network layers already served by cuBLAS/cuDNN or a framework
compiler; plain Warp API questions; a Warp kernel already chosen and only
needing to be written; or unrelated autodiff debugging.

**Gating is three-level.** The description suppresses loading when the *prompt*
reveals a disqualifier. Disqualifiers visible only *in the codebase* — a README
stating no GPU budget, a non-NVIDIA deployment target, a mature CUDA path that
already fits — are caught by the disqualifier checks in the workflow's first
stage, which exit to `ABORT` with no report directory and no prototype. If a
pattern survives, a mandatory authorization checkpoint stops before profiling,
dependency work, prototypes, benchmarks or GPU use.

## Contents

| Path | Purpose |
|---|---|
| `SKILL.md` | 9-stage decision workflow with early exits and hard rules |
| `references/target-patterns.md` | Catalogue of patterns worth porting, what decides each, and the ones that reliably are not |
| `references/rejection-gates.md` | Cheap disqualifiers, and the better non-Warp route for each |
| `references/authorization-checkpoint.md` | Early-findings, resource-preview and explicit-authorization contract |
| `references/semantic-contract.md` | Contract-first design and the correctness hazard catalogue |
| `references/baselines.md` | Algorithm-first ladder and strongest-baseline selection |
| `references/benchmark-protocol.md` | Timing phases, memory accounting, evidence integrity |
| `references/verdicts-and-reporting.md` | Verdict gates, vetoes, clean rejection, report rules |
| `assets/warp-adoption-report-template.md` | Versioned 14-section `warp-adoption/v2` report template |
| `scripts/validate_report_schema.py` | Checks the report directory, artifact links, sections, tables, columns and filled cells |
| `evals/evals.json` | Output-quality evaluation set |
| `evals/trigger_queries.json` | Trigger evaluation set (30 queries, 15/15, 60/40 train/validation split) |
| `evals/files/` | Self-contained CI-friendly code fixtures (24 scenarios) |

References are organized by topic, not by workflow stage, and carry no stage
numbers — the workflow in `SKILL.md` owns the sequencing.

## Evidence prerequisites

Screening, assessment and reporting require **no GPU**. A GO or CONDITIONAL GO
verdict additionally requires an NVIDIA CUDA GPU, the target project's
dependencies, and a representative workload. Without those the skill returns
INCONCLUSIVE with the smallest next experiment.

The evaluation suite is CI-friendly and requires no GPU. Any real-GPU
integration evaluation is a separate optional tier and is not included here.
