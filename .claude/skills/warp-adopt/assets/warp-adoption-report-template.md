<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

<!-- warp-adoption-report-schema: 2 -->
# Warp adoption assessment — <project / subsystem>

> Place this file at `<output-directory>/warp-adoption-report.md`, with sibling
> `solutions/`, `benchmarks/`, and `results/` directories. Keep all numbered
> section titles, `required-table` markers, and required table columns exactly
> as written and in this order. Add project-specific detail only as prose or
> `###` subsections under the nearest numbered section; do not replace a
> required table with a custom one. Fill every required table. Use exactly
> **"not measured"** (in scope, not collected), **"not available"** (the tool or
> counter does not exist), **"no representative data"** (the workload is
> missing), **"unknown"** (a fact is unresolved), or **"n/a"** (the field does
> not apply). Never substitute a plausible number. Keep the HTML markers but
> delete these quoted instruction lines before delivering.

**Report date:** <YYYY-MM-DD>
**Prepared by:** <agent/human> · **Reviewed by:** <human — required before any action>
**Status:** DRAFT — for user review. No production code was modified.
**Assessment authorization:** <date / user response and approved scope>
**Report schema:** `warp-adoption/v2`
**Assessment depth:** full / profile-only / correctness-only / inconclusive-minimal

---

## 1. Scope, requested outcomes, and non-goals

**Requested objective (user's words):** <quote>

**Objective class:** latency / throughput / memory / capacity-from-measured-memory

**Success threshold:** <the number or condition that counts as success, as
supplied by the user; "not supplied" if the user gave none>

**Scope — what was assessed:** <files, modules, stages>

**Non-goals — explicitly not assessed or not counted:**

- Maintainability, developer ergonomics, packaging, Python extensibility,
  autodiff, and new functionality are **not** objectives here unless listed
  below as user-requested.
- User-requested non-performance objectives: <none / list them, with the
  user's own wording and, for any native-code rewrite, the agreed
  performance-regression budget>
- <other exclusions>

---

## 2. Executive verdict

**Report-level summary:** GO / CONDITIONAL GO / INCONCLUSIVE / NO-GO / MIXED
**Summary derivation:** single-decision / unanimous-decisions / dominant-regime / mixed-decisions

> `ABORT` never appears in this report. An assessment disqualified at the cheap
> gates produces no report directory at all — if you are filling this template
> in, an assessment ran and there is evidence to record.
>
> The report-level summary is only a synopsis. The authoritative verdict is per
> candidate, narrow seam, and execution regime in §12.

<Two to five sentences. State what decided it. If the recommendation is to
change nothing, or to do something other than Warp, say that plainly here.>

**Recommended action:** <e.g. "Leave as-is", "Adopt <library> instead",
"Fix the algorithm first", "Authorize a narrow prototype of X", "Gather Y">

<!-- required-table: executive-summary -->
| Candidate | Seam | Regime | Verdict | Recommended action | Decisive evidence |
|---|---|---|---|---|---|
| C1 | | long-lived / one-shot / other | GO / CONDITIONAL GO / INCONCLUSIVE / NO-GO | | |

---

## 3. Assumptions and missing information

<!-- required-table: assumptions-gaps -->
| # | Assumption or gap | Why it matters | How to resolve |
|---|---|---|---|
| 1 | | | |

**Questions the user has not answered:** <list, or "none">

---

## 4. Environment, target workload, and evidence provenance

<!-- required-table: environment -->
| Item | Value |
|---|---|
| Repository / commit | |
| Warp version (project pin) | |
| Framework versions | |
| GPU / driver / CUDA | |
| OS / Python | |
| Cache state during measurement | |
| **Execution regimes present in the product** | list **every** one (e.g. long-lived training loop *and* one-shot CLI). Cold start and JIT are charged per call in some and amortised in others, so §2 and §12 answer per regime |
| Device memory budget assumed | <the card any materiality or capacity judgement is judged against> |
| Device lock held for sweeps | yes / no / n/a |
| Representative dataset | <name, or "none — procedural substitute: ...”> |

**Entry-point surface:** do not leave a user-facing entry point unmentioned.

<!-- required-table: entry-point-surface -->
| Entry point / variant | Regime | Status | Workload | Gate / reason | Evidence |
|---|---|---|---|---|---|
| | | profiled / screened / out of scope | | | fact / measured / hypothesis / unknown |

**Production workload distribution:** <sizes, densities, batch shapes,
frequencies — as supplied or as measured; say which>

### 4.1 Pre-registration record

> Record these five fields before the first candidate measurement. If the study
> ended before a field could be established, keep the row and use a standard
> missing-evidence token.

<!-- required-table: pre-registration -->
| Field | Pre-registered value | Status / evidence |
|---|---|---|
| Workload provenance | captured from a live run / assembled from plausible inputs / no representative data | |
| State variable and production range | <e.g. query-to-reference distance / reference spacing, 15× → 1× over a run; or "n/a, cost is size-only"> | |
| Candidate tuning knobs | | |
| Baseline run-to-run spread | deterministic / measured distribution / not measured | |
| Oracle plan and both-sides scoring | | |

**Evidence status legend used below:** `fact` (observed in source or output) ·
`measured` (this study's own measurement) · `hypothesis` (reasoned, untested) ·
`unknown`.

---

## 5. Baseline profile and strongest alternatives

### 5.1 Entry-point census

> Every user-facing entry point that survived stage 1, measured at one
> representative scale. Entry points screened without measuring appear here too,
> with the gate that justified screening them. The candidate in §6 comes out of
> this ranking.

<!-- required-table: entry-point-census -->
| Entry point / variant | Regime | Workload | Host time (ms) | Device time (ms) | Peak memory (MiB; domain) | Status / gate | Evidence |
|---|---|---|---|---|---|---|---|
| | | | | | | profiled / screened — gate X / out of scope | fact / measured / hypothesis / unknown |

**Pipeline composition** (the stages the product runs together, so the reader
sees which stage owns a call):

<!-- required-table: pipeline-composition -->
| Pipeline | Regime | Workload | Stage | Time (ms) | Share of total | Peak memory (MiB; domain) | Evidence |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

**Cost-driver check:** <what the top entry points' cost should scale with, and
whether the measurements matched — state any workload rebuilt because they did
not>

### 5.2 Profile of the candidate stage (synchronized, per stage)

<!-- required-table: candidate-stage-profile -->
| Stage | Time (ms) | Share of total | Peak memory (MiB; domain) | Evidence |
|---|---|---|---|---|
| | | | | measured / not measured |

**Is the candidate stage a material bottleneck?** yes / no — <evidence>

**Ceiling if this stage became free:** <whole-stage cost without it> →
**<ratio>× maximum end-to-end**. Quote every later speedup as a fraction of
this.

### 5.3 Strongest practical non-Warp alternatives considered

<!-- required-table: alternatives -->
| Candidate | Alternative | Applicable? | Outcome | Evidence |
|---|---|---|---|---|
| C1 | Better algorithm | | | |
| C1 | Chunking / tiling / sparse output / layout | | | |
| C1 | Incumbent compiler or native primitives | | | |
| C1 | **The project's own accelerator backend, same algorithm** | | | |
| C1 | Mature domain library or existing CUDA | | | |

**Baseline used for all comparisons below:** <which implementation, and whether
it is the original or an improved version>

---

## 6. Candidate matrix

> **This table must account for the §5 profile.** Every stage worth ≥10 % of
> the measured total needs a row, including the ones you rejected —
> `rejected — gate D, mature CUDA library` is a complete row. State per
> candidate which metric its gate was applied against; it is the metric that
> candidate's pattern exhausts, not the study's headline objective.

<!-- required-table: candidate-matrix -->
| ID | Candidate | Source location | Share of profile | Objective / gate metric | Proposed seam | Status | Notes |
|---|---|---|---|---|---|---|---|
| C1 | | `path:line` | % | | | `rejected` / `hypothesis` / `measured` / `recommended` | |

For each candidate that is not `rejected`, record:

- **Why Warp could change the outcome** (algorithm, representation, launch
  count, or memory behavior — not "Warp is fast"):
- **Strongest incumbent and non-Warp alternative:**
- **Semantic and lifecycle risks:**
- **Falsifiable acceptance threshold:**
- **Cheapest experiment that could reject it:**

---

## 7. Correctness and semantic-contract results

**Contract written before prototyping:** yes / no / n/a — <link or summary>

**Tolerances fixed before observing Warp output:** yes / no

**Incumbent's own run-to-run spread (the tolerance floor):** <per output, or
"deterministic"> — measured before the contract was written

<!-- required-table: correctness-results -->
| Candidate | Check | Result | Evidence |
|---|---|---|---|
| C1 | Empty / singleton / duplicate | | |
| C1 | Ties and boundary values | | |
| C1 | Capacity, overflow surfaced | | |
| C1 | Degenerate geometry | | |
| C1 | Extreme scales | | |
| C1 | Noncontiguous / unsupported dtype fallback | | |
| C1 | Mutation, refit, rebuild | | |
| C1 | Non-default stream | | |
| C1 | Repeated call / repeated backward | | |
| C1 | Graph capture, replay with changed input | | |
| C1 | Acceleration structure retained across capture + caller release (weakref) | | |
| C1 | Large random audit at production scale | | |
| C1 | Immediate downstream result / shipped artifact | | |
| C1 | Unchanged upstream test suite — and *what* it covers | | |
| C1 | Suite proven able to fail (deliberate break) | | |

**Oracle scoring — both sides:** candidate wrong on <N₁>, incumbent wrong on
<N₂>, against <oracle description>. (A bare "candidate differs from oracle on
N" is not admissible.)

**Ambiguity reachability:** <how often the contested output is even
well-defined in the production data — e.g. "5,959 / 100,000 queries land in the
unambiguous case", or "n/a">

**Hazards characterised rather than asserted:** <e.g. "unbound-stream
mismatch: 0/20 runs", or "none">

**Gradients:** required? yes / no · tested? <how> · result:

**Semantic gate verdict:** PASS / FAIL — <what failed, and whether it is
contractual>

---

## 8. Reproducible experiment and benchmark method

**Commands:** <literal commands from `benchmarks/README.md`, no absolute paths>

**Protocol:** warmups <n>, retained samples <n>, synchronization <where>,
statistic reported <median + p10/p90>.

**End-to-end runs:** repetitions per variant <n ≥ 2>, order alternated
<yes/no>, first run of the session discarded <yes/no>.

**Candidate tuning knobs named before benchmarking:** <list, and which were
swept> — a candidate that lost on unset defaults has not been compared.

**Deviations from the protocol and why:** <list, or "none">

**Benchmark bundle:** `benchmarks/README.md` and the exact scripts it invokes.

**Raw artifact locations:** <paths under `results/`; external datasets are named
by provenance and are not copied>

---

## 9. Optimization experiments and runtime results

> Use milliseconds for every time and MiB for every memory figure, converting
> units rather than changing columns. Put workload, regime, and timed boundary
> in the row; do not compare a kernel-only row with an end-to-end row.

### 9.1 Optimization ledger

> This is the report's authoritative inventory of experiments. Include the
> reference baseline and **every optimization or implementation actually
> timed**, including improved incumbents, non-Warp alternatives, Warp variants,
> incorrect variants, superseded runs, and negative controls. Use one row per
> candidate × implementation × decision workload × regime; split rows whenever
> the verdict changes. A detailed sweep may follow, but it never replaces this
> ledger. `Decision` is one of `BASELINE`, `CARRY FORWARD`, `RECOMMEND`,
> `REJECT`, `INCONCLUSIVE`, or `EXCLUDE INVALID`. Measure both decision-boundary
> time and decision-relevant peak memory for every runnable, semantics-valid
> implementation. Missing either result stays visible and prevents
> `RECOMMEND`; invalid implementations use `EXCLUDE INVALID`. Every
> non-baseline code solution links an independent patch under `solutions/`.
> Every timed row links the exact script under `benchmarks/` and raw output
> under `results/`; several rows may share one solution diff. A
> configuration-only solution uses `Solution type = configuration` and
> `Solution diff = n/a`; its exact change must be present in the benchmark
> command.

<!-- required-table: optimization-ledger -->
| ID | Candidate | Optimization / implementation | Workload / regime | Timed boundary | Correctness | Time (ms) | Peak memory (MiB; domain) | Relative to baseline | Decision | Solution type | Solution diff | Benchmark entry point | Raw result | Decisive note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | C1 | reference baseline | | public call / stage / kernel | PASS / FAIL / PARTIAL / NOT TESTED / N/A | | | time: 1.00×; memory: 0 MiB | BASELINE | baseline | n/a | `benchmarks/<script>` | `results/<file>` | |

### 9.2 Cold costs

<!-- required-table: cold-costs -->
| Experiment | Implementation | Phase | Cache state | Time (ms) | Charged in regime? | Evidence |
|---|---|---|---|---|---|---|
| E1 | | Process/import init | fresh process | | yes / no / n/a | |
| E1 | | Fresh-cache compile | empty | | yes / no / n/a | |
| E1 | | Populated-cache load, fresh process | populated | | yes / no / n/a | |
| E1 | | First launch / structure build | populated / n/a | | yes / no / n/a | |

### 9.3 Totals for representative repetition counts

<!-- required-table: repetition-totals -->
| Workload / regime | Implementation | 1 call (ms) | 10 calls (ms) | 100 calls (ms) | Measurement status | Evidence |
|---|---|---|---|---|---|---|
| | | | | | measured / derived / not measured | |

### 9.4 Crossovers

> State measured brackets, never interpolated points.

<!-- required-table: crossovers -->
| Candidate | Axis | Crossover bracket | Production distribution sits | Evidence |
|---|---|---|---|---|
| C1 | | | winning side / losing side / spans bracket / unknown | |

**Does the production distribution lie on the winning side?** yes / no /
unknown

### 9.5 Optional detailed sweeps

Add workload-, size-, state-, or variant-specific tables here when they explain
the decision. Keep §9.1 complete even when a detailed table repeats its values.

---

## 10. Memory results

> Name the accounting domain for every number. One framework's allocator
> counters do not include another's allocations. Use one row per implementation
> and domain so any number of alternatives fit the same columns. The
> optimization ledger repeats the decision-relevant peak; this table gives its
> accounting detail.

<!-- required-table: memory-results -->
| Experiment | Candidate | Workload | Implementation | Domain | Peak memory (MiB) | Delta vs baseline (MiB) | How measured | Evidence |
|---|---|---|---|---|---|---|---|---|
| E1 | C1 | | reference baseline | Live / allocated | | 0 | | |
| E1 | C1 | | reference baseline | Allocator-reserved | | 0 | | |
| E1 | C1 | | reference baseline | Process-level | | 0 | | |
| E1 | C1 | | reference baseline | Peak temporary | | 0 | | |
| E1 | C1 | | reference baseline | Acceleration structure | | 0 | | |
| E1 | C1 | | reference baseline | Retained output / autograd | | 0 | | |
| E1 | C1 | | reference baseline | Graph pool | | 0 | | |

**Measured in fresh processes:** yes / no · **Polling interval:** <value or n/a>
· **Counters unavailable in this version:** <list, or "none">

---

## 11. Integration, lifecycle, portability, and maintenance costs

<!-- required-table: integration-costs -->
| Dimension | Finding |
|---|---|
| API seam and ownership model | |
| Cache / structure lifetime and invalidation | |
| Streams and synchronization contract | |
| Concurrency and thread/process policy | |
| Graph capture and retention requirements | |
| Fallback path (device, dtype, shape, size) | |
| Optional dependency and lazy import | |
| Kernel cache in deployment (writable FS, prewarming) | |
| Version qualification matrix | |
| Who owns the second implementation and its tests | |
| Dependency friction encountered | |

---

## 12. Decisions by candidate, narrow seam, and regime

> One row per (seam × execution regime). The same seam commonly earns opposite
> verdicts in a long-lived loop and a one-shot command of the same product; a
> single row per seam hides that.

<!-- required-table: seam-decisions -->
| Candidate | Seam | Regime | Verdict | Approved envelope | Dispatch condition | Fallback | Recommended action | Decisive evidence |
|---|---|---|---|---|---|---|---|---|
| C1 | | long-lived / one-shot / other | GO / CONDITIONAL GO / INCONCLUSIVE / NO-GO | | | | | |

**Vetoes applied:** <semantic / gradient / full-stage / transfer / lifecycle /
cold-start — or "none">

**Vetoes explicitly not applicable:** <list the tempting but unsupported
explanations the evidence ruled out, or "none">

**Gates not satisfied, and what they force:** <e.g. "strongest baseline — no
maintained library measured → INCONCLUSIVE below 1 k elements"; or "none">.
An unmet gate constrains the verdict; it is not a caveat printed beside it.

**Re-entry condition:** <what evidence or product change would make a rejected
or inconclusive seam worth reassessing, or "none">

---

## 13. Smallest next steps requiring user review

1. <the single smallest experiment or decision that unblocks the largest
   uncertainty>
2. <next>

**Nothing in this report has been applied.** The user decides whether to stop,
gather missing data, authorize a narrow prototype, or request implementation as
a separate follow-up.

---

## 14. Evidence and raw artifacts

> Every location is relative to the report directory. Do not point at an
> untracked assessment directory elsewhere in the target checkout.

<!-- required-table: evidence-artifacts -->
| Claim | Artifact | Location |
|---|---|---|
| | | |

**Claims that could not be substantiated and were therefore removed:** <list,
or "none">

**Measurements attempted but not accepted, and why:** <list, or "none">
