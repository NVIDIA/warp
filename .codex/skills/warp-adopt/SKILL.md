---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: warp-adopt
description: >
  Use when an existing hot path needs more speed, less memory, or scale and does
  irregular/spatial work—neighbor/mesh/BVH queries, particle/geometry
  simulation, host fallbacks, per-item loops, many small kernels, or large
  intermediates—unless NVIDIA is ruled out, even if neither Warp nor a GPU is
  mentioned. Do not use for CPU-only or required cross-vendor parity, dense
  tensor/NN layers handled by vendor libraries/compilers, general Warp API
  questions, or implementing an already-chosen Warp kernel.
license: Apache-2.0
---

# Warp adoption assessment

Assess whether there is an opportunity where porting a **narrow part** of an
existing codebase to NVIDIA Warp could add real value — and, if there is, what
evidence would justify it. The goal is sound engineering judgment, not Warp
adoption.

**Deliverable:** normally one directory, `warp-adoption-report/` (or a directory
the user names), containing the report, independently applicable diffs for
every tested solution, the exact benchmark scripts, and raw results. Nothing
ships outside that directory, and production code remains unchanged. The
exception is `ABORT` — if a gate rules Warp out before anything is measured,
produce no directory at all (below). A user who declines the stage-1
authorization checkpoint likewise gets no directory; that is a hand-back, not a
verdict.

## You were offered, not asked for

The user usually will not have said "Warp". They described a slow, memory-hungry
or capacity-limited hot path that happens to be Warp-shaped, and this skill is
being proposed as one possible answer. Two consequences:

- **Say so in the first line of your reply.** Name Warp as the option being
  evaluated and that the answer may well be "not Warp". Do not present the
  assessment as something they requested.
- **The bar is higher, not lower.** Because nobody asked for Warp, an
  unjustified GO is a recommendation they never sought.

**Gating happens twice, so you should usually be starting from a plausible
opportunity.** The description already keeps this skill from loading when the
*request itself* rules Warp out (CPU-only, non-NVIDIA portability, dense tensor
algebra, an already-chosen kernel, plain API questions). What it cannot see is
the codebase: the stage-1 disqualifiers below catch what only appears
there — a README stating no GPU budget, a deployment target with no NVIDIA
hardware, a mature CUDA path that already fits.

So a gate firing after load should be the exception rather than the norm. When
one does fire, `ABORT` immediately and say why — a fast exit is a correct, cheap
outcome, not a wasted invocation. And when no gate fires, that is not a
licence to reach for GO: an opportunity worth *assessing* is not yet an
opportunity worth *taking*, and "leave as-is" or "use X instead" stays a
first-class result whenever the evidence points there.

## Non-negotiable rules

Read these before anything else. They override any local reasoning.

1. **Default objectives are only these three.** Lower representative end-to-end
   latency; higher representative throughput; lower peak or retained memory —
   including the larger problem sizes a *measured* memory reduction makes
   feasible, which is a consequence of that objective, not a separate one.
2. **Everything else requires the user to name it first.** Maintainability,
   ergonomics, packaging, Python extensibility, autodiff, new functionality —
   these are valid objectives *only* when the user independently states them as
   a desired outcome. Never infer them, never introduce them to rescue a weak
   case, never count them as upside. They may still appear as *requirements,
   risks, or costs*.
3. **Correctness is a gate, never a reason to migrate.** If the incumbent has a
   bug, recommend fixing or clarifying its contract first — unless the user
   explicitly asked for correctness/functionality as the migration objective.
4. **"It can be written in Warp" is not evidence that it should be.** Static
   inspection produces a *hypothesis*, never a proven opportunity.
5. **Never predict a speedup or memory saving from source shape.** A ratio
   measured on another codebase, GPU, or workload is not a forecast for this
   one — never carry one into a report as an expectation.
6. **Label every statement** as observed fact, measurement, hypothesis, or
   unknown — in your reasoning and in the report.
7. **A fast kernel is not a fast program — overhead can turn a kernel win into
   an end-to-end loss, and that is a NO-GO.** Time the whole user-visible stage,
   not the kernel, and charge the Warp path for every cost the incumbent does
   not pay:
   - **cold start** — process import, runtime init, and first-call JIT
     compilation. Decisive for one-shot CLIs and per-request processes, often
     irrelevant for a long-lived service; decide which regime this is.
   - **host↔device round trips** — H2D input and D2H output copies. If data
     starts and ends on the CPU, this alone can exceed the kernel time.
   - **launch overhead** — per-kernel dispatch, and any per-item Python loop
     around launches.
   - **structure build/refit** — BVH, hash grid or cache construction that a
     real input requires on every call.
   - **allocation, conversion, validation, compaction, synchronization** — the
     wrapper work between the caller and the kernel.

   If the sum of these exceeds the kernel saving at the production workload
   size, say so and return NO-GO. Semantics, required gradients, and
   portability veto independently of any of it.
8. **Measured work requires explicit authorization.** After stage 1, stop and
   ask before profiling, environment setup, dependency work, prototyping,
   benchmarking or GPU use. Automatic skill activation and the original
   performance request are not authorization.
9. **Stop at the report.** Do not begin rollout or edit production code.

If you cannot measure something material, say so and return `INCONCLUSIVE`. Do
not fill gaps with estimates.

## Pin the version before you cite behavior

Use the **target codebase's own Warp pin**. If it has none, use the current
stable release and say which. Never describe an API that exists only on `main`
or in `latest` docs as a released capability — Warp's `latest` documentation
tracks development beyond the shipped release. Warp does not follow semantic
versioning: feature releases can break APIs, only the newest feature line is
maintained, and deprecations run roughly four monthly releases. Two behaviours
worth knowing before you cite them: CPU kernels are serial, so "run it on CPU"
is not a performance fallback; and mesh/geometry queries operate in
float32/int32 even behind a float64 public API.

## Workflow

Work the stages in order. Every stage before the last can end the assessment,
and taking an early exit is a *success*, not a failure.

| Outcome | Reached when | Produces |
|---|---|---|
| `ABORT` | A disqualifier fires in stage 1, before anything is measured | **No report directory** — a sentence naming it, plus the better route |
| `AWAITING AUTHORIZATION` | Candidate patterns survive stage 1 | Early findings and a scoped work/resource preview; **no report directory** |
| `NO-GO` | An assessment ran and the evidence says no | Report directory |
| `INCONCLUSIVE` | A material gate or measurement is missing | Report directory |
| `CONDITIONAL GO` | Evidence supports only a bounded envelope | Report directory |
| `GO` | Every gate passes for the named scope | Report directory |

Definitions and the gates behind each:
[references/verdicts-and-reporting.md](references/verdicts-and-reporting.md).

### 1. Read the codebase, derive the contract, check the disqualifiers

**Infer first; ask only what you genuinely cannot determine.** Do not open with
a questionnaire. Audit the code against
[references/target-patterns.md](references/target-patterns.md) to find the
candidate **areas** — the patterns worth measuring, not yet the seam you will
propose — and derive the rest of the contract from the repository:
objective; devices and residency; dtypes and shapes; workload sizes and call
frequency; cold-start vs steady-service; whether gradients are required; and
whether the packaging can accept an optional compiled dependency.

Pick the objective yourself when the user did not name one: use the resource
the pattern is known to exhaust. Do not demand a success threshold up front —
"faster" is a normal way for a developer to open. Derive one from the baseline
once you have profiled.

**The objective is per candidate, not per project.** A study opened as a latency
question will still turn up candidates whose entire value is memory or reachable
problem size. Re-derive the metric for each seam from its pattern class; do not
carry one headline objective into every screen.

Then **state the contract back as a proposal**, marked as inference, and invite
correction in one line: *"Reading the code, this looks like a throughput
problem in the broad-phase, ~10⁵ points per call, GPU-resident float32, running
in a long-lived worker — correct me if any of that is wrong."*

Three things the repository usually cannot answer. Ask them once, together,
after stating what you inferred — so the user is correcting a draft rather than
filling in a form:

- Is NVIDIA-only deployment acceptable for this path?
- Is there a representative workload or dataset you can share for measurement?
- Who would own a second implementation, its tests and its version matrix?

Infer *intent, scope and shape* freely; never infer a *measurement*. Sizes,
hardware, call counts and tolerances you did not observe are assumptions, and
belong in the report's assumptions section, never in a results table.

#### Disqualifiers — check while you read

Any one of these ends the assessment as `ABORT`, before any profiling or
prototyping: production is CPU-only or needs non-NVIDIA portability with no
acceptable optional CUDA path; data must cross the host/device boundary per
small or infrequent call and the boundary cannot be widened; the region is
ordinary dense tensor algebra already mapped to a tuned framework or vendor
library; a mature CUDA implementation already meets the contract and no
non-performance objective was requested; the project cannot accept Warp's
dependency, compilation, cache and fallback obligations; or the region is
provably too small a share of the requested metric for any backend to move it.

**No pattern, no profiling.** If nothing in the codebase matches a target
pattern, there is no candidate to measure: say so and stop. Stage 2 is scoped
work that begins only once at least one candidate pattern is standing — it is
not a survey you run to find out whether the skill applies.

Detail and the better route for each:
[references/rejection-gates.md](references/rejection-gates.md). **Recommending
NumPy/SciPy, Numba, CuPy, JAX, PyTorch, Triton, cuTile, cuCIM, cuML, cuSpatial,
OptiX, Embree, an existing CUDA library, or no change at all is a correct
result.**

To abort: say in a sentence or two what ruled Warp out, name the better route,
and get on with the request that triggered this.

> "Warp isn't a fit here — this ships CPU-only and Warp's CPU kernels are
> serial, so it can't help. The real cost looks like the per-cell Python loop in
> `collide.py:12`; a vectorized rewrite is the thing to do. Want me to take a
> look at that?"

On `ABORT`, write something only if the user explicitly asked for an
assessment, or if the finding is consequential and non-obvious — and then a
short note, never the report template.

#### Authorization checkpoint — stop here

If a pattern survives, follow
[references/authorization-checkpoint.md](references/authorization-checkpoint.md):
report the early facts and hypotheses, preview the exact next stages and their
known or unknown resource costs, and ask whether to continue. Do not profile,
install or build dependencies, create a harness or report directory, prototype,
benchmark, acquire a GPU lock, or delegate measured work before an explicit
yes. If the user declines, hand back with no verdict and no directory.

### 2. Profile the real application

Begin only after the stage-1 scope is explicitly authorized.

Profile the **application**, not a microbenchmark you wrote. Measure
synchronized end-to-end stages and the relevant memory *before* choosing a
backend, and identify launch count, transfers, synchronization, allocator
retention, intermediate materialization, spatial-structure rebuilds, and the
immediate downstream consumer.

**Name the entry-point surface before you pick one.** List the product's
user-facing entry points, and state which you profiled and which you screened
without measuring, naming the gate that justified each. Screening by gate is
legitimate and often correct; screening *silently* turns a narrow study into
one that reads like a survey. A profiled entry point plus a documented gate
argument for the rest is a complete stage 2; one profiled command with the
others unmentioned is not.

**Look for the project's own profiler first.** Many projects ship timing hooks,
a `--profiler` flag, or telemetry that answers materiality in one flag and no
new harness. Check before building your own, and check the logs you have
already produced — a profile you generated and never read is the cheapest
evidence you will ever leave on the table.

**Rank that surface by measurement, not by reading.** Once stage 1 leaves at
least one candidate pattern standing — and only then; a repository with no
pattern match never reaches this stage — measure every entry point you did not
gate away, at one representative scale each, for time *and* peak memory, and
rank them. A pattern match tells you what to look for; it does not tell you
where the time is, and the operation you found by reading is routinely not the
one that owns the runtime. Then compose the stages the product actually runs
together (aggregate → shade → post-process; load → solve → export) and report
the per-stage split: a per-operation table hides which stage owns a frame. The
seam you carry into stage 3 comes out of that ranking. One scale per entry point
is enough here — size, density and variant sweeps belong to whatever the ranking
puts on top.

**Confirm that the path you timed is the path you think you timed.** Compare a
stage's cost *and* its device-allocation delta between host-resident and
device-resident input. A stage that costs the same either way and allocates
essentially no device memory is running on the host: look inside it for an
`asnumpy`/`to_pandas` conversion, or a capability check that silently downgrades
what was asked for. Grep the incumbent for warnings that quietly change the
request — precision, antialiasing, backend, chunking — and confirm which branch
your workload takes. A number attached to the wrong branch is worse than no
number: it will read as a supported fast path that nobody has.

Do not infer materialized GPU memory from source expressions — a compiler may
fuse them. Use profiler or allocator evidence.

**Compute the ceiling before you prototype.** Once a stage's share is measured,
state what the whole user-visible stage would cost if that stage became free,
and the ratio that implies. It is one line of arithmetic and it sets the
expectation the rest of the study is judged against: a seam at 69 % of an
iteration cannot buy more than ~3.2× end-to-end no matter how good the kernel
is. Carry the ceiling into the report and quote the achieved speedup as a
fraction of it — a result at 87 % of the ceiling is finished work, and one at
20 % says the wrapper is eating the win.

**Stop if** the candidate stage is not a material bottleneck and does not
constrain capacity.

If there is nothing to profile against — no dataset, no runnable entry point,
no production distribution — return `INCONCLUSIVE` and name the one artifact
that would unblock it. A workload you invented can exercise correctness; it
cannot establish materiality.

### 3. Form falsifiable candidate hypotheses

Candidate signals: branch-heavy or irregular per-element work; bounded sparse
output; early exits; atomics; repeated spatial queries; persistent BVHs or hash
grids; Python loops around scalar queries; many launches that could be fused or
captured; large *measured* intermediates a local kernel could eliminate; many
independent problems each too small to occupy the device; and iterative loops
that cross the host boundary every step.

For each candidate state: exact source locations and the measured bottleneck
evidence; the requested objective; the proposed narrow seam; why Warp could
change the algorithm, representation, launch count or memory behavior; the
strongest incumbent and non-Warp alternatives; semantic and lifecycle risks; a
falsifiable acceptance threshold; and the cheapest experiment that could
*reject* it.

**If nothing survives screening, write the report and stop.** Screen each
candidate against [references/target-patterns.md](references/target-patterns.md),
which gives the signals, anti-signals and the evidence each pattern turns on.

### 4. Define semantics before implementation

Write the observable contract *before* a prototype exists: values, dtypes,
shapes, devices, errors, mutation; ordering, ties, capacities, overflow,
partitions, topology, degeneracy; numerical, geometric and task-level
tolerances chosen **before** seeing any Warp output; gradients only where
already required or explicitly requested; stream, ownership, aliasing, cache
invalidation, concurrency, graph capture, teardown and fallback behavior.

Correctness and required gradients veto performance. **Never weaken a contract
after a mismatch to save a promising benchmark.** Hazard catalogue (cutoff ties,
face/type ties, nonsmooth gradients, topology drift, float32 scale,
nondeterministic atomics, output aliasing, mutable external state, stream
lifetime, pointer-keyed graph caches):
[references/semantic-contract.md](references/semantic-contract.md).

#### Pre-registration — five lines, before the first measurement

Most bad assessments are decided here, not in the benchmark. Write these
together in the report's §4.1 pre-registration table before you time anything;
each one is a failure that has inverted a verdict:

1. **Workload provenance.** Captured from a live run of the application, or
   assembled from plausible inputs? If assembled, say so and justify it.
2. **The state variable the cost actually depends on**, and its production
   range. For a relational operation this is a property of the input *pair*,
   not of either input.
3. **The candidate's tuning knobs**, named now — so a loss can be attributed to
   the approach rather than to an unset default.
4. **The baseline's own run-to-run spread**, measured now — it is the floor for
   any equality contract you are about to write.
5. **The oracle plan** — what it is, and the fact that *every* implementation
   including the incumbent will be scored against it.

A contract the incumbent itself cannot satisfy is not a contract. Check that
before you write "bitwise".

### 5. Improve the baseline before building Warp

Apply **algorithm first, backend second**, in this order:

1. an asymptotically better or output-sensitive algorithm;
2. chunking, tiling, sparse output, layout change, rematerialization;
3. the incumbent framework's compiler and native primitives;
4. mature domain libraries and existing CUDA implementations;
5. only then a narrow Warp implementation.

Preserve the improved incumbent as the **oracle**. Compare optimized with
optimized and equivalent work with equivalent work. **If the improved incumbent
or another maintained library meets the objective with less complexity,
recommend it and stop the Warp path.** A prototype that beats only an untuned,
incorrect, or asymptotically inferior baseline cannot receive a positive
verdict. Baseline selection per framework:
[references/baselines.md](references/baselines.md).

### 6. Prototype only the minimum operation

Only when a hypothesis survives stages 1–5. Keep production code unchanged;
work in an isolated worktree, copy, or clearly separated experiment directory,
with any integration seam disabled by default. Implement only enough Warp *and
non-Warp* alternatives to test the hypothesis. Do not let an assessment become
a migration.

Give every timed variant an experiment ID before changing it. Preserve each
code-bearing solution as an independent unified diff against the same pinned
baseline commit—not as a cumulative diff on top of another candidate. Include
new files, but exclude benchmark harnesses, results, caches and generated
binaries. Before leaving a variant, prove its diff applies to a clean disposable
checkout and reproduces the implementation that was measured.

Surface unsupported cases and overflow **before** benchmarking, so you never
time a path that silently drops work.

Run adversarial correctness checks **before** performance tests: empty,
singleton, duplicate, tie, cutoff-boundary, capacity, overflow, degenerate,
extreme-scale, noncontiguous, mutation, stream, repeated-call and large-random
cases as applicable. Compare semantic sets, partitions, topology and downstream
invariants rather than arbitrary IDs where the contract permits variation.

Audit at production scale. Rare semantic failures hide easily: a few hundred
sampled points can agree perfectly while a full-scale audit exposes the
mismatches that decide the verdict, so size the audit to the defect rate you
would need to detect. Prove your tests can fail — break a branch deliberately
and confirm the suite catches it.

### 7. Benchmark the complete boundary

Benchmark the **whole boundary**, not the kernel: cold start, transfers,
structure build/refit, wrapper overhead, the public call, and the immediate
downstream stage all belong in the comparison, alongside crossover behaviour
across realistic sizes and cold/warm regimes.

Four rules that decide whether the numbers mean anything:

- Never compare asynchronous dispatch time with synchronized end-to-end time.
- Name the memory accounting domain for every figure; one framework's allocator
  counters do not include another's.
- **A whole-application run is one sample.** Warm-up and ordering distort it
  exactly as they distort a microbenchmark, but there is no median to hide
  behind. Run each variant at least twice, alternate which one goes first, and
  discard the first run of the session. An unrepeated end-to-end number has
  overstated a speedup by ~20 % in practice.
- Document any deviation from the protocol rather than presenting a weaker one
  as if it were this.

Phases, sampling, memory domains and reproduction record:
[references/benchmark-protocol.md](references/benchmark-protocol.md).

### 8. Apply the verdict gates

`GO` requires every applicable gate to pass for the named scope: semantics,
strongest baseline, representative workload, memory, lifecycle, portability and
ownership. Anything narrower than that is a `CONDITIONAL GO` with its envelope,
dispatch threshold and fallback stated.

A large warm-kernel win is vetoed by any of: rare semantic failures on
contractual outputs; missing or wrong gradients where they are required; the
overheads in rule 7 exceeding the saving at production size; or hidden mutable
state with no safe ownership contract.

Give verdicts **per narrow seam and per execution regime**, never one
aggregate — the same operation is routinely a `GO` for sparse or batched inputs
and a `NO-GO` for dense single ones, and an averaged verdict hides both. Regime
is an axis of that table, not a property of the study: one product commonly
ships both a long-lived loop and a one-shot command, and the *same seam* can be
the dominant cost in one and unreachable noise in the other, behind a cold start
that dwarfs it. Enumerate the regimes the product actually has and answer for
each.

**Close the loop between the profile and the matrix.** Every stage worth ≥10 %
of the measured total needs a row with a status and a one-line reason, including
the rejections. Depth on one seam is right; large stages left unaccounted for
read as missed rather than screened. And check that each gate was applied
against the metric *that candidate's* pattern exhausts — a memory candidate
rejected on a latency share has been mismeasured, not screened.

Gate detail, vetoes and how to close out a rejected candidate:
[references/verdicts-and-reporting.md](references/verdicts-and-reporting.md).

### 9. Hand control back

If you reached this stage you did real work, so write the report from
[assets/warp-adoption-report-template.md](assets/warp-adoption-report-template.md)
using the `warp-adoption/v2` directory contract in
[references/verdicts-and-reporting.md](references/verdicts-and-reporting.md).
Keep every required section, table and artifact link unchanged; record the
authorized scope in the header. Copy artifacts without reconstructing them,
credentials or external datasets. Run
`python scripts/validate_report_schema.py <report-directory>` from this skill
directory and fix every error before delivery. Then stop. Let the user decide
whether to stop, gather missing data, authorize a narrow prototype, or request
implementation as a separate follow-up.

If instead you `ABORT`ed in stage 1, you have already handed back. Do not
circle back to write it up.

## What a good seam looks like

The normal architecture is an **optional kernel island or backend**, never a
rewrite. Preserve the frontend, domain logic, reference implementation and
fallback. Isolate the smallest measured execution layer. Prefer explicit
caller-owned cache and acceleration-structure lifetimes over hidden state. Keep
unsupported devices, dtypes, shapes, semantics and small workloads on the
existing path.

Do **not** prospect inside mature, specialist-maintained C++/CUDA code by
default. Assess such a rewrite only when the user explicitly requests an
allowed non-performance objective *and* defines an acceptable
performance-regression budget.

## Running experiments

Read-only analysis and CPU-only work can run in parallel. **Every GPU
measurement is serialized under an exclusive device lock** — concurrent
benchmarking contaminates results in the direction that favours whichever side
has the longer kernel. Ask before anything unusually expensive or long-running.

**Confirm every step from its artifact, not from its exit status.** Build and
install commands routinely report success while having failed — a redirected
log, a swallowed status, a wrapper that returns 0 regardless. Check the thing
you wanted to exist: the extension imports, the symbol is present, the file has
today's timestamp. An assessment built on a stale binary measures the wrong
code and looks entirely normal while doing it.

The scripts copied to `benchmarks/` are the scripts actually used, including
setup or orchestration code required to reproduce the result. Make them
location-independent, document commands and prerequisites in
`benchmarks/README.md`, and keep host-specific paths, credentials, downloaded
datasets and caches out. Every accepted number links to its raw file under
`results/`.

If you delegate, give each subagent one falsifiable question and reconcile the
answers against raw artifacts yourself; a subagent's confidence is not evidence.
