---
name: warp-eval
description: >
  Evaluate whether an existing hot path is a credible NVIDIA Warp candidate.
  Use for irregular or spatial queries, particle or geometry simulation,
  branch-heavy loops, many small launches, host fallbacks, or large
  intermediates. CPU-only code and absent GPU dependencies are normal unless
  NVIDIA is prohibited. Exclude required cross-vendor or CPU-only deployment,
  vendor-lowered dense or NN layers, general Warp API questions, and
  already-selected Warp kernels. Contribution policy alone is not exclusion.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation <warp-python@nvidia.com>
  tags:
    - warp
    - gpu-acceleration
    - performance
    - simulation
    - evaluation
compatibility: >
  Screening, static evaluation and reporting need no GPU. Measuring Warp
  requires an NVIDIA CUDA GPU, the target project's dependencies and a
  representative workload; without them, abort before profiling.
---

# Warp evaluation

## Purpose

Collect reproducible evidence about how a **narrow seam** in an existing
codebase would behave in NVIDIA Warp. Report facts; the user decides.

Name Warp as the option under evaluation in the first line, state that no
adoption recommendation will follow, and do not treat the triggering performance
request as authorization to experiment.

Measured evaluations produce `warp-evaluation-report/`: the report, one
independently applicable diff per solution, the drivers, and raw results. Never
modify production code. Exits before measured work create no directory.

## Hard rules

These override any local reasoning.

1. **Default objectives are latency, throughput, and peak or retained memory.**
   Count maintainability, ergonomics, packaging, extensibility, autodiff or new
   functionality only when the user names it; otherwise report them as
   constraints or costs, not benefits.
2. **Correctness is a gate.** If the incumbent is buggy or its contract unclear,
   abort that comparison until an independent oracle or clarified contract
   exists. Report the defect without prescribing a response.
3. **Never predict savings from source shape or another workload.** Materialized
   memory comes from profiler or allocator evidence, not source a compiler may
   fuse.
4. **"It can be written in Warp" is a hypothesis, never a proven opportunity.**
5. **Gates are exact, never adjacent or analogous.** Fire a gate only when every
   condition in its definition is established. Gate D requires a maintained
   implementation confirmed to execute with CUDA on an NVIDIA GPU; a fast native
   CPU library is a baseline, not Gate D.
6. **Label every claim** as observed fact, measurement, hypothesis or unknown.
7. **Time the user-visible stage, not the kernel.** Include Warp's cold
   import/init/JIT in the real process regime, transfers, launches, Python
   launch loops, structure build/refit, allocation, conversion, validation,
   compaction and synchronization. Report each cost and the end-to-end
   difference.
8. **No measured work without explicit authorization.** After stage 1, stop and
   ask before profiling, environment changes, prototyping, benchmarking or GPU
   use.
9. **Authorized measured work includes Warp.** If profiling exposes no gate,
   continue through the strongest in-project baseline, minimum Warp prototype
   and end-to-end comparison. If Warp is out of scope, stop before profiling.
10. **Never infer environment intent.** NVIDIA deployment and ownership of an
    optional compiled dependency are product decisions. Abort only on a stated
    constraint; otherwise ask once and stop (`AWAITING INTENT`).
11. **Warp measurements are CUDA-only and synchronized.** Resolve an explicit
    NVIDIA CUDA device; discard CPU resolution and dispatch-only timing.
12. **Never recommend or rank adoption options.** Report facts, measurements,
    hypotheses and unknowns per seam and regime.
13. **Stop at the report.** No rollout or production edits.

## Requirements

Static screening requires the target repository and its stated product
constraints. Measured work additionally requires explicit authorization, an
NVIDIA CUDA GPU, target-project dependencies and a representative workload.

## Limitations

- **Warp CPU kernels are serial and outside this skill's evaluation scope.**
  "Run it on the CPU instead" is not a performance fallback.
- **Mesh and geometry queries compute in float32/int32** even behind a float64
  public API. Absolute error grows with coordinate magnitude.
- **Warp does not follow semver.** Feature releases can break APIs, only the
  newest feature line is maintained, deprecations run roughly four monthly
  releases.
- **`latest` docs track development, not the shipped release.** Pin to the
  target project's own Warp version; if it has none, use the current stable and
  say which.
- **Kernel-only builtins are not resolvable from Python** —
  `hasattr(wp, "mesh_query_point")` is `False` on a release that has it. Probe
  the stub file or that version's docs before concluding a builtin is absent.
- **`enable_backward=False`** (kernel, module or global) removes adjoint
  codegen. If nothing differentiates through the seam, set it before measuring
  compile cost.

## Output Format

| State | Reached when | Report directory |
|---|---|---|
| `ABORT` | Any gate establishes that the evaluated seam cannot satisfy the stated scope | No if nothing was measured; otherwise preserve the evidence already collected |
| `AWAITING INTENT` | The environment gates turn on a fact only the user has | No — one question, both branches concrete |
| `AWAITING AUTHORIZATION` | A candidate pattern survives stage 1 | No — early findings plus scope/resource preview |
| `INCOMPLETE` | Authorized work cannot obtain representative evidence required for the scoped evaluation | Yes — preserve collected evidence and name the one missing artifact |
| Report delivered | Authorized work produced measurements, or stopped after the report directory existed | Yes — facts per seam and regime, with missing evidence explicit |

Use this exact shape for an early exit:
`ABORT — Gate <letter>: <cited fact>; <why the scoped Warp evaluation cannot proceed>.`
Do not name a preferred alternative.
Reporting rules:
[references/evidence-and-reporting.md](references/evidence-and-reporting.md).
A delivered directory follows the template's fixed order: schema and provenance,
authorization and evaluation state, stage census, one `B<n>` evidence section
per seam/regime, caveats, then environment/reproduction. `solutions/`,
`benchmarks/` and `results/` contain every linked artifact.

## Examples

**Gate exit:** `ABORT — Gate A: deployment.md requires one implementation with
AMD, Apple and NVIDIA parity; a Warp-specific path cannot satisfy this scope.`

**Surviving candidate:** Name the seam and pattern, label inferred facts as
assumptions, state that no gate has fired, preview the profile/baseline/Warp
prototype/benchmark scope and its cost, then ask the separate intent and
authorization questions from
[references/authorization-checkpoint.md](references/authorization-checkpoint.md).

## Inputs

**Required:** the target repository and a performance, memory or scale problem
with a candidate seam. **Optional:** explicit deployment/packaging constraints,
existing profiles or logs, representative datasets and acceptance criteria.
Prompt constraints take precedence over repository policy/configuration, then
existing logs. User corrections override inference. Never substitute an
assumption for a stated fact or measurement.

## Available scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/driver-template.py` | Copy once per bottleneck; define workloads and variants | Edit placeholders, then run the copied driver |
| `scripts/measure.py` | Import from drivers for synchronized timing, memory and isolated cases | Python API; do not execute directly |
| `scripts/validate_report_schema.py` | Validate the delivered report and evidence links | `<report-directory>` |

Use `run_script("scripts/validate_report_schema.py", args=["warp-evaluation-report"])`
when supported; otherwise invoke the script with Python and the report directory.

## Troubleshooting

- No representative workload: mark the scope `INCOMPLETE` and name the missing
  artifact; do not invent data or fire Gate F.
- Warp resolves to CPU or no CUDA device: discard the run and stop before
  correctness or timing claims.
- Report validation fails: fix the report or referenced artifact; never waive
  the schema error.

## Instructions

Every stage before the last can end the evaluation. Stop as soon as a gate
fires; do not gather evidence that cannot change the scoped facts.

### 1. Read the code, derive the contract, check the gates

- Identify a candidate and its metric with
  [references/target-patterns.md](references/target-patterns.md).
- Derive devices/residency, dtypes/shapes, sizes, frequency, process lifetime,
  gradients and packaging from the repository. Infer before asking.
- State the inferred contract in one line and invite correction. Unknown
  hardware, counts, sizes and tolerances are assumptions, never measurements.
  Every inference remains open to correction and cannot satisfy a gate that
  requires a stated fact or measurement.
- Check Gates A–E before profiling. Check Gate F now only if representative
  evidence already exists; otherwise carry it into stage 2. Every gate uses only
  the exact boundaries in
  [references/rejection-gates.md](references/rejection-gates.md).

| Gate | Fires when |
|---|---|
| **A** | Production is *stated* CPU-only or to need non-NVIDIA portability, with no acceptable optional CUDA path |
| **B** | Data must cross the host/device boundary per small or infrequent call and the boundary cannot be widened |
| **C** | The region is dense tensor algebra already mapped to a tuned framework or vendor library |
| **D** | A mature CUDA implementation already meets the contract, and no non-performance objective was requested |
| **E** | A stated policy blocks Warp's dependency, compilation, cache or fallback obligations |
| **F** | Representative evidence proves the region too small a share of its requested metric for any backend to move it |

- Gate F can fire in stage 1 only from representative evidence that already
  exists — a supplied profile, structural bound, or arithmetic on figures the
  user quoted. If that evidence does not exist, Gate F remains open until stage
  2 profiling; inferred values never fire it.
- Gates A and E need a stated constraint. A CPU implementation, another
  accelerator, no Warp dependency, or a small dependency list proves nothing.
- When A/E are unresolved and a pattern survives, ask whether an optional NVIDIA
  path is acceptable: named extra, soft import, existing fallback, default
  install unchanged. Every affirmative answer must say explicitly that Warp will
  be prototyped and benchmarked; conditions constrain only that Warp scope. A
  negative or undecided answer means `ABORT`.
- Do not ask when another gate fired, the repository answers, or no pattern
  matched. No pattern means no profiling.
- If a candidate survives, combine any intent question with the
  [authorization checkpoint](references/authorization-checkpoint.md) — early
  findings, exact scope, stages, resource cost — then stop. Stage 2 requires
  both settled intent and explicit authorization.

### 2. Profile the real application

Requires explicit authorization and a settled intent question.

- Profile with the project's own profiler and representative entry points.
- Measure synchronized end-to-end stage time and peak memory before choosing a
  backend. Report which entry points were profiled and which a gate screened.
- Prioritize further measurement by observed cost, not source appearance.
- Name each measurement by the public method and variant actually invoked. A
  fallback is an execution regime of that public seam, not a different
  operation, and a cheaper sibling method cannot screen out the named method.
- Confirm the timed branch ran on the intended device. Unchanged cost and
  near-zero device allocation between host and device inputs exposes a host
  fallback.
- Measure the stage's free-stage ceiling and stubbed floor through the public
  boundary. Do not subtract per-op timings.
- If the measured ceiling proves the candidate cannot move its own metric,
  `ABORT` the affected scope under Gate F, preserve the evidence already
  collected, and stop. The existing authorization already covered this
  materiality check; do not ask for authorization again.
- If representative coverage is unavailable — no representative dataset,
  runnable entry point or production distribution — record the single missing
  artifact, mark the affected scope `INCOMPLETE`, and stop. This is missing
  evidence, not Gate F and not `ABORT`. An invented workload cannot prove
  materiality.

Protocol: [references/benchmark-protocol.md](references/benchmark-protocol.md).

### 3. Form falsifiable hypotheses

Record per candidate: source, bottleneck evidence, objective, narrow seam,
mechanism Warp could change, strongest incumbent, risks, acceptance threshold,
cheapest falsifying experiment. Screen against
[references/target-patterns.md](references/target-patterns.md); if none
survives, write the report and stop.

### 4. Write the contract before the prototype

Define values, dtypes, shapes, devices, errors, mutation, ordering, ties,
capacity/overflow, topology/degeneracy, tolerances, required gradients, streams,
ownership, aliasing, invalidation, concurrency, capture, teardown and fallback.

- Fix tolerances before seeing Warp output. Never weaken a contract after a
  mismatch.
- `ABORT` before prototyping if the proposed seam cannot satisfy a required
  contract.
- Pre-register, before timing: workload provenance, the state variable and
  production range controlling cost, tuning knobs, incumbent run-to-run spread,
  and the oracle applied to every implementation.

Hazards and adversarial checks:
[references/semantic-contract.md](references/semantic-contract.md).

### 5. Improve the baseline first

Algorithm before backend: (1) a better or output-sensitive algorithm;
(2) chunking, tiling, sparse output, layout, rematerialization; (3) the
incumbent framework's compiler and native primitives; (4) **what the project
already depends on** — its own accelerator backend, a parallel idiom it ships
but leaves off, or a capability an existing dependency exposes and nobody wired
up; (5) only then narrow Warp.

- Compare only in-scope options: the improved incumbent, capabilities reachable
  through current dependencies, and Warp. Do not add unrelated libraries.
- Search declared dependencies for dormant backends, flags and bindings before
  calling a route absent.
- Keep the improved incumbent as the oracle. A result against an untuned,
  incorrect or asymptotically inferior baseline does not establish a backend
  comparison.

Close every in-project route before prototyping Warp:

| State | What it takes to claim it |
|---|---|
| **measured** | timed through the same boundary as the baseline |
| **absent** | a cited declaration, symbol table or missing flag proves it is unavailable |
| **waived** | you asked the user and they chose to skip it; record their words |

A capability present but unbound is reachable, not absent. If exposing it costs
no more than the planned Warp seam, measure it first. Ladder details:
[references/baselines.md](references/baselines.md).
Waived routes do not block stage 6; every route not explicitly waived must be
measured or evidenced absent before the Warp prototype begins.

### 6. Prototype the minimum unit

- Work in a separate copy, production unchanged, integration seam off by
  default. Prototype only enough to test the hypothesis.
- Select an explicit CUDA device for every Warp prototype and verify that Warp
  resolves it as CUDA before correctness or performance work. Never exercise or
  report Warp's CPU backend.
- Size the unit by shared data and structure lifetime, not function boundaries:
  build/refit/query costs that amortize together are one unit.
- Name variants before timing. Preserve each solution as an **independent**
  patch against the pinned baseline, and prove it applies cleanly and reproduces
  the measured result.
- Surface unsupported cases and overflow before timing.
- Run the adversarial contract checks, audit correctness at production scale,
  and deliberately break a branch to prove the tests fail. Any failed required
  check returns `ABORT` for the affected scope.
- Before treating a disappointing compute-bound or reuse-heavy result as
  representative, consider the stronger formulations in
  [references/target-patterns.md](references/target-patterns.md). One naive
  kernel does not bound Warp's potential.

### 7. Benchmark the whole boundary

- Copy [scripts/driver-template.py](scripts/driver-template.py) once per
  bottleneck and measure through [scripts/measure.py](scripts/measure.py). Do
  not hand-roll timing or memory.
- Warp launches are asynchronous. The measurement helper must synchronize the
  selected device immediately before starting and after enqueueing every timed
  region, before stopping its wall timer.
- Include every rule-7 cost, the public call, the immediate downstream stage,
  realistic sizes, and cold/warm process regimes.
- Transcribe every report cell from emitted JSON; absent records read
  `not measured`.
- Report `null_test`, and one-time costs both separately and amortized.
  Serialize GPU measurements under an exclusive device lock. Below 1.5× is no
  measured difference.
- `ABORT` for a seam and regime whose predeclared end-to-end performance or
  memory requirement fails, after preserving the measurements.

Full protocol:
[references/benchmark-protocol.md](references/benchmark-protocol.md).

### 8. Summarize the evidence

- Per seam and execution regime, report semantic results, strongest-baseline
  measurements, workload provenance, time, memory, lifecycle, portability and
  ownership facts.
- Use `pass`, `fail`, `not measured`, `not available`, `no representative data`,
  `unknown` or `n/a` only where a stated criterion makes that status objective.
  Missing evidence remains missing.
- Mark the report `complete` only when every evaluated seam and regime includes
  an end-to-end Warp measurement. Use `aborted — <gate and scope>` when a gate
  fired, or `incomplete — <missing evidence and scope>` when representative
  evidence was unavailable. Preserve everything collected. Never deliver an
  incumbent-only report as a completed `warp-eval`.
- Give every stage worth ≥10 % of the measured total a census row with a status
  and a one-line reason, including the screened-out stages.
- Check each gate against the metric *that candidate's* pattern exhausts, not
  the study's headline objective.

### 9. Hand back

- Use
  [assets/warp-evaluation-report-template.md](assets/warp-evaluation-report-template.md)
  unchanged in schema. Record authorization and scope.
- One table per bottleneck: baseline, current-dependency solutions, then Warp;
  absolute time and peak memory, ratios, contract status, evidence gaps.
- Include prose only where it explains the measurements or their bounds.
  Working artifacts belong in `results/`.

```bash
uv run python scripts/validate_report_schema.py <report-directory>
```

- Run the validator once the first measurement establishes a census/table, and
  again before delivery. Fix every error.
- Verify builds and imports from artifacts, not exit status. Ship the exact
  drivers run. Then stop.
