# Evidence and reporting

Report what was observed; the user owns the adoption decision.

## Procedural states

- **`ABORT`** — a gate establishes that the scoped seam or execution regime
  cannot satisfy a stated requirement. Stop immediately. *Before* measured work,
  hand back one sentence naming the gate, citing its source and stating the
  disqualifying fact, and create no report directory. *After* measured work has
  started, preserve the report and artifacts collected so far, mark the affected
  scope `aborted`, record the failed criterion and stop work on it.
- **`AWAITING INTENT`** — the repository does not say whether an NVIDIA path and
  Warp's runtime obligations are acceptable. Ask once and stop.
- **`AWAITING AUTHORIZATION`** — a candidate survived the static screen, but the
  user has not authorized profiling, prototyping or GPU use. Present the scope
  and resource preview, then stop.
- **`INCOMPLETE`** — authorized work started, but representative evidence
  required by the scope was unavailable. Preserve collected artifacts, name the
  one missing artifact and stop. Missing evidence is not a gate.
- **Report delivered** — authorized work produced evidence, or a gate stopped
  work after the report directory existed. This carries no aggregate outcome.

Mark the report itself `complete` — every evaluated seam and execution regime
carries an end-to-end Warp measurement against the strongest in-project baseline
— `aborted — <gate and scope>`, or
`incomplete — <missing evidence and scope>`. A baseline-only or incumbent-only
report cannot be `complete`. Do not keep measuring or tuning an affected seam
after a required criterion fails merely to make the report look complete.

## Report facts per seam and execution regime

Keep one evidence block per narrow seam and per execution regime, never rolled
up into an overall opinion. Each block records:

1. **Scope** — operation, source location, execution regime, device, dtype,
   shapes and production range.
2. **Criterion status** — each predeclared semantic, performance, memory,
   lifecycle, portability and ownership criterion as `pass`, `fail`,
   `not measured`, `not available`, `no representative data`, `unknown` or
   `n/a`, with its evidence.
3. **Strongest baseline** — the best correct route reachable through the
   project's current dependencies, or the evidence showing that no such route
   was available.
4. **Measurements** — absolute end-to-end time and peak memory, ratios,
   cold/warm costs, crossovers and run-to-run context.
5. **Contract findings** — corrections and regressions against an independent
   oracle, unsupported cases, overflow behavior, precision and gradient results.
6. **Operational facts** — dependency, compilation, cache, fallback, ownership,
   concurrency, stream and teardown requirements.
7. **Evidence gaps** — exactly what was not measured and why. Name the smallest
   measurement that would fill a gap only as a procedural fact, not a
   recommendation.

Label candidate status `screened out`, `hypothesis`, `measured` or `aborted`. Do
not use adoption statuses, recommendations, winners, rankings or equivalent
judgments.

## Required gates

A large warm-kernel speedup does not erase a failed requirement. Abort the
affected seam or regime when any of these facts is established:

- contractual values, identities, partitions, topology, gradients, errors or
  fallback behavior fail;
- the user-visible stage or its immediate downstream consumer exceeds a stated
  performance limit;
- cold start, transfers, launches, structure work or wrapper costs violate a
  stated end-to-end limit;
- peak live, reserved or process memory exceeds the stated deployment budget;
- required representative workload coverage is unavailable;
- required lifecycle, concurrency, stream, pointer, invalidation, capture or
  teardown behavior fails;
- required device, dtype or shape coverage lacks the stated fallback;
- the project cannot own a stated dependency or maintenance obligation.

State the failed requirement and the measured fact. Do not translate either into
a preferred implementation.

## Ending an aborted evaluation cleanly

1. Do not weaken the contract after observing a failure.
2. Do not broaden the implementation to rescue sunk cost.
3. Leave integration seams reverted or disabled; retain only standalone
   evidence such as correctness tests, harnesses and independent patches.
4. Record the factual condition under which the failed criterion would need to
   be measured again.
5. Stop spending benchmark time on the affected scope.

## The report directory

Copy the report template named by `SKILL.md` to
`warp-evaluation-report/warp-evaluation-report.md` (the user may name another
directory) and fill it in:

```text
warp-evaluation-report/
├── warp-evaluation-report.md
├── solutions/          # one independent .patch per tested code solution
├── benchmarks/         # the scripts actually run, plus README.md
└── results/            # raw samples and logs
```

Add case-specific explanation under the nearest section. A specialized sweep
table may supplement a required table; it never replaces one.

## Before delivering — what the validator cannot check

- **Authorization is recorded.** The header carries the authorizing response and
  the approved profile, baseline, Warp prototype and Warp benchmark scope. Any
  experiment beyond that scope needs later authorization.
- **No overall assessment.** The opening summary lists the measured scope and
  the most decision-relevant facts. Is there any sentence telling the user what
  to adopt, avoid or prefer? Remove it.
- **The bottleneck table is exhaustive for what was timed**, including slower
  and failed attempts.
- **Only in-scope comparisons appear:** the baseline, routes reachable through
  current dependencies, and Warp — no unrelated libraries added to create a
  leaderboard.
- **The profile is accounted for.** Every stage worth at least 10 % of the
  measured total has a census row with its ID and factual screening status.
- **Empty evidence stays visible.** `not measured`, `not available`,
  `no representative data`, `unknown` or `n/a`; never a plausible substitute.
- **Every number traces to a raw artifact and an exact workload**, with hardware
  and measurement context. Never project a ratio from another codebase, GPU or
  workload, and cite `path:line` for codebase facts.
- **Every device path is confirmed to have run on the intended device**, and
  every Gate D claim cites evidence that the incumbent executed through CUDA on
  an NVIDIA GPU.
- **Every screening claim measures the public method and variant it names**,
  rather than a helper, fallback or cheaper sibling operation.
- **Warp was compared with the project's own strongest correct route** running
  equivalent work.
- **Every comparison is scoped** to a seam, regime, size range and hardware
  target, and ratios below the protocol's resolution read as no measured
  difference rather than being ordered.
- **Contract failures and missing measurements are explicit**, not buried in
  prose, and every aborted scope names the gate, source and observed fact.
- **Non-goals are stated.** Separate observations outside the user's requested
  objectives rather than counting them as benefits.
