<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Verdicts and reporting

The five outcomes, what vetoes a positive one, how to close out a rejected
candidate, and the rules the report must satisfy.

`ABORT` and an authorization hand-back produce no report directory. The latter
is not a verdict: the assessment never started. The four measured outcomes all
produce a directory because each records evidence.

## The outcomes

Give a verdict **per narrow seam**, never one aggregate for a whole project.

### ABORT — no report directory

Not a verdict on Warp's merit; a statement that the question does not apply.
Something in the environment or the workload disqualified Warp before any
assessment happened: CPU-only or non-NVIDIA deployment, a host/device boundary
that cannot be widened, dense tensor algebra already on vendor libraries, a
mature CUDA path that already fits, obligations the project cannot accept, no
representative workload, or a region too small to matter.

Nothing was measured, so **there is nothing to report**. Give the user a
sentence naming the disqualifier and the better route, then return to whatever
they actually asked for. Do not create the report directory or open the report
template. Do not produce a verdict table for a single line of reasoning.

If the user explicitly asked for a written assessment, they still get a short
note — which gate fired, why, what to do instead — but not the full template.

### AWAITING AUTHORIZATION or declined — no report directory

Candidate patterns survived the cheap screen, but static shape is only a
hypothesis. Present the early findings, planned stages and resource preview from
[authorization-checkpoint.md](authorization-checkpoint.md), then stop. If the
user declines or does not answer, give no verdict: neither `ABORT` nor `NO-GO`
is justified because no assessment ran.

Everything below this point applies only after the user explicitly authorized
the scoped measured assessment.

### GO

Every applicable gate passes **for the named scope**:

1. **Semantic** — required values, identities, partitions, topology, gradients,
   errors and fallback behavior pass on adversarial *and* representative data.
2. **Strongest baseline** — the path beats the best practical non-Warp
   algorithm or implementation, not merely the legacy path. This gate is
   **binding, not a caveat**: if a maintained library plausibly covers the
   operation and you did not measure it, the verdict is `INCONCLUSIVE` for the
   size range where it plausibly wins, and you name the library and the
   experiment. "No mature alternative was measured" recorded as a known gap
   next to a positive verdict does not satisfy this gate — a reader takes the
   headline and skips the caveat. The risk is real in both directions: a
   maintained point-face library measured late beat *both* the incumbent and
   the Warp path below ~1 k elements while losing by 200× above it, which
   changes the dispatch floor rather than the verdict.
3. **Workload** — the measured production distribution sits on the winning side
   of every crossover (size, density, batch, reuse, cold start).
4. **Memory/scale** — peak live, reserved and process memory are acceptable,
   and any scale claim includes retained structures, outputs, graph pools and
   framework allocators.
5. **Lifecycle** — streams, pointers, retention, invalidation, graph capture,
   aliasing, concurrency and teardown have explicit tested contracts.
6. **Portability** — unsupported device, dtype and shape cases have a
   maintained fallback or are deliberately excluded from the product.
7. **Ownership** — the benefit justifies custom kernels, custom or reused
   gradients, dependency constraints, platform tests and benchmark maintenance,
   and someone has agreed to own them.

Even a GO is normally scoped: opt-in, experimental, one dtype, one device
class, above a dispatch floor.

### CONDITIONAL GO

Evidence supports only an explicitly bounded envelope — named sizes, dtypes,
batches, reuse levels, devices or semantics. State the conservative dispatch
condition and the fallback. Bound it by what you measured, and set the
threshold **outside** overlapping p10/p90 bands rather than at the observed
crossing point.

### INCONCLUSIVE

A material gate or representative measurement is missing: no GPU, no
representative workload, no real data, untested concurrency, unmeasured
gradients, or no strongest-baseline comparison.

**This is not approval.** Say exactly which evidence is missing and give the
single smallest next experiment that would resolve it.

### NO-GO / LEAVE AS-IS

A required contract fails; a stronger alternative wins; the production
distribution lies below crossover; costs outweigh the requested benefit; or no
material opportunity exists.

This is a successful outcome and should read like one — confident, specific,
and without consolation prizes.

**A rejection at the cheap gates is an `ABORT`, not a `NO-GO`** — see above. Use
`NO-GO` only where an assessment actually ran, so the report has measurements
to record and the rejection is a finding rather than a precondition.

## Vetoes

A large warm-kernel win is overridden by any of:

- rare semantic failures on contractual outputs;
- gradient failures, or gradients that were required and were not tested;
- the whole user-visible stage getting slower even though the kernel got
  faster, or the immediate downstream consumer regressing;
- cold start, host↔device copies, launch overhead, structure build/refit, or
  wrapper allocation/conversion/synchronization summing to more than the kernel
  saving at the production workload size;
- unsafe hidden state, unclear ownership, or unsafe concurrent use;
- cold-start cost that the product's regime actually pays.

**A local kernel speedup alone can never produce GO.**

## Ending a failed assessment cleanly

Name the failed gate and freeze the evidence behind it, then:

1. **Do not weaken the contract.** Do not drop mismatching cases or relabel a
   nonsmooth gradient error without an explicit product decision.
2. **Do not broaden the implementation to rescue sunk cost.** A full rewrite is
   not justified because a narrow prototype lost.
3. **Leave every integration seam reverted or disabled**, and keep only assets
   that stand on their own — a correctness test, a benchmark harness, an
   improved baseline.
4. **Record a re-entry condition**: what would have to change for this to be
   worth revisiting.
5. **Stop spending benchmark time.** Once a required semantic gate fails or a
   stronger algorithm wins, more tuning cannot change the decision.

## Assembling the report

Copy [../assets/warp-adoption-report-template.md](../assets/warp-adoption-report-template.md)
into `warp-adoption-report/warp-adoption-report.md` and fill it in. The user may
name a different output directory.

The deliverable is the **versioned directory schema** below, not only a Markdown
outline:

```text
warp-adoption-report/
├── warp-adoption-report.md
├── solutions/
├── benchmarks/
│   └── README.md
└── results/
```

Keep the report's schema marker, numbered section titles and order,
required-table markers, and required column names unchanged. Add case-specific
explanation as prose or `###` subsections under the nearest numbered section. A
specialised sweep table may supplement a required table; it never replaces one.
This keeps reports machine-checkable and lets a reader compare projects without
relearning the layout.

### Hard rules

- **Record authorization.** A report directory exists only after the user
  explicitly approved the stage-1 scope/resource preview. The header records
  the authorizing response and approved scope; experiments outside that scope
  require a later authorization that is recorded too.
- **Declare assessment depth and summary derivation.** Use `full`,
  `profile-only`, `correctness-only`, or `inconclusive-minimal` so sentinel-heavy
  reports are not mistaken for completed benchmark studies. The report-level
  summary is a synopsis of §12: state whether it comes from one decision,
  unanimous decisions, a named dominant regime, or mixed decisions. If §12
  disagrees across seams or regimes, the summary is `MIXED` unless the user
  explicitly named one dominant regime.
- **Keep the five pre-registration facts together in §4.1.** Workload
  provenance, state variable and production range, tuning knobs, baseline
  spread, and the both-sides oracle plan each retain a row even when the value
  is missing. Do not scatter them across narrative sections.
- **Use stable row IDs.** Candidates are `C1`, `C2`, ... and experiments are
  `E1`, `E2`, ... within a report. Reuse those IDs in the candidate matrix,
  correctness results, optimization ledger, memory results, executive summary,
  decisions and evidence table. Do not identify the same seam by unrelated
  prose in each section.
- **The optimization ledger is exhaustive.** Give the reference baseline and
  every implementation, formulation, tuning variant and negative control that
  was actually timed one row at the decision workload. This includes failed,
  incorrect, superseded and non-Warp alternatives; mark them `REJECT`,
  `INCONCLUSIVE`, or `EXCLUDE INVALID` rather than hiding them in prose. Use one
  row per candidate × implementation × decision workload × execution regime,
  splitting rows when the verdict changes. Detailed sweeps may follow the
  ledger.
- **Every tested code solution has an independent diff.** Store one unified
  `.patch` under `solutions/`, generated against the repository/commit named in
  §4. The patch must include new source files, apply independently to that clean
  baseline, and reproduce the implementation measured. It must not contain the
  report bundle, benchmark harness, raw results, caches, generated binaries,
  credentials or unrelated changes. Several experiment rows may reference the
  same patch when one solution was measured across workloads or regimes. Use
  `Solution type = configuration` and `Solution diff = n/a` only when the
  solution changes no target source; the exact runtime/configuration change must
  then be present in the linked benchmark entry point. The reference baseline
  also uses `n/a`.
- **Ship the exact reproducibility assets.** Copy every setup, benchmark,
  correctness and orchestration script required by the accepted commands into
  `benchmarks/`; do not rewrite a cleaner version after measurement. Document
  prerequisites, patch-application steps and literal commands in
  `benchmarks/README.md`. Put raw sample arrays and logs under `results/`.
  External datasets remain external and are identified by provenance rather
  than copied. Never include credentials or host-specific absolute paths.
- **Ledger artifact paths are binding.** `Solution diff`, `Benchmark entry
  point`, and `Raw result` are relative to the report directory and must exist.
  A rejected, superseded or invalid timed solution still keeps its diff and raw
  result so the decision is auditable.
- **Every optimization-ledger row records correctness, time, memory and the
  decision.** Use milliseconds and MiB. Name the timed boundary and the memory
  accounting domain. Measure both decision-boundary time and
  decision-relevant peak memory for every runnable, semantics-valid
  implementation. If either was not collected, use the standard
  missing-evidence token instead of deleting the column or moving the attempt
  to another table; that row cannot be `RECOMMEND`. A kernel-only timing can be
  recorded, but cannot stand in for an end-to-end result.
- **Use only the standard empty-evidence tokens:** `not measured` (in scope but
  not collected), `not available` (counter/tool absent), `no representative
  data` (workload absent), `unknown` (fact unresolved), and `n/a` (does not
  apply). These states are materially different and must not be collapsed into
  `—`.
- **The candidate matrix must account for the profile.** Every stage worth
  **≥10 % of the measured total** gets a row in the matrix with a status and a
  one-line reason. `rejected — gate C, dense NN layers already on vendor
  libraries` is a complete and good row. A stage that appears in the profile
  table and then never reaches the matrix is an *undocumented gap*, not a
  focused study: the reader cannot tell whether you screened it or missed it.
  Going deep on one seam is correct; leaving the other large stages
  unaccounted for is not.
- **Every quantitative claim states its source, workload, hardware and
  measurement context.** A number without these is not admissible.
- **Empty evidence stays visibly empty.** Write "not measured", "not
  available", or "no representative data" — never a plausible-looking figure.
- **No invented benchmark values. No inferred speedup ranges. No generic
  promotional claims.**
- Label every candidate `rejected`, `hypothesis`, `measured`, or
  `recommended` — and keep the labels honest. A candidate you reasoned about
  but never ran is a `hypothesis`.
- Cite source locations as `path:line` for every claim about the codebase.
- Distinguish observed fact, measurement, hypothesis and unknown throughout.
- State non-goals explicitly, including any non-performance benefit you noticed
  but did not count because the user did not request it.
- Never present a ratio measured on another codebase, GPU or workload as a
  prediction for this one.
- Before delivery run
  `python scripts/validate_report_schema.py <report-directory>` from the skill
  directory, or the equivalent absolute path. Fix every schema error; do not
  waive one because the report reads well.

### Self-check before delivering

- Does the directory retain `warp-adoption/v2`, all four required paths, all 14
  numbered sections, and every required table with the exact columns from the
  template?
- Does the header identify the user's authorization and the scope it covered?
  Did any experiment exceed that scope without a recorded re-authorization?
- Does the header state assessment depth, and does the §2 summary say how it was
  derived from the per-seam/per-regime decisions in §12?
- Are all five pre-registration facts together in §4.1, with missing facts
  represented by standard tokens rather than omitted?
- Does the optimization ledger contain **every tested optimization**, including
  baseline, non-Warp, rejected, incorrect and superseded variants, with a quick
  decision note?
- Does every non-baseline timed solution link to an independent patch that
  applies to the pinned baseline, and do repeated measurements of one solution
  deliberately reuse that path?
- Does every ledger row link to the exact benchmark entry point and raw result
  shipped in the directory?
- Does each ledger row include correctness, decision-boundary time and
  decision-relevant peak memory, or one of the standard missing-evidence
  tokens? Are invalid implementations marked `EXCLUDE INVALID`, so an
  attractive time cannot leak into the recommendation?
- Do `C*` and `E*` IDs refer to the same candidate and experiment throughout?
- Does the report show the **whole measured surface**, or only the seam you
  chose? A reader must be able to see what else was measured and what was
  screened, with the gate named.
- Is every stage quoted as a device path **confirmed** to have run on the
  device, rather than assumed from the input's residency?
- Was the candidate compared against the project's own accelerator backend
  running the same algorithm, and not only against its CPU path?
- Does every stage ≥10 % of the measured total appear in the candidate matrix
  with a status and a reason? If not, add the rows — including the rejections.
- Was each gate applied against the metric that candidate's pattern exhausts,
  rather than the study's headline objective?
- Does any number lack a source, workload, hardware and context? Remove or
  qualify it.
- Does any section imply a measurement that was not taken?
- Is there a benefit counted as upside that the user never named as an
  objective?
- Would a reader mistake a hypothesis for a result?
- Is the verdict per seam **and per execution regime**, with its envelope and
  fallback stated?
- Was the workload **captured from a live run**, or assembled? If assembled, is
  that stated where the numbers are, not only in the assumptions?
- For a relational operation, was the **joint** distribution of the inputs
  checked, and the state variable it turns on swept across its production range?
- Did the candidate lose anywhere on **unset defaults**? If so, that comparison
  is not admissible — tune it or withdraw it.
- Was the incumbent's **own run-to-run spread** measured before any equality
  tolerance was fixed, and is every implementation scored against the oracle?
- Is every end-to-end number **repeated at least twice with the order
  alternated**, and the first run of the session discarded?
- Is the **ceiling** stated, and each speedup quoted as a fraction of it?
- Does any test assert that a hazard *reproduces*? Replace it with a
  characterised frequency.
- Does any unmet gate appear only as a caveat beside a positive verdict, rather
  than constraining it?
- Does the report end at a recommendation, without production changes?
