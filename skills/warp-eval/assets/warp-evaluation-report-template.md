**Report schema:** `warp-evaluation/v1`
# <project> Warp evaluation evidence — <subsystem>

> Place at `<output-directory>/warp-evaluation-report.md`, with sibling
> `solutions/`, `benchmarks/` and `results/` directories. Keep the schema
> declaration, section headings and table columns exactly as written.
> Delete these quoted instruction lines before delivering.
>
> **The report is evidence, not a recommendation.** Show observed facts,
> measurements, hypotheses and unknowns per seam and execution regime. Do not
> issue an aggregate opinion, rank solutions, or tell the user what to adopt.
>
> Use exactly **"not measured"** (in scope, not collected), **"not available"**
> (the counter or tool does not exist), **"no representative data"**,
> **"unknown"**, or **"n/a"**. Never substitute a plausible number.

**Commit** `<sha>` · **Date** `<YYYY-MM-DD>` · **Status** draft, nothing applied, production code unmodified.

**Authorization:** <the user's own words on NVIDIA deployment, dependency
obligations and approved experimental scope.>

**Evaluation state:** complete / aborted — <gate and scope> / incomplete — <missing evidence and scope>

<Two to five sentences stating what was evaluated, which regimes were measured,
the largest observed differences and the material evidence gaps. No overall
opinion or recommendation.>

---

## Where the time goes

> Every user-facing entry point considered in stage 1. Surviving entries are
> measured at one representative scale. The **ID** column names each bottleneck
> and is reused verbatim as the section heading below. Screened-out stages
> remain in this table with the gate and observed fact that stopped them.
>
> Figures here must be the same measurement used as that bottleneck's baseline
> below. If two harnesses timed the same call, reconcile them in a note rather
> than printing both.

<N> operations, one call each, <workload>. Source: `results/<file>`.

| ID | Operation | Time | Peak mem | Evaluation status |
|---|---|---|---|---|
| **B1** | | | | **§ B1 below** |
| — | | | | screened out — <gate and observed fact> |

---

## B1 — <name> (`path:line`)

**Cause.** <What the incumbent does that costs this, in terms of algorithm,
representation, launch count or memory. State what the cost scales with and
confirm the measurements matched it.>

**Workload:** <sizes, distribution, thread or core count>. One warm-up
discarded, one timed run. Rows are ordered by scope: baseline, routes reachable
through current dependencies, then Warp.

> Every measured attempt gets a row, including slower implementations and
> contract failures. Ratio cells use only `N× faster` / `N× slower` / `N× less`
> / `N× more` / `same`. Memory cells name their domain inline
> (`386 MiB host / 13.9 MiB device`).
> The row names the exact public method and variant invoked. A helper, fallback,
> sibling operation or different option set cannot stand in for another
> contract.

| Solution | Time | vs baseline | Peak mem | vs baseline | Evidence |
|---|---|---|---|---|---|
| **Baseline** — current code | | — | | — | |
| <in-project solution> | | | | | <measured facts, contract status, artifact> |
| **Warp** <seam> *(new dep: warp-lang + CUDA)* | | | | | `solutions/<file>.patch`; <measured facts> |

<Any capacity boundary stated plainly: the size at which each implementation
stops fitting the stated memory budget and what becomes unreachable.>

### Comparison by regime (when measurements vary)

> Include when measurements differ by regime, size, distribution or hardware.
> Cold costs belong here as a one-shot regime row.

| Regime | Baseline | Warp | Time difference | Baseline mem | Warp mem | Memory difference |
|---|---|---|---|---|---|---|
| | | | | | | |

<One or two sentences explaining the measured variation, including any regime
where direction changes.>

---

## Correctness (required for prototyped candidates)

**Oracle:** <the independent reference, and confirmation that every
implementation including the incumbent was scored against it>.
**Tolerances fixed before observing candidate output:** yes / no.
**Suite proven able to fail:** <the deliberate breaks and that they were caught>.

**Both sides:** candidate corrected <N₁> incumbent outputs and regressed <N₂>
contractual outputs. <A bare "candidate differs from incumbent on N" is not
admissible.>

<For each failed required criterion, name the scope that aborted, the exact
failure and its source. Do not prescribe a response.>

---

## Caveats that bound these numbers

> One line each. Required: workload provenance; the state variable and its
> production range; any in-project route left unmeasured; memory and core budget;
> measurements discarded and why.

- **Workload provenance:** captured from a live run / assembled — <if assembled, what that prevents establishing>.
- **State variable:** <the ratio cost turns on, its production range and measured points>.
- **Unmeasured route:** <any in-project option not measured and the comparison left unavailable>.
- **Host and budget:** <cores, RAM, GPU and driver; direction of measurement bias>.
- **Numerics:** <precision differences behind the public API and their measured bound>.
- **Untested:** <lifecycle, concurrency, refit, streams, cache obligations>.
- **Measurements discarded, and why:** <every artifact produced under an indefensible protocol, or "none">.

---

## Environment and reproduction

<versions · resolved Warp CUDA device and architecture · GPU/driver/CUDA ·
OS/Python · cores · cache state · wall-timer synchronization boundary · whether
one exclusive device lock was held for the sweep>

Commands in `benchmarks/README.md`.

**Artifacts:** `solutions/` (<n> patches, each applies independently to a clean
tree) · `benchmarks/` (<n> scripts) · `results/` (<n> raw files).

**Nothing in this report has been applied.** The user decides whether to stop,
gather missing data or request implementation as a separate follow-up.
