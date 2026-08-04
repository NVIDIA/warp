# Authorization checkpoint

Mandatory once stage 1 finds at least one candidate pattern, and before any
measured or experimental work.

**Allowed before the checkpoint:** reading source, documentation, configuration
and existing logs; identifying candidate areas; deriving the proposed contract;
checking the cheap gates.

**Not allowed:** running profilers or workloads, installing or building
dependencies, creating environments or harnesses, acquiring a GPU lock,
prototyping, benchmarking, or creating the report directory.

## What to report

One concise early-findings message:

1. **Framing** — Warp is only the option being evaluated; the skill will report
   evidence without an adoption recommendation.
2. **Observed facts** — candidate areas with `path:line`, the pattern each
   matches, existing device/backend facts, and which cheap gates you checked.
3. **Hypotheses and unknowns** — inferred objective and execution regime,
   unmeasured bottleneck hypotheses, and anything that could end the evaluation
   early. **If the environment gates are unresolved, that question rides here**,
   phrased per *Asking about intent* in the stage-1 gate reference named by
   `SKILL.md`, with the consequence for whether the evaluation continues stated
   for each branch. It is blocking.
4. **Proposed next steps** — the entry points and workload to profile; time and
   memory measurements; semantic contract and oracle; improved incumbent and
   non-Warp baselines; the minimum Warp prototype if no gate fires; correctness
   audit; complete-boundary Warp benchmark; the report directory. State that
   these are one evaluation scope, not independently selectable branches.
5. **Resource preview** — required datasets, environment and dependency work,
   expected CPU/GPU use, number and breadth of sweeps, and likely wall-clock,
   storage or token cost. Use `unknown until <probe>` rather than inventing an
   estimate.

Then ask — **using `AskUserQuestion` where available**, not buried in prose.
Keep the environment question (if unresolved) and the scope approval as separate
structured choices, so packaging conditions cannot be mistaken for permission to
omit Warp. Use the intent choices in the stage-1 gate reference, then ask:

> Do you want me to run this scoped measured Warp evaluation? A yes authorizes
> the real-application profile, strongest in-project baseline, minimum Warp
> prototype and end-to-end Warp benchmark described above as one scope. If a
> gate fires I will abort immediately; if scope or resource cost grows
> materially I will stop and ask again.

Only an explicit approval after this preview starts stage 2. The original
performance request, an explicit skill invocation, and silence do not count.
Approval to profile or improve only the incumbent does not start stage 2; leave
`warp-eval` instead.

## Outcomes

| Response | What follows |
|---|---|
| Approved | Record the approved scope and the authorizing response in the eventual report, then begin stage 2 |
| Approved, environment question unanswered | Stage 2 does **not** start. "Go ahead" authorizes the *work*; it does not say the product may require an NVIDIA GPU or take the dependency. Ask again in one line and wait |
| Environment question answered no | `ABORT`. No report directory; cite the stated constraint, whatever the scope answer was |
| Narrowed, Warp still included | Restate the revised Warp profile/prototype/benchmark scope and obtain explicit approval for it |
| Narrowed to incumbent-only work | `ABORT` before profiling; that request belongs outside `warp-eval` |
| Declined or no response | Stop. No report directory. This is **not** `ABORT`: Warp was plausible, but the evaluation never ran |
| Scope or cost grows later | Stop at the discovery point, explain the change, and obtain fresh approval before continuing |
