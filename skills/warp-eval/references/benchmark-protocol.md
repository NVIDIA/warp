# Benchmark protocol

Use only after the user authorizes the measured evaluation. Follow this and
**document every deviation** rather than presenting a weaker protocol as if it
were this one.

Measure through [../scripts/measure.py](../scripts/measure.py), called from one
driver script per bottleneck, rather than hand-rolling timing and memory code:
it implements the rules below, and two evaluations that both "followed the
protocol" by hand are not comparable. Transcribe every table cell from the JSON
it emits; a cell with no record reads "not measured".

## Exclusive execution

Hold one exclusive device lock for a **complete measurement sweep**, not one
sample at a time:

```bash
export GPU_LOCK="${GPU_LOCK:-/tmp/warp-evaluation.gpu.lock}"
flock -x "$GPU_LOCK" -c '<the whole sweep>'
```

- After acquiring the lock, record GPU model, driver, free and used memory,
  utilization and a timestamp.
- **Do not acquire the lock and then wait for 0 % utilization.** Compositors,
  remote-display services and sampling noise can keep it nonzero indefinitely,
  and waiting inside the lock blocks every other queued task. Check for idle
  *before* acquiring, with a finite timeout, then continue or skip. Proceed
  unless the compute-process list shows an unrelated workload or free memory is
  insufficient.
- **Contention distorts comparisons asymmetrically:** a busy device penalizes
  long kernels far more than short ones, which flatters whichever side is
  slower. Re-measuring on an idle device routinely moves baseline medians by
  double-digit percentages and *reduces* quoted speedups. If you cannot get a
  quiet device, say so and mark the numbers.

## Warp device requirement

Run Warp correctness tests, profiles and benchmarks only on an explicitly
selected NVIDIA CUDA device. Resolve the selected device through Warp, verify
that it is CUDA and record its alias and architecture in the raw artifact.
Discard any Warp artifact that resolves to `cpu`. CPU remains valid for the
incumbent and non-Warp candidates; only Warp's CPU backend is excluded.

## Sampling

**You are gauging a ballpark, not producing a benchmark.** Report an order of
magnitude and direction, not a precise performance claim. Precision benchmarking
needs a quiet machine, other tenants killed, raised priority, pinned affinity,
locked clocks and disabled turbo; without those, more samples buy a tighter
estimate of a contaminated number.

`measure.time_it` implements this: one discarded warm-up, then **one** timed
run. Because Warp launches are asynchronous, it synchronizes the selected
device immediately before starting and after enqueueing the timed work, before
stopping the wall timer. `measure.sig2` rounds to two significant figures — a
single sample does not support "1.87×"; it supports "about 1.9×", often only
"about 2×". `measure.env_fingerprint` records the CPU thread count.
What it cannot do for you:

- **Do not raise `samples` to tighten a reported figure.** The default is a
  protocol choice, not a placeholder. Keep the discarded warm-up; skipping it
  can move a baseline enough to invert a comparison.
- Reach for `measure.device_time` **in addition to** wall time where wrapper or
  launch behavior matters; never compare device-event time against synchronized
  end-to-end time.
- For JAX, disable allocator preallocation
  (`XLA_PYTHON_CLIENT_PREALLOCATE=false`) so memory figures mean something.

### Comparison resolution

`measure.comparison` reports the ratio and whether it clears the protocol's
resolution: **at least 1.5×** is a directional difference and remains
approximate; **below 1.5×** is *no measured difference at this precision*.
Neither label is an adoption judgment.

If the user's own criterion turns on distinguishing 1.2× from parity, say the
evaluation cannot resolve it in a shared environment and state the controlled
measurement needed.

**Spend spare budget on coverage, not repetition** — another size, regime or the
cold path is worth far more than a second sample. One exception: if two variants
land close enough that the order matters and the decision depends on it, run
those two once more with the order reversed.

### Prove the harness runs the work

A harness that silently fails to run reports a very fast time, which reads like
a speedup: a kernel never launched, a result served from cache, a missing
synchronization, a test binary that exits 0 without executing.

Before trusting a bottleneck's timings, scale the work and confirm the
measurement scales with it (`measure.null_test`): 4× the work should cost about
4×. Report the observed ratio next to the timings. A measurement that does not
respond to the size of the work is not measuring the work.

## Required timing phases

Measure and report separately, as applicable:

1. process and framework import/initialization;
2. fresh-cache compilation;
3. populated-cache module load in a **fresh process**;
4. first launch and first acceleration-structure construction;
5. warm build, refit, and rebuild;
6. warm direct kernel/query time;
7. wrapper validation, allocation, conversion, interop, transfer, compaction and
   synchronization;
8. graph capture and replay, **including recapture count**;
9. the end-to-end public call;
10. the immediate downstream stage or a full application iteration;
11. measured totals for 1, 10 and 100 calls — measured loops where possible, not
    a single median multiplied.

For one-shot tools, process startup and cold compile are part of the measured
boundary. For persistent services, still report cold recovery and cache-miss
behavior.

```text
total(N) =
    process/import initialization
  + cold compile or cached module load
  + data conversion and validation
  + data-structure build or refit
  + graph capture or recapture
  + N * (interop + transfer + launch + kernel + sync + downstream work)
```

Report the crossover from **measured totals** as a bracket. Never derive
production policy from kernel-only ratios.

### The ceiling

The ceiling is what the whole user-visible stage would cost if the candidate
stage became free, and the ratio that implies. Quote every achieved speedup as a
fraction of it: 87 % of the ceiling is finished work, 20 % says the wrapper is
eating the saving.

- **Measure the floor, never subtract it.** Run the pipeline with the stage
  stubbed and time what is left, rather than subtracting isolated per-op timings
  from a total. The two disagree, and subtraction can produce a "ceiling" the
  real result then exceeds.
- **The ceiling bounds backend substitution only.** A change of representation
  or data structure removes cost elsewhere too, so it can legitimately beat the
  ceiling. If a result exceeds it, say which of the two is happening.

## Memory accounting

Measure in **fresh processes** when allocators retain storage, and name the
domain for every number: live/allocated bytes; allocator-reserved or pool bytes;
peak temporary storage; process-level GPU memory; acceleration-structure
storage; retained outputs and autograd state; graph-private pools and
graph-cache growth; framework preallocation policy; host RSS where host paths or
copies matter.

`measure.host_memory` and `measure.device_memory` take the headline figures from
the two domains that see both sides — process RSS via `/proc` `VmHWM`, and NVML
per-process GPU memory — and `measure.run_isolated` gives each case its own
process. What that mechanism cannot decide for you:

- **A comparison needs a domain that sees both sides.** When the candidate's
  cost lands outside the incumbent's allocator — a device context, a native
  library, a separate process — a per-language counter structurally understates
  it. Python's `tracemalloc` cannot see a CUDA context at all, and a framework
  pool counter can report a bare CUDA context of hundreds of MiB as **0**.
  Framework counters supplement — only they split live from pool-reserved — but
  never replace, and labelling the domain does not fix a blind instrument.
- **Never substitute `ru_maxrss` for `VmHWM`, or one process for two.**
  `ru_maxrss` is inherited across a spawn, so a child reports the parent's peak
  before it allocates anything; peak counters are monotonic *within* a process,
  so two configurations timed in one process are one measurement.
- **Report a missing counter as missing.** Per-process NVML is unavailable
  inside MIG and under some container and driver permissions; a counter absent
  from your version is reported absent, with the conservative process-level
  figure beside it. **Do not invent it.**
- NVML has 1 MiB granularity — a zero delta means "below snapshot resolution",
  not "zero storage". Neither counter polls, so a device-side transient shorter
  than your sampling can still be missed; if you poll for one, state the
  achieved interval, because a sampled figure is not a high-water mark.

## Results do not transfer across GPU architectures

A kernel-level optimization can differ by more than 2× between NVIDIA
generations and occasionally invert — a control-flow change that lets one
architecture unroll a loop can raise register pressure on another. Anything
depending on unrolling, register budget, occupancy, shared memory or tile shape
is architecture-sensitive by default. State the GPU the numbers came from and do
not project them onto another; if production spans several, measure each or
scope the measurements.

## Thread budget

**A CPU baseline measured below the deployment's thread count has not been
compared.** State the thread count of every timed configuration and the core
counts of both the measuring host and the deployment target — the difference
bounds the comparison. Three failure modes, all observed:

- The incumbent exposes parallelism nothing switches on. A shipped `n_threads=1`
  default on a many-core host is an under-parallelized baseline, not a slow
  algorithm.
- The measuring host is shared, so you cap threads to be a good tenant — biasing
  every CPU-versus-device ratio *toward the device*.
- The measuring host has *more* cores than the deployment target, biasing the
  other way.

If you cannot run the baseline at the deployment thread count, measure its
scaling over the counts you can reach, report the achieved count next to every
ratio, and mark the strongest-baseline comparison `not measured` at the
deployment thread count. **Do not extrapolate the scaling curve into a predicted
ratio.**

## Sweeps, crossovers and scale

**Relative performance is a function of scale, not a property of the
implementations.** Fixed per-call costs — context creation, transfer, structure
build, launch — dominate small inputs, while asymptotics and memory traffic
decide large ones, so direction routinely inverts somewhere in between. A
comparison at one size establishes only that size.

- **Sweep across and beyond the expected crossover, at decades of scale —
  10³, 10⁶** — for the incumbent, each in-project alternative and each candidate
  alike: a route dismissed at one size has been dismissed at one size. Sweep
  problem size, density, clustering, batch count, capacity, reuse level,
  cold/warm regime, build vs refit vs rebuild and CPU thread count.
- **Sweep the operation's own variants** — which reduction, dtype, mode,
  interpolation or combination rule the public API exposes. Variants routinely
  differ several-fold and can run through entirely different machinery (a
  reduction needing a per-pixel lock, a mode allocating output proportional to
  category count, an option that silently falls back), so measuring the cheapest
  one understates the surface you would have to reproduce.
- **Name every axis the size runs along, and sweep each of them.** Relational
  and geometric operations have at least two — the query set *and* the reference
  structure it runs against (points versus triangles in a mesh, probes versus
  cells, rows versus index size). Sweeping the convenient axis while holding the
  other fixed measures one axis and *assumes* the other. Include any axis the
  candidate's numerics turn on, such as coordinate magnitude for a float32 seam
  behind a float64 API.
- **Report crossovers and inversions as measured brackets** ("between 100 and
  250 queries"), not interpolated points or endpoints alone. Where a candidate
  is faster throughout, state the measured range; "faster" without a size is not
  a result. Say where the production distribution sits relative to the
  crossover, and if the candidate becomes faster only above the typical
  production range, state both ranges.
- **Watch the metrics separately** — they need not invert together. A candidate
  can be faster at every size while using *more* memory below some size, because
  a fixed device context is a floor the incumbent never pays.

**If the project ships no asset at a size you need, generate one and label it
generated.** Prefer its own assets where they exist and state the rule by which
you selected them, so a second run of this evaluation picks the same ones.

## Representative and adversarial data

Synthetic sweeps isolate variables; they do not establish production behavior.
Add representative real data before comparing production behavior. If real data
was unavailable, say so and name the procedural substitute.

- **Real inputs are not a representative workload.** For a **relational**
  operation — nearest neighbor, closest point, overlap, matching, collision —
  cost depends on the **joint** distribution of the inputs, not on either
  separately. Two real production assets paired with each other can still be a
  workload the product never sees: pair a point set with an unrelated shape
  instead of the one it converges toward, and every query falls through to the
  exact fallback, which can invert the result. So **capture the tensors the
  application actually passes**, from a live run — instrument the loop, dump the
  arguments at several points, benchmark those. Assembling plausible inputs is a
  last resort that must be declared in the report.
- **Sweep the state variable, not just the size.** Relational costs usually turn
  on a ratio rather than N — query-to-reference distance over reference spacing,
  occupancy, clustering, overlap fraction, convergence error — and in an
  optimization or simulation loop that ratio moves by orders of magnitude as the
  run proceeds, so one snapshot measures a single point on a curve. Identify the
  ratio, measure where it starts and ends in production, and report across that
  range. If the candidate needs a parameter derived from that ratio, an
  implementation that adapts it from data the caller already has is part of the
  candidate — hand-picking the best value per measurement point is tuning on the
  test set.
- **Check every synthetic workload against the operation's own cost model.**
  Write down what the work should scale with — triangles times the area each
  covers, non-transparent pixels, points per occupied cell, segments times
  pixels traversed — and confirm the measurement matches. A figure orders of
  magnitude off usually indicts the workload: a mesh whose triangles each span
  the canvas measures fill rate, not mesh rasterization; a fully dense image
  measures a path real sparse input never takes. A degenerate workload invents a
  bottleneck; an unrepresentatively easy one hides a real one.

## Record for reproduction

The delivered directory is the reproduction boundary. **Never reconstruct a
script or raw result from memory after the run.** Beyond the obvious — pinned
commit, environment freeze, literal commands, no credentials or external
datasets — five things matter more than they look:

- **run-unique raw-result filenames** (`results/bench-<sweep>-<runid>.json`) —
  a second lock acquisition that re-runs a phase will otherwise silently
  overwrite files the report already cites;
- **all raw samples**, with the device state they were taken under;
- **full arrays or compressed states** wherever an aggregate could hide a rare
  mismatch;
- **memory time series with the achieved polling interval**;
- **the cache directory, or the procedure for clearing it** — without it no
  cold-start number is reproducible.

## Evidence integrity

If an artifact is lost, overwritten, or produced under a protocol you cannot
defend: **delete the claim, mark the artifact deliberately absent, and say
why.** Do not restate a number from memory or fabricate sample arrays. Dropping
an entire timing table because its raw results were clobbered is the correct
response. A missing measurement is reported as missing — that is what `not
measured` is for.
