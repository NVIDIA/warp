<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmark protocol

Use this protocol only after the user authorizes the scoped measured assessment
at the stage-1 checkpoint.

Follow this and **document every deviation** rather than presenting a weaker
protocol as if it were this one.

## Exclusive execution and environment capture

Hold one exclusive device lock for a **complete measurement sweep**, not one
sample at a time. A portable pattern that hard-codes nothing:

```bash
export GPU_LOCK="${GPU_LOCK:-/tmp/warp-assessment.gpu.lock}"
flock -x "$GPU_LOCK" -c '<the whole sweep>'
```

After acquiring the lock, record GPU model, driver, free and used memory,
utilization and a timestamp.

Treat the lock as the exclusivity gate. **Do not acquire the lock and then wait
for 0% utilization** — compositors, remote-display services and sampling noise
can keep it nonzero indefinitely, and waiting inside the lock blocks every
other queued task. If you want an idle check, do it *before* acquiring, with a
finite timeout, and then continue or skip rather than waiting forever.

Proceed unless the compute-process list shows an unrelated workload or free
memory is insufficient. Preserve raw samples and p10/p90 so environmental
variance stays visible.

**Contention systematically distorts comparisons**, and not symmetrically: a
busy device penalizes long-running kernels far more than short ones, which
usually flatters whichever side is slower. Re-measuring on an idle device
routinely moves baseline medians by double-digit percentages and *reduces*
quoted speedups. If you cannot get a quiet device, say so and mark the numbers
accordingly.

## Synchronization and sampling

- Synchronize the relevant stream or device **before and after every timed
  region**.
- At least **5 warmups** and **20 retained samples** for steady state, when
  practical.
- Report **median plus p10/p90**. Retain every raw sample.
- Distinguish device-event time from synchronized wall time when wrapper or
  launch behavior matters.
- **Never compare unsynchronized dispatch time with synchronized end-to-end
  time.**
- For JAX, disable allocator preallocation (`XLA_PYTHON_CLIENT_PREALLOCATE=false`)
  so memory figures mean something.

Overlapping p10/p90 bands mean "no measured difference", not "slightly
faster". Choose dispatch thresholds outside overlapping bands, not at the
crossing point of two medians.

## Required timing phases

Measure and report separately, as applicable:

1. process and framework import/initialization;
2. fresh-cache compilation;
3. populated-cache module load in a **fresh process**;
4. first launch and first acceleration-structure construction;
5. warm build, refit, and rebuild;
6. warm direct kernel/query time;
7. wrapper validation, allocation, conversion, interop, transfer,
   compaction and synchronization;
8. graph capture and replay, **including recapture count**;
9. the end-to-end public call;
10. the immediate downstream stage or a full application iteration;
11. measured totals for 1, 10 and 100 calls — measured loops where possible,
    not multiplication of a single median.

For one-shot tools, process startup and cold compile are part of the decision.
For persistent services, still report cold recovery and cache-miss behavior.

### Cost model

```text
total(N) =
    process/import initialization
  + cold compile or cached module load
  + data conversion and validation
  + data-structure build or refit
  + graph capture or recapture
  + N * (interop + transfer + launch + kernel + sync + downstream work)
```

Compute the crossover from **measured totals**, then pick a conservative
dispatch threshold. Never derive production policy from kernel-only ratios.

## Memory accounting

Measure in **fresh processes** when allocators retain storage. Name the domain
for every number:

- live/allocated bytes;
- allocator-reserved or pool bytes;
- peak temporary storage;
- process-level GPU memory;
- acceleration-structure storage (BVH, hash grid, winding data);
- retained outputs and autograd state;
- graph-private pools and graph-cache growth;
- framework preallocation policy;
- host RSS where host paths or copies are material.

Rules:

- **One framework's allocator counters do not include another framework's
  allocations.** External allocations show up only in process-level memory.
- If a counter does not exist in the version you are using, report the
  conservative process-level figure and say the counter was unavailable.
  **Do not invent it.**
- Process memory sampled by polling can miss a transient shorter than the
  interval. State the achieved polling interval; do not call a sampled figure a
  driver high-water mark.
- `nvidia-smi` has 1 MiB granularity — a zero delta means "below snapshot
  resolution", not "zero storage".

## Results do not automatically transfer across GPU architectures

A kernel-level optimization can differ by more than 2x between NVIDIA
generations, and occasionally invert: a control-flow change that lets one
architecture unroll a loop can raise register pressure on another. Anything
depending on unrolling, register budget, occupancy, shared-memory capacity or
tile shape is architecture-sensitive by default.

State the GPU the numbers came from, and do not project them onto a different
one. If production spans several architectures, either measure each or scope
the verdict to the one you measured.

## Crossovers and sweeps

Sweep across **and beyond** the expected crossover: problem size, density,
clustering, batch count, capacity, reuse level, cold/warm regime, build vs
refit vs rebuild, and **the operation's own variants** — which reduction,
dtype, mode, interpolation or combination rule the public API exposes. Report
where the production distribution actually sits relative to the crossover — a
win that starts above your users' typical input is a NO-GO for them.

Variants of one entry point routinely differ several-fold and can run through
entirely different machinery: reductions that need a per-pixel lock, modes that
allocate an output proportional to category count, options that silently fall
back. Measuring the cheapest variant and calling it the entry point's cost
understates the surface you would have to reproduce.

Report crossovers as brackets you measured (for example "between 100 and 250
queries"), not as interpolated points.

## Representative and adversarial data

Synthetic sweeps isolate variables; they do not establish production
behavior. Add representative real data before any positive verdict. If real
data was unavailable, **say so and name the procedural substitute** you used
instead, so a reader can judge how far the result generalizes.

### Real inputs are not a representative workload

For a **relational** operation — nearest neighbour, closest point, overlap,
matching, collision — the cost depends on the **joint** distribution of the
inputs, not on either input separately. Two real, shipped, production assets
paired with each other can still be a workload the product never sees, and the
cost difference is not marginal: pairing a point set with an unrelated shape
instead of the one it is converging toward moved a measured result from *2×
slower than the incumbent* to *2.8× faster*, because every query fell through
to the exact fallback. The inputs were real; the relationship was invented.

So: **capture the tensors the application actually passes the operation, from a
live run of it.** Instrument the loop, dump the arguments at several points, and
benchmark those. Assembling plausible inputs is a last resort that must be
declared in the report, not a default.

### Sweep the state variable, not just the size

Relational costs usually turn on a ratio, not on N: query-to-reference distance
over reference spacing, occupancy, clustering, overlap fraction, convergence
error. In an optimisation or simulation loop that ratio **moves by orders of
magnitude while the run proceeds**, so a single snapshot measures one point on a
curve and calls it the answer.

Identify the ratio, measure where it starts and ends in production, and report
the result **across that range**. One study found the same seam ranged from
1.4× to 16× over a single training run; quoting either end alone would have been
wrong, in opposite directions. If the candidate needs a parameter derived from
that ratio, an implementation that adapts it from data the caller already has is
part of the candidate — hand-picking the best value per measurement point is
tuning on the test set and must not be quoted as a result.

**Check every synthetic workload against the operation's own cost model before
you trust its number.** Write down what the work should scale with — triangles
times the area each covers, non-transparent pixels, points per occupied cell,
segments times pixels traversed — and confirm the measurement matches that
expectation. A figure that is orders of magnitude off usually indicts the
workload, not the code: a mesh whose triangles each span the whole canvas
measures fill rate rather than mesh rasterization, and a fully dense image
measures a path that real sparse input never takes. Both directions of error
are dangerous — a degenerate workload invents a bottleneck that does not exist,
and an unrepresentatively easy one hides the bottleneck that does.

Where the cost driver is a property of the data rather than its size — density,
overlap, cardinality, clustering — sweep *that* and report the range, rather
than picking one value and quoting it as the operation's cost.

## Record for reproduction

The delivered report directory is the reproduction boundary. It contains the
exact scripts used under `benchmarks/`, independently applicable solution diffs
under `solutions/`, and raw machine-readable outputs under `results/`. Commands
in the report and `benchmarks/README.md` are relative to that directory and
state the pinned target commit, environment prerequisites, which solution patch
to apply in a disposable checkout, and the expected result file.

Do not copy credentials, caches, generated binaries or external datasets.
Reference datasets by stable provenance and document how the scripts locate
them. Never reconstruct a script or raw result from memory after the run.

Commit, environment freeze, literal commands and raw samples are the baseline.
Four less obvious ones matter as much:

- **all raw samples, not just summaries** — a median hides the variance that
  decides whether two distributions actually differ;
- **full arrays or compressed states** wherever an aggregate metric could hide
  a rare mismatch;
- **memory time series with the achieved polling interval**, since a sampled
  peak is not a driver high-water mark;
- **the cache directory or the procedure for clearing it**, without which no
  cold-start number can be reproduced.

## Evidence integrity

If an artifact is lost, overwritten, or produced under a protocol you cannot
defend: **delete the claim, mark the artifact deliberately absent, and say
why.** Do not restate a number from memory and do not fabricate sample arrays.
Dropping an entire timing table because its raw results were clobbered is the
correct response; so is recording that specific probe files were not retained
and were not reconstructed, while preserving only the aggregates that remain
defensible.

A missing measurement is reported as missing. That is what `INCONCLUSIVE` is
for.
