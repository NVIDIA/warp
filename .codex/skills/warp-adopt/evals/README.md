<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# warp-adopt evals

Eval cases for the `warp-adopt` skill, in the layout described by
[agentskills.io](https://agentskills.io/skill-creation/evaluating-skills):
test cases live in `evals.json`, fixtures in `files/`, and run artifacts go in
a sibling workspace directory (`../warp-adopt-workspace/iteration-N/`) that is
**not** checked in.

Each case is a minimal repro of a way this skill has been observed to go wrong
on a real assessment. They are judgement tests, not code-generation tests: the
skill's output is a decision and a report directory, so the assertions grade
*what the report concluded and what evidence it used*.

Every non-`ABORT` output also has one deterministic structural assertion: it
must pass `scripts/validate_report_schema.py`. The per-case assertions below
continue to grade whether the report's evidence and decision are sound; schema
conformance does not substitute for those judgements.

## Authorization-checkpoint protocol

Run every non-`ABORT` output eval as a two-turn interaction. The first response
must stop after static screening, report early findings and the scoped
work/resource preview, and ask for explicit authorization. It must not profile,
install, prototype, benchmark or create the report directory. Then reply:

> Yes. Proceed with the scoped measured assessment you described, including the
> minimum prototypes and benchmarks. Stop and ask again if scope or cost grows.

Only grade the case-specific report assertions after that follow-up. Case 11
must `ABORT` immediately and never reaches the checkpoint.

## What each case targets

| # | Fixture | Failure mode it catches |
|---|---|---|
| 1 | `e1-non-bottleneck` | Porting a Warp-shaped seam that is not a bottleneck. The user even asserts the wrong culprit; the skill must profile and contradict them. |
| 2 | `e2-memory-candidate` | Screening a memory/capacity candidate on its *latency* share. Also: library default ≠ CLI default, and judging materiality against the wrong device budget. |
| 3 | `e3-profile-coverage` | Going deep on one seam and silently dropping the other stages above 10%, so a narrow study reads like a survey. |
| 4 | `e4-entry-points` | Profiling one entry point and screening the rest by unstated assumption; ignoring the project's own profiler. |
| 5 | `e5-broken-incumbent` | Treating a defective incumbent as the correctness oracle, so "% agreement" inverts and the *correct* candidate scores worst. |
| 6 | `e6-pruning-bound` | Benchmarking a pruning rule before proving it admissible — a plausible, fast, wrong answer that looks like the best variant. |
| 7 | `e7-transferable-opt` | Crediting a backend-neutral reformulation to Warp instead of applying it to the baseline first. |
| 8 | `e8-oneshot-regime` | Ranking on warm steady-state timings for a one-shot CLI that pays cold start on every invocation. |
| 9 | `e9-silent-host-fallback` | Quoting a stage as a device path when it silently converts to host arrays and runs serially — and missing a second option that is silently downgraded. |
| 10 | `e10-existing-gpu-backend` | Comparing a new backend against the *host* loop when the project already ships a device backend that would close the gap. |
| 11 | `e11-abort-cpu-only` | Assessing a workload whose *shape* fits perfectly on a target that has no GPU and forbids runtime compilation — the gate is the environment, and the correct output is an ABORT with no report directory. |
| 12 | `e12-roofline-ceiling` | Quoting an open-ended speedup for a stage that is already close to the floor for reading its own input, and missing that the stage with real headroom is too small to matter. |
| 13 | `e13-wrapper-overhead` | Crediting a backend with per-call wrapper cost that is flat in payload size and belongs to the incumbent. |
| 14 | `e14-degenerate-workload` | Drawing a conclusion from a synthetic workload that exercises a different cost regime than the shipped one. Also: treating a data property as fixed instead of sweeping it and reporting the bracket where the conclusion changes sign. |
| 15 | `e15-unexpressible-contract` | Adopting a stock accelerated primitive that computes a *different quantity* than the documented contract — where the difference is invisible on the toy asset and universal on the shipped one, and frequent but small enough to slip past an absolute tolerance. |
| 16 | `e16-tie-dominated-index` | Judging an index-returning op at the wrong array level, and accepting a downstream check that is either vacuous or no longer contains the condition under test. |
| 17 | `e17-half-entry-point` | Quoting a seam's speedup as the user-visible gain when the public entry point is two symmetric halves and the structure accelerates only one. |
| 18 | `e18-zero-copy-aliasing` | Reporting a harness defect as a property of the backend. A zero-copy bridge whose refit writes through into caller storage passes every single-shot check and only fails across a loop. |
| 19 | `e19-portable-incumbent` | Pricing a vendor-specific kernel as a replacement when the incumbent is one source already built for several vendors, so the port is an *additional* implementation and a portability regression. |
| 20 | `e20-mature-but-wrong-algorithm` | Over-applying the mature-implementation gate. The incumbent is correct, tested and maintained — and still asymptotically wrong, so the win is algorithmic and available with no new dependency. |
| 21 | `e21-assembled-workload` | Benchmarking a *relational* operation on real inputs that are never paired in production. The inputs are genuine shipped assets; the relationship is invented, and it inverts the verdict. |
| 22 | `e22-nondeterministic-baseline` | Grading a candidate against a bitwise contract the incumbent itself cannot satisfy, and reading the resulting diff as a defect. |
| 23 | `e23-topology-dependent-ties` | Reporting a tie-mismatch rate without measuring how often the contested output is well-defined at all, so the same comparison "proves" 0 % or 16 % depending on the layout benchmarked. |
| 24 | `e24-untuned-candidate` | Concluding from a candidate running on an unset default. The knob's shipped heuristic is wrong for the data's intrinsic dimension, so the measured ratio describes the misconfiguration, not the approach. |
| 25 | `e25-adapter-launch-loop` | Reading an adapter's per-call host cost as a property of the algorithm or the backend. One call per batch example, a flat setup each time, and the verdict inverts once the loop is fused. |
| 26 | `e26-scale-sentinel-defect` | Missing an input-scale-triggered silent defect in the incumbent because the workload was never swept over scale — and then treating "our port would fix it" as upside for porting. |
| 27 | `e27-capacity-truncation` | Reporting a fixed-capacity operation's retained subset as a contract when the incumbent's own two backends already disagree about it, and never checking how often the cap binds or that no overflow flag exists. |
| 28 | `e28-gate-e-no-jit` | Assessing a workload on hardware that is present and suitable, for a project that cannot accept a runtime-compiled dependency. The gate is the obligations, not the device. |
| 29 | `e29-instrument-double-count` | Quoting a bundled profiler's absolute figures without checking them against a bound they cannot exceed. |
| 30 | `e30-library-no-application` | Ranking a *library's* seams from its own per-operation benchmark instead of from its shipped consumer, and treating one symbol as one seam when its call sites differ in dimensionality. |

## Ground truth

Measured on the fixtures as committed, single-threaded (see *Running* below).
Timings are machine-dependent; the **shares, ratios and counts** are the stable
part and are what the assertions rely on.

| # | Key facts a correct assessment should recover |
|---|---|
| 1 | `match_points` ≈ **2.0%** of runtime; `parse_scans` ≈ **79%**; `select_keypoints` ≈ 19%. |
| 2 | At the CLI default (`--resolution 32`): fusion is **4.5%** of wall time, peak RSS **105 MiB**. At `--resolution 128`: **84%** of wall time, peak RSS **1127 MiB**. `integrate_volume`'s own signature defaults to **256** — 512× the voxels of the CLI default. |
| 3 | Four stages, all ≥10%: `neighbour_search` **47%**, `shade_vertices` **18%**, `write_obj` **18%**, `pack_atlas` **16%**. |
| 4 | `fit` ≈ **5.4 s** (`fit.forward_backward` 86%); `export` ≈ **0.77 s** (`export.nearest_seed` 99%). The irregular seam is in the *cheaper* command. Both commands accept `--profile`. |
| 5 | 404 templates, of which the chunk loop visits **400** — templates 400-403 are unreachable, and are the true best for **275** points. Zero-area templates at indices 7, 118, 254 poison their chunks. Net: **1458 / 20000 points (7.3%)** differ from the intended contract, **1361 strictly worse**. |
| 6 | `pruned()` is ≈ **2.0×** faster and returns a strictly worse shape for **193 / 6000 queries (3.2%)**, max score inflation **10.4×**. Subtle enough to survive a spot check; decisive at full scale. |
| 7 | Precomputing the affine blend coefficients is worth ≈ **1.52×** *on the numpy baseline alone*, and shifts selection on **231 / 40000** points. Any Warp comparison must be against the improved baseline. |
| 8 | One invocation: **1.015 s** total process wall for **0.906 s** of compute — the rest is interpreter and numpy import, paid per file, on top of whatever a ported backend adds. |
| 9 | At the CLI default: `spread` ≈ **97%** of the call and allocates **0 device bytes**, while `aggregate` and `colorize` each allocate 1,048,576. `quality='high'` warns and reverts to `'fast'` for device input, so the high-quality path is unreachable as advertised. |
| 10 | `relax` ≈ **99.9%** of the call, with **0 device bytes and 0 device launches**, while `normalise` and `histogram` each issue one launch on the backend already in the project. |
| 11 | No measurement to recover — the README states no GPU in any SKU and no runtime compilation on the appliance. Both are stage-1 gates. `assign_zones` really is ~70% of a batch, so the temptation is real and the shape is genuinely Warp-like. |
| 12 | `classify` ≈ **86%** of the pipeline at ≈8-9 GB/s against a ≈13-14 GB/s read-only floor → **≈35-40% headroom** in that stage, ≈30% of the pipeline. `calibrate` is **≈14%** of the pipeline with **≈96%** headroom. Whole-pipeline ceiling ≈45%. |
| 13 | Wrapper cost is **flat at ≈100 ms per call** (index-wide bounds recompute + validation pass): **99.9%** of a 1-query call, **92%** at 64, **≈25%** at the documented 2,000-query frame. `query_prepared` pays none of it. |
| 14 | Same 2,000 triangles and ≈115k covered pixels either way, but the default synthetic workload averages **≈150x canvas coverage** and takes **≈0.71 s**, versus **≈1.0x** coverage and **≈0.06 s** for the shipped mesh — a **12x** difference in cost for the same nominal size. |
| 15 | Shipped asset (n=32, 2 048 triangles, face area 4.883e-4): **100 %** of faces below `min_patch_area`, the stock primitive disagrees on **69.1 %** of points, closest-face index differs on **34.3 %**. Toy asset (n=8, 128 triangles, area 7.812e-3): **0 %** below, **0 %** disagreement. The disagreement is **max abs 8.23e-5 but max rel 1.000**. |
| 16 | Distances **bitwise identical**. Index differs on **250/6 200 (4.0 %)**; of those **50 (20 %)** share a node and **200 (80 %)** do not. Per-segment (index-shaped) max diff **18.0**; per-node (observable) max diff **8.54**; isolating the shared-node ties leaves **5.7e-14**. Trajectory: perturbed start is non-vacuous (**4.3x** reduction) but diverges only **4e-16** because the perturbation broke the ties; exact start diverges **2.2e-3** but is **vacuous** (1.1x). |
| 17 | forward **476.7 ms (50.7 %)**, reverse **464.1 ms (49.3 %)**, public `shape_score` **894.1 ms**. `forward_indexed` **93.1 ms**, values identical — **5.1x** on the seam, **1.60x** on the public call, ceiling **2.03x**. |
| 18 | All **3/3** static single-shot checks PASS; the ownership check FAILS with caller-array drift **0.05** after one refit. Over 40 refits the aliased harness ends at **131.83** against **0.33** for the corrected one — a **~40 000 %** error — and the caller's array has drifted by **8.2**. |
| 19 | `resample_field` **81.4 %** of the batch, `smooth` **12.9 %**, `load` 2.7 %, `emit` 3.0 % (total ≈405 ms). One shared `kernels/*.cu` built for **3 targets** (NVIDIA / AMD / CPU), all in CI, with roughly a third of deployments on AMD. |
| 20 | Maintained backend vs a plain uniform grid, identical results at every size: **2.6x** at N=2 000, **8.5x** at N=8 000, **21.2x** at N=20 000. Scaling confirms O(N·M): **4.2x** the time for **4x** the pairs, twice over. |
| 21 | Target 20 000 points, median spacing **0.0128**. The unrelated asset sits at **49x spacing** — further out than any real state. A captured `fit()` runs **39x → 1.0x spacing** over 40 iterations. Cell list vs brute force: **0.15x** on the assembled pairing, and **0.18x → 0.86x → 2.71x → 8.12x → 8.80x** across the captured iterations 0/3/8/16/39. Exact at every point. |
| 22 | Incumbent vs **itself**, 10 calls on identical inputs: not bitwise equal, max rel self-spread **9.1e-07**. Candidate vs incumbent: **6.1e-07** rel — *smaller than the incumbent's own noise*. Against a float64 reference: incumbent **4.4e-07**, candidate **1.0e-07**, so the candidate is the more accurate. Deterministic variant costs ≈**4.7x** the runtime in this baseline. |
| 23 | Same code, two layouts. Lattice: **16.02 %** of indices differ (9 615 / 60 000), and **31.33 %** of samples are exactly equidistant between ≥2 seeds. Randomised: **0.00 %** differ, **0.00 %** ambiguous. All 9 615 differing choices are exactly equidistant in float64 — worst excess distance **0.0**. Incumbent matches the float64 oracle on 100 %. |
| 24 | `auto_cell()` returns **0.2726** against a measured spacing of **0.0046** — **59.4x** too coarse, because six stray far returns inflate the bounding box. At the default: **11.8x** vs brute force, **7 298** candidates scanned per query. Tuned (8x spacing): **43.6x**, **133** scanned per query — **3.7x** on time, **54.8x** on work. Exact at every cell size. |
| 25 | Per-call setup **~35 ms**, flat. At the shipped 16 x 2048 batch: shipped scan **1130 ms**, per-example adapter **1462 ms (0.77x)**, fused adapter **934 ms (1.21x)** — the verdict inverts. Holding the total at 32 768 points, the gap is **~34-36 ms per call** at every example count while the fused column falls **1330 → 570 ms**; the per-example column bottoms out and then *rises* to **2741 ms** at 64 examples. All three identical. |
| 26 | Portable path returns **256/256** edges at every scale. Accelerated path: **256** up to scale 3e4, **219** at 1e5, **94** at 3e5, **64** (self-pairs only) at 1e6 and 1e9. Trigger is the k-th squared distance crossing the **1e10** sentinel. At scale 1e5, **18 of 64** queries come back short, min degree **1**. The shipped selftest passes **5/5** throughout. |
| 27 | At r=0.10: mean true degree **73.9** (max 196), **2413/3000 (80.4 %)** of queries at or over the cap of 32, **140 227** neighbours discarded. All three backends return exactly **81 441** edges, **100 %** of them genuinely within r. The two *shipped* backends keep the same subset for only **8 of 2413** truncated queries (**43.7 %** edge overlap); the candidate scores **9 of 2413** (**51.8 %**) — no further away than the incumbent's own sibling. **7** queries are exactly full and **2406** truncated, and the API reports 32 edges for all 2413. Sweep: **0 %** at capacity and 100 % identical at r=0.03; **85.8 %** and 14.3 % at r=0.14. |
| 28 | No measurement to recover — the README states a read-only root filesystem, no toolchain in the image, a W^X policy prohibiting runtime code generation, an air-gapped unit, and vendored-prebuilt-only runtime dependencies. Every unit **does** have an RTX A4000 and the project **already ships** a prebuilt CUDA extension, so gates A and D do not fire. `neighbourhood_features` really is **~100 %** of a sweep and scales as n², so the temptation is real. |
| 29 | Tracer total **2.00x** the summed stage wall time, exactly, and **1.54x** the whole-run wall time — impossible for device time on one stream. Per-stage shares are unaffected by the double count and match wall-derived shares: classify **49.1 %**, load **31.7 %**, aggregate **19.1 %**, emit **0.2 %**. |
| 30 | Per-op benchmark: `op_spline` **75.5 %**, `op_scatter` 14.5 %, `op_knn` **6.4 %**, `op_radius` 3.4 %. Shipped consumer: `op_knn` **53.3 %**, `op_scatter` 18.8 %, `op_dense` 18.3 %, `op_radius` 9.6 %, and `op_spline` **never called at all**. Of `op_knn`'s three calls only one is over 3-D positions: **32.4 %** of `op_knn`, **17.3 %** of the step, so a positional index has a ceiling of **~1.21x** against the **2.14x** implied by `op_knn`'s total share. The other two calls run at **D=128**. |

Cases 1-6, 9-30 can be assessed to a verdict **without a GPU** (they turn on
profiling, memory, correctness, semantics, ceilings, portability and gating).
Cases 7-8 need an NVIDIA GPU to complete the backend comparison the prompt
invites; without one, the correct outcome is `INCONCLUSIVE` naming the missing
experiment, and the assertions should be graded against that.

### Cases 15-20 and the extensions to 8, 10 and 14

These came out of a full assessment of `facebookresearch/pytorch3d`, where each
failure below was either hit or narrowly avoided. Rather than adding cases that
overlap existing ones, three existing cases gained assertions instead:

- **8** now requires break-even to be expressed as a **number of invocations**,
  computed against the incumbent's own per-process cost — an absolute cold-start
  figure on its own does not answer the question the user asked.
- **10** now requires the **algorithmic** gain and the **backend** gain to be
  separated rather than bundled into one ratio.
- **14** now requires the cost driver to be swept as a **data property** and the
  result reported as an envelope with the sign-change bracket, not a single
  ratio. In the source assessment the verdict inverted across that sweep: the
  same seam was ~100x faster on near-surface queries and ~2-9x *slower* on
  far-field ones, so a single number would have been wrong in both directions.

### Cases 25-30 and the extensions to 20 and 24

These came out of a full assessment of `pyg-team/pyg-lib`, a library that ships
hand-written CUDA for most of its operations. Each one is a failure that was hit
or narrowly avoided during that assessment, and two of them changed the verdict.

- **25** is the one that nearly inverted the result. The first Warp adapter
  looped over batch examples on the host, which cost ~1.2 ms *per example* and
  made the candidate look several times slower than the incumbent. The device
  work was unchanged. Diagnosing it needs the specific move the case grades:
  hold total work constant and vary only the number of items.
- **26** and **27** are contract findings about the *incumbent*, not the
  candidate. 26 is a hard-coded `1e10` sentinel that silently shortens the
  output above a coordinate scale of ~1e5 while the shipped tests stay green;
  27 is a capacity cap whose retained subset the project's own CPU and CUDA
  backends already disagree about 100 % of the time.
- **29** is the measurement instrument itself being wrong. The profiler's
  device-time totals came out at ~2x synchronized wall time; the shares were
  fine, the absolutes were not, and a report that quoted them would have been
  internally inconsistent.
- **30** is the structural feature of assessing a *library*: there is no
  application to profile, so one has to be composed from the shipped consumer.
  Ranking from the per-operation benchmark picked an operation no consumer
  calls, and missed that the top operation's cost sits mostly in call sites
  where the proposed structure cannot apply.

Two existing cases gained assertions rather than getting near-duplicates:

- **20** now requires the **algorithmic** and **backend** contributions to be
  separated. In the source assessment a control that expressed the incumbent's
  own algorithm in the candidate backend was worth only 1.09x-1.67x, and *below
  parity* at the largest sizes, while the algorithm change was worth up to 132x.
  Reporting one bundled ratio would have credited the backend with all of it.
- **24** now requires a knob tuned on a procedural workload to be re-validated
  on the real asset. Carrying the stand-in's optimum across costs **1.75x** on
  the real scan (`neighbours.py transfer`), because the two workloads differ in
  intrinsic dimension — the same reason the shipped default is wrong.

#### Keeping the new cases apart from their neighbours

| New | Nearest existing | Discriminator |
|---|---|---|
| 25 | 13 (wrapper overhead), 18 (harness defect) | 13 is the *incumbent's* wrapper flattering a candidate; 18 is a harness bug that corrupts *correctness*. 25 is the *candidate's own adapter* costing performance, diagnosed by holding work constant and varying item count. |
| 26 | 5 (broken incumbent) | 5 is a logic defect that invalidates the incumbent as an oracle. 26 is latent and input-scale-triggered: nothing is wrong at test scale, the shipped tests pass, and only a scale sweep exposes it. |
| 27 | 16, 23 (index/tie differences) | 16 and 23 are about *ties* — one answer, several equally valid representations. 27 is about *capacity*: the true answer has more elements than the API can return, so the question is which valid subset survives and whether truncation is even detectable. |
| 28 | 11 (abort, CPU-only), 19 (portable incumbent) | 11 aborts because there is no GPU; 19 continues with portability priced in. 28 has the GPU and a maintained CUDA path, and aborts purely on the *obligations* a runtime-compiled dependency would impose. |
| 29 | 9 (silent host fallback) | 9 is the *code* not running where you think. 29 is the *instrument* misreporting a path that did run. |
| 30 | 1 (non-bottleneck), 4 (entry points), 17 (half an entry point) | 1 and 4 concern which entry point to measure within an application. 30 has no application at all, and adds the per-call-site decomposition of a single symbol whose calls differ in dimensionality. |

Case 16 deliberately has **no clean pass**: neither trajectory regime settles
the question on its own (the non-vacuous one has lost the ties, the one that
keeps them is vacuous). Grade it on whether the report *notices* that, not on
whether it produced a verdict.

Case 19 is the counterpart to case 11: there the environment ruled Warp out and
the answer was `ABORT`; here the environment permits it and the correct answer
is to continue with portability priced into the verdict. A run that aborts case
19 on portability grounds has failed it.

#### Keeping 15, 16, 22 and 23 apart

Four cases now involve a candidate whose output differs from the incumbent's,
and they are only useful if they stay distinct. The discriminator is **what
differs and why**:

| Case | What differs | Root cause | What the report must do |
|---|---|---|---|
| 15 | the **values** | the accelerated primitive computes a *different quantity* — a documented parameter it has no notion of | refuse the drop-in; note an absolute tolerance hides it |
| 16 | only the **index**; values bitwise equal | two valid tie-breaks | compare at the observable level; validate a downstream check that is neither vacuous nor tie-free |
| 22 | the **values** | the incumbent is **nondeterministic**, and differs from itself by more than the candidate differs from it | stop scoring against the incumbent; build a reference |
| 23 | only the **index**; values equidistant | the **input topology** makes the answer genuinely ambiguous | measure the ambiguity rate per layout; don't generalise the easy layout |

15 and 22 are both value differences but with opposite culprits (candidate
computes something else vs incumbent cannot repeat itself). 16 and 23 are both
index differences but ask different questions (at what level does it matter vs
how often is the answer even well-defined). If a run starts passing one of them
purely because another is in the set, collapse the pair rather than keeping
both.

Case 11 is the only one whose correct output is **no report directory at all**. Grade it
on the abort being immediate and correctly attributed — a thorough assessment
that reaches the same conclusion after profiling has failed the case, because
the gate is meant to fire before any measurement.

## Trigger evaluation

`trigger_queries.json` grades the *description* rather than the output: whether
the skill loads at all for a query, given that Warp is offered rather than
requested. 34 queries, 17 positive and 17 negative, `runs_per_query` 3, fixed
train/validation split (20/14) with a proportional mix in each half. No
positive query names Warp or states a desired verdict; several negatives name it
deliberately, because a plain API, packaging or "just write the kernel" request
is exactly what the description has to suppress.

Three are deliberately counter-intuitive, and all three are **positives** that
look like negatives. Query 31 describes a deployment that cannot accept a
runtime-compiled dependency at all — but that is a policy fact living in the
repository, not in the request, and suppressing the skill there would skip the
stage-1 gate that is supposed to fire and produce the `ABORT` (case 28). Query
32 is a library maintainer with no application of their own (case 30). And
query 23 describes a project that
*already* ships `numba.cuda` kernels and is still a **positive**. An existing
accelerator backend is a stage-1 and baselines-ladder judgement, not something a
description can screen on, and treating it as a description-level negative would
suppress the skill on exactly the assessments it should run.

## Running

Fixtures depend only on `numpy` and generate their own inputs on first run
(`scan_a.txt`, `scan_b.txt`, `sample.raw`, `mesh.obj`, `manifest.txt` — all
regenerable, none checked in).

Run single-threaded so timings are reproducible and several cases can share a
host without thrashing:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python files/e3-profile-coverage/run_pipeline.py --stages
```

Run cases **sequentially**, not in parallel: case 2 at raised resolution
allocates over a gigabyte, and concurrent BLAS fan-out distorts every timing in
the suite.

Validate each generated report directory from the skill directory:

```bash
python scripts/validate_report_schema.py /path/to/warp-adoption-report/
```

Each fixture prints its own ground truth. The slower ones, single-threaded:

```bash
python files/e15-unexpressible-contract/patchdist.py --report   # ~5 s
python files/e16-tie-dominated-index/weldpath.py    --report    # ~60 s
python files/e17-half-entry-point/bidist.py         --report    # ~4 s
python files/e18-zero-copy-aliasing/bridge.py       --report    # <1 s
python files/e19-portable-incumbent/pipeline.py     --profile   # <1 s
python files/e20-mature-but-wrong-algorithm/nnkit.py --report   # ~35 s
python files/e24-untuned-candidate/neighbours.py    transfer   # ~23 s
python files/e25-adapter-launch-loop/batchknn.py    sweep      # ~35 s
python files/e25-adapter-launch-loop/batchknn.py    bench      # ~7 s
python files/e27-capacity-truncation/radiusgraph.py sweep      # ~6 s
python files/e28-gate-e-no-jit/voxelise.py     --sweep-size    # ~10 s
python files/e30-library-no-application/bench_ops.py           # ~2 s
```

Case 30's consumer must be run from its own directory so it can import the
library beside it:

```bash
cd files/e30-library-no-application && python examples/train_step.py --profile
```

Per the agentskills workflow, each case is run twice — once with the skill and
once without (or against the previous skill version as a snapshot baseline) —
into `with_skill/` and `without_skill/` under
`../warp-adopt-workspace/iteration-N/<eval-name>/`.

## Notes on these assertions

- They are a **first draft**. The agentskills guidance is to write assertions
  after seeing a first round of outputs; these were derived from an observed
  failure instead, so expect to retune them after iteration 1 — particularly to
  drop any that pass with and without the skill, which measure nothing.
- Case 3's assertions encode the ≥10% coverage rule from
  `references/verdicts-and-reporting.md`. If that threshold changes, change it
  here too.
- Cases 5 and 6 deliberately reward *finding a defect and stopping*, which for
  this skill is a successful outcome, not a failed one. Grade them on whether
  the defect was found and correctly classified, not on whether a port was
  produced.
- Start with cases **1, 2 and 5** if you want the recommended 2-3 case first
  round: they cover the three failures that actually changed a verdict
  (non-bottleneck, wrong-metric screening, defective oracle) and none of them
  needs a GPU.
- For a second round, **15, 18 and 20** are the highest-value additions and are
  also GPU-free: a contract the accelerated primitive cannot express, a harness
  bug masquerading as a result, and a mature implementation that is still the
  wrong algorithm. Each of the three, on its own, produced a wrong answer in the
  assessment they were derived from before it was caught.
- Cases 15, 16, 22 and 23 all involve a candidate that disagrees with the
  incumbent, and 5 and 6 are adjacent again (5 is a *defective* incumbent, 6 an
  *unproved pruning bound*). They are kept apart by *what* differs and *why* —
  see the table in **Keeping 15, 16, 22 and 23 apart** above, which is the
  single source of truth for those boundaries. If any of them starts passing
  purely because another is in the set, collapse the pair.
- Case **25** joins 21-24 as a **false negative** case, and is the most
  valuable of the group: a competently-run benchmark reports the candidate as
  slower, and the number is real — it is just a measurement of the adapter. Run
  it early. Cases **26** and **27** are the cheapest in the whole suite and both
  grade *method* (sweep the input scale; check the incumbent against itself
  before calling a candidate wrong), so watch them in the with/without
  comparison — if they pass without the skill they are measuring nothing.
- Case **28** is the second `ABORT` case after 11, and the harder of the two:
  the hardware is present and the workload shape is genuinely a good fit, so a
  run that produces a full report has failed it even if the report's verdict is
  negative.
- Cases **21-24** are the only ones in the set whose failure mode is a
  **false negative** — each one, left uncaught, ends the assessment with a
  wrongly negative verdict rather than an unjustified GO. That makes them
  harder to grade and more valuable: nothing downstream contradicts a "no".
  21 is the strongest of the four and is the one to run first, because it is
  the only case in the suite where the *correct* answer is the opposite of what
  a competently-executed benchmark reports.
- 23 is deliberately adjacent to 16 and must not collapse into it. 16 asks
  *at what level* an index difference should be judged, given one fixed
  geometry. 23 asks whether the mismatch **rate itself** is even a property of
  the implementation — the same code gives 0 % and 16 % on two plausible
  layouts — and grades whether the reachability statistic was measured at all.
  If 23 starts passing whenever 16 does, the reachability assertions are not
  discriminating and should be tightened rather than dropped.
- 22 and 24 are the two cases that grade **method rather than judgement**:
  measure the baseline's own spread before writing a tolerance, and tune the
  candidate before quoting its ratio. Both are cheap to run and both are easy
  to pass accidentally, so watch them closely in the with/without comparison —
  if they pass without the skill, they are measuring nothing and should be made
  stricter or retired.
