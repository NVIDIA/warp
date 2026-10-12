# CPU Kernel Helper Inlining

**Status**: Implemented

**Issue**: [GH-1994](https://github.com/NVIDIA/warp/issues/1994)

## Motivation

Warp lowers `@wp.func` calls to C++ helpers before Clang and LLVM compile CPU
kernels. Some helpers receive large Warp structs by value. When LLVM leaves
these calls out of line, topology loops repeatedly copy handles to the stack
and lose optimization opportunities across the call. Small changes to nearby
generated code can therefore cause large runtime changes.

The motivating case was a polyhedral kernel that passed a dataset handle
through nested helpers. LLVM should inline profitable calls without changing
the meaning of explicit `@wp.func(inline=True)` or `inline=False` hints.

## Design

### Policy through CPU compiler flags

Set the initial value of `warp.config.cpu_compiler_flags` to:

```text
-march=native -mllvm=-inline-threshold=3000 -mllvm=-inline-cold-callsite-threshold=3000
```

The [LLVM options for CPU compilation](cpu-llvm-compiler-options.md) feature
passes each `-mllvm` argument to LLVM during CPU compilation and resets its
parsed values afterward. The two thresholds are independent LLVM controls
that happened to have the same selected value:

| LLVM option | Bundled LLVM default | Warp CPU default |
| --- | ---: | ---: |
| `inline-threshold` | 225 | 3000 |
| `inline-cold-callsite-threshold` | 45 | 3000 |

The generic threshold alone misses call sites LLVM classifies as cold. Both
budgets remain finite so LLVM can still leave expensive calls out of line.
Debug builds remain at `-O0`, where this inlining heuristic does not run. The
explicit Warp inline hints retain their existing meaning.

Only CPU compilation receives these flags. LLVM-CUDA compilation uses the
default LLVM option state, coordinated with CPU compilation by the existing
option lease. The default NVRTC CUDA path is unaffected.

### Kernel-body linkage

Generate the CPU kernel forward and backward bodies with C++ `static` linkage.
Each body is called only by its generated launch wrapper or block thunk. The
exported `*_cpu_forward` and `*_cpu_backward` entry points retain their
external linkage. Internal linkage lets LLVM discard a body after inlining its
only call, instead of retaining an unused externally visible copy in the
object file.

In bundled LLVM 22.1.8, the [inliner](https://github.com/llvm/llvm-project/blob/llvmorg-22.1.8/llvm/lib/Analysis/InlineCost.cpp)
can apply a 15,000-unit `LastCallToStaticBonus` for a locally linked callee
with one live use. The [default target cost model](https://github.com/llvm/llvm-project/blob/llvmorg-22.1.8/llvm/include/llvm/Analysis/TargetTransformInfoImpl.h)
defines that value. This reduces the inlining cost of the kernel body. It does
not raise either threshold for shared `@wp.func` helpers. It can also be
suppressed for cold calls or callees. Therefore the `static` declaration
complements the 3000 / 3000 helper policy; subtracting 15,000 from either
threshold would not reproduce the same decisions.

### Overrides and caching

A module's `cpu_compiler_flags` string replaces the global string. To use LLVM's
original thresholds for one module while retaining host CPU detection, set its
flags to `"-march=native"`. To do so globally for modules not yet hashed,
set `warp.config.cpu_compiler_flags = None`; the existing resolver maps that
value to `"-march=native"`. A module-level `None` inherits the global setting
when the module is first hashed. After a module has been hashed, use
`wp.set_module_options()` to invalidate its cached hash when changing flags.
Custom CPU flag strings, including portable target strings, use the tuned
policy only if they include both `-mllvm` arguments.
The kernel-body linkage is independent of the flag string, so restoring LLVM's
original thresholds does not restore the former externally visible bodies.

The resolved CPU flag string contributes to the module cache key, and the
ordered tokens contribute to the native PCH key. Tuned and LLVM-default CPU
objects therefore occupy distinct cache entries without a separate policy
revision marker. The Warp release version separates shipped objects generated
with `static` linkage from earlier cached objects.

## Selection evidence and trade-offs

The measurements below selected the numerical thresholds on one Windows
x86-64 host with bundled LLVM 22.1.8. They predate the CPU `-mllvm` delivery
route, so they establish the values and costs rather than final-branch
performance.

Raising only the generic threshold left a focused hex `iso_surface` case
around 84 ms, even at 50,000. Raising only the cold-callsite threshold to
225 brought it to about 64 ms. Raising both to 3000 brought it to about
40 ms. The broader comparisons used the selected pair.

| Generic / cold threshold | CGNS `iso_surface`, ms | CGNS `external_faces`, ms | Hex `iso_surface`, ms |
| --- | ---: | ---: | ---: |
| 1000 / default | 76.5 | 702.2 | 107.5 |
| 5000 / default | 76.4 | 509.9 | 82.2 |
| 1000 / 1000 | 74.7 | 698.6 | 67.3 |
| 2000 / 2000 | 74.6 | 566.6 | 40.8 |
| **3000 / 3000** | **76.2** | **521.0** | **40.2** |

Increasing `locally-hot-callsite-threshold` to 3000 slowed the focused hex
case to about 176 ms, and `-O3` slowed the focused polyhedral and hex cases.
Those options are excluded from the default policy.

Across 53 matching Benchy synthetic CPU cases, the equal-weighted geometric
mean of median speedups was **+14.98%**. Thirty-nine cases improved by more
than 5%, and the worst case regressed by 2.24%. Hex `iso_surface` improved
**126.3%** and hex `external_faces` improved **45.6%**. Output cell and point
counts matched in all 53 cases.

Across seven Newton CPU examples, the equal-weighted geometric mean of
median FPS ratios was **+7.69%**. `diffsim_drone` improved **21.71%**;
`softbody_hanging` measured **-3.43%**, with overlapping run ranges. The
other five examples measured between +0.03% and +14.54%.

An ASV sweep of 26 existing CPU parameter cases found no case more than 5%
slower. CPU BVH AABB queries improved **1.08 to 1.21x**, and a 1024-size CPU GEMM
improved **1.13x**. CPU code generation and 2D register tile load/store at
block size 1 were about 2% slower, with overlapping ASV uncertainty ranges.
The original short `cpu_helper_inlining.StructHelperInliningCPU.time_classify_edges` case,
which repeatedly calls a helper with an array-rich struct, measured
**23.19 ms baseline versus 11.09 ms tuned (2.09x)**. Its first ASV run took
about three seconds including setup and compilation.

The selected policy increased aggregate native compilation time on
representative kernels from **6.666 s to 7.188 s (+7.8%)**, and full Benchy
CPU object bytes from **894,288 to 1,088,768 (+21.7%)**. Both stayed within
the chosen limits of 20% additional representative compilation time and 25%
additional full-matrix object bytes. Benchy first-call medians increased by
9.0% in aggregate. Disassembly of representative workers showed that calls
to struct-taking helpers and associated copies disappeared, consistent with
the intended optimization.

Earlier correctness checks found matching Benchy outputs, passed 16 selected
Newton example tests on CPU and CUDA, and found no additional Warp suite
failures with the selected thresholds. These comparisons do not substitute
for testing the CPU `-mllvm` delivery path.

A focused check of the flag-based implementation on the original ASV helper workload
measured **11.01 ms with the shipped default versus 42.51 ms with
`"-march=native"` (3.86x)**. Both variants produced the same output, and
their distinct module hashes confirmed separate cache entries. Setting the
global flags to `None` selected the LLVM-default variant; restoring the
shipped flags selected the tuned variant again in the same process. This
spot check and the earlier ASV measurement are separate comparisons.

### Combined linkage and threshold evaluation

With the flag-based implementation, three independent passes over the same 53
Benchy CPU cases measured the following variants. Runtime is the geometric
mean speedup of per-case medians relative to external linkage and LLVM's
original thresholds. Object size and native compilation time are totals over
the cases. All variants produced matching output counts.

| Kernel body | Generic / cold threshold | Runtime | CPU object bytes | Native compilation |
| --- | ---: | ---: | ---: | ---: |
| External | LLVM defaults | Baseline | 894,288 | 27.87 s |
| External | 3000 / 3000 | +14.97% | 1,088,768 | 33.03 s |
| `static` | LLVM defaults | +4.06% | 812,680 | 29.18 s |
| `static` | 2000 / 2000 | +16.77% | 865,184 | 31.71 s |
| **`static`** | **3000 / 3000** | **+19.70%** | **897,792** | **32.64 s** |

Relative to the existing tuned policy, `static` with 3000 / 3000 improves the
geometric mean runtime by another 4.11%, reduces aggregate object size by
190,976 bytes (17.54%), and slightly reduces native compilation time. It
eliminates 109,368 of the 194,480 bytes of object growth caused by tuning with
external linkage (56.24%). The tuned `static` objects are only 0.39% larger
than the original LLVM-default objects. Object inspection found no retained
kernel-body symbols in the `static` variants. A 1000 / 1000 trial gave up
substantial runtime gains, while 3000 / 2000 gave no useful size saving.

On the original ASV struct-helper case on Windows, external linkage with LLVM
defaults took 23.62 ms; external 3000 / 3000 took 11.49 ms; `static` 2000 / 2000 took
11.73 ms; and `static` 3000 / 3000 took 11.20 ms. The selected combination is
2.11x faster than the original baseline and slightly faster than the current
tuned build. A second, longer-sample check of 17 potentially regressing Benchy
cases found no case more than 5% slower with `static` 3000 / 3000 than with
external 3000 / 3000.

A seven-example Newton CPU sweep had variable frame-rate results. Follow-up
paired runs reversed an apparent `cloth_bending` slowdown. Repeated
`diffsim_drone` runs from already compiled caches measured median 96.95 FPS
with external 3000 / 3000 and 98.65 FPS with `static` 3000 / 3000. These
checks did not establish a Newton regression, but their variability limits
claims about the size of any gain.

### ASV coverage under both x86-64 ABIs

The original key-only ASV helper already inlined with LLVM defaults under the
Linux ABI. Linux CI measured **18.696 ms baseline versus 17.589 ms tuned
(1.063x)**, much less than the 2.11x Windows gain. The different calling
conventions put the helper on opposite sides of the default inlining threshold.

The benchmark now returns a canonical edge key and an interpolated position.
The kernel consumes both, using distinct keys to select a squared triangle-normal
length calculation. This is representative clamped edge-intersection work that keeps
the helper above the default inlining budget under both ABIs. It uses ordinary
`@wp.func` helpers without forcing their inlining decisions.

With bundled LLVM 22.1.8, diagnostics show helper costs of 385 under the Windows
ABI and 315 with the Linux frontend before kernel-body inlining. Both remain
above threshold 225 afterward, so default compilation retains the helper
calls. The 3000 / 3000 policy inlines them under both ABIs.

Two independent ASV runner processes per variant, each with ten timed launches,
gave the following ranges of process medians and paired speedups. Kernel bodies
remain `static` in both variants to isolate threshold tuning.

| Calling convention | LLVM-default thresholds | 3000 / 3000 | Speedup |
| --- | ---: | ---: | ---: |
| Windows x64 | 10.98-12.82 ms | 3.08-3.19 ms | 3.44-4.16x |
| System V AMD64 | 16.09-16.11 ms | 3.07-3.08 ms | 5.23-5.24x |

Both runtime experiments used the same Windows host, with the System V calling
convention applied to generated helpers and kernel bodies. Separate compilation
with the Linux frontend confirmed the same inlining decisions. These results
establish sensitivity to the thresholds under both ABIs; they do not measure
the revised workload's speedup on a Linux CI host.

Reducing the cell count to 262,144 keeps suite overhead small. Fresh-cache ASV
runner invocations took 1.36-1.56 s including setup, compilation, warmup,
validation, and ten timed launches, excluding interpreter startup. The complete
output is checked against NumPy once during setup. Timing and compiler
decisions are not asserted by the benchmark.

## Decision

Keep both thresholds at 3000 in the public CPU compiler flag default and make
the generated CPU kernel bodies `static`. Reducing both thresholds to 2000
saves another 32,608 object bytes over the full Benchy matrix but gives up
2.93 percentage points of geometric mean runtime gain. The selected combination
offers the best balance of runtime, object size, and compilation time in these
measurements. Users can request LLVM's original thresholds for a module or
future modules globally; kernel bodies retain `static` linkage. Other custom
CPU flag strings replace the threshold policy, and CUDA compilation keeps
LLVM's original defaults.
