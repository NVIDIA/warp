# Warp cold-compile mechanisms

Each section gives a signal, cause, fix, limits, and measured result. Read only
the sections selected by your measurement.

## Contents

| ID | Problem | Backend |
| --- | --- | --- |
| [CS-1](#cs-1-kernels-defined-after-their-module-loaded) | Kernels defined after their module loaded | CPU + CUDA |
| [CS-2](#cs-2-the-cache-is-never-reused) | The cache is never reused | CPU + CUDA |
| [CS-3](#cs-3-module-options-set-after-first-load) | Module options set after first load | CPU + CUDA |
| [CS-4](#cs-4-module-boundaries-too-granular) | Module boundaries too granular | CPU + CUDA |
| [CS-5](#cs-5-one-module-spans-several-cuda-block-dimensions) | One module spans several CUDA block dimensions | CUDA |
| [CS-6](#cs-6-generic-instances-appear-after-first-load) | Generic instances appear after first load | CPU + CUDA |
| [CS-7](#cs-7-mathdx-lto-without-an-amortization-case) | MathDx LTO without an amortization case | CUDA tiles |
| [CS-8](#cs-8-concurrent-first-cpu-load-races-the-native-jit) | Concurrent first CPU load races the native JIT | CPU |
| [CS-9](#cs-9-moduleunique-on-stable-kernels) | `module="unique"` on stable kernels | CPU + CUDA |
| [CS-10](#cs-10-backward-code-generated-but-never-used) | Backward code generated but never used | CPU + CUDA |
| [CS-11](#cs-11-unroll-budget-inflates-source) | Unroll budget inflates source | CPU + CUDA |
| [CS-12](#cs-12-the-precompiled-header-is-not-paying-for-itself) | The precompiled header is not paying for itself | CPU + CUDA |
| [CS-13](#cs-13-independent-modules-are-built-one-at-a-time) | Independent modules are built one at a time | CUDA |

## How to read the numbers below

Every figure comes from one controlled study, on one machine, with one Warp
revision. Use them to rank mechanisms and to sanity-check whether a result is
plausible. Absolute milliseconds do not transfer between machines.

Percentages do not transfer either, and for some mechanisms they cannot: where
the payoff is set by the shape of the program rather than by the fix, one
measurement says nothing about the next application. Those sections say so and
give you a way to bound your own case from your own measurement instead. Do not
quote a figure from this file to a user as the result they should expect.

On CPU, a large fixed native-toolchain cost dominates each run, so the same
structural fix shows a small relative gain on CPU and a large one on CUDA. Do
not conclude a CPU fix failed because the percentage is modest, and do not
extrapolate a CUDA percentage to CPU.

Part of that CPU floor is Warp's precompiled header build, which is what keeps
later modules cheap. Check the header first (CS-12) — it can be wrong in either
direction, and it applies on CUDA too — then focus on structural mechanisms.

## CS-1: Kernels defined after their module loaded

**Signal.** One module name, several hashes. A `Module hash changed,
recompiling` line in verbose output.

**Why.** A module's hash covers its live kernel set. Define a kernel, launch it
(the module builds), then define another kernel in the same module and launch
again. The new kernel changes the identity, so Warp builds the module again
with both kernels.

**Fix**, in order of preference:

1. Move stable kernel definitions to module scope.
2. If they must be created at runtime, create every expected kernel in that
   module *before* the first launch.
3. Give kernels that share one lifecycle a deliberately named module.
4. Only then, `module="unique"` for a kernel that cannot join a
   stable module.

```python
# Before: the module builds once for `first`, then again for `first` + `second`.
def run(values):
    @wp.kernel
    def first(x: wp.array[wp.float32]):
        x[wp.tid()] += 1.0
    wp.launch(first, dim=values.shape, inputs=[values])

    @wp.kernel
    def second(x: wp.array[wp.float32]):
        x[wp.tid()] *= 2.0
    wp.launch(second, dim=values.shape, inputs=[values])
```

```python
# After: one identity, one build.
@wp.kernel
def first(x: wp.array[wp.float32]):
    x[wp.tid()] += 1.0


@wp.kernel
def second(x: wp.array[wp.float32]):
    x[wp.tid()] *= 2.0


def run(values):
    wp.launch(first, dim=values.shape, inputs=[values])
    wp.launch(second, dim=values.shape, inputs=[values])
```

**Does not apply when** runtime data creates a kernel per call, such as a
user-supplied expression, plugin, or varying shape specialization. Keep these
kernels isolated so they do not invalidate a shared module.

**Observed.** Total cold module work fell about 20% on CUDA and about 6% on
CPU; generated source roughly halved. Creating the dynamic kernels upfront
scored the same as module-scope definition.

## CS-2: The cache is never reused

**Signal.** A run that should have been warm still reports `(compiled)`, or the
configured cache directory is empty at process start. The probe's warm pass makes
this visible on every measurement: a repeat run should cost near zero.

**Why.** Warp cannot reuse a module binary without its cache. CUDA also has a
driver cache.

Check application code for:

- `wp.clear_kernel_cache()` or `wp.clear_lto_cache()` left in application code.
  These calls are not safe across processes.
- `wp.config.cache_kernels = False`, which disables reuse outright.
- `wp.config.kernel_cache_dir` pointed at a fresh temporary directory per run,
  or at a path containing a timestamp, so every invocation starts empty.
- The same directory removed on exit, by `shutil.rmtree` or an `atexit` hook.
- `CUDA_CACHE_DISABLE` set in the environment, which defeats the driver-side
  layer even when the Warp cache is healthy.

**Fix.** Persist a cache directory scoped to a compatible application build,
Warp version, module options, target architecture, and driver/toolchain. On
CUDA persist `CUDA_CACHE_PATH` as well as `WARP_CACHE_PATH`. Where persistent
storage is impossible, prewarm during startup and report that the cost moved.
For per-request containers, CI jobs, and serverless handlers, a persisted cache
can reduce module work to near zero.

**Does not apply when** the caller needs a clean build, or the cache
would be shared across incompatible builds, architectures, drivers, or option
sets. Reusing such artifacts is unsafe.

**Observed.** Cold module work of several hundred milliseconds (CUDA) and
several seconds (CPU) fell to roughly 1 ms warm. Process startup and imports
remain regardless.

## CS-3: Module options set after first load

**Signal.** A build, then a hash change and another build, with a
`wp.set_module_options()` call in between.

**Why.** Options are part of module identity. Changing one after the module has
loaded invalidates it.

This is the later of the two option deadlines described under "When a module's
options are fixed" in `SKILL.md`, and the more forgiving one: it costs an extra
build and announces itself as a second hash. The earlier deadline, assigning
`wp.config.*` after the module's file was imported, produces no build and no
symptom at all. If an option appears to have been ignored rather than applied
late, check that one instead.

```python
# Before
wp.launch(first, ...)
wp.set_module_options({"fast_math": True})   # invalidates the module
wp.launch(second, ...)

# After
wp.set_module_options({"fast_math": True})   # resolved before anything loads
wp.launch(first, ...)
wp.launch(second, ...)
```

**Fix.** Resolve mode, `fast_math`, `enable_backward`, `max_unroll`, and MathDx
flags before the module's first load. Set them at module scope or in an init
path before the first launch.

Check the call site of helpers such as `configure_stages()` or `setup_mode()`.
Also check which module a bare `wp.set_module_options()` applies to: it targets
the calling Python module, which is not necessarily where the kernels live.

**Does not apply when** kernels need different option sets. Then the
fix is separate named modules by option set (CS-4).

For a named shared module, `@wp.kernel(module_options={...})` is rejected
unless the kernel is `module="unique"`. `wp.set_module_options()` resolves the
calling Python module's `__name__`; it accepts neither a module-name string nor
a `wp.Module`. Use one of these patterns:

- give the group a real Python module and call a bare `wp.set_module_options()`
  at module scope; or
- set `wp.get_module("pkg.name").options.update({...})` while the module is
  still empty, before any kernel in it loads.

The first is clearer; the second avoids adding files.

**Observed.** A late `fast_math` change cost roughly half of total cold module
work on CUDA (two full builds instead of one); about 6% on CPU.

## CS-4: Module boundaries too granular

**Signal.** One feature loads and emits many small module events with similar
lifecycles.

**Why.** Every module carries its own hash, generated source, compiler
invocation, metadata, and binary load. Small modules repeat that fixed cost.

**Fix.** Group kernels that are deployed, loaded, and invalidated together into
one named module. Choose boundaries by lifecycle, not by a kernels-per-module
rule.

**Does not apply when** a kernel changes independently, is rarely used, needs
different options, or on CUDA uses a different stable block dimension. Merging
kernels with different block dimensions makes each compile at both dimensions.

A broader module also recompiles more code when any member changes, so
aggressive merging can slow down the edit-run loop it was meant to help.

**Observed.** Four one-kernel modules merged into one shared module cut cold
module work by about 29% on CUDA and 9% on CPU.

## CS-5: One module spans several CUDA block dimensions

**Signal.** The same module name compiled at several `block_dim` values. The
generated `.cu` for each variant contains the module's whole live kernel set.

**Why.** Warp compiles a module as a unit for an active `block_dim`. Every live
kernel in it is generated and compiled at every block dimension requested,
even a kernel only ever launched at one of them.

**Fix.** When the kernel-to-block-dimension mapping is stable, put each block
dimension's kernels in their own named module. Alternatively, reduce the number
of requested block dimensions if the launches were not deliberately tuned.

**Does not apply when** the mapping is not stable. A kernel whose block
dimension is chosen from input size at runtime has no fixed home. Keep it in
its own module rather than pulling a shared module into extra variants. It also
does not apply on CPU, where the effective block dimension is 1; splitting
there adds modules and made things *worse* in measurement.

Do not "fix" this by collapsing kernels that share a body into one kernel
launched at a single block dimension. That changes the workload.

**Observed.** Partitioning three stable block-dimension groups cut generated
source by about 60% and cold module work by about 13% on CUDA. The same split
on CPU was a regression.

## CS-6: Generic instances appear after first load

**Signal.** Two launches of the same generic kernel with different types
produce two module hashes.

**Why.** Instantiating a new overload changes the module's live generic kernel
set, so the next launch rebuilds it.

```python
# After: register the types this code is known to use, before the first launch.
scale_f32 = wp.overload(scale, [wp.array[wp.float32], wp.float32])
scale_f64 = wp.overload(scale, [wp.array[wp.float64], wp.float64])
wp.launch(scale, dim=x32.shape, inputs=[x32, wp.float32(2.0)])
wp.launch(scale, dim=x64.shape, inputs=[x64, wp.float64(2.0)])
```

Retain the returned overload objects; letting them be collected defeats the
purpose.

**Fix.** Call `wp.overload()` for the type signatures the application routinely
uses, before the first launch.

**Does not apply to** rare or diagnostic types. Registering a specialization a
run never uses adds work. Two independent runs on a bank of five
specializations where the workload needed two found that registering all five
upfront roughly doubled cold module work
(848 ms and 844 ms, versus 414 ms and 424 ms for registering only the two in
use). Once the unused set is large, the extra generated source outweighs the
fixed per-module cost you were trying to amortize.

Register routine types early. Leave rare specializations lazy or put them in a
module with a separate lifecycle.

**Observed.** About 17% less cold module work on CUDA, 5% on CPU.

## CS-7: MathDx LTO without an amortization case

**Signal.** `.lto` artifacts in the cache, plus a large gap between total module
time and native compiler time. The remainder is dominated by LTO work.

**Why.** MathDx-backed tile operations (`tile_matmul`, `tile_cholesky`,
`tile_fft`, ...) generate and link LTO material at first compile. That cost is
large and paid once per cold identity.

An adjoint `tile_matmul` needs its own linked GEMMs, so a forward tile operation
with backward enabled produces roughly three links instead of one. Check
`enable_backward` first (CS-10).

**Fix.** Set the relevant `enable_mathdx_*` option both ways and measure build
time and steady-state kernel time.

One recent architecture produced these tile GEMM results:

| | Build | Steady state | Results |
| --- | --- | --- | --- |
| float32 | 15-25x more expensive with MathDx | no faster, marginally slower | bit-identical |
| float16 | ~14x more expensive with MathDx | **~2.9x faster** with MathDx | differ slightly |

On this float32 path, disabling MathDx saved build time without a runtime loss.
On float16, disabling it caused a runtime regression. Measure the active
precision.

A wrapper that allocates device arrays and returns NumPy can spend several
times the kernel's own cost on transfers. In one case, wrapper overhead hid a
2.9x kernel slowdown. Time the launch or make wrapper overhead small relative
to the work.

If startup and per-request paths need different options, put them in separate
modules.

**Does not apply when** the MathDx path's runtime behavior is required and you
have measured its benefit.

**Observed.** The enabled path's cold module work was roughly an order of
magnitude above the fallback for one tile GEMM, with most of the difference
outside the native compiler timer. Warm cost was also markedly higher.

## CS-8: Concurrent first CPU load races the native JIT

**Signal.** Modules report a successful load, then `Failed to find module` or
`Failed to find forward kernel` at launch. It may not reproduce every run.

**Why.** Concurrent first-use CPU `Module.load()` calls reach a native JIT
creation path that publishes process-global pointers without synchronization.
Two callers can create different JIT instances; a module registered in the one
that gets overwritten becomes unreachable.

**Fix.** Use `max_workers <= 1` for any module load that can target the CPU:
`device="cpu"`, `device=None`, and mixed device lists all count.

```python
wp.load_module(package, device="cpu", recursive=True, max_workers=1)
```

Do not mask this with launch retries or JIT prewarming. Both can hide a
correctness failure without making the native path safe. There is no
evidence-backed concurrent CPU fallback today.

CUDA-only loading is not subject to this restriction. In measurement,
parallel loading of four balanced CUDA modules performed about the same as
serial, so treat it as unproven rather than free.

## CS-9: `module="unique"` on stable kernels

**Signal.** A separate hash-named module per kernel, instead of one shared
module event.

**Why.** `module="unique"` gives a kernel its own module. That isolates
invalidation, but it also means one build and one load per kernel, and
lifecycle-related kernels can no longer share fixed cost.

**Fix.** Same order as CS-1: module-scope definitions, then shared definitions
created before first load, then a named partition, and `module="unique"` last.

Watch for `module="unique"` appearing as a *default* in a kernel-factory
helper. That is where it silently spreads to stable kernels that never needed
it.

**Does not apply to** a kernel whose runtime captures, options, or lifetime
would otherwise invalidate a useful shared module repeatedly, such as a plugin,
a user-supplied transform, a per-request specialization. Keeping those unique
avoids repeated shared-module invalidation.

**Observed.** Stable module-scope sharing beat unique-per-kernel by about 15%
on CUDA. Unique modules did beat repeated shared-module invalidation.

## CS-10: Backward code generated but never used

**Signal.** Adjoint code in the generated source for a module nothing
differentiates.

**Why.** Backward code generation emits adjoint code and supporting source for
every kernel in the module.

On a tile module, an adjoint `tile_matmul` needs its own linked GEMMs, so
backward generation roughly triples the number of MathDx links. Two measured
cases found it was the largest build cost. For tile operations, check this
first (CS-7).

On a tile module, use the module option. Measured on two tile kernels sharing a
module:

| How backward was disabled | Module time | LTO artifacts | `adj_` in `.cu` |
| --- | ---: | ---: | ---: |
| not disabled | 8902 ms | 3 | 108 |
| `wp.set_module_options({"enable_backward": False})` | **3482 ms** | **1** | 2 |
| `@wp.kernel(enable_backward=False)` | 8958 ms | 3 | 2 |

The per-kernel argument removes adjoint source but still links adjoint GEMMs.
Confirm the module option with LTO artifact count or build time.

**Fix.** `@wp.kernel(enable_backward=False)` on a forward-only kernel, or the
module option set before first load for a forward-only module.

Before disabling backward:

1. Test whether anything differentiates the kernel today. Wrap the entry point
   in a `wp.Tape` and inspect `len(tape.launches)`. A NumPy return does not prove
   that the tape missed the kernel.
2. Check whether the public API promises gradients.
3. Ask who owns the module. Both checks above look at the code in front of you,
   which is the right question for an application and the wrong one for a
   library. See "Where you apply it" below.

If none applies, make backward removal explicit, reversible in one line, and
attach the measured result.

**Where you apply it matters as much as whether.** Writing the module option into
a shared library narrows that module for every consumer, on the strength of one
application's profile. Scope the change to the process you measured by setting
the global at its entry point instead; step 2 of `SKILL.md` covers the reasoning.

The ordering rule is the import deadline in "When a module's options are fixed"
(`SKILL.md`): `enable_backward` is one of the six options a module copies out of
`warp.config` as it is created. A global set *after* the library import therefore
reaches only what that library imports lazily, and misses the rest with no error
and no warning. Measured on a Newton scene:

| `wp.config.enable_backward = False` | Cold module work | `newton._src.sim.articulation` |
| --- | ---: | --- |
| not set | 27440 ms | hash `data0137`, 7803 ms |
| set after `import newton` | 16851 ms | hash `data0137`, 7745 ms — untouched |
| set before `import newton` | 10455 ms | hash `25439b5`, 1313 ms |

The middle row is the trap: a third of the available benefit silently missing,
because that module was already sealed at import. Verify by confirming the hash
moved for every module you meant to change — an unchanged hash means the option
never arrived.

Import order is not always yours to control. When the library is already imported
by the time your code runs, reach the module that exists instead of the default
it was built from:

```python
wp.get_module("pkg.module").options["enable_backward"] = False   # before it loads
```

That is still application-scoped — it configures this process, not the library's
source — and it produced the same module identity as setting the global early
(hash `25439b5`, 1352 ms). Both stop working once the module has loaded, at which
point you are in CS-3.

**Does not apply when** a tape demonstrably traverses the launch and the result
is used. Disabling backward there does not error loudly; it makes
differentiation silently wrong. Where forward-only and differentiable kernels
mix, split them into separate modules rather than disabling backward broadly.

**Observed.** Generated source fell by roughly half. Cold module work fell about
6% on CUDA and 2% on CPU in the original study. Three small-kernel modules
measured 26%, 30%, and 37% reductions.

## CS-11: Unroll budget inflates source

**Signal.** Large generated source from static loop expansion, without a
matching rebuild problem.

**Why.** Static unrolling emits the loop body up to the configured bound.

**Fix.** Use the smallest `max_unroll` that preserves required runtime
performance, and set it before first load.

**Does not apply as a compile-time fix at all, on the current evidence.**
Lowering the bound cut generated source by about 39% and produced *no*
measurable cold-time improvement. More source did not mean a slower compile
here. Treat this as a source-size and runtime knob, not a cold-start fix, and
do not spend a possible kernel-throughput regression to buy a compile-time
improvement that measurement does not support.

If you change it, benchmark steady-state kernel time.

## CS-12: The precompiled header is not paying for itself

`wp.config.use_precompiled_headers` defaults to `True`, and it is **not
CPU-only**: `build_cuda()` receives both the flag and an NVRTC header directory,
so it applies on CUDA as well, for toolkit versions below 13.0. On 13.0 and
above `get_nvrtc_pch_dir()` returns `None` because CUDA manages this itself, so
the knob stops mattering. Check the toolkit version before reasoning about it.

**Signal, in either direction.** Compiles are slow across the board and
something sets `use_precompiled_headers = False`; or the default is in force and
the workload is a handful of small modules, where building the header can cost
more than it saves.

**Why it matters.** Warp builds the header once per compiling thread, then each
module behind it compiles far faster. That trade is good for many modules or
large ones and can invert for a few small ones — and because the header is
per-thread, parallel loading (CS-13) multiplies its cost by the worker count.

**Check.** Search the project, its entry points, and its deployment config for
`use_precompiled_headers`. If it is being set to `False`, find out why. Unless
there is a recorded measurement justifying it on this workload, restore the
default.

Do not disable it from library code. It is a global setting that changes
compilation for every Warp user in the process, so it belongs at an application
entry point if anywhere. Any exception requires an application-level benchmark
rather than an argument.

**Measured against the default.** On a five-module CUDA workload with toolkit
12.9, turning the header off cut cold wall clock from 2.38 s to 1.94 s serially,
and combined with a parallel preload reached 1.36 s against a 2.38 s baseline.
That is the inverted case, not the general one: few modules, all small. Measure
before assuming which side of the trade a workload sits on, in either direction.

**Does not apply** on CUDA toolkit 13.0 and above, where Warp passes no header
directory.

## CS-13: Independent modules are built one at a time

Every other mechanism here compiles *less*. This one compiles the same code in
less time, so it is the only fix available when the build is already minimal —
and the only one that survives a constraint against touching library code.

**Signal.** Several independent modules, each compiled exactly once, with no
mechanism above firing. The probe's `overlap_factor` is about 1.0, meaning summed
module time and compile elapsed agree, so nothing overlapped. Cost is spread
across modules rather than concentrated in one.

**Why.** `warp.config.load_module_max_workers` defaults to `0`, which means
serial. Modules otherwise load lazily at first launch, one at a time, so a
workload touching eleven modules pays them end to end even though they are
independent and the machine has idle cores.

**Fix.** Load them concurrently before the first launch, from the application
entry point:

```python
wp.force_load(device=device, modules=[...], max_workers=min(os.cpu_count(), 8))
```

`wp.load_module(name, device=..., max_workers=N)` does the same for one module by
name. Nothing about the compilation changes: same identities, same hashes, same
generated source. Only the scheduling does. This needs no library patch, which is
what makes it reachable when a pinned dependency owns the expensive modules.

**Observed once.** On a Newton rigid-body scene compiling 11 CUDA modules on 8
cores, preloading 8 of them took compile elapsed from 29.4 s to 18.5 s, with 189
launches, 11 module identities, and 4 280 961 bytes of generated source identical
in both arms.

That figure is evidence the mechanism works. It is not a number to expect, and it
should not be quoted to a user as one. What this mechanism pays out depends
almost entirely on how a particular program's compile cost is distributed across
its modules, and that varies far more between applications than anything else in
this file. A program whose cost sits in one large module gains nothing at all; a
program spread evenly across many modules approaches the worker count. Both are
ordinary. There is no typical case to report.

**Bound your own case before writing any code.** Overlapping builds cannot finish
sooner than the slowest single module, so the per-module breakdown you already
measured gives you the answer's upper limit:

```text
best possible elapsed  =  largest module time
ceiling speedup        =  sum of module times / largest module time
```

For the run above: 29.3 s over 12.3 s is a 2.4x ceiling, so 58% was the most that
was available and 37% is what landed. The shortfall is structural — code
generation serializes under `_codegen_lock`, the surrounding Python is GIL-bound,
and those 8 cores were shared. Expect to land somewhere between that ceiling and
nothing, and report the ceiling alongside the result so the reader can see how
much of the available headroom you captured.

A ratio near 1.0 means one module is the whole cost and this mechanism has
nothing to offer. Compute it first and skip the work.

**Read the result on the clock, not the sum.** Summed module timers went *up*,
29.2 s to 41.7 s, because overlapping builds contend and each module's timer
absorbs the others'. That is the metric inversion described in
`measurement.md`; judge this mechanism on compile elapsed.

**Limits, and they are real.**

- **CUDA only.** Warp's native CPU JIT first-use path is not synchronized, so
  concurrent first loads can report success and then fail kernel lookup. Gate on
  `device.is_cuda`; see CS-8.
- **The ceiling is well under linear.** Code generation is serialized under a
  global `_codegen_lock` and the rest is GIL-bound Python, so only native
  compilation genuinely overlaps. A single dominant module is a hard floor: no
  worker count makes a 12 s module finish sooner.
- **Never pass `modules=None`.** That loads every registered module, not the ones
  you need. In the measured case 94 modules were registered against 11 used, so
  the blanket form would compile roughly 8.5x the code and lose badly.
- **A preload whose block dimension does not match the launch compiles the module
  twice.** With no explicit `block_dim`, `force_load` preloads the variants
  already loaded on that device, falling back to the module default when there
  are none — which is the situation during a prewarm, because nothing has loaded
  yet. So a module whose kernels are launched at a block dimension other than its
  module default gets built at the default first and at the real dimension
  second. This is easy to walk into: a five-module prewarm measured here turned
  five builds into nine and came out slower than no prewarm at all. Either declare
  each module's real block dimension as its module default,
  `wp.set_module_options({"block_dim": N})`, so an unqualified preload lands on
  the right variant, or pass `block_dim` explicitly. Leave modules whose block
  dimension is chosen at runtime to load lazily.
- **Do not swallow the error.** `load_module` raises `RuntimeError` for a module
  that is not imported or holds no Warp code. An `except RuntimeError: pass`
  around the call turns a typo in a module name into a silent no-op that still
  looks like a fix. Let it raise, or log it.

**Maintenance.** A hardcoded module list names another package's internals. When
that dependency reorganizes, the list degrades quietly — you lose the speed-up,
you do not break. Regenerate it from a verbose cold run, and assert the list
still resolves.

**Does not apply when** one module dominates the total, when the target is CPU,
or when the cache is persisted, since a warm run has nothing to overlap.

## Known unknowns

Do not present these as established:

- The MathDx break-even point in launches. Not measured.
- How CS-13 scales at other module counts, size balances, and worker counts. One
  workload was measured; the mechanism is established, the magnitude is not.
- Whether cache artifacts move safely between hosts, containers, drivers,
  architectures, or Warp builds. Only same-host reuse was tested.
- Debug mode, line info, optimization levels, and custom compiler flags. These
  were never compared and remain hypotheses.
