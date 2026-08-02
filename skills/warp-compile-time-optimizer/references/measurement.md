# Measuring Warp cold compilation

Read this when the probe does not fit the application, when you need to explain
a number, or when a result looks too good.

## What the probe does

`scripts/warp_compile_probe.py measure` runs a command in a subprocess with:

- fresh directories for `WARP_CACHE_PATH`, `WARP_CACHE_ROOT`, and
  `CUDA_CACHE_PATH`;
- `warp.config.verbose = True` for module and compiler timers;
- a wrapped `warp.launch` that records each kernel, module, dim, device,
  dtypes, block dimension, and adjoint flag.

Instrumentation is injected as a `sitecustomize` module on `PYTHONPATH`, so it
works with `python -m pkg`, `python script.py`, `uv run`, and test runners. It
chains to any existing `sitecustomize` without writing into the project.

Each sample then re-runs the command against the cache it just populated. That
warm pass is not optional, because two things depend on it: distinguishing
expensive cold compilation from compilation repeated on every run, and supplying
the `warm wall` term that compile elapsed is derived from. `--warm` is still
accepted so older commands keep working, but it does nothing.

`compare` diffs two reports: launch topology first, then cold module work
against a noise band, then structural counts.

## What each metric means

**Cold module work:** the sum of Warp's per-module timers for events
marked `(compiled)`. It excludes process startup, imports, and data loading.
This is the primary number for a change to *what* gets built, and the wrong
number for a change to *when* it gets built — see "Two clocks" below.

**Compile elapsed:** cold wall clock minus warm wall clock. The warm pass runs
the same command against a populated cache, so startup, imports, and the
workload appear in both and cancel, leaving the elapsed time attributable to
compilation. Unlike the sum of per-module timers, it cannot double-count builds
that overlapped, so it is the number a user actually waits on.

**Native compile time:** the sum of Warp's `Compile x86` / `Compile CUDA`
timer lines. The share of cold module work spent inside the native toolchain.

**The remainder** (cold module work minus native compile time) is *not*
"code generation time." It includes source and metadata I/O, cache
installation, binary loading, and LTO work. Module hashing happens before the
timer starts and is not in either number. A large remainder alongside `.lto`
artifacts is the MathDx signature (CS-7); a large remainder otherwise is not
self-explaining.

**Module identities:** distinct `(module, device, block_dim)` triples that
compiled. Unlike timing, this count does not vary with machine load.

**Generated source bytes:** total `.cpp` and `.cu` bytes in the cache. Use this
to corroborate timing inside the noise band.

**LTO artifact count:** `.lto` files left in the cache. This discrete count
shows whether a tile-workload change reached the linker, often moving from 3
to 1 or 1 to 0.

**Warm module work:** the same command re-run against the populated cache.
Near zero means compilation is a cold-start-only cost.

**Launches:** the ordered topology used to verify that the workload did not
change.

## Two clocks, and when each one lies

Warp reports one duration per module. Summing them measures *work* and is stable
against machine load, which is why it is the default. But a sum assumes the
builds did not run at the same time. When they overlap, each module's timer
absorbs the others' contention, so the sum grows while the elapsed time falls.

The result is that a change which compiles the same code concurrently instead of
serially looks like a large regression on cold module work and a large
improvement on the clock. Both numbers are correctly measured; they answer
different questions.

The probe reports `BUILDS OVERLAPPED` when summed module time exceeds compile
elapsed by more than 15%, and `compare` refuses to call a slowdown a regression
when the launches, module identities, and generated source bytes are all
unchanged — identical artifacts mean nothing extra was built, so the extra time
is contention, not work. In that state it judges the change on compile elapsed.

Which to report:

| Your change alters | Headline | Why |
| --- | --- | --- |
| what gets compiled (kernel set, options, boundaries) | cold module work | work really changed; stable against load |
| when or how builds are scheduled (preloading, concurrency) | compile elapsed | the sum double-counts overlap |

When you report elapsed, carry the structural counts with it — identical
identities, hashes, and generated source bytes are what prove no work was added.

## Two diagnoses the probe reports directly

**`CACHE NOT REUSED`:** the warm pass still costs most of the cold pass. The
application is discarding its own compiled kernels between runs, and no amount
of module restructuring will help. See CS-2.

**`LINK-DOMINATED BUILD`:** most module time is outside the native
compiler and `.lto` artifacts are present. The build is bounded by device-code
linking for tile operations. See CS-7, and check `enable_backward` first
(CS-10) because adjoint tile operations need their own links.

## Reading the module log directly

Warp prints, with `warp.config.verbose = True`:

```text
Module <name> <hash> load on device '<device>' (block_dim=<n>) took <ms> ms (compiled|cached|error)
Compile CUDA (<details>) took <ms> ms
```

The hash is the useful field. Two events with the same **name** and different
**hashes** mean the module's kernel set, options, or generic instances changed
after its first load. Same hash at different **block_dim** values means CUDA
per-block-dimension duplication.

Three common shapes, shown as time, device, block dimension, hash, and name:

```text
# Identity churn: one name and block dimension, several hashes (CS-1/3/6).
312.44  cuda:0  256  6eb476a  app.filters
289.10  cuda:0  256  17e6601  app.filters

# Block-dimension duplication: one name and hash, several dimensions (CS-5).
361.16  cuda:0  128  9004220  app.stages
357.95  cuda:0  256  9004220  app.stages

# Nothing structural: one identity, no variants or LTO.
1772.27  cpu    1    b1faa06  __main__
```

The first two can cost the same number of builds but require different fixes.
The third rules out redundant builds, not oversized generated code.

## Protocol

- Use a fresh cache for each cold sample. Reused artifacts invalidate the
  measurement.
- Isolate both cache layers on CUDA. Setting `WARP_CACHE_PATH` alone leaves
  the driver cache at `~/.nv/ComputeCache` donating PTX-to-SASS results, which
  can hide a difference. Set `CUDA_CACHE_PATH` too.
- At least three cold samples; five if you are reporting a small effect.
- Same machine, same device, same Warp version for every arm.
- Report median and MAD because build times have outliers.
- Interleave baseline and candidate runs. Block ordering on a busy machine can
  turn drift into an apparent effect.
- Never carry an absolute millisecond figure from one machine to another.

## The noise band

```text
band = max(0.01 * baseline_median, 2 * baseline_MAD)
```

A change smaller than the band is inconclusive.

At small magnitudes, a percentage band is meaningless. One percent of a
1.5 ms warm measurement is 15 microseconds, far below host jitter, so such a
band can flag noise as a regression. Below a few milliseconds, prefer compiled
event counts, module identities, or LTO artifacts.

## Verifying the workload, in order

1. **Launch topology.** `compare` does this automatically and is the fastest
   check. Order, count, dim, dtypes, device, and block dimension must all
   match. It keys on structure rather than kernel name, because moving a kernel
   from a closure to module scope may rename it. Keep the original name for
   external tooling.
2. **Numeric output.** Run the project's own tests or entry points and diff
   against the baseline output.
3. **Gradients.** If you touched `enable_backward` or moved kernels between
   modules, exercise a `wp.Tape` path and check the adjoints. A broken adjoint
   is silent. Note that a function returning a NumPy array can still have
   recorded launches on a tape.
4. **Steady-state runtime.** Required if you touched `fast_math`, `max_unroll`,
   MathDx, or swapped an implementation. Cold-compile measurement is blind to a
   kernel you made slower.
5. **Uncovered paths.** Re-run modes, dtypes, profiles, and dynamic paths your
   measured command did not exercise.

## Measure the thing you changed

Two common measurement mistakes:

**Timing a wrapper instead of a kernel.** A function that allocates
device arrays and returns NumPy can spend several times the kernel's own cost on
transfers. One measured case had a 2.9x kernel slowdown that was nearly
invisible end to end. Time the launch, or make the wrapper's overhead small
relative to the work, before concluding a swap was free.

**Reading source instead of build output.** On a tile module,
`@wp.kernel(enable_backward=False)` removes adjoint source while leaving the
adjoint device-code links in place. The generated `.cu` looks identical to the
module-option version, so check LTO artifacts or build time instead (CS-10).

Verify at the layer your change affects. Compilation structure appears in
module events; caching appears in the warm pass; generated device code appears
in artifacts and runtime.

## When a result looks too good

For a large reduction, check:

- Did the launch topology change? A dropped launch is the usual cause.
- Did the caches actually stay isolated between arms, **both layers**?
- Did the run exit non-zero, or fewer modules compile because something failed
  early?
- Did work move into a prewarm step or an unmeasured first request?
- Did a fallback silently replace a required implementation?

A verified 5% is worth more than an unverified 80%.

## When cold time does not improve

A cache fix shows no cold-time improvement because cold work is unchanged. The
improvement is that compilation now happens once instead of every invocation,
and it appears in the warm pass and in end-to-end time across repeated runs.
Report that measure instead of the cold comparison.

Similarly, on CPU a large fixed toolchain cost dominates every run, so a real
structural fix can show a small percentage. Removing three of four module builds
can be an 8% improvement if the toolchain accounts for the remaining 92%.

## Reporting results

Report before/after medians, sample counts, fixed and declined mechanisms,
tradeoffs, and behavior you did not verify. Call results inside the noise band
inconclusive. Give both the reduction and residual cost; a verified 3% reduction
does not resolve a startup the user described as painful.

### Bound claims to the search

"I did not find a further reduction" describes the search. "This compiles as
fast as possible" makes an unsupported claim about the code. No repeated hashes,
block-dimension variants, or LTO establish "no structural churn," but do not
prove each build has a minimal kernel set, adjoint footprint, or unroll budget.
Report what you checked, the signal that ruled each mechanism out, and what you
did not reach. Include the per-module breakdown so others need not remeasure.

Do not invent a small change to avoid a negative result. Also do not stop after
the work-removal mechanisms: check CS-13, which can overlap required CUDA builds
without a library patch.

### Measure declined levers

Measure options you decline or cannot ship when practical. Compilation cost is
hard to predict because launching one kernel from a large module still compiles
the module's entire live kernel set. If measurement is impossible, label the
number an untested estimate and explain why.

Before splitting a dominant module, compile a throwaway one-kernel module with
the same options. This estimates the fixed cost repeated by every new module and
bounds the possible benefit. In one measured case, a 167 ms fixed floor inside
a 467 ms module capped a three-way split at about 20%.

Keep a ledger:

| Option | Measured | Why not taken | To take it |
| --- | ---: | --- | --- |
| `enable_backward=False` on `pkg.solver` | -38% cold | a live tape traverses these kernels | set at the entry point, then re-check adjoints |
| `max_unroll=4` | -2%, inside noise | changes generated code for no measured gain | no action |

Include applied options in the same table when verification did not reach the
layer they affect. For example, a runtime knob checked only against compile
time remains an open question.

### Price blocking constraints

If a pinned dependency, required capability, or lack of persistent storage
blocks every fix, measure what relaxing each constraint would buy and present
the values side by side. A rule against persistent state may still permit a
read-only prewarmed artifact. If no constraint is worth reopening, say what
would need to change for another result.
