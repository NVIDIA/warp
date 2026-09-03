---
name: warp-compile-time-optimizer
description: >-
  Use when compile time or startup time is the problem in code that uses Warp:
  a request to improve, optimize, or cut compile times; an app that is slow to
  start or stalls at the first wp.launch; seconds of compiling before real work
  begins; JIT modules recompiling on every run or every CI job. Only applies
  when the code being optimized uses Warp kernels. Not for steady-state kernel
  runtime, memory, correctness, building Warp itself from source, or nvcc/C++
  build times.
license: Apache-2.0
compatibility: Requires Python 3.10+ and an installed warp-lang package. A CUDA device is needed to diagnose CUDA-specific mechanisms.
allowed-tools: Bash, Read, Edit, Write, Glob, Grep, env
metadata:
  version: "0.1.0"
  author: "Warp Team <warp-python@nvidia.com>"
  tags:
    - warp
    - compilation
    - cold-start
    - startup-latency
    - kernel-cache
    - gpu
---

# Warp cold-start compile time

## Start with a runnable command

The probe runs the target in a subprocess, writes only to temporary
directories, and needs no network, external tool servers, or Warp checkout.

| Script | Purpose | Arguments |
| --- | --- | --- |
| `scripts/warp_compile_probe.py` | Measure isolated cold/warm compilation and launches. | `measure [OPTIONS] -- COMMAND...`; use `--help`. |

Use `run_script("scripts/warp_compile_probe.py", args=[...])` when supported;
otherwise use the Python command below. The target must run to completion.

## Compilation model

Warp compiles modules, not individual kernels. A module's identity is:

```text
(live kernel & function set) x (module options) x (CUDA block_dim) x (generic instances)
```

Each identity requires code generation and native compilation for the full
module.

Cold-start cost is roughly:

```text
number of distinct module identities you touch  x  size of each module
```

Reduce it in two ways:

1. Stop identity churn. A module that changes after loading compiles again.
2. Stop source duplication. A module used at three block dimensions compiles
   every kernel three times.

Deleting one kernel from a module that still builds saves only part of one
build. Removing an unnecessary module identity saves the full build.

When neither applies, overlap independent CUDA builds (CS-13). This changes
when work happens, not how much is compiled, so judge it on elapsed time.

## When a module's options are fixed

Options have two deadlines:

1. **Module creation (Python import).** The module copies `enable_backward`,
   `max_unroll`, `lineinfo`, `deterministic`, `deterministic_max_records`, and
   `compile_time_trace` from `warp.config`. Setting a global later is silently
   ignored by that module. `default_grid_stride` is the exception.
2. **First load.** Before compilation, change the existing module with
   `wp.set_module_options()` or `wp.get_module(name).options`. Changing an
   option after load creates a new identity and rebuilds the module (CS-3).

| Missed deadline | Symptom | Cost |
| --- | --- | --- |
| `wp.config.*` set after import | hash unchanged, option silently absent | the entire benefit, invisibly |
| module options set after load | a second hash, module builds twice | one extra build, visible in the trace |

After changing an option, confirm the hash moved for every target module. An
unchanged hash means the option never arrived.

## Preserve behavior

Change how Warp compiles the code, not the workload.

Do not delete or merge kernels to claim a gain. Apparently redundant stages
may preserve ownership, aliasing, retained outputs, numerical boundaries, or
API behavior. Fix duplication at the module level.

Preserve every launch and its order, dimensions, dtypes, devices, block
dimensions, gradients, numerical modes, dynamic/plugin behavior, and public
API signatures. Keep kernel names when moving definitions to module scope
because logs, cache artifacts, and external tools expose them.

## Instructions

### 1. Find the real cost, and confirm it is compilation

Ask what command the user actually waits on, then measure it cold:

```bash
python scripts/warp_compile_probe.py measure --samples 3 \
    --json baseline.json -- <the user's command>
```

The probe gives each sample private `WARP_CACHE_PATH`, `WARP_CACHE_ROOT`, and
`CUDA_CACHE_PATH` directories, enables module timers, and records launches. It
creates those directories under the system temporary directory and removes
them itself, so isolating a sample never requires writing a cache into the
project or deleting anything to re-measure. Isolate a hand-rolled sample the
same way. Never clear a live cache with `wp.clear_kernel_cache()` or
`wp.clear_lto_cache()`; clearing is not isolated and can disrupt other
processes.

Read the probe output before source. If compilation is a small part of wall
time, report the real bottleneck and stop. For libraries and tests, use the
smallest command that compiles the workload's modules.

Modules that each compiled once, with no repeated hashes, block-dimension
variants, or LTO, have no structural churn. This rules out redundant builds,
not oversized builds; still check cache reuse (CS-2), backward codegen
(CS-10), unrolling (CS-11), the precompiled header (CS-12), and overlap when
several CUDA modules remain (CS-13).

Every sample also re-runs the command against the cache it just populated. If
warm module work is not near zero, diagnose cache reuse (CS-2) before changing
module structure.

### 2. Ask about runtime tradeoffs when needed

Apply ordering fixes, lifecycle grouping, and option hoists without asking.
Ask before changing `fast_math`, `max_unroll`, or a MathDx/tile implementation:

> Some of these knobs cut compile time but can make the compiled kernels
> slower or change numerics. Are you optimizing a fast edit-run loop (where
> slower kernels are usually fine), or production startup (where they usually
> are not)?

If the user is unavailable:

- Leave numerics-changing options and implementation swaps alone.
- Before removing a capability such as backward codegen, test whether it is
  used; report what is removed, the measured benefit, and how to revert.
- Set global `wp.config.*` options at application entry points, not in library
  code.

Record declined options and their measured benefits in the step 6 ledger.

#### Match the scope of the change to the scope of the evidence

A profile supports a change to the measured application, not every consumer of
a shared library. Repository searches also miss out-of-tree and future callers.
For example, a forward-only application does not justify disabling gradients
inside a solver library that another application differentiates through.

Scope the option to the measured process, before importing the library:

```python
import warp as wp

wp.config.enable_backward = False   # must precede the library import

import the_library
```

This also reaches every module the application loads. CS-10 covers the silent
import-order trap. If only a library change works, send its maintainers the
measurement and let them decide the contract.

### 3. Diagnose from the measurement, not from reading the source

The probe prints every compiled module identity with its name, hash, device,
and block dimension, then names which modules built more than once. Match what
you see:

| What the probe shows | What it means | Where to look |
| --- | --- | --- |
| One module name, several **hashes** | Identity churn: its kernel set, options, or generic instances changed after it first loaded | CS-1, CS-3, CS-6 |
| One module name, several **block_dim** values | The whole module is recompiled per block dimension (CUDA) | CS-5 |
| Many one-kernel modules in one feature | Fixed per-module cost repeated | CS-4 |
| A hash-named module per kernel | `module="unique"` used on stable kernels | CS-9 |
| Big gap between module time and native compile time, plus `.lto` artifacts | MathDx/LTO setup | CS-7 |
| `(compiled)` on a run that should have been warm | Cache is not being reused | CS-2 |
| Modules load, then "Failed to find module" | Concurrent CPU JIT first-use race | CS-8 |
| Large generated source, no rebuild problem | Unroll budget | CS-11 |
| Adjoint code in a module nothing differentiates | Backward codegen | CS-10 |
| Compiles slow across the board, or a few small modules on CUDA below toolkit 13 | The precompiled header is turned off, or is not paying for itself | CS-12 |
| Several independent modules, each built once, `overlap_factor` near 1.0 | Builds are running one at a time; parallel loading is off by default | CS-13 |
| An option you set changed nothing, and that module's hash is unchanged | It was assigned after the module was created, so it never arrived | "When a module's options are fixed" |
| No row above fires | Nothing is being built redundantly; the cost is the size of the builds themselves | Step 6 |

`references/mechanisms.md` has one section per mechanism: how to confirm it,
the fix, its limits, and its failure mode. Read only the sections selected by
the measurement.

### 4. Choose module boundaries deliberately

Group kernels in one module only when they share:

- lifecycle: they are defined, loaded, and invalidated together;
- option set: they need the same `fast_math`, `enable_backward`, `max_unroll`,
  and MathDx settings;
- stable block dimension on CUDA.

Kernels with the same lifecycle but different stable block dimensions should
not share a module because each would compile twice. Separate kernels with
independent lifecycles too.

Kernels whose block dimension varies at runtime (chosen from input size, say)
have no stable mapping, so keep them in their own module rather than dragging
a whole shared module into an extra variant.

Prefer the least invasive change that removes a build. Ordering fixes and
option hoists are cheaper and safer than re-architecting module layout;
regroup only when fixed per-module cost or block-dimension duplication
dominates.

`wp.set_module_options()` targets its calling Python module, not kernels with
an explicit `module="pkg.name"`. Either use a real Python module or update the
named module before it loads:

```python
wp.set_module_options({"enable_backward": False})  # at module scope

wp.get_module("pkg.name").options.update({"enable_backward": False})
```

Do not pass `wp.get_module()` to `wp.set_module_options(module=...)`, or use
`@wp.kernel(module_options={...})` without `module="unique"`. Per-kernel
`enable_backward=False` has a tile-module exception covered by CS-10. Confirm
the module hash after every option change.

### 5. Verify

```bash
python scripts/warp_compile_probe.py measure --samples 3 \
    --json candidate.json -- <the same command>
python scripts/warp_compile_probe.py compare baseline.json candidate.json
```

`compare` rejects changed launch topology and treats a result inside
`max(1% of baseline, 2 x baseline MAD)` as inconclusive.

For `BUILDS OVERLAPPED`, judge scheduling changes on compile elapsed rather
than summed module timers. The required warm pass supplies that clock. See
`references/measurement.md`.

Then check what the probe cannot see:

- Diff numeric output and run the project's tests or entry points.
- After changing `enable_backward` or boundaries, verify a gradient path.
- After changing `fast_math`, `max_unroll`, MathDx, or an implementation,
  benchmark steady-state runtime.
- Exercise uncovered dynamic kernels, dtypes, profiles, and modes.

### 6. Report results

Read "Reporting results" in `references/measurement.md`. Report:

Describe every optimization in plain language: name the behavior, the evidence,
and the effect. For example, write "moved module options before the first load
to avoid a redundant rebuild," not "applied CS-3." Treat `CS-*` labels as
internal navigation aids, not user-facing explanations.

| Option | Measured | Why not taken | To take it |
| --- | ---: | --- | --- |
| `enable_backward=False` on `pkg.solver` | −38% cold | a live tape traverses these kernels | set at the entry point, then re-check adjoints |
| `max_unroll=4` | −2%, inside noise | changes generated code for no measured gain | — |

- before/after medians and sample counts;
- the reduction and residual cost, sized against the original complaint;
- mechanisms fixed, ruled out, measured and declined, or not reached;
- tradeoffs and behavior not verified;
- the ledger above for declined or incompletely verified options;
- the measured value of relaxing any constraint that blocked a fix.

State only what the evidence supports. "No structural churn" does not mean
"optimal" or "irreducible." Measure declined levers when practical; label any
estimate untested. Before reporting no available fix, check CS-13. For a
possible module split, first measure a one-kernel module with the same options
to establish the repeated fixed cost.

## Troubleshooting

Run the target command directly before debugging the probe. See
`references/measurement.md` for cache/noise issues and
`references/mechanisms.md` for mechanism-specific failures.

## Limitations

Two rules override any gain:

- Isolate both Warp and CUDA caches for every cold sample.
- Keep `max_workers <= 1` when a load can target CPU, including `device=None`
  and mixed device lists. Concurrent CPU first loads can lose kernels; retries
  do not make them safe. CUDA-only loading is unaffected.

Measurements are environment-specific: cold times move with CPU, GPU, driver,
toolchain, and Warp version. Mechanisms transfer; numbers do not. Read the
known unknowns in `references/mechanisms.md` before making broad claims.

## Reference files

- `references/mechanisms.md`: the thirteen compile-time mechanisms, each with
  its confirming signal, fix, applicability limits, and failure mode. Read the
  sections your measurement points to.
- `references/measurement.md`: measurement protocol, what each metric does
  and does not mean, reporting guidance, log examples, and manual measurement.
