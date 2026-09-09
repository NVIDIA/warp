# CPU Block Fiber Implementation Plan

**Status**: Ready for execution

**Issue**: [GH-1638](https://github.com/NVIDIA/warp/issues/1638)

**Design**: [design/cpu-block-dim-fibers.md](design/cpu-block-dim-fibers.md)

**Target worktree**: `C:\src\warp-fiber-block`

## Outcome

Implement opt-in CPU `block_dim > 1` execution with cooperative stackful fibers
on every supported CPU architecture. The first implementation is a correctness
foundation for CPU/CUDA equivalence testing in CI, not a SIMD or general
cooperative-operation optimization project.

The completed series must have these properties:

- `wp.config.enable_cpu_blocks` exists and defaults to `False`;
- valid explicit CPU block dimensions are forced to 1 while the option is
  disabled and honored while it is enabled;
- an explicit CPU `block_dim > 1024` raises `ValueError` before the config can
  force it to 1;
- an enabled CPU block dimension from 2 through 1024 runs only the active lanes
  of each block as fibers on one host thread;
- effective `block_dim=1` emits the historical direct task loop, a literal lane
  index of zero, and compile-time no-op barriers, with no fiber-related reference
  or dispatch overhead;
- Linux x86-64, Linux AArch64, macOS arm64, Windows x86-64, and Windows ARM64 all
  build and pass functional fiber tests;
- AddressSanitizer builds reject enabled CPU blocks greater than 1 with
  `NotImplementedError`; and
- every existing CPU tile primitive used by the supported public API is either
  covered by equivalence tests or fails a development assertion. There is no
  silent fallback to `block_dim=1` after opt-in resolution.

GH-1638 also describes executing shared cooperative operations once per block
where possible. That broader efficiency work remains follow-up scope. In this
series, use a leader section only where it is required to avoid duplicated
externally visible effects or otherwise establish correctness.

## Scope lock

The following localized non-SIMD mechanisms are part of the foundation. Their
benefit justifies including them before the feature is exercised broadly in CI.

| Mechanism | Complexity | Benefit | Decision |
| --- | --- | --- | --- |
| Exact `block_dim=1` code specialization | Medium | Critical: protects every existing CPU launch from added dispatch, TLS, and barrier overhead. | Required. |
| Active-tail fiber creation | Low | High for partial final blocks; avoids up to 1023 useless contexts. | Required. |
| Thread-local fiber pool | Medium | High for ordinary multi-block launches; removes repeated guarded-stack creation. | Required where arena state can be safely rebound. |
| Direct generation-safe bitset scheduler | Medium | High and broad; bounds lane selection to at most sixteen word checks and avoids known epoch-aliasing deadlocks. | Required in the release path. |

Do not add SIMD/SPMD lowering, scalar fission, lane gangs, cross-block OS-thread
parallelism, general allocation/body splitting, cooperative-operation effect
classification, primitive-wide leader-combine optimization, barrier reduction,
adaptive stack sizing, a custom Windows context switch, or ASan fiber hooks.

## Execution rules

1. Execute the commits below in order. Each commit must build and its focused
   tests must pass before starting the next one.
2. Use the exact imperative commit subjects shown below and commit with
   `git commit -s`.
3. Preserve unrelated worktree changes. Do not reset or rewrite user changes.
4. Use `uv run` for Python and test commands. Never use `pytest`, and never clear
   Warp's kernel or LTO cache from a test.
5. Add every new test module under `warp/tests/` to `default_suite` in
   `warp/tests/unittest_suites.py`.
6. Rebuild native libraries after commits that change `warp/native/`. Use
   `uv run build_lib.py --quick` only after confirming the installed CUDA driver
   is at least as new as the selected toolkit; otherwise use
   `uv run build_lib.py`.
7. Run `uvx pre-commit run --files <changed files>` before every commit.
8. The prototype branch is evidence and a porting reference, not a source to
   cherry-pick wholesale. Reapply only the behavior assigned to the current
   commit and adapt it to current `omniverse/main`.
9. If implementation work invalidates a design statement, update
   `design/cpu-block-dim-fibers.md` in the same commit that changes the decision.
   Do not let the implementation silently diverge from the design.

## Preflight and baseline evidence

Before changing runtime code:

1. Confirm the branch is based on the intended `omniverse/main` and that the only
   expected untracked files are this plan and the design document.
2. Build the baseline and run representative CPU kernels at effective
   `block_dim=1`: scalar indexing, array elementwise, shared tile, register tile,
   reduction, and matrix multiplication, including backward kernels.
3. Save generated C++, optimized LLVM IR when available, undefined-symbol lists,
   and normalized disassembly outside the repository. Record the compiler and
   build configuration used. These artifacts are the final zero-overhead
   comparison baseline.
4. Run the existing focused tile tests that the later commits will touch so that
   pre-existing failures are known.

Timing data may be collected as a warning signal, but do not add timing-based
assertions. The acceptance proof for `block_dim=1` is unchanged generated control
flow and absence of fiber/scheduler references.

## Runtime shape

The dependency direction should remain simple:

```text
Python launch resolution
    -> CPU specialization/cache key
        -> generated B=1 direct loop
        `-> generated B>1 block loop and lane thunk
            -> CPU block scheduler
                -> platform fiber context
```

Generated kernels may call the block runtime. The fiber implementation must not
depend on Python, code generation, or tile primitive details. Scheduler state is
new for every block even when worker fiber stacks are reused.

## Prototype reference map

These commits on `ncapens/tile-mem-fibers-rebased` are useful behavioral
references. Inspect their patches, but do not inherit later SIMD machinery or
old GH-1413 wording.

| Area | Reference commits | Porting note |
| --- | --- | --- |
| Fiber and initial block runtime | `e537d37f0` | Split the primitive and scheduler into separate commits in this series. |
| CPU lane/sync hooks | `a8073f4a5`, `673d4a676` | Keep the literal-zero typed lane accessor and make the one-lane barrier disappear at compile time. |
| CPU block code generation | `80b561f0c`, `3f8fcbaac` | Preserve the exact direct loop for `B=1`; add active-count dispatch rather than creating inactive tail fibers. |
| Launch behavior | `bc8fbb130`, `9054e84e2` | Replace unconditional opt-in with the default-disabled resolver and the 1024 limit in the new design. |
| Core correctness tests | `bc3c7e111`, `86c0aaeca` | Keep CPython frames off suspended fiber stacks. |
| Shared tile state | `8e0b8bfa2`, `8455cce9d`, `9867e6540` | Port correctness fixes only. |
| Partial collectives | `f21980d9b`, `0d064c465` | Start with simple barrier-correct implementations; omit later leader/barrier optimizations. |
| Linear algebra and FFT | `f1c67be18`, `762776adf`, `2e3c6ac0c` | Port synchronization and shared-scratch correctness fixes. |
| Sort and queries | `3dc3b272d`, `38ce53ef7` | Prevent duplicate effects and make scratch block-safe. |
| Fiber pooling | `66b94979c` | Retain the bounded lazy pool, but make rearming and architecture behavior explicit. |
| Direct bitset scheduler | `864d591b7`, `0b4c87760`, `3fe2a2649` | Use absolute generations and keep a scalar test oracle. |

Do not port the optimization sequence beginning with `961b5650d` or the later
effect-classification, gang, and SIMD commits.

## Commit series

### Commit 1: Establish the design baseline

**Subject**: `Document CPU block fiber design (GH-1638)`

Include:

- `design/cpu-block-dim-fibers.md`; and
- this implementation plan.

Verify that the design is `Proposed`, links GH-1638, lists every supported CPU
architecture, records the default-disabled configuration, caps CPU blocks at
1024, preserves the exact one-lane path, and defers ASan support.

Validation:

```powershell
uvx pre-commit run --files design/cpu-block-dim-fibers.md CPU_BLOCK_DIM_FIBERS_IMPLEMENTATION_PLAN.md
git diff --check
```

### Commit 2: Add portable fiber contexts

**Subject**: `Add portable CPU fiber contexts (GH-1638)`

Add the internal `wp_fiber_*` abstraction in
`warp/native/cpu_fiber.h` and `warp/native/cpu_fiber.cpp`, plus any narrowly
scoped assembly sources required by the POSIX ABIs. Integrate those sources into
`CMakeLists.txt`, `build_lib.py`, and `build_llvm.py` for every applicable native
target.

Implementation requirements:

- expose create, destroy, switch, active, and finished operations through a
  small internal C ABI;
- use guarded, page-aligned POSIX stacks and `CreateFiberEx` reservations on
  Windows;
- request a 1 MiB usable stack for block workers; low-level tests may request a
  smaller explicit size;
- preserve stack alignment, all ABI-required nonvolatile integer and
  floating/vector state, and floating-point control state;
- handle a Windows thread already converted to a fiber by its host;
- use the native Windows fiber API on both x86-64 and ARM64; do not add a custom
  Windows context switch;
- return a clear failure result on allocation/conversion failure; and
- do not add ASan switching hooks in this series.

Add `warp/tests/test_cpu_fiber.py` and register it in `default_suite`. Run each
crash-prone scenario in a subprocess. Do not suspend a Python callback frame on a
fiber: use native entry functions or a native test harness while a context is
suspended. Cover main-context identity, create/switch/destroy, return-to-main,
repeated ping-pong, a 1024-context stress case, deep calls, stack alignment,
live nonvolatile values, and the Windows pre-converted-thread case. Add a guarded
stack overflow expected-crash scenario where the platform harness can express it
reliably.

Validation:

```powershell
uv run build_lib.py
uv run warp/tests/test_cpu_fiber.py
```

CI gate: the test must run, not merely compile, on Linux x86-64, Linux AArch64,
macOS arm64, Windows x86-64, and Windows ARM64.

### Commit 3: Add the deterministic block scheduler

**Subject**: `Add CPU block fiber scheduler (GH-1638)`

Add `warp/native/cpu_block_runtime.h` and
`warp/native/cpu_block_runtime.cpp`. Register the JIT-resolvable runtime symbols
in `warp/native/clang/clang.cpp` and add the sources to all required native build
targets.

The runtime contract must accept both the logical block dimension and the active
lane count. It must:

- create scheduler records only for lanes `[0, active_count)`;
- assign a distinct lane index and a common block context to each fiber;
- use two 1024-bit sets represented by sixteen `uint64_t` words;
- tag scheduler state with absolute `uint64_t` generations, not parity alone;
- hand off directly to the lowest-numbered unfinished lane that is behind;
- make a resumed waiter compare its own saved generation before consulting a
  possibly rotated bitset;
- remove a returning lane from every outstanding set;
- restore any enclosing TLS/runtime context after the block finishes; and
- report allocation failure through Warp's existing native error mechanism,
  never by retrying at `block_dim=1` and never by throwing C++ exceptions across
  the generated C ABI.

Keep the scalar scheduler as an independent model in tests, not as a release
runtime mode. Add `warp/tests/test_cpu_block_runtime.py`, register it, and compare
native schedules with the model for exhaustive small cases and randomized cases
through 1024 lanes. Cover bitset boundaries, thousands of generations, early
return before/after barriers, lane 0 returning first, sparse survivors, one
survivor, full blocks, and partial active prefixes. Native switching portions of
the tests must remain off suspended CPython frames.

Validation:

```powershell
uv run build_lib.py
uv run warp/tests/test_cpu_fiber.py
uv run warp/tests/test_cpu_block_runtime.py
```

### Commit 4: Pool reusable block fibers

**Subject**: `Pool reusable CPU block fibers (GH-1638)`

Add a lazy thread-local worker pool to `cpu_block_runtime.cpp`:

- grow only to the largest `active_count` seen on that host thread;
- retain at most 1024 worker contexts per host thread;
- rearm a worker only while it is suspended at a known scheduler point;
- recreate all scheduler, generation, participation, task, and block-shared
  state for every block;
- destroy retained contexts at thread exit; and
- expose the retained-context count through an internal diagnostic/test query,
  without adding a public Python API.

On AArch64, the JIT reserves `x28` for the stack-backed tile arena pointer. The
preferred rebind is for the generated `B>1` lane thunk to install the current
arena pointer from its block payload every time a worker is armed. This also
initializes a fresh Windows ARM64 fiber without requiring a custom switch. If a
safe rebind is not demonstrated on a POSIX AArch64 target, use the design's
per-block create/destroy allowance there until it is; do not disable functional
CPU blocks on that architecture. Windows ARM64 must still initialize the arena
before entering tile code.

Extend runtime tests to distinguish cold creation from warm reuse, prove the
pool does not grow for inactive tail lanes, alternate block sizes and payloads,
and verify that completion/generation state does not leak across reuse. Do not
add performance-ratio assertions.

Validation:

```powershell
uv run build_lib.py
uv run warp/tests/test_cpu_fiber.py
uv run warp/tests/test_cpu_block_runtime.py
```

### Commit 5: Generate fiber-aware CPU blocks

**Subject**: `Generate fiber-aware CPU blocks (GH-1638)`

Update `warp/_src/codegen.py`, `warp/native/tile.h`, and
`warp/native/builtin.h` to produce two compile-time CPU paths.

For effective `WP_TILE_BLOCK_DIM == 1`, preserve the current forward and backward
entry-point bodies verbatim except for mechanically necessary surrounding
structure. The compiled specialization must contain:

- the same direct `for (task_index = 0; task_index < dim->size; ...)` loop;
- literal-zero CPU lane access;
- compile-time no-op `WP_TILE_SYNC()`; and
- no lane thunk or reference to `wp_cpu_run_block`, TLS lane access, the pool,
  active masks, or barrier code.

For `WP_TILE_BLOCK_DIM > 1`:

- emit forward and backward lane thunks;
- loop over blocks in increasing order on the calling OS thread;
- compute `block_first`, `active_count`, and `task_index` with overflow-safe
  `size_t` arithmetic;
- skip the dispatcher for an empty launch;
- pass only `active_count` lanes to the runtime; and
- initialize one shared tile arena per block and make its pointer available to
  every fresh or reused lane, including AArch64.

Make `wp.block_dim()` return the effective compile-time CPU block dimension. Add
a typed lane accessor and typed barrier wrapper so the one-lane compiler can
constant-fold them before optimization.

Add focused code-generation tests, preferably in a new registered
`warp/tests/test_cpu_block_codegen.py`. Test source structure for both paths and
execute simple forward/backward non-tile kernels through the internal
specialization mechanism. Inspect the compiled one-lane object for undefined
fiber/runtime symbols. Keep a normalized IR/disassembly comparison as a required
review artifact even if platform tool differences make it unsuitable as a unit
test.

Validation:

```powershell
uv run build_lib.py
uv run warp/tests/test_cpu_block_codegen.py
uv run warp/tests/test_cpu_block_runtime.py
```

### Commit 6: Gate CPU block launch resolution

**Subject**: `Gate CPU block dimensions by config (GH-1638)`

Add the documented `enable_cpu_blocks: bool = False` setting to `warp/config.py`
and one shared resolver in `warp/_src/context.py` used by `wp.launch()` and
`wp.launch_tiled()`.

Use this resolution table before module lookup, specialization/cache selection,
command construction, tape recording, or capture:

| Device/request | Effective result |
| --- | --- |
| CPU, omitted or non-positive | 1 |
| CPU, explicit 1 | 1 |
| CPU, 2..1024, option `False` | 1 |
| CPU, 2..1024, option `True` | requested value |
| CPU, explicit greater than 1024 | `ValueError`, for either option value |
| CUDA | Existing CUDA defaulting and validation, unchanged |

Change the internal/public `wp.launch()` default to a sentinel such as `None` so
an omitted value resolves to 1 on CPU and the existing default of 256 on CUDA.
Keep `wp.launch_tiled()`'s explicit argument requirement. Update annotations,
stubs, docstrings, and internal callers that assumed an integer default,
including autograd, FEM, optimization, sparse, and utility launch wrappers.

When the resolved CPU value is greater than 1 and
`runtime.clang_sanitizer == "address"`, raise `NotImplementedError` before
compilation. The message must identify cooperative CPU fibers and suggest
`block_dim=1` or a non-ASan build.

Store the effective value in the CPU specialization/cache identity and in
recorded commands/captured launches. Replays retain that value if the global
config later changes.

Add and register `warp/tests/test_cpu_block_dim.py`. Cover both config states,
omitted/negative/zero/1/2/1024/1025 inputs, config toggling without stale cache
reuse, CUDA invariance, simple `wp.tid()` and `wp.block_dim()` behavior, forward
and backward execution, recorded commands, capture replay, and the ASan gate.
Always restore the process-global config in `finally` or an equivalent context
manager.

Also update existing launch expectation tests such as
`warp/tests/test_template_launch_bounds.py` rather than weakening assertions.

Validation:

```powershell
uv run warp/tests/test_cpu_block_dim.py
uv run warp/tests/test_template_launch_bounds.py
uv run warp/tests/test_grad_debug.py
uv run warp/tests/test_map.py
uv run warp/tests/test_sanitize.py
```

### Commit 7: Share CPU tile state across fibers

**Subject**: `Share CPU tile state across fibers (GH-1638)`

Audit and update the core machinery in `warp/native/tile.h` and matching type
declarations in `warp/_src/builtins.py` and `warp/__init__.pyi`.

Required behavior:

- all fibers in one block observe the same shared tile arena;
- register tiles and ordinary locals remain lane-local;
- block entry resets allocator offsets for every possible lane, not only lane 0;
- a reused block cannot observe stale offsets or shared values;
- the existing 256 KiB CPU shared arena limit is checked before pointer
  arithmetic and reports a clear failure;
- register/shared/global copies, `tile_extract`, `tile_from_thread`, `untile`,
  view/reshape/squeeze, assignment, masked/scatter operations, atomics, and
  component writes have correct visibility barriers; and
- every `if constexpr (WP_TILE_BLOCK_DIM == 1)` branch retains the old simple
  implementation, so correctness scaffolding for fibers does not enter the
  one-lane object.

Prefer small RAII scratch holders for temporary block-shared values. Their
lifetime must be bracketed by enough barriers that no lane frees/reuses storage
while another still reads it. Do not introduce a general cooperative-operation
gating framework.

Enable representative existing tests explicitly with
`enable_cpu_blocks=True`, including `test_tile.py`, load/store variants,
`test_tile_shared_memory.py`, `test_tile_struct.py`, `test_tile_view.py`,
`test_tile_fused_ops.py`, and out-of-bounds tests. Add the original
block-dimension mismatch regression as a dedicated test if it is not already
represented.

Validation:

```powershell
uv run build_lib.py
uv run warp/tests/tile/test_tile.py
uv run warp/tests/tile/test_tile_shared_memory.py
uv run warp/tests/tile/test_tile_struct.py
uv run warp/tests/tile/test_tile_view.py
```

### Commit 8: Port reductions and scans

**Subject**: `Port CPU tile collectives to fibers (GH-1638)`

Update `warp/native/tile_reduce.h` and `warp/native/tile_scan.h` for correct full
and partial CPU blocks in forward and backward execution.

Use the simplest correct fiber implementation: per-lane contribution/scratch
state, explicit participation, and real barriers. Correctly handle lanes without
a valid tile element and lanes that have returned. Preserve deterministic native
scheduler order without claiming bit-identical floating-point association with
CUDA.

Do not port the prototype's later two-barrier reductions, leader-only combines,
live-mask shortcuts, or barrier-elision experiments. Retain exact one-lane
specializations where the old implementation was simpler.

Tests must cover sum/min/max/custom reduction, inclusive/exclusive scan, min/max
scan, partial tiles, block dimensions on both sides of 64-bit word boundaries,
dynamic loops, adjoints, and CPU/CUDA comparison when CUDA is available.

Validation:

```powershell
uv run build_lib.py
uv run warp/tests/tile/test_tile_reduce.py
uv run warp/tests/tile/test_tile.py
```

Run the repository's scan-specific tile test too if it is separate at execution
time; locate it with `rg --files warp/tests/tile | rg "scan"` rather than assuming
a filename.

### Commit 9: Port tile linear algebra and FFT

**Subject**: `Port CPU tile math to fibers (GH-1638)`

Update `warp/native/tile_matmul.h`, `warp/native/tile_cholesky.h`,
`warp/native/tile_solve.h`, and `warp/native/tile_fft.h`.

Add only synchronization and shared-scratch changes required for correctness.
In particular, preserve ordering between adjoint matrix multiplication reads and
beta scaling, share lower-solve scratch across fibers, and prevent FFT temporary
state from becoming an accidental per-fiber private copy. A local leader section
is acceptable only for an intrinsically sequential algorithm or to avoid
duplicated side effects, and must publish its result with a barrier.

Exercise forward/backward matmul, Cholesky, lower/upper/general solves, FFT/IFFT,
partial tiles, repeated launches, and stack-backed tile storage. Run CPU/CUDA
equivalence where the same primitive is supported on both devices, plus the
no-mathdx variants so CPU correctness does not depend on GPU libraries.

Validation:

```powershell
uv run build_lib.py
uv run warp/tests/tile/test_tile_matmul.py
uv run warp/tests/tile/test_tile_matmul_no_mathdx.py
uv run warp/tests/tile/test_tile_cholesky.py
uv run warp/tests/tile/test_tile_cholesky_no_mathdx.py
uv run warp/tests/tile/test_tile_solve.py
uv run warp/tests/tile/test_tile_solve_no_mathdx.py
uv run warp/tests/tile/test_tile_fft.py
uv run warp/tests/tile/test_tile_fft_no_mathdx.py
```

### Commit 10: Port stacks, sorting, and queries

**Subject**: `Port CPU tile stacks and queries (GH-1638)`

Finish the primitive audit in `warp/native/tile.h`,
`warp/native/tile_radix_sort.h`, `warp/native/tile_mesh.h`, and
`warp/native/tile_bvh.h`.

- Make tile stack initialization, push, pop, clear, count, capacity clamping,
  and adjoints safe with multiple fibers and partial participation.
- Give radix sort block-shared temporary storage with an explicit lifetime; heap
  scratch is acceptable in this foundation when it avoids unsafe fiber stacks.
- Ensure mesh/BVH tiled queries use the logical global lane/register mapping and
  do not emit duplicate externally visible results merely because each fiber
  enters the operation.
- Test sparse/non-contiguous survivors and repeated pool reuse.
- Keep one-lane paths unchanged and do not add general leader gating.

Validation:

```powershell
uv run build_lib.py
uv run warp/tests/tile/test_tile_stack.py
uv run warp/tests/tile/test_tile_sort.py
uv run warp/tests/tile/test_tile.py
```

Locate and run the current geometry tests that exercise tiled mesh and BVH query
builtins. Add focused regressions if existing files do not cover query iteration
with `block_dim > 1`.

### Commit 11: Complete equivalence and CI coverage

**Subject**: `Complete CPU block equivalence tests (GH-1638)`

Close test gaps across the full tile suite and ensure the new registered modules
run in the ordinary CI matrix. Existing tests should continue to use the default
`enable_cpu_blocks=False` unless they deliberately exercise block semantics.
Focused tests scope the option to `True` and restore it afterward.

The enabled suite must cover:

- dimensions `1, 2, 8, 31, 32, 63, 64, 65, 255, 256, 257, 511, 512, 1023,
  1024` as appropriate for test cost;
- full blocks, one-lane tails, arbitrary partial prefixes, empty launches,
  lane 0 returning first, and sparse early returns;
- one- through four-dimensional `wp.tid()` mapping and `wp.block_dim()`;
- forward/backward kernels and nested Warp functions;
- every public tile primitive category;
- multiple sequential launches and warm pool reuse; and
- exact integer/index results plus operation-appropriate floating/adjoint
  tolerances against CUDA.

Use fixed CPU-only `unittest.TestCase` tests for fiber ABI and scheduler behavior.
CUDA-equivalence tests may skip when CUDA is unavailable, but at least one Linux
GPU job and one Windows GPU job must execute them. CPU functional tests must run
on Linux x86-64, Linux AArch64, macOS arm64, Windows x86-64, and Windows ARM64.

The existing GitHub and GitLab full-suite jobs should pick the tests up through
`default_suite`. Change CI configuration only if inspection proves a required
platform does not run that suite. If a lightweight job is added in one CI system
and an equivalent lane exists in the other, keep them in sync and pin external
GitHub Actions to commit hashes.

Final zero-overhead gate for effective `block_dim=1`:

1. Repeat the baseline kernels for omitted block dimensions, explicit 1, and a
   valid larger request forced to 1 by the disabled config.
2. Compare generated direct-loop structure, optimized IR or normalized
   disassembly, and undefined symbols with the preflight artifacts.
3. Reject the series if a one-lane object references the block runtime, fiber
   pool, TLS lane accessor, active mask, or runtime barrier.
4. Run non-asserting benchmarks only as a secondary smoke test.

Validation:

```powershell
uv run --extra dev -m warp.tests -s autodetect -k TestCpuFiber -k TestCpuBlockRuntime -k TestCpuBlockDim
uv run --extra dev -m warp.tests -s autodetect -k TestTile
```

Use the actual class names discovered with `rg "^class Test"` if they differ.

### Commit 12: Document the opt-in feature

**Subject**: `Document experimental CPU blocks (GH-1638)`

Update user-facing documentation only after the behavior and limitations are
settled:

- add `enable_cpu_blocks` to `docs/api_reference/warp_config.rst`;
- document CPU/CUDA `block_dim` defaulting and the 1024 CPU limit in the launch
  and configuration material;
- explain that enabled CPU blocks use cooperative fibers on one host thread,
  are intended initially for correctness/testing, and are not SIMD or
  multi-core acceleration;
- document the 1 MiB-per-retained-worker virtual stack reservation, the bounded
  pool, the 256 KiB shared tile arena, and ASan rejection; and
- add `changelog/1638.added.md` describing the opt-in user-visible capability.
  Do not edit `CHANGELOG.md` directly.

Reconcile the design document with the final implementation, changing its status
only according to the repository's review convention.

Validation:

```powershell
uvx pre-commit run --files docs/api_reference/warp_config.rst docs/user_guide/configuration.rst changelog/1638.added.md design/cpu-block-dim-fibers.md
uv run --extra docs build_docs.py 2>&1 | tee /tmp/build_docs.log
uvx --from towncrier==25.8.0 towncrier build --draft --version X.Y.Z --date YYYY-MM-DD
```

Replace `X.Y.Z` and the date only for the local draft preview; do not commit a
rendered `CHANGELOG.md`.

## Final acceptance checklist

Before handing off the branch:

- the worktree contains twelve signed-off commits in the order above, or fewer
  only where adjacent commits had to be combined to keep the tree buildable;
- each commit subject uses `(GH-1638)` and each body explains why the change is
  needed;
- native debug and release builds succeed on both CPU architectures represented
  by current CI;
- new modules are in `default_suite` and no test uses timing assertions or cache
  clearing;
- default-disabled launches retain legacy CPU results and the exact one-lane
  generated path;
- enabled launches honor every valid block dimension through 1024 and never
  create inactive tail fibers;
- 1025 and larger values fail even while the feature is disabled;
- ASan behavior is an explicit early `NotImplementedError`;
- the scalar scheduler model and native bitset scheduler agree;
- CPU/CUDA equivalence covers forward, backward, partial, early-return, and
  repeated-pool cases;
- all five supported platform combinations execute fiber tests in CI; and
- `git diff --check`, pre-commit, focused tests, the broader tile suite, docs,
  and the changelog preview pass.

The merge request should state plainly that SIMD, general once-per-block
cooperative-operation lowering, ASan fiber hooks, and cross-block CPU parallelism
remain follow-up work under GH-1638.
