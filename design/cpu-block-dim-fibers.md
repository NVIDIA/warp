# Opt-in CPU Block Execution with Cooperative Fibers

**Status**: Implemented (experimental)

**Issue**: [GH-1638](https://github.com/NVIDIA/warp/issues/1638)

## Motivation

Warp exposes a CUDA-style block execution model through ``wp.launch()``,
``wp.launch_tiled()``, ``block_dim``, tile operations, and block-wide
synchronization. CUDA launches execute multiple logical kernel threads in each
block. Those threads have distinct lane indices, share block-scoped tile state,
and rendezvous at barriers.

Historically, the CPU backend did not implement that model. It forced every
launch to ``block_dim=1``, even when the caller requested a larger value.
Consequently, CPU kernels could observe different thread indices and tile layouts
than the same kernel on CUDA. Code that was correct for the requested block shape
could compute an incorrect result or access tile memory out of bounds after the
CPU silently changed that shape.

This design implements opt-in CPU ``block_dim > 1`` execution using
cooperative, stackful fibers. Each logical kernel thread in a CPU block runs as a
fiber on one host OS thread. A tile barrier suspends the current fiber and resumes
a peer until all still-participating lanes have arrived. This preserves ordinary
lane-local control flow without requiring the compiler to split kernels at every
barrier.

The initial purpose is correctness and validation. It provides a portable CPU
oracle for CUDA block semantics and exercises the fiber implementation on Warp's
CI platforms. It also establishes runtime boundaries that later optimizations can
replace or build upon. It is not intended to make the initial ``block_dim > 1``
path competitive with optimized CUDA or a future SIMD CPU backend.

The feature is disabled by default. Existing applications therefore retain the
legacy CPU behavior unless they deliberately enable CPU blocks.

## Terminology

- A **kernel thread** is one logical execution of the kernel body. CUDA commonly
  calls this a thread; it is not necessarily an OS thread on CPU.
- A **CPU thread** is an operating-system thread executing host instructions.
- A **thread block** is a group of kernel threads identified by one block index
  and lane indices from zero through ``block_dim - 1``.
- A **participating lane** is a kernel thread that exists in the logical launch
  and has not returned from the kernel body.
- A **fiber** is a user-space execution context with its own registers and stack.
  It runs until it yields at a barrier or returns; the operating system does not
  schedule fibers independently.
- A **barrier generation** is one logical block-wide rendezvous. Participating
  lanes must encounter compatible barriers in the same logical order.

## Requirements

| ID | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| R1 | When CPU blocks are enabled, an explicit ``block_dim`` from 2 through 1024 is honored by ``wp.launch()`` and ``wp.launch_tiled()``. | Must | The effective value controls lane identity, ``wp.tid()``, ``wp.block_dim()``, tile layout, shared storage, and barriers. |
| R2 | ``wp.config.enable_cpu_blocks`` gates the feature and defaults to ``False``. | Must | ``False`` preserves the legacy forced-1 behavior. |
| R3 | An explicit CPU ``block_dim > 1024`` raises an error before the config override is applied. | Must | Legacy mode must not hide an invalid request. |
| R4 | Omitted or non-positive ``block_dim`` values continue to resolve to 1 on CPU. | Must | The CUDA default remains unchanged. |
| R5 | The effective ``block_dim=1`` CPU specialization has no fiber, scheduler, TLS-lane, barrier-call, thunk, or additional dispatch overhead. | Must | Generated control flow and optimized machine code must remain equivalent to the legacy CPU path. |
| R6 | For data-race-free kernels with valid barrier participation, CPU and CUDA have equivalent observable block semantics. | Must | Exact integer/indexing results; documented tolerances for floating-point results and adjoints. |
| R7 | Full blocks, partial final blocks, forward and backward kernels, and lanes that return early are supported. | Must | Finished lanes permanently leave later barrier generations. |
| R8 | Every supported CPU platform provides functional ``block_dim > 1`` execution. | Must | No architecture-specific silent fallback to 1 when the feature is enabled. |
| R9 | The scheduler is deterministic and has bounded lane-selection work through ``block_dim=1024``. | Must | Use a direct generation-safe bitset, not a scalar scan on the release path. |
| R10 | Fiber creation is amortized where the platform state can be safely rebound, and inactive tail lanes do not receive fibers. | Should | Pooling is an implementation optimization; it does not change semantics. |
| R11 | AddressSanitizer builds reject enabled ``block_dim > 1`` execution with a clear error. | Must | ASan fiber-switch integration is deferred. |
| R12 | The design remains compatible with later cooperative-operation, SIMD, and cross-block parallelism work. | Should | None of those optimizations is required by this proposal. |

**Non-goals:**

- SIMD/SPMD vectorization, implicit LLVM vectorization changes, lane gangs, or
  kernel fission.
- Multi-core execution across blocks or within a block.
- General code generation that executes every cooperative tile operation once
  per block.
- Barrier coalescing, movement, or elimination.
- A custom Windows context-switch implementation when the native Windows fiber
  API is sufficient for correctness.
- AddressSanitizer fiber integration.
- Enabling CPU block execution by default. Any future default change requires
  separate compatibility and performance review.
- Defining behavior for kernels whose participating lanes encounter incompatible
  divergent barrier sequences. Such kernels do not have portable CUDA block
  semantics either.

## Design

### Decision summary

CPU block execution has two compile-time-specialized paths:

```text
effective block_dim == 1
    -> emit the existing direct per-task CPU loop

effective block_dim in [2, 1024]
    -> group logical tasks into blocks
    -> execute the active lanes as fibers on one CPU thread
    -> synchronize fibers with a generation-safe bitset scheduler
```

The configuration decision is made before selecting or compiling a kernel
specialization:

```text
requested CPU block_dim
    |
    +-- greater than 1024 --------------------------> ValueError
    |
    +-- omitted or non-positive --------------------> effective 1
    |
    +-- enable_cpu_blocks is False -----------------> effective 1
    |
    `-- enable_cpu_blocks is True ------------------> requested value
```

CUDA launch resolution is unaffected.

### Configuration and launch resolution

The feature is controlled by the following setting in ``warp/config.py``:

```python
enable_cpu_blocks: bool = False
"""Honor explicit ``block_dim > 1`` values on CPU launches.

When ``False``, CPU launches retain the legacy behavior and force the effective
block dimension to ``1``. When ``True``, explicit values from ``2`` through
``1024`` execute with cooperative fibers. GPU launches are unaffected.
"""
```

``wp.launch()`` must distinguish an omitted value from its CUDA default. Its
internal default becomes a sentinel such as ``None``: omission resolves to 1 on
CPU and 256 on CUDA. This does not change either device's existing default.
``wp.launch_tiled()`` continues to require ``block_dim`` explicitly.

Both entry points use one shared resolution helper so validation and config
behavior cannot drift. An explicit value greater than 1024 raises ``ValueError``
on CPU regardless of ``enable_cpu_blocks``. The helper resolves the effective
value before module lookup, command construction, tape recording, or capture.

The effective value, rather than the config flag, is part of the CPU
specialization and cache identity. Changing the flag between ordinary launches
takes effect immediately and selects the corresponding specialization. A
recorded command or captured launch retains the effective block dimension that
was resolved when it was recorded; replay does not reinterpret it using a later
config value.

### Exact legacy path for ``block_dim=1``

The new runtime must not merely contain a quick ``if (block_dim == 1)`` branch.
Even a quick branch, an indirect thunk call, or an opaque lane getter can regress
the existing CPU path. The JIT must select the legacy template at compile time:

```cpp
extern "C" WP_API void kernel_cpu_forward(launch_bounds* dim, args* a)
{
    tile_shared_storage_t tile_mem;
    // Preserve the existing shared-storage setup exactly.
    for (size_t task_index = 0; task_index < dim->size; ++task_index)
        kernel_cpu_kernel_forward(*dim, task_index, a);
}
```

For this specialization:

- ``WP_TILE_THREAD_IDX`` or its typed replacement reduces to the literal zero;
- ``WP_TILE_SYNC()`` reduces to a compile-time no-op;
- there is no block thunk, ``wp_cpu_run_block()`` reference, TLS lane read, fiber
  pool access, active-mask access, or scheduler state;
- the direct task loop has the same indexing and bounds behavior as before this
  feature; and
- no global compiler-vectorization or optimization flag is changed as part of
  this work.

This condition applies both to explicit ``block_dim=1`` and to any launch forced
to 1 by the default-disabled config. Inspection of generated IR or disassembly is
the primary acceptance test. Timing is a regression signal, not the proof of zero
added work.

### Block dispatcher

For an enabled specialization with ``block_dim > 1``, code generation emits a
per-lane thunk and a per-block loop. For block ``b``:

```text
block_first  = b * block_dim
active_count = min(block_dim, total_tasks - block_first)
task_index   = block_first + lane
```

Only lanes in ``[0, active_count)`` borrow fiber contexts. Creating fibers for
the known-inactive suffix would add stack reservations and lifecycle work for
kernel threads that immediately fail the launch bounds check. This optimization
has no effect on full blocks and can avoid as many as 1023 unused contexts in a
small final block.

The dispatcher initializes block-shared tile state, arms the active fibers with
the same block context and distinct lane indices, transfers control to the first
fiber, and returns after all active fibers finish. Blocks execute serially in
increasing block order on the calling CPU thread in this design.

The forward and backward entry points use the same dispatcher contract. Their
argument payloads may differ, but their participation, barrier, stack, and cleanup
rules do not.

### Fiber execution contexts

Each active lane owns a stackful context containing the callee-saved state
required by the platform ABI and a private stack. The kernel body otherwise uses
the ordinary generated CPU code. Lane-local branches, function calls, loops, and
values live across a barrier naturally remain on that fiber's stack and in its
saved registers.

The low-level interface is internal and deliberately small:

```c
typedef struct wp_fiber wp_fiber_t;

wp_fiber_t* wp_fiber_create(void (*entry)(void*), void* arg,
                            size_t stack_size);
void wp_fiber_destroy(wp_fiber_t* fiber);
void wp_fiber_switch(wp_fiber_t* target);
wp_fiber_t* wp_fiber_active(void);
```

The implementation must preserve all nonvolatile integer, floating-point, and
vector state required by the ABI, the stack pointer and alignment, floating-point
control state where required, and Warp's lane/block context. Context-switch tests
must keep values live across many switches so an incomplete save set cannot pass
by accident.

Fibers share the address space and OS-thread-local storage of their host CPU
thread. The scheduler therefore maintains the currently executing lane and block
context explicitly, restoring them whenever a fiber resumes. Generated
``block_dim > 1`` code obtains its lane identity through this context. The
``block_dim=1`` specialization never reads it.

### Platform backends

The initial feature covers every CPU platform distributed and tested by Warp:

| Platform | Initial switching backend |
| --- | --- |
| Linux x86-64 | Small SysV ABI assembly context switch with guarded POSIX stacks. |
| Linux AArch64 | Small AAPCS64 assembly context switch with guarded POSIX stacks. |
| macOS arm64 | AAPCS64/Darwin assembly context switch with guarded POSIX stacks. |
| Windows x86-64 and ARM64 | ``ConvertThreadToFiberEx``, ``CreateFiberEx``, and ``SwitchToFiber``. |

The abstraction is platform-internal; JIT-generated kernels call the same block
runtime on every platform. A newly supported Warp CPU platform must implement and
pass the fiber ABI tests before it can build with CPU blocks. It must not report
support and silently force an enabled launch back to 1.

AArch64 requires special care because the CPU tile arena pointer is held in a
reserved callee-saved register. A fresh fiber must receive the block's arena
pointer before entering JIT code, and a reused fiber must rebind it for every new
block. The portable mechanism is for the generated ``block_dim > 1``
lane thunk to install the pointer from its block payload before calling the
kernel body; this works with both the POSIX context switch and the Windows fiber
API. The implemented AArch64 backends rebind this pointer on every dispatch and
therefore use the same reusable worker pool as the other supported platforms.

### Stacks and failure behavior

Each worker fiber initially receives a 1 MiB usable stack reservation. This is
large enough for the CUDA-scale lane-local storage exercised by the prototype,
including a 256 KiB register-tile frame, while leaving space for CPU call frames.
It remains a finite resource rather than an unlimited compatibility guarantee.

POSIX backends allocate page-aligned stacks and place a non-accessible guard page
below the downward-growing usable region. Windows uses ``CreateFiberEx`` with an
explicit reserve and demand commitment so the operating system enforces the
stack boundary. Stack overflow must fail loudly rather than corrupting another
fiber or the block scheduler.

A block of 1024 lanes can reserve approximately 1 GiB of virtual address space.
Only touched pages should consume physical memory, but the number and size of
reservations are still operational constraints. The pool is therefore bounded by
the maximum supported block dimension and must expose its reserved-context count
to debug diagnostics. Adaptive stack sizing and a smaller retention budget are
follow-up options; they are not allowed to weaken the guard-page behavior.

Fiber allocation failure raises a clear launch error. It must not retry the
kernel at ``block_dim=1``.

### Fiber pool

Creating a guarded stack for every lane of every block can dominate the kernel.
Where platform state can be rebound safely, use a thread-local pool of reusable
worker fibers:

- grow lazily to the largest active lane count requested on that CPU thread;
- arm each borrowed worker with a new block task only while it is suspended at a
  known scheduler point;
- retain no more than 1024 worker contexts per CPU thread;
- destroy all retained contexts at thread exit; and
- keep cold create/destroy tests separate from warm-pool execution tests.

The 1024-lane execution limit provides a hard upper bound for this first design.
Pooling changes only lifecycle cost. Scheduler state, generation counters,
finished-lane state, and block-shared storage are recreated or reset for every
block and must never leak through a reused fiber.

The prototype measured about 5 microseconds to create a fiber and found pooling
approximately 54 times faster for a creation-bound register-elementwise workload
and 1.8 times faster for a tiled matrix multiplication at ``block_dim=256``.
Reduction-heavy work improved only about 1.1 times because barrier switching, not
creation, dominated it. These figures motivate including the pool but are not
portable performance promises.

### Barrier and scheduler

``WP_TILE_SYNC()`` becomes a real block barrier for the fiber specialization. A
lane arriving at a barrier advances its generation and transfers directly to the
lowest-numbered unfinished lane that is still behind that generation. The call
returns only after every lane participating in that generation has arrived or
finished.

The release scheduler uses two generation-tagged bitsets:

- ``front`` contains unfinished lanes at the current frontier generation;
- ``behind`` contains unfinished lanes one generation behind the frontier; and
- an absolute unsigned generation identifies the logical epoch represented by
  each set.

At most sixteen 64-bit words represent 1024 lanes. Selection scans words in a
fixed order and uses a native count-trailing-zeros operation within the first
non-empty word. Thus discovery is bounded by sixteen word checks rather than a
scalar scan of every lane at every arrival. The design does not describe the word
scan as strict constant time; it is a small fixed bound established by the public
``block_dim`` limit.

Direct fiber-to-fiber handoff avoids routing every arrival through the main
dispatcher. A suspended waiter records its own generation. If the frontier has
advanced past that generation when the waiter resumes, it returns immediately
instead of examining bitset storage that may now represent a newer epoch. This
own-generation rule prevents the aliasing and deadlock observed in a parity-only
two-epoch prototype.

When a lane's kernel body returns, the runtime removes it permanently from every
outstanding set before selecting the next runnable lane. This supports partial
blocks and non-contiguous data-dependent early returns. It does not make
incompatible divergent barrier sequences valid.

Scheduling order is deterministic: word order is fixed, and the lowest set lane
is selected within a word. Data-race-free kernels therefore have repeatable
within-block execution for fixed inputs and a fixed build. This is not a promise
that floating-point results are bit-identical to CUDA, whose legal reduction
association and thread scheduling can differ.

The scalar generation scheduler remains as an independent reference model in
focused tests and instrumented builds. It is not selected by an environment
variable or runtime branch in release execution. Debug builds may retain
state-transition assertions and a bounded deadlock trace, but release barriers do
not contain statistics or diagnostic atomics.

Prototype measurements found the direct generation-safe bitset about 2.1 to 2.5
times faster than the direct scalar scan at high block dimensions, and about 2.2
times faster at ``block_dim=1024`` in the measured reduction workload. More
importantly for this design, it removes a known quadratic lookup term from the
foundation used by CI.

### Shared tile state and primitive coverage

All fibers in a block share one fixed 256 KiB tile arena. Lane-local register
tiles and ordinary locals stay on the fiber stack. CPU tile allocation must
preserve the existing lockstep allocator contract: participating lanes executing
the same logical tile allocation receive the same block-relative storage while
retaining any per-lane allocation bookkeeping required by existing layouts.

Block initialization resets allocator state for every possible lane, not only the
currently executing lane. Reusing an arena for a later block must not expose stale
allocation offsets, participation flags, or shared values from the preceding
block.

Every CPU tile primitive must be audited for assumptions that were valid only
when ``WP_TILE_BLOCK_DIM`` and ``WP_TILE_THREAD_IDX`` were 1 and 0. This includes
forward and backward forms of load/store, register/shared conversion, extraction,
reduction, scan, matrix operations, sorting, stack operations, atomics, and tile
query iteration.

The foundational implementation uses the simplest correct fiber lowering for
each primitive. In general, every participating fiber enters the primitive and
performs its lane's existing work, with real barriers providing visibility.
Operations that are intrinsically sequential or would duplicate externally
visible side effects may require a leader section and a publication barrier for
correctness. Such local correctness fixes do not establish a general
once-per-block optimization framework.

In particular, this design does not split 30-50 cooperative primitives into
allocation and body phases or add a compiler classification for their call sites.
That transformation remains a planned optimization under GH-1638. The initial
runtime and shared-storage contract are chosen so a later implementation can run
a proven block-shared body once, publish its result, and release the waiting
fibers without changing user-visible semantics.

### AddressSanitizer

Custom stacks must be announced to AddressSanitizer before switching and closed
out after resumption. That integration and its platform validation are deferred.
When Warp's CPU JIT/runtime is built with AddressSanitizer, an enabled CPU launch
whose resolved ``block_dim`` is greater than 1 raises ``NotImplementedError``
before compilation or dispatch. The message identifies cooperative fibers as the
unsupported feature and suggests ``block_dim=1`` or a non-ASan build.

Legacy and explicit ``block_dim=1`` launches remain available and retain their
unchanged direct path under ASan.

## Alternatives Considered

### Sequentially call each lane with no-op barriers

This preserves lane identity but not synchronization. A later lane can depend on
shared data written by an earlier or later peer across a barrier. Running one
lane's entire kernel before the next cannot reproduce those visibility edges.

### One OS thread per lane

OS threads provide real barriers but oversubscribe immediately at CUDA-sized
block dimensions, add materially higher creation and scheduling overhead, and
make within-block ordering dependent on the host scheduler. OS-level parallelism
is better applied across independent blocks in a later design.

### C++ coroutines or general stackless continuations

They can represent the same semantics without private stacks, but require the
compiler to transform all barrier-containing control flow and propagate coroutine
state through called Warp functions and tile primitives. That is a much broader
JIT and code-generation change than the self-contained fiber runtime.

### Scalar kernel fission

Splitting the kernel at every barrier and looping over lanes can be much faster
even without SIMD. It still requires barrier convergence analysis, cross-barrier
liveness, continuation storage, and a fallback for control flow the compiler
cannot prove safe. It would also reduce the fiber coverage that this project is
intended to validate. It is therefore deferred with SIMD/gang lowering.

### Scalar-scan scheduler

A lowest-lane scalar scan is simple and remains useful as a correctness oracle.
It performs quadratic lane inspection across a barrier at high ``block_dim``.
The direct generation-safe bitset has already demonstrated the same deterministic
selection order with bounded word lookup through 1024 lanes, so knowingly using
the scalar scan in the release foundation is not justified.

### Enable the feature by default

Correct CPU/CUDA equivalence is desirable, but the unoptimized fiber path can be
substantially slower and reserve significant virtual address space. Default-on
would impose those costs on existing launches that explicitly pass a GPU-oriented
``block_dim`` but have historically run at 1 on CPU. The opt-in flag lets CI and
interested users validate semantics before later performance work and migration
evidence justify revisiting the default.

## Testing Strategy

### Legacy-path invariance

Tests must establish absence of new work, not merely an acceptable benchmark
ratio:

1. Compile representative scalar, array, shared-tile, register-tile, reduction,
   and matrix kernels at effective ``block_dim=1`` before and after the change.
2. Confirm the generated entry point retains the direct per-task loop and has no
   block thunk or reference to fiber, scheduler, TLS-lane, active-mask, or runtime
   barrier symbols.
3. Confirm lane-index expressions fold to zero and CPU barriers disappear.
4. Compare optimized IR or normalized disassembly for the hot kernel bodies.
5. Benchmark the same workloads as a secondary regression check. Unit tests must
   not assert timing ratios.

Run these checks with the config disabled, explicit ``block_dim=1``, omitted
``block_dim``, forward execution, and backward execution.

### Fiber primitive and ABI

Exercise the context-switch layer independently in subprocesses so a stack or ABI
failure does not terminate the test runner. Cover:

- main-fiber identity and repeated create/switch/destroy cycles;
- two-fiber ping-pong and 1024-fiber round-robin stress;
- integer and floating/vector values live in every nonvolatile register class;
- stack alignment and deep nested calls;
- guard-page overflow in a dedicated expected-crash test where the platform
  harness supports it;
- fiber reuse with new entry arguments and block contexts; and
- platform-specific shared-tile arena inheritance, especially AArch64.

The functional suite runs on Linux x86-64, Linux AArch64, macOS arm64, Windows
x86-64, and Windows ARM64 CI. A platform is not considered supported based only
on a successful build.

### Scheduler model and participation

Maintain a scheduler reference model independent of the native switching code.
Exhaustively enumerate small blocks and use randomized traces through 1024 lanes.
For each state transition, verify that:

- the chosen lane is unfinished and behind the requesting generation;
- no waiter consumes state tagged for a newer generation;
- completion removes the lane from every bitset;
- the scheduler does not report a barrier complete while a participant is behind;
- execution either progresses or produces a bounded diagnostic trace; and
- repeated inputs produce the same schedule.

The native tests cover full blocks, one-lane tails, partial prefixes, alternating
early exits, arbitrary sparse survivors, lane 0 finishing first, one survivor,
returns before the first barrier, and thousands of generations. Exercise block
dimensions around bitset and common execution boundaries, including 1, 2, 8, 31,
32, 63, 64, 65, 255, 256, 257, 511, 512, 1023, and 1024.

### CPU/CUDA semantic equivalence

For kernels with valid CUDA barrier participation, run matching CPU and CUDA
launches at the same explicit ``block_dim``. Check:

- ``wp.block_dim()``, lane identity, and one- through four-dimensional
  ``wp.tid()`` mapping;
- full and partial blocks;
- shared allocation identity and lifetime;
- forward and backward tile loads/stores, register/shared conversions,
  extraction, reductions, scans, matrix operations, sorting, stacks, atomics,
  and queries;
- nested Warp functions containing cooperative operations;
- the block-dimension mismatch pattern that originally produced invalid tile
  indexing; and
- repeated execution and fiber-pool reuse.

Use exact comparisons for integer and indexing behavior. Use each operation's
documented tolerance for floating-point values and adjoints; CPU and CUDA need not
choose an identical legal reduction association.

All new test modules belong in ``default_suite``. Fixed CPU-only tests use
ordinary ``unittest.TestCase`` methods. Timing assertions are prohibited.

### Configuration, validation, and recording

Focused tests cover:

- the default value ``False``;
- forced-1 legacy behavior for valid explicit values through 1024;
- enabled behavior for values 2 through 1024;
- ``ValueError`` above 1024 even when the config is disabled;
- unchanged CUDA behavior for both flag values;
- toggling the flag between launches without stale cache reuse;
- recorded command and capture replay retaining their resolved block dimension;
- no worker creation for known-inactive tail lanes; and
- ``NotImplementedError`` for enabled ``block_dim > 1`` in an ASan build, while
  ``block_dim=1`` continues to work.

## Adoption Plan

1. **Land the opt-in surface and exact direct specialization.** Add config and
   shared resolution logic, preserve the legacy generated template, specialize
   lane and barrier access at ``block_dim=1``, and add the greater-than-1024 and
   ASan errors. The default remains disabled.
2. **Land and validate all platform fiber backends.** Add guarded stacks, ABI
   tests, block/lane context, and low-level switch tests on every supported CPU
   platform before exposing enabled execution there.
3. **Add the block runtime.** Create only the known active lane prefix, use the
   direct generation-safe bitset with its scalar reference model, and pool workers
   where the platform context can be safely rebound.
4. **Complete primitive correctness.** Audit every existing CPU tile primitive,
   add CPU/CUDA equivalence tests, and cover forward, backward, partial, and sparse
   participation. An unported primitive is a development assertion, not a shipped
   fallback to ``block_dim=1``.
5. **Exercise the opt-in path in CI.** Targeted suites set
   ``enable_cpu_blocks=True`` explicitly. The ordinary suite also runs with the
   default disabled to continuously protect legacy behavior and performance.
6. **Optimize incrementally under separate evidence.** Start with measured
   barrier-count or cooperative-body pilots. Do not broaden the initial design or
   change the default as part of those optimizations.

## Deferred Optimizations

The following work is compatible with the runtime but deliberately excluded from
the foundation:

| Optimization | Prototype evidence | Reason deferred |
| --- | --- | --- |
| Reduction barrier count reduction | About 1.4-1.9 times faster in reduction-heavy tests. | Primitive-specific participation and scratch-lifetime reasoning. |
| Leader-only whole-tile collectives | About 1.35 times at block dimension 128, 2.2 times at 256, and 3.3 times at 1024 in one reduction benchmark. | Requires live-lane masks, publication barriers, and a platform-measured crossover. |
| General allocation/body split for cooperative operations | No completed end-to-end pilot; potentially removes duplicate bodies and internal barriers. | Touches roughly 30-50 primitives and requires new codegen classification. |
| Barrier coalescing or elimination | Each valid removal avoids a full active-lane handoff chain. | Requires cross-lane dependence and alias analysis. |
| Custom Windows x64 switch | About 2.7-3.5 times faster for the raw switch primitive. | Expected end-to-end gain is much smaller; ABI, TEB, unwind, and build complexity are high. |
| Smaller or adaptive fiber stacks | No measured barrier-speed benefit; potentially large virtual-memory savings. | Must retain CUDA-scale lane-local capacity and reliable guard behavior. |
| Scalar fission and SIMD gangs | Scalar fission prototypes showed large gains before SIMD. | Requires compiler analysis and continuations and would bypass the fibers this phase is intended to validate. |
| Cross-block CPU threading | Independent blocks could use multiple cores. | Changes scheduling, resource ownership, and cross-block determinism; orthogonal to within-block correctness. |
| AddressSanitizer fiber hooks | Enables sanitizer coverage of the opt-in path. | Requires correct custom-stack handoff integration and platform validation. |

The fiber pool, active-tail creation, and direct generation-safe bitset are not in
this table because they are part of the selected foundation. They are localized
runtime mechanisms with broad benefit and prevent allocation churn or quadratic
scheduler lookup from obscuring the behavior CI is intended to validate.
