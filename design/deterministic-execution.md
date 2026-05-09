# Deterministic Execution Mode

**Status**: Implemented

## Motivation

GPU atomic operations on floating-point arrays are inherently non-deterministic:
threads execute in unpredictable order, and since float addition is
non-associative, different execution orderings produce different rounding,
yielding different results each run. This also applies to counter/slot-allocation
patterns (``slot = wp.atomic_add(counter, 0, 1)``) where the thread-to-slot
assignment varies across runs, causing downstream writes to differ.

Customers need bit-exact reproducibility for debugging, regression testing, and
certification workflows. The manual workaround --- rewriting algorithms to use
two-pass count-scan-write patterns or sorted reductions --- is painful and
error-prone. Users want a simple toggle that makes their existing code
deterministic without algorithm rewrites.

## Requirements

| ID  | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| R1  | ``wp.config.deterministic = "run_to_run"`` makes float atomic accumulations bit-exact reproducible across runs | Must | Core value proposition |
| R2  | Counter/allocator pattern (``slot = wp.atomic_add(counter, 0, 1)``) produces deterministic slot assignments | Must | Common in compaction, particle emission |
| R3  | Both patterns work in the same kernel simultaneously | Must | Real workloads mix accumulation and allocation |
| R4  | Integer atomics with unused return values incur no overhead | Must | Already associative+commutative |
| R5  | CPU execution unaffected (already sequential/deterministic) | Must | Zero overhead on CPU |
| R6  | Per-module and per-kernel granularity via ``module_options`` / unique kernels | Should | Allows selective opt-in |
| R7  | Backward pass (autodiff) gradient accumulation is also deterministic | Should | Adjoint atomics are Pattern A |
| R8  | Multiple target arrays in one kernel each get independent buffers | Must | Real kernels write to N arrays |

**Non-goals**:
- ``atomic_cas``/``atomic_exch`` determinism (inherently order-dependent).
- Tile-level atomic operations (``tile_atomic_add``).
- Kernels where counter contributions depend on scratch array writes within the
  same kernel (Phase 0 suppresses all side effects; documented limitation).

## User Configuration

Deterministic mode can be enabled at three scopes:

- **Global**: set ``wp.config.deterministic`` to one of:
  - ``"not_guaranteed"``: default behavior, no deterministic transform.
  - ``"run_to_run"``: bit-exact reproducibility across repeated runs on the
    same GPU architecture.
  - ``"gpu_to_gpu"``: stronger reduction path intended to preserve identical
    results across GPU architectures as well.
- **Global diagnostics**: set ``wp.config.deterministic_debug = True`` to emit
  debug diagnostics for deterministic scatter overflow.
- **Per shared module**: call ``wp.set_module_options({"deterministic": "run_to_run"})``
  in the Python module that defines the kernels, just like other module-level
  options such as ``enable_backward``.
- **Per kernel**: use ``@wp.kernel(deterministic="run_to_run")`` and optionally set a
  per-target, per-thread scatter record limit with
  ``deterministic_max_records=...``. In practice determinism is a compilation
  mode, so the cleanest way to opt in for only one kernel is to place it in a
  unique module (for example with ``module="unique"``).

For backward compatibility, ``True`` and ``False`` are still accepted at the
module and kernel level as aliases for ``"run_to_run"`` and
``"not_guaranteed"``, respectively.

Like ``enable_backward``, the setting participates in module compilation and
hashing. A kernel defined in a shared module inherits that module's
``deterministic`` option; a unique-module kernel can override it independently.
This matters because deterministic lowering affects both kernels and any
reachable ``@wp.func`` call graph compiled into the same module.

## Supported Operators

Deterministic mode currently handles these atomic builtins:

- ``atomic_add``
- ``atomic_sub``
- ``atomic_min``
- ``atomic_max``

Handling depends on how the atomic is used:

- **Pattern A: accumulation, return value unused**
  Examples: ``wp.atomic_add(arr, i, value)``, ``arr[i] += value``.
  Floating-point ``add/sub/min/max`` on scalar and composite leaf types are
  redirected through scatter-sort-reduce. This includes vectors, matrices,
  quaternions, and transforms with floating-point components.
  Integer ``add/sub/min/max`` with unused return values are left on the normal
  atomic path because the final value is already deterministic.
- **Pattern B: counter / allocator, return value consumed**
  Example: ``slot = wp.atomic_add(counter, 0, 1)``.
  This is handled with the two-pass count-scan-execute path.

Operators that are not supported by deterministic mode:

- ``atomic_cas``
- ``atomic_exch``
- Tile atomics such as ``tile_atomic_add``

Bitwise integer atomics (``atomic_and``, ``atomic_or``, ``atomic_xor``) are not
transformed because their final results are already deterministic for the
unused-return case.

## Design

### Approach

Two distinct atomic usage patterns require two strategies, both transparent to
the user:

**Pattern A --- Scatter-Sort-Reduce** (accumulation, return value unused):

Instead of performing ``atomic_add`` in-place during kernel execution, each
thread writes a ``(sort_key, value)`` record to a temporary scatter buffer. The
sort key packs ``(dest_index << 32 | thread_id)``. After the kernel completes,
Warp reads the emitted record count outside CUDA graph capture and a CUB radix
sort orders the valid scatter record prefix by key and record index. The
post-sort reduction depends on the selected determinism guarantee:

- ``"run_to_run"``: scalar reductions use CUB ``DeviceReduce::ReduceByKey`` on
  destination indices for better performance.
- ``"gpu_to_gpu"``: reductions use the simple sequential segmented kernel so
  the left-to-right accumulation order is explicit in Warp's own code.
- Composite leaf types always use the segmented kernel.

During CUDA graph capture, host-side scatter count readbacks are not allowed.
Captured launches therefore still initialize unused buffer slots with the
invalid sentinel key ``-1`` and sort the full fixed-capacity buffer. The
emitted record count is only known on the device after the scatter kernel runs,
while CUB sort/reduce calls take the item count as a host-side launch
parameter when the graph is captured. Using the exact count during replay would
therefore require a host readback, graph rebuild, or a custom device-driven
sort/reduce path. CUB sorts the signed sentinel key before valid non-negative
destination keys; the reducer paths ignore records with invalid destinations,
so sentinel records do not affect outputs.

Because the sort key reserves 32 bits for the linear thread index,
deterministic scatter launches are limited to at most ``2**32`` threads.

**Pattern B --- Two-Pass Execution** (counter/allocator, return value used):

The kernel runs twice:
1. *Phase 0 (counting)*: The kernel executes with all side effects suppressed.
   Counter atomics record per-thread contributions to a scratch array instead
   of performing the actual atomic.
2. *Prefix sum*: ``wp.utils.array_scan(contrib, prefix, inclusive=False)``
   computes deterministic per-thread offsets. The total count is
   ``prefix[-1] + contrib[-1]``.
3. *Phase 1 (execution)*: The kernel re-executes. Counter atomics return the
   deterministic offset from the prefix sum. All other operations (including
   Pattern A scatters) execute normally.

This mirrors the well-established count-scan-write pattern already used by
``warp/_src/marching_cubes.py`` and the FEM geometry code, but applied
automatically.

### Alternatives Considered

| Alternative | Why rejected |
| --- | --- |
| Fixed-point integer accumulation | Loses precision/range; not general for all float types |
| Per-thread output arrays | ``O(threads * output_size)`` memory; doesn't scale |
| Serialized atomics (mutex) | Prohibitively slow on GPU |
| Kahan compensated summation | Reduces error but does not guarantee determinism |
| Taint-based selective side-effect suppression for Phase 0 | Significant codegen complexity for an edge case; deferred to a future version |

### Key Implementation Details

**Atomic classification** happens in ``Adjoint._emit_deterministic_atomic()``
during the codegen build phase. The ``_det_in_assign`` flag on the ``Adjoint``
(set by ``emit_Assign``, cleared after) distinguishes Pattern B (return consumed
in an assignment like ``slot = wp.atomic_add(...)``) from Pattern A (return
discarded, as in ``arr[i] += val`` or bare ``wp.atomic_add(...)``). Only
``atomic_add``, ``atomic_sub``, ``atomic_min``, and ``atomic_max`` are
intercepted. Integer atomics with unused return values are skipped entirely
(already deterministic).

**CPU/CUDA dual compilation** is handled inside deterministic atomic lowering.
On CUDA, generated code uses the deterministic scatter or two-pass path. On
CPU, generated code falls back to the normal Warp atomics so the CPU path
remains unchanged and sequentially deterministic.

**Hidden deterministic ABI** now has two layers:

- **Raw kernel-entry parameters**: CUDA kernels still receive raw launch-time
  pointers and scalars appended after the user arguments. These include shared
  execution state (phase, debug flag, one overflow flag) plus the raw storage
  for each deterministic target.
- **Structured helper objects**: immediately inside the generated kernel body,
  Warp constructs small helper objects with readable names:
  - ``det_ctx`` for shared execution state
  - ``det_<array>`` for each scatter target as ``wp::det_scatter_buf<T>``
  - ``det_<array>`` for each counter target as ``wp::det_counter_buf``

This keeps kernel launches compatible with ctypes/raw parameter packing while
making generated function calls much easier to read and propagate.

**Deterministic ``@wp.func`` support** is implemented by threading the helper
objects through the generated function signature, similar in spirit to how Warp
threads hidden ``adj_*`` parameters for reverse-mode codegen. Only functions
that directly or transitively need deterministic lowering receive the extra
hidden arguments. A function containing no deterministic atomics keeps its
normal signature.

When a deterministic ``@wp.func`` is called, helper arguments are remapped from
the callee's formal parameter names to the caller's actual array arguments. For
example, a callee written as ``foo(arr, ...)`` may receive ``det_output`` or
``det_force`` at a particular call site depending on which array is bound to
``arr``. This avoids relying on anonymous ``_0`` / ``_1`` numbering and is the
mechanism that makes nested deterministic ``@wp.func`` calls work.

**Multiple reduction buffers**: within one function or kernel, each distinct
target array gets at most one normalized deterministic family:
``add`` (shared by ``atomic_add`` and ``atomic_sub``), ``min``, ``max``, or
``counter``. Multiple call sites with the same target array and normalized
family share one helper object and one postpass buffer set. The
``DeterministicMeta`` dataclass on each ``Adjoint`` tracks the scatter and
counter targets required by that function or kernel, and callers include the
transitive requirements of any deterministic callees.

**Scatter sizing**: each scatter target uses a fixed-capacity buffer sized from
a code-generated lower bound (static records-per-thread analysis). The optional
``deterministic_max_records`` setting overrides the per-thread record count when
users know a thread may revisit the same atomic site multiple times, for
example inside a dynamic loop. Warp uses
``max(codegen_lower_bound, deterministic_max_records)`` records per thread for
each target. On overflow, new records are truncated, one shared device-side
overflow flag is set, and optional diagnostics may be emitted when
``wp.config.deterministic_debug`` is enabled. Outside CUDA graph capture, Warp
reads each target's emitted-record counter before the postpass so sort/reduce
work is proportional to actual records rather than static capacity, and checks
the overflow flag after the deterministic launch. During CUDA graph capture
and replay, Warp deliberately skips host-side count readbacks and overflow
checks because reading device flags would synchronize every graph launch and
defeat the asynchronous graph replay path.

**CUDA graph capture support**: deterministic launch orchestration allocates
scatter buffers, counter buffers, and CUB sort/reduce workspaces from Python.
During ordinary CUDA graph capture, Warp retains those arrays on the captured
graph object so replay can safely reuse the device memory after the original
launch call returns. The native sort/reduce entry point receives an explicit
workspace pointer and size; it does not allocate temporary CUB scratch storage
inside the captured native calls.

**Counter total writeback**: after the exclusive prefix sum in Phase 0, the
launch system copies the total count back to the actual counter array so user
code that reads it post-launch sees the correct value. The total is
``prefix[-1] + contrib[-1]``; the implementation may materialize that value via
an inclusive scan scratch buffer to keep the writeback capture-friendly.

**Files added/modified**:

| File | Role |
| --- | --- |
| ``warp/config.py`` | ``deterministic`` global flag |
| ``warp/_src/deterministic.py`` | Dataclasses, deterministic buffer/workspace allocation, sort-reduce orchestration |
| ``warp/_src/codegen.py`` | Atomic classification, scatter/phase codegen, hidden kernel params |
| ``warp/_src/context.py`` | Module option, ``_launch_deterministic()`` orchestrator, ctypes bindings |
| ``warp/native/deterministic.h`` | ``wp::det_ctx``, ``wp::det_scatter_buf<T>``, ``wp::det_counter_buf``, and device-side ``wp::deterministic::scatter()`` |
| ``warp/native/deterministic.cu`` | CUB workspace size queries, radix sort + ``ReduceByKey`` / segmented reduce kernels |
| ``warp/native/deterministic.cpp`` | CPU stubs (linker satisfaction when CUDA unavailable) |

## Current Limitations

- Deterministic lowering is effectively a module compilation mode. It is
  supported on shared modules via ``wp.set_module_options(...)`` and per-kernel
  via unique modules, but mixed deterministic/non-deterministic call graphs
  inside one shared compiled module are not the intended path.
- Within one function or kernel, a given array may participate in only one
  normalized deterministic family. ``atomic_add`` and ``atomic_sub`` share the
  same family; ``min`` and ``max`` are separate. Mixing families such as
  ``atomic_add(out, ...)`` and ``atomic_max(out, ...)`` on the same array in
  deterministic mode is a compile-time error.
- Phase 0 of the counter path suppresses normal side effects. Kernels whose
  correctness depends on writes performed during the counting pass are outside
  the supported model. This suppression is applied conservatively when the
  kernel's statically reachable ``@wp.func`` call graph contains a consumed
  counter atomic.
- The consumed-return counter path currently supports only ``int32`` counter
  arrays with literal counter index ``0``.
- Scatter buffers are fixed-capacity and rely on a static lower bound plus the
  optional ``deterministic_max_records`` override. Dynamic loops that exceed
  that bound will truncate records.
- During CUDA graph capture, Pattern A sorts each target's full fixed-capacity
  scatter buffer, including unused sentinel records. The exact emitted record
  count is device-side state, but CUB receives its item count as a host-side
  launch parameter during capture. Avoiding the extra sentinel work during
  replay would require synchronization, graph rebuild, or a custom
  device-driven sort/reduce path. This limitation is specific to graph capture;
  ordinary deterministic launches read the emitted count and sort only the
  valid prefix.
- Deterministic scatter buffer capacity is limited to ``2**31 - 1`` records
  because the native scatter buffer metadata and CUB sort/reduce counts use
  32-bit signed integers.
- ``"gpu_to_gpu"`` mode currently falls back to Warp's explicit segmented
  reduction path for scalar scatter reductions. Each segment head accumulates
  its segment serially so the accumulation order is fully controlled by Warp.
  This is correct but much slower than the ``"run_to_run"`` CUB ``ReduceByKey``
  fast path in high-contention cases.
- Deterministic scatter overflow detection is disabled for CUDA graph capture
  and replay to keep graph launches asynchronous. Graph workloads should size
  buffers conservatively with ``deterministic_max_records``.
- Deterministic kernels are supported in ordinary CUDA graph capture/replay,
  but not inside CUDA conditional body graphs such as ``wp.capture_while()``
  or ``wp.capture_if()`` when the deterministic launch would need to allocate
  scatter buffers, counter buffers, or sort/reduce workspaces while capturing
  the conditional body. Capture a fixed sequence, run that region outside
  conditional graph nodes, or disable deterministic mode there until Warp
  exposes a caller-provided reusable workspace for conditional body capture.
- APIC serialization (``apic=True`` capture for ``wp.capture_save()``) is not
  currently supported for deterministic CUDA kernels. APIC records
  user-visible launches and arguments, but deterministic mode adds hidden phase
  launches, temporary scatter/counter buffers, CUB workspace arrays, and native
  sort/reduce calls that APIC does not yet serialize as a self-contained
  workload. Use ordinary CUDA graph capture/replay without APIC serialization,
  or disable deterministic mode for kernels captured with ``apic=True``.
- Deterministic scatter launches are limited to ``2**32`` threads because the
  sort key packs the destination index and the linear thread index into one
  64-bit key.

## Testing Strategy

59 tests in ``warp/tests/test_deterministic.py`` cover:

- **Bit-exact reproducibility** (Pattern A): launch the same kernel 10 times
  with ``deterministic="run_to_run"``, assert ``np.array_equal`` across all
  runs for
  float32 scatter-add, ``+=`` syntax, float64, and atomic-sub.
- **Composite leaf types**: deterministic ``wp.vec3`` atomic-add and
  component-wise ``atomic_min``/``atomic_max``, plus deterministic ``wp.mat33``
  atomic-add.
- **Correctness** (Pattern A): compare GPU deterministic output against a
  sequential CPU reference within ``rtol=1e-4``.
- **Multiple arrays**: kernel that atomically adds to three different output
  arrays simultaneously.
- **Deterministic ``@wp.func`` propagation**: direct and nested ``@wp.func``
  calls containing deterministic atomics compile and execute correctly, and
  remain graph-capture-safe.
- **Mixed reduce ops on one array**: deterministic mode rejects mixing
  normalized reduction families on the same destination array.
- **Multi-dimensional indexing**: 2D array ``atomic_add`` with row/col indices.
- **Scatter capacity accounting**: kernels with more than two deterministic
  scatters per thread to the same target do not overflow a fixed heuristic
  buffer, and the native postpass receives the emitted record count instead of
  full scatter capacity outside CUDA graph capture.
- **Counter reproducibility** (Pattern B): ``slot = atomic_add(counter, 0, 1);
  output[slot] = data[tid]`` produces identical output arrays across 10 runs.
- **Phase 0 side-effect suppression**: non-counter array writes are skipped in
  the counting pass, including pure write helper calls and stores that occur
  before helper calls whose reachable call graph contains a consumed-return
  counter.
- **Counter correctness**: verifies counter value equals N and output is a
  permutation of input.
- **Conditional counter**: stream compaction (only elements above threshold),
  verifying correct count and reproducible output.
- **Mixed pattern**: both counter and accumulation in one kernel.
- **Integer passthrough**: integer ``atomic_add`` with unused return incurs no
  transformation; result matches ``np.bincount``.
- **Per-module override**: ``@wp.kernel(module_options={"deterministic": "gpu_to_gpu"},
  module="unique")`` works with global config off.
- **Kernel decorator override**: ``@wp.kernel(deterministic="gpu_to_gpu")``
  works with
  global config off.
- **Recorded launch support**: ``wp.launch(..., record_cmd=True)`` works for
  deterministic CUDA kernels.
- **Graph capture support**: deterministic scatter launches can be captured and
  replayed with CUDA graphs, including composite ``wp.vec3`` reductions.
- All tests run on both CPU and CUDA where applicable.  Existing
  ``test_atomic.py`` (158 tests) passes with zero regressions.
