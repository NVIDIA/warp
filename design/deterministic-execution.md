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
| R6  | Per-module and per-kernel granularity via ``module_options`` | Should | Allows selective opt-in |
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
  ``deterministic_max_records=...``.

For backward compatibility, ``True`` and ``False`` are still accepted at the
module and kernel level as aliases for ``"run_to_run"`` and
``"not_guaranteed"``, respectively.

Like ``enable_backward``, the setting participates in module compilation and
hashing. A kernel defined in a shared module inherits that module's
``deterministic`` option; a unique-module kernel can override it independently.

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
a CUB radix sort orders the fixed-capacity scatter buffer by key and record
index. The post-sort reduction depends on the selected determinism guarantee:

- ``"run_to_run"``: scalar reductions use CUB ``DeviceReduce::ReduceByKey`` on
  destination indices for better performance.
- ``"gpu_to_gpu"``: reductions use the simple sequential segmented kernel so
  the left-to-right accumulation order is explicit in Warp's own code.
- Composite leaf types always use the segmented kernel.

Unused buffer slots are initialized with an invalid sentinel key and sort to
the end. This avoids host-side scatter count readbacks and keeps the path
compatible with CUDA graph capture.

**Pattern B --- Two-Pass Execution** (counter/allocator, return value used):

The kernel runs twice:
1. *Phase 0 (counting)*: The kernel executes with all side effects suppressed.
   Counter atomics record per-thread contributions to a scratch array instead
   of performing the actual atomic.
2. *Prefix sum*: ``wp.utils.array_scan(contrib, prefix, inclusive=False)``
   computes deterministic per-thread offsets.
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

**CPU/CUDA dual compilation** is handled with ``#ifdef __CUDA_ARCH__`` guards
in the generated function body.  On CUDA the scatter or phase-branching code
executes; on CPU the normal ``wp::atomic_add(...)`` call is emitted in the
``#else`` branch.  This is necessary because Warp generates a single function
body that compiles for both targets.

**Hidden kernel parameters** are appended to the CUDA kernel signature by
``codegen_kernel()`` after the user arguments.  Pattern B kernels get
``_wp_det_phase``, ``_wp_det_contrib_N``, ``_wp_det_prefix_N``. Pattern A
targets get ``_wp_scatter_keys_N``, ``_wp_scatter_vals_N``,
``_wp_scatter_ctr_N``, ``_wp_scatter_overflow_N``, ``_wp_scatter_cap_N``, and
``_wp_det_debug``. The launch system
(``_launch_deterministic``) allocates these buffers and appends the
corresponding ctypes params to the launch args.

**Multiple reduction buffers**: each distinct ``(target array, value type,
reduction op)`` combination gets its own scatter buffer set. Multiple call
sites with the same target and reduction op share one buffer. The
``DeterministicMeta`` dataclass on the kernel's ``Adjoint`` tracks all scatter
and counter targets discovered during codegen.

**Scatter sizing**: each scatter target uses a fixed-capacity buffer sized from
a code-generated lower bound (static records-per-thread analysis). The optional
``deterministic_max_records`` setting overrides the per-thread record count when
users know a thread may revisit the same atomic site multiple times, for
example inside a dynamic loop. Warp uses
``max(codegen_lower_bound, deterministic_max_records)`` records per thread for
each target. On overflow, new records are truncated, a device-side overflow
flag is set, and optional diagnostics may be emitted when
``wp.config.deterministic_debug`` is enabled.

**Counter total writeback**: after the prefix sum in Phase 0, the launch system
copies the total count (last element of the inclusive scan) back to the actual
counter array so user code that reads it post-launch sees the correct value.

**Files added/modified**:

| File | Role |
| --- | --- |
| ``warp/config.py`` | ``deterministic`` global flag |
| ``warp/_src/deterministic.py`` | Dataclasses, buffer allocation, sort-reduce orchestration |
| ``warp/_src/codegen.py`` | Atomic classification, scatter/phase codegen, hidden kernel params |
| ``warp/_src/context.py`` | Module option, ``_launch_deterministic()`` orchestrator, ctypes bindings |
| ``warp/native/deterministic.h`` | Device-side ``wp::deterministic::scatter()`` template |
| ``warp/native/deterministic.cu`` | CUB radix sort + ``ReduceByKey`` / segmented reduce kernels |
| ``warp/native/deterministic.cpp`` | CPU stubs (linker satisfaction when CUDA unavailable) |

## Testing Strategy

31 tests in ``warp/tests/test_deterministic.py`` cover:

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
- **Mixed reduce ops on one array**: ``atomic_add`` and ``atomic_max`` targeting
  the same destination array are reduced independently.
- **Multi-dimensional indexing**: 2D array ``atomic_add`` with row/col indices.
- **Scatter capacity accounting**: kernels with more than two deterministic
  scatters per thread to the same target do not overflow a fixed heuristic
  buffer.
- **Counter reproducibility** (Pattern B): ``slot = atomic_add(counter, 0, 1);
  output[slot] = data[tid]`` produces identical output arrays across 10 runs.
- **Phase 0 side-effect suppression**: non-counter array writes are skipped in
  the counting pass.
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
