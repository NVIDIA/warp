.. _determinism:

Determinism
===========

.. currentmodule:: warp

Deterministic execution is useful when you want the same program, with the same
inputs, to produce bit-exact results every time.  This matters for regression
tests, debugging, validation, and simulation workloads where a small numerical
change can make a later frame hard to compare.

The most common source of non-determinism in Warp kernels is a floating-point
atomic operation on CUDA.  If many threads execute ``wp.atomic_add(out, i,
value)`` at the same time, CUDA is free to apply those updates in different
orders.  Floating-point addition is not associative, so different orders can
produce different rounded results.

Warp's deterministic mode recognizes the common atomic patterns and rewrites
them into reproducible algorithms.  You keep writing ordinary Warp kernels; the
deterministic lowering happens when the module is compiled.

Quick Start
-----------

For one kernel, use a unique module and enable deterministic mode directly on
the kernel:

.. code:: python

    import warp as wp


    @wp.kernel(deterministic="run_to_run", module="unique")
    def accumulate(
        values: wp.array(dtype=wp.float32),
        bins: wp.array(dtype=wp.int32),
        out: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        wp.atomic_add(out, bins[tid], values[tid])


    n = 4096
    values = wp.ones(n, dtype=wp.float32, device="cuda")
    bins = wp.zeros(n, dtype=wp.int32, device="cuda")
    out = wp.zeros(1, dtype=wp.float32, device="cuda")

    wp.launch(accumulate, dim=n, inputs=[values, bins], outputs=[out], device="cuda")

For every kernel in a Python module, set a module option near the top of the
file, before the kernels are defined:

.. code:: python

    import warp as wp

    wp.set_module_options({"deterministic": "run_to_run"})


    @wp.kernel
    def accumulate(...):
        ...

For a process-wide default, set :attr:`wp.config.deterministic
<warp.config.deterministic>`:

.. code:: python

    import warp as wp

    wp.config.deterministic = "run_to_run"

The accepted modes are:

``"not_guaranteed"``
    The default.  Warp uses normal atomics.

``"run_to_run"``
    Results are reproducible across repeated runs on the same GPU architecture.
    This is the usual setting for debugging and regression tests.

``"gpu_to_gpu"``
    Uses a stronger reduction path intended to preserve the same result across
    GPU architectures.  It is more conservative and can be slower.

For backward compatibility, ``True`` is treated as ``"run_to_run"`` and
``False`` is treated as ``"not_guaranteed"``.

Choosing a Scope
----------------

Determinism is a compilation option.  In practice that means it belongs to a
compiled module, not just to one line of Python.

Use this rule of thumb:

* Use ``@wp.kernel(deterministic=..., module="unique")`` when you only want to
  opt in one kernel.
* Use :func:`wp.set_module_options() <warp.set_module_options>` when the kernels
  in the current Python file should all share the same deterministic setting.
* Use :attr:`wp.config.deterministic <warp.config.deterministic>` when you want
  a global default for a whole application or test run.

Why ``module="unique"``?  A normal Warp module may contain many kernels and
``@wp.func`` helpers.  Deterministic mode changes generated code and the module
hash.  A unique module keeps one per-kernel experiment from affecting unrelated
kernels that happen to live in the same Python file.

What Warp Supports
------------------

Deterministic mode currently handles:

* ``wp.atomic_add()``
* ``wp.atomic_sub()``
* ``wp.atomic_min()``
* ``wp.atomic_max()``
* The shorthand forms ``arr[i] += value`` and ``arr[i] -= value``

Warp recognizes two patterns.

Pattern 1: Accumulation
~~~~~~~~~~~~~~~~~~~~~~~

This is the most common pattern.  Many threads contribute values to one or more
output arrays, and the atomic return value is ignored:

.. code:: python

    @wp.kernel(deterministic="run_to_run", module="unique")
    def bin_values(
        values: wp.array(dtype=wp.float32),
        bins: wp.array(dtype=wp.int32),
        out: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        wp.atomic_add(out, bins[tid], values[tid])

With deterministic mode enabled, Warp does not apply the floating-point atomic
directly.  Instead, each thread writes a temporary record containing:

* the output element that should be updated, and
* the value to add, subtract, minimize, or maximize.

After the kernel finishes, Warp sorts those records into a stable order and
then reduces each group.  This is similar to the "sort then reduce by key"
pattern used in CUDA programming.  Internally, Warp uses CUB device-wide
primitives from NVIDIA CCCL, including `CUB DeviceRadixSort`_ and
`CUB DeviceReduce`_, for the fast ``"run_to_run"`` scalar path.  Composite
types and the stronger ``"gpu_to_gpu"`` path use a Warp-controlled segmented
reduction so the accumulation order is explicit.

The following operations are handled by this pattern:

.. list-table::
   :header-rows: 1

   * - Atomic
     - Deterministic behavior
   * - ``wp.atomic_add(out, i, value)``
     - Values are added in a fixed order.
   * - ``wp.atomic_sub(out, i, value)``
     - Values are negated and added in a fixed order.
   * - ``wp.atomic_min(out, i, value)``
     - Values are grouped by destination and minimized.
   * - ``wp.atomic_max(out, i, value)``
     - Values are grouped by destination and maximized.

Floating-point scalar arrays, vectors, matrices, quaternions, transforms, and
``wp.Struct`` fields that contain arrays can participate in this pattern.
Integer atomics whose return value is ignored usually do not need rewriting
because the final result is already deterministic for the supported operations.

Pattern 2: Slot Allocation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Some kernels use an atomic return value as a slot number:

.. code:: python

    @wp.kernel(deterministic="run_to_run", module="unique")
    def compact_positive(
        values: wp.array(dtype=wp.float32),
        count: wp.array(dtype=wp.int32),
        compacted: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        v = values[tid]

        if v > 0.0:
            slot = wp.atomic_add(count, 0, 1)
            compacted[slot] = v

Normal atomic execution makes ``slot`` depend on the order in which threads
arrive.  Deterministic mode turns this into the familiar count-scan-write
workflow:

1. Run a counting pass.  Each thread records how many slots it needs.
2. Run a prefix scan.  This computes each thread's deterministic starting slot.
3. Run the kernel again.  The atomic returns the deterministic slot.

The prefix scan is the same concept as an exclusive scan in CUDA libraries.
On CUDA, Warp's scan utilities are implemented with CUB primitives such as
`CUB DeviceScan`_.

This pattern currently supports only:

* ``slot = wp.atomic_add(counter, 0, value)``
* ``counter`` must be an ``int32`` array
* every counter index must be the literal ``0``

Consumed-return ``atomic_sub``, ``atomic_min``, and ``atomic_max`` are rejected
because their semantics are not prefix-sum slot allocation.

Picking the Right Mode
----------------------

Start with ``"run_to_run"``.  It gives bit-exact repeated runs on the same GPU
architecture and uses the fastest deterministic path for scalar reductions.

Use ``"gpu_to_gpu"`` when you need a stronger guarantee across GPU
architectures.  This is more expensive because Warp avoids relying on reduction
orders that may change with architecture-specific CUB policies.

The mode names match the determinism vocabulary used by NVIDIA CCCL.  Recent
CCCL releases also expose determinism controls for some CUB reductions; see
`Controlling Floating-Point Determinism in NVIDIA CCCL`_ for background.  Warp's
feature is different in scope: it targets Warp kernel atomics and automatically
rewrites the supported atomic patterns described above.

Capacity and Dynamic Loops
--------------------------

For Pattern 1, Warp allocates temporary scatter buffers.  The default capacity
comes from a static estimate of how many deterministic atomic records each
thread can emit.

If a thread can revisit the same atomic site inside a dynamic loop, Warp may
not be able to prove the maximum number of records.  In that case, provide a
per-target, per-thread upper bound:

.. code:: python

    @wp.kernel(
        deterministic="run_to_run",
        deterministic_max_records=8,
        module="unique",
    )
    def loop_accumulate(
        values: wp.array(dtype=wp.float32),
        loop_counts: wp.array(dtype=wp.int32),
        out: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()

        for i in range(loop_counts[tid]):
            wp.atomic_add(out, 0, values[tid])

If the runtime count exceeds the capacity, Warp truncates records and sets an
overflow flag.  Outside CUDA graph capture, the launch reports this as an
error.  For debugging capacity problems, enable:

.. code:: python

    wp.config.deterministic_debug = True

Leave this off in performance-sensitive code.

CUDA Graph Capture
------------------

Deterministic accumulation kernels can be captured and replayed with ordinary
CUDA graphs:

.. code:: python

    output.zero_()

    with wp.ScopedCapture(device="cuda", force_module_load=False) as capture:
        wp.launch(accumulate, dim=n, inputs=[values, bins], outputs=[output], device="cuda")

    wp.capture_launch(capture.graph)

There are a few details to know:

* During graph capture, Warp cannot read the scatter record count back to the
  host without synchronizing.  Captured Pattern 1 launches therefore sort the
  full fixed-capacity buffer, including unused sentinel records.  This is
  correct but can be slower than ordinary non-captured launches, which sort
  only the emitted record prefix.
* Overflow checks are disabled during graph capture and replay to keep graph
  launches asynchronous.  Size buffers conservatively with
  ``deterministic_max_records``.
* Counter/slot-allocation kernels use prefix scans.  CUDA graph replay requires
  all temporary scan storage used by captured CUB work to remain valid for the
  lifetime of the graph.  Warp currently rejects Pattern 2 kernels during CUDA
  graph capture; launch them outside capture, or disable deterministic mode for
  that kernel.
* Deterministic kernels are not supported inside CUDA conditional body graphs,
  such as :func:`wp.capture_while() <warp.capture_while>` or
  :func:`wp.capture_if() <warp.capture_if>`, when the deterministic launch
  would need to allocate temporary buffers while capturing the conditional
  body.
* APIC serialization (``apic=True`` with :func:`wp.capture_save()
  <warp.capture_save>`) is not currently supported for deterministic CUDA
  kernels.  Use ordinary CUDA graph capture or disable deterministic mode for
  APIC-captured kernels.

Common Limitations
------------------

Deterministic mode is intentionally narrow.  It supports common reproducibility
patterns without pretending that every parallel program can be made
deterministic automatically.

Unsupported atomics
    ``wp.atomic_cas()``, ``wp.atomic_exch()``, and tile atomics such as
    ``wp.tile_atomic_add()`` are not transformed.

One reduction family per target
    A single output array cannot mix different deterministic families inside
    one function or kernel.  For example, using ``wp.atomic_add(out, ...)`` and
    ``wp.atomic_max(out, ...)`` on the same array in the same deterministic
    kernel is rejected.  Write to separate arrays or split the work into
    separate kernels.

Side effects in the counting pass
    Pattern 2 runs the kernel once just to count.  During this pass, Warp
    suppresses side effects such as array writes, non-counter atomics, and
    ``printf``.  If the number of slots a thread needs depends on scratch-array
    writes performed earlier in the same kernel, rewrite the code to use local
    variables or input arrays for that decision.

Fixed scatter capacity
    Pattern 1 uses fixed-capacity scatter buffers.  Dynamic loops that emit
    more records than the static estimate or ``deterministic_max_records`` can
    overflow.

Large launches
    Deterministic scatter launches are limited to at most ``2**32`` threads.
    The scatter sort key stores the destination index and the linear thread
    index in one 64-bit value.  Deterministic counter launches are limited to
    at most ``2**31 - 1`` threads because their prefix buffers store
    per-thread contributions as ``int32`` values.

Performance cost
    Deterministic mode adds work.  Accumulation atomics sort and reduce
    temporary records.  Slot allocation runs an extra counting pass and a scan.
    Use it where reproducibility matters, not as a default performance mode.

Checklist
---------

When enabling deterministic mode in a new kernel:

1. Start with ``@wp.kernel(deterministic="run_to_run", module="unique")``.
2. Run the kernel several times and compare outputs with
   ``np.testing.assert_array_equal()`` for bit-exact tests.
3. If the kernel has dynamic loops around deterministic atomics, set
   ``deterministic_max_records`` to a real upper bound.
4. If the kernel uses ``slot = wp.atomic_add(counter, 0, value)``, check that
   the counter is ``int32`` and indexed at literal ``0``.
5. If you capture the kernel in a CUDA graph, validate both capture and replay.
   Pattern 2 counter kernels are not currently supported in graph capture.
6. If performance matters, benchmark both normal and deterministic modes.

Related NVIDIA Libraries
------------------------

Warp's implementation builds on CUDA and CUB concepts rather than inventing a
new vocabulary:

* `CUB`_ is part of `NVIDIA CCCL`_ and provides optimized device-wide building
  blocks such as radix sort, reduce-by-key, and scan.
* Pattern 1 follows the same general shape as CUB sort plus reduce-by-key:
  group equal destinations, then reduce each group.
* Pattern 2 follows the standard count-scan-write pattern and uses scan
  primitives conceptually like CUB ``DeviceScan``.
* The ``"not_guaranteed"``, ``"run_to_run"``, and ``"gpu_to_gpu"`` names match
  the determinism levels described for CCCL reductions.

These references are useful if you want to understand the lower-level CUDA
building blocks behind Warp's higher-level switch.

.. _CUB: https://nvidia.github.io/cccl/cub/
.. _NVIDIA CCCL: https://nvidia.github.io/cccl/
.. _CUB DeviceRadixSort: https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
.. _CUB DeviceReduce: https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html
.. _CUB DeviceScan: https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceScan.html
.. _Controlling Floating-Point Determinism in NVIDIA CCCL: https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/
