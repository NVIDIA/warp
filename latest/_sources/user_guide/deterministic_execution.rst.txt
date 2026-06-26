.. _determinism:

Deterministic Execution
=======================

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

For an existing Warp module, set a module option near the top of the Python
file that defines your kernels, before the kernels are defined:

.. code:: python

    import warp as wp

    wp.set_module_options({"deterministic": wp.DeterministicMode.RUN_TO_RUN})


    @wp.kernel
    def accumulate(
        values: wp.array[wp.float32],
        bins: wp.array[wp.int32],
        out: wp.array[wp.float32],
    ):
        tid = wp.tid()
        wp.atomic_add(out, bins[tid], values[tid])


    n = 4096
    values = wp.ones(n, dtype=wp.float32, device="cuda")
    bins = wp.zeros(n, dtype=wp.int32, device="cuda")
    out = wp.zeros(1, dtype=wp.float32, device="cuda")

    wp.launch(accumulate, dim=n, inputs=[values, bins], outputs=[out], device="cuda")

This is the most common way to turn determinism on after the fact: every kernel
defined in that Python module uses deterministic lowering when it contains a
supported atomic pattern.

For a process-wide default, set :attr:`wp.config.deterministic
<warp.config.deterministic>` before modules are compiled:

.. code:: python

    import warp as wp

    wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN

If you only want to experiment with one kernel in an otherwise shared Python
module, put that kernel in a unique module and pass module options:

.. code:: python

    @wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
    def accumulate(...):
        ...

The accepted modes are:

``wp.DeterministicMode.NOT_GUARANTEED``
    The default.  Warp uses normal atomics.

``wp.DeterministicMode.RUN_TO_RUN``
    Results are reproducible across repeated runs on the same GPU architecture.
    This is the usual setting for debugging and regression tests.

``wp.DeterministicMode.GPU_TO_GPU``
    Uses a stronger reduction path intended to preserve the same result across
    GPU architectures.  It is more conservative and can be slower.

Choosing a Scope
----------------

Determinism is a compilation option.  In practice that means it belongs to a
compiled module, not just to one line of Python.

Use this rule of thumb:

* Use :func:`wp.set_module_options() <warp.set_module_options>` when the kernels
  in the current Python file should all share the same deterministic setting.
* Use :attr:`wp.config.deterministic <warp.config.deterministic>` when you want
  a global default for modules created later in a whole application or test run.
* Use ``@wp.kernel(module="unique", module_options={...})`` when one kernel
  should have its own compiled module and deterministic setting.

The kernel decorator does not accept a direct ``deterministic=...`` shorthand.
Determinism is a module-level code-generation option because Warp normally
compiles the kernels and ``@wp.func`` helpers in a Python module into one module
binary.  Deterministic lowering can change generated code for a kernel's
reachable helper functions, especially when supported atomics live in
``@wp.func`` calls.  ``module="unique"`` gives that kernel its own compiled
module, so the deterministic and non-deterministic variants do not share helper
code or module hashes.

Changing :attr:`wp.config.deterministic <warp.config.deterministic>` after a
module has been created does not update that module.  To toggle an existing
module on and off, use ``wp.set_module_options()``.

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

    @wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
    def bin_values(
        values: wp.array[wp.float32],
        bins: wp.array[wp.int32],
        out: wp.array[wp.float32],
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
`CUB DeviceReduce`_, for the fast ``wp.DeterministicMode.RUN_TO_RUN`` scalar
path.  Composite types and the stronger ``wp.DeterministicMode.GPU_TO_GPU``
path use a Warp-controlled segmented reduction so the accumulation order is
explicit.

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

    @wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
    def compact_positive(
        values: wp.array[wp.float32],
        count: wp.array[wp.int32],
        compacted: wp.array[wp.float32],
    ):
        tid = wp.tid()
        v = values[tid]

        if v > 0.0:
            slot = wp.atomic_add(count, 0, 1)
            compacted[slot] = v

Normal atomic execution makes ``slot`` depend on the order in which threads
arrive.  Deterministic mode turns this into the familiar count-scan-write
workflow:

1. Run a counting pass.  Each thread records each slot reservation, including
   the counter element, the reservation order for that thread, and the number
   of slots requested.
2. Sort reservations by counter element and deterministic thread order.
3. Run a segmented prefix scan.  This computes the deterministic starting slot
   for each reservation and the final total for each touched counter element.
4. Run the kernel again.  The atomic returns the deterministic slot.

The prefix scan is the same concept as an exclusive scan in CUDA libraries.
On CUDA, Warp's scan utilities are implemented with CUB primitives such as
`CUB DeviceScan`_.

This pattern currently supports:

* ``slot = wp.atomic_add(counter, index, value)``
* ``counter`` must be an ``int32`` array
* ``index`` may be constant or data-dependent
* Sliced counter views such as ``wp.atomic_add(counters[world], 0, 1)``

Consumed-return ``atomic_sub``, ``atomic_min``, and ``atomic_max`` are rejected
because their semantics are not prefix-sum slot allocation.

When this pattern appears in differentiable code, generated backward replay is
not supported automatically.  The backward pass would need the exact slot that
each thread received in the forward pass, but the counter array has already
been mutated and later launches may have appended more records.  Warp therefore
raises an error before launching the generated adjoint instead of recomputing a
possibly wrong slot.  If gradients need the slot, store the mapping explicitly
and provide a custom replay function, as shown in
:ref:`custom-replay-function`.  If the kernel is only bookkeeping, mark it with
``enable_backward=False`` or disable deterministic mode for that kernel.

Tape and Backward Passes
------------------------

Deterministic mode also applies to generated CUDA backward kernels launched by
:class:`wp.Tape <warp.Tape>`.  This matters for differentiable simulations:
the forward pass may be bit-exact, while the backward pass still scatters many
per-thread contributions into the same input gradients.

The most common example is a gather in the forward pass:

.. code:: python

    @wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
    def gather(
        values: wp.array[wp.float32],
        indices: wp.array[wp.int32],
        out: wp.array[wp.float32],
    ):
        tid = wp.tid()
        out[tid] = values[indices[tid]]


The forward kernel has no atomic operation.  During ``Tape.backward()``,
however, Warp's generated adjoint accumulates ``out.grad[tid]`` into
``values.grad[indices[tid]]``.  If several output lanes read the same input
lane, that gradient accumulation is an atomic add.  With deterministic mode
enabled, Warp routes that reverse accumulation through the same scatter,
sort, and reduce path used for Pattern 1.

User-defined custom adjoints are covered when their gradient accumulation is
written in Warp code using supported atomic patterns:

.. code:: python

    @wp.func_grad(load_value)
    def adj_load_value(values: wp.array[wp.float32], index: int, adj_ret: wp.float32):
        wp.adjoint[values][index] += adj_ret


The deterministic launcher distinguishes forward-only, backward-only, and
shared deterministic targets.  Backward-only gradient targets are part of the
compiled hidden ABI, but forward launches pass inert helper buffers for those
targets and skip their sort/reduce work.  That keeps the forward path from
paying for gradient reductions that only exist during ``Tape.backward()``.
When an input array has no adjoint buffer, for example because it was read by
the forward pass but does not require gradients, deterministic mode treats the
corresponding backward target as inactive and leaves the generated helper as a
no-op.  Other gradients from the same backward kernel are still reduced
deterministically.

The unsupported autodiff case is Pattern 2 slot allocation with a consumed
counter return.  A generated backward pass normally replays the original
forward code before running reverse statements.  Replaying
``slot = wp.atomic_add(counter, index, value)`` would allocate new slots from
the already-mutated counter, which can silently send adjoints to the wrong
elements.  Deterministic mode rejects that launch instead.  The supported fix
is to save the forward slot mapping yourself and use ``@wp.func_replay`` so the
backward replay reads the saved slot rather than running the counter atomic
again.

Writing Manual Deterministic Code
---------------------------------

Warp's deterministic mode is a convenience layer.  You can also write the
deterministic algorithm directly when you want tighter control over memory,
launch boundaries, or performance.

For small output spaces, the simplest deterministic accumulation is to give
each output element one thread and visit the inputs in a fixed order:

.. code:: python

    @wp.kernel
    def reduce_by_bin(
        values: wp.array[wp.float32],
        bins: wp.array[wp.int32],
        out: wp.array[wp.float32],
        n: int,
    ):
        bin_id = wp.tid()
        total = wp.float32(0.0)

        for i in range(n):
            if bins[i] == bin_id:
                total += values[i]

        out[bin_id] = total

This avoids atomics entirely and accumulates each bin in input-index order.  It
is easy to reason about, but it is ``O(num_bins * num_values)``, so it is most
useful for debugging, validation, or small numbers of output bins.

For slot allocation, write the count-scan-write pattern explicitly:

.. code:: python

    @wp.kernel
    def count_positive(
        values: wp.array[wp.float32],
        counts: wp.array[wp.int32],
    ):
        tid = wp.tid()
        counts[tid] = wp.int32(values[tid] > 0.0)


    @wp.kernel
    def write_positive(
        values: wp.array[wp.float32],
        offsets: wp.array[wp.int32],
        out: wp.array[wp.float32],
    ):
        tid = wp.tid()

        if values[tid] > 0.0:
            out[offsets[tid]] = values[tid]


    counts = wp.empty(n, dtype=wp.int32, device="cuda")
    offsets = wp.empty(n, dtype=wp.int32, device="cuda")

    wp.launch(
        count_positive,
        dim=n,
        inputs=[values],
        outputs=[counts],
        device="cuda",
    )
    wp.utils.array_scan(counts, offsets, inclusive=False)
    wp.launch(
        write_positive,
        dim=n,
        inputs=[values, offsets],
        outputs=[compacted],
        device="cuda",
    )

This is the same basic lowering that deterministic consumed-return counters use
internally.  If you also need the final count, it is ``offsets[n - 1] +
counts[n - 1]``.

Picking the Right Mode
----------------------

Start with ``wp.DeterministicMode.RUN_TO_RUN``.  It gives bit-exact repeated
runs on the same GPU architecture and is the fastest deterministic mode.  For
scalar accumulation atomics, Warp writes scatter records, sorts them by
destination and linear thread index, and then uses `CUB DeviceReduce`_
reduce-by-key to combine records with the same destination.  This is stable for
repeated runs on the same GPU architecture, but the internal reduction tree may
vary across architectures.

Use ``wp.DeterministicMode.GPU_TO_GPU`` when the same kernel should produce the
same result across GPU architectures.  Warp still uses CUB radix sort to group
records, but it does not use CUB's scalar reduce-by-key path.  Instead, Warp
launches a segmented reduction kernel that walks each sorted destination
segment in thread-index order.  That fixed accumulation order is more
conservative and can be slower, especially when many threads contribute to the
same destination.

Composite values such as vectors, matrices, quaternions, transforms, and
``wp.Struct`` fields already use Warp's segmented reduction path in both modes.
Pattern 2 slot allocation uses the same count-scan-write algorithm in both
enabled modes.

The enum values correspond to determinism levels used by NVIDIA CCCL.  Recent
CCCL releases also expose determinism controls for some CUB reductions; see
`Controlling Floating-Point Determinism in NVIDIA CCCL`_ for background.  Warp's
feature is different in scope: it targets Warp kernel atomics and automatically
rewrites the supported atomic patterns described above.

Capacity and Dynamic Loops
--------------------------

For Pattern 1, Warp allocates temporary scatter buffers.  The default capacity
comes from a static estimate of how many deterministic atomic records each
thread can emit.

Warp keeps a small floor of 1024 records per scatter target.  Very small
launches can therefore allocate more scatter storage than their exact record
count would require.  This keeps the common path simple and gives repeated
launches enough room without adding another tuning parameter.

If a thread can revisit the same atomic site inside a dynamic loop, Warp may
not be able to prove the maximum number of records.  In that case, provide a
per-target, per-thread upper bound.  For one module or kernel, use the module
option directly:

.. code:: python

    @wp.kernel(
        module="unique",
        module_options={
            "deterministic": wp.DeterministicMode.RUN_TO_RUN,
            "deterministic_max_records": 8,
        },
    )
    def loop_accumulate(
        values: wp.array[wp.float32],
        loop_counts: wp.array[wp.int32],
        out: wp.array[wp.float32],
    ):
        tid = wp.tid()

        for i in range(loop_counts[tid]):
            wp.atomic_add(out, 0, values[tid])

To apply the same bound to modules created after the config is set, use the
global default:

.. code:: python

    wp.config.deterministic_max_records = 8

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
* Counter/slot-allocation kernels use sort and prefix-scan workspaces.  CUDA
  graph replay requires all temporary CUB storage used by captured work to
  remain valid for the lifetime of the graph.  Warp keeps these buffers alive
  on the captured graph object for ordinary CUDA graph capture.
* Deterministic kernels are not supported inside CUDA conditional body graphs,
  such as :func:`wp.capture_while() <warp.capture_while>` or
  :func:`wp.capture_if() <warp.capture_if>`, when the deterministic launch
  would need to allocate temporary buffers while capturing the conditional
  body.
* APIC serialization (``apic=True`` with :func:`wp.capture_save()
  <warp.capture_save>`) is not currently supported for deterministic CUDA
  kernels.  Use ordinary CUDA graph capture or disable deterministic mode for
  APIC-captured kernels.

Performance
-----------

Deterministic mode changes the algorithm, so there is no single overhead
number.  The important question is which pattern the kernel uses and how much
contention the atomics have.  The measurements below use CUDA event timing in
both cases so they measure GPU work rather than Python wall-clock time.

Pattern A: Accumulation Atomics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accumulation atomics can get faster when many threads write to the
same output element.  Normal floating-point atomics serialize at the hot
address.  Deterministic ``wp.DeterministicMode.RUN_TO_RUN`` mode instead writes
temporary records, sorts them by destination, and reduces each group.  That
extra work is not free, but it can remove enough atomic contention to win.
Sparse atomics usually slow down because there is little contention to remove.

The benchmark kernel for Pattern A is deliberately small:

.. code:: python

    @wp.kernel
    def atomic_add_kernel(
        values: wp.array[wp.float32],
        indices: wp.array[wp.int32],
        out: wp.array[wp.float32],
    ):
        tid = wp.tid()
        wp.atomic_add(out, indices[tid], values[tid])


    out.zero_()
    wp.launch(
        atomic_add_kernel,
        dim=num_writes,
        inputs=[values, indices],
        outputs=[out],
        device="cuda",
    )

To compare modes, the benchmark uses the same kernel body compiled normally,
with ``module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN}``,
and with ``module_options={"deterministic": wp.DeterministicMode.GPU_TO_GPU}``.
The timed path is CUDA graph replay of the output zeroing plus the kernel launch,
measured with CUDA events.

The table below is one local measurement, not a performance guarantee.  It was
measured on an NVIDIA GeForce RTX 4090 with CUDA 12.4.  Each number is median
CUDA event time in milliseconds.  ``outputs=1`` means every thread writes to the
same element; ``outputs=65,536`` uses random output indices and much lower
contention.

.. list-table::
   :header-rows: 1

   * - Writes
     - Outputs
     - ``NOT_GUARANTEED``
     - ``RUN_TO_RUN``
     - ``GPU_TO_GPU``
     - ``RUN_TO_RUN`` ratio
   * - 65,536
     - 1
     - 0.099 ms
     - 0.075 ms
     - 7.66 ms
     - 0.76x
   * - 262,144
     - 1
     - 0.364 ms
     - 0.096 ms
     - 31.0 ms
     - 0.26x
   * - 1,048,576
     - 1
     - 1.46 ms
     - 0.193 ms
     - 172 ms
     - 0.13x
   * - 65,536
     - 65,536
     - 0.0058 ms
     - 0.079 ms
     - 0.076 ms
     - 13.6x
   * - 262,144
     - 65,536
     - 0.0081 ms
     - 0.192 ms
     - 0.187 ms
     - 23.9x
   * - 1,048,576
     - 65,536
     - 0.020 ms
     - 0.201 ms
     - 0.228 ms
     - 10.0x

The first three rows are the high-contention case:
``wp.DeterministicMode.RUN_TO_RUN`` gets faster as the number of writes to the
one output grows.  The last three rows are the low-contention case: normal
atomics are already cheap, so sorting dominates.
``wp.DeterministicMode.GPU_TO_GPU`` preserves a stricter order and is
intentionally more expensive for one long destination segment.

Pattern B: Slot Allocation Counters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Slot allocation has a different shape.  The kernel is:

.. code:: python

    @wp.kernel
    def counter_kernel(
        values: wp.array[wp.float32],
        count: wp.array[wp.int32],
        out: wp.array[wp.float32],
    ):
        tid = wp.tid()
        slot = wp.atomic_add(count, 0, 1)
        out[slot] = values[tid]


    count.zero_()
    wp.launch(counter_kernel, dim=num_writes, inputs=[values, count], outputs=[out], device="cuda")

Deterministic mode turns this into a counting pass, a sort by counter element
and deterministic reservation order, a segmented prefix scan, and an execution
pass.  ``wp.DeterministicMode.RUN_TO_RUN`` and
``wp.DeterministicMode.GPU_TO_GPU`` use the same slot allocation algorithm.
This table uses direct launches measured with CUDA events.

.. list-table::
   :header-rows: 1

   * - Writes
     - ``NOT_GUARANTEED``
     - Deterministic
     - Ratio
   * - 65,536
     - 0.020 ms
     - 0.129 ms
     - 6.3x
   * - 262,144
     - 0.024 ms
     - 0.144 ms
     - 6.0x
   * - 1,048,576
     - 0.022 ms
     - 0.165 ms
     - 7.6x

These counter numbers measure the device work only.  Application wall time can
be higher because this direct-launch path still allocates temporary prefix-scan
buffers around the launch.  The practical advice is to use deterministic slot
allocation where stable ordering matters, and benchmark it in the workload that
matters to you.

Limitations
-----------

Deterministic mode is intentionally narrow.  It supports common reproducibility
patterns without pretending that every parallel program can be made
deterministic automatically.

Unsupported atomics
    ``wp.atomic_cas()``, ``wp.atomic_exch()``, and tile atomics such as
    ``wp.tile_atomic_add()`` are not transformed.

    .. code:: python

        @wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
        def not_rewritten(lock: wp.array[wp.int32]):
            # Compare-and-swap is still the ordinary Warp atomic.
            wp.atomic_cas(lock, 0, 0, 1)

Custom/native adjoint boundaries
    User-defined custom adjoints are deterministic when their contended
    accumulation is written in Warp code with supported atomic patterns, such
    as ``wp.adjoint[arr][i] += value``.  Native C++ adjoint helpers and
    unsupported atomics remain outside the automatic rewrite.

One reduction family per target
    A single output array cannot mix different deterministic families inside
    one function or kernel.  For example, using ``wp.atomic_add(out, ...)`` and
    ``wp.atomic_max(out, ...)`` on the same array in the same deterministic
    kernel is rejected.  Write to separate arrays or split the work into
    separate kernels.

    .. code:: python

        @wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
        def rejected(values: wp.array[wp.float32], out: wp.array[wp.float32]):
            tid = wp.tid()
            wp.atomic_add(out, 0, values[tid])
            wp.atomic_max(out, 0, values[tid])

    One workaround is to write each reduction family to a separate target:

    .. code:: python

        @wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
        def allowed(
            values: wp.array[wp.float32],
            sum_out: wp.array[wp.float32],
            max_out: wp.array[wp.float32],
        ):
            tid = wp.tid()
            wp.atomic_add(sum_out, 0, values[tid])
            wp.atomic_max(max_out, 0, values[tid])

Side effects in the counting pass
    Pattern 2 runs the kernel once just to count.  During this pass, Warp
    suppresses side effects such as array writes, non-counter atomics, and
    ``printf``.  If the number of slots a thread needs depends on scratch-array
    writes performed earlier in the same kernel, rewrite the code to use local
    variables or input arrays for that decision.

    .. code:: python

        @wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
        def avoid_scratch_dependency(
            values: wp.array[wp.float32],
            scratch: wp.array[wp.float32],
            count: wp.array[wp.int32],
            out: wp.array[wp.float32],
        ):
            tid = wp.tid()
            scratch[tid] = values[tid]

            # During the counting pass, the scratch write is suppressed.
            if scratch[tid] > 0.0:
                slot = wp.atomic_add(count, 0, 1)
                out[slot] = scratch[tid]

Fixed scatter capacity
    Pattern 1 uses fixed-capacity scatter buffers.  Dynamic loops that emit
    more records than the static estimate or ``deterministic_max_records`` can
    overflow.

    .. code:: python

        @wp.kernel(
            module="unique",
            module_options={
                "deterministic": wp.DeterministicMode.RUN_TO_RUN,
                "deterministic_max_records": 4,
            },
        )
        def can_overflow(loop_counts: wp.array[wp.int32], out: wp.array[wp.float32]):
            tid = wp.tid()

            for i in range(loop_counts[tid]):
                wp.atomic_add(out, 0, 1.0)

Consumed counters in backward replay
    Generated backward replay of Pattern 2 is rejected.  Warp cannot safely
    recompute the consumed slot from a counter array that the forward pass
    already changed.  Store the forward slot mapping yourself and provide a
    ``@wp.func_replay`` implementation when gradients need that slot.  Consumed
    counter atomics inside custom adjoint code are also rejected; move slot
    allocation to the forward pass and replay an explicit slot mapping instead.

Counter arrays are allocation counters
    Pattern 2 snapshots the counter values before the counting pass and writes
    final totals after the execution pass.  This supports pre-existing
    nonzero counters and append-style multi-launch workflows.  It is still an
    allocation-counter pattern, not a general way to make arbitrary
    order-dependent counter mutations deterministic.

Large launches
    Deterministic scatter and counter launches are limited to at most
    ``2**31 - 1`` records.  Scatter buffers use signed 32-bit record slots, and
    counter prefix buffers store per-record contributions as ``int32`` values.

Checklist
---------

When enabling deterministic mode in a new kernel:

1. Start with ``wp.set_module_options({"deterministic": wp.DeterministicMode.RUN_TO_RUN})``
   for an existing module, or a unique module with ``module_options`` for one
   targeted kernel.
2. Run the kernel several times and compare outputs with
   ``np.testing.assert_array_equal()`` for bit-exact tests.
3. If the kernel has dynamic loops around deterministic atomics, set
   ``deterministic_max_records`` to a real upper bound.
4. If the kernel uses ``slot = wp.atomic_add(counter, index, value)``, check
   that the counter is an ``int32`` array initialized for that launch.
5. For differentiable kernels, run ``Tape.backward()`` several times and compare
   the gradients that matter with ``np.testing.assert_array_equal()``.  If the
   kernel consumes a counter return, provide an explicit ``@wp.func_replay``
   slot mapping or keep that kernel out of the generated backward pass.
6. If you capture the kernel in a CUDA graph, validate both capture and replay.
   Ordinary CUDA graph capture is supported for both Pattern 1 and Pattern 2;
   Warp keeps the temporary deterministic buffers alive on the captured graph
   object. Conditional body graphs and APIC serialization still have the
   limitations described above.
7. If performance matters, benchmark both normal and deterministic modes.

Related NVIDIA Libraries
------------------------

Warp's implementation is inspired by CUDA and CUB patterns that are already
familiar in NVIDIA libraries:

* `CUB`_ is part of `NVIDIA CCCL`_ and provides optimized device-wide building
  blocks such as radix sort, reduce-by-key, and scan.
* Pattern 1 follows the same general shape as CUB sort plus reduce-by-key:
  group equal destinations, then reduce each group.
* Pattern 2 follows the standard count-scan-write pattern and uses scan
  primitives conceptually like CUB ``DeviceScan``.
* ``wp.DeterministicMode.NOT_GUARANTEED``,
  ``wp.DeterministicMode.RUN_TO_RUN``, and
  ``wp.DeterministicMode.GPU_TO_GPU`` correspond to the determinism levels
  described for CCCL reductions.

These references are useful if you want to understand the lower-level CUDA
building blocks behind Warp's higher-level switch.

.. _CUB: https://nvidia.github.io/cccl/cub/
.. _NVIDIA CCCL: https://nvidia.github.io/cccl/
.. _CUB DeviceRadixSort: https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
.. _CUB DeviceReduce: https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html
.. _CUB DeviceScan: https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceScan.html
.. _Controlling Floating-Point Determinism in NVIDIA CCCL: https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/
