Allocators
==========

.. currentmodule:: warp

.. _mempool_allocators:

Stream-Ordered Memory Pool Allocators
-------------------------------------

Introduction
~~~~~~~~~~~~

Warp 0.14.0 added support for `stream-ordered memory pool allocators for CUDA arrays <https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1>`_.  As of Warp 0.15.0, these allocators are enabled by default on
all CUDA devices that support them.  "Stream-ordered memory pool allocator" is quite a mouthful, so let's unpack it one bit at a time.

Whenever you create an array, the memory needs to be allocated on the device:

.. code:: python

    a = wp.empty(n, dtype=float, device="cuda:0")
    b = wp.zeros(n, dtype=float, device="cuda:0")
    c = wp.ones(n, dtype=float, device="cuda:0")
    d = wp.full(n, 42.0, dtype=float, device="cuda:0")

Each of the calls above allocates a block of device memory large enough to hold the array and optionally initializes the contents with
the specified values.
:func:`wp.empty() <warp.empty>` is the only function that does not initialize the contents in any way, it just allocates the memory.

Memory pool allocators grab a block of memory from a larger pool of reserved memory, which is generally faster than asking
the operating system for a brand new chunk of storage.  This is an important benefit of these pooled allocators—they are faster.

Stream-ordered means that each allocation is scheduled on a :ref:`CUDA stream<streams>`, which represents a sequence of instructions that execute in order on the GPU.  The main benefit is that it allows memory to be allocated in CUDA graphs, which was previously not possible:

.. code:: python

    with wp.ScopedCapture() as capture:
        a = wp.zeros(n, dtype=float)
        wp.launch(kernel, dim=a.size, inputs=[a])

    wp.capture_launch(capture.graph)

From now on, we will refer to these allocators as *mempool allocators* for short.


Configuration
~~~~~~~~~~~~~

Mempool allocators are a feature of CUDA that is supported on most modern devices and operating systems.  However,
there can be systems where they are not supported, such as certain virtual machine setups.  Warp is designed with resiliency in mind,
so existing code written prior to the introduction of these new allocators should continue to function regardless of whether they
are supported by the underlying system or not.

Warp's startup message gives the status of these allocators, for example:

.. code-block:: text

    Warp 0.15.1 initialized:
    CUDA Toolkit 11.5, Driver 12.2
    Devices:
        "cpu"      : "x86_64"
        "cuda:0"   : "NVIDIA GeForce RTX 4090" (24 GiB, sm_89, mempool enabled)
        "cuda:1"   : "NVIDIA GeForce RTX 3090" (24 GiB, sm_86, mempool enabled)

Note the ``mempool enabled`` text next to each CUDA device.  This means that memory pools are enabled on the device.  Whenever you create
an array on that device, it will be allocated using the mempool allocator.  If you see ``mempool supported``, it means that memory
pools are supported but were not enabled on startup.  If you see ``mempool not supported``, it means that memory pools can't be used
on this device.

There is a configuration flag that controls whether memory pools should be automatically enabled during :func:`wp.init() <warp.init>`:

.. code:: python

    import warp as wp

    wp.config.enable_mempools_at_init = False

    wp.init()

The flag defaults to ``True``, but can be set to ``False`` if desired.  Changing this configuration flag after :func:`wp.init() <warp.init>` is called has no effect.

After :func:`wp.init() <warp.init>`, you can check if the memory pool is enabled on each device like this:

.. code:: python

    if wp.is_mempool_enabled("cuda:0"):
        ...

You can also independently control enablement on each device:

.. code:: python

    if wp.is_mempool_supported("cuda:0"):
        wp.set_mempool_enabled("cuda:0", True)

It's possible to temporarily enable or disable memory pools using a scoped manager:

.. code:: python

    with wp.ScopedMempool("cuda:0", True):
        a = wp.zeros(n, dtype=float, device="cuda:0")

    with wp.ScopedMempool("cuda:0", False):
        b = wp.zeros(n, dtype=float, device="cuda:0")

In the snippet above, array ``a`` will be allocated using the mempool allocator and array ``b`` will be allocated using the default allocator.

In most cases, it shouldn't be necessary to fiddle with these enablement functions, but they are there if you need them.
By default, Warp will enable memory pools on startup if they are supported, which will bring the benefits of improved allocation speed automatically.
Most Warp code should continue to function with or without mempool allocators, with the exception of memory allocations
during CUDA graph capture, which will raise an exception if memory pools are not enabled. CPU APIC graph capture does
not use CUDA-style memory pools; host allocations made during CPU capture are retained for the captured graph's
lifetime and reused on replay.


Querying Memory Usage
~~~~~~~~~~~~~~~~~~~~~

The amount of memory the application is currently using from a specific memory
pool can be queried using :func:`wp.get_mempool_used_mem_current() <warp.get_mempool_used_mem_current>`.
This can be different from the amount of memory reserved for the pool itself.
Similarly, the high-water mark of used memory can be queried using
:func:`wp.get_mempool_used_mem_high() <warp.get_mempool_used_mem_high>`.


Allocation Performance
~~~~~~~~~~~~~~~~~~~~~~

Allocating and releasing memory are rather expensive operations that can add overhead to a program.  We can't avoid them, since we need to allocate storage for our data somewhere, but there are some simple strategies that can reduce the overall impact of allocations on performance.

Consider the following example:

.. code:: python

    for i in range(100):
        a = wp.zeros(n, dtype=float, device="cuda:0")
        wp.launch(kernel, dim=a.size, inputs=[a], device="cuda:0")

On each iteration of the loop, we allocate an array and run a kernel on the data.  This program has 100 allocations and 100 deallocations.  When we assign a new value to ``a``, the previous value gets garbage collected by Python, which triggers the deallocation.

Reusing Memory
^^^^^^^^^^^^^^

If the size of the array remains fixed, consider reusing the memory on subsequent iterations.  We can allocate the array only once and just re-initialize its contents on each iteration:

.. code:: python

    # pre-allocate the array
    a = wp.empty(n, dtype=float, device="cuda:0")
    for i in range(100):
        # reset the contents
        a.zero_()
        wp.launch(kernel, dim=a.size, inputs=[a], device="cuda:0")

This works well if the array size does not change on each iteration.  If the size changes but the upper bound is known, we can still pre-allocate a buffer large enough to store all the elements at any iteration.

.. code:: python

    # pre-allocate a big enough buffer
    buffer = wp.empty(MAX_N, dtype=float, device="cuda:0")
    for i in range(100):
        # get a buffer slice of size n <= MAX_N
        n = get_size(i)
        a = buffer[:n]
        # reset the contents
        a.zero_()
        wp.launch(kernel, dim=a.size, inputs=[a], device="cuda:0")

Reusing memory this way can improve performance, but may also add undesirable complexity to our code.  The mempool allocators have a useful feature that can improve allocation performance without modifying our original code in any way.

Release Threshold
^^^^^^^^^^^^^^^^^

The memory pool release threshold determines how much reserved memory the allocator should hold on to before releasing it back to the operating system.  For programs that frequently allocate and release memory, setting a higher release threshold can improve the performance of allocations.

By default, the release threshold is set to 0.  Setting it to a higher number will reduce the cost of allocations if memory was previously acquired and returned to the pool.

.. code:: python

    # set the release threshold to reduce re-allocation overhead
    wp.set_mempool_release_threshold("cuda:0", 1024**3)

    for i in range(100):
        a = wp.zeros(n, dtype=float, device="cuda:0")
        wp.launch(kernel, dim=a.size, inputs=[a], device="cuda:0")

Threshold values between 0 and 1 are interpreted as fractions of available memory.  For example, 0.5 means half of the device's physical memory and 1.0 means all of the memory.  Greater values are interpreted as an absolute number of bytes.  For example, 1024**3 means one GiB of memory.

This is a simple optimization that can improve the performance of programs without modifying the existing code in any way.


Graph Allocations
~~~~~~~~~~~~~~~~~

Mempool allocators can be used in CUDA graphs, which means that you can capture Warp code that creates arrays:

.. code:: python

    with wp.ScopedCapture() as capture:
        a = wp.full(n, 42, dtype=float)

    wp.capture_launch(capture.graph)

    print(a)

Capturing allocations in CUDA graphs is similar to capturing other operations like kernel launches or memory copies.  During capture, the operations don't actually execute, but are recorded.  To execute the captured operations, we must launch the graph using :func:`wp.capture_launch() <warp.capture_launch>`.  This is important to keep in mind if you want to use an array that was allocated during CUDA graph capture.  The array doesn't actually exist until the captured graph is launched.  In the snippet above, we would get an error if we tried to print the array before calling :func:`wp.capture_launch() <warp.capture_launch>`.

CPU APIC graph capture handles allocations differently: CPU arrays allocated
inside the capture are allocated immediately, retained for the lifetime of the
captured graph, and reused on every replay. This allows temporary CPU buffers,
including FEM temporary storage, to participate in CPU graph capture without a
CUDA-style memory pool.

More generally, the ability to allocate memory during graph capture greatly increases the range of code that can be captured in a graph.  This includes any code that creates temporary allocations.  CUDA graphs can be used to re-run operations with minimal CPU overhead, which can yield dramatic performance improvements.

.. _mempool_access:

Memory Pool Access
~~~~~~~~~~~~~~~~~~

On multi-GPU systems that support :ref:`peer access<peer_access>`, we can enable directly accessing a memory pool from a different device:

.. code:: python

    if wp.is_mempool_access_supported("cuda:0", "cuda:1"):
        wp.set_mempool_access_enabled("cuda:0", "cuda:1", True)

This will allow the memory pool of device ``cuda:0`` to be directly accessed on device ``cuda:1``.  Memory pool access is directional, which means that enabling access to ``cuda:0`` from ``cuda:1`` does not automatically enable access to ``cuda:1`` from ``cuda:0``.

The benefit of enabling memory pool access is that it allows direct memory transfers (DMA) between the devices.  This is generally a faster way to copy data, since otherwise the transfer needs to be done using a CPU staging buffer.

The drawback is that enabling memory pool access can slightly reduce the performance of allocations and deallocations.  However, for applications that rely on copying memory between devices, there should be a net benefit.

It's possible to temporarily enable or disable memory pool access using a scoped manager:

.. code:: python

    with wp.ScopedMempoolAccess("cuda:0", "cuda:1", True):
        a0 = wp.zeros(n, dtype=float, device="cuda:0")
        a1 = wp.empty(n, dtype=float, device="cuda:1")

        # use direct memory transfer between GPUs
        wp.copy(a1, a0)

Note that memory pool access only applies to memory allocated using mempool allocators.
For memory allocated using default CUDA allocators, we can enable CUDA :ref:`peer access<peer_access>` using :func:`wp.set_peer_access_enabled() <warp.set_peer_access_enabled>` to get similar benefits.

Because enabling memory pool access can have drawbacks, Warp does not automatically enable it, even if it's supported.  Programs that don't require copying data between GPUs are therefore not affected in any way.


Limitations
~~~~~~~~~~~

Mempool-to-Mempool Copies Between GPUs During Graph Capture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Copying data between different GPUs will fail during graph capture if the source and destination are allocated using mempool allocators and mempool access is not enabled between devices.  Note that this only applies to capturing mempool-to-mempool copies in a graph.  Copies done outside of graph capture are not affected.  Copies within the same mempool (i.e., same device) are also not affected.

There are two workarounds.  If mempool access is supported, you can simply enable mempool access between the devices prior to graph capture, as shown in :ref:`mempool_access`.

If mempool access is not supported, you will need to pre-allocate the arrays involved in the copy using the default CUDA allocators.  This will need to be done before capture begins:

.. code:: python

    # pre-allocate the arrays with mempools disabled
    with wp.ScopedMempool("cuda:0", False):
        a0 = wp.zeros(n, dtype=float, device="cuda:0")
    with wp.ScopedMempool("cuda:1", False):
        a1 = wp.empty(n, dtype=float, device="cuda:1")

    with wp.ScopedCapture("cuda:1") as capture:
        wp.copy(a1, a0)

    wp.capture_launch(capture.graph)

This is due to a limitation in CUDA, which we envision being fixed in the future.


.. _custom_allocators:

Custom Allocators
-----------------

Warp supports pluggable memory allocators for CUDA devices. You can redirect all
GPU array allocations through a custom allocator by implementing the
:class:`warp.Allocator` protocol, i.e., any object with ``allocate(size_in_bytes)``
and ``deallocate(ptr, size_in_bytes)`` methods. Custom allocators only affect
:class:`warp.array` allocations on CUDA devices; CPU allocations, pinned memory,
and internal native allocations (e.g., BVH construction temporaries) are not
affected.

Setting a Custom Allocator
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :func:`set_cuda_allocator` to set a custom allocator on all CUDA devices, or
:func:`set_device_allocator` for a specific device:

.. code:: python

    wp.set_cuda_allocator(my_allocator)           # all CUDA devices
    wp.set_device_allocator("cuda:0", my_allocator)  # one device

Pass ``None`` to restore the built-in allocator:

.. code:: python

    wp.set_cuda_allocator(None)

Use :func:`get_device_allocator` to query the current allocator:

.. code:: python

    allocator = wp.get_device_allocator("cuda:0")

For temporary allocator changes, use the :class:`ScopedAllocator` context manager:

.. code:: python

    with wp.ScopedAllocator("cuda:0", my_allocator):
        a = wp.zeros(1000, dtype=wp.float32, device="cuda:0")
    # Original allocator is restored here

.. _managed_memory_allocation_options:

Managed Memory Allocator
~~~~~~~~~~~~~~~~~~~~~~~~

Managed memory is CUDA-managed storage that can be addressed from CPU and GPU
code. CUDA Unified Memory manages page placement and migration, so pages may move
between CPU and GPU memory as different processors touch them. Unlike pinned CPU
memory, which remains host memory that a GPU may access through a host mapping,
managed memory gives Warp arrays a different tradeoff from the other allocation
options:

.. list-table::
   :header-rows: 1
   :widths: 18 29 27 26

   * - Allocation option
     - Residency and migration
     - CPU/GPU access
     - Typical use
   * - Default CUDA
     - Device memory with no automatic CPU/GPU migration.
     - CUDA kernels access it directly; CPU code uses explicit copies.
     - General GPU arrays when CPU access is staged explicitly.
   * - CUDA mempool
     - Device memory from CUDA's stream-ordered pool, with no automatic CPU/GPU
       migration.
     - Same CPU/GPU access rules as default CUDA memory, with separate
       memory-pool access controls for peer GPUs.
     - Faster repeated CUDA allocations and graph-captured allocation when
       supported.
   * - Pinned CPU
     - Host memory that does not migrate into device memory as an allocation.
     - CPU code accesses it directly; CUDA devices with unified virtual
       addressing can access it through a host mapping.
     - Asynchronous CPU/GPU copies or zero-copy access to small host-resident
       data.
   * - CUDA managed
     - CUDA Unified Memory whose pages may migrate between CPU and GPU memory.
     - CPU and GPU access follow CUDA managed-memory support and synchronization
       rules.
     - Sharing data across CPU/GPU code when migration is preferable to manual
       copies.

:class:`CudaManagedAllocator` creates CUDA managed-memory arrays through Warp's
allocator interface. Managed arrays keep their CUDA device metadata, but
``wp.can_access()`` and checked launch validation use CUDA managed-memory access
rules for them instead of peer-access or memory-pool-access rules.

One major reason to choose this allocator is CPU/GPU shared work: on systems
where CUDA reports compatible managed-memory access, CPU kernels can directly
read and write managed CUDA arrays instead of maintaining a separate CPU copy.
Standard Warp CUDA arrays remain non-managed and still require explicit copies
before CPU code accesses them.

The allocator object is not bound to one CUDA device and can be constructed
before choosing a CUDA device. Warp invokes it under the target device's CUDA
context, which must support CUDA managed memory:

.. code:: python

    managed = wp.CudaManagedAllocator()
    device = wp.get_device("cuda:0")

    with wp.ScopedAllocator(device, managed):
        a = wp.zeros(1000, dtype=wp.float32, device=device)

Constructing a :class:`CudaManagedAllocator` does not promise that pages initially
reside in any device's physical memory, and it does not bypass the device's
managed-memory capability check. The CUDA device used for each allocation
identifies the array device metadata; CUDA Unified Memory manages physical
placement and migration.

Use :attr:`array.memory_kind <warp.array.memory_kind>` to inspect the observed
memory class backing a concrete :class:`warp.array`:

.. code:: python

    if a.memory_kind is wp.MemoryKind.CUDA_MANAGED:
        ...

The memory kind describes the pointer's memory class as reported by Warp. It
does not describe the current physical residency of CUDA managed memory, and
views report the memory kind of their owner array. Indexed arrays do not expose
a single memory kind because their data and index arrays may have different
backing allocations.

To use managed memory as a persistent allocator for all CUDA devices, install one
allocator instance with :func:`set_cuda_allocator`:

.. code:: python

    managed = wp.CudaManagedAllocator()
    wp.set_cuda_allocator(managed)

If only some CUDA devices should use managed memory, install the same allocator
with :func:`set_device_allocator` on those devices. A single allocator instance
can serve multiple CUDA devices, but allocation fails clearly on any target
device that does not report CUDA managed-memory support.

Direct calls to ``CudaManagedAllocator.allocate()`` require an active CUDA context.
Array factory functions such as :func:`zeros` and :func:`empty` pass the target
device context automatically and perform the same managed-memory support check.

CUDA may reject managed allocations during graph capture because
:class:`CudaManagedAllocator` uses ``cudaMallocManaged()``. If you need managed
arrays with CUDA graphs, allocate them before capture begins and reuse the
existing arrays inside the captured work. This is not a restriction on using
pre-existing managed arrays in captured work. Separately,
:class:`CudaManagedAllocator`-managed arrays cannot be exported with
``array.ipc_handle()``; IPC export is unsupported for managed arrays. If IPC is
required, choose a different allocator for shared data or pre-allocate and
export device arrays before switching allocator state.

CPU access to managed arrays is hardware-dependent. Use :func:`can_access` to
check a specific managed array before CPU code reads or writes it directly:

.. code:: python

    if wp.can_access("cpu", a):
        wp.launch(cpu_kernel, dim=a.size, inputs=[a], device="cpu")
    else:
        a_cpu = a.to("cpu")
        wp.launch(cpu_kernel, dim=a_cpu.size, inputs=[a_cpu], device="cpu")

Writing a Custom Allocator
~~~~~~~~~~~~~~~~~~~~~~~~~~

A custom allocator is any object that implements ``allocate`` and ``deallocate``:

.. code:: python

    class MyAllocator:
        def allocate(self, size_in_bytes: int) -> int:
            # Return a device pointer (int)
            ...

        def deallocate(self, ptr: int, size_in_bytes: int) -> None:
            # Free the device pointer
            ...

Warp enters the array's CUDA context before calling ``deallocate()`` by default.
If an allocator manages the current context itself, set
``deallocate_requires_context_guard = False`` on the allocator object.

Allocators that do not support stream-ordered allocation may not work correctly
during CUDA graph capture.

See ``warp/examples/core/example_custom_allocator.py`` for a complete example.

.. _pytorch-cuda-caching-allocator:

PyTorch CUDA Caching Allocator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch exposes low-level CUDA caching allocator functions that can be used by
other frameworks. If an application wants Warp CUDA arrays to be allocated from
PyTorch's cache, implement a small custom allocator that calls
``torch.cuda.caching_allocator_alloc()`` and releases pointers with
``torch.cuda.caching_allocator_delete()``:

.. code:: python

    import torch
    import warp as wp


    class TorchCachingAllocator:
        """Route Warp CUDA array allocations through PyTorch."""

        def __init__(self):
            self._active_allocations = {}

        @staticmethod
        def _current_warp_device_and_stream():
            device = wp.get_cuda_device()
            stream = device.stream.cuda_stream
            return device.ordinal, int(stream) if stream is not None else 0

        def allocate(self, size_in_bytes: int) -> int:
            if size_in_bytes == 0:
                return 0

            device, stream = self._current_warp_device_and_stream()
            ptr = torch.cuda.caching_allocator_alloc(size_in_bytes, device=device, stream=stream)
            ptr = int(ptr)
            self._active_allocations[ptr] = size_in_bytes
            return ptr

        def deallocate(self, ptr: int, size_in_bytes: int) -> None:
            if ptr == 0:
                return

            allocated_size = self._active_allocations.get(ptr)
            if allocated_size is None:
                raise RuntimeError(f"Unrecognized allocation pointer {ptr:#x}")
            if allocated_size != size_in_bytes:
                raise RuntimeError(
                    f"Allocation size mismatch for pointer {ptr:#x}: "
                    f"allocated {allocated_size}, deallocating {size_in_bytes}"
                )

            del self._active_allocations[ptr]
            torch.cuda.caching_allocator_delete(ptr)


    allocator = TorchCachingAllocator()
    wp.set_cuda_allocator(allocator)
    try:
        a = wp.zeros(1000, dtype=wp.float32, device="cuda:0")
    finally:
        wp.set_cuda_allocator(None)

PyTorch tracks the device and stream for pointers returned by
``caching_allocator_alloc()``, so ``caching_allocator_delete()`` only needs the
pointer. The ``_active_allocations`` dictionary above is for validation and
debugging. Applications can customize this tracking for their own accounting,
thread-safety, or distributed runtime needs.

RAPIDS Memory Manager (RMM) Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`RAPIDS Memory Manager (RMM) <https://github.com/rapidsai/rmm>`_ provides high-performance
pooled allocators for CUDA. Warp includes a built-in adapter, :class:`~warp.utils.AllocatorRmm`, that
routes array allocations through RMM.

Install RMM (Linux only):

.. code:: bash

    pip install rmm-cu12

Set up a shared RMM pool for PyTorch and Warp:

.. code:: python

    import rmm
    rmm.reinitialize(pool_allocator=True, initial_pool_size=2**30)

    # Route PyTorch through RMM
    import torch
    from rmm.allocators.torch import rmm_torch_allocator
    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

    # Route Warp through RMM
    import warp as wp
    wp.set_cuda_allocator(wp.utils.AllocatorRmm())

    # Now both frameworks share the same memory pool
    a = wp.zeros(1000, dtype=wp.float32, device="cuda:0")
