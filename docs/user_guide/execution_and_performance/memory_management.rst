Memory Allocation and Access
============================

.. currentmodule:: warp

.. index:: memory management

Warp handles the lifetime of ordinary array allocations for you. This guide
explains how to choose where arrays are allocated, improve allocation
performance, and share data safely across devices.

The portable default is to allocate arrays on the device that will use them and
copy data explicitly between devices. Pinned CPU memory, CUDA managed memory,
CUDA peer access, and memory-pool access enable additional patterns when the
hardware and driver report the required capabilities.


Choosing a Memory Model
-----------------------

Default and memory-pool CUDA allocations remain device memory, pinned CPU
allocations remain host memory, and CUDA managed allocations may migrate
between CPU and GPU memory. Choose between them based on where code runs,
whether data migration or explicit copies are preferable, and whether
allocation must occur during CUDA graph capture.

.. _managed_memory_allocation_options:

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


.. _mempool_allocators:

Stream-Ordered Memory Pool Allocators
-------------------------------------

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


.. _peer_access:

CUDA Peer Access
~~~~~~~~~~~~~~~~

CUDA allows direct memory access between different GPUs if the system hardware configuration supports it.  Typically, the GPUs should be of the same type and a special interconnect may be required (e.g., NVLINK or PCIe topology).

During initialization, Warp reports whether peer access is supported on multi-GPU systems:

.. code:: text

    Warp 0.15.1 initialized:
       CUDA Toolkit 11.5, Driver 12.2
       Devices:
         "cpu"      : "x86_64"
         "cuda:0"   : "NVIDIA L40" (48 GiB, sm_89, mempool enabled)
         "cuda:1"   : "NVIDIA L40" (48 GiB, sm_89, mempool enabled)
         "cuda:2"   : "NVIDIA L40" (48 GiB, sm_89, mempool enabled)
         "cuda:3"   : "NVIDIA L40" (48 GiB, sm_89, mempool enabled)
       CUDA peer access:
         Supported fully (all-directional)

If the message reports that CUDA peer access is ``Supported fully``, it means that every CUDA device can access every other CUDA device in the system.  If it says ``Supported partially``, it will be followed by the access matrix that shows which devices can access each other.  If it says ``Not supported``, it means that access is not supported between any devices.

In code, we can check support and enable peer access like this:

.. code:: python

    if wp.is_peer_access_supported("cuda:0", "cuda:1"):
        wp.set_peer_access_enabled("cuda:0", "cuda:1", True)

This will allow the memory of device ``cuda:0`` to be directly accessed on device ``cuda:1``.  Peer access is directional, which means that enabling access to ``cuda:0`` from ``cuda:1`` does not automatically enable access to ``cuda:1`` from ``cuda:0``.

The benefit of enabling peer access is that it allows direct memory transfers (DMA) between the devices.  This is generally a faster way to copy data, since otherwise the transfer needs to be done using a CPU staging buffer.

The drawback is that enabling peer access can reduce the performance of allocations and deallocations.  Programs that don't rely on peer-to-peer memory transfers should leave this setting disabled.

It's possible to temporarily enable or disable peer access using a scoped manager:

.. code:: python

    with wp.ScopedPeerAccess("cuda:0", "cuda:1", True):
        ...

.. note::

    Peer access does not accelerate memory transfers between arrays allocated using the :ref:`stream-ordered memory pool allocators<mempool_allocators>` introduced in Warp 0.14.0.
    To accelerate memory pool transfers, :ref:`memory pool access<mempool_access>` should be enabled instead.


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

Managed Memory Allocator
~~~~~~~~~~~~~~~~~~~~~~~~

Managed memory is CUDA-managed storage that can be addressed from CPU and GPU
code. CUDA Unified Memory manages page placement and migration, so pages may move
between CPU and GPU memory as different processors touch them. Unlike pinned CPU
memory, which remains host memory that a GPU may access through a host mapping,
managed memory gives Warp arrays a different tradeoff from the other allocation
options summarized in :ref:`the allocation comparison <managed_memory_allocation_options>`.

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


Cross-Device Memory Access
--------------------------

Warp arrays are associated with an allocation :class:`Device` such as
``"cpu"`` or ``"cuda:0"``, and kernels run on a launch device. The portable
default is to launch kernels on the same device as their array arguments. On
systems with hardware-supported CPU/GPU memory access, some cross-device
patterns can also be valid: a GPU may be able to read or write unpinned CPU
memory directly, and some systems can let CPU code directly access GPU-resident
CUDA managed memory.


Launch Device Versus Allocation Device
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The launch device determines where a kernel runs, and the array device describes
where the array allocation lives:

.. code:: python

    cpu_array = wp.zeros(1024, dtype=float, device="cpu")
    gpu_array = wp.zeros(1024, dtype=float, device="cuda:0")

    wp.launch(kernel, dim=cpu_array.size, inputs=[cpu_array], device="cpu")
    wp.launch(kernel, dim=gpu_array.size, inputs=[gpu_array], device="cuda:0")

The same-device pattern works on all supported systems. Passing an array from
one device to a kernel running on another device depends on the capabilities of
the device that performs the access.


Device Capability Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each device exposes CPU/GPU memory access and managed-memory properties:

- :attr:`Device.is_cpu_memory_access_from_gpu_supported <warp.Device.is_cpu_memory_access_from_gpu_supported>`
- :attr:`Device.is_gpu_memory_access_from_cpu_supported <warp.Device.is_gpu_memory_access_from_cpu_supported>`
- :attr:`Device.is_managed_memory_supported <warp.Device.is_managed_memory_supported>`
- :attr:`Device.is_concurrent_managed_access_supported <warp.Device.is_concurrent_managed_access_supported>`

On CPU devices, these properties are always ``False``. On GPU devices, each
property describes a specific access path or allocation feature; support for one
does not imply support for another. For example, a system can allow GPU access
to CPU memory without allowing CPU access to GPU-resident managed memory.


Common CPU/GPU Memory Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The exact values are reported by the CUDA driver and may vary by platform,
driver, kernel, and GPU generation. The following table summarizes the models
advanced users commonly need to reason about:

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - System model
     - GPU access to CPU arrays
     - CPU access to GPU-resident managed memory
   * - Discrete GPU without HMM
     - Usually no
     - Usually no
   * - Discrete GPU with Linux HMM
     - Yes
     - Usually no
   * - Jetson Thor-style ATS
     - Yes
     - Platform-dependent for managed memory
   * - Host-page-table ATS with distinct CPU/GPU physical memory
     - Yes
     - Only when reported by the driver

HMM stands for Heterogeneous Memory Management; for background, see NVIDIA's
`HMM overview <https://developer.nvidia.com/blog/simplifying-gpu-application-development-with-heterogeneous-memory-management/>`__.
ATS stands for Address Translation Services. Warp does not require users to
classify the platform manually. Query the :class:`Device` properties and branch
on the behavior your program needs.

For the CUDA-level model behind these categories, see the CUDA Programming
Guide's `Unified and System Memory chapter
<https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html>`__,
especially its `Unified Memory paradigms table
<https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#table-unified-memory-levels>`__.

Do not infer CPU access to GPU-resident CUDA managed memory from ATS, C2C, or a
product family name. For example, a DGX Spark-class GB10 system can report ATS
and GPU access to CPU memory while
``device.is_gpu_memory_access_from_cpu_supported`` is ``False``. Query the
property directly before CPU code reads or writes GPU-resident managed memory.


GPU Kernels Using CPU Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``device.is_cpu_memory_access_from_gpu_supported`` is true, a GPU kernel can
directly read or write a CPU array:

.. code:: python

    device = wp.get_device("cuda:0")
    a = wp.zeros(1024, dtype=float, device="cpu")

    if device.is_cpu_memory_access_from_gpu_supported:
        wp.launch(kernel, dim=a.size, inputs=[a], device=device)
    else:
        a_gpu = a.to(device)
        wp.launch(kernel, dim=a_gpu.size, inputs=[a_gpu], device=device)

This can avoid explicit copies on HMM and coherent CPU/GPU systems. If the
capability is false and the kernel actually dereferences the CPU pointer, CUDA
will report a runtime error such as an illegal memory access.


CPU Code Using GPU-Resident Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CPU access to GPU-resident managed memory is a separate capability:

.. code:: python

    device = wp.get_device("cuda:0")
    if device.is_gpu_memory_access_from_cpu_supported:
        ...

CUDA arrays created by standard Warp array constructors, such as
:func:`zeros`, :func:`empty`, and :func:`ones`, are not CUDA managed-memory
allocations. This is true whether the array comes from Warp's :ref:`mempool
allocator <mempool_allocators>` or the built-in default CUDA allocator. For
those arrays, use an explicit copy before CPU code reads or writes the data:

.. code:: python

    a = wp.zeros(1024, dtype=float, device=device)
    a_cpu = a.to("cpu")
    wp.launch(cpu_kernel, dim=a_cpu.size, inputs=[a_cpu], device="cpu")

For explicit CUDA managed-memory arrays, use the allocator construction in
`Managed Memory Allocator`_. The allocator instance is not bound to one CUDA
device, but each allocation still happens under the target device's CUDA context
and that device must report CUDA managed-memory support. See :ref:`the allocation
comparison <managed_memory_allocation_options>` for how managed memory differs
from Warp's default CUDA, CUDA mempool, and pinned CPU allocation options.

Managed arrays remain CUDA arrays in Warp: ``a.device`` is still ``"cuda:0"``,
and CUDA Unified Memory manages physical page migration. This is the opt-in Warp
allocation path for CPU kernels that need to operate directly on CUDA-side data
without maintaining a separate CPU copy. Ordinary Warp CUDA arrays remain
non-managed allocations and still need explicit copies before CPU kernels read
or write them. The `Managed Memory Allocator`_ section also describes the single
canonical graph-capture limitation for managed allocations.

Do not assume that GPU access to CPU memory implies CPU access to GPU-resident
memory. Some systems support the former but not the latter.


Synchronization and Ownership
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CPU and GPU work may execute asynchronously, so synchronize the producer before
code on another device reads or writes the same allocation. When CPU and GPU
code both update an allocation, sequence ownership explicitly or use separate
buffers. Warp atomics do not make overlapping CPU/GPU updates safe.


Validating Memory Access
------------------------

Checking a Concrete Array with ``wp.can_access()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function :func:`warp.can_access` answers whether code running on one device
can directly access a specific Warp array:

.. code:: python

    launch_device = wp.get_device("cuda:0")
    data = wp.empty(1024, dtype=float, device="cpu")

    if wp.can_access(launch_device, data):
        ...

For CPU arrays passed to CUDA kernels, pinned CPU arrays are accepted on CUDA
devices with unified virtual addressing, and unpinned CPU arrays require
``is_cpu_memory_access_from_gpu_supported``. For CUDA arrays, managed-memory
allocations use CUDA managed-memory support on the launch device, default CUDA
allocations use CUDA peer-access state, and memory pool allocations use
memory-pool access state. See :ref:`mempool_access` for the distinction between
peer access for default CUDA allocations and memory-pool access for mempool
allocations.

CPU access to managed arrays is hardware-dependent. Check a specific managed
array before CPU code reads or writes it directly:

.. code:: python

    if wp.can_access("cpu", a):
        wp.launch(cpu_kernel, dim=a.size, inputs=[a], device="cpu")
    else:
        a_cpu = a.to("cpu")
        wp.launch(cpu_kernel, dim=a_cpu.size, inputs=[a_cpu], device="cpu")

``wp.can_access("cpu", a)`` returns ``True`` for a managed CUDA array only when
the owning CUDA device reports concurrent managed access or direct CPU access to
GPU memory. On limited managed-memory systems, Warp returns ``False`` because it
cannot prove that a direct CPU access is synchronized with GPU use.

Use :attr:`array.memory_kind <warp.array.memory_kind>` to inspect the observed
memory class backing a concrete :class:`warp.array`:

.. code:: python

    if a.memory_kind is wp.MemoryKind.CUDA_MANAGED:
        ...

The memory kind describes the pointer's memory class as reported by Warp and is
an input to Warp's access checks. It does not describe the current physical
residency of CUDA managed memory, and views report the memory kind of their
owner array. Indexed arrays do not expose a single memory kind because their
data and index arrays may have different backing allocations; inspect those
constituent arrays directly when diagnostics are needed.

``wp.can_access(device, array)`` returns ``False`` when Warp cannot verify that
the array's memory access requirements are satisfied. This includes
unclassified pointers and classified pointers whose allocation-specific access
cannot be proven. A ``False`` result means "not verified accessible"; it does
not prove that the hardware could never access the pointer.

``wp.can_access()`` is a resource-oriented API. In this release, the second
argument must be a concrete Warp array instance. Annotation-only arrays such as
``wp.array[float]`` and device objects are not supported.


Checking Coarse Device Access with ``Device.can_access()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The method :meth:`Device.can_access` is a coarse device-level query for cases
where no concrete array is available:

.. code:: python

    launch_device = wp.get_device("cuda:0")
    array_device = wp.get_device("cpu")

    if launch_device.can_access(array_device):
        ...

For GPU kernels accessing CPU arrays, this method uses
``is_cpu_memory_access_from_gpu_supported`` because standard Warp CPU arrays use
unpinned CPU memory. For CPU code accessing CUDA arrays, it returns ``False`` for
standard Warp CUDA arrays because the built-in CUDA allocators do not create
CUDA managed-memory allocations. For GPU/GPU pairs, it reflects the target
device's current built-in allocator mode: memory-pool access when memory pools
are enabled on the target device, and peer access otherwise.

``Device.can_access()`` is not authoritative for existing arrays. An array may
have been allocated before memory-pool settings changed, may use a custom
allocator, or may wrap external memory. Code that has an actual array should use
``wp.can_access(device, array)`` instead.


.. _launch_array_access_checks:

Checked Launch Validation
~~~~~~~~~~~~~~~~~~~~~~~~~

By default, Warp launches kernels after type, dtype, and dimension validation
without checking array accessibility. This keeps the launch path lightweight and
allows hardware-supported mixed CPU/GPU launches to work.

If you want a clear Python error before the kernel runs, set
:attr:`warp.config.launch_array_access_mode`:

.. code:: python

    wp.config.launch_array_access_mode = wp.config.LaunchArrayAccessMode.CHECKED

- ``wp.config.LaunchArrayAccessMode.RELAXED`` is the default and performs no pre-launch
  array access checks beyond type, dtype, and dimension validation.
- ``wp.config.LaunchArrayAccessMode.STRICT`` restores Warp's original same-device rule
  and requires every Warp array argument to be allocated on the launch device.
- ``wp.config.LaunchArrayAccessMode.CHECKED`` raises an error before launch when Warp can
  determine that a cross-device Warp array argument is not accessible from the
  launch device. This is useful when debugging mixed-device launches on systems
  that do not support direct CPU/GPU memory access or on multi-GPU systems where
  peer and memory-pool access are configured separately.

Checked validation distinguishes three outcomes for pointer access. Verified
accessible pointers proceed normally. Known inaccessible pointers raise an
error before launch. Unverified pointers, whose memory kind or required access
state Warp cannot prove, emit a ``UserWarning`` and proceed so valid custom or
externally managed access patterns are not rejected.

Custom allocators and external wrappers are a limitation of this diagnostic
only when Warp cannot classify the pointer or cannot prove the specific access
requirements, such as for unowned CUDA memory-pool pointers. In those cases,
``wp.config.LaunchArrayAccessMode.CHECKED`` emits a ``UserWarning`` once per
``(kernel, argument name, source device, launch device)`` pattern and allows
the launch to proceed. Use ``wp.config.LaunchArrayAccessMode.STRICT`` if
unverified cross-device access should be rejected, or
``wp.config.LaunchArrayAccessMode.RELAXED`` to suppress the diagnostic.

Objects exposing ``__array_interface__`` are accepted only for CPU launches.
Warp treats that protocol as a CPU-addressable pointer and does not infer CUDA
memory kind from it, so ``wp.config.LaunchArrayAccessMode.CHECKED`` has no
cross-device access decision to make for that protocol.

Directly passing an object that exposes ``__cuda_array_interface__`` is
different from passing a Warp array. The protocol lets Warp construct the kernel
argument at launch time, but it does not identify the allocation device or
allocation-specific access state. In this phase,
``wp.config.LaunchArrayAccessMode.CHECKED`` does not fully verify directly
passed objects exposing this protocol. Advanced users who know such an
allocation is valid are responsible for ensuring that the launch device can
legally access the pointer.

.. code:: python

    wp.config.launch_array_access_mode = wp.config.LaunchArrayAccessMode.CHECKED
    wp.launch(kernel, dim=a.size, inputs=[a], device="cuda:0")

:attr:`warp.config.launch_array_access_mode` can add launch overhead in
``wp.config.LaunchArrayAccessMode.STRICT`` and
``wp.config.LaunchArrayAccessMode.CHECKED`` modes. Use
``wp.config.LaunchArrayAccessMode.RELAXED`` in performance-sensitive code that
has already validated its launch accessibility assumptions.

Unlike :attr:`warp.config.verify_cuda`,
:attr:`warp.config.launch_array_access_mode` can be used during CUDA graph
capture because ``wp.config.LaunchArrayAccessMode.CHECKED`` checks run before
each launch is recorded. For cross-GPU graph capture, enable peer access or
memory-pool access with Warp APIs before capture begins so verification can use
the recorded access state during capture. When a CUDA graph captures a launch
with CPU array arguments, replay uses the same captured CPU pointers. If the
arrays remain alive, CPU updates made between replays are visible to kernels on
devices that can access CPU memory.


Choosing a Memory Access Pattern
--------------------------------

Use the same-device pattern unless you need zero-copy CPU/GPU access. When you
already have an array, use :func:`wp.can_access(device, array) <warp.can_access>`
to decide whether a specific launch device can directly access that allocation.
Capability flags are most useful before allocation, when deciding what kind of
allocation or access pattern to create:

- GPU kernel reads or writes unpinned CPU arrays: check
  ``device.is_cpu_memory_access_from_gpu_supported``.
- GPU kernel reads or writes pinned CPU arrays: use ``pinned=True`` and check
  ``device.is_uva``.
- CPU code reads or writes arrays backed by non-managed CUDA allocations: copy
  the data to ``"cpu"`` first.
- CPU kernels read or write Warp CUDA arrays directly: allocate those arrays
  with :class:`CudaManagedAllocator` as described in `Managed Memory Allocator`_,
  then use ``wp.can_access("cpu", array)`` before launching the CPU kernel.
- CPU code accesses externally provided GPU-resident CUDA managed memory: check
  ``device.is_gpu_memory_access_from_cpu_supported``.
- CPU and GPU both need to update the same allocation: follow
  `Synchronization and Ownership`_ to sequence access or use separate buffers.
- GPU kernels use arrays from another GPU: enable :ref:`peer access <peer_access>`
  for default CUDA allocations or :ref:`memory-pool access <mempool_access>` for
  CUDA memory-pool allocations, then check the concrete array with
  :func:`wp.can_access(device, array) <warp.can_access>`.
- Debugging mixed-device launch failures: temporarily set
  :attr:`warp.config.launch_array_access_mode` to
  ``wp.config.LaunchArrayAccessMode.CHECKED``.

Prefer capability checks over platform-name checks. They make code portable
across discrete GPUs, HMM-enabled systems, Jetson, Grace, and future coherent
CPU/GPU platforms.
