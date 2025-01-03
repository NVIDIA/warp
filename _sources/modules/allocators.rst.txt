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
:func:`wp.empty() <empty>` is the only function that does not initialize the contents in any way, it just allocates the memory.

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

There is a configuration flag that controls whether memory pools should be automatically enabled during ``wp.init()``:

.. code:: python

    import warp as wp

    wp.config.enable_mempools_at_init = False

    wp.init()

The flag defaults to ``True``, but can be set to ``False`` if desired.  Changing this configuration flag after ``wp.init()`` is called has no effect.

After ``wp.init()``, you can check if the memory pool is enabled on each device like this:

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
during graph capture, which will raise an exception if memory pools are not enabled.

.. autofunction:: warp.is_mempool_supported
.. autofunction:: warp.is_mempool_enabled
.. autofunction:: warp.set_mempool_enabled


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

.. autofunction:: warp.get_mempool_release_threshold
.. autofunction:: warp.set_mempool_release_threshold

Graph Allocations
~~~~~~~~~~~~~~~~~

Mempool allocators can be used in CUDA graphs, which means that you can capture Warp code that creates arrays:

.. code:: python

    with wp.ScopedCapture() as capture:
        a = wp.full(n, 42, dtype=float)

    wp.capture_launch(capture.graph)

    print(a)

Capturing allocations is similar to capturing other operations like kernel launches or memory copies.  During capture, the operations don't actually execute, but are recorded.  To execute the captured operations, we must launch the graph using :func:`wp.capture_launch() <capture_launch>`.  This is important to keep in mind if you want to use an array that was allocated during graph capture.  The array doesn't actually exist until the captured graph is launched.  In the snippet above, we would get an error if we tried to print the array before calling :func:`wp.capture_launch() <capture_launch>`.

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
For memory allocated using default CUDA allocators, we can enable CUDA :ref:`peer access<peer_access>` using :func:`wp.set_peer_access_enabled() <set_peer_access_enabled>` to get similar benefits.

Because enabling memory pool access can have drawbacks, Warp does not automatically enable it, even if it's supported.  Programs that don't require copying data between GPUs are therefore not affected in any way.

.. autofunction:: warp.is_mempool_access_supported
.. autofunction:: warp.is_mempool_access_enabled
.. autofunction:: warp.set_mempool_access_enabled

Limitations
~~~~~~~~~~~

Mempool-to-Mempool Copies Between GPUs During Graph Capture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Copying data between different GPUs will fail during graph capture if the source and destination are allocated using mempool allocators and mempool access is not enabled between devices.  Note that this only applies to capturing mempool-to-mempool copies in a graph; copies done outside of graph capture are not affected.  Copies within the same mempool (i.e., same device) are also not affected.

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
