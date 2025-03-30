Devices
=======

.. currentmodule:: warp

Warp assigns unique string aliases to all supported compute devices in the system.  There is currently a single CPU device exposed as ``"cpu"``.  Each CUDA-capable GPU gets an alias of the form ``"cuda:i"``, where ``i`` is the CUDA device ordinal.  This convention should be familiar to users of other popular frameworks like PyTorch.

It is possible to explicitly target a specific device with each Warp API call using the ``device`` argument::

    a = wp.zeros(n, device="cpu")
    wp.launch(kernel, dim=a.size, inputs=[a], device="cpu")

    b = wp.zeros(n, device="cuda:0")
    wp.launch(kernel, dim=b.size, inputs=[b], device="cuda:0")

    c = wp.zeros(n, device="cuda:1")
    wp.launch(kernel, dim=c.size, inputs=[c], device="cuda:1")

.. note::

    A Warp CUDA device (``"cuda:i"``) corresponds to the primary CUDA context of device ``i``.
    This is compatible with frameworks like PyTorch and other software that uses the CUDA Runtime API.
    It makes interoperability easy because GPU resources like memory can be shared with Warp.

.. autoclass:: warp.context.Device
    :members:

Warp also provides functions that can be used to query the available devices on the system:

.. autofunction:: get_devices
.. autofunction:: get_cuda_devices
.. autofunction:: get_cuda_device_count

Default Device
--------------

To simplify writing code, Warp has the concept of **default device**.  When the ``device`` argument is omitted from a Warp API call, the default device will be used.

Calling :func:`wp.get_device() <warp.get_device>` without an argument
will return an instance of :class:`warp.context.Device` for the default device.

During Warp initialization, the default device is set to ``"cuda:0"`` if CUDA is available.  Otherwise, the default device is ``"cpu"``.
If the default device is changed, :func:`wp.get_preferred_device() <warp.get_preferred_device>` can be used to get
the *original* default device.

:func:`wp.set_device() <warp.set_device>` can be used to change the default device::

    wp.set_device("cpu")
    a = wp.zeros(n)
    wp.launch(kernel, dim=a.size, inputs=[a])
   
    wp.set_device("cuda:0")
    b = wp.zeros(n)
    wp.launch(kernel, dim=b.size, inputs=[b])
   
    wp.set_device("cuda:1")
    c = wp.zeros(n)
    wp.launch(kernel, dim=c.size, inputs=[c])

.. note::

    For CUDA devices, :func:`wp.set_device() <warp.set_device>` does two things: It sets the Warp default device and it makes the device's CUDA context current.  This helps to minimize the number of CUDA context switches in blocks of code targeting a single device.

For PyTorch users, this function is similar to :func:`torch.cuda.set_device()`.
It is still possible to specify a different device in individual API calls, like in this snippet::

    # set default device
    wp.set_device("cuda:0")
   
    # use default device
    a = wp.zeros(n)
   
    # use explicit devices
    b = wp.empty(n, device="cpu")
    c = wp.empty(n, device="cuda:1")
   
    # use default device
    wp.launch(kernel, dim=a.size, inputs=[a])
   
    wp.copy(b, a)
    wp.copy(c, a)

.. autofunction:: set_device
.. autofunction:: get_device
.. autofunction:: get_preferred_device

Scoped Devices
--------------

Another way to manage the default device is using :class:`wp.ScopedDevice <ScopedDevice>` objects.
They can be arbitrarily nested and restore the previous default device on exit::

    with wp.ScopedDevice("cpu"):
        # alloc and launch on "cpu"
        a = wp.zeros(n)
        wp.launch(kernel, dim=a.size, inputs=[a])
 
    with wp.ScopedDevice("cuda:0"):
        # alloc on "cuda:0"
        b = wp.zeros(n)
   
        with wp.ScopedDevice("cuda:1"):
            # alloc and launch on "cuda:1"
            c = wp.zeros(n)
            wp.launch(kernel, dim=c.size, inputs=[c])
   
        # launch on "cuda:0"
        wp.launch(kernel, dim=b.size, inputs=[b])

.. autoclass:: ScopedDevice

Example: Using ``wp.ScopedDevice`` with multiple GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example shows how to allocate arrays and launch kernels on all available CUDA devices.

.. code:: python

    import warp as wp


    @wp.kernel
    def inc(a: wp.array(dtype=float)):
        tid = wp.tid()
        a[tid] = a[tid] + 1.0


    # get all CUDA devices
    devices = wp.get_cuda_devices()
    device_count = len(devices)

    # number of launches
    iters = 1000

    # list of arrays, one per device
    arrs = []

    # loop over all devices
    for device in devices:
        # use a ScopedDevice to set the target device
        with wp.ScopedDevice(device):
            # allocate array
            a = wp.zeros(250 * 1024 * 1024, dtype=float)
            arrs.append(a)

            # launch kernels
            for _ in range(iters):
                wp.launch(inc, dim=a.size, inputs=[a])

    # synchronize all devices
    wp.synchronize()

    # print results
    for i in range(device_count):
        print(f"{arrs[i].device} -> {arrs[i].numpy()}")


Current CUDA Device
-------------------

Warp uses the device alias ``"cuda"`` to target the current CUDA device.  This allows external code to manage the CUDA device on which to execute Warp scripts.  It is analogous to the PyTorch ``"cuda"`` device, which should be familiar to Torch users and simplify interoperation.

In this snippet, we use PyTorch to manage the current CUDA device and invoke a Warp kernel on that device::

    def example_function():
        # create a Torch tensor on the current CUDA device
        t = torch.arange(10, dtype=torch.float32, device="cuda")

        a = wp.from_torch(t)

        # launch a Warp kernel on the current CUDA device
        wp.launch(kernel, dim=a.size, inputs=[a], device="cuda")

    # use Torch to set the current CUDA device and run example_function() on that device
    torch.cuda.set_device(0)
    example_function()

    # use Torch to change the current CUDA device and re-run example_function() on that device
    torch.cuda.set_device(1)
    example_function()

.. note::

    Using the device alias ``"cuda"`` can be problematic if the code runs in an environment where another part of the code can unpredictably change the CUDA context.  Using an explicit CUDA device like ``"cuda:i"`` is recommended to avoid such issues.

Device Synchronization
----------------------

CUDA kernel launches and memory operations can execute asynchronously.
This allows for overlapping compute and memory operations on different devices.
Warp allows synchronizing the host with outstanding asynchronous operations on a specific device::

    wp.synchronize_device("cuda:1")

:func:`wp.synchronize_device() <synchronize_device>` offers more fine-grained synchronization than
:func:`wp.synchronize() <synchronize>`, as the latter waits for *all* devices to complete their work.

.. autofunction:: synchronize_device
.. autofunction:: synchronize

Custom CUDA Contexts
--------------------

Warp is designed to work with arbitrary CUDA contexts so it can easily integrate into different workflows.

Applications built on the CUDA Runtime API target the *primary context* of each device.  The Runtime API hides CUDA context management under the hood.  In Warp, device ``"cuda:i"`` represents the primary context of device ``i``, which aligns with the CUDA Runtime API.

Applications built on the CUDA Driver API work with CUDA contexts directly and can create custom CUDA contexts on any device.  Custom CUDA contexts can be created with specific affinity or interop features that benefit the application.  Warp can work with these CUDA contexts as well.

The special device alias ``"cuda"`` can be used to target the current CUDA context, whether this is a primary or custom context.

In addition, Warp allows registering new device aliases for custom CUDA contexts using
:func:`wp.map_cuda_device() <map_cuda_device>` so that they can be explicitly targeted by name.
If the ``CUcontext`` pointer is available, it can be used to create a new device alias like this::

    wp.map_cuda_device("foo", ctypes.c_void_p(context_ptr))

Alternatively, if the custom CUDA context was made current by the application, the pointer can be omitted::

    wp.map_cuda_device("foo")

In either case, mapping the custom CUDA context allows us to target the context directly using the assigned alias::

    with wp.ScopedDevice("foo"):
        a = wp.zeros(n)
        wp.launch(kernel, dim=a.size, inputs=[a])

.. autofunction:: map_cuda_device
.. autofunction:: unmap_cuda_device

.. _peer_access:

CUDA Peer Access
----------------

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
        wp.set_peer_access_enabled("cuda:0", "cuda:1", True):

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

.. autofunction:: warp.is_peer_access_supported
.. autofunction:: warp.is_peer_access_enabled
.. autofunction:: warp.set_peer_access_enabled
