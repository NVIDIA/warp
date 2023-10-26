.. _devices:

Devices
=======

Warp assigns unique string aliases to all supported compute devices in the system.  There is currently a single CPU device exposed as ``"cpu"``.  Each CUDA-capable GPU gets an alias of the form ``"cuda:i"``, where ``i`` is the CUDA device ordinal.  This convention should be familiar to users of other popular frameworks like PyTorch.

It is possible to explicitly target a specific device with each Warp API call using the ``device`` argument::

    a = wp.zeros(n, device="cpu")
    wp.launch(kernel, dim=a.size, inputs=[a], device="cpu")

    b = wp.zeros(n, device="cuda:0")
    wp.launch(kernel, dim=b.size, inputs=[b], device="cuda:0")

    c = wp.zeros(n, device="cuda:1")
    wp.launch(kernel, dim=c.size, inputs=[c], device="cuda:1")

.. note::

    A Warp CUDA device (``"cuda:i"``) corresponds to the primary CUDA context of device ``i``.  This is compatible with frameworks like PyTorch and other software that uses the CUDA Runtime API.  It makes interoperability easy, because GPU resources like memory can be shared with Warp.

Default Device
--------------

To simplify writing code, Warp has the concept of **default device**.  When the ``device`` argument is omitted from a Warp API call, the default device will be used.

During Warp initialization, the default device is set to be ``"cuda:0"`` if CUDA is available.  Otherwise, the default device is ``"cpu"``.

The function ``wp.set_device()`` can be used to change the default device::

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

    For CUDA devices, ``wp.set_device()`` does two things: it sets the Warp default device and it makes the device's CUDA context current.  This helps to minimize the number of CUDA context switches in blocks of code targeting a single device.

For PyTorch users, this function is similar to ``torch.cuda.set_device()``.  It is still possible to specify a different device in individual API calls, like in this snippet::

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

Scoped Devices
--------------

Another way to manage the default device is using ``wp.ScopedDevice`` objects.  They can be arbitrarily nested and restore the previous default device on exit::

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

.. note::

    For CUDA devices, ``wp.ScopedDevice`` makes the device's CUDA context current and restores the previous CUDA context on exit.  This is handy when running Warp scripts as part of a bigger pipeline, because it avoids any side effects of changing the CUDA context in the enclosed code.

Example: Using ``wp.ScopedDevice`` with multiple GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example shows how to allocate arrays and launch kernels on all available CUDA devices.

.. code:: python

    import warp as wp

    wp.init()


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

CUDA kernel launches and memory operations can execute asynchronously.  This allows for overlapping compute and memory operations on different devices.  Warp allows synchronizing the host with outstanding asynchronous operations on a specific device::

    wp.synchronize_device("cuda:1")

The ``wp.synchronize_device()`` function offers more fine-grained synchronization than ``wp.synchronize()``, as the latter waits for *all* devices to complete their work.

Custom CUDA Contexts
--------------------

Warp is designed to work with arbitrary CUDA contexts so it can easily integrate into different workflows.

Applications built on the CUDA Runtime API target the *primary context* of each device.  The Runtime API hides CUDA context management under the hood.  In Warp, device ``"cuda:i"`` represents the primary context of device ``i``, which aligns with the CUDA Runtime API.

Applications built on the CUDA Driver API work with CUDA contexts directly and can create custom CUDA contexts on any device.  Custom CUDA contexts can be created with specific affinity or interop features that benefit the application.  Warp can work with these CUDA contexts as well.

The special device alias ``"cuda"`` can be used to target the current CUDA context, whether this is a primary or custom context.

In addition, Warp allows registering new device aliases for custom CUDA contexts, so that they can be explicitly targeted by name.  If the ``CUcontext`` pointer is available, it can be used to create a new device alias like this::

    wp.map_cuda_device("foo", ctypes.c_void_p(context_ptr))

Alternatively, if the custom CUDA context was made current by the application, the pointer can be omitted::

    wp.map_cuda_device("foo")

In either case, mapping the custom CUDA context allows us to target the context directly using the assigned alias::

    with wp.ScopedDevice("foo"):
        a = wp.zeros(n)
        wp.launch(kernel, dim=a.size, inputs=[a])
