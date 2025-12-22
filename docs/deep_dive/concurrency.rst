Concurrency
===========

.. currentmodule:: warp

Asynchronous Operations
-----------------------

Kernel Launches
~~~~~~~~~~~~~~~

Kernels launched on a CUDA device are asynchronous with respect to the host (CPU Python thread).  Launching a kernel schedules
its execution on the CUDA device, but the :func:`wp.launch() <launch>` function can return before the kernel execution
completes.  This allows us to run some CPU computations while the CUDA kernel is executing, which is an
easy way to introduce parallelism into our programs.

.. code:: python

    wp.launch(kernel1, dim=n, inputs=[a], device="cuda:0")

    # do some CPU work while the CUDA kernel is running
    do_cpu_work()

Kernels launched on different CUDA devices can execute concurrently.  This can be used to tackle independent sub-tasks in parallel on different GPUs while using the CPU to do other useful work:

.. code:: python

    # launch concurrent kernels on different devices
    wp.launch(kernel1, dim=n, inputs=[a0], device="cuda:0")
    wp.launch(kernel2, dim=n, inputs=[a1], device="cuda:1")

    # do CPU work while kernels are running on both GPUs
    do_cpu_work()

Launching kernels on the CPU is currently a synchronous operation.  In other words, :func:`wp.launch() <launch>` will return only after the kernel has finished executing on the CPU.  To run a CUDA kernel and a CPU kernel concurrently, the CUDA kernel should be launched first:

.. code:: python

    # schedule a kernel on a CUDA device
    wp.launch(kernel1, ..., device="cuda:0")

    # run a kernel on the CPU while the CUDA kernel is running
    wp.launch(kernel2, ..., device="cpu")


Graph Launches
~~~~~~~~~~~~~~

The concurrency rules for CUDA graph launches are similar to CUDA kernel launches, except that graphs are not available on the CPU.

.. code:: python

    # capture work on cuda:0 in a graph
    with wp.ScopedCapture(device="cuda:0") as capture0:
        do_gpu0_work()

    # capture work on cuda:1 in a graph
    with wp.ScopedCapture(device="cuda:1") as capture1:
        do_gpu1_work()

    # launch captured graphs on the respective devices concurrently
    wp.capture_launch(capture0.graph)
    wp.capture_launch(capture1.graph)

    # do some CPU work while the CUDA graphs are running
    do_cpu_work()


Array Creation
~~~~~~~~~~~~~~

Creating CUDA arrays is also asynchronous with respect to the host.  It involves allocating memory on the device
and initializing it, which is done under the hood using a kernel launch or an asynchronous CUDA memset operation.

.. code:: python

    a0 = wp.zeros(n, dtype=float, device="cuda:0")
    b0 = wp.ones(n, dtype=float, device="cuda:0")

    a1 = wp.empty(n, dtype=float, device="cuda:1")
    b1 = wp.full(n, 42.0, dtype=float, device="cuda:1")

In this snippet, arrays ``a0`` and ``b0`` are created on device ``cuda:0`` and arrays ``a1`` and ``b1`` are created
on device ``cuda:1``.  The operations on the same device are sequential, but each device executes them independently of the
other device, so they can run concurrently.


Array Copying
~~~~~~~~~~~~~

Copying arrays between devices can also be asynchronous, but there are some details to be aware of.

Copying from host memory to a CUDA device and copying from a CUDA device to host memory is asynchronous only if the host array is pinned.
Pinned memory allows the CUDA driver to use direct memory transfers (DMA), which are generally faster and can be done without involving the CPU.
There are a couple of drawbacks to using pinned memory: allocation and deallocation is usually slower and there are system-specific limits
on how much pinned memory can be allocated on the system.  For this reason, Warp CPU arrays are not pinned by default.  You can request a pinned
allocation by passing the ``pinned=True`` flag when creating a CPU array.  This is a good option for arrays that are used to copy data
between host and device, especially if asynchronous transfers are desired.

.. code:: python

    h = wp.zeros(n, dtype=float, device="cpu")
    p = wp.zeros(n, dtype=float, device="cpu", pinned=True)
    d = wp.zeros(n, dtype=float, device="cuda:0")

    # host-to-device copy
    wp.copy(d, h)  # synchronous
    wp.copy(d, p)  # asynchronous
    
    # device-to-host copy
    wp.copy(h, d)  # synchronous
    wp.copy(p, d)  # asynchronous

    # wait for asynchronous operations to complete
    wp.synchronize_device("cuda:0")

Copying between CUDA arrays on the same device is always asynchronous with respect to the host, since it does not involve the CPU:

.. code:: python

    a = wp.zeros(n, dtype=float, device="cuda:0")
    b = wp.empty(n, dtype=float, device="cuda:0")

    # asynchronous device-to-device copy
    wp.copy(a, b)

    # wait for transfer to complete
    wp.synchronize_device("cuda:0")

Copying between CUDA arrays on different devices is also asynchronous with respect to the host.  Peer-to-peer transfers require
extra care, because CUDA devices are also asynchronous with respect to each other.  When copying an array from one GPU to another,
the destination GPU is used to perform the copy, so we need to ensure that prior work on the source GPU completes before the transfer.

.. code:: python

    a0 = wp.zeros(n, dtype=float, device="cuda:0")
    a1 = wp.empty(n, dtype=float, device="cuda:1")

    # wait for outstanding work on the source device to complete to ensure the source array is ready
    wp.synchronize_device("cuda:0")

    # asynchronous peer-to-peer copy
    wp.copy(a1, a0)

    # wait for the copy to complete on the destination device
    wp.synchronize_device("cuda:1")

Note that peer-to-peer transfers can be accelerated using :ref:`memory pool access <mempool_access>` or :ref:`peer access <peer_access>`, which enables DMA transfers between CUDA devices on supported systems.

.. _streams:

Streams
-------

A CUDA stream is a sequence of operations that execute in order on the GPU.  Operations from different streams may run concurrently
and may be interleaved by the device scheduler.

Warp automatically creates a stream for each CUDA device during initialization.  This becomes the current stream for the device.
All kernel launches and memory operations issued on that device are placed on the current stream.

Creating Streams
~~~~~~~~~~~~~~~~

A stream is tied to a particular CUDA device.  New streams can be created using the :class:`wp.Stream <Stream>` constructor:

.. code:: python

    s1 = wp.Stream("cuda:0")  # create a stream on a specific CUDA device
    s2 = wp.Stream()          # create a stream on the default device

If the device parameter is omitted, the default device will be used, which can be managed using :class:`wp.ScopedDevice <ScopedDevice>`.

For interoperation with external code, it is possible to pass a CUDA stream handle to wrap an external stream:

.. code:: python

    s3 = wp.Stream("cuda:0", cuda_stream=stream_handle)

The ``cuda_stream`` argument must be a native stream handle (``cudaStream_t`` or ``CUstream``) passed as a Python integer.
This mechanism is used internally for sharing streams with external frameworks like PyTorch or DLPack.  The caller is responsible for ensuring
that the external stream does not get destroyed while it is referenced by a ``wp.Stream`` object.

Using Streams
~~~~~~~~~~~~~

Use :class:`wp.ScopedStream <ScopedStream>` to temporarily change the current stream on a device and schedule a sequence of operations on that stream:

.. code:: python

    stream = wp.Stream("cuda:0")

    with wp.ScopedStream(stream):
        a = wp.zeros(n, dtype=float)
        b = wp.empty(n, dtype=float)
        wp.launch(kernel, dim=n, inputs=[a])
        wp.copy(b, a)

Since streams are tied to a particular device, :class:`wp.ScopedStream <ScopedStream>` subsumes the functionality of :class:`wp.ScopedDevice <ScopedDevice>`.  That's why we don't need to explicitly specify the ``device`` argument to each of the calls.

An important benefit of streams is that they can be used to overlap compute and data transfer operations on the same device,
which can improve the overall throughput of a program by doing those operations in parallel.

.. code:: python

    with wp.ScopedDevice("cuda:0"):
        a = wp.zeros(n, dtype=float)
        b = wp.empty(n, dtype=float)
        c = wp.ones(n, dtype=float, device="cpu", pinned=True)

        compute_stream = wp.Stream()
        transfer_stream = wp.Stream()

        # asynchronous kernel launch on a stream
        with wp.ScopedStream(compute_stream)
            wp.launch(kernel, dim=a.size, inputs=[a])

        # asynchronous host-to-device copy on another stream
        with wp.ScopedStream(transfer_stream)
            wp.copy(b, c)

The :func:`wp.get_stream() <get_stream>` function can be used to get the current stream on a device:

.. code:: python

    s1 = wp.get_stream("cuda:0")  # get the current stream on a specific device
    s2 = wp.get_stream()          # get the current stream on the default device

The :func:`wp.set_stream() <set_stream>` function can be used to set the current stream on a device:

.. code:: python

    wp.set_stream(stream, device="cuda:0")  # set the stream on a specific device
    wp.set_stream(stream)                   # set the stream on the default device

In general, we recommend using :class:`wp.ScopedStream <ScopedStream>` rather than :func:`wp.set_stream() <set_stream>`.

Synchronization
~~~~~~~~~~~~~~~

:func:`wp.synchronize_stream() <synchronize_stream>` can be used to block the host thread until the given stream completes:

.. code:: python

    wp.synchronize_stream(stream)

In a program that uses multiple streams, this gives a more fine-grained level of control over synchronization behavior
than :func:`wp.synchronize_device() <synchronize_device>`, which synchronizes all streams on the device.
For example, if a program has multiple compute and transfer streams, the host might only want to wait for one transfer stream
to complete, without waiting for the other streams.  By synchronizing only one stream, we allow the others to continue running
concurrently with the host thread.

.. _cuda_events:

Events
~~~~~~

Functions like :func:`wp.synchronize_device() <synchronize_device>` or :func:`wp.synchronize_stream() <synchronize_stream>` block the CPU thread until work completes on a CUDA device, but they're not intended to synchronize multiple CUDA streams with each other.

CUDA events provide a mechanism for device-side synchronization between streams.
This kind of synchronization does not block the host thread, but it allows one stream to wait for work on another stream
to complete.

Like streams, events are tied to a particular device:

.. code:: python

    e1 = wp.Event("cuda:0")  # create an event on a specific CUDA device
    e2 = wp.Event()          # create an event on the default device

To wait for a stream to complete some work, we first record the event on that stream.  Then we make another stream
wait on that event:

.. code:: python

    stream1 = wp.Stream("cuda:0")
    stream2 = wp.Stream("cuda:0")
    event = wp.Event("cuda:0")

    stream1.record_event(event)
    stream2.wait_event(event)

Note that when recording events, the event must be from the same device as the recording stream.
When waiting for events, the waiting stream can be from another device.  This allows using events to synchronize streams
on different GPUs.

If the :meth:`Stream.record_event` method is called without an event argument, a temporary event will be created, recorded, and returned:

.. code:: python

    event = stream1.record_event()
    stream2.wait_event(event)

The :meth:`Stream.wait_stream` method combines the acts of recording and waiting on an event in one call:

.. code:: python

    stream2.wait_stream(stream1)

Warp also provides global functions :func:`wp.record_event() <record_event>`, :func:`wp.wait_event() <wait_event>`, and :func:`wp.wait_stream() <wait_stream>` which operate on the current
stream of the default device:

.. code:: python

    wp.record_event(event)  # record an event on the current stream
    wp.wait_event(event)    # make the current stream wait for an event
    wp.wait_stream(stream)  # make the current stream wait for another stream

These variants are convenient to use inside of :class:`wp.ScopedStream <ScopedStream>` and :class:`wp.ScopedDevice <ScopedDevice>` managers.

Here is a more complete example with a producer stream that copies data into an array and a consumer stream
that uses the array in a kernel:

.. code:: python

    with wp.ScopedDevice("cuda:0"):
        a = wp.empty(n, dtype=float)
        b = wp.ones(n, dtype=float, device="cpu", pinned=True)

        producer_stream = wp.Stream()
        consumer_stream = wp.Stream()

        with wp.ScopedStream(producer_stream)
            # asynchronous host-to-device copy
            wp.copy(a, b)

            # record an event to create a synchronization point for the consumer stream
            event = wp.record_event()

            # do some unrelated work in the producer stream
            do_other_producer_work()

        with wp.ScopedStream(consumer_stream)
            # do some unrelated work in the consumer stream
            do_other_consumer_work()

            # wait for the producer copy to complete
            wp.wait_event(event)

            # consume the array in a kernel
            wp.launch(kernel, dim=a.size, inputs=[a])

The function :func:`wp.synchronize_event() <synchronize_event>` can be used to block the host thread until a recorded event completes.  This is useful when the host wants to wait for a specific synchronization point on a stream, while allowing subsequent stream operations to continue executing asynchronously.

.. code:: python

    with wp.ScopedDevice("cpu"):
        # CPU buffers for readback
        a_host = wp.empty(N, dtype=float, pinned=True)
        b_host = wp.empty(N, dtype=float, pinned=True)

    with wp.ScopedDevice("cuda:0"):
        stream = wp.get_stream()

        # initialize first GPU array
        a = wp.full(N, 17, dtype=float)
        # asynchronous readback
        wp.copy(a_host, a)
        # record event
        a_event = stream.record_event()

        # initialize second GPU array
        b = wp.full(N, 42, dtype=float)
        # asynchronous readback
        wp.copy(b_host, b)
        # record event
        b_event = stream.record_event()

        # wait for first array readback to complete
        wp.synchronize_event(a_event)
        # process first array on the CPU
        assert np.array_equal(a_host.numpy(), np.full(N, fill_value=17.0))

        # wait for second array readback to complete
        wp.synchronize_event(b_event)
        # process second array on the CPU
        assert np.array_equal(b_host.numpy(), np.full(N, fill_value=42.0))

Querying Stream and Event Status
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :attr:`Stream.is_complete` and :attr:`Event.is_complete` attributes can be used to query the status of a stream or
event. These queries do not block the host thread unlike :func:`wp.synchronize_stream() <synchronize_stream>` and
:func:`wp.synchronize_event() <synchronize_event>`.

These attributes are useful for running operations on the CPU while waiting for GPU operations to complete:

.. code:: python

    @wp.kernel
    def test_kernel(sum: wp.array(dtype=wp.uint64)):
        wp.atomic_add(sum, 0, wp.uint64(1))


    sum = wp.zeros(1, dtype=wp.uint64)
    wp.launch(test_kernel, dim=8 * 1024 * 1024, outputs=[sum])

    # Have the CPU do some unrelated work while the GPU is computing
    counter = 0
    while not wp.get_stream().is_complete:
        print(f"counter: {counter}")
        counter += 1

:attr:`Stream.is_complete` and :attr:`Event.is_complete` cannot be accessed during a graph capture.

CUDA Default Stream
~~~~~~~~~~~~~~~~~~~

Warp avoids using the synchronous CUDA default stream, which is a special stream that synchronizes with all other streams
on the same device.  This stream is currently only used during readback operations that are provided for convenience, such as ``array.numpy()`` and ``array.list()``.

.. code:: python

    stream1 = wp.Stream("cuda:0")
    stream2 = wp.Stream("cuda:0")

    with wp.ScopedStream(stream1):
        a = wp.zeros(n, dtype=float)

    with wp.ScopedStream(stream2):
        b = wp.ones(n, dtype=float)

    print(a)
    print(b)

In the snippet above, there are two arrays that are initialized on different CUDA streams.  Printing those arrays triggers
a readback, which is done using the ``array.numpy()`` method.  This readback happens on the synchronous CUDA default stream,
which means that no explicit synchronization is required.  The reason for this is convenience - printing an array is useful
for debugging purposes, so it's nice not to worry about synchronization.

The drawback of this approach is that the CUDA default stream (and any methods that use it) cannot be used during graph capture.
The regular :func:`wp.copy() <copy>` function should be used to capture readback operations in a graph.


Explicit Streams Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~

Several Warp functions accept optional ``stream`` arguments.  This allows directly specifying the stream without
using a :class:`wp.ScopedStream <ScopedStream>` manager.  There are benefits and drawbacks to both approaches, which will be discussed below.
Functions that accept stream arguments directly include :func:`wp.launch() <launch>`, :func:`wp.capture_launch() <capture_launch>`, and :func:`wp.copy() <copy>`.

To launch a kernel on a specific stream:

.. code:: python

    wp.launch(kernel, dim=n, inputs=[...], stream=my_stream)

When launching a kernel with an explicit ``stream`` argument, the ``device`` argument should be omitted, since the device is inferred
from the stream.  If both ``stream`` and ``device`` are specified, the ``stream`` argument takes precedence.

To launch a graph on a specific stream:

.. code:: python

    wp.capture_launch(graph, stream=my_stream)

For both kernel and graph launches, specifying the stream directly can be faster than using :class:`wp.ScopedStream <ScopedStream>`.
While :class:`wp.ScopedStream <ScopedStream>` is useful for scheduling a sequence of operations on a specific stream, there is some overhead
in setting and restoring the current stream on the device.  This overhead is negligible for larger workloads,
but performance-sensitive code may benefit from specifying the stream directly instead of using :class:`wp.ScopedStream <ScopedStream>`, especially
for a single kernel or graph launch.

In addition to these performance considerations, specifying the stream directly can be useful when copying arrays between
two CUDA devices.  By default, Warp uses the following rules to determine which stream will be used for the copy:

- If the destination array is on a CUDA device, use the current stream on the destination device.
- Otherwise, if the source array is on a CUDA device, use the current stream on the source device.

In the case of peer-to-peer copies, specifying the ``stream`` argument allows overriding these rules, and the copy can
be performed on a stream from any device.

.. code:: python

    stream0 = wp.get_stream("cuda:0")
    stream1 = wp.get_stream("cuda:1")

    a0 = wp.zeros(n, dtype=float, device="cuda:0")
    a1 = wp.empty(n, dtype=float, device="cuda:1")

    # wait for the destination array to be ready
    stream0.wait_stream(stream1)

    # use the source device stream to do the copy
    wp.copy(a1, a0, stream=stream0)

Notice that we use event synchronization to make the source stream wait for the destination stream prior to the copy.
This is due to the :ref:`stream-ordered memory pool allocators<mempool_allocators>` introduced in Warp 0.14.0.  The allocation of the
empty array ``a1`` is scheduled on stream ``stream1``.  To avoid use-before-alloc errors, we need to wait until the 
allocation completes before using that array on a different stream.

Stream Priorities
~~~~~~~~~~~~~~~~~

Streams can be created with a specified numerical priority using the ``priority`` parameter when creating a new
:class:`Stream`. High-priority streams can be created with a priority of -1, while low-priority streams
have a priority of 0. By scheduling work on streams of different priorities, users can achieve finer-grained
control of how the GPU schedules pending work. Priorities are only a hint to the GPU for how to
process work and do not guarantee that pending work will be executed in a certain order.
Stream priorities currently do not affect host-to-device or device-to-host memory transfers.

Streams created with a priority outside the valid values of -1 and 0 will have
the priority clamped.
The priority of any stream can be queried using the :attr:`Stream.priority` attribute.
If a CUDA device does not support stream priorities, then all streams will have
a priority of 0 regardless of the priority requested when creating the stream.

For more information on stream priorities, see the section in the
`CUDA C++ Programming Guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-priorities>`_.

The following example illustrates the impact of stream priorities:

.. code:: python

    import warp as wp

    wp.config.verify_cuda = True

    wp.init()

    total_size = 256 * 1024 * 1024
    each_size = 128 * 1024 * 1024

    with wp.ScopedDevice("cuda:0"):
        array_lo = wp.zeros(total_size, dtype=wp.float32)
        array_hi = wp.zeros(total_size, dtype=wp.float32)

        stream_lo = wp.Stream(wp.get_device(), 0)  # Low priority
        stream_hi = wp.Stream(wp.get_device(), -1)  # High priority

        start_lo_event = wp.Event(enable_timing=True)
        start_hi_event = wp.Event(enable_timing=True)
        end_lo_event = wp.Event(enable_timing=True)
        end_hi_event = wp.Event(enable_timing=True)

        wp.synchronize_device(wp.get_device())

        stream_lo.record_event(start_lo_event)
        stream_hi.record_event(start_hi_event)

        for copy_offset in range(0, total_size, each_size):
            wp.copy(array_lo, array_lo, copy_offset, copy_offset, each_size, stream_lo)
            wp.copy(array_hi, array_hi, copy_offset, copy_offset, each_size, stream_hi)

        stream_lo.record_event(end_lo_event)
        stream_hi.record_event(end_hi_event)

        # get elapsed time between the two events
        elapsed_lo = wp.get_event_elapsed_time(start_lo_event, end_lo_event)
        elapsed_hi = wp.get_event_elapsed_time(start_hi_event, end_hi_event)

        print(f"elapsed_lo = {elapsed_lo:.6f}")
        print(f"elapsed_hi = {elapsed_hi:.6f}")

The output of the example on a test workstation looks like::

    elapsed_lo = 5.118944
    elapsed_hi = 2.647040

If the example is modified so that both streams have the same priority, the output becomes::

    elapsed_lo = 5.112832
    elapsed_hi = 5.114880

Finally, if we reverse the stream priorities so that ``stream_lo`` has a
a priority of -1 and ``stream_hi`` has a priority of 0, we get::

    elapsed_lo = 2.621440
    elapsed_hi = 5.105664

Stream Usage Guidance
~~~~~~~~~~~~~~~~~~~~~

Stream synchronization can be a tricky business, even for experienced CUDA developers.  Consider the following code:

.. code:: python

    a = wp.zeros(n, dtype=float, device="cuda:0")

    s = wp.Stream("cuda:0")

    wp.launch(kernel, dim=a.size, inputs=[a], stream=s)

This snippet has a stream synchronization problem that is difficult to detect at first glance.
It's quite possible that the code will work just fine, but it introduces undefined behavior,
which may lead to incorrect results that manifest only once in a while.  The issue is that the kernel is launched
on stream ``s``, which is different than the stream used for creating array ``a``.  The array is allocated and
initialized on the current stream of device ``cuda:0``, which means that it might not be ready when stream ``s``
begins executing the kernel that consumes the array.

The solution is to synchronize the streams, which can be done like this:

.. code:: python

    a = wp.zeros(n, dtype=float, device="cuda:0")

    s = wp.Stream("cuda:0")

    # wait for the current stream on cuda:0 to finish initializing the array
    s.wait_stream(wp.get_stream("cuda:0"))

    wp.launch(kernel, dim=a.size, inputs=[a], stream=s)

The :class:`wp.ScopedStream <ScopedStream>` manager is designed to alleviate this common problem.  It synchronizes the new stream with the
previous stream on the device.  Its behavior is equivalent to inserting the ``wait_stream()`` call as shown above.
With :class:`wp.ScopedStream <ScopedStream>`, we don't need to explicitly sync the new stream with the previous stream:

.. code:: python

    a = wp.zeros(n, dtype=float, device="cuda:0")

    s = wp.Stream("cuda:0")

    with wp.ScopedStream(s):
        wp.launch(kernel, dim=a.size, inputs=[a])

This makes :class:`wp.ScopedStream <ScopedStream>` the recommended way of getting started with streams in Warp.  Using explicit stream arguments
might be slightly more performant, but it requires more attention to stream synchronization mechanics.
If you are a stream novice, consider the following trajectory for integrating streams into your Warp programs:

- Level 1:  Don't.  You don't need to use streams to use Warp.  Avoiding streams is a perfectly valid and respectable way to live.  Many interesting and sophisticated algorithms can be developed without fancy stream juggling.  Often it's better to focus on solving a problem in a simple and elegant way, unencumbered by the vagaries of low-level stream management.
- Level 2:  Use :class:`wp.ScopedStream <ScopedStream>`.  It helps to avoid some common hard-to-catch issues.  There's a little bit of overhead, but it should be negligible if the GPU workloads are large enough.  Consider adding streams into your program as a form of targeted optimization, especially if some areas like memory transfers ("feeding the beast") are a known bottleneck.  Streams are great for overlapping memory transfers with compute workloads.
- Level 3:  Use explicit stream arguments for kernel launches, array copying, etc.  This will be the most performant approach that can get you close to the speed of light.  You will need to take care of all stream synchronization yourself, but the results can be rewarding in the benchmarks.

.. _synchronization_guidance:

Synchronization Guidance
------------------------

The general rule with synchronization is to use as little of it as possible, but not less.

Excessive synchronization can severely limit the performance of programs.  Synchronization means that a stream or thread
is waiting for something else to complete.  While it's waiting, it's not doing any useful work, which means that any
outstanding work cannot start until the synchronization point is reached.  This limits parallel execution, which is 
often important for squeezing the most juice out of the collection of hardware components.

On the other hand, insufficient synchronization can lead to errors or incorrect results if operations execute out-of-order.
A fast program is no good if it can't guarantee correct results.

Host-side Synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~

Host-side synchronization blocks the host thread (Python) until GPU work completes.  This is necessary when
you are waiting for some GPU work to complete so that you can access the results on the CPU.

:func:`wp.synchronize() <synchronize>` is the most heavy-handed synchronization function, since it synchronizes all the devices in the system.  It is almost never the right function to call if performance is important.  However, it can sometimes be useful when debugging synchronization-related issues.

:func:`wp.synchronize_device(device) <synchronize_device>` synchronizes a single device, which is generally better and faster.  This synchronizes all the streams on the specified device, including streams created by Warp and those created by any other framework.

:func:`wp.synchronize_stream(stream) <synchronize_stream>` synchronizes a single stream, which is better still.  If the program uses multiple streams, you can wait for a specific one to finish without waiting for the others.  This is handy if you have a readback stream that is copying data from the GPU to the CPU.  You can wait for the transfer to complete and start processing it on the CPU while other streams are still chugging along on the GPU, in parallel with the host code.

:func:`wp.synchronize_event(event) <synchronize_event>` is the most specific host synchronization function.  It blocks the host until an event previously recorded on a CUDA stream completes.  This can be used to wait for a specific stream synchronization point to be reached, while allowing subsequent operations on that stream to continue asynchronously.

Device-side Synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Device-side synchronization uses CUDA events to make one stream wait for a synchronization point recorded on another stream (:func:`wp.record_event() <record_event>`, :func:`wp.wait_event() <wait_event>`, :func:`wp.wait_stream() <wait_stream>`).

These functions don't block the host thread, so the CPU can stay busy doing useful work, like preparing the next batch of data
to feed the beast.  Events can be used to synchronize streams on the same device or even different CUDA devices, so you can
choreograph very sophisticated multi-stream and multi-device workloads that execute entirely on the available GPUs.
This allows keeping host-side synchronization to a minimum, perhaps only when reading back the final results.

Synchronization and Graph Capture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A CUDA graph captures a sequence of operations on a CUDA stream that can be replayed multiple times with low overhead.
During capture, certain CUDA functions are not allowed, which includes host-side synchronization functions.  Using the synchronous
CUDA default stream is also not allowed.  The only form of synchronization allowed in CUDA graphs is event-based synchronization.

A CUDA graph capture must start and end on the same stream, but multiple streams can be used in the middle.  This allows CUDA graphs to encompass multiple streams and even multiple GPUs.  Events play a crucial role with multi-stream graph capture because they are used to fork and join new streams to the main capture stream, in addition to their regular synchronization duties.

Here's an example of capturing a multi-GPU graph using a stream on each device:

.. code:: python

    stream0 = wp.Stream("cuda:0")
    stream1 = wp.Stream("cuda:1")

    # use stream0 as the main capture stream
    with wp.ScopedCapture(stream=stream0) as capture:

        # fork stream1, which adds it to the set of streams being captured
        stream1.wait_stream(stream0)

        # launch a kernel on stream0
        wp.launch(kernel, ..., stream=stream0)

        # launch a kernel on stream1
        wp.launch(kernel, ..., stream=stream1)

        # join stream1
        stream0.wait_stream(stream1)

    # launch the multi-GPU graph, which can execute the captured kernels concurrently
    wp.capture_launch(capture.graph)
