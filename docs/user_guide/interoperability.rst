Interoperability
================

Warp interoperates with other Python-based frameworks through standard interface protocols. Warp accepts external arrays in kernel launches as long as they implement the ``__array__``, ``__array_interface__``, or ``__cuda_array_interface__`` protocols. This works with NumPy, CuPy, PyTorch, JAX, and Paddle.

Quick Reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 18 42 15 25

   * - Framework
     - Conversion
     - Zero-copy
     - Gradient-aware
   * - NumPy
     - :func:`wp.from_numpy <warp.from_numpy>` / :meth:`array.numpy() <warp.array.numpy>`
     - CPU only
     - No
   * - PyTorch
     - :func:`wp.from_torch <warp.from_torch>` / :func:`wp.to_torch <warp.to_torch>`
     - Yes
     - Yes
   * - JAX
     - :func:`wp.from_jax <warp.from_jax>` / :func:`wp.to_jax <warp.to_jax>`
     - Yes
     - Via ``jax_kernel``
   * - Paddle
     - :func:`wp.from_paddle <warp.from_paddle>` / :func:`wp.to_paddle <warp.to_paddle>`
     - Yes
     - Yes
   * - CuPy / Numba
     - ``__cuda_array_interface__`` protocol
     - Yes
     - No
   * - DLPack
     - :func:`wp.from_dlpack <warp.from_dlpack>` / ``framework.from_dlpack``
     - Yes
     - No

For full framework coverage, see:

* :doc:`interoperability_pytorch`
* :doc:`interoperability_jax`

Paddle is documented inline below. NumPy, CuPy, Numba, and DLPack interop is protocol-level and covered in the sections that follow.

Pass Arrays Directly
--------------------

Any object that implements ``__array_interface__`` (CPU) or ``__cuda_array_interface__`` (GPU) can be passed to :func:`wp.launch <warp.launch>` inputs without calling a conversion function. This is the fastest on-ramp for most use cases.

.. code:: python

    import numpy as np
    import warp as wp

    @wp.kernel
    def saxpy(x: wp.array[float], y: wp.array[float], a: float):
        i = wp.tid()
        y[i] = a * x[i] + y[i]

    x = np.arange(n, dtype=np.float32)
    y = np.ones(n, dtype=np.float32)

    wp.launch(saxpy, dim=n, inputs=[x, y, 1.0], device="cpu")

On CUDA, the same pattern works with CuPy, PyTorch, or any framework that exposes ``__cuda_array_interface__``:

.. code:: python

    import cupy as cp

    with cp.cuda.Device(0):
        x = cp.arange(n, dtype=cp.float32)
        y = cp.ones(n, dtype=cp.float32)

    wp.launch(saxpy, dim=n, inputs=[x, y, 1.0], device="cuda:0")

When passing CUDA arrays, ensure the device on which the arrays reside matches the device on which the kernel is launched.

NumPy
-----

Warp arrays may be converted to a NumPy array through the :meth:`array.numpy() <warp.array.numpy>` method. When the Warp array lives on the ``cpu`` device this returns a zero-copy view onto the underlying Warp allocation. If the array lives on a ``cuda`` device, it is first copied back to a temporary buffer and then copied to NumPy.

Warp CPU arrays also implement the ``__array_interface__`` protocol and can be used to construct NumPy arrays directly::

    w = wp.array([1.0, 2.0, 3.0], dtype=float, device="cpu")
    a = np.array(w)
    print(a)
    > [1. 2. 3.]

Data type conversion utilities are also available for convenience:

.. code:: python

    warp_type = wp.float32
    ...
    numpy_type = wp.dtype_to_numpy(warp_type)
    ...
    a = wp.zeros(n, dtype=warp_type)
    b = np.zeros(n, dtype=numpy_type)

To create Warp arrays from NumPy arrays, use :func:`warp.from_numpy` or pass the NumPy array as the ``data`` argument of the :class:`warp.array` constructor directly.

CuPy / Numba
------------

Warp GPU arrays support the ``__cuda_array_interface__`` protocol for sharing data with other Python GPU frameworks. This allows frameworks like CuPy and Numba to use Warp GPU arrays directly.

Likewise, Warp arrays can be created from any object that exposes the ``__cuda_array_interface__``. Such objects can also be passed to Warp kernels directly without creating a Warp array object.

.. _paddle-interop:

Paddle
------

Warp provides helper functions to convert arrays to/from Paddle::

    w = wp.array([1.0, 2.0, 3.0], dtype=float, device="cpu")

    # convert to Paddle tensor
    t = wp.to_paddle(w)

    # convert from Paddle tensor
    w = wp.from_paddle(t)

These helper functions convert Warp arrays to/from Paddle tensors without copying the underlying data. Gradient arrays and tensors are converted to/from Paddle autograd tensors, allowing the use of Warp arrays in Paddle autograd computations.

To convert a Paddle CUDA stream to a Warp CUDA stream and vice versa, Warp provides:

* :func:`warp.stream_from_paddle`

Example: Optimization using :func:`warp.to_paddle`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Less code is needed when we declare the optimization variables directly in Warp and use :func:`warp.to_paddle` to convert them to Paddle tensors. The following example minimizes a loss function over an array of 2D points written in Warp, with a single ``wp.to_paddle`` call supplying Adam with the optimization variables::

    import warp as wp
    import numpy as np
    import paddle

    @wp.kernel()
    def loss(xs: wp.array2d[float], l: wp.array[float]):
        tid = wp.tid()
        wp.atomic_add(l, 0, xs[tid, 0] ** 2.0 + xs[tid, 1] ** 2.0)

    # initialize the optimization variables in Warp
    xs = wp.array(np.random.randn(100, 2), dtype=wp.float32, requires_grad=True)
    l = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    # just a single wp.to_paddle call is needed, Adam optimizes using the Warp array gradients
    opt = paddle.optimizer.Adam(learning_rate=0.1, parameters=[wp.to_paddle(xs)])

    tape = wp.Tape()
    with tape:
        wp.launch(loss, dim=len(xs), inputs=[xs], outputs=[l], device=xs.device)

    for i in range(500):
        tape.zero()
        tape.backward(loss=l)
        opt.step()

        l.zero_()
        wp.launch(loss, dim=len(xs), inputs=[xs], outputs=[l], device=xs.device)
        print(f"{i}\tloss: {l.numpy()[0]}")

.. note::
   Paddle follows the same performance-tuning pattern as PyTorch. See the "Performance Tuning" section of :doc:`interoperability_pytorch` for ``return_ctype``, direct tensor passing, and reuse-don't-reconvert guidance.

.. _DLPack:

DLPack
------

Warp supports the DLPack protocol included in the Python Array API standard v2022.12. See the `Python Specification for DLPack <https://dmlc.github.io/dlpack/latest/python_spec.html>`__ for reference.

The canonical way to import an external array into Warp is using the :func:`warp.from_dlpack()` function::

    warp_array = wp.from_dlpack(external_array)

The external array can be a PyTorch tensor, JAX array, or any other array type compatible with this version of the DLPack protocol. For CUDA arrays, this approach requires the producer to perform stream synchronization which ensures that operations on the array are ordered correctly. The :func:`warp.from_dlpack()` function asks the producer to synchronize the current Warp stream on the device where the array resides. It should therefore be safe to use the array in Warp kernels on that device without any additional synchronization.

The canonical way to export a Warp array to an external framework is to use the ``from_dlpack()`` function in that framework::

    jax_array = jax.dlpack.from_dlpack(warp_array)
    torch_tensor = torch.utils.dlpack.from_dlpack(warp_array)
    paddle_tensor = paddle.utils.dlpack.from_dlpack(warp_array)

For CUDA arrays, this synchronizes the current stream of the consumer framework with the current Warp stream on the array's device. It should therefore be safe to use the wrapped array in the consumer framework, even if the array was previously used in a Warp kernel on the device.

Alternatively, arrays can be shared by explicitly creating PyCapsules using a ``to_dlpack()`` function provided by the producer framework. This approach may be used for older versions of frameworks that do not support the v2022.12 standard::

    warp_array1 = wp.from_dlpack(jax_array)
    warp_array2 = wp.from_dlpack(torch.utils.dlpack.to_dlpack(torch_tensor))
    warp_array3 = wp.from_dlpack(paddle.utils.dlpack.to_dlpack(paddle_tensor))

    jax_array = jax.dlpack.from_dlpack(wp.to_dlpack(warp_array))
    torch_tensor = torch.utils.dlpack.from_dlpack(wp.to_dlpack(warp_array))
    paddle_tensor = paddle.utils.dlpack.from_dlpack(wp.to_dlpack(warp_array))

This approach is generally faster because it skips any stream synchronization, but another solution must be used to ensure correct ordering of operations. In situations where no synchronization is required, using this approach can yield better performance. This may be a good choice in situations like these:

- The external framework is using the synchronous CUDA default stream.
- Warp and the external framework are using the same CUDA stream.
- Another synchronization mechanism is already in place.

The framework-specific converters (:func:`warp.to_torch`, :func:`warp.to_paddle`) are usually a better choice than DLPack when one exists, because DLPack does not carry gradient information. If autograd needs to flow between Warp and the other framework, use the direct converters. DLPack is most useful when no framework-specific converter exists, or when both producer and consumer are DLPack-native.

.. toctree::
   :titlesonly:

   interoperability_pytorch
   interoperability_jax
