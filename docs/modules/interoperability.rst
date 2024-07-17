Interoperability
================

Warp can interoperate with other Python-based frameworks such as NumPy through standard interface protocols.

Warp supports passing external arrays to kernels directly, as long as they implement the ``__array__``, ``__array_interface__``, or ``__cuda_array_interface__`` protocols.  This works with many common frameworks like NumPy, CuPy, or PyTorch.

For example, we can use NumPy arrays directly when launching Warp kernels on the CPU:

.. code:: python

    import numpy as np
    import warp as wp

    @wp.kernel
    def saxpy(x: wp.array(dtype=float), y: wp.array(dtype=float), a: float):
        i = wp.tid()
        y[i] = a * x[i] + y[i]

    x = np.arange(n, dtype=np.float32)
    y = np.ones(n, dtype=np.float32)

    wp.launch(saxpy, dim=n, inputs=[x, y, 1.0], device="cpu")

Likewise, we can use CuPy arrays on a CUDA device:

.. code:: python

    import cupy as cp

    with cp.cuda.Device(0):
        x = cp.arange(n, dtype=cp.float32)
        y = cp.ones(n, dtype=cp.float32)

    wp.launch(saxpy, dim=n, inputs=[x, y, 1.0], device="cuda:0")

Note that with CUDA arrays, it's important to ensure that the device on which the arrays reside is the same as the device on which the kernel is launched.

PyTorch supports both CPU and GPU tensors and both kinds can be passed to Warp kernels on the appropriate device.

.. code:: python

    import random
    import torch

    if random.choice([False, True]):
        device = "cpu"
    else:
        device = "cuda:0"

    x = torch.arange(n, dtype=torch.float32, device=device)
    y = torch.ones(n, dtype=torch.float32, device=device)

    wp.launch(saxpy, dim=n, inputs=[x, y, 1.0], device=device)

NumPy
-----

Warp arrays may be converted to a NumPy array through the ``warp.array.numpy()`` method. When the Warp array lives on
the ``cpu`` device this will return a zero-copy view onto the underlying Warp allocation. If the array lives on a
``cuda`` device then it will first be copied back to a temporary buffer and copied to NumPy.

Warp CPU arrays also implement  the ``__array_interface__`` protocol and so can be used to construct NumPy arrays
directly::

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

To create Warp arrays from NumPy arrays, use :func:`warp.from_numpy` 
or pass the NumPy array as the ``data`` argument of the :class:`warp.array` constructor directly.

.. autofunction:: warp.from_numpy
.. autofunction:: warp.dtype_from_numpy
.. autofunction:: warp.dtype_to_numpy

.. _pytorch-interop:

PyTorch
-------

Warp provides helper functions to convert arrays to/from PyTorch::

    w = wp.array([1.0, 2.0, 3.0], dtype=float, device="cpu")

    # convert to Torch tensor
    t = wp.to_torch(w)

    # convert from Torch tensor
    w = wp.from_torch(t)

These helper functions allow the conversion of Warp arrays to/from PyTorch tensors without copying the underlying data.
At the same time, if available, gradient arrays and tensors are converted to/from PyTorch autograd tensors, allowing the use of Warp arrays
in PyTorch autograd computations.

.. autofunction:: warp.from_torch
.. autofunction:: warp.to_torch
.. autofunction:: warp.device_from_torch
.. autofunction:: warp.device_to_torch
.. autofunction:: warp.dtype_from_torch
.. autofunction:: warp.dtype_to_torch

To convert a PyTorch CUDA stream to a Warp CUDA stream and vice versa, Warp provides the following functions:

.. autofunction:: warp.stream_from_torch
.. autofunction:: warp.stream_to_torch

Example: Optimization using ``warp.from_torch()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example usage of minimizing a loss function over an array of 2D points written in Warp via PyTorch's Adam optimizer
using :func:`warp.from_torch` is as follows::

    import warp as wp
    import torch


    @wp.kernel()
    def loss(xs: wp.array(dtype=float, ndim=2), l: wp.array(dtype=float)):
        tid = wp.tid()
        wp.atomic_add(l, 0, xs[tid, 0] ** 2.0 + xs[tid, 1] ** 2.0)

    # indicate requires_grad so that Warp can accumulate gradients in the grad buffers
    xs = torch.randn(100, 2, requires_grad=True)
    l = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([xs], lr=0.1)

    wp_xs = wp.from_torch(xs)
    wp_l = wp.from_torch(l)

    tape = wp.Tape()
    with tape:
        # record the loss function kernel launch on the tape
        wp.launch(loss, dim=len(xs), inputs=[wp_xs], outputs=[wp_l], device=wp_xs.device)

    for i in range(500):
        tape.zero()
        tape.backward(loss=wp_l)  # compute gradients
        # now xs.grad will be populated with the gradients computed by Warp
        opt.step()  # update xs (and thereby wp_xs)

        # these lines are only needed for evaluating the loss
        # (the optimization just needs the gradient, not the loss value)
        wp_l.zero_()
        wp.launch(loss, dim=len(xs), inputs=[wp_xs], outputs=[wp_l], device=wp_xs.device)
        print(f"{i}\tloss: {l.item()}")

Example: Optimization using ``warp.to_torch``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Less code is needed when we declare the optimization variables directly in Warp and use :func:`warp.to_torch` to convert them to PyTorch tensors.
Here, we revisit the same example from above where now only a single conversion to a torch tensor is needed to supply Adam with the optimization variables::

    import warp as wp
    import numpy as np
    import torch


    @wp.kernel()
    def loss(xs: wp.array(dtype=float, ndim=2), l: wp.array(dtype=float)):
        tid = wp.tid()
        wp.atomic_add(l, 0, xs[tid, 0] ** 2.0 + xs[tid, 1] ** 2.0)

    # initialize the optimization variables in Warp
    xs = wp.array(np.random.randn(100, 2), dtype=wp.float32, requires_grad=True)
    l = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    # just a single wp.to_torch call is needed, Adam optimizes using the Warp array gradients
    opt = torch.optim.Adam([wp.to_torch(xs)], lr=0.1)

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

Example: Optimization using ``torch.autograd.function``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One can insert Warp kernel launches in a PyTorch graph by defining a :class:`torch.autograd.Function` class, which
requires forward and backward functions to be defined. After mapping incoming torch arrays to Warp arrays, a Warp kernel
may be launched in the usual way. In the backward pass, the same kernel's adjoint may be launched by 
setting ``adjoint = True`` in :func:`wp.launch() <launch>`. Alternatively, the user may choose to rely on Warp's tape.
In the following example, we demonstrate how Warp may be used to evaluate the Rosenbrock function in an optimization context::

    import warp as wp
    import numpy as np
    import torch

    pvec2 = wp.types.vector(length=2, dtype=wp.float32)

    # Define the Rosenbrock function
    @wp.func
    def rosenbrock(x: float, y: float):
        return (1.0 - x) ** 2.0 + 100.0 * (y - x**2.0) ** 2.0

    @wp.kernel
    def eval_rosenbrock(
        xs: wp.array(dtype=pvec2),
        # outputs
        z: wp.array(dtype=float),
    ):
        i = wp.tid()
        x = xs[i]
        z[i] = rosenbrock(x[0], x[1])


    class Rosenbrock(torch.autograd.Function):
        @staticmethod
        def forward(ctx, xy, num_points):
            # ensure Torch operations complete before running Warp
            wp.synchronize_device()

            ctx.xy = wp.from_torch(xy, dtype=pvec2, requires_grad=True)
            ctx.num_points = num_points

            # allocate output
            ctx.z = wp.zeros(num_points, requires_grad=True)

            wp.launch(
                kernel=eval_rosenbrock,
                dim=ctx.num_points,
                inputs=[ctx.xy],
                outputs=[ctx.z]
            )

            # ensure Warp operations complete before returning data to Torch
            wp.synchronize_device()

            return wp.to_torch(ctx.z)

        @staticmethod
        def backward(ctx, adj_z):
            # ensure Torch operations complete before running Warp
            wp.synchronize_device()

            # map incoming Torch grads to our output variables
            ctx.z.grad = wp.from_torch(adj_z)

            wp.launch(
                kernel=eval_rosenbrock,
                dim=ctx.num_points,
                inputs=[ctx.xy],
                outputs=[ctx.z],
                adj_inputs=[ctx.xy.grad],
                adj_outputs=[ctx.z.grad],
                adjoint=True
            )

            # ensure Warp operations complete before returning data to Torch
            wp.synchronize_device()

            # return adjoint w.r.t. inputs
            return (wp.to_torch(ctx.xy.grad), None)


    num_points = 1500
    learning_rate = 5e-2

    torch_device = wp.device_to_torch(wp.get_device())

    rng = np.random.default_rng(42)
    xy = torch.tensor(rng.normal(size=(num_points, 2)), dtype=torch.float32, requires_grad=True, device=torch_device)
    opt = torch.optim.Adam([xy], lr=learning_rate)

    for _ in range(10000):
        # step
        opt.zero_grad()
        z = Rosenbrock.apply(xy, num_points)
        z.backward(torch.ones_like(z))

        opt.step()

    # minimum at (1, 1)
    xy_np = xy.numpy(force=True)
    print(np.mean(xy_np, axis=0))

Note that if Warp code is wrapped in a torch.autograd.function that gets called in ``torch.compile()``, it will automatically
exclude that function from compiler optimizations. If your script uses ``torch.compile()``, we recommend using Pytorch version 2.3.0+,
which has improvements that address this scenario.

Performance Notes
^^^^^^^^^^^^^^^^^

The ``wp.from_torch()`` function creates a Warp array object that shares data with a PyTorch tensor.  Although this function does not copy the data, there is always some CPU overhead during the conversion.  If these conversions happen frequently, the overall program performance may suffer.  As a general rule, it's good to avoid repeated conversions of the same tensor.  Instead of:

.. code:: python

    x_t = torch.arange(n, dtype=torch.float32, device=device)
    y_t = torch.ones(n, dtype=torch.float32, device=device)

    for i in range(10):
        x_w = wp.from_torch(x_t)
        y_w = wp.from_torch(y_t)
        wp.launch(saxpy, dim=n, inputs=[x_w, y_w, 1.0], device=device)

Try converting the arrays only once and reuse them:

.. code:: python

    x_t = torch.arange(n, dtype=torch.float32, device=device)
    y_t = torch.ones(n, dtype=torch.float32, device=device)

    x_w = wp.from_torch(x_t)
    y_w = wp.from_torch(y_t)

    for i in range(10):
        wp.launch(saxpy, dim=n, inputs=[x_w, y_w, 1.0], device=device)

If reusing arrays is not possible (e.g., a new PyTorch tensor is constructed on every iteration), passing ``return_ctype=True`` to ``wp.from_torch()`` should yield faster performance.  Setting this argument to True avoids constructing a ``wp.array`` object and instead returns a low-level array descriptor.  This descriptor is a simple C structure that can be passed to Warp kernels instead of a ``wp.array``, but cannot be used in other places that require a ``wp.array``.

.. code:: python

    for n in range(1, 10):
        # get Torch tensors for this iteration
        x_t = torch.arange(n, dtype=torch.float32, device=device)
        y_t = torch.ones(n, dtype=torch.float32, device=device)

        # get Warp array descriptors
        x_ctype = wp.from_torch(x_t, return_ctype=True)
        y_ctype = wp.from_torch(y_t, return_ctype=True)

        wp.launch(saxpy, dim=n, inputs=[x_ctype, y_ctype, 1.0], device=device)

An alternative approach is to pass the PyTorch tensors to Warp kernels directly.  This avoids constructing temporary Warp arrays by leveraging standard array interfaces (like ``__cuda_array_interface__``) supported by both PyTorch and Warp.  The main advantage of this approach is convenience, since there is no need to call any conversion functions.  The main limitation is that it does not handle gradients, because gradient information is not included in the standard array interfaces.  This technique is therefore most suitable for algorithms that do not involve differentiation.

.. code:: python

    x = torch.arange(n, dtype=torch.float32, device=device)
    y = torch.ones(n, dtype=torch.float32, device=device)

    for i in range(10):
        wp.launch(saxpy, dim=n, inputs=[x, y, 1.0], device=device)

.. code:: shell

    python -m warp.examples.benchmarks.benchmark_interop_torch

Sample output:

.. code::

    5095 ms  from_torch(...)
    2113 ms  from_torch(..., return_ctype=True)
    2950 ms  direct from torch

The default ``wp.from_torch()`` conversion is the slowest.  Passing ``return_ctype=True`` is the fastest, because it skips creating temporary Warp array objects.  Passing PyTorch tensors to Warp kernels directly falls somewhere in between.  It skips creating temporary Warp arrays, but accessing the ``__cuda_array_interface__`` attributes of PyTorch tensors adds overhead because they are initialized on-demand.


CuPy/Numba
----------

Warp GPU arrays support the ``__cuda_array_interface__`` protocol for sharing data with other Python GPU frameworks.  This allows frameworks like CuPy to use Warp GPU arrays directly.

Likewise, Warp arrays can be created from any object that exposes the ``__cuda_array_interface__``.  Such objects can also be passed to Warp kernels directly without creating a Warp array object.

.. _jax-interop:

JAX
---

Interoperability with JAX arrays is supported through the following methods.
Internally these use the DLPack protocol to exchange data in a zero-copy way with JAX::

    warp_array = wp.from_jax(jax_array)
    jax_array = wp.to_jax(warp_array)

It may be preferable to use the :ref:`DLPack` protocol directly for better performance and control over stream synchronization behaviour.

.. autofunction:: warp.from_jax
.. autofunction:: warp.to_jax
.. autofunction:: warp.device_from_jax
.. autofunction:: warp.device_to_jax
.. autofunction:: warp.dtype_from_jax
.. autofunction:: warp.dtype_to_jax


Using Warp kernels as JAX primitives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
    This is an experimental feature under development.

Warp kernels can be used as JAX primitives, which can be used to call Warp kernels inside of jitted JAX functions::

    import warp as wp
    import jax
    import jax.numpy as jp

    # import experimental feature
    from warp.jax_experimental import jax_kernel

    @wp.kernel
    def triple_kernel(input: wp.array(dtype=float), output: wp.array(dtype=float)):
        tid = wp.tid()
        output[tid] = 3.0 * input[tid]

    # create a Jax primitive from a Warp kernel
    jax_triple = jax_kernel(triple_kernel)

    # use the Warp kernel in a Jax jitted function
    @jax.jit
    def f():
        x = jp.arange(0, 64, dtype=jp.float32)
        return jax_triple(x)

    print(f())

Since this is an experimental feature, there are some limitations:

    - All kernel arguments must be arrays.
    - Kernel launch dimensions are inferred from the shape of the first argument.
    - Input arguments are followed by output arguments in the Warp kernel definition.
    - There must be at least one input argument and at least one output argument.
    - Output shapes must match the launch dimensions (i.e., output shapes must match the shape of the first argument).
    - All arrays must be contiguous.
    - Only the CUDA backend is supported.

Here is an example of an operation with three inputs and two outputs::

    import warp as wp
    import jax
    import jax.numpy as jp

    # import experimental feature
    from warp.jax_experimental import jax_kernel

    # kernel with multiple inputs and outputs
    @wp.kernel
    def multiarg_kernel(
        # inputs
        a: wp.array(dtype=float),
        b: wp.array(dtype=float),
        c: wp.array(dtype=float),
        # outputs
        ab: wp.array(dtype=float),
        bc: wp.array(dtype=float),
    ):
        tid = wp.tid()
        ab[tid] = a[tid] + b[tid]
        bc[tid] = b[tid] + c[tid]

    # create a Jax primitive from a Warp kernel
    jax_multiarg = jax_kernel(multiarg_kernel)

    # use the Warp kernel in a Jax jitted function with three inputs and two outputs
    @jax.jit
    def f():
        a = jp.full(64, 1, dtype=jp.float32)
        b = jp.full(64, 2, dtype=jp.float32)
        c = jp.full(64, 3, dtype=jp.float32)
        return jax_multiarg(a, b, c)

    x, y = f()

    print(x)
    print(y)

.. _DLPack:

DLPack
------

Warp supports the DLPack protocol included in the Python Array API standard v2022.12.
See the `Python Specification for DLPack <https://dmlc.github.io/dlpack/latest/python_spec.html>`_ for reference.

The canonical way to import an external array into Warp is using the ``warp.from_dlpack()`` function::

    warp_array = wp.from_dlpack(external_array)

The external array can be a PyTorch tensor, Jax array, or any other array type compatible with this version of the DLPack protocol.
For CUDA arrays, this approach requires the producer to perform stream synchronization which ensures that operations on the array
are ordered correctly.  The ``warp.from_dlpack()`` function asks the producer to synchronize the current Warp stream on the device where
the array resides.  Thus it should be safe to use the array in Warp kernels on that device without any additional synchronization.

The canonical way to export a Warp array to an external framework is to use the ``from_dlpack()`` function in that framework::

    jax_array = jax.dlpack.from_dlpack(warp_array)
    torch_tensor = torch.utils.dlpack.from_dlpack(warp_array)

For CUDA arrays, this will synchronize the current stream of the consumer framework with the current Warp stream on the array's device.
Thus it should be safe to use the wrapped array in the consumer framework, even if the array was previously used in a Warp kernel
on the device.

Alternatively, arrays can be shared by explicitly creating PyCapsules using a ``to_dlpack()`` function provided by the producer framework.
This approach may be used for older versions of frameworks that do not support the v2022.12 standard::

    warp_array1 = wp.from_dlpack(jax.dlpack.to_dlpack(jax_array))
    warp_array2 = wp.from_dlpack(torch.utils.dlpack.to_dlpack(torch_tensor))

    jax_array = jax.dlpack.from_dlpack(wp.to_dlpack(warp_array))
    torch_tensor = torch.utils.dlpack.from_dlpack(wp.to_dlpack(warp_array))

This approach is generally faster because it skips any stream synchronization, but another solution must be used to ensure correct
ordering of operations.  In situations where no synchronization is required, using this approach can yield better performance.
This may be a good choice in situations like these:

    - The external framework is using the synchronous CUDA default stream.
    - Warp and the external framework are using the same CUDA stream.
    - Another synchronization mechanism is already in place.

.. autofunction:: warp.from_dlpack
.. autofunction:: warp.to_dlpack
