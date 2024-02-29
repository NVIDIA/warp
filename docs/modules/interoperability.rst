Interoperability
================

Warp can interop with other Python-based frameworks such as NumPy through standard interface protocols.

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

.. _pytorch-interop:

PyTorch
-------

Warp provides helper functions to convert arrays to/from PyTorch. Please see the ``warp.torch`` module for more details. Example usage is shown below::

    import warp.torch

    w = wp.array([1.0, 2.0, 3.0], dtype=float, device="cpu")

    # convert to Torch tensor
    t = warp.to_torch(w)

    # convert from Torch tensor
    w = warp.from_torch(t)

These helper functions allow the conversion of Warp arrays to/from PyTorch tensors without copying the underlying data.
At the same time, if available, gradient arrays and tensors are converted to/from PyTorch autograd tensors, allowing the use of Warp arrays
in PyTorch autograd computations.

Example: Optimization using ``warp.from_torch()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example usage of minimizing a loss function over an array of 2D points written in Warp via PyTorch's Adam optimizer using ``warp.from_torch`` is as follows::

    import warp as wp
    import torch

    wp.init()

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

Less code is needed when we declare the optimization variables directly in Warp and use ``warp.to_torch`` to convert them to PyTorch tensors.
Here, we revisit the same example from above where now only a single conversion to a torch tensor is needed to supply Adam with the optimization variables::

    import warp as wp
    import numpy as np
    import torch

    wp.init()

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

One can insert Warp kernel launches in a PyTorch graph by defining a ``torch.autograd.function`` class, which
requires forward and backward functions to be defined. After mapping incoming torch arrays to Warp arrays, a Warp kernel
may be launched in the usual way. In the backward pass, the same kernel's adjoint may be launched by 
setting ``adjoint = True`` in :func:`wp.launch() <launch>`. Alternatively, the user may choose to rely on Warp's tape.
In the following example, we demonstrate how Warp may be used to evaluate the Rosenbrock function in an optimization context::
    
    import warp as wp
    import numpy as np
    import torch

    wp.init()

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

.. automodule:: warp.torch
    :members:
    :undoc-members:

CuPy/Numba
----------

Warp GPU arrays support the ``__cuda_array_interface__`` protocol for sharing data with other Python GPU frameworks.
Currently this is one-directional, so that Warp arrays can be used as input to any framework that also supports the
``__cuda_array_interface__`` protocol, but not the other way around.

.. _jax-interop:

JAX
---

Interoperability with JAX arrays is supported through the following methods.
Internally these use the DLPack protocol to exchange data in a zero-copy way with JAX.

It may be preferable to use the :ref:`DLPack` protocol directly for better performance and control over stream synchronization behaviour.

.. automodule:: warp.jax
    :members:
    :undoc-members:

.. _DLPack:

DLPack
------

Warp supports the DLPack protocol included in the Python Array API standard v2022.12.
See the `Python Specification for DLPack <https://dmlc.github.io/dlpack/latest/python_spec.html>`_ for reference.

The canonical way to import an external array into Warp is using the ``warp.from_dlpack()`` function::

    warp_array = warp.from_dlpack(external_array)

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

    warp_array1 = warp.from_dlpack(jax.dlpack.to_dlpack(jax_array))
    warp_array2 = warp.from_dlpack(torch.utils.dlpack.to_dlpack(torch_tensor))

    jax_array = jax.dlpack.from_dlpack(warp.to_dlpack(warp_array))
    torch_tensor = torch.utils.dlpack.from_dlpack(warp.to_dlpack(warp_array))

This approach is generally faster because it skips any stream synchronization, but another solution must be used to ensure correct
ordering of operations.  In situations where no synchronization is required, using this approach can yield better performance.
This may be a good choice in situations like these:

    - The external framework is using the synchronous CUDA default stream.
    - Warp and the external framework are using the same CUDA stream.
    - Another synchronization mechanism is already in place.

.. automodule:: warp.dlpack
    :members:
    :undoc-members:
