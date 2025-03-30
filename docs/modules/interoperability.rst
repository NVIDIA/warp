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

Warp arrays may be converted to a NumPy array through the :meth:`array.numpy() <warp.array.numpy>` method. When the Warp array lives on
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

Warp provides helper functions to convert arrays to/from PyTorch:

.. code:: python

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
using :func:`warp.from_torch` is as follows:

.. code:: python

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
Here, we revisit the same example from above where now only a single conversion to a PyTorch tensor is needed to supply Adam with the optimization variables:

.. code:: python

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

Example: Optimization using ``torch.autograd.function`` (PyTorch <= 2.3.1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One can insert Warp kernel launches in a PyTorch graph by defining a :class:`torch.autograd.Function` class, which
requires forward and backward functions to be defined. After mapping incoming PyTorch tensors to Warp arrays, a Warp kernel
may be launched in the usual way. In the backward pass, the same kernel's adjoint may be launched by 
setting ``adjoint = True`` in :func:`wp.launch() <warp.launch>`. Alternatively, the user may choose to rely on Warp's tape.
In the following example, we demonstrate how Warp may be used to evaluate the Rosenbrock function in an optimization context.

.. code:: python

    import warp as wp
    import numpy as np
    import torch

    # Define the Rosenbrock function
    @wp.func
    def rosenbrock(x: float, y: float):
        return (1.0 - x) ** 2.0 + 100.0 * (y - x**2.0) ** 2.0

    @wp.kernel
    def eval_rosenbrock(
        xs: wp.array(dtype=wp.vec2),
        # outputs
        z: wp.array(dtype=float),
    ):
        i = wp.tid()
        x = xs[i]
        z[i] = rosenbrock(x[0], x[1])


    class Rosenbrock(torch.autograd.Function):
        @staticmethod
        def forward(ctx, xy, num_points):
            ctx.xy = wp.from_torch(xy, dtype=wp.vec2, requires_grad=True)
            ctx.num_points = num_points

            # allocate output
            ctx.z = wp.zeros(num_points, requires_grad=True)

            wp.launch(
                kernel=eval_rosenbrock,
                dim=ctx.num_points,
                inputs=[ctx.xy],
                outputs=[ctx.z]
            )

            return wp.to_torch(ctx.z)

        @staticmethod
        def backward(ctx, adj_z):
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

Note that if Warp code is wrapped in a :class:`torch.autograd.Function` that gets called in :func:`torch.compile()`, it will automatically
exclude that function from compiler optimizations. If your script uses :func:`torch.compile()`,
we recommend using PyTorch version 2.3.0+, which has improvements that address this scenario.

.. _pytorch-custom-ops-example:

Example: Optimization using PyTorch custom operators (PyTorch >= 2.4.0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch 2.4+ introduced `custom operators <https://pytorch.org/tutorials/advanced/python_custom_ops.html#python-custom-ops-tutorial>`_ to replace 
PyTorch autograd functions. These treat arbitrary Python functions (including Warp calls) as opaque callables, which prevents
:func:`torch.compile()` from tracing into them. This means that forward PyTorch graph evaluations that include Warp kernel launches can be safely accelerated with
:func:`torch.compile()`. We can re-write the previous example using custom operators as follows:

.. code:: python

    import warp as wp
    import numpy as np
    import torch

    # Define the Rosenbrock function
    @wp.func
    def rosenbrock(x: float, y: float):
        return (1.0 - x) ** 2.0 + 100.0 * (y - x**2.0) ** 2.0


    @wp.kernel
    def eval_rosenbrock(
        xy: wp.array(dtype=wp.vec2),
        # outputs
        z: wp.array(dtype=float),
    ):
        i = wp.tid()
        v = xy[i]
        z[i] = rosenbrock(v[0], v[1])


    @torch.library.custom_op("wp::warp_rosenbrock", mutates_args=())
    def warp_rosenbrock(xy: torch.Tensor, num_points: int) -> torch.Tensor:
        wp_xy = wp.from_torch(xy, dtype=wp.vec2)
        wp_z = wp.zeros(num_points, dtype=wp.float32)

        wp.launch(kernel=eval_rosenbrock, dim=num_points, inputs=[wp_xy], outputs=[wp_z])

        return wp.to_torch(wp_z)


    @warp_rosenbrock.register_fake
    def _(xy, num_points):
        return torch.empty(num_points, dtype=torch.float32)


    @torch.library.custom_op("wp::warp_rosenbrock_backward", mutates_args=())
    def warp_rosenbrock_backward(
        xy: torch.Tensor, num_points: int, z: torch.Tensor, adj_z: torch.Tensor
    ) -> torch.Tensor:
        wp_xy = wp.from_torch(xy, dtype=wp.vec2)
        wp_z = wp.from_torch(z, requires_grad=False)
        wp_adj_z = wp.from_torch(adj_z, requires_grad=False)

        wp.launch(
            kernel=eval_rosenbrock,
            dim=num_points,
            inputs=[wp_xy],
            outputs=[wp_z],
            adj_inputs=[wp_xy.grad],
            adj_outputs=[wp_adj_z],
            adjoint=True,
        )

        return wp.to_torch(wp_xy.grad)


    @warp_rosenbrock_backward.register_fake
    def _(xy, num_points, z, adj_z):
        return torch.empty_like(xy)


    def backward(ctx, adj_z):
        ctx.xy.grad = warp_rosenbrock_backward(ctx.xy, ctx.num_points, ctx.z, adj_z)
        return ctx.xy.grad, None


    def setup_context(ctx, inputs, output):
        ctx.xy, ctx.num_points = inputs
        ctx.z = output


    warp_rosenbrock.register_autograd(backward, setup_context=setup_context)

    num_points = 1500
    learning_rate = 5e-2

    torch_device = wp.device_to_torch(wp.get_device())

    rng = np.random.default_rng(42)
    xy = torch.tensor(rng.normal(size=(num_points, 2)), dtype=torch.float32, requires_grad=True, device=torch_device)
    opt = torch.optim.Adam([xy], lr=learning_rate)

    @torch.compile(fullgraph=True)
    def forward():
        global xy, num_points

        z = warp_rosenbrock(xy, num_points)
        return z

    for _ in range(10000):
        # step
        opt.zero_grad()
        z = forward()
        z.backward(torch.ones_like(z))
        opt.step()

    # minimum at (1, 1)
    xy_np = xy.numpy(force=True)
    print(np.mean(xy_np, axis=0))

Performance Notes
^^^^^^^^^^^^^^^^^

The :func:`wp.from_torch() <warp.from_torch>` function creates a Warp array object that shares data with a PyTorch tensor.
Although this function does not copy the data, there is always some CPU overhead during the conversion.
If these conversions happen frequently, the overall program performance may suffer.
As a general rule, repeated conversions of the same tensor should be avoided.  Instead of:

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

If reusing arrays is not possible (e.g., a new PyTorch tensor is constructed on every iteration), passing ``return_ctype=True``
to :func:`wp.from_torch() <warp.from_torch>` should yield better performance.
Setting this argument to ``True`` avoids constructing a ``wp.array`` object and instead returns a low-level array descriptor.
This descriptor is a simple C structure that can be passed to Warp kernels instead of a ``wp.array``,
but cannot be used in other places that require a ``wp.array``.

.. code:: python

    for n in range(1, 10):
        # get Torch tensors for this iteration
        x_t = torch.arange(n, dtype=torch.float32, device=device)
        y_t = torch.ones(n, dtype=torch.float32, device=device)

        # get Warp array descriptors
        x_ctype = wp.from_torch(x_t, return_ctype=True)
        y_ctype = wp.from_torch(y_t, return_ctype=True)

        wp.launch(saxpy, dim=n, inputs=[x_ctype, y_ctype, 1.0], device=device)

An alternative approach is to pass the PyTorch tensors to Warp kernels directly.
This avoids constructing temporary Warp arrays by leveraging standard array interfaces (like ``__cuda_array_interface__``) supported by both PyTorch and Warp.
The main advantage of this approach is convenience, since there is no need to call any conversion functions.
The main limitation is that it does not handle gradients, because gradient information is not included in the standard array interfaces.
This technique is therefore most suitable for algorithms that do not involve differentiation.

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

The default :func:`wp.from_torch() <warp.from_torch>` conversion is the slowest.
Passing ``return_ctype=True`` is the fastest, because it skips creating temporary Warp array objects.
Passing PyTorch tensors to Warp kernels directly falls somewhere in between.
It skips creating temporary Warp arrays, but accessing the ``__cuda_array_interface__`` attributes of PyTorch tensors
adds overhead because they are initialized on-demand.


CuPy/Numba
----------

Warp GPU arrays support the ``__cuda_array_interface__`` protocol for sharing data with other Python GPU frameworks.
This allows frameworks like CuPy to use Warp GPU arrays directly.

Likewise, Warp arrays can be created from any object that exposes the ``__cuda_array_interface__``.
Such objects can also be passed to Warp kernels directly without creating a Warp array object.

.. _jax-interop:

JAX
---

Interoperability with JAX arrays is supported through the following methods.
Internally these use the DLPack protocol to exchange data in a zero-copy way with JAX::

    warp_array = wp.from_jax(jax_array)
    jax_array = wp.to_jax(warp_array)

It may be preferable to use the :ref:`DLPack` protocol directly for better performance and control over stream synchronization .

.. autofunction:: warp.from_jax
.. autofunction:: warp.to_jax
.. autofunction:: warp.device_from_jax
.. autofunction:: warp.device_to_jax
.. autofunction:: warp.dtype_from_jax
.. autofunction:: warp.dtype_to_jax


Using Warp kernels as JAX primitives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
    This version of :func:`jax_kernel() <warp.jax_experimental.jax_kernel>` is based on JAX features that are now deprecated.

    For JAX version 0.4.31 or newer, users are encouraged to switch to the new version of
    :func:`jax_kernel() <warp.jax_experimental.ffi.jax_kernel>`
    based on the new :ref:`Foreign Function Interface (FFI)<jax-ffi>`.

Warp kernels can be used as JAX primitives, which allows calling them inside of jitted JAX functions::

    import warp as wp
    import jax
    import jax.numpy as jnp

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
        x = jnp.arange(0, 64, dtype=jnp.float32)
        return jax_triple(x)

    print(f())

.. autofunction:: warp.jax_experimental.jax_kernel


Input and Output Semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~

All kernel arguments must be contiguous arrays.
Input arguments must come before output arguments in the kernel definition.
At least one input array and one output array are required.
Here is a kernel with three inputs and two outputs::

    import warp as wp
    import jax
    import jax.numpy as jnp

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
        a = jnp.full(64, 1, dtype=jnp.float32)
        b = jnp.full(64, 2, dtype=jnp.float32)
        c = jnp.full(64, 3, dtype=jnp.float32)
        return jax_multiarg(a, b, c)

    x, y = f()

    print(x)
    print(y)


Kernel Launch and Output Dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the launch dimensions are inferred from the shape of the first input array.
When that's not appropriate, the ``launch_dims`` argument can be used to override this behavior.
The launch dimensions also determine the shape of the output arrays.
Here is a simple matrix multiplication kernel that multiplies an NxK matrix by a KxM matrix.
The launch dimensions and output shape must be (N, M), which is different than the shape of the input arrays::

    import warp as wp
    import jax
    import jax.numpy as jnp

    import warp as wp
    from warp.jax_experimental import jax_kernel

    @wp.kernel
    def matmul_kernel(
        a: wp.array2d(dtype=float),  # NxK input
        b: wp.array2d(dtype=float),  # KxM input
        c: wp.array2d(dtype=float),  # NxM output
    ):
        # launch dims should be (N, M)
        i, j = wp.tid()
        N = a.shape[0]
        K = a.shape[1]
        M = b.shape[1]
        if i < N and j < M:
            s = wp.float32(0)
            for k in range(K):
                s += a[i, k] * b[k, j]
            c[i, j] = s

    N, M, K = 3, 4, 2

    # specify custom launch dimensions
    jax_matmul = jax_kernel(matmul_kernel, launch_dims=(N, M))

    @jax.jit
    def f():
        a = jnp.full((N, K), 2, dtype=jnp.float32)
        b = jnp.full((K, M), 3, dtype=jnp.float32)

        # use default launch dims
        return jax_matmul(a, b)

    print(f())


.. _jax-ffi:

JAX Foreign Function Interface (FFI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 1.7

JAX v0.4.31 introduced a new `foreign function interface <https://docs.jax.dev/en/latest/ffi.html>`_ that supersedes
the older custom call mechanism. One important benefit is that it allows the foreign function to be captured in a CUDA
graph together with other JAX operations. This can lead to significant performance improvements.

Users of newer JAX versions are encouraged to switch to the new implementation of
:func:`jax_kernel() <warp.jax_experimental.ffi.jax_kernel>` based on FFI.
The old implementation is still available to avoid breaking existing code,
but future development will likely focus on the FFI version.

.. code-block:: python

    from warp.jax_experimental.ffi import jax_kernel  # new FFI-based jax_kernel()

The new implementation is likely to be faster and it is also more flexible.

.. autofunction:: warp.jax_experimental.ffi.jax_kernel

Input and Output Semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~

Input arguments must come before output arguments in the kernel definition.
At least one output array is required, but it's ok to have kernels with no inputs.
The new :func:`jax_kernel() <warp.jax_experimental.ffi.jax_kernel>` allows specifying the number of outputs using the
``num_outputs`` argument.
It defaults to one, so this argument is only needed for kernels with multiple outputs.

Here's a kernel with two inputs and one output::

    import jax
    import jax.numpy as jnp

    import warp as wp
    from warp.jax_experimental.ffi import jax_kernel

    @wp.kernel
    def add_kernel(a: wp.array(dtype=int),
                   b: wp.array(dtype=int),
                   output: wp.array(dtype=int)):
        tid = wp.tid()
        output[tid] = a[tid] + b[tid]

    jax_add = jax_kernel(add_kernel)

    @jax.jit
    def f():
        n = 10
        a = jnp.arange(n, dtype=jnp.int32)
        b = jnp.ones(n, dtype=jnp.int32)
        return jax_add(a, b)

    print(f())

One input and two outputs::

    import math

    import jax
    import jax.numpy as jnp

    import warp as wp
    from warp.jax_experimental.ffi import jax_kernel

    @wp.kernel
    def sincos_kernel(angle: wp.array(dtype=float),
                      # outputs
                      sin_out: wp.array(dtype=float),
                      cos_out: wp.array(dtype=float)):
        tid = wp.tid()
        sin_out[tid] = wp.sin(angle[tid])
        cos_out[tid] = wp.cos(angle[tid])

    jax_sincos = jax_kernel(sincos_kernel, num_outputs=2)  # specify multiple outputs

    @jax.jit
    def f():
        a = jnp.linspace(0, 2 * math.pi, 32)
        return jax_sincos(a)

    s, c = f()
    print(s)
    print(c)

Here is a kernel with no inputs that initializes an array of 3x3 matrices with the diagonal values (1, 2, 3).
With no inputs, specifying the launch dimensions is required to determine the shape of the output array::

    @wp.kernel
    def diagonal_kernel(output: wp.array(dtype=wp.mat33)):
        tid = wp.tid()
        output[tid] = wp.mat33(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0)

    jax_diagonal = jax_kernel(diagonal_kernel)

    @jax.jit
    def f():
        # launch dimensions determine the output shape
        return jax_diagonal(launch_dims=4)

    print(f())

Scalar Inputs
.............

Scalar input arguments are supported, although there are some limitations. Currently, scalars passed to Warp kernels must be constant or static values in JAX::

    @wp.kernel
    def scale_kernel(a: wp.array(dtype=float),
                     s: float,  # scalar input
                     output: wp.array(dtype=float)):
        tid = wp.tid()
        output[tid] = a[tid] * s


    jax_scale = jax_kernel(scale_kernel)

    @jax.jit
    def f():
        a = jnp.arange(10, dtype=jnp.float32)
        return jax_scale(a, 2.0)  # ok: constant scalar argument

    print(f())

Trying to use a traced scalar value will result in an exception::

    @jax.jit
    def f(a, s):
        return jax_scale(a, s)  # ERROR: traced scalar argument

    a = jnp.arange(10, dtype=jnp.float32)

    print(f(a, 2.0))

JAX static arguments to the rescue::

    # make scalar arguments static
    @partial(jax.jit, static_argnames=["s"])
    def f(a, s):
        return jax_scale(a, s)  # ok: static scalar argument

    a = jnp.arange(10, dtype=jnp.float32)

    print(f(a, 2.0))


Kernel Launch and Output Dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the launch dimensions are inferred from the shape of the first input array.
When that's not appropriate, the ``launch_dims`` argument can be used to override this behavior.
The launch dimensions also determine the shape of the output arrays.

Here is a simple matrix multiplication kernel that multiplies an NxK matrix by a KxM matrix.
The launch dimensions and output shape must be (N, M), which is different than the shape of the input arrays.

Note that the new :func:`jax_kernel() <warp.jax_experimental.ffi.jax_kernel>` allows specifying custom launch
dimensions with each call, which is more flexible than the old implementation, although the old approach is still supported::

    @wp.kernel
    def matmul_kernel(
        a: wp.array2d(dtype=float),  # NxK input
        b: wp.array2d(dtype=float),  # KxM input
        c: wp.array2d(dtype=float),  # NxM output
    ):
        # launch dimensions should be (N, M)
        i, j = wp.tid()
        N = a.shape[0]
        K = a.shape[1]
        M = b.shape[1]
        if i < N and j < M:
            s = wp.float32(0)
            for k in range(K):
                s += a[i, k] * b[k, j]
            c[i, j] = s

    # no need to specify launch dims here
    jax_matmul = jax_kernel(matmul_kernel)

    @jax.jit
    def f():
        N1, M1, K1 = 3, 4, 2
        a1 = jnp.full((N1, K1), 2, dtype=jnp.float32)
        b1 = jnp.full((K1, M1), 3, dtype=jnp.float32)

        # use custom launch dims
        result1 = jax_matmul(a1, b1, launch_dims=(N1, M1))

        N2, M2, K2 = 4, 3, 2
        a2 = jnp.full((N2, K2), 2, dtype=jnp.float32)
        b2 = jnp.full((K2, M2), 3, dtype=jnp.float32)

        # use custom launch dims
        result2 = jax_matmul(a2, b2, launch_dims=(N2, M2))

        return result1, result2

    r1, r2 = f()
    print(r1)
    print(r2)


By default, output array shapes are determined from the launch dimensions, but it's possible to specify custom output
dimensions using the ``output_dims`` argument. Consider a kernel like this::

    @wp.kernel
    def funky_kernel(a: wp.array(dtype=float),
                     # outputs
                     b: wp.array(dtype=float),
                     c: wp.array(dtype=float)):
        ...

    jax_funky = jax_kernel(funky_kernel, num_outputs=2)

Specify a custom output shape used for all outputs::

    b, c = jax_funky(a, output_dims=n)

Specify different output dimensions for each output using a dictionary::

    b, c = jax_funky(a, output_dims={"b": n, "c": m})

Specify custom launch and output dimensions together::

    b, c = jax_funky(a, launch_dims=k, output_dims={"b": n, "c": m})

One-dimensional shapes can be specified using an integer. Multi-dimensional shapes can be specified using tuples or lists of integers.

Vector and Matrix Arrays
........................

Arrays of Warp vector and matrix types are supported.
Since JAX does not have corresponding data types, the components are packed into extra inner dimensions of JAX arrays.
For example, a Warp array of ``wp.vec3`` will have a JAX array shape of (..., 3) and a Warp array of ``wp.mat22`` will
have a JAX array shape of (..., 2, 2)::

    @wp.kernel
    def vecmat_kernel(a: wp.array(dtype=float),
                      b: wp.array(dtype=wp.vec3),
                      c: wp.array(dtype=wp.mat22),
                      # outputs
                      d: wp.array(dtype=float),
                      e: wp.array(dtype=wp.vec3),
                      f: wp.array(dtype=wp.mat22)):
        ...

    jax_vecmat = jax_kernel(vecmat_kernel, num_outputs=3)

    @jax.jit
    def f():
        n = 10
        a = jnp.zeros(n, dtype=jnp.float32)          # scalar array
        b = jnp.zeros((n, 3), dtype=jnp.float32)     # vec3 array
        c = jnp.zeros((n, 2, 2), dtype=jnp.float32)  # mat22 array

        d, e, f = vecmat_kernel(a, b, c)

It's important to recognize that the Warp and JAX array shapes are different for vector and matrix types.
In the above snippet, Warp sees ``a``, ``b``, and ``c`` as one-dimensional arrays of ``wp.float32``, ``wp.vec3``, and
``wp.mat22``, respectively. In JAX, ``a`` is a one-dimensional array with length n, ``b`` is a two-dimensional array
with shape (n, 3), and ``c`` is a three-dimensional array with shape (n, 2, 2).

When specifying custom output dimensions, it's possible to use either convention. The following calls are equivalent::

    d, e, f = vecmat_kernel(a, b, c, output_dims=n)
    d, e, f = vecmat_kernel(a, b, c, output_dims={"d": n, "e": n, "f": n})
    d, e, f = vecmat_kernel(a, b, c, output_dims={"d": n, "e": (n, 3), "f": (n, 2, 2)})

This is a convenience feature meant to simplify writing code.
For example, when Warp expects the arrays to be of the same shape, we only need to specify the shape once without
worrying about the extra vector and matrix dimensions required by JAX::

    d, e, f = vecmat_kernel(a, b, c, output_dims=n)

On the other hand, JAX dimensions are also accepted to allow passing shapes directly from JAX::

    d, e, f = vecmat_kernel(a, b, c, output_dims={"d": a.shape, "e": b.shape, "f": c.shape})

See `example_jax_kernel.py <https://github.com/NVIDIA/warp/tree/main/warp/examples/interop/example_jax_kernel.py>`_ for examples.


JAX VMAP Support
~~~~~~~~~~~~~~~~

The ``vmap_method`` argument can be used to specify how the callback transforms under :func:`jax.vmap`.
The default is ``"broadcast_all"``. This argument can be passed to :func:`jax_kernel() <warp.jax_experimental.ffi.jax_kernel>`,
and it can also be passed to each call::

    # set default vmap behavior
    jax_callback = jax_kernel(my_kernel, vmap_method="sequential")

    @jax.jit
    def f():
        ...
        b = jax_callback(a)  # uses "sequential"
        ...
        d = jax_callback(c, vmap_method="expand_dims")  # uses "expand_dims"
        ...


Calling Annotated Python Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`jax_kernel() <warp.jax_experimental.ffi.jax_kernel>` mechanism can be used to launch a single Warp kernel
from JAX, but it's also possible to call a Python function that launches multiple kernels.
The target Python function should have argument type annotations as if it were a Warp kernel.
To call this function from JAX, use :func:`jax_callable() <warp.jax_experimental.ffi.jax_callable>`::

    from warp.jax_experimental.ffi import jax_callable

    @wp.kernel
    def scale_kernel(a: wp.array(dtype=float), s: float, output: wp.array(dtype=float)):
        tid = wp.tid()
        output[tid] = a[tid] * s

    @wp.kernel
    def scale_vec_kernel(a: wp.array(dtype=wp.vec2), s: float, output: wp.array(dtype=wp.vec2)):
        tid = wp.tid()
        output[tid] = a[tid] * s


    # The Python function to call.
    # Note the argument type annotations, just like Warp kernels.
    def example_func(
        # inputs
        a: wp.array(dtype=float),
        b: wp.array(dtype=wp.vec2),
        s: float,
        # outputs
        c: wp.array(dtype=float),
        d: wp.array(dtype=wp.vec2),
    ):
        # launch multiple kernels
        wp.launch(scale_kernel, dim=a.shape, inputs=[a, s], outputs=[c])
        wp.launch(scale_vec_kernel, dim=b.shape, inputs=[b, s], outputs=[d])


    jax_func = jax_callable(example_func, num_outputs=2)

    @jax.jit
    def f():
        # inputs
        a = jnp.arange(10, dtype=jnp.float32)
        b = jnp.arange(10, dtype=jnp.float32).reshape((5, 2))  # wp.vec2
        s = 2.0

        # output shapes
        output_dims = {"c": a.shape, "d": b.shape}

        c, d = jax_func(a, b, s, output_dims=output_dims)

        return c, d

    r1, r2 = f()
    print(r1)
    print(r2)

The input and output semantics of :func:`jax_callable() <warp.jax_experimental.ffi.jax_callable>` are similar to
:func:`jax_kernel() <warp.jax_experimental.ffi.jax_kernel>`, so we won't recap everything here,
just focus on the differences:

- :func:`jax_callable() <warp.jax_experimental.ffi.jax_callable>` does not take a ``launch_dims`` argument,
  since the target function is responsible for launching kernels using appropriate dimensions.
- :func:`jax_callable() <warp.jax_experimental.ffi.jax_callable>` takes an optional Boolean
  ``graph_compatible`` argument, which defaults to ``True``.
  This argument determines whether JAX can capture the function in a CUDA graph.
  It is generally desirable, since CUDA graphs can greatly improve the application performance.
  However, if the target function performs operations that are not allowed during graph capture, it may lead to errors.
  This includes any operations that require synchronization with the host. In such cases, pass ``graph_compatible=False``.

See `example_jax_callable.py <https://github.com/NVIDIA/warp/tree/main/warp/examples/interop/example_jax_callable.py>`_ for examples.

.. autofunction:: warp.jax_experimental.ffi.jax_callable

Generic JAX FFI Callbacks
~~~~~~~~~~~~~~~~~~~~~~~~~

Another way to call Python functions is to use
:func:`register_ffi_callback() <warp.jax_experimental.ffi.register_ffi_callback>`::

    from warp.jax_experimental.ffi import register_ffi_callback

This allows calling functions that don't have Warp-style type annotations, but must have the form::

    func(inputs, outputs, attrs, ctx)

where:

- ``inputs`` is a list of input buffers.
- ``outputs`` is a list of output buffers.
- ``attrs`` is a dictionary of attributes.
- ``ctx`` is the execution context, including the CUDA stream.

The input and output buffers are neither JAX nor Warp arrays.
They are objects that expose the ``__cuda_array_interface__``, which can be passed to Warp kernels directly.
Here is an example::

    from warp.jax_experimental.ffi import register_ffi_callback

    @wp.kernel
    def scale_kernel(a: wp.array(dtype=float), s: float, output: wp.array(dtype=float)):
        tid = wp.tid()
        output[tid] = a[tid] * s

    @wp.kernel
    def scale_vec_kernel(a: wp.array(dtype=wp.vec2), s: float, output: wp.array(dtype=wp.vec2)):
        tid = wp.tid()
        output[tid] = a[tid] * s

    # the Python function to call
    def warp_func(inputs, outputs, attrs, ctx):
        # input arrays
        a = inputs[0]
        b = inputs[1]

        # scalar attributes
        s = attrs["scale"]

        # output arrays
        c = outputs[0]
        d = outputs[1]

        device = wp.device_from_jax(get_jax_device())
        stream = wp.Stream(device, cuda_stream=ctx.stream)

        with wp.ScopedStream(stream):
            # launch with arrays of scalars
            wp.launch(scale_kernel, dim=a.shape, inputs=[a, s], outputs=[c])

            # launch with arrays of vec2
            # NOTE: the input shapes are from JAX arrays, so we need to strip the inner dimension for vec2 arrays
            wp.launch(scale_vec_kernel, dim=b.shape[0], inputs=[b, s], outputs=[d])

    # register callback
    register_ffi_callback("warp_func", warp_func)

    n = 10

    # inputs
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.arange(n, dtype=jnp.float32).reshape((n // 2, 2))  # array of wp.vec2
    s = 2.0

    # set up the call
    out_types = [
        jax.ShapeDtypeStruct(a.shape, jnp.float32),
        jax.ShapeDtypeStruct(b.shape, jnp.float32),  # array of wp.vec2
    ]
    call = jax.ffi.ffi_call("warp_func", out_types)

    # call it
    c, d = call(a, b, scale=s)

    print(c)
    print(d)

This is a more low-level approach to JAX FFI callbacks.
A proposal was made to incorporate such a mechanism in JAX, but for now we have a prototype here.
This approach leaves a lot of work up to the user, such as verifying argument types and shapes,
but it can be used when other utilities like :func:`jax_kernel() <warp.jax_experimental.ffi.jax_kernel>` and
:func:`jax_callable() <warp.jax_experimental.ffi.jax_callable>` are not sufficient.

See `example_jax_ffi_callback.py <https://github.com/NVIDIA/warp/tree/main/warp/examples/interop/example_jax_ffi_callback.py>`_ for examples.

.. autofunction:: warp.jax_experimental.ffi.register_ffi_callback

Distributed Computation
^^^^^^^^^^^^^^^^^^^^^^^

Warp can be used in conjunction with JAX's `shard_map <https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html>`__
to perform distributed multi-GPU computations.

To achieve this, the JAX distributed environment must be initialized
(see `Distributed Arrays and Automatic Parallelization <https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html>`__
for more details):

.. code-block:: python

    import jax
    jax.distributed.initialize()

This initialization must be called at the beginning of your program, before any other JAX operations.

Here's an example of how to use ``shard_map`` with a Warp kernel:

.. code-block:: python

    import warp as wp
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from jax.experimental.multihost_utils import process_allgather as allgather
    from jax.experimental.shard_map import shard_map
    from warp.jax_experimental import jax_kernel
    import numpy as np

    # Initialize JAX distributed environment
    jax.distributed.initialize()
    num_gpus = jax.device_count()

    def print_on_process_0(*args, **kwargs):
        if jax.process_index() == 0:
            print(*args, **kwargs)

    print_on_process_0(f"Running on {num_gpus} GPU(s)")

    @wp.kernel
    def multiply_by_two_kernel(
        a_in: wp.array(dtype=wp.float32),
        a_out: wp.array(dtype=wp.float32),
    ):
        index = wp.tid()
        a_out[index] = a_in[index] * 2.0

    jax_warp_multiply = jax_kernel(multiply_by_two_kernel)

    def warp_multiply(x):
        result = jax_warp_multiply(x)
        return result

        # a_in here is the full sharded array with shape (M,)
        # The output will also be a sharded array with shape (M,)
    def warp_distributed_operator(a_in):
        def _sharded_operator(a_in):
            # Inside the sharded operator, a_in is a local shard on each device
            # If we have N devices and input size M, each shard has shape (M/N,)
            
            # warp_multiply applies the Warp kernel to the local shard
            result = warp_multiply(a_in)[0]
            
            # result has the same shape as the input shard (M/N,)
            return result

        # shard_map distributes the computation across devices
        return shard_map(
            _sharded_operator,
            mesh=jax.sharding.Mesh(np.array(jax.devices()), "x"),
            in_specs=(P("x"),),  # Input is sharded along the 'x' axis
            out_specs=P("x"),    # Output is also sharded along the 'x' axis
            check_rep=False,
        )(a_in)

    print_on_process_0("Test distributed multiplication using JAX + Warp")

    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices), "x")
    sharding_spec = jax.sharding.NamedSharding(mesh, P("x"))

    input_size = num_gpus * 5  # 5 elements per device
    single_device_arrays = jnp.arange(input_size, dtype=jnp.float32)

    # Define the shape of the input array based on the total input size
    shape = (input_size,)

    # Create a list of arrays by distributing the single_device_arrays across the available devices
    # Each device will receive a portion of the input data
    arrays = [
        jax.device_put(single_device_arrays[index], d)  # Place each element on the corresponding device
        for d, index in sharding_spec.addressable_devices_indices_map(shape).items()
    ]

    # Combine the individual device arrays into a single sharded array
    sharded_array = jax.make_array_from_single_device_arrays(shape, sharding_spec, arrays)

    # sharded_array has shape (input_size,) but is distributed across devices
    print_on_process_0(f"Input array: {allgather(sharded_array)}")

    # warp_result has the same shape and sharding as sharded_array
    warp_result = warp_distributed_operator(sharded_array)

    # allgather collects results from all devices, resulting in a full array of shape (input_size,)
    print_on_process_0("Warp Output:", allgather(warp_result))

In this example, ``shard_map`` is used to distribute the computation across available devices.
The input array ``a_in`` is sharded along the 'x' axis, and each device processes its local shard.
The Warp kernel ``multiply_by_two_kernel`` is applied to each shard, and the results are combined to form the final output.

This approach allows for efficient parallel processing of large arrays, as each device works on a portion of the data simultaneously.

To run this program on multiple GPUs, you must have Open MPI installed.
You can consult the `OpenMPI installation guide <https://docs.open-mpi.org/en/main/installing-open-mpi/quickstart.html>`__
for instructions on how to install it.
Once Open MPI is installed, you can use ``mpirun`` with the following command:

.. code-block:: bash

    mpirun -np <NUM_OF_GPUS> python <filename>.py


.. _DLPack:

DLPack
------

Warp supports the DLPack protocol included in the Python Array API standard v2022.12.
See the `Python Specification for DLPack <https://dmlc.github.io/dlpack/latest/python_spec.html>`__ for reference.

The canonical way to import an external array into Warp is using the :func:`warp.from_dlpack()` function::

    warp_array = wp.from_dlpack(external_array)

The external array can be a PyTorch tensor, Jax array, or any other array type compatible with this version of the DLPack protocol.
For CUDA arrays, this approach requires the producer to perform stream synchronization which ensures that operations on the array
are ordered correctly.  The :func:`warp.from_dlpack()` function asks the producer to synchronize the current Warp stream on the device where
the array resides.  Thus it should be safe to use the array in Warp kernels on that device without any additional synchronization.

The canonical way to export a Warp array to an external framework is to use the ``from_dlpack()`` function in that framework::

    jax_array = jax.dlpack.from_dlpack(warp_array)
    torch_tensor = torch.utils.dlpack.from_dlpack(warp_array)
    paddle_tensor = paddle.utils.dlpack.from_dlpack(warp_array)

For CUDA arrays, this will synchronize the current stream of the consumer framework with the current Warp stream on the array's device.
Thus it should be safe to use the wrapped array in the consumer framework, even if the array was previously used in a Warp kernel
on the device.

Alternatively, arrays can be shared by explicitly creating PyCapsules using a ``to_dlpack()`` function provided by the producer framework.
This approach may be used for older versions of frameworks that do not support the v2022.12 standard::

    warp_array1 = wp.from_dlpack(jax.dlpack.to_dlpack(jax_array))
    warp_array2 = wp.from_dlpack(torch.utils.dlpack.to_dlpack(torch_tensor))
    warp_array3 = wp.from_dlpack(paddle.utils.dlpack.to_dlpack(paddle_tensor))

    jax_array = jax.dlpack.from_dlpack(wp.to_dlpack(warp_array))
    torch_tensor = torch.utils.dlpack.from_dlpack(wp.to_dlpack(warp_array))
    paddle_tensor = paddle.utils.dlpack.from_dlpack(wp.to_dlpack(warp_array))

This approach is generally faster because it skips any stream synchronization, but another solution must be used to ensure correct
ordering of operations.  In situations where no synchronization is required, using this approach can yield better performance.
This may be a good choice in situations like these:

- The external framework is using the synchronous CUDA default stream.
- Warp and the external framework are using the same CUDA stream.
- Another synchronization mechanism is already in place.

.. autofunction:: warp.from_dlpack
.. autofunction:: warp.to_dlpack

.. _paddle-interop:

Paddle
------

Warp provides helper functions to convert arrays to/from Paddle::

    w = wp.array([1.0, 2.0, 3.0], dtype=float, device="cpu")

    # convert to Paddle tensor
    t = wp.to_paddle(w)

    # convert from Paddle tensor
    w = wp.from_paddle(t)

These helper functions allow the conversion of Warp arrays to/from Paddle tensors without copying the underlying data.
At the same time, if available, gradient arrays and tensors are converted to/from Paddle autograd tensors, allowing the use of Warp arrays
in Paddle autograd computations.

.. autofunction:: warp.from_paddle
.. autofunction:: warp.to_paddle
.. autofunction:: warp.device_from_paddle
.. autofunction:: warp.device_to_paddle
.. autofunction:: warp.dtype_from_paddle
.. autofunction:: warp.dtype_to_paddle

To convert a Paddle CUDA stream to a Warp CUDA stream and vice versa, Warp provides the following function:

.. autofunction:: warp.stream_from_paddle

Example: Optimization using ``warp.from_paddle()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example usage of minimizing a loss function over an array of 2D points written in Warp via Paddle's Adam optimizer
using :func:`warp.from_paddle` is as follows::

    import warp as wp
    import paddle

    # init warp context at beginning
    wp.context.init()

    @wp.kernel()
    def loss(xs: wp.array(dtype=float, ndim=2), l: wp.array(dtype=float)):
        tid = wp.tid()
        wp.atomic_add(l, 0, xs[tid, 0] ** 2.0 + xs[tid, 1] ** 2.0)

    # indicate requires_grad so that Warp can accumulate gradients in the grad buffers
    xs = paddle.randn([100, 2])
    xs.stop_gradient = False
    l = paddle.zeros([1])
    l.stop_gradient = False
    opt = paddle.optimizer.Adam(learning_rate=0.1, parameters=[xs])

    wp_xs = wp.from_paddle(xs)
    wp_l = wp.from_paddle(l)

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

Example: Optimization using ``warp.to_paddle``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Less code is needed when we declare the optimization variables directly in Warp and use :func:`warp.to_paddle` to convert them to Paddle tensors.
Here, we revisit the same example from above where now only a single conversion to a Paddle tensor is needed to supply Adam with the optimization variables::

    import warp as wp
    import numpy as np
    import paddle

    # init warp context at beginning
    wp.context.init()

    @wp.kernel()
    def loss(xs: wp.array(dtype=float, ndim=2), l: wp.array(dtype=float)):
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

Performance Notes
^^^^^^^^^^^^^^^^^

The :func:`wp.from_paddle() <warp.from_paddle>` function creates a Warp array object that shares data with a Paddle tensor.
Although this function does not copy the data, there is always some CPU overhead during the conversion.
If these conversions happen frequently, the overall program performance may suffer.
As a general rule, it's good to avoid repeated conversions of the same tensor.
Instead of:

.. code:: python

    x_t = paddle.arange(n, dtype=paddle.float32).to(device=wp.device_to_paddle(device))
    y_t = paddle.ones([n], dtype=paddle.float32).to(device=wp.device_to_paddle(device))

    for i in range(10):
        x_w = wp.from_paddle(x_t)
        y_w = wp.from_paddle(y_t)
        wp.launch(saxpy, dim=n, inputs=[x_w, y_w, 1.0], device=device)

Try converting the arrays only once and reuse them:

.. code:: python

    x_t = paddle.arange(n, dtype=paddle.float32).to(device=wp.device_to_paddle(device))
    y_t = paddle.ones([n], dtype=paddle.float32).to(device=wp.device_to_paddle(device))

    x_w = wp.from_paddle(x_t)
    y_w = wp.from_paddle(y_t)

    for i in range(10):
        wp.launch(saxpy, dim=n, inputs=[x_w, y_w, 1.0], device=device)

If reusing arrays is not possible (e.g., a new Paddle tensor is constructed on every iteration), passing ``return_ctype=True`` to
:func:`wp.from_paddle() <warp.from_paddle>` should yield faster performance.
Setting this argument to ``True`` avoids constructing a ``wp.array`` object and instead returns a low-level array descriptor.
This descriptor is a simple C structure that can be passed to Warp kernels instead of a ``wp.array``, but cannot be used in other places that require a ``wp.array``.

.. code:: python

    for n in range(1, 10):
        # get Paddle tensors for this iteration
        x_t = paddle.arange(n, dtype=paddle.float32).to(device=wp.device_to_paddle(device))
        y_t = paddle.ones([n], dtype=paddle.float32).to(device=wp.device_to_paddle(device))

        # get Warp array descriptors
        x_ctype = wp.from_paddle(x_t, return_ctype=True)
        y_ctype = wp.from_paddle(y_t, return_ctype=True)

        wp.launch(saxpy, dim=n, inputs=[x_ctype, y_ctype, 1.0], device=device)

An alternative approach is to pass the Paddle tensors to Warp kernels directly.
This avoids constructing temporary Warp arrays by leveraging standard array interfaces (like ``__cuda_array_interface__``) supported by both Paddle and Warp.
The main advantage of this approach is convenience, since there is no need to call any conversion functions.
The main limitation is that it does not handle gradients, because gradient information is not included in the standard array interfaces.
This technique is therefore most suitable for algorithms that do not involve differentiation.

.. code:: python

    x = paddle.arange(n, dtype=paddle.float32).to(device=wp.device_to_paddle(device))
    y = paddle.ones([n], dtype=paddle.float32).to(device=wp.device_to_paddle(device))

    for i in range(10):
        wp.launch(saxpy, dim=n, inputs=[x, y, 1.0], device=device)

.. code:: shell

    python -m warp.examples.benchmarks.benchmark_interop_paddle

Sample output:

.. code::

    13990 ms  from_paddle(...)
     5990 ms  from_paddle(..., return_ctype=True)
    35167 ms  direct from paddle

The default ``wp.from_paddle()`` conversion is the slowest.
Passing ``return_ctype=True`` is the fastest, because it skips creating temporary Warp array objects.
Passing Paddle tensors to Warp kernels directly falls somewhere in between.
It skips creating temporary Warp arrays, but accessing the ``__cuda_array_interface__`` attributes of Paddle tensors adds overhead because they are initialized on-demand.
