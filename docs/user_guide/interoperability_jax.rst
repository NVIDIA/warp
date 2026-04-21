.. _jax-interop:

JAX Interoperability
====================

Introduction
------------

Interoperability with JAX arrays is supported through the following methods.
Internally these use the DLPack protocol to exchange data in a zero-copy way with JAX::

    warp_array = wp.from_jax(jax_array)
    jax_array = wp.to_jax(warp_array)

It may be preferable to use the :ref:`DLPack` protocol directly for better performance and control over stream synchronization.

.. _jax-ffi:
.. _jax-kernel:

Using Warp Kernels as JAX Primitives
------------------------------------

Warp kernels can be used as JAX primitives, which allows calling them inside of jitted JAX functions::

    import warp as wp
    import jax
    import jax.numpy as jnp

    from warp.jax_experimental import jax_kernel

    @wp.kernel
    def triple_kernel(input: wp.array[float], output: wp.array[float]):
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



Input and Output Semantics
^^^^^^^^^^^^^^^^^^^^^^^^^^

Input arguments must come before output arguments in the kernel definition.
At least one output array is required, but it's ok to have kernels with no inputs.
The number of outputs can be specified using the ``num_outputs`` argument, which defaults to one.

Here's a kernel with two inputs and one output::

    import jax
    import jax.numpy as jnp

    import warp as wp
    from warp.jax_experimental import jax_kernel

    @wp.kernel
    def add_kernel(a: wp.array[int],
                   b: wp.array[int],
                   output: wp.array[int]):
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
    from warp.jax_experimental import jax_kernel

    @wp.kernel
    def sincos_kernel(angle: wp.array[float],
                      # outputs
                      sin_out: wp.array[float],
                      cos_out: wp.array[float]):
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
    def diagonal_kernel(output: wp.array[wp.mat33]):
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
    def scale_kernel(a: wp.array[float],
                     s: float,  # scalar input
                     output: wp.array[float]):
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

    from functools import partial

    # make scalar arguments static
    @partial(jax.jit, static_argnames=["s"])
    def f(a, s):
        return jax_scale(a, s)  # ok: static scalar argument

    a = jnp.arange(10, dtype=jnp.float32)

    print(f(a, 2.0))


Kernel Launch and Output Dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the launch dimensions are inferred from the shape of the first input array.
When that's not appropriate, the ``launch_dims`` argument can be used to override this behavior.
The launch dimensions also determine the shape of the output arrays.

Here is a simple matrix multiplication kernel that multiplies an NxK matrix by a KxM matrix.
The launch dimensions and output shape must be (N, M), which is different than the shape of the input arrays::

    @wp.kernel
    def matmul_kernel(
        a: wp.array2d[float],  # NxK input
        b: wp.array2d[float],  # KxM input
        c: wp.array2d[float],  # NxM output
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
    def funky_kernel(a: wp.array[float],
                     # outputs
                     b: wp.array[float],
                     c: wp.array[float]):
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
For example, a Warp array of :class:`wp.vec3 <warp.vec3>` will have a JAX array shape of (..., 3)
and a Warp array of :class:`wp.mat22 <warp.mat22>` will have a JAX array shape of (..., 2, 2):

.. code-block:: python

    @wp.kernel
    def vecmat_kernel(a: wp.array[float],
                      b: wp.array[wp.vec3],
                      c: wp.array[wp.mat22],
                      # outputs
                      d: wp.array[float],
                      e: wp.array[wp.vec3],
                      f: wp.array[wp.mat22]):
        ...

    jax_vecmat = jax_kernel(vecmat_kernel, num_outputs=3)

    @jax.jit
    def f():
        n = 10
        a = jnp.zeros(n, dtype=jnp.float32)          # scalar array
        b = jnp.zeros((n, 3), dtype=jnp.float32)     # vec3 array
        c = jnp.zeros((n, 2, 2), dtype=jnp.float32)  # mat22 array

        d, e, f = jax_vecmat(a, b, c)

It's important to recognize that the Warp and JAX array shapes are different for vector and matrix types.
In the above snippet, Warp sees ``a``, ``b``, and ``c`` as one-dimensional arrays of :class:`wp.float32 <warp.float32>`,
:class:`wp.vec3 <warp.vec3>`, and :class:`wp.mat22 <warp.mat22>`, respectively.
In JAX, ``a`` is a one-dimensional array with length ``n``, ``b`` is a two-dimensional array
with shape ``(n, 3)``, and ``c`` is a three-dimensional array with shape ``(n, 2, 2)``.

When specifying custom output dimensions, it's possible to use either convention. The following calls are equivalent::

    d, e, f = jax_vecmat(a, b, c, output_dims=n)
    d, e, f = jax_vecmat(a, b, c, output_dims={"d": n, "e": n, "f": n})
    d, e, f = jax_vecmat(a, b, c, output_dims={"d": n, "e": (n, 3), "f": (n, 2, 2)})

This is a convenience feature meant to simplify writing code.
For example, when Warp expects the arrays to be of the same shape, we only need to specify the shape once without
worrying about the extra vector and matrix dimensions required by JAX::

    d, e, f = jax_vecmat(a, b, c, output_dims=n)

On the other hand, JAX dimensions are also accepted to allow passing shapes directly from JAX::

    d, e, f = jax_vecmat(a, b, c, output_dims={"d": a.shape, "e": b.shape, "f": c.shape})

See `example_jax_kernel.py <https://github.com/NVIDIA/warp/blob/main/warp/examples/interop/example_jax_kernel.py>`_ for examples.

.. _jax-vmap:

VMAP Support
------------

The ``vmap_method`` argument can be used to specify how the callback transforms under :func:`jax.vmap`.
The default is ``"broadcast_all"``.
This argument can be passed to :func:`jax_kernel() <warp.jax_experimental.ffi.jax_kernel>`,
and it can also be passed to each call:

.. code-block:: python

    # set default vmap behavior
    jax_callback = jax_kernel(my_kernel, vmap_method="sequential")

    @jax.jit
    def f():
        ...
        b = jax_callback(a)  # uses "sequential"
        ...
        d = jax_callback(c, vmap_method="expand_dims")  # uses "expand_dims"
        ...

Basic VMAP Example
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import warp as wp
    from warp.jax_experimental import jax_kernel

    import jax
    import jax.numpy as jnp

    @wp.kernel
    def add_kernel(a: wp.array[float], b: wp.array[float], output: wp.array[float]):
        tid = wp.tid()
        output[tid] = a[tid] + b[tid]

    jax_add = jax_kernel(add_kernel)

    # batched inputs
    a = jnp.arange(3 * 4, dtype=jnp.float32).reshape((3, 4))
    b = jnp.ones(3 * 4, dtype=jnp.float32).reshape((3, 4))

    (output,) = jax.jit(jax.vmap(jax_add))(a, b)
    print(output)


VMAP Example with In-Out Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider the following Warp kernel that sums the rows of a matrix:

.. code-block:: python

    @wp.kernel
    def rowsum_kernel(matrix: wp.array2d[float], sums: wp.array1d[float]):
        i, j = wp.tid()
        wp.atomic_add(sums, i, matrix[i, j])

Note that ``sums`` is an in-out argument that should be initialized to zero prior to launch:

.. code-block:: python

    jax_rowsum = jax_kernel(rowsum_kernel, in_out_argnames=["sums"])

    # batched input with shape (2, 3, 4)
    matrices = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape((2, 3, 4))

    # vmap with batch dim 0: input 2 matrices with shape (3, 4), output shape (2, 3)
    sums = jnp.zeros((2, 3), dtype=jnp.float32)
    (output,) = jax.jit(jax.vmap(jax_rowsum, in_axes=(0, 0)))(matrices, sums)

    # vmap with batch dim 1: input 3 matrices with shape (2, 4), output shape (3, 2)
    sums = jnp.zeros((3, 2), dtype=jnp.float32)
    (output,) = jax.jit(jax.vmap(jax_rowsum, in_axes=(1, 0)))(matrices, sums)

    # vmap with batch dim 2: input 4 matrices with shape (2, 3), output shape (4, 2)
    sums = jnp.zeros((4, 2), dtype=jnp.float32)
    (output,) = jax.jit(jax.vmap(jax_rowsum, in_axes=(2, 0)))(matrices, sums)

VMAP Example with Custom Launch and Output Dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is a kernel that looks up values in a table given the indices:

.. code-block:: python

    @wp.kernel
    def lookup_kernel(table: wp.array[float], indices: wp.array[int], output: wp.array[float]):
        i = wp.tid()
        output[i] = table[indices[i]]

The table itself is not batched, but we will provide batches of indices. By default, ``jax_kernel()`` infers the launch dimensions and output shape from the shape of the first array argument, but in this case the kernel launch dimensions should correspond to the shape of the ``indices`` array. We will need to pass custom ``launch_dims`` when calling the kernel. In order to pass this keyword argument through vmap, we will use ``functools.partial()``.

.. code-block:: python

    from functools import partial

    jax_lookup = jax_kernel(lookup_kernel)

    # lookup table (not batched)
    N = 100
    table = jnp.arange(N, dtype=jnp.float32)

    # batched indices to look up
    key = jax.random.key(42)
    indices = jax.random.randint(key, (20, 50), 0, N, dtype=jnp.int32)

    # vmap with batch dim 0: input 20 sets of 50 indices each, output shape (20, 50)
    (output,) = jax.jit(jax.vmap(partial(jax_lookup, launch_dims=50), in_axes=(None, 0)))(
        table, indices
    )

    # vmap with batch dim 1: input 50 sets of 20 indices each, output shape (50, 20)
    (output,) = jax.jit(jax.vmap(partial(jax_lookup, launch_dims=20), in_axes=(None, 1)))(
        table, indices
    )

Note that ``launch_dims`` should NOT include the batch dimension - batching will be handled automatically. The same is true when passing ``output_dims`` to ``jax_kernel()`` and ``jax_callable()``.

.. _jax-autodiff:

Automatic Differentiation
-------------------------

Warp kernels can be given JAX gradients using a convenience wrapper that wires a custom VJP around a kernel and its adjoint.
To enable autodiff, pass the ``enable_backward=True`` argument to :func:`jax_kernel() <warp.jax_experimental.ffi.jax_kernel>`.

Basic example (one output)::

    from functools import partial
    import jax
    import jax.numpy as jnp
    import warp as wp
    from warp.jax_experimental import jax_kernel

    @wp.kernel
    def scale_sum_square(
        a: wp.array[float],
        b: wp.array[float],
        s: float,
        out: wp.array[float],
    ):
        tid = wp.tid()
        out[tid] = (a[tid] * s + b[tid]) ** 2.0

    jax_scale = jax_kernel(scale_sum_square, num_outputs=1, enable_backward=True)

    # scalars must be static
    @partial(jax.jit, static_argnames=["s"])
    def loss(a, b, s):
        (out,) = jax_scale(a, b, s)
        return jnp.sum(out)

    n = 16
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.ones(n, dtype=jnp.float32)
    s = 2.0

    # gradients w.r.t. array inputs
    da, db = jax.grad(loss, argnums=(0, 1))(a, b, s)
    print(da)
    print(db)


Multiple outputs::

    import jax
    import jax.numpy as jnp
    import warp as wp
    from warp.jax_experimental import jax_kernel

    @wp.kernel
    def multi_output(
        a: wp.array[float],
        b: wp.array[float],
        s: float,
        c: wp.array[float],
        d: wp.array[float],
    ):
        tid = wp.tid()
        c[tid] = a[tid] ** 2.0
        d[tid] = a[tid] * b[tid] * s

    jax_multi = jax_kernel(multi_output, num_outputs=2, enable_backward=True)

    def caller(fn, a, b, s):
        c, d = fn(a, b, s)
        return jnp.sum(c + d)

    n = 16
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.ones(n, dtype=jnp.float32)
    s = 2.0

    # differentiate a batched scalar objective over two inputs
    da, db = jax.grad(lambda a, b, s: caller(jax_multi, a, b, s), argnums=(0, 1))(a, b, s)
    print(da)
    print(db)

Vector and matrix arrays also work. Inner component dimensions are packed in the JAX array and handled automatically::

    from functools import partial
    import jax
    import jax.numpy as jnp
    import warp as wp
    from warp.jax_experimental import jax_kernel

    @wp.kernel
    def scale_vec2(a: wp.array[wp.vec2], s: float, out: wp.array[wp.vec2]):
        tid = wp.tid()
        out[tid] = a[tid] * s

    jax_vec = jax_kernel(scale_vec2, num_outputs=1, enable_backward=True)

    @partial(jax.jit, static_argnames=["s"])
    def vec_loss(a, s):
        (out,) = jax_vec(a, s)
        return jnp.sum(out)

    a2 = jnp.arange(10, dtype=jnp.float32).reshape((5, 2))  # vec2 payload shape
    (da2,) = jax.grad(vec_loss, argnums=(0,))(a2, 3.0)
    print(da2)

Limitations
^^^^^^^^^^^

The autodiff functionality is considered experimental and is still a work in progress.

- Scalar inputs must be static arguments in JAX.
- Gradients are returned for differentiable array inputs (static scalars are excluded from the gradient tuple).
- Input-output arguments (``in_out_argnames``) are not supported when ``enable_backward=True``, because in-place modifications are not differentiable.
- ``output_dims`` is not currently supported when ``enable_backward=True`` (this requires separate output-buffer allocation logic; it is planned as a follow-up).
- ``launch_dims`` **is** supported when ``enable_backward=True``. The captured value is used by both the forward and the adjoint launches. This is required for correct gradient values when the input array has more dimensions than the kernel's ``wp.tid()`` iteration space (for example, an LBM distribution ``(Q, nx, ny, nz)`` with a 3-D spatial ``tid`` or a batched volume ``(B, x, y, z)``). If ``launch_dims`` is omitted, the dimensions are inferred from the shape of the first array argument as before.

Computing gradients with custom launch dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When an input array has more dimensions than the kernel's ``wp.tid()``
iteration space, ``launch_dims`` can be specified together with
``enable_backward=True``. The same ``launch_dims`` is used for both the
forward launch and the adjoint launch, so gradients computed via
``jax.grad`` are scaled correctly.

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import warp as wp
    from warp.jax_experimental import jax_kernel

    wp.init()


    @wp.kernel
    def scale(
        a: wp.array4d(dtype=wp.float32),
        b: wp.array4d(dtype=wp.float32),
    ):
        # tid is 3-D; the outer axis is iterated inside the kernel.
        i, j, k = wp.tid()
        for m in range(a.shape[0]):
            b[m, i, j, k] = a[m, i, j, k] * 2.0


    jax_func = jax_kernel(
        scale,
        num_outputs=1,
        launch_dims=(16, 16, 16),   # spatial dimensions only
        enable_backward=True,
    )

    a = jnp.ones((8, 16, 16, 16), dtype=jnp.float32)
    grad = jax.grad(lambda x: jnp.sum(jax_func(x)[0]))(a)
    # grad is 2.0 everywhere (analytical gradient)

If ``launch_dims`` is omitted, Warp infers it from the shape of the first
input array. For scalar-dtype arrays with ``array.ndim > kernel.tid_ndim``
the inferred shape includes the outer axis. For kernels whose per-location
write is idempotent (for example, ``b[i] = f(a[i])``), the forward kernel
still produces the correct value; kernels using non-idempotent patterns
(e.g. atomic accumulation into the output) may also see wrong forward
values. In either case, the adjoint kernel over-accumulates gradients by
a factor of the outer axis size via ``atomic_add``. Passing
``launch_dims`` explicitly is the recommended way to avoid this.

.. _jax-callable:

``jax_callable`` for Multi-Kernel Functions
-------------------------------------------

The :func:`jax_kernel() <warp.jax_experimental.ffi.jax_kernel>` mechanism can be used to launch a single Warp kernel
from JAX, but it's also possible to call a Python function that launches multiple kernels.
The target Python function should have argument type annotations as if it were a Warp kernel.
To call this function from JAX, use :func:`jax_callable() <warp.jax_experimental.ffi.jax_callable>`::

    from warp.jax_experimental import jax_callable

    @wp.kernel
    def scale_kernel(a: wp.array[float], s: float, output: wp.array[float]):
        tid = wp.tid()
        output[tid] = a[tid] * s

    @wp.kernel
    def scale_vec_kernel(a: wp.array[wp.vec2], s: float, output: wp.array[wp.vec2]):
        tid = wp.tid()
        output[tid] = a[tid] * s


    # The Python function to call.
    # Note the argument type annotations, just like Warp kernels.
    def example_func(
        # inputs
        a: wp.array[float],
        b: wp.array[wp.vec2],
        s: float,
        # outputs
        c: wp.array[float],
        d: wp.array[wp.vec2],
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
- :func:`jax_callable() <warp.jax_experimental.ffi.jax_callable>` takes an optional ``graph_mode`` argument, which determines how the callable can be captured in a CUDA graph.
  Graphs are generally desirable, since they can greatly improve the application performance.
  ``GraphMode.JAX`` (default) lets JAX capture the graph, which may be used as a subgraph in an enclosing capture for maximal benefit.
  ``GraphMode.WARP`` lets Warp capture the graph. Use this mode when the callable cannot be used as a subgraph, such as when the callable uses conditional graph nodes.
  ``GraphMode.NONE`` disables graph capture. Use this mode if the callable performs operations that are not allowed during graph capture, such as host synchronization.

See `example_jax_callable.py <https://github.com/NVIDIA/warp/blob/main/warp/examples/interop/example_jax_callable.py>`_ for examples.

.. _jax-ffi-callbacks:

Generic FFI Callbacks
---------------------

Another way to call Python functions is to use
:func:`register_ffi_callback() <warp.jax_experimental.ffi.register_ffi_callback>`::

    from warp.jax_experimental import register_ffi_callback

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

    import jax

    from warp.jax_experimental import register_ffi_callback

    @wp.kernel
    def scale_kernel(a: wp.array[float], s: float, output: wp.array[float]):
        tid = wp.tid()
        output[tid] = a[tid] * s

    @wp.kernel
    def scale_vec_kernel(a: wp.array[wp.vec2], s: float, output: wp.array[wp.vec2]):
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

        device = wp.device_from_jax(jax.local_devices()[0])
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

See `example_jax_ffi_callback.py <https://github.com/NVIDIA/warp/blob/main/warp/examples/interop/example_jax_ffi_callback.py>`_ for examples.

.. _jax-shard-map:

Distributed Computation with ``shard_map``
------------------------------------------

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
        a_in: wp.array[float],
        a_out: wp.array[float],
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
