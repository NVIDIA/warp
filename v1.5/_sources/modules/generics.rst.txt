Generics
========

.. currentmodule:: warp

Warp supports writing generic kernels and functions, which act as templates that can be instantiated with different concrete types.
This allows you to write code once and reuse it with multiple data types.
The concepts discussed on this page also apply to :ref:`Runtime Kernel Creation`.

Generic Kernels
---------------

Generic kernel definition syntax is the same as regular kernels, but you can use ``typing.Any`` in place of concrete types:

.. testcode::

    from typing import Any

    # generic kernel definition using Any as a placeholder for concrete types
    @wp.kernel
    def scale(x: wp.array(dtype=Any), s: Any):
        i = wp.tid()
        x[i] = s * x[i]

    data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    n = len(data)

    x16 = wp.array(data, dtype=wp.float16)
    x32 = wp.array(data, dtype=wp.float32)
    x64 = wp.array(data, dtype=wp.float64)

    # run the generic kernel with different data types
    wp.launch(scale, dim=n, inputs=[x16, wp.float16(3)])
    wp.launch(scale, dim=n, inputs=[x32, wp.float32(3)])
    wp.launch(scale, dim=n, inputs=[x64, wp.float64(3)])

    print(x16)
    print(x32)
    print(x64)

.. testoutput::

    [ 3.  6.  9. 12. 15. 18. 21. 24. 27.]
    [ 3.  6.  9. 12. 15. 18. 21. 24. 27.]
    [ 3.  6.  9. 12. 15. 18. 21. 24. 27.]

Under the hood, Warp will automatically generate new instances of the generic kernel to match the given argument types.


Type Inference
~~~~~~~~~~~~~~

When a generic kernel is being launched, Warp infers the concrete types from the arguments.
:func:`wp.launch() <launch>` handles generic kernels without any special syntax, but we should be mindful of the data types passed as arguments to make sure that the correct types are inferred:

* Scalars can be passed as regular Python numeric values (e.g., ``42`` or ``0.5``).  Python integers are interpreted as ``wp.int32`` and Python floating point values are interpreted as ``wp.float32``.  To specify a different data type and to avoid ambiguity, Warp data types should be used instead (e.g., ``wp.int64(42)`` or ``wp.float16(0.5)``).
* Vectors and matrices should be passed as Warp types rather than tuples or lists (e.g., ``wp.vec3f(1.0, 2.0, 3.0)`` or ``wp.mat22h([[1.0, 0.0], [0.0, 1.0]])``).
* Warp arrays and structs can be passed normally.

.. _implicit_instantiation:

Implicit Instantiation
~~~~~~~~~~~~~~~~~~~~~~

When you launch a generic kernel with a new set of data types, Warp automatically creates a new instance of this kernel with the given types.  This is convenient, but there are some downsides to this implicit instantiation.

Consider these three generic kernel launches:

.. code:: python

    wp.launch(scale, dim=n, inputs=[x16, wp.float16(3)])
    wp.launch(scale, dim=n, inputs=[x32, wp.float32(3)])
    wp.launch(scale, dim=n, inputs=[x64, wp.float64(3)])

During each one of these launches, a new kernel instance is being generated, which forces the module to be reloaded.  You might see something like this in the output:

.. code:: text

    Module __main__ load on device 'cuda:0' took 170.37 ms
    Module __main__ load on device 'cuda:0' took 171.43 ms
    Module __main__ load on device 'cuda:0' took 179.49 ms

This leads to a couple of potential problems:

* The overhead of repeatedly rebuilding the modules can impact the overall performance of the program.
* Module reloading during graph capture is not allowed on older CUDA drivers, which will cause captures to fail.

Explicit instantiation can be used to overcome these issues.


.. _explicit_instantiation:

Explicit Instantiation
~~~~~~~~~~~~~~~~~~~~~~

Warp allows explicitly declaring instances of generic kernels with different types.  One way is to use the ``@wp.overload`` decorator:

.. code:: python

    @wp.overload
    def scale(x: wp.array(dtype=wp.float16), s: wp.float16):
        ...

    @wp.overload
    def scale(x: wp.array(dtype=wp.float32), s: wp.float32):
        ...

    @wp.overload
    def scale(x: wp.array(dtype=wp.float64), s: wp.float64):
        ...

    wp.launch(scale, dim=n, inputs=[x16, wp.float16(3)])
    wp.launch(scale, dim=n, inputs=[x32, wp.float32(3)])
    wp.launch(scale, dim=n, inputs=[x64, wp.float64(3)])

The ``@wp.overload`` decorator allows redeclaring generic kernels without repeating the kernel code.  The kernel body is just replaced with the ellipsis (``...``).  Warp keeps track of known overloads for each kernel, so if an overload exists it will not be instantiated again.  If all the overloads are declared prior to kernel launches, the module will only load once with all the kernel instances in place.

We can also use :func:`wp.overload() <overload>` as a function for a slightly more concise syntax.  We just need to specify the generic kernel and a list of concrete argument types:

.. code:: python

    wp.overload(scale, [wp.array(dtype=wp.float16), wp.float16])
    wp.overload(scale, [wp.array(dtype=wp.float32), wp.float32])
    wp.overload(scale, [wp.array(dtype=wp.float64), wp.float64])

Instead of an argument list, a dictionary can also be provided:

.. code:: python

    wp.overload(scale, {"x": wp.array(dtype=wp.float16), "s": wp.float16})
    wp.overload(scale, {"x": wp.array(dtype=wp.float32), "s": wp.float32})
    wp.overload(scale, {"x": wp.array(dtype=wp.float64), "s": wp.float64})

A dictionary might be preferred for readability.  With dictionaries, only generic arguments need to be specified, which can be even more concise when overloading kernels where some of the arguments are not generic.

We can easily create overloads in a single loop, like this:

.. code:: python

    for T in [wp.float16, wp.float32, wp.float64]:
        wp.overload(scale, [wp.array(dtype=T), T])

Finally, the :func:`wp.overload() <overload>` function returns the concrete kernel instance, which can be saved in a variable:

.. code:: python

    scale_f16 = wp.overload(scale, [wp.array(dtype=wp.float16), wp.float16])
    scale_f32 = wp.overload(scale, [wp.array(dtype=wp.float32), wp.float32])
    scale_f64 = wp.overload(scale, [wp.array(dtype=wp.float64), wp.float64])

These instances are treated as regular kernels, not generic.  This means that launches should be faster, because Warp doesn't need to infer data types from the arguments like it does when launching generic kernels.  The typing requirements for kernel arguments are also more relaxed than with generic kernels, because Warp can convert scalars, vectors, and matrices to the known required types.

.. code:: python

    # launch concrete kernel instances
    wp.launch(scale_f16, dim=n, inputs=[x16, 3])
    wp.launch(scale_f32, dim=n, inputs=[x32, 3])
    wp.launch(scale_f64, dim=n, inputs=[x64, 3])

.. autofunction:: overload

.. _Generic Functions:

Generic Functions
-----------------

Like Warp kernels, we can also define generic Warp functions:

.. testcode::

    # generic function
    @wp.func
    def f(x: Any):
        return x * x

    # use generic function in a regular kernel
    @wp.kernel
    def square_float(a: wp.array(dtype=float)):
        i = wp.tid()
        a[i] = f(a[i])

    # use generic function in a generic kernel
    @wp.kernel
    def square_any(a: wp.array(dtype=Any)):
        i = wp.tid()
        a[i] = f(a[i])

    data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    n = len(data)

    af = wp.array(data, dtype=float)
    ai = wp.array(data, dtype=int)

    # launch regular kernel
    wp.launch(square_float, dim=n, inputs=[af])
    print(af)

    # launch generic kernel
    wp.launch(square_any, dim=n, inputs=[af])
    print(af)

    wp.launch(square_any, dim=n, inputs=[ai])
    print(ai)

.. testoutput::

    [ 1.  4.  9. 16. 25. 36. 49. 64. 81.]
    [1.000e+00 1.600e+01 8.100e+01 2.560e+02 6.250e+02 1.296e+03 2.401e+03
     4.096e+03 6.561e+03]
    [ 1  4  9 16 25 36 49 64 81]

A generic function can be used in regular and generic kernels.  It's not necessary to explicitly overload generic functions.  All required function overloads are generated automatically when those functions are used in kernels.


type() Operator
---------------

Consider the following generic function:

.. code:: python

    @wp.func
    def triple(x: Any):
        return 3 * x

Using numeric literals like ``3`` is problematic in generic expressions due to Warp's strict typing rules.  Operands in arithmetic expressions must have the same data types, but integer literals are always treated as ``wp.int32``.  This function will fail to compile if ``x`` has a data type other than ``wp.int32``, which means that it's not generic at all.

The ``type()`` operator comes to the rescue here.  The ``type()`` operator returns the type of its argument, which is handy in generic functions or kernels where the data types are not known in advance.  We can rewrite the function like this to make it work with a wider range of types:

.. code:: python

    @wp.func
    def triple(x: Any):
        return type(x)(3) * x

The ``type()`` operator is useful for type conversions in Warp kernels and functions.  For example, here is a simple generic ``arange()`` kernel:

.. code:: python

    @wp.kernel
    def arange(a: wp.array(dtype=Any)):
        i = wp.tid()
        a[i] = type(a[0])(i)

    n = 10
    ai = wp.empty(n, dtype=wp.int32)
    af = wp.empty(n, dtype=wp.float32)

    wp.launch(arange, dim=n, inputs=[ai])
    wp.launch(arange, dim=n, inputs=[af])

``wp.tid()`` returns an integer, but the value gets converted to the array's data type before storing it in the array.  Alternatively, we could write our ``arange()`` kernel like this:

.. code:: python

    @wp.kernel
    def arange(a: wp.array(dtype=Any)):
        i = wp.tid()
        a[i] = a.dtype(i)

This variant uses the ``array.dtype()`` operator, which returns the type of the array's contents.


Limitations and Rough Edges
---------------------------

Warp generics are still in development and there are some limitations.

Module Reloading Behavior
~~~~~~~~~~~~~~~~~~~~~~~~~

As mentioned in the :ref:`implicit instantiation <implicit_instantiation>` section, launching new kernel overloads triggers the recompilation of the kernel module.  This adds overhead and doesn't play well with Warp's current kernel caching strategy.  Kernel caching relies on hashing the contents of the module, which includes all the concrete kernels and functions encountered in the Python program so far.  Whenever a new kernel or a new instance of a generic kernel is added, the module needs to be reloaded.  Re-running the Python program leads to the same sequence of kernels being added to the module, which means that implicit instantiation of generic kernels will trigger the same module reloading on every run.  This is clearly not ideal, and we intend to improve this behavior in the future.

Using :ref:`explicit instantiation <explicit_instantiation>` is usually a good workaround for this, as long as the overloads are added in the same order before any kernel launches.

Note that this issue is not specific to generic kernels.  Adding new regular kernels to a module can also trigger repetitive module reloading if the kernel definitions are intermixed with kernel launches.  For example:

.. code:: python

    @wp.kernel
    def foo(x: float):
        wp.print(x)

    wp.launch(foo, dim=1, inputs=[17])

    @wp.kernel
    def bar(x: float):
        wp.print(x)

    wp.launch(bar, dim=1, inputs=[42])

This code will also trigger module reloading during each kernel launch, even though it doesn't use generics at all:

.. code:: text

    Module __main__ load on device 'cuda:0' took 155.73 ms
    17
    Module __main__ load on device 'cuda:0' took 164.83 ms
    42


Graph Capture
~~~~~~~~~~~~~

Module reloading is not allowed during graph capture in CUDA 12.2 or older.  Kernel instantiation can trigger module reloading, which will cause graph capture to fail on drivers that don't support newer versions of CUDA.  The workaround, again, is to explicitly declare the required overloads before capture begins.


Type Variables
~~~~~~~~~~~~~~

Warp's ``type()`` operator is similar in principle to Python's ``type()`` function, but it's currently not possible to use types as variables in Warp kernels and functions.  For example, the following is currently `not` allowed:

.. code:: python

    @wp.func
    def triple(x: Any):
        # TODO:
        T = type(x)
        return T(3) * x


Kernel Overloading Restrictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It's currently not possible to define multiple kernels with the same name but different argument counts, but this restriction may be lifted in the future.
