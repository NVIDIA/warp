Limitations
===========

.. currentmodule:: warp

This section summarizes various limitations and currently unsupported features in Warp.
Problems, questions, and feature requests can be opened on `GitHub Issues <https://github.com/NVIDIA/warp/issues>`_.

Unsupported Features
--------------------

To achieve good performance on GPUs some dynamic language features are not supported:

* Lambda functions
* List comprehensions
* Exceptions
* Recursion
* Runtime evaluation of expressions, e.g.: eval()
* Dynamic structures such as lists, sets, dictionaries, etc.

Kernels and User Functions
--------------------------

* Strings cannot be passed into kernels.
* :func:`wp.atomic_add() <warp._src.lang.atomic_add>` and :func:`wp.atomic_sub() <warp._src.lang.atomic_sub>`
  do not support :class:`wp.float16 <float16>` or :class:`wp.bfloat16 <bfloat16>`
  on GPUs with compute capability below 7.0.
  Kernels compiled for those GPUs skip these updates without raising an error.
* :func:`wp.atomic_add() <warp._src.lang.atomic_add>` and :func:`wp.atomic_sub() <warp._src.lang.atomic_sub>`
  do not support :class:`wp.float64 <float64>` on GPUs with compute capability below 6.0.
  Kernels compiled for those GPUs skip these updates without raising an error.
* Using ``wp.atomic_add()`` or related functions on the same memory address from
  overlapping CPU and GPU kernels is currently unsupported.
* :func:`wp.tid() <warp._src.lang.tid>` cannot be called from user functions.
* Modifying the value of a :func:`wp.constant() <warp.constant>` during runtime will not trigger
  recompilation of the affected kernels if the modules have already been loaded
  (e.g. through a :func:`wp.launch() <warp.launch>` or a :func:`wp.load_module() <warp.load_module>`).
* A :func:`wp.constant() <warp.constant>` used without an explicit type constructor is treated as
  :class:`wp.float32 <float32>` (for Python floats) or :class:`wp.int32 <int32>` (for Python integers).
  See :ref:`Constants <runtime-constants>` for preserving 64-bit precision.
* Python ``IntFlag`` values behave like raw integers in Warp kernels: bitwise inversion (``~``)
  produces the integer bitwise complement, not a masked combination of flags as in standard Python ``IntFlag`` behavior.
* :ref:`Function parameters <callable-parameters>` in user functions only support direct inline calls with
  user-defined :func:`@wp.func <warp.func>` functions and simple built-in Warp functions such as ``wp.sin``,
  ``wp.cos``, ``wp.sqrt``, ``wp.add``, and ``wp.min``.
  Arbitrary Python callables are not supported. Some built-in Warp functions, such as ``wp.printf``, cannot be used
  as ``wp.Function`` arguments because they need special handling during kernel compilation.
  Rebinding a function-valued local to a different function or to a non-function value is not supported.
  User functions with ``wp.Function`` parameters also cannot define custom gradient or replay functions.

:func:`wp.tid() <warp._src.lang.tid>` returns signed 32-bit thread coordinates. In a launch with a nonzero thread
count, the extent corresponding to each coordinate may be at most :math:`2^{31}`, so the largest coordinate is
:math:`2^{31}-1`. Warp raises a ``ValueError`` if any such extent is larger. This limit applies to scalar and
tuple-valued ``wp.tid()``.

The total number of threads in a multidimensional grid may exceed :math:`2^{31}` when each returned coordinate stays
within these limits. Use 64-bit arithmetic when flattening such coordinates into a linear index. See
:ref:`large-array launch indexing <large_launch_indexing>` for an example.

By default, Warp will try to process one element from the Warp grid in one CUDA thread.
This is not always possible for kernels launched with multi-dimensional grid bounds, as there are
`hardware limitations <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#features-and-technical-specifications>`_
on CUDA block dimensions.

Warp will automatically fall back to using
`grid-stride loops <https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/>`_ when
it is not possible for a CUDA thread to process only one element from the Warp grid.
When this happens, some CUDA threads may process more than one element from the Warp grid.
Users can also set the ``max_blocks`` parameter to fine-tune the grid-striding behavior of kernels, even for kernels that are otherwise
able to process one Warp-grid element per CUDA thread. 

Differentiability
-----------------
Please see the :ref:`Limitations and Workarounds <limitations_and_workarounds>` section in the Differentiability page for auto-differentiation limitations.

Arrays
------

* Arrays can have a maximum of four dimensions.
* Each dimension of a Warp array cannot be greater than the maximum value representable by a 32-bit signed integer,
  :math:`2^{31}-1`. As a result, one-dimensional Warp arrays cannot represent larger logical data sets; split them
  across multiple dimensions instead. See :ref:`large-array launch indexing <large_launch_indexing>` for the launch
  and indexing pattern.
* There is currently no dedicated complex scalar type.
* ``wp.config.launch_array_access_mode = wp.config.LaunchArrayAccessMode.CHECKED``
  only fully verifies cross-device :class:`wp.array <warp.array>` arguments when
  Warp can classify the pointer and prove the relevant access requirements.
  Custom arrays or external wrappers whose pointer kind or specific access state
  cannot be verified warn and proceed in checked mode; use
  ``wp.config.LaunchArrayAccessMode.STRICT`` to reject cross-device launches before
  checking access. Directly passed ``__array_interface__`` or
  ``__cuda_array_interface__`` objects are not fully access-verified. See
  :ref:`launch_array_access_checks` for details.

Structs
-------

* Structs cannot have generic members, i.e. of type ``typing.Any``.
* Structs do not support inheritance. Consider using composition instead.

Volumes
-------

* The sparse topology of a non-rebuildable :class:`Volume` cannot be changed after allocation. Volumes created with
  ``rebuildable=True`` or explicit capacity arguments can change topology within their reserved capacity.

Multiple Processes
------------------

* A CUDA context created in the parent process cannot be used in a *forked* child process.
  Use the spawn start method instead, or avoid creating CUDA contexts in the parent process.
* There can be issues with using same user kernel cache directory when running with multiple processes.
  A workaround is to use a separate cache directory for every process.
  See the :ref:`Configuration` section for how the cache directory may be changed.

Scalar Math Functions
---------------------

This section details some limitations and differences from CPython semantics for scalar math functions.

Modulus Operator
""""""""""""""""

Deviation from Python behavior can occur when the modulus operator (``%``) is used with a negative dividend or divisor
(also see :func:`wp.mod() <warp._src.lang.mod>`).
The behavior of the modulus operator in a Warp kernel follows that of C++11: The sign of the result follows the sign of
*dividend*. In Python, the sign of the result follows the sign of the *divisor*:

.. code-block:: python

    @wp.kernel
    def modulus_test():
        # Kernel-scope behavior:
        a = -3 % 2 # a is -1 
        b = 3 % -2 # b is 1
        c = 3 % 0  # Undefined behavior

    # Python-scope behavior:
    a = -3 % 2 # a is 1
    b = 3 % -2 # b is -1
    c = 3 % 0  # ZeroDivisionError

Power Operator
""""""""""""""

The power operator (``**``) in Warp kernels only works on floating-point numbers (also see :func:`wp.pow() <pow>`).
In Python, the power operator can also be used on integers.

Inverse Sine and Cosine
"""""""""""""""""""""""

:func:`wp.asin() <warp._src.lang.asin>` and :func:`wp.acos() <warp._src.lang.acos>` automatically clamp the input to fall in the range [-1, 1].
In Python, using :external+python:py:func:`math.asin` or :external+python:py:func:`math.acos`
with an input outside [-1, 1] raises a ``ValueError`` exception.

Rounding
""""""""

:func:`wp.round() <warp._src.lang.round>` rounds halfway cases away from zero, but Python's
:external+python:py:func:`round` rounds halfway cases to the nearest even
choice (Banker's rounding). Use :func:`wp.rint() <warp._src.lang.rint>` when Banker's rounding is
desired. Unlike Python, the return type in Warp of both of these rounding
functions is the same type as the input:

.. code-block:: python

    @wp.kernel
    def halfway_rounding_test():
        # Kernel-scope behavior:
        a = wp.round(0.5) # a is 1.0
        b = wp.rint(0.5)  # b is 0.0
        c = wp.round(1.5) # c is 2.0
        d = wp.rint(1.5)  # d is 2.0

    # Python-scope behavior:
    a = round(0.5) # a is 0
    c = round(1.5) # c is 2

Bit Count
"""""""""

:func:`wp.bit_count() <warp._src.lang.bit_count>` counts set bits in an integer's fixed-width binary
representation, including the sign bit for signed inputs. In contrast, Python's
:external+python:py:meth:`int.bit_count` counts set bits in the absolute value.

For example, ``wp.bit_count(wp.int8(-1))`` returns ``8``, whereas Python's ``(-1).bit_count()`` returns ``1``.
The difference applies both in Warp kernels and when calling ``wp.bit_count()`` from Python.

Variable Scope
--------------

When writing Warp kernels, reading uninitialized variables can behave differently than in standard Python.
This can sometimes lead to unexpected results.

In standard Python, variables assigned inside an ``if`` block belong to the enclosing function's scope.
Consider this example:

.. code-block:: python

    @wp.func
    def foo(cond: bool):
        if cond:
            out = 123
        else:
            out = 234

        print(out)

This code works as expected in standard Python.
Regardless of the value of ``cond``, ``out`` is defined before being printed.

However, consider a slightly modified example:

.. code-block:: python

    @wp.func
    def foo(cond: bool, x: int):
        if cond:
            out = x + 123

        print(out) # No error even when `cond` is `False`.

In standard Python, if ``cond`` is ``False``, the call to ``print(out)`` would raise an ``UnboundLocalError`` because
``out`` was never assigned a value.

In Warp, the behavior is different. The call to ``print(out)`` *will not* raise an error, even if ``cond`` is ``False``.
Warp does not check whether ``out`` was assigned before it is read.
However, if ``cond`` is ``False``, ``out`` will be uninitialized, leading to undefined behavior.

.. _limitations-arrays-in-structs:

Arrays in Structs
-----------------

Modifying flags on arrays stored in structs may not trigger an update to the underlying struct memory, e.g.:

.. code-block:: python

    @wp.struct
    class MyStruct:
        arr: wp.array[float]

    a = wp.zeros(10, dtype=float)

    s = MyStruct()
    s.arr = a

    # modify original array
    a.requires_grad = True


In this case, ``s.arr.requires_grad`` is ``True``, but the struct's copied array descriptor still has no gradient pointer,
which could lead to gradients not being computed during backward kernel launches.

Tile reductions and atomics on struct-valued tiles do not accumulate the contents of array fields.
Which array descriptor survives is unspecified. See :ref:`tile-struct-element-types` for details and an example.
