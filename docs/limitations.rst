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
* Short-circuit evaluation is not supported
* :func:`wp.atomic_add() <atomic_add>` does not support ``wp.int64``.
* :func:`wp.tid() <tid>` cannot be called from user functions.
* Modifying the value of a :class:`wp.constant() <constant>` during runtime will not trigger
  recompilation of the affected kernels if the modules have already been loaded
  (e.g. through a :func:`wp.launch() <launch>` or a ``wp.load_module()``).
* A :class:`wp.constant() <constant>` can suffer precision loss if used with ``wp.float64``
  as it is initially assigned to a ``wp.float32`` variable in the generated code.

A limitation of Warp is that each dimension of the grid used to launch a kernel must be representable as a 32-bit
signed integer. Therefore, no single dimension of a grid should exceed :math:`2^{31}-1`.

Warp also currently uses a fixed block size of 256 (CUDA) threads per block.
By default, Warp will try to process one element from the Warp grid in one CUDA thread.
This is not always possible for kernels launched with multi-dimensional grid bounds, as there are
`hardware limitations <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability>`_
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
  :math:`2^{31}-1`.
* There are currently no data types that support complex numbers.

Structs
-------

* Structs cannot have generic members, i.e. of type ``typing.Any``.

Volumes
-------

* The sparse-volume *topology* cannot be changed after the tiles for the :class:`Volume` have been allocated.

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
(also see :func:`wp.mod() <mod>`).
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

:func:`wp.asin() <asin>` and :func:`wp.acos() <acos>` automatically clamp the input to fall in the range [-1, 1].
In Python, using :external+python:py:func:`math.asin` or :external+python:py:func:`math.acos`
with an input outside [-1, 1] raises a ``ValueError`` exception.

Rounding
""""""""

:func:`wp.round() <round>` rounds halfway cases away from zero, but Python's
:external+python:py:func:`round` rounds halfway cases to the nearest even
choice (Banker's rounding). Use :func:`wp.rint() <rint>` when Banker's rounding is
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
