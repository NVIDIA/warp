Limitations
===========

.. currentmodule:: warp

This section summarizes various limitations and currently unsupported features in Warp.
Requests for new features can be made at `GitHub Discussions <https://github.com/NVIDIA/warp/discussions>`_,
and issues can be opened at `GitHub Issues <https://github.com/NVIDIA/warp/issues>`_.

Unsupported Features
--------------------

To achieve good performance on GPUs some dynamic language features are not supported:

* Lambda functions
* List comprehensions
* Exceptions
* Recursion
* Runtime evaluation of expressions, e.g.: eval()
* Dynamic structures such as lists, sets, dictionaries, etc.

Kernels
-------

* Strings cannot be passed into kernels.
* Short-circuit evaluation is not supported.
* :func:`wp.atomic_add() <atomic_add>` does not support ``wp.int64``.
* :func:`wp.tid() <tid>` cannot be called from user functions.
* CUDA thread blocks use a fixed size 256 threads per block.

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
  See :ref:`example-cache-management` for how the cache directory may be changed.
