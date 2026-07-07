Basics
======

.. currentmodule:: warp

Tutorial Notebooks
------------------

The `NVIDIA Accelerated Computing Hub <https://github.com/NVIDIA/accelerated-computing-hub>`_ contains the current,
actively maintained set of Warp tutorials:

.. list-table::
   :header-rows: 1

   * - Notebook
     - Colab Link
   * - `Introduction to NVIDIA Warp <https://github.com/NVIDIA/accelerated-computing-hub/blob/32fe3d5a448446fd52c14a6726e1b867cbfed2d9/Accelerated_Python_User_Guide/notebooks/Chapter_12_Intro_to_NVIDIA_Warp.ipynb>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/32fe3d5a448446fd52c14a6726e1b867cbfed2d9/Accelerated_Python_User_Guide/notebooks/Chapter_12_Intro_to_NVIDIA_Warp.ipynb
          :alt: Open In Colab
   * - `GPU-Accelerated Ising Model Simulation in NVIDIA Warp <https://github.com/NVIDIA/accelerated-computing-hub/blob/32fe3d5a448446fd52c14a6726e1b867cbfed2d9/Accelerated_Python_User_Guide/notebooks/Chapter_12.1_IsingModel_In_Warp.ipynb>`_
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/NVIDIA/accelerated-computing-hub/blob/32fe3d5a448446fd52c14a6726e1b867cbfed2d9/Accelerated_Python_User_Guide/notebooks/Chapter_12.1_IsingModel_In_Warp.ipynb
          :alt: Open In Colab

.. _warp-initialization:

Initialization
--------------

When calling a Warp function like :func:`wp.launch() <warp.launch>` for the first time,
Warp will initialize itself and will print some startup information
about the compute devices available, driver versions, and the location for any
generated kernel code, e.g.:

.. code:: bat

    Warp 1.2.0 initialized:
    CUDA Toolkit 12.5, Driver 12.5
    Devices:
        "cpu"      : "x86_64"
        "cuda:0"   : "NVIDIA GeForce RTX 3090" (24 GiB, sm_86, mempool enabled)
        "cuda:1"   : "NVIDIA GeForce RTX 3090" (24 GiB, sm_86, mempool enabled)
    CUDA peer access:
        Supported fully (all-directional)
    Kernel cache:
        /home/nvidia/.cache/warp/1.2.0


It's also possible to explicitly initialize Warp with the :func:`wp.init() <warp.init>` method::

    import warp as wp

    wp.init()


Kernels
-------

In Warp, compute kernels are defined as Python functions and annotated with the :func:`@wp.kernel <warp.kernel>` decorator::

    import warp as wp

    @wp.kernel
    def simple_kernel(a: wp.array[wp.vec3],
                      b: wp.array[wp.vec3],
                      c: wp.array[float]):

        # get thread index
        tid = wp.tid()

        # load two vec3s
        x = a[tid]
        y = b[tid]

        # compute the dot product between vectors
        r = wp.dot(x, y)

        # write result back to memory
        c[tid] = r

Warp kernels do not return values directly. Write results through array arguments
or other explicit output parameters, and either omit the kernel return annotation
or use ``-> None``. Non-``None`` return annotations and ``return <value>``
statements are invalid for kernels.

Conceptually, Warp kernels are similar to CUDA kernels. When a kernel is *launched* on a GPU,
the body of the kernel is executed a certain number of times in parallel.

Because Warp kernels are compiled to native C++/CUDA code, all the function input arguments must be statically typed. This allows 
Warp to generate fast code that executes at essentially native speeds. Because kernels may be run on either the CPU
or GPU, they cannot access arbitrary global state from the Python environment. Instead, they must read and write data
through their input parameters such as *arrays*.

Warp kernels have a one-to-one correspondence with CUDA kernels. 
To launch a kernel with 1024 threads, we use :func:`wp.launch() <warp.launch>`::

    wp.launch(kernel=simple_kernel, # kernel to launch
              dim=1024,             # number of threads
              inputs=[a, b, c],     # parameters
              device="cuda")        # execution device

Inside the kernel, we retrieve the *thread index* of each thread using the :func:`wp.tid() <warp._src.lang.tid>` built-in function::

    # get thread index
    i = wp.tid()

The full list of built-in functions that may be called in Warp kernels is
documented in the :doc:`/language_reference/builtins`.

Kernels can be launched with 1D, 2D, 3D, or 4D grids of threads.
To launch a 2D grid of threads to process a 1024x1024 image, we could write::

    wp.launch(kernel=compute_image, dim=(1024, 1024), inputs=[img], device="cuda")

We retrieve a 2D thread index inside the kernel by using multiple assignment when calling :func:`wp.tid() <warp._src.lang.tid>`:

.. code-block:: python

    @wp.kernel
    def compute_image(pixel_data: wp.array2d[wp.vec3]):
        # get thread index
        i, j = wp.tid()

.. _large_launch_indexing:

Large array indexing
^^^^^^^^^^^^^^^^^^^^

For multi-dimensional launches, Warp derives the coordinates from a linear
thread order in row-major form, with the last dimension varying fastest. This
matches the default contiguous Warp array layout, so adjacent threads that
differ only in the last index access adjacent array elements. On CUDA devices,
this is the preferred pattern for coalesced global memory access.

This is also useful when a logical data set has more than :math:`2^{31}-1`
elements. Because each array dimension must fit in a signed 32-bit integer,
store the data across multiple dimensions, launch over the array shape, and
construct a 64-bit linear index when the algorithm is naturally expressed in
1D:

.. code-block:: python

    @wp.kernel
    def process_large_array(values: wp.array2d[wp.float32], logical_size: wp.int64):
        i, j = wp.tid()

        linear = wp.int64(i) * wp.int64(values.shape[1]) + wp.int64(j)
        if linear < logical_size:
            values[i, j] = float(linear % wp.int64(1024))

    rows = 1 << 20
    cols = 1 << 12
    logical_size = (1 << 32) - 123
    values = wp.empty((rows, cols), dtype=wp.float32, device="cuda")
    wp.launch(
        process_large_array,
        dim=values.shape,
        inputs=[values, wp.int64(logical_size)],
        device=values.device,
    )

Here, the array shape pads the allocation beyond ``logical_size``. The bounds
check skips those padded elements while preserving coalesced access for the
contiguous region.

Arrays
------

Memory allocations are exposed via the :class:`wp.array <warp.array>` type.
Arrays are multidimensional containers of fixed size that can store homogeneous
elements of any Warp data type either in host (CPU) or device (GPU) memory.
All arrays have an associated data type, which can be a scalar data type
(e.g. ``float``, ``int``) or a composite data type (e.g. :class:`wp.vec3 <warp.vec3>`, :class:`wp.mat33 <warp.mat33>`).
:ref:`Data_Types` lists all of Warp's built-in data types.

Arrays can be allocated similar to NumPy and PyTorch::

    # allocate an uninitialized array of vec3s
    v = wp.empty(shape=n, dtype=wp.vec3, device="cuda")

    # allocate a zero-initialized array of quaternions    
    q = wp.zeros(shape=n, dtype=wp.quat, device="cuda")

    # allocate and initialize an array from a NumPy array
    # will be automatically transferred to the specified device
    a = np.ones((10, 3), dtype=np.float32)
    v = wp.from_numpy(a, dtype=wp.vec3, device="cuda")

Arrays up to four dimensions are supported. The aliases
:func:`wp.array2d <warp.array2d>`, :func:`wp.array3d <warp.array3d>`, :func:`wp.array4d <warp.array4d>` are useful when typing kernel
arguments:

.. code-block:: python

    @wp.kernel
    def make_field(field: wp.array3d[float], center: wp.vec3, radius: float):
        i, j, k = wp.tid()

        p = wp.vec3(float(i), float(j), float(k))

        d = wp.length(p - center) - radius

        field[i, j, k] = d

By default, Warp arrays that are initialized from external data (e.g.: NumPy, Lists, Tuples) will create a copy the data in new memory for the
device specified. However, it is possible for arrays to alias external memory using the ``copy=False`` parameter to the
array constructor provided the input is contiguous and on the same device. See the :doc:`/user_guide/interoperability`
section for more details on sharing memory with external frameworks.

To read GPU array data back to CPU memory we can use :func:`array.numpy`::

    # bring data from device back to host
    view = device_array.numpy()

This will automatically synchronize with the GPU to ensure that any outstanding work has finished, and will
copy the array back to CPU memory where it is passed to NumPy.
Calling :func:`array.numpy` on a CPU array will return a zero-copy NumPy view
onto the Warp data.

At Python scope, slicing a Warp array returns a zero-copy view into the same
allocation. Views can be non-contiguous when the slice changes the memory
stride:

.. testcode::

    a = wp.array(np.arange(12, dtype=np.float32).reshape(3, 4), dtype=wp.float32, device="cpu")

    view = a[:, ::2]
    print(view)
    print(view.is_contiguous)

    view.fill_(-1.0)
    print(a)

.. testoutput::

    [[ 0.  2.]
     [ 4.  6.]
     [ 8. 10.]]
    False
    [[-1.  1. -1.  3.]
     [-1.  5. -1.  7.]
     [-1.  9. -1. 11.]]

Scalar item indexing is intentionally not supported on ``wp.array`` objects at
Python scope, so ``a[0, 0]`` raises an error. Use slicing to create array views,
or convert to NumPy with :meth:`array.numpy <warp.array.numpy>` when reading
individual values on the host. Inside kernels, arrays still support element-wise
indexing.

This keeps host-side behavior consistent across CPU and GPU arrays and avoids
encouraging per-element device synchronization or copies.

Common operators such as ``+``, ``-``, ``*``, and ``/`` are overloaded for Warp arrays.
For example, we can add two arrays together using the ``+`` operator:

.. testcode::

    a = wp.array(np.arange(10), dtype=wp.float32)
    b = wp.array(np.arange(10), dtype=wp.float32)
    # multiply a by 2 and add these two arrays together element-wise
    c = 2.0 * a + b
    # multiply c by 10.0 in-place
    c *= 10.0
    print(c)

.. testoutput::

    [  0.  30.  60.  90. 120. 150. 180. 210. 240. 270.]

For further details on mapping arbitrary functions to Warp arrays, see :func:`warp.map`.

Please see the :ref:`Arrays Reference <Arrays>` for more details.

User Functions
--------------

Users can write their own functions to be called from Warp kernels using the :func:`@wp.func <warp.func>` decorator, for example::

    @wp.func
    def square(x: float):
        return x*x

Kernels can call user functions defined in the same module or defined in a different module.
As the example shows, return type hints for user functions are **optional**.

While :func:`@wp.func <warp.func>` is primarily for functions that are called from kernels, they can also be called directly
from Python. This is an experimental feature with an important distinction: functions called from kernels are compiled by Warp,
while functions called from Python are executed by the native Python interpreter.
This means that any code inside a :func:`@wp.func <warp.func>` that is intended to be called from Python must be
compatible with the standard Python interpreter (e.g., it cannot use Warp's tile API).
See :ref:`Python Scope vs. Kernel Scope API <python-scope-vs-kernel-scope-api>` for more details.

Anything that can be done in a Warp kernel can also be done in a user function **with the exception**
of :func:`wp.tid() <warp._src.lang.tid>`. The thread index can be passed in through the arguments of a user function if it is required.

Functions can accept arrays and structs as inputs:

.. code-block:: python

    @wp.func
    def lookup(foos: wp.array[wp.uint32], index: int):
        return foos[index]

Pass-by-reference parameters can be declared on user functions with
:class:`wp.ref[T] <warp.ref>`. Assigning to a ``wp.ref`` parameter mutates the
caller's storage instead of rebinding a local variable. Calls must pass an
addressable expression, such as a local variable, function parameter, array
element, or struct field:

.. code-block:: python

    @wp.func
    def increment(x: wp.ref[wp.int32]):
        x += 1


    @wp.kernel(enable_backward=False)
    def increment_kernel(values: wp.array[wp.int32]):
        i = wp.tid()
        increment(values[i])

``wp.ref`` parameters are supported on :func:`@wp.func <warp.func>` and
:func:`@wp.func_native <warp.func_native>` functions, but not on
:func:`@wp.kernel <warp.kernel>` signatures. User functions with ``wp.ref``
parameters are not automatically differentiable; use ``enable_backward=False``
on the calling kernel or provide a manually differentiated native function when
the function participates in a tape-recorded computation.

Functions may also return multiple values:

.. code-block:: python

    @wp.func
    def multi_valued_func(a: wp.float32, b: wp.float32):
        return a + b, a - b, a * b, a / b

    @wp.kernel
    def test_multi_valued_kernel(test_data1: wp.array[float], test_data2: wp.array[float]):
        tid = wp.tid()
        d1, d2 = test_data1[tid], test_data2[tid]
        a, b, c, d = multi_valued_func(d1, d2)

User functions may also be overloaded by defining multiple function signatures with the same function name:

.. code-block:: python

    @wp.func
    def custom(x: int):
        return x + 1


    @wp.func
    def custom(x: float):
        return x + 1.0


    @wp.func
    def custom(x: wp.vec3):
        return x + wp.vec3(1.0, 0.0, 0.0)

.. _callable-parameters:

Function Parameters
^^^^^^^^^^^^^^^^^^^

User functions can accept another user-defined Warp function or simple built-in
Warp function by annotating the parameter as :class:`warp.Function`.
Function targets are chosen when Warp generates code, not while the kernel is
running. Warp compiles a separate version of the user function for each distinct
set of target functions used at a call site. This is similar to generic
specialization: many target combinations mean more specialized versions to
compile, including for ``wp.grad()`` calls. The function target is chosen where
the user function is called and can be invoked directly inside the user function
body:

.. code-block:: python

    import warp as wp

    @wp.func
    def square(x: float):
        return x * x


    @wp.func
    def cube(x: float):
        return x * x * x


    @wp.func
    def apply(f: wp.Function, x: float):
        return f(x)


    @wp.kernel
    def apply_kernel(
        values: wp.array[float],
        square_out: wp.array[float],
        cube_out: wp.array[float],
    ):
        i = wp.tid()
        square_out[i] = apply(square, values[i])
        cube_out[i] = apply(cube, values[i])

The :class:`warp.Function` annotation is type-erased. It does not encode or
validate the target function signature. The target is checked only through the
actual calls made in the function body during code generation.

Function parameters may also use defaults and keyword arguments:

.. code-block:: python

    @wp.func
    def apply_default(f: wp.Function = square, x: float = 0.0):
        return f(x)

Pass only user-defined :func:`@wp.func <warp.func>` functions or simple built-in
functions such as ``wp.sin``, ``wp.cos``, ``wp.sqrt``, ``wp.add``, and ``wp.min``
as function targets. See :doc:`limitations` for unsupported function targets and
other restrictions.

Tiles may also be passed to user functions. The function signature tile argument should include
dtype and shape parameters to match the tile type intended to be used in the function. For example:

.. code-block:: python

    @wp.func
    def tile_sum_func(a: wp.tile[float, TILE_M, TILE_N]):
        return wp.tile_sum(a) * 0.5

For convenience, it is recommended that users rely on `typing.Any` to let the compiler automatically
determine the tile argument type:

.. code-block:: python

    @wp.func
    def tile_sum_func(a: Any):
        return wp.tile_sum(a) * 0.5

See :ref:`Generic Functions` for details on using ``typing.Any`` in user function signatures.

See :doc:`differentiability` for details on custom gradient and replay
functions. See :doc:`cpp_cuda_workflows` for native C++/CUDA snippets and
other non-Python integration workflows.

User Structs
--------------

Users can define their own structures using the :func:`@wp.struct <warp.struct>` decorator, for example::

    @wp.struct
    class MyStruct:

        pos: wp.vec3
        vel: wp.vec3
        active: int
        indices: wp.array[int]

As with kernel parameters, all attributes of a struct must have valid type hints at class definition time.

Structs may be used as a ``dtype`` for :class:`wp.array <warp.array>` and may be passed to kernels directly as arguments.
See :ref:`Structs Reference <Structs>` for more details on structs.

.. _python-scope-vs-kernel-scope-api:

Python Scope vs. Kernel Scope API
---------------------------------

Some of the Warp API can only be called from the Python scope (i.e. outside of Warp user functions and kernels),
while others can only be called from the kernel scope.

The Python-scope API is documented in the :doc:`/api_reference/warp`,
while the kernel-scope API is documented in the :doc:`/language_reference/builtins`.
Generally, the kernel-scope API can also be used in the Python scope.

Not all of the Python language is supported inside the kernel scope. Some features haven't been implemented yet, while
other features do not map well to the GPU from a performance perspective.

See the :doc:`Limitations <limitations>` documentation for more details.

.. _Compilation Model:

Compilation Model
-----------------

Warp uses a Python->C++/CUDA compilation model that generates kernel code from Python function definitions.
All kernels belonging to a Python module are runtime compiled into dynamic libraries and PTX.
The result is then cached between application restarts for fast startup times.

Note that compilation is triggered on the first kernel launch for that module.
Any kernels registered in the module with :func:`@wp.kernel <warp.kernel>` will be included in the shared library.

.. image:: ../img/compiler_pipeline.svg

By default, status messages will be printed out after each module has been loaded indicating basic information:

* The name of the module that was just loaded
* The first seven characters of the module hash
* The device on which the module is being loaded for
* How long it took to load the module in milliseconds
* Whether the module was compiled ``(compiled)``, loaded from the cache ``(cached)``, or was unable to be loaded ``(error)``.

For debugging purposes, ``wp.config.log_level = wp.LOG_DEBUG`` can be set to also get a printout when each module load begins.

Here is an example illustrating the functionality of the kernel cache by running ``python -m warp.examples.tile.example_tile_cholesky``
twice. The first time, we see:

.. code:: bat

    Warp 1.10.0.dev0 initialized:
    CUDA Toolkit 13.0, Driver 13.0
    Devices:
        "cpu"      : "x86_64"
        "cuda:0"   : "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition" (95 GiB, sm_120, mempool enabled)
    Kernel cache:
        /home/nvidia/.cache/warp/1.10.0.dev0
    Module __main__ 0b0ecab load on device 'cuda:0' took 4136.19 ms  (compiled)

The second time we run this example, we see that the module-loading message now says ``(cached)`` and that it takes
much less time to load the module since code compilation is skipped:

.. code:: bat

    Warp 1.10.0.dev0 initialized:
    CUDA Toolkit 13.0, Driver 13.0
    Devices:
        "cpu"      : "x86_64"
        "cuda:0"   : "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition" (95 GiB, sm_120, mempool enabled)
    Kernel cache:
        /home/nvidia/.cache/warp/1.10.0.dev0
    Module __main__ 0b0ecab load on device 'cuda:0' took 30.98 ms  (cached)

For more information, see the :doc:`../deep_dive/codegen` section.

Language Details
----------------

To support GPU computation and differentiability, there are some differences from the CPython runtime.

Built-in Types
^^^^^^^^^^^^^^

Warp supports a number of built-in math types similar to high-level shading languages,
e.g. ``vec2, vec3, vec4, mat22, mat33, mat44, quat, array``.
All built-in types have value semantics so that expressions such as ``a = b``
generate a copy of the variable ``b`` rather than a reference.

Strong Typing
^^^^^^^^^^^^^

Unlike Python, in Warp all variables must be typed.
Types are inferred from source expressions and function signatures using the Python typing extensions.
All kernel parameters must be annotated with the appropriate type, for example::

    @wp.kernel
    def simple_kernel(a: wp.array[vec3],
                      b: wp.array[vec3],
                      c: float):

For convenience, ``typing.Any`` may be used in place of concrete types
appearing in function signatures. See the :doc:`/user_guide/generics` documentation
for more information. A generic version of the above kernel could look like::

    from typing import Any

    @wp.kernel
    def generic_kernel(a: wp.array[Any],
                      b: wp.array[Any],
                      c: Any):

Tuple initialization is not supported, instead variables should be explicitly typed::

    # invalid
    a = (1.0, 2.0, 3.0)        

    # valid
    a = wp.vec3(1.0, 2.0, 3.0)

Similarly, Python lists are not supported inside kernels. For small fixed-size collections,
use vector types like :class:`wp.vec3() <warp.vec3>`. For larger collections, use
:func:`wp.zeros() <warp.zeros>` to create a stack-allocated fixed-size array::

    # invalid
    my_data = [1.0, 2.0, 3.0]

    # valid: vector (good for 2-4 elements)
    my_data = wp.vec3(1.0, 2.0, 3.0)

    # valid: fixed-size array (stack-allocated, supports array indexing and slicing)
    my_data = wp.zeros(shape=16, dtype=float)


Limitations and Unsupported Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See :doc:`limitations` for a list of Warp limitations and unsupported features.
