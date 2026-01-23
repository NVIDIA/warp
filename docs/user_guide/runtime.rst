Runtime
=======

.. currentmodule:: warp

This section describes the Warp Python runtime API, how to manage memory, launch kernels, and high-level functionality
for dealing with objects such as meshes and volumes. The APIs described in this section are intended to be used at
the *Python Scope* and run inside the CPython interpreter. For a comprehensive list of functions available at
the *Kernel Scope*, please see the :doc:`/language_reference/builtins` section.

Kernels
-------

Kernels are defined via Python functions that are annotated with the :func:`@wp.kernel <kernel>` decorator.
All arguments of the Python function must be annotated with their respective type.
The following example shows a simple kernel that adds two arrays together::

    import warp as wp

    @wp.kernel
    def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
        tid = wp.tid()
        c[tid] = a[tid] + b[tid]

Kernels are launched with the :func:`wp.launch() <warp.launch>` function on a specific device (CPU/GPU)::

    wp.launch(add_kernel, dim=1024, inputs=[a, b], outputs=[c], device="cuda")

Note that all the kernel inputs and outputs must live on the target device or a runtime exception will be raised.

Unless you are using the :ref:`Graph visualization tool<visualizing_computation_graphs>`, the ``outputs`` argument is optional -- all kernel 
arguments may be passed as inputs, but for readability it is sometimes useful to distinguish between the 
kernel arguments that are read from (``inputs``) and the kernel arguments that are written to (``outputs``). 
So in the above example, it would be equally valid to write ``inputs=[a, b, c]`` but since we are writing to ``c``,
we list it in the ``outputs`` argument. Note that the combined ``inputs`` followed by ``outputs`` list 
should match the ordering of the kernel arguments.

Kernels may be launched with multi-dimensional grid bounds.
In this case, threads are not assigned a single index, but a coordinate in an n-dimensional grid, e.g.::

    wp.launch(complex_kernel, dim=(128, 128, 3), ...)

Launches a 3D grid of threads with dimension 128 x 128 x 3. To retrieve the 3D index for each thread, use the following syntax::

    i,j,k = wp.tid()

.. note::
    Currently, kernels launched on CPU devices will be executed in serial.
    Kernels launched on CUDA devices will be launched in parallel with a fixed block-size.

In the Warp :ref:`Compilation Model`, kernels are just-in-time compiled into dynamic libraries and PTX using
C++/CUDA as an intermediate representation.
To avoid excessive runtime recompilation of kernel code, these files are stored in a cache directory
named with a module-dependent hash to allow for the reuse of previously compiled modules.
The location of the kernel cache is printed when Warp is initialized.
:func:`wp.clear_kernel_cache() <warp.clear_kernel_cache>` can be used to clear the kernel cache of previously
generated compilation artifacts as Warp does not automatically try to keep the cache below a certain size.


.. _Runtime Kernel Creation:

Runtime Kernel Creation
#######################

Warp allows generating kernels on-the-fly with various customizations, including closure support.
Refer to the :ref:`Code Generation<code_generation>` section for the latest features.

Launch Objects
##############

:class:`Launch` objects are one way to reduce the overhead of launching a kernel multiple times.
:class:`Launch` objects are returned from calling :func:`wp.launch() <warp.launch>` with ``record_cmd=True``.
This stores the results of various overhead operations that are needed to launch a kernel
but defers the actual kernel launch until the :meth:`Launch.launch` method is called.

In contrast to :ref:`graphs`, :class:`Launch` objects only record the launch of a single kernel
and do not reduce the driver overhead of preparing the kernel for execution on a GPU.
On the other hand, :class:`Launch` objects do not have the storage and initialization
overheads of CUDA graphs and also allow for the modification of launch
dimensions with :meth:`Launch.set_dim` and
kernel parameters with functions such as :meth:`Launch.set_params` and
:meth:`Launch.set_param_by_name`.
Additionally, :class:`Launch` objects can also be used to reduce the overhead of launching kernels running on the CPU.

.. note::
    Kernels launched via :class:`Launch` objects currently do not get recorded onto the :class:`Tape`.


.. _Arrays:

Arrays
------

Arrays are the fundamental memory abstraction in Warp. They can be created through the following global constructor::

    wp.empty(shape=1024, dtype=wp.vec3, device="cpu")
    wp.zeros(shape=1024, dtype=float, device="cuda")
    wp.full(shape=1024, value=10, dtype=int, device="cuda")


Arrays can also be constructed directly from NumPy ``ndarrays`` as follows::

    r = np.random.rand(1024)

    # copy to Warp owned array
    a = wp.array(r, dtype=float, device="cpu")

    # return a Warp array wrapper around the NumPy data (zero-copy)
    a = wp.array(r, dtype=float, copy=False, device="cpu")

    # return a Warp copy of the array data on the GPU
    a = wp.array(r, dtype=float, device="cuda")

Note that for multi-dimensional data, the ``dtype`` parameter must be specified explicitly, e.g.::

    r = np.random.rand((1024, 3))

    # initialize as an array of vec3 objects
    a = wp.array(r, dtype=wp.vec3, device="cuda")

If the shapes are incompatible, an error will be raised.

Warp arrays can also be constructed from objects that define the ``__cuda_array_interface__`` attribute. For example::

    import cupy
    import warp as wp

    device = wp.get_cuda_device()

    r = cupy.arange(10)

    # return a Warp array wrapper around the cupy data (zero-copy)
    a = wp.array(r, device=device)

.. note::

    When constructing arrays from the ``__cuda_array_interface__``, it is important to pass the correct CUDA device to the Warp array constructor.  The ``__cuda_array_interface__`` protocol does not include the device, hence it is necessary to explicitly specify the device where the array resides.

Arrays can be moved between devices using :meth:`array.to`::

    host_array = wp.array(a, dtype=float, device="cpu")

    # allocate and copy to GPU
    device_array = host_array.to("cuda")

Additionally, data can be copied between arrays in different memory spaces using :func:`wp.copy() <warp.copy>`::

    src_array = wp.array(a, dtype=float, device="cpu")
    dest_array = wp.empty_like(host_array)

    # copy from source CPU buffer to GPU
    wp.copy(dest_array, src_array)

When indexing an array with an array of integers, the result is an :ref:`indexed array<Indexed_Arrays>`:

.. testcode::

    import warp as wp

    arr = wp.array((1, 2, 3, 4, 5, 6))
    sub = arr[wp.array((0, 2, 4), dtype=wp.int32)] # advanced indexing -> wp.indexedarray

    print(type(arr), arr.shape)
    print(type(sub), sub.shape)
    print(sub)

.. testoutput::

    <class 'warp._src.types.array'> (6,)
    <class 'warp._src.types.indexedarray'> (3,)
    [1 3 5]


Multi-dimensional Arrays
########################

Multi-dimensional arrays up to four dimensions can be constructed by passing a tuple of sizes for each dimension.

The following constructs a 2D array of size 1024 x 16::

    wp.zeros(shape=(1024, 16), dtype=float, device="cuda")

When passing multi-dimensional arrays to kernels users must specify the expected array dimension inside the kernel signature,
e.g. to pass a 2D array to a kernel the number of dims is specified using the ``ndim=2`` parameter::

    @wp.kernel
    def test(input: wp.array(dtype=float, ndim=2)):

Type-hint helpers are provided for common array sizes, e.g.: :func:`wp.array2d <warp.array2d>`, :func:`wp.array3d <warp.array3d>`, which are equivalent to calling ``array(..., ndim=2)``, etc.
To index a multi-dimensional array, use the following kernel syntax::

    # returns a float from the 2d array
    value = input[i,j]

To create an array slice, use the following syntax, where the number of indices is less than the array dimensions::

    # returns an 1d array slice representing a row of the 2d array
    row = input[i]

Slice operators can be concatenated, e.g.: ``s = array[i][j][k]``. Slices can be passed to :func:`wp.func <warp.func>` user functions provided
the function also declares the expected array dimension. Currently, only single-index slicing is supported.

The following construction methods are provided for allocating zero-initialized and empty (non-initialized) arrays:

* :func:`wp.zeros() <warp.zeros>`
* :func:`wp.zeros_like() <warp.zeros_like>`
* :func:`wp.ones() <warp.ones>`
* :func:`wp.ones_like() <warp.ones_like>`
* :func:`wp.full() <warp.full>`
* :func:`wp.full_like() <warp.full_like>`
* :func:`wp.empty() <warp.empty>`
* :func:`wp.empty_like() <warp.empty_like>`
* :func:`wp.copy() <warp.copy>`
* :func:`wp.clone() <warp.clone>`



.. _Indexed_Arrays:

Indexed Arrays
##############

An indexed array is a lightweight view into an existing :class:`warp.array` instance that references elements
through an explicit integer index list, thus allowing to run kernels on an arbitrary subset of data without any copy.



Creating an Indexed Array
^^^^^^^^^^^^^^^^^^^^^^^^^

Pass the *data* array together with a list of :class:`wp.int32 <warp.int32>` index arrays, one for each dimension:

.. testcode::

    import warp as wp

    # Base data.
    arr = wp.array((1.23, 2.34, 3.45, 4.56, 5.67, 6.78), device="cuda")

    # Only view elements at odd indices.
    idx = wp.array((1, 3, 5), dtype=wp.int32, device="cuda")
    sub = wp.indexedarray(arr, [idx])  # Same as wp.indexedarray1d(...)
    print(sub)

.. testoutput::

    [2.34 4.56 6.78]


Additionally, ``None`` can be passed to select all elements for any given dimension.

.. testcode::

    import numpy as np
    import warp as wp

    mat = wp.array(np.arange(25, dtype=np.float32).reshape((5, 5)))
    rows = wp.array((1, 3), dtype=wp.int32)

    block = wp.indexedarray2d(mat, (rows, None))  # shape == (2, 5)
    print(block)

.. testoutput::

    [[ 5.  6.  7.  8.  9.]
     [15. 16. 17. 18. 19.]]


The resulting view keeps the ``dtype`` of the source and has a shape given by the lengths of the supplied index arrays.

Alternative constructors are available for convenience:

* :func:`warp.indexedarray1d`
* :func:`warp.indexedarray2d`
* :func:`warp.indexedarray3d`
* :func:`warp.indexedarray4d`



Interoperability With Other Frameworks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Frameworks such as PyTorch or JAX do not have a concept equivalent to
Warp's indexed arrays. Converting an :class:`wp.indexedarray <warp.indexedarray>` directly therefore
raises an exception. Two common workarounds are:

1. Make a contiguous copy and share that::

    import warp as wp

    arr = wp.array((1.0, 2.0, 3.0, 4.0), device="cuda")
    idx = wp.array((0, 3), dtype=int, device="cuda")
    sub = wp.indexedarray1d(arr, idx)
    t = wp.to_torch(sub.contiguous())

2. Share the underlying data and index buffers independently (zero-copy)::

    import warp as wp

    arr = wp.array((1.0, 2.0, 3.0, 4.0), device="cuda")
    idx = wp.array((0, 3), dtype=int, device="cuda")
    sub = wp.indexedarray1d(arr, idx)
    t_data = wp.to_torch(sub.data)
    t_ind = wp.to_torch(sub.indices[0])


PyTorch can index with integer tensors, but doing so always copies the data.


Structured Arrays
#################

Structured arrays in Warp allow you to work with arrays of user-defined structs,
enabling efficient, named access to heterogeneous data fields across the CPU and GPU.

Creating and Viewing Struct Arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you define a Warp struct, you can allocate a Warp array of that type on the CPU and convert it to a NumPy structured array view (zero-copy):

.. testcode::

    import warp as wp
    import numpy as np

    @wp.struct
    class Foo:
        i: int
        f: float

    # allocate a Warp array on the CPU
    a = wp.zeros(5, dtype=Foo, device="cpu")

    # view it in NumPy without copying
    na = a.numpy()

    # modify via NumPy
    na["i"][0] = 42
    na["f"][2] = 13.37

    print(a)
    
.. testoutput::

    [(42,  0.  ) ( 0,  0.  ) ( 0, 13.37) ( 0,  0.  ) ( 0,  0.  )]

Initializing via NumPy and Converting to a Warp Array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also create a NumPy structured array first, then convert it to a Warp array, which works well for batch initialization: ::

    import warp as wp
    import numpy as np
    import math

    rng = np.random.default_rng(123)

    @wp.struct
    class Boid:
        vel: wp.vec3f
        wander_angles: wp.vec2f
        mass: float
        group: int

    num_boids = 3
    npboids = np.zeros(num_boids, dtype=Boid.numpy_dtype())

    angles = math.pi - 2 * math.pi * rng.random(num_boids)
    npboids["vel"][:, 0] = 20 * np.sin(angles)
    npboids["vel"][:, 2] = 20 * np.cos(angles)

    npboids["wander_angles"][:, 0] = math.pi * rng.random(num_boids)
    npboids["wander_angles"][:, 1] = 2 * math.pi * rng.random(num_boids)

    npboids["mass"][:] = 0.5 + 0.5 * rng.random(num_boids)

    # create Warp array from prepared NumPy array
    boids = wp.array(npboids, dtype=Boid)

This approach leverages NumPy's vectorized operations to initialize all array elements efficiently, avoiding Python loops.

Nested Structs and Vector Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Structured arrays fully support nested structs and Warp vector (and matrix) types:

.. testcode::

    import warp as wp
    import numpy as np

    @wp.struct
    class Bar:
        x: wp.vec3

    @wp.struct
    class Foo:
        i: int
        f: float
        bar: Bar

    na = np.zeros(5, dtype=Foo.numpy_dtype())

    na["i"][0] = 42
    na["f"][2] = 13.37
    na["bar"]["x"][4] = wp.vec3(1.0)

    a = wp.array(na, dtype=Foo, device="cuda:0")

    print(a.numpy())

.. testoutput::

    [(42,  0.  , ([0., 0., 0.],)) ( 0,  0.  , ([0., 0., 0.],))
     ( 0, 13.37, ([0., 0., 0.],)) ( 0,  0.  , ([0., 0., 0.],))
     ( 0,  0.  , ([1., 1., 1.],))]


Local Arrays
############

While arrays are typically created at the Python scope and passed to kernels as arguments,
Warp also supports creating arrays directly inside kernels. This capability is limited to two specific approaches:

1. **Creating array views from existing memory**: Initialize an array that references an existing data buffer
   by using ``wp.array(ptr=..., shape=..., dtype=...)``. This is useful to reinterpret memory
   with a different shape or when working with external memory pointers:

    .. testcode::

        @wp.kernel
        def sum_rows_kernel(
            flat_arr: wp.array(dtype=int),
            out: wp.array(dtype=int),
        ):
            tid = wp.tid()

            # Reinterpret the flat array as a 2D array of 3x4 elements.
            arr = wp.array(ptr=flat_arr.ptr, shape=(3, 4), dtype=int)

            # Compute sum of row.
            sum = int(0)
            for j in range(arr.shape[1]):
                sum += arr[tid, j]

            out[tid] = sum

        flat_arr = wp.array(range(12), dtype=int)
        row_sums = wp.zeros(3, dtype=int)
        wp.launch(sum_rows_kernel, dim=3, inputs=(flat_arr, row_sums))
        print(row_sums.numpy())

    .. testoutput::

        [ 6 22 38]

2. **Allocating fixed-size arrays**: Allocate a new zero-initialized array with a compile-time constant shape
   using ``wp.zeros(shape=..., dtype=...)``:

    .. testcode::

        N = 6

        @wp.kernel
        def find_cumsum_avg_crossing_kernel(
            arr: wp.array2d(dtype=float),
            out: wp.array(dtype=int),
        ):
            tid = wp.tid()

            # Create temporary array to store cumulative sums for this column.
            tmp = wp.zeros(shape=(N,), dtype=float)

            # Compute the cumulative sum values.
            tmp[0] = arr[0, tid]
            for i in range(1, N):
                tmp[i] = tmp[i - 1] + arr[i, tid]

            # Calculate the average of the cumulative sum values.
            sum = float(0)
            for i in range(N):
                sum += tmp[i]
            avg = sum / float(N)

            # Find the first index where `cumulative sum value >= avg`.
            # This represents the crossing point where accumulated values exceed
            # the average accumulation.
            out[tid] = wp.lower_bound(tmp, avg)

        arr = wp.array(np.abs(np.sin(np.arange(N * 3))).reshape(N, 3), dtype=float)
        idx = wp.empty(shape=(3,), dtype=int)
        wp.launch(find_cumsum_avg_crossing_kernel, dim=(3,), inputs=(arr,), outputs=(idx,))
        print(idx.numpy())

    .. testoutput::

        [3 3 3]


.. _Data_Types:

Data Types
----------

Scalar Types
############

The following scalar storage types are supported for array structures:

+---------+------------------------+
| bool    | boolean                |
+---------+------------------------+
| int8    | signed byte            |
+---------+------------------------+
| uint8   | unsigned byte          |
+---------+------------------------+
| int16   | signed short           |
+---------+------------------------+
| uint16  | unsigned short         |
+---------+------------------------+
| int32   | signed integer         |
+---------+------------------------+
| uint32  | unsigned integer       |
+---------+------------------------+
| int64   | signed long integer    |
+---------+------------------------+
| uint64  | unsigned long integer  |
+---------+------------------------+
| float16 | half-precision float   |
+---------+------------------------+
| float32 | single-precision float |
+---------+------------------------+
| float64 | double-precision float |
+---------+------------------------+

Warp supports ``float`` and ``int`` as aliases for :class:`wp.float32 <warp.float32>` and :class:`wp.int32 <warp.int32>` respectively.

.. _vec:

Vectors
#######

Warp provides built-in math and geometry types for common simulation and graphics problems.
A full reference for operators and functions for these types is available in the :doc:`/language_reference/builtins`.

Warp supports vectors of numbers with an arbitrary length/numeric type. The built-in concrete types are as follows:

+-----------------------+------------------------------------------------+
| vec2 vec3 vec4        | 2D, 3D, 4D vector of single-precision floats   |
+-----------------------+------------------------------------------------+
| vec2b vec3b vec4b     | 2D, 3D, 4D vector of signed bytes              |
+-----------------------+------------------------------------------------+
| vec2ub vec3ub vec4ub  | 2D, 3D, 4D vector of unsigned bytes            |
+-----------------------+------------------------------------------------+
| vec2s vec3s vec4s     | 2D, 3D, 4D vector of signed shorts             |
+-----------------------+------------------------------------------------+
| vec2us vec3us vec4us  | 2D, 3D, 4D vector of unsigned shorts           |
+-----------------------+------------------------------------------------+
| vec2i vec3i vec4i     | 2D, 3D, 4D vector of signed integers           |
+-----------------------+------------------------------------------------+
| vec2ui vec3ui vec4ui  | 2D, 3D, 4D vector of unsigned integers         |
+-----------------------+------------------------------------------------+
| vec2l vec3l vec4l     | 2D, 3D, 4D vector of signed long integers      |
+-----------------------+------------------------------------------------+
| vec2ul vec3ul vec4ul  | 2D, 3D, 4D vector of unsigned long integers    |
+-----------------------+------------------------------------------------+
| vec2h vec3h vec4h     | 2D, 3D, 4D vector of half-precision floats     |
+-----------------------+------------------------------------------------+
| vec2f vec3f vec4f     | 2D, 3D, 4D vector of single-precision floats   |
+-----------------------+------------------------------------------------+
| vec2d vec3d vec4d     | 2D, 3D, 4D vector of double-precision floats   |
+-----------------------+------------------------------------------------+
| spatial_vector        | 6D vector of single-precision floats           |
+-----------------------+------------------------------------------------+
| spatial_vectorf       | 6D vector of single-precision floats           |
+-----------------------+------------------------------------------------+
| spatial_vectord       | 6D vector of double-precision floats           |
+-----------------------+------------------------------------------------+
| spatial_vectorh       | 6D vector of half-precision floats             |
+-----------------------+------------------------------------------------+

Vectors support most standard linear algebra operations, e.g.: ::

    @wp.kernel
    def compute( ... ):

        # basis vectors
        a = wp.vec3(1.0, 0.0, 0.0)
        b = wp.vec3(0.0, 1.0, 0.0)

        # take the cross product
        c = wp.cross(a, b)

        # compute
        r = wp.dot(c, c)

        ...


It's possible to declare additional vector types with different lengths and data types. This is done in outside of kernels in *Python scope* using ``warp.types.vector()``, for example: ::

    # declare a new vector type for holding 5 double precision floats:
    vec5d = wp.types.vector(length=5, dtype=wp.float64)

Once declared, the new type can be used when allocating arrays or inside kernels: ::

    # create an array of vec5d
    arr = wp.zeros(10, dtype=vec5d)

    # use inside a kernel
    @wp.kernel
    def compute( ... ):

        # zero initialize a custom named vector type
        v = vec5d()
        ...

        # component-wise initialize a named vector type
        v = vec5d(wp.float64(1.0),
                  wp.float64(2.0),
                  wp.float64(3.0),
                  wp.float64(4.0),
                  wp.float64(5.0))
      ...

In addition, it's possible to directly create *anonymously* typed instances of these vectors without declaring their type in advance. In this case the type will be inferred by the constructor arguments. For example: ::

    @wp.kernel
    def compute( ... ):

        # zero initialize vector of 5 doubles:
        v = wp.types.vector(dtype=wp.float64, length=5)

        # scalar initialize a vector of 5 doubles to the same value:
        v = wp.types.vector(wp.float64(1.0), length=5)

        # component-wise initialize a vector of 5 doubles
        v = wp.types.vector(wp.float64(1.0),
                            wp.float64(2.0),
                            wp.float64(3.0),
                            wp.float64(4.0),
                            wp.float64(5.0))


These can be used with all the standard vector arithmetic operators, e.g.: ``+``, ``-``, scalar multiplication, and can also be transformed using matrices with compatible dimensions, potentially returning vectors with a different length.

.. _mat:

Matrices
########

Matrices with arbitrary shapes/numeric types are also supported. The built-in concrete matrix types are as follows:

+--------------------------+-------------------------------------------------+
| mat22 mat33 mat44        | 2x2, 3x3, 4x4 matrix of single-precision floats |
+--------------------------+-------------------------------------------------+
| mat22f mat33f mat44f     | 2x2, 3x3, 4x4 matrix of single-precision floats |
+--------------------------+-------------------------------------------------+
| mat22d mat33d mat44d     | 2x2, 3x3, 4x4 matrix of double-precision floats |
+--------------------------+-------------------------------------------------+
| mat22h mat33h mat44h     | 2x2, 3x3, 4x4 matrix of half-precision floats   |
+--------------------------+-------------------------------------------------+
| spatial_matrix           | 6x6 matrix of single-precision floats           |
+--------------------------+-------------------------------------------------+
| spatial_matrixf          | 6x6 matrix of single-precision floats           |
+--------------------------+-------------------------------------------------+
| spatial_matrixd          | 6x6 matrix of double-precision floats           |
+--------------------------+-------------------------------------------------+
| spatial_matrixh          | 6x6 matrix of half-precision floats             |
+--------------------------+-------------------------------------------------+

Matrices are stored in row-major format and support most standard linear algebra operations: ::

    @wp.kernel
    def compute( ... ):

        # initialize matrix
        m = wp.mat22(1.0, 2.0,
                     3.0, 4.0)

        # compute inverse
        minv = wp.inverse(m)

        # transform vector
        v = minv * wp.vec2(0.5, 0.3)

        ...


In a similar manner to vectors, it's possible to declare new matrix types with arbitrary shapes and data types using ``wp.types.matrix()``, for example: ::

    # declare a new 3x2 half precision float matrix type:
    mat32h = wp.types.matrix(shape=(3,2), dtype=wp.float64)

    # create an array of this type
    a = wp.zeros(10, dtype=mat32h)

These can be used inside a kernel::

    @wp.kernel
    def compute( ... ):
        ...

        # initialize a mat32h matrix
        m = mat32h(wp.float16(1.0), wp.float16(2.0),
                   wp.float16(3.0), wp.float16(4.0),
                   wp.float16(5.0), wp.float16(6.0))

        # declare a 2 component half precision vector
        v2 = wp.vec2h(wp.float16(1.0), wp.float16(1.0))

        # multiply by the matrix, returning a 3 component vector:
        v3 = m * v2
        ...

It's also possible to directly create anonymously typed instances inside kernels where the type is inferred from constructor arguments as follows::

    @wp.kernel
    def compute( ... ):
        ...

        # create a 3x2 half precision matrix from components (row major ordering):
        m = wp.types.matrix(
            wp.float16(1.0), wp.float16(2.0),
            wp.float16(1.0), wp.float16(2.0),
            wp.float16(1.0), wp.float16(2.0),
            shape=(3,2))

        # zero initialize a 3x2 half precision matrix:
        m = wp.types.matrix(wp.float16(0.0),shape=(3,2))

        # create a 5x5 double precision identity matrix:
        m = wp.identity(n=5, dtype=wp.float64)

As with vectors, you can do standard matrix arithmetic with these variables, along with multiplying matrices with compatible shapes and potentially returning a matrix with a new shape.

.. _quat:

Quaternions
###########

Warp supports quaternions with the layout ``i, j, k, w`` where ``w`` is the real part. Here are the built-in concrete quaternion types:

+-----------------+--------------------------------------------+
| quat            | Single-precision floating point quaternion |
+-----------------+--------------------------------------------+
| quatf           | Single-precision floating point quaternion |
+-----------------+--------------------------------------------+
| quatd           | Double-precision floating point quaternion |
+-----------------+--------------------------------------------+
| quath           | Half-precision floating point quaternion   |
+-----------------+--------------------------------------------+

Quaternions can be used to transform vectors as follows::

    @wp.kernel
    def compute( ... ):
        ...

        # construct a 30 degree rotation around the x-axis
        q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.degrees(30.0))

        # rotate an axis by this quaternion
        v = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))


As with vectors and matrices, you can declare quaternion types with an arbitrary numeric type like so::

    quatd = wp.types.quaternion(dtype=wp.float64)

You can also create identity quaternion and anonymously typed instances inside a kernel like so::

    @wp.kernel
    def compute( ... ):
        ...

        # create a double precision identity quaternion:
        qd = wp.quat_identity(dtype=wp.float64)

        # precision defaults to wp.float32 so this creates a single precision identity quaternion:
        qf = wp.quat_identity()

        # create a half precision quaternion from components, or a vector/scalar:
        qh = wp.quaternion(wp.float16(0.0),
                           wp.float16(0.0),
                           wp.float16(0.0),
                           wp.float16(1.0))


        qh = wp.quaternion(
            wp.vector(wp.float16(0.0),wp.float16(0.0),wp.float16(0.0)),
            wp.float16(1.0))

.. _transform:

Transforms
##########

Transforms are 7D vectors of floats representing a spatial rigid body transformation in format (p, q) where p is a 3D vector, and q is a quaternion.

+-----------------+--------------------------------------------+
| transform       | Single-precision floating point transform  |
+-----------------+--------------------------------------------+
| transformf      | Single-precision floating point transform  |
+-----------------+--------------------------------------------+
| transformd      | Double-precision floating point transform  |
+-----------------+--------------------------------------------+
| transformh      | Half-precision floating point transform    |
+-----------------+--------------------------------------------+

Transforms can be constructed inside kernels from translation and rotation parts::

    @wp.kernel
    def compute( ... ):
        ...

        # create a transform from a vector/quaternion:
        t = wp.transform(
                wp.vec3(1.0, 2.0, 3.0),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.degrees(30.0)))

        # transform a point
        p = wp.transform_point(t, wp.vec3(10.0, 0.5, 1.0))

        # transform a vector (ignore translation)
        p = wp.transform_vector(t, wp.vec3(10.0, 0.5, 1.0))


As with vectors and matrices, you can declare transform types with an arbitrary numeric type using ``wp.types.transformation()``, for example::

    transformd = wp.types.transformation(dtype=wp.float64)

You can also create identity transforms and anonymously typed instances inside a kernel like so::

    @wp.kernel
    def compute( ... ):

        # create double precision identity transform:
        qd = wp.transform_identity(dtype=wp.float64)

.. _Structs:

Structs
#######

Users can define custom structure types using the :func:`@wp.struct <warp.struct>` decorator as follows::

    @wp.struct
    class MyStruct:

        param1: int
        param2: float
        param3: wp.array(dtype=wp.vec3)

Struct attributes must be annotated with their respective type. They can be constructed in Python scope and then passed to kernels as arguments::

    @wp.kernel
    def compute(args: MyStruct):

        tid = wp.tid()

        print(args.param1)
        print(args.param2)
        print(args.param3[tid])

    # construct an instance of the struct in Python
    s = MyStruct()
    s.param1 = 10
    s.param2 = 2.5
    s.param3 = wp.zeros(shape=10, dtype=wp.vec3)

    # pass to our compute kernel
    wp.launch(compute, dim=10, inputs=[s])

An array of structs can be zero-initialized as follows::

    a = wp.zeros(shape=10, dtype=MyStruct)

An array of structs can also be initialized from a list of struct objects::

    a = wp.array([MyStruct(), MyStruct(), MyStruct()], dtype=MyStruct)

Example: Using a struct in gradient computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. testcode::

    import numpy as np

    import warp as wp


    @wp.struct
    class TestStruct:
        x: wp.vec3
        a: wp.array(dtype=wp.vec3)
        b: wp.array(dtype=wp.vec3)


    @wp.kernel
    def test_kernel(s: TestStruct):
        tid = wp.tid()

        s.b[tid] = s.a[tid] + s.x


    @wp.kernel
    def loss_kernel(s: TestStruct, loss: wp.array(dtype=float)):
        tid = wp.tid()

        v = s.b[tid]
        wp.atomic_add(loss, 0, float(tid + 1) * (v[0] + 2.0 * v[1] + 3.0 * v[2]))


    # create struct
    ts = TestStruct()

    # set members
    ts.x = wp.vec3(1.0, 2.0, 3.0)
    ts.a = wp.array(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), dtype=wp.vec3, requires_grad=True)
    ts.b = wp.zeros(2, dtype=wp.vec3, requires_grad=True)

    loss = wp.zeros(1, dtype=float, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(test_kernel, dim=2, inputs=[ts])
        wp.launch(loss_kernel, dim=2, inputs=[ts, loss])

    tape.backward(loss)

    print(loss)
    print(ts.a)

.. testoutput::

    [120.]
    [[1. 2. 3.]
     [4. 5. 6.]]

Example: Defining Operator Overloads
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    @wp.struct
    class Complex:
        real: float
        imag: float

    @wp.func
    def add(
        a: Complex,
        b: Complex,
    ) -> Complex:
        return Complex(a.real + b.real, a.imag + b.imag)

    @wp.func
    def mul(
        a: Complex,
        b: Complex,
    ) -> Complex:
        return Complex(
            a.real * b.real - a.imag * b.imag,
            a.real * b.imag + a.imag * b.real,
        )

    @wp.kernel
    def kernel():
        a = Complex(1.0, 2.0)
        b = Complex(3.0, 4.0)

        c = a + b
        wp.printf("%.0f %+.0fi\n", c.real, c.imag)

        d = a * b
        wp.printf("%.0f %+.0fi\n", d.real, d.imag)

    wp.launch(kernel, dim=(1,))
    wp.synchronize()


Indexing and Slicing
####################

Indexing and slicing for vectors, matrices, quaternions, and transforms, follow NumPy-like semantics for element access: ::

    @wp.kernel
    def compute( ... ):
        v = wp.vec3(1.0, 2.0, 3.0)
        wp.expect_eq(v[-1], 3.0) # negative indices wrap
        wp.expect_eq(v[1:], wp.vec2(2.0, 3.0)) # slice returns a new vector

        v[::2] = 0.0 # slice assignment
        wp.expect_eq(v, wp.vec3(0.0, 2.0, 0.0))

        m = wp.matrix_from_rows(
            wp.vec3(1.0, 2.0, 3.0),
            wp.vec3(4.0, 5.0, 6.0),
            wp.vec3(7.0, 8.0, 9.0),
        )
        wp.expect_eq(m[:, 1], wp.vec3(2.0, 5.0, 8.0)) # column vector
        wp.expect_eq(
            m[:2, 1:], # 2x2 sub-matrix
            wp.matrix_from_rows(wp.vec2(2.0, 3.0), wp.vec2(5.0, 6.0))
        )

        m[:, 0] = wp.vec3(10.0, 11.0, 12.0) # column vector assignment
        wp.expect_eq(
            m,
            wp.matrix_from_rows(
                wp.vec3(10.0, 2.0, 3.0),
                wp.vec3(11.0, 5.0, 6.0),
                wp.vec3(12.0, 8.0, 9.0),
            )
        )

Negative indices are wrapped around, such that ``-1`` refers to the last element. Slices always create new copies.

Inside kernels, the ``start / stop / step`` values of a slice must be **compile-time constants**.  Simple element indexing (``v[i]``, ``m[i, j]``) may use run-time
expressions.


Unpacking
#########

Python's unpack operator (``*``) can be used in function calls inside kernels to expand vectors, matrices, quaternions, and 1D array slices into individual arguments:

.. code:: python

    @wp.kernel
    def compute(
        arr: wp.array(dtype=float),
    ):
        # Unpack a 1D array slice into a vector.
        v1 = wp.vec3(*arr[:3])
        wp.expect_eq(v1, wp.vec3(1.0, 2.0, 3.0))

        # Unpack a vector into function arguments.
        v2 = wp.vec2(1.0, 2.0)
        x2 = wp.max(*v2)
        wp.expect_eq(x2, 2.0)

        # Build larger vectors by unpacking smaller ones.
        v3 = wp.vec3(1.0, 2.0, 3.0)
        v4 = wp.vec4(*v3, 4.0)
        wp.expect_eq(v4, wp.vec4(1.0, 2.0, 3.0, 4.0))

        # Combine multiple unpacks.
        v5 = wp.vec2(1.0, 2.0)
        v6 = wp.vec2(3.0, 4.0)
        v7 = wp.vec4(*v5, *v6)
        wp.expect_eq(v7, wp.vec4(1.0, 2.0, 3.0, 4.0))

        # Unpack vector slices.
        v8 = wp.vec4(1.0, 2.0, 3.0, 4.0)
        v9 = wp.vec2(*v8[1:3])
        wp.expect_eq(v9, wp.vec2(2.0, 3.0))

        # Unpack matrix rows.
        m1 = wp.mat33(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        m2 = wp.matrix_from_rows(*m1[:2])
        wp.expect_eq(
            m2,
            wp.matrix_from_rows(
                wp.vec3(1.0, 2.0, 3.0),
                wp.vec3(4.0, 5.0, 6.0),
            )
        )


    arr = wp.array((1, 2, 3, 4, 5, 6, 7, 8, 9), dtype=float)
    wp.launch(compute, dim=1, inputs=(arr,))


When unpacking 1D arrays, the slice indices must be **compile-time constants** and non-negative.
The upper bound is required and negative indices or steps are not allowed since the array
length is not known at compile time.


Type Conversions
################

Warp is particularly strict regarding type conversions and does not perform *any* implicit conversion between numeric types.
The user is responsible for ensuring types for most arithmetic operators match, e.g.: ``x = float(0.0) + int(4)`` will result in an error.
This can be surprising for users that are accustomed to C-style conversions but avoids a class of common bugs that result from implicit conversions.

Users should explicitly cast variables to compatible types using constructors like
``int()``, ``float()``, :class:`wp.float16() <warp.float16>`, :class:`wp.uint8() <warp.uint8>`, etc.

.. note::

    For performance reasons, Warp relies on native compilers to perform numeric conversions (e.g., LLVM for CPU and NVRTC for CUDA).
    This is generally not a problem, but in some cases the results may vary on different devices.
    For example, the conversion ``wp.uint8(-1.0)`` results in undefined behavior, since the floating point value -1.0
    is out of range for unsigned integer types.

    C++ compilers are free to handle such cases as they see fit.
    Numeric conversions are only guaranteed to produce correct results when the value being converted is in the range
    supported by the target data type.

Constants
---------

A Warp kernel can access Python variables defined outside of the kernel, which are treated as compile-time constants inside of the kernel.

.. code:: python

    TYPE_SPHERE = wp.constant(0)
    TYPE_CUBE = wp.constant(1)
    TYPE_CAPSULE = wp.constant(2)

    @wp.kernel
    def collide(geometry: wp.array(dtype=int)):

        t = geometry[wp.tid()]

        if t == TYPE_SPHERE:
            print("sphere")
        elif t == TYPE_CUBE:
            print("cube")
        elif t == TYPE_CAPSULE:
            print("capsule")

Note that using :func:`wp.constant() <warp.constant>` is no longer required, but it performs some type checking and can serve as a reminder that the variables are meant to be used as Warp constants.

The behavior is simple and intuitive when the referenced Python variables never change. For details and more complex scenarios, refer to :ref:`External References and Constants<external_references>`. The :ref:`Code Generation<code_generation>` section contains additional information and tips for advanced usage.

Predefined Constants
####################

For convenience, Warp has a number of predefined mathematical constants that
may be used both inside and outside Warp kernels.
The constants in the following table also have lowercase versions defined,
e.g. :const:`wp.E <warp.E>` and :const:`wp.e <warp.e>` are equivalent.

================ =========================
Name             Value
================ =========================
wp.E             2.71828182845904523536
wp.LOG2E         1.44269504088896340736
wp.LOG10E        0.43429448190325182765
wp.LN2           0.69314718055994530942
wp.LN10          2.30258509299404568402
wp.PHI           1.61803398874989484820
wp.PI            3.14159265358979323846
wp.HALF_PI       1.57079632679489661923
wp.TAU           6.28318530717958647692
wp.INF           math.inf
wp.NAN           float('nan')
================ =========================

The :const:`wp.NAN <warp.NAN>` constant may only be used with floating-point types.
Comparisons involving :const:`wp.NAN <warp.NAN>` follow the IEEE 754 standard,
e.g. ``wp.float32(wp.NAN) == wp.float32(wp.NAN)`` returns ``False``.
The :func:`wp.isnan() <warp._src.lang.isnan>` built-in function can be used to determine whether a
value is a NaN (or if a vector, matrix, or quaternion contains a NaN entry).

The following example shows how positive and negative infinity
can be used with floating-point types in Warp using the :const:`wp.inf <warp.inf>` constant:

.. code-block:: python

    @wp.kernel
    def test_infinity(outputs: wp.array(dtype=wp.float32)):
        outputs[0] = wp.float32(wp.inf)        # inf
        outputs[1] = wp.float32(-wp.inf)       # -inf
        outputs[2] = wp.float32(2.0 * wp.inf)  # inf
        outputs[3] = wp.float32(-2.0 * wp.inf) # -inf
        outputs[4] = wp.float32(2.0 / 0.0)     # inf
        outputs[5] = wp.float32(-2.0 / 0.0)    # -inf

Operators
----------

Boolean Operators
#################

+--------------+--------------------------------------+
|   a and b    | True if a and b are True             |
+--------------+--------------------------------------+
|   a or b     | True if a or b is True               |
+--------------+--------------------------------------+
|   not a      | True if a is False, otherwise False  |
+--------------+--------------------------------------+

.. note::
    Expressions such as ``if (a and b):`` currently do not perform short-circuit evaluation.
    In this case ``b`` will also be evaluated even when ``a`` is ``False``.
    Users should take care to ensure that secondary conditions are safe to evaluate (e.g.: do not index out of bounds) in all cases.


Comparison Operators
####################

+----------+---------------------------------------+
| a > b    | True if a strictly greater than b     |
+----------+---------------------------------------+
| a < b    | True if a strictly less than b        |
+----------+---------------------------------------+
| a >= b   | True if a greater than or equal to b  |
+----------+---------------------------------------+
| a <= b   | True if a less than or equal to b     |
+----------+---------------------------------------+
| a == b   | True if a equals b                    |
+----------+---------------------------------------+
| a != b   | True if a not equal to b              |
+----------+---------------------------------------+

Arithmetic Operators
####################

+-----------+--------------------------+
|  a + b    | Addition                 |
+-----------+--------------------------+
|  a - b    | Subtraction              |
+-----------+--------------------------+
|  a * b    | Multiplication           |
+-----------+--------------------------+
|  a @ b    | Matrix multiplication    |
+-----------+--------------------------+
|  a / b    | Floating point division  |
+-----------+--------------------------+
|  a // b   | Floored division         |
+-----------+--------------------------+
|  a ** b   | Exponentiation           |
+-----------+--------------------------+
|  a % b    | Modulus                  |
+-----------+--------------------------+


Since Warp does not perform implicit type conversions, operands should have compatible data types.
Users should use type constructors such as ``float()``, ``int()``, :class:`wp.int64() <warp.int64>`, etc. to cast variables
to the correct type.

The multiplication expression ``a * b`` can also be used to perform matrix multiplication
between `matrix types <Matrices>`_.

Mapping Functions
#################

The :func:`wp.map() <warp.map>` function can be used to apply a function to each element of an array.


Streams
-------

A CUDA stream is a sequence of operations that execute in order on the GPU.
Operations from different streams may run concurrently and may be interleaved by the device scheduler.
See the :ref:`Streams documentation <streams>` for more information on using streams.


Events
------

Events can be inserted into streams and used to synchronize a stream
with a different one. See the :ref:`Events documentation <cuda_events>` for
information on how to use events for cross-stream synchronization
or the :ref:`CUDA Events Timing documentation <cuda_events_profiling>` for
information on how to use events for measuring GPU performance.


.. _graphs:

Graphs
-----------

Launching kernels from Python introduces significant additional overhead compared to C++ or native programs.
To address this, Warp exposes the concept of `CUDA graphs <https://developer.nvidia.com/blog/cuda-graphs/>`_
to allow recording large batches of kernels and replaying them with very little CPU overhead.

To record a series of kernel launches use the :func:`wp.capture_begin() <warp.capture_begin>` and
:func:`wp.capture_end() <warp.capture_end>` API as follows:

.. code:: python

    # begin capture
    wp.capture_begin(device="cuda")

    try:
        # record launches
        for i in range(100):
            wp.launch(kernel=compute1, inputs=[a, b], device="cuda")
    finally:
        # end capture and return a graph object
        graph = wp.capture_end(device="cuda")

We strongly recommend the use of the try-finally pattern when capturing graphs because the `finally`
statement will ensure :func:`wp.capture_end <warp.capture_end>` gets called, even if an exception occurs during
capture, which would otherwise trap the stream in a capturing state.

Once a graph has been constructed it can be executed: ::

    wp.capture_launch(graph)

The :class:`wp.ScopedCapture <warp.ScopedCapture>` context manager can be used to simplify the code and
ensure that :func:`wp.capture_end <warp.capture_end>` is called regardless of exceptions:

.. code:: python

    with wp.ScopedCapture(device="cuda") as capture:
        # record launches
        for i in range(100):
            wp.launch(kernel=compute1, inputs=[a, b], device="cuda")

    wp.capture_launch(capture.graph)

Note that only launch calls are recorded in the graph; any Python executed outside of the kernel code will not be recorded.
Typically it is only beneficial to use CUDA graphs when the graph will be reused or launched multiple times, as
there is a graph-creation overhead.

Conditional Execution
#####################

CUDA 12.4+ supports conditional graph nodes that enable dynamic control flow in CUDA graphs.

:func:`wp.capture_if <warp.capture_if>` creates a dynamic branch based on a condition. The condition value is read from a single-element ``int`` array, where a non-zero value means that the condition is True.

.. code:: python

    # create condition
    cond = wp.zeros(1, dtype=int)

    with wp.ScopedCapture() as capture:
        wp.launch(foo, ...)

        # execute a branch based on the condition value
        wp.capture_if(cond,
                      on_true=...,
                      on_false=...)

        wp.launch(bar, ...)

The condition value can be updated by kernels launched prior to ``capture_if()`` in the same graph (e.g. kernel ``foo`` above) or it can be updated by other means before the graph is launched. Note that during graph capture, the value of the condition is ignored. It is only used when the graph is launched, making dynamic control flow possible.

.. code:: python

    # this will execute the `on_true` branch
    cond.fill_(1)
    wp.capture_launch(capture.graph)

    # this will execute the `on_false` branch
    cond.fill_(0)
    wp.capture_launch(capture.graph)

The ``on_true`` and ``on_false`` callbacks can be previously captured graph objects or Python callback functions.
These callbacks are captured as child graphs of the enclosing graph.
It's possible to specify only one or both callbacks, as needed.
When the parent graph is launched, the correct child graph is executed based on the value of the condition.
This is done efficiently on the device without involving the CPU. 

Here is an example that uses previously captured graphs:

.. code:: python

    @wp.kernel
    def hello_kernel():
        print("Hello")

    @wp.kernel
    def goodbye_kernel():
        print("Goodbye")

    @wp.kernel
    def yes_kernel():
        print("Yes!")

    @wp.kernel
    def no_kernel():
        print("No!")


    # create condition
    cond = wp.zeros(1, dtype=int)

    # capture the on_true branch
    with wp.ScopedCapture() as yes_capture:
        wp.launch(yes_kernel, dim=1)

    # capture the on_false branch
    with wp.ScopedCapture() as no_capture:
        wp.launch(no_kernel, dim=1)

    # capture the main graph
    with wp.ScopedCapture() as capture:
        wp.launch(hello_kernel, dim=1)

        # specify branches using subgraphs
        wp.capture_if(cond,
                      on_true=yes_capture.graph,
                      on_false=no_capture.graph)

        wp.launch(goodbye_kernel, dim=1)

    # execute on_true branch
    cond.fill_(1)
    wp.capture_launch(capture.graph)

    # execute on_false branch
    cond.fill_(0)
    wp.capture_launch(capture.graph)

    wp.synchronize_device()

Here is an example that uses Python callback functions. These callbacks will be captured as child graphs of the main graph:

.. code:: python

    @wp.kernel
    def hello_kernel():
        print("Hello")

    @wp.kernel
    def goodbye_kernel():
        print("Goodbye")

    @wp.kernel
    def yes_kernel():
        print("Yes!")

    @wp.kernel
    def no_kernel():
        print("No!")


    # create condition
    cond = wp.zeros(1, dtype=int)

    # Python callback for the on_true branch
    def yes_callback():
        wp.launch(yes_kernel, dim=1)

    # Python callback for the on_false branch
    def no_callback():
        wp.launch(no_kernel, dim=1)

    # capture the main graph
    with wp.ScopedCapture() as capture:
        wp.launch(hello_kernel, dim=1)

        # specify branches using Python callback functions
        wp.capture_if(cond,
                      on_true=yes_callback,
                      on_false=no_callback)

        wp.launch(goodbye_kernel, dim=1)

    # execute on_true branch
    cond.fill_(1)
    wp.capture_launch(capture.graph)

    # execute on_false branch
    cond.fill_(0)
    wp.capture_launch(capture.graph)

    wp.synchronize_device()

When using Python callback functions, any extra keyword arguments to :func:`wp.capture_if <warp.capture_if>` are forwarded to the callbacks.

:func:`wp.capture_while <warp.capture_while>` creates a dynamic loop based on a condition. Similarly to :func:`wp.capture_if <warp.capture_if>`, the condition value is read from a single-element ``int`` array, where a non-zero value means that the condition is True.

.. code:: python

    # create condition
    cond = wp.zeros(1, dtype=int)

    with wp.ScopedCapture() as capture:
        wp.launch(foo, ...)

        # execute the while_body while the condition is true
        wp.capture_while(cond, while_body=...)

        wp.launch(bar, ...)

The ``while_body`` callback will be executed as long as the condition is non-zero. The callback is responsible for updating the condition value so that the loop eventually terminates. The ``while_body`` argument can be a previously captured graph or a Python callback function. Here is an example that will run some number of iterations, using the condition value as a counter:

.. code:: python

    @wp.kernel
    def hello_kernel():
        print("Hello")

    @wp.kernel
    def goodbye_kernel():
        print("Goodbye")

    @wp.kernel
    def body_kernel(cond: wp.array(dtype=int)):
        tid = wp.tid()
        print(cond[0])
        # decrement the condition counter
        if tid == 0:
            cond[0] -= 1    


    # create condition
    cond = wp.zeros(1, dtype=int)

    # capture the while_body
    with wp.ScopedCapture() as body_capture:
        wp.launch(body_kernel, dim=1, inputs=[cond])

    # capture the main graph
    with wp.ScopedCapture() as capture:
        wp.launch(hello_kernel, dim=1)

        # dynamic loop
        wp.capture_while(cond, while_body=body_capture.graph)

        wp.launch(goodbye_kernel, dim=1)

    # loop 5 times
    cond.fill_(5)
    wp.capture_launch(capture.graph)

    # loop 2 times
    cond.fill_(2)
    wp.capture_launch(capture.graph)

    wp.synchronize_device()


.. note::
    Conditional graph node support is only available if Warp is built using CUDA Toolkit 12.4+ and the NVIDIA driver supports CUDA 12.4+.

.. note::
    Due to a current CUDA limitation, graphs with conditional nodes cannot be used as child graphs. It means that it's not possible to create nested conditional constructs using previously captured graphs. If nesting is required, using Python callback functions is the way to go.

.. note::
    :func:`wp.capture_if <warp.capture_if>` and :func:`wp.capture_while <warp.capture_while>` will work even without graph capture on any device. If there is no active capture, the condition will be evaluated on the CPU and the correct branch will be executed immediately. This makes it possible to write code that works similarly with and without graph capture.


Spatial Computing Primitives
----------------------------

Spatial computing primitives provide efficient data structures for spatial queries and geometric operations.
These include hash grids for particle neighbor searches, bounding volume hierarchies (BVHs) for ray tracing and
collision detection, and mesh types.
These ready-to-use implementations save significant development time compared to building spatial data structures from
scratch, while providing high-performance on the GPU.

.. caution::
    **Object Lifetime Management**: Spatial computing primitives
    (e.g. :class:`wp.HashGrid <warp.HashGrid>`, :class:`wp.Bvh <warp.Bvh>`, etc.) must remain in scope.
    These acceleration data structures are identified by their ``id`` attribute when passed to kernels, 
    but if the Python object is garbage collected, the memory allocated for the primitive may be
    freed, causing crashes and undefined behavior.

    See the :ref:`Object Lifetime Pitfall<object lifetime pitfall>` section below for more information.

Meshes
######

Warp provides a :class:`wp.Mesh <warp.Mesh>` class to manage triangle mesh data. To create a mesh, users provide points, indices, and optionally a velocity array::

    mesh = wp.Mesh(points, indices, velocities)

.. note::
    Mesh objects maintain references to their input geometry buffers. All buffers should live on the same device.

Meshes can be passed to kernels using their ``id`` attribute, which is ``uint64`` value that uniquely identifies the mesh.
Once inside a kernel, you can perform geometric queries against the mesh such as ray-casts or closest-point lookups::

    @wp.kernel
    def raycast(mesh: wp.uint64,
                ray_origin: wp.array(dtype=wp.vec3),
                ray_dir: wp.array(dtype=wp.vec3),
                ray_hit: wp.array(dtype=wp.vec3)):

        tid = wp.tid()

        t = float(0.0)      # hit distance along ray
        u = float(0.0)      # hit face barycentric u
        v = float(0.0)      # hit face barycentric v
        sign = float(0.0)   # hit face sign
        n = wp.vec3()       # hit face normal
        f = int(0)          # hit face index

        color = wp.vec3()

        # ray cast against the mesh
        if wp.mesh_query_ray(mesh, ray_origin[tid], ray_dir[tid], 1.e+6, t, u, v, sign, n, f):

            # if we got a hit then set color to the face normal
            color = n*0.5 + wp.vec3(0.5, 0.5, 0.5)

        ray_hit[tid] = color


Users may update mesh vertex positions at runtime simply by modifying the points buffer.
After modifying point locations users should call :meth:`Mesh.refit()` to rebuild the bounding volume hierarchy (BVH)
structure and ensure that queries work correctly.

.. note::
    Updating Mesh topology (indices) at runtime is not currently supported. Users should instead recreate a new Mesh object.


Hash Grids
##########

Many particle-based simulation methods such as the Discrete Element Method (DEM), or Smoothed Particle Hydrodynamics (SPH), involve iterating over spatial neighbors to compute force interactions. Hash grids are a well-established data structure to accelerate these nearest neighbor queries, and particularly well-suited to the GPU.

To support spatial neighbor queries Warp provides a ``HashGrid`` object that may be created as follows::

    grid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=128, device="cuda")

    grid.build(points=p, radius=r)

``p`` is an array of :class:`wp.vec3 <warp.vec3>` point positions, and ``r`` is the radius to use when building the grid.
Neighbors can then be iterated over inside the kernel code using :func:`wp.hash_grid_query() <warp._src.lang.hash_grid_query>`
and :func:`wp.hash_grid_query_next() <warp._src.lang.hash_grid_query_next>` as follows:

.. code:: python

    @wp.kernel
    def sum(grid : wp.uint64,
            points: wp.array(dtype=wp.vec3),
            output: wp.array(dtype=wp.vec3),
            radius: float):

        tid = wp.tid()

        # query point
        p = points[tid]

        # create grid query around point
        query = wp.hash_grid_query(grid, p, radius)
        index = int(0)

        sum = wp.vec3()

        while(wp.hash_grid_query_next(query, index)):

            neighbor = points[index]

            # compute distance to neighbor point
            dist = wp.length(p-neighbor)
            if (dist <= radius):
                sum += neighbor

        output[tid] = sum

.. note::
    The ``HashGrid`` query will give back all points in *cells* that fall inside the query radius.
    When there are hash conflicts it means that some points outside of query radius will be returned, and users should
    check the distance themselves inside their kernels. The reason the query doesn't do the check itself for each
    returned point is because it's common for kernels to compute the distance themselves, so it would redundant to
    check/compute the distance twice.



Volumes
#######

Sparse volumes are incredibly useful for representing grid data over large domains, such as signed distance fields
(SDFs) for complex objects, or velocities for large-scale fluid flow. Warp supports reading sparse volumetric grids
stored using the `NanoVDB <https://developer.nvidia.com/nanovdb>`_ standard. Users can access voxels directly
or use built-in closest-point or trilinear interpolation to sample grid data from world or local space.

Volume objects can be created directly from Warp arrays containing a NanoVDB grid, from the contents of a
standard ``.nvdb`` file using :func:`load_from_nvdb() <warp.Volume.load_from_nvdb>`,
from an uncompressed in-memory buffer using :func:`load_from_address() <warp.Volume.load_from_address>`,
or from a dense 3D NumPy array using :func:`load_from_numpy() <warp.Volume.load_from_numpy>`.

Volumes can also be created using :meth:`allocate() <warp.Volume.allocate>`, 
:meth:`allocate_by_tiles() <warp.Volume.allocate_by_tiles>` or :meth:`allocate_by_voxels() <warp.Volume.allocate_by_voxels>`. 
The values for a Volume object can be modified in a Warp kernel using :func:`wp.volume_store() <warp._src.lang.volume_store>`.

.. note::
    Warp does not currently support modifying the topology of sparse volumes at runtime.

Below we give an example of creating a Volume object from an existing NanoVDB file::

    # open NanoVDB file on disk
    file = open("mygrid.nvdb", "rb")

    # create Volume object
    volume = wp.Volume.load_from_nvdb(file, device="cpu")

.. note::
    Files written by the NanoVDB library, commonly marked by the ``.nvdb`` extension, can contain multiple grids with
    various compression methods, but a :class:`Volume` object represents a single NanoVDB grid. 
    The first grid is loaded by default, then  Warp volumes corresponding to the other grids in the file can be created
    using repeated calls to :func:`load_next_grid() <warp.Volume.load_next_grid>`.
    NanoVDB's uncompressed and zip-compressed file formats are supported out-of-the-box, blosc compressed files require
    the `blosc` Python package to be installed.

To sample the volume inside a kernel we pass a reference to it by ID, and use the built-in sampling modes::

    @wp.kernel
    def sample_grid(volume: wp.uint64,
                    points: wp.array(dtype=wp.vec3),
                    samples: wp.array(dtype=float)):

        tid = wp.tid()

        # load sample point in world-space
        p = points[tid]

        # transform position to the volume's local-space
        q = wp.volume_world_to_index(volume, p)

        # sample volume with trilinear interpolation
        f = wp.volume_sample(volume, q, wp.Volume.LINEAR, dtype=float)

        # write result
        samples[tid] = f

Warp also supports NanoVDB index grids, which provide a memory-efficient linearization of voxel indices that can refer 
to values in arbitrarily shaped arrays::

    @wp.kernel
    def sample_index_grid(volume: wp.uint64,
                         points: wp.array(dtype=wp.vec3),
                         voxel_values: wp.array(dtype=Any)):

        tid = wp.tid()

        # load sample point in world-space
        p = points[tid]

        # transform position to the volume's local-space
        q = wp.volume_world_to_index(volume, p)

        # sample volume with trilinear interpolation
        background_value = voxel_values.dtype(0.0)
        f = wp.volume_sample_index(volume, q, wp.Volume.LINEAR, voxel_values, background_value)

The coordinates of all indexable voxels can be recovered using :func:`get_voxels() <warp.Volume.get_voxels>`.
NanoVDB grids may also contain embedded *blind* data arrays; those can be accessed with the 
:func:`feature_array() <warp.Volume.feature_array>` function.


.. seealso:: `Built-Ins <../language_reference/builtins.html#volumes>`__ for the volume functions available in kernels.


Textures
########

Warp provides :class:`Texture2D` and :class:`Texture3D` classes for hardware-accelerated texture sampling
on CUDA devices. Textures support bilinear/trilinear interpolation and various addressing modes
(wrap, clamp, mirror, border), making them ideal for efficiently sampling regularly-gridded data.

Textures can be created from NumPy arrays or Warp arrays::

    import warp as wp
    import numpy as np

    # Create a 256x256 RGBA 2D texture
    data = np.random.rand(256, 256, 4).astype(np.float32)
    tex2d = wp.Texture2D(data, filter_mode=wp.TextureFilterMode.LINEAR, device="cuda:0")

    # Create a 64x64x64 single-channel 3D texture
    data3d = np.random.rand(64, 64, 64).astype(np.float32)
    tex3d = wp.Texture3D(data3d, filter_mode=wp.TextureFilterMode.LINEAR, device="cuda:0")

Textures can be sampled inside kernels using the :func:`wp.texture_sample() <warp._src.lang.texture_sample>` function::

    @wp.kernel
    def sample_texture(
        tex: wp.Texture2D,
        uvs: wp.array(dtype=wp.vec2f),
        output: wp.array(dtype=float),
    ):
        tid = wp.tid()
        uv = uvs[tid]
        output[tid] = wp.texture_sample(tex, uv, dtype=float)

Supported data types include ``uint8``, ``uint16``, and ``float32``. Integer textures (uint8, uint16)
are automatically normalized to the [0, 1] range when sampled.

.. seealso:: `Reference <language_reference/builtins.html#textures>`__ for the texture sampling functions available in kernels.


Bounding Volume Hierarchies (BVH)
#################################

The :class:`wp.Bvh <warp.Bvh>` class can be used to create a BVH for a group of bounding volumes. This object can then be traversed
to determine which parts are intersected by a ray using :func:`wp.bvh_query_ray <warp._src.lang.bvh_query_ray>` and which parts overlap
with a certain bounding volume using :func:`wp.bvh_query_aabb() <warp._src.lang.bvh_query_aabb>`.

The following snippet demonstrates how to create a :class:`wp.Bvh <warp.Bvh>` object from 100 random bounding volumes:

.. code:: python

    rng = np.random.default_rng(123)

    num_bounds = 100
    lowers = rng.random(size=(num_bounds, 3)) * 5.0
    uppers = lowers + rng.random(size=(num_bounds, 3)) * 5.0

    device_lowers = wp.array(lowers, dtype=wp.vec3, device="cuda:0")
    device_uppers = wp.array(uppers, dtype=wp.vec3, device="cuda:0")

    bvh = wp.Bvh(device_lowers, device_uppers)


Example: BVH Ray Traversal
^^^^^^^^^^^^^^^^^^^^^^^^^^

An example of performing a ray traversal on the data structure is as follows:

.. code:: python

    @wp.kernel
    def bvh_query_ray(
        bvh_id: wp.uint64,
        start: wp.vec3,
        dir: wp.vec3,
        bounds_intersected: wp.array(dtype=wp.bool),
    ):
        query = wp.bvh_query_ray(bvh_id, start, dir)
        bounds_nr = wp.int32(0)

        while wp.bvh_query_next(query, bounds_nr):
            # The ray intersects the volume with index bounds_nr
            bounds_intersected[bounds_nr] = True


    bounds_intersected = wp.zeros(shape=(num_bounds), dtype=wp.bool, device="cuda:0")
    query_start = wp.vec3(0.0, 0.0, 0.0)
    query_dir = wp.normalize(wp.vec3(1.0, 1.0, 1.0))

    wp.launch(
        kernel=bvh_query_ray,
        dim=1,
        inputs=[bvh.id, query_start, query_dir, bounds_intersected],
        device="cuda:0",
    )

The Warp kernel ``bvh_query_ray`` is launched with a single thread, provided the unique :class:`wp.uint64 <warp.uint64>`
identifier of the :class:`wp.Bvh <warp.Bvh>` object, parameters describing the ray, and an array to store the results.
In ``bvh_query_ray``, :func:`wp.bvh_query_ray() <warp._src.lang.bvh_query_ray>` is called once to obtain an object that is stored in the
variable ``query``. An integer is also allocated as ``bounds_nr`` to store the volume index of the traversal.
A while statement is used for the actual traversal using :func:`wp.bvh_query_next() <warp._src.lang.bvh_query_next>`,
which returns ``True`` as long as there are intersecting bounds.

Example: BVH Volume Traversal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similar to the ray-traversal example, we can perform volume traversal to find the volumes that are overlapping with
a specified bounding box.

.. code:: python

    @wp.kernel
    def bvh_query_aabb(
        bvh_id: wp.uint64,
        lower: wp.vec3,
        upper: wp.vec3,
        bounds_intersected: wp.array(dtype=wp.bool),
    ):
        query = wp.bvh_query_aabb(bvh_id, lower, upper)
        bounds_nr = wp.int32(0)

        while wp.bvh_query_next(query, bounds_nr):
            # The volume with index bounds_nr overlaps with
            # the (lower,upper) bounding box
            bounds_intersected[bounds_nr] = True


    bounds_intersected = wp.zeros(shape=(num_bounds), dtype=wp.bool, device="cuda:0")
    query_lower = wp.vec3(4.0, 4.0, 4.0)
    query_upper = wp.vec3(6.0, 6.0, 6.0)

    wp.launch(
        kernel=bvh_query_aabb,
        dim=1,
        inputs=[bvh.id, query_lower, query_upper, bounds_intersected],
        device="cuda:0",
    )

The kernel is nearly identical to the ray-traversal example, except we obtain ``query`` using
:func:`wp.bvh_query_aabb() <warp._src.lang.bvh_query_aabb>`.

.. _object lifetime pitfall:

Object Lifetime Pitfall
#######################

When working with spatial computing primitives like :class:`wp.HashGrid <warp.HashGrid>` and :class:`wp.Bvh <warp.Bvh>`,
it's crucial to understand how Python's garbage collection interacts with these objects.
The following example demonstrate a common mistake and how to avoid it.

**Common Pitfall**: Creating objects in loops and only storing their IDs

.. code-block:: python
    
    # WRONG - objects may be garbage collected
    hash_grids = []
    for i in range(10):
        grid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=128)
        grid.build(points=particle_positions[i], radius=search_radius)
        hash_grids.append(grid.id)  # Only storing the ID
    
    # RIGHT - maintain references to the objects
    hash_grid_objects = []
    for i in range(10):
        grid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=128)
        grid.build(points=particle_positions[i], radius=search_radius)
        hash_grid_objects.append(grid)  # Keep the object alive
    
    # Create Warp array for kernel execution when needed
    grid_ids_array = wp.array([x.id for x in hash_grid_objects], dtype=wp.uint64, device="cuda")
    
    wp.launch(my_kernel, dim=10, inputs=[grid_ids_array])

**Why This Happens**: When you only store the ``id`` attribute (which is a :class:`wp.uint64 <warp.uint64>` pointer),
Python's garbage collector may free the original object if no other references exist. This leads to 
undefined behavior when the kernel tries to access the freed memory.

**Common Problematic Scenarios**:

1. **Creating objects in loops** and only storing their IDs
2. **Creating objects in functions** and returning only the ID  
3. **Creating objects as temporary variables** that get overwritten

Always maintain references to spatial computing primitive objects
(like :class:`wp.HashGrid <warp.HashGrid>`, :class:`wp.Bvh <warp.Bvh>`, etc.) rather than just their ID values.
This is especially important in loops, functions, and temporary variables where object scope might be unclear.

Marching Cubes
--------------

The :class:`wp.MarchingCubes <warp.MarchingCubes>` class can be used to extract a 2-D mesh approximating an
isosurface of a 3-D scalar field. The resulting triangle mesh can be saved to a USD
file using the :class:`warp.render.UsdRenderer`.

See :github:`warp/examples/core/example_marching_cubes.py` for a usage example.


Profiling
---------

:class:`wp.ScopedTimer <warp.ScopedTimer>` objects can be used to gain some basic insight into the performance of Warp applications:

.. code:: python

    with wp.ScopedTimer("grid build"):
        self.grid.build(self.x, self.point_radius)

This results in a printout at runtime to the standard output stream like:

.. code:: console

    grid build took 0.06 ms

See :doc:`../deep_dive/profiling` documentation for more information.


Interprocess Communication (IPC)
--------------------------------

Interprocess communication can be used to share Warp arrays and events across
processes without creating copies of the underlying data.

Some basic requirements for using IPC include:

* Linux operating system (note however that integrated devices like NVIDIA
  Jetson do not support CUDA IPC)
* The array must be allocated on a GPU device using the default memory allocator (see :doc:`../deep_dive/allocators`)

  The :class:`wp.ScopedMempool <warp.ScopedMempool>` context manager is useful for temporarily disabling
  memory pools for the purpose of allocating arrays that can be shared using IPC.

Support for IPC on a device is indicated by the :attr:`is_ipc_supported <warp.Device.is_ipc_supported>`
attribute of the :class:`Device <warp.Device>`. This device attribute will be
``None`` to indicate that IPC support could not be determined using the CUDA API.

To share a Warp array between processes, use :meth:`array.ipc_handle` in the
originating process to obtain an IPC handle for the array's memory allocation.
The handle is a ``bytes`` object with a length of 64.
The IPC handle along with information about the array (data type, shape, and
optionally strides) should be shared with another process, e.g. via shared
memory or files.
Another process can use this information to import the original array by
calling :func:`from_ipc_handle`.

Events can be shared in a similar manner, but they must be constructed with
``interprocess=True``. Additionally, events cannot be created with both
``interprocess=True`` and ``enable_timing=True``. Use :meth:`Event.ipc_handle`
in the originating process to obtain an IPC handle for the event. Another
process can use this information to import the original event by calling
:func:`event_from_ipc_handle`.




LTO Cache
---------

:ref:`MathDx <mathdx>` generates Link-Time Optimization (LTO) files for GEMM, Cholesky, and FFT tile operations.
Warp caches these to speed up kernel compilation. Each LTO file maps to a specific Linear Algebra
solver configuration, and is otherwise independent of the kernel in which its corresponding routine
is called. Therefore, LTOs are stored in a cache that is independent of a given module's kernel cache,
and will remain cached even if :func:`wp.clear_kernel_cache() <warp.clear_kernel_cache>` is called.
:func:`wp.clear_lto_cache() <warp.clear_lto_cache>` can be used to clear the LTO cache.


Random Number Generation
------------------------

To generate random numbers in a Warp kernel, use the :func:`wp.rand_init() <warp._src.lang.rand_init>` built-in
inside the kernel to initialize a random number generator, followed by a call to any of Warp's random number
built-ins. For example:

.. testcode::

    import warp as wp

    @wp.kernel
    def rand_kernel(seed: int, out_rand: wp.array(dtype=float)):
        i = wp.tid()
        rng = wp.rand_init(seed, i)
        out_rand[i] = wp.randf(rng)

    seed = 123
    out_rand = wp.empty(3, dtype=float)
    wp.launch(rand_kernel, dim=3, inputs=[seed, out_rand])

    print(out_rand.numpy())

.. testoutput::

    [0.1415146 0.9632247 0.6449367]

Warp uses a PCG (Permuted Congruential Generator) for pseudo-random number generation [1]_.
:func:`wp.rand_init() <warp._src.lang.rand_init>` hashes the seed and offset with two nested calls
to a PCG routine to produce a 32-bit unsigned integer representing the RNG state. The offset
in a kernel is necessary to ensure that each thread generates a unique sequence of numbers. Were it
absent, all threads would share the same RNG state.

All calls to Warp random functions accept a uint32 RNG state, which is internally updated when 
the function is called. Hence, consecutive calls to the same random function will generate different
numbers, e.g.:

.. testcode::

    import warp as wp

    @wp.kernel
    def rand_kernel(seed: int, output: wp.array(dtype=float)):
        i = wp.tid()
        rng = wp.rand_init(seed, i)
        output[0] = wp.randf(rng)
        output[1] = wp.randf(rng)

    output = wp.zeros(2, dtype=float)
    wp.launch(rand_kernel, dim=1, inputs=[42], outputs=[output])
    print(output.numpy())

.. testoutput::

    [0.86597514 0.1859147 ]

Avoiding Correlated Sequences
#############################

Care must be taken when two different kernels generate random numbers. If the same seed is used for both,
any quantities computed from the generated random numbers will be correlated. Similarly, if the same seed
is passed to a kernel that is launched multiple times, the same random numbers will be generated for each launch.
To generate different sequences across launches, use a different seed for each launch:

.. testcode::

    import warp as wp

    def increment_seed(seed: int) -> int:
        return seed + 1

    @wp.kernel
    def rand_kernel(seed: int, out_rand: wp.array(dtype=float)):
        i = wp.tid()
        rng = wp.rand_init(seed, i)
        out_rand[i] = wp.randf(rng)

    seed = 123
    out_rand = wp.empty(3, dtype=float)
    wp.launch(rand_kernel, dim=3, inputs=[seed, out_rand])
    print(out_rand.numpy())

    seed = increment_seed(seed)
    wp.launch(rand_kernel, dim=3, inputs=[seed, out_rand])
    print(out_rand.numpy())

.. testoutput::

    [0.1415146 0.9632247 0.6449367]
    [0.37815237 0.68619    0.7548081 ]

.. [1] Mark Jarzynski and Marc Olano, `Hash Functions for GPU Rendering <https://jcgt.org/published/0009/03/02/>`_,
   Journal of Computer Graphics Techniques (JCGT), vol. 9, no. 3, 20–38, 2020.