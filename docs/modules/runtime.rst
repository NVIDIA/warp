Runtime Reference
=================

.. currentmodule:: warp

This section describes the Warp Python runtime API, how to manage memory, launch kernels, and high-level functionality
for dealing with objects such as meshes and volumes. The APIs described in this section are intended to be used at
the *Python Scope* and run inside the CPython interpreter. For a comprehensive list of functions available at
the *Kernel Scope*, please see the :doc:`functions` section.

Kernels
-------

Kernels are launched with the :func:`wp.launch() <launch>` function on a specific device (CPU/GPU)::

    wp.launch(simple_kernel, dim=1024, inputs=[a, b, c], device="cuda")

Kernels may be launched with multi-dimensional grid bounds. In this case threads are not assigned a single index,
but a coordinate in an n-dimensional grid, e.g.::

    wp.launch(complex_kernel, dim=(128, 128, 3), ...)

Launches a 3D grid of threads with dimension 128 x 128 x 3. To retrieve the 3D index for each thread use the following syntax::

    i,j,k = wp.tid()

.. note::
    Currently kernels launched on CPU devices will be executed in serial.
    Kernels launched on CUDA devices will be launched in parallel with a fixed block-size.

.. note::
    Note that all the kernel inputs must live on the target device, or a runtime exception will be raised.

.. autofunction:: launch

Large Kernel Launches
#####################

A limitation of Warp is that each dimension of the grid used to launch a kernel must be representable as a 32-bit
signed integer. Therefore, no single dimension of a grid should exceed :math:`2^{31}-1`.

Warp also currently uses a fixed block size of 256 (CUDA) threads per block.
By default, Warp will try to process one element from the Warp grid in one CUDA thread.
This is not always possible for kernels launched with multi-dimensional grid bounds, as there are
`hardware limitations <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability>`_
on CUDA block dimensions.

Warp will automatically resort to using
`grid-stride loops <https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/>`_ when
it is not possible for a CUDA thread to process only one element from the Warp grid
When this happens, some CUDA threads may process more than one element from the Warp grid.
Users can also set the ``max_blocks`` parameter to fine-tune the grid-striding behavior of kernels, even for kernels that are otherwise
able to process one Warp-grid element per CUDA thread. 

Runtime Kernel Specialization
#############################

It is often desirable to specialize kernels for different types, constants, or functions.
We can achieve this through the use of runtime kernel specialization using Python closures.

For example, we might require a variety of kernels that execute particular functions for each item in an array.
We might also want this function call to be valid for a variety of data types. Making use of closure and generics, we can generate
these kernels using a single kernel definition: ::

    def make_kernel(func, dtype):
        def closure_kernel_fn(data: wp.array(dtype=dtype), out: wp.array(dtype=dtype)):
            tid = wp.tid()
            out[tid] = func(data[tid])

        return wp.Kernel(closure_kernel_fn)

In practice, we might use our kernel generator, ``make_kernel()`` as follows: ::

    @wp.func
    def sqr(x: Any) -> Any:
        return x * x

    @wp.func
    def cube(x: Any) -> Any:
        return sqr(x) * x

    sqr_float = make_kernel(sqr, wp.float32)
    cube_double = make_kernel(cube, wp.float64)

    arr = [1.0, 2.0, 3.0]
    N = len(arr)

    data_float = wp.array(arr, dtype=wp.float32, device=device)
    data_double = wp.array(arr, dtype=wp.float64, device=device)

    out_float = wp.zeros(N, dtype=wp.float32, device=device)
    out_double = wp.zeros(N, dtype=wp.float64, device=device)

    wp.launch(sqr_float, dim=N, inputs=[data_float], outputs=[out_float], device=device)
    wp.launch(cube_double, dim=N, inputs=[data_double], outputs=[out_double], device=device)

We can specialize kernel definitions over warp constants similarly. The following generates kernels that add a specified constant
to a generic-typed array value: ::

    def make_add_kernel(key, constant):
        def closure_kernel_fn(data: wp.array(dtype=Any), out: wp.array(dtype=Any)):
            tid = wp.tid()
            out[tid] = data[tid] + constant

        return wp.Kernel(closure_kernel_fn, key=key)

    add_ones_int = make_add_kernel("add_one", wp.constant(1))
    add_ones_vec3 = make_add_kernel("add_ones_vec3", wp.constant(wp.vec3(1.0, 1.0, 1.0)))

    a = wp.zeros(2, dtype=int)
    b = wp.zeros(2, dtype=wp.vec3)

    a_out = wp.zeros_like(a)
    b_out = wp.zeros_like(b)

    wp.launch(add_ones_int, dim=a.size, inputs=[a], outputs=[a_out], device=device)
    wp.launch(add_ones_vec3, dim=b.size, inputs=[b], outputs=[b_out], device=device)


Arrays
------

Arrays are the fundamental memory abstraction in Warp; they are created through the following global constructors: ::

    wp.empty(shape=1024, dtype=wp.vec3, device="cpu")
    wp.zeros(shape=1024, dtype=float, device="cuda")
    wp.full(shape=1024, value=10, dtype=int, device="cuda")


Arrays can also be constructed directly from ``numpy`` ndarrays as follows: ::

    r = np.random.rand(1024)

    # copy to Warp owned array
    a = wp.array(r, dtype=float, device="cpu")

    # return a Warp array wrapper around the NumPy data (zero-copy)
    a = wp.array(r, dtype=float, copy=False, device="cpu")

    # return a Warp copy of the array data on the GPU
    a = wp.array(r, dtype=float, device="cuda")

Note that for multi-dimensional data the ``dtype`` parameter must be specified explicitly, e.g.: ::

    r = np.random.rand((1024, 3))

    # initialize as an array of vec3 objects
    a = wp.array(r, dtype=wp.vec3, device="cuda")

If the shapes are incompatible, an error will be raised.


Arrays can be moved between devices using the ``array.to()`` method: ::

    host_array = wp.array(a, dtype=float, device="cpu")

    # allocate and copy to GPU
    device_array = host_array.to("cuda")

Additionally, arrays can be copied directly between memory spaces: ::

    src_array = wp.array(a, dtype=float, device="cpu")
    dest_array = wp.empty_like(host_array)

    # copy from source CPU buffer to GPU
    wp.copy(dest_array, src_array)

.. autoclass:: array
    :members:
    :undoc-members:

Multi-dimensional Arrays
########################

Multi-dimensional arrays can be constructed by passing a tuple of sizes for each dimension, e.g.: the following constructs a 2d array of size 1024x16::

    wp.zeros(shape=(1024, 16), dtype=float, device="cuda")

When passing multi-dimensional arrays to kernels users must specify the expected array dimension inside the kernel signature,
e.g. to pass a 2d array to a kernel the number of dims is specified using the ``ndim=2`` parameter::

    @wp.kernel
    def test(input: wp.array(dtype=float, ndim=2)):

Type-hint helpers are provided for common array sizes, e.g.: ``array2d()``, ``array3d()``, which are equivalent to calling ``array(..., ndim=2)```, etc. To index a multi-dimensional array use a the following kernel syntax::

    # returns a float from the 2d array
    value = input[i,j]

To create an array slice use the following syntax, where the number of indices is less than the array dimensions::

    # returns an 1d array slice representing a row of the 2d array
    row = input[i]

Slice operators can be concatenated, e.g.: ``s = array[i][j][k]``. Slices can be passed to ``wp.func`` user functions provided
the function also declares the expected array dimension. Currently only single-index slicing is supported.

.. note::
    Currently Warp limits arrays to 4 dimensions maximum. This is in addition to the contained datatype, which may be 1-2 dimensional for vector and matrix types such as ``vec3``, and ``mat33``.


The following construction methods are provided for allocating zero-initialized and empty (non-initialized) arrays:

.. autofunction:: zeros
.. autofunction:: zeros_like
.. autofunction:: full
.. autofunction:: full_like
.. autofunction:: empty
.. autofunction:: empty_like
.. autofunction:: copy
.. autofunction:: clone

Matrix Multiplication
#####################

Warp 2D array multiplication is built on NVIDIA's CUTLASS library, which enables fast matrix multiplication of large arrays on the GPU.

If no GPU is detected, matrix multiplication falls back to Numpy's implementation on the CPU.

Matrix multiplication is fully differentiable, and can be recorded on the tape like so::

    tape = wp.Tape()
    with tape:
        wp.matmul(A, B, C, D, device=device)
        wp.launch(loss_kernel, dim=(m, n), inputs=[D, loss], device=device)

    tape.backward(loss=loss)
    A_grad = A.grad.numpy()

Using the ``@`` operator (``D = A @ B``) will default to the same CUTLASS algorithm used in ``wp.matmul``.

.. autofunction:: matmul

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
| float32 | single precision float |
+---------+------------------------+
| float64 | double precision float |
+---------+------------------------+

Warp supports ``float`` and ``int`` as aliases for ``wp.float32`` and ``wp.int32`` respectively.

.. _vec:

Vectors
#######

Warp provides built-in math and geometry types for common simulation and graphics problems.
A full reference for operators and functions for these types is available in the :doc:`/modules/functions`.

Warp supports vectors of numbers with an arbitrary length/numeric type. The built in concrete types are as follows:

+-----------------------+------------------------------------------------+
| vec2 vec3 vec4        | 2d, 3d, 4d vector of default precision floats  |
+-----------------------+------------------------------------------------+
| vec2f vec3f vec4f     | 2d, 3d, 4d vector of single precision floats   |
+-----------------------+------------------------------------------------+
| vec2d vec3d vec4d     | 2d, 3d, 4d vector of double precision floats   |
+-----------------------+------------------------------------------------+
| vec2h vec3h vec4h     | 2d, 3d, 4d vector of half precision floats     |
+-----------------------+------------------------------------------------+
| vec2ub vec3ub vec4ub  | 2d, 3d, 4d vector of half precision floats     |
+-----------------------+------------------------------------------------+
| spatial_vector        | 6d vector of single precision floats           |
+-----------------------+------------------------------------------------+
| spatial_vectorf       | 6d vector of single precision floats           |
+-----------------------+------------------------------------------------+
| spatial_vectord       | 6d vector of double precision floats           |
+-----------------------+------------------------------------------------+
| spatial_vectorh       | 6d vector of half precision floats             |
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
        v = wp.vector(dtype=wp.float64, length=5)

        # scalar initialize a vector of 5 doubles to the same value:
        v = wp.vector(wp.float64(1.0), length=5)

        # component-wise initialize a vector of 5 doubles
        v = wp.vector(wp.float64(1.0),
                      wp.float64(2.0),
                      wp.float64(3.0),
                      wp.float64(4.0),
                      wp.float64(5.0))


These can be used with all the standard vector arithmetic operators, e.g.: ``+``, ``-``, scalar multiplication, and can also be transformed using matrices with compatible dimensions, potentially returning vectors with a different length.

.. _mat:

Matrices
########

Matrices with arbitrary shapes/numeric types are also supported. The built in concrete matrix types are as follows:

+--------------------------+-----------------------------------------------------+
| mat22 mat33 mat44        | 2,3 and 4d square matrix of default precision       |
+--------------------------+-----------------------------------------------------+
| mat22f mat33f mat44f     | 2,3 and 4d square matrix of single precision floats |
+--------------------------+-----------------------------------------------------+
| mat22d mat33d mat44d     | 2,3 and 4d square matrix of double precision floats |
+--------------------------+-----------------------------------------------------+
| mat22h mat33h mat44h     | 2,3 and 4d square matrix of half precision floats   |
+--------------------------+-----------------------------------------------------+
| spatial_matrix           | 6x6 matrix of single precision floats               |
+--------------------------+-----------------------------------------------------+
| spatial_matrixf          | 6x6 matrix of single precision floats               |
+--------------------------+-----------------------------------------------------+
| spatial_matrixd          | 6x6 matrix of double precision floats               |
+--------------------------+-----------------------------------------------------+
| spatial_matrixh          | 6x6 matrix of half precision floats                 |
+--------------------------+-----------------------------------------------------+

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

These can be used inside a kernel: ::

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

It's also possible to directly create anonymously typed instances inside kernels where the type is inferred from constructor arguments as follows: ::

    @wp.kernel
    def compute( ... ):
        ...

        # create a 3x2 half precision matrix from components (row major ordering):
        m = wp.matrix(
            wp.float16(1.0), wp.float16(2.0),
            wp.float16(1.0), wp.float16(2.0),
            wp.float16(1.0), wp.float16(2.0),
            shape=(3,2))

        # zero initialize a 3x2 half precision matrix:
        m = wp.matrix(wp.float16(0.0),shape=(3,2))

        # create a 5x5 double precision identity matrix:
        m = wp.identity(n=5, dtype=wp.float64)

As with vectors, you can do standard matrix arithmetic with these variables, along with multiplying matrices with compatible shapes and potentially returning a matrix with a new shape.

.. _quat:

Quaternions
###########

Warp supports quaternions with the layout ``i, j, k, w`` where ``w`` is the real part. Here are the built in concrete quaternion types:

+-----------------+-----------------------------------------------+
| quat            | Default precision floating point quaternion   |
+-----------------+-----------------------------------------------+
| quatf           | Single precision floating point quaternion    |
+-----------------+-----------------------------------------------+
| quatd           | Double precision floating point quaternion    |
+-----------------+-----------------------------------------------+
| quath           | Half precision floating point quaternion      |
+-----------------+-----------------------------------------------+

Quaternions can be used to transform vectors as follows: ::

    @wp.kernel
    def compute( ... ):
        ...

        # construct a 30 degree rotation around the x-axis
        q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.degrees(30.0))

        # rotate an axis by this quaternion
        v = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))


As with vectors and matrices, you can declare quaternion types with an arbitrary numeric type like so: ::

    quatd = wp.types.quaternion(dtype=wp.float64)

You can also create identity quaternion and anonymously typed instances inside a kernel like so: ::

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

Transforms are 7d vectors of floats representing a spatial rigid body transformation in format (p, q) where p is a 3d vector, and q is a quaternion.

+-----------------+--------------------------------------------+
| transform       | Default precision floating point transform |
+-----------------+--------------------------------------------+
| transformf      | Single precision floating point transform  |
+-----------------+--------------------------------------------+
| transformd      | Double precision floating point transform  |
+-----------------+--------------------------------------------+
| transformh      | Half precision floating point transform    |
+-----------------+--------------------------------------------+

Transforms can be constructed inside kernels from translation and rotation parts: ::

    @wp.kernel
    def compute( ... ):
        ...

        # create a transform from a vector/quaternion:
        t = wp.transform(
                wp.vec3(1.0, 2.0, 3.0),
                wp.quat_from_axis_angle(0.0, 1.0, 0.0, wp.degrees(30.0)))

        # transform a point
        p = wp.transform_point(t, wp.vec3(10.0, 0.5, 1.0))

        # transform a vector (ignore translation)
        p = wp.transform_vector(t, wp.vec3(10.0, 0.5, 1.0))


As with vectors and matrices, you can declare transform types with an arbitrary numeric type using ``wp.types.transformation()``, for example: ::

    transformd = wp.types.transformation(dtype=wp.float64)

You can also create identity transforms and anonymously typed instances inside a kernel like so: ::

    @wp.kernel
    def compute( ... ):

        # create double precision identity transform:
        qd = wp.transform_identity(dtype=wp.float64)

Structs
#######

Users can define custom structure types using the ``@wp.struct`` decorator as follows::

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

Example: Using a custom struct in gradient computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    import numpy as np

    import warp as wp

    wp.init()


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
    print(tape.gradients[ts].a)


Type Conversions
################

Warp is particularly strict regarding type conversions and does not perform *any* implicit conversion between numeric types.
The user is responsible for ensuring types for most arithmetic operators match, e.g.: ``x = float(0.0) + int(4)`` will result in an error.
This can be surprising for users that are accustomed to C-style conversions but avoids a class of common bugs that result from implicit conversions.

.. note::
    Warp does not currently perform implicit type conversions between numeric types.
    Users should explicitly cast variables to compatible types using constructors like
    ``int()``, ``float()``, ``wp.float16()``, ``wp.uint8()``, etc.

Constants
---------

In general, Warp kernels cannot access variables in the global Python interpreter state. One exception to this is for compile-time constants, which may be declared globally (or as class attributes) and folded into the kernel definition.

Constants are defined using the ``wp.constant()`` function. An example is shown below::

    TYPE_SPHERE = wp.constant(0)
    TYPE_CUBE = wp.constant(1)
    TYPE_CAPSULE = wp.constant(2)

    @wp.kernel
    def collide(geometry: wp.array(dtype=int)):

        t = geometry[wp.tid()]

        if (t == TYPE_SPHERE):
            print("sphere")
        if (t == TYPE_CUBE):
            print("cube")
        if (t == TYPE_CAPSULE):
            print("capsule")


.. autoclass:: constant


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
|  a / b    | Floating point division  |
+-----------+--------------------------+
|  a // b   | Floored division         |
+-----------+--------------------------+
|  a ** b   | Exponentiation           |
+-----------+--------------------------+
|  a % b    | Modulus                  |
+-----------+--------------------------+

.. note::
    Since implicit conversions are not performed arguments types to operators should match.
    Users should use type constructors, e.g.: ``float()``, ``int()``, ``wp.int64()``, etc. to cast variables
    to the correct type. Also note that the multiplication expression ``a * b`` is used to represent scalar
    multiplication and matrix multiplication. The ``@`` operator is not currently supported.

Graphs
-----------

Launching kernels from Python introduces significant additional overhead compared to C++ or native programs.
To address this, Warp exposes the concept of `CUDA graphs <https://developer.nvidia.com/blog/cuda-graphs/>`_
to allow recording large batches of kernels and replaying them with very little CPU overhead.

To record a series of kernel launches use the :func:`wp.capture_begin() <capture_begin>` and
:func:`wp.capture_end() <capture_end>` API as follows:

.. code:: python

    # begin capture
    wp.capture_begin()

    try:
        # record launches
        for i in range(100):
            wp.launch(kernel=compute1, inputs=[a, b], device="cuda")
    finally:
        # end capture and return a graph object
        graph = wp.capture_end()

We strongly recommend the use of the the try-finally pattern when capturing graphs because
:func:`wp.capture_begin <capture_begin>` disables Python garbage collection so that Warp objects are not
garbage-collected during graph capture (CUDA does not allow memory to be deallocated during graph capture).
:func:`wp.capture_end <capture_end>` reenables garbage collection.

Once a graph has been constructed it can be executed: ::

    wp.capture_launch(graph)

Note that only launch calls are recorded in the graph, any Python executed outside of the kernel code will not be recorded.
Typically it is only beneficial to use CUDA graphs when the graph will be reused or launched multiple times.

.. autofunction:: capture_begin
.. autofunction:: capture_end
.. autofunction:: capture_launch

Bounding Value Hierarchies (BVH)
--------------------------------

The :class:`wp.Bvh <Bvh>` class can be used to create a BVH for a group of bounding volumes. This object can then be traversed
to determine which parts are intersected by a ray using :func:`bvh_query_ray` and which parts are fully contained
within a certain bounding volume using :func:`bvh_query_aabb`.

The following snippet demonstrates how to create a :class:`wp.Bvh <Bvh>` object from 100 random bounding volumes:

.. code:: python

    rng = np.random.default_rng(123)

    num_bounds = 100
    lowers = rng.random(size=(num_bounds, 3)) * 5.0
    uppers = lowers + rng.random(size=(num_bounds, 3)) * 5.0

    device_lowers = wp.array(lowers, dtype=wp.vec3, device="cuda:0")
    device_uppers = wp.array(uppers, dtype=wp.vec3, device="cuda:0")

    bvh = wp.Bvh(device_lowers, device_uppers)

.. autoclass:: Bvh
    :members:

Example: BVH Ray Traversal
##########################

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

The Warp kernel ``bvh_query_ray`` is launched with a single thread, provided the unique :class:`uint64`
identifier of the :class:`wp.Bvh <Bvh>` object, parameters describing the ray, and an array to store the results.
In ``bvh_query_ray``, :func:`wp.bvh_query_ray() <bvh_query_ray>` is called once to obtain an object that is stored in the
variable ``query``. An integer is also allocated as ``bounds_nr`` to store the volume index of the traversal.
A while statement is used for the actual traversal using :func:`wp.bvh_query_next() <bvh_query_next>`,
which returns ``True`` as long as there are intersecting bounds.

Example: BVH Volume Traversal
#############################

Similar to the ray-traversal example, we can perform volume traversal to find the volumes that are fully contained
within a specified bounding box.

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
            # The volume with index bounds_nr is fully contained
            # in the (lower,upper) bounding box
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
:func:`wp.bvh_query_aabb() <bvh_query_aabb>`.

Meshes
------

Warp provides a ``wp.Mesh`` class to manage triangle mesh data. To create a mesh users provide a points, indices and optionally a velocity array::

    mesh = wp.Mesh(points, indices, velocities)

.. note::
    Mesh objects maintain references to their input geometry buffers. All buffers should live on the same device.

Meshes can be passed to kernels using their ``id`` attribute which uniquely identifies the mesh by a unique ``uint64`` value.
Once inside a kernel you can perform geometric queries against the mesh such as ray-casts or closest point lookups::

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
After modifying point locations users should call ``Mesh.refit()`` to rebuild the bounding volume hierarchy (BVH) structure and ensure that queries work correctly.

.. note::
    Updating Mesh topology (indices) at runtime is not currently supported. Users should instead recreate a new Mesh object.

.. autoclass:: Mesh
    :members:

Hash Grids
----------

Many particle-based simulation methods such as the Discrete Element Method (DEM), or Smoothed Particle Hydrodynamics (SPH), involve iterating over spatial neighbors to compute force interactions. Hash grids are a well-established data structure to accelerate these nearest neighbor queries, and particularly well-suited to the GPU.

To support spatial neighbor queries Warp provides a ``HashGrid`` object that may be created as follows::

    grid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=128, device="cuda")

    grid.build(points=p, radius=r)

``p`` is an array of ``wp.vec3`` point positions, and ``r`` is the radius to use when building the grid.
Neighbors can then be iterated over inside the kernel code using :func:`wp.hash_grid_query() <hash_grid_query>`
and :func:`wp.hash_grid_query_next() <hash_grid_query_next>` as follows:

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


.. autoclass:: HashGrid
    :members:

Volumes
-------

Sparse volumes are incredibly useful for representing grid data over large domains, such as signed distance fields
(SDFs) for complex objects, or velocities for large-scale fluid flow. Warp supports reading sparse volumetric grids
stored using the `NanoVDB <https://developer.nvidia.com/nanovdb>`_ standard. Users can access voxels directly
or use built-in closest-point or trilinear interpolation to sample grid data from world or local space.

Volume objects can be created directly from Warp arrays containing a NanoVDB grid, from the contents of a
standard ``.nvdb`` file using :func:`load_from_nvdb() <warp.Volume.load_from_nvdb>`,
or from a dense 3D NumPy array using :func:`load_from_numpy() <warp.Volume.load_from_numpy>`.

Volumes can also be created using :func:`allocate() <warp.Volume.allocate>` or
:func:`allocate_by_tiles() <warp.Volume.allocate_by_tiles>`. The values for a Volume object can be modified in a Warp
kernel using :func:`wp.volume_store_f() <warp.volume_store_f>`, :func:`wp.volume_store_v() <warp.volume_store_v>`, and
:func:`wp.volume_store_i() <warp.volume_store_i>`.

.. note::
    Warp does not currently support modifying the topology of sparse volumes at runtime.

Below we give an example of creating a Volume object from an existing NanoVDB file::

    # open NanoVDB file on disk
    file = open("mygrid.nvdb", "rb")

    # create Volume object
    volume = wp.Volume.load_from_nvdb(file, device="cpu")

.. note::
    Files written by the NanoVDB library, commonly marked by the ``.nvdb`` extension, can contain multiple grids with
    various compression methods, but a :class:`Volume` object represents a single NanoVDB grid therefore only files with
    a single grid are supported. NanoVDB's uncompressed and zip-compressed file formats are supported.

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
        f = wp.volume_sample_f(volume, q, wp.Volume.LINEAR)

        # write result
        samples[tid] = f

.. autoclass:: Volume
    :members:
    :undoc-members:

.. seealso:: `Reference <functions.html#volumes>`__ for the volume functions available in kernels.

Differentiability
-----------------

By default, Warp generates a forward and backward (adjoint) version of each kernel definition.
Buffers that participate in the chain of computation should be created with ``requires_grad=True``, for example::

    a = wp.zeros(1024, dtype=wp.vec3, device="cuda", requires_grad=True)

The ``wp.Tape`` class can then be used to record kernel launches, and replay them to compute the gradient of a scalar loss function with respect to the kernel inputs::

    tape = wp.Tape()

    # forward pass
    with tape:
        wp.launch(kernel=compute1, inputs=[a, b], device="cuda")
        wp.launch(kernel=compute2, inputs=[c, d], device="cuda")
        wp.launch(kernel=loss, inputs=[d, l], device="cuda")

    # reverse pass
    tape.backward(l)

After the backward pass has completed, the gradients with respect to the inputs are available via a mapping in the :class:`wp.Tape <Tape>` object::

    # gradient of loss with respect to input a
    print(tape.gradients[a])

Note that gradients are accumulated on the participating buffers, so if you wish to reuse the same buffers for multiple backward passes you should first zero the gradients::

    tape.zero()

.. note::

    Warp uses a source-code transformation approach to auto-differentiation.
    In this approach, the backwards pass must keep a record of intermediate values computed during the forward pass.
    This imposes some restrictions on what kernels can do and still be differentiable:

    * Dynamic loops should not mutate any previously declared local variable. This means the loop must be side-effect free.
      A simple way to ensure this is to move the loop body into a separate function.
      Static loops that are unrolled at compile time do not have this restriction and can perform any computation.

    * Kernels should not overwrite any previously used array values except to perform simple linear add/subtract operations (e.g. via :func:`wp.atomic_add() <atomic_add>`)

.. autoclass:: Tape
    :members:

Jacobians
#########

To compute the Jacobian matrix :math:`J\in\mathbb{R}^{m\times n}` of a multi-valued function :math:`f: \mathbb{R}^n \to \mathbb{R}^m`, we can evaluate an entire row of the Jacobian in parallel by finding the Jacobian-vector product :math:`J^\top \mathbf{e}`. The vector :math:`\mathbf{e}\in\mathbb{R}^m` selects the indices in the output buffer to differentiate with respect to.
In Warp, instead of passing a scalar loss buffer to the ``tape.backward()`` method, we pass a dictionary ``grads`` mapping from the function output array to the selection vector :math:`\mathbf{e}` having the same type::

    # compute the Jacobian for a function of single output
    jacobian = np.empty((output_dim, input_dim), dtype=np.float32)
    tape = wp.Tape()
    with tape:
        output_buffer = launch_kernels_to_be_differentiated(input_buffer)
    for output_index in range(output_dim):
        # select which row of the Jacobian we want to compute
        select_index = np.zeros(output_dim)
        select_index[output_index] = 1.0
        e = wp.array(select_index, dtype=wp.float32)
        # pass input gradients to the output buffer to apply selection
        tape.backward(grads={output_buffer: e})
        q_grad_i = tape.gradients[input_buffer]
        jacobian[output_index, :] = q_grad_i.numpy()
        tape.zero()

When we run simulations independently in parallel, the Jacobian corresponding to the entire system dynamics is a block-diagonal matrix. In this case, we can compute the Jacobian in parallel for all environments by choosing a selection vector that has the output indices active for all environment copies. For example, to get the first rows of the Jacobians of all environments, :math:`\mathbf{e}=[\begin{smallmatrix}1 & 0 & 0 & \dots & 1 & 0 & 0 & \dots\end{smallmatrix}]^\top`, to compute the second rows, :math:`\mathbf{e}=[\begin{smallmatrix}0 & 1 & 0 & \dots & 0 & 1 & 0 & \dots\end{smallmatrix}]^\top`, etc.::

    # compute the Jacobian for a function over multiple environments in parallel
    jacobians = np.empty((num_envs, output_dim, input_dim), dtype=np.float32)
    tape = wp.Tape()
    with tape:
        output_buffer = launch_kernels_to_be_differentiated(input_buffer)
    for output_index in range(output_dim):
        # select which row of the Jacobian we want to compute
        select_index = np.zeros(output_dim)
        select_index[output_index] = 1.0
        # assemble selection vector for all environments (can be precomputed)
        e = wp.array(np.tile(select_index, num_envs), dtype=wp.float32)
        tape.backward(grads={output_buffer: e})
        q_grad_i = tape.gradients[input_buffer]
        jacobians[:, output_index, :] = q_grad_i.numpy().reshape(num_envs, input_dim)
        tape.zero()


Custom Gradient Functions
#########################

Warp supports custom gradient function definitions for user-defined Warp functions.
This allows users to define code that should replace the automatically generated derivatives.

To differentiate a function :math:`h(x) = f(g(x))` that has a nested call to function :math:`g(x)`, the chain rule is evaluated in the automatic differentiation of :math:`h(x)`:

.. math::

    h^\prime(x) = f^\prime({\color{green}{\underset{\textrm{replay}}{g(x)}}}) {\color{blue}{\underset{\textrm{grad}}{g^\prime(x)}}}

This implies that a function to be compatible with the autodiff engine needs to provide an implementation of its forward version
:math:`\color{green}{g(x)}`, which we refer to as "replay" function (that matches the original function definition by default),
and its derivative :math:`\color{blue}{g^\prime(x)}`, referred to as "grad".

Both the replay and the grad implementations can be customized by the user. They are defined as follows:

.. list-table:: Customizing the replay and grad versions of function ``myfunc``
    :widths: 100
    :header-rows: 0

    * - Forward Function
    * - .. code-block:: python

            @wp.func
            def myfunc(in1: InType1, ..., inN: InTypeN) -> OutType1, ..., OutTypeM:
                return out1, ..., outM

    * - Custom Replay Function
    * - .. code-block:: python

            @wp.func_replay(myfunc)
            def replay_myfunc(in1: InType1, ..., inN: InTypeN) -> OutType1, ..., OutTypeM:
                # Custom forward computations to be executed in the backward pass of a
                # function calling `myfunc` go here
                # Ensure the output variables match the original forward definition
                return out1, ..., outM

    * - Custom Grad Function
    * - .. code-block:: python

            @wp.func_grad(myfunc)
            def adj_myfunc(in1: InType1, ..., inN: InTypeN, adj_out1: OutType1, ..., adj_outM: OutTypeM):
                # Custom adjoint code goes here
                # Update the partial derivatives for the inputs as follows:
                wp.adjoint[in1] += ...
                ...
                wp.adjoint[inN] += ...

.. note:: It is currently not possible to define custom replay or grad functions for functions that
    have generic arguments, e.g. ``Any`` or ``wp.array(dtype=Any)``. Replay or grad functions that
    themselves use generic arguments are also not yet supported.

Example 1: Custom Grad Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the following, we define a Warp function ``safe_sqrt`` that computes the square root of a number::

    @wp.func
    def safe_sqrt(x: float):
        return wp.sqrt(x)

To evaluate this function, we define a kernel that applies ``safe_sqrt`` to an array of input values::

    @wp.kernel
    def run_safe_sqrt(xs: wp.array(dtype=float), output: wp.array(dtype=float)):
        i = wp.tid()
        output[i] = safe_sqrt(xs[i])

Calling the kernel for an array of values ``[1.0, 2.0, 0.0]`` yields the expected outputs, the gradients are finite except for the zero input::

    xs = wp.array([1.0, 2.0, 0.0], dtype=wp.float32, requires_grad=True)
    ys = wp.zeros_like(xs)
    tape = wp.Tape()
    with tape:
        wp.launch(run_safe_sqrt, dim=len(xs), inputs=[xs], outputs=[ys])
    tape.backward(grads={ys: wp.array(np.ones(len(xs)), dtype=wp.float32)})
    print("ys     ", ys)
    print("xs.grad", xs.grad)

    # ys      [1.   1.4142135   0. ]
    # xs.grad [0.5  0.35355338  inf]

It is often desired to catch nonfinite gradients in the computation graph as they may cause the entire gradient computation to be nonfinite.
To do so, we can define a custom gradient function that replaces the adjoint function for ``safe_sqrt`` which is automatically generated by
decorating the custom gradient code via ``@wp.func_grad(safe_sqrt)``::

    @wp.func_grad(safe_sqrt)
    def adj_safe_sqrt(x: float, adj_ret: float):
        if x > 0.0:
            wp.adjoint[x] += 1.0 / (2.0 * wp.sqrt(x)) * adj_ret

.. note:: The function signature of the custom grad code consists of the input arguments of the forward function plus the adjoint variables of the
    forward function outputs. To access and modify the partial derivatives of the input arguments, we use the ``wp.adjoint`` dictionary.
    The keys of this dictionary are the input arguments of the forward function, and the values are the partial derivatives of the forward function
    output with respect to the input argument.


Example 2: Custom Replay Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the following, we increment an array index in each thread via :func:`wp.atomic_add() <atomic_add>` and compute
the square root of an input array at the incremented index::

    @wp.kernel
    def test_add(counter: wp.array(dtype=int), input: wp.array(dtype=float), output: wp.array(dtype=float)):
        idx = wp.atomic_add(counter, 0, 1)
        output[idx] = wp.sqrt(input[idx])

    def main():
        dim = 16
        use_reversible_increment = False
        input = wp.array(np.arange(1, dim + 1), dtype=wp.float32, requires_grad=True)
        counter = wp.zeros(1, dtype=wp.int32)
        thread_ids = wp.zeros(dim, dtype=wp.int32)
        output = wp.zeros(dim, dtype=wp.float32, requires_grad=True)
        tape = wp.Tape()
        with tape:
            if use_reversible_increment:
                wp.launch(test_add_diff, dim, inputs=[counter, thread_ids, input], outputs=[output])
            else:
                wp.launch(test_add, dim, inputs=[counter, input], outputs=[output])

        print("counter:    ", counter.numpy())
        print("thread_ids: ", thread_ids.numpy())
        print("input:      ", input.numpy())
        print("output:     ", output.numpy())

        tape.backward(grads={
            output: wp.array(np.ones(dim), dtype=wp.float32)
        })
        print("input.grad: ", input.grad.numpy())

    if __name__ == "__main__":
        main()

The output of the above code is:

.. code-block:: js

    counter:     [8]
    thread_ids:  [0 0 0 0 0 0 0 0]
    input:       [1. 2. 3. 4. 5. 6. 7. 8.]
    output:      [1.  1.4142135  1.7320508  2.  2.236068  2.4494898  2.6457512  2.828427]
    input.grad:  [4. 0. 0. 0. 0. 0. 0. 0.]

The gradient of the input is incorrect because the backward pass involving the atomic operation ``wp.atomic_add()`` does not know which thread ID corresponds
to which input value.
The index returned by the adjoint of ``wp.atomic_add()`` is always zero so that the gradient the first entry of the input array,
i.e. :math:`\frac{1}{2\sqrt{1}} = 0.5`, is accumulated ``dim`` times (hence ``input.grad[0] == 4.0`` and all other entries zero).

To fix this, we define a new Warp function ``reversible_increment()`` with a custom *replay* definition that stores the thread ID in a separate array::

    @wp.func
    def reversible_increment(
        buf: wp.array(dtype=int),
        buf_index: int,
        value: int,
        thread_values: wp.array(dtype=int),
        tid: int
    ):
        next_index = wp.atomic_add(buf, buf_index, value)
        # store which thread ID corresponds to which index for the backward pass
        thread_values[tid] = next_index
        return next_index


    @wp.func_replay(reversible_increment)
    def replay_reversible_increment(
        buf: wp.array(dtype=int),
        buf_index: int,
        value: int,
        thread_values: wp.array(dtype=int),
        tid: int
    ):
        return thread_values[tid]


Instead of running ``reversible_increment()``, the custom replay code in ``replay_reversible_increment()`` is now executed
during forward phase in the backward pass of the function calling ``reversible_increment()``.
We first stored the array index to each thread ID in the forward pass, and now we retrieve the array index for each thread ID in the backward pass.
That way, the backward pass can reproduce the same addition operation as in the forward pass with exactly the same operands per thread.

.. warning:: The function signature of the custom replay code must match the forward function signature.

To use our function we write the following kernel::

    @wp.kernel
    def test_add_diff(
        counter: wp.array(dtype=int),
        thread_ids: wp.array(dtype=int),
        input: wp.array(dtype=float),
        output: wp.array(dtype=float)
    ):
        tid = wp.tid()
        idx = reversible_increment(counter, 0, 1, thread_ids, tid)
        output[idx] = wp.sqrt(input[idx])

Running the ``test_add_diff`` kernel via the previous ``main`` function with ``use_reversible_increment = True``, we now compute correct gradients
for the input array:

.. code-block:: js

    counter:     [8]
    thread_ids:  [0 1 2 3 4 5 6 7]
    input:       [1. 2. 3. 4. 5. 6. 7. 8.]
    output:      [1.   1.4142135   1.7320508   2.    2.236068   2.4494898   2.6457512   2.828427  ]
    input.grad:  [0.5  0.35355338  0.28867513  0.25  0.2236068  0.20412414  0.18898225  0.17677669]

Custom Native Functions
-----------------------

Users may insert native C++/CUDA code in Warp kernels using ``@func_native`` decorated functions.
These accept native code as strings that get compiled after code generation, and are called within ``@wp.kernel`` functions.
For example::

    snippet = """
        __shared__ int s[128];

        s[tid] = d[tid];
        __syncthreads();
        d[tid] = s[N - tid - 1];
        """

    @wp.func_native(snippet)
    def reverse(d: wp.array(dtype=int), N: int, tid: int):
        ...

    @wp.kernel
    def reverse_kernel(d: wp.array(dtype=int), N: int):
        tid = wp.tid()
        reverse(d, N, tid)

    N = 128
    x = wp.array(np.arange(N, dtype=int), dtype=int, device=device)

    wp.launch(kernel=reverse_kernel, dim=N, inputs=[x, N], device=device)

Notice the use of shared memory here: the Warp library does not expose shared memory as a feature, but the CUDA compiler will
readily accept the above snippet. This means CUDA features not exposed in Warp are still accessible in Warp scripts.
Warp kernels meant for the CPU won't be able to leverage CUDA features of course, but this same mechanism supports pure C++ snippets as well.

Please bear in mind the following: the thread index in your snippet should be computed in a ``@wp.kernel`` and passed to your snippet,
as in the above example. This means your ``@wp.func_native`` function signature should include the variables used in your snippet, 
as well as a thread index of type ``int``. The function body itself should be stubbed with ``...`` (the snippet will be inserted during compilation).

Should you wish to record your native function on the tape and then subsequently rewind the tape, you must include an adjoint snippet
alongside your snippet as an additional input to the decorator, as in the following example::

    snippet = """
    out[tid] = a * x[tid] + y[tid];
    """
    adj_snippet = """
    adj_a = x[tid] * adj_out[tid];
    adj_x[tid] = a * adj_out[tid];
    adj_y[tid] = adj_out[tid];
    """

    @wp.func_native(snippet, adj_snippet)
    def saxpy(
        a: wp.float32,
        x: wp.array(dtype=wp.float32),
        y: wp.array(dtype=wp.float32),
        out: wp.array(dtype=wp.float32),
        tid: int,
    ):
        ...

    @wp.kernel
    def saxpy_kernel(
        a: wp.float32,
        x: wp.array(dtype=wp.float32),
        y: wp.array(dtype=wp.float32),
        out: wp.array(dtype=wp.float32)
    ):
        tid = wp.tid()
        saxpy(a, x, y, out, tid)

    N = 128
    a = 2.0
    x = wp.array(np.arange(N, dtype=np.float32), dtype=wp.float32, device=device, requires_grad=True)
    y = wp.zeros_like(x1)
    out = wp.array(np.arange(N, dtype=np.float32), dtype=wp.float32, device=device)
    adj_out = wp.array(np.ones(N, dtype=np.float32), dtype=wp.float32, device=device)

    tape = wp.Tape()

    with tape:
        wp.launch(kernel=saxpy_kernel, dim=N, inputs=[a, x, y], outputs=[out], device=device)

    tape.backward(grads={out: adj_out})

Profiling
---------

``wp.ScopedTimer`` objects can be used to gain some basic insight into the performance of Warp applications:

.. code:: python

    with wp.ScopedTimer("grid build"):
        self.grid.build(self.x, self.point_radius)

This results in a printout at runtime to the standard output stream like:

.. code:: console

    grid build took 0.06 ms

The ``wp.ScopedTimer`` object does not synchronize (e.g. by calling ``wp.synchronize()``)
upon exiting the ``with`` statement, so this can lead to misleading numbers if the body
of the ``with`` statement launches device kernels.

When a ``wp.ScopedTimer`` object is passed ``use_nvtx=True`` as an argument, the timing functionality is replaced by calls
to ``nvtx.start_range()`` and ``nvtx.end_range()``:

.. code:: python

    with wp.ScopedTimer("grid build", use_nvtx=True, color="cyan"):
        self.grid.build(self.x, self.point_radius)

These range annotations can then be collected by a tool like `NVIDIA Nsight Systems <https://developer.nvidia.com/nsight-systems>`_
for visualization on a timeline, e.g.:

.. code:: console
   
    $ nsys profile python warp_application.py

This code snippet also demonstrates the use of the ``color`` argument to specify a color
for the range, which may be a number representing the ARGB value or a recognized string
(refer to `colors.py <https://github.com/NVIDIA/NVTX/blob/release-v3/python/nvtx/colors.py>`_ for
additional color examples).
The `nvtx module <https://github.com/NVIDIA/NVTX>`_ must be
installed in the Python environment for this capability to work.
An equivalent way to create an NVTX range without using ``wp.ScopedTimer`` is:

.. code:: python

    import nvtx

    with nvtx.annotate("grid build", color="cyan"):
        self.grid.build(self.x, self.point_radius)

This form may be more convenient if the user does not need to frequently switch
between timer and NVTX capabilities of ``wp.ScopedTimer``. 

.. autoclass:: warp.ScopedTimer
