Runtime Reference
=================

.. currentmodule:: warp

This section describes the Warp Python runtime API, how to manage memory, launch kernels, and high-level functionality
for dealing with objects such as meshes and volumes. The APIs described in this section are intended to be used at
the *Python Scope* and run inside the CPython interpreter. For a comprehensive list of functions available at
the *Kernel Scope*, please see the :doc:`functions` section.

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

Kernels are launched with the :func:`wp.launch() <launch>` function on a specific device (CPU/GPU)::

    wp.launch(add_kernel, dim=1024, inputs=[a, b, c], device="cuda")

Note that all the kernel inputs must live on the target device or a runtime exception will be raised.
Kernels may be launched with multi-dimensional grid bounds. In this case, threads are not assigned a single index,
but a coordinate in an n-dimensional grid, e.g.::

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
:func:`wp.clear_kernel_cache() <clear_kernel_cache>` can be used to clear the kernel cache of previously
generated compilation artifacts as Warp does not automatically try to keep the cache below a certain size.

.. autofunction:: kernel

.. autofunction:: launch
.. autofunction:: launch_tiled
    
.. autofunction:: clear_kernel_cache

.. _Runtime Kernel Creation:

Runtime Kernel Creation
#######################

Warp allows generating kernels on-the-fly with various customizations, including closure support.
Refer to the :ref:`Code Generation<code_generation>` section for the latest features.

Launch Objects
##############

:class:`Launch` objects are one way to reduce the overhead of launching a kernel multiple times.
:class:`Launch` objects are returned from calling :func:`wp.launch() <launch>` with ``record_cmd=True``.
This stores the results of various overhead operations that are needed to launch a kernel
but defers the actual kernel launch until the :func:`Launch.launch() <Launch.launch>` method is called.

In contrast to :ref:`graphs`, :class:`Launch` objects only record the launch of a single kernel
and do not reduce the driver overhead of preparing the kernel for execution on a GPU.
On the other hand, :class:`Launch` objects do not have the storage and initialization
overheads of CUDA graphs and also allow for the modification of launch
dimensions with :func:`Launch.set_dim() <Launch.set_dim>` and
kernel parameters with functions such as :func:`Launch.set_params() <Launch.set_params>` and
:func:`Launch.set_param_by_name() <Launch.set_param_by_name>`.
Additionally, :class:`Launch` objects can also be used to reduce the overhead of launching kernels running on the CPU.

.. note::
    Kernels launched via :class:`Launch` objects currently do not get recorded onto the :class:`Tape`.

.. autoclass:: Launch
    :members:
    :undoc-members:
    :exclude-members: __init__

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

Arrays can be moved between devices using :meth:`array.to`::

    host_array = wp.array(a, dtype=float, device="cpu")

    # allocate and copy to GPU
    device_array = host_array.to("cuda")

Additionally, data can be copied between arrays in different memory spaces using :func:`wp.copy() <warp.copy()>`::

    src_array = wp.array(a, dtype=float, device="cpu")
    dest_array = wp.empty_like(host_array)

    # copy from source CPU buffer to GPU
    wp.copy(dest_array, src_array)

.. autoclass:: array
    :members:
    :undoc-members:
    :exclude-members: vars

Multi-dimensional Arrays
########################

Multi-dimensional arrays up to four dimensions can be constructed by passing a tuple of sizes for each dimension.

The following constructs a 2D array of size 1024 x 16::

    wp.zeros(shape=(1024, 16), dtype=float, device="cuda")

When passing multi-dimensional arrays to kernels users must specify the expected array dimension inside the kernel signature,
e.g. to pass a 2D array to a kernel the number of dims is specified using the ``ndim=2`` parameter::

    @wp.kernel
    def test(input: wp.array(dtype=float, ndim=2)):

Type-hint helpers are provided for common array sizes, e.g.: ``array2d()``, ``array3d()``, which are equivalent to calling ``array(..., ndim=2)```, etc.
To index a multi-dimensional array, use the following kernel syntax::

    # returns a float from the 2d array
    value = input[i,j]

To create an array slice, use the following syntax, where the number of indices is less than the array dimensions::

    # returns an 1d array slice representing a row of the 2d array
    row = input[i]

Slice operators can be concatenated, e.g.: ``s = array[i][j][k]``. Slices can be passed to ``wp.func`` user functions provided
the function also declares the expected array dimension. Currently, only single-index slicing is supported.

The following construction methods are provided for allocating zero-initialized and empty (non-initialized) arrays:

.. autofunction:: zeros
.. autofunction:: zeros_like
.. autofunction:: ones
.. autofunction:: ones_like
.. autofunction:: full
.. autofunction:: full_like
.. autofunction:: empty
.. autofunction:: empty_like
.. autofunction:: copy
.. autofunction:: clone

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

Warp supports ``float`` and ``int`` as aliases for ``wp.float32`` and ``wp.int32`` respectively.

.. _vec:

Vectors
#######

Warp provides built-in math and geometry types for common simulation and graphics problems.
A full reference for operators and functions for these types is available in the :doc:`/modules/functions`.

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

Type Conversions
################

Warp is particularly strict regarding type conversions and does not perform *any* implicit conversion between numeric types.
The user is responsible for ensuring types for most arithmetic operators match, e.g.: ``x = float(0.0) + int(4)`` will result in an error.
This can be surprising for users that are accustomed to C-style conversions but avoids a class of common bugs that result from implicit conversions.

Users should explicitly cast variables to compatible types using constructors like
``int()``, ``float()``, ``wp.float16()``, ``wp.uint8()``, etc.

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

Note that using ``wp.constant()`` is no longer required, but it performs some type checking and can serve as a reminder that the variables are meant to be used as Warp constants.

The behavior is simple and intuitive when the referenced Python variables never change. For details and more complex scenarios, refer to :ref:`External References and Constants<external_references>`. The :ref:`Code Generation<code_generation>` section contains additional information and tips for advanced usage.

Predefined Constants
####################

For convenience, Warp has a number of predefined mathematical constants that
may be used both inside and outside Warp kernels.
The constants in the following table also have lowercase versions defined,
e.g. ``wp.E`` and ``wp.e`` are equivalent.

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

The ``wp.NAN`` constant may only be used with floating-point types.
Comparisons involving ``wp.NAN`` follow the IEEE 754 standard,
e.g. ``wp.float32(wp.NAN) == wp.float32(wp.NAN)`` returns ``False``.
The :func:`wp.isnan() <isnan>` built-in function can be used to determine whether a
value is a NaN (or if a vector, matrix, or quaternion contains a NaN entry).

The following example shows how positive and negative infinity
can be used with floating-point types in Warp using the ``wp.inf`` constant:

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
Users should use type constructors such as ``float()``, ``int()``, ``wp.int64()``, etc. to cast variables
to the correct type.

The multiplication expression ``a * b`` can also be used to perform matrix multiplication
between `matrix types <Matrices>`_.

Streams
-------

A CUDA stream is a sequence of operations that execute in order on the GPU.
Operations from different streams may run concurrently and may be interleaved by the device scheduler.
See the :ref:`Streams documentation <streams>` for more information on using streams.

.. autoclass:: Stream
    :members:
    :exclude-members: cached_event

.. autofunction:: get_stream
.. autofunction:: set_stream
.. autofunction:: wait_stream
.. autofunction:: synchronize_stream

.. autoclass:: ScopedStream

Events
------

Events can be inserted into streams and used to synchronize a stream
with a different one. See the :ref:`Events documentation <cuda_events>` for
information on how to use events for cross-stream synchronization
or the :ref:`CUDA Events Timing documentation <cuda_events_profiling>` for
information on how to use events for measuring GPU performance.

.. autoclass:: Event
    :members:
    :exclude-members: Flags

.. autofunction:: record_event
.. autofunction:: wait_event
.. autofunction:: synchronize_event
.. autofunction:: get_event_elapsed_time

.. _graphs:

Graphs
-----------

Launching kernels from Python introduces significant additional overhead compared to C++ or native programs.
To address this, Warp exposes the concept of `CUDA graphs <https://developer.nvidia.com/blog/cuda-graphs/>`_
to allow recording large batches of kernels and replaying them with very little CPU overhead.

To record a series of kernel launches use the :func:`wp.capture_begin() <capture_begin>` and
:func:`wp.capture_end() <capture_end>` API as follows:

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
statement will ensure :func:`wp.capture_end <capture_end>` gets called, even if an exception occurs during
capture, which would otherwise trap the stream in a capturing state.

Once a graph has been constructed it can be executed: ::

    wp.capture_launch(graph)

The :class:`wp.ScopedCapture <ScopedCapture>` context manager can be used to simplify the code and
ensure that :func:`wp.capture_end <capture_end>` is called regardless of exceptions:

.. code:: python

    with wp.ScopedCapture(device="cuda") as capture:
        # record launches
        for i in range(100):
            wp.launch(kernel=compute1, inputs=[a, b], device="cuda")

    wp.capture_launch(capture.graph)

Note that only launch calls are recorded in the graph; any Python executed outside of the kernel code will not be recorded.
Typically it is only beneficial to use CUDA graphs when the graph will be reused or launched multiple times, as
there is a graph-creation overhead.

.. autofunction:: capture_begin
.. autofunction:: capture_end
.. autofunction:: capture_launch

.. autoclass:: ScopedCapture
    :members:

Meshes
------

Warp provides a :class:`wp.Mesh <Mesh>` class to manage triangle mesh data. To create a mesh, users provide a points, indices and optionally a velocity array::

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

.. autoclass:: Mesh
    :members:
    :exclude-members: vars, Var

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
from an uncompressed in-memory buffer using :func:`load_from_address() <warp.Volume.load_from_address>`,
or from a dense 3D NumPy array using :func:`load_from_numpy() <warp.Volume.load_from_numpy>`.

Volumes can also be created using :func:`allocate() <warp.Volume.allocate>`, 
:func:`allocate_by_tiles() <warp.Volume.allocate_by_tiles>` or :func:`allocate_by_voxels() <warp.Volume.allocate_by_voxels>`. 
The values for a Volume object can be modified in a Warp kernel using :func:`wp.volume_store() <warp.volume_store>`.

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

.. autoclass:: Volume
    :members:
    :undoc-members:

.. seealso:: `Reference <functions.html#volumes>`__ for the volume functions available in kernels.


Bounding Value Hierarchies (BVH)
--------------------------------

The :class:`wp.Bvh <Bvh>` class can be used to create a BVH for a group of bounding volumes. This object can then be traversed
to determine which parts are intersected by a ray using :func:`bvh_query_ray` and which parts overlap
with a certain bounding volume using :func:`bvh_query_aabb`.

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

Marching Cubes
--------------

The :class:`wp.MarchingCubes <MarchingCubes>` class can be used to extract a 2-D mesh approximating an
isosurface of a 3-D scalar field. The resulting triangle mesh can be saved to a USD
file using the :class:`warp.renderer.UsdRenderer`.

See :github:`warp/examples/core/example_marching_cubes.py` for a usage example.

.. autoclass:: MarchingCubes
    :members:

Profiling
---------

``wp.ScopedTimer`` objects can be used to gain some basic insight into the performance of Warp applications:

.. code:: python

    with wp.ScopedTimer("grid build"):
        self.grid.build(self.x, self.point_radius)

This results in a printout at runtime to the standard output stream like:

.. code:: console

    grid build took 0.06 ms

See :doc:`../profiling` documentation for more information.

.. autoclass:: warp.ScopedTimer
    :noindex:

Interprocess Communication (IPC)
--------------------------------

Interprocess communication can be used to share Warp arrays and events across
processes without creating copies of the underlying data.

Some basic requirements for using IPC include:

* Linux operating system (note however that integrated devices like NVIDIA
  Jetson do not support CUDA IPC)
* The array must be allocated on a GPU device using the default memory allocator (see :doc:`allocators`)

  The ``wp.ScopedMempool`` context manager is useful for temporarily disabling
  memory pools for the purpose of allocating arrays that can be shared using IPC.

Support for IPC on a device is indicated by the :attr:`is_ipc_supported <warp.context.Device.is_ipc_supported>`
attribute of the :class:`Device <warp.context.Device>`. If the Warp library has
been compiled with CUDA 11, this device attribute will be ``None`` to indicate
that IPC support could not be determined using the CUDA API.

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



.. autofunction:: from_ipc_handle

.. autofunction:: event_from_ipc_handle

LTO Cache
---------

:ref:`MathDx <mathdx>` generates Link-Time Optimization (LTO) files for GEMM, Cholesky, and FFT tile operations.
Warp caches these to speed up kernel compilation. Each LTO file maps to a specific Linear Algebra
solver configuration, and is otherwise independent of the kernel in which its corresponding routine
is called. Therefore, LTOs are stored in a cache that is independent of a given module's kernel cache,
and will remain cached even if :func:`wp.clear_kernel_cache() <clear_kernel_cache>` is called.
:func:`wp.clear_lto_cache() <clear_lto_cache>` can be used to clear the LTO cache.

.. autofunction:: clear_lto_cache