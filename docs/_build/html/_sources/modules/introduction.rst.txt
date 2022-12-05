Introduction
================

Warp is a Python framework for writing high-performance simulation and graphics code. Kernels are defined in Python syntax and JIT converted to C++/CUDA and compiled at runtime.

Warp is designed to make it easy to write programs for physics simulation, geometry processing, and procedural animation. Below are some examples of simulations implemented using Warp:

.. image:: ../img/header.png

Installation
------------

Please see the Warp `README.md <https://github.com/NVIDIA/warp>`_ for building and installation instructions and where to download pre-built packages.

Example Usage
-------------

Before use Warp should be explicitly initialized as follows: ::

    import warp as wp

    wp.init()

To define a computational kernel use the following syntax with the ``@wp.kernel`` decorator. Note that all input arguments must be typed, and that the function cannot access any global state::

    @wp.kernel
    def simple_kernel(a: wp.array(dtype=wp.vec3),
                      b: wp.array(dtype=wp.vec3),
                      c: wp.array(dtype=float)):

        # get thread index
        tid = wp.tid()

        # load two vec3s
        x = a[tid]
        y = b[tid]

        # compute the dot product between vectors
        r = wp.dot(x, y)

        # write result back to memory
        c[tid] = r

Arrays can be allocated similar to PyTorch: ::

    # allocate an uninitizalized array of vec3s
    v = wp.empty(shape=n, dtype=wp.vec3, device="cuda")

    # allocate a zero-initialized array of quaternions    
    q = wp.zeros(shape=n, dtype=wp.quat, device="cuda")

    # allocate and initialize an array from a numpy array
    # will be automatically transferred to the specified device
    a = np.ones((10, 3), dtype=np.float32)
    v = wp.from_numpy(a, dtype=wp.vec3, device="cuda")


To launch a kernel use the following syntax: ::


    wp.launch(kernel=simple_kernel, # kernel to launch
              dim=1024,             # number of threads
              inputs=[a, b, c],     # parameters
              device="cuda")        # execution device


Note that all input and output buffers must exist on the same device as the one specified for execution.

Often we need to read data back to main (CPU) memory which can be done conveniently as follows: ::

    # automatically bring data from device back to host
    view = device_array.numpy()

This pattern will allocate a temporary CPU buffer, perform a copy from device->host memory, and return a numpy view onto it. To avoid allocating temporary buffers this process can be managed explicitly: ::

    # manually bring data back to host
    wp.copy(dest=host_array, src=device_array)
    wp.synchronize()

    view = host_array.numpy()

All copy operations are performed asynchronously and must be synchronized explicitly to ensure data is visible. For best performance multiple copies should be queued together: ::

    # launch multiple copy operations asynchronously
    wp.copy(dest=host_array_0, src=device_array_0)
    wp.copy(dest=host_array_1, src=device_array_1)
    wp.copy(dest=host_array_2, src=device_array_2)
    wp.synchronize()

Memory Model
------------

Memory allocations are exposed via the ``warp.array`` type. Arrays wrap an underlying memory allocation that may live in either host (CPU), or device (GPU) memory. Arrays are strongly typed and store a linear sequence of built-in structures (``vec3``, ``matrix33``, etc).

Arrays may be constructed from Python lists or numpy arrays; by default, data will be copied to new memory for the device specified. However, it is possible for arrays to alias user memory using the ``copy=False`` parameter to the array constructor.

Compilation Model
-----------------

Warp uses a Python->C++/CUDA compilation model that generates kernel code from Python function definitions. All kernels belonging to a Python module are runtime compiled into dynamic libraries and PTX, the result is then cached between application restarts for fast startup times.

Note that compilation is triggered on the first kernel launch for that module. Any kernels registered in the module with ``@wp.kernel`` will be included in the shared library.

.. image:: ../img/compiler_pipeline.png

Language Details
----------------

To support GPU computation and differentiability, there are some differences from the CPython runtime.

Built-in Types
^^^^^^^^^^^^^^

Warp supports a number of built-in math types similar to high-level shading languages, for example ``vec2, vec3, vec4, mat22, mat33, mat44, quat, array``. All built-in types have value semantics so that expressions such as ``a = b`` generate a copy of the variable b rather than a reference.

Strong Typing
^^^^^^^^^^^^^

Unlike Python, in Warp all variables must be typed. Types are inferred from source expressions and function signatures using the Python typing extensions. All kernel parameters must be annotated with the appropriate type, for example: ::

    @wp.kernel
    def simple_kernel(a: wp.array(dtype=vec3),
                    b: wp.array(dtype=vec3),
                    c: float):

Tuple initialization is not supported, instead variables should be explicitly typed: ::

    # invalid
    a = (1.0, 2.0, 3.0)        

    # valid
    a = wp.vec3(1.0, 2.0, 3.0) 

Immutable Types
^^^^^^^^^^^^^^^

Similar to Python tuples, built-in value types are immutable, and users should use construction syntax to mutate existing variables, for example: ::

    a = wp.vec3(0.0, 0.0, 0.0)

    # invalid
    a[1] = 1.0

    # valid
    a = wp.vec3(0.0, 1.0, 0.0)


Unsupported Features
^^^^^^^^^^^^^^^^^^^^

To achieve good performance on GPUs some dynamic language features are not supported:

* Array slicing notation
* Lambda functions
* Exceptions
* Class definitions
* Runtime evaluation of expressions, e.g.: eval()
* Recursion
* Dynamic allocation, lists, sets, dictionaries

