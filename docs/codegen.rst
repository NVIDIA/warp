.. _code_generation:

Code Generation
===============

Overview
--------

Warp kernels are grouped together by Python module.  Before they can run on a device, they must be translated and compiled for the device architecture.  All kernels in a module are compiled together, which is faster than compiling each one individually.  When a kernel is launched, Warp checks if the module is up-to-date and will compile it if needed.  Adding new kernels to a module at runtime modifies the module, which means that it will need to be reloaded on next launch.

.. code:: python

    @wp.kernel
    def kernel_foo():
        print("foo")

    wp.launch(kernel_foo, dim=1)

    @wp.kernel
    def kernel_bar():
        print("bar")

    wp.launch(kernel_bar, dim=1)

In the snippet above, kernel definitions are interspersed with kernel launches.  To execute ``kernel_foo``, the module is compiled during the first launch.  Defining ``kernel_bar`` modifies the module, so it needs to be recompiled during the second launch:

.. code:: text

    Module __main__ 6cd1d53 load on device 'cuda:0' took 168.19 ms  (compiled)
    foo
    Module __main__ c7c0e9a load on device 'cuda:0' took 160.35 ms  (compiled)
    bar

The compilation can take a long time for modules with numerous complex kernels, so Warp caches the compiled modules and can reuse them on the next run of the program:

.. code:: text

    Module __main__ 6cd1d53 load on device 'cuda:0' took 4.97 ms  (cached)
    foo
    Module __main__ c7c0e9a load on device 'cuda:0' took 0.40 ms  (cached)
    bar

Loading cached modules is much faster, but it's not free.  In addition, module reloading can cause problems during CUDA graph capture, so there are good reasons to try to avoid it.

The best way to avoid module reloading is to define all the kernels before launching any of them.  This way, the module will be compiled only once:

.. code:: python

    @wp.kernel
    def kernel_foo():
        print("foo")

    @wp.kernel
    def kernel_bar():
        print("bar")

    wp.launch(kernel_foo, dim=1)
    wp.launch(kernel_bar, dim=1)

.. code:: text

    Module __main__ c7c0e9a load on device 'cuda:0' took 174.57 ms  (compiled)
    foo
    bar

On subsequent runs it will be loaded from the kernel cache only once:

.. code:: text

    Module __main__ c7c0e9a load on device 'cuda:0' took 4.96 ms  (cached)
    foo
    bar

Warp tries to recognize duplicate kernels to avoid unnecessary module reloading.  For example, this program creates kernels in a loop, but they are always identical, so the module does not need to be recompiled on every launch:

.. code:: python

    for i in range(3):

        @wp.kernel
        def kernel_hello():
            print("hello")

        wp.launch(kernel_hello, dim=1)

Warp filters out the duplicate kernels, so the module is only loaded once:

.. code:: text

    Module __main__ 8194f57 load on device 'cuda:0' took 178.24 ms  (compiled)
    hello
    hello
    hello


Warp generates C++/CUDA source code for CPU/GPU and stores the .cpp/.cu source files under the module directories of the kernel cache.
The kernel cache folder path is printed during the :ref:`Warp initialization <warp-initialization>` and
can be retrieved after Warp has been initialized from the ``warp.config.kernel_cache_dir`` :ref:`configuration setting <global-settings>`.

Consider the following example:

.. code:: python

    @wp.func
    def my_func(a: float, b: float):
        c = wp.sin(b) * a
        return c

The resulting CUDA code looks similar to this:

.. code:: cpp

    // example.py:5
    static CUDA_CALLABLE wp::float32 my_func_0(
        wp::float32 var_a,
        wp::float32 var_b)
    {
        //---------
        // primal vars
        wp::float32 var_0;
        wp::float32 var_1;
        //---------
        // forward
        // def my_func(a: float, b: float):                                                       <L 6>
        // c = wp.sin(b) * a                                                                      <L 7>
        var_0 = wp::sin(var_b);
        var_1 = wp::mul(var_0, var_a);
        // return c                                                                               <L 8>
        return var_1;
    }

The generated code follows `static-single-assignment (SSA) form <https://en.wikipedia.org/wiki/Static_single-assignment_form>`__.
To ease the readability, comments referring to the original Python source code lines are inserted.
Besides the forward pass, the gradient function is also generated, and,
if a :ref:`custom replay function <custom-gradient-functions>` is provided, the replay function is generated as well.

Warp passes the generated source code to native compilers (e.g., LLVM for CPU and NVRTC for CUDA) to produce executable code that is invoked when launching kernels.

.. _static_expressions:

Static Expressions
------------------

We often encounter situations where a kernel needs to be specialized for a given input or where certain parts of the code are static by the time the code is executed.
With static expressions, we can write Python expressions to be evaluated at the time of declaring a Warp function or kernel.

``wp.static(...)`` expressions allow the user to run arbitrary Python code at the time the Warp function or kernel containing the expression is defined.
:func:`wp.static(expr) <static>` accepts a Python expression and replaces it with the result.
Note that the expression can only access variables that can be evaluated at the time the expression is declared.
This includes global variables and variables captured in a closure in which the Warp function or kernel is defined.
Additionally, Warp constants from within the kernel or function can be accessed, such as the constant iteration variable for static for-loops (i.e. when the range is known at the time of code generation).

The result from ``wp.static()`` must be a non-null value of one of the following types:

- A Warp function
- A string
- Any type that is supported by Warp inside kernels (e.g. scalars, structs, matrices, vectors, etc.), excluding Warp arrays or structs containing Warp arrays

Example: Static Math Expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    import warp as wp
    import scipy.linalg

    @wp.kernel
    def my_kernel():
        static_var = wp.static(3 + 2)
        # we can call arbitrary Python code inside wp.static()
        static_norm = wp.static(wp.float64(scipy.linalg.norm([3, 4])))
        wp.printf("static_var = %i\n", static_var)
        wp.printf("static_norm = %f\n", static_norm)

    wp.launch(my_kernel, 1)

The static expressions are evaluated at the time of when the ``@wp.kernel`` decorator is evaluated and replaced in the code by their respective constant result values. The generated code will therefore contain the results of the expressions hard-coded in the source file (shown an abbreviated version):

.. code:: cpp

    const wp::int32 var_0 = 5;
    const wp::float64 var_1 = 5.0;
    const wp::str var_2 = "static_var = %i\n";
    const wp::str var_3 = "static_norm = %f\n";
    
    // wp.printf("static_var = %i\n", static_var)                                             <L 10>
    printf(var_2, var_0);
    // wp.printf("static_norm = %f\n", static_norm)                                           <L 11>
    printf(var_3, var_1);


Example: Static Conditionals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If/else/elif conditions that are constant can be eliminated from the generated code by using ``wp.static()`` inside the branch condition to yield a constant boolean.
This can provide improved performance by avoiding branching and can be useful for generating specialized kernels:

.. code:: python

    import warp as wp

    available_colors = {"red", "green", "blue"}

    @wp.kernel
    def my_kernel():
        if wp.static("red" in available_colors):
            print("red is available")
        else:
            print("red is not available")

The global variable ``available_colors`` is known at the time of declaring the kernel and the generated code will contain only the branch that is taken:

.. code:: cpp

    const wp::str var_1 = "red is available";
    wp::print(var_1);

Example: Static Loop Unrolling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Static expressions can be used to unroll for-loops during code generation. We place ``wp.static()`` expressions inside the loop's ``range`` to yield static for-loops that can be unrolled. The iteration variable becomes a constant and can therefore be accessed from within a static expression in the loop body:

.. code:: python

    import warp as wp

    def loop_limit():
        return 3

    @wp.kernel
    def my_kernel():
        for i in range(wp.static(loop_limit())):
            static_i = wp.static(i)
            wp.printf("i = %i\n", static_i)

    wp.launch(my_kernel, 1)

The generated code will not contain the for-loop but instead the loop body will be repeated three times:

.. code:: cpp

    const wp::int32 var_0 = 3;
    const wp::int32 var_1 = 0;
    const wp::int32 var_2 = 0;
    const wp::str var_3 = "i = %i\n";
    const wp::int32 var_4 = 1;
    const wp::int32 var_5 = 1;
    const wp::str var_6 = "i = %i\n";
    const wp::int32 var_7 = 2;
    const wp::int32 var_8 = 2;
    const wp::str var_9 = "i = %i\n";
    printf(var_3, var_2);
    printf(var_6, var_5);
    printf(var_9, var_8);

Example: Function Pointers
~~~~~~~~~~~~~~~~~~~~~~~~~~

``wp.static(...)`` may also return a Warp function. This can be useful to specialize a kernel or function based on information available at the time of declaring the Warp function or kernel, or to automatically generate overloads for different types.

.. code:: python

    import warp as wp

    @wp.func
    def do_add(a: float, b: float):
        return a + b

    @wp.func
    def do_sub(a: float, b: float):
        return a - b

    @wp.func
    def do_mul(a: float, b: float):
        return a * b

    op_handlers = {
        "add": do_add,
        "sub": do_sub,
        "mul": do_mul,
    }

    inputs = wp.array([[1, 2], [3, 0]], dtype=wp.float32)
    outputs = wp.empty(2, dtype=wp.float32)

    for op in op_handlers.keys():

        @wp.kernel
        def operate(input: wp.array(dtype=inputs.dtype, ndim=2), output: wp.array(dtype=wp.float32)):
            tid = wp.tid()
            a, b = input[tid, 0], input[tid, 1]
            # retrieve the right function to use for the captured dtype variable
            output[tid] = wp.static(op_handlers[op])(a, b)

        wp.launch(operate, dim=2, inputs=[inputs], outputs=[outputs])
        print(outputs.numpy())

The above program uses a static expression to select the right function given the captured ``op`` variable and prints the following output while compiling the module containing the ``operate`` kernel three times:

.. code:: text

    [3. 3.]
    [-1.  3.]
    [2. 0.]


.. _dynamic_generation:

Dynamic Kernel Creation
-----------------------

It is often desirable to dynamically customize kernels with different constants, types, or functions.  We can achieve this through runtime kernel specialization using Python closures.

Kernel Closures
~~~~~~~~~~~~~~~

Constants
^^^^^^^^^

Warp allows references to external constants in kernels:

.. code:: python

    def create_kernel_with_constant(constant):
        @wp.kernel
        def k(a: wp.array(dtype=float)):
            tid = wp.tid()
            a[tid] += constant
        return k

    k1 = create_kernel_with_constant(17.0)
    k2 = create_kernel_with_constant(42.0)

    a = wp.zeros(5, dtype=float)

    wp.launch(k1, dim=a.shape, inputs=[a])
    wp.launch(k2, dim=a.shape, inputs=[a])

    print(a)

Output:

.. code:: text

    [59. 59. 59. 59. 59.]


Data Types
^^^^^^^^^^

Warp data types can also be captured in a closure.  Here is an example of creating kernels that work with different vector dimensions:

.. code:: python

    def create_kernel_with_dtype(vec_type):
        @wp.kernel
        def k(a: wp.array(dtype=vec_type)):
            tid = wp.tid()
            a[tid] += float(tid) * vec_type(1.0)
        return k

    k2 = create_kernel_with_dtype(wp.vec2)
    k4 = create_kernel_with_dtype(wp.vec4)

    a2 = wp.ones(3, dtype=wp.vec2)
    a4 = wp.ones(3, dtype=wp.vec4)

    wp.launch(k2, dim=a2.shape, inputs=[a2])
    wp.launch(k4, dim=a4.shape, inputs=[a4])

    print(a2)
    print(a4)

Output:

.. code:: text

    [[1. 1.]
     [2. 2.]
     [3. 3.]]
    [[1. 1. 1. 1.]
     [2. 2. 2. 2.]
     [3. 3. 3. 3.]]


Functions
^^^^^^^^^

Here's a kernel generator that's parameterized using different functions:

.. code:: python

    def create_kernel_with_function(f):
        @wp.kernel
        def k(a: wp.array(dtype=float)):
            tid = wp.tid()
            a[tid] = f(a[tid])
        return k

    @wp.func
    def square(x: float):
        return x * x

    @wp.func
    def cube(x: float):
        return x * x * x

    k1 = create_kernel_with_function(square)
    k2 = create_kernel_with_function(cube)

    a1 = wp.array([1, 2, 3, 4, 5], dtype=float)
    a2 = wp.array([1, 2, 3, 4, 5], dtype=float)

    wp.launch(k1, dim=a1.shape, inputs=[a1])
    wp.launch(k2, dim=a2.shape, inputs=[a2])

    print(a1)
    print(a2)

Output:

.. code:: text

    [ 1.  4.  9.  16.  25.]
    [ 1.  8.  27.  64.  125.]


Function Closures
~~~~~~~~~~~~~~~~~

Warp functions (``@wp.func``) also support closures, just like kernels:

.. code:: python

    def create_function_with_constant(constant):
        @wp.func
        def f(x: float):
            return constant * x
        return f

    f1 = create_function_with_constant(2.0)
    f2 = create_function_with_constant(3.0)

    @wp.kernel
    def k(a: wp.array(dtype=float)):
        tid = wp.tid()
        x = float(tid)
        a[tid] = f1(x) + f2(x)

    a = wp.ones(5, dtype=float)

    wp.launch(k, dim=a.shape, inputs=[a])

    print(a)

Output:

.. code:: text

    [ 0.  5. 10. 15. 20.]


We can also create related function and kernel closures together like this:

.. code:: python

    def create_fk(a, b):
        @wp.func
        def f(x: float):
            return a * x

        @wp.kernel    
        def k(a: wp.array(dtype=float)):
            tid = wp.tid()
            a[tid] = f(a[tid]) + b

        return f, k

    # create related function and kernel closures
    f1, k1 = create_fk(2.0, 3.0)
    f2, k2 = create_fk(4.0, 5.0)

    # use the functions separately in a new kernel
    @wp.kernel
    def kk(a: wp.array(dtype=float)):
        tid = wp.tid()
        a[tid] = f1(a[tid]) + f2(a[tid])

    a1 = wp.array([1, 2, 3, 4, 5], dtype=float)
    a2 = wp.array([1, 2, 3, 4, 5], dtype=float)
    ak = wp.array([1, 2, 3, 4, 5], dtype=float)

    wp.launch(k1, dim=a1.shape, inputs=[a1])
    wp.launch(k2, dim=a2.shape, inputs=[a2])
    wp.launch(kk, dim=ak.shape, inputs=[ak])

    print(a1)
    print(a2)
    print(ak)

Output:

.. code:: text

    [ 5.  7.  9. 11. 13.]
    [ 9. 13. 17. 21. 25.]
    [ 6. 12. 18. 24. 30.]


Dynamic Structs
~~~~~~~~~~~~~~~

Sometimes it's useful to customize Warp structs with different data types.

Customize Precision
^^^^^^^^^^^^^^^^^^^

For example, we can create structs with different floating point precision:

.. code:: python

    def create_struct_with_precision(dtype):
        @wp.struct
        class S:
            a: dtype
            b: dtype
        return S

    # create structs with different floating point precision
    S16 = create_struct_with_precision(wp.float16)
    S32 = create_struct_with_precision(wp.float32)
    S64 = create_struct_with_precision(wp.float64)

    s16 = S16()
    s32 = S32()
    s64 = S64()

    s16.a, s16.b = 2.0001, 3.0000002
    s32.a, s32.b = 2.0001, 3.0000002
    s64.a, s64.b = 2.0001, 3.0000002

    # create a generic kernel that works with the different types
    @wp.kernel
    def k(s: Any, output: wp.array(dtype=Any)):
        tid = wp.tid()
        x = output.dtype(tid)
        output[tid] = x * s.a + s.b

    a16 = wp.empty(5, dtype=wp.float16)
    a32 = wp.empty(5, dtype=wp.float32)
    a64 = wp.empty(5, dtype=wp.float64)

    wp.launch(k, dim=a16.shape, inputs=[s16, a16])
    wp.launch(k, dim=a32.shape, inputs=[s32, a32])
    wp.launch(k, dim=a64.shape, inputs=[s64, a64])

    print(a16)
    print(a32)
    print(a64)

We can see the effect of using different floating point precision in the output:

.. code:: text

    [ 3.  5.  7.  9. 11.]
    [ 3.0000002  5.0001     7.0002003  9.000299  11.0004   ]
    [ 3.0000002  5.0001002  7.0002002  9.0003002 11.0004002]


Customize Dimensions
^^^^^^^^^^^^^^^^^^^^

Another useful application of dynamic structs is the ability to customize dimensionality.  Here, we create structs that work with 2D and 3D data:

.. code:: python

    # create struct with different vectors and matrix dimensions
    def create_struct_nd(dim):
        @wp.struct
        class S:
            v: wp.types.vector(dim, float)
            m: wp.types.matrix((dim, dim), float)
        return S

    S2 = create_struct_nd(2)
    S3 = create_struct_nd(3)

    s2 = S2()
    s2.v = (1.0, 2.0)
    s2.m = ((2.0, 0.0),
            (0.0, 0.5))

    s3 = S3()
    s3.v = (1.0, 2.0, 3.0)
    s3.m = ((2.0, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.0, 1.0))

    # create a generic kernel that works with the different types
    @wp.kernel
    def k(s: Any, output: wp.array(dtype=Any)):
        tid = wp.tid()
        x = float(tid)
        output[tid] = x * s.v * s.m

    a2 = wp.empty(5, dtype=wp.vec2)
    a3 = wp.empty(5, dtype=wp.vec3)

    wp.launch(k, dim=a2.shape, inputs=[s2, a2])
    wp.launch(k, dim=a3.shape, inputs=[s3, a3])

    print(a2)
    print(a3)

Output:

.. code:: text

    [[0. 0.]
     [2. 1.]
     [4. 2.]
     [6. 3.]
     [8. 4.]]
    [[ 0.  0.  0.]
     [ 2.  1.  3.]
     [ 4.  2.  6.]
     [ 6.  3.  9.]
     [ 8.  4. 12.]]


Module Reloading
~~~~~~~~~~~~~~~~

Frequent recompilation can add overhead to a program, especially if the program is creating kernels at runtime.  Consider this program:

.. code:: python

    def create_kernel_with_constant(constant):
        @wp.kernel
        def k(a: wp.array(dtype=float)):
            tid = wp.tid()
            a[tid] += constant
        return k

    a = wp.zeros(5, dtype=float)

    k1 = create_kernel_with_constant(17.0)
    wp.launch(k1, dim=a.shape, inputs=[a])
    print(a)

    k2 = create_kernel_with_constant(42.0)
    wp.launch(k2, dim=a.shape, inputs=[a])
    print(a)

    k3 = create_kernel_with_constant(-9.0)
    wp.launch(k3, dim=a.shape, inputs=[a])
    print(a)

Kernel creation is interspersed with kernel launches, which forces reloading on each kernel launch:

.. code:: text

    Module __main__ 96db544 load on device 'cuda:0' took 165.46 ms  (compiled)
    [17. 17. 17. 17. 17.]
    Module __main__ 9f609a4 load on device 'cuda:0' took 151.69 ms  (compiled)
    [59. 59. 59. 59. 59.]
    Module __main__ e93fbb9 load on device 'cuda:0' took 167.84 ms  (compiled)
    [50. 50. 50. 50. 50.]

To avoid reloading, all kernels should be created before launching them:

.. code:: python

    def create_kernel_with_constant(constant):
        @wp.kernel
        def k(a: wp.array(dtype=float)):
            tid = wp.tid()
            a[tid] += constant
        return k

    k1 = create_kernel_with_constant(17.0)
    k2 = create_kernel_with_constant(42.0)
    k3 = create_kernel_with_constant(-9.0)

    a = wp.zeros(5, dtype=float)

    wp.launch(k1, dim=a.shape, inputs=[a])
    print(a)

    wp.launch(k2, dim=a.shape, inputs=[a])
    print(a)

    wp.launch(k3, dim=a.shape, inputs=[a])
    print(a)

.. code:: text

    Module __main__ e93fbb9 load on device 'cuda:0' took 164.87 ms  (compiled)
    [17. 17. 17. 17. 17.]
    [59. 59. 59. 59. 59.]
    [50. 50. 50. 50. 50.]

Redefining identical kernels, functions, and structs should not cause module reloading, since Warp is able to detect duplicates:

.. code:: python

    def create_struct(dtype):
        @wp.struct
        class S:
            a: dtype
            b: dtype
        return S

    def create_function(dtype, S):
        @wp.func
        def f(s: S):
            return s.a * s.b
        return f

    def create_kernel(dtype, S, f, C):
        @wp.kernel
        def k(a: wp.array(dtype=dtype)):
            tid = wp.tid()
            s = S(a[tid], C)
            a[tid] = f(s)
        return k

    # create identical struct, function, and kernel in a loop
    for i in range(3):
        S = create_struct(float)
        f = create_function(float, S)
        k = create_kernel(float, S, f, 3.0)

        a = wp.array([1, 2, 3, 4, 5], dtype=float)

        wp.launch(k, dim=a.shape, inputs=[a])
        print(a)

Even though struct ``S``, function ``f``, and kernel ``k`` are re-created in each iteration of the loop, they are duplicates so the module is only loaded once:

.. code:: text

    Module __main__ 4af2d60 load on device 'cuda:0' took 181.34 ms  (compiled)
    [ 3.  6.  9. 12. 15.]
    [ 3.  6.  9. 12. 15.]
    [ 3.  6.  9. 12. 15.]


.. _late_binding:

Late Binding and Static Expressions
-----------------------------------

Python uses late binding, which means that variables can be referenced in a function before they are defined:

.. code:: python

    def k():
        # Function f() and constant C are not defined yet.
        # They will be resolved when k() is called.
        print(f() + C)

    def f():
        return 42

    C = 17

    # late binding occurs in this call
    k()

Warp follows this convention by default, because it's the Pythonic way.  Here is a similar program written in Warp:

.. code:: python

    @wp.kernel
    def k():
        # Function f() and constant C are not defined yet.
        # They will be resolved when k() is called.
        print(f() + C)

    @wp.func
    def f():
        return 42

    C = 17

    # late binding occurs in this launch, when the module is compiled
    wp.launch(k, dim=1)

    # wait for the output
    wp.synchronize_device()

Late binding is often convenient, but it can sometimes lead to surprising results.  Consider this snippet, which creates kernels in a loop.  The kernels reference the loop variable as a constant.

.. code:: python

    # create a list of kernels that use the loop variable
    kernels = []
    for i in range(3):
        @wp.kernel
        def k():
            print(i)
        kernels.append(k)

    # launch the kernels
    for k in kernels:
        wp.launch(k, dim=1)

    wp.synchronize_device()

This prints:

.. code:: text

    2
    2
    2

This might be surprising, but creating a similar program in pure Python would lead to the same results.  Because of late binding, the captured loop variable ``i`` is not evaluated until the kernels are launched.  At that moment, the value of ``i`` is 2 and we see the same output from each kernel.

In Warp, ``wp.static()`` can be used to get around this problem:

.. code:: python

    # create a list of kernels that use the loop variable
    kernels = []
    for i in range(3):
        @wp.kernel
        def k():
            print(wp.static(i))  # wp.static() for the win
        kernels.append(k)

    # launch the kernels
    for k in kernels:
        wp.launch(k, dim=1)

    wp.synchronize_device()

Warp replaces the call to ``wp.static()`` with the value of the expression passed as its argument.  The expression is evaluated immediately at the time of kernel definition.  This is similar to static binding used by languages like C++, which means that all variables referenced by the static expression must already be defined.

To further illustrate the difference between the default late binding behavior and static expressions, consider this program:

.. code:: python

    C = 17

    @wp.kernel
    def k1():
        print(C)

    @wp.kernel
    def k2():
        print(wp.static(C))

    # redefine constant
    C = 42

    wp.launch(k1, dim=1)
    wp.launch(k2, dim=1)

    wp.synchronize_device()

Output:

.. code:: text

    42
    17

Kernel ``k1`` uses late binding of ``C``.  This means that it captures the latest value of ``C``, determined when the module is built during the launch.  Kernel ``k2`` consumes ``C`` in a static expression, thus it captures the value of ``C`` when the kernel is defined.

The same rules apply to resolving Warp functions:

.. code:: python

    @wp.func
    def f():
        return 17

    @wp.kernel
    def k1():
        print(f())

    @wp.kernel
    def k2():
        print(wp.static(f)())

    # redefine function
    @wp.func
    def f():
        return 42

    wp.launch(k1, dim=1)
    wp.launch(k2, dim=1)

    wp.synchronize_device()

Output:

.. code:: text

    42
    17

Kernel ``k1`` uses the latest definition of function ``f``, while kernel ``k2`` uses the definition of ``f`` when the kernel was declared.
