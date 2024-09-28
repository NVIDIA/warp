Code Generation
===============

Warp explicitly generates C++/CUDA code for CPU/GPU and stores the .cpp/.cu source files under the module directories of the kernel cache.
The kernel cache folder path is printed during the `Warp initialization <basics.html#initialization>`_ and can be retrieved after Warp has been initialized from the ``warp.config.kernel_cache_dir`` `configuration <configuration.html#global-settings>`_.

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

The generated code follows `static-single-assignment (SSA) form <https://en.wikipedia.org/wiki/Static_single-assignment_form>`_. To ease the readability, comments referring to the original Python source code lines are inserted. Besides the forward pass, the gradient function is also generated, and, if a `custom replay function <differentiability.html#custom-gradient-functions>`_ is provided, the replay function is generated as well.

Static Expressions
------------------

We often encounter situations where a kernel needs to be specialized for a given input or where certain parts of the code are static by the time the code is executed. With static expressions we can write Python expressions to be evaluated at the time of declaring a Warp function or kernel.

``wp.static(...)`` expressions allow the user to run arbitrary Python code at the time of when the Warp function or kernel containing the expression is defined. :func:`wp.static(expr) <static>` accepts a Python expression and replaces it with the result. Note that the expression can only access variables that can be evaluated at the time the expression is declared. This includes global variables and variables captured in a closure in which the Warp function or kernel is defined. Additionally, Warp constants from within the kernel or function can be accessed, such as the constant iteration variable for static for-loops (i.e. where the range is known at the time of code generation).

The result from `wp.static()` must be a non-null value of one of the following types:

- a Warp function
- a string
- any type that is supported by Warp inside kernels (e.g. scalars, structs, matrices, vectors, etc.), excluding Warp arrays or structs containing Warp arrays.

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

If/else/elif conditions that are constant can be eliminated from the generated code. We can leverage such mechanism by using ``wp.static()`` inside the branch condition to yield a constant boolean. This can provide improved performance by avoiding branching and can be useful for generating specialized kernels:

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

``wp.static(...)`` may also return a Warp function. This can be useful to specialize a kernel or function based on information available at the time of declaring t he Warp function or kernel, or to automatically generate overloads for different types.

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

.. code:: console

    [3. 3.]
    [-1.  3.]
    [2. 0.]
