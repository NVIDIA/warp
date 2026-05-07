Debugging
=========

Printing Values
---------------

Often one of the best debugging methods is to simply print values from kernels. Warp supports printing all built-in
types using the ``print()`` function, e.g.::

    v = wp.vec3(1.0, 2.0, 3.0)

    print(v)   

In addition, formatted C-style printing is available through the ``wp.printf()`` function, e.g.::

    x = 1.0
    i = 2

    wp.printf("A float value %f, an int value: %d", x, i)

.. note:: Formatted printing is only available for scalar types (e.g.: ``int`` and ``float``) not vector types.

Printing Launches
-----------------

For complex applications it can be difficult to understand the order-of-operations that lead to a bug. To help diagnose
these issues Warp supports a simple option to print out all launches and arguments to the console::

    wp.config.print_launches = True


Step-Through Debugging
----------------------

It is possible to attach IDE debuggers such as Visual Studio to Warp processes to step through generated kernel code.
Users should first compile the kernels in debug mode by setting::
   
    wp.config.mode = "debug"

This setting ensures that line numbers, and debug symbols are generated correctly. After launching the Python process,
the debugger should be attached, and a breakpoint inserted into the generated code.

.. note:: Generated kernel code is not a 1:1 correspondence with the original Python code, but individual operations can still be replayed and variables inspected.

Also see :github:`warp/tests/walkthrough_debug.py` for an example of how to debug Warp kernel code running on the CPU.

Generated Code
--------------

The generated code for kernels is stored in a central cache location in the user's home directory, the cache location
is printed at startup when ``wp.init()`` is called, for example:

.. code-block:: console

    Warp 0.8.1 initialized:
        CUDA Toolkit: 11.8, Driver: 11.8
        Devices:
        "cpu"    | AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD
        "cuda:0" | NVIDIA GeForce RTX 3090 (sm_86)
        "cuda:1" | NVIDIA GeForce RTX 2080 Ti (sm_75)
        Kernel cache: C:\Users\LukasW\AppData\Local\NVIDIA Corporation\warp\Cache\0.8.1

The kernel cache has ``gen`` and ``bin`` folders that contain the generated C++/CUDA code and the compiled binaries
respectively. Occasionally it can be useful to inspect the generated code for debugging / profiling.

Bounds Checking
---------------

Warp will perform bounds checking in debug build configurations to ensure that all array accesses lie within the defined
shape.

CUDA Verification
-----------------

It is possible to generate out-of-bounds memory access violations through poorly formed kernel code or inputs. In this
case the CUDA runtime will detect the violation and put the CUDA context into an error state. Subsequent kernel launches
may silently fail which can lead to hard to diagnose issues.

If a CUDA error is suspected a simple verification method is to enable::

    wp.config.verify_cuda = True

This setting will check the CUDA context after every operation to ensure that it is still valid. If an error is
encountered it will raise an exception that often helps to narrow down the problematic kernel.

.. note:: Verifying CUDA state at each launch requires synchronizing CPU and GPU which has a significant overhead. Users should ensure this setting is only used during debugging.
