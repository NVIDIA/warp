Runtime Settings
================

Warp has settings at the global, module, and kernel level that can be used to fine-tune the compilation and verbosity
of Warp programs. In cases in which a setting can be changed at multiple levels (e.g ``enable_backward``),
the setting at the more-specific scope takes precedence.

Global Settings
---------------

To change a setting, prepend ``wp.config.`` to the name of the variable and assign a value to it.
Some settings may be changed on the fly, while others need to be set prior to calling ``wp.init()`` to take effect.

For example, the location of the user kernel cache can be changed with:

.. code-block:: python

    import os

    import warp as wp

    example_dir = os.path.dirname(os.path.realpath(__file__))

    # set default cache directory before wp.init()
    wp.config.kernel_cache_dir = os.path.join(example_dir, "tmp", "warpcache1")

    wp.init()


Basic Global Settings
^^^^^^^^^^^^^^^^^^^^^

+--------------------+---------+-------------+--------------------------------------------------------------------------+
| Field              | Type    |Default Value| Description                                                              |
+====================+=========+=============+==========================================================================+
|``verify_fp``       | Boolean | ``False``   | If ``True``, Warp will check that inputs and outputs are finite before   |
|                    |         |             | and/or after various operations. **Has performance implications.**       |
+--------------------+---------+-------------+--------------------------------------------------------------------------+
|``verify_cuda``     | Boolean | ``False``   | If ``True``, Warp will check for CUDA errors after every launch and      |
|                    |         |             | memory operation. CUDA error verification cannot be used during graph    |
|                    |         |             | capture. **Has performance implications.**                               |              
+--------------------+---------+-------------+--------------------------------------------------------------------------+
|``print_launches``  | Boolean | ``False``   | If ``True``, Warp will print details of every kernel launch to standard  |
|                    |         |             | out (e.g. launch dimensions, inputs, outputs, device, etc.).             |
|                    |         |             | **Has performance implications.**                                        |
+--------------------+---------+-------------+--------------------------------------------------------------------------+
|``mode``            | String  |``"release"``| Controls whether to compile Warp kernels in debug or release mode.       |
|                    |         |             | Valid choices are ``"release"`` or ``"debug"``.                          |
|                    |         |             | **Has performance implications.**                                        |
+--------------------+---------+-------------+--------------------------------------------------------------------------+
|``verbose``         | Boolean | ``False``   | If ``True``, additional information will be printed to standard out      |
|                    |         |             | during code generation, compilation, etc.                                |
+--------------------+---------+-------------+--------------------------------------------------------------------------+
|``quiet``           | Boolean | ``False``   | If ``True``, Warp module initialization messages will be disabled.       |
|                    |         |             | This setting does not affect error messages and warnings.                |
+--------------------+---------+-------------+--------------------------------------------------------------------------+
|``kernel_cache_dir``| String  | ``None``    | The path to the directory used for the user kernel cache. Subdirectories |
|                    |         |             | named ``gen`` and ``bin`` will be created in this directory. If ``None``,|
|                    |         |             | a directory will be automatically determined using                       |
|                    |         |             | `appdirs.user_cache_directory <https://github.com/ActiveState/appdirs>`_ |
|                    |         |             |                                                                          |
+--------------------+---------+-------------+--------------------------------------------------------------------------+
|``enable_backward`` | Boolean | ``True``    | If ``True``, backward passes of kernels will be compiled by default.     |
|                    |         |             | Disabling this setting can reduce kernel compilation times.              |
+--------------------+---------+-------------+--------------------------------------------------------------------------+

Advanced Global Settings
^^^^^^^^^^^^^^^^^^^^^^^^

+--------------------+---------+-------------+--------------------------------------------------------------------------+
| Field              | Type    |Default Value| Description                                                              |
+====================+=========+=============+==========================================================================+
|``cache_kernels``   | Boolean | ``True``    | If ``True``, kernels that have already been compiled from previous       |
|                    |         |             | application launches will not be recompiled.                             |
+--------------------+---------+-------------+--------------------------------------------------------------------------+
|``cuda_output``     | String  | ``None``    | The preferred CUDA output format for kernels. Valid choices are ``None``,|
|                    |         |             | ``"ptx"``, and ``"cubin"``. If ``None``, a format will be determined     |
|                    |         |             | automatically.                                                           |
+--------------------+---------+-------------+--------------------------------------------------------------------------+
|``ptx_target_arch`` | Integer | 70          | The target architecture for PTX generation.                              |
+--------------------+---------+-------------+--------------------------------------------------------------------------+
|``llvm_cuda``       | Boolean | ``False``   | If ``True``, Clang/LLVM will be used to compile CUDA code instead of     |
|                    |         |             | NVTRC.                                                                   |
+--------------------+---------+-------------+--------------------------------------------------------------------------+

Module Settings
---------------

Module-level settings to control runtime compilation and code generation may be changed by passing a dictionary of
option pairs to ``wp.set_module_options()``.

For example, compilation of backward passes for the kernel in an entire module can be disabled with:

.. code:: python

    wp.set_module_options({"enable_backward": False})

The options for a module can also be queried using ``wp.get_module_options()``.

+--------------------+---------+-------------+--------------------------------------------------------------------------+
| Field              | Type    |Default Value| Description                                                              |
+====================+=========+=============+==========================================================================+
|``mode``            | String  | Global      | Controls whether to compile the module's kernels in debug or release     |
|                    |         | setting     | mode by default. Valid choices are ``"release"`` or ``"debug"``.         |
+--------------------+---------+-------------+--------------------------------------------------------------------------+
|``max_unroll``      | Integer | 16          | The maximum fixed-size loop to unroll. Note that ``max_unroll`` does not |
|                    |         |             | consider the total number of iterations in nested loops. This can result |
|                    |         |             | in a large amount of automatically generated code if each nested loop is |
|                    |         |             | below the ``max_unroll`` threshold.                                      |
+--------------------+---------+-------------+--------------------------------------------------------------------------+
|``enable_backward`` | Boolean | Global      | If ``True``, backward passes of kernels will be compiled by default.     |
|                    |         | setting     | Valid choices are ``"release"`` or ``"debug"``.                          |
+--------------------+---------+-------------+--------------------------------------------------------------------------+
|``fast_math``       | Boolean | ``False``   | If ``True``, CUDA kernels will be compiled with the ``--use_fast_math``  |
|                    |         |             | compiler option, which enables some fast math operations that are faster |
|                    |         |             | but less accurate.                                                       |
+--------------------+---------+-------------+--------------------------------------------------------------------------+
|``cuda_output``     | String  | ``None``    | The preferred CUDA output format for kernels. Valid choices are ``None``,|
|                    |         |             | ``"ptx"``, and ``"cubin"``. If ``None``, a format will be determined     |
|                    |         |             | automatically. The module-level setting takes precedence over the global |
|                    |         |             | setting.                                                                 |
+--------------------+---------+-------------+--------------------------------------------------------------------------+

Kernel Settings
---------------

``enable_backward`` is currently the only setting that can also be configured on a per-kernel level.
Backward-pass compilation can be disabled by passing an argument into the ``@wp.kernel`` decorator
as in the following example:

.. code-block:: python

    @wp.kernel(enable_backward=False)
    def scale_2(
        x: wp.array(dtype=float),
        y: wp.array(dtype=float),
    ):
        y[0] = x[0] ** 2.0
