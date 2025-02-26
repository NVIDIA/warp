.. _Configuration:

Configuration
=============

Warp has settings at the global, module, and kernel level that can be used to fine-tune the compilation and verbosity
of Warp programs. In cases in which a setting can be changed at multiple levels (e.g.: ``enable_backward``),
the setting at the more-specific scope takes precedence.

.. _global-settings:

Global Settings
---------------

Settings can be modified by direct assignment before or after calling ``wp.init()``,
though some settings only take effect if set prior to initialization.

For example, the location of the user kernel cache can be changed with:

.. code-block:: python

    import os

    import warp as wp

    example_dir = os.path.dirname(os.path.realpath(__file__))

    # set default cache directory before wp.init()
    wp.config.kernel_cache_dir = os.path.join(example_dir, "tmp", "warpcache1")

    wp.init()

.. automodule:: warp.config
   :members:

Module Settings
---------------

Module-level settings to control runtime compilation and code generation may be changed by passing a dictionary of
option pairs to ``wp.set_module_options()``.

For example, compilation of backward passes for the kernel in an entire module can be disabled with:

.. code:: python

    wp.set_module_options({"enable_backward": False})

The options for a module can also be queried using ``wp.get_module_options()``.

+--------------------------------------+---------+-------------+--------------------------------------------------------------------------+
| Field                                | Type    |Default Value| Description                                                              |
+======================================+=========+=============+==========================================================================+
|``mode``                              | String  | Global      | Controls whether to compile the module's kernels in debug or release     |
|                                      |         | setting     | mode by default. Valid choices are ``"release"`` or ``"debug"``.         |
+--------------------------------------+---------+-------------+--------------------------------------------------------------------------+
|``max_unroll``                        | Integer | Global      | The maximum fixed-size loop to unroll. Note that ``max_unroll`` does not |
|                                      |         | setting     | consider the total number of iterations in nested loops. This can result |
|                                      |         |             | in a large amount of automatically generated code if each nested loop is |
|                                      |         |             | below the ``max_unroll`` threshold.                                      |
+--------------------------------------+---------+-------------+--------------------------------------------------------------------------+
|``enable_backward``                   | Boolean | Global      | If ``True``, backward passes of kernels will be compiled by default.     |
|                                      |         | setting     | Valid choices are ``"release"`` or ``"debug"``.                          |
+--------------------------------------+---------+-------------+--------------------------------------------------------------------------+
|``fast_math``                         | Boolean | ``False``   | If ``True``, CUDA kernels will be compiled with the ``--use_fast_math``  |
|                                      |         |             | compiler option, which enables some fast math operations that are faster |
|                                      |         |             | but less accurate.                                                       |
+--------------------------------------+---------+-------------+--------------------------------------------------------------------------+
|``fuse_fp``                           | Boolean | ``True``    | If ``True``, allow compilers to emit fused floating point operations     |
|                                      |         |             | such as fused-multiply-add. This may improve numerical accuracy and      |
|                                      |         |             | is generally recommended. Setting to ``False`` can help ensuring         |
|                                      |         |             | that functionally equivalent kernels will produce identical results      |
|                                      |         |             | unaffected by the presence or absence of fused operations.               |
+--------------------------------------+---------+-------------+--------------------------------------------------------------------------+
|``lineinfo``                          | Boolean | ``False``   | If ``True``, CUDA kernels will be compiled with the                      |
|                                      |         |             | ``--generate-line-info`` compiler option, which generates line-number    |
|                                      |         |             | information for device code, e.g. to allow NVIDIA Nsight Compute to      |
|                                      |         |             | correlate CUDA-C source and SASS. Line-number information is always      |
|                                      |         |             | included when compiling kernels in ``"debug"`` mode regardless of this   |
|                                      |         |             | setting.                                                                 |
+--------------------------------------+---------+-------------+--------------------------------------------------------------------------+
|``cuda_output``                       | String  | ``None``    | The preferred CUDA output format for kernels. Valid choices are ``None``,|
|                                      |         |             | ``"ptx"``, and ``"cubin"``. If ``None``, a format will be determined     |
|                                      |         |             | automatically. The module-level setting takes precedence over the global |
|                                      |         |             | setting.                                                                 |
+--------------------------------------+---------+-------------+--------------------------------------------------------------------------+

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
