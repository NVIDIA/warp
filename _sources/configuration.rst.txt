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

.. _module-settings:

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
|``mode``                              | String  | Global      | A module-level override of the :attr:`warp.config.mode` setting.         |
|                                      |         | setting     |                                                                          |
+--------------------------------------+---------+-------------+--------------------------------------------------------------------------+
|``max_unroll``                        | Integer | Global      | A module-level override of the :attr:`warp.config.max_unroll` setting.   |
|                                      |         | setting     |                                                                          |
+--------------------------------------+---------+-------------+--------------------------------------------------------------------------+
|``enable_backward``                   | Boolean | Global      | A module-level override of the :attr:`warp.config.enable_backward`       |
|                                      |         | setting     | setting.                                                                 |
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
|``lineinfo``                          | Boolean | Global      | A module-level override of the :attr:`warp.config.lineinfo` setting.     |
|                                      |         | setting     |                                                                          |
+--------------------------------------+---------+-------------+--------------------------------------------------------------------------+
|``cuda_output``                       | String  | ``None``    | A module-level override of the :attr:`warp.config.cuda_output` setting.  |
+--------------------------------------+---------+-------------+--------------------------------------------------------------------------+

Kernel Settings
---------------

Backward-pass compilation can be disabled on a per-kernel basis by passing the ``enable_backward`` argument into the :func:`@wp.kernel <warp.kernel>` decorator
as in the following example:

.. code-block:: python

    @wp.kernel(enable_backward=False)
    def scale_2(
        x: wp.array(dtype=float),
        y: wp.array(dtype=float),
    ):
        y[0] = x[0] ** 2.0
