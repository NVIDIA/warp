Reducing Compilation and Startup Time
=====================================

.. currentmodule:: warp

Warp loads a module's kernels the first time one of them is used on a device. If
no matching binary is cached, Warp compiles the module first. Later runs are
usually faster because Warp can load the cached binary. Applications with many
kernels, generic instances, or MathDx-backed tile operations may still spend
noticeable time compiling.

Measuring compilation and startup time
--------------------------------------

Start by checking the module-load output. By default, Warp prints the module
name, target device, elapsed load time, and whether it compiled the module or
loaded it from cache. These lines show which modules take time to load and
whether each was compiled or loaded from cache.

Set :attr:`warp.config.log_level` to ``wp.LOG_DEBUG`` to print the start of each
module load and details such as ``block_dim``. Use this output to see which
module is loading during a long startup pause.

Compilation can reuse artifacts from several independent caches:

- Warp's kernel cache.
- Warp's LTO cache for MathDx-backed tile operations.
- The NVIDIA CUDA driver's compute cache.

See :ref:`benchmarking-cold-start-compilation` for a cold-start benchmarking
workflow and compile-time tracing when the total module-load time does not show
where the delay occurs.

Reduce compilation work
-----------------------

Define kernels and overloads before first module load
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Warp compiles kernels by module. Once Warp has compiled or loaded a module
variant, adding new kernels or overloads changes the module hash. This includes
kernels defined at runtime inside Python functions. On the next launch, Warp
loads the resulting variant from cache or compiles it.

Define a module's expected kernels and overloads before its first load so later
launches can reuse the loaded module or cached artifacts.

Prefer this structure:

.. code-block:: python

    import warp as wp


    @wp.kernel
    def first_kernel(values: wp.array[float]):
        values[wp.tid()] *= 2.0


    @wp.kernel
    def second_kernel(values: wp.array[float]):
        values[wp.tid()] += 1.0


    def run(values):
        wp.launch(first_kernel, dim=values.shape, inputs=[values])
        wp.launch(second_kernel, dim=values.shape, inputs=[values])

If a kernel must be defined later, use ``module="unique"`` to assign it to a
separate module:

.. code-block:: python

    def make_scale_kernel(scale: float):
        @wp.kernel(module="unique")
        def scale_kernel(values: wp.array[float]):
            values[wp.tid()] *= scale

        return scale_kernel

The separate module still requires compilation, but it does not change the
already loaded module. See :ref:`kernel-settings` for accepted ``module``
values.

For generic kernels, declare the type combinations you expect to launch before
the first launch:

.. code-block:: python

    from typing import Any

    import warp as wp


    @wp.kernel
    def scale(x: wp.array[Any], s: Any):
        i = wp.tid()
        x[i] = x[i] * s


    scale_f32 = wp.overload(scale, {"x": wp.array[wp.float32], "s": wp.float32})
    scale_f64 = wp.overload(scale, {"x": wp.array[wp.float64], "s": wp.float64})

Only declare overloads that the application actually uses. Excessive overloads
increase generated code and compile work.

Limit runtime specializations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Warp normally chooses the CUDA target architecture from the local device, so
architecture variants matter mainly when compiling modules ahead of time.

Each compiled module variant has a fixed ``block_dim``.
:func:`wp.launch() <warp.launch>` and
:func:`wp.launch_tiled() <warp.launch_tiled>` select the variant that matches
the launch's ``block_dim``. If it is not loaded, Warp loads it from cache or
compiles it. When kernels in the same module use different values, Warp may
compile the whole module once for each value. In that case, group kernels by
``block_dim`` in separate modules, or use ``module="unique"`` for a kernel that
needs its own value.

When preloading modules, include only the variants the application expects to
use. Each unused ``block_dim`` or tile combination adds compile work.

Set compilation options before importing or loading modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Module options affect compiled output and cache reuse. Configure global
compilation options before importing modules that define kernels. Changing them
later can cause modules to use different settings or require additional
compilation.

Set module-specific options before the module's first load. Changing them after
a launch or explicit load can trigger another compilation and reduce cache
reuse. Avoid changing the same module's options repeatedly during a run.

Disable backward code generation when gradients are not needed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Warp generates backward code for kernels that may participate in automatic
differentiation. If an application does not need gradients for those kernels,
disable backward code generation to reduce generated code and compile work.

Disable it globally before module creation:

.. code-block:: python

    import warp as wp

    wp.config.enable_backward = False

For modules that do not need gradients, set the module option before the module
loads:

.. code-block:: python

    import warp as wp
    import my_app.kernels

    wp.set_module_options({"enable_backward": False}, module=my_app.kernels)

For individual kernels, use the decorator option:

.. code-block:: python

    @wp.kernel(enable_backward=False)
    def integrate(values: wp.array[float]):
        values[wp.tid()] += 1.0

Only disable backward generation for kernels whose adjoints will not be used by
:class:`Tape <warp.Tape>` or differentiable framework integrations.

Choose module boundaries
^^^^^^^^^^^^^^^^^^^^^^^^

Put kernels that are deployed, loaded, and changed together in the same module.
This can avoid repeating fixed code-generation, compilation, and load costs
across several small modules.

Use separate modules for kernels that change independently, are rarely used,
require different compilation options, or use different CUDA ``block_dim``
values. A broad module can compile more code when its contents or options
change. Many tiny modules can duplicate shared functions and structs and add
fixed load-time overhead. Runtime-created unique modules also cannot always be
preloaded by package name.

Use ``module="unique"`` to isolate one kernel in a separate module. Add
``module_options=...`` when that kernel also needs different compilation
options:

.. code-block:: python

    @wp.kernel(module="unique", module_options={"enable_backward": False})
    def halve_values(values: wp.array[float]):
        i = wp.tid()
        values[i] *= 0.5

Compile known kernels before latency-sensitive work
---------------------------------------------------

Preloading and ahead-of-time compilation work only for kernels and variants
that already exist. An application that creates a kernel or specialization
from runtime data must wait for that data before Warp can compile or load it.
Create known variants during the build or before latency-sensitive work begins.

Load independent modules in parallel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When compilation work is spread across several independent modules,
:func:`wp.load_module() <warp.load_module>` can compile or load them
concurrently:

.. code-block:: python

    import warp as wp
    import my_app.kernels

    wp.load_module(
        my_app.kernels,
        device="cuda:0",
        recursive=True,
        max_workers=4,
    )

The submodules must already be imported and registered with Warp. Parallel
loading offers little benefit when one large module dominates or when module
loads are short.

Without multiple workers, :func:`wp.load_module() <warp.load_module>` runs the
same compilation earlier, before CUDA graph capture or a frame loop.

Compile known modules for deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For an application with a known set of kernels and a controlled deployment
environment, use :func:`wp.compile_aot_module() <warp.compile_aot_module>`
during the build and load its output with
:func:`wp.load_aot_module() <warp.load_aot_module>` at runtime. This example
builds a CUBIN for a deployment GPU with compute capability 9.0:

.. code-block:: python

    # build_kernels.py
    import warp as wp
    import my_app.kernels

    wp.compile_aot_module(
        my_app.kernels,
        arch=90,  # Deployment GPU has compute capability 9.0
        module_dir="build/warp_modules",
        use_ptx=False,
    )

Deploy ``build/warp_modules`` with the application to a machine with that GPU
architecture, then load the module at runtime:

.. code-block:: python

    # run_app.py
    import warp as wp
    import my_app.kernels

    wp.load_aot_module(my_app.kernels, module_dir="build/warp_modules", use_ptx=False)

Choose the output format based on the deployment environment:

- CUBIN avoids driver JIT compilation. Build a CUBIN for each GPU architecture
  the application loads. For a known GPU target, CUBIN may also work with an
  older driver under CUDA minor-version compatibility when PTX produced by the
  same newer Toolkit would not. Features that require a newer driver remain
  unavailable.
- PTX lets the driver compile for compatible GPU architectures when the module
  loads. The deployed driver must support the PTX emitted by the CUDA Toolkit
  used to build Warp. Without a CUDA forward-compatibility package, PTX
  produced by a newer Toolkit does not work on an older driver.

See :doc:`../compatibility` for CUDA driver and GPU architecture compatibility.

Development iteration tradeoffs
-------------------------------

These settings can shorten development compile times, but may reduce runtime
performance or available features. Measure their effect on the target workload.

Lower optimization levels
^^^^^^^^^^^^^^^^^^^^^^^^^

A lower optimization level may shorten compile time while iterating. For CUDA,
this setting affects compilation only when the Warp native library was built
with CUDA Toolkit 12.9 or newer:

.. code-block:: python

    import warp as wp

    wp.config.optimization_level = 0

Restore the application's normal optimization level before benchmarking
runtime performance.

Use the native tile-matmul fallback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MathDx compilation for :func:`wp.tile_matmul() <warp._src.lang.tile_matmul>`
can dominate cold compile time, especially when backward code generation adds
adjoint GEMMs. Different tile dimensions or dtypes can require additional
MathDx LTO compilation. If it slows development, try the Warp/native GEMM
fallback temporarily:

.. code-block:: python

    import warp as wp

    wp.config.enable_mathdx_gemm = False

Use the module option when only one module should use the fallback:

.. code-block:: python

    import my_app.kernels

    wp.set_module_options({"enable_mathdx_gemm": False}, module=my_app.kernels)

Other MathDx-backed tile operations, including solvers and FFTs, may also incur
LTO compilation.

Related documentation
---------------------

- :ref:`Compilation Model` explains Warp's Python-to-C++/CUDA compilation model
  and module-load output.
- :doc:`../runtime` explains kernel-cache behavior and cache clearing.
- :doc:`../configuration` lists global, module, and kernel options.
- :doc:`../programming_model/generics` explains implicit and explicit generic
  instantiation.
- :doc:`../programming_model/tiles` covers tile operations, MathDx requirements,
  and LTO errors.
- :ref:`code_generation` describes generated source, module caching, and
  ahead-of-time workflows.
- :doc:`profiling` describes compile-time tracing and cold-start
  measurement with :ref:`benchmarking-cold-start-compilation`.
