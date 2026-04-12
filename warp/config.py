# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Global configuration settings for Warp.

This module provides settings to control compilation behavior, debugging, performance,
and runtime behavior of Warp kernels and modules. Settings exist at the global, module,
and kernel levels, with more specific scopes taking precedence.

Settings can be modified by direct assignment before or after calling :func:`warp.init`,
though some settings only take effect if set prior to initialization. See individual
setting documentation for details.

For information on module-level and kernel-level settings, see :doc:`/user_guide/configuration`.
"""

_wp_module_name_ = "warp.config"

version: str = "1.13.0.dev0"
"""Warp version string"""

verify_fp: bool = False
"""Enable floating-point verification for inputs and outputs.

When enabled, checks if all values are finite before and after operations.

Note: Enabling this flag impacts performance.
"""

verify_cuda: bool = False
"""Enable CUDA error checking after kernel launches.

This setting cannot be used during graph capture

Note: Enabling this flag impacts performance
"""

print_launches: bool = False
"""Enable detailed kernel launch logging.

Prints information about each kernel launch including:

- Launch dimensions
- Input/output parameters
- Target device

Note: Enabling this flag impacts performance.
"""

mode: str = "release"
"""Compilation mode for Warp kernels.

Args:
    mode: Either ``"release"`` or ``"debug"``.

Note: Debug mode may impact performance.

This setting can be overridden at the module level by setting the ``"mode"`` module option.
"""

optimization_level: int | None = None
"""Optimization level for Warp kernels.

Args:
    optimization_level: An integer representing the optimization level (0-3), or ``None`` for
        target-specific defaults (``-O2`` for CPU, ``-O3`` for CUDA).

Note: Higher optimization levels increase compilation time but may improve run-time performance.

This setting can be overridden at the module level by setting the ``"optimization_level"`` module option.
"""

verbose: bool = False
"""Enable detailed logging during code generation and compilation."""

verbose_warnings: bool = False
"""Enable extended warning messages with source location information."""

quiet: bool = False
"""Disable Warp module initialization messages.

Error messages and warnings remain unaffected.
"""

verify_autograd_array_access: bool = False
"""Enable warnings for array overwrites that may affect gradient computation."""

enable_vector_component_overwrites: bool = False
"""Allow multiple writes to vector/matrix/quaternion components.

Note: Enabling this may significantly increase kernel compilation time.
"""

legacy_scalar_return_types: bool = False
"""Use legacy scalar return types from built-in functions and indexing.

When ``False`` (the default), built-in function calls and vector/matrix
indexing return Warp scalar instances (e.g. ``wp.float64``, ``wp.int16``)
that match the input types. The types ``wp.int32``, ``wp.float32``, and
``wp.bool`` are aliases for Python's ``int``, ``float``, and ``bool``, so
those continue to return Python built-in values.

Set to ``True`` to restore the pre-1.12 behavior where all scalar
operations return Python built-in types (``int``, ``float``, ``bool``).
"""

cache_kernels: bool = True
"""Enable kernel caching between application launches."""

kernel_cache_dir: str | None = None
"""Directory path for storing compiled kernel cache.

If ``None``, the path is determined in the following order:

1. ``WARP_CACHE_PATH`` environment variable.
2. System's user cache directory (via ``appdirs.user_cache_directory``).

A version-specific subdirectory is automatically appended to the resolved
base path to prevent cache collisions between different Warp versions.

Note: Subdirectories prefixed with ``wp_`` will be created in this location.
"""

cuda_output: str | None = None
"""Preferred CUDA output format for kernel compilation.

Args:
    cuda_output: One of {``None``, ``"ptx"``, ``"cubin"``}. If ``None``, format is auto-determined.
"""

ptx_target_arch: int | None = None
"""Target architecture version for PTX generation, e.g., ``ptx_target_arch = 75``.

If ``None``, the architecture is determined by devices present in the system.
"""

cuda_arch_suffix: str | None = None
"""CUDA architecture suffix for kernel compilation.

Controls whether architecture-specific or family-specific suffixes are
appended to the ``--gpu-architecture`` flag passed to NVRTC.

Args:
    cuda_arch_suffix: One of {``None``, ``"a"``, ``"f"``}.
        ``None`` disables suffixes (default, current behavior).
        ``"a"`` enables architecture-specific features (requires sm_90+).
        ``"f"`` enables family-specific features (requires sm_100+ and CUDA 12.9+).
"""

lineinfo: bool = False
"""Enable the compilation of modules with line information.

Modules compiled for GPU execution will be compiled with the
``--generate-line-info`` compiler option, which generates line-number
information for device code. Line-number information is always included when
compiling a module in ``"debug"`` mode regardless of this setting.

This setting can be overridden at the module level by setting the ``"lineinfo"`` module option.
"""

line_directives: bool = True
"""Enable Python source line mapping in generated code.

If ``True``, ``#line`` directives are inserted in generated code for modules
compiled with line information to map back to the original Python source file.
"""

compile_time_trace: bool = False
"""Enable the generation of Trace Event Format files for runtime module compilation.

These are JSON files that can be opened by tools like ``edge://tracing/`` and
``chrome://tracing/``.

This setting is currently only effective when compiling modules for the GPU with NVRTC (CUDA 12.8+).

This setting can be overridden at the module level by setting the ``"compile_time_trace"`` module option.
"""

enable_backward: bool = True
"""Enable compilation of kernel backward passes.

This setting can be overridden at the module level by setting the ``"enable_backward"`` module option.
"""

enable_mathdx_gemm: bool = True
"""Use libmathdx (cuBLASDx) for tile_matmul on GPU when available.

When False, tile_matmul falls back to a scalar GEMM implementation, which avoids
the slow libmathdx LTO compilation at the cost of runtime performance.

This setting can be overridden at the module level by setting the
``"enable_mathdx_gemm"`` module option.
"""

cpu_compiler_flags: str | None = None
"""Flags controlling CPU kernel compilation.

Warp acts as a compiler driver for the embedded Clang frontend. The flag
``-march=native`` is intercepted and triggers host CPU feature detection
(equivalent to ``llvm::sys::getHostCPUName()`` + ``getHostCPUFeatures()``).
All other flags are passed through to the Clang frontend as-is.

The value controls both CPU target detection and extra compiler flags:

- ``None`` (default): detect host CPU features (equivalent to ``"-march=native"``).
- ``""``: disable host CPU detection; compile for a generic target.
- ``"-march=native"``: explicitly detect host CPU features.
- ``"-march=native -fno-vectorize"``: detect host CPU + pass ``-fno-vectorize``.
- ``"-fno-vectorize"``: generic target + pass ``-fno-vectorize``.

Changing this setting invalidates the kernel cache.
"""

llvm_cuda: bool = False
"""Use Clang/LLVM compiler instead of NVRTC for CUDA compilation."""

enable_graph_capture_module_load_by_default: bool = True
"""Enable automatic module loading before graph capture.

Only affects systems with CUDA driver versions below 12.3.
"""

enable_mempools_at_init: bool = True
"""Enable CUDA memory pools during device initialization when supported."""

max_unroll: int = 16
"""Maximum unroll factor for loops.

Note that ``max_unroll`` does not consider the total number of iterations in
nested loops. This can result in a large amount of automatically generated code
if each nested loop is below the ``max_unroll`` threshold.

This setting can be overridden at the module level by setting the ``"max_unroll"`` module option.
"""

enable_tiles_in_stack_memory: bool | None = True
"""Use stack memory instead of static memory for tile allocations on the CPU.

Static memory in kernels is not well supported on some architectures (notably AArch64). We work
around it by reserving stack memory on kernel entry and pointing a reserved callee-saved
register to it (AArch64) or a single static pointer (x86-64).

When set to ``None``, this flag automatically enables stack memory on ``aarch64``
platforms (Linux ARM) and disables it on other architectures. To explicitly enable or disable
this behavior regardless of architecture, set this flag to ``True`` or ``False``.
"""

use_precompiled_headers: bool = True
"""Enable the use of precompiled headers during kernel compilation.
"""

legacy_cpu_linker: bool = False
"""Use the legacy RTDyld linker instead of JITLink for CPU kernel loading.

The default JITLink linker is more robust against virtual address space
fragmentation (e.g. caused by the CUDA driver). Set this to ``True`` to
use the older RTDyld linker, which supports step-through debugging of CPU
kernels with pre-built LLVM libraries that lack the JITLink debug symbols.

This setting can be changed at runtime.  Each linker has its own JIT
instance, created lazily on first use and kept alive so that previously
loaded CPU modules remain valid.

.. note::

   Step-through debugging with JITLink requires building Warp with
   ``--build-llvm`` to get LLVM 21+ with the necessary ORC runtime symbols.

.. warning::

   This flag is experimental and may be removed without warning in a
   future release.
"""

load_module_max_workers: int | None = 0
"""Default number of worker threads for compiling and loading modules in parallel.

For ``wp.load_module()`` and ``wp.force_load()``, if the ``max_workers`` parameter is not specified,
the default number of worker threads is determined by this setting. ``0`` means serial loading.
If ``None``, Warp determines the behavior (currently equal to ``min(os.cpu_count(), 4)``).
"""

_git_commit_hash: str | None = None
"""Git commit hash associated with the Warp installation.

Set automatically by CI, do not modify.
"""
