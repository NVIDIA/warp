# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Global configuration settings for Warp.

This module provides settings to control compilation behavior, debugging, performance,
and runtime behavior of Warp kernels and modules. Settings exist at the global, module,
and kernel levels, with more specific scopes taking precedence.

Settings can be modified by direct assignment before or after calling :func:`warp.init`,
though some settings only take effect if set prior to initialization. See individual
setting documentation for details.

For information on module-level and kernel-level settings, see :doc:`/user_guide/configuration`.
"""

from typing import Optional as _Optional

_wp_module_name_ = "warp.config"

version: str = "1.12.0.dev0"
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

optimization_level: _Optional[int] = None
"""Optimization level for Warp kernels.

Args:
    optimization_level: An integer representing the optimization level (0-3), or ``None`` for default behavior.

Note: Higher optimization levels increase compilation time but may improve run-time performance.

Currently only affects GPU modules.

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

cache_kernels: bool = True
"""Enable kernel caching between application launches."""

kernel_cache_dir: _Optional[str] = None
"""Directory path for storing compiled kernel cache.

If ``None``, the path is determined in the following order:

1. ``WARP_CACHE_PATH`` environment variable.
2. System's user cache directory (via ``appdirs.user_cache_directory``).

Note: Subdirectories prefixed with ``wp_`` will be created in this location.
"""

cuda_output: _Optional[str] = None
"""Preferred CUDA output format for kernel compilation.

Args:
    cuda_output: One of {``None``, ``"ptx"``, ``"cubin"``}. If ``None``, format is auto-determined.
"""

ptx_target_arch: _Optional[int] = None
"""Target architecture version for PTX generation, e.g., ``ptx_target_arch = 75``.

If ``None``, the architecture is determined by devices present in the system.
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

enable_tiles_in_stack_memory: _Optional[bool] = True
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

load_module_max_workers: _Optional[int] = 0
"""Default number of worker threads for compiling and loading modules in parallel.

For ``wp.load_module()`` and ``wp.force_load()``, if the ``max_workers`` parameter is not specified,
the default number of worker threads is determined by this setting. ``0`` means serial loading.
If ``None``, Warp determines the behavior (currently equal to ``min(os.cpu_count(), 4)``).
"""

_git_commit_hash: _Optional[str] = None
"""Git commit hash associated with the Warp installation.

Set automatically by CI, do not modify.
"""
