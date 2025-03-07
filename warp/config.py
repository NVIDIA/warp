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

from typing import Optional

version: str = "1.6.2"
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

kernel_cache_dir: Optional[str] = None
"""Directory path for storing compiled kernel cache.

If ``None``, the path is determined in the following order:

1. ``WARP_CACHE_PATH`` environment variable.
2. System's user cache directory (via ``appdirs.user_cache_directory``).

Note: Subdirectories prefixed with ``wp_`` will be created in this location.
"""

cuda_output: Optional[str] = None
"""Preferred CUDA output format for kernel compilation.

Args:
    cuda_output: One of {``None``, ``"ptx"``, ``"cubin"``}. If ``None``, format is auto-determined.
"""

ptx_target_arch: int = 75
"""Target architecture version for PTX generation.

Defaults to minimum architecture version supporting all Warp features.
"""

enable_backward: bool = True
"""Enable compilation of kernel backward passes."""

llvm_cuda: bool = False
"""Use Clang/LLVM compiler instead of NVRTC for CUDA compilation."""

enable_graph_capture_module_load_by_default: bool = True
"""Enable automatic module loading before graph capture.

Only affects systems with CUDA driver versions below 12.3.
"""

enable_mempools_at_init: bool = True
"""Enable CUDA memory pools during device initialization when supported."""

max_unroll: int = 16
"""Maximum unroll factor for loops."""
