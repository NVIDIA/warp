# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import Optional

version: str = "1.2.2"
"""Warp version string"""

verify_fp: bool = False
"""If `True`, Warp will check that inputs and outputs are finite before and/or after various operations.
Has performance implications.
"""

verify_cuda: bool = False
"""If `True`, Warp will check for CUDA errors after every launch and memory operation.
CUDA error verification cannot be used during graph capture. Has performance implications.
"""

print_launches: bool = False
"""If `True`, Warp will print details of every kernel launch to standard out
(e.g. launch dimensions, inputs, outputs, device, etc.). Has performance implications
"""

mode: str = "release"
"""Controls whether to compile Warp kernels in debug or release mode.
Valid choices are `"release"` or `"debug"`. Has performance implications.
"""

verbose: bool = False
"""If `True`, additional information will be printed to standard out during code generation, compilation, etc."""

verbose_warnings: bool = False
"""If `True`, Warp warnings will include extra information such as the source file and line number."""

quiet: bool = False
"""Suppress all output except errors and warnings."""

cache_kernels: bool = True
"""If `True`, kernels that have already been compiled from previous application launches will not be recompiled."""

kernel_cache_dir: Optional[str] = None
"""Path to kernel cache directory, if `None`, a default path will be used."""

cuda_output: Optional[str] = None
"""Preferred CUDA output format for kernels (`"ptx"` or `"cubin"`), determined automatically if unspecified"""

ptx_target_arch: int = 75
"""Target architecture for PTX generation, defaults to the lowest architecture that supports all of Warp's features."""

enable_backward: bool = True
"""Whether to compiler the backward passes of the kernels."""

llvm_cuda: bool = False
"""Use Clang/LLVM instead of NVRTC to compile CUDA."""

enable_graph_capture_module_load_by_default: bool = True
"""Default value of `force_module_load` for `capture_begin()` if CUDA driver does not support at least CUDA 12.3."""

enable_mempools_at_init: bool = True
"""Whether CUDA devices will be initialized with mempools enabled (if supported)."""

max_unroll: int = 16
"""Maximum unroll factor for loops."""
