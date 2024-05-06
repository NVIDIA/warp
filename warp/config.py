# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import Optional

version: str = "1.1.0"

verify_fp: bool = False  # verify inputs and outputs are finite after each launch
verify_cuda: bool = False  # if true will check CUDA errors after each kernel launch / memory operation
print_launches: bool = False  # if true will print out launch information

mode: str = "release"
verbose: bool = False  # print extra informative messages
verbose_warnings: bool = False  # whether file and line info gets included in Warp warnings
quiet: bool = False  # suppress all output except errors and warnings

cache_kernels: bool = True
kernel_cache_dir: bool = None  # path to kernel cache directory, if None a default path will be used

cuda_output: Optional[str] = (
    None  # preferred CUDA output format for kernels ("ptx" or "cubin"), determined automatically if unspecified
)

ptx_target_arch: int = 70  # target architecture for PTX generation, defaults to the lowest architecture that supports all of Warp's features

enable_backward: bool = True  # whether to compiler the backward passes of the kernels

llvm_cuda: bool = False  # use Clang/LLVM instead of NVRTC to compile CUDA

enable_graph_capture_module_load_by_default: bool = (
    True  # Default value of force_module_load for capture_begin() if CUDA driver does not support at least CUDA 12.3
)

enable_mempools_at_init: bool = True  # Whether CUDA devices will be initialized with mempools enabled (if supported)

max_unroll: int = 16
