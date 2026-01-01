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

"""The ``warp`` package provides array types and functions for creating and manipulating
multi-dimensional data on CPU and CUDA devices. It includes kernel and function decorators
(:func:`kernel`, :func:`func`) for defining parallel code, along with a comprehensive set
of built-in types and functions for use within kernels (see :doc:`/language_reference/builtins`).

The package provides device management, kernel launch and synchronization functions, automatic
differentiation via :class:`Tape` recording, type introspection and construction utilities, and
module compilation and caching.

Additional functionality is available in optional submodules that must be explicitly
imported, such as :mod:`warp.render` for visualization, :mod:`warp.fem` for finite
element methods, and :mod:`warp.sparse` for sparse linear algebra.
"""

# isort: skip_file

# category: Type Annotations

from warp._src.types import Int as Int
from warp._src.types import Float as Float
from warp._src.types import Scalar as Scalar
from warp._src.context import DeviceLike as DeviceLike


# category: Data Types > Scalars

from warp._src.types import bool as bool
from warp._src.types import int8 as int8
from warp._src.types import uint8 as uint8
from warp._src.types import int16 as int16
from warp._src.types import uint16 as uint16
from warp._src.types import int32 as int32
from warp._src.types import uint32 as uint32
from warp._src.types import int64 as int64
from warp._src.types import uint64 as uint64
from warp._src.types import float16 as float16
from warp._src.types import float32 as float32
from warp._src.types import float64 as float64


# category: Data Types > Vectors

from warp._src.types import vec2 as vec2
from warp._src.types import vec2b as vec2b
from warp._src.types import vec2ub as vec2ub
from warp._src.types import vec2s as vec2s
from warp._src.types import vec2us as vec2us
from warp._src.types import vec2i as vec2i
from warp._src.types import vec2ui as vec2ui
from warp._src.types import vec2l as vec2l
from warp._src.types import vec2ul as vec2ul
from warp._src.types import vec2h as vec2h
from warp._src.types import vec2f as vec2f
from warp._src.types import vec2d as vec2d

from warp._src.types import vec3 as vec3
from warp._src.types import vec3b as vec3b
from warp._src.types import vec3ub as vec3ub
from warp._src.types import vec3s as vec3s
from warp._src.types import vec3us as vec3us
from warp._src.types import vec3i as vec3i
from warp._src.types import vec3ui as vec3ui
from warp._src.types import vec3l as vec3l
from warp._src.types import vec3ul as vec3ul
from warp._src.types import vec3h as vec3h
from warp._src.types import vec3f as vec3f
from warp._src.types import vec3d as vec3d

from warp._src.types import vec4 as vec4
from warp._src.types import vec4b as vec4b
from warp._src.types import vec4ub as vec4ub
from warp._src.types import vec4s as vec4s
from warp._src.types import vec4us as vec4us
from warp._src.types import vec4i as vec4i
from warp._src.types import vec4ui as vec4ui
from warp._src.types import vec4l as vec4l
from warp._src.types import vec4ul as vec4ul
from warp._src.types import vec4h as vec4h
from warp._src.types import vec4f as vec4f
from warp._src.types import vec4d as vec4d


# category: Data Types > Matrices

from warp._src.types import mat22 as mat22
from warp._src.types import mat22h as mat22h
from warp._src.types import mat22f as mat22f
from warp._src.types import mat22d as mat22d

from warp._src.types import mat33 as mat33
from warp._src.types import mat33h as mat33h
from warp._src.types import mat33f as mat33f
from warp._src.types import mat33d as mat33d

from warp._src.types import mat44 as mat44
from warp._src.types import mat44h as mat44h
from warp._src.types import mat44f as mat44f
from warp._src.types import mat44d as mat44d

from warp._src.types import matrix_from_cols as matrix_from_cols
from warp._src.types import matrix_from_rows as matrix_from_rows


# category: Data Types > Quaternions

from warp._src.types import quat as quat
from warp._src.types import quath as quath
from warp._src.types import quatf as quatf
from warp._src.types import quatd as quatd


# category: Data Types > Transformations

from warp._src.types import transform as transform
from warp._src.types import transformh as transformh
from warp._src.types import transformf as transformf
from warp._src.types import transformd as transformd


# category: Data Types > Spatial Vectors and Matrices

from warp._src.types import spatial_vector as spatial_vector
from warp._src.types import spatial_vectorh as spatial_vectorh
from warp._src.types import spatial_vectorf as spatial_vectorf
from warp._src.types import spatial_vectord as spatial_vectord

from warp._src.types import spatial_matrix as spatial_matrix
from warp._src.types import spatial_matrixh as spatial_matrixh
from warp._src.types import spatial_matrixf as spatial_matrixf
from warp._src.types import spatial_matrixd as spatial_matrixd


# category: Arrays

from warp._src.types import array as array
from warp._src.types import array1d as array1d
from warp._src.types import array2d as array2d
from warp._src.types import array3d as array3d
from warp._src.types import array4d as array4d

from warp._src.types import fixedarray as fixedarray

from warp._src.types import tile as tile

from warp._src.types import from_ptr as from_ptr

from warp._src.context import zeros as zeros
from warp._src.context import zeros_like as zeros_like
from warp._src.context import ones as ones
from warp._src.context import ones_like as ones_like
from warp._src.context import full as full
from warp._src.context import full_like as full_like
from warp._src.context import clone as clone
from warp._src.context import empty as empty
from warp._src.context import empty_like as empty_like
from warp._src.context import copy as copy


# category: Arrays > Indexed Arrays

from warp._src.types import indexedarray as indexedarray
from warp._src.types import indexedarray1d as indexedarray1d
from warp._src.types import indexedarray2d as indexedarray2d
from warp._src.types import indexedarray3d as indexedarray3d
from warp._src.types import indexedarray4d as indexedarray4d


# category: Spatial Acceleration

from warp._src.types import Bvh as Bvh
from warp._src.types import Mesh as Mesh
from warp._src.types import HashGrid as HashGrid
from warp._src.types import Volume as Volume
from warp._src.types import BvhQuery as BvhQuery
from warp._src.types import BvhQueryTiled as BvhQueryTiled
from warp._src.types import HashGridQuery as HashGridQuery
from warp._src.types import MeshQueryAABB as MeshQueryAABB
from warp._src.types import MeshQueryAABBTiled as MeshQueryAABBTiled
from warp._src.types import MeshQueryPoint as MeshQueryPoint
from warp._src.types import MeshQueryRay as MeshQueryRay


# category: Runtime

from warp._src.context import init as init

from warp._src.context import is_cpu_available as is_cpu_available
from warp._src.context import is_cuda_available as is_cuda_available

from warp._src.build import clear_kernel_cache as clear_kernel_cache
from warp._src.build import clear_lto_cache as clear_lto_cache


# category: Kernel Programming

from warp._src.codegen import WarpCodegenAttributeError as WarpCodegenAttributeError
from warp._src.codegen import WarpCodegenError as WarpCodegenError
from warp._src.codegen import WarpCodegenIndexError as WarpCodegenIndexError
from warp._src.codegen import WarpCodegenKeyError as WarpCodegenKeyError
from warp._src.codegen import WarpCodegenTypeError as WarpCodegenTypeError
from warp._src.codegen import WarpCodegenValueError as WarpCodegenValueError

from warp._src.context import func as func
from warp._src.context import func_grad as func_grad
from warp._src.context import func_replay as func_replay
from warp._src.context import func_native as func_native
from warp._src.context import grad as grad
from warp._src.context import kernel as kernel
from warp._src.context import struct as struct
from warp._src.context import overload as overload
from warp._src.types import constant as constant
from warp._src.utils import map as map
from warp._src.builtins import static as static


# category: Kernel Execution

from warp._src.context import Kernel as Kernel
from warp._src.context import Function as Function
from warp._src.context import Launch as Launch
from warp._src.context import Module as Module

from warp._src.context import launch as launch
from warp._src.context import launch_tiled as launch_tiled
from warp._src.context import synchronize as synchronize


# category: Automatic Differentiation

from warp._src.tape import Tape as Tape


# category: Device Management

from warp._src.context import Device as Device
from warp._src.utils import ScopedDevice as ScopedDevice

from warp._src.context import is_device_available as is_device_available

from warp._src.context import get_devices as get_devices
from warp._src.context import get_preferred_device as get_preferred_device
from warp._src.context import get_cuda_devices as get_cuda_devices
from warp._src.context import get_cuda_device_count as get_cuda_device_count
from warp._src.context import get_cuda_device as get_cuda_device
from warp._src.context import get_cuda_supported_archs as get_cuda_supported_archs

from warp._src.context import map_cuda_device as map_cuda_device
from warp._src.context import unmap_cuda_device as unmap_cuda_device

from warp._src.context import get_device as get_device
from warp._src.context import set_device as set_device

from warp._src.context import synchronize_device as synchronize_device


# category: Module Management

from warp._src.context import set_module_options as set_module_options
from warp._src.context import get_module_options as get_module_options

from warp._src.context import get_module as get_module
from warp._src.context import force_load as force_load
from warp._src.context import load_module as load_module

from warp._src.context import compile_aot_module as compile_aot_module
from warp._src.context import load_aot_module as load_aot_module


# category: CUDA Stream Management

from warp._src.context import Stream as Stream
from warp._src.utils import ScopedStream as ScopedStream


from warp._src.context import get_stream as get_stream
from warp._src.context import set_stream as set_stream
from warp._src.context import wait_stream as wait_stream
from warp._src.context import synchronize_stream as synchronize_stream


# category: CUDA Event Management

from warp._src.context import Event as Event
from warp._src.context import record_event as record_event
from warp._src.context import wait_event as wait_event
from warp._src.context import synchronize_event as synchronize_event
from warp._src.context import get_event_elapsed_time as get_event_elapsed_time


# category: CUDA Memory Management

from warp._src.utils import ScopedMempool as ScopedMempool
from warp._src.utils import ScopedMempoolAccess as ScopedMempoolAccess
from warp._src.utils import ScopedPeerAccess as ScopedPeerAccess

from warp._src.context import is_mempool_supported as is_mempool_supported
from warp._src.context import is_mempool_enabled as is_mempool_enabled
from warp._src.context import set_mempool_enabled as set_mempool_enabled

from warp._src.context import set_mempool_release_threshold as set_mempool_release_threshold
from warp._src.context import get_mempool_release_threshold as get_mempool_release_threshold
from warp._src.context import get_mempool_used_mem_current as get_mempool_used_mem_current
from warp._src.context import get_mempool_used_mem_high as get_mempool_used_mem_high

from warp._src.context import is_mempool_access_supported as is_mempool_access_supported
from warp._src.context import is_mempool_access_enabled as is_mempool_access_enabled
from warp._src.context import set_mempool_access_enabled as set_mempool_access_enabled

from warp._src.context import is_peer_access_supported as is_peer_access_supported
from warp._src.context import is_peer_access_enabled as is_peer_access_enabled
from warp._src.context import set_peer_access_enabled as set_peer_access_enabled


# category: CUDA Graph Management

from warp._src.utils import ScopedCapture as ScopedCapture

from warp._src.context import is_conditional_graph_supported as is_conditional_graph_supported

from warp._src.context import capture_begin as capture_begin
from warp._src.context import capture_end as capture_end
from warp._src.context import capture_launch as capture_launch
from warp._src.context import capture_if as capture_if
from warp._src.context import capture_while as capture_while
from warp._src.context import capture_debug_dot_print as capture_debug_dot_print


# category: CUDA Interprocess Communication

from warp._src.types import from_ipc_handle as from_ipc_handle
from warp._src.context import event_from_ipc_handle as event_from_ipc_handle


# category: Profiling

from warp._src.utils import ScopedTimer as ScopedTimer

from warp._src.utils import TimingResult as TimingResult
from warp._src.utils import timing_begin as timing_begin
from warp._src.utils import timing_end as timing_end
from warp._src.utils import timing_print as timing_print


# category: Profiling > Timing Flags

from warp._src.utils import TIMING_KERNEL as TIMING_KERNEL
from warp._src.utils import TIMING_KERNEL_BUILTIN as TIMING_KERNEL_BUILTIN
from warp._src.utils import TIMING_MEMCPY as TIMING_MEMCPY
from warp._src.utils import TIMING_MEMSET as TIMING_MEMSET
from warp._src.utils import TIMING_GRAPH as TIMING_GRAPH
from warp._src.utils import TIMING_ALL as TIMING_ALL


# category: NumPy Interop

from warp._src.types import dtype_from_numpy as dtype_from_numpy
from warp._src.types import dtype_to_numpy as dtype_to_numpy

from warp._src.context import from_numpy as from_numpy


# category: DLPack Interop

from warp._src.dlpack import from_dlpack as from_dlpack
from warp._src.dlpack import to_dlpack as to_dlpack


# category: JAX Interop

from warp._src.jax import from_jax as from_jax
from warp._src.jax import to_jax as to_jax
from warp._src.jax import dtype_from_jax as dtype_from_jax
from warp._src.jax import dtype_to_jax as dtype_to_jax
from warp._src.jax import device_from_jax as device_from_jax
from warp._src.jax import device_to_jax as device_to_jax


# category: PyTorch Interop

from warp._src.torch import from_torch as from_torch
from warp._src.torch import to_torch as to_torch
from warp._src.torch import dtype_from_torch as dtype_from_torch
from warp._src.torch import dtype_to_torch as dtype_to_torch
from warp._src.torch import device_from_torch as device_from_torch
from warp._src.torch import device_to_torch as device_to_torch
from warp._src.torch import stream_from_torch as stream_from_torch
from warp._src.torch import stream_to_torch as stream_to_torch


# category: Omniverse Runtime Fabric Interop

from warp._src.fabric import fabricarray as fabricarray
from warp._src.fabric import fabricarrayarray as fabricarrayarray
from warp._src.fabric import indexedfabricarray as indexedfabricarray
from warp._src.fabric import indexedfabricarrayarray as indexedfabricarrayarray


# category: Paddle Interop

from warp._src.paddle import from_paddle as from_paddle
from warp._src.paddle import to_paddle as to_paddle
from warp._src.paddle import dtype_from_paddle as dtype_from_paddle
from warp._src.paddle import dtype_to_paddle as dtype_to_paddle
from warp._src.paddle import device_from_paddle as device_from_paddle
from warp._src.paddle import device_to_paddle as device_to_paddle
from warp._src.paddle import stream_from_paddle as stream_from_paddle


# category: Data Types > Transformations

from warp._src.utils import transform_expand as transform_expand


# category: Data Types > Quaternions

from warp._src.utils import quat_between_vectors as quat_between_vectors


# category: Constants

from warp._src.constants import *


# category: Submodules

from . import config as config
from . import types as types
from . import utils as utils


# category: Misc

from warp._src.math import *
from warp._src.marching_cubes import MarchingCubes as MarchingCubes
from warp._src.context import RegisteredGLBuffer as RegisteredGLBuffer


__version__ = config.version


# TODO: Remove after cleaning up the public API.

from warp._src import types as _types


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    if name == "mat":
        return get_deprecated_api(_types, "warp", "matrix", old_attr_path="warp.mat")
    elif name == "vec":
        return get_deprecated_api(_types, "warp", "vector", old_attr_path="warp.vec")

    if name in (
        "build_dll",
        "build",
        "builtins",
        "codegen",
        "constants",
        "context",
        "dlpack",
        "fabric",
        "jax",
        "marching_cubes",
        "math",
        "paddle",
        "tape",
        "torch",
        "types",
        "utils",
    ):
        import importlib  # noqa: PLC0415

        return importlib.import_module(f".{name}", __package__)

    raise AttributeError(f"module 'warp' has no attribute '{name}'")
