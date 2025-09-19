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

# isort: skip_file

from warp.types import array as array
from warp.types import array1d as array1d
from warp.types import array2d as array2d
from warp.types import array3d as array3d
from warp.types import array4d as array4d
from warp.types import constant as constant
from warp.types import from_ptr as from_ptr
from warp.types import fixedarray as fixedarray
from warp.types import indexedarray as indexedarray
from warp.types import indexedarray1d as indexedarray1d
from warp.types import indexedarray2d as indexedarray2d
from warp.types import indexedarray3d as indexedarray3d
from warp.types import indexedarray4d as indexedarray4d
from warp.fabric import fabricarray as fabricarray
from warp.fabric import fabricarrayarray as fabricarrayarray
from warp.fabric import indexedfabricarray as indexedfabricarray
from warp.fabric import indexedfabricarrayarray as indexedfabricarrayarray
from warp.types import tile as tile

from warp.types import bool as bool
from warp.types import int8 as int8
from warp.types import uint8 as uint8
from warp.types import int16 as int16
from warp.types import uint16 as uint16
from warp.types import int32 as int32
from warp.types import uint32 as uint32
from warp.types import int64 as int64
from warp.types import uint64 as uint64
from warp.types import float16 as float16
from warp.types import float32 as float32
from warp.types import float64 as float64

from warp.types import vec2 as vec2
from warp.types import vec2b as vec2b
from warp.types import vec2ub as vec2ub
from warp.types import vec2s as vec2s
from warp.types import vec2us as vec2us
from warp.types import vec2i as vec2i
from warp.types import vec2ui as vec2ui
from warp.types import vec2l as vec2l
from warp.types import vec2ul as vec2ul
from warp.types import vec2h as vec2h
from warp.types import vec2f as vec2f
from warp.types import vec2d as vec2d

from warp.types import vec3 as vec3
from warp.types import vec3b as vec3b
from warp.types import vec3ub as vec3ub
from warp.types import vec3s as vec3s
from warp.types import vec3us as vec3us
from warp.types import vec3i as vec3i
from warp.types import vec3ui as vec3ui
from warp.types import vec3l as vec3l
from warp.types import vec3ul as vec3ul
from warp.types import vec3h as vec3h
from warp.types import vec3f as vec3f
from warp.types import vec3d as vec3d

from warp.types import vec4 as vec4
from warp.types import vec4b as vec4b
from warp.types import vec4ub as vec4ub
from warp.types import vec4s as vec4s
from warp.types import vec4us as vec4us
from warp.types import vec4i as vec4i
from warp.types import vec4ui as vec4ui
from warp.types import vec4l as vec4l
from warp.types import vec4ul as vec4ul
from warp.types import vec4h as vec4h
from warp.types import vec4f as vec4f
from warp.types import vec4d as vec4d

from warp.types import mat22 as mat22
from warp.types import mat22h as mat22h
from warp.types import mat22f as mat22f
from warp.types import mat22d as mat22d

from warp.types import mat33 as mat33
from warp.types import mat33h as mat33h
from warp.types import mat33f as mat33f
from warp.types import mat33d as mat33d

from warp.types import mat44 as mat44
from warp.types import mat44h as mat44h
from warp.types import mat44f as mat44f
from warp.types import mat44d as mat44d

from warp.types import quat as quat
from warp.types import quath as quath
from warp.types import quatf as quatf
from warp.types import quatd as quatd

from warp.types import transform as transform
from warp.types import transformh as transformh
from warp.types import transformf as transformf
from warp.types import transformd as transformd

from warp.types import spatial_vector as spatial_vector
from warp.types import spatial_vectorh as spatial_vectorh
from warp.types import spatial_vectorf as spatial_vectorf
from warp.types import spatial_vectord as spatial_vectord

from warp.types import spatial_matrix as spatial_matrix
from warp.types import spatial_matrixh as spatial_matrixh
from warp.types import spatial_matrixf as spatial_matrixf
from warp.types import spatial_matrixd as spatial_matrixd

# annotation types
from warp.types import Int as Int
from warp.types import Float as Float
from warp.types import Scalar as Scalar

# geometry types
from warp.types import Bvh as Bvh
from warp.types import Mesh as Mesh
from warp.types import HashGrid as HashGrid
from warp.types import Volume as Volume
from warp.types import BvhQuery as BvhQuery
from warp.types import HashGridQuery as HashGridQuery
from warp.types import MeshQueryAABB as MeshQueryAABB
from warp.types import MeshQueryPoint as MeshQueryPoint
from warp.types import MeshQueryRay as MeshQueryRay

# device-wide gemms
from warp.types import matmul as matmul
from warp.types import adj_matmul as adj_matmul
from warp.types import batched_matmul as batched_matmul
from warp.types import adj_batched_matmul as adj_batched_matmul

# discouraged, users should use wp.types.vector, wp.types.matrix
from warp.types import vector as vec
from warp.types import matrix as mat

# matrix construction
from warp.types import matrix_from_cols as matrix_from_cols
from warp.types import matrix_from_rows as matrix_from_rows

# numpy interop
from warp.types import dtype_from_numpy as dtype_from_numpy
from warp.types import dtype_to_numpy as dtype_to_numpy

# ipc interop
from warp.types import from_ipc_handle as from_ipc_handle

from warp.context import init as init
from warp.context import func as func
from warp.context import func_grad as func_grad
from warp.context import func_replay as func_replay
from warp.context import func_native as func_native
from warp.context import kernel as kernel
from warp.context import struct as struct
from warp.context import overload as overload

from warp.context import is_cpu_available as is_cpu_available
from warp.context import is_cuda_available as is_cuda_available
from warp.context import is_device_available as is_device_available
from warp.context import get_devices as get_devices
from warp.context import get_preferred_device as get_preferred_device
from warp.context import get_cuda_devices as get_cuda_devices
from warp.context import get_cuda_device_count as get_cuda_device_count
from warp.context import get_cuda_device as get_cuda_device
from warp.context import map_cuda_device as map_cuda_device
from warp.context import unmap_cuda_device as unmap_cuda_device
from warp.context import get_device as get_device
from warp.context import set_device as set_device
from warp.context import synchronize_device as synchronize_device

# tensor creation
from warp.context import zeros as zeros
from warp.context import zeros_like as zeros_like
from warp.context import ones as ones
from warp.context import ones_like as ones_like
from warp.context import full as full
from warp.context import full_like as full_like
from warp.context import clone as clone
from warp.context import empty as empty
from warp.context import empty_like as empty_like
from warp.context import copy as copy
from warp.context import from_numpy as from_numpy

from warp.context import launch as launch
from warp.context import launch_tiled as launch_tiled
from warp.context import synchronize as synchronize
from warp.context import compile_aot_module as compile_aot_module
from warp.context import force_load as force_load
from warp.context import load_module as load_module
from warp.context import load_aot_module as load_aot_module
from warp.context import event_from_ipc_handle as event_from_ipc_handle

from warp.context import set_module_options as set_module_options
from warp.context import get_module_options as get_module_options
from warp.context import get_module as get_module

from warp.context import capture_begin as capture_begin
from warp.context import capture_end as capture_end
from warp.context import capture_launch as capture_launch
from warp.context import capture_if as capture_if
from warp.context import capture_while as capture_while
from warp.context import capture_debug_dot_print as capture_debug_dot_print

from warp.context import Kernel as Kernel
from warp.context import Function as Function
from warp.context import Launch as Launch

from warp.context import Stream as Stream
from warp.context import get_stream as get_stream
from warp.context import set_stream as set_stream
from warp.context import wait_stream as wait_stream
from warp.context import synchronize_stream as synchronize_stream

from warp.context import Event as Event
from warp.context import record_event as record_event
from warp.context import wait_event as wait_event
from warp.context import synchronize_event as synchronize_event
from warp.context import get_event_elapsed_time as get_event_elapsed_time

from warp.context import RegisteredGLBuffer as RegisteredGLBuffer

from warp.context import is_mempool_supported as is_mempool_supported
from warp.context import is_mempool_enabled as is_mempool_enabled
from warp.context import set_mempool_enabled as set_mempool_enabled

from warp.context import set_mempool_release_threshold as set_mempool_release_threshold
from warp.context import get_mempool_release_threshold as get_mempool_release_threshold
from warp.context import get_mempool_used_mem_current as get_mempool_used_mem_current
from warp.context import get_mempool_used_mem_high as get_mempool_used_mem_high

from warp.context import is_mempool_access_supported as is_mempool_access_supported
from warp.context import is_mempool_access_enabled as is_mempool_access_enabled
from warp.context import set_mempool_access_enabled as set_mempool_access_enabled

from warp.context import is_peer_access_supported as is_peer_access_supported
from warp.context import is_peer_access_enabled as is_peer_access_enabled
from warp.context import set_peer_access_enabled as set_peer_access_enabled

from warp.tape import Tape as Tape

from warp.utils import ScopedTimer as ScopedTimer
from warp.utils import ScopedDevice as ScopedDevice
from warp.utils import ScopedStream as ScopedStream
from warp.utils import ScopedMempool as ScopedMempool
from warp.utils import ScopedMempoolAccess as ScopedMempoolAccess
from warp.utils import ScopedPeerAccess as ScopedPeerAccess
from warp.utils import ScopedCapture as ScopedCapture

from warp.utils import transform_expand as transform_expand
from warp.utils import quat_between_vectors as quat_between_vectors

from warp.utils import TimingResult as TimingResult
from warp.utils import timing_begin as timing_begin
from warp.utils import timing_end as timing_end
from warp.utils import timing_print as timing_print

from warp.utils import TIMING_KERNEL as TIMING_KERNEL
from warp.utils import TIMING_KERNEL_BUILTIN as TIMING_KERNEL_BUILTIN
from warp.utils import TIMING_MEMCPY as TIMING_MEMCPY
from warp.utils import TIMING_MEMSET as TIMING_MEMSET
from warp.utils import TIMING_GRAPH as TIMING_GRAPH
from warp.utils import TIMING_ALL as TIMING_ALL

from warp.utils import map as map

from warp.marching_cubes import MarchingCubes as MarchingCubes

from warp.torch import from_torch as from_torch
from warp.torch import to_torch as to_torch
from warp.torch import dtype_from_torch as dtype_from_torch
from warp.torch import dtype_to_torch as dtype_to_torch
from warp.torch import device_from_torch as device_from_torch
from warp.torch import device_to_torch as device_to_torch
from warp.torch import stream_from_torch as stream_from_torch
from warp.torch import stream_to_torch as stream_to_torch

from warp.jax import from_jax as from_jax
from warp.jax import to_jax as to_jax
from warp.jax import dtype_from_jax as dtype_from_jax
from warp.jax import dtype_to_jax as dtype_to_jax
from warp.jax import device_from_jax as device_from_jax
from warp.jax import device_to_jax as device_to_jax

from warp.dlpack import from_dlpack as from_dlpack
from warp.dlpack import to_dlpack as to_dlpack

from warp.paddle import from_paddle as from_paddle
from warp.paddle import to_paddle as to_paddle
from warp.paddle import dtype_from_paddle as dtype_from_paddle
from warp.paddle import dtype_to_paddle as dtype_to_paddle
from warp.paddle import device_from_paddle as device_from_paddle
from warp.paddle import device_to_paddle as device_to_paddle
from warp.paddle import stream_from_paddle as stream_from_paddle

from warp.build import clear_kernel_cache as clear_kernel_cache
from warp.build import clear_lto_cache as clear_lto_cache

from warp.constants import *

from . import builtins
from warp.builtins import static as static

from warp.math import *

from . import config as config
from . import types as types

__version__ = config.version
