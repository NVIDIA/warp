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

# for autocomplete on builtins
# from warp.stubs import *

from warp.types import array, array1d, array2d, array3d, array4d, constant, from_ptr
from warp.types import indexedarray, indexedarray1d, indexedarray2d, indexedarray3d, indexedarray4d
from warp.fabric import fabricarray, fabricarrayarray, indexedfabricarray, indexedfabricarrayarray

from warp.types import bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64
from warp.types import vec2, vec2b, vec2ub, vec2s, vec2us, vec2i, vec2ui, vec2l, vec2ul, vec2h, vec2f, vec2d
from warp.types import vec3, vec3b, vec3ub, vec3s, vec3us, vec3i, vec3ui, vec3l, vec3ul, vec3h, vec3f, vec3d
from warp.types import vec4, vec4b, vec4ub, vec4s, vec4us, vec4i, vec4ui, vec4l, vec4ul, vec4h, vec4f, vec4d
from warp.types import mat22, mat22h, mat22f, mat22d
from warp.types import mat33, mat33h, mat33f, mat33d
from warp.types import mat44, mat44h, mat44f, mat44d
from warp.types import quat, quath, quatf, quatd
from warp.types import transform, transformh, transformf, transformd
from warp.types import spatial_vector, spatial_vectorh, spatial_vectorf, spatial_vectord
from warp.types import spatial_matrix, spatial_matrixh, spatial_matrixf, spatial_matrixd

# annotation types
from warp.types import Int, Float, Scalar

# geometry types
from warp.types import Bvh, Mesh, HashGrid, Volume, MarchingCubes
from warp.types import BvhQuery, HashGridQuery, MeshQueryAABB, MeshQueryPoint, MeshQueryRay

# device-wide gemms
from warp.types import matmul, adj_matmul, batched_matmul, adj_batched_matmul

# discouraged, users should use wp.types.vector, wp.types.matrix
from warp.types import vector as vec
from warp.types import matrix as mat

# numpy interop
from warp.types import dtype_from_numpy, dtype_to_numpy

from warp.types import from_ipc_handle

from warp.context import init, func, func_grad, func_replay, func_native, kernel, struct, overload
from warp.context import is_cpu_available, is_cuda_available, is_device_available
from warp.context import get_devices, get_preferred_device
from warp.context import get_cuda_devices, get_cuda_device_count, get_cuda_device, map_cuda_device, unmap_cuda_device
from warp.context import get_device, set_device, synchronize_device
from warp.context import (
    zeros,
    zeros_like,
    ones,
    ones_like,
    full,
    full_like,
    clone,
    empty,
    empty_like,
    copy,
    from_numpy,
    launch,
    launch_tiled,
    synchronize,
    force_load,
    load_module,
    event_from_ipc_handle,
)
from warp.context import set_module_options, get_module_options, get_module
from warp.context import capture_begin, capture_end, capture_launch
from warp.context import Kernel, Function, Launch
from warp.context import Stream, get_stream, set_stream, wait_stream, synchronize_stream
from warp.context import Event, record_event, wait_event, synchronize_event, get_event_elapsed_time
from warp.context import RegisteredGLBuffer
from warp.context import is_mempool_supported, is_mempool_enabled, set_mempool_enabled
from warp.context import (
    set_mempool_release_threshold,
    get_mempool_release_threshold,
    get_mempool_used_mem_current,
    get_mempool_used_mem_high,
)
from warp.context import is_mempool_access_supported, is_mempool_access_enabled, set_mempool_access_enabled
from warp.context import is_peer_access_supported, is_peer_access_enabled, set_peer_access_enabled

from warp.tape import Tape
from warp.utils import ScopedTimer, ScopedDevice, ScopedStream
from warp.utils import ScopedMempool, ScopedMempoolAccess, ScopedPeerAccess
from warp.utils import ScopedCapture
from warp.utils import transform_expand, quat_between_vectors
from warp.utils import TimingResult, timing_begin, timing_end, timing_print
from warp.utils import (
    TIMING_KERNEL,
    TIMING_KERNEL_BUILTIN,
    TIMING_MEMCPY,
    TIMING_MEMSET,
    TIMING_GRAPH,
    TIMING_ALL,
)

from warp.torch import from_torch, to_torch
from warp.torch import dtype_from_torch, dtype_to_torch
from warp.torch import device_from_torch, device_to_torch
from warp.torch import stream_from_torch, stream_to_torch

from warp.jax import from_jax, to_jax
from warp.jax import dtype_from_jax, dtype_to_jax
from warp.jax import device_from_jax, device_to_jax

from warp.dlpack import from_dlpack, to_dlpack

from warp.paddle import from_paddle, to_paddle
from warp.paddle import dtype_from_paddle, dtype_to_paddle
from warp.paddle import device_from_paddle, device_to_paddle
from warp.paddle import stream_from_paddle

from warp.build import clear_kernel_cache
from warp.build import clear_lto_cache

from warp.constants import *

from . import builtins
from warp.builtins import static

from warp.math import *

import warp.config as config

__version__ = config.version
