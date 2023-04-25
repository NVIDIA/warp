# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys

if hasattr(sys.stderr, "fileno"):
    import faulthandler

    faulthandler.enable()

# for autocomplete on builtins
# from warp.stubs import *

from warp.types import array, array1d, array2d, array3d, array4d, constant
from warp.types import indexedarray, indexedarray1d, indexedarray2d, indexedarray3d, indexedarray4d

from warp.types import int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64
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

# geometry types
from warp.types import Bvh, Mesh, HashGrid, Volume, MarchingCubes
from warp.types import bvh_query_t, mesh_query_aabb_t, hash_grid_query_t

# device-wide gemms
from warp.types import matmul, adj_matmul, batched_matmul, adj_batched_matmul, from_ptr

# deprecated
from warp.types import vector as vec
from warp.types import matrix as mat

from warp.context import init, func, kernel, struct, overload
from warp.context import is_cpu_available, is_cuda_available, is_device_available
from warp.context import get_devices, get_preferred_device
from warp.context import get_cuda_devices, get_cuda_device_count, get_cuda_device, map_cuda_device, unmap_cuda_device
from warp.context import get_device, set_device, synchronize_device
from warp.context import (
    zeros,
    zeros_like,
    clone,
    empty,
    empty_like,
    copy,
    from_numpy,
    launch,
    synchronize,
    force_load,
    load_module,
)
from warp.context import set_module_options, get_module_options, get_module
from warp.context import capture_begin, capture_end, capture_launch
from warp.context import print_builtins, export_builtins, export_stubs
from warp.context import Kernel, Function
from warp.context import Stream, get_stream, set_stream, synchronize_stream
from warp.context import Event, record_event, wait_event, wait_stream

from warp.tape import Tape
from warp.utils import ScopedTimer, ScopedCudaGuard, ScopedDevice, ScopedStream
from warp.utils import transform_expand

from warp.torch import from_torch, to_torch
from warp.torch import device_from_torch, device_to_torch
from warp.torch import stream_from_torch, stream_to_torch

from warp.jax import from_jax, to_jax
from warp.jax import device_from_jax, device_to_jax

from warp.dlpack import from_dlpack, to_dlpack

from warp.constants import *

from . import builtins, render
