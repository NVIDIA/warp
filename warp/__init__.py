# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# for autocomplete on builtins
#from warp.stubs import *

from warp.types import array, array2d, array3d, array4d, constant
from warp.types import int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64
from warp.types import vec2, vec3, vec4, mat22, mat33, mat44, quat, transform, spatial_vector, spatial_matrix
from warp.types import Bvh, Mesh, HashGrid, Volume, MarchingCubes
from warp.types import bvh_query_t, mesh_query_aabb_t, hash_grid_query_t

from warp.context import init, func, kernel, struct
from warp.context import is_cpu_available, is_cuda_available, is_device_available
from warp.context import get_devices, get_preferred_device
from warp.context import get_cuda_devices, get_cuda_device_count, get_cuda_device, map_cuda_device, unmap_cuda_device
from warp.context import get_device, set_device, synchronize_device
from warp.context import zeros, zeros_like, clone, empty, empty_like, copy, from_numpy, launch, synchronize, force_load
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

from . import builtins, render
