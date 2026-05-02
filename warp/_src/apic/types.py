# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ctypes definitions matching apic_types.h."""

import ctypes

# Constants (must match apic_types.h)
APIC_MAX_DIMS = 4
APIC_LAUNCH_MAX_DIMS = 4

# Operation types
APIC_OP_KERNEL_LAUNCH = 1
APIC_OP_MEMCPY_H2D = 2
APIC_OP_MEMCPY_D2H = 3
APIC_OP_MEMCPY_D2D = 4
APIC_OP_MEMSET = 5
APIC_OP_ALLOC = 6


class APICLaunchParamRecord(ctypes.Structure):
    """One entry per kernel argument (88 bytes, packed). Matches apic_types.h.

    Arrays and scalars share the same 88-byte layout:

    - Arrays: ``(region_id, byte_offset)`` locates the array's data inside a
      captured memory region; ``shape`` / ``strides`` / ``ndim`` /
      ``element_size`` carry the per-launch ``array_t`` view metadata. Shape
      and strides are stored here because the captured region only holds the
      underlying data — the ``array_t`` descriptor is built fresh per launch
      from Python and isn't recoverable at replay time otherwise.
    - Scalars: ``is_array == 0``. ``byte_offset`` is the scalar size in bytes,
      and the scalar value itself is inlined into ``shape`` (first 32 B) and
      ``strides`` (next 32 B). Scalars larger than 64 B are rejected at
      capture time.
    """

    _pack_ = 1
    _fields_ = [
        ("is_array", ctypes.c_uint8),
        ("ndim", ctypes.c_uint8),
        ("param_index", ctypes.c_uint16),
        ("region_id", ctypes.c_int32),
        ("byte_offset", ctypes.c_uint64),
        ("shape", ctypes.c_int64 * APIC_MAX_DIMS),
        ("strides", ctypes.c_int64 * APIC_MAX_DIMS),
        ("element_size", ctypes.c_uint32),
        ("_pad1", ctypes.c_uint32),
    ]


class APICLaunchInfo(ctypes.Structure):
    """Launch info passed to wp_cpu_launch_kernel / wp_cuda_launch_kernel.

    Must match apic_types.h APICLaunchInfo (naturally aligned, not packed).
    """

    _fields_ = [
        ("kernel_key", ctypes.c_char_p),
        ("module_hash", ctypes.c_char_p),
        ("is_forward", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 7),
        ("params", ctypes.POINTER(APICLaunchParamRecord)),
        ("num_params", ctypes.c_int32),
    ]
