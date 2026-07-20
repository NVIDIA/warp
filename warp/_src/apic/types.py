# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ctypes definitions matching apic_types.h."""

import ctypes

# Constants (must match apic_types.h)
APIC_MAX_DIMS = 4
APIC_LAUNCH_MAX_DIMS = 4

# Operation types (must match APICOpType in apic_types.h).
APIC_OP_KERNEL_LAUNCH = 1
APIC_OP_MEMCPY_H2D = 2
APIC_OP_MEMCPY_D2H = 3
APIC_OP_MEMCPY_D2D = 4
APIC_OP_MEMSET = 5
APIC_OP_ALLOC = 6
APIC_OP_IF = 7
APIC_OP_WHILE = 8
APIC_OP_SCAN = 9
APIC_OP_MEMTILE = 10
APIC_OP_SEGMENTED_SORT = 11
APIC_OP_RADIX_SORT = 12
APIC_OP_RUNLENGTH_ENCODE = 13
APIC_OP_BSR_FROM_TRIPLETS = 14
APIC_OP_BSR_TRANSPOSE = 15
APIC_OP_REDUCTION = 16


# Scalar element types (must match APICType in apic_types.h).
APIC_TYPE_INT32 = 1
APIC_TYPE_UINT32 = 2
APIC_TYPE_INT64 = 3
APIC_TYPE_UINT64 = 4
APIC_TYPE_FLOAT32 = 5
APIC_TYPE_FLOAT64 = 6


# Per-value-blob relocation kinds (must match APICRelocKind in apic_types.h).
APIC_RELOC_DATA_PTR = 1  # Region pointer: write resolve_ptr(region_id, region_offset)
APIC_RELOC_HANDLE = 2  # wp.handle / mesh id: write handle_ptr_remap[region_offset]
APIC_RELOC_NULL = 3  # Explicit zero (null array data/grad, absent indexedarray dim)


class APICLaunchParamRecord(ctypes.Structure):
    """One entry per kernel argument (16 bytes, packed). Matches apic_types.h.

    Describes a slice ``[value_offset, value_offset + value_size)`` of the
    per-launch ``value_data`` section, plus the count of relocations that
    patch pointer fields inside the slice. Every Warp argument kind
    (scalars, vec/mat, ``@wp.struct``, ``wp.array``, ``wp.indexedarray``,
    ``wp.handle``) takes this same form.
    """

    _pack_ = 1
    _fields_ = [
        ("param_index", ctypes.c_uint16),
        ("num_relocs", ctypes.c_uint16),
        ("value_offset", ctypes.c_uint32),
        ("value_size", ctypes.c_uint32),
        ("value_align", ctypes.c_uint32),
    ]


class APICLaunchPtrLocation(ctypes.Structure):
    """One relocation entry (24 bytes, packed). Matches apic_types.h.

    Patches one 8-byte slot at ``value_byte_offset`` inside a kernel-launch
    parameter's value blob at replay time. ``kind`` selects how the patched
    pointer is computed:

    - ``APIC_RELOC_DATA_PTR``: ``resolve_ptr(region_id, region_offset)``.
    - ``APIC_RELOC_HANDLE``: ``handle_ptr_remap[region_offset]`` (``region_offset``
      carries the original captured handle id; ``region_id`` is unused / -1).
    - ``APIC_RELOC_NULL``: literal zero (null array.data, absent
      indexedarray dim).
    """

    _pack_ = 1
    _fields_ = [
        ("value_byte_offset", ctypes.c_uint32),
        ("region_id", ctypes.c_int32),
        ("region_offset", ctypes.c_uint64),
        ("kind", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 7),
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
        ("kernel_dim", ctypes.c_int32),
        ("value_data_size", ctypes.c_uint32),
        ("value_data", ctypes.POINTER(ctypes.c_uint8)),
        ("adj_params", ctypes.POINTER(APICLaunchParamRecord)),
        ("num_relocs", ctypes.c_uint32),
        ("_pad2", ctypes.c_uint32),
        ("relocs", ctypes.POINTER(APICLaunchPtrLocation)),
    ]
