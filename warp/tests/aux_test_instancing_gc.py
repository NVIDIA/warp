# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Helper module used in test_codegen_instancing.py"""

import warp as wp


def create_kernel_closure(value: int):
    @wp.kernel
    def k(a: wp.array(dtype=int)):
        a[0] = value

    return k
