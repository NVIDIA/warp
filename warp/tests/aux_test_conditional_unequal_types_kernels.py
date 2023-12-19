# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""This file defines a kernel that fails on codegen.py"""

import warp as wp


@wp.kernel
def unequal_types_kernel():
    x = wp.int32(10)
    y = 10
    z = True

    # Throws a TypeError
    if x == y == z:
        pass
