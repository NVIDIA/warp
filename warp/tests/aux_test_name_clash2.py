# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp


# test clashes with identical struct from another module
@wp.struct
class SameStruct:
    x: float


# test clashes with identically named but different struct from another module
@wp.struct
class DifferentStruct:
    v: wp.vec2


# test clashes with identical function from another module
@wp.func
def same_func():
    return 99


# test clashes with identically named but different function from another module
@wp.func
def different_func():
    return 42
