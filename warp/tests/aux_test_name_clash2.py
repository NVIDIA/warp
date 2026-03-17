# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
