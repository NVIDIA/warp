# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
