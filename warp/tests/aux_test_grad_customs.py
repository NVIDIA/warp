# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""This file is used to test importing a user-defined function with a custom gradient"""

import warp as wp

wp.init()


@wp.func
def aux_custom_fn(x: float, y: float):
    return x * 3.0 + y / 3.0, y**2.5


@wp.func_grad(aux_custom_fn)
def aux_custom_fn_grad(x: float, y: float, adj_ret0: float, adj_ret1: float):
    wp.adjoint[x] += x * adj_ret0 * 42.0 + y * adj_ret1 * 10.0
    wp.adjoint[y] += y * adj_ret1 * 3.0
