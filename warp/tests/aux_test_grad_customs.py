# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""This file is used to test importing a user-defined function with a custom gradient"""

import warp as wp


@wp.func
def aux_custom_fn(x: float, y: float):
    return x * 3.0 + y / 3.0, y**2.5


@wp.func_grad(aux_custom_fn)
def aux_custom_fn_grad(x: float, y: float, adj_ret0: float, adj_ret1: float):
    wp.adjoint[x] += x * adj_ret0 * 42.0 + y * adj_ret1 * 10.0
    wp.adjoint[y] += y * adj_ret1 * 3.0
