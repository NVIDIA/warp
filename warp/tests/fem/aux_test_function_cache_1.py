# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provide same-named Warp functions for FEM evaluator-cache identity tests.

This module is paired with ``aux_test_function_cache_2``. Matching function
names intentionally return different values so tests detect evaluator reuse
across distinct Warp function identities.
"""

import warp as wp


# Unique modules give the paired implicit functions matching Warp keys and
# module names but distinct native identities.
@wp.func(module="unique")
def implicit_func(x: wp.vec2):
    return 1.0


@wp.func(module="unique")
def implicit_grad_func(x: wp.vec2):
    return wp.vec2(3.0, 4.0)


@wp.func(module="unique")
def implicit_div_func(x: wp.vec2):
    return 5.0


# Distinct explicit shared modules exercise named-module identity separately
# from the unique-module case above.
@wp.func(module="warp.tests.fem.function.cache.module")
def point_kernel_func(squared_dist: float, point_index: int):
    return 1.0


@wp.func(module="warp.tests.fem.function.cache.module")
def point_kernel_grad_func(squared_dist: float, point_index: int):
    return 0.0
