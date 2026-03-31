# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper module used in test_codegen_instancing.py"""

import warp as wp


def create_kernel_closure(value: int):
    @wp.kernel
    def k(a: wp.array(dtype=int)):
        a[0] = value

    return k
