# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import warp as wp


@wp.kernel
def add_kernel(a: wp.array[wp.int32], b: wp.array[wp.int32], res: wp.array[wp.int32]):
    i = wp.tid()
    res[i] = a[i] + b[i]
