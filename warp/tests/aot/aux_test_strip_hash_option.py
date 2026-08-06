# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp


@wp.kernel
def add_one(x: wp.array[wp.int32]):
    i = wp.tid()
    x[i] = x[i] + 1
