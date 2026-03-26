# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import warp as wp


@wp.kernel
def scale(x: wp.array(dtype=Any), s: Any):
    i = wp.tid()
    x[i] = s * x[i]


# Add exactly one overload
scale_f32 = wp.overload(scale, [wp.array(dtype=wp.float32), wp.float32])
