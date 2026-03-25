# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import warp as wp


@wp.func
def multiply(x: float):
    return x * x


@wp.kernel
def kern(expect: float):
    wp.expect_eq(multiply(4.0), expect)


def run(expect, device):
    wp.launch(kern, dim=1, inputs=[expect], device=device)
    wp.synchronize_device(device)
