# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import warp as wp


@wp.kernel
def add_kernel(a: wp.array[float], b: wp.array[float], out: wp.array[float]):
    tid = wp.tid()
    out[tid] = a[tid] + b[tid]


with wp.ScopedDevice("cpu"):
    n = 1024
    a = wp.array(np.arange(n, dtype=np.float32))
    b = wp.array(np.ones(n, dtype=np.float32))
    out = wp.zeros(n, dtype=float)
    wp.launch(add_kernel, dim=n, inputs=[a, b], outputs=[out])
    np.testing.assert_allclose(out.numpy(), np.arange(n, dtype=np.float32) + 1.0)
print("CPU JIT smoke test passed")
