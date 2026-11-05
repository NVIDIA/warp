# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import warp as wp

from ..benchmarks_utils import LAUNCHES_PER_SAMPLE, CudaEventTimer, capture_launches, setup_once


def make_lu_kernel(n, dtype, mathdx):
    @wp.kernel(module="unique", enable_backward=False)
    def factor_kernel(
        matrices: wp.array3d[dtype],
        factors: wp.array3d[dtype],
        pivots: wp.array2d[wp.int32],
        status: wp.array2d[wp.int32],
    ):
        batch = wp.tid()
        a = wp.tile_load(matrices[batch], shape=(n, n))
        lu, p, info = wp.tile_lu(a)
        wp.tile_store(factors[batch], lu)
        wp.tile_store(pivots[batch], p)
        wp.tile_store(status[batch], info)

    wp.set_module_options({"enable_mathdx_solver": mathdx}, module=factor_kernel.module)
    return factor_kernel


class TileLU:
    """Time 256 batched LU launches, including input/output traffic, on each backend."""

    params = ([8, 32, 64], [1, 256, 4096], ["float32", "float64"], [False, True])
    param_names = ["tile_size", "batch_count", "precision", "mathdx"]
    timer = CudaEventTimer()

    @setup_once
    def setup(self, n, batches, precision, mathdx):
        wp.init()
        if not wp.is_cuda_available():
            raise NotImplementedError("Requires CUDA")
        if mathdx and not wp._src.context.runtime.core.wp_is_mathdx_enabled():
            raise NotImplementedError("Requires MathDx")
        self.device = wp.get_device("cuda:0")
        dtype = getattr(wp, precision)
        rng = np.random.default_rng(42)
        a = wp.array(rng.standard_normal((batches, n, n)), dtype=dtype, device=self.device)
        lu = wp.empty_like(a)
        p = wp.empty((batches, n), dtype=wp.int32, device=self.device)
        info = wp.empty((batches, 1), dtype=wp.int32, device=self.device)
        self.cmd = wp.launch_tiled(
            make_lu_kernel(n, dtype, mathdx),
            dim=batches,
            inputs=[a, lu, p, info],
            block_dim=128,
            device=self.device,
            record_cmd=True,
        )
        self.cmd.launch()
        wp.synchronize_device(self.device)
        self.graph = capture_launches(self.cmd.launch, LAUNCHES_PER_SAMPLE, self.device)

    def time_cuda(self, n, batches, precision, mathdx):
        wp.capture_launch(self.graph)
