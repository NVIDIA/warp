# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import warp as wp

from ..benchmarks_utils import LAUNCHES_PER_SAMPLE, CudaEventTimer, capture_launches, setup_once


def make_cholesky_kernel(n, dtype, mathdx, upper=False, inplace=False):
    @wp.kernel(module="unique", enable_backward=False)
    def factor_kernel(matrices: wp.array3d[dtype], factors: wp.array3d[dtype]):
        batch = wp.tid()
        matrix = wp.tile_load(matrices[batch], shape=(n, n))
        if wp.static(inplace):
            if wp.static(upper):
                wp.tile_cholesky_inplace(matrix, fill_mode="upper")
            else:
                wp.tile_cholesky_inplace(matrix)
            wp.tile_store(factors[batch], matrix)
        else:
            if wp.static(upper):
                factor = wp.tile_cholesky(matrix, fill_mode="upper")
            else:
                factor = wp.tile_cholesky(matrix)
            wp.tile_store(factors[batch], factor)

    wp.set_module_options({"enable_backward": False, "enable_mathdx_solver": mathdx}, module=factor_kernel.module)
    return factor_kernel


class TileCholesky:
    """Time 256 batched Cholesky launches for representative workloads across precisions and backends.

    Workloads are ``(tile_size, batch_count, block_dim)`` and cover the shared-diagonal path, single-warp
    execution, and the block-size boundary.
    """

    params = ([(8, 1, 32), (8, 4096, 128), (32, 256, 256), (64, 1, 64)], ["float32", "float64"], [False, True])
    param_names = ["workload", "precision", "mathdx"]
    timer = CudaEventTimer()

    @setup_once
    def setup(self, workload, precision, mathdx):
        n, batches, block_dim = workload
        wp.init()
        if not wp.is_cuda_available():
            raise NotImplementedError("Requires CUDA")
        if mathdx and not wp._src.context.runtime.core.wp_is_mathdx_enabled():
            raise NotImplementedError("Requires MathDx")
        self.device = wp.get_device("cuda:0")
        dtype = getattr(wp, precision)
        rng = np.random.default_rng(42)
        normal = rng.standard_normal((batches, n, n))
        matrices = wp.array(normal @ normal.transpose(0, 2, 1) / n + 0.1 * np.eye(n), dtype=dtype, device=self.device)
        factors = wp.empty_like(matrices)
        self.command = wp.launch_tiled(
            make_cholesky_kernel(n, dtype, mathdx),
            dim=batches,
            inputs=[matrices, factors],
            block_dim=block_dim,
            device=self.device,
            record_cmd=True,
        )
        self.command.launch()
        self.graph = capture_launches(self.command.launch, LAUNCHES_PER_SAMPLE, self.device)

    def time_cuda(self, workload, precision, mathdx):
        wp.capture_launch(self.graph)
