# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import warp as wp
from warp._src.context import builtin_functions

from ..benchmarks_utils import LAUNCHES_PER_SAMPLE, CudaEventTimer, capture_launches, setup_once


def make_factor_kernel(n, dtype, algorithm):
    name = "tile_ldlt" if algorithm == "ldlt" else "tile_lu"
    if name not in builtin_functions:
        raise NotImplementedError(f"Requires wp.{name}")
    factorize = builtin_functions[name]

    @wp.kernel(module="unique", enable_backward=False)
    def factor_kernel(
        matrices: wp.array3d[dtype], factors: wp.array3d[dtype], pivots: wp.array2d[int], status: wp.array2d[int]
    ):
        batch = wp.tid()
        packed, p, info = factorize(wp.tile_load(matrices[batch], shape=(n, n)))
        wp.tile_store(factors[batch], packed)
        wp.tile_store(pivots[batch], p)
        wp.tile_store(status[batch], info)

    wp.set_module_options(
        {"enable_backward": False, "enable_mathdx_solver": algorithm == "lu_mathdx"}, module=factor_kernel.module
    )
    return factor_kernel


def symmetric_indefinite_matrices(n, batches, family):
    rng = np.random.default_rng(47 + n + batches)
    if family == "random":
        q, _ = np.linalg.qr(rng.standard_normal((batches, n, n)))
        spectrum = np.linspace(0.5, 2, n) * np.where(np.arange(n) % 2, 1, -1)
        matrices = (q * spectrum) @ q.transpose(0, 2, 1)
    elif family == "diagonal":
        random = rng.standard_normal((batches, n, n)) / n
        matrices = (random + random.transpose(0, 2, 1)) * 0.5
        matrices += np.diag(np.where(np.arange(n) % 2, 2, -2))
    elif family == "kkt":
        primal = n // 2
        random = rng.standard_normal((batches, primal, primal))
        matrices = np.zeros((batches, n, n))
        matrices[:, :primal, :primal] = random @ random.transpose(0, 2, 1) / primal + np.eye(primal)
        constraints = rng.standard_normal((batches, n - primal, primal)) / np.sqrt(primal)
        matrices[:, primal:, :primal] = constraints
        matrices[:, :primal, primal:] = constraints.transpose(0, 2, 1)
        matrices[:, primal:, primal:] = -0.1 * np.eye(n - primal)
    else:
        raise ValueError(f"Unknown matrix family {family!r}")
    return (matrices + matrices.transpose(0, 2, 1)) * 0.5


class TileLDLT:
    """Compare indefinite factorizations with 256-launch CUDA graphs."""

    params = (
        [8, 32, 64],
        [1, 256, 4096],
        ["float32", "float64"],
        [32, 128, 256],
        ["random", "kkt", "diagonal"],
        ["ldlt", "lu_native", "lu_mathdx"],
    )
    param_names = ["tile_size", "batch_count", "precision", "block_dim", "matrix_family", "algorithm"]
    timer = CudaEventTimer()

    @setup_once
    def setup(self, n, batches, precision, block_dim, family, algorithm):
        wp.init()
        if not wp.is_cuda_available():
            raise NotImplementedError("Requires CUDA")
        if algorithm == "lu_mathdx" and not wp._src.context.runtime.core.wp_is_mathdx_enabled():
            raise NotImplementedError("Requires MathDx")
        dtype = getattr(wp, precision)
        kernel = make_factor_kernel(n, dtype, algorithm)
        self.device = wp.get_device("cuda:0")
        self.matrices = wp.array(symmetric_indefinite_matrices(n, batches, family), dtype=dtype, device=self.device)
        self.factors = wp.empty_like(self.matrices)
        self.pivots = wp.empty((batches, n), dtype=wp.int32, device=self.device)
        self.status = wp.empty((batches, 1), dtype=wp.int32, device=self.device)
        self.command = wp.launch_tiled(
            kernel,
            dim=batches,
            inputs=[self.matrices, self.factors, self.pivots, self.status],
            block_dim=block_dim,
            device=self.device,
            record_cmd=True,
        )
        self.command.launch()
        self.graph = capture_launches(self.command.launch, LAUNCHES_PER_SAMPLE, self.device)

    def time_cuda(self, n, batches, precision, block_dim, family, algorithm):
        wp.capture_launch(self.graph)
