# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the dot products and solves of ``warp.optim.linear``."""

import numpy as np

import warp as wp
from warp._src.optim.linear import TiledDot
from warp.optim.linear import cg, preconditioner
from warp.sparse import bsr_from_triplets

from ..benchmarks_utils import setup_once

DOT_LENGTH = 1 << 20

LAPLACIAN_RES = 64
CG_ITERATIONS = 100
# Small enough that every solve runs all CG_ITERATIONS, so the timing doesn't
# depend on when the solver detects convergence.
CG_TOLERANCE = 1.0e-30


def _get_device(device_name):
    if device_name == "cuda" and not wp.is_cuda_available():
        raise NotImplementedError("CUDA is not available")
    return wp.get_device(device_name)


def _laplacian_2d(res, device):
    """Return the 5-point Laplacian of a ``res`` x ``res`` grid as a BSR matrix."""
    n = res * res
    nodes = np.arange(n).reshape(res, res)
    rows = [nodes.ravel()]
    cols = [nodes.ravel()]
    values = [np.full(n, 4.0, dtype=np.float32)]
    for a, b in ((nodes[1:, :], nodes[:-1, :]), (nodes[:, 1:], nodes[:, :-1])):
        for row, col in ((a, b), (b, a)):
            rows.append(row.ravel())
            cols.append(col.ravel())
            values.append(np.full(row.size, -1.0, dtype=np.float32))

    return bsr_from_triplets(
        n,
        n,
        wp.array(np.concatenate(rows), dtype=int, device=device),
        wp.array(np.concatenate(cols), dtype=int, device=device),
        wp.array(np.concatenate(values), dtype=wp.float32, device=device),
    )


class LinearSolverDot:
    """Measure the non-batched dot product that the linear solvers use."""

    params = ["cpu", "cuda"]
    param_names = ["device"]

    # A call takes microseconds on CUDA and up to a second on CPU, so let ASV
    # pick the number of calls per sample.
    repeat = 10

    @setup_once
    def setup(self, device_name):
        wp.init()
        self.device = _get_device(device_name)

        values = np.random.default_rng(42).standard_normal(DOT_LENGTH).astype(np.float32)
        self.a = wp.array(values, device=self.device)
        self.dot = TiledDot(DOT_LENGTH, wp.float32, device=self.device)

        self.dot.compute(self.a, self.a)
        expected = np.dot(values.astype(np.float64), values.astype(np.float64))
        np.testing.assert_allclose(self.dot.col().numpy()[0], expected, rtol=1.0e-4)

    def time_compute(self, device_name):
        self.dot.compute(self.a, self.a)
        wp.synchronize_device(self.device)


class LinearSolverCg:
    """Measure a fixed number of Jacobi-preconditioned CG iterations on a 2D Laplacian.

    ``cpu`` and ``cuda`` use the default ``use_cuda_graph``, and ``cpu_graph``
    passes ``use_cuda_graph=True`` on CPU.
    """

    params = ["cpu", "cpu_graph", "cuda"]
    param_names = ["mode"]

    # A solve takes milliseconds on CUDA and up to a second on CPU, so let ASV
    # pick the number of calls per sample.
    repeat = 10

    @setup_once
    def setup(self, mode):
        wp.init()
        self.device = _get_device("cuda" if mode == "cuda" else "cpu")
        self.use_cuda_graph = True if mode == "cpu_graph" else None

        self.A = _laplacian_2d(LAPLACIAN_RES, self.device)
        self.M = preconditioner(self.A, "diag")
        rhs = np.random.default_rng(42).standard_normal(self.A.shape[0]).astype(np.float32)
        self.b = wp.array(rhs, device=self.device)
        self.x = wp.zeros_like(self.b)

        iterations, residual_norm, _ = self._solve()
        if iterations != CG_ITERATIONS or not residual_norm < 1.0e-2 * np.linalg.norm(rhs):
            raise RuntimeError(f"unexpected CG result: {iterations} iterations, residual norm {residual_norm}")

    def _solve(self):
        self.x.zero_()
        kwargs = {} if self.use_cuda_graph is None else {"use_cuda_graph": self.use_cuda_graph}
        return cg(
            self.A,
            self.b,
            self.x,
            tol=CG_TOLERANCE,
            atol=CG_TOLERANCE,
            maxiter=CG_ITERATIONS,
            M=self.M,
            check_every=10,
            **kwargs,
        )

    def time_solve(self, mode):
        self._solve()
        wp.synchronize_device(self.device)
