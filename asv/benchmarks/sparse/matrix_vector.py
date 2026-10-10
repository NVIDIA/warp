# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import warp as wp
import warp.fem as fem
import warp.sparse as wps
from warp.examples.fem.utils import gen_tetmesh

from ..benchmarks_utils import setup_once


@fem.integrand
def diffusion_form_scalar(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def diffusion_form_vector(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.ddot(fem.D(u, s), fem.D(v, s))


class BsrMvFemMatrix:
    """Utility base class for building FEM matrices to test BSR matrix-vector multiplication."""

    def __init__(self, use_graph: bool = True):
        self._use_graph = use_graph

    def build_system(self, space: fem.FunctionSpace, integrand: fem.Integrand):
        u = fem.make_trial(space)
        v = fem.make_test(space)

        self._mat = fem.integrate(integrand, fields={"u": u, "v": v}, output_dtype=float)
        self._vec = wp.ones(shape=self._mat.shape[0], dtype=wp.float32)
        self._res = wp.zeros(shape=self._mat.shape[0], dtype=wp.float32)

        self._mat.nnz_sync()

        self._run_impl()

        if self._use_graph:
            with wp.ScopedCapture() as capture:
                self._run_impl()
            self._graph = capture.graph

        wp.synchronize_device(self.device)

    def _run_impl(self):
        wps.bsr_mv(self._mat, self._vec, self._res, alpha=1.0, beta=1.0)
        wps.bsr_mv(self._mat, self._vec, self._res, alpha=1.0, beta=1.0, transpose=True)

    def run(self):
        if self._use_graph:
            wp.capture_launch(self._graph)
        else:
            self._run_impl()
        wp.synchronize_device(self.device)


class BsrMvQuadraticTetmeshMatrix(BsrMvFemMatrix):
    """Test BSR matrix-vector multiplication with quadratic tetrahedral elements."""

    rounds = 1
    repeat = 2
    number = 10  # Number of timed calls per sample

    @setup_once
    def setup(self):
        wp.init()
        self.device = wp.get_device("cuda:0")

        res = 32
        with wp.ScopedDevice(self.device):
            pos, cells = gen_tetmesh(res=(res, res, res))
            geo = fem.Tetmesh(cells, pos)
            space = fem.make_polynomial_space(geo, degree=2, dtype=wp.vec3)
            self.build_system(space, diffusion_form_vector)

    def time_cuda(self):
        self.run()


class BsrMvLinearGridMatrix(BsrMvFemMatrix):
    """Test BSR matrix-vector multiplication with linear grid elements."""

    rounds = 1
    repeat = 2
    number = 10  # Number of timed calls per sample

    @setup_once
    def setup(self):
        wp.init()
        self.device = wp.get_device("cuda:0")

        res = 64
        with wp.ScopedDevice(self.device):
            geo = fem.Grid3D(res=(res, res, res))
            space = fem.make_polynomial_space(geo)
            self.build_system(space, diffusion_form_scalar)

    def time_cuda(self):
        self.run()


class BsrMvAlmostDense(BsrMvFemMatrix):
    """Test BSR matrix-vector multiplication with almost dense matrices (high-order elements)."""

    rounds = 1
    repeat = 2
    number = 10  # Number of timed calls per sample

    @setup_once
    def setup(self):
        wp.init()
        self.device = wp.get_device("cuda:0")

        res = 2
        with wp.ScopedDevice(self.device):
            geo = fem.Grid3D(res=(res, res, res))
            space = fem.make_polynomial_space(geo, degree=4, dtype=wp.vec3)
            self.build_system(space, diffusion_form_vector)

    def time_cuda(self):
        self.run()


class BsrMvMixedPrecision:
    """Compare compressed matrix storage with preconverted float64 storage."""

    params = [["float32", "float64"], [8, 32, 128], [-1, 0, 64], [False, True]]
    param_names = ["matrix_dtype", "blocks_per_row", "tile_size", "transpose"]
    rounds = 3
    repeat = 5
    number = 10

    def setup(self, matrix_dtype, blocks_per_row, tile_size, transpose):
        """Prepare a captured product for the requested storage type and dispatch."""
        self._setup(matrix_dtype, blocks_per_row, tile_size, transpose)

    def _setup(self, matrix_dtype, blocks_per_row, tile_size, transpose, rows_per_block=1):
        """Build reproducible coefficients and capture a warmed-up CUDA product."""
        wp.init()
        self.device = wp.get_device("cuda:0")
        nrow = 32768
        rng = np.random.default_rng(2069)
        with wp.ScopedDevice(self.device):
            A = wps.bsr_zeros(nrow, nrow, getattr(wp, matrix_dtype))
            A.nnz = nrow * blocks_per_row
            A.offsets = wp.array(np.arange(nrow + 1, dtype=np.int32) * blocks_per_row)
            A.columns = wp.array(
                ((np.arange(nrow)[:, None] * 17 + np.arange(blocks_per_row) * 13) % nrow).astype(np.int32).ravel()
            )
            # Both storage types represent exactly the same coefficients.
            A.values = wp.array(rng.uniform(-1, 1, size=A.nnz).astype(np.float32), dtype=getattr(wp, matrix_dtype))
            x = wp.array(rng.uniform(-1, 1, size=nrow), dtype=wp.float64)
            y = wp.empty_like(x)
            wps.bsr_mv(A, x, y, tile_size=tile_size, transpose=transpose, rows_per_block=rows_per_block)
            with wp.ScopedCapture() as capture:
                wps.bsr_mv(A, x, y, tile_size=tile_size, transpose=transpose, rows_per_block=rows_per_block)
            self._graph = capture.graph
            self._arrays = A, x, y
        wp.synchronize_device(self.device)

    def time_cuda(self, matrix_dtype, blocks_per_row, tile_size, transpose):
        """Time one captured mixed-precision product through device completion."""
        wp.capture_launch(self._graph)
        wp.synchronize_device(self.device)


class BsrMvRowPacked(BsrMvMixedPrecision):
    """Compare row packing for mixed-precision products of different row lengths."""

    params = [[8, 32, 128], [1, 16, 32, 64]]
    param_names = ["blocks_per_row", "rows_per_block"]

    def setup(self, blocks_per_row, rows_per_block):
        """Prepare a captured product with the requested row length and packing."""
        self._setup("float32", blocks_per_row, 128 if rows_per_block > 1 else 0, False, rows_per_block)

    def time_cuda(self, blocks_per_row, rows_per_block):
        """Time one captured row-packing configuration through device completion."""
        wp.capture_launch(self._graph)
        wp.synchronize_device(self.device)
