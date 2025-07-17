from typing import Optional, Tuple

import numpy as np

import warp as wp
import warp.fem as fem
import warp.sparse as wps
from warp.examples.fem.utils import gen_tetmesh


@fem.integrand
def grad_field(s: fem.Sample, u: fem.Field):
    return fem.grad(u, s)


class BsrMMFemMatrix:
    """Utility base class for building FEM matrices to test BSR matrix-vector multiplication."""

    def __init__(self, use_graph: bool = True):
        self._use_graph = use_graph

    def build_system(
        self, space: fem.FunctionSpace, quadrature: Optional[int] = None, block_shape: Optional[Tuple[int, int]] = None
    ):
        u = fem.make_trial(space)

        if quadrature is None:
            quadrature = fem.RegularQuadrature(u.domain, order=space.degree)

        dof_size = wp.types.type_length(space.dtype)

        self._mat = wps.bsr_zeros(
            rows_of_blocks=quadrature.total_point_count(),
            cols_of_blocks=u.partition_node_count(),
            block_type=wp.types.matrix(shape=(3 * dof_size, dof_size), dtype=wp.float32),
        )

        fem.interpolate(grad_field, dest=self._mat, quadrature=quadrature, fields={"u": u})

        if block_shape is not None:
            self._mat = wps.bsr_copy(self._mat, block_shape=block_shape)

        self._mat.nnz_sync()

        self._id_mat = wps.bsr_identity(
            self._mat.nrow,
            block_type=wp.mat(shape=(self._mat.block_shape[0], self._mat.block_shape[0]), dtype=self._mat.scalar_type),
        )

        self._mat_t = self._mat.transpose()

        self._work_arrays = wps.bsr_mm_work_arrays()
        self._work_arrays_id_left = wps.bsr_mm_work_arrays()
        self._work_arrays_id_right = wps.bsr_mm_work_arrays()

        self._run_impl(reuse_topology=False)

        if self._use_graph:
            with wp.ScopedCapture() as capture:
                self._run_impl(reuse_topology=True)
            self._graph = capture.graph

        wp.synchronize_device(self.device)

    def _run_impl(self, reuse_topology: bool = True):
        # self multiply
        wps.bsr_mm(self._mat_t, self._mat, work_arrays=self._work_arrays, reuse_topology=reuse_topology)
        # multiply with diag on right
        wps.bsr_mm(self._mat_t, self._id_mat, work_arrays=self._work_arrays_id_right, reuse_topology=reuse_topology)
        # multiply with diag on left
        wps.bsr_mm(self._id_mat, self._mat, work_arrays=self._work_arrays_id_left, reuse_topology=reuse_topology)

    def run(self):
        if self._use_graph:
            wp.capture_launch(self._graph)
        else:
            self._run_impl(reuse_topology=True)

        wp.synchronize_device(self.device)


class BsrMMQuadraticTetmeshMatrix(BsrMMFemMatrix):
    """Test BSR matrix-vector multiplication with quadratic tetrahedral elements."""

    rounds = 1
    repeat = 2
    number = 5

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        self.device = wp.get_device("cuda:0")

        res = 32
        with wp.ScopedDevice(self.device):
            pos, cells = gen_tetmesh(res=(res, res, res))
            geo = fem.Tetmesh(cells, pos)
            space = fem.make_polynomial_space(geo, degree=2, dtype=wp.vec3)
            self.build_system(space)

    def time_cuda(self):
        self.run()


class BsrMMLinearGridMatrix(BsrMMFemMatrix):
    """Test BSR matrix-vector multiplication with linear grid elements."""

    rounds = 1
    repeat = 2
    number = 5

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        self.device = wp.get_device("cuda:0")

        res = 64
        with wp.ScopedDevice(self.device):
            geo = fem.Grid3D(res=(res, res, res))
            space = fem.make_polynomial_space(geo)
            self.build_system(space)

    def time_cuda(self):
        self.run()


class BsrMMDeepDense(BsrMMFemMatrix):
    """Test BSR matrix-vector multiplication with almost dense matrices (high-order elements)."""

    rounds = 1
    repeat = 2
    number = 5

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        self.device = wp.get_device("cuda:0")

        res = 1
        n_qp = 6000

        with wp.ScopedDevice(self.device):
            geo = fem.Grid3D(res=(res, res, res))
            space = fem.make_polynomial_space(geo, degree=3, dtype=wp.vec3)

            rng = np.random.default_rng(42)
            qps = wp.array(rng.random((1, n_qp, 3), dtype=np.float32), dtype=wp.vec3)
            weights = wp.full((1, n_qp), value=1.0 / n_qp, dtype=wp.float32)

            quadrature = fem.ExplicitQuadrature(fem.Cells(geo), points=qps, weights=weights)

            self.build_system(space, quadrature=quadrature, block_shape=(9, 12))

    def time_cuda(self):
        self.run()


if __name__ == "__main__":
    for cases in (
        (BsrMMDeepDense, "deep_dense"),
        # (BsrMMQuadraticTetmeshMatrix, "quadratic_tetmesh"),
        # (BsrMMLinearGridMatrix, "linear_grid"),
    ):
        A = cases[0](use_graph=False)
        A.setup()

        for _k in range(3):
            with wp.ScopedTimer(cases[1], synchronize=True):
                A.time_cuda()
