import warp as wp
import warp.fem as fem
import warp.sparse as wps
from warp.examples.fem.utils import gen_tetmesh


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

    def build_system(self, space: fem.FunctionSpace, integrand: fem.operator.Integrand):
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
    number = 10  # Number of measurements to make between a single setup and teardown

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
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
    number = 10  # Number of measurements to make between a single setup and teardown

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
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
    number = 10  # Number of measurements to make between a single setup and teardown

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        self.device = wp.get_device("cuda:0")

        res = 2
        with wp.ScopedDevice(self.device):
            geo = fem.Grid3D(res=(res, res, res))
            space = fem.make_polynomial_space(geo, degree=4, dtype=wp.vec3)
            self.build_system(space, diffusion_form_vector)

    def time_cuda(self):
        self.run()
