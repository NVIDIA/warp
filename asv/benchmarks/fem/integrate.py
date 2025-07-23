import warp as wp
import warp.fem as fem
from warp.examples.fem.utils import gen_tetmesh


@wp.func
def symmetric_strain(sig: wp.vec3, V: wp.mat33):
    return V * wp.diag(sig) * wp.transpose(V)


@wp.func
def symmetric_strain_delta(U: wp.mat33, sig: wp.vec3, V: wp.mat33, dF: wp.mat33):
    # see supplementary of `WRAPD: Weighted Rotation-aware ADMM`, Brown and Narain 21

    Ut = wp.transpose(U)
    Vt = wp.transpose(V)

    dF_loc = Ut * dF * V
    SigdF_loc = wp.diag(sig) * dF_loc

    sig_op = wp.matrix_from_cols(wp.vec3(sig[0]), wp.vec3(sig[1]), wp.vec3(sig[2]))
    dSig = wp.cw_div(SigdF_loc + wp.transpose(SigdF_loc), sig_op + wp.transpose(sig_op))
    dS = V * dSig * Vt

    return dS


@fem.integrand
def defgrad(u: fem.Field, s: fem.Sample):
    return fem.grad(u, s) + wp.identity(n=3, dtype=float)


@wp.func
def hooke_stress(S: wp.mat33, lame: wp.vec2):
    strain = S - wp.identity(n=3, dtype=float)
    return 2.0 * lame[1] * strain + lame[0] * wp.trace(strain) * wp.identity(n=3, dtype=float)


@wp.func
def hooke_energy(S: wp.mat33, lame: wp.vec2):
    strain = S - wp.identity(n=3, dtype=float)
    return 0.5 * wp.ddot(strain, hooke_stress(S, lame))


@wp.func
def hooke_hessian(S: wp.mat33, tau: wp.mat33, sig: wp.mat33, lame: wp.vec2):
    return wp.ddot(hooke_stress(sig + wp.identity(n=3, dtype=float), lame), tau)


@fem.integrand
def cr_elastic_energy(s: fem.Sample, u_cur: fem.Field, lame: fem.Field):
    F = defgrad(u_cur, s)
    U, sig, V = wp.svd3(F)

    S = symmetric_strain(sig, V)
    return hooke_energy(S, lame(s))


@fem.integrand
def cr_elastic_forces(s: fem.Sample, u_cur: fem.Field, v: fem.Field, lame: fem.Field):
    F = defgrad(u_cur, s)
    U, sig, V = wp.svd3(F)

    S = symmetric_strain(sig, V)
    tau = symmetric_strain_delta(U, sig, V, fem.grad(v, s))
    return -wp.ddot(tau, hooke_stress(S, lame(s)))


@fem.integrand
def cr_elasticity_hessian(s: fem.Sample, u_cur: fem.Field, u: fem.Field, v: fem.Field, lame: fem.Field):
    F = defgrad(u_cur, s)
    U, sig, V = wp.svd3(F)

    S_s = symmetric_strain(sig, V)
    tau_s = symmetric_strain_delta(U, sig, V, fem.grad(v, s))
    sig_s = symmetric_strain_delta(U, sig, V, fem.grad(u, s))
    lame_s = lame(s)

    return hooke_hessian(S_s, tau_s, sig_s, lame_s)


@fem.integrand
def diffusion_form_scalar(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def diffusion_form_vector(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.ddot(fem.D(u, s), fem.D(v, s))


class FemCorotatedElasticity:
    """Utility base class for building FEM matrices to test BSR matrix-vector multiplication."""

    def __init__(self, use_graph: bool = True):
        self._use_graph = use_graph

    def make_integrate_func(self, space: fem.FunctionSpace, fn: str, assembly: str):
        u = fem.make_trial(space)
        v = fem.make_test(space)
        u_cur = fem.make_discrete_field(space)

        lame = fem.UniformField(v.domain, value=wp.vec2(1.0, 0.3))
        E = wp.array(shape=1, dtype=float)

        quadrature = fem.RegularQuadrature(v.domain, order=2 * u_cur.degree)

        self._integrate_funcs = {
            "energy": lambda: fem.integrate(
                cr_elastic_energy,
                domain=v.domain,
                fields={"u_cur": u_cur, "lame": lame},
                output=E,
                quadrature=quadrature,
                kernel_options={"enable_backward": False},
            ),
            "forces": lambda: fem.integrate(
                cr_elastic_forces,
                fields={"u_cur": u_cur, "v": v, "lame": lame},
                output_dtype=float,
                assembly=assembly,
                quadrature=quadrature,
                kernel_options={"enable_backward": False},
            ),
            "hessian": lambda: fem.integrate(
                cr_elasticity_hessian,
                fields={"u_cur": u_cur, "u": u, "v": v, "lame": lame},
                output_dtype=float,
                assembly=assembly,
                quadrature=quadrature,
                kernel_options={"enable_backward": False},
            ),
        }

        integrate_fn = self._integrate_funcs[fn]
        integrate_fn()

        if self._use_graph:
            with wp.ScopedCapture() as capture:
                integrate_fn()
            self._graph = capture.graph

    def run(self, fn: str):
        if self._use_graph:
            wp.capture_launch(self._graph)
        else:
            self._integrate_funcs[fn]()
        wp.synchronize_device(self.device)


class FemCorotatedElasticityQuadraticTetmesh(FemCorotatedElasticity):
    """Test corotated elasticity with quadratic tetrahedral elements."""

    rounds = 1
    repeat = 2
    number = 5

    params = (["energy", "forces", "hessian"], ["dispatch"])
    param_names = ["fn", "assembly"]

    def setup(self, fn: str, assembly: str):
        wp.init()
        wp.build.clear_kernel_cache()
        self.device = wp.get_device("cuda:0")

        res = 16
        with wp.ScopedDevice(self.device):
            pos, cells = gen_tetmesh(res=(res, res, res))
            geo = fem.Tetmesh(cells, pos)
            space = fem.make_polynomial_space(geo, degree=2, dtype=wp.vec3)
            self.make_integrate_func(space, fn, assembly)

    def time_cuda(self, fn, assembly):
        self.run(fn)


class FemCorotatedElasticityLinearGrid(FemCorotatedElasticity):
    """Test corotated elasticity with linear grid elements."""

    rounds = 1
    repeat = 2
    number = 5

    params = (["energy", "forces", "hessian"], ["generic", "dispatch"])
    param_names = ["fn", "assembly"]

    def setup(self, fn: str, assembly: str):
        wp.init()
        wp.build.clear_kernel_cache()
        self.device = wp.get_device("cuda:0")

        res = 32
        with wp.ScopedDevice(self.device):
            geo = fem.Grid3D(res=(res, res, res))
            space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec3)
            self.make_integrate_func(space, fn, assembly)

    def time_cuda(self, fn, assembly):
        self.run(fn)


class FemCorotatedElasticityVeryHighOrder(FemCorotatedElasticity):
    """Test corotated elasticity with high order elements."""

    rounds = 1
    repeat = 2
    number = 5

    params = (["energy", "forces", "hessian"], ["dispatch"])
    param_names = ["fn", "assembly"]

    def setup(self, fn: str, assembly: str):
        wp.init()
        wp.build.clear_kernel_cache()
        self.device = wp.get_device("cuda:0")

        res = 2
        with wp.ScopedDevice(self.device):
            geo = fem.Grid3D(res=(res, res, res))
            space = fem.make_polynomial_space(geo, degree=4, dtype=wp.vec3)
            self.make_integrate_func(space, fn, assembly)

    def time_cuda(self, fn, assembly):
        self.run(fn)


if __name__ == "__main__":
    for cases in (
        (FemCorotatedElasticityQuadraticTetmesh, "quadratic_tetmesh"),
        (FemCorotatedElasticityLinearGrid, "linear_grid"),
        (FemCorotatedElasticityVeryHighOrder, "very_high_order"),
    ):
        case = cases[0](use_graph=False)

        for fn in case.params[0]:
            for asm in case.params[1]:
                case.setup(fn, asm)
                for _k in range(3):
                    with wp.ScopedTimer(f"{cases[1]}_{fn}_{asm}", synchronize=True):
                        case.time_cuda(fn, asm)
