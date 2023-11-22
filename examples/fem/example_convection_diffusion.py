"""
This example simulates a convection-diffusion PDE using semi-Lagrangian advection

 D phi / dt - nu d2 phi / dx^2 = 0

"""

import argparse


import warp as wp
import warp.fem as fem

# Import example utilities
# Make sure that works both when imported as module and run as standalone file
try:
    from .bsr_utils import bsr_cg
    from .mesh_utils import gen_trimesh
    from .plot_utils import Plot
except ImportError:
    from bsr_utils import bsr_cg
    from mesh_utils import gen_trimesh
    from plot_utils import Plot


@fem.integrand
def initial_condition(domain: fem.Domain, s: fem.Sample):
    """Initial condition: 1.0 in ]0.6, 0.4[ x ]0.2, 0.8[, 0.0 elsewhere"""
    pos = domain(s)
    if pos[0] > 0.4 and pos[0] < 0.6 and pos[1] > 0.2 and pos[1] < 0.8:
        return 1.0
    return 0.0


@wp.func
def velocity(pos: wp.vec2, ang_vel: float):
    center = wp.vec2(0.5, 0.5)
    offset = pos - center
    return wp.vec2(offset[1], -offset[0]) * ang_vel


@fem.integrand
def inertia_form(s: fem.Sample, phi: fem.Field, psi: fem.Field, dt: float):
    return phi(s) * psi(s) / dt


@fem.integrand
def transported_inertia_form(
    s: fem.Sample, domain: fem.Domain, phi: fem.Field, psi: fem.Field, ang_vel: float, dt: float
):
    pos = domain(s)
    vel = velocity(pos, ang_vel)

    # semi-Lagrangian advection; evaluate phi upstream
    conv_pos = pos - vel * dt
    # lookup opertor constructs a Sample from a world position.
    # the optional last argument provides a initial guess for the lookup
    conv_phi = phi(fem.lookup(domain, conv_pos, s))

    return conv_phi * psi(s) / dt


@fem.integrand
def diffusion_form(
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    return wp.dot(
        fem.grad(u, s),
        fem.grad(v, s),
    )


@fem.integrand
def diffusion_and_inertia_form(s: fem.Sample, phi: fem.Field, psi: fem.Field, dt: float, nu: float):
    return inertia_form(s, phi, psi, dt) + nu * diffusion_form(s, phi, psi)


class Example:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=50)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=250)
    parser.add_argument("--viscosity", type=float, default=0.001)
    parser.add_argument("--ang_vel", type=float, default=1.0)
    parser.add_argument("--tri_mesh", action="store_true", help="Use a triangular mesh")

    def __init__(self, stage=None, quiet=False, args=None, **kwargs):
        if args is None:
            # Read args from kwargs, add default arg values from parser
            args = argparse.Namespace(**kwargs)
            args = Example.parser.parse_args(args=[], namespace=args)
        self._args = args
        self._quiet = quiet

        res = args.resolution
        self.sim_dt = 1.0 / (args.ang_vel * res)
        self.current_frame = 0

        if args.tri_mesh:
            positions, tri_vidx = gen_trimesh(res=wp.vec2i(res))
            geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
        else:
            geo = fem.Grid2D(res=wp.vec2i(res))

        domain = fem.Cells(geometry=geo)
        scalar_space = fem.make_polynomial_space(geo, degree=args.degree)

        # Initial condition
        self._phi_field = scalar_space.make_field()
        fem.interpolate(initial_condition, dest=self._phi_field)

        # Assemble diffusion and inertia matrix
        self._test = fem.make_test(space=scalar_space, domain=domain)
        self._trial = fem.make_trial(space=scalar_space, domain=domain)
        self._matrix = fem.integrate(
            diffusion_and_inertia_form,
            fields={"phi": self._trial, "psi": self._test},
            values={"nu": args.viscosity, "dt": self.sim_dt},
            output_dtype=float,
        )

        self.renderer = Plot(stage)
        self.renderer.add_surface("phi", self._phi_field)

    def update(self):
        self.current_frame += 1

        # right-hand-side -- advected inertia
        rhs = fem.integrate(
            transported_inertia_form,
            fields={"phi": self._phi_field, "psi": self._test},
            values={"ang_vel": self._args.ang_vel, "dt": self.sim_dt},
            output_dtype=float,
        )

        # Solve linear system
        bsr_cg(self._matrix, x=self._phi_field.dof_values, b=rhs, quiet=self._quiet, tol=1.0e-12)

    def render(self):
        self.renderer.begin_frame(time = self.current_frame * self.sim_dt)
        self.renderer.add_surface("phi", self._phi_field)
        self.renderer.end_frame()


if __name__ == "__main__":
    wp.init()
    wp.set_module_options({"enable_backward": False})

    args = Example.parser.parse_args()

    example = Example(args=args)
    for k in range(args.num_frames):
        print(f"Frame {k}:")
        example.update()
        example.render()

    example.renderer.plot()
