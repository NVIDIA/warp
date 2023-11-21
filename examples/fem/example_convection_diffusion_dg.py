"""
This example simulates a convection-diffusion PDE using Discontinuous Galerkin
with upwind transport and Symmetric Interior Penalty

 D phi / dt - nu d2 phi / dx^2 = 0

"""

import argparse

import warp as wp
import warp.fem as fem

from warp.sparse import bsr_axpy

# Import example utilities
# Make sure that works both when imported as module and run as standalone file
try:
    from .bsr_utils import bsr_to_scipy
    from .mesh_utils import gen_trimesh, gen_quadmesh
    from .plot_utils import Plot
    from .example_convection_diffusion import (
        initial_condition,
        velocity,
        inertia_form,
        diffusion_form,
    )
except ImportError:
    from bsr_utils import bsr_to_scipy
    from mesh_utils import gen_trimesh, gen_quadmesh
    from plot_utils import Plot
    from example_convection_diffusion import (
        initial_condition,
        velocity,
        inertia_form,
        diffusion_form,
    )

# Non-SPD system, solve using scipy
from scipy.sparse.linalg import factorized


# Standard transport term, on cells' interior
@fem.integrand
def transport_form(s: fem.Sample, domain: fem.Domain, phi: fem.Field, psi: fem.Field, ang_vel: float):
    pos = domain(s)
    vel = velocity(pos, ang_vel)

    return psi(s) * wp.dot(fem.grad(phi, s), vel)


# Upwind flux, on cell sides
@fem.integrand
def upwind_transport_form(s: fem.Sample, domain: fem.Domain, phi: fem.Field, psi: fem.Field, ang_vel: float):
    pos = domain(s)
    vel = velocity(pos, ang_vel)
    vel_n = wp.dot(vel, fem.normal(domain, s))

    return fem.jump(phi, s) * (-fem.average(psi, s) * vel_n + 0.5 * fem.jump(psi, s) * wp.abs(vel_n))


# Symmetric-Interior-Penalty diffusion term (See Pietro Ern 2012)
@fem.integrand
def sip_diffusion_form(
    s: fem.Sample,
    domain: fem.Domain,
    psi: fem.Field,
    phi: fem.Field,
):
    nor = fem.normal(domain, s)
    penalty = fem.measure_ratio(domain, s) * float(fem.degree(psi) * fem.degree(phi))

    return penalty * fem.jump(phi, s) * fem.jump(psi, s) - (
        wp.dot(fem.grad_average(phi, s), nor) * fem.jump(psi, s)
        + wp.dot(fem.grad_average(psi, s), nor) * fem.jump(phi, s)
    )


class Example:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=50)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--viscosity", type=float, default=0.001)
    parser.add_argument("--ang_vel", type=float, default=1.0)
    parser.add_argument("--mesh", choices=("grid", "tri", "quad"), default="grid", help="Mesh type")

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

        if args.mesh == "tri":
            positions, tri_vidx = gen_trimesh(res=wp.vec2i(args.resolution))
            geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
        elif args.mesh == "quad":
            positions, quad_vidx = gen_quadmesh(res=wp.vec2i(args.resolution))
            geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions)
        else:
            geo = fem.Grid2D(res=wp.vec2i(args.resolution))

        domain = fem.Cells(geometry=geo)
        sides = fem.Sides(geo)
        scalar_space = fem.make_polynomial_space(
            geo,
            discontinuous=True,
            degree=args.degree,
            family=fem.Polynomial.GAUSS_LEGENDRE,
        )

        # Assemble transport, diffusion and inertia matrices

        self._test = fem.make_test(space=scalar_space, domain=domain)
        trial = fem.make_trial(space=scalar_space, domain=domain)

        matrix_inertia = fem.integrate(
            inertia_form,
            fields={"phi": trial, "psi": self._test},
            values={"dt": self.sim_dt},
        )

        matrix_transport = fem.integrate(
            transport_form,
            fields={"phi": trial, "psi": self._test},
            values={"ang_vel": args.ang_vel},
        )

        side_test = fem.make_test(space=scalar_space, domain=sides)
        side_trial = fem.make_trial(space=scalar_space, domain=sides)

        bsr_axpy(
            fem.integrate(
                upwind_transport_form,
                fields={"phi": side_trial, "psi": side_test},
                values={"ang_vel": args.ang_vel},
            ),
            y=matrix_transport,
        )

        matrix_diffusion = fem.integrate(
            diffusion_form,
            fields={"u": trial, "v": self._test},
        )
        bsr_axpy(
            fem.integrate(
                sip_diffusion_form,
                fields={"phi": side_trial, "psi": side_test},
            ),
            y=matrix_diffusion,
        )

        self._matrix = matrix_inertia
        bsr_axpy(x=matrix_transport, y=self._matrix)
        bsr_axpy(x=matrix_diffusion, y=self._matrix, alpha=args.viscosity)
        
        # Compute LU factorization of system matrix
        self._solve_lu = factorized(bsr_to_scipy(self._matrix))

        # Initial condition
        self._phi_field = scalar_space.make_field()
        fem.interpolate(initial_condition, dest=self._phi_field)

        self.renderer = Plot(stage)
        self.renderer.add_surface("phi", self._phi_field)


    def update(self):
        self.current_frame += 1

        rhs = fem.integrate(
            inertia_form,
            fields={"phi": self._phi_field, "psi": self._test},
            values={"dt": self.sim_dt},
        )

        self._phi_field.dof_values = self._solve_lu(rhs.numpy())

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
