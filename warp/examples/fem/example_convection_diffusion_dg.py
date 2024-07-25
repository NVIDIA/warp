# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Convection Diffusion DG
#
# This example simulates a convection-diffusion PDE using Discontinuous
# Galerkin with upwind transport and Symmetric Interior Penalty
#
# D phi / dt - nu d2 phi / dx^2 = 0
###########################################################################

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem
from warp.examples.fem.example_convection_diffusion import (
    diffusion_form,
    inertia_form,
    initial_condition,
    velocity,
)


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

    if wp.min(pos) <= 0.0 or wp.max(pos) >= 1.0:  # boundary side
        return phi(s) * (-psi(s) * vel_n + 0.5 * psi(s) * wp.abs(vel_n))

    # interior side
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
    def __init__(self, quiet=False, degree=2, resolution=50, mesh="grid", viscosity=0.0001, ang_vel=1.0):
        self._quiet = quiet

        res = resolution
        self.sim_dt = 1.0 / (ang_vel * res)
        self.current_frame = 0

        if mesh == "tri":
            positions, tri_vidx = fem_example_utils.gen_trimesh(res=wp.vec2i(resolution))
            geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
        elif mesh == "quad":
            positions, quad_vidx = fem_example_utils.gen_quadmesh(res=wp.vec2i(resolution))
            geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions)
        else:
            geo = fem.Grid2D(res=wp.vec2i(resolution))

        domain = fem.Cells(geometry=geo)
        sides = fem.Sides(geo)
        scalar_space = fem.make_polynomial_space(
            geo,
            discontinuous=True,
            degree=degree,
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
            values={"ang_vel": ang_vel},
        )

        side_test = fem.make_test(space=scalar_space, domain=sides)
        side_trial = fem.make_trial(space=scalar_space, domain=sides)

        matrix_transport += fem.integrate(
            upwind_transport_form,
            fields={"phi": side_trial, "psi": side_test},
            values={"ang_vel": ang_vel},
        )

        matrix_diffusion = fem.integrate(
            diffusion_form,
            fields={"u": trial, "v": self._test},
        )

        matrix_diffusion += fem.integrate(
            sip_diffusion_form,
            fields={"phi": side_trial, "psi": side_test},
        )

        self._matrix = matrix_inertia + matrix_transport + viscosity * matrix_diffusion

        # Initial condition
        self._phi_field = scalar_space.make_field()
        fem.interpolate(initial_condition, dest=self._phi_field)

        self.renderer = fem_example_utils.Plot()
        self.renderer.add_field("phi", self._phi_field)

    def step(self):
        self.current_frame += 1

        rhs = fem.integrate(
            inertia_form,
            fields={"phi": self._phi_field, "psi": self._test},
            values={"dt": self.sim_dt},
        )

        phi = wp.zeros_like(rhs)
        fem_example_utils.bsr_cg(self._matrix, b=rhs, x=phi, method="bicgstab", quiet=self._quiet)

        wp.utils.array_cast(in_array=phi, out_array=self._phi_field.dof_values)

    def render(self):
        self.renderer.begin_frame(time=self.current_frame * self.sim_dt)
        self.renderer.add_field("phi", self._phi_field)
        self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=50, help="Grid resolution.")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree of shape functions.")
    parser.add_argument("--num_frames", type=int, default=100, help="Total number of frames.")
    parser.add_argument("--viscosity", type=float, default=0.001, help="Fluid viscosity parameter.")
    parser.add_argument("--ang_vel", type=float, default=1.0, help="Angular velocity.")
    parser.add_argument("--mesh", choices=("grid", "tri", "quad"), default="grid", help="Mesh type.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            quiet=args.quiet,
            degree=args.degree,
            resolution=args.resolution,
            mesh=args.mesh,
            viscosity=args.viscosity,
            ang_vel=args.ang_vel,
        )

        for k in range(args.num_frames):
            print(f"Frame {k}:")
            example.step()
            example.render()

        if not args.headless:
            example.renderer.plot()
