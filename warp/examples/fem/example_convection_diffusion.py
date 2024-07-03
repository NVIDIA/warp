# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Convection Diffusion
#
# This example simulates a convection-diffusion PDE using
# semi-Lagrangian advection
#
# D phi / dt - nu d2 phi / dx^2 = 0
###########################################################################

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem


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
    # lookup operator constructs a Sample from a world position.
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
    def __init__(self, quiet=False, degree=2, resolution=50, tri_mesh=False, viscosity=0.001, ang_vel=1.0):
        self._quiet = quiet

        self._ang_vel = ang_vel

        res = resolution
        self.sim_dt = 1.0 / (ang_vel * res)
        self.current_frame = 0

        if tri_mesh:
            positions, tri_vidx = fem_example_utils.gen_trimesh(res=wp.vec2i(res))
            geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions, build_bvh=True)
        else:
            geo = fem.Grid2D(res=wp.vec2i(res))

        domain = fem.Cells(geometry=geo)
        scalar_space = fem.make_polynomial_space(geo, degree=degree)

        # Initial condition
        self._phi_field = scalar_space.make_field()
        fem.interpolate(initial_condition, dest=self._phi_field)

        # Assemble diffusion and inertia matrix
        self._test = fem.make_test(space=scalar_space, domain=domain)
        self._trial = fem.make_trial(space=scalar_space, domain=domain)
        self._matrix = fem.integrate(
            diffusion_and_inertia_form,
            fields={"phi": self._trial, "psi": self._test},
            values={"nu": viscosity, "dt": self.sim_dt},
            output_dtype=float,
        )

        self.renderer = fem_example_utils.Plot()
        self.renderer.add_surface("phi", self._phi_field)

    def step(self):
        self.current_frame += 1

        # right-hand-side -- advected inertia
        rhs = fem.integrate(
            transported_inertia_form,
            fields={"phi": self._phi_field, "psi": self._test},
            values={"ang_vel": self._ang_vel, "dt": self.sim_dt},
            output_dtype=float,
        )

        # Solve linear system
        fem_example_utils.bsr_cg(self._matrix, x=self._phi_field.dof_values, b=rhs, quiet=self._quiet, tol=1.0e-12)

    def render(self):
        self.renderer.begin_frame(time=self.current_frame * self.sim_dt)
        self.renderer.add_surface("phi", self._phi_field)
        self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=50, help="Grid resolution.")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree of shape functions.")
    parser.add_argument("--num_frames", type=int, default=250, help="Total number of frames.")
    parser.add_argument("--viscosity", type=float, default=0.001, help="Fluid viscosity parameter.")
    parser.add_argument("--ang_vel", type=float, default=1.0, help="Angular velocity.")
    parser.add_argument("--tri_mesh", action="store_true", help="Use a triangular mesh.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppresses the printing out of iteration residuals.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            quiet=args.quiet,
            degree=args.degree,
            resolution=args.resolution,
            tri_mesh=args.tri_mesh,
            viscosity=args.viscosity,
            ang_vel=args.ang_vel,
        )

        for k in range(args.num_frames):
            print(f"Frame {k}:")
            example.step()
            example.render()

        if not args.headless:
            example.renderer.plot()
