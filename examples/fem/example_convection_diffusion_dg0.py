"""
This example simulates a convection-diffusion PDE using FVM with upwind transport

 D phi / dt + nu Div f = 0
 f = grad phi

"""

import argparse

import warp as wp
import warp.fem as fem

from warp.sparse import bsr_mm, bsr_axpy, bsr_transposed


# Import example utilities
# Make sure that works both when imported as module and run as standalone file
try:
    from .bsr_utils import bsr_to_scipy, invert_diagonal_bsr_mass_matrix
    from .plot_utils import Plot
    from .mesh_utils import gen_trimesh, gen_quadmesh
    from .example_convection_diffusion import initial_condition, velocity, inertia_form
except ImportError:
    from bsr_utils import bsr_to_scipy, invert_diagonal_bsr_mass_matrix
    from plot_utils import Plot
    from mesh_utils import gen_trimesh, gen_quadmesh
    from example_convection_diffusion import initial_condition, velocity, inertia_form

from scipy.sparse.linalg import factorized


@fem.integrand
def vel_mass_form(
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    return wp.dot(v(s), u(s))


@fem.integrand
def half_diffusion_form(
    s: fem.Sample,
    domain: fem.Domain,
    psi: fem.Field,
    u: fem.Field,
):
    return fem.jump(psi, s) * wp.dot(fem.average(u, s), fem.normal(domain, s))


@fem.integrand
def upwind_transport_form(s: fem.Sample, domain: fem.Domain, phi: fem.Field, psi: fem.Field, ang_vel: float):
    pos = domain(s)

    vel = velocity(pos, ang_vel)

    vel_n = wp.dot(vel, fem.normal(domain, s))

    return fem.jump(psi, s) * (fem.average(phi, s) * vel_n + 0.5 * fem.jump(phi, s) * wp.abs(vel_n))


class Example:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=50)
    parser.add_argument("--num_frames", type=int, default=250)
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
        scalar_space = fem.make_polynomial_space(geo, degree=0)

        # Inertia matrix
        self._test = fem.make_test(space=scalar_space, domain=domain)
        trial = fem.make_trial(space=scalar_space, domain=domain)
        matrix_inertia = fem.integrate(
            inertia_form,
            fields={"phi": trial, "psi": self._test},
            values={"dt": self.sim_dt},
        )

        # Upwind transport term
        side_test = fem.make_test(space=scalar_space, domain=sides)
        side_trial = fem.make_trial(space=scalar_space, domain=sides)
        matrix_transport = fem.integrate(
            upwind_transport_form,
            fields={"phi": side_trial, "psi": side_test},
            values={"ang_vel": args.ang_vel},
        )

        # Diffusion bilinear form
        # Since we have piecewise constant element, we cannot use the classical diffusion form
        # Instead we assemble the matrix B M^-1 B^T, with B associated to the form psi div(u)
        # and the diagonal matrix M to the velocity mass form u.v

        velocity_space = fem.make_polynomial_space(geo, degree=0, dtype=wp.vec2)
        side_trial_vel = fem.make_trial(space=velocity_space, domain=sides)
        matrix_half_diffusion = fem.integrate(
            half_diffusion_form,
            fields={"psi": side_test, "u": side_trial_vel},
        )

        # Diagonal velocity mass matrix
        test_vel = fem.make_test(space=velocity_space, domain=domain)
        trial_vel = fem.make_trial(space=velocity_space, domain=domain)
        inv_vel_mass_matrix = fem.integrate(
            vel_mass_form, domain=domain, fields={"u": trial_vel, "v": test_vel}, nodal=True
        )
        invert_diagonal_bsr_mass_matrix(inv_vel_mass_matrix)

        # Assemble system matrix

        matrix = matrix_inertia
        # matrix += matrix_transport
        bsr_axpy(x=matrix_transport, y=matrix)
        # matrix += nu * B M^-1 B^T
        bsr_mm(
            x=bsr_mm(matrix_half_diffusion, inv_vel_mass_matrix),
            y=bsr_transposed(matrix_half_diffusion),
            z=matrix,
            alpha=args.viscosity,
            beta=1.0,
        )

        # Compute LU factorization of system matrix
        self._solve_lu = factorized(bsr_to_scipy(matrix))

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
