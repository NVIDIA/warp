"""
This example simulates a convection-diffusion PDE using FVM with upwind transport

 D phi / dt + nu Div f = 0
 f = grad phi

"""

import argparse

import warp as wp
from warp.sparse import bsr_mm, bsr_axpy, bsr_transposed

from warp.fem import Sample, Field, Domain
from warp.fem import Grid2D, Trimesh2D
from warp.fem import make_test, make_trial
from warp.fem import make_polynomial_space
from warp.fem import RegularQuadrature
from warp.fem import Cells, Sides
from warp.fem import integrate, interpolate
from warp.fem import integrand, jump, average, normal

from bsr_utils import bsr_to_scipy, invert_diagonal_bsr_mass_matrix
from plot_utils import plot_surface
from mesh_utils import gen_trimesh

from example_convection_diffusion import initial_condition, velocity, inertia_form

from scipy.sparse.linalg import factorized

import matplotlib.pyplot as plt
import matplotlib.animation as animation


@integrand
def vel_mass_form(
    s: Sample,
    u: Field,
    v: Field,
):
    return wp.dot(v(s), u(s))


@integrand
def half_diffusion_form(
    s: Sample,
    domain: Domain,
    psi: Field,
    u: Field,
):
    return jump(psi, s) * wp.dot(average(u, s), normal(domain, s))


@integrand
def upwind_transport_form(s: Sample, domain: Domain, phi: Field, psi: Field, ang_vel: float):
    pos = domain(s)

    vel = velocity(pos, ang_vel)

    vel_n = wp.dot(vel, normal(domain, s))

    return jump(psi, s) * (average(phi, s) * vel_n + 0.5 * jump(phi, s) * wp.abs(vel_n))


if __name__ == "__main__":
    wp.init()
    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=50)
    parser.add_argument("--n_frames", type=int, default=250)
    parser.add_argument("--viscosity", type=float, default=0.001)
    parser.add_argument("--ang_vel", type=float, default=1.0)
    parser.add_argument("--tri_mesh", action="store_true", help="Use a triangular mesh")

    args = parser.parse_args()

    res = args.resolution
    dt = 1.0 / (args.ang_vel * res)

    if args.tri_mesh:
        positions, tri_vidx = gen_trimesh(res=wp.vec2i(res))
        geo = Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
    else:
        geo = Grid2D(res=wp.vec2i(res))

    domain = Cells(geometry=geo)
    sides = Sides(geo)
    quadrature = RegularQuadrature(domain=domain, order=2)
    scalar_space = make_polynomial_space(geo, degree=0)

    # Initial condition
    phi0 = scalar_space.make_field()
    interpolate(initial_condition, dest=phi0)

    # Inertia matrix
    test = make_test(space=scalar_space, domain=domain)
    trial = make_trial(space=scalar_space, domain=domain)
    matrix_inertia = integrate(
        inertia_form,
        quadrature=quadrature,
        fields={"phi": trial, "psi": test},
        values={"dt": dt},
    )

    # Upwind transport term
    side_test = make_test(space=scalar_space, domain=sides)
    side_trial = make_trial(space=scalar_space, domain=sides)
    matrix_transport = integrate(
        upwind_transport_form,
        fields={"phi": side_trial, "psi": side_test},
        values={"ang_vel": args.ang_vel},
    )

    # Diffusion bilinear form
    # Since we have piecewise constant element, we cannot use the classical diffusion form
    # Instead we assemble the matrix B M^-1 B^T, with B associated to the form psi div(u)
    # and the diagonal matrix M to the velocity mass form u.v

    velocity_space = make_polynomial_space(geo, degree=0, dtype=wp.vec2)
    side_trial_vel = make_trial(space=velocity_space, domain=sides)
    matrix_half_diffusion = integrate(
        half_diffusion_form,
        fields={"psi": side_test, "u": side_trial_vel},
    )

    # Diagonal velocity mass matrix
    test_vel = make_test(space=velocity_space, domain=domain)
    trial_vel = make_trial(space=velocity_space, domain=domain)
    inv_vel_mass_matrix = integrate(vel_mass_form, domain=domain, fields={"u": trial_vel, "v": test_vel}, nodal=True)
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

    matrix_solve = factorized(bsr_to_scipy(matrix))

    results = [phi0.dof_values.numpy()]
    phik = phi0
    for k in range(args.n_frames):
        # right-hand-side -- standard inertia
        rhs = integrate(
            inertia_form,
            quadrature=quadrature,
            fields={"phi": phik, "psi": test},
            values={"dt": dt},
        )

        # Solve using Scipy
        x = matrix_solve(rhs.numpy().flatten())

        phik.dof_values = x
        results.append(x)

    colormesh = plot_surface(phi0)
    ax = colormesh.axes

    def animate(i):
        ax.clear()
        phik.dof_values = results[i]
        return plot_surface(phik, axes=ax)

    anim = animation.FuncAnimation(
        ax.figure,
        animate,
        interval=30,
        blit=False,
        frames=len(results),
    )
    plt.show()
