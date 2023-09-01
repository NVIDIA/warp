"""
This example simulates a convection-diffusion PDE using semi-Lagrangian advection

 D phi / dt - nu d2 phi / dx^2 = 0

"""

import argparse

import warp as wp

from warp.fem.types import *
from warp.fem.geometry import Grid2D, Trimesh2D
from warp.fem.field import make_test, make_trial
from warp.fem.space import make_polynomial_space
from warp.fem.domain import Cells
from warp.fem.integrate import integrate, interpolate
from warp.fem.operator import grad, integrand, lookup

from bsr_utils import bsr_to_scipy
from plot_utils import plot_surface
from mesh_utils import gen_trimesh

from scipy.sparse.linalg import factorized

import matplotlib.pyplot as plt
import matplotlib.animation as animation


@integrand
def initial_condition(domain: Domain, s: Sample):
    pos = domain(s)
    if pos[0] > 0.4 and pos[0] < 0.6 and pos[1] > 0.2 and pos[1] < 0.8:
        return 1.0
    return 0.0


@wp.func
def velocity(pos: wp.vec2, ang_vel: float):
    center = wp.vec2(0.5, 0.5)
    offset = pos - center
    return wp.vec2(offset[1], -offset[0]) * ang_vel


@integrand
def inertia_form(s: Sample, phi: Field, psi: Field, dt: float):
    return phi(s) * psi(s) / dt


@integrand
def transported_inertia_form(s: Sample, domain: Domain, phi: Field, psi: Field, ang_vel: float, dt: float):
    pos = domain(s)
    vel = velocity(pos, ang_vel)

    # semi-Lagrangian advection; evaluate phi upstream
    conv_pos = pos - vel * dt
    # lookup opertor constructs a Sample from a world position.
    # the optional last argument provides a initial guess for the lookup
    conv_phi = phi(lookup(domain, conv_pos, s))

    return conv_phi * psi(s) / dt


@integrand
def diffusion_form(
    s: Sample,
    u: Field,
    v: Field,
):
    return wp.dot(
        grad(u, s),
        grad(v, s),
    )


@integrand
def diffusion_and_inertia_form(s: Sample, phi: Field, psi: Field, dt: float, nu: float):
    return inertia_form(s, phi, psi, dt) + nu * diffusion_form(s, phi, psi)


if __name__ == "__main__":
    wp.init()
    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=50)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--n_frames", type=int, default=250)
    parser.add_argument("--viscosity", type=float, default=0.001)
    parser.add_argument("--ang_vel", type=float, default=1.0)
    parser.add_argument("--tri_mesh", action="store_true", help="Use a triangular mesh")
    args = parser.parse_args()

    res = args.resolution
    dt = 1.0 / (args.ang_vel * res)

    if args.tri_mesh:
        positions, tri_vidx = gen_trimesh(res=vec2i(res))
        geo = Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
    else:
        geo = Grid2D(res=vec2i(res))

    domain = Cells(geometry=geo)
    scalar_space = make_polynomial_space(geo, degree=args.degree)
    quadrature = None

    # Initial condition
    phi0 = scalar_space.make_field()
    interpolate(initial_condition, dest=phi0)

    # Assemble and factorize diffusion and inertia matrix
    test = make_test(space=scalar_space, domain=domain)
    trial = make_trial(space=scalar_space, domain=domain)
    matrix = integrate(
        diffusion_and_inertia_form,
        quadrature=quadrature,
        fields={"phi": trial, "psi": test},
        values={"nu": args.viscosity, "dt": dt},
    )
    matrix_solve = factorized(bsr_to_scipy(matrix))

    results = [phi0.dof_values.numpy()]
    phik = phi0

    for k in range(args.n_frames):
        # right-hand-side -- advected inertia
        rhs = integrate(
            transported_inertia_form,
            quadrature=quadrature,
            fields={"phi": phik, "psi": test},
            values={"ang_vel": args.ang_vel, "dt": dt},
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
