"""
This example simulates a convection-diffusion PDE using Discontinuous Galerkin
with upwind transport and Symmetric Interior Penalty

 D phi / dt - nu d2 phi / dx^2 = 0

"""

import argparse

import warp as wp
from warp.sparse import bsr_axpy

from warp.fem import Field, Domain, Sample
from warp.fem import Grid2D, Trimesh2D
from warp.fem import make_test, make_trial
from warp.fem import make_polynomial_space
from warp.fem import RegularQuadrature
from warp.fem import Cells, Sides
from warp.fem import integrate, interpolate
from warp.fem import Polynomial
from warp.fem import (
    grad,
    integrand,
    jump,
    average,
    normal,
    grad_average,
    measure_ratio,
    degree,
)

from bsr_utils import bsr_to_scipy
from plot_utils import plot_surface
from mesh_utils import gen_trimesh

from example_convection_diffusion import (
    initial_condition,
    velocity,
    inertia_form,
    diffusion_form,
)

from scipy.sparse.linalg import factorized

import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Standard transport term, on cells' interior
@integrand
def transport_form(s: Sample, domain: Domain, phi: Field, psi: Field, ang_vel: float):
    pos = domain(s)
    vel = velocity(pos, ang_vel)

    return psi(s) * wp.dot(grad(phi, s), vel)


# Upwind flux, on cell sides
@integrand
def upwind_transport_form(s: Sample, domain: Domain, phi: Field, psi: Field, ang_vel: float):
    pos = domain(s)
    vel = velocity(pos, ang_vel)
    vel_n = wp.dot(vel, normal(domain, s))

    return jump(phi, s) * (-average(psi, s) * vel_n + 0.5 * jump(psi, s) * wp.abs(vel_n))


# Symmetric-Interior-Penalty diffusion term (See Pietro Ern 2012)
@integrand
def sip_diffusion_form(
    s: Sample,
    domain: Domain,
    psi: Field,
    phi: Field,
):
    nor = normal(domain, s)
    penalty = measure_ratio(domain, s) * float(degree(psi) * degree(phi))

    return penalty * jump(phi, s) * jump(psi, s) - (
        wp.dot(grad_average(phi, s), nor) * jump(psi, s) + wp.dot(grad_average(psi, s), nor) * jump(phi, s)
    )


@integrand
def identity(s: Sample, phi: Field):
    return phi(s)


if __name__ == "__main__":
    wp.init()
    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=50)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--n_frames", type=int, default=100)
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
    quadrature = RegularQuadrature(domain=domain, order=2 * args.degree)
    scalar_space = make_polynomial_space(
        geo,
        discontinuous=True,
        degree=args.degree,
        family=Polynomial.GAUSS_LEGENDRE,
    )

    # Right-hand-side
    phi0 = scalar_space.make_field()
    interpolate(initial_condition, dest=phi0)

    test = make_test(space=scalar_space, domain=domain)
    trial = make_trial(space=scalar_space, domain=domain)

    side_test = make_test(space=scalar_space, domain=sides)
    side_trial = make_trial(space=scalar_space, domain=sides)

    matrix_inertia = integrate(
        inertia_form,
        quadrature=quadrature,
        fields={"phi": trial, "psi": test},
        values={"dt": dt},
    )

    matrix_transport = integrate(
        transport_form,
        fields={"phi": trial, "psi": test},
        values={"ang_vel": args.ang_vel},
    )

    bsr_axpy(
        integrate(
            upwind_transport_form,
            fields={"phi": side_trial, "psi": side_test},
            values={"ang_vel": args.ang_vel},
        ),
        y=matrix_transport,
    )

    matrix_diffusion = integrate(
        diffusion_form,
        fields={"u": trial, "v": test},
    )
    bsr_axpy(
        integrate(
            sip_diffusion_form,
            fields={"phi": side_trial, "psi": side_test},
        ),
        y=matrix_diffusion,
    )

    matrix = matrix_inertia
    bsr_axpy(x=matrix_transport, y=matrix)
    bsr_axpy(x=matrix_diffusion, y=matrix, alpha=args.viscosity)

    matrix_solve = factorized(bsr_to_scipy(matrix))

    results = [phi0.dof_values.numpy()]
    phik = phi0
    for k in range(args.n_frames):
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

    # Convert to continuous for visualization
    viz_space = make_polynomial_space(geo, degree=args.degree)
    phi_viz = viz_space.make_field()

    def animate(i):
        ax.clear()
        phik.dof_values = results[i]
        interpolate(identity, fields={"phi": phik}, dest=phi_viz)

        return plot_surface(phi_viz, axes=ax)

    anim = animation.FuncAnimation(
        ax.figure,
        animate,
        interval=30,
        blit=False,
        frames=len(results),
    )
    plt.show()
