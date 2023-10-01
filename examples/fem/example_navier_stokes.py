"""
This example solves a 2D Navier-Stokes flow problem

  Du/dt -nu D(u) + grad p = 0
  Div u = 0

with (hard) velocity-Dirichlet boundary conditions
and using semi-Lagrangian advection
"""

import argparse

import warp as wp

import numpy as np

from warp.fem import Field, Sample, Domain
from warp.fem import Grid2D, Trimesh2D
from warp.fem import make_test, make_trial
from warp.fem import make_polynomial_space
from warp.fem import RegularQuadrature
from warp.fem import Cells, BoundarySides
from warp.fem import integrate
from warp.fem import integrand, D, div, lookup
from warp.fem import project_linear_system, normalize_dirichlet_projector
from warp.fem.utils import array_axpy

from warp.sparse import bsr_mm, bsr_mv, bsr_copy

from bsr_utils import bsr_to_scipy
from plot_utils import plot_grid_streamlines, plot_velocities
from mesh_utils import gen_trimesh

from scipy.sparse import bmat
from scipy.sparse.linalg import factorized

import matplotlib.pyplot as plt
import matplotlib.animation as animation


@integrand
def u_boundary_value(s: Sample, domain: Domain, v: Field, top_vel: float):
    # Horizontal velocity on top of domain, zero elsewhere
    if domain(s)[1] == 1.0:
        return wp.dot(wp.vec2f(top_vel, 0.0), v(s))

    return wp.dot(wp.vec2f(0.0, 0.0), v(s))


@integrand
def mass_form(
    s: Sample,
    u: Field,
    v: Field,
):
    return wp.dot(u(s), v(s))


@integrand
def inertia_form(s: Sample, u: Field, v: Field, dt: float):
    return mass_form(s, u, v) / dt


@integrand
def viscosity_form(s: Sample, u: Field, v: Field, nu: float):
    return 2.0 * nu * wp.ddot(D(u, s), D(v, s))


@integrand
def viscosity_and_inertia_form(s: Sample, u: Field, v: Field, dt: float, nu: float):
    return inertia_form(s, u, v, dt) + viscosity_form(s, u, v, nu)


@integrand
def transported_inertia_form(s: Sample, domain: Domain, u: Field, v: Field, dt: float):
    pos = domain(s)
    vel = u(s)

    conv_pos = pos - 0.5 * vel * dt
    conv_s = lookup(domain, conv_pos, s)
    conv_vel = u(conv_s)

    conv_pos = conv_pos - 0.5 * conv_vel * dt
    conv_vel = u(lookup(domain, conv_pos, conv_s))

    return wp.dot(conv_vel, v(s)) / dt


@integrand
def div_form(
    s: Sample,
    u: Field,
    q: Field,
):
    return -q(s) * div(u, s)


if __name__ == "__main__":
    wp.init()
    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=25)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--n_frames", type=int, default=1000)
    parser.add_argument("--top_velocity", type=float, default=1.0)
    parser.add_argument("--Re", type=float, default=1000.0)
    parser.add_argument("--tri_mesh", action="store_true", help="Use a triangular mesh")
    args = parser.parse_args()

    if args.tri_mesh:
        positions, tri_vidx = gen_trimesh(res=wp.vec2i(args.resolution))
        geo = Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
    else:
        geo = Grid2D(res=wp.vec2i(args.resolution))

    boundary = BoundarySides(geo)

    viscosity = args.top_velocity / args.Re
    dt = 1.0 / args.resolution

    domain = Cells(geometry=geo)

    # Functions spaces: Q(d)-Q(d-1)
    u_degree = args.degree
    u_space = make_polynomial_space(geo, degree=u_degree, dtype=wp.vec2)
    p_space = make_polynomial_space(geo, degree=u_degree - 1)
    quadrature = RegularQuadrature(domain=domain, order=2 * u_degree)

    # Viscosity and inertia
    u_test = make_test(space=u_space, domain=domain)
    u_trial = make_trial(space=u_space, domain=domain)

    u_matrix = integrate(
        viscosity_and_inertia_form,
        fields={"u": u_trial, "v": u_test},
        values={"nu": viscosity, "dt": dt},
    )

    # Pressure-velocity coupling
    p_test = make_test(space=p_space, domain=domain)
    div_matrix = integrate(div_form, fields={"u": u_trial, "q": p_test})

    # Enforcing the Dirichlet boundary condition the hard way;
    # build projector for velocity left- and right-hand-sides
    u_bd_test = make_test(space=u_space, domain=boundary)
    u_bd_trial = make_trial(space=u_space, domain=boundary)
    u_bd_projector = integrate(mass_form, fields={"u": u_bd_trial, "v": u_bd_test}, nodal=True)
    u_bd_value = integrate(
        u_boundary_value,
        fields={"v": u_bd_test},
        values={"top_vel": args.top_velocity},
        nodal=True,
        output_dtype=wp.vec2d,
    )

    normalize_dirichlet_projector(u_bd_projector, u_bd_value)

    u_bd_rhs = wp.zeros_like(u_bd_value)
    project_linear_system(u_matrix, u_bd_rhs, u_bd_projector, u_bd_value, normalize_projector=False)

    # div_bd_rhs = div_matrix * u_bd_rhs
    div_bd_rhs = wp.zeros(shape=(div_matrix.nrow,), dtype=div_matrix.scalar_type)
    bsr_mv(div_matrix, u_bd_rhs, y=div_bd_rhs)

    # div_matrix = div_matrix - div_matrix * bd_projector
    bsr_mm(x=bsr_copy(div_matrix), y=u_bd_projector, z=div_matrix, alpha=-1.0, beta=1.0)

    # Assemble saddle system with Scipy
    div_matrix = bsr_to_scipy(div_matrix)
    u_matrix = bsr_to_scipy(u_matrix)
    div_bd_rhs = div_bd_rhs.numpy()

    ones = np.ones(shape=(p_space.node_count(), 1), dtype=float)
    saddle_system = bmat(
        [
            [u_matrix, div_matrix.transpose(), None],
            [div_matrix, None, ones],
            [None, ones.transpose(), None],
        ],
    )

    with wp.ScopedTimer("LU factorization"):
        solve_saddle = factorized(saddle_system)

    u_k = u_space.make_field()
    u_rhs = wp.zeros_like(u_bd_rhs)

    results = [u_k.dof_values.numpy()]

    for k in range(args.n_frames):
        print("Solving step", k)

        u_inertia_rhs = integrate(
            transported_inertia_form,
            quadrature=quadrature,
            fields={"u": u_k, "v": u_test},
            values={"dt": dt},
            output_dtype=wp.vec2d,
        )
        # u_rhs = (I - P) * u_inertia_rhs + u_bd_rhs
        bsr_mv(u_bd_projector, u_inertia_rhs, y=u_rhs, alpha=-1.0, beta=0.0)
        array_axpy(x=u_inertia_rhs, y=u_rhs, alpha=1.0, beta=1.0)
        array_axpy(x=u_bd_rhs, y=u_rhs, alpha=1.0, beta=1.0)

        # Assemble scipy saddle system rhs
        saddle_rhs = np.zeros(saddle_system.shape[0])
        u_slice = slice(0, 2 * u_space.node_count())
        p_slice = slice(2 * u_space.node_count(), 2 * u_space.node_count() + p_space.node_count())
        saddle_rhs[u_slice] = u_rhs.numpy().flatten()
        saddle_rhs[p_slice] = div_bd_rhs

        x = solve_saddle(saddle_rhs)

        # Extract result
        x_u = x[u_slice].reshape((-1, 2))
        results.append(x_u)

        u_k.dof_values = x_u
        # p_field.dof_values = x[p_slice]

    if isinstance(geo, Grid2D):
        plot_grid_streamlines(u_k)

    quiver = plot_velocities(u_k)
    ax = quiver.axes

    def animate(i):
        ax.clear()
        u_k.dof_values = results[i]
        return plot_velocities(u_k, axes=ax)

    anim = animation.FuncAnimation(
        ax.figure,
        animate,
        interval=30,
        blit=False,
        frames=len(results),
    )
    plt.show()
