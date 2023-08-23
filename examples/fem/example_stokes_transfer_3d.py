"""
This example computes a 3D weakly-compressible Stokes flow around a moving object, including:
  - defining active cells from a mask, and restricting the computation domain to those
  - utilizing the PicQuadrature to integrate over unstructured particles 
"""

import warp as wp
import numpy as np

from warp.fem.types import *
from warp.fem.geometry import Grid3D, ExplicitGeometryPartition
from warp.fem.field import make_test, make_trial
from warp.fem.space import make_polynomial_space, make_space_partition
from warp.fem.domain import Cells
from warp.fem.integrate import integrate, interpolate
from warp.fem.operator import integrand, D, div
from warp.fem.quadrature import PicQuadrature
from warp.fem.utils import array_axpy

from warp.sparse import bsr_mv

from plot_utils import plot_3d_scatter, plot_3d_velocities
from bsr_utils import bsr_cg
from example_stokes_transfer import inverse_array_kernel

from warp.utils import array_cast
from warp.sparse import bsr_transposed, bsr_mm, bsr_axpy

import matplotlib.pyplot as plt


@integrand
def vel_from_particles_form(s: Sample, particle_vel: wp.array(dtype=wp.vec3), v: Field):
    vel = particle_vel[s.qp_index]
    return wp.dot(vel, v(s))


@integrand
def viscosity_form(s: Sample, u: Field, v: Field, nu: float):
    return nu * wp.ddot(D(u, s), D(v, s))


@integrand
def mass_form(
    s: Sample,
    u: Field,
    v: Field,
):
    return wp.dot(u(s), v(s))


@integrand
def scalar_mass_form(
    s: Sample,
    p: Field,
    q: Field,
):
    return p(s) * q(s)


@integrand
def div_form(
    s: Sample,
    u: Field,
    q: Field,
):
    return q(s) * div(u, s)


@integrand
def cell_activity(s: Sample, domain: Domain, c1: wp.vec3, c2: wp.vec3, radius: float):
    pos = domain(s)
    if wp.length(pos - c1) < radius:
        return 0.0
    if wp.length(pos - c2) < radius:
        return 0.0
    return 1.0


if __name__ == "__main__":
    wp.init()
    wp.set_module_options({"enable_backward": False})

    res = 20
    geo = Grid3D(
        res=vec3i(res, res, res),
        bounds_lo=wp.vec3(0.0, 0.0, 0.0),
        bounds_hi=wp.vec3(1.0, 1.0, 1.0),
    )

    vel = 1.0
    viscosity = 100.0
    compliance = 0.01
    bd_strength = 100000.0

    # Generate particles defining the transfer displacement
    circle_radius = 0.15
    c1_center = np.array([0.25, 0.5, 0.5])
    c2_center = np.array([0.75, 0.5, 0.5])

    particles_per_side = int(4 * circle_radius * res)

    particles_x = np.linspace(-circle_radius, circle_radius, particles_per_side)
    cube_particles = np.array([[px, py, pz] for px in particles_x for py in particles_x for pz in particles_x])

    particles_per_circle = particles_per_side**3
    n_particles = 2 * particles_per_circle
    particles = np.empty((n_particles, 3), dtype=float)

    particles[:particles_per_circle, :] = cube_particles + c1_center
    particles[particles_per_circle:, :] = cube_particles + c2_center

    particle_areas = np.ones(n_particles) * circle_radius * circle_radius / (res * res)
    particle_velocities = np.zeros_like(particles)
    particle_velocities[:particles_per_circle, 0] = vel
    particle_velocities[particles_per_circle:, 0] = -vel

    particles = wp.array(particles, dtype=wp.vec3)
    particle_areas = wp.array(particle_areas, dtype=float)
    particle_velocities = wp.array(particle_velocities, dtype=wp.vec3)

    # Disable cells that are interior to the circles
    cell_space = make_polynomial_space(geo, degree=0)
    activity = cell_space.make_field()
    interpolate(
        cell_activity,
        dest=activity,
        values={"c1": c1_center, "c2": c2_center, "radius": circle_radius - 1.0 / res},
    )

    active_partition = ExplicitGeometryPartition(geo, wp.array(activity.dof_values.numpy(), dtype=int))
    print("Active cells:", active_partition.cell_count())

    # Function spaces -- Q1 for vel, Q0 for pressure
    u_space = make_polynomial_space(geo, degree=1, dtype=wp.vec3)
    p_space = make_polynomial_space(geo, degree=0)
    active_space_partition = make_space_partition(space=u_space, geometry_partition=active_partition)
    active_p_space_partition = make_space_partition(space=p_space, geometry_partition=active_partition)

    domain = Cells(geometry=active_partition)
    pic_quadrature = PicQuadrature(domain, particles, particle_areas)

    # Boundary condition on particles
    u_test = make_test(space=u_space, space_partition=active_space_partition, domain=domain)
    u_trial = make_trial(space=u_space, space_partition=active_space_partition, domain=domain)

    u_rhs = integrate(
        vel_from_particles_form,
        quadrature=pic_quadrature,
        fields={"v": u_test},
        values={"particle_vel": particle_velocities},
        output_dtype=wp.vec3d
    )

    u_bd_matrix = integrate(mass_form, quadrature=pic_quadrature, fields={"u": u_trial, "v": u_test})

    # Viscosity
    u_visc_matrix = integrate(
        viscosity_form,
        fields={"u": u_trial, "v": u_test},
        values={"nu": viscosity}
    )

    # Pressure-velocity coupling
    p_test = make_test(space=p_space, space_partition=active_p_space_partition, domain=domain)
    p_trial = make_trial(space=p_space, space_partition=active_p_space_partition, domain=domain)

    div_matrix = integrate(div_form, fields={"u": u_trial, "q": p_test})
    inv_p_mass_matrix = integrate(scalar_mass_form, fields={"p": p_trial, "q": p_test})
    wp.launch(
        kernel=inverse_array_kernel,
        dim=inv_p_mass_matrix.values.shape,
        device=inv_p_mass_matrix.values.device,
        inputs=[inv_p_mass_matrix.values],
    )

    # Assemble linear system
    u_matrix = u_visc_matrix
    bsr_axpy(u_bd_matrix, u_matrix, alpha=bd_strength)

    div_matrix_t = bsr_transposed(div_matrix)
    gradient_matrix = bsr_mm(div_matrix_t, inv_p_mass_matrix)
    bsr_mm(gradient_matrix, div_matrix, u_matrix, alpha=1.0 / compliance, beta=1.0)

    array_axpy(u_rhs, u_rhs, alpha=0.0, beta=bd_strength)

    # Solve for displacement
    u_res = wp.zeros_like(u_rhs)
    bsr_cg(u_matrix, x=u_res, b=u_rhs)

    # Recompute pressure
    p_res = wp.zeros(n=active_p_space_partition.node_count(), dtype=wp.float64)
    p_tmp = wp.empty_like(p_res)
    bsr_mv(A=div_matrix, x=u_res, y=p_tmp)
    bsr_mv(A=inv_p_mass_matrix, x=p_tmp, y=p_res, alpha=-1)

    # Display result
    u_field = u_space.make_field()
    p_field = p_space.make_field()

    u_nodes = wp.indexedarray(u_field.dof_values, indices=active_space_partition.space_node_indices())
    p_nodes = wp.indexedarray(p_field.dof_values, indices=active_p_space_partition.space_node_indices())

    array_cast(in_array=u_res, out_array=u_nodes)
    array_cast(in_array=p_res, out_array=p_nodes)

    plot_3d_scatter(p_field)
    plot_3d_velocities(u_field)

    plt.show()
