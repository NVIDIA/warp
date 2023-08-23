"""
This advanced example computes a 3D weakly-compressible flow around a deforming mesh, including:
  - defining active cells from a mask, and restricting the computation domain to those
  - utilizing the PicQuadrature to integrate over unstructured particles 

It can be used from trasnferring USD prims from one reference mesh to another
"""

import argparse
import os
import pathlib

from typing import List

from pxr import Usd, UsdGeom

import warp as wp
import numpy as np

from warp.utils import array_inner

from warp.fem.types import *
from warp.fem.geometry import Grid3D, ExplicitGeometryPartition
from warp.fem.field import make_test, make_trial, TestField, NodalField
from warp.fem.space import make_polynomial_space, make_space_partition
from warp.fem.domain import Cells
from warp.fem.integrate import integrate, interpolate
from warp.fem.operator import integrand, D, div, lookup, grad
from warp.fem.quadrature import PicQuadrature, RegularQuadrature

from warp.sparse import bsr_copy, bsr_transposed, bsr_mm, bsr_mv
from warp.utils import array_cast

from bsr_utils import bsr_cg

wp.set_module_options({"enable_backward": False})


@integrand
def vel_from_particles_form(s: Sample, particle_vel: wp.array(dtype=wp.vec3), v: Field):
    vel = particle_vel[s.qp_index]
    return wp.dot(vel, v(s))


@integrand
def set_cloth_displacement(s: Sample, points: wp.array(dtype=wp.vec3), u: Field):
    points[s.qp_index] = points[s.qp_index] + u(s)
    return 0.0


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
def div_form(s: Sample, u: Field, q: Field):
    return q(s) * div(u, s)


@integrand
def volume_form(s: Sample, u: Field, q: Field):
    Id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    J = wp.determinant(grad(u, s) + Id) - 1.0

    # forgive past volume gain
    return q(s) * wp.min(0.0, J)


@integrand
def cell_activity(s: Sample, domain: Domain, mesh: wp.uint64, interior_bandwidth: float, exterior_bandwidth: float):
    pos = domain(s)

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    max_dist = wp.max(exterior_bandwidth, interior_bandwidth)

    if not wp.mesh_query_point(mesh, pos, max_dist, sign, face_index, face_u, face_v):
        return 0.0

    cp = wp.mesh_eval_position(mesh, face_index, face_u, face_v)

    dist = wp.length(pos - cp)

    if (sign < 0.0 and dist > interior_bandwidth) or (sign >= 0.0 and dist > exterior_bandwidth):
        return 0.0

    return 1.0


@wp.struct
class MeshClosestPoint:
    face_index: wp.array(dtype=int)
    face_u: wp.array(dtype=float)
    face_v: wp.array(dtype=float)
    sign: wp.array(dtype=float)


@integrand
def cache_mesh_closest_point(s: Sample, domain: Domain, mesh: wp.uint64, max_distance: float, cp: MeshClosestPoint):
    pos = domain(s)

    i = s.qp_index

    q_face_index = int(0)
    q_face_u = float(0.0)
    q_face_v = float(0.0)
    q_sign = float(0.0)

    if wp.mesh_query_point(mesh, pos, max_distance, q_sign, q_face_index, q_face_u, q_face_v):
        cp.sign[i] = q_sign
        cp.face_index[i] = q_face_index
        cp.face_u[i] = q_face_u
        cp.face_v[i] = q_face_v
    else:
        cp.face_index[i] = -1

    return 0.0


@integrand
def boundary_mass_form(s: Sample, domain: Domain, mesh: wp.uint64, u: Field, v: Field, cp: MeshClosestPoint):
    i = s.qp_index

    if cp.face_index[i] < 0:
        return 0.0

    pos = wp.mesh_eval_position(mesh, cp.face_index[i], cp.face_u[i], cp.face_v[i])
    sample = lookup(domain, pos)

    if sample.element_index == s.element_index:
        # in same cell, mmove the quadrature point
        bd_sample = Sample(
            sample.element_index, sample.element_coords, s.qp_index, s.qp_weight, s.test_dof, s.trial_dof
        )
        return wp.dot(u(bd_sample), v(bd_sample))
    elif cp.sign[i] < 0.0:
        return wp.dot(u(s), v(s))

    return 0.0


@integrand
def boundary_vel_form(s: Sample, domain: Domain, mesh: wp.uint64, v: Field, cp: MeshClosestPoint):
    i = s.qp_index

    if cp.face_index[i] < 0:
        return 0.0

    pos = wp.mesh_eval_position(mesh, cp.face_index[i], cp.face_u[i], cp.face_v[i])
    vel = wp.mesh_eval_velocity(mesh, cp.face_index[i], cp.face_u[i], cp.face_v[i])

    sample = lookup(domain, pos)

    if sample.element_index == s.element_index:
        # in same cell, mmove the quadrature point
        bd_sample = Sample(
            sample.element_index, sample.element_coords, s.qp_index, s.qp_weight, s.test_dof, s.trial_dof
        )

        return wp.dot(vel, v(bd_sample))
    elif cp.sign[i] < 0.0:
        return wp.dot(vel, v(s))

    return 0.0


@integrand
def bilinear_form(
    s: Sample, domain: Domain, mesh: wp.uint64, u: Field, v: Field, nu: float, bd_strength: float, cp: MeshClosestPoint
):
    return viscosity_form(s, u, v, nu) + bd_strength * boundary_mass_form(s, domain, mesh, u, v, cp)


@wp.kernel
def inverse_array_kernel(m: wp.array(dtype=wp.float32)):
    m[wp.tid()] = 1.0 / m[wp.tid()]


# triangulate a list of polygon face indices
def triangulate(face_counts, face_indices):
    num_tris = np.sum(np.subtract(face_counts, 2))
    num_tri_vtx = num_tris * 3
    tri_indices = np.zeros(num_tri_vtx, dtype=int)
    ctr = 0
    wedgeIdx = 0

    for nb in face_counts:
        for i in range(nb - 2):
            tri_indices[ctr] = face_indices[wedgeIdx]
            tri_indices[ctr + 1] = face_indices[wedgeIdx + i + 1]
            tri_indices[ctr + 2] = face_indices[wedgeIdx + i + 2]
            ctr += 3
        wedgeIdx += nb

    return tri_indices


def find_meshes(stage: Usd.Stage):
    meshes = []
    for prim in stage.Traverse():
        mesh = UsdGeom.Mesh(prim)
        if mesh:
            meshes.append(mesh)
    return meshes


def find_point_based_geoms(stage: Usd.Stage):
    meshes = []
    for prim in stage.Traverse():
        mesh = UsdGeom.PointBased(prim)
        if mesh:
            meshes.append(mesh)
    return meshes


def _read_attr_and_time(attr: Usd.Attribute):
    if not attr:
        raise ValueError("Invalid attribute")

    default_time = Usd.TimeCode.Default()
    at_default_time = attr.Get(default_time)
    if at_default_time:
        return at_default_time, default_time

    if attr.GetNumTimeSamples() == 0:
        raise ValueError(f"Attribute {attr} has no time samples nor value at default time")

    time = attr.GetTimeSamples()[0]
    return attr.Get(time), time


def _read_attr(attr: Usd.Attribute):
    return _read_attr_and_time(attr)[0]


def _get_biggest_mesh(matching_meshes):
    if not matching_meshes:
        return None
    return max(matching_meshes, key=lambda mesh: len(_read_attr(mesh.GetPointsAttr())))


def find_biggest_mesh(stage: Usd.Stage):
    return _get_biggest_mesh(find_meshes(stage))


def find_biggest_mesh_matching(stage: Usd.Stage, needles: List[str]):
    matching_meshes = [
        mesh for mesh in find_meshes(stage) if any(n in str(mesh.GetPrim().GetPath()).split("/")[-1] for n in needles)
    ]

    return _get_biggest_mesh(matching_meshes)


def wrap_geom(cloth_geom: UsdGeom.Mesh, u_field, domain):
    usd_points, usd_time = _read_attr_and_time(cloth_geom.GetPointsAttr())

    cloth_points = wp.array(np.array(usd_points), dtype=wp.vec3)
    cloth_volumes = wp.zeros(n=cloth_points.shape[0], dtype=float)
    cloth_pic_quadrature = PicQuadrature(domain, cloth_points, cloth_volumes)

    integrate(
        set_cloth_displacement,
        quadrature=cloth_pic_quadrature,
        values={"points": cloth_points},
        fields={"u": u_field},
    )

    cloth_geom.GetPointsAttr().Set(cloth_points.numpy(), usd_time)

    # Clear authored extents value if it exists
    cloth_geom.GetExtentAttr().Clear()


def integrate_linear_forms(
    quadrature: RegularQuadrature,
    u_field: NodalField,
    u_test: TestField,
    p_test: TestField,
    mesh: wp.Mesh,
    quadrature_cp: MeshClosestPoint,
):
    with wp.ScopedTimer("Integrate linear forms"):
        u_rhs = integrate(
            boundary_vel_form,
            quadrature=quadrature,
            fields={"v": u_test},
            values={"cp": quadrature_cp, "mesh": mesh.id},
            output_dtype=wp.vec3,
        )

        p_rhs = integrate(
            volume_form,
            quadrature=quadrature,
            fields={"q": p_test, "u": u_field},
            output_dtype=wp.vec(length=1, dtype=wp.float32),
        )

    return u_rhs, p_rhs


def assemble_system_matrices(
    args,
    u_matrix,
    inv_p_mass_matrix,
    div_matrix,
):
    with wp.ScopedTimer("Assemble system matrices"):
        div_matrix_t = bsr_transposed(div_matrix)

        volume_stiffness = 1.0 / args.compliance
        bsr_gradient_matrix = bsr_mm(div_matrix_t, inv_p_mass_matrix, alpha=volume_stiffness)

        bsr_stiffness_matrix = bsr_copy(u_matrix)
        bsr_mm(bsr_gradient_matrix, div_matrix, bsr_stiffness_matrix, alpha=1.0, beta=1.0)

    return bsr_stiffness_matrix, bsr_gradient_matrix


def assemble_system_rhs(
    args,
    quadrature: RegularQuadrature,
    u_field: NodalField,
    u_test: TestField,
    p_test: TestField,
    u_matrix,
    gradient_matrix,
    mesh: wp.Mesh,
    quadrature_cp: MeshClosestPoint,
):
    u_rhs, p_rhs = integrate_linear_forms(quadrature, u_field, u_test, p_test, mesh=mesh, quadrature_cp=quadrature_cp)

    with wp.ScopedTimer("Assemble system rhs"):
        u_values = u_field.dof_values

        # u_rhs = bd_strength*u_rhs - u_matrix @ u_values
        bsr_mv(A=u_matrix, x=u_values, y=u_rhs, alpha=-1, beta=args.bd_strength)

        # rhs = u_rhs - gradient_matrix @ p_rhs
        bsr_mv(A=gradient_matrix, x=p_rhs, y=u_rhs, alpha=-1, beta=1)

    return u_rhs


def solve_linear_system(stiffness_matrix, rhs, x):
    x.zero_()
    with wp.ScopedTimer("Solving with CG"):
        err, end_iter = bsr_cg(stiffness_matrix, b=rhs, x=x, max_iters=1000)

        print(f"CG result norm {np.sqrt(array_inner(x, x))},  error : {err}, end iter {end_iter}")


@wp.kernel
def _compute_aabb_kernel(points: wp.array(dtype=wp.vec3), bounds: wp.array(dtype=wp.vec3), bw: float):
    p = points[wp.tid()]
    p_min = p - wp.vec3(bw)
    p_max = p + wp.vec3(bw)
    wp.atomic_min(bounds, 0, p_min)
    wp.atomic_max(bounds, 1, p_max)


@wp.kernel
def _add_displacement_delta(
    u: wp.array(dtype=wp.vec3),
    du: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    u[i] = u[i] + du[i]


def solve_velocity_field(args, mesh: wp.Mesh):
    bounds = wp.array(
        [
            [1.0e8, 1.0e8, 1.0e8],
            [-1.0e8, -1.0e8, -1.0e8],
        ],
        dtype=wp.vec3,
    )
    wp.launch(kernel=_compute_aabb_kernel, dim=mesh.points.shape, inputs=[mesh.points, bounds, args.band_width])
    bounds = bounds.numpy()

    bounds_lo = wp.vec3(bounds[0])
    bounds_hi = wp.vec3(bounds[1])

    res_f = np.ceil((np.array(bounds_hi) - np.array(bounds_lo)) / args.cell_size)
    res = vec3i(np.array(res_f, dtype=int))
    # res = vec3i(50, 75, 15)
    # res = vec3i(30, 50, 10)

    geo = Grid3D(
        res=res,
        bounds_lo=bounds_lo,
        bounds_hi=bounds_hi,
    )

    # Disable cells that are interior or far from the mesh
    cell_space = make_polynomial_space(geo, degree=0)
    cell_diameter = np.linalg.norm(np.array(geo.cell_size))
    print(f"Cell size is {geo.cell_size}, diameter {cell_diameter}, resolution {res}")

    activity = cell_space.make_field()
    interpolate(
        cell_activity,
        dest=activity,
        values={
            "mesh": mesh.id,
            "exterior_bandwidth": args.band_width + cell_diameter,
            "interior_bandwidth": args.interior_band_width + 0.5 * cell_diameter,
        },
    )

    active_partition_flag = wp.empty(shape=activity.dof_values.shape, dtype=int)
    array_cast(in_array=activity.dof_values, out_array=active_partition_flag)
    active_partition = ExplicitGeometryPartition(geo, active_partition_flag)
    print("Active cells:", active_partition.cell_count())

    # Function spaces -- Q1 for vel, Q0 for pressure
    with wp.ScopedTimer("Build function spaces"):
        u_space = make_polynomial_space(geo, degree=1, dtype=wp.vec3)
        p_space = make_polynomial_space(geo, degree=0)
        active_space_partition = make_space_partition(
            space=u_space, geometry_partition=active_partition, with_halo=False
        )
        active_p_space_partition = make_space_partition(
            space=p_space, geometry_partition=active_partition, with_halo=False
        )

        domain = Cells(geometry=active_partition)

        u_test = make_test(space=u_space, space_partition=active_space_partition, domain=domain)
        u_trial = make_trial(space=u_space, space_partition=active_space_partition, domain=domain)

        p_test = make_test(space=p_space, space_partition=active_p_space_partition, domain=domain)
        p_trial = make_trial(space=p_space, space_partition=active_p_space_partition, domain=domain)

    quadrature = RegularQuadrature(domain, order=2)

    # Cache closest point query results at quadrature points
    with wp.ScopedTimer("Cache closest points"):
        quadrature_cp = MeshClosestPoint()
        quadrature_cp.sign = wp.empty(shape=(quadrature.total_point_count(),), dtype=float)
        quadrature_cp.face_u = wp.empty(shape=(quadrature.total_point_count(),), dtype=float)
        quadrature_cp.face_v = wp.empty(shape=(quadrature.total_point_count(),), dtype=float)
        quadrature_cp.face_index = wp.empty(shape=(quadrature.total_point_count(),), dtype=int)

        integrate(
            cache_mesh_closest_point,
            quadrature=quadrature,
            values={
                "max_distance": cell_diameter,
                "mesh": mesh.id,
                "cp": quadrature_cp,
            },
        )

    with wp.ScopedTimer("Integrate bilinear forms"):
        # Pressure-velocity coupling
        div_matrix = integrate(div_form, fields={"u": u_trial, "q": p_test}, output_dtype=wp.float32)

        # Pressure inverse mass matrix
        inv_p_mass_matrix = integrate(scalar_mass_form, fields={"p": p_trial, "q": p_test}, output_dtype=wp.float32)
        wp.launch(
            kernel=inverse_array_kernel,
            dim=inv_p_mass_matrix.values.shape,
            device=inv_p_mass_matrix.values.device,
            inputs=[inv_p_mass_matrix.values],
        )

        # Viscosity and velocity BC
        u_matrix = integrate(
            bilinear_form,
            quadrature=quadrature,
            fields={"u": u_trial, "v": u_test},
            values={
                "nu": args.viscosity,
                "bd_strength": args.bd_strength,
                "mesh": mesh.id,
                "cp": quadrature_cp,
            },
            output_dtype=wp.float32,
        )

    stiffness_matrix, gradient_matrix = assemble_system_matrices(args, u_matrix, inv_p_mass_matrix, div_matrix)

    u_field = u_space.make_field(space_partition=active_space_partition)
    du = wp.empty_like(u_field.dof_values)

    for ii in range(args.inner_iterations):
        rhs = assemble_system_rhs(
            args,
            quadrature,
            u_field,
            u_test,
            p_test,
            u_matrix,
            gradient_matrix,
            mesh=mesh,
            quadrature_cp=quadrature_cp,
        )

        solve_linear_system(stiffness_matrix, rhs=rhs, x=du)

        wp.launch(kernel=_add_displacement_delta, dim=du.shape[0], inputs=[u_field.dof_values, du])

    return u_field, domain


def transfer(args, ref_geom: UsdGeom.Mesh, tgt_geom: UsdGeom.Mesh, cloth_stages):
    mesh_counts = _read_attr(ref_geom.GetFaceVertexCountsAttr())
    mesh_indices = _read_attr(ref_geom.GetFaceVertexIndicesAttr())

    tri_indices = triangulate(mesh_counts, mesh_indices)

    mesh_points = wp.array(np.array(_read_attr(ref_geom.GetPointsAttr())), dtype=wp.vec3)
    mesh_indices = wp.array(np.array(tri_indices), dtype=int)

    mesh_velocities = wp.array(
        np.array(_read_attr(tgt_geom.GetPointsAttr())) - np.array(_read_attr(ref_geom.GetPointsAttr())), dtype=wp.vec3
    )

    mesh = wp.Mesh(points=mesh_points, velocities=mesh_velocities, indices=mesh_indices)

    u_field, domain = solve_velocity_field(args, mesh)

    # Move reference body
    wrap_geom(ref_geom, u_field, domain)

    # Move other garments
    for garment, cloth_stage in cloth_stages.items():
        for cloth_geom in find_point_based_geoms(cloth_stage):
            print(f"Transferring {cloth_geom.GetPrim().GetPath()} from '{garment}'")

            wrap_geom(cloth_geom, u_field, domain)


if __name__ == "__main__":
    wp.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("ref_body", help="Reference body USD file")
    parser.add_argument("tgt_body", help="Target body USD file")
    parser.add_argument("--out", "-o", default="out", help="Output directory")
    parser.add_argument(
        "--garments", "-g", type=str, nargs="+", metavar="ID", help="USD files contaning garments to transfer"
    )

    parser.add_argument("--iterations", "-i", type=int, default=1)
    parser.add_argument("--inner_iterations", "-ii", type=int, default=1)
    parser.add_argument("--cell_size", "-s", type=float, default=2.0)
    parser.add_argument("--band_width", "-w", type=float, default=6.0)
    parser.add_argument("--interior_band_width", "-iw", type=float, default=0.0)
    parser.add_argument("--viscosity", "-nu", type=float, default=100.0)
    parser.add_argument("--compliance", "-c", type=float, default=1.0)
    parser.add_argument("--bd_strength", type=float, default=1000.0)

    args = parser.parse_args()

    ref_body_stage = Usd.Stage.Open(args.ref_body)
    tgt_body_stage = Usd.Stage.Open(args.tgt_body)

    ref_geom = find_biggest_mesh_matching(ref_body_stage, ["skin", "body"])
    tgt_geom = find_biggest_mesh_matching(tgt_body_stage, ["skin", "body"])

    if not ref_geom:
        ref_geom = find_biggest_mesh(ref_body_stage)
    if not tgt_geom:
        tgt_geom = find_biggest_mesh(tgt_body_stage)

    cloth_stages = {garment: Usd.Stage.Open(garment) for garment in args.garments}

    for it in range(args.iterations):
        transfer(args, ref_geom, tgt_geom, cloth_stages)

    # write output files

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)

    out_file = os.path.join(args.out, os.path.basename(args.ref_body))
    ref_body_stage.Export(out_file)
    print(f"Wrote '{out_file}'")

    for garment, cloth_stage in cloth_stages.items():
        out_file = os.path.join(args.out, os.path.basename(garment))
        cloth_stage.Export(out_file)
        print(f"Wrote '{out_file}'")
