# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Distortion Energy
#
# This example illustrates using a Newton loop to minimize distortion of a
# 3D surface (u,v) parameterization under a Symmetric Dirichlet energy,
#
# E(F) = 1/2 |F|^2 + |F^{-1}|^2
#
# with F := dx/du
###########################################################################

import numpy as np

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem


@fem.integrand
def distortion_gradient_form(s: fem.Sample, u_cur: fem.Field, v: fem.Field):
    # Symmetric Dirichlet energy gradient (linear form)
    # E = 1/2 (F:F + F^-T:F^-T)

    F = fem.grad(u_cur, s)

    F_inv_sq = wp.inverse(F * wp.transpose(F))
    F_inv = F_inv_sq * F

    dE_dF = F - F_inv_sq * F_inv

    return wp.ddot(fem.grad(v, s), dE_dF)


@fem.integrand
def distortion_hessian_form(s: fem.Sample, u_cur: fem.Field, u: fem.Field, v: fem.Field):
    # Symmetric Dirichlet energy approximate hessian (bilinear form)

    # F:F term
    H = wp.ddot(fem.grad(v, s), fem.grad(u, s))

    # F^-T:F^-T term
    F = fem.grad(u_cur, s)
    F_inv_sq = wp.inverse(F * wp.transpose(F))

    # Gauss--Newton (ignore F^-2 derivative)
    H += wp.ddot(F_inv_sq * fem.grad(v, s), F_inv_sq * F_inv_sq * fem.grad(u, s))

    return H


@fem.integrand
def initial_guess(
    s: fem.Sample,
    domain: fem.Domain,
):
    # initialization for UV parameter
    x = domain(s)
    return wp.vec2(x[0], x[1])


@fem.integrand
def boundary_projector_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
):
    # Fix a single point
    # (underconstrained, solution up to a rotation in UV space)
    w = wp.where(s.qp_index == 0, 1.0, 0.0)
    return w * wp.dot(u(s), v(s))


@fem.integrand
def checkerboard(s: fem.Sample, domain: fem.Domain, u: fem.Field):
    # checkerboard pattern for parameter visualization
    u_s = u(s)
    return wp.sign(wp.cos(16.0 * u_s[0]) * wp.sin(16.0 * u_s[1]))


class Example:
    def __init__(
        self,
        quiet=False,
        degree=2,
        resolution=25,
        mesh="grid",
        nonconforming_stresses=False,
    ):
        self._quiet = quiet

        def deform_along_z(positions, z_scale=1.0):
            pos = positions.numpy()
            pos_z = z_scale * np.cos(3.0 * pos[:, 0]) * np.sin(4.0 * pos[:, 1])
            pos = np.hstack((pos, np.expand_dims(pos_z, axis=1)))
            return wp.array(pos, dtype=wp.vec3)

        # Grid or mesh geometry
        if mesh == "tri":
            positions, tri_vidx = fem_example_utils.gen_trimesh(res=wp.vec2i(resolution))
            self._uv_geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=wp.zeros_like(positions))

            positions = deform_along_z(positions)
            self._geo = fem.Trimesh3D(tri_vertex_indices=tri_vidx, positions=positions)
        elif mesh == "quad":
            positions, quad_vidx = fem_example_utils.gen_quadmesh(res=wp.vec2i(resolution))
            self._uv_geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=wp.zeros_like(positions))

            positions = deform_along_z(positions)
            self._geo = fem.Quadmesh3D(quad_vertex_indices=quad_vidx, positions=positions)
        else:
            positions, quad_vidx = fem_example_utils.gen_quadmesh(res=wp.vec2i(resolution))
            self._uv_geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=wp.zeros_like(positions))

            undef_positions = deform_along_z(positions, z_scale=0.0)
            flat_geo = fem.Quadmesh3D(quad_vertex_indices=quad_vidx, positions=undef_positions)

            deformation_field = fem.make_discrete_field(fem.make_polynomial_space(flat_geo, dtype=wp.vec3))
            deformation_field.dof_values = deform_along_z(positions)

            self._geo = deformation_field.make_deformed_geometry(relative=False)

        # parameter space
        self._u_space = fem.make_polynomial_space(self._geo, degree=degree, dtype=wp.vec2)
        self._u_field = self._u_space.make_field()
        self._du_field = self._u_space.make_field()
        fem.interpolate(initial_guess, dest=self._u_field)

        # scalar parameter visualization function
        viz_space = fem.make_polynomial_space(self._geo, degree=3, dtype=float)
        self.viz_field = viz_space.make_field()
        # For visualization of uv in 2D space
        uv_space = fem.make_polynomial_space(self._uv_geo, degree=degree, dtype=wp.vec2)
        self._uv_field = uv_space.make_field()

        self.renderer = fem_example_utils.Plot()

    def step(self):
        boundary = fem.BoundarySides(self._geo)
        domain = fem.Cells(geometry=self._geo)

        # Parameter boundary conditions
        u_bd_test = fem.make_test(space=self._u_space, domain=boundary)
        u_bd_trial = fem.make_trial(space=self._u_space, domain=boundary)
        u_bd_matrix = fem.integrate(
            boundary_projector_form, fields={"u": u_bd_trial, "v": u_bd_test}, nodal=True, output_dtype=float
        )
        fem.normalize_dirichlet_projector(u_bd_matrix)

        u_test = fem.make_test(space=self._u_space, domain=domain)
        u_trial = fem.make_trial(space=self._u_space, domain=domain)

        # Newton iterations (without line-search for simplicity)
        for _newton_iteration in range(10):
            u_matrix = fem.integrate(
                distortion_hessian_form, fields={"u_cur": self._u_field, "u": u_trial, "v": u_test}, output_dtype=float
            )

            u_rhs = fem.integrate(
                distortion_gradient_form, fields={"u_cur": self._u_field, "v": u_test}, output_dtype=wp.vec2
            )

            fem.project_linear_system(u_matrix, u_rhs, u_bd_matrix, normalize_projector=False)

            # Solve for uv increment
            du = self._du_field.dof_values
            du.zero_()
            fem_example_utils.bsr_cg(u_matrix, b=u_rhs, x=du, quiet=self._quiet)

            # Accumulate to UV field
            fem.utils.array_axpy(x=du, y=self._u_field.dof_values, alpha=-1.0, beta=1.0)

    def render(self):
        # Visualization
        fem.interpolate(checkerboard, fields={"u": self._u_field}, dest=self.viz_field)

        self._uv_field.dof_values = wp.clone(self._u_field.dof_values)

        self.renderer.add_field("pattern", self.viz_field)
        self.renderer.add_field("uv", self._uv_field)


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=25, help="Grid resolution.")
    parser.add_argument("--degree", type=int, default=1, help="Polynomial degree of shape functions.")
    parser.add_argument("--mesh", choices=("tri", "quad", "deformed"), default="tri", help="Mesh type")
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
            mesh=args.mesh,
        )
        example.step()
        example.render()

        if not args.headless:
            example.renderer.plot(options={"uv": {"displacement": {}}})
