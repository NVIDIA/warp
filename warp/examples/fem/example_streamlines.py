# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Example Streamlines
#
# Shows how to generate 3D streamlines by tracing through a velocity field
# using the `warp.fem.lookup` operator.
# Also illustrates using `warp.fem.Subdomain` to define subsets of elements.
#
###########################################################################

import numpy as np

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem
from warp.examples.fem.example_apic_fluid import divergence_form, solve_incompressibility


@fem.integrand
def classify_boundary_sides(
    s: fem.Sample,
    domain: fem.Domain,
    outflow: wp.array(dtype=int),
    freeslip: wp.array(dtype=int),
    inflow: wp.array(dtype=int),
):
    x = fem.position(domain, s)
    n = fem.normal(domain, s)

    if n[0] < -0.5:
        # left side
        inflow[s.qp_index] = 1
    elif n[0] > 0.5:
        if x[1] > 0.33 or x[2] < 0.33:
            # right side, top
            freeslip[s.qp_index] = 1
        else:
            # right side, bottom
            outflow[s.qp_index] = 1
    else:
        freeslip[s.qp_index] = 1


@fem.integrand
def inflow_velocity(
    s: fem.Sample,
    domain: fem.Domain,
):
    n = fem.normal(domain, s)
    return -n


@fem.integrand
def noslip_projector_form(
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    return wp.dot(u(s), v(s))


@fem.integrand
def freeslip_projector_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
):
    n = fem.normal(domain, s)
    return wp.dot(u(s), n) * wp.dot(n, v(s))


@fem.integrand
def mass_form(
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    return wp.dot(u(s), v(s))


@fem.integrand
def spawn_streamlines(s: fem.Sample, domain: fem.Domain, jitter: float):
    rng = wp.rand_init(s.qp_index)
    random_offset = wp.vec3(wp.randf(rng), wp.randf(rng), wp.randf(rng)) - wp.vec3(0.5)

    # remove jistter along normal
    n = fem.normal(domain, s)
    random_offset -= wp.dot(random_offset, n) * n

    return domain(s) + jitter * random_offset


@fem.integrand
def gen_streamlines(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    spawn_points: wp.array(dtype=wp.vec3),
    point_count: int,
    dx: float,
    pos: wp.array2d(dtype=wp.vec3),
    speed: wp.array2d(dtype=float),
):
    idx = s.qp_index

    p = spawn_points[idx]
    s = fem.lookup(domain, p)
    for k in range(point_count):
        v = u(s)
        pos[idx, k] = p
        speed[idx, k] = wp.length(v)

        flow_dir = wp.normalize(v)
        adv_p = p + flow_dir * dx
        adv_s = fem.lookup(domain, adv_p, s)

        if adv_s.element_index != fem.NULL_ELEMENT_INDEX:
            # if the lookup result position is different from adv_p,
            # it means we have been projected back onto the domain;
            # align back with flow and terminate streamline
            new_p = domain(adv_s)
            if wp.length_sq(new_p - adv_p) > 0.000001:
                p = p + wp.dot(new_p - p, flow_dir) * flow_dir
                s = fem.lookup(domain, p, s)
                dx = 0.0
            else:
                s = adv_s
                p = new_p


class Example:
    def __init__(self, quiet=False, degree=2, resolution=16, mesh="grid", headless: bool = False):
        self._quiet = quiet
        self._degree = degree

        self._streamline_dx = 0.5 / resolution
        self._streamline_point_count = 4 * resolution

        res = wp.vec3i(resolution)

        if mesh == "tet":
            pos, tet_vtx_indices = fem_example_utils.gen_tetmesh(
                res=res,
            )
            self._geo = fem.Tetmesh(tet_vtx_indices, pos, build_bvh=True)
        elif mesh == "hex":
            pos, hex_vtx_indices = fem_example_utils.gen_hexmesh(
                res=res,
            )
            self._geo = fem.Hexmesh(hex_vtx_indices, pos, assume_parallelepiped_cells=True, build_bvh=True)
        elif mesh == "nano":
            volume = fem_example_utils.gen_volume(
                res=res,
            )
            self._geo = fem.Nanogrid(volume)
        else:
            self._geo = fem.Grid3D(
                res=res,
            )

        # Mark sides with boundary conditions that should apply
        boundary = fem.BoundarySides(self._geo)
        inflow_mask = wp.zeros(shape=boundary.element_count(), dtype=int)
        freeslip_mask = wp.zeros(shape=boundary.element_count(), dtype=int)
        outflow_mask = wp.zeros(shape=boundary.element_count(), dtype=int)

        fem.interpolate(
            classify_boundary_sides,
            quadrature=fem.RegularQuadrature(boundary, order=0),
            values={"outflow": outflow_mask, "freeslip": freeslip_mask, "inflow": inflow_mask},
        )

        self._inflow = fem.Subdomain(boundary, element_mask=inflow_mask)
        self._freeslip = fem.Subdomain(boundary, element_mask=freeslip_mask)
        self._outflow = fem.Subdomain(boundary, element_mask=outflow_mask)

        self.plot = fem_example_utils.Plot()

        self.renderer = None
        if not headless:
            try:
                self.renderer = wp.render.OpenGLRenderer(
                    camera_pos=(2.0, 0.5, 3.0),
                    camera_front=(-0.66, 0.0, -1.0),
                    draw_axis=False,
                )
            except Exception as err:
                wp.utils.warn(f"Could not initialize OpenGL renderer: {err}")
                pass

    def step(self):
        self._generate_incompressible_flow()

        # first generate spawn points for the streamlines
        # we do this by regularly sampling the inflow boundary with a small amount of jitter
        streamline_spawn = fem.RegularQuadrature(
            domain=self._inflow, order=self._degree, family=fem.Polynomial.GAUSS_LEGENDRE
        )
        n_streamlines = streamline_spawn.total_point_count()
        spawn_points = wp.empty(dtype=wp.vec3, shape=n_streamlines)

        jitter_amount = self._streamline_dx / self._degree
        fem.interpolate(
            spawn_streamlines, dest=spawn_points, quadrature=streamline_spawn, values={"jitter": jitter_amount}
        )

        # now forward-trace the velocity field to generate the streamlines
        # here we use a fixed number of points per streamline, otherwise we would need to
        # do a first pass to count points, then array_scan the offsets, then a second pass
        # to populate the per-point data

        point_count = self._streamline_point_count
        points = wp.empty(dtype=wp.vec3, shape=(n_streamlines, point_count))
        speed = wp.empty(dtype=float, shape=(n_streamlines, point_count))

        fem.interpolate(
            gen_streamlines,
            domain=fem.Cells(self._geo),
            dim=n_streamlines,
            fields={"u": self.velocity_field},
            values={
                "spawn_points": spawn_points,
                "point_count": self._streamline_point_count,
                "dx": self._streamline_dx,
                "pos": points,
                "speed": speed,
            },
        )

        self._points = points
        self._speed = speed

    def render(self):
        # self.renderer.add_field("solution", self.pressure_field)
        self.plot.add_field("pressure", self.pressure_field)
        # self.plot.add_field("velocity", self.velocity_field)

        if self.renderer is not None:
            streamline_count = self._points.shape[0]
            point_count = self._streamline_point_count

            vertices = self._points.flatten().numpy()

            line_offsets = np.arange(streamline_count) * point_count
            indices_beg = np.arange(point_count - 1)[np.newaxis, :] + line_offsets[:, np.newaxis]
            indices_end = indices_beg + 1
            indices = np.vstack((indices_beg.flatten(), indices_end.flatten())).T.flatten()

            colors = self._speed.numpy()[:, :-1].flatten()
            colors = [wp.render.bourke_color_map(0.0, 3.0, c) for c in colors]

            self.renderer.begin_frame(0)
            self.renderer.render_line_list("streamlines", vertices, indices)
            self.renderer.render_line_list("streamlines", vertices, indices, colors)

            self.renderer.paused = True
            self.renderer.end_frame()

    def _generate_incompressible_flow(self):
        # Function spaces for velocity and pressure (RT1 / P0)
        u_space = fem.make_polynomial_space(
            geo=self._geo, element_basis=fem.ElementBasis.RAVIART_THOMAS, degree=1, dtype=wp.vec3
        )
        p_space = fem.make_polynomial_space(geo=self._geo, degree=0, dtype=float)

        self.pressure_field = p_space.make_field()
        self.velocity_field = u_space.make_field()

        # Boundary condition projector and matrices
        inflow_test = fem.make_test(u_space, domain=self._inflow)
        inflow_trial = fem.make_trial(u_space, domain=self._inflow)
        dirichlet_projector = fem.integrate(
            noslip_projector_form, fields={"u": inflow_test, "v": inflow_trial}, assembly="nodal", output_dtype=float
        )

        freeslip_test = fem.make_test(u_space, domain=self._freeslip)
        freeslip_trial = fem.make_trial(u_space, domain=self._freeslip)
        dirichlet_projector += fem.integrate(
            freeslip_projector_form,
            fields={"u": freeslip_test, "v": freeslip_trial},
            assembly="nodal",
            output_dtype=float,
        )
        fem.normalize_dirichlet_projector(dirichlet_projector)

        # Initialize velocity field with BC
        fem.interpolate(inflow_velocity, dest=fem.make_restriction(self.velocity_field, domain=self._inflow))

        # (Diagonal) mass matrix
        rho_test = fem.make_test(u_space)
        rho_trial = fem.make_trial(u_space)
        inv_mass_matrix = fem.integrate(
            mass_form, fields={"u": rho_trial, "v": rho_test}, assembly="nodal", output_dtype=float
        )
        fem_example_utils.invert_diagonal_bsr_matrix(inv_mass_matrix)

        # Assemble divergence operator matrix
        p_test = fem.make_test(p_space)
        u_trial = fem.make_trial(u_space)
        divergence_matrix = fem.integrate(
            divergence_form,
            fields={"u": u_trial, "psi": p_test},
            output_dtype=float,
        )

        # Solve incompressibility
        solve_incompressibility(
            divergence_matrix,
            dirichlet_projector,
            inv_mass_matrix.values,
            self.pressure_field.dof_values,
            self.velocity_field.dof_values,
            quiet=self._quiet,
        )


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=8, help="Grid resolution.")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree of shape functions.")
    parser.add_argument("--mesh", choices=("grid", "tet", "hex", "nano"), default="grid", help="Mesh type.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppresses the printing out of iteration residuals.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            quiet=args.quiet, degree=args.degree, resolution=args.resolution, mesh=args.mesh, headless=args.headless
        )

        example.step()
        example.render()
