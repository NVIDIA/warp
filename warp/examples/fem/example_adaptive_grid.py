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
# Example Adaptive Grid
#
# Demonstrates using an adaptive grid to increase the simulation resolition
# near a collider boundary.
#
###########################################################################
import os.path

import numpy as np

import warp as wp
import warp.examples
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem
from warp.examples.fem.example_apic_fluid import divergence_form, solve_incompressibility


@fem.integrand
def inflow_velocity(
    s: fem.Sample,
    domain: fem.Domain,
    bounds_lo: wp.vec3,
    bounds_hi: wp.vec3,
):
    x = fem.position(domain, s)

    if x[1] <= bounds_lo[1] or x[2] <= bounds_lo[2] or x[1] >= bounds_hi[1] or x[2] >= bounds_hi[2]:
        return wp.vec3(0.0)

    if x[0] <= bounds_lo[0] or x[0] >= bounds_hi[0]:
        return wp.vec3(1.0, 0.0, 0.0)

    return wp.vec3(0.0)


@fem.integrand
def noslip_projector_form(
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    return wp.dot(u(s), v(s))


@fem.integrand
def mass_form(
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    return fem.linalg.generalized_inner(u(s), v(s))


@fem.integrand
def side_divergence_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, psi: fem.Field):
    # normal velocity jump (non-zero at resolution boundaries)
    return -wp.dot(fem.jump(u, s), fem.normal(domain, s)) * fem.average(psi, s)


@wp.func
def refinement_field(xyz: wp.vec3, volume: wp.uint64):
    # use distance to collider as refinement function
    uvw = wp.volume_world_to_index(volume, xyz)
    sdf = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)

    if sdf < 0.0:
        return sdf

    # combine with  heuristical distance to keep coarsening past nvdb narrowband
    return 0.5 * wp.max(wp.length(xyz) - 20.0, sdf)


@fem.integrand
def pressure_anomaly_field(s: fem.Sample, domain: fem.Domain, pressure: fem.Field):
    # for visualization, deduce affine part such that grad P = u_x
    x = domain(s)
    return pressure(s) + x[0]


class Example:
    def __init__(
        self, quiet=False, degree=2, div_conforming=False, base_resolution=8, level_count=4, headless: bool = False
    ):
        self._quiet = quiet
        self._degree = degree
        self._div_conforming = div_conforming

        # Start from a coarse, dense grid
        res = wp.vec3i(2 * base_resolution, base_resolution // 2, base_resolution)
        bounds_lo = wp.vec3(-50.0, 0.0, -17.5)
        bounds_hi = wp.vec3(50.0, 12.5, 17.5)
        sim_vol = fem_example_utils.gen_volume(res=res, bounds_lo=bounds_lo, bounds_hi=bounds_hi)

        # load collision volume
        collider_path = os.path.join(warp.examples.get_asset_directory(), "rocks.nvdb")
        with open(collider_path, "rb") as file:
            # create Volume object
            collider = wp.Volume.load_from_nvdb(file)

        # Make adaptive grid from coarse base and refinement field
        refinement = fem.ImplicitField(
            domain=fem.Cells(fem.Nanogrid(sim_vol)), func=refinement_field, values={"volume": collider.id}
        )
        self._geo = fem.adaptivity.adaptive_nanogrid_from_field(
            sim_vol, level_count, refinement_field=refinement, grading="face"
        )

        # Function spaces for velocity, pressure (RTk / Pk-1 or Pk / Pk-1)
        u_space = fem.make_polynomial_space(
            geo=self._geo,
            element_basis=fem.ElementBasis.RAVIART_THOMAS if div_conforming else None,
            degree=self._degree,
            dtype=wp.vec3,
        )
        p_space = fem.make_polynomial_space(geo=self._geo, degree=self._degree - 1, dtype=float)

        self.pressure_field = p_space.make_field()
        self.pressure_anomaly_field = p_space.make_field()
        self.velocity_field = u_space.make_field()

        # Initialize velocity field with BC
        bounds_scale = 0.9999  # account for difference between bounds and actual grid extents
        bounds_center = 0.5 * (bounds_hi + bounds_lo)
        bounds_extent = 0.5 * (bounds_hi - bounds_lo)
        fem.interpolate(
            inflow_velocity,
            dest=fem.make_restriction(self.velocity_field, domain=fem.BoundarySides(self._geo)),
            values={
                "bounds_lo": bounds_center - bounds_scale * bounds_extent,
                "bounds_hi": bounds_center + bounds_scale * bounds_extent,
            },
        )

        self.plot = fem_example_utils.Plot()

    def render(self):
        # self.renderer.add_field("solution", self.pressure_field)
        self.plot.add_field("pressure_anomaly", self.pressure_anomaly_field)

        if self._div_conforming:
            # If using H(div)-conforming elements, interpolate to continuous space
            velocity_field_lagrange = fem.make_polynomial_space(
                self.velocity_field.geometry, dtype=wp.vec3, degree=self._degree
            ).make_field()
            fem.interpolate(self.velocity_field, dest=velocity_field_lagrange)
        else:
            velocity_field_lagrange = self.velocity_field

        self.plot.add_field("velocity", velocity_field_lagrange)

    def step(self):
        u_space = self.velocity_field.space
        p_space = self.pressure_field.space

        # Boundary condition projector and matrices
        boundary = fem.BoundarySides(self._geo)
        bd_test = fem.make_test(u_space, domain=boundary)
        bd_trial = fem.make_trial(u_space, domain=boundary)
        dirichlet_projector = fem.integrate(
            noslip_projector_form, fields={"u": bd_test, "v": bd_trial}, assembly="nodal", output_dtype=float
        )
        fem.normalize_dirichlet_projector(dirichlet_projector)

        # (Diagonal) mass matrix
        if self._div_conforming:
            rho_test = fem.make_test(u_space)
            rho_trial = fem.make_trial(u_space)
        else:
            rho_space = fem.make_polynomial_space(geo=u_space.geometry, degree=self._degree)
            rho_test = fem.make_test(rho_space)
            rho_trial = fem.make_trial(rho_space)

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

        # need to account for discontinuities at resolution boundaries (t-junctions)
        p_side_test = fem.make_test(p_space, domain=fem.Sides(self._geo))
        u_side_trial = fem.make_trial(u_space, domain=fem.Sides(self._geo))
        divergence_matrix += fem.integrate(
            side_divergence_form,
            fields={"u": u_side_trial, "psi": p_side_test},
            output_dtype=float,
            assembly="generic",  # not required, for test coverage purposes
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

        fem.interpolate(
            pressure_anomaly_field,
            dest=self.pressure_anomaly_field,
            fields={"pressure": self.pressure_field},
        )


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=8, help="Grid resolution.")
    parser.add_argument("--degree", type=int, default=1, help="Polynomial degree of shape functions.")
    parser.add_argument(
        "--div_conforming", action="store_true", default=False, help="Use H(div)-conforming function space"
    )
    parser.add_argument("--level_count", type=int, default=4, help="Number of refinement levels.")
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
            div_conforming=args.div_conforming,
            base_resolution=args.resolution,
            level_count=args.level_count,
            headless=args.headless,
        )

        example.step()
        example.render()

        if not args.headless:
            ref_geom = None
            try:
                from pxr import Usd, UsdGeom

                stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "rocks.usd"))
                mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/root/rocks"))
                points = np.array(mesh.GetPointsAttr().Get())
                counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
                indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
                ref_geom = (points, counts, indices)
            except Exception:
                pass

            example.plot.plot(
                {
                    "rows": 2,
                    "ref_geom": ref_geom,
                    "velocity": {"streamlines": {"density": 25, "glyph_scale": 0.01}},
                    "pressure_anomaly": {},
                }
            )
