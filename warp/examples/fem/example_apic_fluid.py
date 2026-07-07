# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example APIC Fluid Simulation
#
# Shows how to implement a minimalist APIC fluid simulation using a NanoVDB
# grid and the PicQuadrature class.
###########################################################################

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem
import warp.render
from warp.fem import Domain, Field, Sample, at_node, div, grad, integrand
from warp.optim.linear import cr, preconditioner
from warp.sparse import BsrMatrix, bsr_mm, bsr_mv, bsr_transposed


@wp.func
def collision_sdf(x: wp.vec3):
    # Arbitrary sdf representing collision geometry
    # Here an inverted half-ball of radius 10
    x[1] = wp.min(x[1], 0.0)
    return 10.0 - wp.length(x), -wp.normalize(x)


@integrand
def integrate_fraction(s: Sample, phi: Field):
    return phi(s)


@integrand
def integrate_velocity(
    s: Sample,
    domain: Domain,
    u: Field,
    velocities: wp.array[wp.vec3],
    velocity_gradients: wp.array[wp.mat33],
    dt: float,
    gravity: wp.vec3,
):
    """Transfer particle velocities to grid"""
    node_offset = domain(at_node(u, s)) - domain(s)
    vel_apic = velocities[s.qp_index] + velocity_gradients[s.qp_index] * node_offset

    vel_adv = vel_apic + dt * gravity

    # if inside collider, remove normal velocity
    sdf, sdf_gradient = collision_sdf(domain(s))
    if sdf <= 0:
        v_n = wp.dot(vel_adv, sdf_gradient)
        vel_adv -= wp.max(v_n, 0.0) * sdf_gradient

    return wp.dot(u(s), vel_adv)


@integrand
def update_particles(
    s: Sample,
    domain: Domain,
    grid_vel: Field,
    dt: float,
    pos: wp.array[wp.vec3],
    pos_prev: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    vel_grad: wp.array[wp.mat33],
):
    """Read particle velocity from grid and advect positions"""
    p_vel = grid_vel(s)
    vel_grad[s.qp_index] = grad(grid_vel, s)

    pos_adv = pos_prev[s.qp_index] + dt * p_vel

    pos[s.qp_index] = pos_adv
    vel[s.qp_index] = p_vel


@integrand
def velocity_boundary_projector_form(s: Sample, domain: Domain, u: Field, v: Field):
    """Projector for velocity-Dirichlet boundary conditions"""

    x = domain(s)
    sdf, sdf_normal = collision_sdf(x)

    if sdf > 0.0:
        # Neuman
        return 0.0

    # Free-slip on boundary
    return wp.dot(u(s), sdf_normal) * wp.dot(v(s), sdf_normal)


@integrand
def divergence_form(s: Sample, domain: Domain, u: Field, psi: Field):
    # Divergence bilinear form
    return div(u, s) * psi(s)


@wp.kernel
def invert_volume_kernel(values: wp.array[float]):
    i = wp.tid()
    m = values[i]
    values[i] = wp.where(m == 0.0, 0.0, 1.0 / m)


@wp.kernel
def scalar_vector_multiply(
    alpha: wp.array[float],
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
):
    i = wp.tid()
    y[i] = alpha[i] * x[i]


@wp.kernel
def scale_transposed_divergence_mat(
    tr_divergence_mat_offsets: wp.array[int],
    tr_divergence_mat_values: wp.array[Any],
    inv_fraction_int: wp.array[float],
):
    # In-place scaling of gradient operator rows with inverse mass

    u_i = wp.tid()
    block_beg = tr_divergence_mat_offsets[u_i]
    block_end = tr_divergence_mat_offsets[u_i + 1]

    for b in range(block_beg, block_end):
        tr_divergence_mat_values[b] = tr_divergence_mat_values[b] * inv_fraction_int[u_i]


def solve_incompressibility(
    divergence_mat: BsrMatrix,
    dirichlet_projector: BsrMatrix,
    inv_volume,
    pressure,
    velocity,
    quiet: bool = False,
    capturable: bool = False,
):
    """Solve for divergence-free velocity delta:

    delta_velocity = inv_volume * transpose(divergence_mat) * pressure
    divergence_mat * (velocity + delta_velocity) = 0
    dirichlet_projector * delta_velocity = 0
    """

    # Constraint-free divergence -- computed *before* projection of divergence_mat
    rhs = wp.empty_like(pressure)
    bsr_mv(A=divergence_mat, x=velocity, y=rhs, alpha=-1.0)

    # Project matrix to enforce boundary conditions
    # divergence_matrix -= divergence_matrix * vel_projector
    if capturable:
        bsr_mm(alpha=-1.0, x=divergence_mat, y=dirichlet_projector, z=divergence_mat, beta=1.0, topology="padded")
    else:
        bsr_mm(alpha=-1.0, x=divergence_mat, y=dirichlet_projector, z=divergence_mat, beta=1.0)

    # Build transposed gradient matrix, scale with inverse fraction
    transposed_divergence_mat = bsr_transposed(divergence_mat)
    wp.launch(
        kernel=scale_transposed_divergence_mat,
        dim=inv_volume.shape[0],
        inputs=[
            transposed_divergence_mat.offsets,
            transposed_divergence_mat.values,
            inv_volume,
        ],
    )

    # For simplicity, assemble Schur complement and solve with CG
    if capturable:
        schur = bsr_mm(divergence_mat, transposed_divergence_mat, max_new_nnz=pressure.shape[0] * 27)
        cr(
            schur,
            b=rhs,
            x=pressure,
            M=preconditioner(schur, "diag"),
            tol=1.0e-6,
            check_every=0,
            use_cuda_graph=True,
            maxiter=1000,
        )
    else:
        schur = bsr_mm(divergence_mat, transposed_divergence_mat)
        fem_example_utils.bsr_cg(schur, b=rhs, x=pressure, quiet=quiet, tol=1.0e-6, method="cr", max_iters=1000)

    # Apply pressure to velocity
    bsr_mv(A=transposed_divergence_mat, x=pressure, y=velocity, alpha=1.0, beta=1.0)


class Example:
    @dataclass
    class State:
        particle_q: wp.array[wp.vec3]
        particle_qd: wp.array[wp.vec3]
        particle_qd_grad: wp.array[wp.mat33]

    def __init__(
        self,
        quiet=False,
        stage_path="example_apic_fluid.usd",
        voxel_size=1.0,
        opengl=False,
        use_cuda_graph=True,
        grid_capacity_ratio=16.0,
        grid_leaf_capacity_ratio=None,
        grid_internal_capacity_ratio=None,
        max_active_voxels=None,
        max_leaf_nodes=None,
        max_lower_nodes=None,
        max_upper_nodes=None,
    ):
        self.gravity = wp.vec3(0.0, -10.0, 0.0)

        fps = 60
        self.sim_substeps = 1
        self.frame_dt = 1.0 / fps
        self.current_frame = 0
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.voxel_size = voxel_size

        self._quiet = quiet

        # particle emission
        PARTICLES_PER_CELL_DIM = 2
        self.radius = float(np.max(voxel_size) / (2 * PARTICLES_PER_CELL_DIM))

        particle_grid_lo = np.full(3, -5)
        particle_grid_hi = np.full(3, 5)
        particle_grid_res = (
            np.array((particle_grid_hi - particle_grid_lo) / voxel_size, dtype=int) * PARTICLES_PER_CELL_DIM
        )

        self.particle_volumes, particle_q = self._spawn_particles(
            particle_grid_res, particle_grid_lo, particle_grid_hi, packing_fraction=1.0
        )
        particle_qd = wp.zeros_like(particle_q)

        particle_count = particle_q.shape[0]
        if not self._quiet:
            print("Particle count:", particle_count)

        self._device = wp.get_device()
        self._use_cuda_graph = (
            use_cuda_graph
            and self._device.is_cuda
            and self._device.is_mempool_supported
            and wp.is_conditional_graph_supported()
        )
        if use_cuda_graph and self._device.is_cuda and not self._use_cuda_graph:
            warnings.warn(
                "CUDA graph capture for example_apic_fluid requires a CUDA memory pool and conditional graphs.",
                stacklevel=2,
            )
        if self._use_cuda_graph and not wp.is_mempool_enabled(self._device):
            wp.set_mempool_enabled(self._device, True)

        explicit_grid_capacity = {
            "max_active_voxels": max_active_voxels,
            "max_leaf_nodes": max_leaf_nodes,
            "max_lower_nodes": max_lower_nodes,
            "max_upper_nodes": max_upper_nodes,
        }
        if all(value is not None for value in explicit_grid_capacity.values()):
            grid_capacity = explicit_grid_capacity
        else:
            grid_capacity = self._estimate_grid_capacity_from_initial_grid(
                particle_q,
                voxel_size=self.voxel_size,
                active_ratio=grid_capacity_ratio,
                leaf_ratio=grid_leaf_capacity_ratio,
                internal_ratio=grid_internal_capacity_ratio,
            )
            grid_capacity.update({key: value for key, value in explicit_grid_capacity.items() if value is not None})

        if not self._quiet:
            print("Grid capacity:", ", ".join(f"{key}={value}" for key, value in grid_capacity.items()))

        # Allocate particle state. The graph-captured path updates this state
        # in-place so captured array pointers remain stable across replays.
        self.state_0 = self.State(
            wp.clone(particle_q),
            wp.clone(particle_qd),
            particle_qd_grad=wp.zeros(shape=(particle_count), dtype=wp.mat33),
        )

        # Storage for temporary variables
        self.temporary_store = fem.TemporaryStore()

        self.grid_status = wp.zeros(1, dtype=wp.uint32)
        self.volume = wp.Volume.allocate_by_voxels(
            voxel_points=self.state_0.particle_q,
            voxel_size=self.voxel_size,
            rebuildable=True,
            **grid_capacity,
            status=self.grid_status,
        )
        self.grid = fem.Nanogrid(self.volume, rebuildable=True)

        self.linear_basis_space = fem.make_polynomial_basis_space(self.grid, degree=1)
        self.velocity_space = fem.make_collocated_function_space(self.linear_basis_space, dtype=wp.vec3)
        self.fraction_space = fem.make_collocated_function_space(self.linear_basis_space, dtype=float)
        self.strain_space = fem.make_polynomial_space(
            self.grid,
            dtype=float,
            degree=0,
            discontinuous=True,
        )

        self._bsr_options = {"construction": "row_compress", "capacity": "auto"}
        self._padded_bsr_options = self._bsr_options | {"topology": "padded"}

        if self._use_cuda_graph:
            import gc  # noqa: PLC0415

            gc.disable()
            try:
                with wp.ScopedCapture(self._device) as capture:
                    self.simulate(capturable=True)
                self.graph = capture.graph
            finally:
                gc.enable()

        # initialize renderers
        self.opengl_renderer = None
        self.usd_renderer = None

        try:
            if opengl:
                self.opengl_renderer = warp.render.OpenGLRenderer(
                    screen_width=1024,
                    screen_height=1024,
                )
        except Exception as err:
            warnings.warn(f"Could not initialize OpenGL renderer: {err}.", stacklevel=2)

        try:
            if stage_path:
                self.usd_renderer = warp.render.UsdRenderer(stage_path)
        except Exception as err:
            print(f"Could not initialize Usd renderer '{stage_path}': {err}.")

    def step(self):
        fem.set_default_temporary_store(self.temporary_store)

        self.current_frame = self.current_frame + 1

        for _s in range(self.sim_substeps):
            if self._use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate(capturable=False)

        fem.set_default_temporary_store(None)

    def simulate(self, capturable: bool):
        # Rebuild the persistent sparse grid from current particle positions.
        self.grid.rebuild(self.state_0.particle_q, status=self.grid_status)
        self.linear_basis_space.topology.rebuild()
        self.strain_space.topology.rebuild()

        # Bin particles on the rebuilt grid, then restrict FEM assembly to
        # cells that actually contain particles.  All counts below are host-side
        # upper bounds, so the captured path does not synchronize for exact
        # topology sizes.
        whole_domain = fem.Cells(self.grid)
        pic = fem.PicQuadrature(
            domain=whole_domain,
            positions=self.state_0.particle_q,
            measures=self.particle_volumes,
            temporary_store=self.temporary_store,
        )

        cell_mask = wp.empty(shape=self.grid.cell_count(), dtype=int)
        pic.fill_element_mask(cell_mask)
        geo_partition = fem.ExplicitGeometryPartition(
            self.grid,
            cell_mask,
            max_cell_count=self.grid.cell_count(),
            max_side_count=0,
            temporary_store=self.temporary_store,
        )
        domain = fem.Cells(geo_partition)
        pic.domain = domain

        velocity_partition = fem.make_space_partition(
            self.velocity_space.topology,
            geometry_partition=geo_partition,
            with_halo=False,
            max_node_count=self.grid.vertex_count(),
            temporary_store=self.temporary_store,
        )
        strain_partition = fem.make_space_partition(
            self.strain_space.topology,
            geometry_partition=geo_partition,
            with_halo=False,
            max_node_count=self.grid.cell_count(),
            temporary_store=self.temporary_store,
        )

        velocity_restriction = fem.make_space_restriction(
            space_partition=velocity_partition,
            domain=domain,
            temporary_store=self.temporary_store,
        )
        strain_restriction = fem.make_space_restriction(
            space_partition=strain_partition,
            domain=domain,
            temporary_store=self.temporary_store,
        )

        velocity_test = fem.make_test(self.velocity_space, space_restriction=velocity_restriction)
        velocity_trial = fem.make_trial(self.velocity_space, space_restriction=velocity_restriction)
        fraction_test = fem.make_test(self.fraction_space, space_restriction=velocity_restriction)
        strain_test = fem.make_test(self.strain_space, space_restriction=strain_restriction)

        pressure_field = self.strain_space.make_field(strain_partition)
        velocity_field = self.velocity_space.make_field(velocity_partition)

        # Build projector for Dirichlet boundary conditions
        vel_projector = fem.integrate(
            velocity_boundary_projector_form,
            fields={"u": velocity_trial, "v": velocity_test},
            assembly="nodal",
            output_dtype=float,
            bsr_options=self._padded_bsr_options if capturable else self._bsr_options,
        )
        fem.normalize_dirichlet_projector(vel_projector)

        # Compute inverse particle volume for each grid node
        inv_volume = fem.integrate(
            integrate_fraction,
            quadrature=pic,
            fields={"phi": fraction_test},
            output_dtype=float,
        )
        wp.launch(kernel=invert_volume_kernel, dim=inv_volume.shape, inputs=[inv_volume])

        # Velocity right-hand side
        velocity_int = fem.integrate(
            integrate_velocity,
            quadrature=pic,
            fields={"u": velocity_test},
            values={
                "velocities": self.state_0.particle_qd,
                "velocity_gradients": self.state_0.particle_qd_grad,
                "dt": self.sim_dt,
                "gravity": self.gravity,
            },
            output_dtype=wp.vec3,
        )

        # Compute constraint-free velocity
        wp.launch(
            kernel=scalar_vector_multiply,
            dim=inv_volume.shape[0],
            inputs=[inv_volume, velocity_int, velocity_field.dof_values],
        )

        # Apply velocity boundary conditions:
        # velocity -= vel_projector * velocity
        bsr_mv(
            A=vel_projector,
            x=velocity_field.dof_values,
            y=velocity_field.dof_values,
            alpha=-1.0,
            beta=1.0,
        )

        # Assemble divergence operator matrix
        divergence_matrix = fem.integrate(
            divergence_form,
            quadrature=pic,
            fields={"u": velocity_trial, "psi": strain_test},
            output_dtype=float,
            bsr_options=self._padded_bsr_options if capturable else None,
        )

        # Solve unilateral incompressibility
        solve_incompressibility(
            divergence_matrix,
            vel_projector,
            inv_volume,
            pressure_field.dof_values,
            velocity_field.dof_values,
            quiet=wp.config.log_level > wp.LOG_DEBUG,
            capturable=capturable,
        )

        # (A)PIC advection.  The update is per-particle, so the graph path can
        # write back in-place and keep captured array pointers stable.
        fem.interpolate(
            update_particles,
            at=pic,
            values={
                "pos": self.state_0.particle_q,
                "pos_prev": self.state_0.particle_q,
                "vel": self.state_0.particle_qd,
                "vel_grad": self.state_0.particle_qd_grad,
                "dt": self.sim_dt,
            },
            fields={"grid_vel": velocity_field},
        )

    @staticmethod
    def _spawn_particles(res, bounds_lo, bounds_hi, packing_fraction):
        Nx = res[0]
        Ny = res[1]
        Nz = res[2]

        px = np.linspace(bounds_lo[0], bounds_hi[0], Nx + 1)
        py = np.linspace(bounds_lo[1], bounds_hi[1], Ny + 1)
        pz = np.linspace(bounds_lo[2], bounds_hi[2], Nz + 1)

        points = np.stack(np.meshgrid(px, py, pz)).reshape(3, -1).T

        cell_size = (bounds_hi - bounds_lo) / res
        cell_volume = np.prod(cell_size)

        radius = np.max(cell_size) * 0.5
        volume = cell_volume * packing_fraction

        rng = np.random.default_rng(42)
        points += 2.0 * radius * (rng.random(points.shape) - 0.5)

        volumes = wp.full(points.shape[0], volume, dtype=float)
        points = wp.array(np.ascontiguousarray(points), dtype=wp.vec3)
        return volumes, points

    @staticmethod
    def _estimate_grid_capacity_from_initial_grid(
        voxel_points,
        voxel_size,
        active_ratio,
        leaf_ratio=None,
        internal_ratio=None,
    ):
        leaf_ratio = active_ratio if leaf_ratio is None else leaf_ratio
        internal_ratio = leaf_ratio if internal_ratio is None else internal_ratio

        for name, ratio in (
            ("grid_capacity_ratio", active_ratio),
            ("grid_leaf_capacity_ratio", leaf_ratio),
            ("grid_internal_capacity_ratio", internal_ratio),
        ):
            if ratio <= 0.0:
                raise ValueError(f"{name} must be positive")

        initial_volume = wp.Volume.allocate_by_voxels(
            voxel_points=voxel_points,
            voxel_size=voxel_size,
            device=voxel_points.device,
        )
        first_counts = initial_volume.get_active_stats()
        if first_counts.voxel_count == 0:
            first_counts = wp.Volume.ActiveStats(1, 1, 1, 1)

        def scale(count, ratio, limit):
            return max(1, min(limit, math.ceil(count * ratio)))

        active_capacity = scale(first_counts.voxel_count, active_ratio, voxel_points.shape[0])
        leaf_capacity = scale(first_counts.leaf_node_count, leaf_ratio, active_capacity)
        lower_capacity = scale(first_counts.lower_node_count, internal_ratio, leaf_capacity)
        upper_capacity = scale(first_counts.upper_node_count, internal_ratio, lower_capacity)

        return {
            "max_active_voxels": active_capacity,
            "max_leaf_nodes": leaf_capacity,
            "max_lower_nodes": lower_capacity,
            "max_upper_nodes": upper_capacity,
        }

    def render(self):
        if self.usd_renderer is None and self.opengl_renderer is None:
            return

        time = self.current_frame * self.frame_dt

        if self.usd_renderer is not None:
            self.usd_renderer.begin_frame(time)
            self.usd_renderer.render_points(
                "particles",
                self.state_0.particle_q.numpy(),
                radius=self.radius,
            )
            self.usd_renderer.end_frame()
        if self.opengl_renderer is not None:
            self.opengl_renderer.begin_frame(time)
            self.opengl_renderer.render_points(
                "particles",
                self.state_0.particle_q,
                radius=self.radius,
            )
            self.opengl_renderer.end_frame()


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_apic_fluid.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=250, help="Total number of frames.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--opengl", action="store_true")
    parser.add_argument("--no-cuda-graph", action="store_true", help="Disable CUDA graph capture.")
    parser.add_argument(
        "--grid-capacity-ratio",
        type=float,
        default=16.0,
        help="Multiplier applied to exact first-frame grid counts for rebuild capacity.",
    )
    parser.add_argument(
        "--grid-leaf-capacity-ratio",
        type=float,
        default=None,
        help="Multiplier applied to exact first-frame leaf-node count; defaults to --grid-capacity-ratio.",
    )
    parser.add_argument(
        "--grid-internal-capacity-ratio",
        type=float,
        default=None,
        help="Multiplier applied to exact first-frame lower and upper node counts; defaults to leaf ratio.",
    )
    parser.add_argument("--max-active-voxels", type=int, default=None, help="Override active voxel rebuild capacity.")
    parser.add_argument("--max-leaf-nodes", type=int, default=None, help="Override leaf node rebuild capacity.")
    parser.add_argument("--max-lower-nodes", type=int, default=None, help="Override lower node rebuild capacity.")
    parser.add_argument("--max-upper-nodes", type=int, default=None, help="Override upper node rebuild capacity.")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.25,
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            quiet=args.quiet,
            stage_path=args.stage_path,
            voxel_size=args.voxel_size,
            opengl=args.opengl,
            use_cuda_graph=not args.no_cuda_graph,
            grid_capacity_ratio=args.grid_capacity_ratio,
            grid_leaf_capacity_ratio=args.grid_leaf_capacity_ratio,
            grid_internal_capacity_ratio=args.grid_internal_capacity_ratio,
            max_active_voxels=args.max_active_voxels,
            max_leaf_nodes=args.max_leaf_nodes,
            max_lower_nodes=args.max_lower_nodes,
            max_upper_nodes=args.max_upper_nodes,
        )

        for _, set_info in fem_example_utils.progress_bar(args.num_frames, quiet=args.quiet):
            with wp.ScopedTimer("step", synchronize=True, print=False) as step_timer:
                example.step()
            with wp.ScopedTimer("render", synchronize=True, print=False) as render_timer:
                example.render()

            set_info("step_time", f"{step_timer.elapsed} ms")
            set_info("render_time", f"{render_timer.elapsed} ms")

        if example.usd_renderer is not None:
            example.usd_renderer.save()
