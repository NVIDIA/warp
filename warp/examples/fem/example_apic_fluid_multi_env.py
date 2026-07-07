# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Multi-Environment APIC Fluid Simulation
#
# Runs colocated APIC fluid environments on a single multi-environment
# Nanogrid. The physical coordinates of all environments overlap; the FEM
# topology is separated by environment ids.
###########################################################################

from dataclasses import dataclass
from typing import Any

import numpy as np

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem
import warp.render
from warp.examples.fem.example_apic_fluid import (
    divergence_form,
    integrate_fraction,
    integrate_velocity,
    invert_volume_kernel,
    scalar_vector_multiply,
    scale_transposed_divergence_mat,
    solve_incompressibility,
    update_particles,
    velocity_boundary_projector_form,
)
from warp.optim.linear import LinearOperator
from warp.sparse import BsrMatrix, bsr_mm, bsr_mv, bsr_transposed


@wp.kernel
def apply_visual_offsets(
    positions: wp.array(dtype=wp.vec3),
    env_indices: wp.array(dtype=int),
    visual_offsets: wp.array(dtype=wp.vec3),
    render_positions: wp.array(dtype=wp.vec3),
):
    p = wp.tid()
    render_positions[p] = positions[p] + visual_offsets[env_indices[p]]


@wp.kernel
def schur_inverse_diagonal_kernel(
    divergence_mat_offsets: wp.array(dtype=int),
    divergence_mat_columns: wp.array(dtype=int),
    divergence_mat_values: wp.array(dtype=Any),
    inv_volume: wp.array(dtype=float),
    inv_diag: wp.array(dtype=float),
):
    p = wp.tid()

    diag = float(0.0)
    for block in range(divergence_mat_offsets[p], divergence_mat_offsets[p + 1]):
        value = divergence_mat_values[block]
        u = divergence_mat_columns[block]
        diag += inv_volume[u] * (value[0, 0] * value[0, 0] + value[0, 1] * value[0, 1] + value[0, 2] * value[0, 2])

    inv_diag[p] = wp.where(diag == 0.0, 0.0, 1.0 / diag)


def solve_incompressibility_matrix_free(
    divergence_mat: BsrMatrix,
    dirichlet_projector: BsrMatrix,
    inv_volume,
    pressure,
    velocity,
    pressure_env_offsets=None,
    quiet: bool = False,
):
    """Solve the APIC pressure system through a Schur-complement operator."""

    rhs = wp.empty_like(pressure)
    bsr_mv(A=divergence_mat, x=velocity, y=rhs, alpha=-1.0)

    bsr_mm(alpha=-1.0, x=divergence_mat, y=dirichlet_projector, z=divergence_mat, beta=1.0)

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

    inv_schur_diagonal = wp.empty_like(pressure)
    wp.launch(
        kernel=schur_inverse_diagonal_kernel,
        dim=pressure.shape[0],
        inputs=[
            divergence_mat.offsets,
            divergence_mat.columns,
            divergence_mat.values,
            inv_volume,
            inv_schur_diagonal,
        ],
    )

    tmp_velocity = wp.empty_like(velocity)

    def schur_matvec(x, y, z, alpha, beta):
        bsr_mv(A=transposed_divergence_mat, x=x, y=tmp_velocity, alpha=1.0, beta=0.0)
        if z.ptr != y.ptr and beta != 0.0:
            wp.copy(src=y, dest=z)
        bsr_mv(A=divergence_mat, x=tmp_velocity, y=z, alpha=alpha, beta=beta)

    # The pressure space is scalar, so partition node offsets are also scalar coefficient offsets.
    pressure_batch_offsets = pressure_env_offsets
    schur = LinearOperator(
        (pressure.shape[0], pressure.shape[0]),
        pressure.dtype,
        pressure.device,
        matvec=schur_matvec,
        batch_offsets=pressure_batch_offsets,
    )

    fem_example_utils.bsr_cg(
        schur,
        b=rhs,
        x=pressure,
        M=inv_schur_diagonal,
        quiet=quiet,
        tol=1.0e-6,
        method="cr",
        max_iters=1000,
    )

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
        stage_path="example_apic_fluid_multi_env.usd",
        voxel_size=0.5,
        env_count=4,
        visual_spacing=12.0,
        solve_mode="matrix-free-batched",
        opengl=False,
    ):
        device = wp.get_device()
        if not device.is_cuda:
            raise RuntimeError("The multi-environment APIC example requires a CUDA device.")

        self.gravity = wp.vec3(0.0, -10.0, 0.0)

        fps = 60
        self.sim_substeps = 1
        self.frame_dt = 1.0 / fps
        self.current_frame = 0
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.voxel_size = float(voxel_size)
        self.env_count = int(env_count)
        self.solve_mode = solve_mode

        if self.env_count < 1:
            raise ValueError("Environment count must be positive")
        if self.solve_mode not in ("assembled", "matrix-free", "matrix-free-batched"):
            raise ValueError("Solve mode must be 'assembled', 'matrix-free', or 'matrix-free-batched'")

        self._quiet = quiet

        particle_volumes, particle_q, particle_env, particle_offsets = self._spawn_multi_env_particles(
            env_count=self.env_count,
            voxel_size=self.voxel_size,
            packing_fraction=1.0,
        )

        self.particle_volumes = wp.array(particle_volumes, dtype=float, device=device)
        self.particle_env = wp.array(particle_env, dtype=int, device=device)
        particle_q = wp.array(particle_q, dtype=wp.vec3, device=device)
        particle_qd = wp.zeros_like(particle_q)

        self.particle_offsets = particle_offsets
        self.particle_counts = np.diff(particle_offsets)

        particle_count = particle_q.shape[0]
        if not self._quiet:
            print("Environment count:", self.env_count)
            print("Particle count:", particle_count)
            print("Particles per environment:", self.particle_counts.tolist())

        self.state_0 = self.State(
            wp.clone(particle_q),
            wp.clone(particle_qd),
            particle_qd_grad=wp.zeros(shape=(particle_count), dtype=wp.mat33, device=device),
        )
        self.state_1 = self.State(
            wp.clone(particle_q),
            wp.clone(particle_qd),
            particle_qd_grad=wp.zeros(shape=(particle_count), dtype=wp.mat33, device=device),
        )

        self.render_q = wp.empty_like(particle_q)
        self.visual_offsets = self._make_visual_offsets(self.env_count, visual_spacing, device)

        self.temporary_store = fem.TemporaryStore()

        self.opengl_renderer = None
        self.usd_renderer = None

        try:
            if opengl:
                self.opengl_renderer = warp.render.OpenGLRenderer(
                    screen_width=1024,
                    screen_height=1024,
                )
        except Exception as err:
            print(f"Could not initialize OpenGL renderer: {err}.")

        try:
            if stage_path:
                self.usd_renderer = warp.render.UsdRenderer(stage_path)
        except Exception as err:
            print(f"Could not initialize Usd renderer '{stage_path}': {err}.")

    def step(self):
        fem.set_default_temporary_store(self.temporary_store)

        self.current_frame = self.current_frame + 1

        for _s in range(self.sim_substeps):
            grid = self._build_multi_env_grid(self.state_0.particle_q)

            linear_basis_space = fem.make_polynomial_basis_space(grid, degree=1)
            velocity_space = fem.make_collocated_function_space(linear_basis_space, dtype=wp.vec3)
            fraction_space = fem.make_collocated_function_space(linear_basis_space, dtype=float)
            strain_space = fem.make_polynomial_space(
                grid,
                dtype=float,
                degree=0,
                discontinuous=True,
            )
            pressure_partition = fem.make_space_partition(
                space_topology=strain_space.topology,
                environment_first=True,
                temporary_store=self.temporary_store,
            )

            pressure_field = strain_space.make_field(space_partition=pressure_partition)
            velocity_field = velocity_space.make_field()

            domain = fem.Cells(grid)
            velocity_test = fem.make_test(velocity_space, domain=domain)
            velocity_trial = fem.make_trial(velocity_space, domain=domain)
            fraction_test = fem.make_test(fraction_space, domain=domain)
            strain_test = fem.make_test(strain_space, domain=domain, space_partition=pressure_partition)

            vel_projector = fem.integrate(
                velocity_boundary_projector_form,
                fields={"u": velocity_trial, "v": velocity_test},
                assembly="nodal",
                output_dtype=float,
            )
            fem.normalize_dirichlet_projector(vel_projector)

            pic = fem.PicQuadrature(
                domain=domain,
                positions=self.state_0.particle_q,
                measures=self.particle_volumes,
                env_indices=self.particle_env,
            )

            inv_volume = fem.integrate(
                integrate_fraction,
                quadrature=pic,
                fields={"phi": fraction_test},
                output_dtype=float,
            )
            wp.launch(kernel=invert_volume_kernel, dim=inv_volume.shape, inputs=[inv_volume])

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

            wp.launch(
                kernel=scalar_vector_multiply,
                dim=inv_volume.shape[0],
                inputs=[inv_volume, velocity_int, velocity_field.dof_values],
            )

            bsr_mv(
                A=vel_projector,
                x=velocity_field.dof_values,
                y=velocity_field.dof_values,
                alpha=-1.0,
                beta=1.0,
            )

            divergence_matrix = fem.integrate(
                divergence_form,
                quadrature=pic,
                fields={"u": velocity_trial, "psi": strain_test},
                output_dtype=float,
            )

            if self.solve_mode == "assembled":
                solve_incompressibility(
                    divergence_matrix,
                    vel_projector,
                    inv_volume,
                    pressure_field.dof_values,
                    velocity_field.dof_values,
                    quiet=not wp.config.verbose,
                )
            else:
                pressure_env_offsets = (
                    pressure_partition.env_offsets if self.solve_mode == "matrix-free-batched" else None
                )
                solve_incompressibility_matrix_free(
                    divergence_matrix,
                    vel_projector,
                    inv_volume,
                    pressure_field.dof_values,
                    velocity_field.dof_values,
                    pressure_env_offsets=pressure_env_offsets,
                    quiet=not wp.config.verbose,
                )

            fem.interpolate(
                update_particles,
                at=pic,
                values={
                    "pos": self.state_1.particle_q,
                    "pos_prev": self.state_0.particle_q,
                    "vel": self.state_1.particle_qd,
                    "vel_grad": self.state_1.particle_qd_grad,
                    "dt": self.sim_dt,
                },
                fields={"grid_vel": velocity_field},
            )

            self.state_0, self.state_1 = self.state_1, self.state_0

        fem.set_default_temporary_store(None)

    def _build_multi_env_grid(self, particle_q):
        return fem.Nanogrid.from_environment_voxels(
            particle_q,
            self.particle_env,
            self.env_count,
            voxel_size=self.voxel_size,
            temporary_store=self.temporary_store,
            device=particle_q.device,
        )

    @staticmethod
    def _spawn_multi_env_particles(env_count, voxel_size, packing_fraction):
        particles_per_cell_dim = 2
        rng = np.random.default_rng(42)

        particle_volumes = []
        particle_positions = []
        particle_env = []
        particle_offsets = [0]

        for env_index in range(env_count):
            width = 4.0 - 0.35 * float(env_index % 3)
            height = 8.0 - 0.5 * float(env_index % 4)
            bounds_lo = np.array([-width, -4.0, -width], dtype=np.float32)
            bounds_hi = np.array([width, -4.0 + height, width], dtype=np.float32)

            volumes, points = Example._spawn_particles(
                res=np.maximum(
                    np.array((bounds_hi - bounds_lo) / voxel_size, dtype=int) * particles_per_cell_dim,
                    1,
                ),
                bounds_lo=bounds_lo,
                bounds_hi=bounds_hi,
                packing_fraction=packing_fraction,
                rng=rng,
            )

            particle_volumes.append(volumes)
            particle_positions.append(points)
            particle_env.append(np.full(points.shape[0], env_index, dtype=np.int32))
            particle_offsets.append(particle_offsets[-1] + points.shape[0])

        return (
            np.concatenate(particle_volumes),
            np.ascontiguousarray(np.concatenate(particle_positions)),
            np.concatenate(particle_env),
            np.array(particle_offsets, dtype=np.int64),
        )

    @staticmethod
    def _spawn_particles(res, bounds_lo, bounds_hi, packing_fraction, rng):
        nx = res[0]
        ny = res[1]
        nz = res[2]

        px = np.linspace(bounds_lo[0], bounds_hi[0], nx + 1)
        py = np.linspace(bounds_lo[1], bounds_hi[1], ny + 1)
        pz = np.linspace(bounds_lo[2], bounds_hi[2], nz + 1)

        points = np.stack(np.meshgrid(px, py, pz)).reshape(3, -1).T

        cell_size = (bounds_hi - bounds_lo) / res
        cell_volume = np.prod(cell_size)

        radius = np.max(cell_size) * 0.5
        volume = cell_volume * packing_fraction

        points += 2.0 * radius * (rng.random(points.shape) - 0.5)

        volumes = np.full(points.shape[0], volume, dtype=np.float32)
        return volumes, np.ascontiguousarray(points, dtype=np.float32)

    @staticmethod
    def _make_visual_offsets(env_count, visual_spacing, device):
        offsets = np.zeros((env_count, 3), dtype=np.float32)
        cols = int(np.ceil(np.sqrt(env_count)))

        for env_index in range(env_count):
            row = env_index // cols
            col = env_index - row * cols
            offsets[env_index, 0] = (col - 0.5 * (cols - 1)) * visual_spacing
            offsets[env_index, 2] = row * visual_spacing

        return wp.array(offsets, dtype=wp.vec3, device=device)

    def render(self):
        if self.usd_renderer is None and self.opengl_renderer is None:
            return

        time = self.current_frame * self.frame_dt
        wp.launch(
            kernel=apply_visual_offsets,
            dim=self.state_0.particle_q.shape,
            inputs=[self.state_0.particle_q, self.particle_env, self.visual_offsets, self.render_q],
        )

        if self.usd_renderer is not None:
            self.usd_renderer.begin_frame(time)
            self.usd_renderer.render_points(
                "particles",
                self.render_q.numpy(),
                radius=self.voxel_size / 4.0,
            )
            self.usd_renderer.end_frame()
        if self.opengl_renderer is not None:
            self.opengl_renderer.begin_frame(time)
            self.opengl_renderer.render_points(
                "particles",
                self.render_q,
                radius=self.voxel_size / 4.0,
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
        default="example_apic_fluid_multi_env.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=250, help="Total number of frames.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--opengl", action="store_true")
    parser.add_argument("--env-count", type=int, default=4, help="Number of colocated environments.")
    parser.add_argument(
        "--solve-mode",
        choices=("assembled", "matrix-free", "matrix-free-batched"),
        default="matrix-free-batched",
        help="Pressure solve implementation.",
    )
    parser.add_argument(
        "--visual-spacing", type=float, default=12.0, help="Renderer-only spacing between environments."
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.5,
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            quiet=args.quiet,
            stage_path=args.stage_path,
            voxel_size=args.voxel_size,
            env_count=args.env_count,
            visual_spacing=args.visual_spacing,
            solve_mode=args.solve_mode,
            opengl=args.opengl,
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
