# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example CUDA Profiler
#
# Runs a direct (all-pairs) N-body gravitational simulation in a loop and
# uses the CUDA profiler control API to restrict an attached profiler's
# capture range to a region of interest. The simulation is warmed up for a
# number of iterations (to settle JIT compilation, allocations, and the GPU
# clocks) before wp.cuda_profiler_range() brackets the steps to be profiled.
#
# Capture the profiled region with an external tool, e.g.:
#
#   Nsight Systems:
#     nsys profile --capture-range=cudaProfilerApi \
#         python example_cuda_profiler.py
#
#   Nsight Compute:
#     ncu --profile-from-start off -o nbody \
#         python example_cuda_profiler.py
#
# See docs/deep_dive/profiling.rst ("Limiting the Profiler Capture Range").
#
###########################################################################

"""Restrict an external profiler's capture range to a region of interest using the CUDA
profiler control API around a warmed-up N-body simulation."""

import numpy as np

import warp as wp


@wp.kernel
def compute_accelerations(
    positions: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    softening_sq: wp.float32,
    gravity: wp.float32,
    accelerations: wp.array(dtype=wp.vec3),
):
    """Compute the gravitational acceleration on each body from all others (O(N) per thread)."""
    i = wp.tid()

    xi = positions[i]
    acc = wp.vec3(0.0, 0.0, 0.0)

    # accumulate the gravitational pull from every other body (O(N) per thread)
    for j in range(positions.shape[0]):
        r = positions[j] - xi
        dist_sq = wp.dot(r, r) + softening_sq

        # inverse cube of the softened distance
        inv_dist = 1.0 / wp.sqrt(dist_sq)
        inv_dist_cube = inv_dist * inv_dist * inv_dist

        acc += r * (gravity * masses[j] * inv_dist_cube)

    accelerations[i] = acc


@wp.kernel
def integrate(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    accelerations: wp.array(dtype=wp.vec3),
    dt: wp.float32,
):
    """Advance positions and velocities by one timestep using semi-implicit Euler integration."""
    i = wp.tid()

    # semi-implicit (symplectic) Euler integration
    v_new = velocities[i] + accelerations[i] * dt
    velocities[i] = v_new
    positions[i] = positions[i] + v_new * dt


class Example:
    """Direct (all-pairs) N-body gravitational simulation used to generate profilable GPU work.

    Args:
        num_bodies: Number of bodies in the simulation.
    """

    def __init__(self, num_bodies=8192):
        self.num_bodies = num_bodies

        self.dt = 1.0e-3
        self.gravity = 1.0
        self.softening_sq = 1.0e-2

        rng = np.random.default_rng(42)

        # initialize bodies in a rotating spherical cloud
        pos = rng.standard_normal((num_bodies, 3)).astype(np.float32)
        radius = np.linalg.norm(pos, axis=1, keepdims=True)
        pos /= np.maximum(radius, 1.0e-3)
        pos *= rng.uniform(0.0, 4.0, (num_bodies, 1)).astype(np.float32) ** (1.0 / 3.0)

        # give each body a velocity perpendicular to its radius for some spin
        vel = np.cross(pos, np.array([0.0, 0.0, 1.0], dtype=np.float32)) * 0.5

        self.positions = wp.array(pos, dtype=wp.vec3)
        self.velocities = wp.array(vel, dtype=wp.vec3)
        self.accelerations = wp.zeros(num_bodies, dtype=wp.vec3)
        self.masses = wp.array(rng.uniform(0.5, 1.5, num_bodies).astype(np.float32), dtype=wp.float32)

    def step(self):
        """Advance the simulation by one timestep (acceleration computation + integration)."""
        wp.launch(
            kernel=compute_accelerations,
            dim=self.num_bodies,
            inputs=[self.positions, self.masses, self.softening_sq, self.gravity],
            outputs=[self.accelerations],
        )
        wp.launch(
            kernel=integrate,
            dim=self.num_bodies,
            inputs=[self.positions, self.velocities, self.accelerations, self.dt],
        )


if __name__ == "__main__":
    import argparse

    wp.config.lineinfo = True
    wp.config.line_directives = False
    wp.init()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num-bodies", type=int, default=8192, help="Number of bodies in the simulation.")
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=200,
        help="Iterations to run before the profiler capture range begins.",
    )
    parser.add_argument(
        "--num-profiled",
        type=int,
        default=100,
        help="Iterations to run inside the profiler capture range.",
    )

    args = parser.parse_known_args()[0]

    # The CUDA profiler capture range maps to cudaProfilerStart()/cudaProfilerStop(),
    # which only affect CUDA execution, so require a CUDA device and fail fast otherwise.
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError(
            f"This example requires a CUDA device, but the selected device is '{device}'. "
            "Run on a CUDA device, e.g. --device cuda:0."
        )

    with wp.ScopedDevice(device):
        example = Example(num_bodies=args.num_bodies)

        # warm up before the region of interest: this triggers kernel compilation,
        # allocations, and lets the GPU clocks ramp up, none of which we want in the
        # captured profile.
        with wp.ScopedTimer("warmup"):
            for _ in range(args.num_warmup):
                example.step()
            wp.synchronize_device()

        print(f"Warmup complete ({args.num_warmup} iterations); starting profiler capture range.")

        # wp.cuda_profiler_range() calls cudaProfilerStart()/cudaProfilerStop() around
        # the block. An attached profiler launched with --capture-range=cudaProfilerApi
        # (Nsight Systems) or --profile-from-start off (Nsight Compute) only collects
        # data for these iterations.
        with wp.cuda_profiler_range():
            with wp.ScopedTimer("profiled region"):
                for _ in range(args.num_profiled):
                    example.step()

                # make sure the trailing asynchronous kernels finish before
                # cudaProfilerStop() so they are included in the capture
                wp.synchronize_device()

        print(f"Profiled region complete ({args.num_profiled} iterations).")
