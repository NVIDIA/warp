# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Example Allocation Tracker
#
# A particle simulation instrumented with allocation tracking to show how
# to diagnose GPU memory usage.  The example demonstrates two APIs:
#
# Part 1 -- Global tracking  (wp.config.track_memory)
#   Enabled before wp.init(), tracks every allocation for the whole
#   process.  Call wp.print_memory_report() at any time to print the
#   current memory state.
#
# Part 2 -- Scoped tracking  (wp.ScopedMemoryTracker)
#   Wraps a specific code region.  Supports nested scopes, periodic
#   reports, and clear() for windowed reporting.
#   Zero overhead when not active.  Also captures C++ internal
#   allocations (BVH, HashGrid, etc.) with descriptive labels.
#
# Run:
#   python -m warp.examples.core.example_alloc_tracker
###########################################################################

import numpy as np

import warp as wp

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@wp.kernel
def apply_gravity(velocities: wp.array(dtype=wp.vec3), dt: float):
    tid = wp.tid()
    velocities[tid] = velocities[tid] + wp.vec3(0.0, -9.81, 0.0) * dt


@wp.kernel
def integrate(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dt: float,
):
    tid = wp.tid()
    positions[tid] = positions[tid] + velocities[tid] * dt


@wp.kernel
def enforce_ground(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    restitution: float,
):
    tid = wp.tid()
    p = positions[tid]
    v = velocities[tid]
    if p[1] < 0.0:
        positions[tid] = wp.vec3(p[0], 0.0, p[2])
        velocities[tid] = wp.vec3(v[0], wp.abs(v[1]) * restitution, v[2])


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


class ParticleSim:
    def __init__(self, n, device):
        self.n = n
        self.device = device
        self.dt = 1.0 / 60.0

        rng = np.random.default_rng(42)
        init_pos = rng.uniform(-5.0, 5.0, (n, 3)).astype(np.float32)
        init_pos[:, 1] = rng.uniform(2.0, 20.0, n).astype(np.float32)

        self.positions = wp.array(init_pos, dtype=wp.vec3, device=device)
        self.velocities = wp.zeros(n, dtype=wp.vec3, device=device)
        self.forces = wp.zeros(n, dtype=wp.vec3, device=device)

        # Regular host buffer for CPU-side analysis
        self.host_buffer = wp.zeros(n, dtype=wp.vec3, device="cpu")
        # Pinned host staging buffer for fast async GPU->CPU readback
        self.staging_buffer = wp.zeros(n, dtype=wp.vec3, device="cpu", pinned=True)

    def step(self):
        wp.launch(apply_gravity, dim=self.n, inputs=[self.velocities, self.dt], device=self.device)
        wp.launch(integrate, dim=self.n, inputs=[self.positions, self.velocities, self.dt], device=self.device)
        wp.launch(enforce_ground, dim=self.n, inputs=[self.positions, self.velocities, 0.6], device=self.device)

    def readback_positions(self):
        """Copy positions to pinned host memory, then to regular host buffer."""
        wp.copy(self.staging_buffer, self.positions)
        wp.copy(self.host_buffer, self.staging_buffer)
        return self.host_buffer

    def save_snapshot(self):
        """Clone the current positions -- intentional per-call allocation."""
        return wp.clone(self.positions)


# ---------------------------------------------------------------------------
# Part 1: Global tracking
# ---------------------------------------------------------------------------


def run_global(device, n_particles=200_000):
    """Demonstrate wp.config.track_memory + wp.print_memory_report()."""

    print("=" * 70)
    print("  Part 1: Global tracking  (wp.config.track_memory = True)")
    print("=" * 70)
    print()

    sim = ParticleSim(n_particles, device)
    sim.step()  # trigger kernel compilation before first report

    print()
    print("After creating simulation + first step:")
    print("(Note: GPU, host, and pinned host memory are listed separately)")
    wp.print_memory_report()

    for _step in range(1, 31):
        sim.step()

    snapshot = sim.save_snapshot()
    host_pos = sim.readback_positions()
    print()
    print(f"After 30 steps + snapshot + readback (mean y = {host_pos.numpy()[:, 1].mean():.2f}):")
    wp.print_memory_report()

    del sim, snapshot


# ---------------------------------------------------------------------------
# Part 2: Scoped tracking
# ---------------------------------------------------------------------------


def run_scoped(device, n_particles=200_000, n_steps=90, report_every=30):
    """Demonstrate wp.ScopedMemoryTracker with nested scopes."""

    print()
    print("=" * 70)
    print("  Part 2: Scoped tracking  (wp.ScopedMemoryTracker)")
    print("=" * 70)
    print()

    with wp.ScopedMemoryTracker("run", print=False) as tracker:
        with wp.ScopedMemoryTracker("init", print=False):
            sim = ParticleSim(n_particles, device)
            sim.step()  # trigger kernel compilation before first report

        print()
        print("After init + first step:")
        tracker.report()

        snapshots = []

        for step in range(1, n_steps + 1):
            sim.step()

            if step % report_every == 0:
                with wp.ScopedMemoryTracker("snapshot", print=False):
                    snapshots.append(sim.save_snapshot())

                print()
                print(f"Step {step}:")
                tracker.report()

        # clear() resets all counters -- useful for isolating a specific phase
        tracker.clear()

        with wp.ScopedMemoryTracker("after_clear", print=False):
            snapshots.append(sim.save_snapshot())

        print()
        print("After clear() + one more snapshot  (only new allocations):")
        tracker.report()

    del sim, snapshots

    print()
    print("After cleanup  (freed arrays gone from live list):")
    tracker.report()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Enable global tracking BEFORE wp.init() -- this is the key step.
    # Every allocation from init() onward is recorded.
    wp.config.track_memory = True
    wp.init()

    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    print(f"Device: {device}\n")

    run_global(device=device)
    run_scoped(device=device)
