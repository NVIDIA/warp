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
# Example DEM with Persistent Contact List
#
# Demonstrates a persistent contact list for bonded-particle (DEM)
# simulations. Instead of rebuilding neighbor queries from scratch every
# substep, a ContactList stores particle pairs and incrementally adds or
# removes contacts as particles move. This preserves contact state across
# substeps, which is essential for accurate bond tracking in cohesive
# materials.
#
# Compare with ``example_dem.py``, which rebuilds all neighbor queries
# every frame via :class:`warp.HashGrid`.
#
# See https://github.com/NVIDIA/warp/issues/1056 for background.
###########################################################################

import numpy as np

import warp as wp
import warp.render

# -----------------------------------------------------------------------
# ContactList — persistent neighbor storage for particle simulations
# -----------------------------------------------------------------------


@wp.kernel
def _contact_build_kernel(
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    radius: float,
    neighbors: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    max_cpn: int,
):
    """Full build: query HashGrid and write all contacts into flat slots."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    x = positions[i]
    slot = i * max_cpn
    count = int(0)

    for j in wp.hash_grid_query(grid, x, radius):
        if j != i and count < max_cpn:
            d = wp.length(x - positions[j])
            if d < radius:
                neighbors[slot + count] = j
                count += 1

    counts[i] = count


@wp.kernel
def _contact_update_kernel(
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    radius: float,
    margin: float,
    neighbors: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    max_cpn: int,
):
    """Incremental update: deactivate broken contacts, discover new ones."""
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    x = positions[i]
    slot = i * max_cpn
    current_count = counts[i]

    # Pass 1 — mark broken contacts by swapping with the last active entry.
    # Walk backwards so that swaps don't skip elements.
    k = current_count - 1
    while k >= 0:
        j = neighbors[slot + k]
        d = wp.length(x - positions[j])
        if d > radius + margin:
            # Swap with last active entry and shrink
            current_count -= 1
            neighbors[slot + k] = neighbors[slot + current_count]
            neighbors[slot + current_count] = -1
        k -= 1

    # Pass 2 — discover new contacts from the grid query.
    for j_new in wp.hash_grid_query(grid, x, radius):
        if j_new != i:
            d = wp.length(x - positions[j_new])
            if d < radius:
                # Check if already stored
                found = int(0)
                for c in range(current_count):
                    if neighbors[slot + c] == j_new:
                        found = 1
                if found == 0 and current_count < max_cpn:
                    neighbors[slot + current_count] = j_new
                    current_count += 1

    counts[i] = current_count


class ContactList:
    """Persistent contact/neighbor list for particle simulations.

    Stores particle pairs discovered by :class:`warp.HashGrid` and
    persists them across timesteps.  Contacts are only added or removed
    when the particle topology actually changes, saving redundant spatial
    queries and preserving per-contact state (e.g. bond age, accumulated
    friction).

    Storage layout (flat CSR-like):
        ``neighbors[i * max_cpn .. i * max_cpn + counts[i])`` holds the
        neighbor indices for particle *i*.

    Args:
        max_particles: Maximum number of particles.
        max_contacts_per_particle: Maximum stored neighbors per particle.
        device: Warp device for array allocation.
    """

    def __init__(self, max_particles: int, max_contacts_per_particle: int = 32, device=None):
        self.max_particles = max_particles
        self.max_cpn = max_contacts_per_particle
        self.device = device

        total_slots = max_particles * max_contacts_per_particle
        self.neighbors = wp.full(total_slots, value=-1, dtype=wp.int32, device=device)
        self.counts = wp.zeros(max_particles, dtype=wp.int32, device=device)

    def build(self, grid: wp.HashGrid, positions: wp.array, radius: float) -> None:
        """Populate the contact list from a full :class:`warp.HashGrid` query.

        This clears any previous contacts and performs a complete rebuild.

        Args:
            grid: A :class:`warp.HashGrid` that has already been built.
            positions: ``wp.array[wp.vec3]`` of particle positions.
            radius: Search radius for neighbor queries.
        """
        self.neighbors.fill_(-1)
        self.counts.zero_()
        wp.launch(
            kernel=_contact_build_kernel,
            dim=len(positions),
            inputs=[grid.id, positions, radius, self.neighbors, self.counts, self.max_cpn],
            device=self.device,
        )

    def update(self, grid: wp.HashGrid, positions: wp.array, radius: float, margin: float = 0.0) -> None:
        """Incrementally update the contact list.

        Contacts whose particles have separated beyond ``radius + margin``
        are removed.  New contacts discovered by the grid query are added.

        Args:
            grid: A :class:`warp.HashGrid` (must be built for this frame).
            positions: ``wp.array[wp.vec3]`` of current positions.
            radius: Search radius for neighbor queries.
            margin: Extra tolerance before a contact is removed.  A small
                positive value prevents contacts from flickering when
                particles oscillate near the boundary.
        """
        wp.launch(
            kernel=_contact_update_kernel,
            dim=len(positions),
            inputs=[grid.id, positions, radius, margin, self.neighbors, self.counts, self.max_cpn],
            device=self.device,
        )


# -----------------------------------------------------------------------
# DEM force / integration kernels
# -----------------------------------------------------------------------


@wp.func
def contact_force(n: wp.vec3, v: wp.vec3, c: float, k_n: float, k_d: float, k_f: float, k_mu: float):
    vn = wp.dot(n, v)
    jn = c * k_n
    jd = wp.min(vn, 0.0) * k_d

    fn = jn + jd

    vt = v - n * vn
    vs = wp.length(vt)
    if vs > 0.0:
        vt = vt / vs

    ft = wp.min(vs * k_f, k_mu * wp.abs(fn))
    return -n * fn - vt * ft


@wp.kernel
def apply_forces_contact_list(
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    neighbors: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    max_cpn: int,
    radius: float,
    k_contact: float,
    k_damp: float,
    k_friction: float,
    k_mu: float,
):
    """Compute DEM contact forces using the persistent contact list."""
    i = wp.tid()
    x = particle_x[i]
    v = particle_v[i]
    f = wp.vec3()

    # Ground contact
    n = wp.vec3(0.0, 1.0, 0.0)
    c = wp.dot(n, x)
    cohesion_ground = 0.02
    if c < cohesion_ground:
        f = f + contact_force(n, v, c, k_contact, k_damp, 100.0, 0.5)

    # Particle contacts from the persistent list
    cohesion_particle = 0.0075
    slot = i * max_cpn
    num_contacts = counts[i]
    for k in range(num_contacts):
        j = neighbors[slot + k]
        if j >= 0:
            n = x - particle_x[j]
            d = wp.length(n)
            err = d - radius * 2.0
            if err <= cohesion_particle:
                n = n / d
                vrel = v - particle_v[j]
                f = f + contact_force(n, vrel, err, k_contact, k_damp, k_friction, k_mu)

    particle_f[i] = f


@wp.kernel
def integrate(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    f: wp.array[wp.vec3],
    gravity: wp.vec3,
    dt: float,
    inv_mass: float,
):
    tid = wp.tid()
    v_new = v[tid] + f[tid] * inv_mass * dt + gravity * dt
    x_new = x[tid] + v_new * dt
    v[tid] = v_new
    x[tid] = x_new


# -----------------------------------------------------------------------
# Example simulation
# -----------------------------------------------------------------------


class Example:
    def __init__(self, stage_path="example_dem_contact_list.usd", num_particles_axis=16):
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 64
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.point_radius = 0.1

        self.k_contact = 8000.0
        self.k_damp = 2.0
        self.k_friction = 1.0
        self.k_mu = 100000.0

        self.inv_mass = 64.0

        self.grid = wp.HashGrid(128, 128, 128)
        self.grid_cell_size = self.point_radius * 5.0

        n = num_particles_axis
        self.points = self.particle_grid(n, n * 2, n, (0.0, 0.5, 0.0), self.point_radius, 0.1)

        self.x = wp.array(self.points, dtype=wp.vec3)
        self.v = wp.array(np.ones([len(self.x), 3]) * np.array([0.0, 0.0, 15.0]), dtype=wp.vec3)
        self.f = wp.zeros_like(self.v)

        # Persistent contact list — replaces per-substep HashGrid queries
        self.contacts = ContactList(len(self.x), max_contacts_per_particle=32)

        self.frame_count = 0

        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)
            self.renderer.render_ground()
        else:
            self.renderer = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            wp.launch(
                kernel=apply_forces_contact_list,
                dim=len(self.x),
                inputs=[
                    self.x,
                    self.v,
                    self.f,
                    self.contacts.neighbors,
                    self.contacts.counts,
                    self.contacts.max_cpn,
                    self.point_radius,
                    self.k_contact,
                    self.k_damp,
                    self.k_friction,
                    self.k_mu,
                ],
            )
            wp.launch(
                kernel=integrate,
                dim=len(self.x),
                inputs=[self.x, self.v, self.f, (0.0, -9.8, 0.0), self.sim_dt, self.inv_mass],
            )

    def step(self):
        with wp.ScopedTimer("step"):
            # Rebuild the spatial hash grid
            self.grid.build(self.x, self.grid_cell_size)

            # Full build on first frame, incremental update afterwards
            query_radius = self.point_radius * 5.0
            if self.frame_count == 0:
                self.contacts.build(self.grid, self.x, query_radius)
            else:
                self.contacts.update(self.grid, self.x, query_radius, margin=self.point_radius)

            self.simulate()

            self.frame_count += 1
            self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_points(
                points=self.x.numpy(), radius=self.point_radius, name="points", colors=(0.8, 0.3, 0.2)
            )
            self.renderer.end_frame()

    def particle_grid(self, dim_x, dim_y, dim_z, lower, radius, jitter):
        rng = np.random.default_rng(42)
        points = np.meshgrid(
            np.linspace(0, dim_x, dim_x),
            np.linspace(0, dim_y, dim_y),
            np.linspace(0, dim_z, dim_z),
        )
        points_t = np.array((points[0], points[1], points[2])).T * radius * 2.0 + np.array(lower)
        points_t = points_t + rng.random(size=points_t.shape) * radius * jitter
        return points_t.reshape((-1, 3))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_dem_contact_list.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=200, help="Total number of frames.")
    parser.add_argument("--num-particles-axis", type=int, default=16, help="Particles per grid axis.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_particles_axis=args.num_particles_axis)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
