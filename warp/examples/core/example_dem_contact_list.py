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
# simulations.  A ``ContactList`` stores particle pairs discovered by
# :class:`warp.HashGrid` and persists them across timesteps, optionally
# carrying per-contact floating-point data (e.g. rest lengths, bond
# flags) so that contact state survives incremental updates.
#
# The example builds a bonded block of particles, drops it under gravity,
# and shows bonds breaking on impact with the ground plane.  Bonds that
# exceed a strain threshold are permanently broken and no longer exert
# forces, causing the block to fracture into fragments.
#
# Compare with ``example_dem.py``, which rebuilds all neighbor queries
# every frame via :class:`warp.HashGrid`.
#
# See https://github.com/NVIDIA/warp/issues/1056 for background.
###########################################################################

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import warp as wp
import warp.render

if TYPE_CHECKING:
    from warp import DeviceLike

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
    data: wp.array[wp.float32],
    max_cpn: int,
    data_width: int,
):
    """Incremental update: deactivate broken contacts, discover new ones.

    Per-contact data entries are swapped/cleared in sync with neighbor
    entries so that metadata stays associated with the correct pair.
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    x = positions[i]
    slot = i * max_cpn
    current_count = counts[i]

    # Pass 1 — evict contacts that have moved out of range (geometric separation).
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
            # Swap associated per-contact data
            for f in range(data_width):
                src = (slot + current_count) * data_width + f
                dst = (slot + k) * data_width + f
                data[dst] = data[src]
                data[src] = 0.0
        k -= 1

    # Pass 2 — discover new contacts from the grid query.
    for j_new in wp.hash_grid_query(grid, x, radius):
        if j_new != i:
            d = wp.length(x - positions[j_new])
            if d < radius:
                # Duplicate check: O(current_count) scan — acceptable for small max_cpn.
                found = int(0)
                for c in range(current_count):
                    if neighbors[slot + c] == j_new:
                        found = 1
                if found == 0 and current_count < max_cpn:
                    # Zero data for the new contact slot
                    for f in range(data_width):
                        data[(slot + current_count) * data_width + f] = 0.0
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

        When ``data_width > 0``, per-contact floating-point data is stored
        in ``data`` with the same slot ordering:
        ``data[(i * max_cpn + k) * data_width + field]``.

    Note:
        If a particle has more neighbors within the search radius than
        ``max_contacts_per_particle``, the excess contacts are silently
        truncated.  Choose a value large enough for your packing density.

    Args:
        max_particles: Maximum number of particles.
        max_contacts_per_particle: Maximum stored neighbors per particle.
        data_width: Number of floats stored per contact.  Set to 0
            (default) to disable per-contact data.
        device: Warp device for array allocation.
    """

    def __init__(
        self,
        max_particles: int,
        max_contacts_per_particle: int = 32,
        data_width: int = 0,
        device: DeviceLike | None = None,
    ):
        self.max_particles = max_particles
        self.max_cpn = max_contacts_per_particle
        self.data_width = data_width
        self.device = device

        total_slots = max_particles * max_contacts_per_particle
        self.neighbors = wp.full(total_slots, value=-1, dtype=wp.int32, device=device)
        self.counts = wp.zeros(max_particles, dtype=wp.int32, device=device)
        # Per-contact data; allocate at least one element to avoid zero-length arrays.
        self.data = wp.zeros(max(total_slots * data_width, 1), dtype=wp.float32, device=device)

    def build(self, grid: wp.HashGrid, positions: wp.array, radius: float) -> None:
        """Populate the contact list from a full :class:`warp.HashGrid` query.

        This clears any previous contacts (and per-contact data) and
        performs a complete rebuild.

        Args:
            grid: A :class:`warp.HashGrid` built with
                ``cell_size >= radius`` to ensure all neighbors are found.
            positions: ``wp.array[wp.vec3]`` of particle positions.
            radius: Search radius for neighbor queries.
        """
        n = len(positions)
        if n > self.max_particles:
            raise ValueError(f"positions length ({n}) exceeds max_particles ({self.max_particles})")
        self.neighbors.fill_(-1)
        self.counts.zero_()
        self.data.zero_()
        wp.launch(
            kernel=_contact_build_kernel,
            dim=n,
            inputs=[grid.id, positions, radius, self.neighbors, self.counts, self.max_cpn],
            device=self.device,
        )

    def update(self, grid: wp.HashGrid, positions: wp.array, radius: float, margin: float = 0.0) -> None:
        """Incrementally update the contact list.

        Contacts whose particles have separated beyond ``radius + margin``
        are removed.  New contacts discovered by the grid query are added
        with zero-initialized per-contact data; callers that require
        non-zero defaults (e.g. rest lengths) must initialize them after
        the update.  Per-contact data entries are swapped/cleared in sync
        with their neighbor entries.

        Args:
            grid: A :class:`warp.HashGrid` built with
                ``cell_size >= radius`` for this frame.
            positions: ``wp.array[wp.vec3]`` of current positions.
            radius: Search radius for neighbor queries.
            margin: Extra tolerance before a contact is removed.  A small
                positive value prevents contacts from flickering when
                particles oscillate near the boundary.
        """
        n = len(positions)
        if n > self.max_particles:
            raise ValueError(f"positions length ({n}) exceeds max_particles ({self.max_particles})")
        wp.launch(
            kernel=_contact_update_kernel,
            dim=n,
            inputs=[
                grid.id,
                positions,
                radius,
                margin,
                self.neighbors,
                self.counts,
                self.data,
                self.max_cpn,
                self.data_width,
            ],
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
def _init_bond_data_kernel(
    positions: wp.array[wp.vec3],
    neighbors: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    data: wp.array[wp.float32],
    max_cpn: int,
    data_width: int,
):
    """Compute rest lengths for all existing contacts and store in data.

    Data layout per contact:
        field 0 -- rest length
        field 1 -- broken flag (0.0 = intact, 1.0 = broken)
    """
    i = wp.tid()
    slot = i * max_cpn
    for k in range(counts[i]):
        j = neighbors[slot + k]
        if j >= 0:
            rest_len = wp.length(positions[i] - positions[j])
            dpos = (slot + k) * data_width
            data[dpos + 0] = rest_len
            data[dpos + 1] = 0.0


@wp.kernel
def apply_bond_forces(
    particle_x: wp.array[wp.vec3],
    particle_v: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    neighbors: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    data: wp.array[wp.float32],
    max_cpn: int,
    data_width: int,
    point_radius: float,
    k_bond: float,
    k_damp_bond: float,
    break_strain: float,
    k_contact: float,
    k_damp: float,
    k_friction: float,
    k_mu: float,
):
    """Compute forces from bonded contacts and ground contact.

    Intact bonds exert spring + damping forces.  Bonds that exceed
    ``break_strain`` are permanently marked as broken.  Broken bonds
    (and contacts with no bond data, e.g. newly discovered by
    ``update()``) fall back to standard DEM repulsion so that
    fragments do not interpenetrate.

    Requires ``data_width >= 2`` with the following per-contact layout:
        field 0 -- rest length,  field 1 -- broken flag (0.0/1.0).
    """
    i = wp.tid()
    x = particle_x[i]
    v = particle_v[i]
    f = wp.vec3()

    # Ground contact (standard DEM)
    n = wp.vec3(0.0, 1.0, 0.0)
    c = wp.dot(n, x)
    cohesion_ground = 0.02
    if c < cohesion_ground:
        f = f + contact_force(n, v, c, k_contact, k_damp, k_friction, k_mu)

    # Bond / contact forces from the persistent contact list
    slot = i * max_cpn
    for k in range(counts[i]):
        j = neighbors[slot + k]
        if j >= 0:
            dpos = (slot + k) * data_width
            rest_len = data[dpos + 0]
            broken = data[dpos + 1]

            diff = x - particle_x[j]
            d = wp.length(diff)

            if d > 1.0e-6:
                if broken < 0.5 and rest_len > 1.0e-6:
                    # Active bond: spring + damping + breakage check
                    n_bond = diff / d
                    stretch = d - rest_len
                    strain = stretch / rest_len

                    if wp.abs(strain) > break_strain:
                        data[dpos + 1] = 1.0
                    else:
                        fn_bond = -k_bond * stretch
                        vrel = v - particle_v[j]
                        vn_bond = wp.dot(vrel, n_bond)
                        fd_bond = -k_damp_bond * vn_bond
                        f = f + n_bond * (fn_bond + fd_bond)
                else:
                    # Broken bond or non-bonded: standard DEM repulsion
                    err = d - point_radius * 2.0
                    if err < 0.0:
                        n_contact = diff / d
                        vrel = v - particle_v[j]
                        f = f + contact_force(n_contact, vrel, err, k_contact, k_damp, k_friction, k_mu)

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
# Example simulation: bonded block fracture
# -----------------------------------------------------------------------


class Example:
    def __init__(self, stage_path="example_dem_contact_list.usd", num_particles_axis=8):
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 64
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.point_radius = 0.1

        # Bond parameters
        self.k_bond = 4000.0
        self.k_damp_bond = 5.0
        self.break_strain = 0.15

        # Ground contact parameters
        self.k_contact = 8000.0
        self.k_damp = 2.0
        self.k_friction = 1.0
        self.k_mu = 100000.0

        self.inv_mass = 64.0

        self.grid = wp.HashGrid(128, 128, 128)
        self.grid_cell_size = self.point_radius * 5.0

        n = num_particles_axis
        self.points = self.particle_grid(n, n * 2, n, (0.0, 1.5, 0.0), self.point_radius, 0.1)

        self.x = wp.array(self.points, dtype=wp.vec3)
        self.v = wp.zeros(len(self.x), dtype=wp.vec3)
        self.f = wp.zeros_like(self.v)

        # Persistent contact list with per-contact data:
        #   field 0: rest length, field 1: broken flag
        self.contacts = ContactList(len(self.x), max_contacts_per_particle=32, data_width=2)

        # Build initial contacts and compute rest lengths
        self.grid.build(self.x, self.grid_cell_size)
        query_radius = self.point_radius * 5.0
        self.contacts.build(self.grid, self.x, query_radius)
        wp.launch(
            kernel=_init_bond_data_kernel,
            dim=len(self.x),
            inputs=[
                self.x,
                self.contacts.neighbors,
                self.contacts.counts,
                self.contacts.data,
                self.contacts.max_cpn,
                self.contacts.data_width,
            ],
        )

        self.frame_count = 0

        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)
            self.renderer.render_ground()
        else:
            self.renderer = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            wp.launch(
                kernel=apply_bond_forces,
                dim=len(self.x),
                inputs=[
                    self.x,
                    self.v,
                    self.f,
                    self.contacts.neighbors,
                    self.contacts.counts,
                    self.contacts.data,
                    self.contacts.max_cpn,
                    self.contacts.data_width,
                    self.point_radius,
                    self.k_bond,
                    self.k_damp_bond,
                    self.break_strain,
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
            # Rebuild the spatial hash grid each frame
            self.grid.build(self.x, self.grid_cell_size)

            # Incremental update -- contacts are only added/removed when
            # particles move in or out of range.  Bond data (rest lengths,
            # broken flags) travels with the contact entries.
            query_radius = self.point_radius * 5.0
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
    parser.add_argument("--num-particles-axis", type=int, default=8, help="Particles per grid axis.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_particles_axis=args.num_particles_axis)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
