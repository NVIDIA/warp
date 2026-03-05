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

"""Tests for the persistent ContactList used in DEM simulations.

The ContactList class is defined in
``warp/examples/core/example_dem_contact_list.py``.  These tests exercise
build, incremental update, and edge-case behaviour independently of the
full DEM simulation loop.
"""

import unittest

import warp as wp
from warp.examples.core.example_dem_contact_list import ContactList
from warp.tests.unittest_utils import add_function_test, get_test_devices


def test_contact_list_build(test, device):
    """Verify build produces correct contacts for a known particle arrangement.

    Four particles in a line (spacing 1.0) plus one isolated particle.
    Query radius = 1.5 means only adjacent particles should be neighbors.
    """
    positions = wp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],  # isolated
        ],
        dtype=wp.vec3,
        device=device,
    )

    grid = wp.HashGrid(32, 32, 32, device)
    grid.build(positions, 1.5)

    contacts = ContactList(5, max_contacts_per_particle=8, device=device)
    contacts.build(grid, positions, 1.5)

    counts = contacts.counts.numpy()
    neighbors = contacts.neighbors.numpy().reshape(5, 8)

    # Particle 0 → neighbor 1 only (d=1.0 < 1.5)
    test.assertEqual(counts[0], 1)
    test.assertEqual(neighbors[0, 0], 1)

    # Particle 1 → neighbors 0 and 2
    test.assertEqual(counts[1], 2)
    test.assertIn(0, neighbors[1, : counts[1]])
    test.assertIn(2, neighbors[1, : counts[1]])

    # Particle 2 → neighbors 1 and 3
    test.assertEqual(counts[2], 2)
    test.assertIn(1, neighbors[2, : counts[2]])
    test.assertIn(3, neighbors[2, : counts[2]])

    # Particle 3 → neighbor 2 only
    test.assertEqual(counts[3], 1)
    test.assertEqual(neighbors[3, 0], 2)

    # Particle 4 → isolated, no contacts
    test.assertEqual(counts[4], 0)


def test_contact_list_update_broken(test, device):
    """Verify update deactivates contacts when particles separate."""
    positions = wp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=wp.vec3,
        device=device,
    )

    grid = wp.HashGrid(32, 32, 32, device)
    grid.build(positions, 1.5)

    contacts = ContactList(2, max_contacts_per_particle=8, device=device)
    contacts.build(grid, positions, 1.5)

    # Sanity: both should see each other
    counts = contacts.counts.numpy()
    test.assertEqual(counts[0], 1)
    test.assertEqual(counts[1], 1)

    # Move particle 1 far away
    new_positions = wp.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=wp.vec3,
        device=device,
    )
    grid.build(new_positions, 1.5)
    contacts.update(grid, new_positions, 1.5, margin=0.5)

    counts = contacts.counts.numpy()
    test.assertEqual(counts[0], 0)
    test.assertEqual(counts[1], 0)


def test_contact_list_update_new(test, device):
    """Verify update discovers new contacts when particles approach."""
    positions = wp.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=wp.vec3,
        device=device,
    )

    grid = wp.HashGrid(32, 32, 32, device)
    grid.build(positions, 1.5)

    contacts = ContactList(2, max_contacts_per_particle=8, device=device)
    contacts.build(grid, positions, 1.5)

    counts = contacts.counts.numpy()
    test.assertEqual(counts[0], 0)  # no contacts initially
    test.assertEqual(counts[1], 0)

    # Move particle 1 close
    new_positions = wp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=wp.vec3,
        device=device,
    )
    grid.build(new_positions, 1.5)
    contacts.update(grid, new_positions, 1.5)

    counts = contacts.counts.numpy()
    test.assertEqual(counts[0], 1)
    test.assertEqual(counts[1], 1)


def test_contact_list_margin_preserves(test, device):
    """Verify margin prevents flickering for particles near the boundary.

    Particle distance is 1.6 (above radius 1.5) but within radius + margin
    (1.5 + 0.5 = 2.0), so the existing contact should survive.
    """
    positions = wp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=wp.vec3,
        device=device,
    )

    grid = wp.HashGrid(32, 32, 32, device)
    grid.build(positions, 1.5)

    contacts = ContactList(2, max_contacts_per_particle=8, device=device)
    contacts.build(grid, positions, 1.5)

    test.assertEqual(contacts.counts.numpy()[0], 1)

    # Move particle 1 slightly beyond radius but within margin
    new_positions = wp.array(
        [
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
        ],
        dtype=wp.vec3,
        device=device,
    )
    grid.build(new_positions, 2.0)
    contacts.update(grid, new_positions, 1.5, margin=0.5)

    # Contact should survive (distance 1.6 < 1.5 + 0.5)
    test.assertEqual(contacts.counts.numpy()[0], 1)


def test_contact_list_single_particle(test, device):
    """A single particle should have zero contacts after build."""
    positions = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device=device)

    grid = wp.HashGrid(32, 32, 32, device)
    grid.build(positions, 1.0)

    contacts = ContactList(1, max_contacts_per_particle=8, device=device)
    contacts.build(grid, positions, 1.0)

    test.assertEqual(contacts.counts.numpy()[0], 0)


def test_contact_list_rebuild_clears(test, device):
    """Calling build() again should clear previous contacts."""
    positions = wp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=wp.vec3,
        device=device,
    )

    grid = wp.HashGrid(32, 32, 32, device)
    grid.build(positions, 1.5)

    contacts = ContactList(2, max_contacts_per_particle=8, device=device)
    contacts.build(grid, positions, 1.5)
    test.assertEqual(contacts.counts.numpy()[0], 1)

    # Move particles far apart, then rebuild (not update)
    far_positions = wp.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=wp.vec3,
        device=device,
    )
    grid.build(far_positions, 1.5)
    contacts.build(grid, far_positions, 1.5)

    test.assertEqual(contacts.counts.numpy()[0], 0)
    test.assertEqual(contacts.counts.numpy()[1], 0)


def test_contact_list_no_duplicates(test, device):
    """Repeated updates should not create duplicate contact entries."""
    positions = wp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=wp.vec3,
        device=device,
    )

    grid = wp.HashGrid(32, 32, 32, device)
    grid.build(positions, 1.5)

    contacts = ContactList(2, max_contacts_per_particle=8, device=device)
    contacts.build(grid, positions, 1.5)

    # Run update several times with unchanged positions
    for _ in range(5):
        contacts.update(grid, positions, 1.5, margin=0.5)

    counts = contacts.counts.numpy()
    test.assertEqual(counts[0], 1)
    test.assertEqual(counts[1], 1)


def test_contact_list_data_persistence(test, device):
    """Verify per-contact data survives incremental updates.

    Writes known values into the per-contact data array, then runs
    several updates with unchanged positions.  The data should remain
    intact because the contacts are neither added nor removed.
    """
    positions = wp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=wp.vec3,
        device=device,
    )

    grid = wp.HashGrid(32, 32, 32, device)
    grid.build(positions, 1.5)

    contacts = ContactList(2, max_contacts_per_particle=8, data_width=2, device=device)
    contacts.build(grid, positions, 1.5)

    test.assertEqual(contacts.counts.numpy()[0], 1)

    # Write known values into particle 0's contact data.
    # Particle 0, contact slot 0, data_width=2: offset = (0*8 + 0)*2 = 0
    data_np = contacts.data.numpy()
    data_np[0] = 42.0
    data_np[1] = 7.0
    contacts.data = wp.array(data_np, dtype=wp.float32, device=device)

    # Run several updates with unchanged positions
    for _ in range(3):
        contacts.update(grid, positions, 1.5, margin=0.5)

    test.assertEqual(contacts.counts.numpy()[0], 1)
    data_after = contacts.data.numpy()
    test.assertAlmostEqual(float(data_after[0]), 42.0)
    test.assertAlmostEqual(float(data_after[1]), 7.0)


def test_contact_list_cluster(test, device):
    """Verify correct contact counts for a tight cluster of 4 particles."""
    # Square arrangement: all pairs within radius 1.5
    positions = wp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=wp.vec3,
        device=device,
    )

    grid = wp.HashGrid(32, 32, 32, device)
    grid.build(positions, 1.5)

    contacts = ContactList(4, max_contacts_per_particle=8, device=device)
    contacts.build(grid, positions, 1.5)

    counts = contacts.counts.numpy()
    # Distance between adjacent corners: 1.0 < 1.5 (within radius)
    # Distance between diagonal corners: sqrt(2) ~ 1.414 < 1.5 (within radius)
    # So every particle should see all 3 others
    for i in range(4):
        test.assertEqual(counts[i], 3, f"Particle {i} should have 3 contacts, got {counts[i]}")


def test_contact_list_capacity_overflow(test, device):
    """Verify that contacts are capped at max_contacts_per_particle.

    When the true neighbor count exceeds max_cpn, excess contacts are
    silently truncated.  This test documents that behaviour.
    """
    # 6 particles in a line with spacing 0.1, query radius 0.45
    # covers up to 4 neighbors per particle, but we cap at 2.
    positions = wp.array(
        [[0.1 * i, 0.0, 0.0] for i in range(6)],
        dtype=wp.vec3,
        device=device,
    )

    grid = wp.HashGrid(32, 32, 32, device)
    grid.build(positions, 0.45)

    contacts = ContactList(6, max_contacts_per_particle=2, device=device)
    contacts.build(grid, positions, 0.45)

    counts = contacts.counts.numpy()
    for i in range(6):
        test.assertLessEqual(counts[i], 2, f"Particle {i} exceeds max_cpn")

    # Middle particles would have 4 true neighbors but are capped at 2
    test.assertEqual(counts[2], 2)
    test.assertEqual(counts[3], 2)


devices = get_test_devices()


class TestContactList(unittest.TestCase):
    pass


add_function_test(TestContactList, "test_contact_list_build", test_contact_list_build, devices=devices)
add_function_test(TestContactList, "test_contact_list_update_broken", test_contact_list_update_broken, devices=devices)
add_function_test(TestContactList, "test_contact_list_update_new", test_contact_list_update_new, devices=devices)
add_function_test(
    TestContactList, "test_contact_list_margin_preserves", test_contact_list_margin_preserves, devices=devices
)
add_function_test(
    TestContactList, "test_contact_list_single_particle", test_contact_list_single_particle, devices=devices
)
add_function_test(
    TestContactList, "test_contact_list_rebuild_clears", test_contact_list_rebuild_clears, devices=devices
)
add_function_test(TestContactList, "test_contact_list_no_duplicates", test_contact_list_no_duplicates, devices=devices)
add_function_test(
    TestContactList, "test_contact_list_data_persistence", test_contact_list_data_persistence, devices=devices
)
add_function_test(TestContactList, "test_contact_list_cluster", test_contact_list_cluster, devices=devices)
add_function_test(
    TestContactList, "test_contact_list_capacity_overflow", test_contact_list_capacity_overflow, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
