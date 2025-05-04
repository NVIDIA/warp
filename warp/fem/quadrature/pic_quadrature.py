# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Optional, Tuple, Union

import warp as wp
from warp.fem.cache import TemporaryStore, borrow_temporary, cached_arg_value, dynamic_kernel
from warp.fem.domain import GeometryDomain
from warp.fem.types import NULL_ELEMENT_INDEX, Coords, ElementIndex, make_free_sample
from warp.fem.utils import compress_node_indices

from .quadrature import Quadrature


class PicQuadrature(Quadrature):
    """Particle-based quadrature formula, using a global set of points unevenly spread out over geometry elements.

    Useful for Particle-In-Cell and derived methods.

    Args:
        domain: Underlying domain for the quadrature
        positions: Either an array containing the world positions of all particles, or a tuple of arrays containing
         the cell indices and coordinates for each particle.
        measures: Array containing the measure (area/volume) of each particle, used to defined the integration weights.
         If ``None``, defaults to the cell measure divided by the number of particles in the cell.
        max_dist: When providing world positions that fall outside of the domain's geometry partition, maximum distance to look up for embedding cells
        requires_grad: Whether gradients should be allocated for the computed quantities
        temporary_store: shared pool from which to allocate temporary arrays
    """

    def __init__(
        self,
        domain: GeometryDomain,
        positions: Union[
            "wp.array(dtype=wp.vecXd)",
            Tuple[
                "wp.array(dtype=ElementIndex)",
                "wp.array(dtype=Coords)",
            ],
        ],
        measures: Optional["wp.array(dtype=float)"] = None,
        requires_grad: bool = False,
        max_dist: float = 0.0,
        temporary_store: TemporaryStore = None,
    ):
        super().__init__(domain)

        self._requires_grad = requires_grad
        self._bin_particles(positions, measures, max_dist=max_dist, temporary_store=temporary_store)
        self._max_particles_per_cell: int = None

    @property
    def name(self):
        return f"{self.__class__.__name__}"

    @Quadrature.domain.setter
    def domain(self, domain: GeometryDomain):
        # Allow changing the quadrature domain as long as underlying geometry and element kind are the same
        if self.domain is not None and (
            domain.element_kind != self.domain.element_kind or domain.geometry.base != self.domain.geometry.base
        ):
            raise RuntimeError(
                "The new domain must use the same base geometry and kind of elements as the current one."
            )

        self._domain = domain

    @wp.struct
    class Arg:
        cell_particle_offsets: wp.array(dtype=int)
        cell_particle_indices: wp.array(dtype=int)
        particle_fraction: wp.array(dtype=float)
        particle_coords: wp.array(dtype=Coords)

    @cached_arg_value
    def arg_value(self, device) -> Arg:
        arg = PicQuadrature.Arg()
        arg.cell_particle_offsets = self._cell_particle_offsets.array.to(device)
        arg.cell_particle_indices = self._cell_particle_indices.array.to(device)
        arg.particle_fraction = self._particle_fraction.to(device)
        arg.particle_coords = self.particle_coords.to(device)
        return arg

    def total_point_count(self):
        return self.particle_coords.shape[0]

    def active_cell_count(self):
        """Number of cells containing at least one particle"""
        return self._cell_count

    def max_points_per_element(self):
        if self._max_particles_per_cell is None:
            max_ppc = wp.zeros(shape=(1,), dtype=int, device=self._cell_particle_offsets.array.device)
            wp.launch(
                PicQuadrature._max_particles_per_cell_kernel,
                self._cell_particle_offsets.array.shape[0] - 1,
                device=max_ppc.device,
                inputs=[self._cell_particle_offsets.array, max_ppc],
            )
            self._max_particles_per_cell = int(max_ppc.numpy()[0])
        return self._max_particles_per_cell

    @wp.func
    def point_count(elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex):
        return qp_arg.cell_particle_offsets[element_index + 1] - qp_arg.cell_particle_offsets[element_index]

    @wp.func
    def point_coords(
        elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex, index: int
    ):
        particle_index = qp_arg.cell_particle_indices[qp_arg.cell_particle_offsets[element_index] + index]
        return qp_arg.particle_coords[particle_index]

    @wp.func
    def point_weight(
        elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex, index: int
    ):
        particle_index = qp_arg.cell_particle_indices[qp_arg.cell_particle_offsets[element_index] + index]
        return qp_arg.particle_fraction[particle_index]

    @wp.func
    def point_index(
        elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex, index: int
    ):
        particle_index = qp_arg.cell_particle_indices[qp_arg.cell_particle_offsets[element_index] + index]
        return particle_index

    @wp.func
    def point_evaluation_index(
        elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex, index: int
    ):
        return qp_arg.cell_particle_offsets[element_index] + index

    def fill_element_mask(self, mask: "wp.array(dtype=int)"):
        """Fills a mask array such that all non-empty elements are set to 1, all empty elements to zero.

        Args:
            mask: Int warp array with size at least equal to `self.domain.geometry_element_count()`
        """

        wp.launch(
            kernel=PicQuadrature._fill_mask_kernel,
            dim=self.domain.geometry_element_count(),
            device=mask.device,
            inputs=[self._cell_particle_offsets.array, mask],
        )

    @wp.kernel
    def _fill_mask_kernel(
        element_particle_offsets: wp.array(dtype=int),
        element_mask: wp.array(dtype=int),
    ):
        i = wp.tid()
        element_mask[i] = wp.where(element_particle_offsets[i] == element_particle_offsets[i + 1], 0, 1)

    @wp.kernel
    def _compute_uniform_fraction(
        cell_index: wp.array(dtype=ElementIndex),
        cell_particle_offsets: wp.array(dtype=int),
        cell_fraction: wp.array(dtype=float),
    ):
        p = wp.tid()

        cell = cell_index[p]
        if cell == NULL_ELEMENT_INDEX:
            cell_fraction[p] = 0.0
        else:
            cell_particle_count = cell_particle_offsets[cell + 1] - cell_particle_offsets[cell]
            cell_fraction[p] = 1.0 / float(cell_particle_count)

    def _bin_particles(self, positions, measures, max_dist: float, temporary_store: TemporaryStore):
        if wp.types.is_array(positions):
            device = positions.device
            if not self.domain.supports_lookup(device):
                raise RuntimeError(
                    "Attempting to build a PicQuadrature from positions on a domain that does not support global lookups"
                )

            cell_lookup = self.domain.element_partition_lookup
            cell_coordinates = self.domain.element_coordinates

            # Initialize from positions
            @dynamic_kernel(suffix=self.domain.name)
            def bin_particles(
                cell_arg_value: self.domain.ElementArg,
                domain_index_arg_value: self.domain.ElementIndexArg,
                positions: wp.array(dtype=positions.dtype),
                max_dist: float,
                cell_index: wp.array(dtype=ElementIndex),
                cell_coords: wp.array(dtype=Coords),
            ):
                p = wp.tid()
                sample = cell_lookup(
                    self.domain.DomainArg(cell_arg_value, domain_index_arg_value), positions[p], max_dist
                )

                cell_index[p] = sample.element_index
                cell_coords[p] = cell_coordinates(cell_arg_value, sample.element_index, positions[p])

            self._cell_index_temp = borrow_temporary(temporary_store, shape=positions.shape, dtype=int, device=device)
            self.cell_indices = self._cell_index_temp.array

            self._particle_coords_temp = borrow_temporary(
                temporary_store, shape=positions.shape, dtype=Coords, device=device, requires_grad=self._requires_grad
            )
            self.particle_coords = self._particle_coords_temp.array

            wp.launch(
                dim=positions.shape[0],
                kernel=bin_particles,
                inputs=[
                    self.domain.element_arg_value(device),
                    self.domain.element_index_arg_value(device),
                    positions,
                    max_dist,
                ],
                outputs=[
                    self.cell_indices,
                    self.particle_coords,
                ],
                device=device,
            )

        else:
            self.cell_indices, self.particle_coords = positions
            if self.cell_indices.shape != self.particle_coords.shape:
                raise ValueError("Cell index and coordinates arrays must have the same shape")

            self._cell_index_temp = None
            self._particle_coords_temp = None

        self._cell_particle_offsets, self._cell_particle_indices, self._cell_count, _ = compress_node_indices(
            self.domain.geometry_element_count(),
            self.cell_indices,
            return_unique_nodes=True,
            temporary_store=temporary_store,
        )

        self._compute_fraction(self.cell_indices, measures, temporary_store)

    def _compute_fraction(self, cell_index, measures, temporary_store: TemporaryStore):
        device = cell_index.device

        self._particle_fraction_temp = borrow_temporary(
            temporary_store, shape=cell_index.shape, dtype=float, device=device, requires_grad=self._requires_grad
        )
        self._particle_fraction = self._particle_fraction_temp.array

        if measures is None:
            # Split fraction uniformly over all particles in cell

            wp.launch(
                dim=cell_index.shape,
                kernel=PicQuadrature._compute_uniform_fraction,
                inputs=[
                    cell_index,
                    self._cell_particle_offsets.array,
                    self._particle_fraction,
                ],
                device=device,
            )

        else:
            # Fraction from particle measure

            if measures.shape != cell_index.shape:
                raise ValueError("Measures should be an 1d array or length equal to particle count")

            @dynamic_kernel(suffix=f"{self.domain.name}")
            def compute_fraction(
                cell_arg_value: self.domain.ElementArg,
                measures: wp.array(dtype=float),
                cell_index: wp.array(dtype=ElementIndex),
                cell_coords: wp.array(dtype=Coords),
                cell_fraction: wp.array(dtype=float),
            ):
                p = wp.tid()

                cell = cell_index[p]
                if cell == NULL_ELEMENT_INDEX:
                    cell_fraction[p] = 0.0
                else:
                    sample = make_free_sample(cell_index[p], cell_coords[p])
                    cell_fraction[p] = measures[p] / self.domain.element_measure(cell_arg_value, sample)

            wp.launch(
                dim=measures.shape[0],
                kernel=compute_fraction,
                inputs=[
                    self.domain.element_arg_value(device),
                    measures,
                    cell_index,
                    self.particle_coords,
                    self._particle_fraction,
                ],
                device=device,
            )

    @wp.kernel
    def _max_particles_per_cell_kernel(offsets: wp.array(dtype=int), max_count: wp.array(dtype=int)):
        cell = wp.tid()
        particle_count = offsets[cell + 1] - offsets[cell]
        wp.atomic_max(max_count, 0, particle_count)
