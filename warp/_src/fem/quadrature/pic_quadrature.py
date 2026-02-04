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

from typing import Any, Optional, Union

import warp as wp
from warp._src.fem.cache import TemporaryStore, borrow_temporary, dynamic_kernel
from warp._src.fem.domain import GeometryDomain
from warp._src.fem.types import (
    NULL_ELEMENT_INDEX,
    NULL_NODE_INDEX,
    NULL_QP_INDEX,
    OUTSIDE,
    Coords,
    ElementIndex,
    make_free_sample,
)
from warp._src.fem.utils import compress_node_indices
from warp.types import is_array

from .quadrature import Quadrature

_wp_module_name_ = "warp.fem.quadrature.pic_quadrature"


class PicQuadrature(Quadrature):
    """Particle-based quadrature formula, using a global set of points unevenly spread out over geometry elements.

    Useful for Particle-In-Cell and derived methods (Material Point Method, Generalized Interpolation Material Point Method, etc).

    Args:
        domain: Underlying domain for the quadrature
        positions: Defines the location of the quadrature points. Cane be:

         - an array containing the world positions of all particles
         - a tuple of arrays containing the cell indices and coordinates for each particle.
         - for GIMP-style integration where particle can span multiple elements: A tuple of 2d array the element indices for each particle

        measures: Array containing the measure (area/volume) of each particle, used to defined the integration weights.
         If ``None``, defaults to the cell measure divided by the number of particles in the cell.
        max_dist: When providing world positions that fall outside of the domain's geometry partition, maximum distance to look up for embedding cells
        requires_grad: Whether gradients should be allocated for the computed quantities
        use_domain_element_indices: Whether to use the domain element indices instead of the full geometry cell indices. This reduces memory usage,
            but prevents changing the quadrature domain a-posteriori.
        temporary_store: shared pool from which to allocate temporary arrays
    """

    def __init__(
        self,
        domain: GeometryDomain,
        positions: Union[
            "wp.array",
            tuple[
                "wp.array(dtype=ElementIndex)",
                "wp.array(dtype=Coords)",
            ],
            tuple[
                "wp.array2d(dtype=ElementIndex)",
                "wp.array2d(dtype=Coords)",
                "wp.array2d(dtype=float)",
            ],
        ],
        measures: Optional["wp.array(dtype=float)"] = None,
        requires_grad: bool = False,
        max_dist: float = 0.0,
        use_domain_element_indices: bool = False,
        temporary_store: TemporaryStore = None,
    ):
        super().__init__(domain)

        self._requires_grad = requires_grad
        self._use_domain_element_indices = use_domain_element_indices

        self._cell_particle_offsets: wp.array = None
        """Postfix sum of the number of particles in each cell"""
        self._cell_particle_indices: wp.array = None
        """Indices of the particles in each cell. Ordered according to cell_particle_offsets."""
        self._cell_particle_fraction: wp.array = None
        """Fraction of the cell occupied by each particle. Ordered according to cell_particle_offsets."""
        self._cell_particle_coords: wp.array = None
        """Coordinates of the particles. Ordered according to cell_particle_offsets."""

        self._unique_point_count: int = 0
        """Number of unique particles"""

        self._max_particles_per_cell: int = None
        """Maximum number of particles per cell. Computed on-demand"""

        self._bin_particles(positions, measures, max_dist=max_dist, temporary_store=temporary_store)

        if self._use_domain_element_indices:
            self.point_count = self._point_count_domain
            self.point_coords = self._point_coords_domain
            self.point_weight = self._point_weight_domain
            self.point_index = self._point_index_domain
            self.point_evaluation_index = self._point_evaluation_index_domain
        else:
            self.point_count = self._point_count_geo
            self.point_coords = self._point_coords_geo
            self.point_weight = self._point_weight_geo
            self.point_index = self._point_index_geo
            self.point_evaluation_index = self._point_evaluation_index_geo

    @property
    def name(self):
        """Unique name of the quadrature rule."""
        return self.__class__.__name__

    @Quadrature.domain.setter
    def domain(self, domain: GeometryDomain):
        """Set the quadrature domain, enforcing compatible geometry."""
        if self._use_domain_element_indices and domain != self.domain:
            raise RuntimeError("Cannot change the domain if use_domain_element_indices is True")

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
        """Structure containing arguments to be passed to device functions."""

        cell_particle_offsets: wp.array(dtype=int)
        cell_particle_indices: wp.array(dtype=int)
        cell_particle_fraction: wp.array(dtype=float)
        cell_particle_coords: wp.array(dtype=Coords)

    def fill_arg(self, args: Arg, device):
        """Fill the quadrature argument structure for device functions."""
        args.cell_particle_offsets = self._cell_particle_offsets.to(device)
        args.cell_particle_indices = self._cell_particle_indices.to(device)
        args.cell_particle_fraction = self._cell_particle_fraction.to(device)
        args.cell_particle_coords = self._cell_particle_coords.to(device)

    def total_point_count(self):
        """Number of unique quadrature points."""
        return self._unique_point_count

    def evaluation_point_count(self):
        """Number of locations at which the quadrature rule is evaluated.
        Quadrature points belonging to multiple cells are counted multiple times.
        """
        return self._cell_particle_indices.shape[0]

    def active_cell_count(self):
        """Number of cells containing at least one particle"""
        return self._active_cell_count.numpy()[0]

    def max_points_per_element(self):
        """Maximum number of quadrature points per element."""
        if self._max_particles_per_cell is None:
            max_ppc = wp.zeros(shape=(1,), dtype=int, device=self._cell_particle_offsets.device)
            wp.launch(
                PicQuadrature._max_particles_per_cell_kernel,
                self._cell_particle_offsets.shape[0] - 1,
                device=max_ppc.device,
                inputs=[self._cell_particle_offsets, max_ppc],
            )
            self._max_particles_per_cell = int(max_ppc.numpy()[0])

        return self._max_particles_per_cell

    @property
    def cell_particle_offsets(self):
        """Postfix sum of the number of particles in each cell"""
        return self._cell_particle_offsets

    @property
    def cell_particle_indices(self):
        """Indices of the particles in each cell"""
        return self._cell_particle_indices

    @wp.func
    def _point_count_geo(elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex):
        return qp_arg.cell_particle_offsets[element_index + 1] - qp_arg.cell_particle_offsets[element_index]

    @wp.func
    def _point_coords_geo(
        elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex, index: int
    ):
        particle_offset = qp_arg.cell_particle_offsets[element_index] + index
        return qp_arg.cell_particle_coords[particle_offset]

    @wp.func
    def _point_weight_geo(
        elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex, index: int
    ):
        particle_offset = qp_arg.cell_particle_offsets[element_index] + index
        return qp_arg.cell_particle_fraction[particle_offset]

    @wp.func
    def _point_index_geo(
        elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex, index: int
    ):
        particle_offset = qp_arg.cell_particle_offsets[element_index] + index
        particle_index = qp_arg.cell_particle_indices[particle_offset]
        return particle_index

    @wp.func
    def _point_evaluation_index_geo(
        elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex, index: int
    ):
        """Return the evaluation index for a given element point."""
        return qp_arg.cell_particle_offsets[element_index] + index

    @wp.func
    def _point_count_domain(elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex):
        return (
            qp_arg.cell_particle_offsets[domain_element_index + 1] - qp_arg.cell_particle_offsets[domain_element_index]
        )

    @wp.func
    def _point_coords_domain(
        elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex, index: int
    ):
        particle_offset = qp_arg.cell_particle_offsets[domain_element_index] + index
        return qp_arg.cell_particle_coords[particle_offset]

    @wp.func
    def _point_weight_domain(
        elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex, index: int
    ):
        particle_offset = qp_arg.cell_particle_offsets[domain_element_index] + index
        return qp_arg.cell_particle_fraction[particle_offset]

    @wp.func
    def _point_index_domain(
        elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex, index: int
    ):
        particle_offset = qp_arg.cell_particle_offsets[domain_element_index] + index
        particle_index = qp_arg.cell_particle_indices[particle_offset]
        return particle_index

    @wp.func
    def _point_evaluation_index_domain(
        elt_arg: Any, qp_arg: Arg, domain_element_index: ElementIndex, element_index: ElementIndex, index: int
    ):
        return qp_arg.cell_particle_offsets[domain_element_index] + index

    def fill_element_mask(self, mask: "wp.array(dtype=int)"):
        """Fills a mask array such that all non-empty elements are set to 1, all empty elements to zero.

        Args:
            mask: Int warp array with size at least equal to `self.domain.geometry_element_count()`
        """

        wp.launch(
            kernel=PicQuadrature._fill_mask_kernel,
            dim=self.domain.geometry_element_count(),
            device=mask.device,
            inputs=[self._cell_particle_offsets, mask],
        )

    @wp.kernel
    def _fill_mask_kernel(
        element_particle_offsets: wp.array(dtype=int),
        element_mask: wp.array(dtype=int),
    ):
        i = wp.tid()
        element_mask[i] = wp.where(element_particle_offsets[i] == element_particle_offsets[i + 1], 0, 1)

    def _bin_particles(self, positions, measures, max_dist: float, temporary_store: TemporaryStore):
        if is_array(positions):
            self.cell_indices, self.particle_coords = self._compute_cell_indices_from_positions(
                positions, max_dist, temporary_store
            )
            self.particle_fraction = None
        elif len(positions) == 2:
            self.cell_indices, self.particle_coords = positions
            self.particle_fraction = None
            if self.cell_indices.shape != self.particle_coords.shape:
                raise ValueError("Cell index and coordinates arrays must have the same shape")
        elif len(positions) == 3:
            self.cell_indices, self.particle_coords, self.particle_fraction = positions
            if (
                self.cell_indices.shape != self.particle_coords.shape
                or self.cell_indices.shape != self.particle_fraction.shape
            ):
                raise ValueError("Cell index, coordinates and fraction arrays must have the same shape")

        self._unique_point_count = self.cell_indices.shape[0]
        cell_count = (
            self.domain.element_count() if self._use_domain_element_indices else self.domain.geometry_element_count()
        )

        self._cell_particle_offsets, self._cell_particle_indices, self._active_cell_count, _ = compress_node_indices(
            cell_count,
            self.cell_indices.flatten(),
            return_unique_nodes=True,
            temporary_store=temporary_store,
        )
        # Keep only the top part, the array was over-allocated for radix-sort
        self._cell_particle_indices = self._cell_particle_indices[: self.cell_indices.size]

        self._finalize_cell_particle_data(measures, temporary_store)

    def _compute_cell_indices_from_positions(self, positions, max_dist: float, temporary_store: TemporaryStore):
        device = positions.device
        if not self.domain.supports_lookup(device):
            raise RuntimeError(
                f"The PicQuadrature's underlying domain of type '{self.domain.geometry.name}.{self.domain.element_kind.name}' does not support global element lookups on this device. "
                "If relevant, check that the geometry's BVH has been built for this device (see `Geometry.build_bvh()`, `Geometry.update_bvh()`)."
            )

        cell_lookup = self.domain.element_partition_lookup
        cell_coordinates = self.domain.element_coordinates

        @dynamic_kernel(suffix=f"{self.domain.name}{self._use_domain_element_indices}")
        def bin_particles(
            cell_arg_value: self.domain.ElementArg,
            domain_index_arg_value: self.domain.ElementIndexArg,
            positions: wp.array(dtype=positions.dtype),
            max_dist: float,
            cell_index: wp.array(dtype=ElementIndex),
            cell_coords: wp.array(dtype=Coords),
        ):
            p = wp.tid()
            sample = cell_lookup(self.domain.DomainArg(cell_arg_value, domain_index_arg_value), positions[p], max_dist)

            if wp.static(self._use_domain_element_indices):
                cell_index[p] = self.domain.element_partition_index(domain_index_arg_value, sample.element_index)
            else:
                cell_index[p] = sample.element_index

            if sample.element_index == NULL_ELEMENT_INDEX:
                cell_coords[p] = Coords(OUTSIDE)
            else:
                cell_coords[p] = cell_coordinates(cell_arg_value, sample.element_index, positions[p])

        cell_indices = borrow_temporary(temporary_store, shape=positions.shape, dtype=int, device=device)
        particle_coords = borrow_temporary(
            temporary_store, shape=positions.shape, dtype=Coords, device=device, requires_grad=self._requires_grad
        )

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
                cell_indices,
                particle_coords,
            ],
            device=device,
        )

        return cell_indices, particle_coords

    def _finalize_cell_particle_data(self, measures: wp.array, temporary_store: TemporaryStore):
        device = self._cell_particle_offsets.device

        @dynamic_kernel(suffix=f"{self.domain.name}{self._use_domain_element_indices}")
        def finalize_cell_particle_data(
            cell_arg_value: self.domain.ElementArg,
            domain_index_arg_value: self.domain.ElementIndexArg,
            elements_per_particle: int,
            particle_measures: wp.array(dtype=float),
            particle_coords: wp.array(dtype=Coords),
            particle_fraction: wp.array(dtype=float),
            particle_cell_indices: wp.array(dtype=int),
            cell_particle_offsets: wp.array(dtype=int),
            cell_particle_index: wp.array(dtype=ElementIndex),
            cell_particle_coords: wp.array(dtype=Coords),
            cell_particle_fraction: wp.array(dtype=float),
        ):
            cp = wp.tid()

            flat_index = cell_particle_index[cp]

            tot_evaluation_point_count = cell_particle_offsets[cell_particle_offsets.shape[0] - 1]
            if flat_index == NULL_NODE_INDEX or flat_index == NULL_ELEMENT_INDEX or cp >= tot_evaluation_point_count:
                cell_particle_index[cp] = NULL_QP_INDEX
                cell_particle_fraction[cp] = 0.0
                cell_particle_coords[cp] = Coords(OUTSIDE)
                return

            particle_index = flat_index // elements_per_particle

            coords = particle_coords[flat_index]
            cell_index = particle_cell_indices[flat_index]

            if particle_measures:
                if wp.static(self._use_domain_element_indices):
                    element_index = self.domain.element_index(domain_index_arg_value, cell_index)
                else:
                    element_index = cell_index
                sample = make_free_sample(element_index, coords)
                fraction = particle_measures[particle_index] / self.domain.element_measure(cell_arg_value, sample)
            else:
                fraction = 1.0 / float(cell_particle_offsets[cell_index + 1] - cell_particle_offsets[cell_index])

            if particle_fraction:
                fraction *= particle_fraction[flat_index]

            cell_particle_fraction[cp] = fraction
            cell_particle_coords[cp] = coords
            cell_particle_index[cp] = particle_index

        if measures is not None:
            if measures.shape != (self.cell_indices.shape[0],):
                raise ValueError("Measures should be an 1d array or length equal to particle count")

        if self.particle_fraction is not None:
            particle_fraction = self.particle_fraction.flatten()
        else:
            particle_fraction = None

        self._cell_particle_coords = borrow_temporary(
            temporary_store,
            shape=self._cell_particle_indices.shape,
            dtype=Coords,
            device=device,
            requires_grad=self._requires_grad,
        )
        self._cell_particle_fraction = borrow_temporary(
            temporary_store,
            shape=self._cell_particle_indices.shape,
            dtype=float,
            device=device,
            requires_grad=self._requires_grad,
        )

        wp.launch(
            dim=self._cell_particle_indices.shape,
            kernel=finalize_cell_particle_data,
            inputs=[
                self.domain.element_arg_value(device),
                self.domain.element_index_arg_value(device),
                1 if self.cell_indices.ndim == 1 else self.cell_indices.shape[1],
                measures,
                self.particle_coords.flatten(),
                particle_fraction,
                self.cell_indices.flatten(),
                self._cell_particle_offsets,
                self._cell_particle_indices,
                self._cell_particle_coords,
                self._cell_particle_fraction,
            ],
        )

    @wp.kernel
    def _max_particles_per_cell_kernel(offsets: wp.array(dtype=int), max_count: wp.array(dtype=int)):
        cell = wp.tid()
        particle_count = offsets[cell + 1] - offsets[cell]
        wp.atomic_max(max_count, 0, particle_count)
