from typing import Any, Optional, Tuple, Union

import warp as wp
from warp.fem.cache import TemporaryStore, borrow_temporary, cached_arg_value, dynamic_kernel
from warp.fem.domain import GeometryDomain
from warp.fem.types import Coords, ElementIndex, make_free_sample
from warp.fem.utils import compress_node_indices

from .quadrature import Quadrature


class PicQuadrature(Quadrature):
    """Particle-based quadrature formula, using a global set of points unevenly spread out over geometry elements.

    Useful for Particle-In-Cell and derived methods.

    Args:
        domain: Underlying domain for the quadrature
        positions: Either an array containing the world positions of all particles, or a tuple of arrays containing
         the cell indices and coordinates for each particle. Note that the former requires the underlying geometry to
         define a global :meth:`Geometry.cell_lookup` method; currently this is only available for :class:`Grid2D` and :class:`Grid3D`.
        measures: Array containing the measure (area/volume) of each particle, used to defined the integration weights.
         If ``None``, defaults to the cell measure divided by the number of particles in the cell.
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
        temporary_store: TemporaryStore = None,
    ):
        super().__init__(domain)

        self._requires_grad = requires_grad
        self._bin_particles(positions, measures, temporary_store)
        self._max_particles_per_cell: int = None

    @property
    def name(self):
        return f"{self.__class__.__name__}"

    @Quadrature.domain.setter
    def domain(self, domain: GeometryDomain):
        # Allow changing the quadrature domain as long as underlying geometry and element kind are the same
        if self.domain is not None and (
            domain.geometry != self.domain.geometry or domain.element_kind != self.domain.element_kind
        ):
            raise RuntimeError(
                "Cannot change the domain to that of a different Geometry and/or using different element kinds."
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
        arg.particle_coords = self._particle_coords.to(device)
        return arg

    def total_point_count(self):
        return self._particle_coords.shape[0]

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
        element_mask[i] = wp.select(element_particle_offsets[i] == element_particle_offsets[i + 1], 1, 0)

    @wp.kernel
    def _compute_uniform_fraction(
        cell_index: wp.array(dtype=ElementIndex),
        cell_particle_offsets: wp.array(dtype=int),
        cell_fraction: wp.array(dtype=float),
    ):
        p = wp.tid()

        cell = cell_index[p]
        cell_particle_count = cell_particle_offsets[cell + 1] - cell_particle_offsets[cell]

        cell_fraction[p] = 1.0 / float(cell_particle_count)

    def _bin_particles(self, positions, measures, temporary_store: TemporaryStore):
        if wp.types.is_array(positions):
            # Initialize from positions
            @dynamic_kernel(suffix=f"{self.domain.name}")
            def bin_particles(
                cell_arg_value: self.domain.ElementArg,
                positions: wp.array(dtype=positions.dtype),
                cell_index: wp.array(dtype=ElementIndex),
                cell_coords: wp.array(dtype=Coords),
            ):
                p = wp.tid()
                sample = self.domain.element_lookup(cell_arg_value, positions[p])

                cell_index[p] = sample.element_index
                cell_coords[p] = sample.element_coords

            device = positions.device

            cell_index_temp = borrow_temporary(temporary_store, shape=positions.shape, dtype=int, device=device)
            cell_index = cell_index_temp.array

            self._particle_coords_temp = borrow_temporary(
                temporary_store, shape=positions.shape, dtype=Coords, device=device, requires_grad=self._requires_grad
            )
            self._particle_coords = self._particle_coords_temp.array

            wp.launch(
                dim=positions.shape[0],
                kernel=bin_particles,
                inputs=[
                    self.domain.element_arg_value(device),
                    positions,
                    cell_index,
                    self._particle_coords,
                ],
                device=device,
            )

        else:
            cell_index, self._particle_coords = positions
            if cell_index.shape != self._particle_coords.shape:
                raise ValueError("Cell index and coordinates arrays must have the same shape")

            cell_index_temp = None
            self._particle_coords_temp = None

        self._cell_particle_offsets, self._cell_particle_indices, self._cell_count, _ = compress_node_indices(
            self.domain.geometry_element_count(), cell_index, return_unique_nodes=True, temporary_store=temporary_store
        )

        self._compute_fraction(cell_index, measures, temporary_store)

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
                sample = make_free_sample(cell_index[p], cell_coords[p])

                cell_fraction[p] = measures[p] / self.domain.element_measure(cell_arg_value, sample)

            wp.launch(
                dim=measures.shape[0],
                kernel=compute_fraction,
                inputs=[
                    self.domain.element_arg_value(device),
                    measures,
                    cell_index,
                    self._particle_coords,
                    self._particle_fraction,
                ],
                device=device,
            )

    @wp.kernel
    def _max_particles_per_cell_kernel(offsets: wp.array(dtype=int), max_count: wp.array(dtype=int)):
        cell = wp.tid()
        particle_count = offsets[cell + 1] - offsets[cell]
        wp.atomic_max(max_count, 0, particle_count)
