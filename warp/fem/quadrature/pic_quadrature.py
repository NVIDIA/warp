import warp as wp

from warp.fem.domain import GeometryDomain
from warp.fem.types import ElementIndex, Coords
from warp.fem.utils import compress_node_indices
from warp.fem.cache import cached_arg_value, TemporaryStore, borrow_temporary

from .quadrature import Quadrature


wp.set_module_options({"enable_backward": False})


class PicQuadrature(Quadrature):
    """Particle-based quadrature formula, using a global set of points irregularely spread out over geometry elements.

    Useful for Particle-In-Cell and derived methods.
    """

    def __init__(
        self,
        domain: GeometryDomain,
        positions: "wp.array()",
        measures: "wp.array(dtype=float)",
        temporary_store: TemporaryStore = None,
    ):
        super().__init__(domain)

        self.positions = positions
        self.measures = measures

        self._bin_particles(temporary_store)

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
        arg.particle_fraction = self._particle_fraction.array.to(device)
        arg.particle_coords = self._particle_coords.array.to(device)
        return arg

    def total_point_count(self):
        return self.positions.shape[0]

    @wp.func
    def point_count(arg: Arg, element_index: ElementIndex):
        return arg.cell_particle_offsets[element_index + 1] - arg.cell_particle_offsets[element_index]

    @wp.func
    def point_coords(arg: Arg, element_index: ElementIndex, index: int):
        particle_index = arg.cell_particle_indices[arg.cell_particle_offsets[element_index] + index]
        return arg.particle_coords[particle_index]

    @wp.func
    def point_weight(arg: Arg, element_index: ElementIndex, index: int):
        particle_index = arg.cell_particle_indices[arg.cell_particle_offsets[element_index] + index]
        return arg.particle_fraction[particle_index]

    @wp.func
    def point_index(arg: Arg, element_index: ElementIndex, index: int):
        particle_index = arg.cell_particle_indices[arg.cell_particle_offsets[element_index] + index]
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

    def _bin_particles(self, temporary_store: TemporaryStore):
        from warp.fem import cache

        @cache.dynamic_kernel(suffix=f"{self.domain.name}")
        def bin_particles(
            cell_arg_value: self.domain.ElementArg,
            positions: wp.array(dtype=self.positions.dtype),
            measures: wp.array(dtype=float),
            cell_index: wp.array(dtype=ElementIndex),
            cell_coords: wp.array(dtype=Coords),
            cell_fraction: wp.array(dtype=float),
        ):
            p = wp.tid()
            sample = self.domain.element_lookup(cell_arg_value, positions[p])

            cell_index[p] = sample.element_index

            cell_coords[p] = sample.element_coords
            cell_fraction[p] = measures[p] / self.domain.element_measure(cell_arg_value, sample)

        device = self.positions.device

        cell_index = borrow_temporary(temporary_store, shape=self.positions.shape, dtype=int, device=device)
        self._particle_coords = borrow_temporary(
            temporary_store, shape=self.positions.shape, dtype=Coords, device=device
        )
        self._particle_fraction = borrow_temporary(
            temporary_store, shape=self.positions.shape, dtype=float, device=device
        )

        wp.launch(
            dim=self.positions.shape[0],
            kernel=bin_particles,
            inputs=[
                self.domain.element_arg_value(device),
                self.positions,
                self.measures,
                cell_index.array,
                self._particle_coords.array,
                self._particle_fraction.array,
            ],
            device=device,
        )

        self._cell_particle_offsets, self._cell_particle_indices, self._cell_count, _ = compress_node_indices(
            self.domain.geometry_element_count(), cell_index.array
        )

        cell_index.release()
