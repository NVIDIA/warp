import warp as wp

from warp.fem.domain import GeometryDomain
from warp.fem.types import ElementIndex, Coords
from warp.fem.utils import compress_node_indices

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
    ):
        super().__init__(domain)

        self.positions = positions
        self.measures = measures

        self._bin_particles()

    @property
    def name(self):
        return f"{self.__class__.__name__}"

    @wp.struct
    class Arg:
        cell_particle_offsets: wp.array(dtype=int)
        cell_particle_indices: wp.array(dtype=int)
        particle_fraction: wp.array(dtype=float)
        particle_coords: wp.array(dtype=Coords)

    def arg_value(self, device) -> Arg:
        arg = PicQuadrature.Arg()
        arg.cell_particle_offsets = self._cell_particle_offsets.to(device)
        arg.cell_particle_indices = self._cell_particle_indices.to(device)
        arg.particle_fraction = self._particle_fraction.to(device)
        arg.particle_coords = self._particle_coords.to(device)
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

    def _bin_particles(self):
        from warp.fem import cache

        def bin_particles_fn(
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

        bin_particles = cache.get_kernel(
            bin_particles_fn,
            suffix=f"{self.domain.name}",
        )

        device = self.positions.device

        cell_index = wp.empty(shape=self.positions.shape, dtype=int, device=device)
        self._particle_coords = wp.empty(shape=self.positions.shape, dtype=Coords, device=device)
        self._particle_fraction = wp.empty(shape=self.positions.shape, dtype=float, device=device)

        wp.launch(
            dim=self.positions.shape[0],
            kernel=bin_particles,
            inputs=[
                self.domain.element_arg_value(device),
                self.positions,
                self.measures,
                cell_index,
                self._particle_coords,
                self._particle_fraction,
            ],
            device=device,
        )

        self._cell_particle_offsets, self._cell_particle_indices, self._cell_count, _ = compress_node_indices(
            self.domain.geometry.cell_count(), cell_index
        )
