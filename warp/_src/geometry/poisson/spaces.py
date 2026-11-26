# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference per-level hats through Warp FEM's public BasisSpace/SpaceTopology interfaces.

An adaptive integration cell carries up to eight functions from one level. This
is a per-level representation; nested levels are never concatenated into an
artificial independent global system.
"""

import warp as wp
import warp.fem as fem
from warp.fem import BasisSpace, ShapeFunction, SpaceTopology, cache

from .field import basis_key, find_key


@wp.struct
class LevelTopologyArg:
    indices: wp.array2d[int]
    corners: wp.array2d[int]
    counts: wp.array[int]
    # Integer lattice offsets and powers of two are exact in float32 through
    # the supported depth 15. Basis arithmetic casts these to geometry precision.
    origins: wp.array[wp.vec3]
    scales: wp.array[float]


@wp.kernel
def _connect_kernel(
    ijk: wp.array[wp.vec3i],
    cell_level: wp.array[wp.uint8],
    keys: wp.array[wp.int64],
    depth: int,
    finest: int,
    arg: LevelTopologyArg,
):
    i = wp.tid()
    scale = float(1 << int(cell_level[i])) / float(1 << (finest - depth))
    start = wp.vec3(ijk[i]) / float(1 << (finest - depth))
    mid = start + wp.vec3(scale * 0.5)
    base = wp.vec3i(int(wp.floor(mid[0])), int(wp.floor(mid[1])), int(wp.floor(mid[2])))
    count = int(0)
    if scale <= 1.0:
        for c in range(8):
            off = base + wp.vec3i((c >> 2) & 1, (c >> 1) & 1, c & 1)
            if wp.min(off) >= 0 and wp.max(off) <= 1 << depth:
                idx = find_key(keys, basis_key(depth, off[0], off[1], off[2]))
                if idx >= 0:
                    arg.indices[i, count] = idx
                    arg.corners[i, count] = c
                    count += 1
    arg.counts[i] = count
    arg.origins[i] = start - wp.vec3(base)
    arg.scales[i] = scale


class LevelTopology(SpaceTopology):
    TopologyArg = LevelTopologyArg

    def __init__(self, geometry, keys, depth, finest):
        self._node_count = len(keys)
        self._depth = depth
        super().__init__(geometry, 8)
        device = keys.device
        count = geometry.cell_count()
        self.arg = LevelTopologyArg()
        self.arg.indices = wp.empty((count, 8), dtype=int, device=device)
        self.arg.corners = wp.empty((count, 8), dtype=int, device=device)
        self.arg.counts = wp.empty(count, dtype=int, device=device)
        self.arg.origins = wp.empty(count, dtype=wp.vec3, device=device)
        self.arg.scales = wp.empty(count, dtype=float, device=device)
        geo_arg = geometry.cell_arg_value(device)
        cell_level = (
            geo_arg.cell_level
            if isinstance(geometry, fem.AdaptiveNanogrid)
            else wp.zeros(count, dtype=wp.uint8, device=device)
        )
        wp.launch(
            _connect_kernel,
            count,
            inputs=[
                geo_arg.cell_ijk,
                cell_level,
                keys,
                depth,
                finest,
                self.arg,
            ],
            device=device,
        )
        self.element_node_count = self._count_func()
        self.element_node_index = self._index_func()

    @property
    def name(self):
        return f"PoissonLevelTopology{self._depth}_{self.geometry.name}_{self.geometry.scalar_type.__name__}"

    def node_count(self):
        return self._node_count

    def fill_topo_arg(self, arg, device):
        for name in ["indices", "corners", "counts", "origins", "scales"]:
            setattr(arg, name, getattr(self.arg, name).to(device))

    def _count_func(self):
        @cache.dynamic_func(suffix=self.name)
        def count(geo: self.ElementArg, arg: LevelTopologyArg, element: int):
            return arg.counts[element]

        return count

    def _index_func(self):
        @cache.dynamic_func(suffix=self.name)
        def index(geo: self.ElementArg, arg: LevelTopologyArg, element: int, local: int):
            return arg.indices[element, local]

        return index


class LevelBasis(BasisSpace):
    ORDER = 1

    @property
    def value(self):
        return ShapeFunction.Value.Scalar

    @property
    def name(self):
        return f"PoissonLevelBasis_{self.topology.name}"

    def make_node_coords_in_element(self):
        dtype = self.geometry.scalar_type
        vec = wp.types.vector(length=3, dtype=dtype)

        @cache.dynamic_func(suffix=self.name)
        def coords(
            geo: self.geometry.CellArg,
            topo: LevelTopologyArg,
            basis: self.BasisArg,
            element: int,
            local: int,
        ):
            c = topo.corners[element, local]
            corner = vec(dtype((c >> 2) & 1), dtype((c >> 1) & 1), dtype(c & 1))
            return (corner - vec(topo.origins[element])) / dtype(topo.scales[element])

        return coords

    def make_node_quadrature_weight(self):
        dtype = self.geometry.scalar_type

        @cache.dynamic_func(suffix=self.name)
        def weight(
            geo: self.geometry.CellArg,
            topo: LevelTopologyArg,
            basis: self.BasisArg,
            element: int,
            local: int,
        ):
            return dtype(0.125)

        return weight

    def make_element_inner_weight(self):
        dtype = self.geometry.scalar_type
        vec = wp.types.vector(length=3, dtype=dtype)

        @cache.dynamic_func(suffix=self.name)
        def weight(
            geo: self.geometry.CellArg,
            topo: LevelTopologyArg,
            basis: self.BasisArg,
            element: int,
            coords: vec,
            local: int,
            qp: int,
        ):
            c = topo.corners[element, local]
            t = vec(topo.origins[element]) + coords * dtype(topo.scales[element])
            h = vec(dtype(1.0)) - t
            if (c & 4) != 0:
                h[0] = t[0]
            if (c & 2) != 0:
                h[1] = t[1]
            if (c & 1) != 0:
                h[2] = t[2]
            return h[0] * h[1] * h[2]

        return weight

    def make_element_inner_weight_gradient(self):
        dtype = self.geometry.scalar_type
        vec = wp.types.vector(length=3, dtype=dtype)

        @cache.dynamic_func(suffix=self.name)
        def gradient(
            geo: self.geometry.CellArg,
            topo: LevelTopologyArg,
            basis: self.BasisArg,
            element: int,
            coords: vec,
            local: int,
            qp: int,
        ):
            c = topo.corners[element, local]
            t = vec(topo.origins[element]) + coords * dtype(topo.scales[element])
            h = vec(dtype(1.0)) - t
            sign = vec(dtype(-1.0))
            if (c & 4) != 0:
                h[0] = t[0]
                sign[0] = dtype(1.0)
            if (c & 2) != 0:
                h[1] = t[1]
                sign[1] = dtype(1.0)
            if (c & 1) != 0:
                h[2] = t[2]
                sign[2] = dtype(1.0)
            return dtype(topo.scales[element]) * vec(
                sign[0] * h[1] * h[2], h[0] * sign[1] * h[2], h[0] * h[1] * sign[2]
            )

        return gradient


def make_level_space(geometry, keys, depth, finest):
    return fem.make_collocated_function_space(
        LevelBasis(LevelTopology(geometry, keys, depth, finest)),
        dtype=geometry.scalar_type,
    )
