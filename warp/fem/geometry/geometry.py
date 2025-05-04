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

from functools import cached_property
from typing import Any

import warp as wp
from warp.fem import cache
from warp.fem.types import NULL_ELEMENT_INDEX, OUTSIDE, Coords, ElementIndex, ElementKind, Sample, make_free_sample

from .element import Element

_mat32 = wp.mat(shape=(3, 2), dtype=float)

_NULL_BVH_ID = wp.uint64(0)
_COORD_LOOKUP_ITERATIONS = 24
_COORD_LOOKUP_STEP = 1.0
_COORD_LOOKUP_EPS = float(2**-20)
_BVH_MIN_PADDING = float(2**-16)
_BVH_MAX_PADDING = float(2**16)


class Geometry:
    """
    Interface class for discrete geometries

    A geometry is composed of cells and sides. Sides may be boundary or interior (between cells).
    """

    dimension: int = 0

    _bvhs = None

    def cell_count(self):
        """Number of cells in the geometry"""
        raise NotImplementedError

    def side_count(self):
        """Number of sides in the geometry"""
        raise NotImplementedError

    def boundary_side_count(self):
        """Number of boundary sides (sides with a single neighbour cell) in the geometry"""
        raise NotImplementedError

    def reference_cell(self) -> Element:
        """Prototypical element for a cell"""
        raise NotImplementedError

    def reference_side(self) -> Element:
        """Prototypical element for a side"""
        raise NotImplementedError

    @property
    def cell_dimension(self) -> int:
        """Manifold dimension of the geometry cells"""
        return self.reference_cell().dimension

    @property
    def base(self) -> "Geometry":
        """Returns the base geometry from which this geometry derives its topology. Usually `self`"""
        return self

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:
        return self.name

    CellArg: wp.codegen.Struct
    """Structure containing arguments to be passed to device functions evaluating cell-related quantities"""

    SideArg: wp.codegen.Struct
    """Structure containing arguments to be passed to device functions evaluating side-related quantities"""

    SideIndexArg: wp.codegen.Struct
    """Structure containing arguments to be passed to device functions for indexing sides"""

    def cell_arg_value(self, device) -> "Geometry.CellArg":
        """Value of the arguments to be passed to cell-related device functions"""
        raise NotImplementedError

    @staticmethod
    def cell_position(args: "Geometry.CellArg", s: "Sample"):
        """Device function returning the world position of a cell sample point"""
        raise NotImplementedError

    @staticmethod
    def cell_deformation_gradient(args: "Geometry.CellArg", s: "Sample"):
        """Device function returning the transpose of the gradient of world position with respect to reference cell"""
        raise NotImplementedError

    @staticmethod
    def cell_inverse_deformation_gradient(args: "Geometry.CellArg", s: "Sample"):
        """Device function returning the matrix right-transforming a gradient w.r.t. cell space to a gradient w.r.t. world space
        (i.e. the inverse deformation gradient)
        """
        raise NotImplementedError

    @staticmethod
    def cell_measure(args: "Geometry.CellArg", s: "Sample"):
        """Device function returning the measure determinant (e.g. volume, area) at a given point"""
        raise NotImplementedError

    @wp.func
    def cell_measure_ratio(args: Any, s: Sample):
        return 1.0

    @staticmethod
    def cell_normal(args: "Geometry.CellArg", s: "Sample"):
        """Device function returning the element normal at a sample point.

        For elements with the same dimension as the embedding space, this will be zero."""
        raise NotImplementedError

    def side_arg_value(self, device) -> "Geometry.SideArg":
        """Value of the arguments to be passed to side-related device functions"""
        raise NotImplementedError

    @staticmethod
    def boundary_side_index(args: "Geometry.SideIndexArg", boundary_side_index: int):
        """Device function returning the side index corresponding to a boundary side"""
        raise NotImplementedError

    @staticmethod
    def side_position(args: "Geometry.SideArg", s: "Sample"):
        """Device function returning the side position at a sample point"""
        raise NotImplementedError

    @staticmethod
    def side_deformation_gradient(args: "Geometry.SideArg", s: "Sample"):
        """Device function returning the gradient of world position with respect to reference side"""
        raise NotImplementedError

    @staticmethod
    def side_inner_inverse_deformation_gradient(args: "Geometry.Siderg", side_index: ElementIndex, coords: Coords):
        """Device function returning the matrix right-transforming a gradient w.r.t. inner cell space to a gradient w.r.t. world space
        (i.e. the inverse deformation gradient)
        """
        raise NotImplementedError

    @staticmethod
    def side_outer_inverse_deformation_gradient(args: "Geometry.CellArg", side_index: ElementIndex, coords: Coords):
        """Device function returning the matrix right-transforming a gradient w.r.t. outer cell space to a gradient w.r.t. world space
        (i.e. the inverse deformation gradient)
        """
        raise NotImplementedError

    @staticmethod
    def side_measure(args: "Geometry.SideArg", s: "Sample"):
        """Device function returning the measure determinant (e.g. volume, area) at a given point"""
        raise NotImplementedError

    @staticmethod
    def side_measure_ratio(args: "Geometry.SideArg", s: "Sample"):
        """Device function returning the ratio of the measure of a side to that of its neighbour cells"""
        raise NotImplementedError

    @staticmethod
    def side_normal(args: "Geometry.SideArg", s: "Sample"):
        """Device function returning the element normal at a sample point"""
        raise NotImplementedError

    @staticmethod
    def side_inner_cell_index(args: "Geometry.SideArg", side_index: ElementIndex):
        """Device function returning the inner cell index for a given side"""
        raise NotImplementedError

    @staticmethod
    def side_outer_cell_index(args: "Geometry.SideArg", side_index: ElementIndex):
        """Device function returning the outer cell index for a given side"""
        raise NotImplementedError

    @staticmethod
    def side_inner_cell_coords(args: "Geometry.SideArg", side_index: ElementIndex, side_coords: Coords):
        """Device function returning the coordinates of a point on a side in the inner cell"""
        raise NotImplementedError

    @staticmethod
    def side_outer_cell_coords(args: "Geometry.SideArg", side_index: ElementIndex, side_coords: Coords):
        """Device function returning the coordinates of a point on a side in the outer cell"""
        raise NotImplementedError

    @staticmethod
    def side_from_cell_coords(
        args: "Geometry.SideArg",
        side_index: ElementIndex,
        element_index: ElementIndex,
        element_coords: Coords,
    ):
        """Device function converting coordinates on a cell to coordinates on a side, or ``OUTSIDE``"""
        raise NotImplementedError

    @staticmethod
    def side_to_cell_arg(side_arg: "Geometry.SideArg"):
        """Device function converting a side-related argument value to a cell-related argument value, for promoting trace samples to the full space"""
        raise NotImplementedError

    # Default implementations for dependent quantities
    # Can be overridden in derived classes if more efficient implementations exist

    def _make_default_dependent_implementations(self):
        self.cell_inverse_deformation_gradient = self._make_cell_inverse_deformation_gradient()
        self.cell_measure = self._make_cell_measure()
        self.cell_normal = self._make_cell_normal()

        self.side_inverse_deformation_gradient = self._make_side_inverse_deformation_gradient()
        self.side_inner_inverse_deformation_gradient = self._make_side_inner_inverse_deformation_gradient()
        self.side_outer_inverse_deformation_gradient = self._make_side_outer_inverse_deformation_gradient()
        self.side_measure = self._make_side_measure()
        self.side_measure_ratio = self._make_side_measure_ratio()
        self.side_normal = self._make_side_normal()

        self.compute_cell_bounds = self._make_compute_cell_bounds()

    @wp.func
    def _element_measure(F: wp.vec2):
        return wp.length(F)

    @wp.func
    def _element_measure(F: wp.vec3):
        return wp.length(F)

    @wp.func
    def _element_measure(F: _mat32):
        Ft = wp.transpose(F)
        Fcross = wp.cross(Ft[0], Ft[1])
        return wp.length(Fcross)

    @wp.func
    def _element_measure(F: wp.mat33):
        return wp.abs(wp.determinant(F))

    @wp.func
    def _element_measure(F: wp.mat22):
        return wp.abs(wp.determinant(F))

    @wp.func
    def _element_normal(F: wp.vec2):
        return wp.normalize(wp.vec2(F[1], -F[0]))

    @wp.func
    def _element_normal(F: _mat32):
        Ft = wp.transpose(F)
        Fcross = wp.cross(Ft[0], Ft[1])
        return wp.normalize(Fcross)

    def _make_cell_measure(self):
        REF_MEASURE = wp.constant(self.reference_cell().measure())

        @cache.dynamic_func(suffix=self.name)
        def cell_measure(args: self.CellArg, s: Sample):
            F = self.cell_deformation_gradient(args, s)
            return Geometry._element_measure(F) * REF_MEASURE

        return cell_measure

    def _make_cell_normal(self):
        cell_dim = self.reference_cell().dimension
        geo_dim = self.dimension
        normal_vec = wp.vec(length=geo_dim, dtype=float)

        @cache.dynamic_func(suffix=self.name)
        def zero_normal(args: self.CellArg, s: Sample):
            return normal_vec(0.0)

        @cache.dynamic_func(suffix=self.name)
        def cell_hyperplane_normal(args: self.CellArg, s: Sample):
            F = self.cell_deformation_gradient(args, s)
            return Geometry._element_normal(F)

        if cell_dim == geo_dim:
            return zero_normal
        if cell_dim == geo_dim - 1:
            return cell_hyperplane_normal

        return None

    def _make_cell_inverse_deformation_gradient(self):
        cell_dim = self.reference_cell().dimension
        geo_dim = self.dimension

        @cache.dynamic_func(suffix=self.name)
        def cell_inverse_deformation_gradient(cell_arg: self.CellArg, s: Sample):
            return wp.inverse(self.cell_deformation_gradient(cell_arg, s))

        @cache.dynamic_func(suffix=self.name)
        def cell_pseudoinverse_deformation_gradient(cell_arg: self.CellArg, s: Sample):
            F = self.cell_deformation_gradient(cell_arg, s)
            Ft = wp.transpose(F)
            return wp.inverse(Ft * F) * Ft

        return cell_inverse_deformation_gradient if cell_dim == geo_dim else cell_pseudoinverse_deformation_gradient

    def _make_side_inverse_deformation_gradient(self):
        side_dim = self.reference_side().dimension
        geo_dim = self.dimension

        if side_dim == geo_dim:

            @cache.dynamic_func(suffix=self.name)
            def side_inverse_deformation_gradient(side_arg: self.SideArg, s: Sample):
                return wp.inverse(self.side_deformation_gradient(side_arg, s))

            return side_inverse_deformation_gradient

        if side_dim == 1:

            @cache.dynamic_func(suffix=self.name)
            def edge_pseudoinverse_deformation_gradient(side_arg: self.SideArg, s: Sample):
                F = self.side_deformation_gradient(side_arg, s)
                return wp.matrix_from_rows(F / wp.dot(F, F))

            return edge_pseudoinverse_deformation_gradient

        @cache.dynamic_func(suffix=self.name)
        def side_pseudoinverse_deformation_gradient(side_arg: self.SideArg, s: Sample):
            F = self.side_deformation_gradient(side_arg, s)
            Ft = wp.transpose(F)
            return wp.inverse(Ft * F) * Ft

        return side_pseudoinverse_deformation_gradient

    def _make_side_measure(self):
        REF_MEASURE = wp.constant(self.reference_side().measure())

        @cache.dynamic_func(suffix=self.name)
        def side_measure(args: self.SideArg, s: Sample):
            F = self.side_deformation_gradient(args, s)
            return Geometry._element_measure(F) * REF_MEASURE

        return side_measure

    def _make_side_measure_ratio(self):
        @cache.dynamic_func(suffix=self.name)
        def side_measure_ratio(args: self.SideArg, s: Sample):
            inner = self.side_inner_cell_index(args, s.element_index)
            outer = self.side_outer_cell_index(args, s.element_index)
            inner_coords = self.side_inner_cell_coords(args, s.element_index, s.element_coords)
            outer_coords = self.side_outer_cell_coords(args, s.element_index, s.element_coords)
            cell_arg = self.side_to_cell_arg(args)
            return self.side_measure(args, s) / wp.min(
                self.cell_measure(cell_arg, make_free_sample(inner, inner_coords)),
                self.cell_measure(cell_arg, make_free_sample(outer, outer_coords)),
            )

        return side_measure_ratio

    def _make_side_normal(self):
        side_dim = self.reference_side().dimension
        geo_dim = self.dimension

        @cache.dynamic_func(suffix=self.name)
        def hyperplane_normal(args: self.SideArg, s: Sample):
            F = self.side_deformation_gradient(args, s)
            return Geometry._element_normal(F)

        if side_dim == geo_dim - 1:
            return hyperplane_normal

        return None

    def _make_side_inner_inverse_deformation_gradient(self):
        @cache.dynamic_func(suffix=self.name)
        def side_inner_inverse_deformation_gradient(args: self.SideArg, s: Sample):
            cell_index = self.side_inner_cell_index(args, s.element_index)
            cell_coords = self.side_inner_cell_coords(args, s.element_index, s.element_coords)
            cell_arg = self.side_to_cell_arg(args)
            return self.cell_inverse_deformation_gradient(cell_arg, make_free_sample(cell_index, cell_coords))

        return side_inner_inverse_deformation_gradient

    def _make_side_outer_inverse_deformation_gradient(self):
        @cache.dynamic_func(suffix=self.name)
        def side_outer_inverse_deformation_gradient(args: self.SideArg, s: Sample):
            cell_index = self.side_outer_cell_index(args, s.element_index)
            cell_coords = self.side_outer_cell_coords(args, s.element_index, s.element_coords)
            cell_arg = self.side_to_cell_arg(args)
            return self.cell_inverse_deformation_gradient(cell_arg, make_free_sample(cell_index, cell_coords))

        return side_outer_inverse_deformation_gradient

    def _make_element_coordinates(self, element_kind: ElementKind, assume_linear: bool = False):
        pos_type = cache.cached_vec_type(self.dimension, dtype=float)

        if element_kind == ElementKind.CELL:
            ref_elt = self.reference_cell()
            arg_type = self.CellArg
            elt_pos = self.cell_position
            elt_inv_grad = self.cell_inverse_deformation_gradient
        else:
            ref_elt = self.reference_side()
            arg_type = self.SideArg
            elt_pos = self.side_position
            elt_inv_grad = self.side_inverse_deformation_gradient

        elt_center = Coords(ref_elt.center())

        ITERATIONS = 1 if assume_linear else _COORD_LOOKUP_ITERATIONS
        STEP = 1.0 if assume_linear else _COORD_LOOKUP_STEP

        @cache.dynamic_func(suffix=f"{self.name}{element_kind}{assume_linear}")
        def element_coordinates(args: arg_type, element_index: ElementIndex, pos: pos_type):
            coords = elt_center

            # Newton loop (single iteration in linear case)
            for _k in range(ITERATIONS):
                s = make_free_sample(element_index, coords)
                x = elt_pos(args, s)
                dc = elt_inv_grad(args, s) * (pos - x)
                if wp.static(not assume_linear):
                    if wp.length_sq(dc) < _COORD_LOOKUP_EPS:
                        break
                coords = coords + ref_elt.coord_delta(STEP * dc)

            return coords

        return element_coordinates

    def _make_cell_coordinates(self, assume_linear: bool = False):
        return self._make_element_coordinates(element_kind=ElementKind.CELL, assume_linear=assume_linear)

    def _make_side_coordinates(self, assume_linear: bool = False):
        return self._make_element_coordinates(element_kind=ElementKind.SIDE, assume_linear=assume_linear)

    def _make_element_closest_point(self, element_kind: ElementKind, assume_linear: bool = False):
        pos_type = cache.cached_vec_type(self.dimension, dtype=float)

        element_coordinates = self._make_element_coordinates(element_kind=element_kind, assume_linear=assume_linear)

        if element_kind == ElementKind.CELL:
            ref_elt = self.reference_cell()
            arg_type = self.CellArg
            elt_pos = self.cell_position
            elt_def_grad = self.cell_deformation_gradient
        else:
            ref_elt = self.reference_side()
            arg_type = self.SideArg
            elt_pos = self.side_position
            elt_def_grad = self.side_deformation_gradient

        @cache.dynamic_func(suffix=f"{self.name}{element_kind}{assume_linear}")
        def cell_closest_point(args: arg_type, cell_index: ElementIndex, pos: pos_type):
            # First get unconstrained coordinates, may use newton for this
            coords = element_coordinates(args, cell_index, pos)

            # Now do projected gradient
            # For interior points should exit at first iteration
            for _k in range(_COORD_LOOKUP_ITERATIONS):
                cur_coords = coords
                s = make_free_sample(cell_index, cur_coords)
                x = elt_pos(args, s)

                F = elt_def_grad(args, s)
                F_scale = wp.ddot(F, F)

                dc = (pos - x) @ F  # gradient step
                coords = ref_elt.project(cur_coords + ref_elt.coord_delta(dc / F_scale))

                if wp.length_sq(coords - cur_coords) < _COORD_LOOKUP_EPS:
                    break

            return cur_coords, wp.length_sq(pos - x)

        return cell_closest_point

    def _make_cell_closest_point(self, assume_linear: bool = False):
        return self._make_element_closest_point(element_kind=ElementKind.CELL, assume_linear=assume_linear)

    def _make_side_closest_point(self, assume_linear: bool = False):
        return self._make_element_closest_point(element_kind=ElementKind.SIDE, assume_linear=assume_linear)

    def make_filtered_cell_lookup(self, filter_func: wp.Function = None):
        suffix = f"{self.name}{filter_func.func.__qualname__ if filter_func is not None else ''}"
        pos_type = cache.cached_vec_type(self.dimension, dtype=float)

        @cache.dynamic_func(suffix=suffix)
        def cell_lookup(args: self.CellArg, pos: pos_type, max_dist: float, filter_data: Any, filter_target: Any):
            closest_cell = int(NULL_ELEMENT_INDEX)
            closest_coords = Coords(OUTSIDE)

            bvh_id = self.cell_bvh_id(args)
            if bvh_id != _NULL_BVH_ID:
                pad = wp.max(max_dist, 1.0) * _BVH_MIN_PADDING

                # query with increasing bbox size until we find an element
                # or reach the max distance bound
                while closest_cell == NULL_ELEMENT_INDEX:
                    query = wp.bvh_query_aabb(bvh_id, _bvh_vec(pos) - wp.vec3(pad), _bvh_vec(pos) + wp.vec3(pad))
                    cell_index = int(0)
                    closest_dist = float(pad * pad)

                    while wp.bvh_query_next(query, cell_index):
                        if wp.static(filter_func is not None):
                            if filter_func(filter_data, cell_index) != filter_target:
                                continue

                        coords, dist = self.cell_closest_point(args, cell_index, pos)
                        if dist <= closest_dist:
                            closest_dist = dist
                            closest_cell = cell_index
                            closest_coords = coords

                    if pad >= _BVH_MAX_PADDING:
                        break
                    pad = wp.min(4.0 * pad, _BVH_MAX_PADDING)

            return make_free_sample(closest_cell, closest_coords)

        return cell_lookup

    @cached_property
    def cell_lookup(self) -> wp.Function:
        unfiltered_cell_lookup = self.make_filtered_cell_lookup(filter_func=None)

        # overloads
        null_filter_data = 0
        null_filter_target = 0

        pos_type = cache.cached_vec_type(self.dimension, dtype=float)

        @cache.dynamic_func(suffix=self.name)
        def cell_lookup(args: self.CellArg, pos: pos_type, max_dist: float):
            return unfiltered_cell_lookup(args, pos, max_dist, null_filter_data, null_filter_target)

        @cache.dynamic_func(suffix=self.name)
        def cell_lookup(args: self.CellArg, pos: pos_type, guess: Sample):
            guess_pos = self.cell_position(args, guess)
            max_dist = wp.length(guess_pos - pos)
            return unfiltered_cell_lookup(args, pos, max_dist, null_filter_data, null_filter_target)

        @cache.dynamic_func(suffix=self.name)
        def cell_lookup(args: self.CellArg, pos: pos_type):
            max_dist = 0.0
            return unfiltered_cell_lookup(args, pos, max_dist, null_filter_data, null_filter_target)

        # array filtering variants
        filtered_cell_lookup = self.make_filtered_cell_lookup(filter_func=_array_load)
        pos_type = cache.cached_vec_type(self.dimension, dtype=float)

        @cache.dynamic_func(suffix=self.name)
        def cell_lookup(
            args: self.CellArg, pos: pos_type, max_dist: float, filter_array: wp.array(dtype=Any), filter_target: Any
        ):
            return filtered_cell_lookup(args, pos, max_dist, filter_array, filter_target)

        @cache.dynamic_func(suffix=self.name)
        def cell_lookup(args: self.CellArg, pos: pos_type, filter_array: wp.array(dtype=Any), filter_target: Any):
            max_dist = 0.0
            return filtered_cell_lookup(args, pos, max_dist, filter_array, filter_target)

        return cell_lookup

    def _make_compute_cell_bounds(self):
        @cache.dynamic_kernel(suffix=self.name)
        def compute_cell_bounds(
            args: self.CellArg,
            lowers: wp.array(dtype=wp.vec3),
            uppers: wp.array(dtype=wp.vec3),
        ):
            i = wp.tid()
            lo, up = self.cell_bounds(args, i)
            lowers[i] = _bvh_vec(lo)
            uppers[i] = _bvh_vec(up)

        return compute_cell_bounds

    def supports_cell_lookup(self, device) -> bool:
        return self.bvh_id(device) != _NULL_BVH_ID

    def update_bvh(self, device=None):
        """
        Refits the BVH, or rebuilds it from scratch if `force_rebuild` is ``True``.
        """

        if self._bvhs is None:
            return self.build_bvh(device)

        device = wp.get_device(device)
        bvh = self._bvhs.get(device.ordinal)
        if bvh is None:
            return self.build_bvh(device)

        wp.launch(
            self.compute_cell_bounds,
            dim=self.cell_count(),
            device=device,
            inputs=[self.cell_arg_value(device=device)],
            outputs=[
                bvh.lowers,
                bvh.uppers,
            ],
        )

        bvh.refit()

    def build_bvh(self, device=None):
        device = wp.get_device(device)

        lowers = wp.array(shape=self.cell_count(), dtype=wp.vec3, device=device)
        uppers = wp.array(shape=self.cell_count(), dtype=wp.vec3, device=device)

        wp.launch(
            self.compute_cell_bounds,
            dim=self.cell_count(),
            device=device,
            inputs=[self.cell_arg_value(device=device)],
            outputs=[
                lowers,
                uppers,
            ],
        )

        if self._bvhs is None:
            self._bvhs = {}
        self._bvhs[device.ordinal] = wp.Bvh(lowers, uppers)

    def bvh_id(self, device):
        if self._bvhs is None:
            return _NULL_BVH_ID

        bvh = self._bvhs.get(wp.get_device(device).ordinal)
        if bvh is None:
            return _NULL_BVH_ID
        return bvh.id


@wp.func
def _bvh_vec(v: wp.vec3):
    return v


@wp.func
def _bvh_vec(v: wp.vec2):
    return wp.vec3(v[0], v[1], 0.0)


@wp.func
def _array_load(arr: wp.array(dtype=Any), idx: int):
    return arr[idx]
