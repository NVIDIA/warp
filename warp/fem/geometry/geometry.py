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

from typing import Any

import warp as wp
from warp.fem import cache
from warp.fem.types import Coords, ElementIndex, Sample, make_free_sample

from .element import Element

_mat32 = wp.mat(shape=(3, 2), dtype=float)


class Geometry:
    """
    Interface class for discrete geometries

    A geometry is composed of cells and sides. Sides may be boundary or interior (between cells).
    """

    dimension: int = 0

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
    def cell_lookup(args: "Geometry.CellArg", pos: Any):
        """Device function returning the cell sample point corresponding to a world position"""
        raise NotImplementedError

    @staticmethod
    def cell_lookup(args: "Geometry.CellArg", pos: Any, guess: "Sample"):
        """Device function returning the cell sample point corresponding to a world position. Can use guess for faster lookup"""
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

        self.side_inner_inverse_deformation_gradient = self._make_side_inner_inverse_deformation_gradient()
        self.side_outer_inverse_deformation_gradient = self._make_side_outer_inverse_deformation_gradient()
        self.side_measure = self._make_side_measure()
        self.side_measure_ratio = self._make_side_measure_ratio()
        self.side_normal = self._make_side_normal()

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
        return wp.normalize(wp.vec2(-F[1], F[0]))

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
