from typing import Any

import warp as wp
from warp.fem.types import Coords, ElementIndex, Sample

from .element import Element


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

    @staticmethod
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
    def cell_inverse_deformation_gradient(args: "Geometry.CellArg", cell_index: ElementIndex, coords: Coords):
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

    @staticmethod
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
    def side_deformation_gradient(args: "Geometry.CellArg", s: "Sample"):
        """Device function returning the gradient of world position with respect to reference cell"""
        raise NotImplementedError

    @staticmethod
    def side_inner_inverse_deformation_gradient(args: "Geometry.CellArg", side_index: ElementIndex, coords: Coords):
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
