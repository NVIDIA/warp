from typing import Any

import warp as wp

from warp.fem.types import Sample, ElementIndex, Coords
from .element import Element


class Geometry:
    """
    Interface class for discrete geometries

    A geometry is composed of cells and sides. Sides may be boundary or interior (between cells).
    """

    dimension: int = 0

    def cell_count(self):
        raise NotImplementedError

    def side_count(self):
        raise NotImplementedError

    def boundary_side_count(self):
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
    """Structure containing arguments to be passed to device function evaluating cell-related quantities"""

    SideArg: wp.codegen.Struct
    """Structure containing arguments to be passed to device function evaluating side-related quantities"""

    def cell_arg_value(self, device) -> "Geometry.CellArg":
        """Value of the arguments to be passed to cell-related device functions"""
        raise NotImplementedError

    def cell_position(args: "Geometry.CellArg", s: "Sample"):
        """Device function returning the cell position at a sample point"""
        raise NotImplementedError

    def cell_lookup(args: "Geometry.CellArg", pos: Any):
        """Device function returning the cell sample point corresponding to a world position"""
        raise NotImplementedError

    def cell_lookup(args: "Geometry.CellArg", pos: Any, guess: "Sample"):
        """Device function returning the cell sample point corresponding to a world position. Can use guess for faster lookup"""
        raise NotImplementedError

    def cell_measure(args: "Geometry.CellArg", cell_index: ElementIndex, coords: Coords):
        """Device function returning the measure determinant (e.g. volume, area) at a given point"""
        raise NotImplementedError

    def cell_measure(args: "Geometry.CellArg", s: "Sample"):
        """Device function returning the measure determinant (e.g. volume, area) at a given point"""
        raise NotImplementedError

    def cell_measure_ratio(args: "Geometry.CellArg", s: "Sample"):
        """Device function returning the ratio of the measure of a side to that of its neighbour cells"""
        raise NotImplementedError

    def cell_normal(args: "Geometry.CellArg", s: "Sample"):
        """Device function returning the element normal at a sample point"""
        raise NotImplementedError

    def side_arg_value(self, device) -> "Geometry.SideArg":
        """Value of the arguments to be passed to side-related device functions"""
        raise NotImplementedError

    def boundary_side_index(args: "Geometry.SideArg", boundary_side_index: int):
        """Device function returning the side index corresponding to a boundary side"""
        raise NotImplementedError

    def side_position(args: "Geometry.SideArg", s: "Sample"):
        """Device function returning the side position at a sample point"""
        raise NotImplementedError

    def side_measure(args: "Geometry.SideArg", cell_index: ElementIndex, coords: Coords):
        """Device function returning the measure determinant (e.g. volume, area) at a given point"""
        raise NotImplementedError

    def side_measure(args: "Geometry.SideArg", s: "Sample"):
        """Device function returning the measure determinant (e.g. volume, area) at a given point"""
        raise NotImplementedError

    def side_measure_ratio(args: "Geometry.SideArg", s: "Sample"):
        """Device function returning the ratio of the measure of a side to that of its neighbour cells"""
        raise NotImplementedError

    def side_normal(args: "Geometry.SideArg", s: "Sample"):
        """Device function returning the element normal at a sample point"""
        raise NotImplementedError

    def side_inner_cell_index(args: "Geometry.SideArg", side_index: ElementIndex):
        """Device function returning the inner cell index for a given side"""
        raise NotImplementedError

    def side_outer_cell_index(args: "Geometry.SideArg", side_index: ElementIndex):
        """Device function returning the outer cell index for a given side"""
        raise NotImplementedError
