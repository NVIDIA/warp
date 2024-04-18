from typing import Any

import warp as wp
from warp.fem.geometry import DeformedGeometry, Geometry
from warp.fem.space import FunctionSpace, SpacePartition
from warp.fem.types import Sample


class FieldLike:
    """Base class for integrable fields"""

    EvalArg: wp.codegen.Struct
    """Structure containing field-level arguments passed to device functions for field evaluation"""

    ElementEvalArg: wp.codegen.Struct
    """Structure combining geometry-level and field-level arguments passed to device functions for field evaluation"""

    def eval_arg_value(self, device) -> "EvalArg":  # noqa: F821
        """Value of the field-level arguments to be passed to device functions"""
        raise NotImplementedError

    @property
    def degree(self) -> int:
        """Polynomial degree of the field, used to estimate necessary quadrature order"""
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def __str__(self) -> str:
        return self.name

    def eval_arg_value(self, device):
        """Value of arguments to be passed to device functions"""
        raise NotImplementedError

    @staticmethod
    def eval_inner(args: "ElementEvalArg", s: "Sample"):  # noqa: F821
        """Device function evaluating the inner field value at a sample point"""
        raise NotImplementedError

    @staticmethod
    def eval_grad_inner(args: "ElementEvalArg", s: "Sample"):  # noqa: F821
        """Device function evaluating the inner field gradient at a sample point"""
        raise NotImplementedError

    @staticmethod
    def eval_div_inner(args: "ElementEvalArg", s: "Sample"):  # noqa: F821
        """Device function evaluating the inner field divergence at a sample point"""
        raise NotImplementedError

    @staticmethod
    def eval_outer(args: "ElementEvalArg", s: "Sample"):  # noqa: F821
        """Device function evaluating the outer field value at a sample point"""
        raise NotImplementedError

    @staticmethod
    def eval_grad_outer(args: "ElementEvalArg", s: "Sample"):  # noqa: F821
        """Device function evaluating the outer field gradient at a sample point"""
        raise NotImplementedError

    @staticmethod
    def eval_div_outer(args: "ElementEvalArg", s: "Sample"):  # noqa: F821
        """Device function evaluating the outer field divergence at a sample point"""
        raise NotImplementedError


class SpaceField(FieldLike):
    """Base class for fields defined over a function space"""

    def __init__(self, space: FunctionSpace, space_partition: SpacePartition):
        self._space = space
        self._space_partition = space_partition

    @property
    def space(self) -> FunctionSpace:
        return self._space

    @property
    def space_partition(self) -> SpacePartition:
        return self._space_partition

    @property
    def degree(self) -> int:
        return self.space.degree

    @property
    def dtype(self) -> type:
        return self.space.dtype

    @property
    def dof_dtype(self) -> type:
        return self.space.dof_dtype

    def gradient_valid(self) -> bool:
        """Whether gradient operator can be computed. Only for scalar and vector fields as higher-order tensors are not support yet"""
        return not wp.types.type_is_matrix(self.dtype)

    def divergence_valid(self) -> bool:
        """Whether divergence of this field can be computed. Only for vector and tensor fields with same dimension as embedding geometry"""
        if wp.types.type_is_vector(self.dtype):
            return wp.types.type_length(self.dtype) == self.space.geometry.dimension
        if wp.types.type_is_matrix(self.dtype):
            return self.dtype._shape_[0] == self.space.geometry.dimension
        return False

    def _make_eval_degree(self):
        ORDER = self.space.ORDER
        from warp.fem import cache

        @cache.dynamic_func(suffix=self.name)
        def degree(args: self.ElementEvalArg):
            return ORDER

        return degree


class DiscreteField(SpaceField):
    """Explicitly-valued field defined over a partition of a discrete function space"""

    @property
    def dof_values(self) -> wp.array:
        """Array of degrees of freedom values"""
        raise NotImplementedError

    @dof_values.setter
    def dof_values(self, values: wp.array):
        """Sets degrees of freedom values from an array"""
        raise NotImplementedError

    def trace(self) -> "DiscreteField":
        """Trace of this field over a lower-dimensional function space"""
        raise NotImplementedError

    @staticmethod
    def set_node_value(args: "FieldLike.EvalArg", node_index: int, value: Any):
        """Device function setting the value at given node"""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return f"{self.__class__.__qualname__}_{self.space.name}_{self.space_partition.name}"

    def make_deformed_geometry(self) -> Geometry:
        """Returns a deformed version of the underlying geometry using this field's values as displacement"""
        return DeformedGeometry(self)
