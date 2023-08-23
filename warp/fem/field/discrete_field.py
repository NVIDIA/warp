from typing import Any

import warp as wp

from warp.fem.types import Sample
from warp.fem.space import FunctionSpace, SpacePartition


class DiscreteField:
    """Field defined over a partition of a discrete function space"""

    EvalArg: wp.codegen.Struct
    """Structure containing arguments passed to device functions for field evaluation"""

    def __init__(self, space: FunctionSpace, space_partition: SpacePartition):
        self.space = space
        self.space_partition = space_partition

        self.dtype = self.space.dtype
        self.dof_dtype = self.space.dof_dtype

    def eval_arg_value(self, device) -> wp.codegen.StructInstance:
        raise NotImplementedError

    @property
    def dof_values(self) -> wp.array:
        """Array of degrees of freedom values"""
        raise NotImplementedError

    @dof_values.setter
    def dof_values(self, values: wp.array):
        """Sets degrees of freedom values from an array"""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return f"{self.__class__.__qualname__}_{self.space.name}_{self.space_partition.name}"

    @property
    def __str__(self) -> str:
        return self.name

    def trace(self) -> FunctionSpace:
        """Trace of this field over a lower-dimensional function space"""
        raise NotImplementedError

    def eval_arg_value(self, device):
        """Value of arguments to be passed to device functions"""
        raise NotImplementedError

    def set_node_value(args: Any, node_index: int, value: Any):
        """Device function setting the value at given node"""
        raise NotImplementedError

    def eval_inner(args: Any, s: "Sample"):
        """Device function evaluating the inner field value at a sample point"""
        raise NotImplementedError

    def eval_grad_inner(args: Any, s: "Sample"):
        """Device function evaluating the inner field gradient at a sample point"""
        raise NotImplementedError

    def eval_outer(args: Any, s: "Sample"):
        """Device function evaluating the outer field value at a sample point"""
        raise NotImplementedError

    def eval_grad_outer(args: Any, s: "Sample"):
        """Device function evaluating the outer field gradient at a sample point"""
        raise NotImplementedError

    @staticmethod
    def _make_eval_degree(EvalArg, space: FunctionSpace):
        ORDER = space.ORDER

        def degree(args: EvalArg):
            return ORDER

        from warp.fem import cache

        return cache.get_func(degree, space)
