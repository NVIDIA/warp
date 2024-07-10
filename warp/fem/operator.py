from typing import Any, Callable

import warp as wp
from warp.fem import utils
from warp.fem.types import Domain, Field, NodeIndex, Sample


class Integrand:
    """An integrand is a device function containing arbitrary expressions over Field and Domain variables.
    It will get transformed to a proper warp.Function by resolving concrete Field types at call time.
    """

    def __init__(self, func: Callable):
        self.func = func
        self.name = wp.codegen.make_full_qualified_name(self.func)
        self.module = wp.get_module(self.func.__module__)
        self.argspec = wp.codegen.get_full_arg_spec(self.func)


class Operator:
    """
    Operators provide syntaxic sugar over Field and Domain evaluation functions and arguments
    """

    def __init__(self, func: Callable, resolver: Callable):
        self.func = func
        self.resolver = resolver


def integrand(func: Callable):
    """Decorator for functions to be integrated (or interpolated) using warp.fem"""
    itg = Integrand(func)
    itg.__doc__ = func.__doc__
    return itg


def operator(resolver: Callable):
    """Decorator for functions operating on Field-like or Domain-like data inside warp.fem integrands"""

    def wrap_operator(func: Callable):
        op = Operator(func, resolver)
        op.__doc__ = func.__doc__
        return op

    return wrap_operator


# Domain operators


@operator(resolver=lambda dmn: dmn.element_position)
def position(domain: Domain, s: Sample):
    """Evaluates the world position of the sample point `s`"""
    pass


@operator(resolver=lambda dmn: dmn.element_normal)
def normal(domain: Domain, s: Sample):
    """Evaluates the element normal at the sample point `s`. Null for interior points."""
    pass


@operator(resolver=lambda dmn: dmn.element_deformation_gradient)
def deformation_gradient(domain: Domain, s: Sample):
    """Evaluates the gradient of the domain position with respect to the element reference space at the sample point `s`"""
    pass


@operator(resolver=lambda dmn: dmn.element_lookup)
def lookup(domain: Domain, x: Any) -> Sample:
    """Looks-up the sample point corresponding to a world position `x`, projecting to the closest point on the domain.

    Args:
        x: world position of the point to look-up in the geometry
        guess: (optional) :class:`Sample` initial guess, may help perform the query

    Note:
        Currently this operator is unsupported for :class:`Hexmesh`, :class:`Quadmesh2D` and deformed geometries.
    """
    pass


@operator(resolver=lambda dmn: dmn.element_measure)
def measure(domain: Domain, s: Sample) -> float:
    """Returns the measure (volume, area, or length) determinant of an element at a sample point `s`"""
    pass


@operator(resolver=lambda dmn: dmn.element_measure_ratio)
def measure_ratio(domain: Domain, s: Sample) -> float:
    """Returns the maximum ratio between the measure of this element and that of higher-dimensional neighbours."""
    pass


# Field operators
# On a side, inner and outer are such that normal goes from inner to outer


@operator(resolver=lambda f: f.eval_inner)
def inner(f: Field, s: Sample):
    """Evaluates the field at a sample point `s`. On oriented sides, uses the inner element"""
    pass


@operator(resolver=lambda f: f.eval_grad_inner)
def grad(f: Field, s: Sample):
    """Evaluates the field gradient at a sample point `s`. On oriented sides, uses the inner element"""
    pass


@operator(resolver=lambda f: f.eval_div_inner)
def div(f: Field, s: Sample):
    """Evaluates the field divergence at a sample point `s`. On oriented sides, uses the inner element"""
    pass


@operator(resolver=lambda f: f.eval_outer)
def outer(f: Field, s: Sample):
    """Evaluates the field at a sample point `s`. On oriented sides, uses the outer element. On interior points and on domain boundaries, this is equivalent to :func:`inner`."""
    pass


@operator(resolver=lambda f: f.eval_grad_outer)
def grad_outer(f: Field, s: Sample):
    """Evaluates the field gradient at a sample point `s`. On oriented sides, uses the outer element. On interior points and on domain boundaries, this is equivalent to :func:`grad`."""
    pass


@operator(resolver=lambda f: f.eval_grad_outer)
def div_outer(f: Field, s: Sample):
    """Evaluates the field divergence at a sample point `s`. On oriented sides, uses the outer element. On interior points and on domain boundaries, this is equivalent to :func:`div`."""
    pass


@operator(resolver=lambda f: f.eval_degree)
def degree(f: Field):
    """Polynomial degree of a field"""
    pass


@operator(resolver=lambda f: f.at_node)
def at_node(f: Field, s: Sample):
    """For a Test or Trial field `f`, returns a copy of the Sample `s` moved to the coordinates of the node being evaluated"""
    pass


@operator(resolver=lambda f: f.node_partition_index)
def node_partition_index(f: Field, node_index: NodeIndex):
    """For a NodalField `f`, returns the index of a given node in the fields's space partition,
    or ``NULL_NODE_INDEX`` if it does not exists"""
    pass


# Common derived operators, for convenience


@integrand
def D(f: Field, s: Sample):
    """Symmetric part of the (inner) gradient of the field at `s`"""
    return utils.symmetric_part(grad(f, s))


@integrand
def curl(f: Field, s: Sample):
    """Skew part of the (inner) gradient of the field at `s`, as a vector such that ``wp.cross(curl(u), v) = skew(grad(u)) v``"""
    return utils.skew_part(grad(f, s))


@integrand
def jump(f: Field, s: Sample):
    """Jump between inner and outer element values on an interior side. Zero for interior points or domain boundaries"""
    return inner(f, s) - outer(f, s)


@integrand
def average(f: Field, s: Sample):
    """Average between inner and outer element values"""
    return 0.5 * (inner(f, s) + outer(f, s))


@integrand
def grad_jump(f: Field, s: Sample):
    """Jump between inner and outer element gradients on an interior side. Zero for interior points or domain boundaries"""
    return grad(f, s) - grad_outer(f, s)


@integrand
def grad_average(f: Field, s: Sample):
    """Average between inner and outer element gradients"""
    return 0.5 * (grad(f, s) + grad_outer(f, s))


# Set default call operators for argument types, so that field(s) = inner(field, s) and domain(s) = position(domain, s)
Field.call_operator = inner
Domain.call_operator = position
