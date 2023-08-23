import inspect
from typing import Callable, Any

import warp as wp

from warp.fem.types import Domain, Field, Sample
from warp.fem import utils


class Integrand:
    """An integrand is a device function containing arbitrary expressions over Field and Domain variables.
    It will get transformed to a proper warp.Function by resolving concrete Field types at call time.
    """

    def __init__(self, func: Callable):
        self.func = func
        self.name = wp.codegen.make_full_qualified_name(self.func)
        self.module = wp.get_module(self.func.__module__)
        self.argspec = inspect.getfullargspec(self.func)


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
def position(domain: Domain, x: Sample):
    """Evaluates the world position of the sample point x"""
    pass


@operator(resolver=lambda dmn: dmn.eval_normal)
def normal(domain: Domain, x: Sample):
    """Evaluates the element normal at the sample point x. Null for interior points."""
    pass


@operator(resolver=lambda dmn: dmn.element_lookup)
def lookup(domain: Domain, x: Any):
    """Look-ups a sample point from a world position, projecting to the closet point on the domain"""
    pass


@operator(resolver=lambda dmn: dmn.element_measure)
def measure(domain: Domain, sample: Sample):
    """Returns the measure (volume, area, or length) of an element"""
    pass


@operator(resolver=lambda dmn: dmn.element_measure_ratio)
def measure_ratio(domain: Domain, sample: Sample):
    """Returns the maximum ratio between the measure of this element and that of higher-dimensional neighbours."""
    pass


# Field operators
# On a side, inner and outer are such that normal goes from inner to outer


@operator(resolver=lambda f: f.eval_inner)
def inner(f: Field, x: Sample):
    """Evaluates the field at a sample point x. On oriented sides, use the inner element"""
    pass


@operator(resolver=lambda f: f.eval_grad_inner)
def grad(f: Field, x: Sample):
    """Evaluates the field gradient at a sample point x. On oriented sides, use the inner element"""
    pass


@operator(resolver=lambda f: f.eval_outer)
def outer(f: Field, x: Sample):
    """Evaluates the field at a sample point x. On oriented sides, use the outer element. On interior points and on domain boundaries, this is equivalent to inner."""
    pass


@operator(resolver=lambda f: f.eval_grad_outer)
def grad_outer(f: Field, x: Sample):
    """Evaluates the field gradient at a sample point x. On oriented sides, use the outer element. On interior points and on domain boundaries, this is equivalent to grad."""
    pass


@operator(resolver=lambda f: f.eval_degree)
def degree(f: Field):
    """Polynomial degree of a field"""
    pass


# Common derived operators, for convenience


@integrand
def D(f: Field, x: Sample):
    """Symmetric part of the (inner) gradient of the field at x"""
    return utils.symmetric_part(grad(f, x))


@integrand
def div(f: Field, x: Sample):
    """(Inner) divergence of the field at x"""
    return wp.trace(grad(f, x))


@integrand
def jump(f: Field, x: Sample):
    """Jump between inner and outer element values on an interior side. Zero for interior points or domain boundaries"""
    return inner(f, x) - outer(f, x)


@integrand
def average(f: Field, x: Sample):
    """Average between inner and outer element values"""
    return 0.5 * (inner(f, x) + outer(f, x))


@integrand
def grad_jump(f: Field, x: Sample):
    """Jump between inner and outer element gradients on an interior side. Zero for interior points or domain boundaries"""
    return grad(f, x) - grad_outer(f, x)


@integrand
def grad_average(f: Field, x: Sample):
    """Average between inner and outer element gradients"""
    return 0.5 * (grad(f, x) + grad_outer(f, x))


# Set default call operators for argument types, so that field(s) = inner(field, s) and domain(s) = position(domain, s)
Field.call_operator = inner
Domain.call_operator = position
