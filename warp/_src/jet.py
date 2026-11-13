# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-order forward-mode jets as ordinary differentiable Warp types.

A jet carries a value together with its derivatives along ``width`` directions
at once::

    x = value + sum_q grad[q] * eps_q

Evaluating a function over jets instead of floats produces its derivatives as a
side effect of evaluating it, with no hand-differentiation. Because the jet
arithmetic is itself ordinary Warp code, Warp's reverse-mode autodiff can
differentiate through it, which makes reverse-over-forward -- and therefore
Hessians -- available without a second-order tape.

:func:`JetSpace` specializes the types at code-generation time. It is described
by the inputs being differentiated, called slots:

    ============================ ======= ===================================
    ``JetSpace(slots)``          width   meaning
    ============================ ======= ===================================
    ``(wp.vec3, wp.vec3)``       6       two 3D points
    ``(wp.vec2, wp.mat22)``      6       a 2D point and a 2x2 matrix
    ``4``                        4       four scalars
    ============================ ======= ===================================

and generates:

    ========== ================== ====================
    Name       ``value``          ``grad``
    ========== ================== ====================
    ``scalar`` ``dtype``          ``vector(width)``
    ``vec2``   ``vector(2)``      ``matrix(2, width)``
    ``vec3``   ``vector(3)``      ``matrix(3, width)``
    ========== ================== ====================

(plus ``mat22``, ``mat33``, ``mat32`` and ``mat23``). The derivative width is
independent of the geometric vector dimension.

The generated arithmetic is registered as overloads of Warp's own builtins, so
jets are written with ordinary operator and builtin syntax::

    J = wp.JetSpace((wp.vec3, wp.vec3))


    @wp.func
    def spring_energy(x0: J.vec3, x1: J.vec3) -> J.scalar:
        d = x1 - x0
        r = wp.length(d) - 1.0
        return 0.5 * r * r


    @wp.kernel
    def local_gradient(
        x0: wp.array[wp.vec3],
        x1: wp.array[wp.vec3],
        grad: wp.array[J.grad_type_scalar],
    ):
        i = wp.tid()

        # Six local scalar variables: x0.x/y/z -> directions 0/1/2 and
        # x1.x/y/z -> directions 3/4/5.
        a, b = J.seed(x0[i], x1[i])

        grad[i] = spring_energy(a, b).grad

Note that ``spring_energy`` is a plain :func:`warp.func`. Operators resolve
through Warp's builtin overload table, so nothing has to be imported or bound
into the calling module beyond the space itself.

Component access follows the same route: ``v[0]`` on a ``J.vec3`` resolves
through the ``extract`` builtin and yields a ``J.scalar``.

Seeding and construction have no builtin counterpart and stay on the returned
namespace, as do the 2D-only helpers :attr:`perp` and :attr:`cross2`, which
Warp does not define for ``vec2``.
"""

# NOTE: deliberately no `from __future__ import annotations` here. The structs
# and functions below are annotated with types created inside _make_jet_space,
# and Warp resolves struct annotations with inspect.get_annotations(eval_str=True),
# which evaluates them against module globals. Deferred annotations would turn
# them into strings that no longer resolve.

import itertools
import linecache
import numbers
import operator
from types import SimpleNamespace

import warp as wp
import warp._src.context
from warp._src.types import float_types, matrix, vector

# Specialized namespaces, keyed by (width, dtype): first order and second order.
_CACHE = {}
_CACHE2 = {}


def _check_space_args(width: int, dtype) -> None:
    """Validate a jet space's parameters before any type is generated.

    Checked up front because generating a space mutates Warp's global builtin
    overload table, which cannot be undone for the life of the process.
    """
    if width <= 0:
        raise ValueError("Jet width must be positive")
    if dtype not in float_types:
        raise TypeError(
            f"Jet dtype must be a Warp floating-point type, got {getattr(dtype, '__name__', dtype)}. "
            "Derivatives are not representable in an integer type: division in the chain rule "
            "would truncate."
        )


def _register(name: str, fn) -> None:
    """Add every signature of ``fn`` as an overload of the builtin ``name``.

    A Warp function defined several times under one name accumulates its
    signatures in ``user_overloads``. Builtin overload resolution matches
    against a function's own ``input_types`` and does not recurse into that
    dict, so each signature has to be appended individually.
    """
    builtin = warp._src.context.builtin_functions[name]

    for overload in tuple(fn.user_overloads.values()) or (fn,):
        builtin.add_overload(overload)


@wp.func
def _component_index(i: int, n: int) -> int:
    # Count a negative index from the end, as native vector and matrix indexing does.
    if i < 0:
        return i + n
    return i


def _make_jet_space(width: int, dtype):
    _check_space_args(width, dtype)

    # Concrete Warp types generated at Python scope.
    Grad = vector(width, dtype)

    NativeVec2 = vector(2, dtype)
    NativeVec3 = vector(3, dtype)

    GradVec2 = matrix((2, width), dtype)
    GradVec3 = matrix((3, width), dtype)

    # ------------------------------------------------------------------
    # Core data types
    # ------------------------------------------------------------------

    @wp.struct
    class JetScalar:
        value: dtype
        grad: Grad

    @wp.struct
    class JetVec2:
        value: NativeVec2
        grad: GradVec2

    @wp.struct
    class JetVec3:
        value: NativeVec3
        grad: GradVec3

    # ------------------------------------------------------------------
    # Scalar construction / seeding
    # ------------------------------------------------------------------

    @wp.func
    def scalar_constant(x: dtype) -> JetScalar:
        return JetScalar(x, Grad())

    @wp.func
    def scalar_with_grad(x: dtype, grad: Grad) -> JetScalar:
        return JetScalar(x, grad)

    @wp.func
    def scalar_variable(x: dtype, direction: int) -> JetScalar:
        c = Grad()
        c[direction] = dtype(1.0)
        return JetScalar(x, c)

    # ------------------------------------------------------------------
    # Arithmetic
    #
    # Warp maps the AST operators to the builtin names add/sub/mul/div/pow/
    # pos/neg, so registering these below makes ordinary operator syntax work
    # on jets in any module.
    # ------------------------------------------------------------------

    @wp.func
    def jet_add(a: JetScalar, b: JetScalar) -> JetScalar:
        return JetScalar(a.value + b.value, a.grad + b.grad)

    @wp.func
    def jet_add(a: JetScalar, b: dtype) -> JetScalar:
        return JetScalar(a.value + b, a.grad)

    @wp.func
    def jet_add(a: dtype, b: JetScalar) -> JetScalar:
        return JetScalar(a + b.value, b.grad)

    @wp.func
    def jet_sub(a: JetScalar, b: JetScalar) -> JetScalar:
        return JetScalar(a.value - b.value, a.grad - b.grad)

    @wp.func
    def jet_sub(a: JetScalar, b: dtype) -> JetScalar:
        return JetScalar(a.value - b, a.grad)

    @wp.func
    def jet_sub(a: dtype, b: JetScalar) -> JetScalar:
        return JetScalar(a - b.value, -b.grad)

    @wp.func
    def jet_pos(a: JetScalar) -> JetScalar:
        return a

    @wp.func
    def jet_neg(a: JetScalar) -> JetScalar:
        return JetScalar(-a.value, -a.grad)

    @wp.func
    def jet_mul(a: JetScalar, b: JetScalar) -> JetScalar:
        return JetScalar(
            a.value * b.value,
            a.grad * b.value + b.grad * a.value,
        )

    @wp.func
    def jet_mul(a: JetScalar, b: dtype) -> JetScalar:
        return JetScalar(a.value * b, a.grad * b)

    @wp.func
    def jet_mul(a: dtype, b: JetScalar) -> JetScalar:
        return JetScalar(a * b.value, a * b.grad)

    @wp.func
    def jet_div(a: JetScalar, b: JetScalar) -> JetScalar:
        # Divide directly, as native division does: forming 1 / b first overflows for
        # tiny denominators even when the quotient is representable.
        value = a.value / b.value
        return JetScalar(value, (a.grad - b.grad * value) / b.value)

    @wp.func
    def jet_div(a: JetScalar, b: dtype) -> JetScalar:
        return JetScalar(a.value / b, a.grad / b)

    @wp.func
    def jet_div(a: dtype, b: JetScalar) -> JetScalar:
        value = a / b.value
        return JetScalar(value, (-value / b.value) * b.grad)

    # ------------------------------------------------------------------
    # Scalar elementary functions
    #
    # Each unary function is the chain rule f(a) = f(value) with derivative
    # f'(value) scaling the gradients, so they all route through _lift1.
    # ------------------------------------------------------------------

    @wp.func
    def _lift1(value: dtype, fp: dtype, a: JetScalar) -> JetScalar:
        # Single-argument chain rule: value = f(a.value), fp = f'(a.value).
        return JetScalar(value, fp * a.grad)

    @wp.func
    def jet_pow(a: JetScalar, p: dtype) -> JetScalar:
        # p == 0 makes this the constant 1, whose derivative is 0. The general
        # p * a**(p-1) form would evaluate it as 0 * inf at a == 0 and give NaN.
        # Non-integer p < 1 is left alone: its derivative really is infinite
        # there, and inf is the honest answer.
        if p == dtype(0.0):
            return JetScalar(dtype(1.0), Grad())
        return _lift1(wp.pow(a.value, p), p * wp.pow(a.value, p - dtype(1.0)), a)

    @wp.func
    def jet_pow(a: JetScalar, p: int) -> JetScalar:
        if p == 0:
            return JetScalar(dtype(1.0), Grad())
        return _lift1(wp.pow(a.value, dtype(p)), dtype(p) * wp.pow(a.value, dtype(p) - dtype(1.0)), a)

    @wp.func
    def jet_pow(a: JetScalar, b: JetScalar) -> JetScalar:
        # a**b: d = b a**(b-1) da + a**b log(a) db.
        #
        # Both partials are formed to stay finite at a.value == 0. Factoring
        # them out of the value instead -- value * (b / a) and value * log(a) --
        # each evaluates as 0 * inf there and gives NaN, even though the limits
        # are finite for b > 0. The da partial below is the same expression the
        # constant-exponent overload uses; the db partial takes its limit, which
        # is 0 as a -> 0 from above.
        value = wp.pow(a.value, b.value)
        # As in the constant-exponent overload, an exponent of 0 makes the da partial 0
        # rather than 0 * inf at a.value == 0.
        da = dtype(0.0)
        if b.value != dtype(0.0):
            da = b.value * wp.pow(a.value, b.value - dtype(1.0))
        if a.value > dtype(0.0):
            return JetScalar(value, da * a.grad + (value * wp.log(a.value)) * b.grad)
        return JetScalar(value, da * a.grad)

    @wp.func
    def jet_pow(a: dtype, b: JetScalar) -> JetScalar:
        value = wp.pow(a, b.value)
        # A zero base gives a constant result, so the derivative is 0 rather than 0 * log(0).
        if a == dtype(0.0):
            return JetScalar(value, Grad())
        return JetScalar(value, (value * wp.log(a)) * b.grad)

    @wp.func
    def jet_sin(a: JetScalar) -> JetScalar:
        return _lift1(wp.sin(a.value), wp.cos(a.value), a)

    @wp.func
    def jet_cos(a: JetScalar) -> JetScalar:
        return _lift1(wp.cos(a.value), -wp.sin(a.value), a)

    @wp.func
    def jet_tan(a: JetScalar) -> JetScalar:
        c = wp.cos(a.value)
        return _lift1(wp.tan(a.value), dtype(1.0) / (c * c), a)

    @wp.func
    def jet_asin(a: JetScalar) -> JetScalar:
        # Like native asin(), which clamps its argument to [-1, 1], the derivative is zero
        # unless the argument lies strictly inside the interval. Test the interval before
        # taking the square root: its adjoint is NaN outside, even where the result is unused.
        if a.value > dtype(-1.0) and a.value < dtype(1.0):
            return _lift1(wp.asin(a.value), dtype(1.0) / wp.sqrt(dtype(1.0) - a.value * a.value), a)
        return JetScalar(wp.asin(a.value), Grad())

    @wp.func
    def jet_acos(a: JetScalar) -> JetScalar:
        if a.value > dtype(-1.0) and a.value < dtype(1.0):
            return _lift1(wp.acos(a.value), -dtype(1.0) / wp.sqrt(dtype(1.0) - a.value * a.value), a)
        return JetScalar(wp.acos(a.value), Grad())

    @wp.func
    def jet_atan(a: JetScalar) -> JetScalar:
        return _lift1(wp.atan(a.value), dtype(1.0) / (dtype(1.0) + a.value * a.value), a)

    @wp.func
    def jet_exp(a: JetScalar) -> JetScalar:
        e = wp.exp(a.value)
        return _lift1(e, e, a)

    @wp.func
    def jet_log(a: JetScalar) -> JetScalar:
        return _lift1(wp.log(a.value), dtype(1.0) / a.value, a)

    @wp.func
    def jet_sqrt(a: JetScalar) -> JetScalar:
        s = wp.sqrt(a.value)
        return _lift1(s, dtype(0.5) / s, a)

    @wp.func
    def jet_atan2(y: JetScalar, x: JetScalar) -> JetScalar:
        # Like native atan2(), the derivatives are zero at the origin.
        d = x.value * x.value + y.value * y.value
        if d > dtype(0.0):
            return JetScalar(wp.atan2(y.value, x.value), (x.value * y.grad - y.value * x.grad) / d)
        return JetScalar(wp.atan2(y.value, x.value), Grad())

    @wp.func
    def jet_atan2(y: JetScalar, x: dtype) -> JetScalar:
        return jet_atan2(y, scalar_constant(x))

    @wp.func
    def jet_atan2(y: dtype, x: JetScalar) -> JetScalar:
        return jet_atan2(scalar_constant(y), x)

    @wp.func
    def jet_abs(a: JetScalar) -> JetScalar:
        # Native adj_abs() takes the derivative +1 at zero, and abs(NaN) is NaN.
        if a.value < dtype(0.0):
            return -a
        return JetScalar(wp.abs(a.value), a.grad)

    @wp.func
    def jet_sign(a: JetScalar) -> JetScalar:
        return JetScalar(wp.sign(a.value), Grad())

    # ------------------------------------------------------------------
    # Selection / branching. Comparisons are value-only (a jet has no order),
    # so branch on ``.value`` and carry the chosen jet's derivatives through.
    # The choice follows native min()/max()/clamp() and their adjoints: a NaN
    # operand is treated as missing, and a tie selects the second operand.
    # ------------------------------------------------------------------

    @wp.func
    def jet_min(a: JetScalar, b: JetScalar) -> JetScalar:
        if wp.isnan(a.value):
            return b
        if wp.isnan(b.value):
            return a
        if a.value < b.value:
            return a
        return b

    @wp.func
    def jet_max(a: JetScalar, b: JetScalar) -> JetScalar:
        if wp.isnan(a.value):
            return b
        if wp.isnan(b.value):
            return a
        if a.value > b.value:
            return a
        return b

    @wp.func
    def jet_min(a: JetScalar, b: dtype) -> JetScalar:
        return jet_min(a, scalar_constant(b))

    @wp.func
    def jet_min(a: dtype, b: JetScalar) -> JetScalar:
        return jet_min(scalar_constant(a), b)

    @wp.func
    def jet_max(a: JetScalar, b: dtype) -> JetScalar:
        return jet_max(a, scalar_constant(b))

    @wp.func
    def jet_max(a: dtype, b: JetScalar) -> JetScalar:
        return jet_max(scalar_constant(a), b)

    @wp.func
    def jet_clamp(a: JetScalar, lo: dtype, hi: dtype) -> JetScalar:
        # Native clamp(x, lo, hi) is min(max(lo, x), hi), so the same composition
        # reproduces its NaN handling and its derivative at either bound.
        return jet_min(jet_max(scalar_constant(lo), a), scalar_constant(hi))

    @wp.func
    def jet_where(cond: bool, a: JetScalar, b: JetScalar) -> JetScalar:
        if cond:
            return a
        return b

    # ------------------------------------------------------------------
    # Vec2 / Vec3 construction
    # ------------------------------------------------------------------

    @wp.func
    def vec2_constant(v: NativeVec2) -> JetVec2:
        return JetVec2(v, GradVec2())

    @wp.func
    def vec3_constant(v: NativeVec3) -> JetVec3:
        return JetVec3(v, GradVec3())

    @wp.func
    def vec2_from_scalars(x: JetScalar, y: JetScalar) -> JetVec2:
        c = GradVec2()
        for q in range(wp.static(width)):
            c[0, q] = x.grad[q]
            c[1, q] = y.grad[q]
        return JetVec2(NativeVec2(x.value, y.value), c)

    @wp.func
    def vec3_from_scalars(x: JetScalar, y: JetScalar, z: JetScalar) -> JetVec3:
        c = GradVec3()
        for q in range(wp.static(width)):
            c[0, q] = x.grad[q]
            c[1, q] = y.grad[q]
            c[2, q] = z.grad[q]
        return JetVec3(NativeVec3(x.value, y.value, z.value), c)

    # ------------------------------------------------------------------
    # Component access
    # ------------------------------------------------------------------

    @wp.func
    def jet_extract(v: JetVec2, i: int) -> JetScalar:
        i = _component_index(i, 2)
        c = Grad()
        for q in range(wp.static(width)):
            c[q] = v.grad[i, q]
        return JetScalar(v.value[i], c)

    @wp.func
    def jet_extract(v: JetVec3, i: int) -> JetScalar:
        i = _component_index(i, 3)
        c = Grad()
        for q in range(wp.static(width)):
            c[q] = v.grad[i, q]
        return JetScalar(v.value[i], c)

    # ------------------------------------------------------------------
    # Vec2 arithmetic
    # ------------------------------------------------------------------

    @wp.func
    def jet_add(a: JetVec2, b: JetVec2) -> JetVec2:
        return JetVec2(a.value + b.value, a.grad + b.grad)

    @wp.func
    def jet_add(a: JetVec2, b: NativeVec2) -> JetVec2:
        return JetVec2(a.value + b, a.grad)

    @wp.func
    def jet_add(a: NativeVec2, b: JetVec2) -> JetVec2:
        return JetVec2(a + b.value, b.grad)

    @wp.func
    def jet_sub(a: JetVec2, b: JetVec2) -> JetVec2:
        return JetVec2(a.value - b.value, a.grad - b.grad)

    @wp.func
    def jet_sub(a: JetVec2, b: NativeVec2) -> JetVec2:
        return JetVec2(a.value - b, a.grad)

    @wp.func
    def jet_sub(a: NativeVec2, b: JetVec2) -> JetVec2:
        return JetVec2(a - b.value, -b.grad)

    @wp.func
    def jet_pos(a: JetVec2) -> JetVec2:
        return a

    @wp.func
    def jet_neg(a: JetVec2) -> JetVec2:
        return JetVec2(-a.value, -a.grad)

    @wp.func
    def jet_mul(a: JetVec2, s: dtype) -> JetVec2:
        return JetVec2(a.value * s, a.grad * s)

    @wp.func
    def jet_mul(s: dtype, a: JetVec2) -> JetVec2:
        return JetVec2(s * a.value, s * a.grad)

    @wp.func
    def jet_mul(a: JetVec2, s: JetScalar) -> JetVec2:
        c = GradVec2()
        for q in range(wp.static(width)):
            c[0, q] = a.grad[0, q] * s.value + a.value[0] * s.grad[q]
            c[1, q] = a.grad[1, q] * s.value + a.value[1] * s.grad[q]
        return JetVec2(a.value * s.value, c)

    @wp.func
    def jet_mul(s: JetScalar, a: JetVec2) -> JetVec2:
        c = GradVec2()
        for q in range(wp.static(width)):
            c[0, q] = s.value * a.grad[0, q] + s.grad[q] * a.value[0]
            c[1, q] = s.value * a.grad[1, q] + s.grad[q] * a.value[1]
        return JetVec2(s.value * a.value, c)

    @wp.func
    def jet_mul(a: NativeVec2, s: JetScalar) -> JetVec2:
        c = GradVec2()
        for q in range(wp.static(width)):
            c[0, q] = a[0] * s.grad[q]
            c[1, q] = a[1] * s.grad[q]
        return JetVec2(a * s.value, c)

    @wp.func
    def jet_mul(s: JetScalar, a: NativeVec2) -> JetVec2:
        c = GradVec2()
        for q in range(wp.static(width)):
            c[0, q] = s.grad[q] * a[0]
            c[1, q] = s.grad[q] * a[1]
        return JetVec2(s.value * a, c)

    @wp.func
    def jet_div(a: JetVec2, s: dtype) -> JetVec2:
        return JetVec2(a.value / s, a.grad / s)

    @wp.func
    def jet_div(a: NativeVec2, s: JetScalar) -> JetVec2:
        value = a / s.value
        c = GradVec2()
        for q in range(wp.static(width)):
            c[0, q] = -value[0] * s.grad[q] / s.value
            c[1, q] = -value[1] * s.grad[q] / s.value
        return JetVec2(value, c)

    @wp.func
    def jet_div(a: JetVec2, s: JetScalar) -> JetVec2:
        value = a.value / s.value
        c = GradVec2()
        for q in range(wp.static(width)):
            c[0, q] = (a.grad[0, q] - value[0] * s.grad[q]) / s.value
            c[1, q] = (a.grad[1, q] - value[1] * s.grad[q]) / s.value
        return JetVec2(value, c)

    # ------------------------------------------------------------------
    # Vec3 arithmetic
    # ------------------------------------------------------------------

    @wp.func
    def jet_add(a: JetVec3, b: JetVec3) -> JetVec3:
        return JetVec3(a.value + b.value, a.grad + b.grad)

    @wp.func
    def jet_add(a: JetVec3, b: NativeVec3) -> JetVec3:
        return JetVec3(a.value + b, a.grad)

    @wp.func
    def jet_add(a: NativeVec3, b: JetVec3) -> JetVec3:
        return JetVec3(a + b.value, b.grad)

    @wp.func
    def jet_sub(a: JetVec3, b: JetVec3) -> JetVec3:
        return JetVec3(a.value - b.value, a.grad - b.grad)

    @wp.func
    def jet_sub(a: JetVec3, b: NativeVec3) -> JetVec3:
        return JetVec3(a.value - b, a.grad)

    @wp.func
    def jet_sub(a: NativeVec3, b: JetVec3) -> JetVec3:
        return JetVec3(a - b.value, -b.grad)

    @wp.func
    def jet_pos(a: JetVec3) -> JetVec3:
        return a

    @wp.func
    def jet_neg(a: JetVec3) -> JetVec3:
        return JetVec3(-a.value, -a.grad)

    @wp.func
    def jet_mul(a: JetVec3, s: dtype) -> JetVec3:
        return JetVec3(a.value * s, a.grad * s)

    @wp.func
    def jet_mul(s: dtype, a: JetVec3) -> JetVec3:
        return JetVec3(s * a.value, s * a.grad)

    @wp.func
    def jet_mul(a: JetVec3, s: JetScalar) -> JetVec3:
        c = GradVec3()
        for q in range(wp.static(width)):
            c[0, q] = a.grad[0, q] * s.value + a.value[0] * s.grad[q]
            c[1, q] = a.grad[1, q] * s.value + a.value[1] * s.grad[q]
            c[2, q] = a.grad[2, q] * s.value + a.value[2] * s.grad[q]
        return JetVec3(a.value * s.value, c)

    @wp.func
    def jet_mul(s: JetScalar, a: JetVec3) -> JetVec3:
        c = GradVec3()
        for q in range(wp.static(width)):
            c[0, q] = s.value * a.grad[0, q] + s.grad[q] * a.value[0]
            c[1, q] = s.value * a.grad[1, q] + s.grad[q] * a.value[1]
            c[2, q] = s.value * a.grad[2, q] + s.grad[q] * a.value[2]
        return JetVec3(s.value * a.value, c)

    @wp.func
    def jet_mul(a: NativeVec3, s: JetScalar) -> JetVec3:
        c = GradVec3()
        for q in range(wp.static(width)):
            c[0, q] = a[0] * s.grad[q]
            c[1, q] = a[1] * s.grad[q]
            c[2, q] = a[2] * s.grad[q]
        return JetVec3(a * s.value, c)

    @wp.func
    def jet_mul(s: JetScalar, a: NativeVec3) -> JetVec3:
        c = GradVec3()
        for q in range(wp.static(width)):
            c[0, q] = s.grad[q] * a[0]
            c[1, q] = s.grad[q] * a[1]
            c[2, q] = s.grad[q] * a[2]
        return JetVec3(s.value * a, c)

    @wp.func
    def jet_div(a: JetVec3, s: dtype) -> JetVec3:
        return JetVec3(a.value / s, a.grad / s)

    @wp.func
    def jet_div(a: NativeVec3, s: JetScalar) -> JetVec3:
        value = a / s.value
        c = GradVec3()
        for q in range(wp.static(width)):
            c[0, q] = -value[0] * s.grad[q] / s.value
            c[1, q] = -value[1] * s.grad[q] / s.value
            c[2, q] = -value[2] * s.grad[q] / s.value
        return JetVec3(value, c)

    @wp.func
    def jet_div(a: JetVec3, s: JetScalar) -> JetVec3:
        value = a.value / s.value
        c = GradVec3()
        for q in range(wp.static(width)):
            c[0, q] = (a.grad[0, q] - value[0] * s.grad[q]) / s.value
            c[1, q] = (a.grad[1, q] - value[1] * s.grad[q]) / s.value
            c[2, q] = (a.grad[2, q] - value[2] * s.grad[q]) / s.value
        return JetVec3(value, c)

    # ------------------------------------------------------------------
    # Dot products
    # ------------------------------------------------------------------

    @wp.func
    def jet_dot(a: JetVec2, b: JetVec2) -> JetScalar:
        c = Grad()
        for q in range(wp.static(width)):
            c[q] = (
                a.grad[0, q] * b.value[0]
                + a.grad[1, q] * b.value[1]
                + a.value[0] * b.grad[0, q]
                + a.value[1] * b.grad[1, q]
            )
        return JetScalar(wp.dot(a.value, b.value), c)

    @wp.func
    def jet_dot(a: JetVec2, b: NativeVec2) -> JetScalar:
        c = Grad()
        for q in range(wp.static(width)):
            c[q] = a.grad[0, q] * b[0] + a.grad[1, q] * b[1]
        return JetScalar(wp.dot(a.value, b), c)

    @wp.func
    def jet_dot(a: NativeVec2, b: JetVec2) -> JetScalar:
        c = Grad()
        for q in range(wp.static(width)):
            c[q] = a[0] * b.grad[0, q] + a[1] * b.grad[1, q]
        return JetScalar(wp.dot(a, b.value), c)

    @wp.func
    def jet_dot(a: JetVec3, b: JetVec3) -> JetScalar:
        c = Grad()
        for q in range(wp.static(width)):
            c[q] = (
                a.grad[0, q] * b.value[0]
                + a.grad[1, q] * b.value[1]
                + a.grad[2, q] * b.value[2]
                + a.value[0] * b.grad[0, q]
                + a.value[1] * b.grad[1, q]
                + a.value[2] * b.grad[2, q]
            )
        return JetScalar(wp.dot(a.value, b.value), c)

    @wp.func
    def jet_dot(a: JetVec3, b: NativeVec3) -> JetScalar:
        c = Grad()
        for q in range(wp.static(width)):
            c[q] = a.grad[0, q] * b[0] + a.grad[1, q] * b[1] + a.grad[2, q] * b[2]
        return JetScalar(wp.dot(a.value, b), c)

    @wp.func
    def jet_dot(a: NativeVec3, b: JetVec3) -> JetScalar:
        c = Grad()
        for q in range(wp.static(width)):
            c[q] = a[0] * b.grad[0, q] + a[1] * b.grad[1, q] + a[2] * b.grad[2, q]
        return JetScalar(wp.dot(a, b.value), c)

    # ------------------------------------------------------------------
    # Component-wise products
    # ------------------------------------------------------------------

    @wp.func
    def jet_cw_mul(a: JetVec2, b: JetVec2) -> JetVec2:
        c = GradVec2()
        for i in range(2):
            for q in range(wp.static(width)):
                c[i, q] = a.grad[i, q] * b.value[i] + a.value[i] * b.grad[i, q]
        return JetVec2(wp.cw_mul(a.value, b.value), c)

    @wp.func
    def jet_cw_mul(a: JetVec2, b: NativeVec2) -> JetVec2:
        c = GradVec2()
        for i in range(2):
            for q in range(wp.static(width)):
                c[i, q] = a.grad[i, q] * b[i]
        return JetVec2(wp.cw_mul(a.value, b), c)

    @wp.func
    def jet_cw_mul(a: NativeVec2, b: JetVec2) -> JetVec2:
        c = GradVec2()
        for i in range(2):
            for q in range(wp.static(width)):
                c[i, q] = a[i] * b.grad[i, q]
        return JetVec2(wp.cw_mul(a, b.value), c)

    @wp.func
    def jet_cw_div(a: JetVec2, b: JetVec2) -> JetVec2:
        value = wp.cw_div(a.value, b.value)
        c = GradVec2()
        for i in range(2):
            inv_b = dtype(1.0) / b.value[i]
            for q in range(wp.static(width)):
                c[i, q] = (a.grad[i, q] - value[i] * b.grad[i, q]) * inv_b
        return JetVec2(value, c)

    @wp.func
    def jet_cw_div(a: JetVec2, b: NativeVec2) -> JetVec2:
        c = GradVec2()
        for i in range(2):
            inv_b = dtype(1.0) / b[i]
            for q in range(wp.static(width)):
                c[i, q] = a.grad[i, q] * inv_b
        return JetVec2(wp.cw_div(a.value, b), c)

    @wp.func
    def jet_cw_div(a: NativeVec2, b: JetVec2) -> JetVec2:
        value = wp.cw_div(a, b.value)
        c = GradVec2()
        for i in range(2):
            inv_b = dtype(1.0) / b.value[i]
            for q in range(wp.static(width)):
                c[i, q] = -value[i] * b.grad[i, q] * inv_b
        return JetVec2(value, c)

    @wp.func
    def jet_cw_mul(a: JetVec3, b: JetVec3) -> JetVec3:
        c = GradVec3()
        for i in range(3):
            for q in range(wp.static(width)):
                c[i, q] = a.grad[i, q] * b.value[i] + a.value[i] * b.grad[i, q]
        return JetVec3(wp.cw_mul(a.value, b.value), c)

    @wp.func
    def jet_cw_mul(a: JetVec3, b: NativeVec3) -> JetVec3:
        c = GradVec3()
        for i in range(3):
            for q in range(wp.static(width)):
                c[i, q] = a.grad[i, q] * b[i]
        return JetVec3(wp.cw_mul(a.value, b), c)

    @wp.func
    def jet_cw_mul(a: NativeVec3, b: JetVec3) -> JetVec3:
        c = GradVec3()
        for i in range(3):
            for q in range(wp.static(width)):
                c[i, q] = a[i] * b.grad[i, q]
        return JetVec3(wp.cw_mul(a, b.value), c)

    @wp.func
    def jet_cw_div(a: JetVec3, b: JetVec3) -> JetVec3:
        value = wp.cw_div(a.value, b.value)
        c = GradVec3()
        for i in range(3):
            inv_b = dtype(1.0) / b.value[i]
            for q in range(wp.static(width)):
                c[i, q] = (a.grad[i, q] - value[i] * b.grad[i, q]) * inv_b
        return JetVec3(value, c)

    @wp.func
    def jet_cw_div(a: JetVec3, b: NativeVec3) -> JetVec3:
        c = GradVec3()
        for i in range(3):
            inv_b = dtype(1.0) / b[i]
            for q in range(wp.static(width)):
                c[i, q] = a.grad[i, q] * inv_b
        return JetVec3(wp.cw_div(a.value, b), c)

    @wp.func
    def jet_cw_div(a: NativeVec3, b: JetVec3) -> JetVec3:
        value = wp.cw_div(a, b.value)
        c = GradVec3()
        for i in range(3):
            inv_b = dtype(1.0) / b.value[i]
            for q in range(wp.static(width)):
                c[i, q] = -value[i] * b.grad[i, q] * inv_b
        return JetVec3(value, c)

    # ------------------------------------------------------------------
    # Length / normalization
    # ------------------------------------------------------------------

    @wp.func
    def jet_length_sq(v: JetVec2) -> JetScalar:
        return jet_dot(v, v)

    @wp.func
    def jet_length_sq(v: JetVec3) -> JetScalar:
        return jet_dot(v, v)

    @wp.func
    def jet_length(v: JetVec2) -> JetScalar:
        # Like native wp.length(), a zero-length vector has a zero gradient. The square
        # root's derivative is infinite there and would turn the zero gradient into NaN.
        # Only an exact zero is special: a NaN length stays NaN, as in native.
        length_sq = jet_dot(v, v)
        if length_sq.value == dtype(0.0):
            return JetScalar(dtype(0.0), Grad())
        return jet_sqrt(length_sq)

    @wp.func
    def jet_length(v: JetVec3) -> JetScalar:
        # Like native wp.length(), a zero-length vector has a zero gradient. The square
        # root's derivative is infinite there and would turn the zero gradient into NaN.
        # Only an exact zero is special: a NaN length stays NaN, as in native.
        length_sq = jet_dot(v, v)
        if length_sq.value == dtype(0.0):
            return JetScalar(dtype(0.0), Grad())
        return jet_sqrt(length_sq)

    @wp.func
    def jet_normalize(v: JetVec2) -> JetVec2:
        # Like the native normalize(), a zero-length vector normalizes to zero.
        length_sq = jet_dot(v, v)
        if length_sq.value > dtype(0.0):
            return jet_div(v, jet_sqrt(length_sq))
        return JetVec2(NativeVec2(), GradVec2())

    @wp.func
    def jet_normalize(v: JetVec3) -> JetVec3:
        # Like the native normalize(), a zero-length vector normalizes to zero.
        length_sq = jet_dot(v, v)
        if length_sq.value > dtype(0.0):
            return jet_div(v, jet_sqrt(length_sq))
        return JetVec3(NativeVec3(), GradVec3())

    # ------------------------------------------------------------------
    # 2D geometry
    #
    # Warp defines neither of these for vec2, so they stay on the namespace
    # instead of becoming builtin overloads.
    # ------------------------------------------------------------------

    @wp.func
    def jet_perp(v: JetVec2) -> JetVec2:
        c = GradVec2()
        for q in range(wp.static(width)):
            c[0, q] = -v.grad[1, q]
            c[1, q] = v.grad[0, q]
        return JetVec2(NativeVec2(-v.value[1], v.value[0]), c)

    @wp.func
    def jet_cross2(a: JetVec2, b: JetVec2) -> JetScalar:
        # Scalar 2D cross product: ax*by - ay*bx
        c = Grad()
        for q in range(wp.static(width)):
            c[q] = (
                a.grad[0, q] * b.value[1]
                + a.value[0] * b.grad[1, q]
                - a.grad[1, q] * b.value[0]
                - a.value[1] * b.grad[0, q]
            )
        return JetScalar(
            a.value[0] * b.value[1] - a.value[1] * b.value[0],
            c,
        )

    @wp.func
    def jet_cross2(a: JetVec2, b: NativeVec2) -> JetScalar:
        c = Grad()
        for q in range(wp.static(width)):
            c[q] = a.grad[0, q] * b[1] - a.grad[1, q] * b[0]
        return JetScalar(a.value[0] * b[1] - a.value[1] * b[0], c)

    @wp.func
    def jet_cross2(a: NativeVec2, b: JetVec2) -> JetScalar:
        c = Grad()
        for q in range(wp.static(width)):
            c[q] = a[0] * b.grad[1, q] - a[1] * b.grad[0, q]
        return JetScalar(a[0] * b.value[1] - a[1] * b.value[0], c)

    # ------------------------------------------------------------------
    # 3D cross product
    # ------------------------------------------------------------------

    @wp.func
    def jet_cross(a: JetVec3, b: JetVec3) -> JetVec3:
        c = GradVec3()

        ax = a.value[0]
        ay = a.value[1]
        az = a.value[2]

        bx = b.value[0]
        by = b.value[1]
        bz = b.value[2]

        for q in range(wp.static(width)):
            dax = a.grad[0, q]
            day = a.grad[1, q]
            daz = a.grad[2, q]

            dbx = b.grad[0, q]
            dby = b.grad[1, q]
            dbz = b.grad[2, q]

            # d(a x b) = da x b + a x db
            c[0, q] = day * bz - daz * by + ay * dbz - az * dby
            c[1, q] = daz * bx - dax * bz + az * dbx - ax * dbz
            c[2, q] = dax * by - day * bx + ax * dby - ay * dbx

        return JetVec3(wp.cross(a.value, b.value), c)

    @wp.func
    def jet_cross(a: JetVec3, b: NativeVec3) -> JetVec3:
        c = GradVec3()

        bx = b[0]
        by = b[1]
        bz = b[2]

        for q in range(wp.static(width)):
            dax = a.grad[0, q]
            day = a.grad[1, q]
            daz = a.grad[2, q]

            c[0, q] = day * bz - daz * by
            c[1, q] = daz * bx - dax * bz
            c[2, q] = dax * by - day * bx

        return JetVec3(wp.cross(a.value, b), c)

    @wp.func
    def jet_cross(a: NativeVec3, b: JetVec3) -> JetVec3:
        c = GradVec3()

        ax = a[0]
        ay = a[1]
        az = a[2]

        for q in range(wp.static(width)):
            dbx = b.grad[0, q]
            dby = b.grad[1, q]
            dbz = b.grad[2, q]

            c[0, q] = ay * dbz - az * dby
            c[1, q] = az * dbx - ax * dbz
            c[2, q] = ax * dby - ay * dbx

        return JetVec3(wp.cross(a, b.value), c)

    # ------------------------------------------------------------------
    # Square matrix jets (mat22 / mat33)
    #
    # ``grad`` flattens the matrix row-major: entry (r, c) is row r*n + c of a
    # (n*n, width) gradient block. Enough to write FEM energies with a native
    # deformation gradient -- construction, transpose, matmul, matrix-vector
    # product, determinant, and trace.
    # ------------------------------------------------------------------

    NativeMat22 = matrix((2, 2), dtype)
    NativeMat33 = matrix((3, 3), dtype)
    NativeMat32 = matrix((3, 2), dtype)
    NativeMat23 = matrix((2, 3), dtype)
    GradMat4 = matrix((4, width), dtype)
    GradMat9 = matrix((9, width), dtype)
    GradMat6 = matrix((6, width), dtype)  # a 3x2 or 2x3 block, row-major

    @wp.struct
    class JetMat22:
        value: NativeMat22
        grad: GradMat4

    @wp.struct
    class JetMat33:
        value: NativeMat33
        grad: GradMat9

    @wp.struct
    class JetMat32:  # 3x2, e.g. a triangle deformation gradient
        value: NativeMat32
        grad: GradMat6

    @wp.struct
    class JetMat23:  # 2x3
        value: NativeMat23
        grad: GradMat6

    @wp.func
    def mat22_from_cols(c0: JetVec2, c1: JetVec2) -> JetMat22:
        v = NativeMat22()
        m = GradMat4()
        for r in range(2):
            v[r, 0] = c0.value[r]
            v[r, 1] = c1.value[r]
            for q in range(wp.static(width)):
                m[r * 2 + 0, q] = c0.grad[r, q]
                m[r * 2 + 1, q] = c1.grad[r, q]
        return JetMat22(v, m)

    @wp.func
    def mat33_from_cols(c0: JetVec3, c1: JetVec3, c2: JetVec3) -> JetMat33:
        v = NativeMat33()
        m = GradMat9()
        for r in range(3):
            v[r, 0] = c0.value[r]
            v[r, 1] = c1.value[r]
            v[r, 2] = c2.value[r]
            for q in range(wp.static(width)):
                m[r * 3 + 0, q] = c0.grad[r, q]
                m[r * 3 + 1, q] = c1.grad[r, q]
                m[r * 3 + 2, q] = c2.grad[r, q]
        return JetMat33(v, m)

    @wp.func
    def mat32_from_cols(c0: JetVec3, c1: JetVec3) -> JetMat32:
        v = NativeMat32()
        m = GradMat6()
        for r in range(3):
            v[r, 0] = c0.value[r]
            v[r, 1] = c1.value[r]
            for q in range(wp.static(width)):
                m[r * 2 + 0, q] = c0.grad[r, q]
                m[r * 2 + 1, q] = c1.grad[r, q]
        return JetMat32(v, m)

    @wp.func
    def jet_transpose(a: JetMat32) -> JetMat23:
        m = GradMat6()
        for r in range(3):
            for c in range(2):
                for q in range(wp.static(width)):
                    m[c * 3 + r, q] = a.grad[r * 2 + c, q]
        return JetMat23(wp.transpose(a.value), m)

    @wp.func
    def jet_transpose(a: JetMat23) -> JetMat32:
        m = GradMat6()
        for r in range(2):
            for c in range(3):
                for q in range(wp.static(width)):
                    m[c * 2 + r, q] = a.grad[r * 3 + c, q]
        return JetMat32(wp.transpose(a.value), m)

    @wp.func
    def jet_mul(a: JetMat23, b: JetMat32) -> JetMat22:
        # (2x3)(3x2) = 2x2, product rule over the shared dimension.
        m = GradMat4()
        for i in range(2):
            for j in range(2):
                for q in range(wp.static(width)):
                    acc = dtype(0.0)
                    for k in range(3):
                        acc += a.grad[i * 3 + k, q] * b.value[k, j] + a.value[i, k] * b.grad[k * 2 + j, q]
                    m[i * 2 + j, q] = acc
        return JetMat22(a.value * b.value, m)

    @wp.func
    def jet_mul(a: JetMat23, b: NativeMat32) -> JetMat22:
        m = GradMat4()
        for i in range(2):
            for j in range(2):
                for q in range(wp.static(width)):
                    acc = dtype(0.0)
                    for k in range(3):
                        acc += a.grad[i * 3 + k, q] * b[k, j]
                    m[i * 2 + j, q] = acc
        return JetMat22(a.value * b, m)

    @wp.func
    def jet_mul(a: NativeMat23, b: JetMat32) -> JetMat22:
        m = GradMat4()
        for i in range(2):
            for j in range(2):
                for q in range(wp.static(width)):
                    acc = dtype(0.0)
                    for k in range(3):
                        acc += a[i, k] * b.grad[k * 2 + j, q]
                    m[i * 2 + j, q] = acc
        return JetMat22(a * b.value, m)

    @wp.func
    def jet_add(a: JetMat22, b: JetMat22) -> JetMat22:
        return JetMat22(a.value + b.value, a.grad + b.grad)

    @wp.func
    def jet_add(a: JetMat22, b: NativeMat22) -> JetMat22:
        return JetMat22(a.value + b, a.grad)

    @wp.func
    def jet_add(a: NativeMat22, b: JetMat22) -> JetMat22:
        return JetMat22(a + b.value, b.grad)

    @wp.func
    def jet_add(a: JetMat33, b: JetMat33) -> JetMat33:
        return JetMat33(a.value + b.value, a.grad + b.grad)

    @wp.func
    def jet_add(a: JetMat33, b: NativeMat33) -> JetMat33:
        return JetMat33(a.value + b, a.grad)

    @wp.func
    def jet_add(a: NativeMat33, b: JetMat33) -> JetMat33:
        return JetMat33(a + b.value, b.grad)

    @wp.func
    def jet_sub(a: JetMat22, b: JetMat22) -> JetMat22:
        return JetMat22(a.value - b.value, a.grad - b.grad)

    @wp.func
    def jet_sub(a: JetMat22, b: NativeMat22) -> JetMat22:
        return JetMat22(a.value - b, a.grad)

    @wp.func
    def jet_sub(a: NativeMat22, b: JetMat22) -> JetMat22:
        return JetMat22(a - b.value, -b.grad)

    @wp.func
    def jet_sub(a: JetMat33, b: JetMat33) -> JetMat33:
        return JetMat33(a.value - b.value, a.grad - b.grad)

    @wp.func
    def jet_sub(a: JetMat33, b: NativeMat33) -> JetMat33:
        return JetMat33(a.value - b, a.grad)

    @wp.func
    def jet_sub(a: NativeMat33, b: JetMat33) -> JetMat33:
        return JetMat33(a - b.value, -b.grad)

    @wp.func
    def jet_mul(a: JetMat22, s: dtype) -> JetMat22:
        return JetMat22(a.value * s, a.grad * s)

    @wp.func
    def jet_mul(s: dtype, a: JetMat22) -> JetMat22:
        return JetMat22(s * a.value, s * a.grad)

    @wp.func
    def jet_mul(a: JetMat33, s: dtype) -> JetMat33:
        return JetMat33(a.value * s, a.grad * s)

    @wp.func
    def jet_mul(s: dtype, a: JetMat33) -> JetMat33:
        return JetMat33(s * a.value, s * a.grad)

    @wp.func
    def jet_transpose(a: JetMat33) -> JetMat33:
        m = GradMat9()
        for r in range(3):
            for c in range(3):
                for q in range(wp.static(width)):
                    m[c * 3 + r, q] = a.grad[r * 3 + c, q]
        return JetMat33(wp.transpose(a.value), m)

    @wp.func
    def jet_transpose(a: JetMat22) -> JetMat22:
        m = GradMat4()
        for r in range(2):
            for c in range(2):
                for q in range(wp.static(width)):
                    m[c * 2 + r, q] = a.grad[r * 2 + c, q]
        return JetMat22(wp.transpose(a.value), m)

    @wp.func
    def jet_mul(a: JetMat33, b: JetMat33) -> JetMat33:
        m = GradMat9()
        for i in range(3):
            for j in range(3):
                for q in range(wp.static(width)):
                    acc = dtype(0.0)
                    for k in range(3):
                        acc += a.grad[i * 3 + k, q] * b.value[k, j] + a.value[i, k] * b.grad[k * 3 + j, q]
                    m[i * 3 + j, q] = acc
        return JetMat33(a.value * b.value, m)

    @wp.func
    def jet_mul(a: JetMat22, b: JetMat22) -> JetMat22:
        m = GradMat4()
        for i in range(2):
            for j in range(2):
                for q in range(wp.static(width)):
                    acc = dtype(0.0)
                    for k in range(2):
                        acc += a.grad[i * 2 + k, q] * b.value[k, j] + a.value[i, k] * b.grad[k * 2 + j, q]
                    m[i * 2 + j, q] = acc
        return JetMat22(a.value * b.value, m)

    @wp.func
    def jet_mul(a: JetMat33, b: NativeMat33) -> JetMat33:
        m = GradMat9()
        for i in range(3):
            for j in range(3):
                for q in range(wp.static(width)):
                    acc = dtype(0.0)
                    for k in range(3):
                        acc += a.grad[i * 3 + k, q] * b[k, j]
                    m[i * 3 + j, q] = acc
        return JetMat33(a.value * b, m)

    @wp.func
    def jet_mul(a: NativeMat33, b: JetMat33) -> JetMat33:
        m = GradMat9()
        for i in range(3):
            for j in range(3):
                for q in range(wp.static(width)):
                    acc = dtype(0.0)
                    for k in range(3):
                        acc += a[i, k] * b.grad[k * 3 + j, q]
                    m[i * 3 + j, q] = acc
        return JetMat33(a * b.value, m)

    @wp.func
    def jet_mul(a: JetMat22, b: NativeMat22) -> JetMat22:
        m = GradMat4()
        for i in range(2):
            for j in range(2):
                for q in range(wp.static(width)):
                    acc = dtype(0.0)
                    for k in range(2):
                        acc += a.grad[i * 2 + k, q] * b[k, j]
                    m[i * 2 + j, q] = acc
        return JetMat22(a.value * b, m)

    @wp.func
    def jet_mul(a: NativeMat22, b: JetMat22) -> JetMat22:
        m = GradMat4()
        for i in range(2):
            for j in range(2):
                for q in range(wp.static(width)):
                    acc = dtype(0.0)
                    for k in range(2):
                        acc += a[i, k] * b.grad[k * 2 + j, q]
                    m[i * 2 + j, q] = acc
        return JetMat22(a * b.value, m)

    @wp.func
    def jet_mul(a: JetMat22, v: JetVec2) -> JetVec2:
        c = GradVec2()
        for i in range(2):
            for q in range(wp.static(width)):
                acc = dtype(0.0)
                for k in range(2):
                    acc += a.grad[i * 2 + k, q] * v.value[k] + a.value[i, k] * v.grad[k, q]
                c[i, q] = acc
        return JetVec2(a.value * v.value, c)

    @wp.func
    def jet_mul(a: JetMat33, v: JetVec3) -> JetVec3:
        c = GradVec3()
        for i in range(3):
            for q in range(wp.static(width)):
                acc = dtype(0.0)
                for k in range(3):
                    acc += a.grad[i * 3 + k, q] * v.value[k] + a.value[i, k] * v.grad[k, q]
                c[i, q] = acc
        return JetVec3(a.value * v.value, c)

    @wp.func
    def jet_mul(a: JetMat22, v: NativeVec2) -> JetVec2:
        c = GradVec2()
        for i in range(2):
            for q in range(wp.static(width)):
                acc = dtype(0.0)
                for k in range(2):
                    acc += a.grad[i * 2 + k, q] * v[k]
                c[i, q] = acc
        return JetVec2(a.value * v, c)

    @wp.func
    def jet_mul(a: NativeMat22, v: JetVec2) -> JetVec2:
        c = GradVec2()
        for i in range(2):
            for q in range(wp.static(width)):
                acc = dtype(0.0)
                for k in range(2):
                    acc += a[i, k] * v.grad[k, q]
                c[i, q] = acc
        return JetVec2(a * v.value, c)

    @wp.func
    def jet_mul(a: JetMat33, v: NativeVec3) -> JetVec3:
        c = GradVec3()
        for i in range(3):
            for q in range(wp.static(width)):
                acc = dtype(0.0)
                for k in range(3):
                    acc += a.grad[i * 3 + k, q] * v[k]
                c[i, q] = acc
        return JetVec3(a.value * v, c)

    @wp.func
    def jet_mul(a: NativeMat33, v: JetVec3) -> JetVec3:
        c = GradVec3()
        for i in range(3):
            for q in range(wp.static(width)):
                acc = dtype(0.0)
                for k in range(3):
                    acc += a[i, k] * v.grad[k, q]
                c[i, q] = acc
        return JetVec3(a * v.value, c)

    @wp.func
    def jet_trace(a: JetMat33) -> JetScalar:
        c = Grad()
        for q in range(wp.static(width)):
            c[q] = a.grad[0, q] + a.grad[4, q] + a.grad[8, q]
        return JetScalar(a.value[0, 0] + a.value[1, 1] + a.value[2, 2], c)

    @wp.func
    def jet_trace(a: JetMat22) -> JetScalar:
        c = Grad()
        for q in range(wp.static(width)):
            c[q] = a.grad[0, q] + a.grad[3, q]
        return JetScalar(a.value[0, 0] + a.value[1, 1], c)

    @wp.func
    def jet_determinant(a: JetMat33) -> JetScalar:
        # d(det A) = sum_ij cofactor(i,j) * dA[i,j]  (Jacobi's formula).
        m = a.value
        cof0 = m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]
        cof1 = -(m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
        cof2 = m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]
        cof3 = -(m[0, 1] * m[2, 2] - m[0, 2] * m[2, 1])
        cof4 = m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]
        cof5 = -(m[0, 0] * m[2, 1] - m[0, 1] * m[2, 0])
        cof6 = m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]
        cof7 = -(m[0, 0] * m[1, 2] - m[0, 2] * m[1, 0])
        cof8 = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
        c = Grad()
        for q in range(wp.static(width)):
            c[q] = (
                cof0 * a.grad[0, q]
                + cof1 * a.grad[1, q]
                + cof2 * a.grad[2, q]
                + cof3 * a.grad[3, q]
                + cof4 * a.grad[4, q]
                + cof5 * a.grad[5, q]
                + cof6 * a.grad[6, q]
                + cof7 * a.grad[7, q]
                + cof8 * a.grad[8, q]
            )
        return JetScalar(wp.determinant(m), c)

    @wp.func
    def jet_determinant(a: JetMat22) -> JetScalar:
        m = a.value
        c = Grad()
        for q in range(wp.static(width)):
            c[q] = m[1, 1] * a.grad[0, q] - m[1, 0] * a.grad[1, q] - m[0, 1] * a.grad[2, q] + m[0, 0] * a.grad[3, q]
        return JetScalar(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0], c)

    @wp.func
    def jet_inverse(a: JetMat33) -> JetMat33:
        # d(A^-1) = -A^-1 (dA) A^-1.
        ainv = wp.inverse(a.value)
        m = GradMat9()
        for q in range(wp.static(width)):
            da = NativeMat33()
            for r in range(3):
                for c in range(3):
                    da[r, c] = a.grad[r * 3 + c, q]
            dinv = -(ainv * da * ainv)
            for r in range(3):
                for c in range(3):
                    m[r * 3 + c, q] = dinv[r, c]
        return JetMat33(ainv, m)

    @wp.func
    def jet_inverse(a: JetMat22) -> JetMat22:
        ainv = wp.inverse(a.value)
        m = GradMat4()
        for q in range(wp.static(width)):
            da = NativeMat22()
            for r in range(2):
                for c in range(2):
                    da[r, c] = a.grad[r * 2 + c, q]
            dinv = -(ainv * da * ainv)
            for r in range(2):
                for c in range(2):
                    m[r * 2 + c, q] = dinv[r, c]
        return JetMat22(ainv, m)

    # ------------------------------------------------------------------
    # More scalar functions: chain rule with f'.
    # ------------------------------------------------------------------

    @wp.func
    def jet_sinh(a: JetScalar) -> JetScalar:
        return _lift1(wp.sinh(a.value), wp.cosh(a.value), a)

    @wp.func
    def jet_cosh(a: JetScalar) -> JetScalar:
        return _lift1(wp.cosh(a.value), wp.sinh(a.value), a)

    @wp.func
    def jet_tanh(a: JetScalar) -> JetScalar:
        t = wp.tanh(a.value)
        return _lift1(t, dtype(1.0) - t * t, a)

    @wp.func
    def jet_log2(a: JetScalar) -> JetScalar:
        return _lift1(wp.log2(a.value), dtype(1.0) / (a.value * dtype(0.6931471805599453)), a)

    @wp.func
    def jet_log10(a: JetScalar) -> JetScalar:
        return _lift1(wp.log10(a.value), dtype(1.0) / (a.value * dtype(2.302585092994046)), a)

    @wp.func
    def jet_cbrt(a: JetScalar) -> JetScalar:
        # Infinite slope at zero, as for sqrt.
        c = wp.cbrt(a.value)
        return _lift1(c, dtype(1.0) / (dtype(3.0) * c * c), a)

    @wp.func
    def jet_erf(a: JetScalar) -> JetScalar:
        # erf'(x) = 2 / sqrt(pi) * exp(-x^2)
        return _lift1(wp.erf(a.value), dtype(1.1283791670955126) * wp.exp(-a.value * a.value), a)

    @wp.func
    def jet_floor(a: JetScalar) -> JetScalar:
        # Piecewise constant: zero derivative away from the integers.
        return JetScalar(wp.floor(a.value), Grad())

    @wp.func
    def jet_lerp(a: JetScalar, b: JetScalar, t: JetScalar) -> JetScalar:
        return a + t * (b - a)

    @wp.func
    def jet_lerp(a: JetScalar, b: JetScalar, t: dtype) -> JetScalar:
        return a + t * (b - a)

    @wp.func
    def jet_lerp(a: dtype, b: dtype, t: JetScalar) -> JetScalar:
        return a + t * (b - a)

    @wp.func
    def jet_smoothstep(a: dtype, b: dtype, x: JetScalar) -> JetScalar:
        # Constant outside [a, b]; t^2 (3 - 2t) inside, as wp.smoothstep.
        t = (x - a) / (b - a)
        if t.value <= dtype(0.0):
            return JetScalar(dtype(0.0), Grad())
        if t.value >= dtype(1.0):
            return JetScalar(dtype(1.0), Grad())
        return t * t * (dtype(3.0) - dtype(2.0) * t)

    @wp.func
    def jet_smoothstep(a: JetScalar, b: JetScalar, x: JetScalar) -> JetScalar:
        t = (x - a) / (b - a)
        if t.value <= dtype(0.0):
            return JetScalar(dtype(0.0), Grad())
        if t.value >= dtype(1.0):
            return JetScalar(dtype(1.0), Grad())
        return t * t * (dtype(3.0) - dtype(2.0) * t)

    # ------------------------------------------------------------------
    # Vector reductions, component assignment, and outer products
    # ------------------------------------------------------------------

    @wp.func
    def jet_abs(v: JetVec2) -> JetVec2:
        return vec2_from_scalars(jet_abs(jet_extract(v, 0)), jet_abs(jet_extract(v, 1)))

    @wp.func
    def jet_abs(v: JetVec3) -> JetVec3:
        return vec3_from_scalars(jet_abs(jet_extract(v, 0)), jet_abs(jet_extract(v, 1)), jet_abs(jet_extract(v, 2)))

    @wp.func
    def jet_min(v: JetVec2) -> JetScalar:
        # The smallest component, carrying that component's derivatives.
        return jet_extract(v, int(wp.argmin(v.value)))

    @wp.func
    def jet_min(v: JetVec3) -> JetScalar:
        return jet_extract(v, int(wp.argmin(v.value)))

    @wp.func
    def jet_max(v: JetVec2) -> JetScalar:
        return jet_extract(v, int(wp.argmax(v.value)))

    @wp.func
    def jet_max(v: JetVec3) -> JetScalar:
        return jet_extract(v, int(wp.argmax(v.value)))

    @wp.func
    def jet_assign_copy(v: JetVec2, i: int, x: JetScalar) -> JetVec2:
        # Built entry by entry into new values: writing into a copy of v's
        # fields would alias v and break the reverse pass.
        i = _component_index(i, 2)
        value = NativeVec2()
        c = GradVec2()
        for k in range(2):
            if k == i:
                value[k] = x.value
                for q in range(wp.static(width)):
                    c[k, q] = x.grad[q]
            else:
                value[k] = v.value[k]
                for q in range(wp.static(width)):
                    c[k, q] = v.grad[k, q]
        return JetVec2(value, c)

    @wp.func
    def jet_assign_copy(v: JetVec3, i: int, x: JetScalar) -> JetVec3:
        # Built entry by entry into new values: writing into a copy of v's
        # fields would alias v and break the reverse pass.
        i = _component_index(i, 3)
        value = NativeVec3()
        c = GradVec3()
        for k in range(3):
            if k == i:
                value[k] = x.value
                for q in range(wp.static(width)):
                    c[k, q] = x.grad[q]
            else:
                value[k] = v.value[k]
                for q in range(wp.static(width)):
                    c[k, q] = v.grad[k, q]
        return JetVec3(value, c)

    @wp.func
    def jet_assign_copy(v: JetVec2, i: int, x: dtype) -> JetVec2:
        return jet_assign_copy(v, i, scalar_constant(x))

    @wp.func
    def jet_assign_copy(v: JetVec3, i: int, x: dtype) -> JetVec3:
        return jet_assign_copy(v, i, scalar_constant(x))

    @wp.func
    def jet_skew(v: JetVec3) -> JetMat33:
        # [v]_x, the cross-product matrix: entries are signed copies of v's components.
        value = wp.skew(v.value)
        m = GradMat9()
        for q in range(wp.static(width)):
            m[1, q] = -v.grad[2, q]
            m[2, q] = v.grad[1, q]
            m[3, q] = v.grad[2, q]
            m[5, q] = -v.grad[0, q]
            m[6, q] = -v.grad[1, q]
            m[7, q] = v.grad[0, q]
        return JetMat33(value, m)

    # Vector jets by length, matrix jets by shape: (jet type, grad type, plain type).
    vec_types = {2: (JetVec2, GradVec2, NativeVec2), 3: (JetVec3, GradVec3, NativeVec3)}
    mat_types = {
        (2, 2): (JetMat22, GradMat4, NativeMat22),
        (3, 3): (JetMat33, GradMat9, NativeMat33),
        (3, 2): (JetMat32, GradMat6, NativeMat32),
        (2, 3): (JetMat23, GradMat6, NativeMat23),
    }

    def _matrix_ops(rows: int, cols: int):
        JM, CM, NM = mat_types[(rows, cols)]
        JRow, CRow, NRow = vec_types[cols]
        JCol, CCol, NCol = vec_types[rows]

        @wp.func
        def jet_neg(a: JM) -> JM:
            return JM(-a.value, -a.grad)

        @wp.func
        def jet_pos(a: JM) -> JM:
            return a

        @wp.func
        def jet_div(a: JM, s: dtype) -> JM:
            return JM(a.value / s, a.grad / s)

        @wp.func
        def jet_mul(s: JetScalar, a: JM) -> JM:
            m = CM()
            for r in range(rows):
                for c in range(cols):
                    for q in range(wp.static(width)):
                        m[r * cols + c, q] = s.value * a.grad[r * cols + c, q] + a.value[r, c] * s.grad[q]
            return JM(s.value * a.value, m)

        @wp.func
        def jet_mul(a: JM, s: JetScalar) -> JM:
            return jet_mul(s, a)

        @wp.func
        def jet_mul(s: JetScalar, a: NM) -> JM:
            m = CM()
            for r in range(rows):
                for c in range(cols):
                    for q in range(wp.static(width)):
                        m[r * cols + c, q] = a[r, c] * s.grad[q]
            return JM(s.value * a, m)

        @wp.func
        def jet_mul(a: NM, s: JetScalar) -> JM:
            return jet_mul(s, a)

        @wp.func
        def jet_div(a: JM, s: JetScalar) -> JM:
            return jet_mul(dtype(1.0) / s, a)

        @wp.func
        def jet_div(a: NM, s: JetScalar) -> JM:
            return jet_mul(dtype(1.0) / s, a)

        @wp.func
        def jet_ddot(a: JM, b: JM) -> JetScalar:
            g = Grad()
            for r in range(rows):
                for c in range(cols):
                    for q in range(wp.static(width)):
                        g[q] += a.grad[r * cols + c, q] * b.value[r, c] + a.value[r, c] * b.grad[r * cols + c, q]
            return JetScalar(wp.ddot(a.value, b.value), g)

        @wp.func
        def jet_ddot(a: JM, b: NM) -> JetScalar:
            g = Grad()
            for r in range(rows):
                for c in range(cols):
                    for q in range(wp.static(width)):
                        g[q] += a.grad[r * cols + c, q] * b[r, c]
            return JetScalar(wp.ddot(a.value, b), g)

        @wp.func
        def jet_ddot(a: NM, b: JM) -> JetScalar:
            return jet_ddot(b, a)

        @wp.func
        def jet_cw_mul(a: JM, b: JM) -> JM:
            m = CM()
            for r in range(rows):
                for c in range(cols):
                    for q in range(wp.static(width)):
                        m[r * cols + c, q] = (
                            a.grad[r * cols + c, q] * b.value[r, c] + a.value[r, c] * b.grad[r * cols + c, q]
                        )
            return JM(wp.cw_mul(a.value, b.value), m)

        @wp.func
        def jet_cw_mul(a: JM, b: NM) -> JM:
            m = CM()
            for r in range(rows):
                for c in range(cols):
                    for q in range(wp.static(width)):
                        m[r * cols + c, q] = a.grad[r * cols + c, q] * b[r, c]
            return JM(wp.cw_mul(a.value, b), m)

        @wp.func
        def jet_cw_mul(a: NM, b: JM) -> JM:
            return jet_cw_mul(b, a)

        @wp.func
        def jet_cw_div(a: JM, b: JM) -> JM:
            value = wp.cw_div(a.value, b.value)
            m = CM()
            for r in range(rows):
                for c in range(cols):
                    for q in range(wp.static(width)):
                        m[r * cols + c, q] = (
                            a.grad[r * cols + c, q] - value[r, c] * b.grad[r * cols + c, q]
                        ) / b.value[r, c]
            return JM(value, m)

        @wp.func
        def jet_cw_div(a: JM, b: NM) -> JM:
            m = CM()
            for r in range(rows):
                for c in range(cols):
                    for q in range(wp.static(width)):
                        m[r * cols + c, q] = a.grad[r * cols + c, q] / b[r, c]
            return JM(wp.cw_div(a.value, b), m)

        @wp.func
        def jet_cw_div(a: NM, b: JM) -> JM:
            value = wp.cw_div(a, b.value)
            m = CM()
            for r in range(rows):
                for c in range(cols):
                    for q in range(wp.static(width)):
                        m[r * cols + c, q] = -value[r, c] * b.grad[r * cols + c, q] / b.value[r, c]
            return JM(value, m)

        @wp.func
        def jet_extract(a: JM, i: int, j: int) -> JetScalar:
            i = _component_index(i, rows)
            j = _component_index(j, cols)
            g = Grad()
            for q in range(wp.static(width)):
                g[q] = a.grad[i * cols + j, q]
            return JetScalar(a.value[i, j], g)

        @wp.func
        def jet_extract(a: JM, i: int) -> JRow:
            i = _component_index(i, rows)
            m = CRow()
            for c in range(cols):
                for q in range(wp.static(width)):
                    m[c, q] = a.grad[i * cols + c, q]
            return JRow(a.value[i], m)

        @wp.func
        def jet_assign_copy(a: JM, i: int, j: int, x: JetScalar) -> JM:
            # Built entry by entry into new values, as for vectors above.
            i = _component_index(i, rows)
            j = _component_index(j, cols)
            value = NM()
            m = CM()
            for r in range(rows):
                for c in range(cols):
                    if r == i and c == j:
                        value[r, c] = x.value
                        for q in range(wp.static(width)):
                            m[r * cols + c, q] = x.grad[q]
                    else:
                        value[r, c] = a.value[r, c]
                        for q in range(wp.static(width)):
                            m[r * cols + c, q] = a.grad[r * cols + c, q]
            return JM(value, m)

        @wp.func
        def jet_assign_copy(a: JM, i: int, j: int, x: dtype) -> JM:
            return jet_assign_copy(a, i, j, scalar_constant(x))

        @wp.func
        def jet_assign_copy(a: JM, i: int, row: JRow) -> JM:
            i = _component_index(i, rows)
            value = NM()
            m = CM()
            for r in range(rows):
                for c in range(cols):
                    if r == i:
                        value[r, c] = row.value[c]
                        for q in range(wp.static(width)):
                            m[r * cols + c, q] = row.grad[c, q]
                    else:
                        value[r, c] = a.value[r, c]
                        for q in range(wp.static(width)):
                            m[r * cols + c, q] = a.grad[r * cols + c, q]
            return JM(value, m)

        @wp.func
        def jet_outer(a: JCol, b: JRow) -> JM:
            m = CM()
            for r in range(rows):
                for c in range(cols):
                    for q in range(wp.static(width)):
                        m[r * cols + c, q] = a.grad[r, q] * b.value[c] + a.value[r] * b.grad[c, q]
            return JM(wp.outer(a.value, b.value), m)

        @wp.func
        def jet_outer(a: JCol, b: NRow) -> JM:
            m = CM()
            for r in range(rows):
                for c in range(cols):
                    for q in range(wp.static(width)):
                        m[r * cols + c, q] = a.grad[r, q] * b[c]
            return JM(wp.outer(a.value, b), m)

        @wp.func
        def jet_outer(a: NCol, b: JRow) -> JM:
            m = CM()
            for r in range(rows):
                for c in range(cols):
                    for q in range(wp.static(width)):
                        m[r * cols + c, q] = a[r] * b.grad[c, q]
            return JM(wp.outer(a, b.value), m)

        @wp.func
        def jet_mul(v: JCol, a: JM) -> JRow:
            # Row vector times matrix: (v^T A)_c = sum_r v_r A_rc.
            m = CRow()
            for c in range(cols):
                for q in range(wp.static(width)):
                    acc = dtype(0.0)
                    for r in range(rows):
                        acc += v.grad[r, q] * a.value[r, c] + v.value[r] * a.grad[r * cols + c, q]
                    m[c, q] = acc
            return JRow(v.value * a.value, m)

        @wp.func
        def jet_mul(v: JCol, a: NM) -> JRow:
            m = CRow()
            for c in range(cols):
                for q in range(wp.static(width)):
                    acc = dtype(0.0)
                    for r in range(rows):
                        acc += v.grad[r, q] * a[r, c]
                    m[c, q] = acc
            return JRow(v.value * a, m)

        @wp.func
        def jet_mul(v: NCol, a: JM) -> JRow:
            m = CRow()
            for c in range(cols):
                for q in range(wp.static(width)):
                    acc = dtype(0.0)
                    for r in range(rows):
                        acc += v[r] * a.grad[r * cols + c, q]
                    m[c, q] = acc
            return JRow(v * a.value, m)

        defined = {
            "neg": jet_neg,
            "pos": jet_pos,
            "mul": jet_mul,
            "div": jet_div,
            "ddot": jet_ddot,
            "cw_mul": jet_cw_mul,
            "cw_div": jet_cw_div,
            "extract": jet_extract,
            "assign_copy": jet_assign_copy,
            "outer": jet_outer,
        }

        # Square and vector-length-matched overloads that only some shapes take.
        if rows == cols:

            @wp.func
            def jet_get_diag(a: JM) -> JCol:
                m = CCol()
                for r in range(rows):
                    for q in range(wp.static(width)):
                        m[r, q] = a.grad[r * cols + r, q]
                return JCol(wp.get_diag(a.value), m)

            @wp.func
            def jet_diag(v: JCol) -> JM:
                m = CM()
                for r in range(rows):
                    for q in range(wp.static(width)):
                        m[r * cols + r, q] = v.grad[r, q]
                return JM(wp.diag(v.value), m)

            defined["get_diag"] = jet_get_diag
            defined["diag"] = jet_diag

        else:
            # mat22 and mat33 already have these above.

            @wp.func
            def jet_add(a: JM, b: JM) -> JM:
                return JM(a.value + b.value, a.grad + b.grad)

            @wp.func
            def jet_add(a: JM, b: NM) -> JM:
                return JM(a.value + b, a.grad)

            @wp.func
            def jet_add(a: NM, b: JM) -> JM:
                return JM(a + b.value, b.grad)

            @wp.func
            def jet_sub(a: JM, b: JM) -> JM:
                return JM(a.value - b.value, a.grad - b.grad)

            @wp.func
            def jet_sub(a: JM, b: NM) -> JM:
                return JM(a.value - b, a.grad)

            @wp.func
            def jet_sub(a: NM, b: JM) -> JM:
                return JM(a - b.value, -b.grad)

            @wp.func
            def jet_mul(a: JM, s: dtype) -> JM:
                return JM(a.value * s, a.grad * s)

            @wp.func
            def jet_mul(s: dtype, a: JM) -> JM:
                return JM(s * a.value, s * a.grad)

            defined["add"] = jet_add
            defined["sub"] = jet_sub
            defined["mul"] = jet_mul

        # Each call defines a separate overload set, so register it per shape.
        for name, fn in defined.items():
            _register(name, fn)

    for shape in mat_types:
        _matrix_ops(*shape)

    @wp.func
    def jet_mul(a: JetMat32, b: JetMat23) -> JetMat33:
        # (3x2)(2x3) = 3x3, e.g. A^T A for a 2x3 A.
        m = GradMat9()
        for i in range(3):
            for j in range(3):
                for q in range(wp.static(width)):
                    acc = dtype(0.0)
                    for k in range(2):
                        acc += a.grad[i * 2 + k, q] * b.value[k, j] + a.value[i, k] * b.grad[k * 3 + j, q]
                    m[i * 3 + j, q] = acc
        return JetMat33(a.value * b.value, m)

    @wp.func
    def jet_mul(a: JetMat32, b: NativeMat23) -> JetMat33:
        m = GradMat9()
        for i in range(3):
            for j in range(3):
                for q in range(wp.static(width)):
                    acc = dtype(0.0)
                    for k in range(2):
                        acc += a.grad[i * 2 + k, q] * b[k, j]
                    m[i * 3 + j, q] = acc
        return JetMat33(a.value * b, m)

    @wp.func
    def jet_mul(a: NativeMat32, b: JetMat23) -> JetMat33:
        m = GradMat9()
        for i in range(3):
            for j in range(3):
                for q in range(wp.static(width)):
                    acc = dtype(0.0)
                    for k in range(2):
                        acc += a[i, k] * b.grad[k * 3 + j, q]
                    m[i * 3 + j, q] = acc
        return JetMat33(a * b.value, m)

    # ------------------------------------------------------------------
    # Decompositions. The factors come from the float decomposition of the
    # value; their derivatives use the analytic rules that Warp's reverse
    # mode uses (the transpose of adj_svd3, adj_svd2 and adj_eig3), with the
    # same clamping of nearly repeated singular values or eigenvalues, rather
    # than differentiating the iterations.
    # ------------------------------------------------------------------

    svd_epsilon = 1.0e-12 if dtype == wp.float64 else 1.0e-6

    @wp.func
    def jet_svd3(a: JetMat33):
        U, sigma, V = wp.svd3(a.value)
        eps = dtype(svd_epsilon)
        sx2 = sigma[0] * sigma[0]
        sy2 = sigma[1] * sigma[1]
        sz2 = sigma[2] * sigma[2]
        f01 = dtype(1.0) / wp.min(sy2 - sx2, -eps)
        f02 = dtype(1.0) / wp.min(sz2 - sx2, -eps)
        f12 = dtype(1.0) / wp.min(sz2 - sy2, -eps)
        zero = dtype(0.0)
        F = NativeMat33(zero, f01, f02, -f01, zero, f12, -f02, -f12, zero)
        S = wp.diag(sigma)
        UT = wp.transpose(U)
        dU = GradMat9()
        dsigma = GradVec3()
        dV = GradMat9()
        for q in range(wp.static(width)):
            dA = NativeMat33()
            for r in range(3):
                for c in range(3):
                    dA[r, c] = a.grad[r * 3 + c, q]
            # P = U^T dA V: dsigma = diag(P), dU = U W_u, dV = V W_v.
            P = UT * dA * V
            Yu = wp.cw_mul(F, P * S)
            Yv = wp.cw_mul(F, S * P)
            dUq = U * (Yu - wp.transpose(Yu))
            dVq = V * (Yv - wp.transpose(Yv))
            for r in range(3):
                dsigma[r, q] = P[r, r]
                for c in range(3):
                    dU[r * 3 + c, q] = dUq[r, c]
                    dV[r * 3 + c, q] = dVq[r, c]
        return JetMat33(U, dU), JetVec3(sigma, dsigma), JetMat33(V, dV)

    @wp.func
    def jet_svd2(a: JetMat22):
        # The transpose of native adj_svd2(). With P = U^T dA V, the singular vectors turn
        # as dU = U [[0, alpha], [-alpha, 0]] and dV = V [[0, beta], [-beta, 0]], where
        # alpha and beta combine a skew term (P01 - P10) / (s0 + s1), from rotating A, and a
        # symmetric term (P01 + P10) / (s0 - s1). Like adj_svd2(), keep the skew term, which
        # is well defined unless A = 0, and drop only the symmetric one at nearly repeated
        # singular values, so that derivatives of U V^T stay exact there.
        U, sigma, V = wp.svd2(a.value)
        eps = dtype(svd_epsilon)
        s_sum = sigma[0] + sigma[1]
        s_diff = sigma[0] - sigma[1]
        UT = wp.transpose(U)
        dU = GradMat4()
        dsigma = GradVec2()
        dV = GradMat4()
        for q in range(wp.static(width)):
            dA = NativeMat22()
            for r in range(2):
                for c in range(2):
                    dA[r, c] = a.grad[r * 2 + c, q]
            P = UT * dA * V
            skew = dtype(0.0)
            if s_sum > dtype(0.0):
                skew = (P[0, 1] - P[1, 0]) / (dtype(2.0) * s_sum)
            sym = dtype(0.0)
            if s_diff > eps * s_sum:
                sym = (P[0, 1] + P[1, 0]) / (dtype(2.0) * s_diff)
            alpha = skew - sym
            beta = -skew - sym
            dUq = U * NativeMat22(dtype(0.0), alpha, -alpha, dtype(0.0))
            dVq = V * NativeMat22(dtype(0.0), beta, -beta, dtype(0.0))
            for r in range(2):
                dsigma[r, q] = P[r, r]
                for c in range(2):
                    dU[r * 2 + c, q] = dUq[r, c]
                    dV[r * 2 + c, q] = dVq[r, c]
        return JetMat22(U, dU), JetVec2(sigma, dsigma), JetMat22(V, dV)

    @wp.func
    def _eigen_gap(x: dtype) -> dtype:
        # adj_eig3 keeps eigenvalue gaps at least 1e-6 in magnitude. It leaves an
        # exact zero alone, which forward mode cannot afford: without knowing
        # whether dQ is used, it would turn dQ into inf or NaN at every matrix
        # with a repeated eigenvalue, such as the identity.
        if x < dtype(0.0) and x > dtype(-1.0e-6):
            return dtype(-1.0e-6)
        if x >= dtype(0.0) and x < dtype(1.0e-6):
            return dtype(1.0e-6)
        return x

    @wp.func
    def jet_eig3(a: JetMat33):
        Q, d = wp.eig3(a.value)
        f01 = dtype(1.0) / _eigen_gap(d[1] - d[0])
        f02 = dtype(1.0) / _eigen_gap(d[2] - d[0])
        f12 = dtype(1.0) / _eigen_gap(d[2] - d[1])
        zero = dtype(0.0)
        F = NativeMat33(zero, f01, f02, -f01, zero, f12, -f02, -f12, zero)
        QT = wp.transpose(Q)
        dQ = GradMat9()
        dd = GradVec3()
        for q in range(wp.static(width)):
            dA = NativeMat33()
            for r in range(3):
                for c in range(3):
                    dA[r, c] = a.grad[r * 3 + c, q]
            # P = Q^T dA Q: dd = diag(P), dQ = Q (F o P).
            P = QT * dA * Q
            dQq = Q * wp.cw_mul(F, P)
            for r in range(3):
                dd[r, q] = P[r, r]
                for c in range(3):
                    dQ[r * 3 + c, q] = dQq[r, c]
        return JetMat33(Q, dQ), JetVec3(d, dd)

    # ------------------------------------------------------------------
    # Publish into Warp's builtin overload table.
    #
    # From here on, ordinary Warp syntax resolves on jets by argument type,
    # in any module, with nothing bound into the caller's namespace.
    # ------------------------------------------------------------------

    @wp.func
    def scalar_value(x: JetScalar) -> dtype:
        return x.value

    _register("value", scalar_value)

    _register("add", jet_add)
    _register("sub", jet_sub)
    _register("mul", jet_mul)
    _register("div", jet_div)
    _register("pow", jet_pow)
    _register("neg", jet_neg)
    _register("pos", jet_pos)

    _register("sin", jet_sin)
    _register("cos", jet_cos)
    _register("tan", jet_tan)
    _register("asin", jet_asin)
    _register("acos", jet_acos)
    _register("atan", jet_atan)
    _register("atan2", jet_atan2)
    _register("exp", jet_exp)
    _register("log", jet_log)
    _register("sqrt", jet_sqrt)
    _register("abs", jet_abs)
    _register("sign", jet_sign)

    _register("min", jet_min)
    _register("max", jet_max)
    _register("clamp", jet_clamp)
    _register("where", jet_where)

    _register("extract", jet_extract)

    _register("dot", jet_dot)
    _register("cw_mul", jet_cw_mul)
    _register("cw_div", jet_cw_div)
    _register("length", jet_length)
    _register("length_sq", jet_length_sq)
    _register("normalize", jet_normalize)
    _register("cross", jet_cross)

    _register("transpose", jet_transpose)
    _register("determinant", jet_determinant)
    _register("trace", jet_trace)
    _register("inverse", jet_inverse)

    _register("sinh", jet_sinh)
    _register("cosh", jet_cosh)
    _register("tanh", jet_tanh)
    _register("log2", jet_log2)
    _register("log10", jet_log10)
    _register("cbrt", jet_cbrt)
    _register("erf", jet_erf)
    _register("floor", jet_floor)
    _register("lerp", jet_lerp)
    _register("smoothstep", jet_smoothstep)
    _register("assign_copy", jet_assign_copy)
    _register("skew", jet_skew)
    # Multiple assignment (U, sigma, V = wp.svd3(A)) picks a builtin overload by
    # its number of outputs, which a @wp.func overload reports through value_func.
    for fn, outputs in (
        (jet_svd3, (JetMat33, JetVec3, JetMat33)),
        (jet_svd2, (JetMat22, JetVec2, JetMat22)),
        (jet_eig3, (JetMat33, JetVec3)),
    ):
        for overload in tuple(fn.user_overloads.values()) or (fn,):
            overload.value_func = lambda arg_types, arg_values, outputs=outputs: outputs
    _register("svd3", jet_svd3)
    _register("svd2", jet_svd2)
    _register("eig3", jet_eig3)

    return SimpleNamespace(
        width=width,
        dtype=dtype,
        scalar=JetScalar,
        vec2=JetVec2,
        vec3=JetVec3,
        mat22=JetMat22,
        mat33=JetMat33,
        mat32=JetMat32,
        mat23=JetMat23,
        grad_type_scalar=Grad,
        grad_type_vec2=GradVec2,
        grad_type_vec3=GradVec3,
        grad_type_mat22=GradMat4,
        grad_type_mat33=GradMat9,
        grad_type_mat32=GradMat6,
        grad_type_mat23=GradMat6,
        native_vec2=NativeVec2,
        native_vec3=NativeVec3,
        native_mat22=NativeMat22,
        native_mat33=NativeMat33,
        constant=scalar_constant,
        with_grad=scalar_with_grad,
        variable=scalar_variable,
        make_vec2=vec2_from_scalars,
        make_vec3=vec3_from_scalars,
        constant_vec2=vec2_constant,
        constant_vec3=vec3_constant,
        make_mat22=mat22_from_cols,
        make_mat33=mat33_from_cols,
        make_mat32=mat32_from_cols,
        perp=jet_perp,
        cross2=jet_cross2,
        _slot_jet_types={
            "scalar": (JetScalar, Grad),
            "vec2": (JetVec2, GradVec2),
            "vec3": (JetVec3, GradVec3),
            "mat22": (JetMat22, GradMat4),
            "mat33": (JetMat33, GradMat9),
            "mat32": (JetMat32, GradMat6),
            "mat23": (JetMat23, GradMat6),
        },
    )


# Shape of each supported slot kind, and the order of the components it
# contributes to the derivative directions: row-major for matrices.
_SLOT_SHAPES = {
    "scalar": (),
    "vec2": (2,),
    "vec3": (3,),
    "mat22": (2, 2),
    "mat33": (3, 3),
    "mat32": (3, 2),
    "mat23": (2, 3),
}
_SHAPE_TO_KIND = {shape: kind for kind, shape in _SLOT_SHAPES.items()}

_SLOT_SUMMARY = "scalar, vec2, vec3, mat22, mat33, mat32 and mat23"

# Layout-specific namespaces (what JetSpace and JetSpace2 return), keyed
# by (order, width, dtype, slot kinds). Types and arithmetic live in _CACHE and
# _CACHE2 and are shared by every layout of the same width and dtype.
_CACHE_LAYOUT = {}

_GENERATED_IDS = itertools.count()


class _JetNamespace(SimpleNamespace):
    """Namespace of a jet space that explains the helpers it deliberately lacks."""

    def __getattr__(self, name):
        # Only reached when normal lookup fails.
        if name == "seed":
            raise AttributeError(
                f"seed() needs one derivative direction per component of the slots, "
                f"but this space has width {self.width} for {self.slot_components} components. "
                "Use directional() to supply the directions explicitly."
            )
        raise AttributeError(name)


def _slot_components(kind: str):
    """Return the index tuples of the components a slot of ``kind`` contributes."""
    shape = _SLOT_SHAPES[kind]
    if len(shape) == 0:
        return [()]
    if len(shape) == 1:
        return [(i,) for i in range(shape[0])]
    return [(r, c) for r in range(shape[0]) for c in range(shape[1])]


def _classify_slot(slot):
    """Return ``(kind, scalar_dtype)`` for a slot type.

    ``scalar_dtype`` is ``None`` for Python ``float``, which does not commit to a precision.
    """
    if slot is float:
        return "scalar", None
    if slot in float_types:
        return "scalar", slot

    shape = getattr(slot, "_shape_", None)
    scalar_dtype = getattr(slot, "_wp_scalar_type_", None)
    if shape is not None and scalar_dtype is not None and tuple(shape) in _SHAPE_TO_KIND:
        if scalar_dtype not in float_types:
            raise TypeError(f"Jet slot {slot.__name__} must have a floating-point scalar type")
        return _SHAPE_TO_KIND[tuple(shape)], scalar_dtype

    raise TypeError(
        f"Unsupported jet slot type {getattr(slot, '__name__', slot)!r}. Supported slot types are {_SLOT_SUMMARY}."
    )


def _normalize_slots(slots, dtype):
    """Resolve the ``slots`` argument of :func:`JetSpace` into ``(kinds, dtype)``."""
    if isinstance(slots, numbers.Integral) and not isinstance(slots, bool):
        count = int(slots)
        if count <= 0:
            raise ValueError("Jet width must be positive")
        return ("scalar",) * count, wp.float32 if dtype is None else dtype

    if not isinstance(slots, (tuple, list)):
        slots = (slots,)
    if len(slots) == 0:
        raise ValueError("A jet space needs at least one slot")

    kinds = []
    inferred = set()
    for slot in slots:
        kind, scalar_dtype = _classify_slot(slot)
        kinds.append(kind)
        if scalar_dtype is not None:
            inferred.add(scalar_dtype)

    if len(inferred) > 1:
        names = ", ".join(sorted(t.__name__ for t in inferred))
        raise TypeError(f"Jet slots mix scalar types ({names}); all slots must share one precision")
    if dtype is not None and inferred and dtype not in inferred:
        raise TypeError(
            f"dtype={dtype.__name__} conflicts with the slot types, which are {next(iter(inferred)).__name__}"
        )

    if dtype is None:
        dtype = next(iter(inferred)) if inferred else wp.float32
    return tuple(kinds), dtype


def _native_slot_type(kind: str, dtype):
    shape = _SLOT_SHAPES[kind]
    if len(shape) == 0:
        return dtype
    if len(shape) == 1:
        return vector(shape[0], dtype)
    return matrix(shape, dtype)


def _define_generated(name: str, source: str, namespace: dict):
    """Compile ``source`` as a Warp function named ``name``.

    The number of slots varies per space, so the signature and body of the
    helpers below cannot be written out once. The source is registered with
    :mod:`linecache` so Warp's codegen can read it back like any other function.
    """
    filename = f"<warp-jet-{name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {"__name__": __name__, "wp": wp, **namespace}
    exec(compile(source, filename, "exec"), namespace)
    return namespace[name]


def _grad_assignment(kind: str, component: int, column: int, value: str, var: str) -> str:
    # A scalar slot's gradient is a vector over directions; every other slot's is
    # a (components, directions) matrix, with matrices flattened row-major.
    if kind == "scalar":
        return f"{var}[{column}] = {value}"
    return f"{var}[{component}, {column}] = {value}"


def _build_slot_helpers(base, kinds, width: int, dtype, second_order: bool, include_directional: bool):
    """Generate ``seed`` (and ``directional``) for one slot layout.

    ``seed`` places a unit direction on every component of every slot, in order.
    ``directional`` takes the directions explicitly, so ``width`` may be smaller
    than the number of components -- one direction gives a Hessian-vector product.

    Both take their operands as separate positional arguments rather than as
    tuples. Warp does not propagate adjoints into a tuple argument, which would
    silently zero the reverse-over-forward derivatives these helpers exist to
    support.
    """
    uid = next(_GENERATED_IDS)
    nslots = len(kinds)

    natives = tuple(_native_slot_type(kind, dtype) for kind in kinds)

    namespace = {"dtype": dtype}
    for kind, (jet_type, grad_type) in base._slot_jet_types.items():
        namespace[f"Jet_{kind}"] = jet_type
        namespace[f"Grad_{kind}"] = grad_type
    if second_order:
        namespace["Hess"] = base.hess_type_scalar
        namespace["variable"] = base.variable

    # Annotation types are passed by name so the source stays readable.
    for s_, native in enumerate(natives):
        namespace[f"Native_{s_}"] = native

    value_params = [f"v{s}: Native_{s}" for s in range(nslots)]

    def construct(slot: int) -> str:
        kind = kinds[slot]
        extra = ", Hess()" if second_order else ""
        return f"    j{slot} = Jet_{kind}(v{slot}, c{slot}{extra})"

    def returns():
        return "    return " + ", ".join(f"j{s}" for s in range(nslots))

    # seed: component k of every slot gets its own unit direction.
    lines = ["@wp.func", f"def jet_seed_{uid}({', '.join(value_params)}):"]
    offset = 0
    for s, kind in enumerate(kinds):
        components = _slot_components(kind)
        if second_order and kind != "scalar":
            # A second-order vector jet is one scalar jet per component.
            scalars = ", ".join(f"variable(v{s}[{k}], {offset + k})" for k in range(len(components)))
            lines.append(f"    j{s} = Jet_{kind}({scalars})")
        else:
            lines.append(f"    c{s} = Grad_{kind}()")
            for k, _ in enumerate(components):
                lines.append("    " + _grad_assignment(kind, k, offset + k, "dtype(1.0)", f"c{s}"))
            lines.append(construct(s))
        offset += len(components)
    lines.append(returns())

    total = offset
    seed = _define_generated(f"jet_seed_{uid}", "\n".join(lines) + "\n", namespace) if total == width else None

    directional = None
    if include_directional:
        # One direction per column. After the values come the directions, flat and
        # direction-major: argument d{q * nslots + s} is slot s's value along
        # direction q.
        direction_params = [f"d{q * nslots + s}: Native_{s}" for q in range(width) for s in range(nslots)]

        lines = ["@wp.func", f"def jet_directional_{uid}({', '.join(value_params + direction_params)}):"]
        for s, kind in enumerate(kinds):
            lines.append(f"    c{s} = Grad_{kind}()")
            for q in range(width):
                for k, component in enumerate(_slot_components(kind)):
                    access = f"d{q * nslots + s}" + (f"[{', '.join(map(str, component))}]" if component else "")
                    lines.append("    " + _grad_assignment(kind, k, q, access, f"c{s}"))
            lines.append(construct(s))
        lines.append(returns())
        directional = _define_generated(f"jet_directional_{uid}", "\n".join(lines) + "\n", namespace)

    return seed, directional, natives, total


def _type_lookup(base, dtype, index: int, what: str):
    """Return a function mapping a Warp type to entry ``index`` of ``base._slot_jet_types``."""

    def lookup(t):
        kind, scalar_dtype = _classify_slot(t)
        if scalar_dtype is not None and scalar_dtype is not dtype:
            raise TypeError(
                f"{getattr(t, '__name__', t)} has scalar type {scalar_dtype.__name__}, "
                f"but this jet space is built on {dtype.__name__}"
            )
        entry = base._slot_jet_types.get(kind)
        if entry is None or entry[index] is None:
            raise TypeError(f"This jet space has no {what} for {getattr(t, '__name__', t)}")
        return entry[index]

    return lookup


def _layout_namespace(base, kinds, width: int, dtype, second_order: bool):
    key = (second_order, width, dtype, kinds)
    J = _CACHE_LAYOUT.get(key)
    if J is None:
        seed, directional, natives, total = _build_slot_helpers(
            base, kinds, width, dtype, second_order, include_directional=not second_order
        )
        members = {k: v for k, v in vars(base).items() if not k.startswith("_")}
        members.update(slots=natives, slot_components=total)
        members["jet_type"] = _type_lookup(base, dtype, 0, "jet type")
        if not second_order:
            members["grad_type"] = _type_lookup(base, dtype, 1, "gradient type")
        if directional is not None:
            members["directional"] = directional
        if seed is not None:
            members["seed"] = seed
        J = _JetNamespace(**members)
        _CACHE_LAYOUT[key] = J
    return J


def JetSpace(slots, dtype=None, *, width=None):
    """Return the jet types and helpers for a set of differentiated inputs.

    .. admonition:: Experimental

        Forward-mode jets are experimental and subject to change without a formal deprecation
        cycle.

    A jet is a value paired with its first derivatives along a fixed set of
    directions. Arithmetic on jets updates the derivatives alongside the value,
    so evaluating a function on jets computes its derivatives in the same pass.
    This is forward-mode automatic differentiation, in contrast to the
    reverse-mode differentiation of :func:`warp.grad` and :class:`warp.Tape`.
    Jets are ordinary Warp types, so reverse mode can also differentiate through
    code that uses them.

    The first call for a given width and scalar type generates the types and
    registers their arithmetic as overloads of Warp's builtins; later calls
    return a namespace over the same types.

    Args:
        slots: The inputs differentiated with respect to. Either an integer
            ``n``, meaning ``n`` scalar inputs, or a sequence of types, one per
            input: ``wp.float32``, ``wp.vec2``, ``wp.vec3``, ``wp.mat22``,
            ``wp.mat33``, ``wp.mat32`` or ``wp.mat23``. A spring over two nodes
            is ``(wp.vec3, wp.vec3)``.
        dtype: Scalar type the jets are built on. Inferred from ``slots`` when
            they are Warp types; ``wp.float32`` when ``slots`` is an integer
            or contains only Python ``float``.
        width: Number of derivative directions each jet carries. Defaults to the
            total number of scalar components in ``slots``, in which case
            ``seed`` seeds the unit directions and a jet carries the full
            gradient. A smaller width carries only the directions passed to
            ``directional``, which is cheaper when only a few projections of the
            gradient are needed.

    Returns:
        A namespace holding the generated jet types (``scalar``, ``vec2``,
        ``vec3``, ``mat22``, ``mat33``, ``mat32``, ``mat23``), the type of
        each one's ``grad`` field (``grad_type_scalar``, ``grad_type_vec2``,
        and so on through ``grad_type_mat23``), the helpers that construct
        jets, and ``slots``, the native type of each slot.

    Jets are created inside a kernel or function with ``seed`` or
    ``directional``. With ``spring_energy`` an ordinary ``@wp.func`` that is
    generic over its argument types::

        J = wp.JetSpace((wp.vec3, wp.vec3))  # two 3D points, width 6


        @wp.func
        def spring_gradient(x0: wp.vec3, x1: wp.vec3):
            # x0_jet and x1_jet are J.vec3: x0_jet carries directions 0-2, x1_jet 3-5.
            x0_jet, x1_jet = J.seed(x0, x1)
            return spring_energy(x0_jet, x1_jet).grad  # the 6-component gradient

    ``seed(*values)`` takes one value per slot and returns one jet per slot,
    with the derivative directions numbered slot by slot in the order the
    components appear (row-major for matrices). With a single slot it takes and
    returns the bare value. It is available only when ``width`` equals the
    number of components; otherwise use ``directional``.

    ``directional(*values, *directions)`` takes the same values, followed by
    ``width * len(slots)`` direction vectors ordered direction-major: direction
    ``q`` of slot ``s`` is argument ``q * len(slots) + s`` after the values. For
    example, for a width-1 space the directions are one value per slot, and the
    jets carry the derivative along the combined direction::

        J1 = wp.JetSpace((wp.vec3, wp.vec3), width=1)


        @wp.func
        def spring_directional_derivative(x0: wp.vec3, x1: wp.vec3, v0: wp.vec3, v1: wp.vec3):
            # Derivative along the combined direction (v0, v1).
            x0_jet, x1_jet = J1.directional(x0, x1, v0, v1)
            return spring_energy(x0_jet, x1_jet).grad[0]

    ``variable(x, direction)`` is the underlying scalar primitive: a scalar jet
    with a unit derivative along ``direction``. Use it to seed in a loop, where
    the number of scalars is not a fixed argument list.

    ``jet_type(t)`` and ``grad_type(t)`` look the types up by the Warp type of
    the value instead: ``J.grad_type(wp.vec2)`` is ``J.grad_type_vec2``. They
    are Python functions, so they work in annotations and host code, such as
    ``wp.array[J.grad_type(wp.vec2)]``, but cannot be called inside a kernel
    or ``@wp.func``, where the attributes are needed. A type whose scalar type
    differs from the space's, or that the space does not support, raises
    :class:`TypeError`.

    ``width`` is fixed when the types are specialized, not when a kernel runs:
    it becomes the length of ``grad`` and is unrolled into the generated code.
    Every intermediate value carries all ``width`` derivative components, so
    cost in registers and compile time scales with the width requested,
    regardless of how many components are read back. Different widths are
    independent specializations that may coexist in one process.

    This makes jets suited to *local* derivatives of fixed arity -- a spring
    over two ``vec3`` nodes has width 6, a triangle 9, a tetrahedron 12 -- where
    the layout is known where the kernel is written and the dense local gradient
    or Hessian block is scattered into a sparse global matrix. They are not
    suited to differentiating with respect to an unbounded or runtime number of
    variables: a problem-sized width cannot specialize a type, and even when it
    could, forward mode would cost one direction per variable against reverse
    mode's single backward pass. Use :class:`warp.Tape` or :func:`warp.grad` for
    that outer derivative, and jets for the fixed-arity term inside it.

    Registering the arithmetic mutates Warp's global builtin overload table,
    which is what lets ``a * b`` resolve on jets from any module. The effect is
    additive and lasts for the lifetime of the process.
    """
    kinds, dtype = _normalize_slots(slots, dtype)
    total = sum(len(_slot_components(kind)) for kind in kinds)

    if width is None:
        width = total
    else:
        width = operator.index(width)

    key = (width, dtype)
    base = _CACHE.get(key)
    if base is None:
        base = _make_jet_space(width, dtype)
        _CACHE[key] = base

    return _layout_namespace(base, kinds, width, dtype, second_order=False)


# ==========================================================================
# Second-order (forward-over-forward) jets.
#
# A width-k second-order jet carries value, gradient, and the full k x k
# Hessian of one scalar with respect to k variables:
#
#     x = value + grad . eps + 0.5 eps^T hess eps
#
# Propagating it through a scalar function in a single forward pass yields the
# local Hessian directly -- no reverse pass, no tape. The per-intermediate
# state is O(k^2), so this is a small-k strategy: register pressure grows
# quadratically where the first-order jet's grows linearly.
#
# Scalars, plus vec2/vec3 built from one scalar jet per component; no matrices.
# ==========================================================================


def _make_jet_space2(width: int, dtype):
    _check_space_args(width, dtype)

    Grad = vector(width, dtype)
    Hess = matrix((width, width), dtype)

    @wp.struct
    class Jet2Scalar:
        value: dtype
        grad: Grad
        hess: Hess

    # ---- construction / seeding ----

    @wp.func
    def scalar_constant(x: dtype) -> Jet2Scalar:
        return Jet2Scalar(x, Grad(), Hess())

    @wp.func
    def scalar_variable(x: dtype, direction: int) -> Jet2Scalar:
        g = Grad()
        g[direction] = dtype(1.0)
        return Jet2Scalar(x, g, Hess())

    # ---- linear ops ----

    @wp.func
    def jet_add(a: Jet2Scalar, b: Jet2Scalar) -> Jet2Scalar:
        return Jet2Scalar(a.value + b.value, a.grad + b.grad, a.hess + b.hess)

    @wp.func
    def jet_add(a: Jet2Scalar, b: dtype) -> Jet2Scalar:
        return Jet2Scalar(a.value + b, a.grad, a.hess)

    @wp.func
    def jet_add(a: dtype, b: Jet2Scalar) -> Jet2Scalar:
        return Jet2Scalar(a + b.value, b.grad, b.hess)

    @wp.func
    def jet_sub(a: Jet2Scalar, b: Jet2Scalar) -> Jet2Scalar:
        return Jet2Scalar(a.value - b.value, a.grad - b.grad, a.hess - b.hess)

    @wp.func
    def jet_sub(a: Jet2Scalar, b: dtype) -> Jet2Scalar:
        return Jet2Scalar(a.value - b, a.grad, a.hess)

    @wp.func
    def jet_sub(a: dtype, b: Jet2Scalar) -> Jet2Scalar:
        return Jet2Scalar(a - b.value, -b.grad, -b.hess)

    @wp.func
    def jet_pos(a: Jet2Scalar) -> Jet2Scalar:
        return a

    @wp.func
    def jet_neg(a: Jet2Scalar) -> Jet2Scalar:
        return Jet2Scalar(-a.value, -a.grad, -a.hess)

    @wp.func
    def jet_mul(a: Jet2Scalar, s: dtype) -> Jet2Scalar:
        return Jet2Scalar(a.value * s, a.grad * s, a.hess * s)

    @wp.func
    def jet_mul(s: dtype, a: Jet2Scalar) -> Jet2Scalar:
        return Jet2Scalar(s * a.value, s * a.grad, s * a.hess)

    @wp.func
    def jet_div(a: Jet2Scalar, s: dtype) -> Jet2Scalar:
        return Jet2Scalar(a.value / s, a.grad / s, a.hess / s)

    # ---- product: hess picks up the grad outer products ----

    @wp.func
    def jet_mul(a: Jet2Scalar, b: Jet2Scalar) -> Jet2Scalar:
        outer_ab = wp.outer(a.grad, b.grad)
        return Jet2Scalar(
            a.value * b.value,
            a.value * b.grad + b.value * a.grad,
            a.value * b.hess + b.value * a.hess + outer_ab + wp.transpose(outer_ab),
        )

    @wp.func
    def jet_div(a: Jet2Scalar, b: Jet2Scalar) -> Jet2Scalar:
        # Differentiating q * b = a twice gives each derivative of q divided by b once.
        # Dividing directly, as native division does, avoids forming 1 / b and its
        # derivatives, which overflow for small denominators.
        value = a.value / b.value
        grad = (a.grad - value * b.grad) / b.value
        outer_gb = wp.outer(grad, b.grad)
        hess = (a.hess - value * b.hess - outer_gb - wp.transpose(outer_gb)) / b.value
        return Jet2Scalar(value, grad, hess)

    # ---- elementary functions: chain rule with f' and f'' ----

    @wp.func
    def _lift(value: dtype, fp: dtype, fpp: dtype, a: Jet2Scalar) -> Jet2Scalar:
        return Jet2Scalar(value, fp * a.grad, fp * a.hess + fpp * wp.outer(a.grad, a.grad))

    @wp.func
    def jet_sin(a: Jet2Scalar) -> Jet2Scalar:
        return _lift(wp.sin(a.value), wp.cos(a.value), -wp.sin(a.value), a)

    @wp.func
    def jet_cos(a: Jet2Scalar) -> Jet2Scalar:
        return _lift(wp.cos(a.value), -wp.sin(a.value), -wp.cos(a.value), a)

    @wp.func
    def jet_exp(a: Jet2Scalar) -> Jet2Scalar:
        e = wp.exp(a.value)
        return _lift(e, e, e, a)

    @wp.func
    def jet_log(a: Jet2Scalar) -> Jet2Scalar:
        inv = dtype(1.0) / a.value
        return _lift(wp.log(a.value), inv, -inv * inv, a)

    @wp.func
    def jet_sqrt(a: Jet2Scalar) -> Jet2Scalar:
        # The second derivative -1 / (4 s**3) overflows for small arguments even when its
        # contraction with the gradient is representable, so scale the gradient by 1 / s
        # first: f'' g g^T = -(1 / (4 s)) (g / s) (g / s)^T.
        s = wp.sqrt(a.value)
        fp = dtype(0.5) / s
        g = a.grad / s
        return Jet2Scalar(s, fp * a.grad, fp * a.hess - (dtype(0.25) / s) * wp.outer(g, g))

    @wp.func
    def jet_tan(a: Jet2Scalar) -> Jet2Scalar:
        c = wp.cos(a.value)
        sec2 = dtype(1.0) / (c * c)
        return _lift(wp.tan(a.value), sec2, dtype(2.0) * wp.tan(a.value) * sec2, a)

    @wp.func
    def jet_asin(a: Jet2Scalar) -> Jet2Scalar:
        # Like native asin(), which clamps its argument to [-1, 1], the derivatives are
        # zero unless the argument lies strictly inside the interval. Test the interval
        # before taking the square root, whose adjoint is NaN outside it.
        if a.value > dtype(-1.0) and a.value < dtype(1.0):
            r = dtype(1.0) / wp.sqrt(dtype(1.0) - a.value * a.value)
            return _lift(wp.asin(a.value), r, a.value * r * r * r, a)
        return Jet2Scalar(wp.asin(a.value), Grad(), Hess())

    @wp.func
    def jet_acos(a: Jet2Scalar) -> Jet2Scalar:
        if a.value > dtype(-1.0) and a.value < dtype(1.0):
            r = dtype(1.0) / wp.sqrt(dtype(1.0) - a.value * a.value)
            return _lift(wp.acos(a.value), -r, -a.value * r * r * r, a)
        return Jet2Scalar(wp.acos(a.value), Grad(), Hess())

    @wp.func
    def jet_atan(a: Jet2Scalar) -> Jet2Scalar:
        u = dtype(1.0) / (dtype(1.0) + a.value * a.value)
        return _lift(wp.atan(a.value), u, -dtype(2.0) * a.value * u * u, a)

    @wp.func
    def _pow_const(a: Jet2Scalar, p: dtype) -> Jet2Scalar:
        """a**p for a constant exponent, with the degenerate exponents guarded.

        The general term p (p-1) a**(p-2) evaluates as 0 * inf at a == 0 for
        p == 1, and both derivative terms do so for p == 0, where the true
        derivatives are zero. Non-integer p below those is left alone: its
        derivatives really are infinite at zero, so inf is the honest answer.
        """
        if p == dtype(0.0):
            return Jet2Scalar(dtype(1.0), Grad(), Hess())

        v = wp.pow(a.value, p)
        fp = p * wp.pow(a.value, p - dtype(1.0))

        if p == dtype(1.0):
            return _lift(v, fp, dtype(0.0), a)

        return _lift(v, fp, p * (p - dtype(1.0)) * wp.pow(a.value, p - dtype(2.0)), a)

    @wp.func
    def jet_pow(a: Jet2Scalar, p: int) -> Jet2Scalar:
        return _pow_const(a, dtype(p))

    @wp.func
    def jet_pow(a: Jet2Scalar, p: dtype) -> Jet2Scalar:
        return _pow_const(a, p)

    @wp.func
    def jet_pow(a: dtype, b: Jet2Scalar) -> Jet2Scalar:
        v = wp.pow(a, b.value)
        # A zero base gives a constant result, so the derivatives are 0 rather than 0 * log(0).
        if a == dtype(0.0):
            return Jet2Scalar(v, Grad(), Hess())
        la = wp.log(a)
        return _lift(v, v * la, v * la * la, b)

    @wp.func
    def jet_div(a: dtype, b: Jet2Scalar) -> Jet2Scalar:
        return jet_div(scalar_constant(a), b)

    # ---- two-argument chain rule: value = F(u, v), F's partials as scalars ----

    @wp.func
    def _lift2(
        value: dtype,
        fu: dtype,
        fv: dtype,
        fuu: dtype,
        fvv: dtype,
        fuv: dtype,
        u: Jet2Scalar,
        v: Jet2Scalar,
    ) -> Jet2Scalar:
        ouv = wp.outer(u.grad, v.grad)
        hess = (
            fu * u.hess
            + fv * v.hess
            + fuu * wp.outer(u.grad, u.grad)
            + fvv * wp.outer(v.grad, v.grad)
            + fuv * (ouv + wp.transpose(ouv))
        )
        return Jet2Scalar(value, fu * u.grad + fv * v.grad, hess)

    @wp.func
    def jet_atan2(y: Jet2Scalar, x: Jet2Scalar) -> Jet2Scalar:
        # Like native atan2(), the derivatives are zero at the origin.
        d = x.value * x.value + y.value * y.value
        if not (d > dtype(0.0)):
            return Jet2Scalar(wp.atan2(y.value, x.value), Grad(), Hess())
        d2 = d * d
        return _lift2(
            wp.atan2(y.value, x.value),
            x.value / d,  # d/dy
            -y.value / d,  # d/dx
            -dtype(2.0) * x.value * y.value / d2,  # d2/dy2
            dtype(2.0) * x.value * y.value / d2,  # d2/dx2
            (y.value * y.value - x.value * x.value) / d2,  # d2/dydx
            y,
            x,
        )

    @wp.func
    def jet_atan2(y: Jet2Scalar, x: dtype) -> Jet2Scalar:
        return jet_atan2(y, scalar_constant(x))

    @wp.func
    def jet_atan2(y: dtype, x: Jet2Scalar) -> Jet2Scalar:
        return jet_atan2(scalar_constant(y), x)

    @wp.func
    def jet_pow(a: Jet2Scalar, b: Jet2Scalar) -> Jet2Scalar:
        # Partials written against wp.pow rather than as g / a, for the same
        # reason as the first-order overload: at a.value == 0 the 1 / a form
        # gives 0 * inf = NaN where the limit is finite. The three partials
        # carrying log(a) are zero in that limit for b > 0.
        g = wp.pow(a.value, b.value)
        # The da and d2/da2 partials take the same guards as the constant-exponent
        # overload: they are 0, not 0 * inf, for an exponent of 0, and d2/da2 also for 1.
        da = dtype(0.0)
        if b.value != dtype(0.0):
            da = b.value * wp.pow(a.value, b.value - dtype(1.0))
        data = dtype(0.0)
        if b.value != dtype(0.0) and b.value != dtype(1.0):
            data = b.value * (b.value - dtype(1.0)) * wp.pow(a.value, b.value - dtype(2.0))
        if a.value > dtype(0.0):
            la = wp.log(a.value)
            return _lift2(
                g,
                da,  # d/da
                g * la,  # d/db
                data,  # d2/da2
                g * la * la,  # d2/db2
                wp.pow(a.value, b.value - dtype(1.0)) * (dtype(1.0) + b.value * la),  # d2/dadb
                a,
                b,
            )
        return _lift2(g, da, dtype(0.0), data, dtype(0.0), dtype(0.0), a, b)

    # ---- branching: value-only comparisons carry the chosen jet's derivatives ----

    @wp.func
    def jet_abs(a: Jet2Scalar) -> Jet2Scalar:
        # Native adj_abs() takes the derivative +1 at zero, and abs(NaN) is NaN.
        if a.value < dtype(0.0):
            return -a
        return Jet2Scalar(wp.abs(a.value), a.grad, a.hess)

    @wp.func
    def jet_sign(a: Jet2Scalar) -> Jet2Scalar:
        return Jet2Scalar(wp.sign(a.value), Grad(), Hess())

    # min/max/clamp follow native Warp as in the first-order space: a NaN operand is
    # treated as missing, and a tie selects the second operand.

    @wp.func
    def jet_min(a: Jet2Scalar, b: Jet2Scalar) -> Jet2Scalar:
        if wp.isnan(a.value):
            return b
        if wp.isnan(b.value):
            return a
        if a.value < b.value:
            return a
        return b

    @wp.func
    def jet_min(a: Jet2Scalar, b: dtype) -> Jet2Scalar:
        return jet_min(a, scalar_constant(b))

    @wp.func
    def jet_min(a: dtype, b: Jet2Scalar) -> Jet2Scalar:
        return jet_min(scalar_constant(a), b)

    @wp.func
    def jet_max(a: Jet2Scalar, b: Jet2Scalar) -> Jet2Scalar:
        if wp.isnan(a.value):
            return b
        if wp.isnan(b.value):
            return a
        if a.value > b.value:
            return a
        return b

    @wp.func
    def jet_max(a: Jet2Scalar, b: dtype) -> Jet2Scalar:
        return jet_max(a, scalar_constant(b))

    @wp.func
    def jet_max(a: dtype, b: Jet2Scalar) -> Jet2Scalar:
        return jet_max(scalar_constant(a), b)

    @wp.func
    def jet_clamp(a: Jet2Scalar, lo: dtype, hi: dtype) -> Jet2Scalar:
        # Native clamp(x, lo, hi) is min(max(lo, x), hi).
        return jet_min(jet_max(scalar_constant(lo), a), scalar_constant(hi))

    @wp.func
    def jet_where(cond: bool, a: Jet2Scalar, b: Jet2Scalar) -> Jet2Scalar:
        if cond:
            return a
        return b

    # ---- vectors: one scalar jet per component ----
    #
    # Each operation is composed from the scalar operations above, so a vector
    # expression performs exactly the arithmetic of the same expression written
    # out per component, and needs no second-order rules of its own.

    NativeVec2 = vector(2, dtype)
    NativeVec3 = vector(3, dtype)

    @wp.struct
    class Jet2Vec2:
        x: Jet2Scalar
        y: Jet2Scalar

    @wp.struct
    class Jet2Vec3:
        x: Jet2Scalar
        y: Jet2Scalar
        z: Jet2Scalar

    @wp.func
    def vec2_constant(v: NativeVec2) -> Jet2Vec2:
        return Jet2Vec2(scalar_constant(v[0]), scalar_constant(v[1]))

    @wp.func
    def vec3_constant(v: NativeVec3) -> Jet2Vec3:
        return Jet2Vec3(scalar_constant(v[0]), scalar_constant(v[1]), scalar_constant(v[2]))

    @wp.func
    def vec2_from_scalars(x: Jet2Scalar, y: Jet2Scalar) -> Jet2Vec2:
        return Jet2Vec2(x, y)

    @wp.func
    def vec3_from_scalars(x: Jet2Scalar, y: Jet2Scalar, z: Jet2Scalar) -> Jet2Vec3:
        return Jet2Vec3(x, y, z)

    # Indexing goes through a native vector of component numbers so that negative
    # and out-of-range indices behave exactly as they do on a native vector.

    @wp.func
    def jet_extract(v: Jet2Vec2, i: int) -> Jet2Scalar:
        if wp.vec2i(0, 1)[i] == 0:
            return v.x
        return v.y

    @wp.func
    def jet_extract(v: Jet2Vec3, i: int) -> Jet2Scalar:
        component = wp.vec3i(0, 1, 2)[i]
        if component == 0:
            return v.x
        if component == 1:
            return v.y
        return v.z

    @wp.func
    def jet_add(a: Jet2Vec2, b: Jet2Vec2) -> Jet2Vec2:
        return Jet2Vec2(a.x + b.x, a.y + b.y)

    @wp.func
    def jet_add(a: Jet2Vec2, b: NativeVec2) -> Jet2Vec2:
        return Jet2Vec2(a.x + b[0], a.y + b[1])

    @wp.func
    def jet_add(a: NativeVec2, b: Jet2Vec2) -> Jet2Vec2:
        return Jet2Vec2(a[0] + b.x, a[1] + b.y)

    @wp.func
    def jet_add(a: Jet2Vec3, b: Jet2Vec3) -> Jet2Vec3:
        return Jet2Vec3(a.x + b.x, a.y + b.y, a.z + b.z)

    @wp.func
    def jet_add(a: Jet2Vec3, b: NativeVec3) -> Jet2Vec3:
        return Jet2Vec3(a.x + b[0], a.y + b[1], a.z + b[2])

    @wp.func
    def jet_add(a: NativeVec3, b: Jet2Vec3) -> Jet2Vec3:
        return Jet2Vec3(a[0] + b.x, a[1] + b.y, a[2] + b.z)

    @wp.func
    def jet_sub(a: Jet2Vec2, b: Jet2Vec2) -> Jet2Vec2:
        return Jet2Vec2(a.x - b.x, a.y - b.y)

    @wp.func
    def jet_sub(a: Jet2Vec2, b: NativeVec2) -> Jet2Vec2:
        return Jet2Vec2(a.x - b[0], a.y - b[1])

    @wp.func
    def jet_sub(a: NativeVec2, b: Jet2Vec2) -> Jet2Vec2:
        return Jet2Vec2(a[0] - b.x, a[1] - b.y)

    @wp.func
    def jet_sub(a: Jet2Vec3, b: Jet2Vec3) -> Jet2Vec3:
        return Jet2Vec3(a.x - b.x, a.y - b.y, a.z - b.z)

    @wp.func
    def jet_sub(a: Jet2Vec3, b: NativeVec3) -> Jet2Vec3:
        return Jet2Vec3(a.x - b[0], a.y - b[1], a.z - b[2])

    @wp.func
    def jet_sub(a: NativeVec3, b: Jet2Vec3) -> Jet2Vec3:
        return Jet2Vec3(a[0] - b.x, a[1] - b.y, a[2] - b.z)

    @wp.func
    def jet_pos(a: Jet2Vec2) -> Jet2Vec2:
        return a

    @wp.func
    def jet_pos(a: Jet2Vec3) -> Jet2Vec3:
        return a

    @wp.func
    def jet_neg(a: Jet2Vec2) -> Jet2Vec2:
        return Jet2Vec2(-a.x, -a.y)

    @wp.func
    def jet_neg(a: Jet2Vec3) -> Jet2Vec3:
        return Jet2Vec3(-a.x, -a.y, -a.z)

    @wp.func
    def jet_mul(a: Jet2Vec2, s: dtype) -> Jet2Vec2:
        return Jet2Vec2(a.x * s, a.y * s)

    @wp.func
    def jet_mul(s: dtype, a: Jet2Vec2) -> Jet2Vec2:
        return Jet2Vec2(s * a.x, s * a.y)

    @wp.func
    def jet_mul(a: Jet2Vec2, s: Jet2Scalar) -> Jet2Vec2:
        return Jet2Vec2(a.x * s, a.y * s)

    @wp.func
    def jet_mul(s: Jet2Scalar, a: Jet2Vec2) -> Jet2Vec2:
        return Jet2Vec2(s * a.x, s * a.y)

    @wp.func
    def jet_mul(a: NativeVec2, s: Jet2Scalar) -> Jet2Vec2:
        return Jet2Vec2(a[0] * s, a[1] * s)

    @wp.func
    def jet_mul(s: Jet2Scalar, a: NativeVec2) -> Jet2Vec2:
        return Jet2Vec2(s * a[0], s * a[1])

    @wp.func
    def jet_mul(a: Jet2Vec3, s: dtype) -> Jet2Vec3:
        return Jet2Vec3(a.x * s, a.y * s, a.z * s)

    @wp.func
    def jet_mul(s: dtype, a: Jet2Vec3) -> Jet2Vec3:
        return Jet2Vec3(s * a.x, s * a.y, s * a.z)

    @wp.func
    def jet_mul(a: Jet2Vec3, s: Jet2Scalar) -> Jet2Vec3:
        return Jet2Vec3(a.x * s, a.y * s, a.z * s)

    @wp.func
    def jet_mul(s: Jet2Scalar, a: Jet2Vec3) -> Jet2Vec3:
        return Jet2Vec3(s * a.x, s * a.y, s * a.z)

    @wp.func
    def jet_mul(a: NativeVec3, s: Jet2Scalar) -> Jet2Vec3:
        return Jet2Vec3(a[0] * s, a[1] * s, a[2] * s)

    @wp.func
    def jet_mul(s: Jet2Scalar, a: NativeVec3) -> Jet2Vec3:
        return Jet2Vec3(s * a[0], s * a[1], s * a[2])

    # A plain matrix times a vector jet, one row at a time.

    NativeMat22 = matrix((2, 2), dtype)
    NativeMat33 = matrix((3, 3), dtype)

    @wp.func
    def jet_mul(m: NativeMat22, a: Jet2Vec2) -> Jet2Vec2:
        return Jet2Vec2(m[0, 0] * a.x + m[0, 1] * a.y, m[1, 0] * a.x + m[1, 1] * a.y)

    @wp.func
    def jet_mul(m: NativeMat33, a: Jet2Vec3) -> Jet2Vec3:
        return Jet2Vec3(
            m[0, 0] * a.x + m[0, 1] * a.y + m[0, 2] * a.z,
            m[1, 0] * a.x + m[1, 1] * a.y + m[1, 2] * a.z,
            m[2, 0] * a.x + m[2, 1] * a.y + m[2, 2] * a.z,
        )

    # Division by a jet takes one reciprocal and multiplies, rather than dividing
    # every component.

    @wp.func
    def jet_div(a: Jet2Vec2, s: dtype) -> Jet2Vec2:
        return Jet2Vec2(a.x / s, a.y / s)

    @wp.func
    def jet_div(a: Jet2Vec2, s: Jet2Scalar) -> Jet2Vec2:
        # Divide each component, rather than multiply by 1 / s, whose second derivative
        # overflows for small s.
        return Jet2Vec2(a.x / s, a.y / s)

    @wp.func
    def jet_div(a: NativeVec2, s: Jet2Scalar) -> Jet2Vec2:
        return Jet2Vec2(a[0] / s, a[1] / s)

    @wp.func
    def jet_div(a: Jet2Vec3, s: dtype) -> Jet2Vec3:
        return Jet2Vec3(a.x / s, a.y / s, a.z / s)

    @wp.func
    def jet_div(a: Jet2Vec3, s: Jet2Scalar) -> Jet2Vec3:
        # Divide each component, rather than multiply by 1 / s, whose second derivative
        # overflows for small s.
        return Jet2Vec3(a.x / s, a.y / s, a.z / s)

    @wp.func
    def jet_div(a: NativeVec3, s: Jet2Scalar) -> Jet2Vec3:
        return Jet2Vec3(a[0] / s, a[1] / s, a[2] / s)

    @wp.func
    def jet_dot(a: Jet2Vec2, b: Jet2Vec2) -> Jet2Scalar:
        return a.x * b.x + a.y * b.y

    @wp.func
    def jet_dot(a: Jet2Vec2, b: NativeVec2) -> Jet2Scalar:
        return a.x * b[0] + a.y * b[1]

    @wp.func
    def jet_dot(a: NativeVec2, b: Jet2Vec2) -> Jet2Scalar:
        return a[0] * b.x + a[1] * b.y

    @wp.func
    def jet_dot(a: Jet2Vec3, b: Jet2Vec3) -> Jet2Scalar:
        return a.x * b.x + a.y * b.y + a.z * b.z

    @wp.func
    def jet_dot(a: Jet2Vec3, b: NativeVec3) -> Jet2Scalar:
        return a.x * b[0] + a.y * b[1] + a.z * b[2]

    @wp.func
    def jet_dot(a: NativeVec3, b: Jet2Vec3) -> Jet2Scalar:
        return a[0] * b.x + a[1] * b.y + a[2] * b.z

    @wp.func
    def jet_cw_mul(a: Jet2Vec2, b: Jet2Vec2) -> Jet2Vec2:
        return Jet2Vec2(a.x * b.x, a.y * b.y)

    @wp.func
    def jet_cw_mul(a: Jet2Vec2, b: NativeVec2) -> Jet2Vec2:
        return Jet2Vec2(a.x * b[0], a.y * b[1])

    @wp.func
    def jet_cw_mul(a: NativeVec2, b: Jet2Vec2) -> Jet2Vec2:
        return Jet2Vec2(a[0] * b.x, a[1] * b.y)

    @wp.func
    def jet_cw_mul(a: Jet2Vec3, b: Jet2Vec3) -> Jet2Vec3:
        return Jet2Vec3(a.x * b.x, a.y * b.y, a.z * b.z)

    @wp.func
    def jet_cw_mul(a: Jet2Vec3, b: NativeVec3) -> Jet2Vec3:
        return Jet2Vec3(a.x * b[0], a.y * b[1], a.z * b[2])

    @wp.func
    def jet_cw_mul(a: NativeVec3, b: Jet2Vec3) -> Jet2Vec3:
        return Jet2Vec3(a[0] * b.x, a[1] * b.y, a[2] * b.z)

    @wp.func
    def jet_cw_div(a: Jet2Vec2, b: Jet2Vec2) -> Jet2Vec2:
        return Jet2Vec2(a.x / b.x, a.y / b.y)

    @wp.func
    def jet_cw_div(a: Jet2Vec2, b: NativeVec2) -> Jet2Vec2:
        return Jet2Vec2(a.x / b[0], a.y / b[1])

    @wp.func
    def jet_cw_div(a: NativeVec2, b: Jet2Vec2) -> Jet2Vec2:
        return Jet2Vec2(a[0] / b.x, a[1] / b.y)

    @wp.func
    def jet_cw_div(a: Jet2Vec3, b: Jet2Vec3) -> Jet2Vec3:
        return Jet2Vec3(a.x / b.x, a.y / b.y, a.z / b.z)

    @wp.func
    def jet_cw_div(a: Jet2Vec3, b: NativeVec3) -> Jet2Vec3:
        return Jet2Vec3(a.x / b[0], a.y / b[1], a.z / b[2])

    @wp.func
    def jet_cw_div(a: NativeVec3, b: Jet2Vec3) -> Jet2Vec3:
        return Jet2Vec3(a[0] / b.x, a[1] / b.y, a[2] / b.z)

    @wp.func
    def jet_length_sq(v: Jet2Vec2) -> Jet2Scalar:
        return jet_dot(v, v)

    @wp.func
    def jet_length_sq(v: Jet2Vec3) -> Jet2Scalar:
        return jet_dot(v, v)

    @wp.func
    def jet_length(v: Jet2Vec2) -> Jet2Scalar:
        # Like native wp.length(), a zero-length vector has a zero gradient, and its
        # Hessian, which is undefined there, is taken as zero too.
        length_sq = jet_dot(v, v)
        if length_sq.value == dtype(0.0):
            return Jet2Scalar(dtype(0.0), Grad(), Hess())
        return jet_sqrt(length_sq)

    @wp.func
    def jet_length(v: Jet2Vec3) -> Jet2Scalar:
        # Like native wp.length(), a zero-length vector has a zero gradient, and its
        # Hessian, which is undefined there, is taken as zero too.
        length_sq = jet_dot(v, v)
        if length_sq.value == dtype(0.0):
            return Jet2Scalar(dtype(0.0), Grad(), Hess())
        return jet_sqrt(length_sq)

    @wp.func
    def jet_normalize(v: Jet2Vec2) -> Jet2Vec2:
        # Like the native normalize(), a zero-length vector normalizes to zero.
        length_sq = jet_dot(v, v)
        if length_sq.value > dtype(0.0):
            return jet_div(v, jet_sqrt(length_sq))
        zero = scalar_constant(dtype(0.0))
        return Jet2Vec2(zero, zero)

    @wp.func
    def jet_normalize(v: Jet2Vec3) -> Jet2Vec3:
        # Like the native normalize(), a zero-length vector normalizes to zero.
        length_sq = jet_dot(v, v)
        if length_sq.value > dtype(0.0):
            return jet_div(v, jet_sqrt(length_sq))
        zero = scalar_constant(dtype(0.0))
        return Jet2Vec3(zero, zero, zero)

    # Warp defines neither of these for vec2, so they stay on the namespace
    # instead of becoming builtin overloads, as in the first-order space.

    @wp.func
    def jet_perp(v: Jet2Vec2) -> Jet2Vec2:
        return Jet2Vec2(-v.y, v.x)

    @wp.func
    def jet_cross2(a: Jet2Vec2, b: Jet2Vec2) -> Jet2Scalar:
        return a.x * b.y - a.y * b.x

    @wp.func
    def jet_cross2(a: Jet2Vec2, b: NativeVec2) -> Jet2Scalar:
        return a.x * b[1] - a.y * b[0]

    @wp.func
    def jet_cross2(a: NativeVec2, b: Jet2Vec2) -> Jet2Scalar:
        return a[0] * b.y - a[1] * b.x

    @wp.func
    def jet_cross(a: Jet2Vec3, b: Jet2Vec3) -> Jet2Vec3:
        return Jet2Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)

    @wp.func
    def jet_cross(a: Jet2Vec3, b: NativeVec3) -> Jet2Vec3:
        return Jet2Vec3(a.y * b[2] - a.z * b[1], a.z * b[0] - a.x * b[2], a.x * b[1] - a.y * b[0])

    @wp.func
    def jet_cross(a: NativeVec3, b: Jet2Vec3) -> Jet2Vec3:
        return Jet2Vec3(a[1] * b.z - a[2] * b.y, a[2] * b.x - a[0] * b.z, a[0] * b.y - a[1] * b.x)

    # ---- more scalar functions: chain rule with f' and f'' ----

    @wp.func
    def jet_sinh(a: Jet2Scalar) -> Jet2Scalar:
        s = wp.sinh(a.value)
        c = wp.cosh(a.value)
        return _lift(s, c, s, a)

    @wp.func
    def jet_cosh(a: Jet2Scalar) -> Jet2Scalar:
        s = wp.sinh(a.value)
        c = wp.cosh(a.value)
        return _lift(c, s, c, a)

    @wp.func
    def jet_tanh(a: Jet2Scalar) -> Jet2Scalar:
        t = wp.tanh(a.value)
        fp = dtype(1.0) - t * t
        return _lift(t, fp, -dtype(2.0) * t * fp, a)

    @wp.func
    def jet_log2(a: Jet2Scalar) -> Jet2Scalar:
        inv = dtype(1.0) / a.value
        k = dtype(1.0) / dtype(0.6931471805599453)
        return _lift(wp.log2(a.value), k * inv, -k * inv * inv, a)

    @wp.func
    def jet_log10(a: Jet2Scalar) -> Jet2Scalar:
        inv = dtype(1.0) / a.value
        k = dtype(1.0) / dtype(2.302585092994046)
        return _lift(wp.log10(a.value), k * inv, -k * inv * inv, a)

    @wp.func
    def jet_cbrt(a: Jet2Scalar) -> Jet2Scalar:
        c = wp.cbrt(a.value)
        fp = dtype(1.0) / (dtype(3.0) * c * c)
        return _lift(c, fp, -dtype(2.0) * fp / (dtype(3.0) * a.value), a)

    @wp.func
    def jet_erf(a: Jet2Scalar) -> Jet2Scalar:
        fp = dtype(1.1283791670955126) * wp.exp(-a.value * a.value)
        return _lift(wp.erf(a.value), fp, -dtype(2.0) * a.value * fp, a)

    @wp.func
    def jet_floor(a: Jet2Scalar) -> Jet2Scalar:
        return Jet2Scalar(wp.floor(a.value), Grad(), Hess())

    @wp.func
    def jet_lerp(a: Jet2Scalar, b: Jet2Scalar, t: Jet2Scalar) -> Jet2Scalar:
        return a + t * (b - a)

    @wp.func
    def jet_lerp(a: Jet2Scalar, b: Jet2Scalar, t: dtype) -> Jet2Scalar:
        return a + t * (b - a)

    @wp.func
    def jet_lerp(a: dtype, b: dtype, t: Jet2Scalar) -> Jet2Scalar:
        return a + t * (b - a)

    @wp.func
    def jet_smoothstep(a: dtype, b: dtype, x: Jet2Scalar) -> Jet2Scalar:
        t = (x - a) / (b - a)
        if t.value <= dtype(0.0):
            return Jet2Scalar(dtype(0.0), Grad(), Hess())
        if t.value >= dtype(1.0):
            return Jet2Scalar(dtype(1.0), Grad(), Hess())
        return t * t * (dtype(3.0) - dtype(2.0) * t)

    @wp.func
    def jet_smoothstep(a: Jet2Scalar, b: Jet2Scalar, x: Jet2Scalar) -> Jet2Scalar:
        t = (x - a) / (b - a)
        if t.value <= dtype(0.0):
            return Jet2Scalar(dtype(0.0), Grad(), Hess())
        if t.value >= dtype(1.0):
            return Jet2Scalar(dtype(1.0), Grad(), Hess())
        return t * t * (dtype(3.0) - dtype(2.0) * t)

    # ---- vector reductions and component assignment ----

    @wp.func
    def jet_abs(v: Jet2Vec2) -> Jet2Vec2:
        return Jet2Vec2(jet_abs(v.x), jet_abs(v.y))

    @wp.func
    def jet_abs(v: Jet2Vec3) -> Jet2Vec3:
        return Jet2Vec3(jet_abs(v.x), jet_abs(v.y), jet_abs(v.z))

    @wp.func
    def jet_min(v: Jet2Vec2) -> Jet2Scalar:
        return jet_extract(v, int(wp.argmin(wp.vector(v.x.value, v.y.value))))

    @wp.func
    def jet_min(v: Jet2Vec3) -> Jet2Scalar:
        return jet_extract(v, int(wp.argmin(wp.vector(v.x.value, v.y.value, v.z.value))))

    @wp.func
    def jet_max(v: Jet2Vec2) -> Jet2Scalar:
        return jet_extract(v, int(wp.argmax(wp.vector(v.x.value, v.y.value))))

    @wp.func
    def jet_max(v: Jet2Vec3) -> Jet2Scalar:
        return jet_extract(v, int(wp.argmax(wp.vector(v.x.value, v.y.value, v.z.value))))

    @wp.func
    def jet_assign_copy(v: Jet2Vec2, i: int, s: Jet2Scalar) -> Jet2Vec2:
        i = _component_index(i, 2)
        if i == 0:
            return Jet2Vec2(s, v.y)
        return Jet2Vec2(v.x, s)

    @wp.func
    def jet_assign_copy(v: Jet2Vec3, i: int, s: Jet2Scalar) -> Jet2Vec3:
        i = _component_index(i, 3)
        if i == 0:
            return Jet2Vec3(s, v.y, v.z)
        if i == 1:
            return Jet2Vec3(v.x, s, v.z)
        return Jet2Vec3(v.x, v.y, s)

    @wp.func
    def jet_assign_copy(v: Jet2Vec2, i: int, s: dtype) -> Jet2Vec2:
        return jet_assign_copy(v, i, scalar_constant(s))

    @wp.func
    def jet_assign_copy(v: Jet2Vec3, i: int, s: dtype) -> Jet2Vec3:
        return jet_assign_copy(v, i, scalar_constant(s))

    @wp.func
    def scalar_value(x: Jet2Scalar) -> dtype:
        return x.value

    _register("value", scalar_value)

    _register("add", jet_add)
    _register("sub", jet_sub)
    _register("mul", jet_mul)
    _register("div", jet_div)
    _register("pow", jet_pow)
    _register("neg", jet_neg)
    _register("pos", jet_pos)

    _register("sin", jet_sin)
    _register("cos", jet_cos)
    _register("tan", jet_tan)
    _register("asin", jet_asin)
    _register("acos", jet_acos)
    _register("atan", jet_atan)
    _register("atan2", jet_atan2)
    _register("exp", jet_exp)
    _register("log", jet_log)
    _register("sqrt", jet_sqrt)
    _register("abs", jet_abs)
    _register("sign", jet_sign)

    _register("min", jet_min)
    _register("max", jet_max)
    _register("clamp", jet_clamp)
    _register("where", jet_where)

    _register("extract", jet_extract)

    _register("dot", jet_dot)
    _register("cw_mul", jet_cw_mul)
    _register("cw_div", jet_cw_div)
    _register("length", jet_length)
    _register("length_sq", jet_length_sq)
    _register("normalize", jet_normalize)
    _register("cross", jet_cross)

    _register("sinh", jet_sinh)
    _register("cosh", jet_cosh)
    _register("tanh", jet_tanh)
    _register("log2", jet_log2)
    _register("log10", jet_log10)
    _register("cbrt", jet_cbrt)
    _register("erf", jet_erf)
    _register("floor", jet_floor)
    _register("lerp", jet_lerp)
    _register("smoothstep", jet_smoothstep)
    _register("assign_copy", jet_assign_copy)

    return SimpleNamespace(
        width=width,
        dtype=dtype,
        scalar=Jet2Scalar,
        vec2=Jet2Vec2,
        vec3=Jet2Vec3,
        grad_type_scalar=Grad,
        hess_type_scalar=Hess,
        native_vec2=NativeVec2,
        native_vec3=NativeVec3,
        constant=scalar_constant,
        variable=scalar_variable,
        make_vec2=vec2_from_scalars,
        make_vec3=vec3_from_scalars,
        constant_vec2=vec2_constant,
        constant_vec3=vec3_constant,
        perp=jet_perp,
        cross2=jet_cross2,
        _slot_jet_types={"scalar": (Jet2Scalar, Grad), "vec2": (Jet2Vec2, None), "vec3": (Jet2Vec3, None)},
    )


def JetSpace2(slots, dtype=None):
    """Return second-order (forward-over-forward) jet types for a set of differentiated inputs.

    .. admonition:: Experimental

        Forward-mode jets are experimental and subject to change without a formal deprecation
        cycle.

    A width-k second-order jet tracks value, gradient, and the full k x k
    Hessian; propagating it through a scalar function in one forward pass yields
    the local Hessian with no reverse pass. Per-intermediate state is O(k^2), so
    this suits small k.

    Args:
        slots: The inputs differentiated with respect to. Either an integer
            ``n``, meaning ``n`` scalar inputs, or a sequence of types, one per
            input: ``wp.float32``, ``wp.vec2`` or ``wp.vec3``. A spring over two
            nodes is ``(wp.vec3, wp.vec3)``, with width 6.
        dtype: Scalar type the jets are built on. Inferred from ``slots`` when
            they are Warp types; ``wp.float32`` when ``slots`` is an integer
            or contains only Python ``float``.

    Returns:
        A namespace holding the generated jet types (``scalar``, ``vec2``,
        ``vec3``), the types of a scalar jet's ``grad`` and ``hess`` fields
        (``grad_type_scalar``, ``hess_type_scalar``), the helpers that
        construct jets, and ``jet_type(t)``, which looks a jet type up by the
        Warp type of the value as in :func:`warp.JetSpace`.

    The width is always the total number of scalar components in ``slots``:
    unlike :func:`warp.JetSpace`, there is no ``width`` argument and no
    ``directional``.

    ``seed(*values)`` takes one value per slot and returns one jet per slot,
    with the derivative directions numbered slot by slot in the order the
    components appear, as in :func:`warp.JetSpace`. With ``spring_energy`` an
    ordinary ``@wp.func`` that is generic over its argument types::

        J2 = wp.JetSpace2((wp.vec3, wp.vec3))  # two 3D points, width 6


        @wp.func
        def spring_hessian(x0: wp.vec3, x1: wp.vec3):
            x0_jet, x1_jet = J2.seed(x0, x1)
            return spring_energy(x0_jet, x1_jet).hess  # the 6 x 6 Hessian

    Vector jets support the same arithmetic and geometry as first-order ones
    (``dot``, ``cross``, ``length``, ``normalize``, indexing), so the same
    ``spring_energy`` also runs on :func:`warp.JetSpace` jets. A second-order
    vector jet is one scalar jet per component, so it carries the same state as
    the function written out per component. ``variable(x, direction)`` seeds a
    single scalar, for use in a loop.

    As with :func:`warp.JetSpace`, the width is a compile-time constant and the
    types are for local derivatives of fixed, statically known arity. The
    quadratic state makes that limit tighter here: register pressure and compile
    time both grow with ``width**2``, so for wider energies prefer a first-order
    jet with a reverse sweep over it (via :class:`warp.Tape`, or in-kernel with
    :func:`warp.grad`), which reaches the same Hessian with linear state.
    """
    kinds, dtype = _normalize_slots(slots, dtype)
    if any(kind not in ("scalar", "vec2", "vec3") for kind in kinds):
        raise TypeError(
            "Second-order jets support scalar, vec2 and vec3 slots, not matrix slots. Pass a matrix as its "
            "columns or scalar components, or use a first-order JetSpace with a reverse sweep."
        )
    width = sum(len(_slot_components(kind)) for kind in kinds)

    key = (width, dtype)
    base = _CACHE2.get(key)
    if base is None:
        base = _make_jet_space2(width, dtype)
        _CACHE2[key] = base

    return _layout_namespace(base, kinds, width, dtype, second_order=True)
