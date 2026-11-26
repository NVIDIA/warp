# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fields whose values can be perturbed along test and trial directions."""

from typing import ClassVar, NamedTuple

import warp as wp
from warp._src.fem import cache, operator
from warp._src.fem.field.field import GeometryField
from warp._src.fem.field.virtual import LocalAdjointField, LocalTestField, LocalTrialField, TestField, TrialField
from warp._src.fem.types import NULL_DOF_INDEX, NULL_ELEMENT_INDEX, DofIndex
from warp._src.jet import JetSpace, JetSpace2
from warp._src.types import type_is_matrix, type_is_vector, type_scalar_type, types_equal


class BoundField(GeometryField):
    """A valued field and up to two finite-element variation directions.

    Args:
        current: Valued field at which the energy is evaluated.
        test: Test direction for residual rows.
        trial: Trial direction for tangent columns.
    """

    _caches_current_values: ClassVar[bool] = True

    _dynamic_attribute_constructors: ClassVar = {
        "EvalArg": lambda obj: obj._make_eval_arg(),
        "ElementEvalArg": lambda obj: obj._make_element_eval_arg(),
        "fill_cache": lambda obj: obj._make_fill_cache(),
        "eval_degree": lambda obj: obj._make_eval_degree(),
        "eval_inner": lambda obj: obj._make_eval("eval_inner"),
        "eval_outer": lambda obj: obj._make_eval("eval_outer"),
        "eval_grad_inner": lambda obj: obj._make_eval("eval_grad_inner"),
        "eval_grad_outer": lambda obj: obj._make_eval("eval_grad_outer"),
        "eval_div_inner": lambda obj: obj._make_eval("eval_div_inner"),
        "eval_div_outer": lambda obj: obj._make_eval("eval_div_outer"),
        "eval_reference_grad_inner": lambda obj: obj._make_eval("eval_reference_grad_inner"),
        "eval_reference_grad_outer": lambda obj: obj._make_eval("eval_reference_grad_outer"),
    }

    def __init__(self, current: GeometryField, test: TestField | None, trial: TrialField | None):
        if not isinstance(current, GeometryField) or current.dtype is None:
            raise TypeError("The current field must be a valued geometry field with a known value type")
        if test is None and trial is None:
            raise ValueError("A binding requires a test or trial field")
        # Local directions replace the global ones for dispatched assembly
        if test is not None and not isinstance(test, (TestField, LocalTestField)):
            raise TypeError("The test direction must be a FEM test field")
        if trial is not None and not isinstance(trial, (TrialField, LocalTrialField)):
            raise TypeError("The trial direction must be a FEM trial field")

        for role, direction in (("test", test), ("trial", trial)):
            if direction is None:
                continue
            if direction.geometry != current.geometry or direction.element_kind != current.element_kind:
                raise ValueError(f"The {role} direction must use the current field's geometry and element kind")
            if not types_equal(direction.dtype, current.dtype):
                raise TypeError(f"The {role} direction must have the current field's value type")

        self.current = current
        self.test = test
        self.trial = trial
        self._test_field = test if test is not None else current
        self._trial_field = trial if trial is not None else current
        self.SampleType = current.geometry.sample_type
        self._cached_evals = ()
        cache.setup_dynamic_attributes(self)

    @property
    def name(self):
        cached = "_".join(_CACHE_MEMBERS[name] for name in self._cached_evals)
        return f"Bound_{self.current.name}_{self.test.name if self.test else 'None'}_{self.trial.name if self.trial else 'None'}_C{cached}"

    @property
    def geometry(self):
        return self.current.geometry

    @property
    def element_kind(self):
        return self.current.element_kind

    @property
    def dtype(self):
        return self.current.dtype

    @property
    def degree(self):
        # The test direction stands in for the separate test field of a hand-written residual.
        # The trial direction adds nothing, so tangents use the same default quadrature as residuals.
        return self.current.degree + (self.test.degree if self.test is not None else 0)

    def gradient_valid(self):
        return all(f.gradient_valid() for f in (self.current, self.test, self.trial) if f is not None)

    def divergence_valid(self):
        return all(f.divergence_valid() for f in (self.current, self.test, self.trial) if f is not None)

    def notify_operator_usage(self, ops):
        for field in (self.current, self.test, self.trial):
            if field is not None:
                field.notify_operator_usage(ops)
        if self._caches_current_values:
            self._cached_evals = tuple(
                name for op, name in _CACHED_OPERATORS.items() if op in ops and getattr(self.current, name, None)
            )
        # Local directions change their layout and evaluators with the operators in use
        cache.setup_dynamic_attributes(self)

    def _rebind(self, test, trial, jet_layout=None) -> "BoundField":
        """Return this binding with its directions replaced, for instance by local dispatch fields."""
        return BoundField(self.current, test, trial)

    def fill_eval_arg(self, arg, device):
        self.current.fill_eval_arg(arg.current, device)
        self._test_field.fill_eval_arg(arg.test, device)
        self._trial_field.fill_eval_arg(arg.trial, device)
        arg.test_coeff = self.geometry.scalar_type(0.0)
        arg.trial_coeff = self.geometry.scalar_type(0.0)
        arg.cache_element = NULL_ELEMENT_INDEX

    def _make_eval_arg(self):
        cached = self._cached_evals
        coords_type = self.SampleType.vars["element_coords"].type
        entry_types = {name: _cache_entry_type(self, name) for name in cached}
        inner_entry = entry_types.get("eval_inner")
        outer_entry = entry_types.get("eval_outer")
        grad_inner_entry = entry_types.get("eval_grad_inner")
        grad_outer_entry = entry_types.get("eval_grad_outer")
        div_inner_entry = entry_types.get("eval_div_inner")
        div_outer_entry = entry_types.get("eval_div_outer")

        @cache.dynamic_struct(suffix=self.name)
        class EvalArg:
            current: self.current.EvalArg
            test: self._test_field.EvalArg
            trial: self._trial_field.EvalArg
            test_coeff: self.geometry.scalar_type
            trial_coeff: self.geometry.scalar_type
            # Field and direction values at one sample, reused by the perturbed evaluations of an energy
            cache_element: int
            cache_coords: coords_type
            cache_test_dof: DofIndex
            cache_trial_dof: DofIndex
            if "eval_inner" in cached:
                cache_inner: inner_entry
            if "eval_outer" in cached:
                cache_outer: outer_entry
            if "eval_grad_inner" in cached:
                cache_grad_inner: grad_inner_entry
            if "eval_grad_outer" in cached:
                cache_grad_outer: grad_outer_entry
            if "eval_div_inner" in cached:
                cache_div_inner: div_inner_entry
            if "eval_div_outer" in cached:
                cache_div_outer: div_outer_entry

        return EvalArg

    def _make_fill_cache(self):
        """Return a function storing the field and direction values at a sample into the runtime argument."""
        cached = self._cached_evals
        element_arg_type = self.ElementEvalArg.vars["elt_arg"].type
        fill = {name: _make_cache_entry_filler(self, name) for name in cached}
        fill_inner = fill.get("eval_inner")
        fill_outer = fill.get("eval_outer")
        fill_grad_inner = fill.get("eval_grad_inner")
        fill_grad_outer = fill.get("eval_grad_outer")
        fill_div_inner = fill.get("eval_div_inner")
        fill_div_outer = fill.get("eval_div_outer")

        @cache.dynamic_func(suffix=self.name)
        def fill_cache(elt_arg: element_arg_type, eval_arg: self.EvalArg, s: self.SampleType):
            filled = eval_arg
            filled.cache_element = s.element_index
            filled.cache_coords = s.element_coords
            filled.cache_test_dof = s.test_dof
            filled.cache_trial_dof = s.trial_dof
            if wp.static("eval_inner" in cached):
                filled.cache_inner = fill_inner(elt_arg, eval_arg, s)
            if wp.static("eval_outer" in cached):
                filled.cache_outer = fill_outer(elt_arg, eval_arg, s)
            if wp.static("eval_grad_inner" in cached):
                filled.cache_grad_inner = fill_grad_inner(elt_arg, eval_arg, s)
            if wp.static("eval_grad_outer" in cached):
                filled.cache_grad_outer = fill_grad_outer(elt_arg, eval_arg, s)
            if wp.static("eval_div_inner" in cached):
                filled.cache_div_inner = fill_div_inner(elt_arg, eval_arg, s)
            if wp.static("eval_div_outer" in cached):
                filled.cache_div_outer = fill_div_outer(elt_arg, eval_arg, s)
            return filled

        return fill_cache

    def _with_jets(self, order: int) -> "BoundField":
        """Return a copy of this binding whose evaluations are jets of the given derivative order."""
        return _JetBoundField(self.current, self.test, self.trial, order)

    def _with_reverse_over_forward(self) -> "BoundField":
        """Return a copy of this binding whose evaluations are width-1 jets over differentiable coefficients."""
        return _RofBoundField(self.current, self.test, self.trial)

    def _make_element_eval_arg(self):
        element_arg_type = self.current.ElementEvalArg.vars["elt_arg"].type

        @cache.dynamic_struct(suffix=self.name)
        class ElementEvalArg:
            elt_arg: element_arg_type
            eval_arg: self.EvalArg

        return ElementEvalArg

    def _make_eval_degree(self):
        # The energy sees the current field; its degree does not depend on the bound directions
        @cache.dynamic_func(suffix=self.name)
        def eval_degree(args: self.ElementEvalArg):
            return self.current.eval_degree(self.current.ElementEvalArg(args.elt_arg, args.eval_arg.current))

        return eval_degree

    def _make_eval(self, operator_name):
        current_op = getattr(self.current, operator_name, None)
        test_op = getattr(self._test_field, operator_name, None)
        trial_op = getattr(self._trial_field, operator_name, None)
        if (
            current_op is None
            or (self.test is not None and test_op is None)
            or (self.trial is not None and trial_op is None)
        ):
            return None

        has_test = self.test is not None
        has_trial = self.trial is not None
        cached = operator_name in self._cached_evals
        read_cache = _make_cache_reader(self, operator_name) if cached else None

        @cache.dynamic_func(suffix=(self.name, operator_name))
        def evaluate(args: self.ElementEvalArg, s: self.SampleType):
            if wp.static(cached):
                # The cache only holds values for the sample it was filled at
                if (
                    args.eval_arg.cache_element == s.element_index
                    and args.eval_arg.cache_coords == s.element_coords
                    and args.eval_arg.cache_test_dof == s.test_dof
                    and args.eval_arg.cache_trial_dof == s.trial_dof
                ):
                    entry = read_cache(args.eval_arg)
                    value = entry.value
                    if wp.static(has_test):
                        value += args.eval_arg.test_coeff * entry.test
                    if wp.static(has_trial):
                        value += args.eval_arg.trial_coeff * entry.trial
                    return value
            value = current_op(self.current.ElementEvalArg(args.elt_arg, args.eval_arg.current), s)
            if wp.static(has_test):
                value += args.eval_arg.test_coeff * test_op(
                    self._test_field.ElementEvalArg(args.elt_arg, args.eval_arg.test), s
                )
            if wp.static(has_trial):
                value += args.eval_arg.trial_coeff * trial_op(
                    self._trial_field.ElementEvalArg(args.elt_arg, args.eval_arg.trial), s
                )
            return value

        return evaluate


_CACHED_OPERATORS = {
    operator.inner: "eval_inner",
    operator.outer: "eval_outer",
    operator.grad: "eval_grad_inner",
    operator.grad_outer: "eval_grad_outer",
    operator.div: "eval_div_inner",
    operator.div_outer: "eval_div_outer",
}

# EvalArg member holding the cache entry of each cached evaluation
_CACHE_MEMBERS = {
    "eval_inner": "cache_inner",
    "eval_outer": "cache_outer",
    "eval_grad_inner": "cache_grad_inner",
    "eval_grad_outer": "cache_grad_outer",
    "eval_div_inner": "cache_div_inner",
    "eval_div_outer": "cache_div_outer",
}


def _cache_entry_type(field: BoundField, operator_name: str):
    """Struct holding the field, test, and trial values of one operator at a sample."""
    value_type = getattr(field.current, _OPERATOR_VALUE_TYPES[operator_name])

    @cache.dynamic_struct(suffix=cache.pod_type_key(value_type))
    class CacheEntry:
        value: value_type
        test: value_type
        trial: value_type

    return CacheEntry


def _make_cache_entry_filler(field: BoundField, operator_name: str):
    current_op = getattr(field.current, operator_name)
    test_op = getattr(field._test_field, operator_name)
    trial_op = getattr(field._trial_field, operator_name)
    has_test = field.test is not None
    has_trial = field.trial is not None
    Entry = _cache_entry_type(field, operator_name)
    element_arg_type = field.ElementEvalArg.vars["elt_arg"].type

    @cache.dynamic_func(suffix=(field.name, operator_name))
    def fill_entry(elt_arg: element_arg_type, eval_arg: field.EvalArg, s: field.SampleType):
        entry = Entry()
        entry.value = current_op(field.current.ElementEvalArg(elt_arg, eval_arg.current), s)
        if wp.static(has_test):
            entry.test = test_op(field._test_field.ElementEvalArg(elt_arg, eval_arg.test), s)
        if wp.static(has_trial):
            entry.trial = trial_op(field._trial_field.ElementEvalArg(elt_arg, eval_arg.trial), s)
        return entry

    return fill_entry


def _make_cache_reader(field: BoundField, operator_name: str):
    @cache.dynamic_func(suffix=(field.name, operator_name))
    def read_cache(eval_arg: field.EvalArg):
        if wp.static(operator_name == "eval_inner"):
            return eval_arg.cache_inner
        elif wp.static(operator_name == "eval_outer"):
            return eval_arg.cache_outer
        elif wp.static(operator_name == "eval_grad_inner"):
            return eval_arg.cache_grad_inner
        elif wp.static(operator_name == "eval_grad_outer"):
            return eval_arg.cache_grad_outer
        elif wp.static(operator_name == "eval_div_inner"):
            return eval_arg.cache_div_inner
        else:
            return eval_arg.cache_div_outer

    return read_cache


_OPERATOR_VALUE_TYPES = {
    "eval_inner": "dtype",
    "eval_outer": "dtype",
    "eval_grad_inner": "gradient_dtype",
    "eval_grad_outer": "gradient_dtype",
    "eval_div_inner": "divergence_dtype",
    "eval_div_outer": "divergence_dtype",
    "eval_reference_grad_inner": "reference_gradient_dtype",
    "eval_reference_grad_outer": "reference_gradient_dtype",
}


_BATCHED_DOF_INDEX = wp.constant(DofIndex(0, 0))
"""Test and trial degree of freedom of the quadrature samples of batched dispatch assembly.

Batched evaluators carry all local components at once and ignore its value. Any non-null value
marks the sample as carrying the directions; samples from ``lookup()`` or a discrete field's
``at_node()`` have null degrees of freedom, so their values get no direction, as with generic assembly.
"""


class _JetLayout(NamedTuple):
    """Assignment of the test and trial directions to jet derivative directions.

    Per-pair layouts carry one test direction and one trial direction, taken from
    the sample's test and trial degrees of freedom. Batched layouts carry every
    local value and gradient component of dispatched test and trial fields at
    once, numbered ``taylor_dof * value_dof_count + value_dof`` from each offset.
    Shared layouts reuse the test directions for the trial field when both are
    the same local functions of the same binding.
    """

    width: int
    test_offset: int
    trial_offset: int
    shared: bool = False
    batched: bool = False

    @property
    def key(self) -> str:
        return f"{self.width}_{self.test_offset}_{self.trial_offset}_{int(self.shared)}{int(self.batched)}"


class _JetBoundField(BoundField):
    """Binding whose evaluations are jets along its test and trial directions.

    Order 1 evaluates ``current`` as a first-order jet whose derivatives are the
    test directions. Order 2 uses a second-order jet over the test and trial
    directions, so mixed second derivatives of an energy are its tangent entries.
    """

    _caches_current_values: ClassVar[bool] = False

    def __init__(self, current: GeometryField, test, trial, order: int, layout: _JetLayout | None = None):
        self._jet_order = order
        if layout is None:
            layout = _JetLayout(width=order, test_offset=0, trial_offset=order - 1)
        self._jet_layout = layout
        super().__init__(current, test, trial)

    @property
    def name(self):
        return f"{super().name}_Jet{self._jet_order}_{self._jet_layout.key}"

    def _make_fill_cache(self):
        return None

    def _rebind(self, test, trial, jet_layout: _JetLayout | None = None) -> BoundField:
        return _JetBoundField(self.current, test, trial, self._jet_order, jet_layout)

    def fill_eval_arg(self, arg, device):
        self.current.fill_eval_arg(arg.current, device)
        self._test_field.fill_eval_arg(arg.test, device)
        self._trial_field.fill_eval_arg(arg.trial, device)

    def _make_eval_arg(self):
        @cache.dynamic_struct(suffix=self.name)
        class EvalArg:
            current: self.current.EvalArg
            test: self._test_field.EvalArg
            trial: self._trial_field.EvalArg

        return EvalArg

    def _make_eval(self, operator_name):
        current_op = getattr(self.current, operator_name, None)
        test_op = getattr(self._test_field, operator_name, None)
        trial_op = getattr(self._trial_field, operator_name, None)
        if (
            current_op is None
            or (self.test is not None and test_op is None)
            or (self.trial is not None and trial_op is None)
        ):
            return None

        layout = self._jet_layout
        batched = layout.batched
        if batched and operator_name not in _LOCAL_OPERATOR_DOFS:
            # Dispatched directions only carry values and world-space gradients
            return None

        jet_ops = self._make_jet_ops(getattr(self.current, _OPERATOR_VALUE_TYPES[operator_name]))
        if jet_ops is None:
            return None
        Coeff, set_direction, make_jet = jet_ops

        has_test = self.test is not None
        has_trial = self.trial is not None and not layout.shared
        TEST_OFFSET = layout.test_offset
        TRIAL_OFFSET = layout.trial_offset
        TEST_BEGIN, TEST_END, TEST_VALUE_DOFS = _local_direction_range(self.test, operator_name, batched)
        TRIAL_BEGIN, TRIAL_END, TRIAL_VALUE_DOFS = _local_direction_range(self.trial, operator_name, batched)

        @cache.dynamic_func(suffix=(self.name, operator_name))
        def evaluate(args: self.ElementEvalArg, s: self.SampleType):
            value = current_op(self.current.ElementEvalArg(args.elt_arg, args.eval_arg.current), s)
            coeff = Coeff()
            if wp.static(has_test):
                test_arg = self._test_field.ElementEvalArg(args.elt_arg, args.eval_arg.test)
                if wp.static(batched):
                    # Every local component gets a direction, except at samples without a test degree of freedom
                    if not (s.test_dof == NULL_DOF_INDEX):
                        for taylor_dof in range(TEST_BEGIN, TEST_END):
                            for value_dof in range(TEST_VALUE_DOFS):
                                direction_sample = self.SampleType(
                                    s.element_index,
                                    s.element_coords,
                                    s.qp_index,
                                    s.qp_weight,
                                    DofIndex(taylor_dof, value_dof),
                                    s.trial_dof,
                                )
                                coeff = set_direction(
                                    coeff,
                                    TEST_OFFSET + taylor_dof * TEST_VALUE_DOFS + value_dof,
                                    test_op(test_arg, direction_sample),
                                )
                else:
                    coeff = set_direction(coeff, TEST_OFFSET, test_op(test_arg, s))
            if wp.static(has_trial):
                trial_arg = self._trial_field.ElementEvalArg(args.elt_arg, args.eval_arg.trial)
                if wp.static(batched):
                    if not (s.trial_dof == NULL_DOF_INDEX):
                        for taylor_dof in range(TRIAL_BEGIN, TRIAL_END):
                            for value_dof in range(TRIAL_VALUE_DOFS):
                                direction_sample = self.SampleType(
                                    s.element_index,
                                    s.element_coords,
                                    s.qp_index,
                                    s.qp_weight,
                                    s.test_dof,
                                    DofIndex(taylor_dof, value_dof),
                                )
                                coeff = set_direction(
                                    coeff,
                                    TRIAL_OFFSET + taylor_dof * TRIAL_VALUE_DOFS + value_dof,
                                    trial_op(trial_arg, direction_sample),
                                )
                else:
                    coeff = set_direction(coeff, TRIAL_OFFSET, trial_op(trial_arg, s))
            return make_jet(value, coeff)

        return evaluate

    def _make_jet_ops(self, value_type):
        """Return the derivative coefficient type and functions filling it and building jets.

        The coefficients hold one column per jet direction for each scalar
        component of ``value_type``, with matrices flattened row-major. Returns
        ``None`` when no jet type of this order can hold ``value_type``.
        """
        if value_type is None:
            return None

        width = self._jet_layout.width
        scalar_type = type_scalar_type(value_type)
        suffix = (self.name, cache.pod_type_key(value_type))

        if type_is_matrix(value_type):
            kind = "matrix"
            ROWS, COLS = value_type._shape_
        elif type_is_vector(value_type):
            kind = "vector"
            ROWS, COLS = value_type._length_, 1
        else:
            kind = "scalar"
            ROWS, COLS = 1, 1

        if self._jet_order == 1:
            try:
                J = JetSpace((value_type,), width=width)
            except TypeError:
                return None
            Jet = J.jet_type(value_type)
            Coeff = J.grad_type(value_type)

            @cache.dynamic_func(suffix=suffix)
            def make_jet(value: value_type, coeff: Coeff):
                return Jet(value, coeff)

        else:
            if kind == "matrix" or (kind == "vector" and ROWS not in (2, 3)):
                return None
            J = JetSpace2(width, dtype=scalar_type)
            Hess = J.hess_type_scalar
            if kind == "scalar":
                Coeff = J.grad_type_scalar

                @cache.dynamic_func(suffix=suffix)
                def make_jet(value: value_type, coeff: Coeff):
                    return J.scalar(value, coeff, Hess())

            else:
                Coeff = cache.cached_mat_type(shape=(ROWS, width), dtype=scalar_type)
                if ROWS == 2:

                    @cache.dynamic_func(suffix=suffix)
                    def make_jet(value: value_type, coeff: Coeff):
                        return J.vec2(J.scalar(value[0], coeff[0], Hess()), J.scalar(value[1], coeff[1], Hess()))

                else:

                    @cache.dynamic_func(suffix=suffix)
                    def make_jet(value: value_type, coeff: Coeff):
                        return J.vec3(
                            J.scalar(value[0], coeff[0], Hess()),
                            J.scalar(value[1], coeff[1], Hess()),
                            J.scalar(value[2], coeff[2], Hess()),
                        )

        @cache.dynamic_func(suffix=suffix)
        def set_direction(coeff: Coeff, direction: int, d: value_type):
            if wp.static(kind == "scalar"):
                coeff[direction] = d
            elif wp.static(kind == "vector"):
                for i in range(ROWS):
                    coeff[i, direction] = d[i]
            else:
                for i in range(ROWS):
                    for j in range(COLS):
                        coeff[i * COLS + j, direction] = d[i, j]
            return coeff

        return Coeff, set_direction, make_jet


class _RofBoundField(BoundField):
    """Binding for reverse-over-forward tangents.

    Under generic assembly, the binding follows the single test and trial directions of the sample's
    degrees of freedom: ``a`` is one scalar and the value is ``current + a * test``. Dispatched assembly,
    with local test and trial fields, instead gives every local component at once.

    Evaluates ``current + sum_i a[i] * test_i`` as the value of a width-1 first-order jet whose
    derivative is the trial direction selected by ``trial_index``. The test directions ``test_i`` are the
    local value and gradient components of the dispatched test field. The coefficients ``a`` are the
    variables that Warp's reverse mode differentiates, so the adjoint of ``a`` after differentiating the
    jet's derivative coefficient is one column of the tangent matrix. Only first-order jets are needed,
    so matrix-valued operators such as ``grad`` of a vector field are supported.
    """

    def __init__(self, current: GeometryField, test, trial):
        self._dispatched = isinstance(test, LocalTestField) or isinstance(trial, LocalTrialField)
        # Generic assembly evaluates one sample many times, so it stores the field and direction values once
        self._caches_current_values = not self._dispatched
        super().__init__(current, test, trial)

    @property
    def name(self):
        return f"{super().name}_Rof"

    @property
    def test_direction_count(self) -> int:
        """Number of local test components, i.e. the length of the differentiable coefficient vector."""
        # Global test fields, before dispatch rebinds them to local ones, have no local components yet
        if not isinstance(self.test, LocalTestField):
            return 1
        return self.test.TAYLOR_DOF_COUNT * self.test.value_dof_count

    @property
    def trial_direction_count(self) -> int:
        """Number of local trial components that a tangent pass loops over."""
        if not isinstance(self.trial, LocalTrialField):
            return 0
        return self.trial.TAYLOR_DOF_COUNT * self.trial.value_dof_count

    def _make_fill_cache(self):
        return None if self._dispatched else super()._make_fill_cache()

    def _rebind(self, test, trial, jet_layout: _JetLayout | None = None) -> BoundField:
        return _RofBoundField(self.current, test, trial)

    def fill_eval_arg(self, arg, device):
        if not self._dispatched:
            # The perturbation coefficient `test_coeff` is the variable that reverse mode differentiates
            super().fill_eval_arg(arg, device)
            return
        self.current.fill_eval_arg(arg.current, device)
        self._test_field.fill_eval_arg(arg.test, device)
        self._trial_field.fill_eval_arg(arg.trial, device)
        arg.trial_index = 0

    def _make_eval_arg(self):
        if not self._dispatched:
            return super()._make_eval_arg()

        coeff_type = cache.cached_vec_type(self.test_direction_count, type_scalar_type(self.current.dtype))

        @cache.dynamic_struct(suffix=self.name)
        class EvalArg:
            current: self.current.EvalArg
            test: self._test_field.EvalArg
            trial: self._trial_field.EvalArg
            # Differentiable coefficients of the test directions; their adjoint is a tangent column
            a: coeff_type
            # Local trial component that the jet's derivative follows
            trial_index: int

        return EvalArg

    def _make_eval(self, operator_name):
        current_op = getattr(self.current, operator_name, None)
        test_op = getattr(self._test_field, operator_name, None)
        trial_op = getattr(self._trial_field, operator_name, None)
        generic = not self._dispatched
        if (
            current_op is None
            or (self.test is not None and test_op is None)
            or (self.trial is not None and trial_op is None)
            or (not generic and operator_name not in _LOCAL_OPERATOR_DOFS)
        ):
            return None

        # Width-1 jet constructors; the layout only serves to name them
        jet_ops = _JetBoundField(self.current, self.test, self.trial, 1, _JetLayout(1, 0, 0))._make_jet_ops(
            getattr(self.current, _OPERATOR_VALUE_TYPES[operator_name])
        )
        if jet_ops is None:
            return None
        Coeff, set_direction, make_jet = jet_ops

        has_test = self.test is not None
        has_trial = self.trial is not None

        if generic:
            cached = operator_name in self._cached_evals
            read_cache = _make_cache_reader(self, operator_name) if cached else None

            @cache.dynamic_func(suffix=(self.name, operator_name))
            def evaluate_pair(args: self.ElementEvalArg, s: self.SampleType):
                if wp.static(cached):
                    # The cache only holds values for the sample it was filled at
                    if (
                        args.eval_arg.cache_element == s.element_index
                        and args.eval_arg.cache_coords == s.element_coords
                        and args.eval_arg.cache_test_dof == s.test_dof
                        and args.eval_arg.cache_trial_dof == s.trial_dof
                    ):
                        entry = read_cache(args.eval_arg)
                        cached_value = entry.value
                        if wp.static(has_test):
                            cached_value += args.eval_arg.test_coeff * entry.test
                        cached_coeff = Coeff()
                        if wp.static(has_trial):
                            cached_coeff = set_direction(cached_coeff, 0, entry.trial)
                        return make_jet(cached_value, cached_coeff)

                value = current_op(self.current.ElementEvalArg(args.elt_arg, args.eval_arg.current), s)
                if wp.static(has_test):
                    value += args.eval_arg.test_coeff * test_op(
                        self._test_field.ElementEvalArg(args.elt_arg, args.eval_arg.test), s
                    )
                coeff = Coeff()
                if wp.static(has_trial):
                    coeff = set_direction(
                        coeff,
                        0,
                        trial_op(self._trial_field.ElementEvalArg(args.elt_arg, args.eval_arg.trial), s),
                    )
                return make_jet(value, coeff)

            return evaluate_pair

        TEST_BEGIN, TEST_END, TEST_VALUE_DOFS = _local_direction_range(self.test, operator_name, True)
        TRIAL_BEGIN, TRIAL_END, TRIAL_VALUE_DOFS = _local_direction_range(self.trial, operator_name, True)

        @cache.dynamic_func(suffix=(self.name, operator_name))
        def evaluate(args: self.ElementEvalArg, s: self.SampleType):
            value = current_op(self.current.ElementEvalArg(args.elt_arg, args.eval_arg.current), s)
            # Samples without a test or trial degree of freedom, such as from lookup(), get no direction
            if wp.static(has_test):
                test_arg = self._test_field.ElementEvalArg(args.elt_arg, args.eval_arg.test)
                if not (s.test_dof == NULL_DOF_INDEX):
                    for taylor_dof in range(TEST_BEGIN, TEST_END):
                        for value_dof in range(TEST_VALUE_DOFS):
                            direction_sample = self.SampleType(
                                s.element_index,
                                s.element_coords,
                                s.qp_index,
                                s.qp_weight,
                                DofIndex(taylor_dof, value_dof),
                                s.trial_dof,
                            )
                            value += args.eval_arg.a[taylor_dof * TEST_VALUE_DOFS + value_dof] * test_op(
                                test_arg, direction_sample
                            )

            coeff = Coeff()
            if wp.static(has_trial):
                trial_arg = self._trial_field.ElementEvalArg(args.elt_arg, args.eval_arg.trial)
                trial_taylor_dof = args.eval_arg.trial_index // TRIAL_VALUE_DOFS
                trial_value_dof = args.eval_arg.trial_index - trial_taylor_dof * TRIAL_VALUE_DOFS
                # Components that this operator does not read have no direction
                if (
                    not (s.trial_dof == NULL_DOF_INDEX)
                    and trial_taylor_dof >= TRIAL_BEGIN
                    and trial_taylor_dof < TRIAL_END
                ):
                    direction_sample = self.SampleType(
                        s.element_index,
                        s.element_coords,
                        s.qp_index,
                        s.qp_weight,
                        s.test_dof,
                        DofIndex(trial_taylor_dof, trial_value_dof),
                    )
                    coeff = set_direction(coeff, 0, trial_op(trial_arg, direction_sample))
            return make_jet(value, coeff)

        return evaluate


def _local_direction_range(field, operator_name: str, batched: bool):
    """Return the Taylor degrees of freedom of a dispatched field that an operator reads, and its value count."""
    if field is None or not batched:
        return 0, 0, 0

    dof_type = _LOCAL_OPERATOR_DOFS[operator_name]
    if operator_name == "eval_outer" and not field._is_discontinuous:
        dof_type = LocalAdjointField.INNER_DOF
    begin = int(field._TAYLOR_DOF_OFFSETS[dof_type])
    return begin, begin + int(field._TAYLOR_DOF_COUNTS[dof_type]), field.value_dof_count


_LOCAL_OPERATOR_DOFS = {
    "eval_inner": LocalAdjointField.INNER_DOF,
    "eval_outer": LocalAdjointField.OUTER_DOF,
    "eval_grad_inner": LocalAdjointField.INNER_GRAD_DOF,
    "eval_grad_outer": LocalAdjointField.OUTER_GRAD_DOF,
    "eval_div_inner": LocalAdjointField.INNER_GRAD_DOF,
    "eval_div_outer": LocalAdjointField.OUTER_GRAD_DOF,
}


def bind(current: GeometryField, *, test: TestField | None = None, trial: TrialField | None = None) -> BoundField:
    """Bind a valued field to test and trial variation directions.

    Args:
        current: Field at which to evaluate the energy.
        test: Direction that supplies residual rows.
        trial: Direction that supplies tangent columns.

    Returns:
        A field accepted by ``fields=`` in :func:`warp.fem.integrate`.
    """
    return BoundField(current, test, trial)
