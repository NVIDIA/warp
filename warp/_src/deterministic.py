# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic execution mode helpers for Warp codegen and launch.

This module implements the scatter-sort-reduce and two-pass strategies for
making atomic operations produce bit-exact reproducible results.

Two forms of atomic usage are supported:

Scatter/reduce atomics (return value unused):
    ``wp.atomic_add(arr, idx, value)`` or ``arr[idx] += value``
    Strategy: scatter records to a temporary buffer during kernel execution,
    then sort by ``(dest_index, thread_id)`` and reduce in fixed order.

Consumed-return counter atomics (return value used):
    ``slot = wp.atomic_add(counter, index, value)``
    Strategy: two-pass execution. Phase 0 records each reservation with all
    side effects suppressed. A deterministic sort/scan computes per-target
    offsets. Phase 1 re-executes with deterministic slot assignments.

See ``warp.config.deterministic`` for the user-facing configuration modes.
"""

from __future__ import annotations

import ast
import ctypes
import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from warp._src.types import float_types, int32, is_array, type_repr, type_size_in_bytes
from warp.config import DeterministicMode

if TYPE_CHECKING:
    from warp._src.context import Device, KernelHooks, Stream
    from warp._src.types import launch_bounds_t


def _det_dest_size_elements(arr) -> int:
    """Upper bound on element offsets reachable from ``arr.ptr`` for any stride layout.

    ``arr.capacity`` underestimates the span for transposed/non-row-major views.
    Compute the byte offset of the last reachable element from the view pointer
    instead.  Fully broadcast zero-stride views still address one physical
    element.
    """
    if arr.ndim == 0 or arr.size == 0:
        return 0

    element_size = type_size_in_bytes(arr.dtype)
    max_offset_bytes = 0
    for k in range(arr.ndim):
        stride = arr.strides[k]
        if stride > 0:
            max_offset_bytes += (arr.shape[k] - 1) * stride

    if max_offset_bytes == 0:
        return 1
    return int(max_offset_bytes // element_size) + 1


# Reduction operation constants (must match C++ ReduceOp enum in deterministic.cu).
REDUCE_OP_ADD = 0
REDUCE_OP_MIN = 1
REDUCE_OP_MAX = 2

DETERMINISTIC_NOT_GUARANTEED = DeterministicMode.NOT_GUARANTEED
DETERMINISTIC_RUN_TO_RUN = DeterministicMode.RUN_TO_RUN
DETERMINISTIC_GPU_TO_GPU = DeterministicMode.GPU_TO_GPU

DETERMINISTIC_FAMILY_ADD = "add"
DETERMINISTIC_FAMILY_MIN = "min"
DETERMINISTIC_FAMILY_MAX = "max"
DETERMINISTIC_FAMILY_COUNTER = "counter"


def reduce_op_to_family(reduce_op: int) -> str:
    """Map a deterministic reduce op to its normalized family name."""
    if reduce_op == REDUCE_OP_ADD:
        return DETERMINISTIC_FAMILY_ADD
    if reduce_op == REDUCE_OP_MIN:
        return DETERMINISTIC_FAMILY_MIN
    if reduce_op == REDUCE_OP_MAX:
        return DETERMINISTIC_FAMILY_MAX
    raise ValueError(f"Unsupported deterministic reduce op: {reduce_op}")


def sanitize_det_name(name: str) -> str:
    """Sanitize a source array label for deterministic helper names."""
    sanitized = re.sub(r"\W+", "_", name)
    sanitized = sanitized.strip("_")
    if not sanitized:
        sanitized = "target"
    return sanitized


def scatter_helper_name(array_var_label: str, index: int) -> str:
    """Return the deterministic scatter helper name for an array."""
    return f"det_scatter_{index}_{sanitize_det_name(array_var_label)}"


def counter_helper_name(array_var_label: str, index: int) -> str:
    """Return the deterministic counter helper name for an array."""
    return f"det_counter_{index}_{sanitize_det_name(array_var_label)}"


def target_label(array_var_label: str, attr_path=()) -> str:
    """Return a stable display/helper label for a deterministic target."""
    attr_path = tuple(attr_path)
    if not attr_path:
        return array_var_label
    return ".".join((array_var_label, *attr_path))


def scatter_cpp_type(target: ScatterTarget) -> str:
    """Return the C++ helper type for a scatter target."""
    return f"wp::det_scatter_buf_t<{target.value_ctype}>"


def counter_cpp_type(_target: CounterTarget) -> str:
    """Return the C++ helper type for a counter target."""
    return "wp::det_counter_buf_t"


def kernel_raw_scatter_param_names(target: ScatterTarget) -> dict[str, str]:
    """Return raw kernel-entry parameter names for a scatter target."""
    return {"buf": f"_wp_{target.helper_name}"}


def kernel_raw_counter_param_names(target: CounterTarget) -> dict[str, str]:
    """Return raw kernel-entry parameter names for a counter target."""
    return {"buf": f"_wp_{target.helper_name}"}


@dataclass(eq=False)
class ScatterTarget:
    """Canonical scatter/reduce target identity for one module build.

    A target is the destination array, or struct-field array, whose atomics are
    scattered and later reduced. The registry owns the stable helper name.
    ``adjoint`` marks reductions into the argument's adjoint buffer during a
    generated backward pass.
    """

    array_var_label: str
    helper_name: str
    family: str
    value_dtype: type
    value_ctype: str
    scalar_dtype: type
    reduce_op: int
    index: int = 0
    attr_path: tuple[str, ...] = ()
    adjoint: bool = False
    use_forward: bool = True
    use_backward: bool = False

    @property
    def target_label(self) -> str:
        label = target_label(self.array_var_label, self.attr_path)
        return f"adjoint({label})" if self.adjoint else label


@dataclass(eq=False)
class CounterTarget:
    """Canonical consumed-return counter identity for one module build."""

    array_var_label: str
    helper_name: str
    value_ctype: str
    index: int = 0
    attr_path: tuple[str, ...] = ()
    adjoint: bool = False
    use_forward: bool = True
    use_backward: bool = False

    @property
    def target_label(self) -> str:
        label = target_label(self.array_var_label, self.attr_path)
        return f"adjoint({label})" if self.adjoint else label


@dataclass
class DeterministicMeta:
    """Per-adjoint deterministic requirements discovered during codegen.

    Nested ``@wp.func`` adjoints each collect local requirements here; callers
    remap and merge those requirements when emitting the call site.
    """

    determinism_mode: DeterministicMode = DeterministicMode.NOT_GUARANTEED
    max_records: int = 0
    scatter_targets: list[ScatterTarget] = field(default_factory=list)
    counter_targets: list[CounterTarget] = field(default_factory=list)
    scatter_records_per_thread: dict[ScatterTarget, int] = field(default_factory=dict)
    counter_records_per_thread: dict[CounterTarget, int] = field(default_factory=dict)
    needs_context: bool = False
    # Set by Adjoint.build() once the body has been pre-scanned.
    has_consumed_atomic: bool = False
    has_side_effect_store: bool = False

    def add_scatter_target(self, target: ScatterTarget, records_per_thread: int = 1):
        """Record a scatter target requirement for this function or kernel."""
        if target not in self.scatter_records_per_thread:
            self.scatter_targets.append(target)
            self.scatter_targets.sort(key=lambda x: x.index)
            self.scatter_records_per_thread[target] = 0
        self.scatter_records_per_thread[target] += records_per_thread

    def add_counter_target(self, target: CounterTarget, records_per_thread: int = 1):
        """Record a counter target requirement for this function or kernel."""
        if target not in self.counter_records_per_thread:
            self.counter_targets.append(target)
            self.counter_targets.sort(key=lambda x: x.index)
            self.counter_records_per_thread[target] = 0
        self.counter_records_per_thread[target] += records_per_thread

    def include(self, other: DeterministicMeta):
        """Merge the transitive deterministic requirements of another adjoint."""
        self.needs_context |= other.needs_context
        for target, count in other.scatter_records_per_thread.items():
            self.add_scatter_target(target, records_per_thread=count)
        for target, count in other.counter_records_per_thread.items():
            self.add_counter_target(target, records_per_thread=count)

    @property
    def has_scatter(self):
        return len(self.scatter_targets) > 0

    @property
    def has_counter(self):
        return len(self.counter_targets) > 0

    @property
    def needs_deterministic(self):
        return self.needs_context or self.has_scatter or self.has_counter


@dataclass
class DeterministicRegistry:
    """Build-wide registry that assigns stable helper names to targets."""

    scatter_targets: list[ScatterTarget] = field(default_factory=list)
    counter_targets: list[CounterTarget] = field(default_factory=list)
    _scatter_targets_by_array: dict[tuple[str, tuple[str, ...], bool], ScatterTarget] = field(default_factory=dict)
    _counter_targets_by_array: dict[tuple[str, tuple[str, ...], bool], CounterTarget] = field(default_factory=dict)
    # Scatter and counter helpers share one namespace in generated C++.
    _next_target_index: int = 0


def get_or_create_scatter_target(
    registry,
    meta,
    array_var_label,
    value_dtype,
    value_ctype,
    scalar_dtype,
    reduce_op,
    attr_path=(),
    *,
    adjoint=False,
    use_forward=True,
    use_backward=False,
):
    """Get or create a stable scatter target and attach it to ``meta``.

    Deterministic mode currently supports only one normalized reduction family
    per target array. ``atomic_add`` and ``atomic_sub`` share the same family.
    Mixing families such as ``atomic_add`` and ``atomic_max`` on the same array
    is rejected as a first-pass limitation.
    """
    family = reduce_op_to_family(reduce_op)
    attr_path = tuple(attr_path)
    key = (array_var_label, attr_path, bool(adjoint))
    label = target_label(array_var_label, attr_path)
    helper_label = f"adj_{label}" if adjoint else label
    if key in registry._counter_targets_by_array:
        raise ValueError(
            f"Deterministic mode does not support using array '{helper_label}' as both a counter target "
            "and a scatter target in the same function or kernel."
        )
    target = registry._scatter_targets_by_array.get(key)

    if target is not None:
        if target.family != family:
            raise ValueError(
                f"Deterministic mode does not support mixing '{target.family}' and '{family}' reductions on array "
                f"'{helper_label}' in the same function or kernel."
            )
        if (
            target.value_dtype != value_dtype
            or target.value_ctype != value_ctype
            or target.scalar_dtype != scalar_dtype
            or target.reduce_op != reduce_op
        ):
            raise ValueError(
                f"Deterministic mode does not support multiple value layouts for array '{label}' "
                f"within the same reduction family."
            )
        target.use_forward |= bool(use_forward)
        target.use_backward |= bool(use_backward)
    else:
        index = registry._next_target_index
        registry._next_target_index += 1
        target = ScatterTarget(
            array_var_label=array_var_label,
            helper_name=scatter_helper_name(helper_label, index),
            family=family,
            value_dtype=value_dtype,
            value_ctype=value_ctype,
            scalar_dtype=scalar_dtype,
            reduce_op=reduce_op,
            index=index,
            attr_path=attr_path,
            adjoint=bool(adjoint),
            use_forward=bool(use_forward),
            use_backward=bool(use_backward),
        )
        registry.scatter_targets.append(target)
        registry._scatter_targets_by_array[key] = target

    meta.add_scatter_target(target)
    return target


def get_or_create_counter_target(
    registry, meta, array_var_label, value_ctype, attr_path=(), *, adjoint=False, use_forward=True, use_backward=False
):
    """Get or create a stable counter target and attach it to ``meta``."""
    attr_path = tuple(attr_path)
    key = (array_var_label, attr_path, bool(adjoint))
    label = target_label(array_var_label, attr_path)
    helper_label = f"adj_{label}" if adjoint else label
    if key in registry._scatter_targets_by_array:
        raise ValueError(
            f"Deterministic mode does not support using array '{helper_label}' as both a counter target "
            "and a scatter target in the same function or kernel."
        )
    target = registry._counter_targets_by_array.get(key)
    if target is None:
        index = registry._next_target_index
        registry._next_target_index += 1
        target = CounterTarget(
            array_var_label=array_var_label,
            helper_name=counter_helper_name(helper_label, index),
            value_ctype=value_ctype,
            index=index,
            attr_path=attr_path,
            adjoint=bool(adjoint),
            use_forward=bool(use_forward),
            use_backward=bool(use_backward),
        )
        registry.counter_targets.append(target)
        registry._counter_targets_by_array[key] = target
    else:
        target.use_forward |= bool(use_forward)
        target.use_backward |= bool(use_backward)
    meta.add_counter_target(target)
    return target


DETERMINISTIC_SCATTER_MAX_CAPACITY = (1 << 31) - 1


class _ScalarTypeIds:
    def _data(self):
        import warp  # noqa: PLC0415

        return {
            warp.float16: 0,
            warp.float32: 1,
            warp.float64: 2,
            warp.int32: 3,
            warp.uint32: 4,
            warp.int64: 5,
            warp.uint64: 6,
            warp.bfloat16: 7,
        }

    def __getitem__(self, dtype):
        return self._data()[dtype]

    def get(self, dtype, default=None):
        return self._data().get(dtype, default)


_SCALAR_TYPE_IDS = _ScalarTypeIds()


def warp_scalar_type_to_id(dtype) -> int:
    """Map a Warp scalar type to the native deterministic reducer enum."""
    type_id = _SCALAR_TYPE_IDS.get(dtype)
    if type_id is None:
        raise ValueError(f"Unsupported scalar type for deterministic atomic: {dtype}")
    return type_id


# ---------------------------------------------------------------------------
# Codegen-time helpers
# ---------------------------------------------------------------------------

# Atomic builtins that can be intercepted by deterministic mode.
_DET_INTERCEPTABLE_ATOMICS = frozenset(
    {
        "atomic_add",
        "atomic_sub",
        "atomic_min",
        "atomic_max",
    }
)

# Atomic builtins that deterministic mode does not intercept, but which still
# mutate user state and must not double-execute under two-pass counter replay.
_DET_UNINTERCEPTED_SIDE_EFFECT_ATOMICS = frozenset(
    {
        "atomic_xor",
        "atomic_and",
        "atomic_or",
    }
)


def _det_needs_store_guard(adj) -> bool:
    """Return whether stores in this body must be phase-gated."""
    if adj.det_meta is None:
        return False
    return adj.det_meta.has_counter or adj.det_meta.has_consumed_atomic or adj.det_meta.has_side_effect_store


def _det_wrap_slot_store(adj, slot_lvalue: str, value_expr: str) -> str:
    """Return the forward C++ statement for ``slot_lvalue = value_expr``."""
    if _det_needs_store_guard(adj):
        adj.det_meta.needs_context = True
        return f"WP_DET_SLOT_STORE_IF_ACTIVE(det_ctx, {slot_lvalue}, {value_expr});"
    return f"{slot_lvalue} = {value_expr};"


def _det_wrap_side_effect_call(adj, statement: str) -> str:
    """Return the forward C++ statement for a non-counter atomic side effect."""
    if _det_needs_store_guard(adj):
        adj.det_meta.needs_context = True
        return f"WP_DET_SIDE_EFFECT_IF_ACTIVE(det_ctx, {statement});"
    return statement


def _deterministic_call_name(node):
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _deterministic_contains_atomic_call(node):
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and _deterministic_call_name(child.func) in _DET_INTERCEPTABLE_ATOMICS:
            return True
    return False


def _deterministic_contains_subscript_target(node):
    if isinstance(node, ast.Subscript):
        return True
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(_deterministic_contains_subscript_target(element) for element in node.elts)
    return False


def _deterministic_has_propagating_side_effect(tree):
    """Return whether ``tree`` contains an array store or unintercepted bitwise atomic."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(_deterministic_contains_subscript_target(target) for target in node.targets):
                return True
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if _deterministic_contains_subscript_target(node.target):
                return True
        elif isinstance(node, ast.Call):
            if _deterministic_call_name(node.func) in _DET_UNINTERCEPTED_SIDE_EFFECT_ATOMICS:
                return True

    return False


def _deterministic_has_consumed_atomic(tree):
    """Return whether codegen will treat any atomic return value as consumed."""
    return _deterministic_has_consumed_atomic_impl(tree, adj=None, seen=None)


def _deterministic_collect_target_names(target, names):
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            _deterministic_collect_target_names(element, names)


def _deterministic_local_names(tree):
    names = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = node.args
            for arg in itertools.chain(args.posonlyargs, args.args, args.kwonlyargs):
                names.add(arg.arg)
            if args.vararg is not None:
                names.add(args.vararg.arg)
            if args.kwarg is not None:
                names.add(args.kwarg.arg)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                _deterministic_collect_target_names(target, names)
        elif isinstance(node, ast.AnnAssign):
            _deterministic_collect_target_names(node.target, names)
        elif isinstance(node, ast.NamedExpr):
            _deterministic_collect_target_names(node.target, names)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            _deterministic_collect_target_names(node.target, names)
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars is not None:
                    _deterministic_collect_target_names(item.optional_vars, names)
        elif isinstance(node, ast.ExceptHandler) and isinstance(node.name, str):
            names.add(node.name)

    return names


def _deterministic_call_path(node):
    path = []

    while isinstance(node, ast.Attribute):
        path.append(node.attr)
        node = node.value

    if isinstance(node, ast.Name):
        path.append(node.id)
        path.reverse()
        return path

    return None


def _deterministic_resolve_static_call(adj, node, local_names):
    if adj is None:
        return None

    path = _deterministic_call_path(node)
    if not path or path[0] in local_names:
        return None

    obj = adj.resolve_external_reference(path[0])

    if obj is None:
        import warp  # noqa: PLC0415

        obj = getattr(warp, path[0], None)

    if obj is None:
        builtins_obj = __builtins__
        if isinstance(builtins_obj, dict):
            obj = builtins_obj.get(path[0])
        else:
            obj = getattr(builtins_obj, path[0], None)

    if obj is None:
        return None

    for attr in path[1:]:
        if not hasattr(obj, attr):
            return None
        obj = getattr(obj, attr)

    return obj


def _deterministic_has_consumed_atomic_impl(tree, adj, seen):
    """Return whether this AST or any statically called Warp function consumes an atomic return."""
    if seen is None:
        seen = set()

    local_names = _deterministic_local_names(tree)

    stack = [(tree, None)]
    while stack:
        node, parent = stack.pop()

        if isinstance(node, ast.Call):
            if _deterministic_call_name(node.func) in _DET_INTERCEPTABLE_ATOMICS:
                if not (isinstance(parent, ast.Expr) and parent.value is node):
                    return True
            else:
                func = _deterministic_resolve_static_call(adj, node.func, local_names)
                is_warp_func = getattr(func, "adj", None) is not None and callable(getattr(func, "is_builtin", None))
                if is_warp_func and not func.is_builtin():
                    func_id = id(func)
                    if func_id not in seen:
                        seen.add(func_id)
                        if _deterministic_has_consumed_atomic_impl(func.adj.tree, func.adj, seen):
                            return True

        for child in ast.iter_child_nodes(node):
            stack.append((child, node))

    return False


def emit_deterministic_atomic(adj, func, bound_args, return_type, output, output_list):
    """Emit deterministic scatter or two-pass code for an atomic builtin.

    Returns the output Var if the atomic was handled, or None only for
    atomic operations whose ordinary implementation is already
    deterministic.
    """
    from warp._src.codegen import Var, WarpCodegenError  # noqa: PLC0415

    args_list = list(bound_args.values())
    arr_var = args_list[0]  # the target array, possibly a view such as arr[i]
    view_prefix_vars = []

    while getattr(arr_var, "_det_view_parent", None) is not None:
        view_prefix_vars = list(arr_var._det_view_indices) + view_prefix_vars
        arr_var = arr_var._det_view_parent

    try:
        target_info = _deterministic_target_info(arr_var)
        if target_info is None:
            raise WarpCodegenError(f"Deterministic mode could not resolve the target array for {func.key}.")

        target_root_label, target_attr_path, arr_type, target_is_adjoint = target_info
        if not is_array(arr_type):
            raise WarpCodegenError(
                f"Deterministic mode expected {func.key} to target an array, got {type_repr(arr_type)}."
            )

        value_dtype = arr_type.dtype
        scalar_dtype = value_dtype
        if hasattr(scalar_dtype, "_wp_scalar_type_"):
            scalar_dtype = scalar_dtype._wp_scalar_type_

        if not any(arg.label == target_root_label for arg in adj.args):
            raise WarpCodegenError(
                f"Deterministic mode could not map atomic target '{target_root_label}' "
                f"for {func.key} to a kernel argument."
            )
    except WarpCodegenError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        raise WarpCodegenError(f"Deterministic mode could not lower {func.key}: {e}") from e

    # Determine if the return value is actually consumed by the caller.
    # When called from emit_AugAssign (arr[i] += val) or a bare expression,
    # the return is discarded and the atomic can be handled by scatter/reduce.
    # Any other expression context consumes the return value.
    return_is_discarded = getattr(adj, "_det_atomic_return_discarded", False)
    return_is_consumed = not return_is_discarded

    value_ctype = Var.dtype_to_ctype(value_dtype)
    scalar_ctype = Var.dtype_to_ctype(scalar_dtype)

    zero_expr = _cinit_expr(return_type)

    # Map from builtin name to reduction op
    op_map = {
        "atomic_add": REDUCE_OP_ADD,
        "atomic_sub": REDUCE_OP_ADD,  # value is negated before scatter
        "atomic_min": REDUCE_OP_MIN,
        "atomic_max": REDUCE_OP_MAX,
    }
    reduce_op = op_map.get(func.key, REDUCE_OP_ADD)

    # Classify whether this atomic needs deterministic slot assignment or
    # can be deferred into the post-kernel scatter/reduce pass.
    # All deterministic codegen is wrapped in #ifdef __CUDA_ARCH__ so that
    # CPU compilation falls through to normal atomic calls (CPU execution
    # is already sequential/deterministic).
    # We return None for the #else branch, telling add_call to emit the
    # normal atomic call below. However, since we can only return once,
    # we handle both CUDA and CPU paths here: emit the CUDA path as
    # raw C++ with #ifdef, then return output to skip the normal codegen.

    # Build the CPU fallback: a normal atomic call string.
    # This is used in the #else branch for CPU compilation.
    loaded_args = [adj.load(a) for a in args_list]
    cpu_args_str = ", ".join(a.emit() for a in loaded_args)
    if output is not None:
        cpu_call = f"var_{output} = wp::{func.native_func}({cpu_args_str});"
    else:
        cpu_call = f"wp::{func.native_func}({cpu_args_str});"

    # Integer accumulation atomics are deterministic in a single pass, but
    # in two-pass counter kernels they are side effects and must not run
    # during phase 0.  The pre-scan handles atomics that appear before the
    # counter atomic in source order.
    if scalar_dtype not in float_types and not return_is_consumed:
        if adj.det_meta.has_consumed_atomic:
            adj.add_forward(f"WP_DET_SIDE_EFFECT_IF_ACTIVE(det_ctx, {cpu_call});")
            return output
        return None  # fall through to normal codegen

    if return_is_consumed:
        # Consumed-return counter: assign slots by replaying after a prefix sum.
        if func.key != "atomic_add":
            raise WarpCodegenError(
                "Deterministic mode currently supports consumed-return counter atomics only for atomic_add, "
                f"got {func.key}."
            )

        if scalar_dtype != int32:
            raise WarpCodegenError(
                "Deterministic mode currently supports consumed-return counter atomics only for int32 counter arrays."
            )

        try:
            target = get_or_create_counter_target(
                adj.det_registry,
                adj.det_meta,
                target_root_label,
                scalar_ctype,
                attr_path=target_attr_path,
                adjoint=target_is_adjoint,
                use_forward=not adj.custom_reverse_mode and not target_is_adjoint,
                use_backward=adj.custom_reverse_mode or target_is_adjoint,
            )
        except ValueError as e:
            raise WarpCodegenError(str(e)) from e
        helper_name = target.helper_name

        val_loaded = loaded_args[-1]  # already loaded above
        loaded_prefix = [adj.load(var) for var in view_prefix_vars]
        idx_loaded_list = loaded_prefix + list(loaded_args[1:-1])
        target_expr = _deterministic_array_expr(adj, target_root_label, target_attr_path)
        if target_expr is None:
            raise WarpCodegenError(
                f"Deterministic mode could not build a generated target expression for {func.key} "
                f"on '{target_root_label}'."
            )
        flat_idx_expr = _deterministic_flat_index_expr(adj, target_expr, idx_loaded_list, func.key, target.value_ctype)

        adj.add_forward(
            f"WP_DET_COUNTER_OR_FALLBACK(var_{output}, det_ctx, {helper_name}, {flat_idx_expr}, "
            f"{val_loaded.emit()}, {cpu_call});",
            replay="// deterministic counter replay (skipped)",
        )
        return output

    # Return value unused: record the update and reduce it after the kernel.
    try:
        target = get_or_create_scatter_target(
            adj.det_registry,
            adj.det_meta,
            target_root_label,
            value_dtype,
            value_ctype,
            scalar_dtype,
            reduce_op,
            attr_path=target_attr_path,
            adjoint=target_is_adjoint,
            use_forward=not adj.custom_reverse_mode and not target_is_adjoint,
            use_backward=adj.custom_reverse_mode or target_is_adjoint,
        )
    except ValueError as e:
        raise WarpCodegenError(str(e)) from e
    helper_name = target.helper_name

    val_loaded = loaded_args[-1]
    loaded_prefix = [adj.load(var) for var in view_prefix_vars]
    idx_loaded_list = loaded_prefix + list(loaded_args[1:-1])

    target_expr = _deterministic_array_expr(adj, target_root_label, target_attr_path)
    if target_expr is None:
        raise WarpCodegenError(
            f"Deterministic mode could not build a generated target expression for {func.key} on '{target_root_label}'."
        )

    flat_idx_expr = _deterministic_flat_index_expr(adj, target_expr, idx_loaded_list, func.key, target.value_ctype)

    val_expr = val_loaded.emit()
    if func.key == "atomic_sub":
        val_expr = f"(-{val_expr})"

    adj.add_forward(
        f"WP_DET_SCATTER_OR_FALLBACK(det_ctx, {helper_name}, {flat_idx_expr}, {val_expr}, {cpu_call});",
        replay="// deterministic scatter replay (skipped)",
    )
    if output is not None:
        adj.add_forward(f"var_{output} = {zero_expr};")

    if target_is_adjoint:
        return output

    target_adj_expr = _deterministic_array_expr(adj, target_root_label, target_attr_path, adjoint=True)
    if target_adj_expr is None:
        return output

    original_val_arg = args_list[-1]
    fwd_parts = [target_expr] + [var.emit() for var in idx_loaded_list] + [val_loaded.emit()]
    adj_ret_str = f"adj_{output}" if output is not None else zero_expr
    adj_parts = (
        [target_adj_expr] + [var.emit_adj() for var in idx_loaded_list] + [original_val_arg.emit_adj(), adj_ret_str]
    )
    adj.add_reverse(f"wp::adj_{func.native_func}({', '.join(fwd_parts + adj_parts)});")
    return output


def _deterministic_function_args(adj):
    """Return hidden deterministic parameters for generated ``@wp.func`` calls."""
    if adj.det_meta is None or not adj.det_meta.needs_deterministic:
        return []

    det_args = ["wp::det_ctx det_ctx"]
    for target in adj.det_meta.counter_targets:
        det_args.append(f"{counter_cpp_type(target)} {target.helper_name}")
    for target in adj.det_meta.scatter_targets:
        det_args.append(f"{scatter_cpp_type(target)} {target.helper_name}")
    return det_args


def _deterministic_kernel_args(adj):
    """Return raw deterministic launch parameters for generated CUDA kernels."""
    if adj.det_meta is None or not adj.det_meta.needs_deterministic:
        return []

    det_args = [
        "int _wp_det_phase",
        "int _wp_det_debug",
        "int* _wp_det_overflow",
    ]
    for target in adj.det_meta.counter_targets:
        names = kernel_raw_counter_param_names(target)
        det_args.append(f"wp::det_counter_buf_t {names['buf']}")
    for target in adj.det_meta.scatter_targets:
        names = kernel_raw_scatter_param_names(target)
        det_args.append(f"wp::det_scatter_buf_t<{target.value_ctype}> {names['buf']}")
    return det_args


def _deterministic_kernel_locals(adj, device, *, use_launch_buffers=True):
    """Declare ``det_ctx`` plus helper aliases/stand-ins inside generated kernels."""
    if adj.det_meta is None or not adj.det_meta.needs_deterministic:
        return ""

    if device == "cuda" and use_launch_buffers:
        decls = []
        n_counter_targets = len(adj.det_meta.counter_targets)
        if n_counter_targets > 0:
            counter_buf_names = [
                kernel_raw_counter_param_names(target)["buf"] for target in adj.det_meta.counter_targets
            ]
            ptr_inits = ", ".join(f"{name}.target_ptr" for name in counter_buf_names)
            size_inits = ", ".join(f"{name}.target_size" for name in counter_buf_names)
            decls.append(f"uint64_t _wp_det_counter_target_ptrs[{n_counter_targets}] = {{{ptr_inits}}};")
            decls.append(f"int _wp_det_counter_target_sizes[{n_counter_targets}] = {{{size_inits}}};")
            decls.append(
                "wp::det_ctx det_ctx{_wp_det_phase, _wp_det_debug, _idx, _wp_det_overflow, "
                f"_wp_det_counter_target_ptrs, _wp_det_counter_target_sizes, {n_counter_targets}}};"
            )
        else:
            decls.append(
                "wp::det_ctx det_ctx{_wp_det_phase, _wp_det_debug, _idx, _wp_det_overflow, nullptr, nullptr, 0};"
            )
        for target in adj.det_meta.counter_targets:
            names = kernel_raw_counter_param_names(target)
            decls.append(f"auto& {target.helper_name} = {names['buf']};")
        for target in adj.det_meta.scatter_targets:
            names = kernel_raw_scatter_param_names(target)
            decls.append(f"auto& {target.helper_name} = {names['buf']};")
    else:
        decls = [
            "wp::det_ctx det_ctx{1, 0, 0, nullptr, nullptr, nullptr, 0};",
        ]
        for target in adj.det_meta.counter_targets:
            decls.append(
                f"wp::det_counter_buf_t {target.helper_name}{{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0}};"
            )
        for target in adj.det_meta.scatter_targets:
            decls.append(
                f"wp::det_scatter_buf_t<{target.value_ctype}> {target.helper_name}{{nullptr, nullptr, nullptr, 0}};"
            )
    return "".join(f"    {line}\n" for line in decls)


def _deterministic_reference_origin(var):
    """Return the root argument and field path for a tracked array reference."""
    root_label = getattr(var, "_det_ref_root_label", None)
    if root_label is None:
        return None
    return root_label, tuple(getattr(var, "_det_ref_attr_path", ()))


def _deterministic_array_expr(adj, root_label, attr_path, adjoint=False):
    """Build a generated C++ expression for a direct or struct-field array."""
    root = None
    for arg in adj.args:
        if arg.label == root_label:
            root = arg
            break

    if root is None:
        return None

    expr = root.emit_adj() if adjoint else root.emit()
    for attr in attr_path:
        expr = f"{expr}.{attr}"
    return expr


def _deterministic_target_info(var):
    """Return ``(root_arg, attr_path, array_type, adjoint)`` for a deterministic target."""
    from warp._src.codegen import is_reference, strip_reference  # noqa: PLC0415

    var_type = getattr(var, "type", None)
    if var_type is None:
        return None

    if getattr(var, "_det_adjoint_target", False):
        root_label = getattr(var, "_det_adjoint_root_label", None)
        if root_label is None:
            return None
        attr_path = tuple(getattr(var, "_det_ref_attr_path", ()))
        array_type = getattr(var, "_det_ref_array_type", strip_reference(var_type))
        return root_label, attr_path, array_type, True

    origin = _deterministic_reference_origin(var)
    if origin is not None:
        root_label, attr_path = origin
        array_type = getattr(var, "_det_ref_array_type", strip_reference(var_type))
        return root_label, attr_path, array_type, False

    if is_reference(var_type):
        return None

    return var.label, (), var_type, False


def _add_deterministic_array_store(adj, target, indices, rhs):
    """Emit ``array_store`` guarded so counter phase 0 has no side effects."""
    from warp._src.codegen import (  # noqa: PLC0415
        Reference,
        compute_type_str,
        get_arg_type,
        get_arg_value,
        strip_reference,
    )
    from warp._src.context import Function, builtin_functions  # noqa: PLC0415

    store_args_raw = (target, *indices, rhs)
    store_func = adj.resolve_func(
        builtin_functions["array_store"],
        tuple(get_arg_type(x) for x in store_args_raw),
        {},
        min_outputs=None,
    )
    bound_args = store_func.signature.bind(*store_args_raw)
    bound_arg_types = {k: get_arg_type(v) for k, v in bound_args.arguments.items()}
    bound_arg_values = {k: get_arg_value(v) for k, v in bound_args.arguments.items()}
    return_type = store_func.value_func(
        {k: strip_reference(v) for k, v in bound_arg_types.items()},
        bound_arg_values,
    )

    if store_func.dispatch_func is not None:
        func_args, template_args = store_func.dispatch_func(store_func.input_types, return_type, bound_args)
    else:
        func_args = tuple(bound_args.arguments.values())
        template_args = ()

    func_args = tuple(adj.register_var(x) for x in func_args)
    func_name = compute_type_str(store_func.native_func, template_args)
    use_initializer_list = store_func.initializer_list_func(bound_args, return_type)

    # Reuse normal array_store dispatch/loading so references, views, and
    # adjoint code match the non-deterministic path exactly.
    fwd_args = []
    for func_arg in func_args:
        loaded_arg = func_arg
        if not isinstance(func_arg, (Reference, Function)):
            loaded_arg = adj.load(func_arg)
        fwd_args.append(strip_reference(loaded_arg))

    store_args = ", ".join(f"var_{x}" for x in fwd_args)
    adj.add_forward(f"WP_DET_STORE_IF_ACTIVE(det_ctx, {store_args});", skip_replay=True)

    if store_func.is_differentiable:
        adj_args = tuple(strip_reference(x) for x in func_args)
        arg_str = adj.format_reverse_call_args(
            fwd_args,
            adj_args,
            [],
            use_initializer_list,
            has_output_args=False,
        )
        if arg_str is not None:
            adj.add_reverse(f"{store_func.namespace}adj_{func_name}({arg_str});")


def _deterministic_adjoint_address_call(adj, fwd_args, output_list, fallback_call):
    """Return a deterministic reverse ``address`` call when it scatters into an array adjoint.

    Reverse-mode array reads accumulate into ``adj_arr[index]`` (or
    ``arr.grad[index]``) with native floating-point atomics.  Under deterministic
    mode, route those updates through the same scatter/sort/reduce helper used
    by forward scatter atomics so tape backward passes get a fixed reduction
    order.
    """
    from warp._src.codegen import Var, WarpCodegenError  # noqa: PLC0415

    if not fwd_args or not output_list:
        return fallback_call

    arr_var = fwd_args[0]
    view_prefix_vars = []
    while getattr(arr_var, "_det_view_parent", None) is not None:
        view_prefix_vars = list(arr_var._det_view_indices) + view_prefix_vars
        arr_var = arr_var._det_view_parent

    try:
        target_info = _deterministic_target_info(arr_var)
        if target_info is None:
            return fallback_call

        target_root_label, target_attr_path, arr_type, _target_is_adjoint = target_info
        if not is_array(arr_type):
            return fallback_call

        value_dtype = arr_type.dtype
        scalar_dtype = value_dtype
        if hasattr(scalar_dtype, "_wp_scalar_type_"):
            scalar_dtype = scalar_dtype._wp_scalar_type_
        if scalar_dtype not in float_types:
            return fallback_call

        if not any(arg.label == target_root_label for arg in adj.args):
            return fallback_call

        value_ctype = Var.dtype_to_ctype(value_dtype)
        target = get_or_create_scatter_target(
            adj.det_registry,
            adj.det_meta,
            target_root_label,
            value_dtype,
            value_ctype,
            scalar_dtype,
            REDUCE_OP_ADD,
            attr_path=target_attr_path,
            adjoint=True,
            use_forward=False,
            use_backward=True,
        )
        # Validate that the scalar value has a deterministic reducer.  The
        # result is used by launch-time post-reduce, but checking here gives a
        # source-adjacent error if an unsupported gradient type appears.
        warp_scalar_type_to_id(scalar_dtype)
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        raise WarpCodegenError(f"Deterministic mode could not lower adjoint address accumulation: {e}") from e

    loaded_prefix = [adj.load(var) for var in view_prefix_vars]
    idx_loaded_list = loaded_prefix + list(fwd_args[1:])
    target_expr = _deterministic_array_expr(adj, target_root_label, target_attr_path)
    if target_expr is None:
        return fallback_call

    flat_idx_expr = _deterministic_flat_index_expr(adj, target_expr, idx_loaded_list, "address", target.value_ctype)
    adj_output = output_list[0].emit_adj()
    return f"WP_DET_SCATTER_OR_FALLBACK(det_ctx, {target.helper_name}, {flat_idx_expr}, {adj_output}, {fallback_call});"


def _deterministic_flat_index_expr(adj, target_expr, idx_loaded_list, func_key, value_ctype):
    """Element offset from ``ptr`` via ``strides``, so non-contiguous views work."""
    from warp._src.codegen import WarpCodegenError  # noqa: PLC0415

    del adj

    ndim = len(idx_loaded_list)
    if ndim < 1 or ndim > 4:
        raise WarpCodegenError(
            f"Deterministic mode currently supports arrays up to 4 dimensions, got {ndim}D indexing for "
            f"{func_key} on {target_expr}."
        )

    terms = " + ".join(f"var_{idx_loaded_list[k]} * {target_expr}.strides[{k}]" for k in range(ndim))
    return f"(({terms}) / static_cast<int>(sizeof({value_ctype})))"


def _deterministic_bound_target(bound_var, extra_attr_path=()):
    """Map a formal target to the actual bound argument at a call site."""
    from warp._src.codegen import is_reference  # noqa: PLC0415

    bound_type = getattr(bound_var, "type", None)
    if bound_type is None:
        return None

    extra_attr_path = tuple(extra_attr_path)
    origin = _deterministic_reference_origin(bound_var)
    if origin is not None:
        root_label, attr_path = origin
        return root_label, attr_path + extra_attr_path

    if is_reference(bound_type):
        return None

    return bound_var.label, extra_attr_path


def _deterministic_map_target(target, bound_args):
    """Remap callee metadata targets into the caller's argument namespace."""
    from warp._src.codegen import strip_reference  # noqa: PLC0415

    attr_path = tuple(getattr(target, "attr_path", ()))
    actual = bound_args.get(target.array_var_label)
    if actual is None:
        return target.array_var_label, attr_path

    mapped = _deterministic_bound_target(strip_reference(actual), attr_path)
    if mapped is None:
        return target.array_var_label, attr_path

    return mapped


def _deterministic_find_target(targets, array_var_label, attr_path, *, adjoint=False):
    attr_path = tuple(attr_path)
    for target in targets:
        if (
            target.array_var_label == array_var_label
            and tuple(getattr(target, "attr_path", ())) == attr_path
            and bool(getattr(target, "adjoint", False)) == bool(adjoint)
        ):
            return target
    return None


def _include_deterministic_call_meta(adj, meta, bound_args):
    """Merge deterministic requirements from a called ``@wp.func`` into ``adj``."""
    if adj.det_meta is None or adj.det_registry is None or meta is None or not meta.needs_deterministic:
        return

    adj.det_meta.needs_context |= meta.needs_context

    for target, count in meta.scatter_records_per_thread.items():
        mapped_label, mapped_attr_path = _deterministic_map_target(target, bound_args)
        mapped_target = get_or_create_scatter_target(
            adj.det_registry,
            adj.det_meta,
            mapped_label,
            target.value_dtype,
            target.value_ctype,
            target.scalar_dtype,
            target.reduce_op,
            attr_path=mapped_attr_path,
            adjoint=getattr(target, "adjoint", False),
            use_forward=getattr(target, "use_forward", True),
            use_backward=getattr(target, "use_backward", False),
        )
        adj.det_meta.scatter_records_per_thread[mapped_target] += count - 1

    for target, count in meta.counter_records_per_thread.items():
        mapped_label, mapped_attr_path = _deterministic_map_target(target, bound_args)
        mapped_target = get_or_create_counter_target(
            adj.det_registry,
            adj.det_meta,
            mapped_label,
            target.value_ctype,
            attr_path=mapped_attr_path,
            adjoint=getattr(target, "adjoint", False),
            use_forward=getattr(target, "use_forward", True),
            use_backward=getattr(target, "use_backward", False),
        )
        adj.det_meta.counter_records_per_thread[mapped_target] += count - 1


def _deterministic_call_args(adj, meta, bound_args):
    """Return helper arguments to pass from a caller into a deterministic ``@wp.func``."""
    if adj.det_meta is None or meta is None or not meta.needs_deterministic:
        return []

    det_args = ["det_ctx"]
    for target in meta.counter_targets:
        mapped_label, mapped_attr_path = _deterministic_map_target(target, bound_args)
        mapped_target = _deterministic_find_target(
            adj.det_meta.counter_targets, mapped_label, mapped_attr_path, adjoint=getattr(target, "adjoint", False)
        )
        if mapped_target is None and adj.det_registry is not None:
            mapped_target = get_or_create_counter_target(
                adj.det_registry,
                adj.det_meta,
                mapped_label,
                target.value_ctype,
                attr_path=mapped_attr_path,
                adjoint=getattr(target, "adjoint", False),
                use_forward=getattr(target, "use_forward", True),
                use_backward=getattr(target, "use_backward", False),
            )
            adj.det_meta.counter_records_per_thread[mapped_target] += meta.counter_records_per_thread.get(target, 1) - 1
        det_args.append(mapped_target.helper_name if mapped_target is not None else target.helper_name)
    for target in meta.scatter_targets:
        mapped_label, mapped_attr_path = _deterministic_map_target(target, bound_args)
        mapped_target = _deterministic_find_target(
            adj.det_meta.scatter_targets, mapped_label, mapped_attr_path, adjoint=getattr(target, "adjoint", False)
        )
        if mapped_target is None and adj.det_registry is not None:
            mapped_target = get_or_create_scatter_target(
                adj.det_registry,
                adj.det_meta,
                mapped_label,
                target.value_dtype,
                target.value_ctype,
                target.scalar_dtype,
                target.reduce_op,
                attr_path=mapped_attr_path,
                adjoint=getattr(target, "adjoint", False),
                use_forward=getattr(target, "use_forward", True),
                use_backward=getattr(target, "use_backward", False),
            )
            adj.det_meta.scatter_records_per_thread[mapped_target] += meta.scatter_records_per_thread.get(target, 1) - 1
        det_args.append(mapped_target.helper_name if mapped_target is not None else target.helper_name)
    return det_args


def _cinit_expr(dtype):
    from warp._src.codegen import Var  # noqa: PLC0415

    if hasattr(dtype, "cinit"):
        return dtype.cinit(requires_grad=False)
    return f"{Var.type_to_ctype(dtype)}{{}}"


# ---------------------------------------------------------------------------
# Launch-time helpers
# ---------------------------------------------------------------------------


def allocate_scatter_buffers(scatter_targets, meta, dim_size, device, max_records=0, initialize_unused=False):
    """Allocate buffers for atomics reduced after the kernel finishes.

    Returns a list of ``(keys, values, counter, capacity)`` tuples, one per
    scatter target in ``scatter_targets``.
    """
    import warp  # noqa: PLC0415

    buffers = []
    for target in scatter_targets:
        records_per_thread = max(meta.scatter_records_per_thread.get(target, 0), max_records)
        capacity = max(dim_size * records_per_thread, 1024)
        if capacity > DETERMINISTIC_SCATTER_MAX_CAPACITY:
            raise RuntimeError(
                "Deterministic scatter buffer capacity exceeds the supported int32 limit "
                f"({capacity} > {DETERMINISTIC_SCATTER_MAX_CAPACITY}). "
                "Reduce the launch size or deterministic_max_records."
            )
        if initialize_unused:
            keys = warp.full(shape=(capacity,), value=-1, dtype=warp.int64, device=device)
            values = warp.zeros(shape=(capacity,), dtype=target.value_dtype, device=device)
        else:
            keys = warp.empty(shape=(capacity,), dtype=warp.int64, device=device)
            values = warp.empty(shape=(capacity,), dtype=target.value_dtype, device=device)
        counter = warp.zeros(shape=(1,), dtype=warp.int32, device=device)
        buffers.append((keys, values, counter, capacity))
    return buffers


def get_scatter_record_count(scatter_buffer, stream_is_capturing=False):
    """Return the number of scatter records that should be sorted.

    Outside CUDA graph capture, the emitted-record counter can be read back
    after the scatter kernel completes, so the postpass only needs to sort the
    valid prefix of the buffer. During capture, host readbacks are illegal, so
    replay still sorts the full sentinel-initialized capacity.
    """
    _keys, _values, counter, capacity = scatter_buffer
    if stream_is_capturing:
        return capacity

    emitted_count = int(counter.numpy()[0])
    if emitted_count < 0:
        return capacity
    return min(emitted_count, capacity)


def get_counter_record_count(counter_buffer, stream_is_capturing=False):
    """Return the number of counter records that should be sorted/scanned."""
    _keys, _values, _prefixes, _record_slots, _cursors, count, capacity, _records_per_thread = counter_buffer
    if stream_is_capturing:
        return capacity

    emitted_count = int(count.numpy()[0])
    if emitted_count < 0:
        return capacity
    return min(emitted_count, capacity)


def allocate_counter_state_buffers(runtime, counter_arrays, device, stream):
    """Snapshot raw counter spans and allocate delayed writeback totals."""
    import warp  # noqa: PLC0415

    buffers = []
    element_size = type_size_in_bytes(warp.int32)

    for counter_arr in counter_arrays:
        if counter_arr is None:
            buffers.append((None, None, 0))
            continue

        counter_size_elements = _det_dest_size_elements(counter_arr)
        if counter_size_elements <= 0:
            buffers.append((None, None, 0))
            continue

        bases = warp.empty(shape=(counter_size_elements,), dtype=warp.int32, device=device)
        totals = warp.empty(shape=(counter_size_elements,), dtype=warp.int32, device=device)
        bytes_to_copy = counter_size_elements * element_size
        if not runtime.core.wp_memcpy_d2d(
            device.context, bases.ptr, counter_arr.ptr, bytes_to_copy, stream.cuda_stream
        ):
            raise RuntimeError(f"Warp deterministic counter snapshot error: {runtime.get_error_string()}")

        buffers.append((bases, totals, counter_size_elements))

    return buffers


def allocate_counter_buffers(counter_targets, meta, dim_size, device, max_records=0, initialize_unused=False):
    """Allocate record buffers for deterministic consumed-return counters."""
    import warp  # noqa: PLC0415

    buffers = []
    for target in counter_targets:
        records_per_thread = max(meta.counter_records_per_thread.get(target, 0), max_records, 1)
        capacity = dim_size * records_per_thread
        if capacity > DETERMINISTIC_SCATTER_MAX_CAPACITY:
            raise RuntimeError(
                "Deterministic counter buffer capacity exceeds the supported int32 limit "
                f"({capacity} > {DETERMINISTIC_SCATTER_MAX_CAPACITY}). "
                "Reduce the launch size or deterministic_max_records."
            )
        if initialize_unused:
            keys = warp.full(shape=(capacity,), value=-1, dtype=warp.int64, device=device)
            values = warp.zeros(shape=(capacity,), dtype=warp.int32, device=device)
        else:
            keys = warp.empty(shape=(capacity,), dtype=warp.int64, device=device)
            values = warp.empty(shape=(capacity,), dtype=warp.int32, device=device)
        prefixes = warp.empty(shape=(capacity,), dtype=warp.int32, device=device)
        record_slots = warp.full(shape=(capacity,), value=-1, dtype=warp.int32, device=device)
        cursors = warp.zeros(shape=(dim_size,), dtype=warp.int32, device=device)
        count = warp.zeros(shape=(1,), dtype=warp.int32, device=device)
        buffers.append((keys, values, prefixes, record_slots, cursors, count, capacity, records_per_thread))
    return buffers


def run_sort_reduce(
    runtime, scatter_targets, scatter_buffers, dest_arrays, device, determinism_mode, record_counts=None
):
    """Execute post-kernel sort-reduce for all scatter targets."""
    import warp  # noqa: PLC0415

    workspaces = []
    determinism_mode_id = int(determinism_mode)

    for i, target in enumerate(scatter_targets):
        keys, values, _counter, capacity = scatter_buffers[i]
        record_count = capacity if record_counts is None else record_counts[i]
        dest_arr = dest_arrays[i]

        # Optional route-specific outputs may be passed as ``None``. Their
        # guarded kernel paths must remain inactive, so there is no destination
        # buffer to reduce into.
        if dest_arr is None:
            continue
        if record_count <= 0:
            continue

        try:
            scalar_type_id = warp_scalar_type_to_id(target.scalar_dtype)
        except ValueError:
            from warp._src import utils as warp_utils  # noqa: PLC0415

            warp_utils.warn(f"Unsupported value type '{target.value_ctype}' for deterministic sort-reduce.")
            continue

        components = getattr(target.value_dtype, "_length_", 1)
        workspace_size = runtime.core.wp_deterministic_sort_reduce_workspace_size(
            record_count,
            target.reduce_op,
            scalar_type_id,
            components,
            determinism_mode_id,
        )
        workspace = warp.empty(shape=(workspace_size,), dtype=warp.uint8, device=device)
        workspaces.append(workspace)

        dest_size_elements = _det_dest_size_elements(dest_arr)
        runtime.core.wp_deterministic_sort_reduce_device(
            keys.ptr,
            values.ptr,
            record_count,
            dest_arr.ptr,
            dest_size_elements,
            target.reduce_op,
            scalar_type_id,
            components,
            determinism_mode_id,
            workspace.ptr,
            workspace_size,
        )

    return workspaces


def run_counter_scan(runtime, counter_buffers, counter_arrays, counter_state_buffers, device, record_counts=None):
    """Compute deterministic consumed-return prefixes for counter records."""
    import warp  # noqa: PLC0415

    workspaces = []

    for i, counter_buffer in enumerate(counter_buffers):
        keys, values, prefixes, _record_slots, _cursors, _count, capacity, _records_per_thread = counter_buffer
        counter_arr = counter_arrays[i]
        counter_bases, counter_totals, counter_size_elements = counter_state_buffers[i]
        record_count = capacity if record_counts is None else record_counts[i]

        if counter_arr is None:
            continue
        if counter_bases is None or counter_totals is None:
            continue
        if record_count <= 0:
            continue

        workspace_size = runtime.core.wp_deterministic_counter_scan_workspace_size(record_count)
        workspace = warp.empty(shape=(workspace_size,), dtype=warp.uint8, device=device)
        workspaces.append(workspace)

        runtime.core.wp_deterministic_counter_scan_device(
            keys.ptr,
            values.ptr,
            record_count,
            prefixes.ptr,
            counter_bases.ptr,
            counter_totals.ptr,
            counter_size_elements,
            workspace.ptr,
            workspace_size,
        )

    return workspaces


def run_counter_writeback(runtime, counter_buffers, counter_arrays, counter_state_buffers, record_counts=None):
    """Publish deterministic counter totals after phase 1 replay."""
    for i, counter_buffer in enumerate(counter_buffers):
        keys, _values, _prefixes, _record_slots, _cursors, _count, capacity, _records_per_thread = counter_buffer
        counter_arr = counter_arrays[i]
        _counter_bases, counter_totals, counter_size_elements = counter_state_buffers[i]
        record_count = capacity if record_counts is None else record_counts[i]

        if counter_arr is None or counter_totals is None:
            continue
        if record_count <= 0:
            continue

        runtime.core.wp_deterministic_counter_writeback_device(
            keys.ptr,
            record_count,
            counter_totals.ptr,
            counter_arr.ptr,
            counter_size_elements,
        )


_DeterministicLaunchClass = None


def _deterministic_launch_class():
    global _DeterministicLaunchClass
    if _DeterministicLaunchClass is not None:
        return _DeterministicLaunchClass

    from warp._src.context import Launch  # noqa: PLC0415

    class DeterministicLaunch(Launch):
        """Recorded launch wrapper that replays through the deterministic launcher."""

        def __init__(
            self,
            kernel,
            device: Device,
            det_meta,
            fwd_args: Sequence[Any],
            adj_args: Sequence[Any] | None = None,
            hooks: KernelHooks | None = None,
            params: Sequence[Any] | None = None,
            bounds: launch_bounds_t | None = None,
            max_blocks: int = 0,
            block_dim: int = 256,
            adjoint: bool = False,
        ):
            super().__init__(
                kernel=kernel,
                device=device,
                hooks=hooks,
                params=params,
                params_addr=None,
                bounds=bounds,
                max_blocks=max_blocks,
                block_dim=block_dim,
                adjoint=adjoint,
            )
            self.det_meta = det_meta
            self.fwd_args = list(fwd_args)
            self.adj_args = list(adj_args) if adj_args is not None else []

        def _is_deterministic_target_arg(self, index: int) -> bool:
            if index < 0 or index >= len(self.kernel.adj.args):
                return False

            arg_label = self.kernel.adj.args[index].label
            targets = (*self.det_meta.scatter_targets, *self.det_meta.counter_targets)
            return any(target.array_var_label == arg_label for target in targets)

        def set_param_at_index(self, index: int, value: Any, adjoint: bool = False):
            super().set_param_at_index(index, value, adjoint)
            if adjoint and index < len(self.adj_args):
                self.adj_args[index] = value
            elif not adjoint and index < len(self.fwd_args):
                self.fwd_args[index] = value

        def set_param_at_index_from_ctype(
            self, index: int, value: ctypes.Structure | int | float, adjoint: bool = False
        ):
            if self._is_deterministic_target_arg(index):
                arg_label = self.kernel.adj.args[index].label
                raise RuntimeError(
                    "Updating deterministic target argument "
                    f"'{arg_label}' from a raw ctypes value is not supported. "
                    "Use set_param_at_index() with a Warp array or struct object instead."
                )

            super().set_param_at_index_from_ctype(index, value, adjoint)
            if adjoint and index < len(self.adj_args):
                self.adj_args[index] = value
            elif not adjoint and index < len(self.fwd_args):
                self.fwd_args[index] = value

        def launch(self, stream: Stream | None = None) -> None:
            if stream is None:
                stream = self.device.stream

            launch_deterministic(
                self.kernel,
                self.hooks,
                self.params,
                self.bounds,
                self.device,
                stream,
                self.max_blocks,
                self.block_dim,
                self.det_meta,
                self.fwd_args,
                adj_args=self.adj_args,
                adjoint=self.adjoint,
                module_exec=self.module_exec,
            )

    DeterministicLaunch.__module__ = __name__
    _DeterministicLaunchClass = DeterministicLaunch
    return DeterministicLaunch


def create_deterministic_launch(*args, **kwargs):
    return _deterministic_launch_class()(*args, **kwargs)


def launch_deterministic(
    kernel,
    hooks,
    user_params,
    bounds,
    device,
    stream,
    max_blocks,
    block_dim,
    det_meta,
    fwd_args,
    *,
    adj_args=None,
    adjoint=False,
    module_exec=None,
):
    """Orchestrate a deterministic kernel launch with scatter-sort-reduce and/or two-pass execution.

    This is called from launch() when deterministic mode is active and the
    kernel has atomic operations that require deterministic treatment.
    Scatter/reduce atomics use one kernel pass plus sort-reduce; consumed-return
    counter atomics add a counting pass.
    """
    import warp  # noqa: PLC0415
    from warp._src import context as warp_context  # noqa: PLC0415
    from warp._src.context import (  # noqa: PLC0415
        _build_cuda_kernel_params,
        _cuda_launch_kernel,
        det_counter_buf_t,
        det_scatter_buf_t,
    )

    runtime = warp_context.runtime

    launch_kind = "backward" if adjoint else "forward"

    if hooks is None:
        raise RuntimeError(
            f"Failed to find {launch_kind} kernel '{kernel.key}' from module '{kernel.module.name}' for device '{device}'"
        )

    launch_hook = hooks.backward if adjoint else hooks.forward
    shared_memory_bytes = hooks.backward_smem_bytes if adjoint else hooks.forward_smem_bytes

    if launch_hook is None:
        raise RuntimeError(
            f"Failed to find {launch_kind} kernel '{kernel.key}' from module '{kernel.module.name}' for device '{device}'"
        )

    if runtime._apic_capture is not None:
        raise RuntimeError(
            "APIC serialization is not currently supported for deterministic CUDA kernels. "
            "Capture with apic=False, or disable deterministic mode for kernels captured with apic=True."
        )

    adj_args = [] if adj_args is None else list(adj_args)

    def resolve_det_target_array(target):
        resolved = None
        for j, arg in enumerate(kernel.adj.args):
            if arg.label == target.array_var_label:
                if getattr(target, "adjoint", False):
                    resolved = adj_args[j] if j < len(adj_args) else None
                else:
                    resolved = fwd_args[j]
                break

        if resolved is None:
            if not getattr(target, "adjoint", False):
                return None

            # Manual adjoint launches may rely on the forward array's embedded
            # grad pointer instead of passing an explicit adjoint array. The
            # deterministic post-pass needs the Python array object, so recover
            # the same fallback here when possible.
            primal = None
            for j, arg in enumerate(kernel.adj.args):
                if arg.label == target.array_var_label:
                    primal = fwd_args[j]
                    break
            for attr in getattr(target, "attr_path", ()):
                if primal is None or not hasattr(primal, attr):
                    primal = None
                    break
                primal = getattr(primal, attr)
            return getattr(primal, "grad", None)

        for attr in getattr(target, "attr_path", ()):
            if resolved is None:
                return None
            if not hasattr(resolved, attr):
                raise RuntimeError(
                    "Deterministic target "
                    f"{target.target_label!r} could not resolve attribute {attr!r} "
                    f"on {type(resolved).__name__}"
                )
            resolved = getattr(resolved, attr)

        if getattr(target, "adjoint", False) and resolved is None:
            primal = None
            for j, arg in enumerate(kernel.adj.args):
                if arg.label == target.array_var_label:
                    primal = fwd_args[j]
                    break
            for attr in getattr(target, "attr_path", ()):
                if primal is None or not hasattr(primal, attr):
                    primal = None
                    break
                primal = getattr(primal, attr)
            resolved = getattr(primal, "grad", None)

        return resolved

    def target_has_resolvable_destination(target):
        array = resolve_det_target_array(target)
        if array is None:
            return False
        if not hasattr(array, "ptr"):
            raise RuntimeError(
                f"Deterministic target {target.target_label!r} resolved to non-array object ({type(array).__name__})"
            )
        return getattr(array, "ptr", None) is not None and getattr(array, "size", 0) > 0

    active_scatter_targets = []
    for target in det_meta.scatter_targets:
        if not (target.use_backward if adjoint else target.use_forward):
            continue
        if adjoint and not target_has_resolvable_destination(target):
            continue
        active_scatter_targets.append(target)

    active_counter_targets = []
    for target in det_meta.counter_targets:
        if not (target.use_backward if adjoint else target.use_forward):
            continue
        if adjoint and not target_has_resolvable_destination(target):
            continue
        active_counter_targets.append(target)

    dim_size = bounds.size
    if active_scatter_targets and dim_size > (1 << 32):
        raise RuntimeError(
            "Deterministic scatter atomics support launch sizes up to 2^32 threads because sort keys pack the "
            "linear thread index into 32 bits."
        )
    if active_counter_targets and dim_size > ((1 << 31) - 1):
        raise RuntimeError(
            "Deterministic consumed-return counter atomics support launch sizes up to 2^31 - 1 threads because "
            "counter prefix buffers store per-thread contributions as int32 values."
        )

    determinism_mode = det_meta.determinism_mode
    max_records = det_meta.max_records
    det_debug = int(warp.config.deterministic_debug)
    stream_is_capturing = len(runtime.captures) > 0 and runtime.core.wp_cuda_stream_is_capturing(stream.cuda_stream)
    capture_graph = None
    if stream_is_capturing:
        capture_id = runtime.core.wp_cuda_stream_get_capture_id(stream.cuda_stream)
        capture_graph = runtime.captures.get(capture_id)

    # Allocate buffers.
    scatter_bufs = (
        allocate_scatter_buffers(
            active_scatter_targets,
            det_meta,
            dim_size,
            device,
            max_records=max_records,
            initialize_unused=stream_is_capturing,
        )
        if active_scatter_targets
        else []
    )
    counter_bufs = (
        allocate_counter_buffers(
            active_counter_targets,
            det_meta,
            dim_size,
            device,
            max_records=max_records,
            initialize_unused=stream_is_capturing,
        )
        if active_counter_targets
        else []
    )
    scatter_bufs_by_target = dict(zip(active_scatter_targets, scatter_bufs, strict=True))
    counter_bufs_by_target = dict(zip(active_counter_targets, counter_bufs, strict=True))
    overflow_buf = warp.zeros(shape=(1,), dtype=warp.int32, device=device) if scatter_bufs or counter_bufs else None
    sort_reduce_workspaces = []
    counter_scan_workspaces = []
    counter_state_buffers = []
    counter_arr_by_target = {}

    # Build the extra deterministic parameters (must match codegen_kernel order).
    def build_det_params(phase, use_scatter):
        det_params = [
            ctypes.c_int(phase),
            ctypes.c_int(det_debug),
            ctypes.c_void_p(overflow_buf.ptr if overflow_buf is not None else 0),
        ]
        for ct in det_meta.counter_targets:
            counter_buf = counter_bufs_by_target.get(ct)
            counter_arr = counter_arr_by_target.get(ct)
            target_ptr = counter_arr.ptr if counter_arr is not None else 0
            target_size = _det_dest_size_elements(counter_arr) if counter_arr is not None else 0
            if counter_buf is None:
                det_params.append(det_counter_buf_t(0, 0, 0, 0, 0, 0, 0, 0, target_ptr, target_size))
                continue

            keys, values, prefixes, record_slots, cursors, count, capacity, records_per_thread = counter_buf
            det_params.append(
                det_counter_buf_t(
                    keys.ptr,
                    values.ptr,
                    prefixes.ptr,
                    record_slots.ptr,
                    cursors.ptr,
                    count.ptr,
                    capacity,
                    records_per_thread,
                    target_ptr,
                    target_size,
                )
            )
        for st in det_meta.scatter_targets:
            scatter_buf = scatter_bufs_by_target.get(st)
            if use_scatter and scatter_buf is not None:
                keys, values, counter, capacity = scatter_buf
                det_params.append(det_scatter_buf_t(keys.ptr, values.ptr, counter.ptr, capacity))
            else:
                # Null scatter buffers (phase 0 doesn't scatter).
                det_params.append(det_scatter_buf_t(0, 0, 0, 0))
        return det_params

    def do_cuda_launch(hook, params_list):
        _cuda_launch_kernel(
            device,
            module_exec,
            hook,
            bounds.size,
            max_blocks,
            block_dim,
            shared_memory_bytes,
            _build_cuda_kernel_params(params_list),
            stream,
        )

    if active_counter_targets:
        # === Two-pass execution ===

        counter_arrays = []
        for ct in active_counter_targets:
            counter_arr = resolve_det_target_array(ct)
            if counter_arr is None:
                counter_arrays.append(None)
                continue
            if not hasattr(counter_arr, "ptr"):
                raise RuntimeError(
                    "Deterministic counter target "
                    f"{ct.target_label!r} resolved to non-array object "
                    f"({type(counter_arr).__name__})"
                )
            if getattr(counter_arr, "ptr", None) is None or getattr(counter_arr, "size", 0) <= 0:
                counter_arrays.append(None)
                continue
            counter_arrays.append(counter_arr)

        for ct, counter_arr in zip(active_counter_targets, counter_arrays, strict=True):
            counter_arr_by_target[ct] = counter_arr

        with warp.ScopedStream(stream, sync_enter=False):
            for counter_buf in counter_bufs:
                keys, values, _prefixes, record_slots, cursors, count, _capacity, _records_per_thread = counter_buf
                count.zero_()
                cursors.zero_()
                record_slots.fill_(-1)
                if stream_is_capturing:
                    keys.fill_(-1)
                    values.zero_()

        # Phase 0: counting pass (side effects suppressed, scatter disabled).
        det_params_p0 = build_det_params(phase=0, use_scatter=False)
        # Append det params after all user args (matches codegen_kernel order:
        # dim, user_args..., det_params...).
        params_p0 = [*user_params, *det_params_p0]
        do_cuda_launch(launch_hook, params_p0)

        with warp.ScopedStream(stream, sync_enter=False):
            counter_state_buffers = allocate_counter_state_buffers(runtime, counter_arrays, device, stream)

        # Sort counter records by destination and deterministic record order,
        # then compute the exclusive prefix each atomic should return.
        counter_record_counts = []
        for i, counter_arr in enumerate(counter_arrays):
            ct = active_counter_targets[i]
            if counter_arr is None:
                if not stream_is_capturing:
                    _keys, _values, _prefixes, _record_slots, _cursors, count, _capacity, _records_per_thread = (
                        counter_bufs[i]
                    )
                    emitted_count = int(count.numpy()[0])
                    if emitted_count != 0:
                        raise RuntimeError(
                            f"Deterministic counter target {ct.target_label!r} resolved to None "
                            f"but emitted {emitted_count} records."
                        )
                counter_record_counts.append(0)
                continue
            if stream_is_capturing:
                record_count = get_counter_record_count(counter_bufs[i], stream_is_capturing=True)
            else:
                _keys, _values, _prefixes, _record_slots, _cursors, count, capacity, _records_per_thread = counter_bufs[
                    i
                ]
                emitted_count = int(count.numpy()[0])
                if emitted_count < 0 or emitted_count > capacity:
                    raise RuntimeError(
                        f"Deterministic counter buffer overflow in kernel '{kernel.key}'. "
                        "Increase 'deterministic_max_records' or reduce the per-thread atomic count."
                    )
                record_count = emitted_count
            counter_record_counts.append(record_count)

        if overflow_buf is not None and not stream_is_capturing and int(overflow_buf.numpy()[0]) != 0:
            raise RuntimeError(
                f"Deterministic counter buffer overflow in kernel '{kernel.key}'. "
                "Increase 'deterministic_max_records' or reduce the per-thread atomic count."
            )

        with warp.ScopedStream(stream, sync_enter=False):
            counter_scan_workspaces = run_counter_scan(
                runtime,
                counter_bufs,
                counter_arrays,
                counter_state_buffers,
                device,
                record_counts=counter_record_counts,
            )
            for counter_buf in counter_bufs:
                _keys, _values, _prefixes, _record_slots, cursors, _count, _capacity, _records_per_thread = counter_buf
                cursors.zero_()

        # Phase 1: execution pass with deterministic slots.
        det_params_p1 = build_det_params(phase=1, use_scatter=True)
        params_p1 = [*user_params, *det_params_p1]
        do_cuda_launch(launch_hook, params_p1)

        with warp.ScopedStream(stream, sync_enter=False):
            run_counter_writeback(
                runtime,
                counter_bufs,
                counter_arrays,
                counter_state_buffers,
                record_counts=counter_record_counts,
            )

    else:
        # === Single-pass (scatter only, or inactive deterministic params only) ===
        det_params = build_det_params(phase=1, use_scatter=True)
        params_all = [*user_params, *det_params]
        do_cuda_launch(launch_hook, params_all)

    # Post-kernel: reduce scatter records in a fixed order.
    if active_scatter_targets:
        # Identify the destination arrays from fwd_args.
        scatter_targets = []
        scatter_buffers = []
        dest_arrays = []
        record_counts = []
        for scatter_target, scatter_buffer in zip(active_scatter_targets, scatter_bufs, strict=True):
            dest_array = resolve_det_target_array(scatter_target)
            if dest_array is None:
                continue
            if not hasattr(dest_array, "ptr"):
                raise RuntimeError(
                    "Deterministic scatter target "
                    f"{scatter_target.target_label!r} resolved to non-array object "
                    f"({type(dest_array).__name__})"
                )
            if getattr(dest_array, "ptr", None) is None or getattr(dest_array, "size", 0) <= 0:
                continue
            scatter_targets.append(scatter_target)
            scatter_buffers.append(scatter_buffer)
            dest_arrays.append(dest_array)
            record_counts.append(get_scatter_record_count(scatter_buffer, stream_is_capturing=stream_is_capturing))

        with warp.ScopedStream(stream, sync_enter=False):
            sort_reduce_workspaces = run_sort_reduce(
                runtime,
                scatter_targets,
                scatter_buffers,
                dest_arrays,
                device,
                determinism_mode,
                record_counts=record_counts,
            )

    if capture_graph is not None:
        # Captured graphs replay after this function returns, so keep temporary
        # deterministic buffers alive for the lifetime of the graph object.
        capture_graph._deterministic_buffer_refs.extend(
            buffer
            for buffer in (
                *scatter_bufs,
                *counter_bufs,
                overflow_buf,
                *sort_reduce_workspaces,
                *counter_scan_workspaces,
                *(array for state_buffer in counter_state_buffers for array in state_buffer[:2] if array is not None),
            )
            if buffer is not None
        )

    try:
        runtime.verify_cuda_device(device)
    except Exception:
        from warp._src import utils as warp_utils  # noqa: PLC0415

        warp_utils.warn(f"Error in deterministic kernel launch: {kernel.key} on device {device}")
        raise

    if overflow_buf is not None and not stream_is_capturing and int(overflow_buf.numpy()[0]) != 0:
        overflow_kind = "scatter" if det_meta.has_scatter else "counter"
        raise RuntimeError(
            f"Deterministic {overflow_kind} buffer overflow in kernel '{kernel.key}'. "
            "Increase 'deterministic_max_records' or reduce the per-thread atomic count."
        )
