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

import operator
import re
from dataclasses import dataclass, field

import numpy as np

import warp
from warp._src import utils as warp_utils
from warp._src.types import type_size_in_bytes


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

DETERMINISTIC_NOT_GUARANTEED = "not_guaranteed"
DETERMINISTIC_RUN_TO_RUN = "run_to_run"
DETERMINISTIC_GPU_TO_GPU = "gpu_to_gpu"

DETERMINISTIC_FAMILY_ADD = "add"
DETERMINISTIC_FAMILY_MIN = "min"
DETERMINISTIC_FAMILY_MAX = "max"
DETERMINISTIC_FAMILY_COUNTER = "counter"

_VALID_DETERMINISTIC_MODES = {
    DETERMINISTIC_NOT_GUARANTEED,
    DETERMINISTIC_RUN_TO_RUN,
    DETERMINISTIC_GPU_TO_GPU,
}

_DETERMINISTIC_MODE_IDS = {
    DETERMINISTIC_NOT_GUARANTEED: 0,
    DETERMINISTIC_RUN_TO_RUN: 1,
    DETERMINISTIC_GPU_TO_GPU: 2,
}


def normalize_deterministic_mode(value, option_name="deterministic", allow_none=False):
    """Normalize user-facing deterministic mode values.

    The public API accepts the explicit mode strings plus ``True``/``False``
    for ease of use:

    - ``False`` -> ``"not_guaranteed"``
    - ``True`` -> ``"run_to_run"``
    """
    if value is None:
        if allow_none:
            return None
        return DETERMINISTIC_NOT_GUARANTEED

    if isinstance(value, bool):
        return DETERMINISTIC_RUN_TO_RUN if value else DETERMINISTIC_NOT_GUARANTEED

    if isinstance(value, str):
        if value in _VALID_DETERMINISTIC_MODES:
            return value
        valid_modes = ", ".join(repr(mode) for mode in sorted(_VALID_DETERMINISTIC_MODES))
        raise ValueError(f"{option_name} must be one of {valid_modes}, got {value!r}")

    raise TypeError(f"{option_name} must be a bool or string, got {type(value).__name__}")


def normalize_deterministic_max_records(value, option_name="deterministic_max_records", allow_none=False) -> int | None:
    """Normalize the deterministic scatter record override.

    ``deterministic_max_records`` is a per-thread record count, so it must be
    a non-negative integer-like value. ``operator.index`` accepts NumPy integer
    scalars while rejecting lossy conversions from floats and strings.
    """
    if value is None:
        if allow_none:
            return None
        return 0

    if isinstance(value, bool):
        raise TypeError(f"{option_name} must be a non-negative integer, got bool")

    try:
        normalized = operator.index(value)
    except TypeError as e:
        raise TypeError(f"{option_name} must be a non-negative integer, got {type(value).__name__}") from e

    if normalized < 0:
        raise ValueError(f"{option_name} must be non-negative, got {normalized}")

    return normalized


def is_deterministic_mode_enabled(value) -> bool:
    """Return ``True`` if a deterministic mode stronger than default is enabled."""
    return normalize_deterministic_mode(value) != DETERMINISTIC_NOT_GUARANTEED


def deterministic_mode_to_id(value) -> int:
    """Map a normalized deterministic mode to the native enum id."""
    return _DETERMINISTIC_MODE_IDS[normalize_deterministic_mode(value)]


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

    scatter_targets: list[ScatterTarget] = field(default_factory=list)
    counter_targets: list[CounterTarget] = field(default_factory=list)
    scatter_records_per_thread: dict[ScatterTarget, int] = field(default_factory=dict)
    counter_records_per_thread: dict[CounterTarget, int] = field(default_factory=dict)
    needs_context: bool = False

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


# ---------------------------------------------------------------------------
# Warp type -> C++ type string mapping for scatter buffer value types
# ---------------------------------------------------------------------------

_WARP_TO_CTYPE = {
    warp.float16: "wp::half",
    warp.float32: "float",
    warp.float64: "double",
    warp.int32: "int",
    warp.uint32: "unsigned int",
    warp.int64: "int64_t",
    warp.uint64: "uint64_t",
}

_SCALAR_TYPE_IDS = {
    warp.float16: 0,
    warp.float32: 1,
    warp.float64: 2,
    warp.int32: 3,
    warp.uint32: 4,
    warp.int64: 5,
    warp.uint64: 6,
}

DETERMINISTIC_SCATTER_MAX_CAPACITY = (1 << 31) - 1


def warp_type_to_ctype(dtype) -> str:
    """Map a Warp scalar type to its C++ type string."""
    ctype = _WARP_TO_CTYPE.get(dtype)
    if ctype is None:
        raise ValueError(f"Unsupported scalar type for deterministic atomic: {dtype}")
    return ctype


def is_float_type(dtype) -> bool:
    """Return ``True`` if ``dtype`` is a Warp floating-point type."""
    return dtype in (warp.float16, warp.float32, warp.float64)


def warp_scalar_type_to_id(dtype) -> int:
    """Map a Warp scalar type to the native deterministic reducer enum."""
    type_id = _SCALAR_TYPE_IDS.get(dtype)
    if type_id is None:
        raise ValueError(f"Unsupported scalar type for deterministic atomic: {dtype}")
    return type_id


# ---------------------------------------------------------------------------
# Launch-time helpers
# ---------------------------------------------------------------------------


def allocate_scatter_buffers(scatter_targets, meta, dim_size, device, max_records=0, initialize_unused=False):
    """Allocate buffers for atomics reduced after the kernel finishes.

    Returns a list of ``(keys, values, counter, capacity)`` tuples, one per
    scatter target in ``scatter_targets``.
    """
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


def allocate_counter_target_table(counter_arrays, device):
    """Allocate the device table used to recognize counter stores in phase 0."""
    target_ptrs = []
    target_sizes = []

    for counter_arr in counter_arrays:
        if counter_arr is None:
            target_ptrs.append(0)
            target_sizes.append(0)
            continue

        counter_size_elements = _det_dest_size_elements(counter_arr)
        target_ptrs.append(counter_arr.ptr if counter_size_elements > 0 else 0)
        target_sizes.append(counter_size_elements)

    if not target_ptrs or not any(target_ptrs):
        return None, None, 0

    ptrs = warp.array(np.asarray(target_ptrs, dtype=np.uint64), dtype=warp.uint64, device=device)
    sizes = warp.array(np.asarray(target_sizes, dtype=np.int32), dtype=warp.int32, device=device)
    return ptrs, sizes, len(target_ptrs)


def allocate_counter_buffers(counter_targets, meta, dim_size, device, max_records=0, initialize_unused=False):
    """Allocate record buffers for deterministic consumed-return counters."""
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
    workspaces = []
    determinism_mode_id = deterministic_mode_to_id(determinism_mode)

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
