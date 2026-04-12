# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic execution mode for Warp.

This module implements the scatter-sort-reduce and two-pass strategies for
making atomic operations produce bit-exact reproducible results.

Two patterns of atomic usage are supported:

Pattern A — Accumulation (return value unused):
    ``wp.atomic_add(arr, idx, value)`` or ``arr[idx] += value``
    Strategy: Scatter records to a temporary buffer during the kernel, then
    sort by (dest_index, thread_id) and reduce in fixed order post-kernel.

Pattern B — Counter/Allocator (return value used):
    ``slot = wp.atomic_add(counter, 0, 1)``
    Strategy: Two-pass execution. Phase 0 records each thread's contribution
    with all side effects suppressed. Prefix sum computes deterministic
    offsets. Phase 1 re-executes with deterministic slot assignments.

See ``warp.config.deterministic`` for the user-facing configuration flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import warp
from warp._src import utils as warp_utils

# Reduction operation constants (must match C++ ReduceOp enum in deterministic.cu).
REDUCE_OP_ADD = 0
REDUCE_OP_MIN = 1
REDUCE_OP_MAX = 2

# Map from Warp builtin names to (is_accumulation, reduce_op).
# Atomics whose return value is consumed are always Pattern B (counter);
# this table is used only for Pattern A classification.
_ATOMIC_OP_INFO = {
    "atomic_add": (True, REDUCE_OP_ADD),
    "atomic_sub": (True, REDUCE_OP_ADD),  # codegen negates value before scattering
    "atomic_min": (True, REDUCE_OP_MIN),
    "atomic_max": (True, REDUCE_OP_MAX),
}

# Atomics that are associative+commutative on integers (no transform needed).
_ALREADY_DETERMINISTIC_OPS = {"atomic_and", "atomic_or", "atomic_xor"}

# Atomics that are inherently order-dependent (cannot be made deterministic).
_ORDER_DEPENDENT_OPS = {"atomic_cas", "atomic_exch"}


@dataclass
class ScatterTarget:
    """Tracks a Pattern A (accumulation) atomic target array during codegen."""

    array_var_label: str  # label of the target array Var
    value_dtype: type  # Warp dtype of the accumulated value (e.g., wp.float32, wp.vec3)
    value_ctype: str  # C type of the value (e.g., "float", "wp::vec_t<3, float>")
    scalar_dtype: type  # scalar component dtype (e.g., wp.float32)
    reduce_op: int  # REDUCE_OP_ADD, REDUCE_OP_MIN, or REDUCE_OP_MAX
    index: int = 0  # scatter buffer index (assigned during codegen)
    records_per_thread: int = 1  # static estimate of emitted records per thread


@dataclass
class CounterTarget:
    """Tracks a Pattern B (counter/allocator) atomic target array during codegen."""

    array_var_label: str  # label of the target array Var
    value_ctype: str  # C type of the counter value (e.g., "int")
    index: int = 0  # counter buffer index (assigned during codegen)


@dataclass
class DeterministicMeta:
    """Metadata attached to a kernel's Adjoint after codegen in deterministic mode.

    Used by the launch system to allocate scatter/counter buffers and
    orchestrate the multi-pass execution.
    """

    scatter_targets: list[ScatterTarget] = field(default_factory=list)
    counter_targets: list[CounterTarget] = field(default_factory=list)

    @property
    def has_scatter(self):
        return len(self.scatter_targets) > 0

    @property
    def has_counter(self):
        return len(self.counter_targets) > 0

    @property
    def needs_deterministic(self):
        return self.has_scatter or self.has_counter


def get_or_create_scatter_target(meta, array_var_label, value_dtype, value_ctype, scalar_dtype, reduce_op):
    """Get existing scatter target for an array, or create a new one.

    Multiple atomic call sites targeting the same array and reduction op share
    one scatter buffer.
    """
    for target in meta.scatter_targets:
        if (
            target.array_var_label == array_var_label
            and target.value_dtype == value_dtype
            and target.value_ctype == value_ctype
            and target.scalar_dtype == scalar_dtype
            and target.reduce_op == reduce_op
        ):
            target.records_per_thread += 1
            return target
    target = ScatterTarget(
        array_var_label=array_var_label,
        value_dtype=value_dtype,
        value_ctype=value_ctype,
        scalar_dtype=scalar_dtype,
        reduce_op=reduce_op,
        index=len(meta.scatter_targets),
    )
    meta.scatter_targets.append(target)
    return target


def get_or_create_counter_target(meta, array_var_label, value_ctype):
    """Get existing counter target for an array, or create a new one."""
    for target in meta.counter_targets:
        if target.array_var_label == array_var_label:
            return target
    target = CounterTarget(
        array_var_label=array_var_label,
        value_ctype=value_ctype,
        index=len(meta.counter_targets),
    )
    meta.counter_targets.append(target)
    return target


# ---------------------------------------------------------------------------
# Warp type → C++ type string mapping for scatter buffer value types
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


def warp_type_to_ctype(dtype) -> str:
    """Map a Warp scalar type to its C++ type string."""
    ctype = _WARP_TO_CTYPE.get(dtype)
    if ctype is None:
        raise ValueError(f"Unsupported scalar type for deterministic atomic: {dtype}")
    return ctype


def is_float_type(dtype) -> bool:
    """Return True if dtype is a Warp floating-point type."""
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


def allocate_scatter_buffers(scatter_targets, dim_size, device, max_records=0):
    """Allocate scatter buffers for Pattern A targets.

    Args:
        scatter_targets: Deterministic scatter target metadata collected during
            code generation.
        dim_size: Launch dimension size. This corresponds to the number of
            threads that may emit scatter records.
        device: Target device for the temporary buffers.
        max_records: Optional per-target, per-thread override for the maximum
            number of scatter records a thread may emit. The final buffer size
            uses ``max(codegen_lower_bound, max_records)`` records per thread.

    Returns:
        A list of ``(keys, values, counter, overflow, capacity)`` tuples, one
        per scatter target.
    """
    buffers = []
    for target in scatter_targets:
        records_per_thread = max(target.records_per_thread, max_records)
        capacity = max(dim_size * records_per_thread, 1024)
        keys = warp.full(shape=(capacity,), value=-1, dtype=warp.int64, device=device)
        values = warp.zeros(shape=(capacity,), dtype=target.value_dtype, device=device)
        counter = warp.zeros(shape=(1,), dtype=warp.int32, device=device)
        overflow = warp.zeros(shape=(1,), dtype=warp.int32, device=device)
        buffers.append((keys, values, counter, overflow, capacity))
    return buffers


def allocate_counter_buffers(counter_targets, dim_size, device):
    """Allocate counter buffers for Pattern B targets.

    Returns a list of (contrib, prefix) tuples.
    """
    buffers = []
    for _target in counter_targets:
        contrib = warp.zeros(shape=(dim_size,), dtype=warp.int32, device=device)
        prefix = warp.empty(shape=(dim_size,), dtype=warp.int32, device=device)
        buffers.append((contrib, prefix))
    return buffers


def run_sort_reduce(runtime, scatter_targets, scatter_buffers, dest_arrays, device):
    """Execute post-kernel sort-reduce for all Pattern A scatter targets.

    Args:
        runtime: The Warp Runtime object with native function bindings.
        scatter_targets: List of ScatterTarget metadata.
        scatter_buffers: List of (keys, values, counter, capacity) tuples.
        dest_arrays: List of destination warp.array objects (parallel to scatter_targets).
        device: The target device.
    """
    for i, target in enumerate(scatter_targets):
        keys, values, _counter, _overflow, capacity = scatter_buffers[i]
        dest_arr = dest_arrays[i]

        try:
            scalar_type_id = warp_scalar_type_to_id(target.scalar_dtype)
        except ValueError:
            warp_utils.warn(f"Unsupported value type '{target.value_ctype}' for deterministic sort-reduce.")
            continue

        runtime.core.wp_deterministic_sort_reduce_device(
            keys.ptr,
            values.ptr,
            capacity,
            dest_arr.ptr,
            dest_arr.size,
            target.reduce_op,
            scalar_type_id,
            getattr(target.value_dtype, "_length_", 1),
        )
