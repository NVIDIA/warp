# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reverse the actual ordered Gauss--Seidel iteration map (opt-in substrate).

This module is not imported by reconstruction. It is the solver building block
for the checkpointed finite reconstruction backward pass.
"""

import threading
from functools import lru_cache
from types import SimpleNamespace

import warp as wp

from .relaxation import SlicedGraph, wavefront_groups

wp.set_module_options({"fuse_fp": False})


@lru_cache(None)
def precision_kernels(dtype):
    pair = wp.types.vector(length=2, dtype=dtype)

    @wp.kernel
    def copy_kernel(src: wp.array[dtype], dst: wp.array[dtype]):
        i = wp.tid()
        dst[i] = src[i]

    @wp.kernel
    def add_kernel(src: wp.array[dtype], dst: wp.array[dtype]):
        i = wp.tid()
        wp.atomic_add(dst, i, src[i])

    @wp.func
    def forward_row(
        i: int,
        visit: int,
        offsets: wp.array[int],
        columns: wp.array[int],
        values: wp.array[dtype],
        b: wp.array[dtype],
        x: wp.array[dtype],
        history: wp.array2d[pair],
    ):
        r = b[i]
        diagonal = dtype(0)
        for k in range(offsets[i], offsets[i + 1]):
            col = columns[k]
            r -= x[col] * values[k]
            if col == i:
                diagonal = values[k]
        # Store the old component and the actual rounded residual. Rewinding
        # each independent group reconstructs all its input neighbor values.
        history[visit, i] = pair(x[i], r)
        if diagonal != dtype(0):
            x[i] += r * (dtype(1) / diagonal)

    @wp.func
    def reverse_row(
        i: int,
        visit: int,
        offsets: wp.array[int],
        columns: wp.array[int],
        values: wp.array[dtype],
        x: wp.array[dtype],
        history: wp.array2d[pair],
        adj_x: wp.array[dtype],
        adj_b: wp.array[dtype],
        adj_values: wp.array[dtype],
    ):
        diagonal = dtype(0)
        diagonal_index = int(-1)
        for k in range(offsets[i], offsets[i + 1]):
            if columns[k] == i:
                diagonal = values[k]
                diagonal_index = k
        old = history[visit, i]
        x[i] = old[0]
        if diagonal != dtype(0):
            seed = adj_x[i]
            inv = dtype(1) / diagonal
            adj_r = seed * inv
            if adj_b.shape[0] != 0:
                wp.atomic_add(adj_b, i, adj_r)
            # Keep both contributions from the actual update expression:
            # x_i + (b_i - sum_j A_ij*x_j) * (1/A_ii).
            # This is the iteration derivative, not an implicit solve.
            if adj_values.shape[0] != 0:
                wp.atomic_add(adj_values, diagonal_index, -(seed * old[1]) / (diagonal * diagonal))
            for k in range(offsets[i], offsets[i + 1]):
                col = columns[k]
                if adj_values.shape[0] != 0:
                    wp.atomic_add(adj_values, k, -adj_r * x[col])
                wp.atomic_add(adj_x, col, -adj_r * values[k])

    @wp.kernel
    def forward_kernel(
        rows: wp.array[int],
        group_offsets: wp.array[int],
        first_group: int,
        first_visit: int,
        group_stride: int,
        offsets: wp.array[int],
        columns: wp.array[int],
        values: wp.array[dtype],
        b: wp.array[dtype],
        x: wp.array[dtype],
        history: wp.array2d[pair],
    ):
        batch, row = wp.tid()
        group = first_group + batch * group_stride
        j = group_offsets[group] + row
        if j < group_offsets[group + 1]:
            forward_row(rows[j], first_visit + batch, offsets, columns, values, b, x, history)

    @wp.kernel
    def reverse_kernel(
        rows: wp.array[int],
        group_offsets: wp.array[int],
        first_group: int,
        first_visit: int,
        group_stride: int,
        offsets: wp.array[int],
        columns: wp.array[int],
        values: wp.array[dtype],
        x: wp.array[dtype],
        history: wp.array2d[pair],
        adj_x: wp.array[dtype],
        adj_b: wp.array[dtype],
        adj_values: wp.array[dtype],
    ):
        batch, row = wp.tid()
        group = first_group + batch * group_stride
        j = group_offsets[group] + row
        if j < group_offsets[group + 1]:
            reverse_row(rows[j], first_visit + batch, offsets, columns, values, x, history, adj_x, adj_b, adj_values)

    state_type = wp.struct(
        type(
            f"RelaxationAdjointState_{dtype.__name__}",
            (),
            {
                "__annotations__": {
                    "rows": wp.array[int],
                    "group_offsets": wp.array[int],
                    "offsets": wp.array[int],
                    "columns": wp.array[int],
                    "values": wp.array[dtype],
                    "b": wp.array[dtype],
                    "x": wp.array[dtype],
                    "history": wp.array2d[pair],
                    "adj_x": wp.array[dtype],
                    "adj_b": wp.array[dtype],
                    "adj_values": wp.array[dtype],
                }
            },
        )
    )

    @wp.kernel
    def bind_kernel(state: state_type, descriptor: wp.array[state_type]):
        descriptor[0] = state

    @wp.kernel
    def graph_forward_kernel(descriptor: wp.array[state_type], first_group: int, first_visit: int, stride: int):
        batch, row = wp.tid()
        state = descriptor[0]
        group = first_group + batch * stride
        j = state.group_offsets[group] + row
        if j < state.group_offsets[group + 1]:
            forward_row(
                state.rows[j],
                first_visit + batch,
                state.offsets,
                state.columns,
                state.values,
                state.b,
                state.x,
                state.history,
            )

    @wp.kernel
    def graph_reverse_kernel(descriptor: wp.array[state_type], first_group: int, first_visit: int, stride: int):
        batch, row = wp.tid()
        state = descriptor[0]
        group = first_group + batch * stride
        j = state.group_offsets[group] + row
        if j < state.group_offsets[group + 1]:
            reverse_row(
                state.rows[j],
                first_visit + batch,
                state.offsets,
                state.columns,
                state.values,
                state.x,
                state.history,
                state.adj_x,
                state.adj_b,
                state.adj_values,
            )

    return SimpleNamespace(
        copy=copy_kernel,
        add=add_kernel,
        forward=forward_kernel,
        reverse=reverse_kernel,
        state_type=state_type,
        bind=bind_kernel,
        graph_forward=graph_forward_kernel,
        graph_reverse=graph_reverse_kernel,
    )


@lru_cache(maxsize=32)
def _cached_graph(dtype, device, stream, program, capacity, reverse):
    kernels = precision_kernels(dtype)
    descriptor = wp.empty(1, dtype=kernels.state_type, device=device)
    kernel = kernels.graph_reverse if reverse else kernels.graph_forward
    command = wp.launch(
        kernel, (1, capacity), inputs=[descriptor, 0, 0, 0], device=device, record_cmd=True, record_tape=False
    )
    wp.launch(
        kernels.bind, 1, inputs=[kernels.state_type(), descriptor], device=device, record_cmd=True, record_tape=False
    )
    with wp.ScopedCapture(device=device) as capture:
        for group, visit, count, stride in reversed(program) if reverse else program:
            command.set_dim((count, capacity))
            command.set_param_at_index(1, group)
            command.set_param_at_index(2, visit)
            command.set_param_at_index(3, stride)
            command.launch()
    return SlicedGraph(descriptor, capture.graph, wp.get_stream(device), threading.Lock())


def _launch_graph(dtype, device, program, maximum, state, reverse=False):
    capacity = 1 << max(0, maximum - 1).bit_length()
    cached = _cached_graph(dtype, str(device), wp.get_stream(device).cuda_stream, tuple(program), capacity, reverse)
    with cached.lock:
        wp.launch(precision_kernels(dtype).bind, 1, inputs=[state, cached.descriptor], device=device, record_tape=False)
        wp.capture_launch(cached.graph)


def differentiable_relaxation(matrix, b, initial, schedule, *, tape, iters, forward=True, sliced_depth=None):
    """Return a tracked finite-iteration solution without mutating ``initial``.

    Differentiate with respect to tracked CSR ``matrix.values``, ``b``, and
    ``initial``. The caller supplies its ``wp.Tape`` explicitly. The row schedule
    and CSR topology are fixed. As in production relaxation, rows in a schedule
    group must be uncoupled except for their diagonals, and CSR columns must be
    unique within each row. Inputs, schedules, and returned outputs must remain
    unchanged until backward completes. Each invocation owns its saved history;
    independent tapes can run on independent streams. Gradients accumulate into
    the caller's buffers, so concurrent backwards must use distinct inputs.

    ``sliced_depth=None`` uses regular forward/reverse color sweeps; otherwise
    use the reference's sliced fine-level schedule. The latter has the same
    arithmetic/dependencies as the production wavefront schedule and requires
    the degree-one stencil (no coupling across more than one z slice). Save two scalar
    values per row visit, rather than a whole vector per group. Reverse execution
    rewinds a private solution copy, leaving the output and caller arrays intact.

    This deliberately separate API adds no allocations, launches, tape handling,
    or imports to forward-only reconstruction. Full reconstruction gradients,
    multigrid composition, root CG, higher-order derivatives, and external CUDA
    graph capture are not implemented by this function.
    """
    if not isinstance(tape, wp.Tape):
        raise ValueError("an explicit wp.Tape is required")
    if not isinstance(iters, int) or isinstance(iters, bool) or iters < 0:
        raise ValueError("iters must be a nonnegative integer")
    if (
        b.dtype not in (wp.float32, wp.float64)
        or initial.dtype != b.dtype
        or matrix.values.dtype != b.dtype
        or b.ndim != 1
        or initial.shape != b.shape
        or not len(b)
        or matrix.nrow != len(b)
        or matrix.ncol != len(b)
    ):
        raise ValueError("require a nonempty square scalar CSR matrix and matching float32/float64 vectors")
    device = b.device
    if device.is_capturing:
        raise ValueError("differentiable relaxation does not support external CUDA graph capture")
    if any(
        a.device != device
        for a in [initial, matrix.values, matrix.offsets, matrix.columns, schedule.rows, schedule.offsets]
    ):
        raise ValueError("matrix, vectors, and schedule must share one device")
    if sliced_depth is not None:
        if not isinstance(sliced_depth, int) or isinstance(sliced_depth, bool) or not 0 <= sliced_depth <= 15:
            raise ValueError("sliced_depth must be an integer in 0..15")
        if not forward:
            raise ValueError("sliced relaxation uses the reference forward schedule")
        # Production's commuting wavefront batches retain each coupled row's
        # order. The visit index increases with z within one wave.
        program = []
        for step, (z, count, color) in enumerate(wavefront_groups(sliced_depth, iters)):
            wave = step // 4
            visit = (wave - (1 << sliced_depth) + z) // 2
            program.append((4 * z + color, visit, count, 8))
        group_count = 4 * ((1 << sliced_depth) + 1)
    else:
        program = [(g, visit, 1, 0) for visit in range(iters) for g in (range(8) if forward else range(7, -1, -1))]
        group_count = 8
    if schedule.offsets.shape != (group_count + 1,):
        raise ValueError("row schedule does not match the requested relaxation mode")
    kernels = precision_kernels(b.dtype)
    pair = wp.types.vector(length=2, dtype=b.dtype)
    output = wp.empty_like(initial, requires_grad=True)
    history = wp.empty((iters, len(b)), dtype=pair, device=device)
    wp.launch(kernels.copy, len(b), inputs=[initial, output], device=device, record_tape=False)
    schedule_arrays = [schedule.rows, schedule.offsets]
    matrix_arrays = [matrix.offsets, matrix.columns, matrix.values]
    state = kernels.state_type()
    state.rows, state.group_offsets = schedule_arrays
    state.offsets, state.columns, state.values = matrix_arrays
    state.b, state.x, state.history = b, output, history
    if device.is_cuda and program:
        _launch_graph(b.dtype, device, program, schedule.maximum, state)
    else:
        for group, visit, count, stride in program:
            wp.launch(
                kernels.forward,
                (count, schedule.maximum),
                inputs=[*schedule_arrays, group, visit, stride, *matrix_arrays, b, output, history],
                device=device,
                record_tape=False,
            )
    event = None
    if device.is_cuda:
        event = wp.Event(device)
        wp.get_stream(device).record_event(event)

    def backward():
        with wp.ScopedDevice(device):
            if event is not None:
                wp.wait_event(event)
            work = wp.clone(output, requires_grad=False)
            adjoint = wp.clone(output.grad, requires_grad=False)
            empty = wp.empty(0, dtype=b.dtype, device=device)
            adj_b = b.grad if b.requires_grad else empty
            adj_values = matrix.values.grad if matrix.values.requires_grad else empty
            reverse_state = kernels.state_type()
            reverse_state.rows, reverse_state.group_offsets = schedule_arrays
            reverse_state.offsets, reverse_state.columns, reverse_state.values = matrix_arrays
            reverse_state.x, reverse_state.history = work, history
            reverse_state.adj_x, reverse_state.adj_b, reverse_state.adj_values = adjoint, adj_b, adj_values
            if device.is_cuda and program:
                _launch_graph(b.dtype, device, program, schedule.maximum, reverse_state, reverse=True)
            else:
                for group, visit, count, stride in reversed(program):
                    wp.launch(
                        kernels.reverse,
                        (count, schedule.maximum),
                        inputs=[
                            *schedule_arrays,
                            group,
                            visit,
                            stride,
                            *matrix_arrays,
                            work,
                            history,
                            adjoint,
                            adj_b,
                            adj_values,
                        ],
                        device=device,
                        record_tape=False,
                    )
            if initial.requires_grad:
                wp.launch(kernels.add, len(initial), inputs=[adjoint, initial.grad], device=device, record_tape=False)

    tape.record_func(backward, [a for a in [matrix.values, b, initial, output] if a.requires_grad])
    return output
