# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``wp_stf.task`` with logical-data deps and array-backed tokens.

Token deps are sync-only; logical-data deps additionally hand the task body
a :class:`warp.array` view of the dep's storage. ``wp_stf.task`` builds those
views internally from the cai object returned by ``stf_task.args_cai()``.
These tests exercise that path so the cai-> ``wp.array`` aliasing (and the
underlying numpy-dtype -> warp-dtype inference) is validated end-to-end, and
also cover ``ctx.dep(wp.array)`` as an ordering-only token helper.
"""

import unittest

import numpy as np

import warp as wp
from warp import stf_experimental as wp_stf

N = 1024


@wp.kernel
def _scale_kernel(arr: wp.array[float], factor: float):
    arr[wp.tid()] = arr[wp.tid()] * factor


@wp.kernel
def _fill_kernel(arr: wp.array[float], value: float):
    arr[wp.tid()] = value


@wp.kernel
def _copy_kernel(src: wp.array[float], dst: wp.array[float]):
    i = wp.tid()
    dst[i] = src[i]


@unittest.skipUnless(wp.is_cuda_available() and wp_stf.is_available(), "CUDASTF is not available")
class TestSTFExperimentalLogicalData(unittest.TestCase):
    def test_dep_cuda_array_stream_context(self):
        """``ctx.dep(wp.array)`` memoizes ordering tokens for Warp arrays."""
        wp.init()
        device = wp.get_device("cuda:0")

        a = wp.zeros(N, dtype=wp.float32, device=device)
        b = wp.zeros(N, dtype=wp.float32, device=device)

        with wp_stf.context(stream=wp.get_stream(device)) as ctx:
            tok = ctx.token()
            dep_a = ctx.dep(a)

            self.assertIs(dep_a, ctx.dep(a))
            self.assertIs(ctx.dep(tok), tok)

            with ctx.task(dep_a.write(), tok.write()) as (s,):
                wp.launch(_fill_kernel, dim=N, inputs=[a, 5.0], stream=s)

            with ctx.task(ctx.dep(a).read(), ctx.dep(b).write(), tok.read()) as (s,):
                wp.launch(_copy_kernel, dim=N, inputs=[a, b], stream=s)

        np.testing.assert_allclose(b.numpy(), np.full(N, 5.0, dtype=np.float32))

    def test_initialized_logical_data(self):
        """``ctx.logical_data(numpy_array)`` wraps existing host data.

        The task body sees a :class:`warp.array` view of STF's device shadow
        and can mutate it via ``wp.launch``. STF stages the result back to
        the original host numpy array on ``finalize()``. This exercises the
        cai -> ``wp.array`` path with a non-empty initial value.
        """
        wp.init()
        device = wp.get_device("cuda:0")

        X = np.ones(N, dtype=np.float32)

        with wp_stf.context(stream=wp.get_stream(device)) as ctx:
            lX = ctx.logical_data(X)

            with ctx.task(lX.rw()) as (s, wX):
                wp.launch(_scale_kernel, dim=N, inputs=[wX, 3.0], stream=s)

        np.testing.assert_allclose(X, np.full(N, 3.0, dtype=np.float32))

    def test_dep_passes_through_logical_data(self):
        """``ctx.dep(logical_data)`` is a no-op for explicitly managed logical data."""
        wp.init()
        device = wp.get_device("cuda:0")

        X = np.ones(N, dtype=np.float32)

        with wp_stf.context(stream=wp.get_stream(device)) as ctx:
            lX = ctx.logical_data(X)
            self.assertIs(ctx.dep(lX), lX)

            with ctx.task(ctx.dep(lX).rw()) as (s, wX):
                wp.launch(_scale_kernel, dim=N, inputs=[wX, 4.0], stream=s)

        np.testing.assert_allclose(X, np.full(N, 4.0, dtype=np.float32))

    def test_empty_logical_data_scratchpad(self):
        """``ctx.logical_data_empty(shape, dtype)`` creates uninitialized storage.

        STF requires its first task-side access to be ``.write()`` since
        there is no source data to read. Used here as a sibling-task
        scratchpad: task 1 fills it write-only, task 2 reads it and writes
        a host-backed result. The assertion validates that the cai ->
        ``wp.array`` views in the two tasks alias the same backing storage,
        and that ``_np_to_wp_dtype`` correctly infers ``wp.float32`` from
        ``np.float32``.
        """
        wp.init()
        device = wp.get_device("cuda:0")

        result = np.zeros(N, dtype=np.float32)

        with wp_stf.context(stream=wp.get_stream(device)) as ctx:
            scratch = ctx.logical_data_empty((N,), dtype=np.float32)
            lresult = ctx.logical_data(result)

            # Task 1: write-only fill of the uninitialized scratchpad.
            with ctx.task(scratch.write()) as (s, w_scratch):
                wp.launch(_fill_kernel, dim=N, inputs=[w_scratch, 7.0], stream=s)

            # Task 2: read the scratchpad, write the result. STF orders this
            # task after task 1 because of the read/write dep on ``scratch``.
            with ctx.task(scratch.read(), lresult.write()) as (s, w_scratch, w_result):
                wp.launch(_copy_kernel, dim=N, inputs=[w_scratch, w_result], stream=s)

        np.testing.assert_allclose(result, np.full(N, 7.0, dtype=np.float32))

    def test_dep_exact_view_cache_key(self):
        """Only identical Warp array views share cached ordering tokens."""
        wp.init()
        device = wp.get_device("cuda:0")

        a = wp.zeros(N, dtype=wp.float32, device=device)
        same_view = wp.array(ptr=a.ptr, dtype=a.dtype, shape=a.shape, strides=a.strides, device=device)
        reshaped_view = wp.array(
            ptr=a.ptr,
            dtype=a.dtype,
            shape=(N // 2, 2),
            strides=(a.strides[0] * 2, a.strides[0]),
            device=device,
        )

        with wp_stf.context(stream=wp.get_stream(device)) as ctx:
            self.assertIs(ctx.dep(a), ctx.dep(same_view))
            self.assertIsNot(ctx.dep(a), ctx.dep(reshaped_view))

    def test_dep_rejects_invalid_inputs(self):
        """``ctx.dep`` only accepts CUDA Warp arrays and CUDASTF logical data."""
        wp.init()
        device = wp.get_device("cuda:0")

        with wp_stf.context(stream=wp.get_stream(device)) as ctx:
            with self.assertRaisesRegex(TypeError, "CUDA wp.array"):
                ctx.dep(wp.zeros(N, dtype=wp.float32, device="cpu"))

            with self.assertRaisesRegex(TypeError, "wp.array or cuda.stf.logical_data"):
                ctx.dep(42)

    def test_dep_task_graph_inline_array_tokens(self):
        """Task graphs can create array-backed token deps inline while recording."""
        wp.init()
        device = wp.get_device("cuda:0")

        with wp.ScopedDevice(device):
            graph = wp_stf.task_graph()
            ctx = graph.context
            a = wp.zeros(N, dtype=wp.float32, device=device)
            b = wp.zeros(N, dtype=wp.float32, device=device)

            with graph:
                with ctx.task(ctx.dep(a).write()) as (s,):
                    wp.launch(_fill_kernel, dim=N, inputs=[a, 9.0], stream=s)
                with ctx.task(ctx.dep(a).read(), ctx.dep(b).write()) as (s,):
                    wp.launch(_copy_kernel, dim=N, inputs=[a, b], stream=s)

            graph.launch()
            graph.launch()
            graph.finalize()

        np.testing.assert_allclose(b.numpy(), np.full(N, 9.0, dtype=np.float32))

    def test_dep_stackable_graph_scope_inline_array_tokens(self):
        """Stackable graph scopes can create array-backed token deps inline."""
        wp.init()
        device = wp.get_device("cuda:0")

        with wp.ScopedDevice(device), wp_stf.context() as ctx:
            a = wp.zeros(N, dtype=wp.float32, device=device)
            b = wp.zeros(N, dtype=wp.float32, device=device)

            with ctx.graph_scope():
                with ctx.task(ctx.dep(a).write()) as (s,):
                    wp.launch(_fill_kernel, dim=N, inputs=[a, 3.0], stream=s)
                with ctx.task(ctx.dep(a).read(), ctx.dep(b).write()) as (s,):
                    wp.launch(_copy_kernel, dim=N, inputs=[a, b], stream=s)

        np.testing.assert_allclose(b.numpy(), np.full(N, 3.0, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
