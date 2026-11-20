# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test OpenGL renderer logic that does not require a display or OpenGL context."""

import ctypes
import unittest
from unittest import mock

import numpy as np

import warp as wp
from warp._src import context
from warp.render import OpenGLRenderer


class TestRegisteredGLBuffer(unittest.TestCase):
    @staticmethod
    def _make_gl(data):
        gl = mock.Mock()
        uploaded = []

        def read_buffer(target, offset, nbytes, ptr):
            ctypes.memmove(ptr, data.ctypes.data + offset, nbytes)

        def write_buffer(target, nbytes, ptr, usage):
            uploaded.append(np.frombuffer(ctypes.string_at(ptr, nbytes), dtype=data.dtype).copy())

        gl.glGetBufferSubData.side_effect = read_buffer
        gl.glBufferData.side_effect = write_buffer
        return gl, uploaded

    def test_cpu_buffer_skips_cuda_registration(self):
        device = wp.get_device("cpu")
        with (
            mock.patch.object(
                context.runtime.core, "wp_cuda_graphics_register_gl_buffer", return_value=None
            ) as register,
            mock.patch.object(context, "log_warning"),
        ):
            buffer = wp.RegisteredGLBuffer(17, device)
            self.assertIsNone(buffer.resource)
            register.assert_not_called()

    def test_cpu_buffer_requires_copy_fallback(self):
        device = wp.get_device("cpu")
        with mock.patch.object(
            context.runtime.core, "wp_cuda_graphics_register_gl_buffer", return_value=None
        ) as register:
            with self.assertRaisesRegex(RuntimeError, "CPU.*fallback_to_copy"):
                wp.RegisteredGLBuffer(17, device, fallback_to_copy=False)
            register.assert_not_called()

    def test_cpu_buffer_copy_flags(self):
        device = wp.get_device("cpu")
        data = np.arange(4, dtype=np.float32)
        replacement = data + 10.0
        real_empty = wp.empty

        def allocate(shape, dtype, **kwargs):
            self.assertFalse(kwargs.get("pinned", False))
            return real_empty(shape, dtype, **kwargs)

        for flags in (wp.RegisteredGLBuffer.NONE, wp.RegisteredGLBuffer.READ_ONLY, wp.RegisteredGLBuffer.WRITE_DISCARD):
            with self.subTest(flags=flags):
                gl, uploaded = self._make_gl(data)
                with (
                    mock.patch.dict("sys.modules", {"pyglet": mock.Mock(gl=gl)}),
                    mock.patch.object(context.runtime.core, "wp_cuda_graphics_register_gl_buffer", return_value=None),
                    mock.patch.object(context, "log_warning"),
                    mock.patch.object(wp, "empty", side_effect=allocate) as empty,
                    mock.patch.object(wp, "copy", wraps=wp.copy) as copy,
                ):
                    buffer = wp.RegisteredGLBuffer(17, device, flags)
                    array = buffer.map(wp.float32, (4,))
                    if flags != wp.RegisteredGLBuffer.WRITE_DISCARD:
                        np.testing.assert_array_equal(array.numpy(), data)
                    self.assertIs(array, buffer.warp_buffer_cpu)
                    self.assertIs(array, buffer.map(wp.float32, (4,)))
                    empty.assert_called_once_with((4,), wp.float32, device=device)
                    copy.assert_not_called()
                    array.assign(replacement)
                    buffer.unmap()

                    self.assertEqual(gl.glGetBufferSubData.call_count, 0 if flags == buffer.WRITE_DISCARD else 2)
                    self.assertEqual(len(uploaded), 0 if flags == buffer.READ_ONLY else 1)
                    if uploaded:
                        np.testing.assert_array_equal(uploaded[0], replacement)

    def test_cuda_buffer_copy_fallback(self):
        wp.get_device("cpu")
        device = mock.Mock(is_cpu=False, is_cuda=True, context=123)
        data = np.arange(4, dtype=np.float32)
        replacement = data + 10.0
        warning_flag = "_RegisteredGLBuffer__fallback_warning_shown"
        previous_warning_shown = getattr(wp.RegisteredGLBuffer, warning_flag)

        for flags in (wp.RegisteredGLBuffer.NONE, wp.RegisteredGLBuffer.READ_ONLY, wp.RegisteredGLBuffer.WRITE_DISCARD):
            with self.subTest(flags=flags):
                gl, uploaded = self._make_gl(data)
                staging = wp.empty((4,), wp.float32, device="cpu")
                array = mock.Mock(dtype=wp.float32, shape=(4,), size=4)
                array.numpy.return_value = replacement
                with (
                    mock.patch.object(wp.RegisteredGLBuffer, warning_flag, False),
                    mock.patch.dict("sys.modules", {"pyglet": mock.Mock(gl=gl)}),
                    mock.patch.object(context, "get_device", return_value=device),
                    mock.patch.object(
                        context.runtime.core, "wp_cuda_graphics_register_gl_buffer", return_value=None
                    ) as register,
                    mock.patch.object(context, "log_warning") as warning,
                    mock.patch.object(wp, "empty", side_effect=(array, staging)) as empty,
                    mock.patch.object(wp, "copy") as copy,
                ):
                    buffer = wp.RegisteredGLBuffer(17, device, flags)
                    warning.assert_called_once()
                    self.assertIs(buffer.map(wp.float32, (4,)), array)
                    register.assert_called_once_with(device.context, 17, flags)
                    self.assertEqual(
                        empty.call_args_list,
                        [
                            mock.call((4,), wp.float32, device=device),
                            mock.call((4,), wp.float32, device="cpu", pinned=True),
                        ],
                    )
                    if flags == buffer.WRITE_DISCARD:
                        copy.assert_not_called()
                        gl.glGetBufferSubData.assert_not_called()
                    else:
                        copy.assert_called_once_with(array, staging)
                        np.testing.assert_array_equal(staging.numpy(), data)
                    buffer.unmap()
                    self.assertEqual(len(uploaded), 0 if flags == buffer.READ_ONLY else 1)
                    if uploaded:
                        np.testing.assert_array_equal(uploaded[0], replacement)
                self.assertIs(getattr(wp.RegisteredGLBuffer, warning_flag), previous_warning_shown)


class TestOpenGLRenderer(unittest.TestCase):
    @staticmethod
    def _make_renderer():
        renderer = OpenGLRenderer.__new__(OpenGLRenderer)
        renderer._device = wp.get_device("cpu")
        renderer._instances = {}
        renderer._shape_geo_hash = {}
        renderer.register_shape = mock.Mock(return_value=0)
        return renderer

    def test_render_mesh_rejects_out_of_range_indices(self):
        """Verify that mesh rendering rejects indices outside the point array."""
        points = np.array(
            [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ],
            dtype=np.float32,
        )

        invalid_indices = (
            ((0, 1, -1), -1),
            ((0, 1, len(points)), len(points)),
            ((0, 1, np.iinfo(np.int32).max), np.iinfo(np.int32).max),
            (np.array((0, 1, 1 << 32), dtype=np.uint64), 1 << 32),
        )

        for smooth_shading in (False, True):
            for indices, invalid_index in invalid_indices:
                renderer = self._make_renderer()

                with self.subTest(smooth_shading=smooth_shading, invalid_index=invalid_index):
                    with mock.patch.object(wp, "launch") as launch:
                        with self.assertRaises(ValueError) as context:
                            renderer.render_mesh(
                                "mesh",
                                points,
                                indices,
                                update_topology=True,
                                is_template=True,
                                smooth_shading=smooth_shading,
                            )

                    launch.assert_not_called()
                    message = str(context.exception)
                    self.assertIn(str(invalid_index), message)
                    self.assertIn(f"[0, {len(points)})", message)

    def test_render_mesh_does_not_validate_unchanged_topology(self):
        """Verify that unchanged topology skips index validation and shading kernels."""
        for indices in ((-1, -1, -1), (0, 0, 1 << 32)):
            renderer = self._make_renderer()
            renderer._instances["mesh"] = (0, None, 7)
            renderer.update_shape_instance = mock.Mock()
            renderer.update_shape_vertices = mock.Mock()

            with self.subTest(indices=indices):
                with mock.patch.object(wp, "launch") as launch:
                    try:
                        shape = renderer.render_mesh(
                            "mesh",
                            ((0.0, 0.0, 0.0),),
                            indices,
                            update_topology=False,
                        )
                    except Exception as error:
                        self.fail(f"Expected unchanged topology to be reused, but got {type(error).__name__}: {error}")

                self.assertEqual(shape, 7)
                launch.assert_not_called()

    def test_render_mesh_reports_wide_python_indices_as_value_error(self):
        """Verify that wide Python integers raise a mesh index range ValueError."""
        renderer = self._make_renderer()

        with mock.patch.object(wp, "launch") as launch:
            try:
                renderer.render_mesh(
                    "mesh",
                    ((0.0, 0.0, 0.0),),
                    (0, 0, 1 << 32),
                    update_topology=True,
                    is_template=True,
                )
            except ValueError as error:
                message = str(error)
            except Exception as error:
                self.fail(f"Expected ValueError, but got {type(error).__name__}: {error}")
            else:
                self.fail("Expected ValueError, but no exception was raised")

        launch.assert_not_called()
        self.assertIn(str(1 << 32), message)
        self.assertIn("[0, 1)", message)

    def test_render_mesh_rejects_indices_outside_int32_range(self):
        """Verify that mesh indices must be representable by Warp's int32 index type."""
        renderer = self._make_renderer()
        points = np.array(((0.0, 0.0, 0.0),), dtype=np.float32)
        max_index = 1 << 31
        point_count = max_index + 1

        def simulated_len(value):
            if isinstance(value, np.ndarray) and value.dtype == np.float32:
                return point_count
            return len(value)

        with mock.patch.dict(OpenGLRenderer.render_mesh.__globals__, {"len": simulated_len}):
            with mock.patch.object(wp, "launch", side_effect=AssertionError("Kernel launched before validation")):
                with self.assertRaises(ValueError) as context:
                    renderer.render_mesh(
                        "mesh",
                        points,
                        np.array((0, 1, max_index), dtype=np.uint64),
                        update_topology=True,
                        is_template=True,
                        smooth_shading=False,
                    )

        message = str(context.exception)
        self.assertIn(str(max_index), message)
        self.assertIn(f"[0, {max_index})", message)

    def test_render_mesh_accepts_empty_indices(self):
        """Verify that mesh rendering accepts meshes without faces."""
        points = np.array(
            [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ],
            dtype=np.float32,
        )

        for smooth_shading in (False, True):
            renderer = self._make_renderer()
            with self.subTest(smooth_shading=smooth_shading):
                with mock.patch.object(wp, "launch"):
                    shape = renderer.render_mesh(
                        "mesh",
                        points,
                        (),
                        update_topology=True,
                        is_template=True,
                        smooth_shading=smooth_shading,
                    )

                self.assertEqual(shape, 0)

    def test_setup_tiled_rendering_requires_list(self):
        """Verify that tiled rendering rejects a non-list instances container."""
        renderer = OpenGLRenderer.__new__(OpenGLRenderer)

        with self.assertRaisesRegex(TypeError, r"instances must be a list of lists, got tuple"):
            renderer.setup_tiled_rendering(([0],))

    def test_setup_tiled_rendering_requires_nonempty_instances(self):
        """Verify that tiled rendering rejects an empty instances list."""
        renderer = OpenGLRenderer.__new__(OpenGLRenderer)

        with self.assertRaisesRegex(ValueError, r"instances must be non-empty, got \[\]"):
            renderer.setup_tiled_rendering([])

    def test_setup_tiled_rendering_requires_list_elements(self):
        """Verify that tiled rendering rejects non-list instance entries."""
        renderer = OpenGLRenderer.__new__(OpenGLRenderer)

        with self.assertRaisesRegex(TypeError, r"instances elements must be lists, got tuple at index 1"):
            renderer.setup_tiled_rendering([[0], (1,)])

    def test_get_pixels_requires_tiled_rendering_setup(self):
        """Verify that tiled pixel reads require configured viewports."""
        renderer = OpenGLRenderer.__new__(OpenGLRenderer)
        renderer._switch_context = mock.Mock()
        renderer._tile_instances = None
        renderer._tile_viewports = None
        renderer._tile_width = 0
        renderer._tile_height = 0

        with self.assertRaisesRegex(
            RuntimeError,
            r"get_pixels\(\) requires setup_tiled_rendering\(\) when split_up_tiles=True",
        ):
            renderer.get_pixels(object())

    def test_update_tile_reports_out_of_bounds_index(self):
        """Verify that updating an invalid tile reports an indexing error."""
        renderer = OpenGLRenderer.__new__(OpenGLRenderer)
        renderer._tile_instances = [[0]]

        for tile_id in (-1, 1):
            with self.subTest(tile_id=tile_id):
                with self.assertRaisesRegex(IndexError, rf"tile_id {tile_id} is out of bounds for 1 tile"):
                    renderer.update_tile(tile_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
