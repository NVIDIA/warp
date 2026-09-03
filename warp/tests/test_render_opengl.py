# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test OpenGL renderer logic that does not require a display or OpenGL context."""

import unittest
from unittest import mock

import numpy as np

import warp as wp
from warp.render import OpenGLRenderer


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
