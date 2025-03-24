# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the mesh deformation sample scene."""

import unittest
from typing import Tuple

import numpy as np
import omni.graph.core as og
import omni.kit
import omni.usd
import omni.warp

from ._common import (
    FrameRange,
    array_are_almost_equal,
    attr_disconnect_all,
    open_sample,
    validate_render,
)

TEST_ID = "mesh_deformation"


class TestSampleMeshDeformation(omni.kit.test.AsyncTestCase):
    async def _test_eval(self, enable_fsd: bool) -> None:
        await open_sample(f"{TEST_ID}.usda", enable_fsd=enable_fsd)

        graph = og.Controller.graph("/World/ActionGraph")

        # Force writing to USD so that we can read resulting values from the stage.
        write_node = og.Controller.node("write_prims", graph)
        write_usd_attr = og.Controller.attribute("inputs:usdWriteBack", write_node)
        attr_disconnect_all(write_usd_attr)
        write_usd_attr.set(True)

        grid_node = og.Controller.node("grid_create", graph)
        grid_size_attr = og.Controller.attribute("inputs:size", grid_node)
        grid_dims_attr = og.Controller.attribute("inputs:dims", grid_node)

        stage = omni.usd.get_context().get_stage()
        mesh_prim = stage.GetPrimAtPath("/World/Mesh")
        points_attr = mesh_prim.GetAttribute("points")

        async def test_variant(
            grid_size: Tuple[int, int],
            grid_dims: Tuple[int, int],
        ) -> None:
            grid_size_attr.set(grid_size)
            grid_dims_attr.set(grid_dims)

            prev_points_hash = None
            curr_points_hash = None

            with FrameRange(30) as frames:
                async for _ in frames:
                    points = np.array(points_attr.Get())
                    assert np.isfinite(points).all()
                    assert points.shape == ((grid_dims[0] + 1) * (grid_dims[1] + 1), 3)
                    array_are_almost_equal(
                        np.min(points, axis=0), (-grid_size[0] * 0.5, -10.0, -grid_size[1] * 0.5), atol=1.0
                    )
                    array_are_almost_equal(
                        np.max(points, axis=0), (grid_size[0] * 0.5, 10.0, grid_size[1] * 0.5), atol=1.0
                    )

                    curr_points_hash = hash(points.tobytes())
                    assert curr_points_hash != prev_points_hash
                    prev_points_hash = curr_points_hash

        await test_variant(grid_size=(100.0, 100.0), grid_dims=(32, 32))
        await test_variant(grid_size=(50.0, 10.0), grid_dims=(64, 8))

    async def test_eval_fsd_off(self) -> None:
        await self._test_eval(enable_fsd=False)

    async def test_eval_fsd_on(self) -> None:
        await self._test_eval(enable_fsd=True)

    async def _test_capture(self, enable_fsd: bool) -> None:
        await open_sample(f"{TEST_ID}.usda", enable_fsd=enable_fsd)

        with FrameRange(30) as frames:
            async for _ in frames:
                pass

        fsd_str = "fsd_on" if enable_fsd else "fsd_off"
        await validate_render(f"{TEST_ID}_{fsd_str}")

    @unittest.skipIf(omni.kit.test.utils.is_etm_run(), "Regression in Kit")
    async def test_capture_fsd_off(self) -> None:
        await self._test_capture(enable_fsd=False)

    @unittest.skipIf(omni.kit.test.utils.is_etm_run(), "Regression in Kit")
    async def test_capture_fsd_on(self) -> None:
        await self._test_capture(enable_fsd=True)
