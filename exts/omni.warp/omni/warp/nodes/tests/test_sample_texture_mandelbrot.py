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

"""Tests for the texture mandelbrot sample scene."""

import unittest

import omni.kit
import omni.warp

from ._common import (
    FrameRange,
    open_sample,
    validate_render,
)

TEST_ID = "texture_mandelbrot"


class TestSampleTextureMandelbrot(omni.kit.test.AsyncTestCase):
    @unittest.skipIf(omni.kit.test.utils.is_etm_run(), "Inconsistencies across ETM matrix")
    async def test_capture(self) -> None:
        await open_sample(f"{TEST_ID}.usda")

        with FrameRange(30) as frames:
            async for _ in frames:
                pass

        await validate_render(TEST_ID)
