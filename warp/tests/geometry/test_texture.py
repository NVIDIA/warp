# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def sample_texture2d_f_kernel(
    tex_id: wp.uint64,
    uvs: wp.array(dtype=wp.vec2),
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    out[tid] = wp.texture2d_sample_f(tex_id, uvs[tid])


def sample_point_clamp_1ch(data_1d: np.ndarray, width: int, height: int, uv: np.ndarray) -> float:
    # Match the typical normalized texture mapping:
    #   texel space coordinate: x = u*width - 0.5
    x = float(uv[0]) * width - 0.5
    y = float(uv[1]) * height - 0.5

    # Nearest: round to nearest integer in texel space (ties avoided by choosing UVs accordingly)
    xi = int(np.floor(x + 0.5))
    yi = int(np.floor(y + 0.5))

    xi = np.clip(xi, 0, width - 1)
    yi = np.clip(yi, 0, height - 1)

    return float(data_1d[yi * width + xi])


def sample_linear_clamp_1ch(data_1d: np.ndarray, width: int, height: int, uv: np.ndarray) -> float:
    x = float(uv[0]) * width - 0.5
    y = float(uv[1]) * height - 0.5

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    fx = x - float(x0)
    fy = y - float(y0)

    x1 = x0 + 1
    y1 = y0 + 1

    # clamp addressing
    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y1, 0, height - 1)

    v00 = float(data_1d[y0c * width + x0c])
    v10 = float(data_1d[y0c * width + x1c])
    v01 = float(data_1d[y1c * width + x0c])
    v11 = float(data_1d[y1c * width + x1c])

    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    return v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11


def test_texture_filtering(test, device):
    width, height, channels = 2, 2, 1
    tex_data_1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    data_arr = wp.array(tex_data_1d, dtype=wp.float32, device=device)

    tex_point = wp.Texture(
        data_arr,
        width=width,
        height=height,
        channels=channels,
        normalized_coords=True,
        address_mode=wp.Texture.ADDRESS_CLAMP,
        filter_mode=wp.Texture.FILTER_POINT,
    )
    tex_linear = wp.Texture(
        data_arr,
        width=width,
        height=height,
        channels=channels,
        normalized_coords=True,
        address_mode=wp.Texture.ADDRESS_CLAMP,
        filter_mode=wp.Texture.FILTER_LINEAR,
    )

    uv_between = np.array([0.49, 0.49], dtype=np.float32)

    uvs = wp.array([uv_between], dtype=wp.vec2, device=device)

    out_point = wp.empty(1, dtype=wp.float32, device=device)
    out_linear = wp.empty(1, dtype=wp.float32, device=device)

    wp.launch(
        sample_texture2d_f_kernel,
        dim=1,
        inputs=[tex_point.id, uvs, out_point],
        device=device,
    )

    wp.launch(
        sample_texture2d_f_kernel,
        dim=1,
        inputs=[tex_linear.id, uvs, out_linear],
        device=device,
    )

    got_point = out_point.numpy()
    got_linear = out_linear.numpy()

    test.assertNotAlmostEqual(got_point[0], got_linear[0], places=6)

    point_expected = sample_point_clamp_1ch(tex_data_1d, width, height, uv_between)
    linear_expected = sample_linear_clamp_1ch(tex_data_1d, width, height, uv_between)

    test.assertAlmostEqual(got_point[0], point_expected, places=6)
    test.assertAlmostEqual(got_linear[0], linear_expected, places=2)


devices = get_test_devices()


class TestTexture(unittest.TestCase):
    pass


add_function_test(TestTexture, "test_texture_filtering", test_texture_filtering, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
