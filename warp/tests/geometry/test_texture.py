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
def test_texture2d_sample_v4_kernel(
    tex_id: wp.uint64,
    uvs: wp.array(dtype=wp.vec2),
    results: wp.array(dtype=wp.vec4),
):
    tid = wp.tid()
    uv = uvs[tid]
    results[tid] = wp.texture2d_sample_v4(tex_id, uv)


@wp.kernel
def test_texture2d_sample_v3_kernel(
    tex_id: wp.uint64,
    uvs: wp.array(dtype=wp.vec2),
    results: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    uv = uvs[tid]
    results[tid] = wp.texture2d_sample_v3(tex_id, uv)


@wp.kernel
def test_texture2d_sample_f_kernel(
    tex_id: wp.uint64,
    uvs: wp.array(dtype=wp.vec2),
    results: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    uv = uvs[tid]
    results[tid] = wp.texture2d_sample_f(tex_id, uv)


@wp.kernel
def test_texture3d_sample_v4_kernel(
    tex_id: wp.uint64,
    uvws: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=wp.vec4),
):
    tid = wp.tid()
    uvw = uvws[tid]
    results[tid] = wp.texture3d_sample_v4(tex_id, uvw)


@wp.kernel
def test_texture3d_sample_f_kernel(
    tex_id: wp.uint64,
    uvws: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    uvw = uvws[tid]
    results[tid] = wp.texture3d_sample_f(tex_id, uvw)


def create_test_texture_2d_uint8(width, height, channels):
    data = np.zeros((height, width, channels), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                data[y, x, c] = ((x + y * width) * (c + 1)) % 256
    return data


def create_test_texture_2d_uint16(width, height, channels):
    data = np.zeros((height, width, channels), dtype=np.uint16)
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                data[y, x, c] = ((x + y * width) * (c + 1) * 256) % 65536
    return data


def create_test_texture_2d_float32(width, height, channels):
    data = np.zeros((height, width, channels), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                data[y, x, c] = float((x + y * width) * (c + 1)) / float(width * height)
    return data


def create_test_texture_3d_uint8(width, height, depth, channels):
    data = np.zeros((depth, height, width, channels), dtype=np.uint8)
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    data[z, y, x, c] = ((x + y * width + z * width * height) * (c + 1)) % 256
    return data


def create_test_texture_3d_float32(width, height, depth, channels):
    data = np.zeros((depth, height, width, channels), dtype=np.float32)
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    data[z, y, x, c] = float((x + y * width + z * width * height) * (c + 1)) / float(
                        width * height * depth
                    )
    return data


def test_texture2d_uint8_4ch(test, device):
    width, height, channels = 4, 4, 4
    tex_data = create_test_texture_2d_uint8(width, height, channels)

    data_arr = wp.array(tex_data, dtype=wp.uint8, device=device)
    texture = wp.Texture(data_arr, dtype=wp.uint8)

    uvs_np = np.array([[0.5, 0.5]], dtype=np.float32)
    uvs = wp.array(uvs_np, dtype=wp.vec2, device=device)
    results = wp.empty(len(uvs_np), dtype=wp.vec4, device=device)

    wp.launch(test_texture2d_sample_v4_kernel, dim=len(uvs_np), inputs=[texture.id, uvs, results], device=device)

    results_np = results.numpy()
    test.assertEqual(results_np.shape, (1, 4))
    test.assertTrue(np.sum(np.abs(results_np)) > 0)


def test_texture2d_uint16_4ch(test, device):
    width, height, channels = 4, 4, 4
    tex_data = create_test_texture_2d_uint16(width, height, channels)

    data_arr = wp.array(tex_data, dtype=wp.uint16, device=device)
    texture = wp.Texture(data_arr, dtype=wp.uint16)

    uvs_np = np.array([[0.5, 0.5]], dtype=np.float32)
    uvs = wp.array(uvs_np, dtype=wp.vec2, device=device)
    results = wp.empty(len(uvs_np), dtype=wp.vec4, device=device)

    wp.launch(test_texture2d_sample_v4_kernel, dim=len(uvs_np), inputs=[texture.id, uvs, results], device=device)

    results_np = results.numpy()
    test.assertEqual(results_np.shape, (1, 4))
    test.assertTrue(np.sum(np.abs(results_np)) > 0)


def test_texture2d_float32_4ch(test, device):
    width, height, channels = 4, 4, 4
    tex_data = create_test_texture_2d_float32(width, height, channels)

    data_arr = wp.array(tex_data, dtype=wp.float32, device=device)
    texture = wp.Texture(data_arr)

    uvs_np = np.array([[0.5, 0.5]], dtype=np.float32)
    uvs = wp.array(uvs_np, dtype=wp.vec2, device=device)
    results = wp.empty(len(uvs_np), dtype=wp.vec4, device=device)

    wp.launch(test_texture2d_sample_v4_kernel, dim=len(uvs_np), inputs=[texture.id, uvs, results], device=device)

    results_np = results.numpy()
    test.assertEqual(results_np.shape, (1, 4))
    test.assertTrue(np.sum(np.abs(results_np)) > 0)


def test_texture2d_1ch(test, device):
    width, height, channels = 4, 4, 1
    tex_data = create_test_texture_2d_float32(width, height, channels)

    data_arr = wp.array(tex_data, dtype=wp.float32, device=device)
    texture = wp.Texture(data_arr)

    uvs_np = np.array([[0.5, 0.5]], dtype=np.float32)
    uvs = wp.array(uvs_np, dtype=wp.vec2, device=device)
    results = wp.empty(len(uvs_np), dtype=wp.float32, device=device)

    wp.launch(test_texture2d_sample_f_kernel, dim=len(uvs_np), inputs=[texture.id, uvs, results], device=device)

    results_np = results.numpy()
    test.assertEqual(results_np.shape, (1,))
    test.assertTrue(np.sum(np.abs(results_np)) > 0)


def test_texture2d_2ch(test, device):
    width, height, channels = 4, 4, 2
    tex_data = create_test_texture_2d_float32(width, height, channels)

    data_arr = wp.array(tex_data, dtype=wp.float32, device=device)
    texture = wp.Texture(data_arr)

    uvs_np = np.array([[0.5, 0.5]], dtype=np.float32)
    uvs = wp.array(uvs_np, dtype=wp.vec2, device=device)
    results = wp.empty(len(uvs_np), dtype=wp.vec4, device=device)

    wp.launch(test_texture2d_sample_v4_kernel, dim=len(uvs_np), inputs=[texture.id, uvs, results], device=device)

    results_np = results.numpy()
    test.assertEqual(results_np.shape, (1, 4))


def test_texture2d_3ch(test, device):
    width, height, channels = 4, 4, 3
    tex_data = create_test_texture_2d_float32(width, height, channels)

    data_arr = wp.array(tex_data, dtype=wp.float32, device=device)
    texture = wp.Texture(data_arr)

    uvs_np = np.array([[0.5, 0.5]], dtype=np.float32)
    uvs = wp.array(uvs_np, dtype=wp.vec2, device=device)
    results = wp.empty(len(uvs_np), dtype=wp.vec3, device=device)

    wp.launch(test_texture2d_sample_v3_kernel, dim=len(uvs_np), inputs=[texture.id, uvs, results], device=device)

    results_np = results.numpy()
    test.assertEqual(results_np.shape, (1, 3))
    test.assertTrue(np.sum(np.abs(results_np)) > 0)


def test_texture3d_uint8_4ch(test, device):
    width, height, depth, channels = 4, 4, 4, 4
    tex_data = create_test_texture_3d_uint8(width, height, depth, channels)

    data_arr = wp.array(tex_data, dtype=wp.uint8, device=device)
    texture = wp.Texture(data_arr, dtype=wp.uint8)

    test.assertTrue(texture.is_3d)

    uvws_np = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    uvws = wp.array(uvws_np, dtype=wp.vec3, device=device)
    results = wp.empty(len(uvws_np), dtype=wp.vec4, device=device)

    wp.launch(test_texture3d_sample_v4_kernel, dim=len(uvws_np), inputs=[texture.id, uvws, results], device=device)

    results_np = results.numpy()
    test.assertEqual(results_np.shape, (1, 4))
    test.assertTrue(np.sum(np.abs(results_np)) > 0)


def test_texture3d_float32_4ch(test, device):
    width, height, depth, channels = 4, 4, 4, 4
    tex_data = create_test_texture_3d_float32(width, height, depth, channels)

    data_arr = wp.array(tex_data, dtype=wp.float32, device=device)
    texture = wp.Texture(data_arr)

    test.assertTrue(texture.is_3d)

    uvws_np = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    uvws = wp.array(uvws_np, dtype=wp.vec3, device=device)
    results = wp.empty(len(uvws_np), dtype=wp.vec4, device=device)

    wp.launch(test_texture3d_sample_v4_kernel, dim=len(uvws_np), inputs=[texture.id, uvws, results], device=device)

    results_np = results.numpy()
    test.assertEqual(results_np.shape, (1, 4))
    test.assertTrue(np.sum(np.abs(results_np)) > 0)


def test_texture3d_1ch(test, device):
    width, height, depth, channels = 4, 4, 4, 1
    tex_data = create_test_texture_3d_float32(width, height, depth, channels)

    data_arr = wp.array(tex_data, dtype=wp.float32, device=device)
    texture = wp.Texture(data_arr)

    test.assertTrue(texture.is_3d)

    uvws_np = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    uvws = wp.array(uvws_np, dtype=wp.vec3, device=device)
    results = wp.empty(len(uvws_np), dtype=wp.float32, device=device)

    wp.launch(test_texture3d_sample_f_kernel, dim=len(uvws_np), inputs=[texture.id, uvws, results], device=device)

    results_np = results.numpy()
    test.assertEqual(results_np.shape, (1,))
    test.assertTrue(np.sum(np.abs(results_np)) > 0)


def test_texture2d_corner_sampling(test, device):
    width, height, channels = 2, 2, 4
    tex_data = np.array(
        [
            [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )

    data_arr = wp.array(tex_data, dtype=wp.float32, device=device)
    texture = wp.Texture(data_arr)

    test.assertFalse(texture.is_3d)

    uvs_np = np.array(
        [
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.75, 0.75],
        ],
        dtype=np.float32,
    )
    uvs = wp.array(uvs_np, dtype=wp.vec2, device=device)
    results = wp.empty(len(uvs_np), dtype=wp.vec4, device=device)

    wp.launch(test_texture2d_sample_v4_kernel, dim=len(uvs_np), inputs=[texture.id, uvs, results], device=device)

    results_np = results.numpy()
    test.assertEqual(results_np.shape, (4, 4))


def test_texture2d_inferred_dimensions(test, device):
    width, height, channels = 8, 6, 4
    tex_data = np.random.Generator(height, width, channels).astype(np.float32)

    data_arr = wp.array(tex_data, dtype=wp.float32, device=device)
    texture = wp.Texture(data_arr)

    test.assertEqual(texture.width, width)
    test.assertEqual(texture.height, height)
    test.assertEqual(texture.channels, channels)
    test.assertFalse(texture.is_3d)


def test_texture3d_inferred_dimensions(test, device):
    width, height, depth, channels = 8, 6, 4, 2
    tex_data = np.random.Generator(depth, height, width, channels).astype(np.float32)

    data_arr = wp.array(tex_data, dtype=wp.float32, device=device)
    texture = wp.Texture(data_arr)

    test.assertEqual(texture.width, width)
    test.assertEqual(texture.height, height)
    test.assertEqual(texture.depth, depth)
    test.assertEqual(texture.channels, channels)
    test.assertTrue(texture.is_3d)


def test_texture_type_aliases(test, device):
    test.assertIs(wp.Texture, wp.Texture2D)
    test.assertIs(wp.Texture, wp.Texture3D)


class TestTexture(unittest.TestCase):
    def test_texture_new_del(self):
        instance = wp.Texture.__new__(wp.Texture)
        instance.__del__()


devices = get_test_devices()

add_function_test(TestTexture, "test_texture2d_uint8_4ch", test_texture2d_uint8_4ch, devices=devices)
add_function_test(TestTexture, "test_texture2d_uint16_4ch", test_texture2d_uint16_4ch, devices=devices)
add_function_test(TestTexture, "test_texture2d_float32_4ch", test_texture2d_float32_4ch, devices=devices)
add_function_test(TestTexture, "test_texture2d_1ch", test_texture2d_1ch, devices=devices)
add_function_test(TestTexture, "test_texture2d_2ch", test_texture2d_2ch, devices=devices)
add_function_test(TestTexture, "test_texture2d_3ch", test_texture2d_3ch, devices=devices)
add_function_test(TestTexture, "test_texture3d_uint8_4ch", test_texture3d_uint8_4ch, devices=devices)
add_function_test(TestTexture, "test_texture3d_float32_4ch", test_texture3d_float32_4ch, devices=devices)
add_function_test(TestTexture, "test_texture3d_1ch", test_texture3d_1ch, devices=devices)
add_function_test(TestTexture, "test_texture2d_corner_sampling", test_texture2d_corner_sampling, devices=devices)
add_function_test(
    TestTexture, "test_texture2d_inferred_dimensions", test_texture2d_inferred_dimensions, devices=devices
)
add_function_test(
    TestTexture, "test_texture3d_inferred_dimensions", test_texture3d_inferred_dimensions, devices=devices
)
add_function_test(TestTexture, "test_texture_type_aliases", test_texture_type_aliases, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
