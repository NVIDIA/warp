# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for 2D and 3D texture functionality on both CPU and CUDA devices."""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices

# ============================================================================
# 2D Texture Kernels
# ============================================================================


@wp.kernel
def sample_texture2d_f_at_centers(
    tex: wp.Texture2D,
    output: wp.array(dtype=float),
    width: int,
    height: int,
):
    """Sample a 1-channel 2D texture at texel centers."""
    tid = wp.tid()
    x = tid % width
    y = tid // width

    # Compute normalized coordinates at texel centers
    # For a texture of width W, texel i has center at (i + 0.5) / W
    u = (wp.float(x) + 0.5) / wp.float(width)
    v = (wp.float(y) + 0.5) / wp.float(height)

    output[tid] = wp.texture_sample(tex, wp.vec2f(u, v), dtype=float)


@wp.kernel
def sample_texture2d_v2_at_centers(
    tex: wp.Texture2D,
    output: wp.array(dtype=wp.vec2f),
    width: int,
    height: int,
):
    """Sample a 2-channel 2D texture at texel centers."""
    tid = wp.tid()
    x = tid % width
    y = tid // width

    u = (wp.float(x) + 0.5) / wp.float(width)
    v = (wp.float(y) + 0.5) / wp.float(height)

    output[tid] = wp.texture_sample(tex, wp.vec2f(u, v), dtype=wp.vec2f)


@wp.kernel
def sample_texture2d_v4_at_centers(
    tex: wp.Texture2D,
    output: wp.array(dtype=wp.vec4f),
    width: int,
    height: int,
):
    """Sample a 4-channel 2D texture at texel centers."""
    tid = wp.tid()
    x = tid % width
    y = tid // width

    u = (wp.float(x) + 0.5) / wp.float(width)
    v = (wp.float(y) + 0.5) / wp.float(height)

    output[tid] = wp.texture_sample(tex, wp.vec2f(u, v), dtype=wp.vec4f)


@wp.kernel
def test_texture2d_resolution(
    tex: wp.Texture2D,
    expected_width: int,
    expected_height: int,
):
    """Test resolution query using texture.width and texture.height."""
    w = tex.width
    h = tex.height

    wp.expect_eq(w, expected_width)
    wp.expect_eq(h, expected_height)


# ============================================================================
# 3D Texture Kernels
# ============================================================================


@wp.kernel
def sample_texture3d_f_at_centers(
    tex: wp.Texture3D,
    output: wp.array(dtype=float),
    width: int,
    height: int,
    depth: int,
):
    """Sample a 1-channel 3D texture at voxel centers."""
    tid = wp.tid()
    x = tid % width
    y = (tid // width) % height
    z = tid // (width * height)

    # Compute normalized coordinates at voxel centers
    u = (wp.float(x) + 0.5) / wp.float(width)
    v = (wp.float(y) + 0.5) / wp.float(height)
    ww = (wp.float(z) + 0.5) / wp.float(depth)

    output[tid] = wp.texture_sample(tex, wp.vec3f(u, v, ww), dtype=float)


@wp.kernel
def sample_texture3d_v2_at_centers(
    tex: wp.Texture3D,
    output: wp.array(dtype=wp.vec2f),
    width: int,
    height: int,
    depth: int,
):
    """Sample a 2-channel 3D texture at voxel centers."""
    tid = wp.tid()
    x = tid % width
    y = (tid // width) % height
    z = tid // (width * height)

    u = (wp.float(x) + 0.5) / wp.float(width)
    v = (wp.float(y) + 0.5) / wp.float(height)
    ww = (wp.float(z) + 0.5) / wp.float(depth)

    output[tid] = wp.texture_sample(tex, wp.vec3f(u, v, ww), dtype=wp.vec2f)


@wp.kernel
def sample_texture3d_v4_at_centers(
    tex: wp.Texture3D,
    output: wp.array(dtype=wp.vec4f),
    width: int,
    height: int,
    depth: int,
):
    """Sample a 4-channel 3D texture at voxel centers."""
    tid = wp.tid()
    x = tid % width
    y = (tid // width) % height
    z = tid // (width * height)

    u = (wp.float(x) + 0.5) / wp.float(width)
    v = (wp.float(y) + 0.5) / wp.float(height)
    ww = (wp.float(z) + 0.5) / wp.float(depth)

    output[tid] = wp.texture_sample(tex, wp.vec3f(u, v, ww), dtype=wp.vec4f)


@wp.kernel
def test_texture3d_resolution(
    tex: wp.Texture3D,
    expected_width: int,
    expected_height: int,
    expected_depth: int,
):
    """Test resolution query using texture.width, texture.height, texture.depth."""
    w = tex.width
    h = tex.height
    d = tex.depth

    wp.expect_eq(w, expected_width)
    wp.expect_eq(h, expected_height)
    wp.expect_eq(d, expected_depth)


# ============================================================================
# Texture Array Kernels
# ============================================================================


@wp.kernel
def sample_texture2d_array(
    textures: wp.array(dtype=wp.Texture2D),
    uv: wp.vec2f,
    output: wp.array(dtype=float),
):
    """Sample from an array of 2D textures, one texture per thread."""
    tid = wp.tid()
    tex = textures[tid]
    output[tid] = wp.texture_sample(tex, uv, dtype=float)


@wp.kernel
def sample_texture3d_array(
    textures: wp.array(dtype=wp.Texture3D),
    uvw: wp.vec3f,
    output: wp.array(dtype=float),
):
    """Sample from an array of 3D textures, one texture per thread."""
    tid = wp.tid()
    tex = textures[tid]
    output[tid] = wp.texture_sample(tex, uvw, dtype=float)


# ============================================================================
# Test Data Generation
# ============================================================================


def generate_sin_pattern_2d(width: int, height: int, num_channels: int) -> np.ndarray:
    """Generate a 2D sin pattern for testing.

    Creates a pattern based on: sin(2*pi*x/width) * sin(2*pi*y/height)
    Values are scaled to [0, 1] range.
    """
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # Create base sin pattern
    pattern = np.sin(2 * np.pi * xx / width) * np.sin(2 * np.pi * yy / height)
    # Scale to [0, 1]
    pattern = (pattern + 1.0) * 0.5

    if num_channels == 1:
        return pattern.astype(np.float32)
    else:
        # Create multi-channel pattern
        result = np.zeros((height, width, num_channels), dtype=np.float32)
        for c in range(num_channels):
            # Each channel has a slightly different phase
            phase = c * 0.25
            channel_pattern = np.sin(2 * np.pi * (xx / width + phase)) * np.sin(2 * np.pi * (yy / height + phase))
            result[:, :, c] = (channel_pattern + 1.0) * 0.5
        return result


def generate_sin_pattern_3d(width: int, height: int, depth: int, num_channels: int) -> np.ndarray:
    """Generate a 3D sin pattern for testing.

    Creates a pattern based on: sin(2*pi*x/width) * sin(2*pi*y/height) * sin(2*pi*z/depth)
    Values are scaled to [0, 1] range.
    """
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    z = np.arange(depth, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    # Transpose to get (depth, height, width) order
    xx = xx.transpose(2, 1, 0)
    yy = yy.transpose(2, 1, 0)
    zz = zz.transpose(2, 1, 0)

    # Create base sin pattern
    pattern = np.sin(2 * np.pi * xx / width) * np.sin(2 * np.pi * yy / height) * np.sin(2 * np.pi * zz / depth)
    # Scale to [0, 1]
    pattern = (pattern + 1.0) * 0.5

    if num_channels == 1:
        return pattern.astype(np.float32)
    else:
        # Create multi-channel pattern
        result = np.zeros((depth, height, width, num_channels), dtype=np.float32)
        for c in range(num_channels):
            phase = c * 0.25
            channel_pattern = (
                np.sin(2 * np.pi * (xx / width + phase))
                * np.sin(2 * np.pi * (yy / height + phase))
                * np.sin(2 * np.pi * (zz / depth + phase))
            )
            result[:, :, :, c] = (channel_pattern + 1.0) * 0.5
        return result


# ============================================================================
# Test Functions
# ============================================================================


def test_texture2d_1channel(test, device):
    """Test 2D texture with 1 channel, sampling at texel centers."""
    width, height = 32, 32
    num_channels = 1

    # Generate test data
    data = generate_sin_pattern_2d(width, height, num_channels)

    # Create texture
    tex = wp.Texture2D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Create output array
    output = wp.zeros(width * height, dtype=float, device=device)

    # Sample texture at texel centers
    wp.launch(
        sample_texture2d_f_at_centers,
        dim=width * height,
        inputs=[tex, output, width, height],
        device=device,
    )

    # Compare results
    expected = data.flatten()
    result = output.numpy()

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_texture2d_2channel(test, device):
    """Test 2D texture with 2 channels, sampling at texel centers."""
    width, height = 32, 32
    num_channels = 2

    # Generate test data
    data = generate_sin_pattern_2d(width, height, num_channels)

    tex = wp.Texture2D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Create output array
    output = wp.zeros(width * height, dtype=wp.vec2f, device=device)

    # Sample texture at texel centers
    wp.launch(
        sample_texture2d_v2_at_centers,
        dim=width * height,
        inputs=[tex, output, width, height],
        device=device,
    )

    # Compare results
    expected = data.reshape(-1, 2)
    result = output.numpy()

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_texture2d_4channel(test, device):
    """Test 2D texture with 4 channels, sampling at texel centers."""
    width, height = 32, 32
    num_channels = 4

    # Generate test data
    data = generate_sin_pattern_2d(width, height, num_channels)

    tex = wp.Texture2D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Create output array
    output = wp.zeros(width * height, dtype=wp.vec4f, device=device)

    # Sample texture at texel centers
    wp.launch(
        sample_texture2d_v4_at_centers,
        dim=width * height,
        inputs=[tex, output, width, height],
        device=device,
    )

    # Compare results
    expected = data.reshape(-1, 4)
    result = output.numpy()

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_texture2d_linear_filter(test, device):
    """Test 2D texture with linear filtering at texel centers.

    At texel centers, linear filtering should give the same result as nearest.
    """
    width, height = 16, 16
    num_channels = 1

    # Generate test data
    data = generate_sin_pattern_2d(width, height, num_channels)

    tex = wp.Texture2D(
        data,
        filter_mode=wp.TextureFilterMode.LINEAR,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Create output array
    output = wp.zeros(width * height, dtype=float, device=device)

    # Sample texture at texel centers
    wp.launch(
        sample_texture2d_f_at_centers,
        dim=width * height,
        inputs=[tex, output, width, height],
        device=device,
    )

    # At texel centers, linear filtering should give exact values
    expected = data.flatten()
    result = output.numpy()

    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


def test_texture2d_resolution_query(test, device):
    """Test resolution query functions for 2D texture."""
    width, height = 64, 128

    data = np.zeros((height, width, 4), dtype=np.float32)

    tex = wp.Texture2D(data, device=device)

    # Test resolution queries in kernel
    wp.launch(
        test_texture2d_resolution,
        dim=1,
        inputs=[tex, width, height],
        device=device,
    )


def test_texture3d_1channel(test, device):
    """Test 3D texture with 1 channel, sampling at voxel centers."""
    width, height, depth = 16, 16, 16
    num_channels = 1

    # Generate test data
    data = generate_sin_pattern_3d(width, height, depth, num_channels)

    tex = wp.Texture3D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Create output array
    output = wp.zeros(width * height * depth, dtype=float, device=device)

    # Sample texture at voxel centers
    wp.launch(
        sample_texture3d_f_at_centers,
        dim=width * height * depth,
        inputs=[tex, output, width, height, depth],
        device=device,
    )

    # Compare results
    expected = data.flatten()
    result = output.numpy()

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_texture3d_2channel(test, device):
    """Test 3D texture with 2 channels, sampling at voxel centers."""
    width, height, depth = 8, 8, 8
    num_channels = 2

    # Generate test data
    data = generate_sin_pattern_3d(width, height, depth, num_channels)

    tex = wp.Texture3D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Create output array
    output = wp.zeros(width * height * depth, dtype=wp.vec2f, device=device)

    # Sample texture at voxel centers
    wp.launch(
        sample_texture3d_v2_at_centers,
        dim=width * height * depth,
        inputs=[tex, output, width, height, depth],
        device=device,
    )

    # Compare results
    expected = data.reshape(-1, 2)
    result = output.numpy()

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_texture3d_4channel(test, device):
    """Test 3D texture with 4 channels, sampling at voxel centers."""
    width, height, depth = 8, 8, 8
    num_channels = 4

    # Generate test data
    data = generate_sin_pattern_3d(width, height, depth, num_channels)

    tex = wp.Texture3D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Create output array
    output = wp.zeros(width * height * depth, dtype=wp.vec4f, device=device)

    # Sample texture at voxel centers
    wp.launch(
        sample_texture3d_v4_at_centers,
        dim=width * height * depth,
        inputs=[tex, output, width, height, depth],
        device=device,
    )

    # Compare results
    expected = data.reshape(-1, 4)
    result = output.numpy()

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_texture3d_linear_filter(test, device):
    """Test 3D texture with linear filtering at voxel centers."""
    width, height, depth = 8, 8, 8
    num_channels = 1

    # Generate test data
    data = generate_sin_pattern_3d(width, height, depth, num_channels)

    tex = wp.Texture3D(
        data,
        filter_mode=wp.TextureFilterMode.LINEAR,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Create output array
    output = wp.zeros(width * height * depth, dtype=float, device=device)

    # Sample texture at voxel centers
    wp.launch(
        sample_texture3d_f_at_centers,
        dim=width * height * depth,
        inputs=[tex, output, width, height, depth],
        device=device,
    )

    # At voxel centers, linear filtering should give exact values
    expected = data.flatten()
    result = output.numpy()

    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


def test_texture3d_resolution_query(test, device):
    """Test resolution query functions for 3D texture."""
    width, height, depth = 32, 64, 16

    data = np.zeros((depth, height, width), dtype=np.float32)

    tex = wp.Texture3D(data, device=device)

    # Test resolution queries in kernel
    wp.launch(
        test_texture3d_resolution,
        dim=1,
        inputs=[tex, width, height, depth],
        device=device,
    )


def test_texture2d_new_del(test, device):
    """Test proper handling of uninitialized texture (created with __new__ but not __init__)."""
    instance = wp.Texture2D.__new__(wp.Texture2D)
    instance.__del__()


def test_texture3d_new_del(test, device):
    """Test proper handling of uninitialized texture (created with __new__ but not __init__)."""
    instance = wp.Texture3D.__new__(wp.Texture3D)
    instance.__del__()


# ============================================================================
# Interpolation Tests - Kernels
# ============================================================================


@wp.kernel
def sample_texture2d_at_uv(
    tex: wp.Texture2D,
    uvs: wp.array(dtype=wp.vec2f),
    output: wp.array(dtype=float),
):
    """Sample a 2D texture at specified UV coordinates."""
    tid = wp.tid()
    uv = uvs[tid]
    output[tid] = wp.texture_sample(tex, uv, dtype=float)


@wp.kernel
def sample_texture3d_at_uvw(
    tex: wp.Texture3D,
    uvws: wp.array(dtype=wp.vec3f),
    output: wp.array(dtype=float),
):
    """Sample a 3D texture at specified UVW coordinates."""
    tid = wp.tid()
    uvw = uvws[tid]
    output[tid] = wp.texture_sample(tex, uvw, dtype=float)


# ============================================================================
# Interpolation Tests - Functions
# ============================================================================


def test_texture2d_nearest_interpolation(test, device):
    """Test that NEAREST filtering returns the nearest texel value when sampling between texels."""
    # Create a simple 2x2 texture with distinct values:
    # [0, 1]
    # [2, 3]
    width, height = 2, 2
    data = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    tex = wp.Texture2D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Sample at texel centers - should return exact values
    # Texel centers for 2x2 texture: (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)
    uvs_np = np.array(
        [
            [0.25, 0.25],  # texel (0,0) -> 0.0
            [0.75, 0.25],  # texel (1,0) -> 1.0
            [0.25, 0.75],  # texel (0,1) -> 2.0
            [0.75, 0.75],  # texel (1,1) -> 3.0
        ],
        dtype=np.float32,
    )

    uvs = wp.array(uvs_np, dtype=wp.vec2f, device=device)
    output = wp.zeros(4, dtype=float, device=device)

    wp.launch(
        sample_texture2d_at_uv,
        dim=4,
        inputs=[tex, uvs, output],
        device=device,
    )

    result = output.numpy()
    expected = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    # Sample between texels - NEAREST should snap to one of the neighbors
    # Sample at center (0.5, 0.5) - with NEAREST, result should be one of 0, 1, 2, or 3
    uvs_between_np = np.array([[0.5, 0.5]], dtype=np.float32)
    uvs_between = wp.array(uvs_between_np, dtype=wp.vec2f, device=device)
    output_between = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        sample_texture2d_at_uv,
        dim=1,
        inputs=[tex, uvs_between, output_between],
        device=device,
    )

    result_between = output_between.numpy()[0]
    # NEAREST should return one of the texel values, not an interpolated value
    test.assertIn(
        result_between,
        [0.0, 1.0, 2.0, 3.0],
        f"NEAREST filtering returned {result_between}, expected one of [0, 1, 2, 3]",
    )


def test_texture2d_linear_interpolation(test, device):
    """Test that LINEAR filtering correctly interpolates between texels."""
    # Create a simple 2x2 texture with distinct values:
    # [0, 1]
    # [2, 3]
    width, height = 2, 2
    data = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    tex = wp.Texture2D(
        data,
        filter_mode=wp.TextureFilterMode.LINEAR,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Sample at the exact center (0.5, 0.5) - should be average of all 4 texels
    # With bilinear interpolation: (0 + 1 + 2 + 3) / 4 = 1.5
    uvs_np = np.array([[0.5, 0.5]], dtype=np.float32)
    uvs = wp.array(uvs_np, dtype=wp.vec2f, device=device)
    output = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        sample_texture2d_at_uv,
        dim=1,
        inputs=[tex, uvs, output],
        device=device,
    )

    result = output.numpy()[0]
    expected = 1.5  # average of 0, 1, 2, 3
    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"LINEAR interpolation at center: expected {expected}, got {result}",
    )

    # Test interpolation along X axis (halfway between texels 0 and 1)
    # At UV (0.5, 0.25): interpolate between texel (0,0)=0 and texel (1,0)=1
    # Expected: 0.5
    uvs_x_np = np.array([[0.5, 0.25]], dtype=np.float32)
    uvs_x = wp.array(uvs_x_np, dtype=wp.vec2f, device=device)
    output_x = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        sample_texture2d_at_uv,
        dim=1,
        inputs=[tex, uvs_x, output_x],
        device=device,
    )

    result_x = output_x.numpy()[0]
    expected_x = 0.5
    np.testing.assert_allclose(
        result_x,
        expected_x,
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"LINEAR interpolation along X: expected {expected_x}, got {result_x}",
    )

    # Test interpolation along Y axis (halfway between texels 0 and 2)
    # At UV (0.25, 0.5): interpolate between texel (0,0)=0 and texel (0,1)=2
    # Expected: 1.0
    uvs_y_np = np.array([[0.25, 0.5]], dtype=np.float32)
    uvs_y = wp.array(uvs_y_np, dtype=wp.vec2f, device=device)
    output_y = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        sample_texture2d_at_uv,
        dim=1,
        inputs=[tex, uvs_y, output_y],
        device=device,
    )

    result_y = output_y.numpy()[0]
    expected_y = 1.0
    np.testing.assert_allclose(
        result_y,
        expected_y,
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"LINEAR interpolation along Y: expected {expected_y}, got {result_y}",
    )


def test_texture3d_nearest_interpolation(test, device):
    """Test that NEAREST filtering returns the nearest voxel value in 3D."""
    # Create a 2x2x2 texture with values 0-7
    width, height, depth = 2, 2, 2
    data = np.arange(8, dtype=np.float32).reshape((2, 2, 2))
    # data[z, y, x] layout:
    # z=0: [[0, 1], [2, 3]]
    # z=1: [[4, 5], [6, 7]]

    tex = wp.Texture3D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Sample at voxel centers
    uvws_np = np.array(
        [
            [0.25, 0.25, 0.25],  # voxel (0,0,0) -> 0
            [0.75, 0.25, 0.25],  # voxel (1,0,0) -> 1
            [0.25, 0.75, 0.25],  # voxel (0,1,0) -> 2
            [0.75, 0.75, 0.25],  # voxel (1,1,0) -> 3
            [0.25, 0.25, 0.75],  # voxel (0,0,1) -> 4
            [0.75, 0.25, 0.75],  # voxel (1,0,1) -> 5
            [0.25, 0.75, 0.75],  # voxel (0,1,1) -> 6
            [0.75, 0.75, 0.75],  # voxel (1,1,1) -> 7
        ],
        dtype=np.float32,
    )

    uvws = wp.array(uvws_np, dtype=wp.vec3f, device=device)
    output = wp.zeros(8, dtype=float, device=device)

    wp.launch(
        sample_texture3d_at_uvw,
        dim=8,
        inputs=[tex, uvws, output],
        device=device,
    )

    result = output.numpy()
    expected = np.arange(8, dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_texture3d_linear_interpolation(test, device):
    """Test that LINEAR filtering correctly interpolates in 3D."""
    # Create a 2x2x2 texture with values 0-7
    width, height, depth = 2, 2, 2
    data = np.arange(8, dtype=np.float32).reshape((2, 2, 2))

    tex = wp.Texture3D(
        data,
        filter_mode=wp.TextureFilterMode.LINEAR,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Sample at the center (0.5, 0.5, 0.5) - should be average of all 8 voxels
    # (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7) / 8 = 3.5
    uvws_np = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    uvws = wp.array(uvws_np, dtype=wp.vec3f, device=device)
    output = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        sample_texture3d_at_uvw,
        dim=1,
        inputs=[tex, uvws, output],
        device=device,
    )

    result = output.numpy()[0]
    expected = 3.5  # average of 0-7
    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"3D LINEAR interpolation at center: expected {expected}, got {result}",
    )

    # Test interpolation along Z axis only (at x=0.25, y=0.25)
    # Interpolate between voxel (0,0,0)=0 and voxel (0,0,1)=4
    # At z=0.5: expected = 2.0
    uvws_z_np = np.array([[0.25, 0.25, 0.5]], dtype=np.float32)
    uvws_z = wp.array(uvws_z_np, dtype=wp.vec3f, device=device)
    output_z = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        sample_texture3d_at_uvw,
        dim=1,
        inputs=[tex, uvws_z, output_z],
        device=device,
    )

    result_z = output_z.numpy()[0]
    expected_z = 2.0
    np.testing.assert_allclose(
        result_z,
        expected_z,
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"3D LINEAR interpolation along Z: expected {expected_z}, got {result_z}",
    )


# ============================================================================
# Compressed Texture Tests (uint8, uint16)
# ============================================================================


def test_texture2d_uint8(test, device):
    """Test 2D texture with uint8 data, which should be read as normalized floats [0, 1]."""
    width, height = 4, 4

    # Create uint8 data with values 0, 128, 255
    data = np.array(
        [
            [0, 64, 128, 192],
            [32, 96, 160, 224],
            [16, 80, 144, 208],
            [48, 112, 176, 240],
        ],
        dtype=np.uint8,
    )

    tex = wp.Texture2D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    test.assertEqual(tex.dtype, np.uint8)

    # Sample at texel centers
    uvs_np = np.array(
        [
            [0.125, 0.125],  # texel (0,0) -> 0/255 = 0.0
            [0.375, 0.125],  # texel (1,0) -> 64/255 ≈ 0.251
            [0.625, 0.125],  # texel (2,0) -> 128/255 ≈ 0.502
            [0.875, 0.125],  # texel (3,0) -> 192/255 ≈ 0.753
        ],
        dtype=np.float32,
    )

    uvs = wp.array(uvs_np, dtype=wp.vec2f, device=device)
    output = wp.zeros(4, dtype=float, device=device)

    wp.launch(
        sample_texture2d_at_uv,
        dim=4,
        inputs=[tex, uvs, output],
        device=device,
    )

    result = output.numpy()
    expected = np.array([0.0, 64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0])
    np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)


def test_texture2d_uint16(test, device):
    """Test 2D texture with uint16 data, which should be read as normalized floats [0, 1]."""
    width, height = 2, 2

    # Create uint16 data
    data = np.array(
        [
            [0, 32768],
            [16384, 65535],
        ],
        dtype=np.uint16,
    )

    tex = wp.Texture2D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    test.assertEqual(tex.dtype, np.uint16)

    # Sample at texel centers
    uvs_np = np.array(
        [
            [0.25, 0.25],  # texel (0,0) -> 0/65535 = 0.0
            [0.75, 0.25],  # texel (1,0) -> 32768/65535 ≈ 0.5
            [0.25, 0.75],  # texel (0,1) -> 16384/65535 ≈ 0.25
            [0.75, 0.75],  # texel (1,1) -> 65535/65535 = 1.0
        ],
        dtype=np.float32,
    )

    uvs = wp.array(uvs_np, dtype=wp.vec2f, device=device)
    output = wp.zeros(4, dtype=float, device=device)

    wp.launch(
        sample_texture2d_at_uv,
        dim=4,
        inputs=[tex, uvs, output],
        device=device,
    )

    result = output.numpy()
    expected = np.array([0.0, 32768.0 / 65535.0, 16384.0 / 65535.0, 1.0])
    np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)


def test_texture3d_uint8(test, device):
    """Test 3D texture with uint8 data."""
    width, height, depth = 2, 2, 2

    # Create uint8 data with values scaling from 0 to 255
    data = np.array(
        [
            [[0, 36], [73, 109]],
            [[146, 182], [219, 255]],
        ],
        dtype=np.uint8,
    )

    tex = wp.Texture3D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    test.assertEqual(tex.dtype, np.uint8)

    # Sample at voxel centers
    uvws_np = np.array(
        [
            [0.25, 0.25, 0.25],  # voxel (0,0,0) -> 0/255 = 0.0
            [0.75, 0.75, 0.75],  # voxel (1,1,1) -> 255/255 = 1.0
        ],
        dtype=np.float32,
    )

    uvws = wp.array(uvws_np, dtype=wp.vec3f, device=device)
    output = wp.zeros(2, dtype=float, device=device)

    wp.launch(
        sample_texture3d_at_uvw,
        dim=2,
        inputs=[tex, uvws, output],
        device=device,
    )

    result = output.numpy()
    expected = np.array([0.0, 1.0])
    np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)


def test_texture2d_uint8_linear_interpolation(test, device):
    """Test that LINEAR filtering works correctly with uint8 textures."""
    width, height = 2, 2

    # Create uint8 data: values 0, 128, 128, 255
    # At center with linear interpolation: (0 + 128 + 128 + 255) / 4 / 255 ≈ 0.5
    data = np.array(
        [
            [0, 128],
            [128, 255],
        ],
        dtype=np.uint8,
    )

    tex = wp.Texture2D(
        data,
        filter_mode=wp.TextureFilterMode.LINEAR,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Sample at center - should interpolate
    uvs_np = np.array([[0.5, 0.5]], dtype=np.float32)
    uvs = wp.array(uvs_np, dtype=wp.vec2f, device=device)
    output = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        sample_texture2d_at_uv,
        dim=1,
        inputs=[tex, uvs, output],
        device=device,
    )

    result = output.numpy()[0]
    expected = (0.0 + 128.0 / 255.0 + 128.0 / 255.0 + 1.0) / 4.0
    np.testing.assert_allclose(result, expected, rtol=0.05, atol=0.05)


def test_texture3d_uint16(test, device):
    """Test 3D texture with uint16 data."""
    width, height, depth = 2, 2, 2

    # Create uint16 data with values scaling from 0 to 65535
    data = np.array(
        [
            [[0, 9362], [18725, 28087]],
            [[37449, 46811], [56174, 65535]],
        ],
        dtype=np.uint16,
    )

    tex = wp.Texture3D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    test.assertEqual(tex.dtype, np.uint16)

    # Sample at voxel centers
    uvws_np = np.array(
        [
            [0.25, 0.25, 0.25],  # voxel (0,0,0) -> 0/65535 = 0.0
            [0.75, 0.75, 0.75],  # voxel (1,1,1) -> 65535/65535 = 1.0
        ],
        dtype=np.float32,
    )

    uvws = wp.array(uvws_np, dtype=wp.vec3f, device=device)
    output = wp.zeros(2, dtype=float, device=device)

    wp.launch(
        sample_texture3d_at_uvw,
        dim=2,
        inputs=[tex, uvws, output],
        device=device,
    )

    result = output.numpy()
    expected = np.array([0.0, 1.0])
    np.testing.assert_allclose(result, expected, rtol=1e-2, atol=1e-2)


def test_texture3d_uint8_linear_interpolation(test, device):
    """Test that LINEAR filtering works correctly with uint8 3D textures."""
    width, height, depth = 2, 2, 2

    # Create uint8 data: corners 0 and 255, others in between
    data = np.array(
        [
            [[0, 36], [73, 109]],
            [[146, 182], [219, 255]],
        ],
        dtype=np.uint8,
    )

    tex = wp.Texture3D(
        data,
        filter_mode=wp.TextureFilterMode.LINEAR,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Sample at center - should interpolate all 8 voxels
    uvws_np = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    uvws = wp.array(uvws_np, dtype=wp.vec3f, device=device)
    output = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        sample_texture3d_at_uvw,
        dim=1,
        inputs=[tex, uvws, output],
        device=device,
    )

    result = output.numpy()[0]
    # Expected is average of all normalized values
    expected = np.mean(data.astype(np.float32).flatten() / 255.0)
    np.testing.assert_allclose(result, expected, rtol=0.05, atol=0.05)


def test_texture2d_uint16_linear_interpolation(test, device):
    """Test that LINEAR filtering works correctly with uint16 textures."""
    width, height = 2, 2

    # Create uint16 data
    data = np.array(
        [
            [0, 32768],
            [32768, 65535],
        ],
        dtype=np.uint16,
    )

    tex = wp.Texture2D(
        data,
        filter_mode=wp.TextureFilterMode.LINEAR,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Sample at center - should interpolate
    uvs_np = np.array([[0.5, 0.5]], dtype=np.float32)
    uvs = wp.array(uvs_np, dtype=wp.vec2f, device=device)
    output = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        sample_texture2d_at_uv,
        dim=1,
        inputs=[tex, uvs, output],
        device=device,
    )

    result = output.numpy()[0]
    # Expected is average of all normalized values
    expected = np.mean(data.astype(np.float32).flatten() / 65535.0)
    np.testing.assert_allclose(result, expected, rtol=0.05, atol=0.05)


def test_texture3d_uint16_linear_interpolation(test, device):
    """Test that LINEAR filtering works correctly with uint16 3D textures."""
    width, height, depth = 2, 2, 2

    # Create uint16 data
    data = np.array(
        [
            [[0, 9362], [18725, 28087]],
            [[37449, 46811], [56174, 65535]],
        ],
        dtype=np.uint16,
    )

    tex = wp.Texture3D(
        data,
        filter_mode=wp.TextureFilterMode.LINEAR,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Sample at center - should interpolate all 8 voxels
    uvws_np = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    uvws = wp.array(uvws_np, dtype=wp.vec3f, device=device)
    output = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        sample_texture3d_at_uvw,
        dim=1,
        inputs=[tex, uvws, output],
        device=device,
    )

    result = output.numpy()[0]
    # Expected is average of all normalized values
    expected = np.mean(data.astype(np.float32).flatten() / 65535.0)
    np.testing.assert_allclose(result, expected, rtol=0.05, atol=0.05)


# ============================================================================
# Per-Axis Address Mode Tests
# ============================================================================


@wp.kernel
def sample_texture2d_outside_bounds(
    tex: wp.Texture2D,
    uvs: wp.array(dtype=wp.vec2f),
    output: wp.array(dtype=float),
):
    """Sample a 2D texture at specified UV coordinates (may be outside [0,1])."""
    tid = wp.tid()
    uv = uvs[tid]
    output[tid] = wp.texture_sample(tex, uv, dtype=float)


@wp.kernel
def sample_texture3d_outside_bounds(
    tex: wp.Texture3D,
    uvws: wp.array(dtype=wp.vec3f),
    output: wp.array(dtype=float),
):
    """Sample a 3D texture at specified UVW coordinates (may be outside [0,1])."""
    tid = wp.tid()
    uvw = uvws[tid]
    output[tid] = wp.texture_sample(tex, uvw, dtype=float)


def test_texture2d_per_axis_address_modes(test, device):
    """Test 2D texture with different address modes per axis.

    Creates a 2x2 texture and tests WRAP on U, CLAMP on V.
    """
    width, height = 2, 2
    # Create texture with distinct values at each corner:
    # [0, 1]
    # [2, 3]
    data = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    # WRAP on U (horizontal), CLAMP on V (vertical)
    tex = wp.Texture2D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode_u=wp.TextureAddressMode.WRAP,
        address_mode_v=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Verify properties are set correctly
    test.assertEqual(tex.address_mode_u, wp.TextureAddressMode.WRAP)
    test.assertEqual(tex.address_mode_v, wp.TextureAddressMode.CLAMP)
    test.assertTrue(tex.normalized_coords)

    # Test sampling at U=1.25 (should wrap to 0.25 -> texel 0),
    # V=0.25 (in bounds -> texel 0)
    # Expected: texel (0, 0) = 0.0
    uvs_wrap = np.array([[1.25, 0.25]], dtype=np.float32)
    uvs = wp.array(uvs_wrap, dtype=wp.vec2f, device=device)
    output = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        sample_texture2d_outside_bounds,
        dim=1,
        inputs=[tex, uvs, output],
        device=device,
    )

    result = output.numpy()[0]
    # With WRAP on U, u=1.25 wraps to u=0.25, which is texel 0
    # With CLAMP on V, v=0.25 is in texel 0
    test.assertAlmostEqual(result, 0.0, places=4)

    # Test sampling at U=0.25 (in bounds -> texel 0),
    # V=1.5 (clamped to 1.0 -> texel 1)
    # Expected: texel (0, 1) = 2.0
    uvs_clamp = np.array([[0.25, 1.5]], dtype=np.float32)
    uvs2 = wp.array(uvs_clamp, dtype=wp.vec2f, device=device)
    output2 = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        sample_texture2d_outside_bounds,
        dim=1,
        inputs=[tex, uvs2, output2],
        device=device,
    )

    result2 = output2.numpy()[0]
    # With CLAMP on V, v=1.5 is clamped, so we get texel (0, 1) = 2.0
    test.assertAlmostEqual(result2, 2.0, places=4)


def test_texture2d_address_mode_tuple(test, device):
    """Test 2D texture with address_mode as a tuple (u, v)."""
    width, height = 2, 2
    data = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    # Use tuple syntax for address modes
    tex = wp.Texture2D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=(wp.TextureAddressMode.CLAMP, wp.TextureAddressMode.WRAP),
        device=device,
    )

    # Verify properties
    test.assertEqual(tex.address_mode_u, wp.TextureAddressMode.CLAMP)
    test.assertEqual(tex.address_mode_v, wp.TextureAddressMode.WRAP)

    # Test sampling with V wrapping: V=1.25 should wrap to 0.25 -> texel 0
    uvs_np = np.array([[0.25, 1.25]], dtype=np.float32)
    uvs = wp.array(uvs_np, dtype=wp.vec2f, device=device)
    output = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        sample_texture2d_outside_bounds,
        dim=1,
        inputs=[tex, uvs, output],
        device=device,
    )

    result = output.numpy()[0]
    # V wraps from 1.25 to 0.25 (texel 0), U=0.25 is texel 0
    # So we get texel (0, 0) = 0.0
    test.assertAlmostEqual(result, 0.0, places=4)


def test_texture3d_per_axis_address_modes(test, device):
    """Test 3D texture with different address modes per axis."""
    width, height, depth = 2, 2, 2
    # Create 2x2x2 texture with values 0-7
    data = np.arange(8, dtype=np.float32).reshape((2, 2, 2))

    # Different mode for each axis
    tex = wp.Texture3D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode_u=wp.TextureAddressMode.WRAP,
        address_mode_v=wp.TextureAddressMode.CLAMP,
        address_mode_w=wp.TextureAddressMode.WRAP,
        device=device,
    )

    # Verify properties
    test.assertEqual(tex.address_mode_u, wp.TextureAddressMode.WRAP)
    test.assertEqual(tex.address_mode_v, wp.TextureAddressMode.CLAMP)
    test.assertEqual(tex.address_mode_w, wp.TextureAddressMode.WRAP)
    test.assertTrue(tex.normalized_coords)

    # Sample at voxel center (0,0,0)
    uvws_np = np.array([[0.25, 0.25, 0.25]], dtype=np.float32)
    uvws = wp.array(uvws_np, dtype=wp.vec3f, device=device)
    output = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        sample_texture3d_outside_bounds,
        dim=1,
        inputs=[tex, uvws, output],
        device=device,
    )

    result = output.numpy()[0]
    test.assertAlmostEqual(result, 0.0, places=4)


def test_texture3d_address_mode_tuple(test, device):
    """Test 3D texture with address_mode as a tuple (u, v, w)."""
    width, height, depth = 2, 2, 2
    data = np.arange(8, dtype=np.float32).reshape((2, 2, 2))

    # Use tuple syntax
    tex = wp.Texture3D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=(
            wp.TextureAddressMode.WRAP,
            wp.TextureAddressMode.CLAMP,
            wp.TextureAddressMode.MIRROR,
        ),
        device=device,
    )

    # Verify properties
    test.assertEqual(tex.address_mode_u, wp.TextureAddressMode.WRAP)
    test.assertEqual(tex.address_mode_v, wp.TextureAddressMode.CLAMP)
    test.assertEqual(tex.address_mode_w, wp.TextureAddressMode.MIRROR)


def test_texture2d_wrap_linear_edge(test, device):
    """Test WRAP mode with LINEAR filtering at texture edge.

    This tests that bilinear interpolation correctly wraps neighbor indices
    at the texture boundary. At u=0.9 on a 4-wide texture with WRAP mode,
    the neighbors should include texel 0 (wrapped), not just clamped to texel 3.
    """
    width, height = 4, 4
    # Create texture where each column has a distinct value
    # Column 0: 0, Column 1: 1, Column 2: 2, Column 3: 3
    data = np.zeros((4, 4), dtype=np.float32)
    for x in range(4):
        data[:, x] = float(x)

    tex = wp.Texture2D(
        data,
        filter_mode=wp.TextureFilterMode.LINEAR,
        address_mode=wp.TextureAddressMode.WRAP,
        device=device,
    )

    # Sample at u=0.9375 (texel center of x=3), v=0.5
    # At texel center, should get exact value = 3.0
    uvs_center = np.array([[0.875, 0.5]], dtype=np.float32)
    uvs = wp.array(uvs_center, dtype=wp.vec2f, device=device)
    output = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        sample_texture2d_outside_bounds,
        dim=1,
        inputs=[tex, uvs, output],
        device=device,
    )

    result_center = output.numpy()[0]
    test.assertAlmostEqual(result_center, 3.0, places=3)

    # Sample at u=0.96875 (between texel 3 and wrapped texel 0)
    # With WRAP: should interpolate between value 3 and value 0
    # With CLAMP (bug): would interpolate between value 3 and value 3
    uvs_edge = np.array([[0.96875, 0.5]], dtype=np.float32)
    uvs2 = wp.array(uvs_edge, dtype=wp.vec2f, device=device)
    output2 = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        sample_texture2d_outside_bounds,
        dim=1,
        inputs=[tex, uvs2, output2],
        device=device,
    )

    result_edge = output2.numpy()[0]
    # With correct WRAP: result should be < 3.0 (interpolating toward 0)
    # With incorrect CLAMP: result would be exactly 3.0
    test.assertLess(result_edge, 2.9, f"WRAP mode not working correctly at edge: got {result_edge}")


def test_texture2d_mirror_linear_edge(test, device):
    """Test MIRROR mode with LINEAR filtering at texture edge.

    At the edge with MIRROR mode, neighbors should mirror back into the texture.
    """
    width, height = 4, 4
    # Create texture where each column has a distinct value
    data = np.zeros((4, 4), dtype=np.float32)
    for x in range(4):
        data[:, x] = float(x)

    tex = wp.Texture2D(
        data,
        filter_mode=wp.TextureFilterMode.LINEAR,
        address_mode=wp.TextureAddressMode.MIRROR,
        device=device,
    )

    # Sample at u=0.96875 (between texel 3 and mirrored texel 3 or 2)
    # With MIRROR: at edge, should mirror back so neighbor is texel 2
    uvs_edge = np.array([[0.96875, 0.5]], dtype=np.float32)
    uvs = wp.array(uvs_edge, dtype=wp.vec2f, device=device)
    output = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        sample_texture2d_outside_bounds,
        dim=1,
        inputs=[tex, uvs, output],
        device=device,
    )

    result_edge = output.numpy()[0]
    # With correct MIRROR: result should be close to 3.0 but may interpolate with mirrored neighbor
    # The key is it should match CUDA behavior
    test.assertGreater(result_edge, 2.0, f"MIRROR mode result unexpected: got {result_edge}")


# ============================================================================
# Non-Normalized Coordinates Tests
# ============================================================================


@wp.kernel
def sample_texture2d_texel_coords(
    tex: wp.Texture2D,
    coords: wp.array(dtype=wp.vec2f),
    output: wp.array(dtype=float),
):
    """Sample a 2D texture using texel-space coordinates."""
    tid = wp.tid()
    coord = coords[tid]
    output[tid] = wp.texture_sample(tex, coord, dtype=float)


@wp.kernel
def sample_texture3d_texel_coords(
    tex: wp.Texture3D,
    coords: wp.array(dtype=wp.vec3f),
    output: wp.array(dtype=float),
):
    """Sample a 3D texture using texel-space coordinates."""
    tid = wp.tid()
    coord = coords[tid]
    output[tid] = wp.texture_sample(tex, coord, dtype=float)


def test_texture2d_non_normalized_coords(test, device):
    """Test 2D texture with non-normalized (texel-space) coordinates.

    With normalized_coords=False, coordinates are in [0, width] x [0, height]
    instead of [0, 1] x [0, 1].
    """
    width, height = 4, 4
    data = np.arange(16, dtype=np.float32).reshape((4, 4))

    tex = wp.Texture2D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        normalized_coords=False,
        device=device,
    )

    # Verify property
    test.assertFalse(tex.normalized_coords)

    # Sample at texel center (1.5, 1.5) in texel space
    # This corresponds to texel (1, 1) = data[1, 1] = 5.0
    coords_np = np.array([[1.5, 1.5]], dtype=np.float32)
    coords = wp.array(coords_np, dtype=wp.vec2f, device=device)
    output = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        sample_texture2d_texel_coords,
        dim=1,
        inputs=[tex, coords, output],
        device=device,
    )

    result = output.numpy()[0]
    expected = data[1, 1]  # 5.0
    test.assertAlmostEqual(result, expected, places=4)

    # Sample at texel center (0.5, 0.5) -> texel (0, 0) = 0.0
    coords_np2 = np.array([[0.5, 0.5]], dtype=np.float32)
    coords2 = wp.array(coords_np2, dtype=wp.vec2f, device=device)
    output2 = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        sample_texture2d_texel_coords,
        dim=1,
        inputs=[tex, coords2, output2],
        device=device,
    )

    result2 = output2.numpy()[0]
    test.assertAlmostEqual(result2, 0.0, places=4)


def test_texture2d_non_normalized_at_all_texels(test, device):
    """Test 2D texture with non-normalized coords sampling all texels."""
    width, height = 4, 4
    data = np.arange(16, dtype=np.float32).reshape((4, 4))

    tex = wp.Texture2D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        normalized_coords=False,
        device=device,
    )

    # Sample at all texel centers using texel-space coordinates
    coords_list = []
    for y in range(height):
        for x in range(width):
            # Texel center in texel space
            coords_list.append([x + 0.5, y + 0.5])

    coords_np = np.array(coords_list, dtype=np.float32)
    coords = wp.array(coords_np, dtype=wp.vec2f, device=device)
    output = wp.zeros(width * height, dtype=float, device=device)

    wp.launch(
        sample_texture2d_texel_coords,
        dim=width * height,
        inputs=[tex, coords, output],
        device=device,
    )

    result = output.numpy()
    expected = data.flatten()
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


def test_texture3d_non_normalized_coords(test, device):
    """Test 3D texture with non-normalized (texel-space) coordinates."""
    width, height, depth = 2, 2, 2
    data = np.arange(8, dtype=np.float32).reshape((2, 2, 2))

    tex = wp.Texture3D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        normalized_coords=False,
        device=device,
    )

    # Verify property
    test.assertFalse(tex.normalized_coords)

    # Sample at voxel centers using texel-space coordinates
    coords_list = []
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                coords_list.append([x + 0.5, y + 0.5, z + 0.5])

    coords_np = np.array(coords_list, dtype=np.float32)
    coords = wp.array(coords_np, dtype=wp.vec3f, device=device)
    output = wp.zeros(width * height * depth, dtype=float, device=device)

    wp.launch(
        sample_texture3d_texel_coords,
        dim=width * height * depth,
        inputs=[tex, coords, output],
        device=device,
    )

    result = output.numpy()
    expected = data.flatten()
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


def test_texture2d_non_normalized_linear(test, device):
    """Test 2D texture with non-normalized coords and linear filtering."""
    width, height = 2, 2
    data = np.array([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32)

    tex = wp.Texture2D(
        data,
        filter_mode=wp.TextureFilterMode.LINEAR,
        address_mode=wp.TextureAddressMode.CLAMP,
        normalized_coords=False,
        device=device,
    )

    # Sample at center (1.0, 1.0) in texel space - this is between all 4 texels
    # With linear filtering, should average all 4: (0 + 2 + 2 + 4) / 4 = 2.0
    coords_np = np.array([[1.0, 1.0]], dtype=np.float32)
    coords = wp.array(coords_np, dtype=wp.vec2f, device=device)
    output = wp.zeros(1, dtype=float, device=device)

    wp.launch(
        sample_texture2d_texel_coords,
        dim=1,
        inputs=[tex, coords, output],
        device=device,
    )

    result = output.numpy()[0]
    expected = 2.0  # Average of 0, 2, 2, 4
    test.assertAlmostEqual(result, expected, places=3)


def test_texture2d_backward_compat_address_mode(test, device):
    """Test that single address_mode parameter still works (backward compatibility)."""
    width, height = 2, 2
    data = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    # Old-style single address_mode should apply to all axes
    tex = wp.Texture2D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.WRAP,
        device=device,
    )

    # Both axes should have WRAP
    test.assertEqual(tex.address_mode_u, wp.TextureAddressMode.WRAP)
    test.assertEqual(tex.address_mode_v, wp.TextureAddressMode.WRAP)
    # Default should be normalized
    test.assertTrue(tex.normalized_coords)


def test_texture3d_backward_compat_address_mode(test, device):
    """Test that single address_mode parameter still works for 3D (backward compatibility)."""
    width, height, depth = 2, 2, 2
    data = np.arange(8, dtype=np.float32).reshape((2, 2, 2))

    # Old-style single address_mode should apply to all axes
    tex = wp.Texture3D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.MIRROR,
        device=device,
    )

    # All axes should have MIRROR
    test.assertEqual(tex.address_mode_u, wp.TextureAddressMode.MIRROR)
    test.assertEqual(tex.address_mode_v, wp.TextureAddressMode.MIRROR)
    test.assertEqual(tex.address_mode_w, wp.TextureAddressMode.MIRROR)
    # Default should be normalized
    test.assertTrue(tex.normalized_coords)


# ============================================================================
# Texture as Struct Member Tests
# ============================================================================


@wp.struct
class TextureStruct2D:
    """Struct containing a 2D texture member."""

    tex: wp.Texture2D
    scale: float


@wp.struct
class TextureStruct3D:
    """Struct containing a 3D texture member."""

    tex: wp.Texture3D
    offset: float


@wp.struct
class TextureStructBoth:
    """Struct containing both 2D and 3D texture members."""

    tex2d: wp.Texture2D
    tex3d: wp.Texture3D
    multiplier: float


@wp.kernel
def sample_texture2d_from_struct(
    s: TextureStruct2D,
    uv: wp.vec2f,
    output: wp.array(dtype=float),
):
    """Sample a 2D texture from a struct member."""
    tid = wp.tid()
    value = wp.texture_sample(s.tex, uv, dtype=float)
    output[tid] = value * s.scale


@wp.kernel
def sample_texture3d_from_struct(
    s: TextureStruct3D,
    uvw: wp.vec3f,
    output: wp.array(dtype=float),
):
    """Sample a 3D texture from a struct member."""
    tid = wp.tid()
    value = wp.texture_sample(s.tex, uvw, dtype=float)
    output[tid] = value + s.offset


@wp.kernel
def sample_both_textures_from_struct(
    s: TextureStructBoth,
    uv: wp.vec2f,
    uvw: wp.vec3f,
    output: wp.array(dtype=float),
):
    """Sample both 2D and 3D textures from a struct."""
    tid = wp.tid()
    val2d = wp.texture_sample(s.tex2d, uv, dtype=float)
    val3d = wp.texture_sample(s.tex3d, uvw, dtype=float)
    output[tid] = (val2d + val3d) * s.multiplier


def test_texture2d_struct_member(test, device):
    """Test that wp.Texture2D can be a member of a warp struct."""
    width, height = 4, 4

    # Create a texture with a constant value
    data = np.full((height, width), 0.5, dtype=np.float32)
    tex = wp.Texture2D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Create struct instance with texture
    s = TextureStruct2D()
    s.tex = tex
    s.scale = 2.0

    # Output array
    output = wp.zeros(1, dtype=float, device=device)

    # Sample at center
    uv = wp.vec2f(0.5, 0.5)

    wp.launch(
        sample_texture2d_from_struct,
        dim=1,
        inputs=[s, uv, output],
        device=device,
    )

    result = output.numpy()[0]
    expected = 0.5 * 2.0  # texture value * scale
    test.assertAlmostEqual(result, expected, places=4)


def test_texture3d_struct_member(test, device):
    """Test that wp.Texture3D can be a member of a warp struct."""
    width, height, depth = 4, 4, 4

    # Create a texture with a constant value
    data = np.full((depth, height, width), 0.25, dtype=np.float32)
    tex = wp.Texture3D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Create struct instance with texture
    s = TextureStruct3D()
    s.tex = tex
    s.offset = 0.75

    # Output array
    output = wp.zeros(1, dtype=float, device=device)

    # Sample at center
    uvw = wp.vec3f(0.5, 0.5, 0.5)

    wp.launch(
        sample_texture3d_from_struct,
        dim=1,
        inputs=[s, uvw, output],
        device=device,
    )

    result = output.numpy()[0]
    expected = 0.25 + 0.75  # texture value + offset
    test.assertAlmostEqual(result, expected, places=4)


def test_texture_struct_both_members(test, device):
    """Test that both wp.Texture2D and wp.Texture3D can be members of the same struct."""
    # Create 2D texture
    data2d = np.full((4, 4), 0.3, dtype=np.float32)
    tex2d = wp.Texture2D(
        data2d,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Create 3D texture
    data3d = np.full((4, 4, 4), 0.2, dtype=np.float32)
    tex3d = wp.Texture3D(
        data3d,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Create struct instance with both textures
    s = TextureStructBoth()
    s.tex2d = tex2d
    s.tex3d = tex3d
    s.multiplier = 2.0

    # Output array
    output = wp.zeros(1, dtype=float, device=device)

    # Sample coordinates
    uv = wp.vec2f(0.5, 0.5)
    uvw = wp.vec3f(0.5, 0.5, 0.5)

    wp.launch(
        sample_both_textures_from_struct,
        dim=1,
        inputs=[s, uv, uvw, output],
        device=device,
    )

    result = output.numpy()[0]
    expected = (0.3 + 0.2) * 2.0  # (tex2d + tex3d) * multiplier
    test.assertAlmostEqual(result, expected, places=4)


# ============================================================================
# Texture Array Tests
# ============================================================================


def test_texture2d_array(test, device):
    """Test sampling from an array of 2D textures.

    Creates multiple 2D textures with different constant values and verifies
    that each thread correctly samples from its corresponding texture.
    """
    num_textures = 4
    width, height = 4, 4

    # Create textures with different constant values (0.25, 0.5, 0.75, 1.0)
    textures = []
    expected_values = []
    for i in range(num_textures):
        value = (i + 1) * 0.25
        data = np.full((height, width), value, dtype=np.float32)
        tex = wp.Texture2D(
            data,
            filter_mode=wp.TextureFilterMode.CLOSEST,
            address_mode=wp.TextureAddressMode.CLAMP,
            device=device,
        )
        textures.append(tex)
        expected_values.append(value)

    # Create array of textures
    tex_array = wp.array(textures, dtype=wp.Texture2D, device=device)

    # Output array
    output = wp.zeros(num_textures, dtype=float, device=device)

    # Sample at center of each texture (same UV for all)
    uv = wp.vec2f(0.5, 0.5)

    wp.launch(
        sample_texture2d_array,
        dim=num_textures,
        inputs=[tex_array, uv, output],
        device=device,
    )

    result = output.numpy()
    expected = np.array(expected_values, dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_texture3d_array(test, device):
    """Test sampling from an array of 3D textures.

    Creates multiple 3D textures with different constant values and verifies
    that each thread correctly samples from its corresponding texture.
    """
    num_textures = 4
    width, height, depth = 4, 4, 4

    # Create textures with different constant values (0.1, 0.2, 0.3, 0.4)
    textures = []
    expected_values = []
    for i in range(num_textures):
        value = (i + 1) * 0.1
        data = np.full((depth, height, width), value, dtype=np.float32)
        tex = wp.Texture3D(
            data,
            filter_mode=wp.TextureFilterMode.CLOSEST,
            address_mode=wp.TextureAddressMode.CLAMP,
            device=device,
        )
        textures.append(tex)
        expected_values.append(value)

    # Create array of textures
    tex_array = wp.array(textures, dtype=wp.Texture3D, device=device)

    # Output array
    output = wp.zeros(num_textures, dtype=float, device=device)

    # Sample at center of each texture (same UVW for all)
    uvw = wp.vec3f(0.5, 0.5, 0.5)

    wp.launch(
        sample_texture3d_array,
        dim=num_textures,
        inputs=[tex_array, uvw, output],
        device=device,
    )

    result = output.numpy()
    expected = np.array(expected_values, dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


# ============================================================================
# Test Class
# ============================================================================


class TestTexture(unittest.TestCase):
    pass


# Register tests - textures work on both CPU and CUDA devices
cuda_devices = get_selected_cuda_test_devices()
all_devices = get_test_devices()

# Core texture tests - run on all devices (CPU + CUDA)
add_function_test(TestTexture, "test_texture2d_1channel", test_texture2d_1channel, devices=all_devices)
add_function_test(TestTexture, "test_texture2d_2channel", test_texture2d_2channel, devices=all_devices)
add_function_test(TestTexture, "test_texture2d_4channel", test_texture2d_4channel, devices=all_devices)
add_function_test(TestTexture, "test_texture2d_linear_filter", test_texture2d_linear_filter, devices=all_devices)
add_function_test(TestTexture, "test_texture2d_resolution_query", test_texture2d_resolution_query, devices=all_devices)
add_function_test(TestTexture, "test_texture3d_1channel", test_texture3d_1channel, devices=all_devices)
add_function_test(TestTexture, "test_texture3d_2channel", test_texture3d_2channel, devices=all_devices)
add_function_test(TestTexture, "test_texture3d_4channel", test_texture3d_4channel, devices=all_devices)
add_function_test(TestTexture, "test_texture3d_linear_filter", test_texture3d_linear_filter, devices=all_devices)
add_function_test(TestTexture, "test_texture3d_resolution_query", test_texture3d_resolution_query, devices=all_devices)

# Interpolation tests - run on all devices
add_function_test(
    TestTexture, "test_texture2d_nearest_interpolation", test_texture2d_nearest_interpolation, devices=all_devices
)
add_function_test(
    TestTexture, "test_texture2d_linear_interpolation", test_texture2d_linear_interpolation, devices=all_devices
)
add_function_test(
    TestTexture, "test_texture3d_nearest_interpolation", test_texture3d_nearest_interpolation, devices=all_devices
)
add_function_test(
    TestTexture, "test_texture3d_linear_interpolation", test_texture3d_linear_interpolation, devices=all_devices
)

# Compressed texture tests (uint8, uint16) - run on all devices
add_function_test(TestTexture, "test_texture2d_uint8", test_texture2d_uint8, devices=all_devices)
add_function_test(TestTexture, "test_texture2d_uint16", test_texture2d_uint16, devices=all_devices)
add_function_test(TestTexture, "test_texture3d_uint8", test_texture3d_uint8, devices=all_devices)
add_function_test(TestTexture, "test_texture3d_uint16", test_texture3d_uint16, devices=all_devices)
add_function_test(
    TestTexture,
    "test_texture2d_uint8_linear_interpolation",
    test_texture2d_uint8_linear_interpolation,
    devices=all_devices,
)
add_function_test(
    TestTexture,
    "test_texture2d_uint16_linear_interpolation",
    test_texture2d_uint16_linear_interpolation,
    devices=all_devices,
)
add_function_test(
    TestTexture,
    "test_texture3d_uint8_linear_interpolation",
    test_texture3d_uint8_linear_interpolation,
    devices=all_devices,
)
add_function_test(
    TestTexture,
    "test_texture3d_uint16_linear_interpolation",
    test_texture3d_uint16_linear_interpolation,
    devices=all_devices,
)

# These tests don't need a device
add_function_test(TestTexture, "test_texture2d_new_del", test_texture2d_new_del, devices=[None])
add_function_test(TestTexture, "test_texture3d_new_del", test_texture3d_new_del, devices=[None])

# Per-axis address mode tests - run on all devices
add_function_test(
    TestTexture, "test_texture2d_per_axis_address_modes", test_texture2d_per_axis_address_modes, devices=all_devices
)
add_function_test(
    TestTexture, "test_texture2d_address_mode_tuple", test_texture2d_address_mode_tuple, devices=all_devices
)
add_function_test(
    TestTexture, "test_texture3d_per_axis_address_modes", test_texture3d_per_axis_address_modes, devices=all_devices
)
add_function_test(
    TestTexture, "test_texture3d_address_mode_tuple", test_texture3d_address_mode_tuple, devices=all_devices
)
add_function_test(TestTexture, "test_texture2d_wrap_linear_edge", test_texture2d_wrap_linear_edge, devices=all_devices)
add_function_test(
    TestTexture, "test_texture2d_mirror_linear_edge", test_texture2d_mirror_linear_edge, devices=all_devices
)

# Non-normalized coordinates tests - run on all devices
add_function_test(
    TestTexture, "test_texture2d_non_normalized_coords", test_texture2d_non_normalized_coords, devices=all_devices
)
add_function_test(
    TestTexture,
    "test_texture2d_non_normalized_at_all_texels",
    test_texture2d_non_normalized_at_all_texels,
    devices=all_devices,
)
add_function_test(
    TestTexture, "test_texture3d_non_normalized_coords", test_texture3d_non_normalized_coords, devices=all_devices
)
add_function_test(
    TestTexture, "test_texture2d_non_normalized_linear", test_texture2d_non_normalized_linear, devices=all_devices
)

# Backward compatibility tests - run on all devices
add_function_test(
    TestTexture,
    "test_texture2d_backward_compat_address_mode",
    test_texture2d_backward_compat_address_mode,
    devices=all_devices,
)
add_function_test(
    TestTexture,
    "test_texture3d_backward_compat_address_mode",
    test_texture3d_backward_compat_address_mode,
    devices=all_devices,
)

# Texture array tests - run on all devices
add_function_test(TestTexture, "test_texture2d_array", test_texture2d_array, devices=all_devices)
add_function_test(TestTexture, "test_texture3d_array", test_texture3d_array, devices=all_devices)

# Texture as struct member tests - run on all devices
add_function_test(TestTexture, "test_texture2d_struct_member", test_texture2d_struct_member, devices=all_devices)
add_function_test(TestTexture, "test_texture3d_struct_member", test_texture3d_struct_member, devices=all_devices)
add_function_test(
    TestTexture, "test_texture_struct_both_members", test_texture_struct_both_members, devices=all_devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
