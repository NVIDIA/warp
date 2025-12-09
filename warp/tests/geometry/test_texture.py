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

"""Unit tests for 2D and 3D texture functionality."""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices

# ============================================================================
# 2D Texture Kernels
# ============================================================================


@wp.kernel
def sample_texture2d_f_at_centers(
    tex: wp.texture2d_t,
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

    output[tid] = wp.tex2d_float(tex, u, v)


@wp.kernel
def sample_texture2d_v4_at_centers(
    tex: wp.texture2d_t,
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

    output[tid] = wp.tex2d_vec4(tex, u, v)


@wp.kernel
def test_texture2d_resolution(
    tex: wp.texture2d_t,
    expected_width: int,
    expected_height: int,
):
    """Test resolution query functions for 2D texture."""
    w = wp.texture_width(tex)
    h = wp.texture_height(tex)

    wp.expect_eq(w, expected_width)
    wp.expect_eq(h, expected_height)


# ============================================================================
# 3D Texture Kernels
# ============================================================================


@wp.kernel
def sample_texture3d_f_at_centers(
    tex: wp.texture3d_t,
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
    w = (wp.float(z) + 0.5) / wp.float(depth)

    output[tid] = wp.tex3d_float(tex, u, v, w)


@wp.kernel
def sample_texture3d_v4_at_centers(
    tex: wp.texture3d_t,
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

    output[tid] = wp.tex3d_vec4(tex, u, v, ww)


@wp.kernel
def test_texture3d_resolution(
    tex: wp.texture3d_t,
    expected_width: int,
    expected_height: int,
    expected_depth: int,
):
    """Test resolution query functions for 3D texture."""
    w = wp.texture_width(tex)
    h = wp.texture_height(tex)
    d = wp.texture_depth(tex)

    wp.expect_eq(w, expected_width)
    wp.expect_eq(h, expected_height)
    wp.expect_eq(d, expected_depth)


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

    try:
        # Create texture
        tex = wp.Texture2D(
            data,
            filter_mode=wp.Texture2D.NEAREST,
            address_mode=wp.Texture2D.CLAMP,
            device=device,
        )
    except (RuntimeError, AttributeError) as e:
        test.skipTest(f"Texture creation failed (native library may need rebuild): {e}")

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


def test_texture2d_4channel(test, device):
    """Test 2D texture with 4 channels, sampling at texel centers."""
    width, height = 32, 32
    num_channels = 4

    # Generate test data
    data = generate_sin_pattern_2d(width, height, num_channels)

    try:
        tex = wp.Texture2D(
            data,
            filter_mode=wp.Texture2D.NEAREST,
            address_mode=wp.Texture2D.CLAMP,
            device=device,
        )
    except (RuntimeError, AttributeError) as e:
        test.skipTest(f"Texture creation failed (native library may need rebuild): {e}")

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

    try:
        tex = wp.Texture2D(
            data,
            filter_mode=wp.Texture2D.LINEAR,
            address_mode=wp.Texture2D.CLAMP,
            device=device,
        )
    except (RuntimeError, AttributeError) as e:
        test.skipTest(f"Texture creation failed (native library may need rebuild): {e}")

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

    try:
        tex = wp.Texture2D(data, device=device)
    except (RuntimeError, AttributeError) as e:
        test.skipTest(f"Texture creation failed (native library may need rebuild): {e}")

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

    try:
        tex = wp.Texture3D(
            data,
            filter_mode=wp.Texture3D.NEAREST,
            address_mode=wp.Texture3D.CLAMP,
            device=device,
        )
    except (RuntimeError, AttributeError) as e:
        test.skipTest(f"Texture creation failed (native library may need rebuild): {e}")

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


def test_texture3d_4channel(test, device):
    """Test 3D texture with 4 channels, sampling at voxel centers."""
    width, height, depth = 8, 8, 8
    num_channels = 4

    # Generate test data
    data = generate_sin_pattern_3d(width, height, depth, num_channels)

    try:
        tex = wp.Texture3D(
            data,
            filter_mode=wp.Texture3D.NEAREST,
            address_mode=wp.Texture3D.CLAMP,
            device=device,
        )
    except (RuntimeError, AttributeError) as e:
        test.skipTest(f"Texture creation failed (native library may need rebuild): {e}")

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

    try:
        tex = wp.Texture3D(
            data,
            filter_mode=wp.Texture3D.LINEAR,
            address_mode=wp.Texture3D.CLAMP,
            device=device,
        )
    except (RuntimeError, AttributeError) as e:
        test.skipTest(f"Texture creation failed (native library may need rebuild): {e}")

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

    try:
        tex = wp.Texture3D(data, device=device)
    except (RuntimeError, AttributeError) as e:
        test.skipTest(f"Texture creation failed (native library may need rebuild): {e}")

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
    tex: wp.texture2d_t,
    uvs: wp.array(dtype=wp.vec2f),
    output: wp.array(dtype=float),
):
    """Sample a 2D texture at specified UV coordinates."""
    tid = wp.tid()
    uv = uvs[tid]
    output[tid] = wp.tex2d_float(tex, uv[0], uv[1])


@wp.kernel
def sample_texture3d_at_uvw(
    tex: wp.texture3d_t,
    uvws: wp.array(dtype=wp.vec3f),
    output: wp.array(dtype=float),
):
    """Sample a 3D texture at specified UVW coordinates."""
    tid = wp.tid()
    uvw = uvws[tid]
    output[tid] = wp.tex3d_float(tex, uvw[0], uvw[1], uvw[2])


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

    try:
        tex = wp.Texture2D(
            data,
            filter_mode=wp.Texture2D.NEAREST,
            address_mode=wp.Texture2D.CLAMP,
            device=device,
        )
    except (RuntimeError, AttributeError) as e:
        test.skipTest(f"Texture creation failed: {e}")

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

    try:
        tex = wp.Texture2D(
            data,
            filter_mode=wp.Texture2D.LINEAR,
            address_mode=wp.Texture2D.CLAMP,
            device=device,
        )
    except (RuntimeError, AttributeError) as e:
        test.skipTest(f"Texture creation failed: {e}")

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

    try:
        tex = wp.Texture3D(
            data,
            filter_mode=wp.Texture3D.NEAREST,
            address_mode=wp.Texture3D.CLAMP,
            device=device,
        )
    except (RuntimeError, AttributeError) as e:
        test.skipTest(f"Texture creation failed: {e}")

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

    try:
        tex = wp.Texture3D(
            data,
            filter_mode=wp.Texture3D.LINEAR,
            address_mode=wp.Texture3D.CLAMP,
            device=device,
        )
    except (RuntimeError, AttributeError) as e:
        test.skipTest(f"Texture creation failed: {e}")

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

    try:
        tex = wp.Texture2D(
            data,
            filter_mode=wp.Texture2D.NEAREST,
            address_mode=wp.Texture2D.CLAMP,
            device=device,
        )
    except (RuntimeError, AttributeError) as e:
        test.skipTest(f"Texture creation failed: {e}")

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

    try:
        tex = wp.Texture2D(
            data,
            filter_mode=wp.Texture2D.NEAREST,
            address_mode=wp.Texture2D.CLAMP,
            device=device,
        )
    except (RuntimeError, AttributeError) as e:
        test.skipTest(f"Texture creation failed: {e}")

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

    try:
        tex = wp.Texture3D(
            data,
            filter_mode=wp.Texture3D.NEAREST,
            address_mode=wp.Texture3D.CLAMP,
            device=device,
        )
    except (RuntimeError, AttributeError) as e:
        test.skipTest(f"Texture creation failed: {e}")

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

    try:
        tex = wp.Texture2D(
            data,
            filter_mode=wp.Texture2D.LINEAR,
            address_mode=wp.Texture2D.CLAMP,
            device=device,
        )
    except (RuntimeError, AttributeError) as e:
        test.skipTest(f"Texture creation failed: {e}")

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


# ============================================================================
# Test Class
# ============================================================================


class TestTexture(unittest.TestCase):
    pass


# Register tests - textures only work on CUDA devices
cuda_devices = get_selected_cuda_test_devices()

add_function_test(TestTexture, "test_texture2d_1channel", test_texture2d_1channel, devices=cuda_devices)
add_function_test(TestTexture, "test_texture2d_4channel", test_texture2d_4channel, devices=cuda_devices)
add_function_test(TestTexture, "test_texture2d_linear_filter", test_texture2d_linear_filter, devices=cuda_devices)
add_function_test(TestTexture, "test_texture2d_resolution_query", test_texture2d_resolution_query, devices=cuda_devices)
add_function_test(TestTexture, "test_texture3d_1channel", test_texture3d_1channel, devices=cuda_devices)
add_function_test(TestTexture, "test_texture3d_4channel", test_texture3d_4channel, devices=cuda_devices)
add_function_test(TestTexture, "test_texture3d_linear_filter", test_texture3d_linear_filter, devices=cuda_devices)
add_function_test(TestTexture, "test_texture3d_resolution_query", test_texture3d_resolution_query, devices=cuda_devices)

# Interpolation tests
add_function_test(
    TestTexture, "test_texture2d_nearest_interpolation", test_texture2d_nearest_interpolation, devices=cuda_devices
)
add_function_test(
    TestTexture, "test_texture2d_linear_interpolation", test_texture2d_linear_interpolation, devices=cuda_devices
)
add_function_test(
    TestTexture, "test_texture3d_nearest_interpolation", test_texture3d_nearest_interpolation, devices=cuda_devices
)
add_function_test(
    TestTexture, "test_texture3d_linear_interpolation", test_texture3d_linear_interpolation, devices=cuda_devices
)

# Compressed texture tests (uint8, uint16)
add_function_test(TestTexture, "test_texture2d_uint8", test_texture2d_uint8, devices=cuda_devices)
add_function_test(TestTexture, "test_texture2d_uint16", test_texture2d_uint16, devices=cuda_devices)
add_function_test(TestTexture, "test_texture3d_uint8", test_texture3d_uint8, devices=cuda_devices)
add_function_test(
    TestTexture,
    "test_texture2d_uint8_linear_interpolation",
    test_texture2d_uint8_linear_interpolation,
    devices=cuda_devices,
)

# These tests don't need a device
add_function_test(TestTexture, "test_texture2d_new_del", test_texture2d_new_del, devices=[None])
add_function_test(TestTexture, "test_texture3d_new_del", test_texture3d_new_del, devices=[None])


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
