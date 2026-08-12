# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for 1D, 2D, and 3D texture functionality on both CPU and CUDA devices."""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices

# ============================================================================
# 1D Texture Kernels
# ============================================================================


@wp.kernel
def sample_texture1d_f_at_centers(
    tex: wp.Texture1D,
    output: wp.array[float],
    width: int,
):
    """Sample a 1-channel 1D texture at texel centers."""
    tid = wp.tid()

    # Compute normalized coordinates at texel centers
    # For a texture of width W, texel i has center at (i + 0.5) / W
    u = (wp.float(tid) + 0.5) / wp.float(width)

    output[tid] = wp.texture_sample(tex, u, dtype=float)


@wp.kernel
def sample_texture1d_v2_at_centers(
    tex: wp.Texture1D,
    output: wp.array[wp.vec2f],
    width: int,
):
    """Sample a 2-channel 1D texture at texel centers."""
    tid = wp.tid()

    u = (wp.float(tid) + 0.5) / wp.float(width)

    output[tid] = wp.texture_sample(tex, u, dtype=wp.vec2f)


@wp.kernel
def sample_texture1d_v4_at_centers(
    tex: wp.Texture1D,
    output: wp.array[wp.vec4f],
    width: int,
):
    """Sample a 4-channel 1D texture at texel centers."""
    tid = wp.tid()

    u = (wp.float(tid) + 0.5) / wp.float(width)

    output[tid] = wp.texture_sample(tex, u, dtype=wp.vec4f)


@wp.kernel
def test_texture1d_resolution(
    tex: wp.Texture1D,
    expected_width: int,
):
    """Test resolution query using texture.width."""
    w = tex.width

    wp.expect_eq(w, expected_width)


# ============================================================================
# 2D Texture Kernels
# ============================================================================


@wp.kernel
def sample_texture2d_f_at_centers(
    tex: wp.Texture2D,
    output: wp.array[float],
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
    output: wp.array[wp.vec2f],
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
    output: wp.array[wp.vec4f],
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
    output: wp.array[float],
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
    output: wp.array[wp.vec2f],
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
    output: wp.array[wp.vec4f],
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
    textures: wp.array[wp.Texture2D],
    uv: wp.vec2f,
    output: wp.array[float],
):
    """Sample from an array of 2D textures, one texture per thread."""
    tid = wp.tid()
    tex = textures[tid]
    output[tid] = wp.texture_sample(tex, uv, dtype=float)


@wp.kernel
def sample_texture3d_array(
    textures: wp.array[wp.Texture3D],
    uvw: wp.vec3f,
    output: wp.array[float],
):
    """Sample from an array of 3D textures, one texture per thread."""
    tid = wp.tid()
    tex = textures[tid]
    output[tid] = wp.texture_sample(tex, uvw, dtype=float)


# ============================================================================
# Test Data Generation
# ============================================================================


def generate_sin_pattern_1d(width: int, num_channels: int) -> np.ndarray:
    """Generate a 1D sin pattern for testing.

    Creates a pattern based on: sin(2*pi*x/width)
    Values are scaled to [0, 1] range.
    """
    x = np.arange(width, dtype=np.float32)

    # Create base sin pattern
    pattern = np.sin(2 * np.pi * x / width)
    # Scale to [0, 1]
    pattern = (pattern + 1.0) * 0.5

    if num_channels == 1:
        return pattern.astype(np.float32)
    else:
        # Create multi-channel pattern
        result = np.zeros((width, num_channels), dtype=np.float32)
        for c in range(num_channels):
            # Each channel has a slightly different phase
            phase = c * 0.25
            channel_pattern = np.sin(2 * np.pi * (x / width + phase))
            result[:, c] = (channel_pattern + 1.0) * 0.5
        return result


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


def test_texture_dtype_prefers_warp_types(test, device):
    """Texture dtype property should report canonical Warp scalar types."""
    data_u8 = np.zeros((4, 4), dtype=np.uint8)
    tex_u8 = wp.Texture2D(data_u8, device=device)
    test.assertIs(tex_u8.dtype, wp.uint8)

    data_f32 = np.zeros((2, 2, 2), dtype=np.float32)
    tex_f32 = wp.Texture3D(data_f32, device=device)
    test.assertIs(tex_f32.dtype, wp.float32)


def test_texture_dtype_float_alias_maps_to_float32(test, device):
    """Python float in constructor args should map to Warp float32."""
    tex = wp.Texture1D(width=4, num_channels=1, dtype=float, device=device)
    test.assertIs(tex.dtype, wp.float32)


def test_texture_dtype_int_alias_maps_to_int32(test, device):
    """Python int in constructor args should map to Warp int32."""
    tex = wp.Texture1D(width=4, num_channels=1, dtype=int, device=device)
    test.assertIs(tex.dtype, wp.int32)


# ============================================================================
# 1D Texture Test Functions
# ============================================================================


def test_texture1d_1channel(test, device):
    """Test 1D texture with 1 channel, sampling at texel centers."""
    width = 32
    num_channels = 1

    # Generate test data
    data = generate_sin_pattern_1d(width, num_channels)

    # Create texture
    tex = wp.Texture1D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Create output array
    output = wp.zeros(width, dtype=float, device=device)

    # Sample texture at texel centers
    wp.launch(
        sample_texture1d_f_at_centers,
        dim=width,
        inputs=[tex, output, width],
        device=device,
    )

    # Compare results
    expected = data.flatten()
    result = output.numpy()

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_texture1d_2channel(test, device):
    """Test 1D texture with 2 channels, sampling at texel centers."""
    width = 32
    num_channels = 2

    # Generate test data
    data = generate_sin_pattern_1d(width, num_channels)

    tex = wp.Texture1D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Create output array
    output = wp.zeros(width, dtype=wp.vec2f, device=device)

    # Sample texture at texel centers
    wp.launch(
        sample_texture1d_v2_at_centers,
        dim=width,
        inputs=[tex, output, width],
        device=device,
    )

    # Compare results
    expected = data.reshape(-1, 2)
    result = output.numpy()

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_texture1d_4channel(test, device):
    """Test 1D texture with 4 channels, sampling at texel centers."""
    width = 32
    num_channels = 4

    # Generate test data
    data = generate_sin_pattern_1d(width, num_channels)

    tex = wp.Texture1D(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Create output array
    output = wp.zeros(width, dtype=wp.vec4f, device=device)

    # Sample texture at texel centers
    wp.launch(
        sample_texture1d_v4_at_centers,
        dim=width,
        inputs=[tex, output, width],
        device=device,
    )

    # Compare results
    expected = data.reshape(-1, 4)
    result = output.numpy()

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_texture1d_linear_filter(test, device):
    """Test 1D texture with linear filtering at texel centers.

    At texel centers, linear filtering should give the same result as nearest.
    """
    width = 16
    num_channels = 1

    # Generate test data
    data = generate_sin_pattern_1d(width, num_channels)

    tex = wp.Texture1D(
        data,
        filter_mode=wp.TextureFilterMode.LINEAR,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )

    # Create output array
    output = wp.zeros(width, dtype=float, device=device)

    # Sample texture at texel centers
    wp.launch(
        sample_texture1d_f_at_centers,
        dim=width,
        inputs=[tex, output, width],
        device=device,
    )

    # At texel centers, linear filtering should give exact values
    expected = data.flatten()
    result = output.numpy()

    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


def test_texture1d_resolution_query(test, device):
    """Test resolution query functions for 1D texture."""
    width = 64

    data = np.zeros((width, 4), dtype=np.float32)

    tex = wp.Texture1D(data, device=device)

    # Test resolution queries in kernel
    wp.launch(
        test_texture1d_resolution,
        dim=1,
        inputs=[tex, width],
        device=device,
    )


def test_texture1d_new_del(test, device):
    """Test proper handling of uninitialized texture (created with __new__ but not __init__)."""
    instance = wp.Texture1D.__new__(wp.Texture1D)
    instance.__del__()


def test_texture2d_constructor_from_same_device_array(test, device):
    """Texture2D constructor should accept same-device wp.array input."""
    h, w = 8, 16
    data = np.random.default_rng(1234).random((h, w, 4), dtype=np.float32)
    src = wp.array(data, dtype=wp.vec4, device=device)

    tex = wp.Texture2D(src, filter_mode=wp.TextureFilterMode.CLOSEST, device=device)
    output = wp.zeros(w * h, dtype=wp.vec4f, device=device)

    wp.launch(
        sample_texture2d_v4_at_centers,
        dim=w * h,
        inputs=[tex, output, w, h],
        device=device,
    )

    expected = data.reshape(-1, 4)
    np.testing.assert_allclose(output.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_texture2d_constructor_transfers_cross_device(test, device):
    """Constructor should transparently transfer wp.array data to the target device."""
    data = np.random.default_rng(42).random((4, 4), dtype=np.float32)
    src = wp.array(data, dtype=float, device="cpu")
    tex = wp.Texture2D(src, filter_mode=wp.TextureFilterMode.CLOSEST, device=device)
    output = wp.zeros(4 * 4, dtype=float, device=device)
    wp.launch(
        sample_texture2d_f_at_centers,
        dim=4 * 4,
        inputs=[tex, output, 4, 4],
        device=device,
    )
    np.testing.assert_allclose(output.numpy(), data.flatten(), rtol=1e-5, atol=1e-5)


def test_texture2d_cuda_interop_handles(test, device):
    """Test CUDA interop handles for 2D textures."""
    data = np.zeros((4, 4, 4), dtype=np.float32)
    tex = wp.Texture2D(data, device=device)

    test.assertGreater(tex.cuda_texture, 0)
    test.assertGreater(tex.cuda_array, 0)


def test_texture2d_cuda_array_wraps_non_mipmapped(test, device):
    """Verify Texture2D.cuda_array exposes a wrappable cudaArray_t for non-mipmapped textures."""
    data = np.zeros((4, 4, 4), dtype=np.float32)
    tex = wp.Texture2D(data, device=device)

    wrapped = wp.Texture2D(cuda_array=tex.cuda_array, device=device)

    test.assertEqual(wrapped.width, 4)
    test.assertEqual(wrapped.height, 4)
    test.assertEqual(wrapped.depth, 1)
    test.assertEqual(wrapped.num_channels, 4)
    test.assertEqual(wrapped.dtype, wp.float32)
    test.assertFalse(wrapped.is_mipmapped)


def test_texture2d_mipmapped_cuda_array_current_limitation(test, device):
    """Verify Texture2D.cuda_array rejects mipmapped CUDA textures as a current limitation."""
    data = np.zeros((4, 4, 4), dtype=np.float32)
    tex = wp.Texture2D(data, num_mip_levels=2, device=device)

    test.assertTrue(tex.is_mipmapped)
    with test.assertRaisesRegex(RuntimeError, "currently limited to non-mipmapped CUDA textures"):
        _ = tex.cuda_array


def test_texture3d_cuda_interop_handles(test, device):
    """Test CUDA interop handles for 3D textures."""
    data = np.zeros((4, 4, 4), dtype=np.float32)
    tex = wp.Texture3D(data, device=device)

    test.assertGreater(tex.cuda_texture, 0)
    test.assertGreater(tex.cuda_array, 0)


def test_texture_id_device_independent(test, device):
    """Texture.id should be valid on both CPU and CUDA."""
    data = np.zeros((4, 4, 4), dtype=np.float32)
    tex = wp.Texture2D(data, device=device)

    test.assertGreater(tex.id, 0)
    if device.is_cuda:
        test.assertEqual(tex.id, tex.cuda_texture)
    else:
        with test.assertRaisesRegex(RuntimeError, "only supported for CUDA textures"):
            _ = tex.cuda_texture


def test_texture_handle_properties_host_vs_cuda(test, device):
    """Validate texture handle properties for host and CUDA textures."""
    data = np.zeros((4, 4, 4), dtype=np.float32)
    tex = wp.Texture3D(data, device=device)

    test.assertGreater(tex.id, 0)
    if device.is_cuda:
        test.assertGreater(tex.cuda_texture, 0)
        test.assertGreater(tex.cuda_array, 0)
        with test.assertRaisesRegex(RuntimeError, "surface_access=True"):
            _ = tex.cuda_surface
    else:
        with test.assertRaisesRegex(RuntimeError, "only supported for CUDA textures"):
            _ = tex.cuda_array
        with test.assertRaisesRegex(RuntimeError, "only supported for CUDA textures"):
            _ = tex.cuda_surface
        with test.assertRaisesRegex(RuntimeError, "only supported for CUDA textures"):
            _ = tex.cuda_texture


def test_texture2d_cuda_array_copy_api(test, device):
    """Test Warp-native CUDA array copy helpers on Texture2D."""
    h, w = 8, 16
    data = np.random.default_rng(1234).random((h, w, 4), dtype=np.float32)
    src = wp.array(data, dtype=wp.vec4, device=device)
    dst = wp.zeros((h, w), dtype=wp.vec4, device=device)
    tex = wp.Texture2D(np.zeros_like(data), device=device)

    tex.copy_from(src)
    tex.copy_to(dst)

    np.testing.assert_allclose(dst.numpy(), data, rtol=1e-6, atol=1e-6)


def test_texture3d_cuda_array_copy_api(test, device):
    """Test Warp-native CUDA array copy helpers on Texture3D."""
    d, h, w = 6, 8, 16
    data = np.random.default_rng(1234).random((d, h, w, 4), dtype=np.float32)
    src = wp.array(data, dtype=wp.vec4, device=device)
    dst = wp.zeros((d, h, w), dtype=wp.vec4, device=device)
    tex = wp.Texture3D(np.zeros_like(data), device=device)

    tex.copy_from(src)
    tex.copy_to(dst)

    np.testing.assert_allclose(dst.numpy(), data, rtol=1e-6, atol=1e-6)


def test_texture_copy_validation_messages(test, device):
    tex = wp.Texture2D(np.zeros((2, 3), dtype=np.float32), device=device)

    with test.assertRaisesRegex(
        ValueError,
        r"Incompatible array shape for copy: texture shape=\(2, 3\), source shape=\(2, 4\)",
    ):
        tex.copy_from(np.zeros((2, 4), dtype=np.float32))

    with test.assertRaisesRegex(
        ValueError,
        "Incompatible array data type for copy: texture dtype=float32, source dtype=int32",
    ):
        tex.copy_from(np.zeros((2, 3), dtype=np.int32))

    dst = np.zeros((2, 4), dtype=np.float32)
    with test.assertRaisesRegex(
        ValueError,
        r"Incompatible array shape for copy: texture shape=\(2, 3\), destination shape=\(2, 4\)",
    ):
        tex.copy_to(dst)

    dst = np.zeros((2, 3), dtype=np.int32)
    with test.assertRaisesRegex(
        ValueError,
        "Incompatible array data type for copy: texture dtype=float32, destination dtype=int32",
    ):
        tex.copy_to(dst)

    other = wp.Texture2D(np.zeros((2, 4), dtype=np.float32), device=device)
    with test.assertRaisesRegex(
        ValueError,
        r"Incompatible texture shapes for copy: destination shape=\(2, 3\), source shape=\(2, 4\)",
    ):
        tex.copy_from(other)


def test_texture2d_cuda_array_copy_api_rejects_indexedarray(test, device):
    """Texture2D CUDA copy helpers should reject non-``wp.array`` inputs."""
    h, w = 8, 16
    data = np.random.default_rng(1234).random((h, w, 4), dtype=np.float32)
    src = wp.array(data, dtype=wp.vec4, device=device)
    indices = wp.array(np.arange(h, dtype=np.int32), dtype=int, device=device)
    indexed_src = wp.indexedarray(src, [indices, None])
    tex = wp.Texture2D(np.zeros_like(data), device=device)

    with test.assertRaisesRegex(ValueError, "Expected contiguous array"):
        tex.copy_from(indexed_src)


def test_texture3d_cuda_array_copy_api_rejects_indexedarray(test, device):
    """Texture3D CUDA copy helpers should reject non-``wp.array`` inputs."""
    d, h, w = 6, 8, 16
    data = np.random.default_rng(1234).random((d, h, w, 4), dtype=np.float32)
    dst = wp.zeros((d, h, w), dtype=wp.vec4, device=device)
    indices = wp.array(np.arange(d, dtype=np.int32), dtype=int, device=device)
    indexed_dst = wp.indexedarray(dst, [indices, None, None])
    tex = wp.Texture3D(np.zeros_like(data), device=device)

    with test.assertRaisesRegex(ValueError, "Expected contiguous array"):
        tex.copy_to(indexed_dst)


def test_texture2d_cuda_array_copy_api_graph_capture(test, device):
    """Validate Texture2D CUDA array copy helpers under CUDA graph capture."""
    h, w = 32, 64
    rng = np.random.default_rng(1234)
    data0 = rng.random((h, w, 4), dtype=np.float32)
    data1 = rng.random((h, w, 4), dtype=np.float32)

    src = wp.array(data0, dtype=wp.vec4, device=device)
    dst = wp.zeros((h, w), dtype=wp.vec4, device=device)
    tex = wp.Texture2D(np.zeros_like(data0), device=device)

    with wp.ScopedCapture(device, force_module_load=False) as capture:
        tex.copy_from(src)
        tex.copy_to(dst)

    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(dst.numpy(), data0, rtol=1e-6, atol=1e-6)

    src.assign(data1)
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(dst.numpy(), data1, rtol=1e-6, atol=1e-6)


def test_texture3d_cuda_array_copy_api_graph_capture(test, device):
    """Validate Texture3D CUDA array copy helpers under CUDA graph capture."""
    d, h, w = 8, 16, 32
    rng = np.random.default_rng(1234)
    data0 = rng.random((d, h, w, 4), dtype=np.float32)
    data1 = rng.random((d, h, w, 4), dtype=np.float32)

    src = wp.array(data0, dtype=wp.vec4, device=device)
    dst = wp.zeros((d, h, w), dtype=wp.vec4, device=device)
    tex = wp.Texture3D(np.zeros_like(data0), device=device)

    with wp.ScopedCapture(device, force_module_load=False) as capture:
        tex.copy_from(src)
        tex.copy_to(dst)

    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(dst.numpy(), data0, rtol=1e-6, atol=1e-6)

    src.assign(data1)
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(dst.numpy(), data1, rtol=1e-6, atol=1e-6)


def test_texture2d_cuda_array_copy_api_graph_capture_explicit_stream(test, device):
    """Validate 2D copy helpers in graph capture on an explicit stream."""
    h, w = 24, 48
    rng = np.random.default_rng(1234)
    data0 = rng.random((h, w, 4), dtype=np.float32)
    data1 = rng.random((h, w, 4), dtype=np.float32)

    src = wp.array(data0, dtype=wp.vec4, device=device)
    dst = wp.zeros((h, w), dtype=wp.vec4, device=device)
    tex = wp.Texture2D(np.zeros_like(data0), device=device)
    stream = wp.Stream(device)

    with wp.ScopedStream(stream):
        wp.capture_begin(stream=stream, force_module_load=False)
        tex.copy_from(src)
        tex.copy_to(dst)
        graph = wp.capture_end(stream=stream)

        wp.capture_launch(graph, stream=stream)
        wp.synchronize_stream(stream)
        np.testing.assert_allclose(dst.numpy(), data0, rtol=1e-6, atol=1e-6)

        src.assign(data1)
        wp.capture_launch(graph, stream=stream)
        wp.synchronize_stream(stream)
        np.testing.assert_allclose(dst.numpy(), data1, rtol=1e-6, atol=1e-6)


def test_texture3d_cuda_array_copy_api_graph_capture_explicit_stream(test, device):
    """Validate 3D copy helpers in graph capture on an explicit stream."""
    d, h, w = 6, 8, 16
    rng = np.random.default_rng(1234)
    data0 = rng.random((d, h, w, 4), dtype=np.float32)
    data1 = rng.random((d, h, w, 4), dtype=np.float32)

    src = wp.array(data0, dtype=wp.vec4, device=device)
    dst = wp.zeros((d, h, w), dtype=wp.vec4, device=device)
    tex = wp.Texture3D(np.zeros_like(data0), device=device)
    stream = wp.Stream(device)

    with wp.ScopedStream(stream):
        wp.capture_begin(stream=stream, force_module_load=False)
        tex.copy_from(src)
        tex.copy_to(dst)
        graph = wp.capture_end(stream=stream)

        wp.capture_launch(graph, stream=stream)
        wp.synchronize_stream(stream)
        np.testing.assert_allclose(dst.numpy(), data0, rtol=1e-6, atol=1e-6)

        src.assign(data1)
        wp.capture_launch(graph, stream=stream)
        wp.synchronize_stream(stream)
        np.testing.assert_allclose(dst.numpy(), data1, rtol=1e-6, atol=1e-6)


def test_texture2d_cuda_surface_property_graph_capture_stability(test, device):
    """Ensure lazy surface handle stays stable across graph-captured copy launches."""
    h, w = 16, 32
    rng = np.random.default_rng(1234)
    data0 = rng.random((h, w, 4), dtype=np.float32)
    data1 = rng.random((h, w, 4), dtype=np.float32)

    src = wp.array(data0, dtype=wp.vec4, device=device)
    dst = wp.zeros((h, w), dtype=wp.vec4, device=device)
    tex = wp.Texture2D(np.zeros_like(data0), device=device, surface_access=True)
    surface_before = tex.cuda_surface

    with wp.ScopedCapture(device, force_module_load=False) as capture:
        tex.copy_from(src)
        tex.copy_to(dst)

    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(dst.numpy(), data0, rtol=1e-6, atol=1e-6)
    test.assertEqual(tex.cuda_surface, surface_before)

    src.assign(data1)
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(dst.numpy(), data1, rtol=1e-6, atol=1e-6)
    test.assertEqual(tex.cuda_surface, surface_before)


def test_texture2d_cuda_surface_property_api(test, device):
    """Test lazy CUDA surface object creation via Texture2D.cuda_surface."""
    data = np.zeros((4, 4, 4), dtype=np.float32)
    tex = wp.Texture2D(data, device=device, surface_access=True)

    surface0 = tex.cuda_surface
    surface1 = tex.cuda_surface
    test.assertGreater(surface0, 0)
    test.assertEqual(surface0, surface1)
    test.assertEqual(surface0, tex.cuda_surface)


def test_texture3d_cuda_surface_property_api(test, device):
    """Test lazy CUDA surface object creation via Texture3D.cuda_surface."""
    data = np.zeros((4, 4, 4), dtype=np.float32)
    tex = wp.Texture3D(data, device=device, surface_access=True)

    surface0 = tex.cuda_surface
    surface1 = tex.cuda_surface
    test.assertGreater(surface0, 0)
    test.assertEqual(surface0, surface1)
    test.assertEqual(surface0, tex.cuda_surface)


def test_texture2d_cuda_surface_property_requires_surface_access(test, device):
    """cuda_surface should fail unless surface access was enabled at texture creation."""
    data = np.zeros((4, 4, 4), dtype=np.float32)
    tex = wp.Texture2D(data, device=device)

    with test.assertRaisesRegex(RuntimeError, "surface_access=True"):
        _ = tex.cuda_surface


def test_texture3d_cuda_surface_property_requires_surface_access(test, device):
    """cuda_surface should fail unless surface access was enabled at texture creation."""
    data = np.zeros((4, 4, 4), dtype=np.float32)
    tex = wp.Texture3D(data, device=device)

    with test.assertRaisesRegex(RuntimeError, "surface_access=True"):
        _ = tex.cuda_surface


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
    uvs: wp.array[wp.vec2f],
    output: wp.array[float],
):
    """Sample a 2D texture at specified UV coordinates."""
    tid = wp.tid()
    uv = uvs[tid]
    output[tid] = wp.texture_sample(tex, uv, dtype=float)


@wp.kernel
def sample_texture3d_at_uvw(
    tex: wp.Texture3D,
    uvws: wp.array[wp.vec3f],
    output: wp.array[float],
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

    test.assertEqual(tex.dtype, wp.uint8)

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

    test.assertEqual(tex.dtype, wp.uint16)

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

    test.assertEqual(tex.dtype, wp.uint8)

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

    test.assertEqual(tex.dtype, wp.uint16)

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
    uvs: wp.array[wp.vec2f],
    output: wp.array[float],
):
    """Sample a 2D texture at specified UV coordinates (may be outside [0,1])."""
    tid = wp.tid()
    uv = uvs[tid]
    output[tid] = wp.texture_sample(tex, uv, dtype=float)


@wp.kernel
def sample_texture3d_outside_bounds(
    tex: wp.Texture3D,
    uvws: wp.array[wp.vec3f],
    output: wp.array[float],
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
    coords: wp.array[wp.vec2f],
    output: wp.array[float],
):
    """Sample a 2D texture using texel-space coordinates."""
    tid = wp.tid()
    coord = coords[tid]
    output[tid] = wp.texture_sample(tex, coord, dtype=float)


@wp.kernel
def sample_texture3d_texel_coords(
    tex: wp.Texture3D,
    coords: wp.array[wp.vec3f],
    output: wp.array[float],
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
    output: wp.array[float],
):
    """Sample a 2D texture from a struct member."""
    tid = wp.tid()
    value = wp.texture_sample(s.tex, uv, dtype=float)
    output[tid] = value * s.scale


@wp.kernel
def sample_texture3d_from_struct(
    s: TextureStruct3D,
    uvw: wp.vec3f,
    output: wp.array[float],
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
    output: wp.array[float],
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
# Mipmap Tests
# ============================================================================


@wp.kernel
def sample_texture2d_mipmap(
    tex: wp.Texture2D,
    output: wp.array2d[wp.vec4f],
    width: int,
    height: int,
    lod: float,
):
    i, j = wp.tid()
    u = (wp.float(j) + 0.5) / wp.float(width)
    v = (wp.float(i) + 0.5) / wp.float(height)
    output[i, j] = wp.texture_sample(tex, wp.vec2f(u, v), dtype=wp.vec4f, lod=lod)


@wp.kernel
def sample_texture1d_mipmap(
    tex: wp.Texture1D,
    output: wp.array[float],
    width: int,
    lod: float,
):
    tid = wp.tid()
    u = (wp.float(tid) + 0.5) / wp.float(width)
    output[tid] = wp.texture_sample(tex, u, dtype=float, lod=lod)


@wp.kernel
def sample_texture3d_mipmap(
    tex: wp.Texture3D,
    output: wp.array3d[float],
    width: int,
    height: int,
    depth: int,
    lod: float,
):
    i, j, k = wp.tid()
    u = (wp.float(k) + 0.5) / wp.float(width)
    v = (wp.float(j) + 0.5) / wp.float(height)
    w = (wp.float(i) + 0.5) / wp.float(depth)
    output[i, j, k] = wp.texture_sample(tex, wp.vec3f(u, v, w), dtype=float, lod=lod)


def test_texture2d_mipmap_full_chain(test, device):
    """A texture created with num_mip_levels=0 should expose the full chain down to 1x1."""
    width = height = 16
    data = np.zeros((height, width, 4), dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    data[..., 0] = xx
    data[..., 1] = yy
    data[..., 2] = 1.0 - xx
    data[..., 3] = 1.0

    tex = wp.Texture2D(data=data, num_mip_levels=0, device=device)

    test.assertTrue(tex.is_mipmapped)
    # log2(16) + 1 == 5 levels.
    test.assertEqual(tex.num_mip_levels, 5)

    for lod in (0.0, 1.0, 2.0):
        output = wp.zeros((height, width), dtype=wp.vec4f, device=device)
        wp.launch(
            sample_texture2d_mipmap,
            dim=(height, width),
            inputs=[tex, output, width, height, lod],
            device=device,
        )
        arr = output.numpy()
        # Mean color of each mip level should stay close to the original mean (0.5, 0.5, 0.5, 1).
        mean = arr.mean(axis=(0, 1))
        np.testing.assert_allclose(mean, np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32), atol=5e-2)


def test_texture2d_mipmap_lod_selects_constant_level(test, device):
    """Sampling at an integer LOD should return the constant color of that level."""
    width = height = 8

    base = np.full((height, width, 4), fill_value=0.1, dtype=np.float32)
    base[..., 3] = 1.0

    num_levels = 4
    tex = wp.Texture2D(
        data=base,
        num_mip_levels=num_levels,
        mip_filter_mode=wp.TextureFilterMode.CLOSEST,
        device=device,
    )

    test.assertEqual(tex.num_mip_levels, num_levels)

    for lod in range(num_levels):
        output = wp.zeros((height, width), dtype=wp.vec4f, device=device)
        wp.launch(
            sample_texture2d_mipmap,
            dim=(height, width),
            inputs=[tex, output, width, height, float(lod)],
            device=device,
        )
        arr = output.numpy()
        np.testing.assert_allclose(arr.mean(axis=(0, 1)), np.array([0.1, 0.1, 0.1, 1.0]), atol=5e-3)


def test_texture1d_mipmap(test, device):
    width = 16
    data = np.linspace(0.0, 1.0, width, dtype=np.float32)
    tex = wp.Texture1D(data=data, num_mip_levels=0, device=device)

    test.assertTrue(tex.is_mipmapped)
    test.assertEqual(tex.num_mip_levels, 5)

    output = wp.zeros(width, dtype=float, device=device)
    wp.launch(sample_texture1d_mipmap, dim=width, inputs=[tex, output, width, 0.0], device=device)
    np.testing.assert_allclose(output.numpy(), data, atol=1e-5)


def test_texture3d_mipmap(test, device):
    size = 8
    data = np.full((size, size, size), fill_value=0.25, dtype=np.float32)

    tex = wp.Texture3D(
        data=data,
        num_mip_levels=3,
        mip_filter_mode=wp.TextureFilterMode.LINEAR,
        device=device,
    )

    test.assertTrue(tex.is_mipmapped)
    test.assertEqual(tex.num_mip_levels, 3)

    output = wp.zeros((size, size, size), dtype=float, device=device)
    wp.launch(
        sample_texture3d_mipmap,
        dim=(size, size, size),
        inputs=[tex, output, size, size, size, 1.5],
        device=device,
    )
    np.testing.assert_allclose(output.numpy(), np.full_like(data, 0.25), atol=1e-3)


def test_texture2d_mipmap_rejects_copy(test, device):
    data = np.zeros((8, 8), dtype=np.float32)
    tex = wp.Texture2D(data=data, num_mip_levels=2, device=device)
    with test.assertRaises(RuntimeError):
        tex.copy_from(data)
    with test.assertRaises(RuntimeError):
        tex.copy_to(np.zeros_like(data))


def test_texture_mipmap_invalid_levels(test, device):
    data = np.zeros((8, 8), dtype=np.float32)
    # 5 levels for an 8x8 texture is the maximum, 6 should fail.
    with test.assertRaises(ValueError):
        wp.Texture2D(data=data, num_mip_levels=6, device=device)
    with test.assertRaises(ValueError):
        wp.Texture2D(data=data, num_mip_levels=-1, device=device)


# ============================================================================
# Adjoint (Gradient) Tests
# ============================================================================
#
# texture_sample gradients are checked against central finite differences of
# Warp's own forward sampler. The forward is piecewise multilinear, so a
# finite difference whose stencil stays inside one interpolation cell is the
# exact derivative; all sample points below are chosen accordingly (cells span
# [n, n+1) in texel space t = u_texel - 0.5). Checks run under every address
# mode so that boundary behavior (CLAMP's zero slope, WRAP's periodicity,
# MIRROR's sign flip, BORDER's zero padding) is exercised on both backends.

_GRAD_ADDRESS_MODES = (
    wp.TextureAddressMode.WRAP,
    wp.TextureAddressMode.CLAMP,
    wp.TextureAddressMode.MIRROR,
    wp.TextureAddressMode.BORDER,
)

# Tolerance for adjoint vs. finite differences of the forward pass. The
# adjoint reconstructs the exact lerp slope, but the CUDA forward interpolates
# with 9-bit fixed-point fractions, so the finite difference itself carries
# O(1/256 / h) noise. In the 2D/3D tests the sample points are quarter-aligned
# (texel fractions that are multiples of 0.25) so that every interpolation
# weight, including the off-axis ones, is exact in the hardware fixed-point
# format; a misaligned off-axis weight couples through the cell's mixed
# derivative and the half-texel stencil amplifies it to percent-level noise.
_GRAD_FD_TOL = 2e-2

# Mipmapped sampling stacks two levels of interpolation, so the finite
# difference carries more quantization noise than the single-level case.
_GRAD_MIP_FD_TOL = 3e-2


@wp.kernel
def sample_grad_1d_f(tex: wp.Texture1D, pos: wp.array[float], out: wp.array[float]):
    tid = wp.tid()
    out[tid] = wp.texture_sample(tex, pos[tid], dtype=float)


@wp.kernel
def sample_grad_1d_v2(tex: wp.Texture1D, pos: wp.array[float], out: wp.array[wp.vec2f]):
    tid = wp.tid()
    out[tid] = wp.texture_sample(tex, pos[tid], dtype=wp.vec2f)


@wp.kernel
def sample_grad_1d_v4(tex: wp.Texture1D, pos: wp.array[float], out: wp.array[wp.vec4f]):
    tid = wp.tid()
    out[tid] = wp.texture_sample(tex, pos[tid], dtype=wp.vec4f)


@wp.kernel
def sample_grad_1d_lod_f(tex: wp.Texture1D, pos: wp.array[float], lod: wp.array[float], out: wp.array[float]):
    tid = wp.tid()
    out[tid] = wp.texture_sample(tex, pos[tid], dtype=float, lod=lod[tid])


@wp.kernel
def sample_grad_2d_f(tex: wp.Texture2D, pos: wp.array[wp.vec2f], out: wp.array[float]):
    tid = wp.tid()
    out[tid] = wp.texture_sample(tex, pos[tid], dtype=float)


@wp.kernel
def sample_grad_2d_scalar_f(tex: wp.Texture2D, pos_u: wp.array[float], pos_v: wp.array[float], out: wp.array[float]):
    tid = wp.tid()
    out[tid] = wp.texture_sample(tex, pos_u[tid], pos_v[tid], dtype=float)


@wp.kernel
def sample_grad_3d_f(tex: wp.Texture3D, pos: wp.array[wp.vec3f], out: wp.array[float]):
    tid = wp.tid()
    out[tid] = wp.texture_sample(tex, pos[tid], dtype=float)


@wp.kernel
def sample_grad_3d_scalar_f(
    tex: wp.Texture3D,
    pos_u: wp.array[float],
    pos_v: wp.array[float],
    pos_w: wp.array[float],
    out: wp.array[float],
):
    tid = wp.tid()
    out[tid] = wp.texture_sample(tex, pos_u[tid], pos_v[tid], pos_w[tid], dtype=float)


@wp.struct
class TextureGradSubject:
    tex: wp.Texture3D


@wp.kernel
def sample_grad_struct_3d_f(subject: TextureGradSubject, pos: wp.array[wp.vec3f], out: wp.array[float]):
    tid = wp.tid()
    out[tid] = wp.texture_sample(subject.tex, pos[tid], dtype=float)


def _forward_values(kernel, tex, coord_arrays, out_dtype, device):
    """Run a sampling kernel and return the outputs as a NumPy array."""
    n = len(coord_arrays[0])
    inputs = [tex] + [wp.array(np.asarray(c, dtype=np.float32), dtype=float, device=device) for c in coord_arrays]
    out = wp.zeros(n, dtype=out_dtype, device=device)
    wp.launch(kernel, dim=n, inputs=inputs, outputs=[out], device=device)
    return out.numpy()


def _tape_gradients(kernel, tex, coord_arrays, out_dtype, device):
    """Backpropagate ones through a sampling kernel; return per-input gradients."""
    n = len(coord_arrays[0])
    pos = [
        wp.array(np.asarray(c, dtype=np.float32), dtype=float, requires_grad=True, device=device) for c in coord_arrays
    ]
    out = wp.zeros(n, dtype=out_dtype, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(kernel, dim=n, inputs=[tex, *pos], outputs=[out], device=device)
    if out_dtype is float:
        out.grad = wp.full(n, 1.0, dtype=float, device=device)
    else:
        out.grad = wp.array(np.ones((n, out_dtype._length_), dtype=np.float32), dtype=out_dtype, device=device)
    tape.backward()
    return [p.grad.numpy() for p in pos]


def _check_grad_1d(test, tex, u, device, h=0.25, tol=_GRAD_FD_TOL):
    f = _forward_values(sample_grad_1d_f, tex, [[u - h, u + h]], float, device)
    fd = (f[1] - f[0]) / (2.0 * h)
    (g,) = _tape_gradients(sample_grad_1d_f, tex, [[u]], float, device)
    np.testing.assert_allclose(g[0], fd, rtol=tol, atol=tol, err_msg=f"1D gradient mismatch at u={u}")


def test_texture_sample_grad_1d_address_modes(test, device):
    """Adjoint matches finite differences under every address mode, both coordinate conventions."""
    width = 8
    rng = np.random.default_rng(0)
    data = rng.standard_normal(width).astype(np.float32)

    # Texel-space points: interior, both half-texel boundary bands, and cells
    # beyond the texture (wraparound / mirrored / clamped / border regions).
    points_texel = (0.2, 3.25, 4.8, 7.75, 8.2, -0.25)

    for address_mode in _GRAD_ADDRESS_MODES:
        for normalized in (False, True):
            tex = wp.Texture1D(
                data,
                normalized_coords=normalized,
                filter_mode=wp.TextureFilterMode.LINEAR,
                address_mode=address_mode,
                device=device,
            )
            for u_texel in points_texel:
                scale = 1.0 / width if normalized else 1.0
                _check_grad_1d(test, tex, u_texel * scale, device, h=0.25 * scale)


def test_texture_sample_grad_1d_analytic(test, device):
    """Hand-computed boundary gradients for each address mode.

    Data is ``v[i] = i + 1`` so interior cells have slope 1 (in texel units).
    Expectations match PyTorch ``grid_sample``: ``padding_mode="zeros"`` for
    BORDER and ``padding_mode="border"`` for CLAMP. WRAP/MIRROR run with
    normalized coordinates because CUDA treats them as CLAMP otherwise.
    """
    width = 8
    data = np.arange(1, width + 1, dtype=np.float32)

    def grad_at(tex, u):
        (g,) = _tape_gradients(sample_grad_1d_f, tex, [[u]], float, device)
        return g[0]

    def make_tex(address_mode, normalized):
        return wp.Texture1D(
            data,
            normalized_coords=normalized,
            filter_mode=wp.TextureFilterMode.LINEAR,
            address_mode=address_mode,
            device=device,
        )

    # (address_mode, normalized, u_texel, expected gradient in texel units)
    cases = (
        # interior cell [2, 3]: slope v[3] - v[2] = 1
        (wp.TextureAddressMode.BORDER, False, 3.25, 1.0),
        (wp.TextureAddressMode.CLAMP, False, 3.25, 1.0),
        (wp.TextureAddressMode.WRAP, True, 3.25, 1.0),
        (wp.TextureAddressMode.MIRROR, True, 3.25, 1.0),
        # left boundary band, cell [-1, 0]: BORDER blends with zero padding
        (wp.TextureAddressMode.BORDER, False, 0.25, data[0]),
        (wp.TextureAddressMode.CLAMP, False, 0.25, 0.0),
        # right boundary band, cell [7, 8]
        (wp.TextureAddressMode.BORDER, False, 7.75, -data[width - 1]),
        (wp.TextureAddressMode.CLAMP, False, 7.75, 0.0),
        (wp.TextureAddressMode.WRAP, True, 7.75, data[0] - data[width - 1]),
        (wp.TextureAddressMode.MIRROR, True, 7.75, 0.0),  # x1 = 8 mirrors back to 7: flat cell
        # fully outside, cell [-2, -1]: BORDER is identically zero
        (wp.TextureAddressMode.BORDER, False, -0.75, 0.0),
        # mirrored descending segment, cell [8, 9]: v[map(9)] - v[map(8)] = v[6] - v[7]
        (wp.TextureAddressMode.MIRROR, True, 8.75, data[6] - data[7]),
        # wrapped cell [8, 9] repeats cell [0, 1]
        (wp.TextureAddressMode.WRAP, True, 8.75, data[1] - data[0]),
    )

    for address_mode, normalized, u_texel, expected_texel in cases:
        tex = make_tex(address_mode, normalized)
        scale = 1.0 / width if normalized else 1.0
        g = grad_at(tex, u_texel * scale)
        np.testing.assert_allclose(
            g * scale,
            expected_texel,
            atol=1e-4,
            err_msg=f"mode={address_mode.name} normalized={normalized} u_texel={u_texel}",
        )


def test_texture_sample_grad_closest_zero(test, device):
    """CLOSEST filtering is piecewise constant: coordinate gradients are zero."""
    data = np.arange(1, 9, dtype=np.float32)
    for address_mode in _GRAD_ADDRESS_MODES:
        tex = wp.Texture1D(
            data,
            normalized_coords=False,
            filter_mode=wp.TextureFilterMode.CLOSEST,
            address_mode=address_mode,
            device=device,
        )
        (g,) = _tape_gradients(sample_grad_1d_f, tex, [[3.3]], float, device)
        test.assertEqual(g[0], 0.0)


def test_texture_sample_grad_1d_multichannel(test, device):
    """vec2f/vec4f samples: the coordinate gradient sums per-channel slopes."""
    width = 8
    rng = np.random.default_rng(1)
    for num_channels, out_dtype, kernel in ((2, wp.vec2f, sample_grad_1d_v2), (4, wp.vec4f, sample_grad_1d_v4)):
        data = rng.standard_normal((width, num_channels)).astype(np.float32)
        tex = wp.Texture1D(
            data,
            normalized_coords=False,
            filter_mode=wp.TextureFilterMode.LINEAR,
            address_mode=wp.TextureAddressMode.CLAMP,
            device=device,
        )
        for u in (0.2, 3.25, 6.8):
            h = 0.25
            f = _forward_values(kernel, tex, [[u - h, u + h]], out_dtype, device)
            fd = (f[1] - f[0]) / (2.0 * h)  # per-channel slopes
            (g,) = _tape_gradients(kernel, tex, [[u]], out_dtype, device)
            np.testing.assert_allclose(g[0], fd.sum(), rtol=_GRAD_FD_TOL, atol=_GRAD_FD_TOL)


def test_texture_sample_grad_2d(test, device):
    """2D vec2 and scalar-coordinate overloads match finite differences per axis."""
    height, width = 6, 8
    rng = np.random.default_rng(2)
    data = rng.standard_normal((height, width)).astype(np.float32)

    points = ((3.25, 2.75), (0.25, 0.25), (7.75, 5.75), (-0.25, 4.25))

    for address_mode in _GRAD_ADDRESS_MODES:
        # WRAP and MIRROR run with normalized coordinates because CUDA treats
        # them as CLAMP otherwise.
        normalized = address_mode in (wp.TextureAddressMode.WRAP, wp.TextureAddressMode.MIRROR)
        su = 1.0 / width if normalized else 1.0
        sv = 1.0 / height if normalized else 1.0
        tex = wp.Texture2D(
            data,
            normalized_coords=normalized,
            filter_mode=wp.TextureFilterMode.LINEAR,
            address_mode=address_mode,
            device=device,
        )
        for u_texel, v_texel in points:
            u, v = u_texel * su, v_texel * sv
            hu, hv = 0.25 * su, 0.25 * sv
            fu = _forward_values(sample_grad_2d_scalar_f, tex, [[u - hu, u + hu], [v, v]], float, device)
            fv = _forward_values(sample_grad_2d_scalar_f, tex, [[u, u], [v - hv, v + hv]], float, device)
            fd = ((fu[1] - fu[0]) / (2.0 * hu), (fv[1] - fv[0]) / (2.0 * hv))

            # scalar-coordinate overload
            gu, gv = _tape_gradients(sample_grad_2d_scalar_f, tex, [[u], [v]], float, device)
            np.testing.assert_allclose(
                (gu[0], gv[0]),
                fd,
                rtol=_GRAD_FD_TOL,
                atol=_GRAD_FD_TOL,
                err_msg=f"mode={address_mode.name} u_texel={u_texel} v_texel={v_texel}",
            )

            # vec2 overload
            pos = wp.array([wp.vec2f(u, v)], dtype=wp.vec2f, requires_grad=True, device=device)
            out = wp.zeros(1, dtype=float, requires_grad=True, device=device)
            tape = wp.Tape()
            with tape:
                wp.launch(sample_grad_2d_f, dim=1, inputs=[tex, pos], outputs=[out], device=device)
            out.grad = wp.ones(1, dtype=float, device=device)
            tape.backward()
            np.testing.assert_allclose(pos.grad.numpy()[0], fd, rtol=_GRAD_FD_TOL, atol=_GRAD_FD_TOL)


def test_texture_sample_grad_3d(test, device):
    """3D vec3 and scalar-coordinate overloads match finite differences per axis."""
    depth, height, width = 4, 6, 8
    rng = np.random.default_rng(3)
    data = rng.standard_normal((depth, height, width)).astype(np.float32)

    points = ((3.25, 2.75, 1.25), (0.25, 0.25, 3.75), (7.75, 5.75, 0.75))

    for address_mode in _GRAD_ADDRESS_MODES:
        # WRAP and MIRROR run with normalized coordinates because CUDA treats
        # them as CLAMP otherwise.
        normalized = address_mode in (wp.TextureAddressMode.WRAP, wp.TextureAddressMode.MIRROR)
        scales = (1.0 / width, 1.0 / height, 1.0 / depth) if normalized else (1.0, 1.0, 1.0)
        tex = wp.Texture3D(
            data,
            normalized_coords=normalized,
            filter_mode=wp.TextureFilterMode.LINEAR,
            address_mode=address_mode,
            device=device,
        )
        for point_texel in points:
            coords = tuple(c * s for c, s in zip(point_texel, scales, strict=True))
            u, v, w = coords
            fd = []
            for axis in range(3):
                h = 0.25 * scales[axis]
                lo = list(coords)
                hi = list(coords)
                lo[axis] -= h
                hi[axis] += h
                f = _forward_values(
                    sample_grad_3d_scalar_f,
                    tex,
                    [[lo[0], hi[0]], [lo[1], hi[1]], [lo[2], hi[2]]],
                    float,
                    device,
                )
                fd.append((f[1] - f[0]) / (2.0 * h))

            gu, gv, gw = _tape_gradients(sample_grad_3d_scalar_f, tex, [[u], [v], [w]], float, device)
            np.testing.assert_allclose(
                (gu[0], gv[0], gw[0]),
                fd,
                rtol=_GRAD_FD_TOL,
                atol=_GRAD_FD_TOL,
                err_msg=f"mode={address_mode.name} point_texel={point_texel}",
            )

            pos = wp.array([wp.vec3f(u, v, w)], dtype=wp.vec3f, requires_grad=True, device=device)
            out = wp.zeros(1, dtype=float, requires_grad=True, device=device)
            tape = wp.Tape()
            with tape:
                wp.launch(sample_grad_3d_f, dim=1, inputs=[tex, pos], outputs=[out], device=device)
            out.grad = wp.ones(1, dtype=float, device=device)
            tape.backward()
            np.testing.assert_allclose(pos.grad.numpy()[0], fd, rtol=_GRAD_FD_TOL, atol=_GRAD_FD_TOL)


def test_texture_sample_grad_lod(test, device):
    """Mipmapped gradients: per-level coordinate slopes and the LOD gradient.

    Sample points are chosen so the finite-difference stencil (h = 0.125 base
    texels) stays inside one interpolation cell at every mip level.
    """
    width = 16
    rng = np.random.default_rng(4)
    data = rng.standard_normal(width).astype(np.float32)

    points_texel = (3.3, 5.2, 10.2)
    lods = (0.0, 1.0, 2.0, 0.5, 1.25, 1.75)

    for mip_filter_mode in (wp.TextureFilterMode.LINEAR, wp.TextureFilterMode.CLOSEST):
        tex = wp.Texture1D(
            data,
            normalized_coords=True,
            filter_mode=wp.TextureFilterMode.LINEAR,
            mip_filter_mode=mip_filter_mode,
            address_mode=wp.TextureAddressMode.CLAMP,
            num_mip_levels=3,  # widths 16, 8, 4
            device=device,
        )
        for lod in lods:
            for u_texel in points_texel:
                u = u_texel / width
                h = 0.125 / width

                f = _forward_values(sample_grad_1d_lod_f, tex, [[u - h, u + h], [lod, lod]], float, device)
                fd_u = (f[1] - f[0]) / (2.0 * h)
                gu, glod = _tape_gradients(sample_grad_1d_lod_f, tex, [[u], [lod]], float, device)
                np.testing.assert_allclose(
                    gu[0],
                    fd_u,
                    rtol=_GRAD_MIP_FD_TOL,
                    atol=_GRAD_MIP_FD_TOL,
                    err_msg=f"mip={mip_filter_mode.name} lod={lod} u_texel={u_texel}",
                )

                if mip_filter_mode == wp.TextureFilterMode.LINEAR:
                    # At an integer LOD the mip blend has a kink, so the adjoint
                    # returns the right-hand derivative; a central difference
                    # there is not a valid reference.
                    if lod % 1.0 != 0.0:
                        hl = 0.125
                        f = _forward_values(sample_grad_1d_lod_f, tex, [[u, u], [lod - hl, lod + hl]], float, device)
                        fd_lod = (f[1] - f[0]) / (2.0 * hl)
                        np.testing.assert_allclose(
                            glod[0],
                            fd_lod,
                            rtol=_GRAD_MIP_FD_TOL,
                            atol=_GRAD_MIP_FD_TOL,
                            err_msg=f"lod gradient at lod={lod} u_texel={u_texel}",
                        )
                else:
                    # CLOSEST mip filter: piecewise constant in lod
                    test.assertEqual(glod[0], 0.0, f"lod gradient should be zero at lod={lod}")


def test_texture_sample_grad_struct_member(test, device):
    """Gradients flow through a texture held inside a wp.struct (interop pattern)."""
    depth, height, width = 4, 4, 4
    rng = np.random.default_rng(5)
    data = rng.standard_normal((depth, height, width)).astype(np.float32)

    subject = TextureGradSubject()
    subject.tex = wp.Texture3D(
        data,
        normalized_coords=False,
        filter_mode=wp.TextureFilterMode.LINEAR,
        address_mode=wp.TextureAddressMode.BORDER,
        device=device,
    )

    u, v, w = 1.25, 2.2, 0.8
    h = 0.25
    fd = []
    for axis in range(3):
        lo = [u, v, w]
        hi = [u, v, w]
        lo[axis] -= h
        hi[axis] += h
        pos = wp.array([wp.vec3f(*lo), wp.vec3f(*hi)], dtype=wp.vec3f, device=device)
        out = wp.zeros(2, dtype=float, device=device)
        wp.launch(sample_grad_struct_3d_f, dim=2, inputs=[subject, pos], outputs=[out], device=device)
        f = out.numpy()
        fd.append((f[1] - f[0]) / (2.0 * h))

    pos = wp.array([wp.vec3f(u, v, w)], dtype=wp.vec3f, requires_grad=True, device=device)
    out = wp.zeros(1, dtype=float, requires_grad=True, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(sample_grad_struct_3d_f, dim=1, inputs=[subject, pos], outputs=[out], device=device)
    out.grad = wp.ones(1, dtype=float, device=device)
    tape.backward()
    np.testing.assert_allclose(pos.grad.numpy()[0], fd, rtol=_GRAD_FD_TOL, atol=_GRAD_FD_TOL)


# ============================================================================
# Test Class
# ============================================================================


class TestTexture(unittest.TestCase):
    pass


# Register tests - textures work on both CPU and CUDA devices
cuda_devices = get_selected_cuda_test_devices()
all_devices = get_test_devices()

# Core texture tests - run on all devices (CPU + CUDA)
add_function_test(TestTexture, "test_texture1d_1channel", test_texture1d_1channel, devices=all_devices)
add_function_test(TestTexture, "test_texture1d_2channel", test_texture1d_2channel, devices=all_devices)
add_function_test(TestTexture, "test_texture1d_4channel", test_texture1d_4channel, devices=all_devices)
add_function_test(TestTexture, "test_texture1d_linear_filter", test_texture1d_linear_filter, devices=all_devices)
add_function_test(TestTexture, "test_texture1d_resolution_query", test_texture1d_resolution_query, devices=all_devices)
add_function_test(TestTexture, "test_texture1d_new_del", test_texture1d_new_del, devices=all_devices)
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
add_function_test(
    TestTexture, "test_texture_dtype_prefers_warp_types", test_texture_dtype_prefers_warp_types, devices=all_devices
)
add_function_test(
    TestTexture,
    "test_texture_dtype_float_alias_maps_to_float32",
    test_texture_dtype_float_alias_maps_to_float32,
    devices=all_devices,
)
add_function_test(
    TestTexture,
    "test_texture_dtype_int_alias_maps_to_int32",
    test_texture_dtype_int_alias_maps_to_int32,
    devices=all_devices,
)
add_function_test(
    TestTexture,
    "test_texture2d_constructor_from_same_device_array",
    test_texture2d_constructor_from_same_device_array,
    devices=cuda_devices,
)
add_function_test(
    TestTexture,
    "test_texture2d_constructor_transfers_cross_device",
    test_texture2d_constructor_transfers_cross_device,
    devices=cuda_devices,
)
add_function_test(
    TestTexture, "test_texture2d_cuda_interop_handles", test_texture2d_cuda_interop_handles, devices=cuda_devices
)
add_function_test(
    TestTexture,
    "test_texture2d_cuda_array_wraps_non_mipmapped",
    test_texture2d_cuda_array_wraps_non_mipmapped,
    devices=cuda_devices,
)
add_function_test(
    TestTexture,
    "test_texture2d_mipmapped_cuda_array_current_limitation",
    test_texture2d_mipmapped_cuda_array_current_limitation,
    devices=cuda_devices,
)
add_function_test(
    TestTexture, "test_texture3d_cuda_interop_handles", test_texture3d_cuda_interop_handles, devices=cuda_devices
)
add_function_test(
    TestTexture, "test_texture_id_device_independent", test_texture_id_device_independent, devices=all_devices
)
add_function_test(
    TestTexture,
    "test_texture_handle_properties_host_vs_cuda",
    test_texture_handle_properties_host_vs_cuda,
    devices=all_devices,
)
add_function_test(
    TestTexture, "test_texture2d_cuda_array_copy_api", test_texture2d_cuda_array_copy_api, devices=cuda_devices
)
add_function_test(
    TestTexture, "test_texture3d_cuda_array_copy_api", test_texture3d_cuda_array_copy_api, devices=cuda_devices
)
add_function_test(
    TestTexture, "test_texture_copy_validation_messages", test_texture_copy_validation_messages, devices=all_devices
)
add_function_test(
    TestTexture,
    "test_texture2d_cuda_array_copy_api_rejects_indexedarray",
    test_texture2d_cuda_array_copy_api_rejects_indexedarray,
    devices=cuda_devices,
)
add_function_test(
    TestTexture,
    "test_texture3d_cuda_array_copy_api_rejects_indexedarray",
    test_texture3d_cuda_array_copy_api_rejects_indexedarray,
    devices=cuda_devices,
)
add_function_test(
    TestTexture,
    "test_texture2d_cuda_array_copy_api_graph_capture",
    test_texture2d_cuda_array_copy_api_graph_capture,
    devices=cuda_devices,
)
add_function_test(
    TestTexture,
    "test_texture3d_cuda_array_copy_api_graph_capture",
    test_texture3d_cuda_array_copy_api_graph_capture,
    devices=cuda_devices,
)
add_function_test(
    TestTexture,
    "test_texture2d_cuda_array_copy_api_graph_capture_explicit_stream",
    test_texture2d_cuda_array_copy_api_graph_capture_explicit_stream,
    devices=cuda_devices,
)
add_function_test(
    TestTexture,
    "test_texture3d_cuda_array_copy_api_graph_capture_explicit_stream",
    test_texture3d_cuda_array_copy_api_graph_capture_explicit_stream,
    devices=cuda_devices,
)
add_function_test(
    TestTexture,
    "test_texture2d_cuda_surface_property_graph_capture_stability",
    test_texture2d_cuda_surface_property_graph_capture_stability,
    devices=cuda_devices,
)
add_function_test(
    TestTexture,
    "test_texture2d_cuda_surface_property_api",
    test_texture2d_cuda_surface_property_api,
    devices=cuda_devices,
)
add_function_test(
    TestTexture,
    "test_texture3d_cuda_surface_property_api",
    test_texture3d_cuda_surface_property_api,
    devices=cuda_devices,
)
add_function_test(
    TestTexture,
    "test_texture2d_cuda_surface_property_requires_surface_access",
    test_texture2d_cuda_surface_property_requires_surface_access,
    devices=cuda_devices,
)
add_function_test(
    TestTexture,
    "test_texture3d_cuda_surface_property_requires_surface_access",
    test_texture3d_cuda_surface_property_requires_surface_access,
    devices=cuda_devices,
)

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

# Mipmap tests - run on all devices
add_function_test(
    TestTexture,
    "test_texture2d_mipmap_full_chain",
    test_texture2d_mipmap_full_chain,
    devices=all_devices,
)
add_function_test(
    TestTexture,
    "test_texture2d_mipmap_lod_selects_constant_level",
    test_texture2d_mipmap_lod_selects_constant_level,
    devices=all_devices,
)
add_function_test(
    TestTexture,
    "test_texture1d_mipmap",
    test_texture1d_mipmap,
    devices=all_devices,
)
add_function_test(
    TestTexture,
    "test_texture3d_mipmap",
    test_texture3d_mipmap,
    devices=all_devices,
)
add_function_test(
    TestTexture,
    "test_texture2d_mipmap_rejects_copy",
    test_texture2d_mipmap_rejects_copy,
    devices=all_devices,
)
add_function_test(
    TestTexture,
    "test_texture_mipmap_invalid_levels",
    test_texture_mipmap_invalid_levels,
    devices=all_devices,
)

# Adjoint (gradient) tests - run on all devices
add_function_test(
    TestTexture,
    "test_texture_sample_grad_1d_address_modes",
    test_texture_sample_grad_1d_address_modes,
    devices=all_devices,
)
add_function_test(
    TestTexture, "test_texture_sample_grad_1d_analytic", test_texture_sample_grad_1d_analytic, devices=all_devices
)
add_function_test(
    TestTexture, "test_texture_sample_grad_closest_zero", test_texture_sample_grad_closest_zero, devices=all_devices
)
add_function_test(
    TestTexture,
    "test_texture_sample_grad_1d_multichannel",
    test_texture_sample_grad_1d_multichannel,
    devices=all_devices,
)
add_function_test(TestTexture, "test_texture_sample_grad_2d", test_texture_sample_grad_2d, devices=all_devices)
add_function_test(TestTexture, "test_texture_sample_grad_3d", test_texture_sample_grad_3d, devices=all_devices)
add_function_test(TestTexture, "test_texture_sample_grad_lod", test_texture_sample_grad_lod, devices=all_devices)
add_function_test(
    TestTexture,
    "test_texture_sample_grad_struct_member",
    test_texture_sample_grad_struct_member,
    devices=all_devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
