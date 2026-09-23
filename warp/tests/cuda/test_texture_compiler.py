# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for CUDA mixed-width texture sampling.

CUDA can miscompile a function that selects between scalar and vector texture
fetches at runtime when targeting ``sm_89`` and older architectures. A selected
fetch can return zero even though the other branches are never executed. Direct
texture-sampling kernels do not reproduce the bug. The kernels below retain the
smallest known source shapes that do.

The 1D, 2D, and 3D kernels are deliberately explicit. Factoring their shared
code into another function changes the generated CUDA call graph and can hide
the compiler regression this module is meant to detect. The all-width case is
smaller: it only requires a texture loaded from an array, a runtime selector,
and two calls to the same sampling function.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices


@wp.struct
class TextureSampleData1D:
    texture: wp.Texture1D
    kind: int
    component: int


@wp.struct
class TextureSampleData2D:
    texture: wp.Texture2D
    kind: int
    component: int


@wp.struct
class TextureSampleData3D:
    texture: wp.Texture3D
    kind: int
    component: int


@wp.func
def sample_all_widths_1d(texture: wp.Texture1D, kind: int) -> float:
    if kind == 1:
        return wp.texture_sample(texture, 0.5, dtype=wp.vec2f)[kind]
    if kind == 2:
        return wp.texture_sample(texture, 0.5, dtype=wp.vec4f)[kind]
    return wp.texture_sample(texture, 0.5, dtype=float)


@wp.func
def sample_scalar_or_vec2_1d(texture: wp.Texture1D, u: float, kind: int, component: int) -> float:
    if kind == 1:
        return wp.texture_sample(texture, u, dtype=wp.vec2f)[component]
    return wp.texture_sample(texture, u, dtype=float)


@wp.func
def sample_scalar_or_vec2_2d(texture: wp.Texture2D, uv: wp.vec2f, kind: int, component: int) -> float:
    if kind == 1:
        return wp.texture_sample(texture, uv, dtype=wp.vec2f)[component]
    return wp.texture_sample(texture, uv, dtype=float)


@wp.func
def sample_scalar_or_vec2_3d(texture: wp.Texture3D, uvw: wp.vec3f, kind: int, component: int) -> float:
    if kind == 1:
        return wp.texture_sample(texture, uvw, dtype=wp.vec2f)[component]
    return wp.texture_sample(texture, uvw, dtype=float)


@wp.func
def sample_scalar_or_vec4_1d(texture: wp.Texture1D, u: float, kind: int, component: int) -> float:
    if kind == 1:
        return wp.texture_sample(texture, u, dtype=wp.vec4f)[component]
    return wp.texture_sample(texture, u, dtype=float)


@wp.func
def sample_scalar_or_vec4_2d(texture: wp.Texture2D, uv: wp.vec2f, kind: int, component: int) -> float:
    if kind == 1:
        return wp.texture_sample(texture, uv, dtype=wp.vec4f)[component]
    return wp.texture_sample(texture, uv, dtype=float)


@wp.func
def sample_scalar_or_vec4_3d(texture: wp.Texture3D, uvw: wp.vec3f, kind: int, component: int) -> float:
    if kind == 1:
        return wp.texture_sample(texture, uvw, dtype=wp.vec4f)[component]
    return wp.texture_sample(texture, uvw, dtype=float)


def sample_scalar_vec2_1d_kernel(
    scale: wp.vec3f,
    table: wp.array[TextureSampleData1D],
    out_value: wp.array[float],
):
    """Sample a runtime-selected scalar or ``vec2`` value from a 1D texture."""
    data = table[0]
    local = wp.cw_div(wp.vec3f(1.0, 0.0, 0.0), scale)
    clamped = wp.vec3f(
        wp.clamp(local[0], -0.6, 0.6),
        wp.clamp(local[1], -0.6, 0.6),
        wp.clamp(local[2], -0.6, 0.6),
    )
    diff = local - clamped
    f = (clamped[0] + 0.6) * 6.6666665
    bx = wp.clamp(int(wp.floor(f)), 0, 7)
    tx = f - float(bx)
    v0 = sample_scalar_or_vec2_1d(data.texture, float(bx) + 0.5, data.kind, data.component)
    v1 = sample_scalar_or_vec2_1d(data.texture, float(bx) + 1.5, data.kind, data.component)
    value = v0 + (v1 - v0) * tx
    out_value[0] = value + wp.length(diff)


def sample_scalar_vec2_2d_kernel(
    scale: wp.vec3f,
    table: wp.array[TextureSampleData2D],
    out_value: wp.array[float],
):
    """Sample a runtime-selected scalar or ``vec2`` value from a 2D texture."""
    data = table[0]
    local = wp.cw_div(wp.vec3f(1.0, 0.0, 0.0), scale)
    clamped = wp.vec3f(
        wp.clamp(local[0], -0.6, 0.6),
        wp.clamp(local[1], -0.6, 0.6),
        wp.clamp(local[2], -0.6, 0.6),
    )
    diff = local - clamped
    f = (clamped[0] + 0.6) * 6.6666665
    bx = wp.clamp(int(wp.floor(f)), 0, 7)
    tx = f - float(bx)
    v0 = sample_scalar_or_vec2_2d(
        data.texture,
        wp.vec2f(float(bx) + 0.5, 4.5),
        data.kind,
        data.component,
    )
    v1 = sample_scalar_or_vec2_2d(
        data.texture,
        wp.vec2f(float(bx) + 1.5, 4.5),
        data.kind,
        data.component,
    )
    value = v0 + (v1 - v0) * tx
    out_value[0] = value + wp.length(diff)


def sample_scalar_vec2_3d_kernel(
    scale: wp.vec3f,
    table: wp.array[TextureSampleData3D],
    out_value: wp.array[float],
):
    """Sample a runtime-selected scalar or ``vec2`` value from a 3D texture."""
    data = table[0]
    local = wp.cw_div(wp.vec3f(1.0, 0.0, 0.0), scale)
    clamped = wp.vec3f(
        wp.clamp(local[0], -0.6, 0.6),
        wp.clamp(local[1], -0.6, 0.6),
        wp.clamp(local[2], -0.6, 0.6),
    )
    diff = local - clamped
    f = (clamped[0] + 0.6) * 6.6666665
    bx = wp.clamp(int(wp.floor(f)), 0, 7)
    tx = f - float(bx)
    v0 = sample_scalar_or_vec2_3d(
        data.texture,
        wp.vec3f(float(bx) + 0.5, 4.5, 4.5),
        data.kind,
        data.component,
    )
    v1 = sample_scalar_or_vec2_3d(
        data.texture,
        wp.vec3f(float(bx) + 1.5, 4.5, 4.5),
        data.kind,
        data.component,
    )
    value = v0 + (v1 - v0) * tx
    out_value[0] = value + wp.length(diff)


def sample_scalar_vec4_1d_kernel(
    scale: wp.vec3f,
    table: wp.array[TextureSampleData1D],
    out_value: wp.array[float],
):
    """Sample a runtime-selected scalar or ``vec4`` value from a 1D texture."""
    data = table[0]
    local = wp.cw_div(wp.vec3f(1.0, 0.0, 0.0), scale)
    clamped = wp.vec3f(
        wp.clamp(local[0], -0.6, 0.6),
        wp.clamp(local[1], -0.6, 0.6),
        wp.clamp(local[2], -0.6, 0.6),
    )
    diff = local - clamped
    f = (clamped[0] + 0.6) * 6.6666665
    bx = wp.clamp(int(wp.floor(f)), 0, 7)
    tx = f - float(bx)
    v0 = sample_scalar_or_vec4_1d(data.texture, float(bx) + 0.5, data.kind, data.component)
    v1 = sample_scalar_or_vec4_1d(data.texture, float(bx) + 1.5, data.kind, data.component)
    value = v0 + (v1 - v0) * tx
    out_value[0] = value + wp.length(diff)


def sample_scalar_vec4_2d_kernel(
    scale: wp.vec3f,
    table: wp.array[TextureSampleData2D],
    out_value: wp.array[float],
):
    """Sample a runtime-selected scalar or ``vec4`` value from a 2D texture."""
    data = table[0]
    local = wp.cw_div(wp.vec3f(1.0, 0.0, 0.0), scale)
    clamped = wp.vec3f(
        wp.clamp(local[0], -0.6, 0.6),
        wp.clamp(local[1], -0.6, 0.6),
        wp.clamp(local[2], -0.6, 0.6),
    )
    diff = local - clamped
    f = (clamped[0] + 0.6) * 6.6666665
    bx = wp.clamp(int(wp.floor(f)), 0, 7)
    tx = f - float(bx)
    v0 = sample_scalar_or_vec4_2d(
        data.texture,
        wp.vec2f(float(bx) + 0.5, 4.5),
        data.kind,
        data.component,
    )
    v1 = sample_scalar_or_vec4_2d(
        data.texture,
        wp.vec2f(float(bx) + 1.5, 4.5),
        data.kind,
        data.component,
    )
    value = v0 + (v1 - v0) * tx
    out_value[0] = value + wp.length(diff)


def sample_scalar_vec4_3d_kernel(
    scale: wp.vec3f,
    table: wp.array[TextureSampleData3D],
    out_value: wp.array[float],
):
    """Sample a runtime-selected scalar or ``vec4`` value from a 3D texture."""
    data = table[0]
    local = wp.cw_div(wp.vec3f(1.0, 0.0, 0.0), scale)
    clamped = wp.vec3f(
        wp.clamp(local[0], -0.6, 0.6),
        wp.clamp(local[1], -0.6, 0.6),
        wp.clamp(local[2], -0.6, 0.6),
    )
    diff = local - clamped
    f = (clamped[0] + 0.6) * 6.6666665
    bx = wp.clamp(int(wp.floor(f)), 0, 7)
    tx = f - float(bx)
    v0 = sample_scalar_or_vec4_3d(
        data.texture,
        wp.vec3f(float(bx) + 0.5, 4.5, 4.5),
        data.kind,
        data.component,
    )
    v1 = sample_scalar_or_vec4_3d(
        data.texture,
        wp.vec3f(float(bx) + 1.5, 4.5, 4.5),
        data.kind,
        data.component,
    )
    value = v0 + (v1 - v0) * tx
    out_value[0] = value + wp.length(diff)


def sample_all_widths_1d_kernel(
    textures: wp.array[wp.Texture1D],
    out_value: wp.array[float],
):
    """Sample twice from a function containing scalar, ``vec2``, and ``vec4`` fetches."""
    texture = textures[0]
    # Keep the selector and vector component dynamic without another input.
    kind = texture.width
    out_value[0] = sample_all_widths_1d(texture, kind) + sample_all_widths_1d(texture, kind)


KERNEL_FUNCTIONS = {
    ("scalar_vec2", "1d"): sample_scalar_vec2_1d_kernel,
    ("scalar_vec2", "2d"): sample_scalar_vec2_2d_kernel,
    ("scalar_vec2", "3d"): sample_scalar_vec2_3d_kernel,
    ("scalar_vec4", "1d"): sample_scalar_vec4_1d_kernel,
    ("scalar_vec4", "2d"): sample_scalar_vec4_2d_kernel,
    ("scalar_vec4", "3d"): sample_scalar_vec4_3d_kernel,
    ("all_widths", "1d"): sample_all_widths_1d_kernel,
}

COMPILER_CONFIGS = (
    ("cubin", 0),
    ("cubin", 3),
    ("ptx", 0),
    ("ptx", 3),
)

# Register each combination as a named case while compiling only one shared
# module for each compiler configuration.
TEXTURE_COMPILER_KERNELS = {}
for cuda_output, optimization_level in COMPILER_CONFIGS:
    module = wp.Module(f"test_texture_compiler_{cuda_output}_{optimization_level}")
    wp.set_module_options(
        {"cuda_output": cuda_output, "optimization_level": optimization_level},
        module=module,
    )
    for kernel_key, kernel_func in KERNEL_FUNCTIONS.items():
        TEXTURE_COMPILER_KERNELS[(cuda_output, optimization_level, *kernel_key)] = wp.kernel(
            kernel_func,
            module=module,
        )


DIMENSION_CASES = (
    ("1d", wp.Texture1D, TextureSampleData1D, (9,)),
    ("2d", wp.Texture2D, TextureSampleData2D, (9, 9)),
    ("3d", wp.Texture3D, TextureSampleData3D, (9, 9, 9)),
)

WIDTH_CASES = (
    ("scalar_from_scalar_vec2", "scalar_vec2"),
    ("scalar_from_scalar_vec4", "scalar_vec4"),
)


def make_constant_scalar_texture(texture_cls, shape, device):
    return texture_cls(
        np.full(shape, 0.1, dtype=np.float32),
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        normalized_coords=False,
        device=device,
    )


def test_mixed_width_texture_sampling(
    test,
    device,
    kernel,
    texture_cls,
    data_cls,
    shape,
):
    texture = make_constant_scalar_texture(texture_cls, shape, device)
    data = data_cls()
    data.texture = texture
    data.kind = 0
    data.component = 0
    table = wp.array([data], dtype=data_cls, device=device)
    out_value = wp.empty(1, dtype=float, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[wp.vec3f(1.0), table],
        outputs=[out_value],
        device=device,
    )

    # The sampled first component is 0.1. Clamping local x from 1.0 to 0.6
    # contributes the remaining distance of 0.4.
    np.testing.assert_allclose(
        out_value.numpy(),
        np.array([0.5], dtype=np.float32),
        rtol=0.0,
        atol=1.0e-6,
    )


def test_all_widths_texture_sampling(test, device, kernel):
    texture = wp.Texture1D(
        np.full((2, 4), 0.1, dtype=np.float32),
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        normalized_coords=False,
        device=device,
    )
    textures = wp.array([texture], dtype=wp.Texture1D, device=device)
    out_value = wp.empty(1, dtype=float, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[textures],
        outputs=[out_value],
        device=device,
    )

    # A width of two selects component two of the vec4 branch. Each of the two
    # samples contributes 0.1.
    np.testing.assert_allclose(
        out_value.numpy(),
        np.array([0.2], dtype=np.float32),
        rtol=0.0,
        atol=1.0e-6,
    )


class TestTextureCompiler(unittest.TestCase):
    pass


devices = get_selected_cuda_test_devices()
for cuda_output, optimization_level in COMPILER_CONFIGS:
    for dimension, texture_cls, data_cls, shape in DIMENSION_CASES:
        for selected_sample, width_pair in WIDTH_CASES:
            kernel = TEXTURE_COMPILER_KERNELS[(cuda_output, optimization_level, width_pair, dimension)]
            add_function_test(
                TestTextureCompiler,
                f"test_{selected_sample}_{dimension}_{cuda_output}_o{optimization_level}",
                test_mixed_width_texture_sampling,
                devices=devices,
                kernel=kernel,
                texture_cls=texture_cls,
                data_cls=data_cls,
                shape=shape,
            )

    kernel = TEXTURE_COMPILER_KERNELS[(cuda_output, optimization_level, "all_widths", "1d")]
    add_function_test(
        TestTextureCompiler,
        f"test_vec4_from_all_widths_1d_{cuda_output}_o{optimization_level}",
        test_all_widths_texture_sampling,
        devices=devices,
        kernel=kernel,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
