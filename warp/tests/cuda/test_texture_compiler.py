# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for CUDA texture compiler workarounds.

The mixed-width miscompile depends on the surrounding generated code. The
separate sampling helpers and repeated coordinate calculations below preserve
source shapes that fail without the native call boundary.
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
def sample_scalar_vec2_1d(texture: wp.Texture1D, u: float, kind: int, component: int) -> float:
    if kind == 1:
        return wp.texture_sample(texture, u, dtype=wp.vec2f)[component]
    return wp.texture_sample(texture, u, dtype=float)


@wp.func
def sample_scalar_vec2_2d(texture: wp.Texture2D, uv: wp.vec2f, kind: int, component: int) -> float:
    if kind == 1:
        return wp.texture_sample(texture, uv, dtype=wp.vec2f)[component]
    return wp.texture_sample(texture, uv, dtype=float)


@wp.func
def sample_scalar_vec2_3d(texture: wp.Texture3D, uvw: wp.vec3f, kind: int, component: int) -> float:
    if kind == 1:
        return wp.texture_sample(texture, uvw, dtype=wp.vec2f)[component]
    return wp.texture_sample(texture, uvw, dtype=float)


@wp.func
def sample_all_widths_1d(texture: wp.Texture1D, u: float, kind: int, component: int) -> float:
    if kind == 1:
        return wp.texture_sample(texture, u, dtype=wp.vec2f)[component]
    if kind == 2:
        return wp.texture_sample(texture, u, dtype=wp.vec4f)[component]
    return wp.texture_sample(texture, u, dtype=float)


@wp.func
def sample_all_widths_2d(texture: wp.Texture2D, uv: wp.vec2f, kind: int, component: int) -> float:
    if kind == 1:
        return wp.texture_sample(texture, uv, dtype=wp.vec2f)[component]
    if kind == 2:
        return wp.texture_sample(texture, uv, dtype=wp.vec4f)[component]
    return wp.texture_sample(texture, uv, dtype=float)


@wp.func
def sample_all_widths_3d(texture: wp.Texture3D, uvw: wp.vec3f, kind: int, component: int) -> float:
    if kind == 1:
        return wp.texture_sample(texture, uvw, dtype=wp.vec2f)[component]
    if kind == 2:
        return wp.texture_sample(texture, uvw, dtype=wp.vec4f)[component]
    return wp.texture_sample(texture, uvw, dtype=float)


def sample_scalar_vec2_1d_kernel(
    scale: wp.vec3f,
    table: wp.array[TextureSampleData1D],
    out_value: wp.array[float],
    out_gradient: wp.array[wp.vec3f],
):
    tid = wp.tid()
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
    v0 = sample_scalar_vec2_1d(data.texture, float(bx) + 0.5, data.kind, data.component)
    v1 = sample_scalar_vec2_1d(data.texture, float(bx) + 1.5, data.kind, data.component)
    value = v0 + (v1 - v0) * tx
    gradient = wp.vec3f(0.0)
    diff_len = wp.length(diff)
    if diff_len > 0.0:
        value = value + diff_len
        gradient = diff / diff_len
    inv_scale = wp.vec3f(1.0 / scale[0], 1.0 / scale[1], 1.0 / scale[2])
    scaled_gradient = wp.cw_mul(gradient, inv_scale)
    gradient_len = wp.length(scaled_gradient)
    if gradient_len > 0.0:
        scaled_gradient = scaled_gradient / gradient_len
    else:
        scaled_gradient = gradient
    out_value[tid] = value
    out_gradient[tid] = scaled_gradient


def sample_scalar_vec2_2d_kernel(
    scale: wp.vec3f,
    table: wp.array[TextureSampleData2D],
    out_value: wp.array[float],
    out_gradient: wp.array[wp.vec3f],
):
    tid = wp.tid()
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
    v0 = sample_scalar_vec2_2d(data.texture, wp.vec2f(float(bx) + 0.5, 4.5), data.kind, data.component)
    v1 = sample_scalar_vec2_2d(data.texture, wp.vec2f(float(bx) + 1.5, 4.5), data.kind, data.component)
    value = v0 + (v1 - v0) * tx
    gradient = wp.vec3f(0.0)
    diff_len = wp.length(diff)
    if diff_len > 0.0:
        value = value + diff_len
        gradient = diff / diff_len
    inv_scale = wp.vec3f(1.0 / scale[0], 1.0 / scale[1], 1.0 / scale[2])
    scaled_gradient = wp.cw_mul(gradient, inv_scale)
    gradient_len = wp.length(scaled_gradient)
    if gradient_len > 0.0:
        scaled_gradient = scaled_gradient / gradient_len
    else:
        scaled_gradient = gradient
    out_value[tid] = value
    out_gradient[tid] = scaled_gradient


def sample_scalar_vec2_3d_kernel(
    scale: wp.vec3f,
    table: wp.array[TextureSampleData3D],
    out_value: wp.array[float],
    out_gradient: wp.array[wp.vec3f],
):
    tid = wp.tid()
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
    v0 = sample_scalar_vec2_3d(data.texture, wp.vec3f(float(bx) + 0.5, 4.5, 4.5), data.kind, data.component)
    v1 = sample_scalar_vec2_3d(data.texture, wp.vec3f(float(bx) + 1.5, 4.5, 4.5), data.kind, data.component)
    value = v0 + (v1 - v0) * tx
    gradient = wp.vec3f(0.0)
    diff_len = wp.length(diff)
    if diff_len > 0.0:
        value = value + diff_len
        gradient = diff / diff_len
    inv_scale = wp.vec3f(1.0 / scale[0], 1.0 / scale[1], 1.0 / scale[2])
    scaled_gradient = wp.cw_mul(gradient, inv_scale)
    gradient_len = wp.length(scaled_gradient)
    if gradient_len > 0.0:
        scaled_gradient = scaled_gradient / gradient_len
    else:
        scaled_gradient = gradient
    out_value[tid] = value
    out_gradient[tid] = scaled_gradient


def sample_all_widths_1d_kernel(
    scale: wp.vec3f,
    table: wp.array[TextureSampleData1D],
    out_value: wp.array[float],
    out_gradient: wp.array[wp.vec3f],
):
    tid = wp.tid()
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
    v0 = sample_all_widths_1d(data.texture, float(bx) + 0.5, data.kind, data.component)
    v1 = sample_all_widths_1d(data.texture, float(bx) + 1.5, data.kind, data.component)
    value = v0 + (v1 - v0) * tx
    gradient = wp.vec3f(0.0)
    diff_len = wp.length(diff)
    if diff_len > 0.0:
        value = value + diff_len
        gradient = diff / diff_len
    inv_scale = wp.vec3f(1.0 / scale[0], 1.0 / scale[1], 1.0 / scale[2])
    scaled_gradient = wp.cw_mul(gradient, inv_scale)
    gradient_len = wp.length(scaled_gradient)
    if gradient_len > 0.0:
        scaled_gradient = scaled_gradient / gradient_len
    else:
        scaled_gradient = gradient
    out_value[tid] = value
    out_gradient[tid] = scaled_gradient


def sample_all_widths_2d_kernel(
    scale: wp.vec3f,
    table: wp.array[TextureSampleData2D],
    out_value: wp.array[float],
    out_gradient: wp.array[wp.vec3f],
):
    tid = wp.tid()
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
    v0 = sample_all_widths_2d(data.texture, wp.vec2f(float(bx) + 0.5, 4.5), data.kind, data.component)
    v1 = sample_all_widths_2d(data.texture, wp.vec2f(float(bx) + 1.5, 4.5), data.kind, data.component)
    value = v0 + (v1 - v0) * tx
    gradient = wp.vec3f(0.0)
    diff_len = wp.length(diff)
    if diff_len > 0.0:
        value = value + diff_len
        gradient = diff / diff_len
    inv_scale = wp.vec3f(1.0 / scale[0], 1.0 / scale[1], 1.0 / scale[2])
    scaled_gradient = wp.cw_mul(gradient, inv_scale)
    gradient_len = wp.length(scaled_gradient)
    if gradient_len > 0.0:
        scaled_gradient = scaled_gradient / gradient_len
    else:
        scaled_gradient = gradient
    out_value[tid] = value
    out_gradient[tid] = scaled_gradient


def sample_all_widths_3d_kernel(
    scale: wp.vec3f,
    table: wp.array[TextureSampleData3D],
    out_value: wp.array[float],
    out_gradient: wp.array[wp.vec3f],
):
    tid = wp.tid()
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
    v0 = sample_all_widths_3d(data.texture, wp.vec3f(float(bx) + 0.5, 4.5, 4.5), data.kind, data.component)
    v1 = sample_all_widths_3d(data.texture, wp.vec3f(float(bx) + 1.5, 4.5, 4.5), data.kind, data.component)
    value = v0 + (v1 - v0) * tx
    gradient = wp.vec3f(0.0)
    diff_len = wp.length(diff)
    if diff_len > 0.0:
        value = value + diff_len
        gradient = diff / diff_len
    inv_scale = wp.vec3f(1.0 / scale[0], 1.0 / scale[1], 1.0 / scale[2])
    scaled_gradient = wp.cw_mul(gradient, inv_scale)
    gradient_len = wp.length(scaled_gradient)
    if gradient_len > 0.0:
        scaled_gradient = scaled_gradient / gradient_len
    else:
        scaled_gradient = gradient
    out_value[tid] = value
    out_gradient[tid] = scaled_gradient


SLOT_LINEAR = wp.uint32(0xFFFFFFFE)


@wp.struct
class SdfTextureData:
    coarse_texture: wp.Texture3D
    subgrid_texture: wp.Texture3D
    slots: wp.array3d[wp.uint32]
    lower: wp.vec3f
    upper: wp.vec3f
    inv_dx: wp.vec3f
    subgrid_size: float
    subgrid_samples: float
    fine_to_coarse: float
    subgrid_min: float
    subgrid_range: float
    use_vector: wp.bool
    component: int


@wp.struct
class SdfTextureCell:
    ix: int
    iy: int
    iz: int
    tx: float
    ty: float
    tz: float
    bx: int
    by: int
    bz: int
    slot: wp.uint32


@wp.func
def sample_sdf_scalar_vec2(texture: wp.Texture3D, uvw: wp.vec3f, use_vector: bool, component: int) -> float:
    if use_vector:
        return wp.texture_sample(texture, uvw, dtype=wp.vec2f)[component]
    return wp.texture_sample(texture, uvw, dtype=float)


@wp.func
def sample_sdf_scalar_vec4(texture: wp.Texture3D, uvw: wp.vec3f, use_vector: bool, component: int) -> float:
    if use_vector:
        return wp.texture_sample(texture, uvw, dtype=wp.vec4f)[component]
    return wp.texture_sample(texture, uvw, dtype=float)


@wp.func
def locate_sdf_texture_cell(sdf: SdfTextureData, f: wp.vec3f) -> SdfTextureCell:
    nx = float(sdf.coarse_texture.width - 1) * sdf.subgrid_size
    ny = float(sdf.coarse_texture.height - 1) * sdf.subgrid_size
    nz = float(sdf.coarse_texture.depth - 1) * sdf.subgrid_size
    fx = wp.clamp(f[0], 0.0, nx)
    fy = wp.clamp(f[1], 0.0, ny)
    fz = wp.clamp(f[2], 0.0, nz)
    cell = SdfTextureCell()
    cell.ix = wp.clamp(int(wp.floor(fx)), 0, int(nx) - 1)
    cell.iy = wp.clamp(int(wp.floor(fy)), 0, int(ny) - 1)
    cell.iz = wp.clamp(int(wp.floor(fz)), 0, int(nz) - 1)
    cell.tx = fx - float(cell.ix)
    cell.ty = fy - float(cell.iy)
    cell.tz = fz - float(cell.iz)
    cell.bx = wp.clamp(int(float(cell.ix) * sdf.fine_to_coarse), 0, sdf.coarse_texture.width - 2)
    cell.by = wp.clamp(int(float(cell.iy) * sdf.fine_to_coarse), 0, sdf.coarse_texture.height - 2)
    cell.bz = wp.clamp(int(float(cell.iz) * sdf.fine_to_coarse), 0, sdf.coarse_texture.depth - 2)
    cell.slot = sdf.slots[cell.bx, cell.by, cell.bz]
    return cell


# Keep separate vec2 and vec4 SDF paths: this exact scalar/vec4 source shape is
# required to expose the independent vec4 failure with CUDA 12.9.
@wp.func
def sample_sdf_pair_vec2(sdf: SdfTextureData, f: wp.vec3f):
    cell = locate_sdf_texture_cell(sdf, f)
    v0 = float(0.0)
    v1 = float(0.0)
    tx = cell.tx
    if cell.slot >= SLOT_LINEAR:
        x = float(cell.bx)
        y = float(cell.by)
        z = float(cell.bz)
        tx = (float(cell.ix) + cell.tx) * sdf.fine_to_coarse - x
        v0 = sample_sdf_scalar_vec2(
            sdf.coarse_texture, wp.vec3f(x + 0.5, y + 0.5, z + 0.5), sdf.use_vector, sdf.component
        )
        v1 = sample_sdf_scalar_vec2(
            sdf.coarse_texture, wp.vec3f(x + 1.5, y + 0.5, z + 0.5), sdf.use_vector, sdf.component
        )
    else:
        x = (
            float(cell.slot & wp.uint32(0x3FF)) * sdf.subgrid_samples
            + float(cell.ix)
            - float(cell.bx) * sdf.subgrid_size
            + 0.5
        )
        y = (
            float((cell.slot >> wp.uint32(10)) & wp.uint32(0x3FF)) * sdf.subgrid_samples
            + float(cell.iy)
            - float(cell.by) * sdf.subgrid_size
            + 0.5
        )
        z = (
            float((cell.slot >> wp.uint32(20)) & wp.uint32(0x3FF)) * sdf.subgrid_samples
            + float(cell.iz)
            - float(cell.bz) * sdf.subgrid_size
            + 0.5
        )
        v0 = (
            sample_sdf_scalar_vec2(sdf.subgrid_texture, wp.vec3f(x, y, z), sdf.use_vector, sdf.component)
            * sdf.subgrid_range
            + sdf.subgrid_min
        )
        v1 = (
            sample_sdf_scalar_vec2(sdf.subgrid_texture, wp.vec3f(x + 1.0, y, z), sdf.use_vector, sdf.component)
            * sdf.subgrid_range
            + sdf.subgrid_min
        )
    return v0, v1, tx


def sample_sdf_vec2_kernel(
    scale: wp.vec3f,
    table: wp.array[SdfTextureData],
    out_value: wp.array[float],
    out_gradient: wp.array[wp.vec3f],
):
    sdf = table[0]
    local = wp.cw_div(wp.vec3f(1.0, 0.0, 0.0), scale)
    clamped = wp.vec3f(
        wp.clamp(local[0], sdf.lower[0], sdf.upper[0]),
        wp.clamp(local[1], sdf.lower[1], sdf.upper[1]),
        wp.clamp(local[2], sdf.lower[2], sdf.upper[2]),
    )
    diff = local - clamped
    v0, v1, tx = sample_sdf_pair_vec2(sdf, wp.cw_mul(clamped - sdf.lower, sdf.inv_dx))
    value = v0 + (v1 - v0) * tx
    gradient = wp.vec3f(0.0)
    diff_len = wp.length(diff)
    if diff_len > 0.0:
        value = value + diff_len
        gradient = diff / diff_len
    inv_scale = wp.vec3f(1.0 / scale[0], 1.0 / scale[1], 1.0 / scale[2])
    scaled_gradient = wp.cw_mul(gradient, inv_scale)
    gradient_len = wp.length(scaled_gradient)
    if gradient_len > 0.0:
        scaled_gradient = scaled_gradient / gradient_len
    else:
        scaled_gradient = gradient
    out_value[0] = value
    out_gradient[0] = scaled_gradient


@wp.func
def sample_sdf_pair_vec4(sdf: SdfTextureData, f: wp.vec3f):
    cell = locate_sdf_texture_cell(sdf, f)
    v0 = float(0.0)
    v1 = float(0.0)
    tx = cell.tx
    if cell.slot >= SLOT_LINEAR:
        x = float(cell.bx)
        y = float(cell.by)
        z = float(cell.bz)
        tx = (float(cell.ix) + cell.tx) * sdf.fine_to_coarse - x
        v0 = sample_sdf_scalar_vec4(
            sdf.coarse_texture, wp.vec3f(x + 0.5, y + 0.5, z + 0.5), sdf.use_vector, sdf.component
        )
        v1 = sample_sdf_scalar_vec4(
            sdf.coarse_texture, wp.vec3f(x + 1.5, y + 0.5, z + 0.5), sdf.use_vector, sdf.component
        )
    else:
        x = (
            float(cell.slot & wp.uint32(0x3FF)) * sdf.subgrid_samples
            + float(cell.ix)
            - float(cell.bx) * sdf.subgrid_size
            + 0.5
        )
        y = (
            float((cell.slot >> wp.uint32(10)) & wp.uint32(0x3FF)) * sdf.subgrid_samples
            + float(cell.iy)
            - float(cell.by) * sdf.subgrid_size
            + 0.5
        )
        z = (
            float((cell.slot >> wp.uint32(20)) & wp.uint32(0x3FF)) * sdf.subgrid_samples
            + float(cell.iz)
            - float(cell.bz) * sdf.subgrid_size
            + 0.5
        )
        v0 = (
            sample_sdf_scalar_vec4(sdf.subgrid_texture, wp.vec3f(x, y, z), sdf.use_vector, sdf.component)
            * sdf.subgrid_range
            + sdf.subgrid_min
        )
        v1 = (
            sample_sdf_scalar_vec4(sdf.subgrid_texture, wp.vec3f(x + 1.0, y, z), sdf.use_vector, sdf.component)
            * sdf.subgrid_range
            + sdf.subgrid_min
        )
    return v0, v1, tx


def sample_sdf_vec4_kernel(
    scale: wp.vec3f,
    table: wp.array[SdfTextureData],
    out_value: wp.array[float],
    out_gradient: wp.array[wp.vec3f],
):
    sdf = table[0]
    local = wp.cw_div(wp.vec3f(1.0, 0.0, 0.0), scale)
    clamped = wp.vec3f(
        wp.clamp(local[0], sdf.lower[0], sdf.upper[0]),
        wp.clamp(local[1], sdf.lower[1], sdf.upper[1]),
        wp.clamp(local[2], sdf.lower[2], sdf.upper[2]),
    )
    diff = local - clamped
    v0, v1, tx = sample_sdf_pair_vec4(sdf, wp.cw_mul(clamped - sdf.lower, sdf.inv_dx))
    value = v0 + (v1 - v0) * tx
    gradient = wp.vec3f(0.0)
    diff_len = wp.length(diff)
    if diff_len > 0.0:
        value = value + diff_len
        gradient = diff / diff_len
    inv_scale = wp.vec3f(1.0 / scale[0], 1.0 / scale[1], 1.0 / scale[2])
    scaled_gradient = wp.cw_mul(gradient, inv_scale)
    gradient_len = wp.length(scaled_gradient)
    if gradient_len > 0.0:
        scaled_gradient = scaled_gradient / gradient_len
    else:
        scaled_gradient = gradient
    out_value[0] = value
    out_gradient[0] = scaled_gradient


KERNEL_FUNCTIONS = {
    "scalar_vec2_1d": sample_scalar_vec2_1d_kernel,
    "scalar_vec2_2d": sample_scalar_vec2_2d_kernel,
    "scalar_vec2_3d": sample_scalar_vec2_3d_kernel,
    "all_widths_1d": sample_all_widths_1d_kernel,
    "all_widths_2d": sample_all_widths_2d_kernel,
    "all_widths_3d": sample_all_widths_3d_kernel,
    "sdf_vec2": sample_sdf_vec2_kernel,
    "sdf_vec4": sample_sdf_vec4_kernel,
}

TEXTURE_COMPILER_KERNELS = {}
for cuda_output in ("cubin", "ptx"):
    for optimization_level in (0, 3):
        module = wp.Module(f"test_texture_compiler_{cuda_output}_{optimization_level}")
        wp.set_module_options(
            {"cuda_output": cuda_output, "optimization_level": optimization_level},
            module=module,
        )
        TEXTURE_COMPILER_KERNELS[(cuda_output, optimization_level)] = {
            name: wp.kernel(kernel_func, module=module) for name, kernel_func in KERNEL_FUNCTIONS.items()
        }


DIMENSION_CASES = (
    ("1d", wp.Texture1D, TextureSampleData1D, (9,)),
    ("2d", wp.Texture2D, TextureSampleData2D, (9, 9)),
    ("3d", wp.Texture3D, TextureSampleData3D, (9, 9, 9)),
)


def make_constant_texture(texture_cls, shape, values, device):
    if isinstance(values, tuple):
        data = np.empty((*shape, len(values)), dtype=np.float32)
        data[...] = np.asarray(values, dtype=np.float32)
    else:
        data = np.full(shape, values, dtype=np.float32)
    return texture_cls(
        data,
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        normalized_coords=False,
        device=device,
    )


def run_dimension_case(test, device, kernel, texture_cls, data_cls, shape, values, kind):
    texture = make_constant_texture(texture_cls, shape, values, device)
    data = data_cls()
    data.texture = texture
    data.kind = kind
    data.component = 0
    table = wp.array([data], dtype=data_cls, device=device)
    out_value = wp.empty(1, dtype=float, device=device)
    out_gradient = wp.empty(1, dtype=wp.vec3f, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[wp.vec3f(1.0), table],
        outputs=[out_value, out_gradient],
        device=device,
    )

    np.testing.assert_allclose(out_value.numpy(), np.array([0.5], dtype=np.float32), rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(
        out_gradient.numpy(), np.array([[1.0, 0.0, 0.0]], dtype=np.float32), rtol=0.0, atol=1.0e-6
    )


def make_sdf_texture_data(device):
    coarse_texture = make_constant_texture(wp.Texture3D, (9, 9, 9), (0.1, 0.2), device)
    subgrid_texture = wp.Texture3D(
        np.zeros((1, 1, 1, 2), dtype=np.uint16),
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        normalized_coords=False,
        device=device,
    )
    sdf = SdfTextureData()
    sdf.coarse_texture = coarse_texture
    sdf.subgrid_texture = subgrid_texture
    sdf.slots = wp.array(np.full((8, 8, 8), SLOT_LINEAR), dtype=wp.uint32, device=device)
    sdf.lower = wp.vec3f(-0.6)
    sdf.upper = wp.vec3f(0.6)
    sdf.inv_dx = wp.vec3f(53.333332)
    sdf.subgrid_size = 8.0
    sdf.subgrid_samples = 9.0
    sdf.fine_to_coarse = 0.125
    sdf.subgrid_min = -0.22990382
    sdf.subgrid_range = 0.45980763
    sdf.use_vector = True
    sdf.component = 0
    return sdf, coarse_texture, subgrid_texture


def test_texture_mixed_width_compiler_regression(test, device):
    for (cuda_output, optimization_level), kernels in TEXTURE_COMPILER_KERNELS.items():
        with test.subTest(cuda_output=cuda_output, optimization_level=optimization_level):
            for dimension, texture_cls, data_cls, shape in DIMENSION_CASES:
                with test.subTest(dimension=dimension, selected_type="scalar"):
                    run_dimension_case(
                        test,
                        device,
                        kernels[f"scalar_vec2_{dimension}"],
                        texture_cls,
                        data_cls,
                        shape,
                        0.1,
                        0,
                    )
                with test.subTest(dimension=dimension, selected_type="vec4"):
                    run_dimension_case(
                        test,
                        device,
                        kernels[f"all_widths_{dimension}"],
                        texture_cls,
                        data_cls,
                        shape,
                        (0.1, 0.2, 0.3, 0.4),
                        2,
                    )

            for vector_type in ("vec2", "vec4"):
                with test.subTest(source_shape="sdf", selected_type=vector_type):
                    sdf, coarse_texture, subgrid_texture = make_sdf_texture_data(device)
                    table = wp.array([sdf], dtype=SdfTextureData, device=device)
                    out_value = wp.empty(1, dtype=float, device=device)
                    out_gradient = wp.empty(1, dtype=wp.vec3f, device=device)
                    wp.launch(
                        kernels[f"sdf_{vector_type}"],
                        dim=1,
                        inputs=[wp.vec3f(1.0), table],
                        outputs=[out_value, out_gradient],
                        device=device,
                    )
                    np.testing.assert_allclose(
                        out_value.numpy(), np.array([0.5], dtype=np.float32), rtol=0.0, atol=1.0e-6
                    )
                    np.testing.assert_allclose(
                        out_gradient.numpy(),
                        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                        rtol=0.0,
                        atol=1.0e-6,
                    )
                    del coarse_texture, subgrid_texture


class TestTextureCompiler(unittest.TestCase):
    pass


add_function_test(
    TestTextureCompiler,
    "test_texture_mixed_width_compiler_regression",
    test_texture_mixed_width_compiler_regression,
    devices=get_selected_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
