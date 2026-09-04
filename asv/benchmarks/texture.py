# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for representative CUDA texture-sampling workloads."""

import numpy as np

import warp as wp

from .benchmarks_utils import setup_once

NUM_QUERIES = 1 << 20
NUM_TEXTURES = 4

TEXTURE_1D_WIDTH = 1 << 16

TEXTURE_2D_WIDTH = 512
TEXTURE_2D_HEIGHT = 512

TEXTURE_3D_WIDTH = 128
TEXTURE_3D_HEIGHT = 128
TEXTURE_3D_DEPTH = 64


@wp.func
def query_1d(tid: int) -> float:
    return (float(tid % TEXTURE_1D_WIDTH) + 0.5) / float(TEXTURE_1D_WIDTH)


@wp.func
def query_2d(tid: int) -> wp.vec2f:
    x = tid % TEXTURE_2D_WIDTH
    y = (tid // TEXTURE_2D_WIDTH) % TEXTURE_2D_HEIGHT
    return wp.vec2f(
        (float(x) + 0.5) / float(TEXTURE_2D_WIDTH),
        (float(y) + 0.5) / float(TEXTURE_2D_HEIGHT),
    )


@wp.func
def query_3d(tid: int) -> wp.vec3f:
    x = tid % TEXTURE_3D_WIDTH
    y = (tid // TEXTURE_3D_WIDTH) % TEXTURE_3D_HEIGHT
    z = (tid // (TEXTURE_3D_WIDTH * TEXTURE_3D_HEIGHT)) % TEXTURE_3D_DEPTH
    return wp.vec3f(
        float(x) + 0.5,
        float(y) + 0.5,
        float(z) + 0.5,
    )


@wp.func
def triplanar_sample(tex: wp.Texture2D, local: wp.vec3f) -> wp.vec4f:
    normal = wp.vec3f(local[0] + 0.25, local[1] + 0.5, local[2] + 1.0)
    weights = wp.vec3f(wp.abs(normal[0]), wp.abs(normal[1]), wp.abs(normal[2]))
    weights /= weights[0] + weights[1] + weights[2]

    color_x = wp.texture_sample(tex, wp.vec2f(local[1], local[2]), dtype=wp.vec4f)
    color_y = wp.texture_sample(tex, wp.vec2f(local[0], local[2]), dtype=wp.vec4f)
    color_z = wp.texture_sample(tex, wp.vec2f(local[0], local[1]), dtype=wp.vec4f)
    return color_x * weights[0] + color_y * weights[1] + color_z * weights[2]


@wp.func
def finite_difference_gradient(tex: wp.Texture3D, uvw: wp.vec3f) -> wp.vec3f:
    dx = wp.vec3f(0.5, 0.0, 0.0)
    dy = wp.vec3f(0.0, 0.5, 0.0)
    dz = wp.vec3f(0.0, 0.0, 0.5)
    gx = wp.texture_sample(tex, uvw + dx, dtype=float) - wp.texture_sample(tex, uvw - dx, dtype=float)
    gy = wp.texture_sample(tex, uvw + dy, dtype=float) - wp.texture_sample(tex, uvw - dy, dtype=float)
    gz = wp.texture_sample(tex, uvw + dz, dtype=float) - wp.texture_sample(tex, uvw - dz, dtype=float)
    return wp.vec3f(gx, gy, gz)


@wp.func
def texture_index(tid: int, lane_varying: bool) -> int:
    if lane_varying:
        return tid % NUM_TEXTURES
    return (tid // 32) % NUM_TEXTURES


@wp.kernel
def sample_texture2d_uv(tex: wp.Texture2D, output: wp.array[wp.vec4f]):
    tid = wp.tid()
    output[tid] = wp.texture_sample(tex, query_2d(tid), dtype=wp.vec4f)


@wp.kernel
def sample_texture1d_vec2(tex: wp.Texture1D, output: wp.array[wp.vec2f]):
    tid = wp.tid()
    output[tid] = wp.texture_sample(tex, query_1d(tid), dtype=wp.vec2f)


@wp.kernel
def sample_texture1d_vec4(tex: wp.Texture1D, output: wp.array[wp.vec4f]):
    tid = wp.tid()
    output[tid] = wp.texture_sample(tex, query_1d(tid), dtype=wp.vec4f)


@wp.kernel
def sample_texture2d_vec2(tex: wp.Texture2D, output: wp.array[wp.vec2f]):
    tid = wp.tid()
    output[tid] = wp.texture_sample(tex, query_2d(tid), dtype=wp.vec2f)


@wp.kernel
def sample_texture3d_vec2(tex: wp.Texture3D, output: wp.array[wp.vec2f]):
    tid = wp.tid()
    output[tid] = wp.texture_sample(tex, query_3d(tid), dtype=wp.vec2f)


@wp.kernel
def sample_texture3d_vec4(tex: wp.Texture3D, output: wp.array[wp.vec4f]):
    tid = wp.tid()
    output[tid] = wp.texture_sample(tex, query_3d(tid), dtype=wp.vec4f)


@wp.kernel
def sample_texture2d_triplanar(tex: wp.Texture2D, output: wp.array[wp.vec4f]):
    tid = wp.tid()
    uv = query_2d(tid)
    local = wp.vec3f(uv[0], uv[1], float((tid * 17) % TEXTURE_2D_WIDTH) / float(TEXTURE_2D_WIDTH))
    output[tid] = triplanar_sample(tex, local)


@wp.kernel
def sample_texture3d_value(tex: wp.Texture3D, output: wp.array[float]):
    tid = wp.tid()
    output[tid] = wp.texture_sample(tex, query_3d(tid), dtype=float)


@wp.kernel
def sample_texture3d_finite_difference_gradient(tex: wp.Texture3D, output: wp.array[wp.vec3f]):
    tid = wp.tid()
    output[tid] = finite_difference_gradient(tex, query_3d(tid))


@wp.kernel
def sample_texture2d_array(
    textures: wp.array[wp.Texture2D],
    lane_varying: bool,
    output: wp.array[wp.vec4f],
):
    tid = wp.tid()
    tex = textures[texture_index(tid, lane_varying)]
    uv = query_2d(tid)
    local = wp.vec3f(uv[0], uv[1], float((tid * 17) % TEXTURE_2D_WIDTH) / float(TEXTURE_2D_WIDTH))
    output[tid] = triplanar_sample(tex, local)


@wp.kernel
def sample_texture3d_array(
    textures: wp.array[wp.Texture3D],
    lane_varying: bool,
    output: wp.array[wp.vec3f],
):
    tid = wp.tid()
    tex = textures[texture_index(tid, lane_varying)]
    output[tid] = finite_difference_gradient(tex, query_3d(tid))


def _make_texture1d_data(seed: int, num_channels: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((TEXTURE_1D_WIDTH, num_channels), dtype=np.float32)


def _make_texture2d_data(seed: int, num_channels: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((TEXTURE_2D_HEIGHT, TEXTURE_2D_WIDTH, num_channels), dtype=np.float32)


def _make_texture3d_data(seed: int, num_channels: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (TEXTURE_3D_DEPTH, TEXTURE_3D_HEIGHT, TEXTURE_3D_WIDTH)
    if num_channels == 1:
        return rng.random(shape, dtype=np.float32)
    return rng.random((*shape, num_channels), dtype=np.float32)


def _make_texture1d(data: np.ndarray, device: wp.Device) -> wp.Texture1D:
    return wp.Texture1D(
        data,
        filter_mode=wp.TextureFilterMode.LINEAR,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )


def _make_texture2d(data: np.ndarray, device: wp.Device) -> wp.Texture2D:
    return wp.Texture2D(
        data,
        filter_mode=wp.TextureFilterMode.LINEAR,
        address_mode=wp.TextureAddressMode.CLAMP,
        device=device,
    )


def _make_texture3d(data: np.ndarray, device: wp.Device) -> wp.Texture3D:
    return wp.Texture3D(
        data,
        filter_mode=wp.TextureFilterMode.LINEAR,
        address_mode=wp.TextureAddressMode.CLAMP,
        normalized_coords=False,
        device=device,
    )


class TextureBaseVector:
    """Sample one base-level vector texture value per query."""

    params = ["1d_vec2", "1d_vec4", "2d_vec2", "2d_vec4", "3d_vec2", "3d_vec4"]
    param_names = ["sample"]
    number = 256
    repeat = 10

    @setup_once
    def setup(self, sample):
        wp.init()
        self.device = wp.get_device("cuda:0")
        dimension = sample[:2]
        num_channels = int(sample[-1])
        dtype = wp.vec2f if num_channels == 2 else wp.vec4f

        if dimension == "1d":
            self.texture = _make_texture1d(_make_texture1d_data(42, num_channels), self.device)
            kernel = sample_texture1d_vec2 if num_channels == 2 else sample_texture1d_vec4
        elif dimension == "2d":
            self.texture = _make_texture2d(_make_texture2d_data(42, num_channels), self.device)
            kernel = sample_texture2d_vec2 if num_channels == 2 else sample_texture2d_uv
        else:
            self.texture = _make_texture3d(_make_texture3d_data(42, num_channels), self.device)
            kernel = sample_texture3d_vec2 if num_channels == 2 else sample_texture3d_vec4

        self.output = wp.empty(NUM_QUERIES, dtype=dtype, device=self.device)
        self.cmd = wp.launch(
            kernel,
            dim=NUM_QUERIES,
            inputs=[self.texture],
            outputs=[self.output],
            device=self.device,
            record_cmd=True,
        )
        self.cmd.launch()
        wp.synchronize_device(self.device)

    def time_cuda(self, sample):
        self.cmd.launch()
        wp.synchronize_device(self.device)


class Texture2DUv:
    """Sample one RGBA 2D texture value per query, as in UV-mapped rendering."""

    number = 256
    repeat = 10

    @setup_once
    def setup(self):
        wp.init()
        self.device = wp.get_device("cuda:0")
        self.texture = _make_texture2d(_make_texture2d_data(42), self.device)
        self.output = wp.empty(NUM_QUERIES, dtype=wp.vec4f, device=self.device)
        self.cmd = wp.launch(
            sample_texture2d_uv,
            dim=NUM_QUERIES,
            inputs=[self.texture],
            outputs=[self.output],
            device=self.device,
            record_cmd=True,
        )
        self.cmd.launch()
        wp.synchronize_device(self.device)

    def time_cuda(self):
        self.cmd.launch()
        wp.synchronize_device(self.device)


class Texture2DTriplanar:
    """Blend three RGBA 2D samples for triplanar material projection."""

    number = 100
    repeat = 5

    @setup_once
    def setup(self):
        wp.init()
        self.device = wp.get_device("cuda:0")
        self.texture = _make_texture2d(_make_texture2d_data(42), self.device)
        self.output = wp.empty(NUM_QUERIES, dtype=wp.vec4f, device=self.device)
        self.cmd = wp.launch(
            sample_texture2d_triplanar,
            dim=NUM_QUERIES,
            inputs=[self.texture],
            outputs=[self.output],
            device=self.device,
            record_cmd=True,
        )
        self.cmd.launch()
        wp.synchronize_device(self.device)

    def time_cuda(self):
        self.cmd.launch()
        wp.synchronize_device(self.device)


class Texture3DValue:
    """Sample one scalar 3D texture value per query, as in hardware-filtered SDF lookup."""

    number = 512
    repeat = 15

    @setup_once
    def setup(self):
        wp.init()
        self.device = wp.get_device("cuda:0")
        self.texture = _make_texture3d(_make_texture3d_data(42), self.device)
        self.output = wp.empty(NUM_QUERIES, dtype=float, device=self.device)
        self.cmd = wp.launch(
            sample_texture3d_value,
            dim=NUM_QUERIES,
            inputs=[self.texture],
            outputs=[self.output],
            device=self.device,
            record_cmd=True,
        )
        self.cmd.launch()
        wp.synchronize_device(self.device)

    def time_cuda(self):
        self.cmd.launch()
        wp.synchronize_device(self.device)


class Texture3DFiniteDifferenceGradient:
    """Compute an SDF-style finite-difference gradient from six 3D samples."""

    number = 256
    repeat = 10

    @setup_once
    def setup(self):
        wp.init()
        self.device = wp.get_device("cuda:0")
        self.texture = _make_texture3d(_make_texture3d_data(42), self.device)
        self.output = wp.empty(NUM_QUERIES, dtype=wp.vec3f, device=self.device)
        self.cmd = wp.launch(
            sample_texture3d_finite_difference_gradient,
            dim=NUM_QUERIES,
            inputs=[self.texture],
            outputs=[self.output],
            device=self.device,
            record_cmd=True,
        )
        self.cmd.launch()
        wp.synchronize_device(self.device)

    def time_cuda(self):
        self.cmd.launch()
        wp.synchronize_device(self.device)


class Texture2DArray:
    """Blend three 2D samples selected from warp-uniform or lane-varying handles."""

    params = ["warp_uniform", "lane_varying"]
    param_names = ["handle_pattern"]
    number = 100
    repeat = 5

    @setup_once
    def setup(self, handle_pattern):
        wp.init()
        self.device = wp.get_device("cuda:0")
        self.textures = [_make_texture2d(_make_texture2d_data(42 + i), self.device) for i in range(NUM_TEXTURES)]
        self.texture_array = wp.array(self.textures, dtype=wp.Texture2D, device=self.device)
        self.output = wp.empty(NUM_QUERIES, dtype=wp.vec4f, device=self.device)
        self.cmd = wp.launch(
            sample_texture2d_array,
            dim=NUM_QUERIES,
            inputs=[self.texture_array, handle_pattern == "lane_varying"],
            outputs=[self.output],
            device=self.device,
            record_cmd=True,
        )
        self.cmd.launch()
        wp.synchronize_device(self.device)

    def time_cuda(self, handle_pattern):
        self.cmd.launch()
        wp.synchronize_device(self.device)


class Texture3DArray:
    """Compute six-sample gradients from warp-uniform or lane-varying 3D handles."""

    params = ["warp_uniform", "lane_varying"]
    param_names = ["handle_pattern"]
    number = 75
    repeat = 5

    @setup_once
    def setup(self, handle_pattern):
        wp.init()
        if handle_pattern == "warp_uniform":
            self.number = 200
            self.repeat = 10
        else:
            self.number = 75
            self.repeat = 5
        self.device = wp.get_device("cuda:0")
        self.textures = [_make_texture3d(_make_texture3d_data(42 + i), self.device) for i in range(NUM_TEXTURES)]
        self.texture_array = wp.array(self.textures, dtype=wp.Texture3D, device=self.device)
        self.output = wp.empty(NUM_QUERIES, dtype=wp.vec3f, device=self.device)
        self.cmd = wp.launch(
            sample_texture3d_array,
            dim=NUM_QUERIES,
            inputs=[self.texture_array, handle_pattern == "lane_varying"],
            outputs=[self.output],
            device=self.device,
            record_cmd=True,
        )
        self.cmd.launch()
        wp.synchronize_device(self.device)

    def time_cuda(self, handle_pattern):
        self.cmd.launch()
        wp.synchronize_device(self.device)
