# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Sample node deforming an ocean surface."""

import traceback

import omni.graph.core as og
import omni.warp.nodes
from omni.warp.nodes.ogn.OgnSampleOceanDeformDatabase import OgnSampleOceanDeformDatabase

import warp as wp

PROFILE_EXTENT = 410.0
PROFILE_RES = 8192
PROFILE_WAVENUM = 1000
MIN_WAVE_LENGTH = 0.1
MAX_WAVE_LENGTH = 250.0


#   Kernels
# ------------------------------------------------------------------------------


# fractional part of a (w.r.t. floor(a))
@wp.func
def frac(a: float):
    return a - wp.floor(a)


# square of a
@wp.func
def sqr(a: float):
    return a * a


@wp.func
def alpha_beta_spectrum(omega: float, peak_omega: float, alpha: float, beta: float, gravity: float):
    return (alpha * gravity * gravity / wp.pow(omega, 5.0)) * wp.exp(-beta * wp.pow(peak_omega / omega, 4.0))


@wp.func
def jonswap_peak_sharpening(omega: float, peak_omega: float, gamma: float):
    sigma = float(0.07)
    if omega > peak_omega:
        sigma = float(0.09)
    return wp.pow(gamma, wp.exp(-0.5 * sqr((omega - peak_omega) / (sigma * peak_omega))))


@wp.func
def jonswap_spectrum(omega: float, gravity: float, wind_speed: float, fetch_km: float, gamma: float):
    # https://wikiwaves.org/Ocean-Wave_Spectra#JONSWAP_Spectrum
    fetch = 1000.0 * fetch_km
    alpha = 0.076 * wp.pow(wind_speed * wind_speed / (gravity * fetch), 0.22)
    peak_omega = 22.0 * wp.pow(wp.abs(gravity * gravity / (wind_speed * fetch)), 1.0 / 3.0)
    return jonswap_peak_sharpening(omega, peak_omega, gamma) * alpha_beta_spectrum(
        omega, peak_omega, alpha, 1.25, gravity
    )


@wp.func
def TMA_spectrum(omega: float, gravity: float, wind_speed: float, fetch_km: float, gamma: float, water_depth: float):
    # https://dl.acm.org/doi/10.1145/2791261.2791267
    omegaH = omega * wp.sqrt(water_depth / gravity)
    omegaH = wp.max(0.0, wp.min(2.2, omegaH))
    phi = 0.5 * omegaH * omegaH
    if omegaH > 1.0:
        phi = 1.0 - 0.5 * sqr(2.0 - omegaH)
    return phi * jonswap_spectrum(omega, gravity, wind_speed, fetch_km, gamma)


# warp kernel definitions
@wp.kernel
def update_profile(
    profile: wp.array(dtype=wp.vec3),
    profile_res: int,
    profile_data_num: int,
    min_lambda: float,
    max_lambda: float,
    profile_extend: float,
    time: float,
    wind_speed: float,
    water_depth: float,
):
    x = wp.tid()
    randState = wp.rand_init(7)
    # sampling parameters
    omega0 = wp.sqrt(wp.tau * 9.80665 / min_lambda)
    omega1 = wp.sqrt(wp.tau * 9.80665 / max_lambda)
    omega_delta = wp.abs(omega1 - omega0) / float(profile_data_num)
    # we blend three displacements for seamless spatial profile tiling
    space_pos_1 = profile_extend * float(x) / float(profile_res)
    space_pos_2 = space_pos_1 + profile_extend
    space_pos_3 = space_pos_1 - profile_extend
    p1 = wp.vec2(0.0, 0.0)
    p2 = wp.vec2(0.0, 0.0)
    p3 = wp.vec2(0.0, 0.0)
    for i in range(profile_data_num):
        omega = wp.abs(omega0 + (omega1 - omega0) * float(i) / float(profile_data_num))  # linear sampling of omega
        k = omega * omega / 9.80665
        phase = -time * omega + wp.randf(randState) * 2.0 * wp.pi
        amplitude = float(10000.0) * wp.sqrt(
            wp.abs(2.0 * omega_delta * TMA_spectrum(omega, 9.80665, wind_speed, 100.0, 3.3, water_depth))
        )
        p1 = wp.vec2(
            p1[0] + amplitude * wp.sin(phase + space_pos_1 * k), p1[1] - amplitude * wp.cos(phase + space_pos_1 * k)
        )
        p2 = wp.vec2(
            p2[0] + amplitude * wp.sin(phase + space_pos_2 * k), p2[1] - amplitude * wp.cos(phase + space_pos_2 * k)
        )
        p3 = wp.vec2(
            p3[0] + amplitude * wp.sin(phase + space_pos_3 * k), p3[1] - amplitude * wp.cos(phase + space_pos_3 * k)
        )
    # cubic blending coefficients
    s = float(float(x) / float(profile_res))
    c1 = float(2.0 * s * s * s - 3.0 * s * s + 1.0)
    c2 = float(-2.0 * s * s * s + 3.0 * s * s)
    disp_out = wp.vec3(
        (p1[0] + c1 * p2[0] + c2 * p3[0]) / float(profile_data_num),
        (p1[1] + c1 * p2[1] + c2 * p3[1]) / float(profile_data_num),
        0.0,
    )
    profile[x] = disp_out


@wp.kernel
def update_points(
    points: wp.array(dtype=wp.vec3),
    profile: wp.array(dtype=wp.vec3),
    profile_res: int,
    profile_extent: float,
    amplitude: float,
    directionality: float,
    direction: float,
    cam_pos: wp.vec3,
    clipmap_cell_size: float,
    out_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    p_crd = wp.vec3(
        points[tid][0] + wp.floor(cam_pos[0] / clipmap_cell_size) * clipmap_cell_size,
        points[tid][1],
        points[tid][2] + wp.floor(cam_pos[2] / clipmap_cell_size) * clipmap_cell_size,
    )

    randState = wp.rand_init(7)
    disp_x = float(0.0)
    disp_y = float(0.0)
    disp_z = float(0.0)
    w_sum = float(0.0)
    direction_count = 128
    for d in range(0, direction_count):
        r = float(d) * wp.tau / float(direction_count) + 0.02
        dir_x = wp.cos(r)
        dir_y = wp.sin(r)
        # directional amplitude
        t = wp.abs(direction - r)
        if t > wp.pi:
            t = wp.tau - t
        t = pow(t, 1.2)
        dir_amp = (2.0 * t * t * t - 3.0 * t * t + 1.0) * 1.0 + (-2.0 * t * t * t + 3.0 * t * t) * (
            1.0 - directionality
        )
        dir_amp = dir_amp / (1.0 + 10.0 * directionality)
        rand_phase = wp.randf(randState)
        x_crd = (p_crd[0] * dir_x + p_crd[2] * dir_y) / profile_extent + rand_phase
        pos_0 = int(wp.floor(x_crd * float(profile_res))) % profile_res
        if x_crd < 0.0:
            pos_0 = pos_0 + profile_res - 1
        pos_1 = int(pos_0 + 1) % profile_res
        p_disp_0 = profile[pos_0]
        p_disp_1 = profile[pos_1]
        w = frac(x_crd * float(profile_res))
        prof_height_x = dir_amp * float((1.0 - w) * p_disp_0[0] + w * p_disp_1[0])
        prof_height_y = dir_amp * float((1.0 - w) * p_disp_0[1] + w * p_disp_1[1])
        disp_x = disp_x + dir_x * prof_height_x
        disp_y = disp_y + prof_height_y
        disp_z = disp_z + dir_y * prof_height_x
        w_sum = w_sum + 1.0

    # write output vertex position
    out_points[tid] = wp.vec3(
        p_crd[0] + amplitude * disp_x / w_sum,
        p_crd[1] + amplitude * disp_y / w_sum,
        p_crd[2] + amplitude * disp_z / w_sum,
    )


#   Internal State
# ------------------------------------------------------------------------------


class InternalState:
    """Convenience class for maintaining per-node state information"""

    def __init__(self):
        self.profile = wp.zeros(PROFILE_RES, dtype=wp.vec3)


#   Compute
# ------------------------------------------------------------------------------


def compute(db: OgnSampleOceanDeformDatabase) -> None:
    """Evaluates the node."""
    if not db.inputs.mesh.valid or not db.outputs.mesh.valid:
        return

    state = db.per_instance_state

    amplitude = max(0.0001, min(1000.0, db.inputs.amplitude))
    direction = db.inputs.direction % 6.28318530718
    directionality = max(0.0, min(1.0, 0.02 * db.inputs.directionality))
    wind_speed = max(0.0, min(30.0, db.inputs.windSpeed))
    water_depth = max(1.0, min(1000.0, db.inputs.waterDepth))
    scale = min(10000.0, max(0.001, db.inputs.scale))

    # create 1D profile buffer for this timestep using wave parameters
    wp.launch(
        kernel=update_profile,
        dim=(PROFILE_RES,),
        inputs=(
            state.profile,
            PROFILE_RES,
            PROFILE_WAVENUM,
            MIN_WAVE_LENGTH,
            MAX_WAVE_LENGTH,
            PROFILE_EXTENT,
            db.inputs.time,
            wind_speed,
            water_depth,
        ),
    )

    # Copy the input geometry mesh bundle and read its contents.
    db.outputs.mesh = db.inputs.mesh

    # Retrieve the input and output point data.
    points = omni.warp.nodes.mesh_get_points(db.inputs.mesh)
    out_points = omni.warp.nodes.mesh_get_points(db.outputs.mesh)

    # update point positions using the profile buffer created above
    wp.launch(
        kernel=update_points,
        dim=len(points),
        inputs=(
            points,
            state.profile,
            PROFILE_RES,
            PROFILE_EXTENT * scale,
            amplitude,
            directionality,
            direction,
            db.inputs.cameraPos,
            db.inputs.clipmapCellSize,
        ),
        outputs=(out_points,),
    )


#   Node Entry Point
# ------------------------------------------------------------------------------


class OgnSampleOceanDeform:
    """
    Mesh deformer modeling ocean waves.
    """

    @staticmethod
    def internal_state():
        """Returns an object that will contain per-node state information"""
        return InternalState()

    @staticmethod
    def compute(db) -> bool:
        """Compute the outputs from the current input"""
        device = omni.warp.nodes.device_get_cuda_compute()

        try:
            with wp.ScopedDevice(device):
                compute(db)
        except Exception:
            db.log_error(traceback.format_exc())
            return

        # Fire the execution for the downstream nodes.
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
