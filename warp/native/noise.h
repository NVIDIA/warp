/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

namespace wp
{

inline CUDA_CALLABLE float smootherstep(float t)
{
    return t * t * t * (t * (t * 6.f - 15.f) + 10.f);
}

inline CUDA_CALLABLE float smootherstep_gradient(float t)
{
    return 30.f * t * t * (t * (t - 2.f) + 1.f);
}

inline CUDA_CALLABLE float smoothstep(float t)
{
    return t * t * (3.f - t * 2.f);
}

inline CUDA_CALLABLE float smoothstep_gradient(float t)
{
    return 6.f * t * (1.f - t);
}

inline CUDA_CALLABLE float interpolate(float a0, float a1, float t)
{
    return (a1 - a0) * smootherstep(t) + a0;
    // return (a1 - a0) * smoothstep(t) + a0;
    // return (a1 - a0) * t + a0;
}

inline CUDA_CALLABLE float interpolate_gradient(float a0, float a1, float t, float d_a0, float d_a1, float d_t)
{
    return (d_a1 - d_a0) * smootherstep(t) + (a1 - a0) * smootherstep_gradient(t) * d_t + d_a0;
    // return (d_a1 - d_a0) * smoothstep(t) + (a1 - a0) * smoothstep_gradient(t) * d_t + d_a0;
    // return (d_a1 - d_a0) * t + (a1 - a0) * d_t + d_a0;
}

inline CUDA_CALLABLE vec2 interpolate_gradient_2d(float a0, float a1, float t, vec2& d_a0, vec2& d_a1, vec2& d_t)
{
    return (d_a1 - d_a0) * smootherstep(t) + (a1 - a0) * smootherstep_gradient(t) * d_t + d_a0;
}

inline CUDA_CALLABLE vec3 interpolate_gradient_3d(float a0, float a1, float t, vec3& d_a0, vec3& d_a1, vec3& d_t)
{
    return (d_a1 - d_a0) * smootherstep(t) + (a1 - a0) * smootherstep_gradient(t) * d_t + d_a0;
}

inline CUDA_CALLABLE vec4 interpolate_gradient_4d(float a0, float a1, float t, vec4& d_a0, vec4& d_a1, vec4& d_t)
{
    return (d_a1 - d_a0) * smootherstep(t) + (a1 - a0) * smootherstep_gradient(t) * d_t + d_a0;
}

inline CUDA_CALLABLE float random_gradient_1d(uint32 state, int ix)
{
    const uint32 p1 = 73856093;
    uint32 idx = ix*p1 + state;
    return randf(idx, -1.f, 1.f);
}

inline CUDA_CALLABLE vec2 random_gradient_2d(uint32 state, int ix, int iy)
{
    const uint32 p1 = 73856093;
    const uint32 p2 = 19349663;
    uint32 idx = ix*p1 ^ iy*p2 + state;
    
    return normalize(sample_unit_square(idx));
}

inline CUDA_CALLABLE vec3 random_gradient_3d(uint32 state, int ix, int iy, int iz)
{
    const uint32 p1 = 73856093;
	const uint32 p2 = 19349663;
    const uint32 p3 = 53471161;
    uint32 idx = ix*p1 ^ iy*p2 ^ iz*p3 + state;

    return normalize(sample_unit_cube(idx));
}

inline CUDA_CALLABLE vec4 random_gradient_4d(uint32 state, int ix, int iy, int iz, int it)
{
    const uint32 p1 = 73856093;
	const uint32 p2 = 19349663;
	const uint32 p3 = 53471161;
    const uint32 p4 = 10000019;
    uint32 idx = ix*p1 ^ iy*p2 ^ iz*p3 ^ it*p4 + state;

    return normalize(sample_unit_hypercube(idx));
}

inline CUDA_CALLABLE float dot_grid_gradient_1d(uint32 state, int ix, float dx)
{
    float gradient = random_gradient_1d(state, ix);
    return dx*gradient;
}

inline CUDA_CALLABLE float dot_grid_gradient_2d(uint32 state, int ix, int iy, float dx, float dy)
{
    vec2 gradient = random_gradient_2d(state, ix, iy);
    return (dx*gradient[0] + dy*gradient[1]);
}

inline CUDA_CALLABLE float dot_grid_gradient_3d(uint32 state, int ix, int iy, int iz, float dx, float dy, float dz)
{
    vec3 gradient = random_gradient_3d(state, ix, iy, iz);
    return (dx*gradient[0] + dy*gradient[1] + dz*gradient[2]);
}

inline CUDA_CALLABLE float dot_grid_gradient_4d(uint32 state, int ix, int iy, int iz, int it, float dx, float dy, float dz, float dt)
{
    vec4 gradient = random_gradient_4d(state, ix, iy, iz, it);
    return (dx*gradient[0] + dy*gradient[1] + dz*gradient[2] + dt*gradient[3]);
}

inline CUDA_CALLABLE float noise_1d(uint32 state, int x0, int x1, float dx)
{
    //vX
    float v0 = dot_grid_gradient_1d(state, x0, dx);
    float v1 = dot_grid_gradient_1d(state, x1, dx-1.f);

    return interpolate(v0, v1, dx);
}

inline CUDA_CALLABLE float noise_1d_gradient(uint32 state, int x0, int x1, float dx)
{
    float gradient_x0 = random_gradient_1d(state, x0);
    float v0 = dx * gradient_x0;

    float gradient_x1 = random_gradient_1d(state, x1);
    float v1 = (dx-1.f) * gradient_x1;

    return interpolate_gradient(v0, v1, dx, gradient_x0, gradient_x1, 1.f);
}

inline CUDA_CALLABLE float noise_2d(uint32 state, int x0, int y0, int x1, int y1, float dx, float dy)
{
    //vXY
    float v00 = dot_grid_gradient_2d(state, x0, y0, dx, dy);
    float v10 = dot_grid_gradient_2d(state, x1, y0, dx-1.f, dy);
    float xi0 = interpolate(v00, v10, dx);

    float v01 = dot_grid_gradient_2d(state, x0, y1, dx, dy-1.f);
    float v11 = dot_grid_gradient_2d(state, x1, y1, dx-1.f, dy-1.f);
    float xi1 = interpolate(v01, v11, dx);

    return interpolate(xi0, xi1, dy);
}

inline CUDA_CALLABLE vec2 noise_2d_gradient(uint32 state, int x0, int y0, int x1, int y1, float dx, float dy)
{
    vec2 d00 = vec2(dx, dy);
    vec2 gradient_v00 = random_gradient_2d(state, x0, y0);
    float v00 = dot(d00, gradient_v00);

    vec2 d10 = vec2(dx-1.f, dy);
    vec2 gradient_v10 = random_gradient_2d(state, x1, y0);
    float v10 = dot(d10, gradient_v10);

    vec2 d01 = vec2(dx, dy-1.f);
    vec2 gradient_v01 = random_gradient_2d(state, x0, y1);
    float v01 = dot(d01, gradient_v01);

    vec2 d11 = vec2(dx-1.f, dy-1.f);
    vec2 gradient_v11 = random_gradient_2d(state, x1, y1);
    float v11 = dot(d11, gradient_v11);

    vec2 dx_dt = vec2(1.f, 0.f);

    float xi0 = interpolate(v00, v10, dx);
    vec2 gradient_xi0 = interpolate_gradient_2d(v00, v10, dx, gradient_v00, gradient_v10, dx_dt);

    float xi1 = interpolate(v01, v11, dx);
    vec2 gradient_xi1 = interpolate_gradient_2d(v01, v11, dx, gradient_v01, gradient_v11, dx_dt);

    vec2 dy_dt = vec2(0.f, 1.f);

    vec2 gradient = interpolate_gradient_2d(xi0, xi1, dy, gradient_xi0, gradient_xi1, dy_dt);

    return gradient;
}

inline CUDA_CALLABLE float noise_3d(uint32 state, int x0, int y0, int z0, int x1, int y1, int z1, float dx, float dy, float dz)
{
    //vXYZ
    float v000 = dot_grid_gradient_3d(state, x0, y0, z0, dx, dy, dz);
    float v100 = dot_grid_gradient_3d(state, x1, y0, z0, dx-1.f, dy, dz);
    float xi00 = interpolate(v000, v100, dx);

    float v010 = dot_grid_gradient_3d(state, x0, y1, z0, dx, dy-1.f, dz);
    float v110 = dot_grid_gradient_3d(state, x1, y1, z0, dx-1.f, dy-1.f, dz);
    float xi10 = interpolate(v010, v110, dx);

    float yi0 = interpolate(xi00, xi10, dy);

    float v001 = dot_grid_gradient_3d(state, x0, y0, z1, dx, dy, dz-1.f);
    float v101 = dot_grid_gradient_3d(state, x1, y0, z1, dx-1.f, dy, dz-1.f);
    float xi01 = interpolate(v001, v101, dx);

    float v011 = dot_grid_gradient_3d(state, x0, y1, z1, dx, dy-1.f, dz-1.f);
    float v111 = dot_grid_gradient_3d(state, x1, y1, z1, dx-1.f, dy-1.f, dz-1.f);
    float xi11 = interpolate(v011, v111, dx);

    float yi1 = interpolate(xi01, xi11, dy);

    return interpolate(yi0, yi1, dz);
}

inline CUDA_CALLABLE vec3 noise_3d_gradient(uint32 state, int x0, int y0, int z0, int x1, int y1, int z1, float dx, float dy, float dz)
{
    vec3 d000 = vec3(dx, dy, dz);
    vec3 gradient_v000 = random_gradient_3d(state, x0, y0, z0);
    float v000 = dot(d000, gradient_v000);

    vec3 d100 = vec3(dx-1.f, dy, dz);
    vec3 gradient_v100 = random_gradient_3d(state, x1, y0, z0);
    float v100 = dot(d100, gradient_v100);

    vec3 d010 = vec3(dx, dy-1.f, dz);
    vec3 gradient_v010 = random_gradient_3d(state, x0, y1, z0);
    float v010 = dot(d010, gradient_v010);
    
    vec3 d110 = vec3(dx-1.f, dy-1.f, dz);
    vec3 gradient_v110 = random_gradient_3d(state, x1, y1, z0);
    float v110 = dot(d110, gradient_v110);

    vec3 d001 = vec3(dx, dy, dz-1.f);
    vec3 gradient_v001 = random_gradient_3d(state, x0, y0, z1);
    float v001 = dot(d001, gradient_v001);

    vec3 d101 = vec3(dx-1.f, dy, dz-1.f);
    vec3 gradient_v101 = random_gradient_3d(state, x1, y0, z1);
    float v101 = dot(d101, gradient_v101);

    vec3 d011 = vec3(dx, dy-1.f, dz-1.f);
    vec3 gradient_v011 = random_gradient_3d(state, x0, y1, z1);
    float v011 = dot(d011, gradient_v011);

    vec3 d111 = vec3(dx-1.f, dy-1.f, dz-1.f);
    vec3 gradient_v111 = random_gradient_3d(state, x1, y1, z1);
    float v111 = dot(d111, gradient_v111);

    vec3 dx_dt = vec3(1.f, 0.f, 0.f);

    float xi00 = interpolate(v000, v100, dx);
    vec3 gradient_xi00 = interpolate_gradient_3d(v000, v100, dx, gradient_v000, gradient_v100, dx_dt);

    float xi10 = interpolate(v010, v110, dx);
    vec3 gradient_xi10 = interpolate_gradient_3d(v010, v110, dx, gradient_v010, gradient_v110, dx_dt);
    
    float xi01 = interpolate(v001, v101, dx);
    vec3 gradient_xi01 = interpolate_gradient_3d(v001, v101, dx, gradient_v001, gradient_v101, dx_dt);

    float xi11 = interpolate(v011, v111, dx);
    vec3 gradient_xi11 = interpolate_gradient_3d(v011, v111, dx, gradient_v011, gradient_v111, dx_dt);

    vec3 dy_dt = vec3(0.f, 1.f, 0.f);

    float yi0 = interpolate(xi00, xi10, dy);
    vec3 gradient_yi0 = interpolate_gradient_3d(xi00, xi10, dy, gradient_xi00, gradient_xi10, dy_dt);

    float yi1 = interpolate(xi01, xi11, dy);    
    vec3 gradient_yi1 = interpolate_gradient_3d(xi01, xi11, dy, gradient_xi01, gradient_xi11, dy_dt);

    vec3 dz_dt = vec3(0.f, 0.f, 1.f);

    vec3 gradient = interpolate_gradient_3d(yi0, yi1, dz, gradient_yi0, gradient_yi1, dz_dt);

    return gradient;
}

inline CUDA_CALLABLE float noise_4d(uint32 state, int x0, int y0, int z0, int t0, int x1, int y1, int z1, int t1, float dx, float dy, float dz, float dt)
{
    //vXYZT
    float v0000 = dot_grid_gradient_4d(state, x0, y0, z0, t0, dx, dy, dz, dt);
    float v1000 = dot_grid_gradient_4d(state, x1, y0, z0, t0, dx-1.f, dy, dz, dt);
    float xi000 = interpolate(v0000, v1000, dx);

    float v0100 = dot_grid_gradient_4d(state, x0, y1, z0, t0, dx, dy-1.f, dz, dt);
    float v1100 = dot_grid_gradient_4d(state, x1, y1, z0, t0, dx-1.f, dy-1.f, dz, dt);
    float xi100 = interpolate(v0100, v1100, dx);

    float yi00 = interpolate(xi000, xi100, dy);

    float v0010 = dot_grid_gradient_4d(state, x0, y0, z1, t0, dx, dy, dz-1.f, dt);
    float v1010 = dot_grid_gradient_4d(state, x1, y0, z1, t0, dx-1.f, dy, dz-1.f, dt);
    float xi010 = interpolate(v0010, v1010, dx);

    float v0110 = dot_grid_gradient_4d(state, x0, y1, z1, t0, dx, dy-1.f, dz-1.f, dt);
    float v1110 = dot_grid_gradient_4d(state, x1, y1, z1, t0, dx-1.f, dy-1.f, dz-1.f, dt);
    float xi110 = interpolate(v0110, v1110, dx);

    float yi10 = interpolate(xi010, xi110, dy);

    float zi0 = interpolate(yi00, yi10, dz);

    float v0001 = dot_grid_gradient_4d(state, x0, y0, z0, t1, dx, dy, dz, dt-1.f);
    float v1001 = dot_grid_gradient_4d(state, x1, y0, z0, t1, dx-1.f, dy, dz, dt-1.f);
    float xi001 = interpolate(v0001, v1001, dx);

    float v0101 = dot_grid_gradient_4d(state, x0, y1, z0, t1, dx, dy-1.f, dz, dt-1.f);
    float v1101 = dot_grid_gradient_4d(state, x1, y1, z0, t1, dx-1.f, dy-1.f, dz, dt-1.f);
    float xi101 = interpolate(v0101, v1101, dx);

    float yi01 = interpolate(xi001, xi101, dy);

    float v0011 = dot_grid_gradient_4d(state, x0, y0, z1, t1, dx, dy, dz-1.f, dt-1.f);
    float v1011 = dot_grid_gradient_4d(state, x1, y0, z1, t1, dx-1.f, dy, dz-1.f, dt-1.f);
    float xi011 = interpolate(v0011, v1011, dx);

    float v0111 = dot_grid_gradient_4d(state, x0, y1, z1, t1, dx, dy-1.f, dz-1.f, dt-1.f);
    float v1111 = dot_grid_gradient_4d(state, x1, y1, z1, t1, dx-1.f, dy-1.f, dz-1.f, dt-1.f);
    float xi111 = interpolate(v0111, v1111, dx);

    float yi11 = interpolate(xi011, xi111, dy);

    float zi1 = interpolate(yi01, yi11, dz);

    return interpolate(zi0, zi1, dt);
}

inline CUDA_CALLABLE vec4 noise_4d_gradient(uint32 state, int x0, int y0, int z0, int t0, int x1, int y1, int z1, int t1, float dx, float dy, float dz, float dt)
{
    vec4 d0000 = vec4(dx, dy, dz, dt);
    vec4 gradient_v0000 = random_gradient_4d(state, x0, y0, z0, t0);
    float v0000 = dot(d0000, gradient_v0000);

    vec4 d1000 = vec4(dx-1.f, dy, dz, dt);
    vec4 gradient_v1000 = random_gradient_4d(state, x1, y0, z0, t0);
    float v1000 = dot(d1000, gradient_v1000);

    vec4 d0100 = vec4(dx, dy-1.f, dz, dt);
    vec4 gradient_v0100 = random_gradient_4d(state, x0, y1, z0, t0);
    float v0100 = dot(d0100, gradient_v0100);

    vec4 d1100 = vec4(dx-1.f, dy-1.f, dz, dt);
    vec4 gradient_v1100 = random_gradient_4d(state, x1, y1, z0, t0);
    float v1100 = dot(d1100, gradient_v1100);

    vec4 d0010 = vec4(dx, dy, dz-1.f, dt);
    vec4 gradient_v0010 = random_gradient_4d(state, x0, y0, z1, t0);
    float v0010 = dot(d0010, gradient_v0010);

    vec4 d1010 = vec4(dx-1.f, dy, dz-1.f, dt);
    vec4 gradient_v1010 = random_gradient_4d(state, x1, y0, z1, t0);
    float v1010 = dot(d1010, gradient_v1010);

    vec4 d0110 = vec4(dx, dy-1.f, dz-1.f, dt);
    vec4 gradient_v0110 = random_gradient_4d(state, x0, y1, z1, t0);
    float v0110 = dot(d0110, gradient_v0110);
    
    vec4 d1110 = vec4(dx-1.f, dy-1.f, dz-1.f, dt);
    vec4 gradient_v1110 = random_gradient_4d(state, x1, y1, z1, t0);
    float v1110 = dot(d1110, gradient_v1110);

    vec4 d0001 = vec4(dx, dy, dz, dt-1.f);
    vec4 gradient_v0001 = random_gradient_4d(state, x0, y0, z0, t1);
    float v0001 = dot(d0001, gradient_v0001);

    vec4 d1001 = vec4(dx-1.f, dy, dz, dt-1.f);
    vec4 gradient_v1001 = random_gradient_4d(state, x1, y0, z0, t1);
    float v1001 = dot(d1001, gradient_v1001);

    vec4 d0101 = vec4(dx, dy-1.f, dz, dt-1.f);
    vec4 gradient_v0101 = random_gradient_4d(state, x0, y1, z0, t1);
    float v0101 = dot(d0101, gradient_v0101);

    vec4 d1101 = vec4(dx-1.f, dy-1.f, dz, dt-1.f);
    vec4 gradient_v1101 = random_gradient_4d(state, x1, y1, z0, t1);
    float v1101 = dot(d1101, gradient_v1101);

    vec4 d0011 = vec4(dx, dy, dz-1.f, dt-1.f);
    vec4 gradient_v0011 = random_gradient_4d(state, x0, y0, z1, t1);
    float v0011 = dot(d0011, gradient_v0011);

    vec4 d1011 = vec4(dx-1.f, dy, dz-1.f, dt-1.f);
    vec4 gradient_v1011 = random_gradient_4d(state, x1, y0, z1, t1);
    float v1011 = dot(d1011, gradient_v1011);

    vec4 d0111 = vec4(dx, dy-1.f, dz-1.f, dt-1.f);
    vec4 gradient_v0111 = random_gradient_4d(state, x0, y1, z1, t1);
    float v0111 = dot(d0111, gradient_v0111);
    
    vec4 d1111 = vec4(dx-1.f, dy-1.f, dz-1.f, dt-1.f);
    vec4 gradient_v1111 = random_gradient_4d(state, x1, y1, z1, t1);
    float v1111 = dot(d1111, gradient_v1111);

    vec4 dx_dt = vec4(1.f, 0.f, 0.f, 0.f);

    float xi000 = interpolate(v0000, v1000, dx);
    vec4 gradient_xi000 = interpolate_gradient_4d(v0000, v1000, dx, gradient_v0000, gradient_v1000, dx_dt);

    float xi100 = interpolate(v0100, v1100, dx);
    vec4 gradient_xi100 = interpolate_gradient_4d(v0100, v1100, dx, gradient_v0100, gradient_v1100, dx_dt);

    float xi010 = interpolate(v0010, v1010, dx);
    vec4 gradient_xi010 = interpolate_gradient_4d(v0010, v1010, dx, gradient_v0010, gradient_v1010, dx_dt);

    float xi110 = interpolate(v0110, v1110, dx);
    vec4 gradient_xi110 = interpolate_gradient_4d(v0110, v1110, dx, gradient_v0110, gradient_v1110, dx_dt);

    float xi001 = interpolate(v0001, v1001, dx);
    vec4 gradient_xi001 = interpolate_gradient_4d(v0001, v1001, dx, gradient_v0001, gradient_v1001, dx_dt);

    float xi101 = interpolate(v0101, v1101, dx);
    vec4 gradient_xi101 = interpolate_gradient_4d(v0101, v1101, dx, gradient_v0101, gradient_v1101, dx_dt);

    float xi011 = interpolate(v0011, v1011, dx);
    vec4 gradient_xi011 = interpolate_gradient_4d(v0011, v1011, dx, gradient_v0011, gradient_v1011, dx_dt);
    
    float xi111 = interpolate(v0111, v1111, dx);
    vec4 gradient_xi111 = interpolate_gradient_4d(v0111, v1111, dx, gradient_v0111, gradient_v1111, dx_dt);
    
    vec4 dy_dt = vec4(0.f, 1.f, 0.f, 0.f);

    float yi00 = interpolate(xi000, xi100, dy);
    vec4 gradient_yi00 = interpolate_gradient_4d(xi000, xi100, dy, gradient_xi000, gradient_xi100, dy_dt);

    float yi10 = interpolate(xi010, xi110, dy);
    vec4 gradient_yi10 = interpolate_gradient_4d(xi010, xi110, dy, gradient_xi010, gradient_xi110, dy_dt);

    float yi01 = interpolate(xi001, xi101, dy);
    vec4 gradient_yi01 = interpolate_gradient_4d(xi001, xi101, dy, gradient_xi001, gradient_xi101, dy_dt);

    float yi11 = interpolate(xi011, xi111, dy);
    vec4 gradient_yi11 = interpolate_gradient_4d(xi011, xi111, dy, gradient_xi011, gradient_xi111, dy_dt);

    vec4 dz_dt = vec4(0.f, 0.f, 1.f, 0.f);

    float zi0 = interpolate(yi00, yi10, dz);
    vec4 gradient_zi0 = interpolate_gradient_4d(yi00, yi10, dz, gradient_yi00, gradient_yi10, dz_dt);

    float zi1 = interpolate(yi01, yi11, dz);
    vec4 gradient_zi1 = interpolate_gradient_4d(yi01, yi11, dz, gradient_yi01, gradient_yi11, dz_dt);

    vec4 dt_dt = vec4(0.f, 0.f, 0.f, 1.f);

    vec4 gradient = interpolate_gradient_4d(zi0, zi1, dt, gradient_zi0, gradient_zi1, dt_dt);

    return gradient;
}

// non-periodic Perlin noise

inline CUDA_CALLABLE float noise(uint32 state, float x)
{
    float dx = x - floor(x);

    int x0 = (int)floor(x);
    int x1 = x0 + 1;

    return noise_1d(state, x0, x1, dx);
}

inline CUDA_CALLABLE void adj_noise(uint32 state, float x, uint32& adj_state, float& adj_x, const float adj_ret)
{
    float dx = x - floor(x);

    int x0 = (int)floor(x);
    int x1 = x0 + 1;

    float gradient = noise_1d_gradient(state, x0, x1, dx);
    adj_x += gradient * adj_ret;
}

inline CUDA_CALLABLE float noise(uint32 state, const vec2& xy)
{
    float dx = xy[0] - floor(xy[0]);
    float dy = xy[1] - floor(xy[1]);

    int x0 = (int)floor(xy[0]); 
    int y0 = (int)floor(xy[1]); 

    int x1 = x0 + 1;
    int y1 = y0 + 1;

    return noise_2d(state, x0, y0, x1, y1, dx, dy);
}

inline CUDA_CALLABLE void adj_noise(uint32 state, const vec2& xy, uint32& adj_state, vec2& adj_xy, const float adj_ret)
{
    float dx = xy[0] - floor(xy[0]);
    float dy = xy[1] - floor(xy[1]);

    int x0 = (int)floor(xy[0]); 
    int y0 = (int)floor(xy[1]); 

    int x1 = x0 + 1;
    int y1 = y0 + 1;

    vec2 gradient = noise_2d_gradient(state, x0, y0, x1, y1, dx, dy);

    adj_xy[0] += gradient[0] * adj_ret;
    adj_xy[1] += gradient[1] * adj_ret;
}

inline CUDA_CALLABLE float noise(uint32 state, const vec3& xyz)
{
    float dx = xyz[0] - floor(xyz[0]);
    float dy = xyz[1] - floor(xyz[1]);
    float dz = xyz[2] - floor(xyz[2]);

    int x0 = (int)floor(xyz[0]);
    int y0 = (int)floor(xyz[1]);
    int z0 = (int)floor(xyz[2]);

    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    return noise_3d(state, x0, y0, z0, x1, y1, z1, dx, dy, dz);
}

inline CUDA_CALLABLE void adj_noise(uint32 state, const vec3& xyz, uint32& adj_state, vec3& adj_xyz, const float adj_ret)
{
    float dx = xyz[0] - floor(xyz[0]);
    float dy = xyz[1] - floor(xyz[1]);
    float dz = xyz[2] - floor(xyz[2]);

    int x0 = (int)floor(xyz[0]);
    int y0 = (int)floor(xyz[1]);
    int z0 = (int)floor(xyz[2]);

    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    vec3 gradient = noise_3d_gradient(state, x0, y0, z0, x1, y1, z1, dx, dy, dz);
    adj_xyz[0] += gradient[0] * adj_ret;
    adj_xyz[1] += gradient[1] * adj_ret;
    adj_xyz[2] += gradient[2] * adj_ret;
}

inline CUDA_CALLABLE float noise(uint32 state, const vec4& xyzt)
{
    float dx = xyzt[0] - floor(xyzt[0]);
    float dy = xyzt[1] - floor(xyzt[1]);
    float dz = xyzt[2] - floor(xyzt[2]);
    float dt = xyzt[3] - floor(xyzt[3]);

    int x0 = (int)floor(xyzt[0]);
    int y0 = (int)floor(xyzt[1]);
    int z0 = (int)floor(xyzt[2]);
    int t0 = (int)floor(xyzt[3]);

    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;
    int t1 = t0 + 1;

    return noise_4d(state, x0, y0, z0, t0, x1, y1, z1, t1, dx, dy, dz, dt);
}

inline CUDA_CALLABLE void adj_noise(uint32 state, const vec4& xyzt, uint32& adj_state, vec4& adj_xyzt, const float adj_ret)
{
    float dx = xyzt[0] - floor(xyzt[0]);
    float dy = xyzt[1] - floor(xyzt[1]);
    float dz = xyzt[2] - floor(xyzt[2]);
    float dt = xyzt[3] - floor(xyzt[3]);

    int x0 = (int)floor(xyzt[0]);
    int y0 = (int)floor(xyzt[1]);
    int z0 = (int)floor(xyzt[2]);
    int t0 = (int)floor(xyzt[3]);

    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;
    int t1 = t0 + 1;

    vec4 gradient = noise_4d_gradient(state, x0, y0, z0, t0, x1, y1, z1, t1, dx, dy, dz, dt);

    adj_xyzt[0] += gradient[0] * adj_ret;
    adj_xyzt[1] += gradient[1] * adj_ret;
    adj_xyzt[2] += gradient[2] * adj_ret;
    adj_xyzt[3] += gradient[3] * adj_ret;
}

// periodic Perlin noise

inline CUDA_CALLABLE float pnoise(uint32 state, float x, int px)
{
    float dx = x - floor(x);

    int x0 = mod(((int)floor(x)), px);
    int x1 = mod((x0 + 1), px);

    return noise_1d(state, x0, x1, dx);
}

inline CUDA_CALLABLE void adj_pnoise(uint32 state, float x, int px, uint32& adj_state, float& adj_x, int& adj_px, const float adj_ret)
{
    float dx = x - floor(x);

    int x0 = mod(((int)floor(x)), px);
    int x1 = mod((x0 + 1), px);

    float gradient = noise_1d_gradient(state, x0, x1, dx);
    adj_x += gradient * adj_ret;
}

inline CUDA_CALLABLE float pnoise(uint32 state, const vec2& xy, int px, int py)
{
    float dx = xy[0] - floor(xy[0]);
    float dy = xy[1] - floor(xy[1]);

    int x0 = mod(((int)floor(xy[0])), px); 
    int y0 = mod(((int)floor(xy[1])), py); 

    int x1 = mod((x0 + 1), px);
    int y1 = mod((y0 + 1), py);

    return noise_2d(state, x0, y0, x1, y1, dx, dy);
}

inline CUDA_CALLABLE void adj_pnoise(uint32 state, const vec2& xy, int px, int py, uint32& adj_state, vec2& adj_xy, int& adj_px, int& adj_py, const float adj_ret)
{
    float dx = xy[0] - floor(xy[0]);
    float dy = xy[1] - floor(xy[1]);

    int x0 = mod(((int)floor(xy[0])), px); 
    int y0 = mod(((int)floor(xy[1])), py); 

    int x1 = mod((x0 + 1), px);
    int y1 = mod((y0 + 1), py);

    vec2 gradient = noise_2d_gradient(state, x0, y0, x1, y1, dx, dy);

    adj_xy[0] += gradient[0] * adj_ret;
    adj_xy[1] += gradient[1] * adj_ret;
}

inline CUDA_CALLABLE float pnoise(uint32 state, const vec3& xyz, int px, int py, int pz)
{
    float dx = xyz[0] - floor(xyz[0]);
    float dy = xyz[1] - floor(xyz[1]);
    float dz = xyz[2] - floor(xyz[2]);

    int x0 = mod(((int)floor(xyz[0])), px); 
    int y0 = mod(((int)floor(xyz[1])), py); 
    int z0 = mod(((int)floor(xyz[2])), pz); 

    int x1 = mod((x0 + 1), px);
    int y1 = mod((y0 + 1), py);
    int z1 = mod((z0 + 1), pz);

    return noise_3d(state, x0, y0, z0, x1, y1, z1, dx, dy, dz);
}

inline CUDA_CALLABLE void adj_pnoise(uint32 state, const vec3& xyz, int px, int py, int pz, uint32& adj_state, vec3& adj_xyz, int& adj_px, int& adj_py, int& adj_pz, const float adj_ret)
{
    float dx = xyz[0] - floor(xyz[0]);
    float dy = xyz[1] - floor(xyz[1]);
    float dz = xyz[2] - floor(xyz[2]);

    int x0 = mod(((int)floor(xyz[0])), px); 
    int y0 = mod(((int)floor(xyz[1])), py); 
    int z0 = mod(((int)floor(xyz[2])), pz); 

    int x1 = mod((x0 + 1), px);
    int y1 = mod((y0 + 1), py);
    int z1 = mod((z0 + 1), pz);

    vec3 gradient = noise_3d_gradient(state, x0, y0, z0, x1, y1, z1, dx, dy, dz);
    adj_xyz[0] += gradient[0] * adj_ret;
    adj_xyz[1] += gradient[1] * adj_ret;
    adj_xyz[2] += gradient[2] * adj_ret;
}

inline CUDA_CALLABLE float pnoise(uint32 state, const vec4& xyzt, int px, int py, int pz, int pt)
{
    float dx = xyzt[0] - floor(xyzt[0]);
    float dy = xyzt[1] - floor(xyzt[1]);
    float dz = xyzt[2] - floor(xyzt[2]);
    float dt = xyzt[3] - floor(xyzt[3]);

    int x0 = mod(((int)floor(xyzt[0])), px);
    int y0 = mod(((int)floor(xyzt[1])), py);
    int z0 = mod(((int)floor(xyzt[2])), pz);
    int t0 = mod(((int)floor(xyzt[3])), pt);

    int x1 = mod((x0 + 1), px);
    int y1 = mod((y0 + 1), py);
    int z1 = mod((z0 + 1), pz);
    int t1 = mod((t0 + 1), pt);

    return noise_4d(state, x0, y0, z0, t0, x1, y1, z1, t1, dx, dy, dz, dt);
}

inline CUDA_CALLABLE void adj_pnoise(uint32 state, const vec4& xyzt, int px, int py, int pz, int pt, uint32& adj_state, vec4& adj_xyzt, int& adj_px, int& adj_py, int& adj_pz, int& adj_pt, const float adj_ret)
{
    float dx = xyzt[0] - floor(xyzt[0]);
    float dy = xyzt[1] - floor(xyzt[1]);
    float dz = xyzt[2] - floor(xyzt[2]);
    float dt = xyzt[3] - floor(xyzt[3]);

    int x0 = mod(((int)floor(xyzt[0])), px);
    int y0 = mod(((int)floor(xyzt[1])), py);
    int z0 = mod(((int)floor(xyzt[2])), pz);
    int t0 = mod(((int)floor(xyzt[3])), pt);

    int x1 = mod((x0 + 1), px);
    int y1 = mod((y0 + 1), py);
    int z1 = mod((z0 + 1), pz);
    int t1 = mod((t0 + 1), pt);

    vec4 gradient = noise_4d_gradient(state, x0, y0, z0, t0, x1, y1, z1, t1, dx, dy, dz, dt);

    adj_xyzt[0] += gradient[0] * adj_ret;
    adj_xyzt[1] += gradient[1] * adj_ret;
    adj_xyzt[2] += gradient[2] * adj_ret;
    adj_xyzt[3] += gradient[3] * adj_ret;
}

// curl noise

inline CUDA_CALLABLE vec2 curlnoise(uint32 state, const vec2& xy, const uint32 octaves, const float lacunarity, const float gain)
{
    vec2 curl_sum = vec2(0.f);
    float freq = 1.f;
    float amplitude = 1.f;

    for (int i = 0; i < octaves; i++)
    {
        vec2 pt = freq * xy;
        float dx = pt[0] - floor(pt[0]);
        float dy = pt[1] - floor(pt[1]);

        int x0 = (int)floor(pt[0]); 
        int y0 = (int)floor(pt[1]); 

        int x1 = x0 + 1;
        int y1 = y0 + 1;

        vec2 grad_field = noise_2d_gradient(state, x0, y0, x1, y1, dx, dy);
        curl_sum += amplitude * grad_field;

        amplitude *= gain;
        freq *= lacunarity;
    }
    return vec2(-curl_sum[1], curl_sum[0]);
}
inline CUDA_CALLABLE void adj_curlnoise(uint32 state, const vec2& xy, const uint32 octaves, const float lacunarity, const float gain, uint32& adj_state, vec2& adj_xy, const uint32& adj_octaves, const float& adj_lacunarity, const float& adj_gain, const vec2& adj_ret) {}

inline CUDA_CALLABLE vec3 curlnoise(uint32 state, const vec3& xyz, const uint32 octaves, const float lacunarity, const float gain)
{
    vec3 curl_sum_1 = vec3(0.f);
    vec3 curl_sum_2 = vec3(0.f);
    vec3 curl_sum_3 = vec3(0.f);
    
    float freq = 1.f;
    float amplitude = 1.f;
    
    for(int i = 0; i < octaves; i++)
    {
        vec3 pt = freq * xyz;
        float dx = pt[0] - floor(pt[0]);
        float dy = pt[1] - floor(pt[1]);
        float dz = pt[2] - floor(pt[2]);

        int x0 = (int)floor(pt[0]);
        int y0 = (int)floor(pt[1]);
        int z0 = (int)floor(pt[2]);

        int x1 = x0 + 1;
        int y1 = y0 + 1;
        int z1 = z0 + 1;

        vec3 grad_field_1 = noise_3d_gradient(state, x0, y0, z0, x1, y1, z1, dx, dy, dz);
        state = rand_init(state, 10019689);
        vec3 grad_field_2 = noise_3d_gradient(state, x0, y0, z0, x1, y1, z1, dx, dy, dz);
        state = rand_init(state, 13112221);
        vec3 grad_field_3 = noise_3d_gradient(state, x0, y0, z0, x1, y1, z1, dx, dy, dz);
        
        curl_sum_1 += amplitude * grad_field_1;
        curl_sum_2 += amplitude * grad_field_2;
        curl_sum_3 += amplitude * grad_field_3;

        amplitude *= gain;
        freq *= lacunarity;
    }
    
    return vec3(
        curl_sum_3[1] - curl_sum_2[2],
        curl_sum_1[2] - curl_sum_3[0],
        curl_sum_2[0] - curl_sum_1[1]);
}
inline CUDA_CALLABLE void adj_curlnoise(uint32 state, const vec3& xyz, const uint32 octaves, const float lacunarity, const float gain, uint32& adj_state, vec3& adj_xyz, const uint32& adj_octaves, const float& adj_lacunarity, const float& adj_gain, vec3& adj_ret) {}

inline CUDA_CALLABLE vec3 curlnoise(uint32 state, const vec4& xyzt, const uint32 octaves, const float lacunarity, const float gain)
{
    vec4 curl_sum_1 = vec4(0.f);
    vec4 curl_sum_2 = vec4(0.f);
    vec4 curl_sum_3 = vec4(0.f);
    
    float freq = 1.f;
    float amplitude = 1.f;

    for(int i = 0; i < octaves; i++)
    {
        vec4 pt = freq * xyzt;
        float dx = pt[0] - floor(pt[0]);
        float dy = pt[1] - floor(pt[1]);
        float dz = pt[2] - floor(pt[2]);
        float dt = pt[3] - floor(pt[3]);

        int x0 = (int)floor(pt[0]);
        int y0 = (int)floor(pt[1]);
        int z0 = (int)floor(pt[2]);
        int t0 = (int)floor(pt[3]);

        int x1 = x0 + 1;
        int y1 = y0 + 1;
        int z1 = z0 + 1;
        int t1 = t0 + 1;

        vec4 grad_field_1 = noise_4d_gradient(state, x0, y0, z0, t0, x1, y1, z1, t1, dx, dy, dz, dt);
        state = rand_init(state, 10019689);
        vec4 grad_field_2 = noise_4d_gradient(state, x0, y0, z0, t0, x1, y1, z1, t1, dx, dy, dz, dt);
        state = rand_init(state, 13112221);
        vec4 grad_field_3 = noise_4d_gradient(state, x0, y0, z0, t0, x1, y1, z1, t1, dx, dy, dz, dt);

        curl_sum_1 += amplitude * grad_field_1;
        curl_sum_2 += amplitude * grad_field_2;
        curl_sum_3 += amplitude * grad_field_3;

        amplitude *= gain;
        freq *= lacunarity;        
    }

    return vec3(
        curl_sum_3[1] - curl_sum_2[2],
        curl_sum_1[2] - curl_sum_3[0],
        curl_sum_2[0] - curl_sum_1[1]);
}
inline CUDA_CALLABLE void adj_curlnoise(uint32 state, const vec4& xyzt, const uint32 octaves, const float lacunarity, const float gain, uint32& adj_state, vec4& adj_xyzt, const uint32& adj_octaves, const float& adj_lacunarity, const float& adj_gain, const vec3& adj_ret) {}

} // namespace wp
