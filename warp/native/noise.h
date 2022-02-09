#pragma once

#ifndef M_PI
#define M_PI 3.14159265358979323846f
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

inline CUDA_CALLABLE float interpolate(float a0, float a1, float t)
{
    return (a1 - a0) * smootherstep(t) + a0;
}

inline CUDA_CALLABLE float interpolate_gradient(float a0, float a1, float t, float d_a0, float d_a1, float d_t)
{
    return (d_a1 - d_a0) * smootherstep(t) + (a1 - a0) * smootherstep_gradient(t) * d_t + d_a0;
}

inline CUDA_CALLABLE float random_gradient_1d(uint32 seed, int ix)
{
    uint32 state = seed + uint32(ix);
    return randf(state, -1.f, 1.f);
}

inline CUDA_CALLABLE vec2 random_gradient_2d(uint32 seed, int ix, int iy, int px)
{
    int idx = ix + px * iy;
    uint32 state = seed + uint32(idx);
    float phi = randf(state, 0.f, 2.f*M_PI);
    float x = cos(phi);
    float y = sin(phi);
    return vec2(x, y);
}

inline CUDA_CALLABLE vec3 random_gradient_3d(uint32 seed, int ix, int iy, int iz, int px, int py)
{
    int idx = ix + px * (iy + py * iz);
    uint32 state = seed + uint32(idx);
    float theta = randf(state, 0.f, M_PI);
    float phi = randf(state, 0.f, 2.f*M_PI);
    float x = sin(theta) * cos(phi);
    float y = sin(theta) * sin(phi);
    float z = cos(theta);
    return vec3(x, y, z);
}

inline CUDA_CALLABLE vec4 random_gradient_4d(uint32 seed, int ix, int iy, int iz, int it, int px, int py, int pz)
{
    int idx = ix + px * (iy + py * (iz + it * pz));
    uint32 state = seed + uint32(idx);
    float psi = randf(state, 0.f, M_PI);
    float theta = randf(state, 0.f, M_PI);
    float phi = randf(state, 0.f, 2.f*M_PI);
    float x = sin(psi) * sin(theta) * cos(phi);
    float y = sin(psi) * sin(theta) * sin(phi);
    float z = sin(psi) * cos(theta);
    float t = cos(psi);
    return vec4(x, y, z, t);
}

inline CUDA_CALLABLE float dot_grid_gradient_1d(uint32 seed, int ix, float dx)
{
    float gradient = random_gradient_1d(seed, ix);
    return dx*gradient;
}

inline CUDA_CALLABLE float dot_grid_gradient_1d_gradient(uint32 seed, int ix, float d_dx)
{
    float gradient = random_gradient_1d(seed, ix);
    return d_dx*gradient;
}

inline CUDA_CALLABLE float dot_grid_gradient_2d(uint32 seed, int ix, int iy, float dx, float dy, int px)
{
    vec2 gradient = random_gradient_2d(seed, ix, iy, px);
    return (dx*gradient.x + dy*gradient.y);
}

inline CUDA_CALLABLE float dot_grid_gradient_2d_gradient(uint32 seed, int ix, int iy, float d_dx, float d_dy, int px)
{
    vec2 gradient = random_gradient_2d(seed, ix, iy, px);
    return (d_dx*gradient.x + d_dy*gradient.y);
}

inline CUDA_CALLABLE float dot_grid_gradient_3d(uint32 seed, int ix, int iy, int iz, float dx, float dy, float dz, int px, int py)
{
    vec3 gradient = random_gradient_3d(seed, ix, iy, iz, px, py);
    return (dx*gradient.x + dy*gradient.y + dz*gradient.z);
}

inline CUDA_CALLABLE float dot_grid_gradient_3d_gradient(uint32 seed, int ix, int iy, int iz, float d_dx, float d_dy, float d_dz, int px, int py)
{
    vec3 gradient = random_gradient_3d(seed, ix, iy, iz, px, py);
    return (d_dx*gradient.x + d_dy*gradient.y + d_dz*gradient.z);
}

inline CUDA_CALLABLE float dot_grid_gradient_4d(uint32 seed, int ix, int iy, int iz, int it, float dx, float dy, float dz, float dt, int px, int py, int pz)
{
    vec4 gradient = random_gradient_4d(seed, ix, iy, iz, it, px, py, pz);
    return (dx*gradient.x + dy*gradient.y + dz*gradient.z + dt*gradient.w);
}

inline CUDA_CALLABLE float dot_grid_gradient_4d_gradient(uint32 seed, int ix, int iy, int iz, int it, float d_dx, float d_dy, float d_dz, float d_dt, int px, int py, int pz)
{
    vec4 gradient = random_gradient_4d(seed, ix, iy, iz, it, px, py, pz);
    return (d_dx*gradient.x + d_dy*gradient.y + d_dz*gradient.z + d_dt*gradient.w);    
}

inline CUDA_CALLABLE float pnoise(uint32 seed, float x, int px)
{
    float dx = x - floor(x);

    int x0 = mod(((int)floor(x)), px);
    int x1 = mod((x0 + 1), px);

    float v0 = dot_grid_gradient_1d(seed, x0, dx);
    float v1 = dot_grid_gradient_1d(seed, x1, dx-1.f);

    return interpolate(v0, v1, dx);
}

inline CUDA_CALLABLE void adj_pnoise(uint32 seed, float x, int px, uint32& adj_seed, float& adj_x, int& adj_px, const float adj_ret)
{
    float dx = x - floor(x);

    float heaviside_x = 1.f;
    if (dx < _EPSILON) heaviside_x = 0.f;

    int x0 = mod(((int)floor(x)), px);
    int x1 = mod((x0 + 1), px);

    float v0 = dot_grid_gradient_1d(seed, x0, dx);
    float d_v0_dx = dot_grid_gradient_1d_gradient(seed, x0, heaviside_x);

    float v1 = dot_grid_gradient_1d(seed, x1, dx-1.f);
    float d_v1_dx = dot_grid_gradient_1d_gradient(seed, x1, heaviside_x);

    adj_x += interpolate_gradient(v0, v1, dx, d_v0_dx, d_v1_dx, heaviside_x) * adj_ret;
}

inline CUDA_CALLABLE float pnoise(uint32 seed, const vec2& xy, int px, int py)
{
    float dx = xy.x - floor(xy.x);
    float dy = xy.y - floor(xy.y);

    int x0 = mod(((int)floor(xy.x)), px); 
    int y0 = mod(((int)floor(xy.y)), py); 

    int x1 = mod((x0 + 1), px);
    int y1 = mod((y0 + 1), py);

    //vXY
    float v00 = dot_grid_gradient_2d(seed, x0, y0, dx, dy, px);
    float v10 = dot_grid_gradient_2d(seed, x1, y0, dx-1.f, dy, px);
    float xi0 = interpolate(v00, v10, dx);

    float v01 = dot_grid_gradient_2d(seed, x0, y1, dx, dy-1.f, px);
    float v11 = dot_grid_gradient_2d(seed, x1, y1, dx-1.f, dy-1.f, px);
    float xi1 = interpolate(v01, v11, dx);

    return interpolate(xi0, xi1, dy);
}

inline CUDA_CALLABLE void adj_pnoise(uint32 seed, const vec2& xy, int px, int py, uint32& adj_seed, vec2& adj_xy, int& adj_px, int& adj_py, const float adj_ret)
{
    float dx = xy.x - floor(xy.x);
    float dy = xy.y - floor(xy.y);

    float heaviside_x = 1.f;
    float heaviside_y = 1.f;
    if (dx < _EPSILON) heaviside_x = 0.f;
    if (dy < _EPSILON) heaviside_y = 0.f;

    int x0 = mod(((int)floor(xy.x)), px); 
    int y0 = mod(((int)floor(xy.y)), py); 

    int x1 = mod((x0 + 1), px);
    int y1 = mod((y0 + 1), py);

    float v00 = dot_grid_gradient_2d(seed, x0, y0, dx, dy, px);
    float d_v00_dx = dot_grid_gradient_2d_gradient(seed, x0, y0, heaviside_x, 0.f, px);
    float d_v00_dy = dot_grid_gradient_2d_gradient(seed, x0, y0, 0.0, heaviside_y, px);
    
    float v10 = dot_grid_gradient_2d(seed, x1, y0, dx-1.f, dy, px);
    float d_v10_dx = dot_grid_gradient_2d_gradient(seed, x1, y0, heaviside_x, 0.f, px);
    float d_v10_dy = dot_grid_gradient_2d_gradient(seed, x1, y0, 0.0, heaviside_y, px);

    float v01 = dot_grid_gradient_2d(seed, x0, y1, dx, dy-1.f, px);
    float d_v01_dx = dot_grid_gradient_2d_gradient(seed, x0, y1, heaviside_x, 0.f, px);
    float d_v01_dy = dot_grid_gradient_2d_gradient(seed, x0, y1, 0.0, heaviside_y, px);

    float v11 = dot_grid_gradient_2d(seed, x1, y1, dx-1.f, dy-1.f, px);
    float d_v11_dx = dot_grid_gradient_2d_gradient(seed, x1, y1, heaviside_x, 0.f, px);
    float d_v11_dy = dot_grid_gradient_2d_gradient(seed, x1, y1, 0.0, heaviside_y, px);

    float xi0 = interpolate(v00, v10, dx);
    float d_xi0_dx = interpolate_gradient(v00, v10, dx, d_v00_dx, d_v10_dx, heaviside_x);
    float d_xi0_dy = interpolate_gradient(v00, v10, dx, d_v00_dy, d_v10_dy, 0.0);

    float xi1 = interpolate(v01, v11, dx);
    float d_xi1_dx = interpolate_gradient(v01, v11, dx, d_v01_dx, d_v11_dx, heaviside_x);
    float d_xi1_dy = interpolate_gradient(v01, v11, dx, d_v01_dy, d_v11_dy, 0.0);

    adj_xy.x += interpolate_gradient(xi0, xi1, dy, d_xi0_dx, d_xi1_dx, 0.0) * adj_ret;
    adj_xy.y += interpolate_gradient(xi0, xi1, dy, d_xi0_dy, d_xi1_dy, heaviside_y) * adj_ret;
}

inline CUDA_CALLABLE float pnoise(uint32 seed, const vec3& xyz, int px, int py, int pz)
{
    float dx = xyz.x - floor(xyz.x);
    float dy = xyz.y - floor(xyz.y);
    float dz = xyz.z - floor(xyz.z);

    int x0 = mod(((int)floor(xyz.x)), px); 
    int y0 = mod(((int)floor(xyz.y)), py); 
    int z0 = mod(((int)floor(xyz.z)), pz); 

    int x1 = mod((x0 + 1), px);
    int y1 = mod((y0 + 1), py);
    int z1 = mod((z0 + 1), pz);

    //vXYZ
    float v000 = dot_grid_gradient_3d(seed, x0, y0, z0, dx, dy, dz, px, py);
    float v100 = dot_grid_gradient_3d(seed, x1, y0, z0, dx-1.f, dy, dz, px, py);
    float xi00 = interpolate(v000, v100, dx);

    float v010 = dot_grid_gradient_3d(seed, x0, y1, z0, dx, dy-1.f, dz, px, py);
    float v110 = dot_grid_gradient_3d(seed, x1, y1, z0, dx-1.f, dy-1.f, dz, px, py);
    float xi10 = interpolate(v010, v110, dx);

    float yi0 = interpolate(xi00, xi10, dy);

    float v001 = dot_grid_gradient_3d(seed, x0, y0, z1, dx, dy, dz-1.f, px, py);
    float v101 = dot_grid_gradient_3d(seed, x1, y0, z1, dx-1.f, dy, dz-1.f, px, py);
    float xi01 = interpolate(v001, v101, dx);

    float v011 = dot_grid_gradient_3d(seed, x0, y1, z1, dx, dy-1.f, dz-1.f, px, py);
    float v111 = dot_grid_gradient_3d(seed, x1, y1, z1, dx-1.f, dy-1.f, dz-1.f, px, py);
    float xi11 = interpolate(v011, v111, dx);

    float yi1 = interpolate(xi01, xi11, dy);

    return interpolate(yi0, yi1, dz);
}

inline CUDA_CALLABLE void adj_pnoise(uint32 seed, const vec3& xyz, int px, int py, int pz, uint32& adj_seed, vec3& adj_xyz, int& adj_px, int& adj_py, int& adj_pz, const float adj_ret)
{
    float dx = xyz.x - floor(xyz.x);
    float dy = xyz.y - floor(xyz.y);
    float dz = xyz.z - floor(xyz.z);

    float heaviside_x = 1.f;
    float heaviside_y = 1.f;
    float heaviside_z = 1.f;
    if (dx < _EPSILON) heaviside_x = 0.f;
    if (dy < _EPSILON) heaviside_y = 0.f;
    if (dz < _EPSILON) heaviside_z = 0.f;

    int x0 = mod(((int)floor(xyz.x)), px); 
    int y0 = mod(((int)floor(xyz.y)), py); 
    int z0 = mod(((int)floor(xyz.z)), pz); 

    int x1 = mod((x0 + 1), px);
    int y1 = mod((y0 + 1), py);
    int z1 = mod((z0 + 1), pz);

    float v000 = dot_grid_gradient_3d(seed, x0, y0, z0, dx, dy, dz, px, py);
    float d_v000_dx = dot_grid_gradient_3d_gradient(seed, x0, y0, z0, heaviside_x, 0.f, 0.f, px, py);
    float d_v000_dy = dot_grid_gradient_3d_gradient(seed, x0, y0, z0, 0.f, heaviside_y, 0.f, px, py);
    float d_v000_dz = dot_grid_gradient_3d_gradient(seed, x0, y0, z0, 0.f, 0.f, heaviside_z, px, py);

    float v100 = dot_grid_gradient_3d(seed, x1, y0, z0, dx-1.f, dy, dz, px, py);
    float d_v100_dx = dot_grid_gradient_3d_gradient(seed, x1, y0, z0, heaviside_x, 0.f, 0.f, px, py);
    float d_v100_dy = dot_grid_gradient_3d_gradient(seed, x1, y0, z0, 0.f, heaviside_y, 0.f, px, py);
    float d_v100_dz = dot_grid_gradient_3d_gradient(seed, x1, y0, z0, 0.f, 0.f, heaviside_z, px, py);

    float v010 = dot_grid_gradient_3d(seed, x0, y1, z0, dx, dy-1.f, dz, px, py);
    float d_v010_dx = dot_grid_gradient_3d_gradient(seed, x0, y1, z0, heaviside_x, 0.f, 0.f, px, py);
    float d_v010_dy = dot_grid_gradient_3d_gradient(seed, x0, y1, z0, 0.f, heaviside_y, 0.f, px, py);
    float d_v010_dz = dot_grid_gradient_3d_gradient(seed, x0, y1, z0, 0.f, 0.f, heaviside_z, px, py);
        
    float v110 = dot_grid_gradient_3d(seed, x1, y1, z0, dx-1.f, dy-1.f, dz, px, py);
    float d_v110_dx = dot_grid_gradient_3d_gradient(seed, x1, y1, z0, heaviside_x, 0.f, 0.f, px, py);
    float d_v110_dy = dot_grid_gradient_3d_gradient(seed, x1, y1, z0, 0.f, heaviside_y, 0.f, px, py);
    float d_v110_dz = dot_grid_gradient_3d_gradient(seed, x1, y1, z0, 0.f, 0.f, heaviside_z, px, py);

    float v001 = dot_grid_gradient_3d(seed, x0, y0, z1, dx, dy, dz-1.f, px, py);
    float d_v001_dx = dot_grid_gradient_3d_gradient(seed, x0, y0, z1, heaviside_x, 0.f, 0.f, px, py);
    float d_v001_dy = dot_grid_gradient_3d_gradient(seed, x0, y0, z1, 0.f, heaviside_y, 0.f, px, py);
    float d_v001_dz = dot_grid_gradient_3d_gradient(seed, x0, y0, z1, 0.f, 0.f, heaviside_z, px, py);

    float v101 = dot_grid_gradient_3d(seed, x1, y0, z1, dx-1.f, dy, dz-1.f, px, py);
    float d_v101_dx = dot_grid_gradient_3d_gradient(seed, x1, y0, z1, heaviside_x, 0.f, 0.f, px, py);
    float d_v101_dy = dot_grid_gradient_3d_gradient(seed, x1, y0, z1, 0.f, heaviside_y, 0.f, px, py);
    float d_v101_dz = dot_grid_gradient_3d_gradient(seed, x1, y0, z1, 0.f, 0.f, heaviside_z, px, py);

    float v011 = dot_grid_gradient_3d(seed, x0, y1, z1, dx, dy-1.f, dz-1.f, px, py);
    float d_v011_dx = dot_grid_gradient_3d_gradient(seed, x0, y1, z1, heaviside_x, 0.f, 0.f, px, py);
    float d_v011_dy = dot_grid_gradient_3d_gradient(seed, x0, y1, z1, 0.f, heaviside_y, 0.f, px, py);
    float d_v011_dz = dot_grid_gradient_3d_gradient(seed, x0, y1, z1, 0.f, 0.f, heaviside_z, px, py);

    float v111 = dot_grid_gradient_3d(seed, x1, y1, z1, dx-1.f, dy-1.f, dz-1.f, px, py);
    float d_v111_dx = dot_grid_gradient_3d_gradient(seed, x1, y1, z1, heaviside_x, 0.f, 0.f, px, py);
    float d_v111_dy = dot_grid_gradient_3d_gradient(seed, x1, y1, z1, 0.f, heaviside_y, 0.f, px, py);
    float d_v111_dz = dot_grid_gradient_3d_gradient(seed, x1, y1, z1, 0.f, 0.f, heaviside_z, px, py);

    float xi00 = interpolate(v000, v100, dx);
    float d_xi00_dx = interpolate_gradient(v000, v100, dx, d_v000_dx, d_v100_dx, heaviside_x);
    float d_xi00_dy = interpolate_gradient(v000, v100, dx, d_v000_dy, d_v100_dy, 0.f);
    float d_xi00_dz = interpolate_gradient(v000, v100, dx, d_v000_dz, d_v100_dz, 0.f);

    float xi10 = interpolate(v010, v110, dx);
    float d_xi10_dx = interpolate_gradient(v010, v110, dx, d_v010_dx, d_v110_dx, heaviside_x);
    float d_xi10_dy = interpolate_gradient(v010, v110, dx, d_v010_dy, d_v110_dy, 0.f);
    float d_xi10_dz = interpolate_gradient(v010, v110, dx, d_v010_dz, d_v110_dz, 0.f);
    
    float xi01 = interpolate(v001, v101, dx);
    float d_xi01_dx = interpolate_gradient(v001, v101, dx, d_v001_dx, d_v101_dx, heaviside_x);
    float d_xi01_dy = interpolate_gradient(v001, v101, dx, d_v001_dy, d_v101_dy, 0.f);
    float d_xi01_dz = interpolate_gradient(v001, v101, dx, d_v001_dz, d_v101_dz, 0.f);  
    
    float xi11 = interpolate(v011, v111, dx);
    float d_xi11_dx = interpolate_gradient(v011, v111, dx, d_v011_dx, d_v111_dx, heaviside_x);
    float d_xi11_dy = interpolate_gradient(v011, v111, dx, d_v011_dy, d_v111_dy, 0.f);
    float d_xi11_dz = interpolate_gradient(v011, v111, dx, d_v011_dz, d_v111_dz, 0.f);

    float yi0 = interpolate(xi00, xi10, dy);
    float d_yi0_dx = interpolate_gradient(xi00, xi10, dy, d_xi00_dx, d_xi10_dx, 0.f);
    float d_yi0_dy = interpolate_gradient(xi00, xi10, dy, d_xi00_dy, d_xi10_dy, heaviside_y);
    float d_yi0_dz = interpolate_gradient(xi00, xi10, dy, d_xi00_dz, d_xi10_dz, 0.f);

    float yi1 = interpolate(xi01, xi11, dy);    
    float d_yi1_dx = interpolate_gradient(xi01, xi11, dy, d_xi01_dx, d_xi11_dx, 0.f);
    float d_yi1_dy = interpolate_gradient(xi01, xi11, dy, d_xi01_dy, d_xi11_dy, heaviside_y);
    float d_yi1_dz = interpolate_gradient(xi01, xi11, dy, d_xi01_dz, d_xi11_dz, 0.f);

    adj_xyz.y += interpolate_gradient(yi0, yi1, dz, d_yi0_dy, d_yi1_dy, 0.f) * adj_ret;
    adj_xyz.x += interpolate_gradient(yi0, yi1, dz, d_yi0_dx, d_yi1_dx, 0.f) * adj_ret;
    adj_xyz.z += interpolate_gradient(yi0, yi1, dz, d_yi0_dz, d_yi1_dz, heaviside_z) * adj_ret;
}

inline CUDA_CALLABLE float pnoise(uint32 seed, const vec4& xyzt, int px, int py, int pz, int pt)
{
    float dx = xyzt.x - floor(xyzt.x);
    float dy = xyzt.y - floor(xyzt.y);
    float dz = xyzt.z - floor(xyzt.z);
    float dt = xyzt.w - floor(xyzt.w);

    int x0 = mod(((int)floor(xyzt.x)), px);
    int y0 = mod(((int)floor(xyzt.y)), py);
    int z0 = mod(((int)floor(xyzt.z)), pz);
    int t0 = mod(((int)floor(xyzt.w)), pt);

    int x1 = mod((x0 + 1), px);
    int y1 = mod((y0 + 1), py);
    int z1 = mod((z0 + 1), pz);
    int t1 = mod((t0 + 1), pt);

    //vXYZT
    float v0000 = dot_grid_gradient_4d(seed, x0, y0, z0, t0, dx, dy, dz, dt, px, py, pz);
    float v1000 = dot_grid_gradient_4d(seed, x1, y0, z0, t0, dx-1.f, dy, dz, dt, px, py, pz);
    float xi000 = interpolate(v0000, v1000, dx);

    float v0100 = dot_grid_gradient_4d(seed, x0, y1, z0, t0, dx, dy-1.f, dz, dt, px, py, pz);
    float v1100 = dot_grid_gradient_4d(seed, x1, y1, z0, t0, dx-1.f, dy-1.f, dz, dt, px, py, pz);
    float xi100 = interpolate(v0100, v1100, dx);

    float yi00 = interpolate(xi000, xi100, dy);

    float v0010 = dot_grid_gradient_4d(seed, x0, y0, z1, t0, dx, dy, dz-1.f, dt, px, py, pz);
    float v1010 = dot_grid_gradient_4d(seed, x1, y0, z1, t0, dx-1.f, dy, dz-1.f, dt, px, py, pz);
    float xi010 = interpolate(v0010, v1010, dx);

    float v0110 = dot_grid_gradient_4d(seed, x0, y1, z1, t0, dx, dy-1.f, dz-1.f, dt, px, py, pz);
    float v1110 = dot_grid_gradient_4d(seed, x1, y1, z1, t0, dx-1.f, dy-1.f, dz-1.f, dt, px, py, pz);
    float xi110 = interpolate(v0110, v1110, dx);

    float yi10 = interpolate(xi010, xi110, dy);

    float zi0 = interpolate(yi00, yi10, dz);

    float v0001 = dot_grid_gradient_4d(seed, x0, y0, z0, t1, dx, dy, dz, dt-1.f, px, py, pz);
    float v1001 = dot_grid_gradient_4d(seed, x1, y0, z0, t1, dx-1.f, dy, dz, dt-1.f, px, py, pz);
    float xi001 = interpolate(v0001, v1001, dx);

    float v0101 = dot_grid_gradient_4d(seed, x0, y1, z0, t1, dx, dy-1.f, dz, dt-1.f, px, py, pz);
    float v1101 = dot_grid_gradient_4d(seed, x1, y1, z0, t1, dx-1.f, dy-1.f, dz, dt-1.f, px, py, pz);
    float xi101 = interpolate(v0101, v1101, dx);

    float yi01 = interpolate(xi001, xi101, dy);

    float v0011 = dot_grid_gradient_4d(seed, x0, y0, z1, t1, dx, dy, dz-1.f, dt-1.f, px, py, pz);
    float v1011 = dot_grid_gradient_4d(seed, x1, y0, z1, t1, dx-1.f, dy, dz-1.f, dt-1.f, px, py, pz);
    float xi011 = interpolate(v0011, v1011, dx);

    float v0111 = dot_grid_gradient_4d(seed, x0, y1, z1, t1, dx, dy-1.f, dz-1.f, dt-1.f, px, py, pz);
    float v1111 = dot_grid_gradient_4d(seed, x1, y1, z1, t1, dx-1.f, dy-1.f, dz-1.f, dt-1.f, px, py, pz);
    float xi111 = interpolate(v0111, v1111, dx);

    float yi11 = interpolate(xi011, xi111, dy);

    float zi1 = interpolate(yi01, yi11, dz);

    return interpolate(zi0, zi1, dt);
}

inline CUDA_CALLABLE void adj_pnoise(uint32 seed, const vec4& xyzt, int px, int py, int pz, int pt, uint32& adj_seed, vec4& adj_xyzt, int& adj_px, int& adj_py, int& adj_pz, int& adj_pt, const float adj_ret)
{
    float dx = xyzt.x - floor(xyzt.x);
    float dy = xyzt.y - floor(xyzt.y);
    float dz = xyzt.z - floor(xyzt.z);
    float dt = xyzt.w - floor(xyzt.w);

    float heaviside_x = 1.f;
    float heaviside_y = 1.f;
    float heaviside_z = 1.f;
    float heaviside_t = 1.f;
    if (dx < _EPSILON) heaviside_x = 0.f;
    if (dy < _EPSILON) heaviside_y = 0.f;
    if (dz < _EPSILON) heaviside_z = 0.f;
    if (dt < _EPSILON) heaviside_t = 0.f;

    int x0 = mod(((int)floor(xyzt.x)), px);
    int y0 = mod(((int)floor(xyzt.y)), py);
    int z0 = mod(((int)floor(xyzt.z)), pz);
    int t0 = mod(((int)floor(xyzt.w)), pt);

    int x1 = mod((x0 + 1), px);
    int y1 = mod((y0 + 1), py);
    int z1 = mod((z0 + 1), pz);
    int t1 = mod((t0 + 1), pt);

    float v0000 = dot_grid_gradient_4d(seed, x0, y0, z0, t0, dx, dy, dz, dt, px, py, pz);
    float d_v0000_dx = dot_grid_gradient_4d_gradient(seed, x0, y0, z0, t0, heaviside_x, 0.f, 0.f, 0.f, px, py, pz);
    float d_v0000_dy = dot_grid_gradient_4d_gradient(seed, x0, y0, z0, t0, 0.f, heaviside_y, 0.f, 0.f, px, py, pz);
    float d_v0000_dz = dot_grid_gradient_4d_gradient(seed, x0, y0, z0, t0, 0.f, 0.f, heaviside_z, 0.f, px, py, pz);
    float d_v0000_dt = dot_grid_gradient_4d_gradient(seed, x0, y0, z0, t0, 0.f, 0.f, 0.f, heaviside_t, px, py, pz);

    float v1000 = dot_grid_gradient_4d(seed, x1, y0, z0, t0, dx-1.f, dy, dz, dt, px, py, pz);
    float d_v1000_dx = dot_grid_gradient_4d_gradient(seed, x1, y0, z0, t0, heaviside_x, 0.f, 0.f, 0.f, px, py, pz);
    float d_v1000_dy = dot_grid_gradient_4d_gradient(seed, x1, y0, z0, t0, 0.f, heaviside_y, 0.f, 0.f, px, py, pz);
    float d_v1000_dz = dot_grid_gradient_4d_gradient(seed, x1, y0, z0, t0, 0.f, 0.f, heaviside_z, 0.f, px, py, pz);
    float d_v1000_dt = dot_grid_gradient_4d_gradient(seed, x1, y0, z0, t0, 0.f, 0.f, 0.f, heaviside_t, px, py, pz);

    float v0100 = dot_grid_gradient_4d(seed, x0, y1, z0, t0, dx, dy-1.f, dz, dt, px, py, pz);
    float d_v0100_dx = dot_grid_gradient_4d_gradient(seed, x0, y1, z0, t0, heaviside_x, 0.f, 0.f, 0.f, px, py, pz);
    float d_v0100_dy = dot_grid_gradient_4d_gradient(seed, x0, y1, z0, t0, 0.f, heaviside_y, 0.f, 0.f, px, py, pz);
    float d_v0100_dz = dot_grid_gradient_4d_gradient(seed, x0, y1, z0, t0, 0.f, 0.f, heaviside_z, 0.f, px, py, pz);
    float d_v0100_dt = dot_grid_gradient_4d_gradient(seed, x0, y1, z0, t0, 0.f, 0.f, 0.f, heaviside_t, px, py, pz);

    float v1100 = dot_grid_gradient_4d(seed, x1, y1, z0, t0, dx-1.f, dy-1.f, dz, dt, px, py, pz);
    float d_v1100_dx = dot_grid_gradient_4d_gradient(seed, x1, y1, z0, t0, heaviside_x, 0.f, 0.f, 0.f, px, py, pz);
    float d_v1100_dy = dot_grid_gradient_4d_gradient(seed, x1, y1, z0, t0, 0.f, heaviside_y, 0.f, 0.f, px, py, pz);
    float d_v1100_dz = dot_grid_gradient_4d_gradient(seed, x1, y1, z0, t0, 0.f, 0.f, heaviside_z, 0.f, px, py, pz);
    float d_v1100_dt = dot_grid_gradient_4d_gradient(seed, x1, y1, z0, t0, 0.f, 0.f, 0.f, heaviside_t, px, py, pz);

    float v0010 = dot_grid_gradient_4d(seed, x0, y0, z1, t0, dx, dy, dz-1.f, dt, px, py, pz);
    float d_v0010_dx = dot_grid_gradient_4d_gradient(seed, x0, y0, z1, t0, heaviside_x, 0.f, 0.f, 0.f, px, py, pz);
    float d_v0010_dy = dot_grid_gradient_4d_gradient(seed, x0, y0, z1, t0, 0.f, heaviside_y, 0.f, 0.f, px, py, pz);
    float d_v0010_dz = dot_grid_gradient_4d_gradient(seed, x0, y0, z1, t0, 0.f, 0.f, heaviside_z, 0.f, px, py, pz);
    float d_v0010_dt = dot_grid_gradient_4d_gradient(seed, x0, y0, z1, t0, 0.f, 0.f, 0.f, heaviside_t, px, py, pz);

    float v1010 = dot_grid_gradient_4d(seed, x1, y0, z1, t0, dx-1.f, dy, dz-1.f, dt, px, py, pz);
    float d_v1010_dx = dot_grid_gradient_4d_gradient(seed, x1, y0, z1, t0, heaviside_x, 0.f, 0.f, 0.f, px, py, pz);
    float d_v1010_dy = dot_grid_gradient_4d_gradient(seed, x1, y0, z1, t0, 0.f, heaviside_y, 0.f, 0.f, px, py, pz);
    float d_v1010_dz = dot_grid_gradient_4d_gradient(seed, x1, y0, z1, t0, 0.f, 0.f, heaviside_z, 0.f, px, py, pz);
    float d_v1010_dt = dot_grid_gradient_4d_gradient(seed, x1, y0, z1, t0, 0.f, 0.f, 0.f, heaviside_t, px, py, pz);
    
    float v0110 = dot_grid_gradient_4d(seed, x0, y1, z1, t0, dx, dy-1.f, dz-1.f, dt, px, py, pz);
    float d_v0110_dx = dot_grid_gradient_4d_gradient(seed, x0, y1, z1, t0, heaviside_x, 0.f, 0.f, 0.f, px, py, pz);
    float d_v0110_dy = dot_grid_gradient_4d_gradient(seed, x0, y1, z1, t0, 0.f, heaviside_y, 0.f, 0.f, px, py, pz);
    float d_v0110_dz = dot_grid_gradient_4d_gradient(seed, x0, y1, z1, t0, 0.f, 0.f, heaviside_z, 0.f, px, py, pz);
    float d_v0110_dt = dot_grid_gradient_4d_gradient(seed, x0, y1, z1, t0, 0.f, 0.f, 0.f, heaviside_t, px, py, pz);

    float v1110 = dot_grid_gradient_4d(seed, x1, y1, z1, t0, dx-1.f, dy-1.f, dz-1.f, dt, px, py, pz);
    float d_v1110_dx = dot_grid_gradient_4d_gradient(seed, x1, y1, z1, t0, heaviside_x, 0.f, 0.f, 0.f, px, py, pz);
    float d_v1110_dy = dot_grid_gradient_4d_gradient(seed, x1, y1, z1, t0, 0.f, heaviside_y, 0.f, 0.f, px, py, pz);
    float d_v1110_dz = dot_grid_gradient_4d_gradient(seed, x1, y1, z1, t0, 0.f, 0.f, heaviside_z, 0.f, px, py, pz);
    float d_v1110_dt = dot_grid_gradient_4d_gradient(seed, x1, y1, z1, t0, 0.f, 0.f, 0.f, heaviside_t, px, py, pz);
    
    float v0001 = dot_grid_gradient_4d(seed, x0, y0, z0, t1, dx, dy, dz, dt-1.f, px, py, pz);
    float d_v0001_dx = dot_grid_gradient_4d_gradient(seed, x0, y0, z0, t1, heaviside_x, 0.f, 0.f, 0.f, px, py, pz);
    float d_v0001_dy = dot_grid_gradient_4d_gradient(seed, x0, y0, z0, t1, 0.f, heaviside_y, 0.f, 0.f, px, py, pz);
    float d_v0001_dz = dot_grid_gradient_4d_gradient(seed, x0, y0, z0, t1, 0.f, 0.f, heaviside_z, 0.f, px, py, pz);
    float d_v0001_dt = dot_grid_gradient_4d_gradient(seed, x0, y0, z0, t1, 0.f, 0.f, 0.f, heaviside_t, px, py, pz);

    float v1001 = dot_grid_gradient_4d(seed, x1, y0, z0, t1, dx-1.f, dy, dz, dt-1.f, px, py, pz);
    float d_v1001_dx = dot_grid_gradient_4d_gradient(seed, x1, y0, z0, t1, heaviside_x, 0.f, 0.f, 0.f, px, py, pz);
    float d_v1001_dy = dot_grid_gradient_4d_gradient(seed, x1, y0, z0, t1, 0.f, heaviside_y, 0.f, 0.f, px, py, pz);
    float d_v1001_dz = dot_grid_gradient_4d_gradient(seed, x1, y0, z0, t1, 0.f, 0.f, heaviside_z, 0.f, px, py, pz);
    float d_v1001_dt = dot_grid_gradient_4d_gradient(seed, x1, y0, z0, t1, 0.f, 0.f, 0.f, heaviside_t, px, py, pz);
    
    float v0101 = dot_grid_gradient_4d(seed, x0, y1, z0, t1, dx, dy-1.f, dz, dt-1.f, px, py, pz);
    float d_v0101_dx = dot_grid_gradient_4d_gradient(seed, x0, y1, z0, t1, heaviside_x, 0.f, 0.f, 0.f, px, py, pz);
    float d_v0101_dy = dot_grid_gradient_4d_gradient(seed, x0, y1, z0, t1, 0.f, heaviside_y, 0.f, 0.f, px, py, pz);
    float d_v0101_dz = dot_grid_gradient_4d_gradient(seed, x0, y1, z0, t1, 0.f, 0.f, heaviside_z, 0.f, px, py, pz);
    float d_v0101_dt = dot_grid_gradient_4d_gradient(seed, x0, y1, z0, t1, 0.f, 0.f, 0.f, heaviside_t, px, py, pz);

    float v1101 = dot_grid_gradient_4d(seed, x1, y1, z0, t1, dx-1.f, dy-1.f, dz, dt-1.f, px, py, pz);
    float d_v1101_dx = dot_grid_gradient_4d_gradient(seed, x1, y1, z0, t1, heaviside_x, 0.f, 0.f, 0.f, px, py, pz);
    float d_v1101_dy = dot_grid_gradient_4d_gradient(seed, x1, y1, z0, t1, 0.f, heaviside_y, 0.f, 0.f, px, py, pz);
    float d_v1101_dz = dot_grid_gradient_4d_gradient(seed, x1, y1, z0, t1, 0.f, 0.f, heaviside_z, 0.f, px, py, pz);
    float d_v1101_dt = dot_grid_gradient_4d_gradient(seed, x1, y1, z0, t1, 0.f, 0.f, 0.f, heaviside_t, px, py, pz);
    
    float v0011 = dot_grid_gradient_4d(seed, x0, y0, z1, t1, dx, dy, dz-1.f, dt-1.f, px, py, pz);
    float d_v0011_dx = dot_grid_gradient_4d_gradient(seed, x0, y0, z1, t1, heaviside_x, 0.f, 0.f, 0.f, px, py, pz);
    float d_v0011_dy = dot_grid_gradient_4d_gradient(seed, x0, y0, z1, t1, 0.f, heaviside_y, 0.f, 0.f, px, py, pz);
    float d_v0011_dz = dot_grid_gradient_4d_gradient(seed, x0, y0, z1, t1, 0.f, 0.f, heaviside_z, 0.f, px, py, pz);
    float d_v0011_dt = dot_grid_gradient_4d_gradient(seed, x0, y0, z1, t1, 0.f, 0.f, 0.f, heaviside_t, px, py, pz);

    float v1011 = dot_grid_gradient_4d(seed, x1, y0, z1, t1, dx-1.f, dy, dz-1.f, dt-1.f, px, py, pz);
    float d_v1011_dx = dot_grid_gradient_4d_gradient(seed, x1, y0, z1, t1, heaviside_x, 0.f, 0.f, 0.f, px, py, pz);
    float d_v1011_dy = dot_grid_gradient_4d_gradient(seed, x1, y0, z1, t1, 0.f, heaviside_y, 0.f, 0.f, px, py, pz);
    float d_v1011_dz = dot_grid_gradient_4d_gradient(seed, x1, y0, z1, t1, 0.f, 0.f, heaviside_z, 0.f, px, py, pz);
    float d_v1011_dt = dot_grid_gradient_4d_gradient(seed, x1, y0, z1, t1, 0.f, 0.f, 0.f, heaviside_t, px, py, pz);

    float v0111 = dot_grid_gradient_4d(seed, x0, y1, z1, t1, dx, dy-1.f, dz-1.f, dt-1.f, px, py, pz);
    float d_v0111_dx = dot_grid_gradient_4d_gradient(seed, x0, y1, z1, t1, heaviside_x, 0.f, 0.f, 0.f, px, py, pz);
    float d_v0111_dy = dot_grid_gradient_4d_gradient(seed, x0, y1, z1, t1, 0.f, heaviside_y, 0.f, 0.f, px, py, pz);
    float d_v0111_dz = dot_grid_gradient_4d_gradient(seed, x0, y1, z1, t1, 0.f, 0.f, heaviside_z, 0.f, px, py, pz);
    float d_v0111_dt = dot_grid_gradient_4d_gradient(seed, x0, y1, z1, t1, 0.f, 0.f, 0.f, heaviside_t, px, py, pz);

    float v1111 = dot_grid_gradient_4d(seed, x1, y1, z1, t1, dx-1.f, dy-1.f, dz-1.f, dt-1.f, px, py, pz);
    float d_v1111_dx = dot_grid_gradient_4d_gradient(seed, x1, y1, z1, t1, heaviside_x, 0.f, 0.f, 0.f, px, py, pz);
    float d_v1111_dy = dot_grid_gradient_4d_gradient(seed, x1, y1, z1, t1, 0.f, heaviside_y, 0.f, 0.f, px, py, pz);
    float d_v1111_dz = dot_grid_gradient_4d_gradient(seed, x1, y1, z1, t1, 0.f, 0.f, heaviside_z, 0.f, px, py, pz);
    float d_v1111_dt = dot_grid_gradient_4d_gradient(seed, x1, y1, z1, t1, 0.f, 0.f, 0.f, heaviside_t, px, py, pz);

    float xi000 = interpolate(v0000, v1000, dx);
    float d_xi000_dx = interpolate_gradient(v0000, v1000, dx, d_v0000_dx, d_v1000_dx, heaviside_x);
    float d_xi000_dy = interpolate_gradient(v0000, v1000, dx, d_v0000_dy, d_v1000_dy, 0.f);
    float d_xi000_dz = interpolate_gradient(v0000, v1000, dx, d_v0000_dz, d_v1000_dz, 0.f);
    float d_xi000_dt = interpolate_gradient(v0000, v1000, dx, d_v0000_dt, d_v1000_dt, 0.f);

    float xi100 = interpolate(v0100, v1100, dx);
    float d_xi100_dx = interpolate_gradient(v0100, v1100, dx, d_v0100_dx, d_v1100_dx, heaviside_x);
    float d_xi100_dy = interpolate_gradient(v0100, v1100, dx, d_v0100_dy, d_v1100_dy, 0.f);
    float d_xi100_dz = interpolate_gradient(v0100, v1100, dx, d_v0100_dz, d_v1100_dz, 0.f);
    float d_xi100_dt = interpolate_gradient(v0100, v1100, dx, d_v0100_dt, d_v1100_dt, 0.f);

    float xi010 = interpolate(v0010, v1010, dx);
    float d_xi010_dx = interpolate_gradient(v0010, v1010, dx, d_v0010_dx, d_v1010_dx, heaviside_x);
    float d_xi010_dy = interpolate_gradient(v0010, v1010, dx, d_v0010_dy, d_v1010_dy, 0.f);
    float d_xi010_dz = interpolate_gradient(v0010, v1010, dx, d_v0010_dz, d_v1010_dz, 0.f);
    float d_xi010_dt = interpolate_gradient(v0010, v1010, dx, d_v0010_dt, d_v1010_dt, 0.f);

    float xi110 = interpolate(v0110, v1110, dx);
    float d_xi110_dx = interpolate_gradient(v0110, v1110, dx, d_v0110_dx, d_v1110_dx, heaviside_x);
    float d_xi110_dy = interpolate_gradient(v0110, v1110, dx, d_v0110_dy, d_v1110_dy, 0.f);
    float d_xi110_dz = interpolate_gradient(v0110, v1110, dx, d_v0110_dz, d_v1110_dz, 0.f);
    float d_xi110_dt = interpolate_gradient(v0110, v1110, dx, d_v0110_dt, d_v1110_dt, 0.f);

    float xi001 = interpolate(v0001, v1001, dx);
    float d_xi001_dx = interpolate_gradient(v0001, v1001, dx, d_v0001_dx, d_v1001_dx, heaviside_x);
    float d_xi001_dy = interpolate_gradient(v0001, v1001, dx, d_v0001_dy, d_v1001_dy, 0.f);
    float d_xi001_dz = interpolate_gradient(v0001, v1001, dx, d_v0001_dz, d_v1001_dz, 0.f);
    float d_xi001_dt = interpolate_gradient(v0001, v1001, dx, d_v0001_dt, d_v1001_dt, 0.f);

    float xi101 = interpolate(v0101, v1101, dx);
    float d_xi101_dx = interpolate_gradient(v0101, v1101, dx, d_v0101_dx, d_v1101_dx, heaviside_x);
    float d_xi101_dy = interpolate_gradient(v0101, v1101, dx, d_v0101_dy, d_v1101_dy, 0.f);
    float d_xi101_dz = interpolate_gradient(v0101, v1101, dx, d_v0101_dz, d_v1101_dz, 0.f);
    float d_xi101_dt = interpolate_gradient(v0101, v1101, dx, d_v0101_dt, d_v1101_dt, 0.f);

    float xi011 = interpolate(v0011, v1011, dx);
    float d_xi011_dx = interpolate_gradient(v0011, v1011, dx, d_v0011_dx, d_v1011_dx, heaviside_x);
    float d_xi011_dy = interpolate_gradient(v0011, v1011, dx, d_v0011_dy, d_v1011_dy, 0.f);
    float d_xi011_dz = interpolate_gradient(v0011, v1011, dx, d_v0011_dz, d_v1011_dz, 0.f);
    float d_xi011_dt = interpolate_gradient(v0011, v1011, dx, d_v0011_dt, d_v1011_dt, 0.f);

    float xi111 = interpolate(v0111, v1111, dx);
    float d_xi111_dx = interpolate_gradient(v0111, v1111, dx, d_v0111_dx, d_v1111_dx, heaviside_x);
    float d_xi111_dy = interpolate_gradient(v0111, v1111, dx, d_v0111_dy, d_v1111_dy, 0.f);
    float d_xi111_dz = interpolate_gradient(v0111, v1111, dx, d_v0111_dz, d_v1111_dz, 0.f);
    float d_xi111_dt = interpolate_gradient(v0111, v1111, dx, d_v0111_dt, d_v1111_dt, 0.f);

    float yi00 = interpolate(xi000, xi100, dy);
    float d_yi00_dx = interpolate_gradient(xi000, xi100, dy, d_xi000_dx, d_xi100_dx, 0.f);
    float d_yi00_dy = interpolate_gradient(xi000, xi100, dy, d_xi000_dy, d_xi100_dy, heaviside_y);
    float d_yi00_dz = interpolate_gradient(xi000, xi100, dy, d_xi000_dz, d_xi100_dz, 0.f);
    float d_yi00_dt = interpolate_gradient(xi000, xi100, dy, d_xi000_dt, d_xi100_dt, 0.f);

    float yi10 = interpolate(xi010, xi110, dy);
    float d_yi10_dx = interpolate_gradient(xi010, xi110, dy, d_xi010_dx, d_xi110_dx, 0.f);
    float d_yi10_dy = interpolate_gradient(xi010, xi110, dy, d_xi010_dy, d_xi110_dy, heaviside_y);
    float d_yi10_dz = interpolate_gradient(xi010, xi110, dy, d_xi010_dz, d_xi110_dz, 0.f);
    float d_yi10_dt = interpolate_gradient(xi010, xi110, dy, d_xi010_dt, d_xi110_dt, 0.f);

    float yi01 = interpolate(xi001, xi101, dy);
    float d_yi01_dx = interpolate_gradient(xi001, xi101, dy, d_xi001_dx, d_xi101_dx, 0.f);
    float d_yi01_dy = interpolate_gradient(xi001, xi101, dy, d_xi001_dy, d_xi101_dy, heaviside_y);
    float d_yi01_dz = interpolate_gradient(xi001, xi101, dy, d_xi001_dz, d_xi101_dz, 0.f);
    float d_yi01_dt = interpolate_gradient(xi001, xi101, dy, d_xi001_dt, d_xi101_dt, 0.f);

    float yi11 = interpolate(xi011, xi111, dy);
    float d_yi11_dx = interpolate_gradient(xi011, xi111, dy, d_xi011_dx, d_xi111_dx, 0.f);
    float d_yi11_dy = interpolate_gradient(xi011, xi111, dy, d_xi011_dy, d_xi111_dy, heaviside_y);
    float d_yi11_dz = interpolate_gradient(xi011, xi111, dy, d_xi011_dz, d_xi111_dz, 0.f);
    float d_yi11_dt = interpolate_gradient(xi011, xi111, dy, d_xi011_dt, d_xi111_dt, 0.f);

    float zi0 = interpolate(yi00, yi10, dz);
    float d_zi0_dx = interpolate_gradient(yi00, yi10, dz, d_yi00_dx, d_yi10_dx, 0.f);
    float d_zi0_dy = interpolate_gradient(yi00, yi10, dz, d_yi00_dy, d_yi10_dy, 0.f);
    float d_zi0_dz = interpolate_gradient(yi00, yi10, dz, d_yi00_dz, d_yi10_dz, heaviside_z);
    float d_zi0_dt = interpolate_gradient(yi00, yi10, dz, d_yi00_dt, d_yi10_dt, 0.f);

    float zi1 = interpolate(yi01, yi11, dz);
    float d_zi1_dx = interpolate_gradient(yi01, yi11, dz, d_yi01_dx, d_yi11_dx, 0.f);
    float d_zi1_dy = interpolate_gradient(yi01, yi11, dz, d_yi01_dy, d_yi11_dy, 0.f);
    float d_zi1_dz = interpolate_gradient(yi01, yi11, dz, d_yi01_dz, d_yi11_dz, heaviside_z);
    float d_zi1_dt = interpolate_gradient(yi01, yi11, dz, d_yi01_dt, d_yi11_dt, 0.f);

    adj_xyzt.x += interpolate_gradient(zi0, zi1, dt, d_zi0_dx, d_zi1_dx, 0.f) * adj_ret;
    adj_xyzt.y += interpolate_gradient(zi0, zi1, dt, d_zi0_dy, d_zi1_dy, 0.f) * adj_ret;
    adj_xyzt.z += interpolate_gradient(zi0, zi1, dt, d_zi0_dz, d_zi1_dz, 0.f) * adj_ret;
    adj_xyzt.w += interpolate_gradient(zi0, zi1, dt, d_zi0_dt, d_zi1_dt, heaviside_t) * adj_ret;
}

inline CUDA_CALLABLE vec2 curlnoise(uint32 seed, const vec2& xy) { return vec2(); }
inline CUDA_CALLABLE void adj_curlnoise(uint32 seed, const vec2& xy, uint32& adj_seed, vec2& adj_xy, const float adj_ret) {}

inline CUDA_CALLABLE vec3 curlnoise(uint32 seed, const vec3& xyz) { return vec3(); }
inline CUDA_CALLABLE void adj_curlnoise(uint32 seed, const vec3& xyz, uint32& adj_seed, vec3& adj_xyz, const float adj_ret) {}

inline CUDA_CALLABLE vec3 curlnoise(uint32 seed, const vec4& xyzt) { return vec3(); }
inline CUDA_CALLABLE void adj_curlnoise(uint32 seed, const vec4& xyzt, uint32& adj_seed, vec4& adj_xyzt, const float adj_ret) {}

} // namespace wp
