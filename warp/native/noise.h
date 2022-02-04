#pragma once

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

namespace wp
{

inline CUDA_CALLABLE float interpolate(float a0, float a1, float t)
{
    // Perlin smootherstep
    return (a1 - a0) * ((t * (t * 6.f - 15.f) + 10.f) * t * t * t) + a0;
}

inline CUDA_CALLABLE float randomGradient1D(uint32 seed, int ix)
{
    uint32 state = seed + uint32(ix);
    return randf(state, -1.f, 1.f);
}

inline CUDA_CALLABLE vec2 randomGradient2D(uint32 seed, int ix, int iy, int px)
{
    int idx = ix + px * iy;
    uint32 state = seed + uint32(idx);
    float phi = randf(state, 0.f, 2.f*M_PI);
    float x = cos(phi);
    float y = sin(phi);
    return vec2(x, y);
}

inline CUDA_CALLABLE vec3 randomGradient3D(uint32 seed, int ix, int iy, int iz, int px, int py)
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

inline CUDA_CALLABLE vec4 randomGradient4D(uint32 seed, int ix, int iy, int iz, int it, int px, int py, int pz)
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

inline CUDA_CALLABLE float dotGridGradient1D(uint32 seed, int ix, float dx)
{
    float gradient = randomGradient1D(seed, ix);
    return dx*gradient;
}

inline CUDA_CALLABLE float dotGridGradient2D(uint32 seed, int ix, int iy, float dx, float dy, int px)
{
    vec2 gradient = randomGradient2D(seed, ix, iy, px);
    return (dx*gradient.x + dy*gradient.y);
}

inline CUDA_CALLABLE float dotGridGradient3D(uint32 seed, int ix, int iy, int iz, float dx, float dy, float dz, int px, int py)
{
    vec3 gradient = randomGradient3D(seed, ix, iy, iz, px, py);
    return (dx*gradient.x + dy*gradient.y + dz*gradient.z);
}

inline CUDA_CALLABLE float dotGridGradient4D(uint32 seed, int ix, int iy, int iz, int it, float dx, float dy, float dz, float dt, int px, int py, int pz)
{
    vec4 gradient = randomGradient4D(seed, ix, iy, iz, it, px, py, pz);
    return (dx*gradient.x + dy*gradient.y + dz*gradient.z + dt*gradient.w);
}

inline CUDA_CALLABLE float pnoise(uint32 seed, float x, int px)
{
    float dx = x - floor(x);

    int x0 = mod(((int)floor(x)), px);

    int x1 = mod((x0 + 1), px);

    float v0 = dotGridGradient1D(seed, x0, dx);
    float v1 = dotGridGradient1D(seed, x1, dx-1.f);

    return interpolate(v0, v1, dx);
}

inline CUDA_CALLABLE void adj_pnoise(uint32 seed, float x, int px, uint32& adj_seed, float& adj_x, int& adj_px, const float adj_ret) {}

inline CUDA_CALLABLE float pnoise(uint32 seed, const vec2& xy, int px, int py)
{
    float dx = xy.x - floor(xy.x);
    float dy = xy.y - floor(xy.y);

    int x0 = mod(((int)floor(xy.x)), px); 
    int y0 = mod(((int)floor(xy.y)), py); 

    int x1 = mod((x0 + 1), px);
    int y1 = mod((y0 + 1), py);

    //vXY
    float v00 = dotGridGradient2D(seed, x0, y0, dx, dy, px);
    float v10 = dotGridGradient2D(seed, x1, y0, dx-1.f, dy, px);
    float xi0 = interpolate(v00, v10, dx);

    float v01 = dotGridGradient2D(seed, x0, y1, dx, dy-1.f, px);
    float v11 = dotGridGradient2D(seed, x1, y1, dx-1.f, dy-1.f, px);
    float xi1 = interpolate(v01, v11, dx);

    return interpolate(xi0, xi1, dy);
}

inline CUDA_CALLABLE void adj_pnoise(uint32 seed, const vec2& xy, int px, int py, uint32& adj_seed, const vec2& adj_xy, int& adj_px, int& adj_py, const float adj_ret) {}

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
    float v000 = dotGridGradient3D(seed, x0, y0, z0, dx, dy, dz, px, py);
    float v100 = dotGridGradient3D(seed, x1, y0, z0, dx-1.f, dy, dz, px, py);
    float xi00 = interpolate(v000, v100, dx);

    float v010 = dotGridGradient3D(seed, x0, y1, z0, dx, dy-1.f, dz, px, py);
    float v110 = dotGridGradient3D(seed, x1, y1, z0, dx-1.f, dy-1.f, dz, px, py);
    float xi10 = interpolate(v010, v110, dx);

    float yi0 = interpolate(xi00, xi10, dy);

    float v001 = dotGridGradient3D(seed, x0, y0, z1, dx, dy, dz-1.f, px, py);
    float v101 = dotGridGradient3D(seed, x1, y0, z1, dx-1.f, dy, dz-1.f, px, py);
    float xi01 = interpolate(v001, v101, dx);

    float v011 = dotGridGradient3D(seed, x0, y1, z1, dx, dy-1.f, dz-1.f, px, py);
    float v111 = dotGridGradient3D(seed, x1, y1, z1, dx-1.f, dy-1.f, dz-1.f, px, py);
    float xi11 = interpolate(v011, v111, dx);

    float yi1 = interpolate(xi01, xi11, dy);

    return interpolate(yi0, yi1, dz);
}

inline CUDA_CALLABLE void adj_pnoise(uint32 seed, const vec3& xyz, int px, int py, int pz, uint32& adj_seed, const vec3& adj_xyz, int& adj_px, int& adj_py, int& adj_pz, const float adj_ret) {}

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
    float v0000 = dotGridGradient4D(seed, x0, y0, z0, t0, dx, dy, dz, dt, px, py, pz);
    float v1000 = dotGridGradient4D(seed, x1, y0, z0, t0, dx-1.f, dy, dz, dt, px, py, pz);
    float xi000 = interpolate(v0000, v1000, dx);

    float v0100 = dotGridGradient4D(seed, x0, y1, z0, t0, dx, dy-1.f, dz, dt, px, py, pz);
    float v1100 = dotGridGradient4D(seed, x1, y1, z0, t0, dx-1.f, dy-1.f, dz, dt, px, py, pz);
    float xi100 = interpolate(v0100, v1100, dx);

    float yi00 = interpolate(xi000, xi100, dy);

    float v0010 = dotGridGradient4D(seed, x0, y0, z1, t0, dx, dy, dz-1.f, dt, px, py, pz);
    float v1010 = dotGridGradient4D(seed, x1, y0, z1, t0, dx-1.f, dy, dz-1.f, dt, px, py, pz);
    float xi010 = interpolate(v0010, v1010, dx);

    float v0110 = dotGridGradient4D(seed, x0, y1, z1, t0, dx, dy-1.f, dz-1.f, dt, px, py, pz);
    float v1110 = dotGridGradient4D(seed, x1, y1, z1, t0, dx-1.f, dy-1.f, dz-1.f, dt, px, py, pz);
    float xi110 = interpolate(v0110, v1110, dx);

    float yi10 = interpolate(xi010, xi110, dy);

    float zi0 = interpolate(yi00, yi10, dz);

    float v0001 = dotGridGradient4D(seed, x0, y0, z0, t1, dx, dy, dz, dt-1.f, px, py, pz);
    float v1001 = dotGridGradient4D(seed, x1, y0, z0, t1, dx-1.f, dy, dz, dt-1.f, px, py, pz);
    float xi001 = interpolate(v0001, v1001, dx);

    float v0101 = dotGridGradient4D(seed, x0, y1, z0, t1, dx, dy-1.f, dz, dt-1.f, px, py, pz);
    float v1101 = dotGridGradient4D(seed, x1, y1, z0, t1, dx-1.f, dy-1.f, dz, dt-1.f, px, py, pz);
    float xi101 = interpolate(v0101, v1101, dx);

    float yi01 = interpolate(xi001, xi101, dy);

    float v0011 = dotGridGradient4D(seed, x0, y0, z1, t1, dx, dy, dz-1.f, dt-1.f, px, py, pz);
    float v1011 = dotGridGradient4D(seed, x1, y0, z1, t1, dx-1.f, dy, dz-1.f, dt-1.f, px, py, pz);
    float xi011 = interpolate(v0011, v1011, dx);

    float v0111 = dotGridGradient4D(seed, x0, y1, z1, t1, dx, dy-1.f, dz-1.f, dt-1.f, px, py, pz);
    float v1111 = dotGridGradient4D(seed, x1, y1, z1, t1, dx-1.f, dy-1.f, dz-1.f, dt-1.f, px, py, pz);
    float xi111 = interpolate(v0111, v1111, dx);

    float yi11 = interpolate(xi011, xi111, dy);

    float zi1 = interpolate(yi01, yi11, dz);

    return interpolate(zi0, zi1, dt);
}

inline CUDA_CALLABLE void adj_pnoise(uint32 seed, const vec4& xyzt, int px, int py, int pz, int pt, uint32& adj_seed, const vec4& adj_xyzt, int& adj_px, int& adj_py, int& adj_pz, int& adj_pt, const float adj_ret) {}

inline CUDA_CALLABLE vec2 curlnoise(const vec2& p) { return vec2(); }
inline CUDA_CALLABLE vec3 curlnoise(const vec3& p) { return vec3(); }
inline CUDA_CALLABLE vec3 curlnoise(const vec4& p) { return vec3(); }

} // namespace wp
