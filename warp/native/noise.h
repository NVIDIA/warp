#pragma once

namespace wp
{

inline CUDA_CALLABLE float interpolate(float a0, float a1, float t)
{
    // Perlin smootherstep
    return (a1 - a0) * ((t * (t * 6.f - 15.f) + 10.f) * t * t * t) + a0;
}

inline CUDA_CALLABLE float randomGradient1D(int ix)
{
    uint32 state = uint32(ix);
    return randf(state, -1.f, 1.f);
}

inline CUDA_CALLABLE vec2 randomGradient2D(uint32 seed, int ix, int iy, int px)
{
    int idx = ix + px * iy;
    uint32 state = seed + uint32(idx);
    // M_PI
    return vec2();
}

inline CUDA_CALLABLE vec3 randomGradient3D(uint32 seed, int ix, int iy, int iz, int px, int py)
{
    int idx = ix + px * (iy + py * iz);
    uint32 state = seed + uint32(idx);
    return vec3();
}

inline CUDA_CALLABLE vec4 randomGradient4D(uint32 seed, int ix, int iy, int iz, int it, int px, int py, int pz)
{
    int idx = ix + px * (iy + py * (iz + it * pz));
    uint32 state = seed + uint32(idx);
    return vec4();
}

inline CUDA_CALLABLE float dotGridGradient1D(uint32 seed, int ix, float dx)
{
    float gradient = randomGradient1D(ix);
    return dx*gradient;
}

inline CUDA_CALLABLE float dotGridGradient2D(uint32 seed, int ix, int iy, float dx, float dy, int px)
{
    vec2 gradient = randomGradient2D(seed, ix, iy);
    return (dx*gradient.x + dy*gradient.y);
}

inline CUDA_CALLABLE float dotGridGradient3D(uint32 seed, int ix, int iy, int iz, float dx, float dy, float dz, int px, int py)
{
    vec3 gradient = randomGradient3D(seed, ix, iy, iz);
    return (dx*gradient.x + dy*gradient.y + dz*gradient.z);
}

inline CUDA_CALLABLE float dotGridGradient4D(uint32 seed, int ix, int iy, int iz, int it, float dx, float dy, float dz, float dt, int px, int py, int pz)
{
    vec4 gradient = randomGradient4D(seed, ix, iy, iz, it);
    return (dx*gradient.x + dy*gradient.y + dz*gradient.z + dt*gradient.w);
}

inline CUDA_CALLABLE float pnoise(uint32 seed, float x, int px)
{
    float dx = x - std::floorf(x);

    int x0 = ((int)std::floorf(x)) & px;

    float x = float(x0) + dx;

    int x1 = (x0 + 1) & px;

    float v0 = dotGridGradient1D(seed, x0, dx);
    float v1 = dotGridGradient1D(seed, x1, dx-1.f);

    return interpolate(v0, v1, dx);
}

inline CUDA_CALLABLE float pnoise(uint32 seed, const vec2& xy, int px, int py)
{
    float dx = xy.x - std::floorf(xy.x);
    float dy = xy.y - std::floorf(xy.y);

    int x0 = ((int)std::floorf(xy.x)) & px; 
    int y0 = ((int)std::floorf(xy.y)) & py; 

    float x = float(x0) + dx;
    float y = float(y0) + dy;

    int x1 = (x0 + 1) & px;
    int y1 = (y0 + 1) & py;

    //vXY
    float v00 = dotGridGradient2D(seed, x0, y0, dx, dy, px);
    float v10 = dotGridGradient2D(seed, x1, y0, dx-1.f, dy, px);
    float xi0 = interpolate(v00, v10, dx);

    float v01 = dotGridGradient2D(seed, x0, y1, dx, dy-1.f, px);
    float v11 = dotGridGradient2D(seed, x1, y1, dx-1.f, dy-1.f, px);
    float xi1 = interpolate(v01, v11, dx);

    return interpolate(xi0, xi1, dy);
}

inline CUDA_CALLABLE float pnoise(uint32 seed, const vec3& xyz, int px, int py, int pz)
{
    float dx = xyz.x - std::floorf(xyz.x);
    float dy = xyz.y - std::floorf(xyz.y);
    float dz = xyz.z - std::floorf(xyz.z);

    int x0 = ((int)std::floorf(xyz.x)) & px; 
    int y0 = ((int)std::floorf(xyz.y)) & py; 
    int z0 = ((int)std::floorf(xyz.z)) & pz; 

    float x = float(x0) + dx;
    float y = float(y0) + dy;
    float z = float(z0) + dz;

    int x1 = (x0 + 1) & px;
    int y1 = (y0 + 1) & py;
    int z1 = (z0 + 1) & pz;

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

inline CUDA_CALLABLE float pnoise(uint32 seed, const vec4& xyzt, int px, int py, int pz, int pt)
{
    float dx = xyzt.x - std::floorf(xyzt.x);
    float dy = xyzt.y - std::floorf(xyzt.y);
    float dz = xyzt.z - std::floorf(xyzt.z);
    float dt = xyzt.w - std::floorf(xyzt.w);

    int x0 = ((int)std::floorf(xyzt.x)) & px; 
    int y0 = ((int)std::floorf(xyzt.y)) & py; 
    int z0 = ((int)std::floorf(xyzt.z)) & pz;
    int t0 = ((int)std::floorf(xyzt.w)) & pt;

    float x = float(x0) + dx;
    float y = float(y0) + dy;
    float z = float(z0) + dz;
    float t = float(t0) + dt;

    int x1 = (x0 + 1) & px;
    int y1 = (y0 + 1) & py;
    int z1 = (z0 + 1) & pz;
    int t1 = (t0 + 1) & pt;

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

inline CUDA_CALLABLE vec2 curlnoise(const vec2& p) { return vec2(); }
inline CUDA_CALLABLE vec3 curlnoise(const vec3& p) { return vec3(); }
inline CUDA_CALLABLE vec3 curlnoise(const vec4& p) { return vec3(); }

} // namespace wp
