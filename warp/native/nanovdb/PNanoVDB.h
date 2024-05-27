
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file   nanovdb/PNanoVDB.h

    \author Andrew Reidmeyer

    \brief  This file is a portable (e.g. pointer-less) C99/GLSL/HLSL port
            of NanoVDB.h, which is compatible with most graphics APIs.
*/

#ifndef NANOVDB_PNANOVDB_H_HAS_BEEN_INCLUDED
#define NANOVDB_PNANOVDB_H_HAS_BEEN_INCLUDED

// ------------------------------------------------ Configuration -----------------------------------------------------------

// platforms
//#define PNANOVDB_C
//#define PNANOVDB_HLSL
//#define PNANOVDB_GLSL

// addressing mode
// PNANOVDB_ADDRESS_32
// PNANOVDB_ADDRESS_64
#if defined(PNANOVDB_C)
#ifndef PNANOVDB_ADDRESS_32
#define PNANOVDB_ADDRESS_64
#endif
#elif defined(PNANOVDB_HLSL)
#ifndef PNANOVDB_ADDRESS_64
#define PNANOVDB_ADDRESS_32
#endif
#elif defined(PNANOVDB_GLSL)
#ifndef PNANOVDB_ADDRESS_64
#define PNANOVDB_ADDRESS_32
#endif
#endif

// bounds checking
//#define PNANOVDB_BUF_BOUNDS_CHECK

// enable HDDA by default on HLSL/GLSL, make explicit on C
#if defined(PNANOVDB_C)
//#define PNANOVDB_HDDA
#ifdef PNANOVDB_HDDA
#ifndef PNANOVDB_CMATH
#define PNANOVDB_CMATH
#endif
#endif
#elif defined(PNANOVDB_HLSL)
#define PNANOVDB_HDDA
#elif defined(PNANOVDB_GLSL)
#define PNANOVDB_HDDA
#endif

#ifdef PNANOVDB_CMATH
#ifndef __CUDACC_RTC__
#include <math.h>
#endif
#endif

// ------------------------------------------------ Buffer -----------------------------------------------------------

#if defined(PNANOVDB_BUF_CUSTOM)
// NOP
#elif defined(PNANOVDB_C)
#define PNANOVDB_BUF_C
#elif defined(PNANOVDB_HLSL)
#define PNANOVDB_BUF_HLSL
#elif defined(PNANOVDB_GLSL)
#define PNANOVDB_BUF_GLSL
#endif

#if defined(PNANOVDB_BUF_C)
#ifndef __CUDACC_RTC__
#include <stdint.h>
#endif
#if defined(__CUDACC__)
#define PNANOVDB_BUF_FORCE_INLINE static __host__ __device__ __forceinline__
#elif defined(_WIN32)
#define PNANOVDB_BUF_FORCE_INLINE static inline __forceinline
#else
#define PNANOVDB_BUF_FORCE_INLINE static inline __attribute__((always_inline))
#endif
typedef struct pnanovdb_buf_t
{
    uint32_t* data;
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
    uint64_t size_in_words;
#endif
}pnanovdb_buf_t;
PNANOVDB_BUF_FORCE_INLINE pnanovdb_buf_t pnanovdb_make_buf(uint32_t* data, uint64_t size_in_words)
{
    pnanovdb_buf_t ret;
    ret.data = data;
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
    ret.size_in_words = size_in_words;
#endif
    return ret;
}
#if defined(PNANOVDB_ADDRESS_32)
PNANOVDB_BUF_FORCE_INLINE uint32_t pnanovdb_buf_read_uint32(pnanovdb_buf_t buf, uint32_t byte_offset)
{
    uint32_t wordaddress = (byte_offset >> 2u);
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
    return wordaddress < buf.size_in_words ? buf.data[wordaddress] : 0u;
#else
    return buf.data[wordaddress];
#endif
}
PNANOVDB_BUF_FORCE_INLINE uint64_t pnanovdb_buf_read_uint64(pnanovdb_buf_t buf, uint32_t byte_offset)
{
    uint64_t* data64 = (uint64_t*)buf.data;
    uint32_t wordaddress64 = (byte_offset >> 3u);
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
    uint64_t size_in_words64 = buf.size_in_words >> 1u;
    return wordaddress64 < size_in_words64 ? data64[wordaddress64] : 0llu;
#else
    return data64[wordaddress64];
#endif
}
PNANOVDB_BUF_FORCE_INLINE void pnanovdb_buf_write_uint32(pnanovdb_buf_t buf, uint32_t byte_offset, uint32_t value)
{
    uint32_t wordaddress = (byte_offset >> 2u);
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
    if (wordaddress < buf.size_in_words)
    {
        buf.data[wordaddress] = value;
}
#else
    buf.data[wordaddress] = value;
#endif
}
PNANOVDB_BUF_FORCE_INLINE void pnanovdb_buf_write_uint64(pnanovdb_buf_t buf, uint32_t byte_offset, uint64_t value)
{
    uint64_t* data64 = (uint64_t*)buf.data;
    uint32_t wordaddress64 = (byte_offset >> 3u);
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
    uint64_t size_in_words64 = buf.size_in_words >> 1u;
    if (wordaddress64 < size_in_words64)
    {
        data64[wordaddress64] = value;
    }
#else
    data64[wordaddress64] = value;
#endif
}
#elif defined(PNANOVDB_ADDRESS_64)
PNANOVDB_BUF_FORCE_INLINE uint32_t pnanovdb_buf_read_uint32(pnanovdb_buf_t buf, uint64_t byte_offset)
{
    uint64_t wordaddress = (byte_offset >> 2u);
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
    return wordaddress < buf.size_in_words ? buf.data[wordaddress] : 0u;
#else
    return buf.data[wordaddress];
#endif
}
PNANOVDB_BUF_FORCE_INLINE uint64_t pnanovdb_buf_read_uint64(pnanovdb_buf_t buf, uint64_t byte_offset)
{
    uint64_t* data64 = (uint64_t*)buf.data;
    uint64_t wordaddress64 = (byte_offset >> 3u);
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
    uint64_t size_in_words64 = buf.size_in_words >> 1u;
    return wordaddress64 < size_in_words64 ? data64[wordaddress64] : 0llu;
#else
    return data64[wordaddress64];
#endif
}
PNANOVDB_BUF_FORCE_INLINE void pnanovdb_buf_write_uint32(pnanovdb_buf_t buf, uint64_t byte_offset, uint32_t value)
{
    uint64_t wordaddress = (byte_offset >> 2u);
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
    if (wordaddress < buf.size_in_words)
    {
        buf.data[wordaddress] = value;
    }
#else
    buf.data[wordaddress] = value;
#endif
}
PNANOVDB_BUF_FORCE_INLINE void pnanovdb_buf_write_uint64(pnanovdb_buf_t buf, uint64_t byte_offset, uint64_t value)
{
    uint64_t* data64 = (uint64_t*)buf.data;
    uint64_t wordaddress64 = (byte_offset >> 3u);
#ifdef PNANOVDB_BUF_BOUNDS_CHECK
    uint64_t size_in_words64 = buf.size_in_words >> 1u;
    if (wordaddress64 < size_in_words64)
    {
        data64[wordaddress64] = value;
    }
#else
    data64[wordaddress64] = value;
#endif
}
#endif
typedef uint32_t pnanovdb_grid_type_t;
#define PNANOVDB_GRID_TYPE_GET(grid_typeIn, nameIn) pnanovdb_grid_type_constants[grid_typeIn].nameIn
#elif defined(PNANOVDB_BUF_HLSL)
#if defined(PNANOVDB_ADDRESS_32)
#define pnanovdb_buf_t StructuredBuffer<uint>
uint pnanovdb_buf_read_uint32(pnanovdb_buf_t buf, uint byte_offset)
{
    return buf[(byte_offset >> 2u)];
}
uint2 pnanovdb_buf_read_uint64(pnanovdb_buf_t buf, uint byte_offset)
{
    uint2 ret;
    ret.x = pnanovdb_buf_read_uint32(buf, byte_offset + 0u);
    ret.y = pnanovdb_buf_read_uint32(buf, byte_offset + 4u);
    return ret;
}
void pnanovdb_buf_write_uint32(pnanovdb_buf_t buf, uint byte_offset, uint value)
{
    // NOP, by default no write in HLSL
}
void pnanovdb_buf_write_uint64(pnanovdb_buf_t buf, uint byte_offset, uint2 value)
{
    // NOP, by default no write in HLSL
}
#elif defined(PNANOVDB_ADDRESS_64)
#define pnanovdb_buf_t StructuredBuffer<uint>
uint pnanovdb_buf_read_uint32(pnanovdb_buf_t buf, uint64_t byte_offset)
{
    return buf[uint(byte_offset >> 2u)];
}
uint64_t pnanovdb_buf_read_uint64(pnanovdb_buf_t buf, uint64_t byte_offset)
{
    uint64_t ret;
    ret = pnanovdb_buf_read_uint32(buf, byte_offset + 0u);
    ret = ret + (uint64_t(pnanovdb_buf_read_uint32(buf, byte_offset + 4u)) << 32u);
    return ret;
}
void pnanovdb_buf_write_uint32(pnanovdb_buf_t buf, uint64_t byte_offset, uint value)
{
    // NOP, by default no write in HLSL
}
void pnanovdb_buf_write_uint64(pnanovdb_buf_t buf, uint64_t byte_offset, uint64_t value)
{
    // NOP, by default no write in HLSL
}
#endif
#define pnanovdb_grid_type_t uint
#define PNANOVDB_GRID_TYPE_GET(grid_typeIn, nameIn) pnanovdb_grid_type_constants[grid_typeIn].nameIn
#elif defined(PNANOVDB_BUF_GLSL)
struct pnanovdb_buf_t
{
    uint unused;    // to satisfy min struct size?
};
uint pnanovdb_buf_read_uint32(pnanovdb_buf_t buf, uint byte_offset)
{
    return pnanovdb_buf_data[(byte_offset >> 2u)];
}
uvec2 pnanovdb_buf_read_uint64(pnanovdb_buf_t buf, uint byte_offset)
{
    uvec2 ret;
    ret.x = pnanovdb_buf_read_uint32(buf, byte_offset + 0u);
    ret.y = pnanovdb_buf_read_uint32(buf, byte_offset + 4u);
    return ret;
}
void pnanovdb_buf_write_uint32(pnanovdb_buf_t buf, uint byte_offset, uint value)
{
    // NOP, by default no write in HLSL
}
void pnanovdb_buf_write_uint64(pnanovdb_buf_t buf, uint byte_offset, uvec2 value)
{
    // NOP, by default no write in HLSL
}
#define pnanovdb_grid_type_t uint
#define PNANOVDB_GRID_TYPE_GET(grid_typeIn, nameIn) pnanovdb_grid_type_constants[grid_typeIn].nameIn
#endif

// ------------------------------------------------ Basic Types -----------------------------------------------------------

// force inline
#if defined(PNANOVDB_C)
#if defined(__CUDACC__)
#define PNANOVDB_FORCE_INLINE static __host__ __device__ __forceinline__
#elif defined(_WIN32)
#define PNANOVDB_FORCE_INLINE static inline __forceinline
#else
#define PNANOVDB_FORCE_INLINE static inline __attribute__((always_inline))
#endif
#elif defined(PNANOVDB_HLSL)
#define PNANOVDB_FORCE_INLINE
#elif defined(PNANOVDB_GLSL)
#define PNANOVDB_FORCE_INLINE
#endif

// struct typedef, static const, inout
#if defined(PNANOVDB_C)
#define PNANOVDB_STRUCT_TYPEDEF(X) typedef struct X X;
#if defined(__CUDA_ARCH__)
#define PNANOVDB_STATIC_CONST constexpr __constant__
#else
#define PNANOVDB_STATIC_CONST static const
#endif
#define PNANOVDB_INOUT(X) X*
#define PNANOVDB_IN(X) const X*
#define PNANOVDB_DEREF(X) (*X)
#define PNANOVDB_REF(X) &X
#elif defined(PNANOVDB_HLSL)
#define PNANOVDB_STRUCT_TYPEDEF(X)
#define PNANOVDB_STATIC_CONST static const
#define PNANOVDB_INOUT(X) inout X
#define PNANOVDB_IN(X) X
#define PNANOVDB_DEREF(X) X
#define PNANOVDB_REF(X) X
#elif defined(PNANOVDB_GLSL)
#define PNANOVDB_STRUCT_TYPEDEF(X)
#define PNANOVDB_STATIC_CONST const
#define PNANOVDB_INOUT(X) inout X
#define PNANOVDB_IN(X) X
#define PNANOVDB_DEREF(X) X
#define PNANOVDB_REF(X) X
#endif

// basic types, type conversion
#if defined(PNANOVDB_C)
#define PNANOVDB_NATIVE_64
#ifndef __CUDACC_RTC__
#include <stdint.h>
#endif
#if !defined(PNANOVDB_MEMCPY_CUSTOM)
#ifndef __CUDACC_RTC__
#include <string.h>
#endif
#define pnanovdb_memcpy memcpy
#endif
typedef uint32_t pnanovdb_uint32_t;
typedef int32_t pnanovdb_int32_t;
typedef int32_t pnanovdb_bool_t;
#define PNANOVDB_FALSE 0
#define PNANOVDB_TRUE 1
typedef uint64_t pnanovdb_uint64_t;
typedef int64_t pnanovdb_int64_t;
typedef struct pnanovdb_coord_t
{
    pnanovdb_int32_t x, y, z;
}pnanovdb_coord_t;
typedef struct pnanovdb_vec3_t
{
    float x, y, z;
}pnanovdb_vec3_t;
PNANOVDB_FORCE_INLINE pnanovdb_int32_t pnanovdb_uint32_as_int32(pnanovdb_uint32_t v) { return (pnanovdb_int32_t)v; }
PNANOVDB_FORCE_INLINE pnanovdb_int64_t pnanovdb_uint64_as_int64(pnanovdb_uint64_t v) { return (pnanovdb_int64_t)v; }
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_int64_as_uint64(pnanovdb_int64_t v) { return (pnanovdb_uint64_t)v; }
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_int32_as_uint32(pnanovdb_int32_t v) { return (pnanovdb_uint32_t)v; }
PNANOVDB_FORCE_INLINE float pnanovdb_uint32_as_float(pnanovdb_uint32_t v) { float vf; pnanovdb_memcpy(&vf, &v, sizeof(vf)); return vf; }
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_float_as_uint32(float v) { return *((pnanovdb_uint32_t*)(&v)); }
PNANOVDB_FORCE_INLINE double pnanovdb_uint64_as_double(pnanovdb_uint64_t v) { double vf; pnanovdb_memcpy(&vf, &v, sizeof(vf)); return vf; }
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_double_as_uint64(double v) { return *((pnanovdb_uint64_t*)(&v)); }
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_uint64_low(pnanovdb_uint64_t v) { return (pnanovdb_uint32_t)v; }
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_uint64_high(pnanovdb_uint64_t v) { return (pnanovdb_uint32_t)(v >> 32u); }
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_uint32_as_uint64(pnanovdb_uint32_t x, pnanovdb_uint32_t y) { return ((pnanovdb_uint64_t)x) | (((pnanovdb_uint64_t)y) << 32u); }
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_uint32_as_uint64_low(pnanovdb_uint32_t x) { return ((pnanovdb_uint64_t)x); }
PNANOVDB_FORCE_INLINE pnanovdb_int32_t pnanovdb_uint64_is_equal(pnanovdb_uint64_t a, pnanovdb_uint64_t b) { return a == b; }
PNANOVDB_FORCE_INLINE pnanovdb_int32_t pnanovdb_int64_is_zero(pnanovdb_int64_t a) { return a == 0; }
#ifdef PNANOVDB_CMATH
PNANOVDB_FORCE_INLINE float pnanovdb_floor(float v) { return floorf(v); }
#endif
PNANOVDB_FORCE_INLINE pnanovdb_int32_t pnanovdb_float_to_int32(float v) { return (pnanovdb_int32_t)v; }
PNANOVDB_FORCE_INLINE float pnanovdb_int32_to_float(pnanovdb_int32_t v) { return (float)v; }
PNANOVDB_FORCE_INLINE float pnanovdb_uint32_to_float(pnanovdb_uint32_t v) { return (float)v; }
PNANOVDB_FORCE_INLINE float pnanovdb_min(float a, float b) { return a < b ? a : b; }
PNANOVDB_FORCE_INLINE float pnanovdb_max(float a, float b) { return a > b ? a : b; }
#elif defined(PNANOVDB_HLSL)
typedef uint pnanovdb_uint32_t;
typedef int pnanovdb_int32_t;
typedef bool pnanovdb_bool_t;
#define PNANOVDB_FALSE false
#define PNANOVDB_TRUE true
typedef int3 pnanovdb_coord_t;
typedef float3 pnanovdb_vec3_t;
pnanovdb_int32_t pnanovdb_uint32_as_int32(pnanovdb_uint32_t v) { return int(v); }
pnanovdb_uint32_t pnanovdb_int32_as_uint32(pnanovdb_int32_t v) { return uint(v); }
float pnanovdb_uint32_as_float(pnanovdb_uint32_t v) { return asfloat(v); }
pnanovdb_uint32_t pnanovdb_float_as_uint32(float v) { return asuint(v); }
float pnanovdb_floor(float v) { return floor(v); }
pnanovdb_int32_t pnanovdb_float_to_int32(float v) { return int(v); }
float pnanovdb_int32_to_float(pnanovdb_int32_t v) { return float(v); }
float pnanovdb_uint32_to_float(pnanovdb_uint32_t v) { return float(v); }
float pnanovdb_min(float a, float b) { return min(a, b); }
float pnanovdb_max(float a, float b) { return max(a, b); }
#if defined(PNANOVDB_ADDRESS_32)
typedef uint2 pnanovdb_uint64_t;
typedef int2 pnanovdb_int64_t;
pnanovdb_int64_t pnanovdb_uint64_as_int64(pnanovdb_uint64_t v) { return int2(v); }
pnanovdb_uint64_t pnanovdb_int64_as_uint64(pnanovdb_int64_t v) { return uint2(v); }
double pnanovdb_uint64_as_double(pnanovdb_uint64_t v) { return asdouble(v.x, v.y); }
pnanovdb_uint64_t pnanovdb_double_as_uint64(double v) { uint2 ret; asuint(v, ret.x, ret.y); return ret; }
pnanovdb_uint32_t pnanovdb_uint64_low(pnanovdb_uint64_t v) { return v.x; }
pnanovdb_uint32_t pnanovdb_uint64_high(pnanovdb_uint64_t v) { return v.y; }
pnanovdb_uint64_t pnanovdb_uint32_as_uint64(pnanovdb_uint32_t x, pnanovdb_uint32_t y) { return uint2(x, y); }
pnanovdb_uint64_t pnanovdb_uint32_as_uint64_low(pnanovdb_uint32_t x) { return uint2(x, 0); }
bool pnanovdb_uint64_is_equal(pnanovdb_uint64_t a, pnanovdb_uint64_t b) { return (a.x == b.x) && (a.y == b.y); }
bool pnanovdb_int64_is_zero(pnanovdb_int64_t a) { return a.x == 0 && a.y == 0; }
#else
typedef uint64_t pnanovdb_uint64_t;
typedef int64_t pnanovdb_int64_t;
pnanovdb_int64_t pnanovdb_uint64_as_int64(pnanovdb_uint64_t v) { return int64_t(v); }
pnanovdb_uint64_t pnanovdb_int64_as_uint64(pnanovdb_int64_t v) { return uint64_t(v); }
double pnanovdb_uint64_as_double(pnanovdb_uint64_t v) { return asdouble(uint(v), uint(v >> 32u)); }
pnanovdb_uint64_t pnanovdb_double_as_uint64(double v) { uint2 ret; asuint(v, ret.x, ret.y); return uint64_t(ret.x) + (uint64_t(ret.y) << 32u); }
pnanovdb_uint32_t pnanovdb_uint64_low(pnanovdb_uint64_t v) { return uint(v); }
pnanovdb_uint32_t pnanovdb_uint64_high(pnanovdb_uint64_t v) { return uint(v >> 32u); }
pnanovdb_uint64_t pnanovdb_uint32_as_uint64(pnanovdb_uint32_t x, pnanovdb_uint32_t y) { return uint64_t(x) + (uint64_t(y) << 32u); }
pnanovdb_uint64_t pnanovdb_uint32_as_uint64_low(pnanovdb_uint32_t x) { return uint64_t(x); }
bool pnanovdb_uint64_is_equal(pnanovdb_uint64_t a, pnanovdb_uint64_t b) { return a == b; }
bool pnanovdb_int64_is_zero(pnanovdb_int64_t a) { return a == 0; }
#endif
#elif defined(PNANOVDB_GLSL)
#define pnanovdb_uint32_t uint
#define pnanovdb_int32_t int
#define pnanovdb_bool_t bool
#define PNANOVDB_FALSE false
#define PNANOVDB_TRUE true
#define pnanovdb_uint64_t uvec2
#define pnanovdb_int64_t ivec2
#define pnanovdb_coord_t ivec3
#define pnanovdb_vec3_t vec3
pnanovdb_int32_t pnanovdb_uint32_as_int32(pnanovdb_uint32_t v) { return int(v); }
pnanovdb_int64_t pnanovdb_uint64_as_int64(pnanovdb_uint64_t v) { return ivec2(v); }
pnanovdb_uint64_t pnanovdb_int64_as_uint64(pnanovdb_int64_t v) { return uvec2(v); }
pnanovdb_uint32_t pnanovdb_int32_as_uint32(pnanovdb_int32_t v) { return uint(v); }
float pnanovdb_uint32_as_float(pnanovdb_uint32_t v) { return uintBitsToFloat(v); }
pnanovdb_uint32_t pnanovdb_float_as_uint32(float v) { return floatBitsToUint(v); }
double pnanovdb_uint64_as_double(pnanovdb_uint64_t v) { return packDouble2x32(uvec2(v.x, v.y)); }
pnanovdb_uint64_t pnanovdb_double_as_uint64(double v) { return unpackDouble2x32(v); }
pnanovdb_uint32_t pnanovdb_uint64_low(pnanovdb_uint64_t v) { return v.x; }
pnanovdb_uint32_t pnanovdb_uint64_high(pnanovdb_uint64_t v) { return v.y; }
pnanovdb_uint64_t pnanovdb_uint32_as_uint64(pnanovdb_uint32_t x, pnanovdb_uint32_t y) { return uvec2(x, y); }
pnanovdb_uint64_t pnanovdb_uint32_as_uint64_low(pnanovdb_uint32_t x) { return uvec2(x, 0); }
bool pnanovdb_uint64_is_equal(pnanovdb_uint64_t a, pnanovdb_uint64_t b) { return (a.x == b.x) && (a.y == b.y); }
bool pnanovdb_int64_is_zero(pnanovdb_int64_t a) { return a.x == 0 && a.y == 0; }
float pnanovdb_floor(float v) { return floor(v); }
pnanovdb_int32_t pnanovdb_float_to_int32(float v) { return int(v); }
float pnanovdb_int32_to_float(pnanovdb_int32_t v) { return float(v); }
float pnanovdb_uint32_to_float(pnanovdb_uint32_t v) { return float(v); }
float pnanovdb_min(float a, float b) { return min(a, b); }
float pnanovdb_max(float a, float b) { return max(a, b); }
#endif

// ------------------------------------------------ Coord/Vec3 Utilties -----------------------------------------------------------

#if defined(PNANOVDB_C)
PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_vec3_uniform(float a)
{
    pnanovdb_vec3_t v;
    v.x = a;
    v.y = a;
    v.z = a;
    return v;
}
PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_vec3_add(const pnanovdb_vec3_t a, const pnanovdb_vec3_t b)
{
    pnanovdb_vec3_t v;
    v.x = a.x + b.x;
    v.y = a.y + b.y;
    v.z = a.z + b.z;
    return v;
}
PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_vec3_sub(const pnanovdb_vec3_t a, const pnanovdb_vec3_t b)
{
    pnanovdb_vec3_t v;
    v.x = a.x - b.x;
    v.y = a.y - b.y;
    v.z = a.z - b.z;
    return v;
}
PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_vec3_mul(const pnanovdb_vec3_t a, const pnanovdb_vec3_t b)
{
    pnanovdb_vec3_t v;
    v.x = a.x * b.x;
    v.y = a.y * b.y;
    v.z = a.z * b.z;
    return v;
}
PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_vec3_div(const pnanovdb_vec3_t a, const pnanovdb_vec3_t b)
{
    pnanovdb_vec3_t v;
    v.x = a.x / b.x;
    v.y = a.y / b.y;
    v.z = a.z / b.z;
    return v;
}
PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_vec3_min(const pnanovdb_vec3_t a, const pnanovdb_vec3_t b)
{
    pnanovdb_vec3_t v;
    v.x = a.x < b.x ? a.x : b.x;
    v.y = a.y < b.y ? a.y : b.y;
    v.z = a.z < b.z ? a.z : b.z;
    return v;
}
PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_vec3_max(const pnanovdb_vec3_t a, const pnanovdb_vec3_t b)
{
    pnanovdb_vec3_t v;
    v.x = a.x > b.x ? a.x : b.x;
    v.y = a.y > b.y ? a.y : b.y;
    v.z = a.z > b.z ? a.z : b.z;
    return v;
}
PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_coord_to_vec3(const pnanovdb_coord_t coord)
{
    pnanovdb_vec3_t v;
    v.x = pnanovdb_int32_to_float(coord.x);
    v.y = pnanovdb_int32_to_float(coord.y);
    v.z = pnanovdb_int32_to_float(coord.z);
    return v;
}
PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_coord_uniform(const pnanovdb_int32_t a)
{
    pnanovdb_coord_t v;
    v.x = a;
    v.y = a;
    v.z = a;
    return v;
}
PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_coord_add(pnanovdb_coord_t a, pnanovdb_coord_t b)
{
    pnanovdb_coord_t v;
    v.x = a.x + b.x;
    v.y = a.y + b.y;
    v.z = a.z + b.z;
    return v;
}
#elif defined(PNANOVDB_HLSL)
pnanovdb_vec3_t pnanovdb_vec3_uniform(float a) { return float3(a, a, a); }
pnanovdb_vec3_t pnanovdb_vec3_add(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return a + b; }
pnanovdb_vec3_t pnanovdb_vec3_sub(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return a - b; }
pnanovdb_vec3_t pnanovdb_vec3_mul(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return a * b; }
pnanovdb_vec3_t pnanovdb_vec3_div(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return a / b; }
pnanovdb_vec3_t pnanovdb_vec3_min(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return min(a, b); }
pnanovdb_vec3_t pnanovdb_vec3_max(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return max(a, b); }
pnanovdb_vec3_t pnanovdb_coord_to_vec3(pnanovdb_coord_t coord) { return float3(coord); }
pnanovdb_coord_t pnanovdb_coord_uniform(pnanovdb_int32_t a) { return int3(a, a, a); }
pnanovdb_coord_t pnanovdb_coord_add(pnanovdb_coord_t a, pnanovdb_coord_t b) { return a + b; }
#elif defined(PNANOVDB_GLSL)
pnanovdb_vec3_t pnanovdb_vec3_uniform(float a) { return vec3(a, a, a); }
pnanovdb_vec3_t pnanovdb_vec3_add(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return a + b; }
pnanovdb_vec3_t pnanovdb_vec3_sub(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return a - b; }
pnanovdb_vec3_t pnanovdb_vec3_mul(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return a * b; }
pnanovdb_vec3_t pnanovdb_vec3_div(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return a / b; }
pnanovdb_vec3_t pnanovdb_vec3_min(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return min(a, b); }
pnanovdb_vec3_t pnanovdb_vec3_max(pnanovdb_vec3_t a, pnanovdb_vec3_t b) { return max(a, b); }
pnanovdb_vec3_t pnanovdb_coord_to_vec3(const pnanovdb_coord_t coord) { return vec3(coord); }
pnanovdb_coord_t pnanovdb_coord_uniform(pnanovdb_int32_t a) { return ivec3(a, a, a); }
pnanovdb_coord_t pnanovdb_coord_add(pnanovdb_coord_t a, pnanovdb_coord_t b) { return a + b; }
#endif

// ------------------------------------------------ Uint64 Utils -----------------------------------------------------------

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_uint32_countbits(pnanovdb_uint32_t value)
{
#if defined(PNANOVDB_C)
#if defined(_MSC_VER) && (_MSC_VER >= 1928) && defined(PNANOVDB_USE_INTRINSICS)
    return __popcnt(value);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(PNANOVDB_USE_INTRINSICS)
    return __builtin_popcount(value);
#else
    value = value - ((value >> 1) & 0x55555555);
    value = (value & 0x33333333) + ((value >> 2) & 0x33333333);
    value = (value + (value >> 4)) & 0x0F0F0F0F;
    return (value * 0x01010101) >> 24;
#endif
#elif defined(PNANOVDB_HLSL)
    return countbits(value);
#elif defined(PNANOVDB_GLSL)
    return bitCount(value);
#endif
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_uint64_countbits(pnanovdb_uint64_t value)
{
    return pnanovdb_uint32_countbits(pnanovdb_uint64_low(value)) + pnanovdb_uint32_countbits(pnanovdb_uint64_high(value));
}

#if defined(PNANOVDB_ADDRESS_32)
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_uint64_offset(pnanovdb_uint64_t a, pnanovdb_uint32_t b)
{
    pnanovdb_uint32_t low = pnanovdb_uint64_low(a);
    pnanovdb_uint32_t high = pnanovdb_uint64_high(a);
    low += b;
    if (low < b)
    {
        high += 1u;
    }
    return pnanovdb_uint32_as_uint64(low, high);
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_uint64_dec(pnanovdb_uint64_t a)
{
    pnanovdb_uint32_t low = pnanovdb_uint64_low(a);
    pnanovdb_uint32_t high = pnanovdb_uint64_high(a);
    if (low == 0u)
    {
        high -= 1u;
    }
    low -= 1u;
    return pnanovdb_uint32_as_uint64(low, high);
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_uint64_to_uint32_lsr(pnanovdb_uint64_t a, pnanovdb_uint32_t b)
{
    pnanovdb_uint32_t low = pnanovdb_uint64_low(a);
    pnanovdb_uint32_t high = pnanovdb_uint64_high(a);
    return (b >= 32u) ?
        (high >> (b - 32)) :
        ((low >> b) | ((b > 0) ? (high << (32u - b)) : 0u));
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_uint64_bit_mask(pnanovdb_uint32_t bit_idx)
{
    pnanovdb_uint32_t mask_low = bit_idx < 32u ? 1u << bit_idx : 0u;
    pnanovdb_uint32_t mask_high = bit_idx >= 32u ? 1u << (bit_idx - 32u) : 0u;
    return pnanovdb_uint32_as_uint64(mask_low, mask_high);
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_uint64_and(pnanovdb_uint64_t a, pnanovdb_uint64_t b)
{
    return pnanovdb_uint32_as_uint64(
        pnanovdb_uint64_low(a) & pnanovdb_uint64_low(b),
        pnanovdb_uint64_high(a) & pnanovdb_uint64_high(b)
    );
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_uint64_any_bit(pnanovdb_uint64_t a)
{
    return pnanovdb_uint64_low(a) != 0u || pnanovdb_uint64_high(a) != 0u;
}

#else
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_uint64_offset(pnanovdb_uint64_t a, pnanovdb_uint32_t b)
{
    return a + b;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_uint64_dec(pnanovdb_uint64_t a)
{
    return a - 1u;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_uint64_to_uint32_lsr(pnanovdb_uint64_t a, pnanovdb_uint32_t b)
{
    return pnanovdb_uint64_low(a >> b);
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_uint64_bit_mask(pnanovdb_uint32_t bit_idx)
{
    return 1llu << bit_idx;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_uint64_and(pnanovdb_uint64_t a, pnanovdb_uint64_t b)
{
    return a & b;
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_uint64_any_bit(pnanovdb_uint64_t a)
{
    return a != 0llu;
}
#endif

// ------------------------------------------------ Address Type -----------------------------------------------------------

#if defined(PNANOVDB_ADDRESS_32)
struct pnanovdb_address_t
{
    pnanovdb_uint32_t byte_offset;
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_address_t)

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_offset(pnanovdb_address_t address, pnanovdb_uint32_t byte_offset)
{
    pnanovdb_address_t ret = address;
    ret.byte_offset += byte_offset;
    return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_offset_neg(pnanovdb_address_t address, pnanovdb_uint32_t byte_offset)
{
    pnanovdb_address_t ret = address;
    ret.byte_offset -= byte_offset;
    return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_offset_product(pnanovdb_address_t address, pnanovdb_uint32_t byte_offset, pnanovdb_uint32_t multiplier)
{
    pnanovdb_address_t ret = address;
    ret.byte_offset += byte_offset * multiplier;
    return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_offset64(pnanovdb_address_t address, pnanovdb_uint64_t byte_offset)
{
    pnanovdb_address_t ret = address;
    // lose high bits on 32-bit
    ret.byte_offset += pnanovdb_uint64_low(byte_offset);
    return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_offset64_product(pnanovdb_address_t address, pnanovdb_uint64_t byte_offset, pnanovdb_uint32_t multiplier)
{
    pnanovdb_address_t ret = address;
    ret.byte_offset += pnanovdb_uint64_low(byte_offset) * multiplier;
    return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_address_mask(pnanovdb_address_t address, pnanovdb_uint32_t mask)
{
    return address.byte_offset & mask;
}
PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_mask_inv(pnanovdb_address_t address, pnanovdb_uint32_t mask)
{
    pnanovdb_address_t ret = address;
    ret.byte_offset &= (~mask);
    return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_null()
{
    pnanovdb_address_t ret = { 0 };
    return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_address_is_null(pnanovdb_address_t address)
{
    return address.byte_offset == 0u;
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_address_in_interval(pnanovdb_address_t address, pnanovdb_address_t min_address, pnanovdb_address_t max_address)
{
    return address.byte_offset >= min_address.byte_offset && address.byte_offset < max_address.byte_offset;
}
#elif defined(PNANOVDB_ADDRESS_64)
struct pnanovdb_address_t
{
    pnanovdb_uint64_t byte_offset;
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_address_t)

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_offset(pnanovdb_address_t address, pnanovdb_uint32_t byte_offset)
{
    pnanovdb_address_t ret = address;
    ret.byte_offset += byte_offset;
    return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_offset_neg(pnanovdb_address_t address, pnanovdb_uint32_t byte_offset)
{
    pnanovdb_address_t ret = address;
    ret.byte_offset -= byte_offset;
    return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_offset_product(pnanovdb_address_t address, pnanovdb_uint32_t byte_offset, pnanovdb_uint32_t multiplier)
{
    pnanovdb_address_t ret = address;
    ret.byte_offset += pnanovdb_uint32_as_uint64_low(byte_offset) * pnanovdb_uint32_as_uint64_low(multiplier);
    return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_offset64(pnanovdb_address_t address, pnanovdb_uint64_t byte_offset)
{
    pnanovdb_address_t ret = address;
    ret.byte_offset += byte_offset;
    return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_offset64_product(pnanovdb_address_t address, pnanovdb_uint64_t byte_offset, pnanovdb_uint32_t multiplier)
{
    pnanovdb_address_t ret = address;
    ret.byte_offset += byte_offset * pnanovdb_uint32_as_uint64_low(multiplier);
    return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_address_mask(pnanovdb_address_t address, pnanovdb_uint32_t mask)
{
    return pnanovdb_uint64_low(address.byte_offset) & mask;
}
PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_mask_inv(pnanovdb_address_t address, pnanovdb_uint32_t mask)
{
    pnanovdb_address_t ret = address;
    ret.byte_offset &= (~pnanovdb_uint32_as_uint64_low(mask));
    return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_address_null()
{
    pnanovdb_address_t ret = { 0 };
    return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_address_is_null(pnanovdb_address_t address)
{
    return address.byte_offset == 0llu;
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_address_in_interval(pnanovdb_address_t address, pnanovdb_address_t min_address, pnanovdb_address_t max_address)
{
    return address.byte_offset >= min_address.byte_offset && address.byte_offset < max_address.byte_offset;
}
#endif

// ------------------------------------------------ High Level Buffer Read -----------------------------------------------------------

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_read_uint32(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
    return pnanovdb_buf_read_uint32(buf, address.byte_offset);
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_read_uint64(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
    return pnanovdb_buf_read_uint64(buf, address.byte_offset);
}
PNANOVDB_FORCE_INLINE pnanovdb_int32_t pnanovdb_read_int32(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
    return pnanovdb_uint32_as_int32(pnanovdb_read_uint32(buf, address));
}
PNANOVDB_FORCE_INLINE float pnanovdb_read_float(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
    return pnanovdb_uint32_as_float(pnanovdb_read_uint32(buf, address));
}
PNANOVDB_FORCE_INLINE pnanovdb_int64_t pnanovdb_read_int64(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
    return pnanovdb_uint64_as_int64(pnanovdb_read_uint64(buf, address));
}
PNANOVDB_FORCE_INLINE double pnanovdb_read_double(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
    return pnanovdb_uint64_as_double(pnanovdb_read_uint64(buf, address));
}
PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_read_coord(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
    pnanovdb_coord_t ret;
    ret.x = pnanovdb_uint32_as_int32(pnanovdb_read_uint32(buf, pnanovdb_address_offset(address, 0u)));
    ret.y = pnanovdb_uint32_as_int32(pnanovdb_read_uint32(buf, pnanovdb_address_offset(address, 4u)));
    ret.z = pnanovdb_uint32_as_int32(pnanovdb_read_uint32(buf, pnanovdb_address_offset(address, 8u)));
    return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_read_vec3(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
    pnanovdb_vec3_t ret;
    ret.x = pnanovdb_read_float(buf, pnanovdb_address_offset(address, 0u));
    ret.y = pnanovdb_read_float(buf, pnanovdb_address_offset(address, 4u));
    ret.z = pnanovdb_read_float(buf, pnanovdb_address_offset(address, 8u));
    return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_read_uint16(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
    pnanovdb_uint32_t raw = pnanovdb_read_uint32(buf, pnanovdb_address_mask_inv(address, 3u));
    return (raw >> (pnanovdb_address_mask(address, 2) << 3));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_read_uint8(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
    pnanovdb_uint32_t raw = pnanovdb_read_uint32(buf, pnanovdb_address_mask_inv(address, 3u));
    return (raw >> (pnanovdb_address_mask(address, 3) << 3)) & 255;
}
PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_read_vec3u16(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
    pnanovdb_vec3_t ret;
    const float scale = 1.f / 65535.f;
    ret.x = scale * pnanovdb_uint32_to_float(pnanovdb_read_uint16(buf, pnanovdb_address_offset(address, 0u))) - 0.5f;
    ret.y = scale * pnanovdb_uint32_to_float(pnanovdb_read_uint16(buf, pnanovdb_address_offset(address, 2u))) - 0.5f;
    ret.z = scale * pnanovdb_uint32_to_float(pnanovdb_read_uint16(buf, pnanovdb_address_offset(address, 4u))) - 0.5f;
    return ret;
}
PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_read_vec3u8(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
    pnanovdb_vec3_t ret;
    const float scale = 1.f / 255.f;
    ret.x = scale * pnanovdb_uint32_to_float(pnanovdb_read_uint8(buf, pnanovdb_address_offset(address, 0u))) - 0.5f;
    ret.y = scale * pnanovdb_uint32_to_float(pnanovdb_read_uint8(buf, pnanovdb_address_offset(address, 1u))) - 0.5f;
    ret.z = scale * pnanovdb_uint32_to_float(pnanovdb_read_uint8(buf, pnanovdb_address_offset(address, 2u))) - 0.5f;
    return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_read_bit(pnanovdb_buf_t buf, pnanovdb_address_t address, pnanovdb_uint32_t bit_offset)
{
    pnanovdb_address_t word_address = pnanovdb_address_mask_inv(address, 3u);
    pnanovdb_uint32_t bit_index = (pnanovdb_address_mask(address, 3u) << 3u) + bit_offset;
    pnanovdb_uint32_t value_word = pnanovdb_buf_read_uint32(buf, word_address.byte_offset);
    return ((value_word >> bit_index) & 1) != 0u;
}

#if defined(PNANOVDB_C)
PNANOVDB_FORCE_INLINE short pnanovdb_read_half(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
    pnanovdb_uint32_t raw = pnanovdb_read_uint32(buf, address);
    return (short)(raw >> (pnanovdb_address_mask(address, 2) << 3));
}
#elif defined(PNANOVDB_HLSL)
PNANOVDB_FORCE_INLINE float pnanovdb_read_half(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
    pnanovdb_uint32_t raw = pnanovdb_read_uint32(buf, address);
    return f16tof32(raw >> (pnanovdb_address_mask(address, 2) << 3));
}
#elif defined(PNANOVDB_GLSL)
PNANOVDB_FORCE_INLINE float pnanovdb_read_half(pnanovdb_buf_t buf, pnanovdb_address_t address)
{
    pnanovdb_uint32_t raw = pnanovdb_read_uint32(buf, address);
    return unpackHalf2x16(raw >> (pnanovdb_address_mask(address, 2) << 3)).x;
}
#endif

// ------------------------------------------------ High Level Buffer Write -----------------------------------------------------------

PNANOVDB_FORCE_INLINE void pnanovdb_write_uint32(pnanovdb_buf_t buf, pnanovdb_address_t address, pnanovdb_uint32_t value)
{
    pnanovdb_buf_write_uint32(buf, address.byte_offset, value);
}
PNANOVDB_FORCE_INLINE void pnanovdb_write_uint64(pnanovdb_buf_t buf, pnanovdb_address_t address, pnanovdb_uint64_t value)
{
    pnanovdb_buf_write_uint64(buf, address.byte_offset, value);
}
PNANOVDB_FORCE_INLINE void pnanovdb_write_int32(pnanovdb_buf_t buf, pnanovdb_address_t address, pnanovdb_int32_t value)
{
    pnanovdb_write_uint32(buf, address, pnanovdb_int32_as_uint32(value));
}
PNANOVDB_FORCE_INLINE void pnanovdb_write_int64(pnanovdb_buf_t buf, pnanovdb_address_t address, pnanovdb_int64_t value)
{
    pnanovdb_buf_write_uint64(buf, address.byte_offset, pnanovdb_int64_as_uint64(value));
}
PNANOVDB_FORCE_INLINE void pnanovdb_write_float(pnanovdb_buf_t buf, pnanovdb_address_t address, float value)
{
    pnanovdb_write_uint32(buf, address, pnanovdb_float_as_uint32(value));
}
PNANOVDB_FORCE_INLINE void pnanovdb_write_double(pnanovdb_buf_t buf, pnanovdb_address_t address, double value)
{
    pnanovdb_write_uint64(buf, address, pnanovdb_double_as_uint64(value));
}
PNANOVDB_FORCE_INLINE void pnanovdb_write_coord(pnanovdb_buf_t buf, pnanovdb_address_t address, PNANOVDB_IN(pnanovdb_coord_t) value)
{
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(address, 0u), pnanovdb_int32_as_uint32(PNANOVDB_DEREF(value).x));
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(address, 4u), pnanovdb_int32_as_uint32(PNANOVDB_DEREF(value).y));
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(address, 8u), pnanovdb_int32_as_uint32(PNANOVDB_DEREF(value).z));
}
PNANOVDB_FORCE_INLINE void pnanovdb_write_vec3(pnanovdb_buf_t buf, pnanovdb_address_t address, PNANOVDB_IN(pnanovdb_vec3_t) value)
{
    pnanovdb_write_float(buf, pnanovdb_address_offset(address, 0u), PNANOVDB_DEREF(value).x);
    pnanovdb_write_float(buf, pnanovdb_address_offset(address, 4u), PNANOVDB_DEREF(value).y);
    pnanovdb_write_float(buf, pnanovdb_address_offset(address, 8u), PNANOVDB_DEREF(value).z);
}

// ------------------------------------------------ Core Structures -----------------------------------------------------------

#define PNANOVDB_MAGIC_NUMBER 0x304244566f6e614eUL// "NanoVDB0" in hex - little endian (uint64_t)
#define PNANOVDB_MAGIC_GRID   0x314244566f6e614eUL// "NanoVDB1" in hex - little endian (uint64_t)
#define PNANOVDB_MAGIC_FILE   0x324244566f6e614eUL// "NanoVDB2" in hex - little endian (uint64_t)

#define PNANOVDB_MAJOR_VERSION_NUMBER 32// reflects changes to the ABI
#define PNANOVDB_MINOR_VERSION_NUMBER  7// reflects changes to the API but not ABI
#define PNANOVDB_PATCH_VERSION_NUMBER  0// reflects bug-fixes with no ABI or API changes

#define PNANOVDB_GRID_TYPE_UNKNOWN 0
#define PNANOVDB_GRID_TYPE_FLOAT 1
#define PNANOVDB_GRID_TYPE_DOUBLE 2
#define PNANOVDB_GRID_TYPE_INT16 3
#define PNANOVDB_GRID_TYPE_INT32 4
#define PNANOVDB_GRID_TYPE_INT64 5
#define PNANOVDB_GRID_TYPE_VEC3F 6
#define PNANOVDB_GRID_TYPE_VEC3D 7
#define PNANOVDB_GRID_TYPE_MASK 8
#define PNANOVDB_GRID_TYPE_HALF 9
#define PNANOVDB_GRID_TYPE_UINT32 10
#define PNANOVDB_GRID_TYPE_BOOLEAN 11
#define PNANOVDB_GRID_TYPE_RGBA8 12
#define PNANOVDB_GRID_TYPE_FP4 13
#define PNANOVDB_GRID_TYPE_FP8 14
#define PNANOVDB_GRID_TYPE_FP16 15
#define PNANOVDB_GRID_TYPE_FPN 16
#define PNANOVDB_GRID_TYPE_VEC4F 17
#define PNANOVDB_GRID_TYPE_VEC4D 18
#define PNANOVDB_GRID_TYPE_INDEX 19
#define PNANOVDB_GRID_TYPE_ONINDEX 20
#define PNANOVDB_GRID_TYPE_INDEXMASK 21
#define PNANOVDB_GRID_TYPE_ONINDEXMASK 22
#define PNANOVDB_GRID_TYPE_POINTINDEX 23
#define PNANOVDB_GRID_TYPE_VEC3U8 24
#define PNANOVDB_GRID_TYPE_VEC3U16 25
#define PNANOVDB_GRID_TYPE_UINT8 26
#define PNANOVDB_GRID_TYPE_END 27

#define PNANOVDB_GRID_CLASS_UNKNOWN 0
#define PNANOVDB_GRID_CLASS_LEVEL_SET 1     // narrow band level set, e.g. SDF
#define PNANOVDB_GRID_CLASS_FOG_VOLUME 2    // fog volume, e.g. density
#define PNANOVDB_GRID_CLASS_STAGGERED 3     // staggered MAC grid, e.g. velocity
#define PNANOVDB_GRID_CLASS_POINT_INDEX 4   // point index grid
#define PNANOVDB_GRID_CLASS_POINT_DATA 5    // point data grid
#define PNANOVDB_GRID_CLASS_TOPOLOGY 6      // grid with active states only (no values)
#define PNANOVDB_GRID_CLASS_VOXEL_VOLUME 7  // volume of geometric cubes, e.g. minecraft
#define PNANOVDB_GRID_CLASS_INDEX_GRID 8    // grid whose values are offsets, e.g. into an external array
#define PNANOVDB_GRID_CLASS_TENSOR_GRID 9 // grid which can have extra metadata and features
#define PNANOVDB_GRID_CLASS_END 10

#define PNANOVDB_GRID_FLAGS_HAS_LONG_GRID_NAME (1 << 0)
#define PNANOVDB_GRID_FLAGS_HAS_BBOX (1 << 1)
#define PNANOVDB_GRID_FLAGS_HAS_MIN_MAX (1 << 2)
#define PNANOVDB_GRID_FLAGS_HAS_AVERAGE (1 << 3)
#define PNANOVDB_GRID_FLAGS_HAS_STD_DEVIATION (1 << 4)
#define PNANOVDB_GRID_FLAGS_IS_BREADTH_FIRST (1 << 5)
#define PNANOVDB_GRID_FLAGS_END (1 << 6)

#define PNANOVDB_LEAF_TYPE_DEFAULT 0
#define PNANOVDB_LEAF_TYPE_LITE 1
#define PNANOVDB_LEAF_TYPE_FP 2
#define PNANOVDB_LEAF_TYPE_INDEX 3
#define PNANOVDB_LEAF_TYPE_INDEXMASK 4
#define PNANOVDB_LEAF_TYPE_POINTINDEX 5

// BuildType = Unknown, float, double, int16_t, int32_t, int64_t, Vec3f, Vec3d, Mask, ...
// bit count of values in leaf nodes, i.e. 8*sizeof(*nanovdb::LeafNode<BuildType>::mValues) or zero if no values are stored
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_grid_type_value_strides_bits[PNANOVDB_GRID_TYPE_END]  = {  0, 32, 64, 16, 32, 64,  96, 192,  0, 16, 32,  1, 32,  4,  8, 16,  0, 128, 256,  0,  0,  0,  0, 16, 24, 48,  8 };
// bit count of the Tile union in InternalNodes, i.e. 8*sizeof(nanovdb::InternalData::Tile)
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_grid_type_table_strides_bits[PNANOVDB_GRID_TYPE_END]  = { 64, 64, 64, 64, 64, 64, 128, 192, 64, 64, 64, 64, 64, 64, 64, 64, 64, 128, 256, 64, 64, 64, 64, 64, 64, 64, 64 };
// bit count of min/max values, i.e. 8*sizeof(nanovdb::LeafData::mMinimum) or zero if no min/max exists
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_grid_type_minmax_strides_bits[PNANOVDB_GRID_TYPE_END] = {  0, 32, 64, 16, 32, 64,  96, 192,  8, 16, 32,  8, 32, 32, 32, 32, 32, 128, 256, 64, 64, 64, 64, 64, 24, 48,  8 };
// bit alignment of the value type, controlled by the smallest native type, which is why it is always 0, 8, 16, 32, or 64, e.g. for Vec3f it is 32
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_grid_type_minmax_aligns_bits[PNANOVDB_GRID_TYPE_END]  = {  0, 32, 64, 16, 32, 64,  32,  64,  8, 16, 32,  8, 32, 32, 32, 32, 32,  32,  64, 64, 64, 64, 64, 64,  8, 16,  8 };
// bit alignment of the stats (avg/std-dev) types, e.g. 8*sizeof(nanovdb::LeafData::mAverage)
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_grid_type_stat_strides_bits[PNANOVDB_GRID_TYPE_END]   = {  0, 32, 64, 32, 32, 64,  32,  64,  8, 32, 32,  8, 32, 32, 32, 32, 32,  32,  64, 64, 64, 64, 64, 64, 32, 32, 32 };
// one of the 4 leaf types defined above, e.g. PNANOVDB_LEAF_TYPE_INDEX = 3
PNANOVDB_STATIC_CONST pnanovdb_uint32_t pnanovdb_grid_type_leaf_type[PNANOVDB_GRID_TYPE_END]           = {  0,  0,  0,  0,  0,  0,  0,    0,  1,  0,  0,  1,  0,  2,  2,  2,  2,   0,   0,  3,  3,  4,  4,  5,  0,  0,  0 };

struct pnanovdb_map_t
{
    float matf[9];
    float invmatf[9];
    float vecf[3];
    float taperf;
    double matd[9];
    double invmatd[9];
    double vecd[3];
    double taperd;
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_map_t)
struct pnanovdb_map_handle_t { pnanovdb_address_t address; };
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_map_handle_t)

#define PNANOVDB_MAP_SIZE 264

#define PNANOVDB_MAP_OFF_MATF 0
#define PNANOVDB_MAP_OFF_INVMATF 36
#define PNANOVDB_MAP_OFF_VECF 72
#define PNANOVDB_MAP_OFF_TAPERF 84
#define PNANOVDB_MAP_OFF_MATD 88
#define PNANOVDB_MAP_OFF_INVMATD 160
#define PNANOVDB_MAP_OFF_VECD 232
#define PNANOVDB_MAP_OFF_TAPERD 256

PNANOVDB_FORCE_INLINE float pnanovdb_map_get_matf(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index) {
    return pnanovdb_read_float(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_MATF + 4u * index));
}
PNANOVDB_FORCE_INLINE float pnanovdb_map_get_invmatf(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index) {
    return pnanovdb_read_float(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_INVMATF + 4u * index));
}
PNANOVDB_FORCE_INLINE float pnanovdb_map_get_vecf(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index) {
    return pnanovdb_read_float(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_VECF + 4u * index));
}
PNANOVDB_FORCE_INLINE float pnanovdb_map_get_taperf(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index) {
    return pnanovdb_read_float(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_TAPERF));
}
PNANOVDB_FORCE_INLINE double pnanovdb_map_get_matd(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index) {
    return pnanovdb_read_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_MATD + 8u * index));
}
PNANOVDB_FORCE_INLINE double pnanovdb_map_get_invmatd(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index) {
    return pnanovdb_read_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_INVMATD + 8u * index));
}
PNANOVDB_FORCE_INLINE double pnanovdb_map_get_vecd(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index) {
    return pnanovdb_read_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_VECD + 8u * index));
}
PNANOVDB_FORCE_INLINE double pnanovdb_map_get_taperd(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index) {
    return pnanovdb_read_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_TAPERD));
}

PNANOVDB_FORCE_INLINE void pnanovdb_map_set_matf(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index, float matf) {
    pnanovdb_write_float(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_MATF + 4u * index), matf);
}
PNANOVDB_FORCE_INLINE void pnanovdb_map_set_invmatf(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index, float invmatf) {
    pnanovdb_write_float(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_INVMATF + 4u * index), invmatf);
}
PNANOVDB_FORCE_INLINE void pnanovdb_map_set_vecf(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index, float vecf) {
    pnanovdb_write_float(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_VECF + 4u * index), vecf);
}
PNANOVDB_FORCE_INLINE void pnanovdb_map_set_taperf(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index, float taperf) {
    pnanovdb_write_float(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_TAPERF), taperf);
}
PNANOVDB_FORCE_INLINE void pnanovdb_map_set_matd(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index, double matd) {
    pnanovdb_write_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_MATD + 8u * index), matd);
}
PNANOVDB_FORCE_INLINE void pnanovdb_map_set_invmatd(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index, double invmatd) {
    pnanovdb_write_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_INVMATD + 8u * index), invmatd);
}
PNANOVDB_FORCE_INLINE void pnanovdb_map_set_vecd(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index, double vecd) {
    pnanovdb_write_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_VECD + 8u * index), vecd);
}
PNANOVDB_FORCE_INLINE void pnanovdb_map_set_taperd(pnanovdb_buf_t buf, pnanovdb_map_handle_t p, pnanovdb_uint32_t index, double taperd) {
    pnanovdb_write_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_MAP_OFF_TAPERD), taperd);
}

struct pnanovdb_grid_t
{
    pnanovdb_uint64_t magic;                    // 8 bytes,     0
    pnanovdb_uint64_t checksum;                 // 8 bytes,     8
    pnanovdb_uint32_t version;                  // 4 bytes,     16
    pnanovdb_uint32_t flags;                    // 4 bytes,     20
    pnanovdb_uint32_t grid_index;               // 4 bytes,     24
    pnanovdb_uint32_t grid_count;               // 4 bytes,     28
    pnanovdb_uint64_t grid_size;                // 8 bytes,     32
    pnanovdb_uint32_t grid_name[256 / 4];       // 256 bytes,   40
    pnanovdb_map_t map;                         // 264 bytes,   296
    double world_bbox[6];                       // 48 bytes,    560
    double voxel_size[3];                       // 24 bytes,    608
    pnanovdb_uint32_t grid_class;               // 4 bytes,     632
    pnanovdb_uint32_t grid_type;                // 4 bytes,     636
    pnanovdb_int64_t blind_metadata_offset;     // 8 bytes,     640
    pnanovdb_uint32_t blind_metadata_count;     // 4 bytes,     648
    pnanovdb_uint32_t pad[5];                   // 20 bytes,    652
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_grid_t)
struct pnanovdb_grid_handle_t { pnanovdb_address_t address; };
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_grid_handle_t)

#define PNANOVDB_GRID_SIZE 672

#define PNANOVDB_GRID_OFF_MAGIC 0
#define PNANOVDB_GRID_OFF_CHECKSUM 8
#define PNANOVDB_GRID_OFF_VERSION 16
#define PNANOVDB_GRID_OFF_FLAGS 20
#define PNANOVDB_GRID_OFF_GRID_INDEX 24
#define PNANOVDB_GRID_OFF_GRID_COUNT 28
#define PNANOVDB_GRID_OFF_GRID_SIZE 32
#define PNANOVDB_GRID_OFF_GRID_NAME 40
#define PNANOVDB_GRID_OFF_MAP 296
#define PNANOVDB_GRID_OFF_WORLD_BBOX 560
#define PNANOVDB_GRID_OFF_VOXEL_SIZE 608
#define PNANOVDB_GRID_OFF_GRID_CLASS 632
#define PNANOVDB_GRID_OFF_GRID_TYPE 636
#define PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET 640
#define PNANOVDB_GRID_OFF_BLIND_METADATA_COUNT 648

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_grid_get_magic(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) {
    return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_MAGIC));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_grid_get_checksum(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) {
    return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_CHECKSUM));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_grid_get_version(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_VERSION));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_grid_get_flags(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_FLAGS));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_grid_get_grid_index(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_INDEX));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_grid_get_grid_count(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_COUNT));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_grid_get_grid_size(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) {
    return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_SIZE));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_grid_get_grid_name(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t index) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_NAME + 4u * index));
}
PNANOVDB_FORCE_INLINE pnanovdb_map_handle_t pnanovdb_grid_get_map(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) {
    pnanovdb_map_handle_t ret;
    ret.address = pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_MAP);
    return ret;
}
PNANOVDB_FORCE_INLINE double pnanovdb_grid_get_world_bbox(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t index) {
    return pnanovdb_read_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_WORLD_BBOX + 8u * index));
}
PNANOVDB_FORCE_INLINE double pnanovdb_grid_get_voxel_size(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t index) {
    return pnanovdb_read_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_VOXEL_SIZE + 8u * index));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_grid_get_grid_class(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_CLASS));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_grid_get_grid_type(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_TYPE));
}
PNANOVDB_FORCE_INLINE pnanovdb_int64_t pnanovdb_grid_get_blind_metadata_offset(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) {
    return pnanovdb_read_int64(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_grid_get_blind_metadata_count(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_BLIND_METADATA_COUNT));
}

PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_magic(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint64_t magic) {
    pnanovdb_write_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_MAGIC), magic);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_checksum(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint64_t checksum) {
    pnanovdb_write_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_CHECKSUM), checksum);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_version(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t version) {
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_VERSION), version);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_flags(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t flags) {
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_FLAGS), flags);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_grid_index(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t grid_index) {
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_INDEX), grid_index);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_grid_count(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t grid_count) {
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_COUNT), grid_count);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_grid_size(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint64_t grid_size) {
    pnanovdb_write_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_SIZE), grid_size);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_grid_name(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t index, pnanovdb_uint32_t grid_name) {
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_NAME + 4u * index), grid_name);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_world_bbox(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t index, double world_bbox) {
    pnanovdb_write_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_WORLD_BBOX + 8u * index), world_bbox);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_voxel_size(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t index, double voxel_size) {
    pnanovdb_write_double(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_VOXEL_SIZE + 8u * index), voxel_size);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_grid_class(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t grid_class) {
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_CLASS), grid_class);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_grid_type(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t grid_type) {
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_GRID_TYPE), grid_type);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_blind_metadata_offset(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint64_t blind_metadata_offset) {
    pnanovdb_write_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET), blind_metadata_offset);
}
PNANOVDB_FORCE_INLINE void pnanovdb_grid_set_blind_metadata_count(pnanovdb_buf_t buf, pnanovdb_grid_handle_t p, pnanovdb_uint32_t metadata_count) {
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRID_OFF_BLIND_METADATA_COUNT), metadata_count);
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_make_version(pnanovdb_uint32_t major, pnanovdb_uint32_t minor, pnanovdb_uint32_t patch_num)
{
    return (major << 21u) | (minor << 10u) | patch_num;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_version_get_major(pnanovdb_uint32_t version)
{
    return (version >> 21u) & ((1u << 11u) - 1u);
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_version_get_minor(pnanovdb_uint32_t version)
{
    return (version >> 10u) & ((1u << 11u) - 1u);
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_version_get_patch(pnanovdb_uint32_t version)
{
    return version & ((1u << 10u) - 1u);
}

struct pnanovdb_gridblindmetadata_t
{
    pnanovdb_int64_t data_offset;       // 8 bytes,     0
    pnanovdb_uint64_t value_count;      // 8 bytes,     8
    pnanovdb_uint32_t value_size;       // 4 bytes,     16
    pnanovdb_uint32_t semantic;         // 4 bytes,     20
    pnanovdb_uint32_t data_class;       // 4 bytes,     24
    pnanovdb_uint32_t data_type;        // 4 bytes,     28
    pnanovdb_uint32_t name[256 / 4];    // 256 bytes,   32
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_gridblindmetadata_t)
struct pnanovdb_gridblindmetadata_handle_t { pnanovdb_address_t address; };
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_gridblindmetadata_handle_t)

#define PNANOVDB_GRIDBLINDMETADATA_SIZE 288

#define PNANOVDB_GRIDBLINDMETADATA_OFF_DATA_OFFSET 0
#define PNANOVDB_GRIDBLINDMETADATA_OFF_VALUE_COUNT 8
#define PNANOVDB_GRIDBLINDMETADATA_OFF_VALUE_SIZE 16
#define PNANOVDB_GRIDBLINDMETADATA_OFF_SEMANTIC 20
#define PNANOVDB_GRIDBLINDMETADATA_OFF_DATA_CLASS 24
#define PNANOVDB_GRIDBLINDMETADATA_OFF_DATA_TYPE 28
#define PNANOVDB_GRIDBLINDMETADATA_OFF_NAME 32

PNANOVDB_FORCE_INLINE pnanovdb_int64_t pnanovdb_gridblindmetadata_get_data_offset(pnanovdb_buf_t buf, pnanovdb_gridblindmetadata_handle_t p) {
    return pnanovdb_read_int64(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRIDBLINDMETADATA_OFF_DATA_OFFSET));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_gridblindmetadata_get_value_count(pnanovdb_buf_t buf, pnanovdb_gridblindmetadata_handle_t p) {
    return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRIDBLINDMETADATA_OFF_VALUE_COUNT));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_gridblindmetadata_get_value_size(pnanovdb_buf_t buf, pnanovdb_gridblindmetadata_handle_t p) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRIDBLINDMETADATA_OFF_VALUE_SIZE));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_gridblindmetadata_get_semantic(pnanovdb_buf_t buf, pnanovdb_gridblindmetadata_handle_t p) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRIDBLINDMETADATA_OFF_SEMANTIC));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_gridblindmetadata_get_data_class(pnanovdb_buf_t buf, pnanovdb_gridblindmetadata_handle_t p) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRIDBLINDMETADATA_OFF_DATA_CLASS));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_gridblindmetadata_get_data_type(pnanovdb_buf_t buf, pnanovdb_gridblindmetadata_handle_t p) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRIDBLINDMETADATA_OFF_DATA_TYPE));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_gridblindmetadata_get_name(pnanovdb_buf_t buf, pnanovdb_gridblindmetadata_handle_t p, pnanovdb_uint32_t index) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_GRIDBLINDMETADATA_OFF_NAME + 4u * index));
}

struct pnanovdb_tree_t
{
    pnanovdb_uint64_t node_offset_leaf;
    pnanovdb_uint64_t node_offset_lower;
    pnanovdb_uint64_t node_offset_upper;
    pnanovdb_uint64_t node_offset_root;
    pnanovdb_uint32_t node_count_leaf;
    pnanovdb_uint32_t node_count_lower;
    pnanovdb_uint32_t node_count_upper;
    pnanovdb_uint32_t tile_count_leaf;
    pnanovdb_uint32_t tile_count_lower;
    pnanovdb_uint32_t tile_count_upper;
    pnanovdb_uint64_t voxel_count;
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_tree_t)
struct pnanovdb_tree_handle_t { pnanovdb_address_t address; };
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_tree_handle_t)

#define PNANOVDB_TREE_SIZE 64

#define PNANOVDB_TREE_OFF_NODE_OFFSET_LEAF 0
#define PNANOVDB_TREE_OFF_NODE_OFFSET_LOWER 8
#define PNANOVDB_TREE_OFF_NODE_OFFSET_UPPER 16
#define PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT 24
#define PNANOVDB_TREE_OFF_NODE_COUNT_LEAF 32
#define PNANOVDB_TREE_OFF_NODE_COUNT_LOWER 36
#define PNANOVDB_TREE_OFF_NODE_COUNT_UPPER 40
#define PNANOVDB_TREE_OFF_TILE_COUNT_LEAF 44
#define PNANOVDB_TREE_OFF_TILE_COUNT_LOWER 48
#define PNANOVDB_TREE_OFF_TILE_COUNT_UPPER 52
#define PNANOVDB_TREE_OFF_VOXEL_COUNT 56

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_tree_get_node_offset_leaf(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p) {
    return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_OFFSET_LEAF));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_tree_get_node_offset_lower(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p) {
    return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_OFFSET_LOWER));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_tree_get_node_offset_upper(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p) {
    return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_OFFSET_UPPER));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_tree_get_node_offset_root(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p) {
    return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_tree_get_node_count_leaf(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_COUNT_LEAF));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_tree_get_node_count_lower(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_COUNT_LOWER));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_tree_get_node_count_upper(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_COUNT_UPPER));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_tree_get_tile_count_leaf(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_TILE_COUNT_LEAF));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_tree_get_tile_count_lower(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_TILE_COUNT_LOWER));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_tree_get_tile_count_upper(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_TILE_COUNT_UPPER));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_tree_get_voxel_count(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p) {
    return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_VOXEL_COUNT));
}

PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_node_offset_leaf(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint64_t node_offset_leaf) {
    pnanovdb_write_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_OFFSET_LEAF), node_offset_leaf);
}
PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_node_offset_lower(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint64_t node_offset_lower) {
    pnanovdb_write_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_OFFSET_LOWER), node_offset_lower);
}
PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_node_offset_upper(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint64_t node_offset_upper) {
    pnanovdb_write_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_OFFSET_UPPER), node_offset_upper);
}
PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_node_offset_root(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint64_t node_offset_root) {
    pnanovdb_write_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT), node_offset_root);
}
PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_node_count_leaf(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint32_t node_count_leaf) {
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_COUNT_LEAF), node_count_leaf);
}
PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_node_count_lower(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint32_t node_count_lower) {
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_COUNT_LOWER), node_count_lower);
}
PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_node_count_upper(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint32_t node_count_upper) {
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_NODE_COUNT_UPPER), node_count_upper);
}
PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_tile_count_leaf(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint32_t tile_count_leaf) {
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_TILE_COUNT_LEAF), tile_count_leaf);
}
PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_tile_count_lower(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint32_t tile_count_lower) {
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_TILE_COUNT_LOWER), tile_count_lower);
}
PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_tile_count_upper(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint32_t tile_count_upper) {
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_TILE_COUNT_UPPER), tile_count_upper);
}
PNANOVDB_FORCE_INLINE void pnanovdb_tree_set_voxel_count(pnanovdb_buf_t buf, pnanovdb_tree_handle_t p, pnanovdb_uint64_t voxel_count) {
    pnanovdb_write_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_TREE_OFF_VOXEL_COUNT), voxel_count);
}

struct pnanovdb_root_t
{
    pnanovdb_coord_t bbox_min;
    pnanovdb_coord_t bbox_max;
    pnanovdb_uint32_t table_size;
    pnanovdb_uint32_t pad1;         // background can start here
    // background, min, max
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_root_t)
struct pnanovdb_root_handle_t { pnanovdb_address_t address; };
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_root_handle_t)

#define PNANOVDB_ROOT_BASE_SIZE 28

#define PNANOVDB_ROOT_OFF_BBOX_MIN 0
#define PNANOVDB_ROOT_OFF_BBOX_MAX 12
#define PNANOVDB_ROOT_OFF_TABLE_SIZE 24

PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_root_get_bbox_min(pnanovdb_buf_t buf, pnanovdb_root_handle_t p) {
    return pnanovdb_read_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_OFF_BBOX_MIN));
}
PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_root_get_bbox_max(pnanovdb_buf_t buf, pnanovdb_root_handle_t p) {
    return pnanovdb_read_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_OFF_BBOX_MAX));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_root_get_tile_count(pnanovdb_buf_t buf, pnanovdb_root_handle_t p) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_OFF_TABLE_SIZE));
}

PNANOVDB_FORCE_INLINE void pnanovdb_root_set_bbox_min(pnanovdb_buf_t buf, pnanovdb_root_handle_t p, PNANOVDB_IN(pnanovdb_coord_t) bbox_min) {
    pnanovdb_write_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_OFF_BBOX_MIN), bbox_min);
}
PNANOVDB_FORCE_INLINE void pnanovdb_root_set_bbox_max(pnanovdb_buf_t buf, pnanovdb_root_handle_t p, PNANOVDB_IN(pnanovdb_coord_t) bbox_max) {
    pnanovdb_write_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_OFF_BBOX_MAX), bbox_max);
}
PNANOVDB_FORCE_INLINE void pnanovdb_root_set_tile_count(pnanovdb_buf_t buf, pnanovdb_root_handle_t p, pnanovdb_uint32_t tile_count) {
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_OFF_TABLE_SIZE), tile_count);
}

struct pnanovdb_root_tile_t
{
    pnanovdb_uint64_t key;
    pnanovdb_int64_t child;         // signed byte offset from root to the child node, 0 means it is a constant tile, so use value
    pnanovdb_uint32_t state;
    pnanovdb_uint32_t pad1;         // value can start here
    // value
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_root_tile_t)
struct pnanovdb_root_tile_handle_t { pnanovdb_address_t address; };
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_root_tile_handle_t)

#define PNANOVDB_ROOT_TILE_BASE_SIZE 20

#define PNANOVDB_ROOT_TILE_OFF_KEY 0
#define PNANOVDB_ROOT_TILE_OFF_CHILD 8
#define PNANOVDB_ROOT_TILE_OFF_STATE 16

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_root_tile_get_key(pnanovdb_buf_t buf, pnanovdb_root_tile_handle_t p) {
    return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_TILE_OFF_KEY));
}
PNANOVDB_FORCE_INLINE pnanovdb_int64_t pnanovdb_root_tile_get_child(pnanovdb_buf_t buf, pnanovdb_root_tile_handle_t p) {
    return pnanovdb_read_int64(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_TILE_OFF_CHILD));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_root_tile_get_state(pnanovdb_buf_t buf, pnanovdb_root_tile_handle_t p) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_TILE_OFF_STATE));
}

PNANOVDB_FORCE_INLINE void pnanovdb_root_tile_set_key(pnanovdb_buf_t buf, pnanovdb_root_tile_handle_t p, pnanovdb_uint64_t key) {
    pnanovdb_write_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_TILE_OFF_KEY), key);
}
PNANOVDB_FORCE_INLINE void pnanovdb_root_tile_set_child(pnanovdb_buf_t buf, pnanovdb_root_tile_handle_t p, pnanovdb_int64_t child) {
    pnanovdb_write_int64(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_TILE_OFF_CHILD), child);
}
PNANOVDB_FORCE_INLINE void pnanovdb_root_tile_set_state(pnanovdb_buf_t buf, pnanovdb_root_tile_handle_t p, pnanovdb_uint32_t state) {
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_ROOT_TILE_OFF_STATE), state);
}

struct pnanovdb_upper_t
{
    pnanovdb_coord_t bbox_min;
    pnanovdb_coord_t bbox_max;
    pnanovdb_uint64_t flags;
    pnanovdb_uint32_t value_mask[1024];
    pnanovdb_uint32_t child_mask[1024];
    // min, max
    // alignas(32) pnanovdb_uint32_t table[];
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_upper_t)
struct pnanovdb_upper_handle_t { pnanovdb_address_t address; };
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_upper_handle_t)

#define PNANOVDB_UPPER_TABLE_COUNT 32768
#define PNANOVDB_UPPER_BASE_SIZE 8224

#define PNANOVDB_UPPER_OFF_BBOX_MIN 0
#define PNANOVDB_UPPER_OFF_BBOX_MAX 12
#define PNANOVDB_UPPER_OFF_FLAGS 24
#define PNANOVDB_UPPER_OFF_VALUE_MASK 32
#define PNANOVDB_UPPER_OFF_CHILD_MASK 4128

PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_upper_get_bbox_min(pnanovdb_buf_t buf, pnanovdb_upper_handle_t p) {
    return pnanovdb_read_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_UPPER_OFF_BBOX_MIN));
}
PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_upper_get_bbox_max(pnanovdb_buf_t buf, pnanovdb_upper_handle_t p) {
    return pnanovdb_read_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_UPPER_OFF_BBOX_MAX));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_upper_get_flags(pnanovdb_buf_t buf, pnanovdb_upper_handle_t p) {
    return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_UPPER_OFF_FLAGS));
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_upper_get_value_mask(pnanovdb_buf_t buf, pnanovdb_upper_handle_t p, pnanovdb_uint32_t bit_index) {
    pnanovdb_uint32_t value = pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_UPPER_OFF_VALUE_MASK + 4u * (bit_index >> 5u)));
    return ((value >> (bit_index & 31u)) & 1) != 0u;
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_upper_get_child_mask(pnanovdb_buf_t buf, pnanovdb_upper_handle_t p, pnanovdb_uint32_t bit_index) {
    pnanovdb_uint32_t value = pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_UPPER_OFF_CHILD_MASK + 4u * (bit_index >> 5u)));
    return ((value >> (bit_index & 31u)) & 1) != 0u;
}

PNANOVDB_FORCE_INLINE void pnanovdb_upper_set_bbox_min(pnanovdb_buf_t buf, pnanovdb_upper_handle_t p, PNANOVDB_IN(pnanovdb_coord_t) bbox_min) {
    pnanovdb_write_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_UPPER_OFF_BBOX_MIN), bbox_min);
}
PNANOVDB_FORCE_INLINE void pnanovdb_upper_set_bbox_max(pnanovdb_buf_t buf, pnanovdb_upper_handle_t p, PNANOVDB_IN(pnanovdb_coord_t) bbox_max) {
    pnanovdb_write_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_UPPER_OFF_BBOX_MAX), bbox_max);
}
PNANOVDB_FORCE_INLINE void pnanovdb_upper_set_child_mask(pnanovdb_buf_t buf, pnanovdb_upper_handle_t p, pnanovdb_uint32_t bit_index, pnanovdb_bool_t value) {
    pnanovdb_address_t addr = pnanovdb_address_offset(p.address, PNANOVDB_UPPER_OFF_CHILD_MASK + 4u * (bit_index >> 5u));
    pnanovdb_uint32_t valueMask = pnanovdb_read_uint32(buf, addr);
    if (!value) { valueMask &= ~(1u << (bit_index & 31u)); }
    if (value) valueMask |= (1u << (bit_index & 31u));
    pnanovdb_write_uint32(buf, addr, valueMask);
}

struct pnanovdb_lower_t
{
    pnanovdb_coord_t bbox_min;
    pnanovdb_coord_t bbox_max;
    pnanovdb_uint64_t flags;
    pnanovdb_uint32_t value_mask[128];
    pnanovdb_uint32_t child_mask[128];
    // min, max
    // alignas(32) pnanovdb_uint32_t table[];
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_lower_t)
struct pnanovdb_lower_handle_t { pnanovdb_address_t address; };
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_lower_handle_t)

#define PNANOVDB_LOWER_TABLE_COUNT 4096
#define PNANOVDB_LOWER_BASE_SIZE 1056

#define PNANOVDB_LOWER_OFF_BBOX_MIN 0
#define PNANOVDB_LOWER_OFF_BBOX_MAX 12
#define PNANOVDB_LOWER_OFF_FLAGS 24
#define PNANOVDB_LOWER_OFF_VALUE_MASK 32
#define PNANOVDB_LOWER_OFF_CHILD_MASK 544

PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_lower_get_bbox_min(pnanovdb_buf_t buf, pnanovdb_lower_handle_t p) {
    return pnanovdb_read_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_LOWER_OFF_BBOX_MIN));
}
PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_lower_get_bbox_max(pnanovdb_buf_t buf, pnanovdb_lower_handle_t p) {
    return pnanovdb_read_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_LOWER_OFF_BBOX_MAX));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_lower_get_flags(pnanovdb_buf_t buf, pnanovdb_lower_handle_t p) {
    return pnanovdb_read_uint64(buf, pnanovdb_address_offset(p.address, PNANOVDB_LOWER_OFF_FLAGS));
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_lower_get_value_mask(pnanovdb_buf_t buf, pnanovdb_lower_handle_t p, pnanovdb_uint32_t bit_index) {
    pnanovdb_uint32_t value = pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_LOWER_OFF_VALUE_MASK + 4u * (bit_index >> 5u)));
    return ((value >> (bit_index & 31u)) & 1) != 0u;
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_lower_get_child_mask(pnanovdb_buf_t buf, pnanovdb_lower_handle_t p, pnanovdb_uint32_t bit_index) {
    pnanovdb_uint32_t value = pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_LOWER_OFF_CHILD_MASK + 4u * (bit_index >> 5u)));
    return ((value >> (bit_index & 31u)) & 1) != 0u;
}

PNANOVDB_FORCE_INLINE void pnanovdb_lower_set_bbox_min(pnanovdb_buf_t buf, pnanovdb_lower_handle_t p, PNANOVDB_IN(pnanovdb_coord_t) bbox_min) {
    pnanovdb_write_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_LOWER_OFF_BBOX_MIN), bbox_min);
}
PNANOVDB_FORCE_INLINE void pnanovdb_lower_set_bbox_max(pnanovdb_buf_t buf, pnanovdb_lower_handle_t p, PNANOVDB_IN(pnanovdb_coord_t) bbox_max) {
    pnanovdb_write_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_LOWER_OFF_BBOX_MAX), bbox_max);
}
PNANOVDB_FORCE_INLINE void pnanovdb_lower_set_child_mask(pnanovdb_buf_t buf, pnanovdb_lower_handle_t p, pnanovdb_uint32_t bit_index, pnanovdb_bool_t value) {
    pnanovdb_address_t addr = pnanovdb_address_offset(p.address, PNANOVDB_LOWER_OFF_CHILD_MASK + 4u * (bit_index >> 5u));
    pnanovdb_uint32_t valueMask = pnanovdb_read_uint32(buf, addr);
    if (!value) { valueMask &= ~(1u << (bit_index & 31u)); }
    if (value) valueMask |= (1u << (bit_index & 31u));
    pnanovdb_write_uint32(buf, addr, valueMask);
}

struct pnanovdb_leaf_t
{
    pnanovdb_coord_t bbox_min;
    pnanovdb_uint32_t bbox_dif_and_flags;
    pnanovdb_uint32_t value_mask[16];
    // min, max
    // alignas(32) pnanovdb_uint32_t values[];
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_leaf_t)
struct pnanovdb_leaf_handle_t { pnanovdb_address_t address; };
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_leaf_handle_t)

#define PNANOVDB_LEAF_TABLE_COUNT 512
#define PNANOVDB_LEAF_BASE_SIZE 80

#define PNANOVDB_LEAF_OFF_BBOX_MIN 0
#define PNANOVDB_LEAF_OFF_BBOX_DIF_AND_FLAGS 12
#define PNANOVDB_LEAF_OFF_VALUE_MASK 16

#define PNANOVDB_LEAF_TABLE_NEG_OFF_BBOX_DIF_AND_FLAGS 84
#define PNANOVDB_LEAF_TABLE_NEG_OFF_MINIMUM 16
#define PNANOVDB_LEAF_TABLE_NEG_OFF_QUANTUM 12

PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_leaf_get_bbox_min(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t p) {
    return pnanovdb_read_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_LEAF_OFF_BBOX_MIN));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_leaf_get_bbox_dif_and_flags(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t p) {
    return pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_LEAF_OFF_BBOX_DIF_AND_FLAGS));
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_leaf_get_value_mask(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t p, pnanovdb_uint32_t bit_index) {
    pnanovdb_uint32_t value = pnanovdb_read_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_LEAF_OFF_VALUE_MASK + 4u * (bit_index >> 5u)));
    return ((value >> (bit_index & 31u)) & 1) != 0u;
}

PNANOVDB_FORCE_INLINE void pnanovdb_leaf_set_bbox_min(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t p, PNANOVDB_IN(pnanovdb_coord_t) bbox_min) {
    pnanovdb_write_coord(buf, pnanovdb_address_offset(p.address, PNANOVDB_LEAF_OFF_BBOX_MIN), bbox_min);
}
PNANOVDB_FORCE_INLINE void pnanovdb_leaf_set_bbox_dif_and_flags(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t p, pnanovdb_uint32_t bbox_dif_and_flags) {
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(p.address, PNANOVDB_LEAF_OFF_BBOX_DIF_AND_FLAGS), bbox_dif_and_flags);
}

struct pnanovdb_grid_type_constants_t
{
    pnanovdb_uint32_t root_off_background;
    pnanovdb_uint32_t root_off_min;
    pnanovdb_uint32_t root_off_max;
    pnanovdb_uint32_t root_off_ave;
    pnanovdb_uint32_t root_off_stddev;
    pnanovdb_uint32_t root_size;
    pnanovdb_uint32_t value_stride_bits;
    pnanovdb_uint32_t table_stride;
    pnanovdb_uint32_t root_tile_off_value;
    pnanovdb_uint32_t root_tile_size;
    pnanovdb_uint32_t upper_off_min;
    pnanovdb_uint32_t upper_off_max;
    pnanovdb_uint32_t upper_off_ave;
    pnanovdb_uint32_t upper_off_stddev;
    pnanovdb_uint32_t upper_off_table;
    pnanovdb_uint32_t upper_size;
    pnanovdb_uint32_t lower_off_min;
    pnanovdb_uint32_t lower_off_max;
    pnanovdb_uint32_t lower_off_ave;
    pnanovdb_uint32_t lower_off_stddev;
    pnanovdb_uint32_t lower_off_table;
    pnanovdb_uint32_t lower_size;
    pnanovdb_uint32_t leaf_off_min;
    pnanovdb_uint32_t leaf_off_max;
    pnanovdb_uint32_t leaf_off_ave;
    pnanovdb_uint32_t leaf_off_stddev;
    pnanovdb_uint32_t leaf_off_table;
    pnanovdb_uint32_t leaf_size;
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_grid_type_constants_t)

// The following table with offsets will nedd to be updates as new GridTypes are added in NanoVDB.h
PNANOVDB_STATIC_CONST pnanovdb_grid_type_constants_t pnanovdb_grid_type_constants[PNANOVDB_GRID_TYPE_END] =
{
{28, 28, 28, 28, 28, 32,  0, 8, 20, 32,  8224, 8224, 8224, 8224, 8224, 270368,  1056, 1056, 1056, 1056, 1056, 33824,  80, 80, 80, 80, 96, 96},
{28, 32, 36, 40, 44, 64,  32, 8, 20, 32,  8224, 8228, 8232, 8236, 8256, 270400,  1056, 1060, 1064, 1068, 1088, 33856,  80, 84, 88, 92, 96, 2144},
{32, 40, 48, 56, 64, 96,  64, 8, 24, 32,  8224, 8232, 8240, 8248, 8256, 270400,  1056, 1064, 1072, 1080, 1088, 33856,  80, 88, 96, 104, 128, 4224},
{28, 30, 32, 36, 40, 64,  16, 8, 20, 32,  8224, 8226, 8228, 8232, 8256, 270400,  1056, 1058, 1060, 1064, 1088, 33856,  80, 82, 84, 88, 96, 1120},
{28, 32, 36, 40, 44, 64,  32, 8, 20, 32,  8224, 8228, 8232, 8236, 8256, 270400,  1056, 1060, 1064, 1068, 1088, 33856,  80, 84, 88, 92, 96, 2144},
{32, 40, 48, 56, 64, 96,  64, 8, 24, 32,  8224, 8232, 8240, 8248, 8256, 270400,  1056, 1064, 1072, 1080, 1088, 33856,  80, 88, 96, 104, 128, 4224},
{28, 40, 52, 64, 68, 96,  96, 16, 20, 32,  8224, 8236, 8248, 8252, 8256, 532544,  1056, 1068, 1080, 1084, 1088, 66624,  80, 92, 104, 108, 128, 6272},
{32, 56, 80, 104, 112, 128,  192, 24, 24, 64,  8224, 8248, 8272, 8280, 8288, 794720,  1056, 1080, 1104, 1112, 1120, 99424,  80, 104, 128, 136, 160, 12448},
{28, 29, 30, 31, 32, 64,  0, 8, 20, 32,  8224, 8225, 8226, 8227, 8256, 270400,  1056, 1057, 1058, 1059, 1088, 33856,  80, 80, 80, 80, 96, 96},
{28, 30, 32, 36, 40, 64,  16, 8, 20, 32,  8224, 8226, 8228, 8232, 8256, 270400,  1056, 1058, 1060, 1064, 1088, 33856,  80, 82, 84, 88, 96, 1120},
{28, 32, 36, 40, 44, 64,  32, 8, 20, 32,  8224, 8228, 8232, 8236, 8256, 270400,  1056, 1060, 1064, 1068, 1088, 33856,  80, 84, 88, 92, 96, 2144},
{28, 29, 30, 31, 32, 64,  1, 8, 20, 32,  8224, 8225, 8226, 8227, 8256, 270400,  1056, 1057, 1058, 1059, 1088, 33856,  80, 80, 80, 80, 96, 160},
{28, 32, 36, 40, 44, 64,  32, 8, 20, 32,  8224, 8228, 8232, 8236, 8256, 270400,  1056, 1060, 1064, 1068, 1088, 33856,  80, 84, 88, 92, 96, 2144},
{28, 32, 36, 40, 44, 64,  0, 8, 20, 32,  8224, 8228, 8232, 8236, 8256, 270400,  1056, 1060, 1064, 1068, 1088, 33856,  88, 90, 92, 94, 96, 352},
{28, 32, 36, 40, 44, 64,  0, 8, 20, 32,  8224, 8228, 8232, 8236, 8256, 270400,  1056, 1060, 1064, 1068, 1088, 33856,  88, 90, 92, 94, 96, 608},
{28, 32, 36, 40, 44, 64,  0, 8, 20, 32,  8224, 8228, 8232, 8236, 8256, 270400,  1056, 1060, 1064, 1068, 1088, 33856,  88, 90, 92, 94, 96, 1120},
{28, 32, 36, 40, 44, 64,  0, 8, 20, 32,  8224, 8228, 8232, 8236, 8256, 270400,  1056, 1060, 1064, 1068, 1088, 33856,  88, 90, 92, 94, 96, 96},
{28, 44, 60, 76, 80, 96,  128, 16, 20, 64,  8224, 8240, 8256, 8260, 8288, 532576,  1056, 1072, 1088, 1092, 1120, 66656,  80, 96, 112, 116, 128, 8320},
{32, 64, 96, 128, 136, 160,  256, 32, 24, 64,  8224, 8256, 8288, 8296, 8320, 1056896,  1056, 1088, 1120, 1128, 1152, 132224,  80, 112, 144, 152, 160, 16544},
{32, 40, 48, 56, 64, 96,  0, 8, 24, 32,  8224, 8232, 8240, 8248, 8256, 270400,  1056, 1064, 1072, 1080, 1088, 33856,  80, 80, 80, 80, 80, 96},
{32, 40, 48, 56, 64, 96,  0, 8, 24, 32,  8224, 8232, 8240, 8248, 8256, 270400,  1056, 1064, 1072, 1080, 1088, 33856,  80, 80, 80, 80, 80, 96},
{32, 40, 48, 56, 64, 96,  0, 8, 24, 32,  8224, 8232, 8240, 8248, 8256, 270400,  1056, 1064, 1072, 1080, 1088, 33856,  80, 80, 80, 80, 80, 160},
{32, 40, 48, 56, 64, 96,  0, 8, 24, 32,  8224, 8232, 8240, 8248, 8256, 270400,  1056, 1064, 1072, 1080, 1088, 33856,  80, 80, 80, 80, 80, 160},
{32, 40, 48, 56, 64, 96,  16, 8, 24, 32,  8224, 8232, 8240, 8248, 8256, 270400,  1056, 1064, 1072, 1080, 1088, 33856,  80, 88, 96, 96, 96, 1120},
{28, 31, 34, 40, 44, 64,  24, 8, 20, 32,  8224, 8227, 8232, 8236, 8256, 270400,  1056, 1059, 1064, 1068, 1088, 33856,  80, 83, 88, 92, 96, 1632},
{28, 34, 40, 48, 52, 64,  48, 8, 20, 32,  8224, 8230, 8236, 8240, 8256, 270400,  1056, 1062, 1068, 1072, 1088, 33856,  80, 86, 92, 96, 128, 3200},
{28, 29, 30, 32, 36, 64,  8, 8, 20, 32,  8224, 8225, 8228, 8232, 8256, 270400,  1056, 1057, 1060, 1064, 1088, 33856,  80, 81, 84, 88, 96, 608},
};

// ------------------------------------------------ Basic Lookup -----------------------------------------------------------

PNANOVDB_FORCE_INLINE pnanovdb_gridblindmetadata_handle_t pnanovdb_grid_get_gridblindmetadata(pnanovdb_buf_t buf, pnanovdb_grid_handle_t grid, pnanovdb_uint32_t index)
{
    pnanovdb_gridblindmetadata_handle_t meta = { grid.address };
    pnanovdb_uint64_t byte_offset = pnanovdb_grid_get_blind_metadata_offset(buf, grid);
    meta.address = pnanovdb_address_offset64(meta.address, byte_offset);
    meta.address = pnanovdb_address_offset_product(meta.address, PNANOVDB_GRIDBLINDMETADATA_SIZE, index);
    return meta;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_grid_get_gridblindmetadata_value_address(pnanovdb_buf_t buf, pnanovdb_grid_handle_t grid, pnanovdb_uint32_t index)
{
    pnanovdb_gridblindmetadata_handle_t meta = pnanovdb_grid_get_gridblindmetadata(buf, grid, index);
    pnanovdb_int64_t byte_offset = pnanovdb_gridblindmetadata_get_data_offset(buf, meta);
    pnanovdb_address_t address = pnanovdb_address_offset64(meta.address, pnanovdb_int64_as_uint64(byte_offset));
    return address;
}

PNANOVDB_FORCE_INLINE pnanovdb_tree_handle_t pnanovdb_grid_get_tree(pnanovdb_buf_t buf, pnanovdb_grid_handle_t grid)
{
    pnanovdb_tree_handle_t tree = { grid.address };
    tree.address = pnanovdb_address_offset(tree.address, PNANOVDB_GRID_SIZE);
    return tree;
}

PNANOVDB_FORCE_INLINE pnanovdb_root_handle_t pnanovdb_tree_get_root(pnanovdb_buf_t buf, pnanovdb_tree_handle_t tree)
{
    pnanovdb_root_handle_t root = { tree.address };
    pnanovdb_uint64_t byte_offset = pnanovdb_tree_get_node_offset_root(buf, tree);
    root.address = pnanovdb_address_offset64(root.address, byte_offset);
    return root;
}

PNANOVDB_FORCE_INLINE pnanovdb_root_tile_handle_t pnanovdb_root_get_tile(pnanovdb_grid_type_t grid_type, pnanovdb_root_handle_t root, pnanovdb_uint32_t n)
{
    pnanovdb_root_tile_handle_t tile = { root.address };
    tile.address = pnanovdb_address_offset(tile.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_size));
    tile.address = pnanovdb_address_offset_product(tile.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_tile_size), n);
    return tile;
}

PNANOVDB_FORCE_INLINE pnanovdb_root_tile_handle_t pnanovdb_root_get_tile_zero(pnanovdb_grid_type_t grid_type, pnanovdb_root_handle_t root)
{
    pnanovdb_root_tile_handle_t tile = { root.address };
    tile.address = pnanovdb_address_offset(tile.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_size));
    return tile;
}

PNANOVDB_FORCE_INLINE pnanovdb_upper_handle_t pnanovdb_root_get_child(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root, pnanovdb_root_tile_handle_t tile)
{
    pnanovdb_upper_handle_t upper = { root.address };
    upper.address = pnanovdb_address_offset64(upper.address, pnanovdb_int64_as_uint64(pnanovdb_root_tile_get_child(buf, tile)));
    return upper;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_coord_to_key(PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
#if defined(PNANOVDB_NATIVE_64)
    pnanovdb_uint64_t iu = pnanovdb_int32_as_uint32(PNANOVDB_DEREF(ijk).x) >> 12u;
    pnanovdb_uint64_t ju = pnanovdb_int32_as_uint32(PNANOVDB_DEREF(ijk).y) >> 12u;
    pnanovdb_uint64_t ku = pnanovdb_int32_as_uint32(PNANOVDB_DEREF(ijk).z) >> 12u;
    return (ku) | (ju << 21u) | (iu << 42u);
#else
    pnanovdb_uint32_t iu = pnanovdb_int32_as_uint32(PNANOVDB_DEREF(ijk).x) >> 12u;
    pnanovdb_uint32_t ju = pnanovdb_int32_as_uint32(PNANOVDB_DEREF(ijk).y) >> 12u;
    pnanovdb_uint32_t ku = pnanovdb_int32_as_uint32(PNANOVDB_DEREF(ijk).z) >> 12u;
    pnanovdb_uint32_t key_x = ku | (ju << 21);
    pnanovdb_uint32_t key_y = (iu << 10) | (ju >> 11);
    return pnanovdb_uint32_as_uint64(key_x, key_y);
#endif
}

PNANOVDB_FORCE_INLINE pnanovdb_root_tile_handle_t pnanovdb_root_find_tile(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    pnanovdb_uint32_t tile_count = pnanovdb_uint32_as_int32(pnanovdb_root_get_tile_count(buf, root));
    pnanovdb_root_tile_handle_t tile = pnanovdb_root_get_tile_zero(grid_type, root);
    pnanovdb_uint64_t key = pnanovdb_coord_to_key(ijk);
    for (pnanovdb_uint32_t i = 0u; i < tile_count; i++)
    {
        if (pnanovdb_uint64_is_equal(key, pnanovdb_root_tile_get_key(buf, tile)))
        {
            return tile;
        }
        tile.address = pnanovdb_address_offset(tile.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_tile_size));
    }
    pnanovdb_root_tile_handle_t null_handle = { pnanovdb_address_null() };
    return null_handle;
}

// ----------------------------- Leaf Node ---------------------------------------

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_leaf_coord_to_offset(PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    return (((PNANOVDB_DEREF(ijk).x & 7) >> 0) << (2 * 3)) +
        (((PNANOVDB_DEREF(ijk).y & 7) >> 0) << (3)) +
        ((PNANOVDB_DEREF(ijk).z & 7) >> 0);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_leaf_get_min_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_leaf_handle_t node)
{
    pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, leaf_off_min);
    return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_leaf_get_max_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_leaf_handle_t node)
{
    pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, leaf_off_max);
    return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_leaf_get_ave_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_leaf_handle_t node)
{
    pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, leaf_off_ave);
    return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_leaf_get_stddev_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_leaf_handle_t node)
{
    pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, leaf_off_stddev);
    return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_leaf_get_table_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_leaf_handle_t node, pnanovdb_uint32_t n)
{
    pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, leaf_off_table) + ((PNANOVDB_GRID_TYPE_GET(grid_type, value_stride_bits) * n) >> 3u);
    return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_leaf_get_value_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    pnanovdb_uint32_t n = pnanovdb_leaf_coord_to_offset(ijk);
    return pnanovdb_leaf_get_table_address(grid_type, buf, leaf, n);
}

// ----------------------------- Leaf FP Types Specialization ---------------------------------------

PNANOVDB_FORCE_INLINE float pnanovdb_leaf_fp_read_float(pnanovdb_buf_t buf, pnanovdb_address_t address, PNANOVDB_IN(pnanovdb_coord_t) ijk, pnanovdb_uint32_t value_log_bits)
{
    //  value_log_bits                                                          //   2     3       4
    pnanovdb_uint32_t value_bits = 1u << value_log_bits;                        //   4     8      16
    pnanovdb_uint32_t value_mask = (1u << value_bits) - 1u;                     // 0xF  0xFF  0xFFFF
    pnanovdb_uint32_t values_per_word_bits = 5u - value_log_bits;               //   3     2       1
    pnanovdb_uint32_t values_per_word_mask = (1u << values_per_word_bits) - 1u; //   7     3       1

    pnanovdb_uint32_t n = pnanovdb_leaf_coord_to_offset(ijk);
    float minimum = pnanovdb_read_float(buf, pnanovdb_address_offset_neg(address, PNANOVDB_LEAF_TABLE_NEG_OFF_MINIMUM));
    float quantum = pnanovdb_read_float(buf, pnanovdb_address_offset_neg(address, PNANOVDB_LEAF_TABLE_NEG_OFF_QUANTUM));
    pnanovdb_uint32_t raw = pnanovdb_read_uint32(buf, pnanovdb_address_offset(address, ((n >> values_per_word_bits) << 2u)));
    pnanovdb_uint32_t value_compressed = (raw >> ((n & values_per_word_mask) << value_log_bits)) & value_mask;
    return pnanovdb_uint32_to_float(value_compressed) * quantum + minimum;
}

PNANOVDB_FORCE_INLINE float pnanovdb_leaf_fp4_read_float(pnanovdb_buf_t buf, pnanovdb_address_t address, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    return pnanovdb_leaf_fp_read_float(buf, address, ijk, 2u);
}

PNANOVDB_FORCE_INLINE float pnanovdb_leaf_fp8_read_float(pnanovdb_buf_t buf, pnanovdb_address_t address, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    return pnanovdb_leaf_fp_read_float(buf, address, ijk, 3u);
}

PNANOVDB_FORCE_INLINE float pnanovdb_leaf_fp16_read_float(pnanovdb_buf_t buf, pnanovdb_address_t address, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    return pnanovdb_leaf_fp_read_float(buf, address, ijk, 4u);
}

PNANOVDB_FORCE_INLINE float pnanovdb_leaf_fpn_read_float(pnanovdb_buf_t buf, pnanovdb_address_t address, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    pnanovdb_uint32_t bbox_dif_and_flags = pnanovdb_read_uint32(buf, pnanovdb_address_offset_neg(address, PNANOVDB_LEAF_TABLE_NEG_OFF_BBOX_DIF_AND_FLAGS));
    pnanovdb_uint32_t flags = bbox_dif_and_flags >> 24u;
    pnanovdb_uint32_t value_log_bits = flags >> 5; // b = 0, 1, 2, 3, 4 corresponding to 1, 2, 4, 8, 16 bits
    return pnanovdb_leaf_fp_read_float(buf, address, ijk, value_log_bits);
}

// ----------------------------- Leaf Index Specialization ---------------------------------------

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_leaf_index_has_stats(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf)
{
    return (pnanovdb_leaf_get_bbox_dif_and_flags(buf, leaf) & (1u << 28u)) != 0u;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_index_get_min_index(pnanovdb_buf_t buf, pnanovdb_address_t min_address)
{
    return pnanovdb_uint64_offset(pnanovdb_read_uint64(buf, min_address), 512u);
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_index_get_max_index(pnanovdb_buf_t buf, pnanovdb_address_t max_address)
{
    return pnanovdb_uint64_offset(pnanovdb_read_uint64(buf, max_address), 513u);
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_index_get_ave_index(pnanovdb_buf_t buf, pnanovdb_address_t ave_address)
{
    return pnanovdb_uint64_offset(pnanovdb_read_uint64(buf, ave_address), 514u);
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_index_get_dev_index(pnanovdb_buf_t buf, pnanovdb_address_t dev_address)
{
    return pnanovdb_uint64_offset(pnanovdb_read_uint64(buf, dev_address), 515u);
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_index_get_value_index(pnanovdb_buf_t buf, pnanovdb_address_t value_address, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    pnanovdb_uint32_t n = pnanovdb_leaf_coord_to_offset(ijk);
    pnanovdb_uint64_t offset = pnanovdb_read_uint64(buf, value_address);
    return pnanovdb_uint64_offset(offset, n);
}

// ----------------------------- Leaf IndexMask Specialization ---------------------------------------

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_leaf_indexmask_has_stats(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf)
{
    return pnanovdb_leaf_index_has_stats(buf, leaf);
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_indexmask_get_min_index(pnanovdb_buf_t buf, pnanovdb_address_t min_address)
{
    return pnanovdb_leaf_index_get_min_index(buf, min_address);
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_indexmask_get_max_index(pnanovdb_buf_t buf, pnanovdb_address_t max_address)
{
    return pnanovdb_leaf_index_get_max_index(buf, max_address);
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_indexmask_get_ave_index(pnanovdb_buf_t buf, pnanovdb_address_t ave_address)
{
    return pnanovdb_leaf_index_get_ave_index(buf, ave_address);
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_indexmask_get_dev_index(pnanovdb_buf_t buf, pnanovdb_address_t dev_address)
{
    return pnanovdb_leaf_index_get_dev_index(buf, dev_address);
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_indexmask_get_value_index(pnanovdb_buf_t buf, pnanovdb_address_t value_address, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    return pnanovdb_leaf_index_get_value_index(buf, value_address, ijk);
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_leaf_indexmask_get_mask_bit(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf, pnanovdb_uint32_t n)
{
    pnanovdb_uint32_t word_idx = n >> 5;
    pnanovdb_uint32_t bit_idx = n & 31;
    pnanovdb_uint32_t val_mask =
        pnanovdb_read_uint32(buf, pnanovdb_address_offset(leaf.address, 96u + 4u * word_idx));
    return (val_mask & (1u << bit_idx)) != 0u;
}
PNANOVDB_FORCE_INLINE void pnanovdb_leaf_indexmask_set_mask_bit(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf, pnanovdb_uint32_t n, pnanovdb_bool_t v)
{
    pnanovdb_uint32_t word_idx = n >> 5;
    pnanovdb_uint32_t bit_idx = n & 31;
    pnanovdb_uint32_t val_mask =
        pnanovdb_read_uint32(buf, pnanovdb_address_offset(leaf.address, 96u + 4u * word_idx));
    if (v)
    {
        val_mask = val_mask | (1u << bit_idx);
    }
    else
    {
        val_mask = val_mask & ~(1u << bit_idx);
    }
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(leaf.address, 96u + 4u * word_idx), val_mask);
}

// ----------------------------- Leaf OnIndex Specialization ---------------------------------------

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_leaf_onindex_get_value_count(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf)
{
    pnanovdb_uint64_t val_mask = pnanovdb_read_uint64(buf, pnanovdb_address_offset(leaf.address, PNANOVDB_LEAF_OFF_VALUE_MASK + 8u * 7u));
    pnanovdb_uint64_t prefix_sum = pnanovdb_read_uint64(
        buf, pnanovdb_address_offset(leaf.address, PNANOVDB_GRID_TYPE_GET(PNANOVDB_GRID_TYPE_ONINDEX, leaf_off_table) + 8u));
    return pnanovdb_uint64_countbits(val_mask) + (pnanovdb_uint64_to_uint32_lsr(prefix_sum, 54u) & 511u);
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_onindex_get_last_offset(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf)
{
    return pnanovdb_uint64_offset(
        pnanovdb_read_uint64(buf, pnanovdb_address_offset(leaf.address, PNANOVDB_GRID_TYPE_GET(PNANOVDB_GRID_TYPE_ONINDEX, leaf_off_table))),
        pnanovdb_leaf_onindex_get_value_count(buf, leaf) - 1u);
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_leaf_onindex_has_stats(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf)
{
    return (pnanovdb_leaf_get_bbox_dif_and_flags(buf, leaf) & (1u << 28u)) != 0u;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_onindex_get_min_index(pnanovdb_buf_t buf, pnanovdb_address_t min_address)
{
    pnanovdb_leaf_handle_t leaf = { pnanovdb_address_offset_neg(min_address, PNANOVDB_GRID_TYPE_GET(PNANOVDB_GRID_TYPE_ONINDEX, leaf_off_table)) };
    pnanovdb_uint64_t idx = pnanovdb_uint32_as_uint64_low(0u);
    if (pnanovdb_leaf_onindex_has_stats(buf, leaf))
    {
        idx = pnanovdb_uint64_offset(pnanovdb_leaf_onindex_get_last_offset(buf, leaf), 1u);
    }
    return idx;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_onindex_get_max_index(pnanovdb_buf_t buf, pnanovdb_address_t max_address)
{
    pnanovdb_leaf_handle_t leaf = { pnanovdb_address_offset_neg(max_address, PNANOVDB_GRID_TYPE_GET(PNANOVDB_GRID_TYPE_ONINDEX, leaf_off_table)) };
    pnanovdb_uint64_t idx = pnanovdb_uint32_as_uint64_low(0u);
    if (pnanovdb_leaf_onindex_has_stats(buf, leaf))
    {
        idx = pnanovdb_uint64_offset(pnanovdb_leaf_onindex_get_last_offset(buf, leaf), 2u);
    }
    return idx;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_onindex_get_ave_index(pnanovdb_buf_t buf, pnanovdb_address_t ave_address)
{
    pnanovdb_leaf_handle_t leaf = { pnanovdb_address_offset_neg(ave_address, PNANOVDB_GRID_TYPE_GET(PNANOVDB_GRID_TYPE_ONINDEX, leaf_off_table)) };
    pnanovdb_uint64_t idx = pnanovdb_uint32_as_uint64_low(0u);
    if (pnanovdb_leaf_onindex_has_stats(buf, leaf))
    {
        idx = pnanovdb_uint64_offset(pnanovdb_leaf_onindex_get_last_offset(buf, leaf), 3u);
    }
    return idx;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_onindex_get_dev_index(pnanovdb_buf_t buf, pnanovdb_address_t dev_address)
{
    pnanovdb_leaf_handle_t leaf = { pnanovdb_address_offset_neg(dev_address, PNANOVDB_GRID_TYPE_GET(PNANOVDB_GRID_TYPE_ONINDEX, leaf_off_table)) };
    pnanovdb_uint64_t idx = pnanovdb_uint32_as_uint64_low(0u);
    if (pnanovdb_leaf_onindex_has_stats(buf, leaf))
    {
        idx = pnanovdb_uint64_offset(pnanovdb_leaf_onindex_get_last_offset(buf, leaf), 4u);
    }
    return idx;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_onindex_get_value_index(pnanovdb_buf_t buf, pnanovdb_address_t value_address, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    pnanovdb_uint32_t n = pnanovdb_leaf_coord_to_offset(ijk);
    pnanovdb_leaf_handle_t leaf = { pnanovdb_address_offset_neg(value_address, PNANOVDB_GRID_TYPE_GET(PNANOVDB_GRID_TYPE_ONINDEX, leaf_off_table)) };

    pnanovdb_uint32_t word_idx = n >> 6u;
    pnanovdb_uint32_t bit_idx = n & 63u;
    pnanovdb_uint64_t val_mask = pnanovdb_read_uint64(buf, pnanovdb_address_offset(leaf.address, PNANOVDB_LEAF_OFF_VALUE_MASK + 8u * word_idx));
    pnanovdb_uint64_t mask = pnanovdb_uint64_bit_mask(bit_idx);
    pnanovdb_uint64_t value_index = pnanovdb_uint32_as_uint64_low(0u);
    if (pnanovdb_uint64_any_bit(pnanovdb_uint64_and(val_mask, mask)))
    {
        pnanovdb_uint32_t sum = 0u;
        sum += pnanovdb_uint64_countbits(pnanovdb_uint64_and(val_mask, pnanovdb_uint64_dec(mask)));
        if (word_idx > 0u)
        {
            pnanovdb_uint64_t prefix_sum = pnanovdb_read_uint64(buf, pnanovdb_address_offset(value_address, 8u));
            sum += pnanovdb_uint64_to_uint32_lsr(prefix_sum, 9u * (word_idx - 1u)) & 511u;
        }
        pnanovdb_uint64_t offset = pnanovdb_read_uint64(buf, value_address);
        value_index = pnanovdb_uint64_offset(offset, sum);
    }
    return value_index;
}

// ----------------------------- Leaf OnIndexMask Specialization ---------------------------------------

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_leaf_onindexmask_get_value_count(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf)
{
    return pnanovdb_leaf_onindex_get_value_count(buf, leaf);
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_onindexmask_get_last_offset(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf)
{
    return pnanovdb_leaf_onindex_get_last_offset(buf, leaf);
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_leaf_onindexmask_has_stats(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf)
{
    return pnanovdb_leaf_onindex_has_stats(buf, leaf);
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_onindexmask_get_min_index(pnanovdb_buf_t buf, pnanovdb_address_t min_address)
{
    return pnanovdb_leaf_onindex_get_min_index(buf, min_address);
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_onindexmask_get_max_index(pnanovdb_buf_t buf, pnanovdb_address_t max_address)
{
    return pnanovdb_leaf_onindex_get_max_index(buf, max_address);
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_onindexmask_get_ave_index(pnanovdb_buf_t buf, pnanovdb_address_t ave_address)
{
    return pnanovdb_leaf_onindex_get_ave_index(buf, ave_address);
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_onindexmask_get_dev_index(pnanovdb_buf_t buf, pnanovdb_address_t dev_address)
{
    return pnanovdb_leaf_onindex_get_dev_index(buf, dev_address);
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_onindexmask_get_value_index(pnanovdb_buf_t buf, pnanovdb_address_t value_address, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    return pnanovdb_leaf_onindex_get_value_index(buf, value_address, ijk);
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_leaf_onindexmask_get_mask_bit(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf, pnanovdb_uint32_t n)
{
    pnanovdb_uint32_t word_idx = n >> 5;
    pnanovdb_uint32_t bit_idx = n & 31;
    pnanovdb_uint32_t val_mask =
        pnanovdb_read_uint32(buf, pnanovdb_address_offset(leaf.address, 96u + 4u * word_idx));
    return (val_mask & (1u << bit_idx)) != 0u;
}
PNANOVDB_FORCE_INLINE void pnanovdb_leaf_onindexmask_set_mask_bit(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf, pnanovdb_uint32_t n, pnanovdb_bool_t v)
{
    pnanovdb_uint32_t word_idx = n >> 5;
    pnanovdb_uint32_t bit_idx = n & 31;
    pnanovdb_uint32_t val_mask =
        pnanovdb_read_uint32(buf, pnanovdb_address_offset(leaf.address, 96u + 4u * word_idx));
    if (v)
    {
        val_mask = val_mask | (1u << bit_idx);
    }
    else
    {
        val_mask = val_mask & ~(1u << bit_idx);
    }
    pnanovdb_write_uint32(buf, pnanovdb_address_offset(leaf.address, 96u + 4u * word_idx), val_mask);
}

// ----------------------------- Leaf PointIndex Specialization ---------------------------------------

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_pointindex_get_offset(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf)
{
    return pnanovdb_read_uint64(buf, pnanovdb_leaf_get_min_address(PNANOVDB_GRID_TYPE_POINTINDEX, buf, leaf));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_pointindex_get_point_count(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf)
{
    return pnanovdb_read_uint64(buf, pnanovdb_leaf_get_max_address(PNANOVDB_GRID_TYPE_POINTINDEX, buf, leaf));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_pointindex_get_first(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf, pnanovdb_uint32_t i)
{
    return pnanovdb_uint64_offset(pnanovdb_leaf_pointindex_get_offset(buf, leaf),
        (i == 0u ? 0u : pnanovdb_read_uint16(buf, pnanovdb_leaf_get_table_address(PNANOVDB_GRID_TYPE_POINTINDEX, buf, leaf, i - 1u))));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_pointindex_get_last(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf, pnanovdb_uint32_t i)
{
    return pnanovdb_uint64_offset(pnanovdb_leaf_pointindex_get_offset(buf, leaf),
        pnanovdb_read_uint16(buf, pnanovdb_leaf_get_table_address(PNANOVDB_GRID_TYPE_POINTINDEX, buf, leaf, i)));
}
PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_leaf_pointindex_get_value(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf, pnanovdb_uint32_t i)
{
    return pnanovdb_uint32_as_uint64_low(pnanovdb_read_uint16(buf, pnanovdb_leaf_get_table_address(PNANOVDB_GRID_TYPE_POINTINDEX, buf, leaf, i)));
}
PNANOVDB_FORCE_INLINE void pnanovdb_leaf_pointindex_set_value_only(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf, pnanovdb_uint32_t i, pnanovdb_uint32_t value)
{
    pnanovdb_address_t addr = pnanovdb_leaf_get_table_address(PNANOVDB_GRID_TYPE_POINTINDEX, buf, leaf, i);
    pnanovdb_uint32_t raw32 = pnanovdb_read_uint32(buf, pnanovdb_address_mask_inv(addr, 3u));
    if ((i & 1) == 0u)
    {
        raw32 = (raw32 & 0xFFFF0000) | (value & 0x0000FFFF);
    }
    else
    {
        raw32 = (raw32 & 0x0000FFFF) | (value << 16u);
    }
    pnanovdb_write_uint32(buf, addr, raw32);
}
PNANOVDB_FORCE_INLINE void pnanovdb_leaf_pointindex_set_on(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf, pnanovdb_uint32_t i)
{
    pnanovdb_uint32_t word_idx = i >> 5;
    pnanovdb_uint32_t bit_idx = i & 31;
    pnanovdb_address_t addr = pnanovdb_address_offset(leaf.address, PNANOVDB_LEAF_OFF_VALUE_MASK + 4u * word_idx);
    pnanovdb_uint32_t val_mask = pnanovdb_read_uint32(buf, addr);
    val_mask = val_mask | (1u << bit_idx);
    pnanovdb_write_uint32(buf, addr, val_mask);
}
PNANOVDB_FORCE_INLINE void pnanovdb_leaf_pointindex_set_value(pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf, pnanovdb_uint32_t i, pnanovdb_uint32_t value)
{
    pnanovdb_leaf_pointindex_set_on(buf, leaf, i);
    pnanovdb_leaf_pointindex_set_value_only(buf, leaf, i, value);
}

// ------------------------------------------------ Lower Node -----------------------------------------------------------

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_lower_coord_to_offset(PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    return (((PNANOVDB_DEREF(ijk).x & 127) >> 3) << (2 * 4)) +
           (((PNANOVDB_DEREF(ijk).y & 127) >> 3) << (4)) +
            ((PNANOVDB_DEREF(ijk).z & 127) >> 3);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_lower_get_min_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_lower_handle_t node)
{
    pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, lower_off_min);
    return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_lower_get_max_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_lower_handle_t node)
{
    pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, lower_off_max);
    return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_lower_get_ave_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_lower_handle_t node)
{
    pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, lower_off_ave);
    return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_lower_get_stddev_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_lower_handle_t node)
{
    pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, lower_off_stddev);
    return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_lower_get_table_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_lower_handle_t node, pnanovdb_uint32_t n)
{
    pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, lower_off_table) + PNANOVDB_GRID_TYPE_GET(grid_type, table_stride) * n;
    return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_int64_t pnanovdb_lower_get_table_child(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_lower_handle_t node, pnanovdb_uint32_t n)
{
    pnanovdb_address_t table_address = pnanovdb_lower_get_table_address(grid_type, buf, node, n);
    return pnanovdb_read_int64(buf, table_address);
}

PNANOVDB_FORCE_INLINE pnanovdb_leaf_handle_t pnanovdb_lower_get_child(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_lower_handle_t lower, pnanovdb_uint32_t n)
{
    pnanovdb_leaf_handle_t leaf = { lower.address };
    leaf.address = pnanovdb_address_offset64(leaf.address, pnanovdb_int64_as_uint64(pnanovdb_lower_get_table_child(grid_type, buf, lower, n)));
    return leaf;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_lower_get_value_address_and_level(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_lower_handle_t lower, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_uint32_t) level)
{
    pnanovdb_uint32_t n = pnanovdb_lower_coord_to_offset(ijk);
    pnanovdb_address_t value_address;
    if (pnanovdb_lower_get_child_mask(buf, lower, n))
    {
        pnanovdb_leaf_handle_t child = pnanovdb_lower_get_child(grid_type, buf, lower, n);
        value_address = pnanovdb_leaf_get_value_address(grid_type, buf, child, ijk);
        PNANOVDB_DEREF(level) = 0u;
    }
    else
    {
        value_address = pnanovdb_lower_get_table_address(grid_type, buf, lower, n);
        PNANOVDB_DEREF(level) = 1u;
    }
    return value_address;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_lower_get_value_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_lower_handle_t lower, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    pnanovdb_uint32_t level;
    return pnanovdb_lower_get_value_address_and_level(grid_type, buf, lower, ijk, PNANOVDB_REF(level));
}

// ------------------------------------------------ Upper Node -----------------------------------------------------------

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_upper_coord_to_offset(PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    return (((PNANOVDB_DEREF(ijk).x & 4095) >> 7) << (2 * 5)) +
           (((PNANOVDB_DEREF(ijk).y & 4095) >> 7) << (5)) +
            ((PNANOVDB_DEREF(ijk).z & 4095) >> 7);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_upper_get_min_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_upper_handle_t node)
{
    pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, upper_off_min);
    return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_upper_get_max_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_upper_handle_t node)
{
    pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, upper_off_max);
    return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_upper_get_ave_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_upper_handle_t node)
{
    pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, upper_off_ave);
    return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_upper_get_stddev_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_upper_handle_t node)
{
    pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, upper_off_stddev);
    return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_upper_get_table_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_upper_handle_t node, pnanovdb_uint32_t n)
{
    pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, upper_off_table) + PNANOVDB_GRID_TYPE_GET(grid_type, table_stride) * n;
    return pnanovdb_address_offset(node.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_int64_t pnanovdb_upper_get_table_child(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_upper_handle_t node, pnanovdb_uint32_t n)
{
    pnanovdb_address_t bufAddress = pnanovdb_upper_get_table_address(grid_type, buf, node, n);
    return pnanovdb_read_int64(buf, bufAddress);
}

PNANOVDB_FORCE_INLINE pnanovdb_lower_handle_t pnanovdb_upper_get_child(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_upper_handle_t upper, pnanovdb_uint32_t n)
{
    pnanovdb_lower_handle_t lower = { upper.address };
    lower.address = pnanovdb_address_offset64(lower.address, pnanovdb_int64_as_uint64(pnanovdb_upper_get_table_child(grid_type, buf, upper, n)));
    return lower;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_upper_get_value_address_and_level(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_upper_handle_t upper, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_uint32_t) level)
{
    pnanovdb_uint32_t n = pnanovdb_upper_coord_to_offset(ijk);
    pnanovdb_address_t value_address;
    if (pnanovdb_upper_get_child_mask(buf, upper, n))
    {
        pnanovdb_lower_handle_t child = pnanovdb_upper_get_child(grid_type, buf, upper, n);
        value_address = pnanovdb_lower_get_value_address_and_level(grid_type, buf, child, ijk, level);
    }
    else
    {
        value_address = pnanovdb_upper_get_table_address(grid_type, buf, upper, n);
        PNANOVDB_DEREF(level) = 2u;
    }
    return value_address;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_upper_get_value_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_upper_handle_t upper, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    pnanovdb_uint32_t level;
    return pnanovdb_upper_get_value_address_and_level(grid_type, buf, upper, ijk, PNANOVDB_REF(level));
}

PNANOVDB_FORCE_INLINE void pnanovdb_upper_set_table_child(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_upper_handle_t node, pnanovdb_uint32_t n, pnanovdb_int64_t child)
{
    pnanovdb_address_t bufAddress = pnanovdb_upper_get_table_address(grid_type, buf, node, n);
    pnanovdb_write_int64(buf, bufAddress, child);
}

// ------------------------------------------------ Root -----------------------------------------------------------

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_root_get_min_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root)
{
    pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, root_off_min);
    return pnanovdb_address_offset(root.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_root_get_max_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root)
{
    pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, root_off_max);
    return pnanovdb_address_offset(root.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_root_get_ave_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root)
{
    pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, root_off_ave);
    return pnanovdb_address_offset(root.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_root_get_stddev_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root)
{
    pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, root_off_stddev);
    return pnanovdb_address_offset(root.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_root_tile_get_value_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_tile_handle_t root_tile)
{
    pnanovdb_uint32_t byte_offset = PNANOVDB_GRID_TYPE_GET(grid_type, root_tile_off_value);
    return pnanovdb_address_offset(root_tile.address, byte_offset);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_root_get_value_address_and_level(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_uint32_t) level)
{
    pnanovdb_root_tile_handle_t tile = pnanovdb_root_find_tile(grid_type, buf, root, ijk);
    pnanovdb_address_t ret;
    if (pnanovdb_address_is_null(tile.address))
    {
        ret = pnanovdb_address_offset(root.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_off_background));
        PNANOVDB_DEREF(level) = 4u;
    }
    else if (pnanovdb_int64_is_zero(pnanovdb_root_tile_get_child(buf, tile)))
    {
        ret = pnanovdb_address_offset(tile.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_tile_off_value));
        PNANOVDB_DEREF(level) = 3u;
    }
    else
    {
        pnanovdb_upper_handle_t child = pnanovdb_root_get_child(grid_type, buf, root, tile);
        ret = pnanovdb_upper_get_value_address_and_level(grid_type, buf, child, ijk, level);
    }
    return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_root_get_value_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    pnanovdb_uint32_t level;
    return pnanovdb_root_get_value_address_and_level(grid_type, buf, root, ijk, PNANOVDB_REF(level));
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_root_get_value_address_bit(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_uint32_t) bit_index)
{
    pnanovdb_uint32_t level;
    pnanovdb_address_t address = pnanovdb_root_get_value_address_and_level(grid_type, buf, root, ijk, PNANOVDB_REF(level));
    PNANOVDB_DEREF(bit_index) = level == 0u ? pnanovdb_int32_as_uint32(PNANOVDB_DEREF(ijk).x & 7) : 0u;
    return address;
}

PNANOVDB_FORCE_INLINE float pnanovdb_root_fp4_read_float(pnanovdb_buf_t buf, pnanovdb_address_t address, PNANOVDB_IN(pnanovdb_coord_t) ijk, pnanovdb_uint32_t level)
{
    float ret;
    if (level == 0)
    {
        ret = pnanovdb_leaf_fp4_read_float(buf, address, ijk);
    }
    else
    {
        ret = pnanovdb_read_float(buf, address);
    }
    return ret;
}

PNANOVDB_FORCE_INLINE float pnanovdb_root_fp8_read_float(pnanovdb_buf_t buf, pnanovdb_address_t address, PNANOVDB_IN(pnanovdb_coord_t) ijk, pnanovdb_uint32_t level)
{
    float ret;
    if (level == 0)
    {
        ret = pnanovdb_leaf_fp8_read_float(buf, address, ijk);
    }
    else
    {
        ret = pnanovdb_read_float(buf, address);
    }
    return ret;
}

PNANOVDB_FORCE_INLINE float pnanovdb_root_fp16_read_float(pnanovdb_buf_t buf, pnanovdb_address_t address, PNANOVDB_IN(pnanovdb_coord_t) ijk, pnanovdb_uint32_t level)
{
    float ret;
    if (level == 0)
    {
        ret = pnanovdb_leaf_fp16_read_float(buf, address, ijk);
    }
    else
    {
        ret = pnanovdb_read_float(buf, address);
    }
    return ret;
}

PNANOVDB_FORCE_INLINE float pnanovdb_root_fpn_read_float(pnanovdb_buf_t buf, pnanovdb_address_t address, PNANOVDB_IN(pnanovdb_coord_t) ijk, pnanovdb_uint32_t level)
{
    float ret;
    if (level == 0)
    {
        ret = pnanovdb_leaf_fpn_read_float(buf, address, ijk);
    }
    else
    {
        ret = pnanovdb_read_float(buf, address);
    }
    return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_root_index_get_value_index(pnanovdb_buf_t buf, pnanovdb_address_t address, PNANOVDB_IN(pnanovdb_coord_t) ijk, pnanovdb_uint32_t level)
{
    pnanovdb_uint64_t ret;
    if (level == 0)
    {
        ret = pnanovdb_leaf_index_get_value_index(buf, address, ijk);
    }
    else
    {
        ret = pnanovdb_read_uint64(buf, address);
    }
    return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_root_onindex_get_value_index(pnanovdb_buf_t buf, pnanovdb_address_t address, PNANOVDB_IN(pnanovdb_coord_t) ijk, pnanovdb_uint32_t level)
{
    pnanovdb_uint64_t ret;
    if (level == 0)
    {
        ret = pnanovdb_leaf_onindex_get_value_index(buf, address, ijk);
    }
    else
    {
        ret = pnanovdb_read_uint64(buf, address);
    }
    return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_root_pointindex_get_point_range(
    pnanovdb_buf_t buf,
    pnanovdb_address_t value_address,
    PNANOVDB_IN(pnanovdb_coord_t) ijk,
    pnanovdb_uint32_t level,
    PNANOVDB_INOUT(pnanovdb_uint64_t)range_begin,
    PNANOVDB_INOUT(pnanovdb_uint64_t)range_end
)
{
    pnanovdb_uint32_t local_range_begin = 0u;
    pnanovdb_uint32_t local_range_end = 0u;
    pnanovdb_uint64_t offset = pnanovdb_uint32_as_uint64_low(0u);
    if (level == 0)
    {
        pnanovdb_uint32_t n = pnanovdb_leaf_coord_to_offset(ijk);
        // recover leaf address
        pnanovdb_leaf_handle_t leaf = { pnanovdb_address_offset_neg(value_address, PNANOVDB_GRID_TYPE_GET(PNANOVDB_GRID_TYPE_POINTINDEX, leaf_off_table) + 2u * n) };
        if (n > 0u)
        {
            local_range_begin = pnanovdb_read_uint16(buf, pnanovdb_address_offset_neg(value_address, 2u));
        }
        local_range_end = pnanovdb_read_uint16(buf, value_address);
        offset = pnanovdb_leaf_pointindex_get_offset(buf, leaf);
    }
    PNANOVDB_DEREF(range_begin) = pnanovdb_uint64_offset(offset, local_range_begin);
    PNANOVDB_DEREF(range_end) = pnanovdb_uint64_offset(offset, local_range_end);
    return pnanovdb_uint32_as_uint64_low(local_range_end - local_range_begin);
}

PNANOVDB_FORCE_INLINE pnanovdb_uint64_t pnanovdb_root_pointindex_get_point_address_range(
    pnanovdb_buf_t buf,
    pnanovdb_grid_type_t value_type,
    pnanovdb_address_t value_address,
    pnanovdb_address_t blindmetadata_value_address,
    PNANOVDB_IN(pnanovdb_coord_t) ijk,
    pnanovdb_uint32_t level,
    PNANOVDB_INOUT(pnanovdb_address_t)address_begin,
    PNANOVDB_INOUT(pnanovdb_address_t)address_end
)
{
    pnanovdb_uint64_t range_begin;
    pnanovdb_uint64_t range_end;
    pnanovdb_uint64_t range_size = pnanovdb_root_pointindex_get_point_range(buf, value_address, ijk, level, PNANOVDB_REF(range_begin), PNANOVDB_REF(range_end));

    pnanovdb_uint32_t stride = 12u; // vec3f
    if (value_type == PNANOVDB_GRID_TYPE_VEC3U8)
    {
        stride = 3u;
    }
    else if (value_type == PNANOVDB_GRID_TYPE_VEC3U16)
    {
        stride = 6u;
    }
    PNANOVDB_DEREF(address_begin) = pnanovdb_address_offset64_product(blindmetadata_value_address, range_begin, stride);
    PNANOVDB_DEREF(address_end) = pnanovdb_address_offset64_product(blindmetadata_value_address, range_end, stride);
    return range_size;
}

// ------------------------------------------------ ReadAccessor -----------------------------------------------------------

struct pnanovdb_readaccessor_t
{
    pnanovdb_coord_t key;
    pnanovdb_leaf_handle_t leaf;
    pnanovdb_lower_handle_t lower;
    pnanovdb_upper_handle_t upper;
    pnanovdb_root_handle_t root;
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_readaccessor_t)

PNANOVDB_FORCE_INLINE void pnanovdb_readaccessor_init(PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, pnanovdb_root_handle_t root)
{
    PNANOVDB_DEREF(acc).key.x = 0x7FFFFFFF;
    PNANOVDB_DEREF(acc).key.y = 0x7FFFFFFF;
    PNANOVDB_DEREF(acc).key.z = 0x7FFFFFFF;
    PNANOVDB_DEREF(acc).leaf.address = pnanovdb_address_null();
    PNANOVDB_DEREF(acc).lower.address = pnanovdb_address_null();
    PNANOVDB_DEREF(acc).upper.address = pnanovdb_address_null();
    PNANOVDB_DEREF(acc).root = root;
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_readaccessor_iscached0(PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, int dirty)
{
    if (pnanovdb_address_is_null(PNANOVDB_DEREF(acc).leaf.address)) { return PNANOVDB_FALSE; }
    if ((dirty & ~((1u << 3) - 1u)) != 0)
    {
        PNANOVDB_DEREF(acc).leaf.address = pnanovdb_address_null();
        return PNANOVDB_FALSE;
    }
    return PNANOVDB_TRUE;
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_readaccessor_iscached1(PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, int dirty)
{
    if (pnanovdb_address_is_null(PNANOVDB_DEREF(acc).lower.address)) { return PNANOVDB_FALSE; }
    if ((dirty & ~((1u << 7) - 1u)) != 0)
    {
        PNANOVDB_DEREF(acc).lower.address = pnanovdb_address_null();
        return PNANOVDB_FALSE;
    }
    return PNANOVDB_TRUE;
}
PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_readaccessor_iscached2(PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, int dirty)
{
    if (pnanovdb_address_is_null(PNANOVDB_DEREF(acc).upper.address)) { return PNANOVDB_FALSE; }
    if ((dirty & ~((1u << 12) - 1u)) != 0)
    {
        PNANOVDB_DEREF(acc).upper.address = pnanovdb_address_null();
        return PNANOVDB_FALSE;
    }
    return PNANOVDB_TRUE;
}
PNANOVDB_FORCE_INLINE int pnanovdb_readaccessor_computedirty(PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    return (PNANOVDB_DEREF(ijk).x ^ PNANOVDB_DEREF(acc).key.x) | (PNANOVDB_DEREF(ijk).y ^ PNANOVDB_DEREF(acc).key.y) | (PNANOVDB_DEREF(ijk).z ^ PNANOVDB_DEREF(acc).key.z);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_leaf_get_value_address_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
    pnanovdb_uint32_t n = pnanovdb_leaf_coord_to_offset(ijk);
    return pnanovdb_leaf_get_table_address(grid_type, buf, leaf, n);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_lower_get_value_address_and_level_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_lower_handle_t lower, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, PNANOVDB_INOUT(pnanovdb_uint32_t) level)
{
    pnanovdb_uint32_t n = pnanovdb_lower_coord_to_offset(ijk);
    pnanovdb_address_t value_address;
    if (pnanovdb_lower_get_child_mask(buf, lower, n))
    {
        pnanovdb_leaf_handle_t child = pnanovdb_lower_get_child(grid_type, buf, lower, n);
        PNANOVDB_DEREF(acc).leaf = child;
        PNANOVDB_DEREF(acc).key = PNANOVDB_DEREF(ijk);
        value_address = pnanovdb_leaf_get_value_address_and_cache(grid_type, buf, child, ijk, acc);
        PNANOVDB_DEREF(level) = 0u;
    }
    else
    {
        value_address = pnanovdb_lower_get_table_address(grid_type, buf, lower, n);
        PNANOVDB_DEREF(level) = 1u;
    }
    return value_address;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_lower_get_value_address_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_lower_handle_t lower, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
    pnanovdb_uint32_t level;
    return pnanovdb_lower_get_value_address_and_level_and_cache(grid_type, buf, lower, ijk, acc, PNANOVDB_REF(level));
}

PNANOVDB_FORCE_INLINE void pnanovdb_lower_set_table_child(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_lower_handle_t node, pnanovdb_uint32_t n, pnanovdb_int64_t child)
{
    pnanovdb_address_t table_address = pnanovdb_lower_get_table_address(grid_type, buf, node, n);
    pnanovdb_write_int64(buf, table_address, child);
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_upper_get_value_address_and_level_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_upper_handle_t upper, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, PNANOVDB_INOUT(pnanovdb_uint32_t) level)
{
    pnanovdb_uint32_t n = pnanovdb_upper_coord_to_offset(ijk);
    pnanovdb_address_t value_address;
    if (pnanovdb_upper_get_child_mask(buf, upper, n))
    {
        pnanovdb_lower_handle_t child = pnanovdb_upper_get_child(grid_type, buf, upper, n);
        PNANOVDB_DEREF(acc).lower = child;
        PNANOVDB_DEREF(acc).key = PNANOVDB_DEREF(ijk);
        value_address = pnanovdb_lower_get_value_address_and_level_and_cache(grid_type, buf, child, ijk, acc, level);
    }
    else
    {
        value_address = pnanovdb_upper_get_table_address(grid_type, buf, upper, n);
        PNANOVDB_DEREF(level) = 2u;
    }
    return value_address;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_upper_get_value_address_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_upper_handle_t upper, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
    pnanovdb_uint32_t level;
    return pnanovdb_upper_get_value_address_and_level_and_cache(grid_type, buf, upper, ijk, acc, PNANOVDB_REF(level));
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_root_get_value_address_and_level_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, PNANOVDB_INOUT(pnanovdb_uint32_t) level)
{
    pnanovdb_root_tile_handle_t tile = pnanovdb_root_find_tile(grid_type, buf, root, ijk);
    pnanovdb_address_t ret;
    if (pnanovdb_address_is_null(tile.address))
    {
        ret = pnanovdb_address_offset(root.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_off_background));
        PNANOVDB_DEREF(level) = 4u;
    }
    else if (pnanovdb_int64_is_zero(pnanovdb_root_tile_get_child(buf, tile)))
    {
        ret = pnanovdb_address_offset(tile.address, PNANOVDB_GRID_TYPE_GET(grid_type, root_tile_off_value));
        PNANOVDB_DEREF(level) = 3u;
    }
    else
    {
        pnanovdb_upper_handle_t child = pnanovdb_root_get_child(grid_type, buf, root, tile);
        PNANOVDB_DEREF(acc).upper = child;
        PNANOVDB_DEREF(acc).key = PNANOVDB_DEREF(ijk);
        ret = pnanovdb_upper_get_value_address_and_level_and_cache(grid_type, buf, child, ijk, acc, level);
    }
    return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_root_get_value_address_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
    pnanovdb_uint32_t level;
    return pnanovdb_root_get_value_address_and_level_and_cache(grid_type, buf, root, ijk, acc, PNANOVDB_REF(level));
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_readaccessor_get_value_address_and_level(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_uint32_t) level)
{
    int dirty = pnanovdb_readaccessor_computedirty(acc, ijk);

    pnanovdb_address_t value_address;
    if (pnanovdb_readaccessor_iscached0(acc, dirty))
    {
        value_address = pnanovdb_leaf_get_value_address_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).leaf, ijk, acc);
        PNANOVDB_DEREF(level) = 0u;
    }
    else if (pnanovdb_readaccessor_iscached1(acc, dirty))
    {
        value_address = pnanovdb_lower_get_value_address_and_level_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).lower, ijk, acc, level);
    }
    else if (pnanovdb_readaccessor_iscached2(acc, dirty))
    {
        value_address = pnanovdb_upper_get_value_address_and_level_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).upper, ijk, acc, level);
    }
    else
    {
        value_address = pnanovdb_root_get_value_address_and_level_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).root, ijk, acc, level);
    }
    return value_address;
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_readaccessor_get_value_address(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    pnanovdb_uint32_t level;
    return pnanovdb_readaccessor_get_value_address_and_level(grid_type, buf, acc, ijk, PNANOVDB_REF(level));
}

PNANOVDB_FORCE_INLINE pnanovdb_address_t pnanovdb_readaccessor_get_value_address_bit(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_uint32_t) bit_index)
{
    pnanovdb_uint32_t level;
    pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address_and_level(grid_type, buf, acc, ijk, PNANOVDB_REF(level));
    PNANOVDB_DEREF(bit_index) = level == 0u ? pnanovdb_int32_as_uint32(PNANOVDB_DEREF(ijk).x & 7) : 0u;
    return address;
}

// ------------------------------------------------ ReadAccessor GetDim -----------------------------------------------------------

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_leaf_get_dim_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
    return 1u;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_lower_get_dim_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_lower_handle_t lower, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
    pnanovdb_uint32_t n = pnanovdb_lower_coord_to_offset(ijk);
    pnanovdb_uint32_t ret;
    if (pnanovdb_lower_get_child_mask(buf, lower, n))
    {
        pnanovdb_leaf_handle_t child = pnanovdb_lower_get_child(grid_type, buf, lower, n);
        PNANOVDB_DEREF(acc).leaf = child;
        PNANOVDB_DEREF(acc).key = PNANOVDB_DEREF(ijk);
        ret = pnanovdb_leaf_get_dim_and_cache(grid_type, buf, child, ijk, acc);
    }
    else
    {
        ret = (1u << (3u)); // node 0 dim
    }
    return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_upper_get_dim_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_upper_handle_t upper, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
    pnanovdb_uint32_t n = pnanovdb_upper_coord_to_offset(ijk);
    pnanovdb_uint32_t ret;
    if (pnanovdb_upper_get_child_mask(buf, upper, n))
    {
        pnanovdb_lower_handle_t child = pnanovdb_upper_get_child(grid_type, buf, upper, n);
        PNANOVDB_DEREF(acc).lower = child;
        PNANOVDB_DEREF(acc).key = PNANOVDB_DEREF(ijk);
        ret = pnanovdb_lower_get_dim_and_cache(grid_type, buf, child, ijk, acc);
    }
    else
    {
        ret = (1u << (4u + 3u)); // node 1 dim
    }
    return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_root_get_dim_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
    pnanovdb_root_tile_handle_t tile = pnanovdb_root_find_tile(grid_type, buf, root, ijk);
    pnanovdb_uint32_t ret;
    if (pnanovdb_address_is_null(tile.address))
    {
        ret = 1u << (5u + 4u + 3u); // background, node 2 dim
    }
    else if (pnanovdb_int64_is_zero(pnanovdb_root_tile_get_child(buf, tile)))
    {
        ret = 1u << (5u + 4u + 3u); // tile value, node 2 dim
    }
    else
    {
        pnanovdb_upper_handle_t child = pnanovdb_root_get_child(grid_type, buf, root, tile);
        PNANOVDB_DEREF(acc).upper = child;
        PNANOVDB_DEREF(acc).key = PNANOVDB_DEREF(ijk);
        ret = pnanovdb_upper_get_dim_and_cache(grid_type, buf, child, ijk, acc);
    }
    return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_uint32_t pnanovdb_readaccessor_get_dim(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    int dirty = pnanovdb_readaccessor_computedirty(acc, ijk);

    pnanovdb_uint32_t dim;
    if (pnanovdb_readaccessor_iscached0(acc, dirty))
    {
        dim = pnanovdb_leaf_get_dim_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).leaf, ijk, acc);
    }
    else if (pnanovdb_readaccessor_iscached1(acc, dirty))
    {
        dim = pnanovdb_lower_get_dim_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).lower, ijk, acc);
    }
    else if (pnanovdb_readaccessor_iscached2(acc, dirty))
    {
        dim = pnanovdb_upper_get_dim_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).upper, ijk, acc);
    }
    else
    {
        dim = pnanovdb_root_get_dim_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).root, ijk, acc);
    }
    return dim;
}

// ------------------------------------------------ ReadAccessor IsActive -----------------------------------------------------------

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_leaf_is_active_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_leaf_handle_t leaf, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
    pnanovdb_uint32_t n = pnanovdb_leaf_coord_to_offset(ijk);
    return pnanovdb_leaf_get_value_mask(buf, leaf, n);
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_lower_is_active_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_lower_handle_t lower, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
    pnanovdb_uint32_t n = pnanovdb_lower_coord_to_offset(ijk);
    pnanovdb_bool_t is_active;
    if (pnanovdb_lower_get_child_mask(buf, lower, n))
    {
        pnanovdb_leaf_handle_t child = pnanovdb_lower_get_child(grid_type, buf, lower, n);
        PNANOVDB_DEREF(acc).leaf = child;
        PNANOVDB_DEREF(acc).key = PNANOVDB_DEREF(ijk);
        is_active = pnanovdb_leaf_is_active_and_cache(grid_type, buf, child, ijk, acc);
    }
    else
    {
        is_active = pnanovdb_lower_get_value_mask(buf, lower, n);
    }
    return is_active;
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_upper_is_active_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_upper_handle_t upper, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
    pnanovdb_uint32_t n = pnanovdb_upper_coord_to_offset(ijk);
    pnanovdb_bool_t is_active;
    if (pnanovdb_upper_get_child_mask(buf, upper, n))
    {
        pnanovdb_lower_handle_t child = pnanovdb_upper_get_child(grid_type, buf, upper, n);
        PNANOVDB_DEREF(acc).lower = child;
        PNANOVDB_DEREF(acc).key = PNANOVDB_DEREF(ijk);
        is_active = pnanovdb_lower_is_active_and_cache(grid_type, buf, child, ijk, acc);
    }
    else
    {
        is_active = pnanovdb_upper_get_value_mask(buf, upper, n);
    }
    return is_active;
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_root_is_active_and_cache(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, pnanovdb_root_handle_t root, PNANOVDB_IN(pnanovdb_coord_t) ijk, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc)
{
    pnanovdb_root_tile_handle_t tile = pnanovdb_root_find_tile(grid_type, buf, root, ijk);
    pnanovdb_bool_t is_active;
    if (pnanovdb_address_is_null(tile.address))
    {
        is_active = PNANOVDB_FALSE; // background
    }
    else if (pnanovdb_int64_is_zero(pnanovdb_root_tile_get_child(buf, tile)))
    {
        pnanovdb_uint32_t state = pnanovdb_root_tile_get_state(buf, tile);
        is_active = state != 0u; // tile value
    }
    else
    {
        pnanovdb_upper_handle_t child = pnanovdb_root_get_child(grid_type, buf, root, tile);
        PNANOVDB_DEREF(acc).upper = child;
        PNANOVDB_DEREF(acc).key = PNANOVDB_DEREF(ijk);
        is_active = pnanovdb_upper_is_active_and_cache(grid_type, buf, child, ijk, acc);
    }
    return is_active;
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_readaccessor_is_active(pnanovdb_grid_type_t grid_type, pnanovdb_buf_t buf, PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc, PNANOVDB_IN(pnanovdb_coord_t) ijk)
{
    int dirty = pnanovdb_readaccessor_computedirty(acc, ijk);

    pnanovdb_bool_t is_active;
    if (pnanovdb_readaccessor_iscached0(acc, dirty))
    {
        is_active = pnanovdb_leaf_is_active_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).leaf, ijk, acc);
    }
    else if (pnanovdb_readaccessor_iscached1(acc, dirty))
    {
        is_active = pnanovdb_lower_is_active_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).lower, ijk, acc);
    }
    else if (pnanovdb_readaccessor_iscached2(acc, dirty))
    {
        is_active = pnanovdb_upper_is_active_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).upper, ijk, acc);
    }
    else
    {
        is_active = pnanovdb_root_is_active_and_cache(grid_type, buf, PNANOVDB_DEREF(acc).root, ijk, acc);
    }
    return is_active;
}

// ------------------------------------------------ Map Transforms -----------------------------------------------------------

PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_map_apply(pnanovdb_buf_t buf, pnanovdb_map_handle_t map, PNANOVDB_IN(pnanovdb_vec3_t) src)
{
    pnanovdb_vec3_t  dst;
    float sx = PNANOVDB_DEREF(src).x;
    float sy = PNANOVDB_DEREF(src).y;
    float sz = PNANOVDB_DEREF(src).z;
    dst.x = sx * pnanovdb_map_get_matf(buf, map, 0) + sy * pnanovdb_map_get_matf(buf, map, 1) + sz * pnanovdb_map_get_matf(buf, map, 2) + pnanovdb_map_get_vecf(buf, map, 0);
    dst.y = sx * pnanovdb_map_get_matf(buf, map, 3) + sy * pnanovdb_map_get_matf(buf, map, 4) + sz * pnanovdb_map_get_matf(buf, map, 5) + pnanovdb_map_get_vecf(buf, map, 1);
    dst.z = sx * pnanovdb_map_get_matf(buf, map, 6) + sy * pnanovdb_map_get_matf(buf, map, 7) + sz * pnanovdb_map_get_matf(buf, map, 8) + pnanovdb_map_get_vecf(buf, map, 2);
    return dst;
}

PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_map_apply_inverse(pnanovdb_buf_t buf, pnanovdb_map_handle_t map, PNANOVDB_IN(pnanovdb_vec3_t) src)
{
    pnanovdb_vec3_t  dst;
    float sx = PNANOVDB_DEREF(src).x - pnanovdb_map_get_vecf(buf, map, 0);
    float sy = PNANOVDB_DEREF(src).y - pnanovdb_map_get_vecf(buf, map, 1);
    float sz = PNANOVDB_DEREF(src).z - pnanovdb_map_get_vecf(buf, map, 2);
    dst.x = sx * pnanovdb_map_get_invmatf(buf, map, 0) + sy * pnanovdb_map_get_invmatf(buf, map, 1) + sz * pnanovdb_map_get_invmatf(buf, map, 2);
    dst.y = sx * pnanovdb_map_get_invmatf(buf, map, 3) + sy * pnanovdb_map_get_invmatf(buf, map, 4) + sz * pnanovdb_map_get_invmatf(buf, map, 5);
    dst.z = sx * pnanovdb_map_get_invmatf(buf, map, 6) + sy * pnanovdb_map_get_invmatf(buf, map, 7) + sz * pnanovdb_map_get_invmatf(buf, map, 8);
    return dst;
}

PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_map_apply_jacobi(pnanovdb_buf_t buf, pnanovdb_map_handle_t map, PNANOVDB_IN(pnanovdb_vec3_t) src)
{
    pnanovdb_vec3_t  dst;
    float sx = PNANOVDB_DEREF(src).x;
    float sy = PNANOVDB_DEREF(src).y;
    float sz = PNANOVDB_DEREF(src).z;
    dst.x = sx * pnanovdb_map_get_matf(buf, map, 0) + sy * pnanovdb_map_get_matf(buf, map, 1) + sz * pnanovdb_map_get_matf(buf, map, 2);
    dst.y = sx * pnanovdb_map_get_matf(buf, map, 3) + sy * pnanovdb_map_get_matf(buf, map, 4) + sz * pnanovdb_map_get_matf(buf, map, 5);
    dst.z = sx * pnanovdb_map_get_matf(buf, map, 6) + sy * pnanovdb_map_get_matf(buf, map, 7) + sz * pnanovdb_map_get_matf(buf, map, 8);
    return dst;
}

PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_map_apply_inverse_jacobi(pnanovdb_buf_t buf, pnanovdb_map_handle_t map, PNANOVDB_IN(pnanovdb_vec3_t) src)
{
    pnanovdb_vec3_t  dst;
    float sx = PNANOVDB_DEREF(src).x;
    float sy = PNANOVDB_DEREF(src).y;
    float sz = PNANOVDB_DEREF(src).z;
    dst.x = sx * pnanovdb_map_get_invmatf(buf, map, 0) + sy * pnanovdb_map_get_invmatf(buf, map, 1) + sz * pnanovdb_map_get_invmatf(buf, map, 2);
    dst.y = sx * pnanovdb_map_get_invmatf(buf, map, 3) + sy * pnanovdb_map_get_invmatf(buf, map, 4) + sz * pnanovdb_map_get_invmatf(buf, map, 5);
    dst.z = sx * pnanovdb_map_get_invmatf(buf, map, 6) + sy * pnanovdb_map_get_invmatf(buf, map, 7) + sz * pnanovdb_map_get_invmatf(buf, map, 8);
    return dst;
}

PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_grid_world_to_indexf(pnanovdb_buf_t buf, pnanovdb_grid_handle_t grid, PNANOVDB_IN(pnanovdb_vec3_t) src)
{
    pnanovdb_map_handle_t map = pnanovdb_grid_get_map(buf, grid);
    return pnanovdb_map_apply_inverse(buf, map, src);
}

PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_grid_index_to_worldf(pnanovdb_buf_t buf, pnanovdb_grid_handle_t grid, PNANOVDB_IN(pnanovdb_vec3_t) src)
{
    pnanovdb_map_handle_t map = pnanovdb_grid_get_map(buf, grid);
    return pnanovdb_map_apply(buf, map, src);
}

PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_grid_world_to_index_dirf(pnanovdb_buf_t buf, pnanovdb_grid_handle_t grid, PNANOVDB_IN(pnanovdb_vec3_t) src)
{
    pnanovdb_map_handle_t map = pnanovdb_grid_get_map(buf, grid);
    return pnanovdb_map_apply_inverse_jacobi(buf, map, src);
}

PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_grid_index_to_world_dirf(pnanovdb_buf_t buf, pnanovdb_grid_handle_t grid, PNANOVDB_IN(pnanovdb_vec3_t) src)
{
    pnanovdb_map_handle_t map = pnanovdb_grid_get_map(buf, grid);
    return pnanovdb_map_apply_jacobi(buf, map, src);
}

// ------------------------------------------------ DitherLUT -----------------------------------------------------------

// This table was generated with
/**************

static constexpr inline uint32
SYSwang_inthash(uint32 key)
{
    // From http://www.concentric.net/~Ttwang/tech/inthash.htm
    key += ~(key << 16);
    key ^=  (key >> 5);
    key +=  (key << 3);
    key ^=  (key >> 13);
    key += ~(key << 9);
    key ^=  (key >> 17);
    return key;
}

static void
ut_initDitherR(float *pattern, float offset,
    int x, int y, int z, int res, int goalres)
{
    // These offsets are designed to maximize the difference between
    // dither values in nearby voxels within a given 2x2x2 cell, without
    // producing axis-aligned artifacts.  The are organized in row-major
    // order.
    static const float  theDitherOffset[] = {0,4,6,2,5,1,3,7};
    static const float  theScale = 0.125F;
    int         key = (((z << res) + y) << res) + x;

    if (res == goalres)
    {
    pattern[key] = offset;
    return;
    }

    // Randomly flip (on each axis) the dithering patterns used by the
    // subcells.  This key is xor'd with the subcell index below before
    // looking up in the dither offset list.
    key = SYSwang_inthash(key) & 7;

    x <<= 1;
    y <<= 1;
    z <<= 1;

    offset *= theScale;
    for (int i = 0; i < 8; i++)
    ut_initDitherR(pattern, offset+theDitherOffset[i ^ key]*theScale,
        x+(i&1), y+((i&2)>>1), z+((i&4)>>2), res+1, goalres);
}

// This is a compact algorithm that accomplishes essentially the same thing
// as ut_initDither() above.  We should eventually switch to use this and
// clean the dead code.
static fpreal32 *
ut_initDitherRecursive(int goalres)
{
    const int nfloat = 1 << (goalres*3);
    float   *pattern = new float[nfloat];
    ut_initDitherR(pattern, 1.0F, 0, 0, 0, 0, goalres);

    // This has built an even spacing from 1/nfloat to 1.0.
    // however, our dither pattern should be 1/(nfloat+1) to nfloat/(nfloat+1)
    // So we do a correction here.  Note that the earlier calculations are
    // done with powers of 2 so are exact, so it does make sense to delay
    // the renormalization to this pass.
    float correctionterm = nfloat / (nfloat+1.0F);
    for (int i = 0; i < nfloat; i++)
        pattern[i] *= correctionterm;
    return pattern;
}

    theDitherMatrix = ut_initDitherRecursive(3);

    for (int i = 0; i < 512/8; i ++)
    {
        for (int j = 0; j < 8; j ++)
            std::cout << theDitherMatrix[i*8+j] << "f, ";
        std::cout << std::endl;
    }

 **************/

PNANOVDB_STATIC_CONST float pnanovdb_dither_lut[512] =
{
    0.14425f, 0.643275f, 0.830409f, 0.331384f, 0.105263f, 0.604289f, 0.167641f, 0.666667f,
    0.892788f, 0.393762f, 0.0818713f, 0.580897f, 0.853801f, 0.354776f, 0.916179f, 0.417154f,
    0.612086f, 0.11306f, 0.79922f, 0.300195f, 0.510721f, 0.0116959f, 0.947368f, 0.448343f,
    0.362573f, 0.861598f, 0.0506823f, 0.549708f, 0.261209f, 0.760234f, 0.19883f, 0.697856f,
    0.140351f, 0.639376f, 0.576998f, 0.0779727f, 0.522417f, 0.0233918f, 0.460039f, 0.959064f,
    0.888889f, 0.389864f, 0.327485f, 0.826511f, 0.272904f, 0.77193f, 0.709552f, 0.210526f,
    0.483431f, 0.982456f, 0.296296f, 0.795322f, 0.116959f, 0.615984f, 0.0545809f, 0.553606f,
    0.732943f, 0.233918f, 0.545809f, 0.0467836f, 0.865497f, 0.366472f, 0.803119f, 0.304094f,
    0.518519f, 0.0194932f, 0.45614f, 0.955166f, 0.729045f, 0.230019f, 0.54191f, 0.042885f,
    0.269006f, 0.768031f, 0.705653f, 0.206628f, 0.479532f, 0.978558f, 0.292398f, 0.791423f,
    0.237817f, 0.736842f, 0.424951f, 0.923977f, 0.136452f, 0.635478f, 0.323587f, 0.822612f,
    0.986355f, 0.487329f, 0.674464f, 0.175439f, 0.88499f, 0.385965f, 0.573099f, 0.0740741f,
    0.51462f, 0.0155945f, 0.202729f, 0.701754f, 0.148148f, 0.647174f, 0.834308f, 0.335283f,
    0.265107f, 0.764133f, 0.951267f, 0.452242f, 0.896686f, 0.397661f, 0.08577f, 0.584795f,
    0.8577f, 0.358674f, 0.920078f, 0.421053f, 0.740741f, 0.241715f, 0.678363f, 0.179337f,
    0.109162f, 0.608187f, 0.17154f, 0.670565f, 0.491228f, 0.990253f, 0.42885f, 0.927875f,
    0.0662768f, 0.565302f, 0.62768f, 0.128655f, 0.183236f, 0.682261f, 0.744639f, 0.245614f,
    0.814815f, 0.315789f, 0.378168f, 0.877193f, 0.931774f, 0.432749f, 0.495127f, 0.994152f,
    0.0350877f, 0.534113f, 0.97076f, 0.471735f, 0.214425f, 0.71345f, 0.526316f, 0.0272904f,
    0.783626f, 0.2846f, 0.222222f, 0.721248f, 0.962963f, 0.463938f, 0.276803f, 0.775828f,
    0.966862f, 0.467836f, 0.405458f, 0.904483f, 0.0701754f, 0.569201f, 0.881092f, 0.382066f,
    0.218324f, 0.717349f, 0.654971f, 0.155945f, 0.818713f, 0.319688f, 0.132554f, 0.631579f,
    0.0623782f, 0.561404f, 0.748538f, 0.249513f, 0.912281f, 0.413255f, 0.974659f, 0.475634f,
    0.810916f, 0.311891f, 0.499025f, 0.998051f, 0.163743f, 0.662768f, 0.226121f, 0.725146f,
    0.690058f, 0.191033f, 0.00389864f, 0.502924f, 0.557505f, 0.0584795f, 0.120858f, 0.619883f,
    0.440546f, 0.939571f, 0.752437f, 0.253411f, 0.307992f, 0.807018f, 0.869396f, 0.37037f,
    0.658869f, 0.159844f, 0.346979f, 0.846004f, 0.588694f, 0.0896686f, 0.152047f, 0.651072f,
    0.409357f, 0.908382f, 0.596491f, 0.0974659f, 0.339181f, 0.838207f, 0.900585f, 0.401559f,
    0.34308f, 0.842105f, 0.779727f, 0.280702f, 0.693957f, 0.194932f, 0.25731f, 0.756335f,
    0.592593f, 0.0935673f, 0.0311891f, 0.530214f, 0.444444f, 0.94347f, 0.506823f, 0.00779727f,
    0.68616f, 0.187135f, 0.124756f, 0.623782f, 0.288499f, 0.787524f, 0.350877f, 0.849903f,
    0.436647f, 0.935673f, 0.873294f, 0.374269f, 0.538012f, 0.0389864f, 0.60039f, 0.101365f,
    0.57115f, 0.0721248f, 0.758285f, 0.259259f, 0.719298f, 0.220273f, 0.532164f, 0.0331384f,
    0.321637f, 0.820663f, 0.00974659f, 0.508772f, 0.469786f, 0.968811f, 0.282651f, 0.781676f,
    0.539961f, 0.0409357f, 0.727096f, 0.22807f, 0.500975f, 0.00194932f, 0.563353f, 0.0643275f,
    0.290448f, 0.789474f, 0.477583f, 0.976608f, 0.251462f, 0.750487f, 0.31384f, 0.812865f,
    0.94152f, 0.442495f, 0.879142f, 0.380117f, 0.37232f, 0.871345f, 0.309942f, 0.808967f,
    0.192982f, 0.692008f, 0.130604f, 0.62963f, 0.621832f, 0.122807f, 0.559454f, 0.0604289f,
    0.660819f, 0.161793f, 0.723197f, 0.224172f, 0.403509f, 0.902534f, 0.840156f, 0.341131f,
    0.411306f, 0.910331f, 0.473684f, 0.97271f, 0.653021f, 0.153996f, 0.0916179f, 0.590643f,
    0.196881f, 0.695906f, 0.384016f, 0.883041f, 0.0955166f, 0.594542f, 0.157895f, 0.65692f,
    0.945419f, 0.446394f, 0.633528f, 0.134503f, 0.844055f, 0.345029f, 0.906433f, 0.407407f,
    0.165692f, 0.664717f, 0.103314f, 0.602339f, 0.126706f, 0.625731f, 0.189084f, 0.688109f,
    0.91423f, 0.415205f, 0.851852f, 0.352827f, 0.875244f, 0.376218f, 0.937622f, 0.438596f,
    0.317739f, 0.816764f, 0.255361f, 0.754386f, 0.996101f, 0.497076f, 0.933723f, 0.434698f,
    0.567251f, 0.0682261f, 0.504873f, 0.00584795f, 0.247563f, 0.746589f, 0.185185f, 0.684211f,
    0.037037f, 0.536062f, 0.0994152f, 0.598441f, 0.777778f, 0.278752f, 0.465887f, 0.964912f,
    0.785575f, 0.28655f, 0.847953f, 0.348928f, 0.0292398f, 0.528265f, 0.7154f, 0.216374f,
    0.39961f, 0.898636f, 0.961014f, 0.461988f, 0.0487329f, 0.547758f, 0.111111f, 0.610136f,
    0.649123f, 0.150097f, 0.212476f, 0.711501f, 0.797271f, 0.298246f, 0.859649f, 0.360624f,
    0.118908f, 0.617934f, 0.0565302f, 0.555556f, 0.329435f, 0.82846f, 0.516569f, 0.0175439f,
    0.867446f, 0.368421f, 0.805068f, 0.306043f, 0.578947f, 0.079922f, 0.267057f, 0.766082f,
    0.270955f, 0.76998f, 0.707602f, 0.208577f, 0.668616f, 0.169591f, 0.606238f, 0.107212f,
    0.520468f, 0.0214425f, 0.45809f, 0.957115f, 0.419103f, 0.918129f, 0.356725f, 0.855751f,
    0.988304f, 0.489279f, 0.426901f, 0.925926f, 0.450292f, 0.949318f, 0.512671f, 0.0136452f,
    0.239766f, 0.738791f, 0.676413f, 0.177388f, 0.699805f, 0.20078f, 0.263158f, 0.762183f,
    0.773879f, 0.274854f, 0.337232f, 0.836257f, 0.672515f, 0.173489f, 0.734893f, 0.235867f,
    0.0253411f, 0.524366f, 0.586745f, 0.0877193f, 0.423002f, 0.922027f, 0.48538f, 0.984405f,
    0.74269f, 0.243665f, 0.680312f, 0.181287f, 0.953216f, 0.454191f, 0.1423f, 0.641326f,
    0.493177f, 0.992203f, 0.430799f, 0.929825f, 0.204678f, 0.703704f, 0.890838f, 0.391813f,
    0.894737f, 0.395712f, 0.0838207f, 0.582846f, 0.0448343f, 0.54386f, 0.231969f, 0.730994f,
    0.146199f, 0.645224f, 0.832359f, 0.333333f, 0.793372f, 0.294347f, 0.980507f, 0.481481f,
    0.364522f, 0.863548f, 0.80117f, 0.302144f, 0.824561f, 0.325536f, 0.138402f, 0.637427f,
    0.614035f, 0.11501f, 0.0526316f, 0.551657f, 0.0760234f, 0.575049f, 0.88694f, 0.387914f,
};

PNANOVDB_FORCE_INLINE float pnanovdb_dither_lookup(pnanovdb_bool_t enabled, int offset)
{
    return enabled ? pnanovdb_dither_lut[offset & 511] : 0.5f;
}

// ------------------------------------------------ HDDA -----------------------------------------------------------

#ifdef PNANOVDB_HDDA

// Comment out to disable this explicit round-off check
#define PNANOVDB_ENFORCE_FORWARD_STEPPING

#define PNANOVDB_HDDA_FLOAT_MAX 1e38f

struct pnanovdb_hdda_t
{
    pnanovdb_int32_t dim;
    float tmin;
    float tmax;
    pnanovdb_coord_t voxel;
    pnanovdb_coord_t step;
    pnanovdb_vec3_t delta;
    pnanovdb_vec3_t next;
};
PNANOVDB_STRUCT_TYPEDEF(pnanovdb_hdda_t)

PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_hdda_pos_to_ijk(PNANOVDB_IN(pnanovdb_vec3_t) pos)
{
    pnanovdb_coord_t voxel;
    voxel.x = pnanovdb_float_to_int32(pnanovdb_floor(PNANOVDB_DEREF(pos).x));
    voxel.y = pnanovdb_float_to_int32(pnanovdb_floor(PNANOVDB_DEREF(pos).y));
    voxel.z = pnanovdb_float_to_int32(pnanovdb_floor(PNANOVDB_DEREF(pos).z));
    return voxel;
}

PNANOVDB_FORCE_INLINE pnanovdb_coord_t pnanovdb_hdda_pos_to_voxel(PNANOVDB_IN(pnanovdb_vec3_t) pos, int dim)
{
    pnanovdb_coord_t voxel;
    voxel.x = pnanovdb_float_to_int32(pnanovdb_floor(PNANOVDB_DEREF(pos).x)) & (~(dim - 1));
    voxel.y = pnanovdb_float_to_int32(pnanovdb_floor(PNANOVDB_DEREF(pos).y)) & (~(dim - 1));
    voxel.z = pnanovdb_float_to_int32(pnanovdb_floor(PNANOVDB_DEREF(pos).z)) & (~(dim - 1));
    return voxel;
}

PNANOVDB_FORCE_INLINE pnanovdb_vec3_t pnanovdb_hdda_ray_start(PNANOVDB_IN(pnanovdb_vec3_t) origin, float tmin, PNANOVDB_IN(pnanovdb_vec3_t) direction)
{
    pnanovdb_vec3_t pos = pnanovdb_vec3_add(
        pnanovdb_vec3_mul(PNANOVDB_DEREF(direction), pnanovdb_vec3_uniform(tmin)),
        PNANOVDB_DEREF(origin)
    );
    return pos;
}

PNANOVDB_FORCE_INLINE void pnanovdb_hdda_init(PNANOVDB_INOUT(pnanovdb_hdda_t) hdda, PNANOVDB_IN(pnanovdb_vec3_t) origin, float tmin, PNANOVDB_IN(pnanovdb_vec3_t) direction, float tmax, int dim)
{
    PNANOVDB_DEREF(hdda).dim = dim;
    PNANOVDB_DEREF(hdda).tmin = tmin;
    PNANOVDB_DEREF(hdda).tmax = tmax;

    pnanovdb_vec3_t pos = pnanovdb_hdda_ray_start(origin, tmin, direction);
    pnanovdb_vec3_t dir_inv = pnanovdb_vec3_div(pnanovdb_vec3_uniform(1.f), PNANOVDB_DEREF(direction));

    PNANOVDB_DEREF(hdda).voxel = pnanovdb_hdda_pos_to_voxel(PNANOVDB_REF(pos), dim);

    // x
    if (PNANOVDB_DEREF(direction).x == 0.f)
    {
        PNANOVDB_DEREF(hdda).next.x = PNANOVDB_HDDA_FLOAT_MAX;
        PNANOVDB_DEREF(hdda).step.x = 0;
        PNANOVDB_DEREF(hdda).delta.x = 0.f;
    }
    else if (dir_inv.x > 0.f)
    {
        PNANOVDB_DEREF(hdda).step.x = 1;
        PNANOVDB_DEREF(hdda).next.x = PNANOVDB_DEREF(hdda).tmin + (PNANOVDB_DEREF(hdda).voxel.x + dim - pos.x) * dir_inv.x;
        PNANOVDB_DEREF(hdda).delta.x = dir_inv.x;
    }
    else
    {
        PNANOVDB_DEREF(hdda).step.x = -1;
        PNANOVDB_DEREF(hdda).next.x = PNANOVDB_DEREF(hdda).tmin + (PNANOVDB_DEREF(hdda).voxel.x - pos.x) * dir_inv.x;
        PNANOVDB_DEREF(hdda).delta.x = -dir_inv.x;
    }

    // y
    if (PNANOVDB_DEREF(direction).y == 0.f)
    {
        PNANOVDB_DEREF(hdda).next.y = PNANOVDB_HDDA_FLOAT_MAX;
        PNANOVDB_DEREF(hdda).step.y = 0;
        PNANOVDB_DEREF(hdda).delta.y = 0.f;
    }
    else if (dir_inv.y > 0.f)
    {
        PNANOVDB_DEREF(hdda).step.y = 1;
        PNANOVDB_DEREF(hdda).next.y = PNANOVDB_DEREF(hdda).tmin + (PNANOVDB_DEREF(hdda).voxel.y + dim - pos.y) * dir_inv.y;
        PNANOVDB_DEREF(hdda).delta.y = dir_inv.y;
    }
    else
    {
        PNANOVDB_DEREF(hdda).step.y = -1;
        PNANOVDB_DEREF(hdda).next.y = PNANOVDB_DEREF(hdda).tmin + (PNANOVDB_DEREF(hdda).voxel.y - pos.y) * dir_inv.y;
        PNANOVDB_DEREF(hdda).delta.y = -dir_inv.y;
    }

    // z
    if (PNANOVDB_DEREF(direction).z == 0.f)
    {
        PNANOVDB_DEREF(hdda).next.z = PNANOVDB_HDDA_FLOAT_MAX;
        PNANOVDB_DEREF(hdda).step.z = 0;
        PNANOVDB_DEREF(hdda).delta.z = 0.f;
    }
    else if (dir_inv.z > 0.f)
    {
        PNANOVDB_DEREF(hdda).step.z = 1;
        PNANOVDB_DEREF(hdda).next.z = PNANOVDB_DEREF(hdda).tmin + (PNANOVDB_DEREF(hdda).voxel.z + dim - pos.z) * dir_inv.z;
        PNANOVDB_DEREF(hdda).delta.z = dir_inv.z;
    }
    else
    {
        PNANOVDB_DEREF(hdda).step.z = -1;
        PNANOVDB_DEREF(hdda).next.z = PNANOVDB_DEREF(hdda).tmin + (PNANOVDB_DEREF(hdda).voxel.z - pos.z) * dir_inv.z;
        PNANOVDB_DEREF(hdda).delta.z = -dir_inv.z;
    }
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_hdda_update(PNANOVDB_INOUT(pnanovdb_hdda_t) hdda, PNANOVDB_IN(pnanovdb_vec3_t) origin, PNANOVDB_IN(pnanovdb_vec3_t) direction, int dim)
{
    if (PNANOVDB_DEREF(hdda).dim == dim)
    {
        return PNANOVDB_FALSE;
    }
    PNANOVDB_DEREF(hdda).dim = dim;

    pnanovdb_vec3_t pos = pnanovdb_vec3_add(
        pnanovdb_vec3_mul(PNANOVDB_DEREF(direction), pnanovdb_vec3_uniform(PNANOVDB_DEREF(hdda).tmin)),
        PNANOVDB_DEREF(origin)
    );
    pnanovdb_vec3_t dir_inv = pnanovdb_vec3_div(pnanovdb_vec3_uniform(1.f), PNANOVDB_DEREF(direction));

    PNANOVDB_DEREF(hdda).voxel = pnanovdb_hdda_pos_to_voxel(PNANOVDB_REF(pos), dim);

    if (PNANOVDB_DEREF(hdda).step.x != 0)
    {
        PNANOVDB_DEREF(hdda).next.x = PNANOVDB_DEREF(hdda).tmin + (PNANOVDB_DEREF(hdda).voxel.x - pos.x) * dir_inv.x;
        if (PNANOVDB_DEREF(hdda).step.x > 0)
        {
            PNANOVDB_DEREF(hdda).next.x += dim * dir_inv.x;
        }
    }
    if (PNANOVDB_DEREF(hdda).step.y != 0)
    {
        PNANOVDB_DEREF(hdda).next.y = PNANOVDB_DEREF(hdda).tmin + (PNANOVDB_DEREF(hdda).voxel.y - pos.y) * dir_inv.y;
        if (PNANOVDB_DEREF(hdda).step.y > 0)
        {
            PNANOVDB_DEREF(hdda).next.y += dim * dir_inv.y;
        }
    }
    if (PNANOVDB_DEREF(hdda).step.z != 0)
    {
        PNANOVDB_DEREF(hdda).next.z = PNANOVDB_DEREF(hdda).tmin + (PNANOVDB_DEREF(hdda).voxel.z - pos.z) * dir_inv.z;
        if (PNANOVDB_DEREF(hdda).step.z > 0)
        {
            PNANOVDB_DEREF(hdda).next.z += dim * dir_inv.z;
        }
    }

    return PNANOVDB_TRUE;
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_hdda_step(PNANOVDB_INOUT(pnanovdb_hdda_t) hdda)
{
    pnanovdb_bool_t ret;
    if (PNANOVDB_DEREF(hdda).next.x < PNANOVDB_DEREF(hdda).next.y && PNANOVDB_DEREF(hdda).next.x < PNANOVDB_DEREF(hdda).next.z)
    {
#ifdef PNANOVDB_ENFORCE_FORWARD_STEPPING
        if (PNANOVDB_DEREF(hdda).next.x <= PNANOVDB_DEREF(hdda).tmin)
        {
            PNANOVDB_DEREF(hdda).next.x += PNANOVDB_DEREF(hdda).tmin - 0.999999f * PNANOVDB_DEREF(hdda).next.x + 1.0e-6f;
        }
#endif
        PNANOVDB_DEREF(hdda).tmin = PNANOVDB_DEREF(hdda).next.x;
        PNANOVDB_DEREF(hdda).next.x += PNANOVDB_DEREF(hdda).dim * PNANOVDB_DEREF(hdda).delta.x;
        PNANOVDB_DEREF(hdda).voxel.x += PNANOVDB_DEREF(hdda).dim * PNANOVDB_DEREF(hdda).step.x;
        ret = PNANOVDB_DEREF(hdda).tmin <= PNANOVDB_DEREF(hdda).tmax;
    }
    else if (PNANOVDB_DEREF(hdda).next.y < PNANOVDB_DEREF(hdda).next.z)
    {
#ifdef PNANOVDB_ENFORCE_FORWARD_STEPPING
        if (PNANOVDB_DEREF(hdda).next.y <= PNANOVDB_DEREF(hdda).tmin)
        {
            PNANOVDB_DEREF(hdda).next.y += PNANOVDB_DEREF(hdda).tmin - 0.999999f * PNANOVDB_DEREF(hdda).next.y + 1.0e-6f;
        }
#endif
        PNANOVDB_DEREF(hdda).tmin = PNANOVDB_DEREF(hdda).next.y;
        PNANOVDB_DEREF(hdda).next.y += PNANOVDB_DEREF(hdda).dim * PNANOVDB_DEREF(hdda).delta.y;
        PNANOVDB_DEREF(hdda).voxel.y += PNANOVDB_DEREF(hdda).dim * PNANOVDB_DEREF(hdda).step.y;
        ret = PNANOVDB_DEREF(hdda).tmin <= PNANOVDB_DEREF(hdda).tmax;
    }
    else
    {
#ifdef PNANOVDB_ENFORCE_FORWARD_STEPPING
        if (PNANOVDB_DEREF(hdda).next.z <= PNANOVDB_DEREF(hdda).tmin)
        {
            PNANOVDB_DEREF(hdda).next.z += PNANOVDB_DEREF(hdda).tmin - 0.999999f * PNANOVDB_DEREF(hdda).next.z + 1.0e-6f;
        }
#endif
        PNANOVDB_DEREF(hdda).tmin = PNANOVDB_DEREF(hdda).next.z;
        PNANOVDB_DEREF(hdda).next.z += PNANOVDB_DEREF(hdda).dim * PNANOVDB_DEREF(hdda).delta.z;
        PNANOVDB_DEREF(hdda).voxel.z += PNANOVDB_DEREF(hdda).dim * PNANOVDB_DEREF(hdda).step.z;
        ret = PNANOVDB_DEREF(hdda).tmin <= PNANOVDB_DEREF(hdda).tmax;
    }
    return ret;
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_hdda_ray_clip(
    PNANOVDB_IN(pnanovdb_vec3_t) bbox_min,
    PNANOVDB_IN(pnanovdb_vec3_t) bbox_max,
    PNANOVDB_IN(pnanovdb_vec3_t) origin, PNANOVDB_INOUT(float) tmin,
    PNANOVDB_IN(pnanovdb_vec3_t) direction, PNANOVDB_INOUT(float) tmax
)
{
    pnanovdb_vec3_t dir_inv = pnanovdb_vec3_div(pnanovdb_vec3_uniform(1.f), PNANOVDB_DEREF(direction));
    pnanovdb_vec3_t t0 = pnanovdb_vec3_mul(pnanovdb_vec3_sub(PNANOVDB_DEREF(bbox_min), PNANOVDB_DEREF(origin)), dir_inv);
    pnanovdb_vec3_t t1 = pnanovdb_vec3_mul(pnanovdb_vec3_sub(PNANOVDB_DEREF(bbox_max), PNANOVDB_DEREF(origin)), dir_inv);
    pnanovdb_vec3_t tmin3 = pnanovdb_vec3_min(t0, t1);
    pnanovdb_vec3_t tmax3 = pnanovdb_vec3_max(t0, t1);
    float tnear = pnanovdb_max(tmin3.x, pnanovdb_max(tmin3.y, tmin3.z));
    float tfar = pnanovdb_min(tmax3.x, pnanovdb_min(tmax3.y, tmax3.z));
    pnanovdb_bool_t hit = tnear <= tfar;
    PNANOVDB_DEREF(tmin) = pnanovdb_max(PNANOVDB_DEREF(tmin), tnear);
    PNANOVDB_DEREF(tmax) = pnanovdb_min(PNANOVDB_DEREF(tmax), tfar);
    return hit;
}

PNANOVDB_FORCE_INLINE pnanovdb_bool_t pnanovdb_hdda_zero_crossing(
    pnanovdb_grid_type_t grid_type,
    pnanovdb_buf_t buf,
    PNANOVDB_INOUT(pnanovdb_readaccessor_t) acc,
    PNANOVDB_IN(pnanovdb_vec3_t) origin, float tmin,
    PNANOVDB_IN(pnanovdb_vec3_t) direction, float tmax,
    PNANOVDB_INOUT(float) thit,
    PNANOVDB_INOUT(float) v
)
{
    pnanovdb_coord_t bbox_min = pnanovdb_root_get_bbox_min(buf, PNANOVDB_DEREF(acc).root);
    pnanovdb_coord_t bbox_max = pnanovdb_root_get_bbox_max(buf, PNANOVDB_DEREF(acc).root);
    pnanovdb_vec3_t bbox_minf = pnanovdb_coord_to_vec3(bbox_min);
    pnanovdb_vec3_t bbox_maxf = pnanovdb_coord_to_vec3(pnanovdb_coord_add(bbox_max, pnanovdb_coord_uniform(1)));

    pnanovdb_bool_t hit = pnanovdb_hdda_ray_clip(PNANOVDB_REF(bbox_minf), PNANOVDB_REF(bbox_maxf), origin, PNANOVDB_REF(tmin), direction, PNANOVDB_REF(tmax));
    if (!hit || tmax > 1.0e20f)
    {
        return PNANOVDB_FALSE;
    }

    pnanovdb_vec3_t pos = pnanovdb_hdda_ray_start(origin, tmin, direction);
    pnanovdb_coord_t ijk = pnanovdb_hdda_pos_to_ijk(PNANOVDB_REF(pos));

    pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, PNANOVDB_REF(ijk));
    float v0 = pnanovdb_read_float(buf, address);

    pnanovdb_int32_t dim = pnanovdb_uint32_as_int32(pnanovdb_readaccessor_get_dim(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, PNANOVDB_REF(ijk)));
    pnanovdb_hdda_t hdda;
    pnanovdb_hdda_init(PNANOVDB_REF(hdda), origin, tmin, direction, tmax, dim);
    while (pnanovdb_hdda_step(PNANOVDB_REF(hdda)))
    {
        pnanovdb_vec3_t pos_start = pnanovdb_hdda_ray_start(origin, hdda.tmin + 1.0001f, direction);
        ijk = pnanovdb_hdda_pos_to_ijk(PNANOVDB_REF(pos_start));
        dim = pnanovdb_uint32_as_int32(pnanovdb_readaccessor_get_dim(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, PNANOVDB_REF(ijk)));
        pnanovdb_hdda_update(PNANOVDB_REF(hdda), origin, direction, dim);
        if (hdda.dim > 1 || !pnanovdb_readaccessor_is_active(grid_type, buf, acc, PNANOVDB_REF(ijk)))
        {
            continue;
        }
        while (pnanovdb_hdda_step(PNANOVDB_REF(hdda)) && pnanovdb_readaccessor_is_active(grid_type, buf, acc, PNANOVDB_REF(hdda.voxel)))
        {
            ijk = hdda.voxel;
            pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, buf, acc, PNANOVDB_REF(ijk));
            PNANOVDB_DEREF(v) = pnanovdb_read_float(buf, address);
            if (PNANOVDB_DEREF(v) * v0 < 0.f)
            {
                PNANOVDB_DEREF(thit) = hdda.tmin;
                return PNANOVDB_TRUE;
            }
        }
    }
    return PNANOVDB_FALSE;
}

#endif

#endif // end of NANOVDB_PNANOVDB_H_HAS_BEEN_INCLUDED
