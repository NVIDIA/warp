/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

// All built-in types and functions. To be compatible with runtime NVRTC compilation
// this header must be independently compilable (i.e.: without external SDK headers)
// to achieve this we redefine a subset of CRT functions (printf, pow, sin, cos, etc)

#include "crt.h"

#ifdef _WIN32
#define __restrict__ __restrict
#endif

#if !defined(__CUDACC__)
    #define CUDA_CALLABLE
    #define CUDA_CALLABLE_DEVICE
#else
    #define CUDA_CALLABLE __host__ __device__ 
    #define CUDA_CALLABLE_DEVICE __device__
#endif

#ifdef WP_VERIFY_FP
#define FP_CHECK 1
#define DO_IF_FPCHECK(X) {X}
#define DO_IF_NO_FPCHECK(X)
#else
#define FP_CHECK 0
#define DO_IF_FPCHECK(X)
#define DO_IF_NO_FPCHECK(X) {X}
#endif

#define RAD_TO_DEG 57.29577951308232087679
#define DEG_TO_RAD  0.01745329251994329577

#if defined(__CUDACC__) && !defined(_MSC_VER)
__device__ void __debugbreak() {}
#endif

namespace wp
{


// numeric types (used from generated kernels)
typedef float float32;
typedef double float64;

typedef int8_t int8;
typedef uint8_t uint8;

typedef int16_t int16;
typedef uint16_t uint16;

typedef int32_t int32;
typedef uint32_t uint32;

typedef int64_t int64;
typedef uint64_t uint64;


// matches Python string type for constant strings
typedef const char* str;



struct half;

CUDA_CALLABLE half float_to_half(float x);
CUDA_CALLABLE float half_to_float(half x);

struct half
{
    CUDA_CALLABLE inline half() : u(0) {}

    CUDA_CALLABLE inline half(float f)
    {
        *this = float_to_half(f);
    }

    unsigned short u;

    CUDA_CALLABLE inline bool operator==(const half& h) const { return u == h.u; }
    CUDA_CALLABLE inline bool operator!=(const half& h) const { return u != h.u; }
    CUDA_CALLABLE inline bool operator>(const half& h) const { return half_to_float(*this) > half_to_float(h); }
    CUDA_CALLABLE inline bool operator>=(const half& h) const { return half_to_float(*this) >= half_to_float(h); }
    CUDA_CALLABLE inline bool operator<(const half& h) const { return half_to_float(*this) < half_to_float(h); }
    CUDA_CALLABLE inline bool operator<=(const half& h) const { return half_to_float(*this) <= half_to_float(h); }

    CUDA_CALLABLE inline bool operator!() const
    {
        return float32(*this) == 0;
    }

    CUDA_CALLABLE inline half operator*=(const half& h)
    {
        half prod = half(float32(*this) * float32(h));
        this->u = prod.u;
        return *this;
    }

    CUDA_CALLABLE inline half operator/=(const half& h)
    {
        half quot = half(float32(*this) / float32(h));
        this->u = quot.u;
        return *this;
    }

    CUDA_CALLABLE inline half operator+=(const half& h)
    {
        half sum = half(float32(*this) + float32(h));
        this->u = sum.u;
        return *this;
    }
    
    CUDA_CALLABLE inline half operator-=(const half& h)
    {
        half diff = half(float32(*this) - float32(h));
        this->u = diff.u;
        return *this;
    }

    CUDA_CALLABLE inline operator float32() const { return float32(half_to_float(*this)); }
    CUDA_CALLABLE inline operator float64() const { return float64(half_to_float(*this)); }
    CUDA_CALLABLE inline operator int8() const { return int8(half_to_float(*this)); }
    CUDA_CALLABLE inline operator uint8() const { return uint8(half_to_float(*this)); }
    CUDA_CALLABLE inline operator int16() const { return int16(half_to_float(*this)); }
    CUDA_CALLABLE inline operator uint16() const { return uint16(half_to_float(*this)); }
    CUDA_CALLABLE inline operator int32() const { return int32(half_to_float(*this)); }
    CUDA_CALLABLE inline operator uint32() const { return uint32(half_to_float(*this)); }
    CUDA_CALLABLE inline operator int64() const { return int64(half_to_float(*this)); }
    CUDA_CALLABLE inline operator uint64() const { return uint64(half_to_float(*this)); }
};

static_assert(sizeof(half) == 2, "Size of half / float16 type must be 2-bytes");

typedef half float16;

#if __CUDA_ARCH__

CUDA_CALLABLE inline half float_to_half(float x)
{
    half h;
    asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(h.u) : "f"(x));  
    return h;
}

CUDA_CALLABLE inline float half_to_float(half x)
{
    float val;
    asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(x.u));
    return val;
}

#else

// adapted from Fabien Giesen's post: https://gist.github.com/rygorous/2156668
inline half float_to_half(float x)
{
    union fp32
    {
        uint32 u;
        float f;

        struct
        {
            unsigned int mantissa : 23;
            unsigned int exponent : 8;
            unsigned int sign : 1;
        };
    };

    fp32 f;
    f.f = x;

    fp32 f32infty = { 255 << 23 };
    fp32 f16infty = { 31 << 23 };
    fp32 magic = { 15 << 23 };
    uint32 sign_mask = 0x80000000u;
    uint32 round_mask = ~0xfffu; 
    half o;

    uint32 sign = f.u & sign_mask;
    f.u ^= sign;

    // NOTE all the integer compares in this function can be safely
    // compiled into signed compares since all operands are below
    // 0x80000000. Important if you want fast straight SSE2 code
    // (since there's no unsigned PCMPGTD).

    if (f.u >= f32infty.u) // Inf or NaN (all exponent bits set)
        o.u = (f.u > f32infty.u) ? 0x7e00 : 0x7c00; // NaN->qNaN and Inf->Inf
    else // (De)normalized number or zero
    {
        f.u &= round_mask;
        f.f *= magic.f;
        f.u -= round_mask;
        if (f.u > f16infty.u) f.u = f16infty.u; // Clamp to signed infinity if overflowed

        o.u = f.u >> 13; // Take the bits!
    }

    o.u |= sign >> 16;
    return o;
}


inline float half_to_float(half h)
{
    union fp32
    {
        uint32 u;
        float f;

        struct
        {
            unsigned int mantissa : 23;
            unsigned int exponent : 8;
            unsigned int sign : 1;
        };
    };

    static const fp32 magic = { 113 << 23 };
    static const uint32 shifted_exp = 0x7c00 << 13; // exponent mask after shift
    fp32 o;

    o.u = (h.u & 0x7fff) << 13;     // exponent/mantissa bits
    uint32 exp = shifted_exp & o.u;   // just the exponent
    o.u += (127 - 15) << 23;        // exponent adjust

    // handle exponent special cases
    if (exp == shifted_exp) // Inf/NaN?
        o.u += (128 - 16) << 23;    // extra exp adjust
    else if (exp == 0) // Zero/Denormal?
    {
        o.u += 1 << 23;             // extra exp adjust
        o.f -= magic.f;             // renormalize
    }

    o.u |= (h.u & 0x8000) << 16;    // sign bit
    return o.f;
}


#endif


// BAD operator implementations for fp16 arithmetic...

// negation:
inline CUDA_CALLABLE half operator - (half a)
{
    return float_to_half( -half_to_float(a) );
}

inline CUDA_CALLABLE half operator + (half a,half b)
{
    return float_to_half( half_to_float(a) + half_to_float(b) );
}

inline CUDA_CALLABLE half operator - (half a,half b)
{
    return float_to_half( half_to_float(a) - half_to_float(b) );
}

inline CUDA_CALLABLE half operator * (half a,half b)
{
    return float_to_half( half_to_float(a) * half_to_float(b) );
}

inline CUDA_CALLABLE half operator * (half a,double b)
{
    return float_to_half( half_to_float(a) * b );
}

inline CUDA_CALLABLE half operator * (double a,half b)
{
    return float_to_half( a * half_to_float(b) );
}

inline CUDA_CALLABLE half operator / (half a,half b)
{
    return float_to_half( half_to_float(a) / half_to_float(b) );
}





template <typename T>
CUDA_CALLABLE float cast_float(T x) { return (float)(x); }

template <typename T>
CUDA_CALLABLE int cast_int(T x) { return (int)(x); }

template <typename T>
CUDA_CALLABLE void adj_cast_float(T x, T& adj_x, float adj_ret) { adj_x += T(adj_ret); }

template <typename T>
CUDA_CALLABLE void adj_cast_int(T x, T& adj_x, int adj_ret) { adj_x += adj_ret; }

template <typename T>
CUDA_CALLABLE inline void adj_int8(T, T&, int8) {}
template <typename T>
CUDA_CALLABLE inline void adj_uint8(T, T&, uint8) {}
template <typename T>
CUDA_CALLABLE inline void adj_int16(T, T&, int16) {}
template <typename T>
CUDA_CALLABLE inline void adj_uint16(T, T&, uint16) {}
template <typename T>
CUDA_CALLABLE inline void adj_int32(T, T&, int32) {}
template <typename T>
CUDA_CALLABLE inline void adj_uint32(T, T&, uint32) {}
template <typename T>
CUDA_CALLABLE inline void adj_int64(T, T&, int64) {}
template <typename T>
CUDA_CALLABLE inline void adj_uint64(T, T&, uint64) {}


template <typename T>
CUDA_CALLABLE inline void adj_float16(T x, T& adj_x, float16 adj_ret) { adj_x += T(adj_ret); }
template <typename T>
CUDA_CALLABLE inline void adj_float32(T x, T& adj_x, float32 adj_ret) { adj_x += T(adj_ret); }
template <typename T>
CUDA_CALLABLE inline void adj_float64(T x, T& adj_x, float64 adj_ret) { adj_x += T(adj_ret); }


#define kEps 0.0f

// basic ops for integer types
#define DECLARE_INT_OPS(T) \
inline CUDA_CALLABLE T mul(T a, T b) { return a*b; } \
inline CUDA_CALLABLE T div(T a, T b) { return a/b; } \
inline CUDA_CALLABLE T add(T a, T b) { return a+b; } \
inline CUDA_CALLABLE T sub(T a, T b) { return a-b; } \
inline CUDA_CALLABLE T mod(T a, T b) { return a%b; } \
inline CUDA_CALLABLE T min(T a, T b) { return a<b?a:b; } \
inline CUDA_CALLABLE T max(T a, T b) { return a>b?a:b; } \
inline CUDA_CALLABLE T clamp(T x, T a, T b) { return min(max(a, x), b); } \
inline CUDA_CALLABLE T floordiv(T a, T b) { return a/b; } \
inline CUDA_CALLABLE T nonzero(T x) { return x == T(0) ? T(0) : T(1); } \
inline CUDA_CALLABLE T sqrt(T x) { return 0; } \
inline CUDA_CALLABLE T bit_and(T a, T b) { return a&b; } \
inline CUDA_CALLABLE T bit_or(T a, T b) { return a|b; } \
inline CUDA_CALLABLE T bit_xor(T a, T b) { return a^b; } \
inline CUDA_CALLABLE T lshift(T a, T b) { return a<<b; } \
inline CUDA_CALLABLE T rshift(T a, T b) { return a>>b; } \
inline CUDA_CALLABLE T invert(T x) { return ~x; } \
inline CUDA_CALLABLE bool isfinite(T x) { return true; } \
inline CUDA_CALLABLE void adj_mul(T a, T b, T& adj_a, T& adj_b, T adj_ret) { } \
inline CUDA_CALLABLE void adj_div(T a, T b, T& adj_a, T& adj_b, T adj_ret) { } \
inline CUDA_CALLABLE void adj_add(T a, T b, T& adj_a, T& adj_b, T adj_ret) { } \
inline CUDA_CALLABLE void adj_sub(T a, T b, T& adj_a, T& adj_b, T adj_ret) { } \
inline CUDA_CALLABLE void adj_mod(T a, T b, T& adj_a, T& adj_b, T adj_ret) { } \
inline CUDA_CALLABLE void adj_min(T a, T b, T& adj_a, T& adj_b, T adj_ret) { } \
inline CUDA_CALLABLE void adj_max(T a, T b, T& adj_a, T& adj_b, T adj_ret) { } \
inline CUDA_CALLABLE void adj_abs(T x, T adj_x, T& adj_ret) { } \
inline CUDA_CALLABLE void adj_sign(T x, T adj_x, T& adj_ret) { } \
inline CUDA_CALLABLE void adj_clamp(T x, T a, T b, T& adj_x, T& adj_a, T& adj_b, T adj_ret) { } \
inline CUDA_CALLABLE void adj_floordiv(T a, T b, T& adj_a, T& adj_b, T adj_ret) { } \
inline CUDA_CALLABLE void adj_step(T x, T& adj_x, T adj_ret) { } \
inline CUDA_CALLABLE void adj_nonzero(T x, T& adj_x, T adj_ret) { } \
inline CUDA_CALLABLE void adj_sqrt(T x, T adj_x, T& adj_ret) { } \
inline CUDA_CALLABLE void adj_bit_and(T a, T b, T& adj_a, T& adj_b, T adj_ret) { } \
inline CUDA_CALLABLE void adj_bit_or(T a, T b, T& adj_a, T& adj_b, T adj_ret) { } \
inline CUDA_CALLABLE void adj_bit_xor(T a, T b, T& adj_a, T& adj_b, T adj_ret) { } \
inline CUDA_CALLABLE void adj_lshift(T a, T b, T& adj_a, T& adj_b, T adj_ret) { } \
inline CUDA_CALLABLE void adj_rshift(T a, T b, T& adj_a, T& adj_b, T adj_ret) { } \
inline CUDA_CALLABLE void adj_invert(T x, T adj_x, T& adj_ret) { }

inline CUDA_CALLABLE int8 abs(int8 x) { return ::abs(x); }
inline CUDA_CALLABLE int16 abs(int16 x) { return ::abs(x); }
inline CUDA_CALLABLE int32 abs(int32 x) { return ::abs(x); }
inline CUDA_CALLABLE int64 abs(int64 x) { return ::llabs(x); }
inline CUDA_CALLABLE uint8 abs(uint8 x) { return x; }
inline CUDA_CALLABLE uint16 abs(uint16 x) { return x; }
inline CUDA_CALLABLE uint32 abs(uint32 x) { return x; }
inline CUDA_CALLABLE uint64 abs(uint64 x) { return x; }

 DECLARE_INT_OPS(int8)
 DECLARE_INT_OPS(int16)
 DECLARE_INT_OPS(int32)
 DECLARE_INT_OPS(int64)
 DECLARE_INT_OPS(uint8)
 DECLARE_INT_OPS(uint16)
 DECLARE_INT_OPS(uint32)
 DECLARE_INT_OPS(uint64)
 

inline CUDA_CALLABLE int8 step(int8 x) { return x < 0 ? 1 : 0; }
inline CUDA_CALLABLE int16 step(int16 x) { return x < 0 ? 1 : 0; }
inline CUDA_CALLABLE int32 step(int32 x) { return x < 0 ? 1 : 0; }
inline CUDA_CALLABLE int64 step(int64 x) { return x < 0 ? 1 : 0; }
inline CUDA_CALLABLE uint8 step(uint8 x) { return 0; }
inline CUDA_CALLABLE uint16 step(uint16 x) { return 0; }
inline CUDA_CALLABLE uint32 step(uint32 x) { return 0; }
inline CUDA_CALLABLE uint64 step(uint64 x) { return 0; }


inline CUDA_CALLABLE int8 sign(int8 x) { return x < 0 ? -1 : 1; }
inline CUDA_CALLABLE int8 sign(int16 x) { return x < 0 ? -1 : 1; }
inline CUDA_CALLABLE int8 sign(int32 x) { return x < 0 ? -1 : 1; }
inline CUDA_CALLABLE int8 sign(int64 x) { return x < 0 ? -1 : 1; }
inline CUDA_CALLABLE uint8 sign(uint8 x) { return 1; }
inline CUDA_CALLABLE uint16 sign(uint16 x) { return 1; }
inline CUDA_CALLABLE uint32 sign(uint32 x) { return 1; }
inline CUDA_CALLABLE uint64 sign(uint64 x) { return 1; }



inline bool CUDA_CALLABLE isfinite(half x)
{
    return ::isfinite(float(x));
}
inline bool CUDA_CALLABLE isfinite(float x)
{
    return ::isfinite(x);
}
inline bool CUDA_CALLABLE isfinite(double x)
{
    return ::isfinite(x);
}

inline CUDA_CALLABLE void print(float16 f)
{
    printf("%g\n", half_to_float(f));
}

inline CUDA_CALLABLE void print(float f)
{
    printf("%g\n", f);
}

inline CUDA_CALLABLE void print(double f)
{
    printf("%g\n", f);
}


// basic ops for float types
#define DECLARE_FLOAT_OPS(T) \
inline CUDA_CALLABLE T mul(T a, T b) { return a*b; } \
inline CUDA_CALLABLE T add(T a, T b) { return a+b; } \
inline CUDA_CALLABLE T sub(T a, T b) { return a-b; } \
inline CUDA_CALLABLE T min(T a, T b) { return a<b?a:b; } \
inline CUDA_CALLABLE T max(T a, T b) { return a>b?a:b; } \
inline CUDA_CALLABLE T sign(T x) { return x < T(0) ? -1 : 1; } \
inline CUDA_CALLABLE T step(T x) { return x < T(0) ? T(1) : T(0); }\
inline CUDA_CALLABLE T nonzero(T x) { return x == T(0) ? T(0) : T(1); }\
inline CUDA_CALLABLE T clamp(T x, T a, T b) { return min(max(a, x), b); }\
inline CUDA_CALLABLE void adj_abs(T x, T& adj_x, T adj_ret) \
{\
    if (x < T(0))\
        adj_x -= adj_ret;\
    else\
        adj_x += adj_ret;\
}\
inline CUDA_CALLABLE void adj_mul(T a, T b, T& adj_a, T& adj_b, T adj_ret) { adj_a += b*adj_ret; adj_b += a*adj_ret; } \
inline CUDA_CALLABLE void adj_add(T a, T b, T& adj_a, T& adj_b, T adj_ret) { adj_a += adj_ret; adj_b += adj_ret; } \
inline CUDA_CALLABLE void adj_sub(T a, T b, T& adj_a, T& adj_b, T adj_ret) { adj_a += adj_ret; adj_b -= adj_ret; } \
inline CUDA_CALLABLE void adj_min(T a, T b, T& adj_a, T& adj_b, T adj_ret) \
{ \
    if (a < b) \
        adj_a += adj_ret; \
    else \
        adj_b += adj_ret; \
} \
inline CUDA_CALLABLE void adj_max(T a, T b, T& adj_a, T& adj_b, T adj_ret) \
{ \
    if (a > b) \
        adj_a += adj_ret; \
    else \
        adj_b += adj_ret; \
} \
inline CUDA_CALLABLE void adj_floordiv(T a, T b, T& adj_a, T& adj_b, T adj_ret) { } \
inline CUDA_CALLABLE void adj_mod(T a, T b, T& adj_a, T& adj_b, T adj_ret){ adj_a += adj_ret; }\
inline CUDA_CALLABLE void adj_sign(T x, T adj_x, T& adj_ret) { }\
inline CUDA_CALLABLE void adj_step(T x, T& adj_x, T adj_ret) { }\
inline CUDA_CALLABLE void adj_nonzero(T x, T& adj_x, T adj_ret) { }\
inline CUDA_CALLABLE void adj_clamp(T x, T a, T b, T& adj_x, T& adj_a, T& adj_b, T adj_ret)\
{\
    if (x < a)\
        adj_a += adj_ret;\
    else if (x > b)\
        adj_b += adj_ret;\
    else\
        adj_x += adj_ret;\
}\
inline CUDA_CALLABLE void adj_round(T x, T& adj_x, T adj_ret){ }\
inline CUDA_CALLABLE void adj_rint(T x, T& adj_x, T adj_ret){ }\
inline CUDA_CALLABLE void adj_trunc(T x, T& adj_x, T adj_ret){ }\
inline CUDA_CALLABLE void adj_floor(T x, T& adj_x, T adj_ret){ }\
inline CUDA_CALLABLE void adj_ceil(T x, T& adj_x, T adj_ret){ }\
inline CUDA_CALLABLE T div(T a, T b)\
{\
    DO_IF_FPCHECK(\
    if (!isfinite(a) || !isfinite(b) || b == T(0))\
    {\
        printf("%s:%d div(%f, %f)\n", __FILE__, __LINE__, float(a), float(b));\
        assert(0);\
    })\
    return a/b;\
}\
inline CUDA_CALLABLE void adj_div(T a, T b, T& adj_a, T& adj_b, T adj_ret)\
{\
    adj_a += adj_ret/b;\
    adj_b -= adj_ret*(a/b)/b;\
    DO_IF_FPCHECK(\
    if (!isfinite(adj_a) || !isfinite(adj_b))\
    {\
        printf("%s:%d - adj_div(%f, %f, %f, %f, %f)\n", __FILE__, __LINE__, float(a), float(b), float(adj_a), float(adj_b), float(adj_ret));\
        assert(0);\
    })\
}\

DECLARE_FLOAT_OPS(float16)
DECLARE_FLOAT_OPS(float32)
DECLARE_FLOAT_OPS(float64)



// basic ops for float types
inline CUDA_CALLABLE float16 mod(float16 a, float16 b)
{
#if FP_CHECK
    if (!isfinite(a) || !isfinite(b) || float(b) == 0.0f)
    {
        printf("%s:%d mod(%f, %f)\n", __FILE__, __LINE__, float(a), float(b));
        assert(0);
    }
#endif
    return fmodf(float(a), float(b));
}

inline CUDA_CALLABLE float32 mod(float32 a, float32 b)
{
#if FP_CHECK
    if (!isfinite(a) || !isfinite(b) || b == 0.0f)
    {
        printf("%s:%d mod(%f, %f)\n", __FILE__, __LINE__, a, b);
        assert(0);
    }
#endif
    return fmodf(a, b);
}

inline CUDA_CALLABLE double mod(double a, double b)
{
#if FP_CHECK
    if (!isfinite(a) || !isfinite(b) || b == 0.0f)
    {
        printf("%s:%d mod(%f, %f)\n", __FILE__, __LINE__, a, b);
        assert(0);
    }
#endif
    return fmod(a, b);
}

inline CUDA_CALLABLE half log(half a)
{
#if FP_CHECK
    if (!isfinite(a) || float(a) < 0.0f)
    {
        printf("%s:%d log(%f)\n", __FILE__, __LINE__, float(a));
        assert(0);
    }
#endif
    return ::logf(a);
}

inline CUDA_CALLABLE float log(float a)
{
#if FP_CHECK
    if (!isfinite(a) || a < 0.0f)
    {
        printf("%s:%d log(%f)\n", __FILE__, __LINE__, a);
        assert(0);
    }
#endif
    return ::logf(a);
}

inline CUDA_CALLABLE double log(double a)
{
#if FP_CHECK
    if (!isfinite(a) || a < 0.0)
    {
        printf("%s:%d log(%f)\n", __FILE__, __LINE__, a);
        assert(0);
    }
#endif
    return ::log(a);
}

inline CUDA_CALLABLE half log2(half a) 
{
#if FP_CHECK
    if (!isfinite(a) || float(a) < 0.0f)
    {
        printf("%s:%d log2(%f)\n", __FILE__, __LINE__, float(a));
        assert(0);
    }
#endif

    return ::log2f(float(a));    
}

inline CUDA_CALLABLE float log2(float a) 
{
#if FP_CHECK
    if (!isfinite(a) || a < 0.0f)
    {
        printf("%s:%d log2(%f)\n", __FILE__, __LINE__, a);
        assert(0);
    }
#endif

    return ::log2f(a);    
}

inline CUDA_CALLABLE double log2(double a) 
{
#if FP_CHECK
    if (!isfinite(a) || a < 0.0)
    {
        printf("%s:%d log2(%f)\n", __FILE__, __LINE__, a);
        assert(0);
    }
#endif

    return ::log2(a);    
}

inline CUDA_CALLABLE half log10(half a) 
{
#if FP_CHECK
    if (!isfinite(a) || float(a) < 0.0f)
    {
        printf("%s:%d log10(%f)\n", __FILE__, __LINE__, float(a));
        assert(0);
    }
#endif

    return ::log10f(float(a)); 
}

inline CUDA_CALLABLE float log10(float a) 
{
#if FP_CHECK
    if (!isfinite(a) || a < 0.0f)
    {
        printf("%s:%d log10(%f)\n", __FILE__, __LINE__, a);
        assert(0);
    }
#endif

    return ::log10f(a); 
}

inline CUDA_CALLABLE double log10(double a) 
{
#if FP_CHECK
    if (!isfinite(a) || a < 0.0)
    {
        printf("%s:%d log10(%f)\n", __FILE__, __LINE__, a);
        assert(0);
    }
#endif

    return ::log10(a); 
}

inline CUDA_CALLABLE half exp(half a)
{
    half result = ::expf(float(a));
#if FP_CHECK
    if (!isfinite(a) || !isfinite(result))
    {
        printf("%s:%d exp(%f) = %f\n", __FILE__, __LINE__, float(a), float(result));
        assert(0);
    }
#endif
    return result;
}
inline CUDA_CALLABLE float exp(float a)
{
    float result = ::expf(a);
#if FP_CHECK
    if (!isfinite(a) || !isfinite(result))
    {
        printf("%s:%d exp(%f) = %f\n", __FILE__, __LINE__, a, result);
        assert(0);
    }
#endif
    return result;
}
inline CUDA_CALLABLE double exp(double a)
{
    double result = ::exp(a);
#if FP_CHECK
    if (!isfinite(a) || !isfinite(result))
    {
        printf("%s:%d exp(%f) = %f\n", __FILE__, __LINE__, a, result);
        assert(0);
    }
#endif
    return result;
}

inline CUDA_CALLABLE half pow(half a, half b)
{
    float result = ::powf(float(a), float(b));
#if FP_CHECK
    if (!isfinite(float(a)) || !isfinite(float(b)) || !isfinite(result))
    {
        printf("%s:%d pow(%f, %f) = %f\n", __FILE__, __LINE__, float(a), float(b), result);
        assert(0);
    }
#endif
    return result;
}

inline CUDA_CALLABLE float pow(float a, float b)
{
    float result = ::powf(a, b);
#if FP_CHECK
    if (!isfinite(a) || !isfinite(b) || !isfinite(result))
    {
        printf("%s:%d pow(%f, %f) = %f\n", __FILE__, __LINE__, a, b, result);
        assert(0);
    }
#endif
    return result;
}

inline CUDA_CALLABLE double pow(double a, double b)
{
    double result = ::pow(a, b);
#if FP_CHECK
    if (!isfinite(a) || !isfinite(b) || !isfinite(result))
    {
        printf("%s:%d pow(%f, %f) = %f\n", __FILE__, __LINE__, a, b, result);
        assert(0);
    }
#endif
    return result;
}

inline CUDA_CALLABLE half floordiv(half a, half b)
{
#if FP_CHECK
    if (!isfinite(a) || !isfinite(b) || float(b) == 0.0f)
    {
        printf("%s:%d mod(%f, %f)\n", __FILE__, __LINE__, float(a), float(b));
        assert(0);
    }
#endif
    return floorf(float(a/b));
}
inline CUDA_CALLABLE float floordiv(float a, float b)
{
#if FP_CHECK
    if (!isfinite(a) || !isfinite(b) || b == 0.0f)
    {
        printf("%s:%d mod(%f, %f)\n", __FILE__, __LINE__, a, b);
        assert(0);
    }
#endif
    return floorf(a/b);
}
inline CUDA_CALLABLE double floordiv(double a, double b)
{
#if FP_CHECK
    if (!isfinite(a) || !isfinite(b) || b == 0.0)
    {
        printf("%s:%d mod(%f, %f)\n", __FILE__, __LINE__, a, b);
        assert(0);
    }
#endif
    return ::floor(a/b);
}

inline CUDA_CALLABLE float leaky_min(float a, float b, float r) { return min(a, b); }
inline CUDA_CALLABLE float leaky_max(float a, float b, float r) { return max(a, b); }

inline CUDA_CALLABLE half abs(half x) { return ::fabsf(float(x)); }
inline CUDA_CALLABLE float abs(float x) { return ::fabsf(x); }
inline CUDA_CALLABLE double abs(double x) { return ::fabs(x); }

inline CUDA_CALLABLE float acos(float x){ return ::acosf(min(max(x, -1.0f), 1.0f)); }
inline CUDA_CALLABLE float asin(float x){ return ::asinf(min(max(x, -1.0f), 1.0f)); }
inline CUDA_CALLABLE float atan(float x) { return ::atanf(x); }
inline CUDA_CALLABLE float atan2(float y, float x) { return ::atan2f(y, x); }
inline CUDA_CALLABLE float sin(float x) { return ::sinf(x); }
inline CUDA_CALLABLE float cos(float x) { return ::cosf(x); }

inline CUDA_CALLABLE double acos(double x){ return ::acos(min(max(x, -1.0), 1.0)); }
inline CUDA_CALLABLE double asin(double x){ return ::asin(min(max(x, -1.0), 1.0)); }
inline CUDA_CALLABLE double atan(double x) { return ::atan(x); }
inline CUDA_CALLABLE double atan2(double y, double x) { return ::atan2(y, x); }
inline CUDA_CALLABLE double sin(double x) { return ::sin(x); }
inline CUDA_CALLABLE double cos(double x) { return ::cos(x); }

inline CUDA_CALLABLE half acos(half x){ return ::acosf(min(max(float(x), -1.0f), 1.0f)); }
inline CUDA_CALLABLE half asin(half x){ return ::asinf(min(max(float(x), -1.0f), 1.0f)); }
inline CUDA_CALLABLE half atan(half x) { return ::atanf(float(x)); }
inline CUDA_CALLABLE half atan2(half y, half x) { return ::atan2f(float(y), float(x)); }
inline CUDA_CALLABLE half sin(half x) { return ::sinf(float(x)); }
inline CUDA_CALLABLE half cos(half x) { return ::cosf(float(x)); }


inline CUDA_CALLABLE float sqrt(float x)
{
#if FP_CHECK
    if (x < 0.0f)
    {
        printf("%s:%d sqrt(%f)\n", __FILE__, __LINE__, x);
        assert(0);
    }
#endif
    return ::sqrtf(x);
}
inline CUDA_CALLABLE double sqrt(double x)
{
#if FP_CHECK
    if (x < 0.0)
    {
        printf("%s:%d sqrt(%f)\n", __FILE__, __LINE__, x);
        assert(0);
    }
#endif
    return ::sqrt(x);
}
inline CUDA_CALLABLE half sqrt(half x)
{
#if FP_CHECK
    if (float(x) < 0.0f)
    {
        printf("%s:%d sqrt(%f)\n", __FILE__, __LINE__, float(x));
        assert(0);
    }
#endif
    return ::sqrtf(float(x));
}

inline CUDA_CALLABLE float tan(float x) { return ::tanf(x); }
inline CUDA_CALLABLE float sinh(float x) { return ::sinhf(x);}
inline CUDA_CALLABLE float cosh(float x) { return ::coshf(x);}
inline CUDA_CALLABLE float tanh(float x) { return ::tanhf(x);}
inline CUDA_CALLABLE float degrees(float x) { return x * RAD_TO_DEG;}
inline CUDA_CALLABLE float radians(float x) { return x * DEG_TO_RAD;}

inline CUDA_CALLABLE double tan(double x) { return ::tan(x); }
inline CUDA_CALLABLE double sinh(double x) { return ::sinh(x);}
inline CUDA_CALLABLE double cosh(double x) { return ::cosh(x);}
inline CUDA_CALLABLE double tanh(double x) { return ::tanh(x);}
inline CUDA_CALLABLE double degrees(double x) { return x * RAD_TO_DEG;}
inline CUDA_CALLABLE double radians(double x) { return x * DEG_TO_RAD;}

inline CUDA_CALLABLE half tan(half x) { return ::tanf(float(x)); }
inline CUDA_CALLABLE half sinh(half x) { return ::sinhf(float(x));}
inline CUDA_CALLABLE half cosh(half x) { return ::coshf(float(x));}
inline CUDA_CALLABLE half tanh(half x) { return ::tanhf(float(x));}
inline CUDA_CALLABLE half degrees(half x) { return x * RAD_TO_DEG;}
inline CUDA_CALLABLE half radians(half x) { return x * DEG_TO_RAD;}

inline CUDA_CALLABLE float round(float x) { return ::roundf(x); }
inline CUDA_CALLABLE float rint(float x) { return ::rintf(x); }
inline CUDA_CALLABLE float trunc(float x) { return ::truncf(x); }
inline CUDA_CALLABLE float floor(float x) { return ::floorf(x); }
inline CUDA_CALLABLE float ceil(float x) { return ::ceilf(x); }

#define DECLARE_ADJOINTS(T)\
inline CUDA_CALLABLE void adj_log(T a, T& adj_a, T adj_ret)\
{\
    adj_a += (T(1)/a)*adj_ret;\
    DO_IF_FPCHECK(if (!isfinite(adj_a))\
    {\
        printf("%s:%d - adj_log(%f, %f, %f)\n", __FILE__, __LINE__, float(a), float(adj_a), float(adj_ret));\
        assert(0);\
    })\
}\
inline CUDA_CALLABLE void adj_log2(T a, T& adj_a, T adj_ret)\
{ \
    adj_a += (T(1)/a)*(T(1)/log(T(2)))*adj_ret; \
    DO_IF_FPCHECK(if (!isfinite(adj_a))\
    {\
        printf("%s:%d - adj_log2(%f, %f, %f)\n", __FILE__, __LINE__, float(a), float(adj_a), float(adj_ret));\
        assert(0);\
    })   \
}\
inline CUDA_CALLABLE void adj_log10(T a, T& adj_a, T adj_ret)\
{\
    adj_a += (T(1)/a)*(T(1)/log(T(10)))*adj_ret; \
    DO_IF_FPCHECK(if (!isfinite(adj_a))\
    {\
        printf("%s:%d - adj_log10(%f, %f, %f)\n", __FILE__, __LINE__, float(a), float(adj_a), float(adj_ret));\
        assert(0);\
    })\
}\
inline CUDA_CALLABLE void adj_exp(T a, T& adj_a, T adj_ret) { adj_a += exp(a)*adj_ret; }\
inline CUDA_CALLABLE void adj_pow(T a, T b, T& adj_a, T& adj_b, T adj_ret)\
{ \
    adj_a += b*pow(a, b-T(1))*adj_ret;\
    adj_b += log(a)*pow(a, b)*adj_ret;\
    DO_IF_FPCHECK(if (!isfinite(adj_a) || !isfinite(adj_b))\
    {\
        printf("%s:%d - adj_pow(%f, %f, %f, %f, %f)\n", __FILE__, __LINE__, float(a), float(b), float(adj_a), float(adj_b), float(adj_ret));\
        assert(0);\
    })\
}\
inline CUDA_CALLABLE void adj_leaky_min(T a, T b, T r, T& adj_a, T& adj_b, T& adj_r, T adj_ret)\
{\
    if (a < b)\
        adj_a += adj_ret;\
    else\
    {\
        adj_a += r*adj_ret;\
        adj_b += adj_ret;\
    }\
}\
inline CUDA_CALLABLE void adj_leaky_max(T a, T b, T r, T& adj_a, T& adj_b, T& adj_r, T adj_ret)\
{\
    if (a > b)\
        adj_a += adj_ret;\
    else\
    {\
        adj_a += r*adj_ret;\
        adj_b += adj_ret;\
    }\
}\
inline CUDA_CALLABLE void adj_acos(T x, T& adj_x, T adj_ret)\
{\
    T d = sqrt(T(1)-x*x);\
    DO_IF_FPCHECK(adj_x -= (T(1)/d)*adj_ret;\
    if (!isfinite(d) || !isfinite(adj_x))\
    {\
        printf("%s:%d - adj_acos(%f, %f, %f)\n", __FILE__, __LINE__, float(x), float(adj_x), float(adj_ret));        \
        assert(0);\
    })\
    DO_IF_NO_FPCHECK(if (d > T(0))\
        adj_x -= (T(1)/d)*adj_ret;)\
}\
inline CUDA_CALLABLE void adj_asin(T x, T& adj_x, T adj_ret)\
{\
    T d = sqrt(T(1)-x*x);\
    DO_IF_FPCHECK(adj_x += (T(1)/d)*adj_ret;\
    if (!isfinite(d) || !isfinite(adj_x))\
    {\
        printf("%s:%d - adj_asin(%f, %f, %f)\n", __FILE__, __LINE__, float(x), float(adj_x), float(adj_ret));   \
        assert(0);\
    })\
    DO_IF_NO_FPCHECK(if (d > T(0))\
        adj_x += (T(1)/d)*adj_ret;)\
}\
inline CUDA_CALLABLE void adj_tan(T x, T& adj_x, T adj_ret)\
{\
    T cos_x = cos(x);\
    DO_IF_FPCHECK(adj_x += (T(1)/(cos_x*cos_x))*adj_ret;\
    if (!isfinite(adj_x) || cos_x == T(0))\
    {\
        printf("%s:%d - adj_tan(%f, %f, %f)\n", __FILE__, __LINE__, float(x), float(adj_x), float(adj_ret));\
        assert(0);\
    })\
    DO_IF_NO_FPCHECK(if (cos_x != T(0))\
        adj_x += (T(1)/(cos_x*cos_x))*adj_ret;)\
}\
inline CUDA_CALLABLE void adj_atan(T x, T& adj_x, T adj_ret)\
{\
    adj_x += adj_ret /(x*x + T(1));\
}\
inline CUDA_CALLABLE void adj_atan2(T y, T x, T& adj_y, T& adj_x, T adj_ret)\
{\
    T d = x*x + y*y;\
    DO_IF_FPCHECK(adj_x -= y/d*adj_ret;\
    adj_y += x/d*adj_ret;\
    if (!isfinite(adj_x) || !isfinite(adj_y) || d == T(0))\
    {\
        printf("%s:%d - adj_atan2(%f, %f, %f, %f, %f)\n", __FILE__, __LINE__, float(y), float(x), float(adj_y), float(adj_x), float(adj_ret));\
        assert(0);\
    })\
    DO_IF_NO_FPCHECK(if (d > T(0))\
    {\
        adj_x -= (y/d)*adj_ret;\
        adj_y += (x/d)*adj_ret;\
    })\
}\
inline CUDA_CALLABLE void adj_sin(T x, T& adj_x, T adj_ret)\
{\
    adj_x += cos(x)*adj_ret;\
}\
inline CUDA_CALLABLE void adj_cos(T x, T& adj_x, T adj_ret)\
{\
    adj_x -= sin(x)*adj_ret;\
}\
inline CUDA_CALLABLE void adj_sinh(T x, T& adj_x, T adj_ret)\
{\
    adj_x += cosh(x)*adj_ret;\
}\
inline CUDA_CALLABLE void adj_cosh(T x, T& adj_x, T adj_ret)\
{\
    adj_x += sinh(x)*adj_ret;\
}\
inline CUDA_CALLABLE void adj_tanh(T x, T& adj_x, T adj_ret)\
{\
    T tanh_x = tanh(x);\
    adj_x += (T(1) - tanh_x*tanh_x)*adj_ret;\
}\
inline CUDA_CALLABLE void adj_sqrt(T x, T& adj_x, T adj_ret)\
{\
    adj_x += T(0.5)*(T(1)/sqrt(x))*adj_ret;\
    DO_IF_FPCHECK(if (!isfinite(adj_x))\
    {\
        printf("%s:%d - adj_sqrt(%f, %f, %f)\n", __FILE__, __LINE__, float(x), float(adj_x), float(adj_ret));\
        assert(0);\
    })\
}\
inline CUDA_CALLABLE void adj_degrees(T x, T& adj_x, T adj_ret)\
{\
    adj_x += RAD_TO_DEG * adj_ret;\
}\
inline CUDA_CALLABLE void adj_radians(T x, T& adj_x, T adj_ret)\
{\
    adj_x += DEG_TO_RAD * adj_ret;\
}

DECLARE_ADJOINTS(float16)
DECLARE_ADJOINTS(float32)
DECLARE_ADJOINTS(float64)

template <typename C, typename T>
CUDA_CALLABLE inline T select(const C& cond, const T& a, const T& b)
{
    // The double NOT operator !! casts to bool without compiler warnings.
    return (!!cond) ? b : a;
}

template <typename C, typename T>
CUDA_CALLABLE inline void adj_select(const C& cond, const T& a, const T& b, C& adj_cond, T& adj_a, T& adj_b, const T& adj_ret)
{
    // The double NOT operator !! casts to bool without compiler warnings.
    if (!!cond)
        adj_b += adj_ret;
    else
        adj_a += adj_ret;
}

template <typename T>
CUDA_CALLABLE inline void copy(T& dest, const T& src)
{
    dest = src;
}

template <typename T>
CUDA_CALLABLE inline void adj_copy(T& dest, const T& src, T& adj_dest, T& adj_src)
{
    // nop, this is non-differentiable operation since it violates SSA
    adj_src = adj_dest;
    adj_dest = T(0);
}


// some helpful operator overloads (just for C++ use, these are not adjointed)

template <typename T>
CUDA_CALLABLE inline T& operator += (T& a, const T& b) { a = add(a, b); return a; }

template <typename T>
CUDA_CALLABLE inline T& operator -= (T& a, const T& b) { a = sub(a, b); return a; }

template <typename T>
CUDA_CALLABLE inline T operator+(const T& a, const T& b) { return add(a, b); }

template <typename T>
CUDA_CALLABLE inline T operator-(const T& a, const T& b) { return sub(a, b); }

template <typename T>
CUDA_CALLABLE inline T pos(const T& x) { return x; }
template <typename T>
CUDA_CALLABLE inline void adj_pos(const T& x, T& adj_x, const T& adj_ret) { adj_x += T(adj_ret); }

// unary negation implemented as negative multiply, not sure the fp implications of this
// may be better as 0.0 - x?
template <typename T>
CUDA_CALLABLE inline T neg(const T& x) { return T(0.0) - x; }
template <typename T>
CUDA_CALLABLE inline void adj_neg(const T& x, T& adj_x, const T& adj_ret) { adj_x += T(-adj_ret); }

// unary boolean negation
template <typename T>
CUDA_CALLABLE inline bool unot(const T& b) { return !b; }
template <typename T>
CUDA_CALLABLE inline void adj_unot(const T& b, T& adj_b, const bool& adj_ret) { }

const int LAUNCH_MAX_DIMS = 4;   // should match types.py

struct launch_bounds_t
{
    int shape[LAUNCH_MAX_DIMS]; // size of each dimension
    int ndim;                   // number of valid dimension
    size_t size;                // total number of threads
};

#ifdef __CUDACC__

// store launch bounds in shared memory so
// we can access them from any user func
// this is to avoid having to explicitly
// set another piece of __constant__ memory
// from the host
__shared__ launch_bounds_t s_launchBounds;

__device__ inline void set_launch_bounds(const launch_bounds_t& b)
{
    if (threadIdx.x == 0)
        s_launchBounds = b;

    __syncthreads();
}

#else

// for single-threaded CPU we store launch
// bounds in static memory to share globally
static launch_bounds_t s_launchBounds;
static size_t s_threadIdx;

inline void set_launch_bounds(const launch_bounds_t& b)
{
    s_launchBounds = b;
}
#endif

inline CUDA_CALLABLE size_t grid_index()
{
#ifdef __CUDACC__
    // Need to cast at least one of the variables being multiplied so that type promotion happens before the multiplication
    size_t grid_index = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
    return grid_index;
#else
    return s_threadIdx;
#endif
}

inline CUDA_CALLABLE int tid()
{
    const size_t index = grid_index();

    // For the 1-D tid() we need to warn the user if we're about to provide a truncated index
    // Only do this in _DEBUG when called from device to avoid excessive register allocation
#if defined(_DEBUG) || !defined(__CUDA_ARCH__)
    if (index > 2147483647) {
        printf("Warp warning: tid() is returning an overflowed int\n");
    }
#endif
    return static_cast<int>(index);
}

inline CUDA_CALLABLE_DEVICE void tid(int& i, int& j)
{
    const size_t index = grid_index();

    const size_t n = s_launchBounds.shape[1];

    // convert to work item
    i = index/n;
    j = index%n;
}

inline CUDA_CALLABLE_DEVICE void tid(int& i, int& j, int& k)
{
    const size_t index = grid_index();

    const size_t n = s_launchBounds.shape[1];
    const size_t o = s_launchBounds.shape[2];

    // convert to work item
    i = index/(n*o);
    j = index%(n*o)/o;
    k = index%o;
}

inline CUDA_CALLABLE_DEVICE void tid(int& i, int& j, int& k, int& l)
{
    const size_t index = grid_index();

    const size_t n = s_launchBounds.shape[1];
    const size_t o = s_launchBounds.shape[2];
    const size_t p = s_launchBounds.shape[3];

    // convert to work item
    i = index/(n*o*p);
    j = index%(n*o*p)/(o*p);
    k = index%(o*p)/p;
    l = index%p;
}

template<typename T>
inline CUDA_CALLABLE T atomic_add(T* buf, T value)
{
#if !defined(__CUDA_ARCH__)
    T old = buf[0];
    buf[0] += value;
    return old;
#else
    return atomicAdd(buf, value);
#endif
}

template<>
inline CUDA_CALLABLE float16 atomic_add(float16* buf, float16 value)
{
#if !defined(__CUDA_ARCH__)
    float16 old = buf[0];
    buf[0] += value;
    return old;
#elif defined(__clang__)  // CUDA compiled by Clang
	__half r = atomicAdd(reinterpret_cast<__half*>(buf), *reinterpret_cast<__half*>(&value));
    return *reinterpret_cast<float16*>(&r);
#else  // CUDA compiled by NVRTC
    //return atomicAdd(buf, value);
    
    /* Define __PTR for atomicAdd prototypes below, undef after done */
    #if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)
    #define __PTR   "l"
    #else
    #define __PTR   "r"
    #endif /*(defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)*/
   
    half r = 0.0;

    #if __CUDA_ARCH__ >= 700

        asm volatile ("{ atom.add.noftz.f16 %0,[%1],%2; }\n"
                    : "=h"(r.u)
                    : __PTR(buf), "h"(value.u)
                    : "memory");
    #endif

    return r;

    #undef __PTR

#endif  // CUDA compiled by NVRTC

}

// emulate atomic float max
inline CUDA_CALLABLE float atomic_max(float* address, float val)
{
#if defined(__CUDA_ARCH__)
    int *address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    
	while (val > __int_as_float(old))
	{
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
    }

    return __int_as_float(old);

#else
    float old = *address;
    *address = max(old, val);
    return old;
#endif
}

// emulate atomic float min/max with atomicCAS()
inline CUDA_CALLABLE float atomic_min(float* address, float val)
{
#if defined(__CUDA_ARCH__)
    int *address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    while (val < __int_as_float(old)) 
	{
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
    }

    return __int_as_float(old);

#else
    float old = *address;
    *address = min(old, val);
    return old;
#endif
}

inline CUDA_CALLABLE int atomic_max(int* address, int val)
{
#if defined(__CUDA_ARCH__)
    return atomicMax(address, val);

#else
    int old = *address;
    *address = max(old, val);
    return old;
#endif
}

// atomic int min
inline CUDA_CALLABLE int atomic_min(int* address, int val)
{
#if defined(__CUDA_ARCH__)
    return atomicMin(address, val);

#else
    int old = *address;
    *address = min(old, val);
    return old;
#endif
}


} // namespace wp

#include "vec.h"
#include "mat.h"
#include "quat.h"
#include "spatial.h"
#include "intersect.h"
#include "intersect_adj.h"

//--------------
namespace wp
{


// dot for scalar types just to make some templates compile for scalar/vector
inline CUDA_CALLABLE float dot(float a, float b) { return mul(a, b); }
inline CUDA_CALLABLE void adj_dot(float a, float b, float& adj_a, float& adj_b, float adj_ret) { adj_mul(a, b, adj_a, adj_b, adj_ret); }
inline CUDA_CALLABLE float tensordot(float a, float b) { return mul(a, b); }


#define DECLARE_INTERP_FUNCS(T) \
CUDA_CALLABLE inline T smoothstep(T edge0, T edge1, T x)\
{\
    x = clamp((x - edge0) / (edge1 - edge0), T(0), T(1));\
    return x * x * (T(3) - T(2) * x);\
}\
CUDA_CALLABLE inline void adj_smoothstep(T edge0, T edge1, T x, T& adj_edge0, T& adj_edge1, T& adj_x, T adj_ret)\
{\
    T ab = edge0 - edge1;\
    T ax = edge0 - x;\
    T bx = edge1 - x;\
    T xb = x - edge1;\
\
    if (bx / ab >= T(0) || ax / ab <= T(0))\
    {\
        return;\
    }\
\
    T ab3 = ab * ab * ab;\
    T ab4 = ab3 * ab;\
    adj_edge0 += adj_ret * ((T(6) * ax * bx * bx) / ab4);\
    adj_edge1 += adj_ret * ((T(6) * ax * ax * xb) / ab4);\
    adj_x     += adj_ret * ((T(6) * ax * bx     ) / ab3);\
}\
CUDA_CALLABLE inline T lerp(const T& a, const T& b, T t)\
{\
    return a*(T(1)-t) + b*t;\
}\
CUDA_CALLABLE inline void adj_lerp(const T& a, const T& b, T t, T& adj_a, T& adj_b, T& adj_t, const T& adj_ret)\
{\
    adj_a += adj_ret*(T(1)-t);\
    adj_b += adj_ret*t;\
    adj_t += b*adj_ret - a*adj_ret;\
}

DECLARE_INTERP_FUNCS(float16)
DECLARE_INTERP_FUNCS(float32)
DECLARE_INTERP_FUNCS(float64)

inline CUDA_CALLABLE void print(const str s)
{
    printf("%s\n", s);
}

inline CUDA_CALLABLE void print(int i)
{
    printf("%d\n", i);
}

inline CUDA_CALLABLE void print(short i)
{
    printf("%hd\n", i);
}

inline CUDA_CALLABLE void print(long i)
{
    printf("%ld\n", i);
}

inline CUDA_CALLABLE void print(long long i)
{
    printf("%lld\n", i);
}

inline CUDA_CALLABLE void print(unsigned i)
{
    printf("%u\n", i);
}

inline CUDA_CALLABLE void print(unsigned short i)
{
    printf("%hu\n", i);
}

inline CUDA_CALLABLE void print(unsigned long i)
{
    printf("%lu\n", i);
}

inline CUDA_CALLABLE void print(unsigned long long i)
{
    printf("%llu\n", i);
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void print(vec_t<Length, Type> v)
{
    for( unsigned i=0; i < Length; ++i )
    {
        printf("%g ", float(v[i]));
    }
    printf("\n");
}

template<typename Type>
inline CUDA_CALLABLE void print(quat_t<Type> i)
{
    printf("%g %g %g %g\n", float(i.x), float(i.y), float(i.z), float(i.w));
}

template<unsigned Rows,unsigned Cols,typename Type>
inline CUDA_CALLABLE void print(const mat_t<Rows,Cols,Type> &m)
{
    for( unsigned i=0; i< Rows; ++i )
    {
        for( unsigned j=0; j< Cols; ++j )
        {
            printf("%g ",float(m.data[i][j]));
        }
        printf("\n");
    }
}

template<typename Type>
inline CUDA_CALLABLE void print(transform_t<Type> t)
{
    printf("(%g %g %g) (%g %g %g %g)\n", float(t.p[0]), float(t.p[1]), float(t.p[2]), float(t.q.x), float(t.q.y), float(t.q.z), float(t.q.w));
}

inline CUDA_CALLABLE void adj_print(int i, int adj_i) { printf("%d adj: %d\n", i, adj_i); }
inline CUDA_CALLABLE void adj_print(float f, float adj_f) { printf("%g adj: %g\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(short f, short adj_f) { printf("%hd adj: %hd\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(long f, long adj_f) { printf("%ld adj: %ld\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(long long f, long long adj_f) { printf("%lld adj: %lld\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(unsigned f, unsigned adj_f) { printf("%u adj: %u\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(unsigned short f, unsigned short adj_f) { printf("%hu adj: %hu\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(unsigned long f, unsigned long adj_f) { printf("%lu adj: %lu\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(unsigned long long f, unsigned long long adj_f) { printf("%llu adj: %llu\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(half h, half adj_h) { printf("%g adj: %g\n", half_to_float(h), half_to_float(adj_h)); }
inline CUDA_CALLABLE void adj_print(double f, double adj_f) { printf("%g adj: %g\n", f, adj_f); }

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_print(vec_t<Length, Type> v, vec_t<Length, Type>& adj_v) { printf("%g %g adj: %g %g \n", v[0], v[1], adj_v[0], adj_v[1]); }

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_print(mat_t<Rows, Cols, Type> m, mat_t<Rows, Cols, Type>& adj_m) { }

template<typename Type>
inline CUDA_CALLABLE void adj_print(quat_t<Type> q, quat_t<Type>& adj_q) { printf("%g %g %g %g adj: %g %g %g %g\n", q.x, q.y, q.z, q.w, adj_q.x, adj_q.y, adj_q.z, adj_q.w); }

template<typename Type>
inline CUDA_CALLABLE void adj_print(transform_t<Type> t, transform_t<Type>& adj_t) {}

inline CUDA_CALLABLE void adj_print(str t, str& adj_t) {}


// printf defined globally in crt.h
inline CUDA_CALLABLE void adj_printf(const char* fmt, ...) {}


template <typename T>
inline CUDA_CALLABLE void expect_eq(const T& actual, const T& expected)
{
    if (!(actual == expected))
    {
        printf("Error, expect_eq() failed:\n");
        printf("\t Expected: "); print(expected); 
        printf("\t Actual: "); print(actual);
    }
}

template <typename T>
inline CUDA_CALLABLE void adj_expect_eq(const T& a, const T& b, T& adj_a, T& adj_b)
{
    // nop
}

template <typename T>
inline CUDA_CALLABLE void expect_neq(const T& actual, const T& expected)
{
    if (actual == expected)
    {
        printf("Error, expect_neq() failed:\n");
        printf("\t Expected: "); print(expected); 
        printf("\t Actual: "); print(actual);
    }
}

template <typename T>
inline CUDA_CALLABLE void adj_expect_neq(const T& a, const T& b, T& adj_a, T& adj_b)
{
    // nop
}

template <typename T>
inline CUDA_CALLABLE void expect_near(const T& actual, const T& expected, const T& tolerance)
{
    if (abs(actual - expected) > tolerance)
    {
        printf("Error, expect_near() failed with tolerance "); print(tolerance);
        printf("\t Expected: "); print(expected); 
        printf("\t Actual: "); print(actual);
    }
}

inline CUDA_CALLABLE void expect_near(const vec3& actual, const vec3& expected, const float& tolerance)
{
    const float diff = max(max(abs(actual[0] - expected[0]), abs(actual[1] - expected[1])), abs(actual[2] - expected[2]));
    if (diff > tolerance)
    {
        printf("Error, expect_near() failed with tolerance "); print(tolerance);
        printf("\t Expected: "); print(expected); 
        printf("\t Actual: "); print(actual);
    }
}

template <typename T>
inline CUDA_CALLABLE void adj_expect_near(const T& actual, const T& expected, const T& tolerance, T& adj_actual, T& adj_expected, T& adj_tolerance)
{
    // nop
}

inline CUDA_CALLABLE void adj_expect_near(const vec3& actual, const vec3& expected, float tolerance, vec3& adj_actual, vec3& adj_expected, float adj_tolerance)
{
    // nop
}


} // namespace wp

// include array.h so we have the print, isfinite functions for the inner array types defined
#include "array.h"
#include "mesh.h"
#include "bvh.h" 
#include "svd.h"
#include "hashgrid.h"
#include "volume.h"
#include "range.h"
#include "rand.h"
#include "noise.h"
#include "matnn.h"
