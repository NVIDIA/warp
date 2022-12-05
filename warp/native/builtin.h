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

#if !defined(__CUDA_ARCH__)
    #if defined(_WIN32)
        #define WP_API __declspec(dllexport)
    #else
        #define WP_API __attribute__ ((visibility ("default")))
    #endif
#else
    #define WP_API
#endif

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
#else
#define FP_CHECK 0
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
typedef char* str;



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

    CUDA_CALLABLE inline half operator+=(const half& h)
    {
        half sum = half(float32(*this) + float32(h));
        this->u = sum.u;
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
inline CUDA_CALLABLE int mul(int a, int b) { return a*b; }
inline CUDA_CALLABLE int div(int a, int b) { return a/b; }
inline CUDA_CALLABLE int add(int a, int b) { return a+b; }
inline CUDA_CALLABLE int sub(int a, int b) { return a-b; }
inline CUDA_CALLABLE int mod(int a, int b) { return a%b; }
inline CUDA_CALLABLE int min(int a, int b) { return a<b?a:b; }
inline CUDA_CALLABLE int max(int a, int b) { return a>b?a:b; }
inline CUDA_CALLABLE int abs(int x) { return ::abs(x); }
inline CUDA_CALLABLE int sign(int x) { return x < 0 ? -1 : 1; }
inline CUDA_CALLABLE int clamp(int x, int a, int b) { return min(max(a, x), b); }
inline CUDA_CALLABLE int floordiv(int a, int b) { return a/b; }


inline CUDA_CALLABLE void adj_mul(int a, int b, int& adj_a, int& adj_b, int adj_ret) { }
inline CUDA_CALLABLE void adj_div(int a, int b, int& adj_a, int& adj_b, int adj_ret) { }
inline CUDA_CALLABLE void adj_add(int a, int b, int& adj_a, int& adj_b, int adj_ret) { }
inline CUDA_CALLABLE void adj_sub(int a, int b, int& adj_a, int& adj_b, int adj_ret) { }
inline CUDA_CALLABLE void adj_mod(int a, int b, int& adj_a, int& adj_b, int adj_ret) { }
inline CUDA_CALLABLE void adj_min(int a, int b, int& adj_a, int& adj_b, int adj_ret) { }
inline CUDA_CALLABLE void adj_max(int a, int b, int& adj_a, int& adj_b, int adj_ret) { }
inline CUDA_CALLABLE void adj_abs(int x, int adj_x, int& adj_ret) { }
inline CUDA_CALLABLE void adj_sign(int x, int adj_x, int& adj_ret) { }
inline CUDA_CALLABLE void adj_clamp(int x, int a, int b, int& adj_x, int& adj_a, int& adj_b, int adj_ret) { }
inline CUDA_CALLABLE void adj_floordiv(int a, int b, int& adj_a, int& adj_b, int adj_ret) { }

// basic ops for float types
inline CUDA_CALLABLE float mul(float a, float b) { return a*b; }
inline CUDA_CALLABLE float div(float a, float b)
{
#if FP_CHECK
    if (!isfinite(a) || !isfinite(b) || b == 0.0f)
    {
        printf("%s:%d div(%f, %f)\n", __FILE__, __LINE__, a, b);
        assert(0);
    }
#endif
    return a/b;
}
inline CUDA_CALLABLE float add(float a, float b) { return a+b; }
inline CUDA_CALLABLE float sub(float a, float b) { return a-b; }
inline CUDA_CALLABLE float min(float a, float b) { return a<b?a:b; }
inline CUDA_CALLABLE float max(float a, float b) { return a>b?a:b; }
inline CUDA_CALLABLE float mod(float a, float b)
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
inline CUDA_CALLABLE float log(float a)
{
#if FP_CHECK
    if (!isfinite(a) || a < 0.0f)
    {
        printf("%s:%d log(%f)\n", __FILE__, __LINE__, a);
        assert(0);
    }
#endif
    return logf(a);
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

    return log2f(a);    
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

    return log10f(a); 
}


inline CUDA_CALLABLE float exp(float a)
{
    float result = expf(a);
#if FP_CHECK
    if (!isfinite(a) || !isfinite(result))
    {
        printf("%s:%d exp(%f) = %f\n", __FILE__, __LINE__, a, result);
        assert(0);
    }
#endif
    return result;
}
inline CUDA_CALLABLE float pow(float a, float b)
{
    float result = powf(a, b);
#if FP_CHECK
    if (!isfinite(a) || !isfinite(b) || !isfinite(result))
    {
        printf("%s:%d pow(%f, %f) = %f\n", __FILE__, __LINE__, a, b, result);
        assert(0);
    }
#endif
    return result;
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

inline CUDA_CALLABLE float leaky_min(float a, float b, float r) { return min(a, b); }
inline CUDA_CALLABLE float leaky_max(float a, float b, float r) { return max(a, b); }
inline CUDA_CALLABLE float clamp(float x, float a, float b) { return min(max(a, x), b); }
inline CUDA_CALLABLE float step(float x) { return x < 0.0f ? 1.0f : 0.0f; }
inline CUDA_CALLABLE float sign(float x) { return x < 0.0f ? -1.0f : 1.0f; }
inline CUDA_CALLABLE float abs(float x) { return ::fabs(x); }
inline CUDA_CALLABLE float nonzero(float x) { return x == 0.0f ? 0.0f : 1.0f; }

inline CUDA_CALLABLE float acos(float x)
{
    return ::acos(min(max(x, -1.0f), 1.0f));
}
inline CUDA_CALLABLE float asin(float x)
{
    return ::asin(min(max(x, -1.0f), 1.0f));
}
inline CUDA_CALLABLE float atan(float x) { return ::atan(x); }
inline CUDA_CALLABLE float atan2(float y, float x) { return ::atan2(y, x); }
inline CUDA_CALLABLE float sin(float x) { return ::sin(x); }
inline CUDA_CALLABLE float cos(float x) { return ::cos(x); }
inline CUDA_CALLABLE float sqrt(float x)
{
#if FP_CHECK
    if (x < 0.0f)
    {
        printf("%s:%d sqrt(%f)\n", __FILE__, __LINE__, x);
        assert(0);
    }
#endif
    return ::sqrt(x);
}
inline CUDA_CALLABLE float tan(float x) { return ::tan(x); }
inline CUDA_CALLABLE float sinh(float x) { return ::sinhf(x);}
inline CUDA_CALLABLE float cosh(float x) { return ::coshf(x);}
inline CUDA_CALLABLE float tanh(float x) { return ::tanhf(x);}

inline CUDA_CALLABLE float round(float x) { return ::roundf(x); }
inline CUDA_CALLABLE float rint(float x) { return ::rintf(x); }
inline CUDA_CALLABLE float trunc(float x) { return ::truncf(x); }
inline CUDA_CALLABLE float floor(float x) { return ::floorf(x); }
inline CUDA_CALLABLE float ceil(float x) { return ::ceilf(x); }

inline CUDA_CALLABLE void adj_mul(float a, float b, float& adj_a, float& adj_b, float adj_ret) { adj_a += b*adj_ret; adj_b += a*adj_ret; }
inline CUDA_CALLABLE void adj_div(float a, float b, float& adj_a, float& adj_b, float adj_ret)
{
    adj_a += adj_ret/b;
    adj_b -= adj_ret*(a/b)/b;
#if FP_CHECK
    if (!isfinite(adj_a) || !isfinite(adj_b))
    {
        printf("%s:%d - adj_div(%f, %f, %f, %f, %f)\n", __FILE__, __LINE__, a, b, adj_a, adj_b, adj_ret);
        assert(0);
    }
#endif
}
inline CUDA_CALLABLE void adj_add(float a, float b, float& adj_a, float& adj_b, float adj_ret) { adj_a += adj_ret; adj_b += adj_ret; }
inline CUDA_CALLABLE void adj_sub(float a, float b, float& adj_a, float& adj_b, float adj_ret) { adj_a += adj_ret; adj_b -= adj_ret; }
inline CUDA_CALLABLE void adj_mod(float a, float b, float& adj_a, float& adj_b, float adj_ret)
{
    printf("adj_mod not implemented for floating point types\n");
}
inline CUDA_CALLABLE void adj_log(float a, float& adj_a, float adj_ret)
{
    adj_a += (1.f/a)*adj_ret;
#if FP_CHECK
    if (!isfinite(adj_a))
    {
        printf("%s:%d - adj_log(%f, %f, %f)\n", __FILE__, __LINE__, a, adj_a, adj_ret);
        assert(0);
    }
#endif
}

inline CUDA_CALLABLE void adj_log2(float a, float& adj_a, float adj_ret) 
{ 
    adj_a += (1.f/a)*(1.f/log(2.f))*adj_ret; 
    
#if FP_CHECK
    if (!isfinite(adj_a))
    {
        printf("%s:%d - adj_log2(%f, %f, %f)\n", __FILE__, __LINE__, a, adj_a, adj_ret);
        assert(0);
    }
#endif    
}

inline CUDA_CALLABLE void adj_log10(float a, float& adj_a, float adj_ret)
{
    adj_a += (1.f/a)*(1.f/log(10.f))*adj_ret; 
    
#if FP_CHECK
    if (!isfinite(adj_a))
    {
        printf("%s:%d - adj_log10(%f, %f, %f)\n", __FILE__, __LINE__, a, adj_a, adj_ret);
        assert(0);
    }
#endif
}

inline CUDA_CALLABLE void adj_exp(float a, float& adj_a, float adj_ret) { adj_a += exp(a)*adj_ret; }
inline CUDA_CALLABLE void adj_pow(float a, float b, float& adj_a, float& adj_b, float adj_ret)
{ 
    adj_a += b*pow(a, b-1.f)*adj_ret;
    adj_b += log(a)*pow(a, b)*adj_ret;
#if FP_CHECK
    if (!isfinite(adj_a) || !isfinite(adj_b))
    {
        printf("%s:%d - adj_pow(%f, %f, %f, %f, %f)\n", __FILE__, __LINE__, a, b, adj_a, adj_b, adj_ret);
        assert(0);
    }
#endif
}
inline CUDA_CALLABLE void adj_floordiv(float a, float b, float& adj_a, float& adj_b, float adj_ret) { }

inline CUDA_CALLABLE void adj_min(float a, float b, float& adj_a, float& adj_b, float adj_ret)
{
    if (a < b)
        adj_a += adj_ret;
    else
        adj_b += adj_ret;
}

inline CUDA_CALLABLE void adj_max(float a, float b, float& adj_a, float& adj_b, float adj_ret)
{
    if (a > b)
        adj_a += adj_ret;
    else
        adj_b += adj_ret;
}

inline CUDA_CALLABLE void adj_leaky_min(float a, float b, float r, float& adj_a, float& adj_b, float& adj_r, float adj_ret)
{
    if (a < b)
        adj_a += adj_ret;
    else
    {
        adj_a += r*adj_ret;
        adj_b += adj_ret;
    }
}

inline CUDA_CALLABLE void adj_leaky_max(float a, float b, float r, float& adj_a, float& adj_b, float& adj_r, float adj_ret)
{
    if (a > b)
        adj_a += adj_ret;
    else
    {
        adj_a += r*adj_ret;
        adj_b += adj_ret;
    }
}

inline CUDA_CALLABLE void adj_clamp(float x, float a, float b, float& adj_x, float& adj_a, float& adj_b, float adj_ret)
{
    if (x < a)
        adj_a += adj_ret;
    else if (x > b)
        adj_b += adj_ret;
    else
        adj_x += adj_ret;
}

inline CUDA_CALLABLE void adj_step(float x, float& adj_x, float adj_ret)
{
    // nop
}

inline CUDA_CALLABLE void adj_nonzero(float x, float& adj_x, float adj_ret)
{
    // nop
}

inline CUDA_CALLABLE void adj_sign(float x, float& adj_x, float adj_ret)
{
    // nop
}

inline CUDA_CALLABLE void adj_abs(float x, float& adj_x, float adj_ret)
{
    if (x < 0.0f)
        adj_x -= adj_ret;
    else
        adj_x += adj_ret;                
}

inline CUDA_CALLABLE void adj_acos(float x, float& adj_x, float adj_ret)
{
    float d = sqrt(1.0f-x*x);
#if FP_CHECK
    adj_x -= (1.0f/d)*adj_ret;
    if (!isfinite(d) || !isfinite(adj_x))
    {
        printf("%s:%d - adj_acos(%f, %f, %f)\n", __FILE__, __LINE__, x, adj_x, adj_ret);        
        assert(0);
    }
#else    
    if (d > 0.0f)
        adj_x -= (1.0f/d)*adj_ret;
#endif
}

inline CUDA_CALLABLE void adj_asin(float x, float& adj_x, float adj_ret)
{
    float d = sqrt(1.0f-x*x);
#if FP_CHECK
    adj_x += (1.0f/d)*adj_ret;
    if (!isfinite(d) || !isfinite(adj_x))
    {
        printf("%s:%d - adj_asin(%f, %f, %f)\n", __FILE__, __LINE__, x, adj_x, adj_ret);   
        assert(0);
    }
#else    
    if (d > 0.0f)
        adj_x += (1.0f/d)*adj_ret;
#endif
}

inline CUDA_CALLABLE void adj_tan(float x, float& adj_x, float adj_ret)
{
    float cos_x = cos(x);
#if FP_CHECK
    adj_x += (1.0f/(cos_x*cos_x))*adj_ret;
    if (!isfinite(adj_x) || cos_x == 0.0f)
    {
        printf("%s:%d - adj_tan(%f, %f, %f)\n", __FILE__, __LINE__, x, adj_x, adj_ret);
        assert(0);
    }
#else    
    if (cos_x > 0.0f)
        adj_x += (1.0f/(cos_x*cos_x))*adj_ret;
#endif
}

inline CUDA_CALLABLE void adj_atan(float x, float& adj_x, float adj_ret)
{
    adj_x += (x*x + 1.0f)*adj_ret;
}

inline CUDA_CALLABLE void adj_atan2(float y, float x, float& adj_y, float& adj_x, float adj_ret)
{
    float d = x*x + y*y;
#if FP_CHECK
    adj_x -= y/d*adj_ret;
    adj_y += x/d*adj_ret;
    if (!isfinite(adj_x) || !isfinite(adj_y) || d == 0.0f)
    {
        printf("%s:%d - adj_atan2(%f, %f, %f, %f, %f)\n", __FILE__, __LINE__, y, x, adj_y, adj_x, adj_ret);
        assert(0);
    }
#else    
    if (d > 0.0f)
    {
        adj_x -= y/d*adj_ret;
        adj_y += x/d*adj_ret;
    }
#endif
}

inline CUDA_CALLABLE void adj_sin(float x, float& adj_x, float adj_ret)
{
    adj_x += cos(x)*adj_ret;
}

inline CUDA_CALLABLE void adj_cos(float x, float& adj_x, float adj_ret)
{
    adj_x -= sin(x)*adj_ret;
}

inline CUDA_CALLABLE void adj_sinh(float x, float& adj_x, float adj_ret)
{
    adj_x += cosh(x)*adj_ret;
}

inline CUDA_CALLABLE void adj_cosh(float x, float& adj_x, float adj_ret)
{
    adj_x += sinh(x)*adj_ret;
}

inline CUDA_CALLABLE void adj_tanh(float x, float& adj_x, float adj_ret)
{
    float tanh_x = tanh(x);
    adj_x += (1.0f - tanh_x*tanh_x)*adj_ret;
}

inline CUDA_CALLABLE void adj_sqrt(float x, float& adj_x, float adj_ret)
{
    adj_x += 0.5f*(1.0f/sqrt(x))*adj_ret;
#if FP_CHECK    
    if (!isfinite(adj_x))
    {
        printf("%s:%d - adj_sqrt(%f, %f, %f)\n", __FILE__, __LINE__, x, adj_x, adj_ret);
        assert(0);
    }
#endif
}

inline CUDA_CALLABLE void adj_round(float x, float& adj_x, float adj_ret)
{
    // nop
}

inline CUDA_CALLABLE void adj_rint(float x, float& adj_x, float adj_ret)
{
    // nop
}

inline CUDA_CALLABLE void adj_trunc(float x, float& adj_x, float adj_ret)
{
    // nop
}

inline CUDA_CALLABLE void adj_floor(float x, float& adj_x, float adj_ret)
{
    // nop
}

inline CUDA_CALLABLE void adj_ceil(float x, float& adj_x, float adj_ret)
{
    // nop
}

// basic arithmetic for fp16, this is only called for adjoint accumulation
inline CUDA_CALLABLE half add(half a, half b) { return half(float(a)+float(b)); }


template <typename T>
CUDA_CALLABLE inline T select(bool cond, const T& a, const T& b) { return cond?b:a; }

template <typename T>
CUDA_CALLABLE inline void adj_select(bool cond, const T& a, const T& b, bool& adj_cond, T& adj_a, T& adj_b, const T& adj_ret)
{
    if (cond)
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
    adj_dest = 0.0;
}


// some helpful operator overloads (just for C++ use, these are not adjointed)

template <typename T>
CUDA_CALLABLE inline T& operator += (T& a, const T& b) { a = add(a, b); return a; }

template <typename T>
CUDA_CALLABLE inline T& operator -= (T& a, const T& b) { a = sub(a, b); return a; }

template <typename T>
CUDA_CALLABLE inline T operator*(const T& a, float s) { return mul(a, s); }

template <typename T>
CUDA_CALLABLE inline T operator*(float s, const T& a) { return mul(a, s); }

template <typename T>
CUDA_CALLABLE inline T operator/(const T& a, float s) { return div(a, s); }

template <typename T>
CUDA_CALLABLE inline T operator+(const T& a, const T& b) { return add(a, b); }

template <typename T>
CUDA_CALLABLE inline T operator-(const T& a, const T& b) { return sub(a, b); }

// unary negation implementated as negative multiply, not sure the fp implications of this
// may be better as 0.0 - x?
template <typename T>
CUDA_CALLABLE inline T neg(const T& x) { return T(0.0) - x; }
template <typename T>
CUDA_CALLABLE inline void adj_neg(const T& x, T& adj_x, const T& adj_ret) { adj_x += T(-adj_ret); }

// unary boolean negation
CUDA_CALLABLE inline bool unot(const bool& b) { return !b; }
CUDA_CALLABLE inline void adj_unot(const bool& b, bool& adj_b, const bool& adj_ret) { }

const int LAUNCH_MAX_DIMS = 4;   // should match types.py

struct launch_bounds_t
{
    int shape[LAUNCH_MAX_DIMS];  // size of each dimension
    int ndim;                   // number of valid dimension
    int size;                   // total number of threads
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
static int s_threadIdx;

void set_launch_bounds(const launch_bounds_t& b)
{
    s_launchBounds = b;
}
#endif



inline CUDA_CALLABLE int tid()
{
#ifdef __CUDACC__
    return blockDim.x * blockIdx.x + threadIdx.x;
#else
    return s_threadIdx;
#endif
}

inline CUDA_CALLABLE_DEVICE void tid(int& i, int& j)
{
    const int index = tid();

    const int n = s_launchBounds.shape[1];

    // convert to work item
    i = index/n;
    j = index%n;
}

inline CUDA_CALLABLE_DEVICE void tid(int& i, int& j, int& k)
{
    const int index = tid();

    const int n = s_launchBounds.shape[1];
    const int o = s_launchBounds.shape[2];

    // convert to work item
    i = index/(n*o);
    j = index%(n*o)/o;
    k = index%o;
}

inline CUDA_CALLABLE_DEVICE void tid(int& i, int& j, int& k, int& l)
{
    const int index = tid();

    const int n = s_launchBounds.shape[1];
    const int o = s_launchBounds.shape[2];
    const int p = s_launchBounds.shape[3];

    // convert to work item
    i = index/(n*o*p);
    j = index%(n*o*p)/(o*p);
    k = index%(o*p)/p;
    l = index%p;
}

template<typename T>
inline CUDA_CALLABLE T atomic_add(T* buf, T value)
{
#if defined(WP_CPU)
    T old = buf[0];
    buf[0] += value;
    return old;
#elif defined(WP_CUDA)
    return atomicAdd(buf, value);
#endif
}

template<>
inline CUDA_CALLABLE float16 atomic_add(float16* buf, float16 value)
{
#if defined(WP_CPU)
    float16 old = buf[0];
    buf[0] += value;
    return old;
#elif defined(WP_CUDA)
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

#endif // defined(WP_CUDA)

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

inline bool CUDA_CALLABLE isfinite(float x)
{
    return ::isfinite(x);
}

} // namespace wp

#include "vec2.h"
#include "vec3.h"
#include "vec4.h"
#include "mat22.h"
#include "mat33.h"
#include "quat.h"
#include "mat44.h"
#include "spatial.h"
#include "intersect.h"
#include "intersect_adj.h"

//--------------
namespace wp
{


// define scalar multiplication in reverse order i.e.: s*M, individual types just implement M*s
template <typename T>
T mul(float s, const T& x) { return mul(x, s); }

template <typename T>
void adj_mul(float s, const T& x, float& adj_s, T& adj_x, const T& adj_ret) { adj_mul(x, s, adj_x, adj_s, adj_ret); }


// dot for scalar types just to make some templates compile for scalar/vector
inline CUDA_CALLABLE float dot(float a, float b) { return mul(a, b); }
inline CUDA_CALLABLE void adj_dot(float a, float b, float& adj_a, float& adj_b, float adj_ret) { adj_mul(a, b, adj_a, adj_b, adj_ret); }
inline CUDA_CALLABLE float tensordot(float a, float b) { return mul(a, b); }


template <typename T>
CUDA_CALLABLE inline T lerp(const T& a, const T& b, float t)
{
    return a*(1.0-t) + b*t;
}

template <typename T>
CUDA_CALLABLE inline void adj_lerp(const T& a, const T& b, float t, T& adj_a, T& adj_b, float& adj_t, const T& adj_ret)
{
    adj_a += adj_ret*(1.0-t);
    adj_b += adj_ret*t;
    adj_t += tensordot(b, adj_ret) - tensordot(a, adj_ret);
}

CUDA_CALLABLE inline float smoothstep(float edge0, float edge1, float x)
{
    // remap x from the range [edge0, edge1] to [0, 1]
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

CUDA_CALLABLE inline void adj_smoothstep(float edge0, float edge1, float x, float& adj_edge0, float& adj_edge1, float& adj_x, float adj_ret)
{
    float ab = edge0 - edge1;
    float ax = edge0 - x;
    float bx = edge1 - x;
    float xb = x - edge1;

    if (bx / ab >= 0 || ax / ab <= 0)
    {
        return;
    }

    float ab3 = ab * ab * ab;
    float ab4 = ab3 * ab;
    adj_edge0 += adj_ret * ((6 * ax * bx * bx) / ab4);
    adj_edge1 += adj_ret * ((6 * ax * ax * xb) / ab4);
    adj_x     += adj_ret * ((6 * ax * bx     ) / ab3);
}

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

inline CUDA_CALLABLE void print(vec2 v)
{
    printf("%g %g\n", v.x, v.y);
}

inline CUDA_CALLABLE void print(vec3 v)
{
    printf("%g %g %g\n", v.x, v.y, v.z);
}

inline CUDA_CALLABLE void print(vec4 v)
{
    printf("%g %g %g %g\n", v.x, v.y, v.z, v.w);
}

inline CUDA_CALLABLE void print(quat i)
{
    printf("%g %g %g %g\n", i.x, i.y, i.z, i.w);
}

inline CUDA_CALLABLE void print(mat22 m)
{
    printf("%g %g\n%g %g\n", m.data[0][0], m.data[0][1], 
                             m.data[1][0], m.data[1][1]);
}

inline CUDA_CALLABLE void print(mat33 m)
{
    printf("%g %g %g\n%g %g %g\n%g %g %g\n", m.data[0][0], m.data[0][1], m.data[0][2], 
                                             m.data[1][0], m.data[1][1], m.data[1][2], 
                                             m.data[2][0], m.data[2][1], m.data[2][2]);
}

inline CUDA_CALLABLE void print(mat44 m)
{
    printf("%g %g %g %g\n%g %g %g %g\n%g %g %g %g\n%g %g %g %g\n", m.data[0][0], m.data[0][1], m.data[0][2], m.data[0][3],
                                                                   m.data[1][0], m.data[1][1], m.data[1][2], m.data[1][3],
                                                                   m.data[2][0], m.data[2][1], m.data[2][2], m.data[2][3],
                                                                   m.data[3][0], m.data[3][1], m.data[3][2], m.data[3][3]);
}

inline CUDA_CALLABLE void print(transform t)
{
    printf("(%g %g %g) (%g %g %g %g)\n", t.p.x, t.p.y, t.p.z, t.q.x, t.q.y, t.q.z, t.q.w);
}

inline CUDA_CALLABLE void print(spatial_vector v)
{
    printf("(%g %g %g) (%g %g %g)\n", v.w.x, v.w.y, v.w.z, v.v.x, v.v.y, v.v.z);
}

inline CUDA_CALLABLE void print(spatial_matrix m)
{
    printf("%g %g %g %g %g %g\n"
           "%g %g %g %g %g %g\n"
           "%g %g %g %g %g %g\n"
           "%g %g %g %g %g %g\n"
           "%g %g %g %g %g %g\n"
           "%g %g %g %g %g %g\n", 
           m.data[0][0], m.data[0][1], m.data[0][2],  m.data[0][3], m.data[0][4], m.data[0][5], 
           m.data[1][0], m.data[1][1], m.data[1][2],  m.data[1][3], m.data[1][4], m.data[1][5], 
           m.data[2][0], m.data[2][1], m.data[2][2],  m.data[2][3], m.data[2][4], m.data[2][5], 
           m.data[3][0], m.data[3][1], m.data[3][2],  m.data[3][3], m.data[3][4], m.data[3][5], 
           m.data[4][0], m.data[4][1], m.data[4][2],  m.data[4][3], m.data[4][4], m.data[4][5], 
           m.data[5][0], m.data[5][1], m.data[5][2],  m.data[5][3], m.data[5][4], m.data[5][5]);
}


inline CUDA_CALLABLE void adj_print(int i, int& adj_i) { printf("%d adj: %d\n", i, adj_i); }
inline CUDA_CALLABLE void adj_print(float f, float& adj_f) { printf("%g adj: %g\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(short f, short& adj_f) { printf("%hd adj: %hd\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(long f, long& adj_f) { printf("%ld adj: %ld\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(long long f, long long& adj_f) { printf("%lld adj: %lld\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(unsigned f, unsigned& adj_f) { printf("%u adj: %u\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(unsigned short f, unsigned short& adj_f) { printf("%hu adj: %hu\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(unsigned long f, unsigned long& adj_f) { printf("%lu adj: %lu\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(unsigned long long f, unsigned long long& adj_f) { printf("%llu adj: %llu\n", f, adj_f); }
inline CUDA_CALLABLE void adj_print(half h, half& adj_h) { printf("%g adj: %g\n", half_to_float(h), half_to_float(adj_h)); }
inline CUDA_CALLABLE void adj_print(vec2 v, vec2& adj_v) { printf("%g %g adj: %g %g \n", v.x, v.y, adj_v.x, adj_v.y); }
inline CUDA_CALLABLE void adj_print(vec3 v, vec3& adj_v) { printf("%g %g %g adj: %g %g %g \n", v.x, v.y, v.z, adj_v.x, adj_v.y, adj_v.z); }
inline CUDA_CALLABLE void adj_print(vec4 v, vec4& adj_v) { printf("%g %g %g %g adj: %g %g %g %g\n", v.x, v.y, v.z, v.w, adj_v.x, adj_v.y, adj_v.z, adj_v.w); }
inline CUDA_CALLABLE void adj_print(quat q, quat& adj_q) { printf("%g %g %g %g adj: %g %g %g %g\n", q.x, q.y, q.z, q.w, adj_q.x, adj_q.y, adj_q.z, adj_q.w); }
inline CUDA_CALLABLE void adj_print(mat22 m, mat22& adj_m) { }
inline CUDA_CALLABLE void adj_print(mat33 m, mat33& adj_m) { }
inline CUDA_CALLABLE void adj_print(mat44 m, mat44& adj_m) { }
inline CUDA_CALLABLE void adj_print(transform t, transform& adj_t) {}
inline CUDA_CALLABLE void adj_print(spatial_vector t, spatial_vector& adj_t) {}
inline CUDA_CALLABLE void adj_print(spatial_matrix t, spatial_matrix& adj_t) {}
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
inline CUDA_CALLABLE void expect_near(const T& actual, const T& expected, const float& tolerance)
{
    if (abs(actual - expected) > tolerance)
    {
        printf("Error, expect_near() failed with torerance "); print(tolerance);
        printf("\t Expected: "); print(expected); 
        printf("\t Actual: "); print(actual);
    }
}

template <>
inline CUDA_CALLABLE void expect_near<vec3>(const vec3& actual, const vec3& expected, const float& tolerance)
{
    const float diff = max(max(abs(actual.x - expected.x), abs(actual.y - expected.y)), abs(actual.z - expected.z));
    if (diff > tolerance)
    {
        printf("Error, expect_near() failed with torerance "); print(tolerance);
        printf("\t Expected: "); print(expected); 
        printf("\t Actual: "); print(actual);
    }
}

template <typename T>
inline CUDA_CALLABLE void adj_expect_near(const T& actual, const T& expected, const float& tolerance, T& adj_actual, T& adj_expected, float& adj_tolerance)
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
