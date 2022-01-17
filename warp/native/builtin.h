#pragma once

// All built-in types and functions. To be compatible with runtime NVRTC compilation
// this header must be independently compilable (without external headers)
// to achieve this we redefine a subset of CRT functions (printf, pow, sin, cos, etc)

#include "crt.h"

#if _WIN32
#define WP_API __declspec(dllexport)
#else
#define WP_API
#endif

#ifdef _WIN32
#define __restrict__ __restrict
#endif

#if !defined(__CUDACC__)
    #define CUDA_CALLABLE

#else
    #define CUDA_CALLABLE __host__ __device__ 
#endif


#define FP_CHECK 0


namespace wp
{

template <typename T>
CUDA_CALLABLE float cast_float(T x) { return (float)(x); }

template <typename T>
CUDA_CALLABLE int cast_int(T x) { return (int)(x); }

template <typename T>
CUDA_CALLABLE void adj_cast_float(T x, T& adj_x, float adj_ret) { adj_x += adj_ret; }

template <typename T>
CUDA_CALLABLE void adj_cast_int(T x, T& adj_x, int adj_ret) { adj_x += adj_ret; }


// 64bit address for an array
typedef uint64_t array;

template <typename T>
CUDA_CALLABLE T cast(wp::array addr)
{
    return (T)(addr);
}

// numeric types (used from generated kernels)
typedef float float32;
typedef double float64;

typedef int64_t int64;
typedef int32_t int32;

typedef uint64_t uint64;
typedef uint32_t uint32;

#define kEps 0.0f

// basic ops for integer types
inline CUDA_CALLABLE int mul(int a, int b) { return a*b; }
inline CUDA_CALLABLE int div(int a, int b) { return a/b; }
inline CUDA_CALLABLE int add(int a, int b) { return a+b; }
inline CUDA_CALLABLE int sub(int a, int b) { return a-b; }
inline CUDA_CALLABLE int mod(int a, int b) { return a % b; }
inline CUDA_CALLABLE int min(int a, int b) { return a<b?a:b; }
inline CUDA_CALLABLE int max(int a, int b) { return a>b?a:b; }
inline CUDA_CALLABLE int clamp(int x, int a, int b) { return min(max(a, x), b); }

inline CUDA_CALLABLE void adj_mul(int a, int b, int& adj_a, int& adj_b, int adj_ret) { }
inline CUDA_CALLABLE void adj_div(int a, int b, int& adj_a, int& adj_b, int adj_ret) { }
inline CUDA_CALLABLE void adj_add(int a, int b, int& adj_a, int& adj_b, int adj_ret) { }
inline CUDA_CALLABLE void adj_sub(int a, int b, int& adj_a, int& adj_b, int adj_ret) { }
inline CUDA_CALLABLE void adj_mod(int a, int b, int& adj_a, int& adj_b, int adj_ret) { }
inline CUDA_CALLABLE void adj_min(int a, int b, int& adj_a, int& adj_b, int adj_ret) { }
inline CUDA_CALLABLE void adj_max(int a, int b, int& adj_a, int& adj_b, int adj_ret) { }
inline CUDA_CALLABLE void adj_clamp(int x, int a, int b, int& adj_x, int& adj_a, int& adj_b, int adj_ret) { }

// basic ops for float types
inline CUDA_CALLABLE float mul(float a, float b) { return a*b; }
inline CUDA_CALLABLE float div(float a, float b) { return a/b; }
inline CUDA_CALLABLE float add(float a, float b) { return a+b; }
inline CUDA_CALLABLE float sub(float a, float b) { return a-b; }
inline CUDA_CALLABLE float min(float a, float b) { return a<b?a:b; }
inline CUDA_CALLABLE float max(float a, float b) { return a>b?a:b; }
inline CUDA_CALLABLE float mod(float a, float b) { return fmodf(a, b); }
inline CUDA_CALLABLE float log(float a) { return logf(a); }
inline CUDA_CALLABLE float exp(float a) { return expf(a); }
inline CUDA_CALLABLE float pow(float a, float b) { return powf(a, b); }

inline CUDA_CALLABLE float leaky_min(float a, float b, float r) { return min(a, b); }
inline CUDA_CALLABLE float leaky_max(float a, float b, float r) { return max(a, b); }
inline CUDA_CALLABLE float clamp(float x, float a, float b) { return min(max(a, x), b); }
inline CUDA_CALLABLE float step(float x) { return x < 0.0f ? 1.0f : 0.0f; }
inline CUDA_CALLABLE float sign(float x) { return x < 0.0f ? -1.0f : 1.0f; }
inline CUDA_CALLABLE float abs(float x) { return ::fabs(x); }
inline CUDA_CALLABLE float nonzero(float x) { return x == 0.0f ? 0.0f : 1.0f; }

inline CUDA_CALLABLE float acos(float x) { return ::acos(min(max(x, -1.0f), 1.0f)); }
inline CUDA_CALLABLE float asin(float x) { return ::asin(min(max(x, -1.0f), 1.0f)); }
inline CUDA_CALLABLE float atan(float x) { return ::atan(x); }
inline CUDA_CALLABLE float atan2(float y, float x) { return ::atan2(y, x); }
inline CUDA_CALLABLE float sin(float x) { return ::sin(x); }
inline CUDA_CALLABLE float cos(float x) { return ::cos(x); }
inline CUDA_CALLABLE float sqrt(float x) { return ::sqrt(x); }
inline CUDA_CALLABLE float tan(float x) { return ::tan(x); }

inline CUDA_CALLABLE void adj_mul(float a, float b, float& adj_a, float& adj_b, float adj_ret) { adj_a += b*adj_ret; adj_b += a*adj_ret; }
inline CUDA_CALLABLE void adj_div(float a, float b, float& adj_a, float& adj_b, float adj_ret) { adj_a += adj_ret/b; adj_b -= adj_ret*(a/b)/b; }
inline CUDA_CALLABLE void adj_add(float a, float b, float& adj_a, float& adj_b, float adj_ret) { adj_a += adj_ret; adj_b += adj_ret; }
inline CUDA_CALLABLE void adj_sub(float a, float b, float& adj_a, float& adj_b, float adj_ret) { adj_a += adj_ret; adj_b -= adj_ret; }
inline CUDA_CALLABLE void adj_mod(float a, float b, float& adj_a, float& adj_b, float adj_ret)
{
    printf("adj_mod not implemented for floating point types\n");
}
inline CUDA_CALLABLE void adj_log(float a, float& adj_a, float adj_ret) { adj_a += (1.f/a)*adj_ret; }
inline CUDA_CALLABLE void adj_exp(float a, float& adj_a, float adj_ret) { adj_a += exp(a)*adj_ret; }
inline CUDA_CALLABLE void adj_pow(float a, float b, float& adj_a, float& adj_b, float adj_ret) { adj_a += b*pow(a, b-1.f)*adj_ret; adj_b += log(a)*pow(a, b)*adj_ret; }


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
    if (d > 0.0f)
        adj_x -= (1.0f/d)*adj_ret;
}

inline CUDA_CALLABLE void adj_asin(float x, float& adj_x, float adj_ret)
{
    float d = sqrt(1.0f-x*x);
    if (d > 0.0f)
        adj_x += (1.0f/d)*adj_ret;
}

inline CUDA_CALLABLE void adj_tan(float x, float& adj_x, float adj_ret)
{
    float cos_x = cos(x);
    adj_x += (1.0f/(cos_x*cos_x))*adj_ret;
}

inline CUDA_CALLABLE void adj_atan(float x, float& adj_x, float adj_ret)
{
    adj_x += (x*x + 1.0f)*adj_ret;
}

inline CUDA_CALLABLE void adj_atan2(float y, float x, float& adj_y, float& adj_x, float adj_ret)
{
    printf("arctan2 adjoint not implemented");
}


inline CUDA_CALLABLE void adj_sin(float x, float& adj_x, float adj_ret)
{
    adj_x += cos(x)*adj_ret;
}

inline CUDA_CALLABLE void adj_cos(float x, float& adj_x, float adj_ret)
{
    adj_x -= sin(x)*adj_ret;
}

inline CUDA_CALLABLE void adj_sqrt(float x, float& adj_x, float adj_ret)
{
    adj_x += 0.5f*(1.0f/sqrt(x))*adj_ret;
}


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


// for single thread CPU only
static int s_threadIdx;

inline CUDA_CALLABLE int tid()
{
#ifdef __CUDACC__
    return blockDim.x * blockIdx.x + threadIdx.x;
#else
    return s_threadIdx;
#endif
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


} // namespace wp

#include "vec2.h"
#include "vec3.h"
#include "vec4.h"
#include "mat22.h"
#include "mat33.h"
#include "mat44.h"
#include "matnn.h"
#include "quat.h"
#include "spatial.h"
#include "intersect.h"
#include "mesh.h"
#include "svd.h"
#include "hashgrid.h"
#include "rand.h"

//--------------
namespace wp
{

template<typename T>
inline CUDA_CALLABLE T atomic_add(T* buf, int index, T value)
{
    return atomic_add(buf + index, value);
}

template<typename T>
inline CUDA_CALLABLE T atomic_sub(T* buf, int index, T value)
{
    return atomic_add(buf + index, -value);
}


template<typename T>
inline CUDA_CALLABLE T load(T* buf, int index)
{
    assert(buf);
    return buf[index];
}

template<typename T>
inline CUDA_CALLABLE void store(T* buf, int index, T value)
{
    // allow NULL buffers for case where gradients are not required
    if (buf)
    {
        buf[index] = value;
    }
}


template <typename T>
inline CUDA_CALLABLE void adj_load(T* buf, int index, T* adj_buf, int& adj_index, const T& adj_output)
{
    // allow NULL buffers for case where gradients are not required
    if (adj_buf) {

#if defined(WP_CPU)
        adj_buf[index] += adj_output;  // does not need to be atomic if single-threaded
#elif defined(WP_CUDA)
        atomic_add(adj_buf, index, adj_output);
#endif

    }
}

template <typename T>
inline CUDA_CALLABLE void adj_store(T* buf, int index, T value, T* adj_buf, int& adj_index, T& adj_value)
{   
    if (adj_buf)
        adj_value += adj_buf[index];
}

template<typename T>
inline CUDA_CALLABLE void adj_atomic_add(T* buf, int index, T value, T* adj_buf, int& adj_index, T& adj_value, const T& adj_ret)
{
    if (adj_buf) 
        adj_value += adj_buf[index];
}

template<typename T>
inline CUDA_CALLABLE void adj_atomic_sub(T* buf, int index, T value, T* adj_buf, int& adj_index, T& adj_value, const T& adj_ret)
{
    if (adj_buf) 
        adj_value -= adj_buf[index];
}

//-------------------------
// Texture methods

inline CUDA_CALLABLE float sdf_sample(vec3 x)
{
    return 0.0;
}

inline CUDA_CALLABLE vec3 sdf_grad(vec3 x)
{
    return vec3();
}

inline CUDA_CALLABLE void adj_sdf_sample(vec3 x, vec3& adj_x, float adj_ret)
{

}

inline CUDA_CALLABLE void adj_sdf_grad(vec3 x, vec3& adj_x, vec3& adj_ret)
{

}

// // based on https://arxiv.org/abs/2004.06278
// inline CUDA_CALLABLE int rand_int(int count)
// {
//     return clock()
// }

// inline CUDA_CALLABLE float rand_float(int count)
// {

// }

inline CUDA_CALLABLE void print(int i)
{
    printf("%d\n", i);
}

inline CUDA_CALLABLE void print(float f)
{
    printf("%f\n", f);
}

inline CUDA_CALLABLE void print(vec2 v)
{
    printf("%f %f\n", v.x, v.y);
}

inline CUDA_CALLABLE void print(vec3 v)
{
    printf("%f %f %f\n", v.x, v.y, v.z);
}

inline CUDA_CALLABLE void print(vec4 v)
{
    printf("%f %f %f %f\n", v.x, v.y, v.z, v.w);
}

inline CUDA_CALLABLE void print(quat i)
{
    printf("%f %f %f %f\n", i.x, i.y, i.z, i.w);
}

inline CUDA_CALLABLE void print(mat22 m)
{
    printf("%f %f\n%f %f\n", m.data[0][0], m.data[0][1], 
                             m.data[1][0], m.data[1][1]);
}

inline CUDA_CALLABLE void print(mat33 m)
{
    printf("%f %f %f\n%f %f %f\n%f %f %f\n", m.data[0][0], m.data[0][1], m.data[0][2], 
                                             m.data[1][0], m.data[1][1], m.data[1][2], 
                                             m.data[2][0], m.data[2][1], m.data[2][2]);
}

inline CUDA_CALLABLE void print(mat44 m)
{
    printf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n", m.data[0][0], m.data[0][1], m.data[0][2], m.data[0][3],
                                                                   m.data[1][0], m.data[1][1], m.data[1][2], m.data[1][3],
                                                                   m.data[2][0], m.data[2][1], m.data[2][2], m.data[2][3],
                                                                   m.data[3][0], m.data[3][1], m.data[3][2], m.data[3][3]);
}

inline CUDA_CALLABLE void print(transform t)
{
    printf("(%f %f %f) (%f %f %f %f)\n", t.p.x, t.p.y, t.p.z, t.q.x, t.q.y, t.q.z, t.q.w);
}

inline CUDA_CALLABLE void print(spatial_vector v)
{
    printf("(%f %f %f) (%f %f %f)\n", v.w.x, v.w.y, v.w.z, v.v.x, v.v.y, v.v.z);
}

inline CUDA_CALLABLE void print(spatial_matrix m)
{
    printf("%f %f %f %f %f %f\n"
           "%f %f %f %f %f %f\n"
           "%f %f %f %f %f %f\n"
           "%f %f %f %f %f %f\n"
           "%f %f %f %f %f %f\n"
           "%f %f %f %f %f %f\n", 
           m.data[0][0], m.data[0][1], m.data[0][2],  m.data[0][3], m.data[0][4], m.data[0][5], 
           m.data[1][0], m.data[1][1], m.data[1][2],  m.data[1][3], m.data[1][4], m.data[1][5], 
           m.data[2][0], m.data[2][1], m.data[2][2],  m.data[2][3], m.data[2][4], m.data[2][5], 
           m.data[3][0], m.data[3][1], m.data[3][2],  m.data[3][3], m.data[3][4], m.data[3][5], 
           m.data[4][0], m.data[4][1], m.data[4][2],  m.data[4][3], m.data[4][4], m.data[4][5], 
           m.data[5][0], m.data[5][1], m.data[5][2],  m.data[5][3], m.data[5][4], m.data[5][5]);
}


inline CUDA_CALLABLE void adj_print(int i, int& adj_i) { printf("%d adj: %d\n", i, adj_i); }
inline CUDA_CALLABLE void adj_print(float i, float& adj_i) { printf("%f adj: %f\n", i, adj_i); }
inline CUDA_CALLABLE void adj_print(vec2 v, vec2& adj_v) { printf("%f %f adj: %f %f \n", v.x, v.y, adj_v.x, adj_v.y); }
inline CUDA_CALLABLE void adj_print(vec3 v, vec3& adj_v) { printf("%f %f %f adj: %f %f %f \n", v.x, v.y, v.z, adj_v.x, adj_v.y, adj_v.z); }
inline CUDA_CALLABLE void adj_print(vec4 v, vec4& adj_v) { printf("%f %f %f %f adj: %f %f %f %f\n", v.x, v.y, v.z, v.w, adj_v.x, adj_v.y, adj_v.z, adj_v.w); }
inline CUDA_CALLABLE void adj_print(quat q, quat& adj_q) { printf("%f %f %f %f adj: %f %f %f %f\n", q.x, q.y, q.z, q.w, adj_q.x, adj_q.y, adj_q.z, adj_q.w); }
inline CUDA_CALLABLE void adj_print(mat22 m, mat22& adj_m) { }
inline CUDA_CALLABLE void adj_print(mat33 m, mat33& adj_m) { }
inline CUDA_CALLABLE void adj_print(mat44 m, mat44& adj_m) { }
inline CUDA_CALLABLE void adj_print(transform t, transform& adj_t) {}
inline CUDA_CALLABLE void adj_print(spatial_vector t, spatial_vector& adj_t) {}
inline CUDA_CALLABLE void adj_print(spatial_matrix t, spatial_matrix& adj_t) {}


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


} // namespace wp

