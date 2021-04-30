#pragma once

#include <math.h>
#include <float.h>

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#if _WIN32
#define OG_API __declspec(dllexport)
#else
#define OG_API
#endif

#if defined(CPU)
    #define CUDA_CALLABLE

#elif defined(CUDA)
    #define CUDA_CALLABLE __host__ __device__ 

    #include <cuda.h>
    #include <cuda_runtime_api.h>

    #if _DEBUG
        #define check_cuda(code) { check_cuda_impl(code, __FILE__, __LINE__); }
    #else
        #define check_cuda(code)
    #endif

    void check_cuda_impl(cudaError_t code, const char* file, int line)
    {
        if (code != cudaSuccess) 
        {
            printf("CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        }
    }

    void print_device()
    {
        int currentDevice;
        cudaError_t err = cudaGetDevice(&currentDevice);        

        cudaDeviceProp props;
        err = cudaGetDeviceProperties(&props, currentDevice);
        if (err != cudaSuccess)
            printf("CUDA error: %d\n", err);
        else
            printf("%s\n", props.name);
    }


#endif

#ifdef _WIN32
#define __restrict__ __restrict
#endif

#define FP_CHECK 0

#define OG_DEVICE_CPU -1
#define OG_DEVICE_CUDA = 0

namespace og
{

template <typename T>
CUDA_CALLABLE float cast_float(T x) { return (float)(x); }

template <typename T>
CUDA_CALLABLE void adj_cast_float(T x, T& adj_x, float adj_ret) { adj_x += adj_ret; }


// 64bit address for an array
typedef uint64_t array;

template <typename T>
CUDA_CALLABLE T cast(og::array addr)
{
    return (T)(addr);
}

// numeric types
typedef float float32;
typedef double float64;
typedef int64_t int64;
typedef int32_t int32;

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
inline CUDA_CALLABLE float leaky_min(float a, float b, float r) { return min(a, b); }
inline CUDA_CALLABLE float leaky_max(float a, float b, float r) { return max(a, b); }
inline CUDA_CALLABLE float clamp(float x, float a, float b) { return min(max(a, x), b); }
inline CUDA_CALLABLE float step(float x) { return x < 0.0f ? 1.0f : 0.0f; }
inline CUDA_CALLABLE float sign(float x) { return x < 0.0f ? -1.0f : 1.0f; }
inline CUDA_CALLABLE float abs(float x) { return fabsf(x); }
inline CUDA_CALLABLE float nonzero(float x) { return x == 0.0f ? 0.0f : 1.0f; }

inline CUDA_CALLABLE float acos(float x) { return acosf(min(max(x, -1.0f), 1.0f)); }
inline CUDA_CALLABLE float sin(float x) { return sinf(x); }
inline CUDA_CALLABLE float cos(float x) { return cosf(x); }
inline CUDA_CALLABLE float sqrt(float x) { return sqrtf(x); }

inline CUDA_CALLABLE void adj_mul(float a, float b, float& adj_a, float& adj_b, float adj_ret) { adj_a += b*adj_ret; adj_b += a*adj_ret; }
inline CUDA_CALLABLE void adj_div(float a, float b, float& adj_a, float& adj_b, float adj_ret) { adj_a += adj_ret/b; adj_b -= adj_ret*(a/b)/b; }
inline CUDA_CALLABLE void adj_add(float a, float b, float& adj_a, float& adj_b, float adj_ret) { adj_a += adj_ret; adj_b += adj_ret; }
inline CUDA_CALLABLE void adj_sub(float a, float b, float& adj_a, float& adj_b, float adj_ret) { adj_a += adj_ret; adj_b -= adj_ret; }


// inline CUDA_CALLABLE bool lt(float a, float b) { return a < b; }
// inline CUDA_CALLABLE bool gt(float a, float b) { return a > b; }
// inline CUDA_CALLABLE bool lte(float a, float b) { return a <= b; }
// inline CUDA_CALLABLE bool gte(float a, float b) { return a >= b; }
// inline CUDA_CALLABLE bool eq(float a, float b) { return a == b; }
// inline CUDA_CALLABLE bool neq(float a, float b) { return a != b; }

// inline CUDA_CALLABLE bool adj_lt(float a, float b, float & adj_a, float & adj_b, bool & adj_ret) { }
// inline CUDA_CALLABLE bool adj_gt(float a, float b, float & adj_a, float & adj_b, bool & adj_ret) {  }
// inline CUDA_CALLABLE bool adj_lte(float a, float b, float & adj_a, float & adj_b, bool & adj_ret) {  }
// inline CUDA_CALLABLE bool adj_gte(float a, float b, float & adj_a, float & adj_b, bool & adj_ret) {  }
// inline CUDA_CALLABLE bool adj_eq(float a, float b, float & adj_a, float & adj_b, bool & adj_ret) {  }
// inline CUDA_CALLABLE bool adj_neq(float a, float b, float & adj_a, float & adj_b, bool & adj_ret) {  }

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
        adj_x += adj_ret;
    else
        adj_x -= adj_ret;                
}

inline CUDA_CALLABLE void adj_acos(float x, float& adj_x, float adj_ret)
{
    float d = sqrtf(1.0f-x*x);
    if (d > 0.0f)
        adj_x -= (1.0f/d)*adj_ret;
}

inline CUDA_CALLABLE void adj_sin(float x, float& adj_x, float adj_ret)
{
    adj_x += cosf(x)*adj_ret;
}

inline CUDA_CALLABLE void adj_cos(float x, float& adj_x, float adj_ret)
{
    adj_x -= sinf(x)*adj_ret;
}

inline CUDA_CALLABLE void adj_sqrt(float x, float& adj_x, float adj_ret)
{
    adj_x += 0.5f*(1.0f/sqrtf(x))*adj_ret;
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



// some helpful operator overloads (just for C++ use, these are not adjointed)

template <typename T>
CUDA_CALLABLE T& operator += (T& a, const T& b) { a = add(a, b); return a; }

template <typename T>
CUDA_CALLABLE T& operator -= (T& a, const T& b) { a = sub(a, b); return a; }

template <typename T>
CUDA_CALLABLE T operator*(const T& a, float s) { return mul(a, s); }

template <typename T>
CUDA_CALLABLE T operator*(float s, const T& a) { return mul(a, s); }

template <typename T>
CUDA_CALLABLE T operator*(const T& a, const T& b) { return mul(a, b); }

template <typename T>
CUDA_CALLABLE T operator/(const T& a, float s) { return div(a, s); }

template <typename T>
CUDA_CALLABLE T operator+(const T& a, const T& b) { return add(a, b); }

template <typename T>
CUDA_CALLABLE T operator-(const T& a, const T& b) { return sub(a, b); }


// for single thread CPU only
static int s_threadIdx;

inline CUDA_CALLABLE int tid()
{
#ifdef CPU
    return s_threadIdx;
#elif defined(CUDA)
    return blockDim.x * blockIdx.x + threadIdx.x;
#endif
}


#include "vec2.h"
#include "vec3.h"
#include "vec4.h"
#include "mat22.h"
#include "mat33.h"
#include "matnn.h"
#include "quat.h"
#include "spatial.h"
#include "intersect.h"


//--------------

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


template<typename T>
inline CUDA_CALLABLE void atomic_add(T* buf, T value)
{
#if defined(CPU)
    buf[0] += value;
#elif defined(CUDA)
    atomicAdd(buf, value);
#endif
}

template<typename T>
inline CUDA_CALLABLE void atomic_add(T* buf, int index, T value)
{
    if (buf)
    {        
#if defined(CPU)
        buf[index] += value;        // does not need to be atomic if single-threaded
#elif defined(CUDA)
        atomic_add(buf + index, value);
#endif
    }
}

template<typename T>
inline CUDA_CALLABLE void atomic_sub(T* buf, int index, T value)
{
    if (buf)
    {
#ifdef CPU
        buf[index] -= value;        // does not need to be atomic if single-threaded
#elif defined(CUDA)
        atomic_add(buf + index, -value);
#endif
    }
}

template <typename T>
inline CUDA_CALLABLE void adj_load(T* buf, int index, T* adj_buf, int& adj_index, const T& adj_output)
{
    // allow NULL buffers for case where gradients are not required
    if (adj_buf) {
#ifdef CPU
        adj_buf[index] += adj_output;  // does not need to be atomic if single-threaded
#elif defined(CUDA)
        atomic_add(adj_buf, index, adj_output);
#endif

    }
}

template <typename T>
inline CUDA_CALLABLE void adj_store(T* buf, int index, T value, T* adj_buf, int& adj_index, T& adj_value)
{   
    adj_value += adj_buf[index]; // doesn't need to be atomic because it's used to load from a buffer onto the stack
}

template<typename T>
inline CUDA_CALLABLE void adj_atomic_add(T* buf, int index, T value, T* adj_buf, int& adj_index, T& adj_value)
{
    if (adj_buf) {  // cannot be atomic because used locally
        adj_value += adj_buf[index];
    }
}

template<typename T>
inline CUDA_CALLABLE void adj_atomic_sub(T* buf, int index, T value, T* adj_buf, int& adj_index, T& adj_value)
{
    if (adj_buf) { // cannot be atomic because used locally
        adj_value -= adj_buf[index];
    }
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

inline CUDA_CALLABLE void print(int i)
{
    printf("%d\n", i);
}

inline CUDA_CALLABLE void print(float i)
{
    printf("%f\n", i);
}

inline CUDA_CALLABLE void print(vec3 i)
{
    printf("%f %f %f\n", i.x, i.y, i.z);
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

inline CUDA_CALLABLE void print(spatial_transform t)
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
inline CUDA_CALLABLE void adj_print(vec3 i, vec3& adj_i) { printf("%f %f %f adj: %f %f %f \n", i.x, i.y, i.z, adj_i.x, adj_i.y, adj_i.z); }
inline CUDA_CALLABLE void adj_print(quat i, quat& adj_i) { }
inline CUDA_CALLABLE void adj_print(mat22 m, mat22& adj_m) { }
inline CUDA_CALLABLE void adj_print(mat33 m, mat33& adj_m) { }
inline CUDA_CALLABLE void adj_print(spatial_transform t, spatial_transform& adj_t) {}
inline CUDA_CALLABLE void adj_print(spatial_vector t, spatial_vector& adj_t) {}
inline CUDA_CALLABLE void adj_print(spatial_matrix t, spatial_matrix& adj_t) {}

} // namespace og



// this is the core runtime API exposed on the DLL level
extern "C"
{
    OG_API void init();
    OG_API void shutdown();

    OG_API void* alloc_host(size_t s);
    OG_API void* alloc_device(size_t s);

    OG_API void free_host(void* ptr);
    OG_API void free_device(void* ptr);

    // all memcpys are performed asynchronously
    OG_API void memcpy_h2h(void* dest, void* src, size_t n);
    OG_API void memcpy_h2d(void* dest, void* src, size_t n);
    OG_API void memcpy_d2h(void* dest, void* src, size_t n);
    OG_API void memcpy_d2d(void* dest, void* src, size_t n);

    // all memsets are performed asynchronously
    OG_API void memset_host(void* dest, int value, size_t n);
    OG_API void memset_device(void* dest, int value, size_t n);

    // create a user-accesible copy of the mesh, it is the 
    // users reponsibility to keep-alive the points/tris data for the duration of the mesh lifetime
	OG_API uint64_t mesh_create_host(og::vec3* points, int* tris, int num_points, int num_tris);
	OG_API void mesh_destroy_host(uint64_t id);
    OG_API void mesh_update_host(uint64_t id, og::vec3* points, int* tris, bool refit);

	OG_API uint64_t mesh_create_device(og::vec3* points, int* tris, int num_points, int num_tris);
	OG_API void mesh_destroy_device(uint64_t id);
    OG_API void mesh_update_device(uint64_t id, og::vec3* points, int* tris, bool refit);

    // ensures all device side operations have completed
    OG_API void synchronize();

}


#include "mesh.h"
#include "volume.h"
