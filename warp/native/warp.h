#pragma once

#if defined(__CUDACC__)

    #if _DEBUG
        #define check_cuda(code) { cuda_report_error(code, __FILE__, __LINE__); }
    #else
        #define check_cuda(code) code;
    #endif

#endif

// defines all crt + builtin types
#include "builtin.h"

// this is the core runtime API exposed on the DLL level
extern "C"
{
    WP_API int init();
    //WP_API void shutdown();

    WP_API void* alloc_host(size_t s);
    WP_API void* alloc_device(size_t s);

    WP_API void free_host(void* ptr);
    WP_API void free_device(void* ptr);

    // all memcpys are performed asynchronously
    WP_API void memcpy_h2h(void* dest, void* src, size_t n);
    WP_API void memcpy_h2d(void* dest, void* src, size_t n);
    WP_API void memcpy_d2h(void* dest, void* src, size_t n);
    WP_API void memcpy_d2d(void* dest, void* src, size_t n);

    // all memsets are performed asynchronously
    WP_API void memset_host(void* dest, int value, size_t n);
    WP_API void memset_device(void* dest, int value, size_t n);

    // create a user-accesible copy of the mesh, it is the 
    // users reponsibility to keep-alive the points/tris data for the duration of the mesh lifetime
	WP_API uint64_t mesh_create_host(wp::vec3* points, wp::vec3* velocities, int* tris, int num_points, int num_tris);
	WP_API void mesh_destroy_host(uint64_t id);
    WP_API void mesh_refit_host(uint64_t id);

	WP_API uint64_t mesh_create_device(wp::vec3* points, wp::vec3* velocities, int* tris, int num_points, int num_tris);
	WP_API void mesh_destroy_device(uint64_t id);
    WP_API void mesh_refit_device(uint64_t id);

    WP_API void array_inner_host(uint64_t a, uint64_t b, uint64_t out, int len);
    WP_API void array_sum_host(uint64_t a, uint64_t out, int len);

    WP_API void array_inner_device(uint64_t a, uint64_t b, uint64_t out, int len);
    WP_API void array_sum_device(uint64_t a, uint64_t out, int len);

    // ensures all device side operations have completed
    WP_API void synchronize();

    // return cudaError_t code
    WP_API uint64_t cuda_check_device();
    WP_API void cuda_report_error(int code, const char* file, int line);

    WP_API void cuda_acquire_context();
    WP_API void cuda_restore_context();
    WP_API void* cuda_get_context();
    WP_API void cuda_set_context(void* ctx);
    WP_API void* cuda_get_stream();
    WP_API const char* cuda_get_device_name();

    WP_API void cuda_graph_begin_capture();
    WP_API void* cuda_graph_end_capture();
    WP_API void cuda_graph_launch(void* graph);
    WP_API void cuda_graph_destroy(void* graph);

    WP_API size_t cuda_compile_program(const char* cuda_src, const char* include_dir, bool debug, bool verbose, const char* output_file);
    WP_API void* cuda_load_module(const char* ptx);
    WP_API void* cuda_get_kernel(void* module, const char* name);
    WP_API size_t cuda_launch_kernel(void* kernel, int dim, void** args);

} // extern "C"


