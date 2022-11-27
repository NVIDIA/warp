/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

// defines all crt + builtin types
#include "builtin.h"

// this is the core runtime API exposed on the DLL level
extern "C"
{
    WP_API int init();
    //WP_API void shutdown();

    WP_API void* alloc_host(size_t s);
    WP_API void* alloc_pinned(size_t s);
    WP_API void* alloc_device(void* context, size_t s);

    WP_API void free_host(void* ptr);
    WP_API void free_pinned(void* ptr);
    WP_API void free_device(void* context, void* ptr);

    // all memcpys are performed asynchronously
    WP_API void memcpy_h2h(void* dest, void* src, size_t n);
    WP_API void memcpy_h2d(void* context, void* dest, void* src, size_t n);
    WP_API void memcpy_d2h(void* context, void* dest, void* src, size_t n);
    WP_API void memcpy_d2d(void* context, void* dest, void* src, size_t n);
    WP_API void memcpy_peer(void* context, void* dest, void* src, size_t n);

    // all memsets are performed asynchronously
    WP_API void memset_host(void* dest, int value, size_t n);
    WP_API void memset_device(void* context, void* dest, int value, size_t n);

	WP_API uint64_t bvh_create_host(wp::vec3* lowers, wp::vec3* uppers, int num_bounds);
	WP_API void bvh_destroy_host(uint64_t id);
    WP_API void bvh_refit_host(uint64_t id);

	WP_API uint64_t bvh_create_device(void* context, wp::vec3* lowers, wp::vec3* uppers, int num_bounds);
	WP_API void bvh_destroy_device(uint64_t id);
    WP_API void bvh_refit_device(uint64_t id);

    // create a user-accessible copy of the mesh, it is the 
    // users responsibility to keep-alive the points/tris data for the duration of the mesh lifetime
	WP_API uint64_t mesh_create_host(wp::vec3* points, wp::vec3* velocities, int* tris, int num_points, int num_tris);
	WP_API void mesh_destroy_host(uint64_t id);
    WP_API void mesh_refit_host(uint64_t id);

	WP_API uint64_t mesh_create_device(void* context, wp::vec3* points, wp::vec3* velocities, int* tris, int num_points, int num_tris);
	WP_API void mesh_destroy_device(uint64_t id);
    WP_API void mesh_refit_device(uint64_t id);

    WP_API uint64_t hash_grid_create_host(int dim_x, int dim_y, int dim_z);
    WP_API void hash_grid_reserve_host(uint64_t id, int num_points);
    WP_API void hash_grid_destroy_host(uint64_t id);
    WP_API void hash_grid_update_host(uint64_t id, float cell_width, const wp::vec3* positions, int num_points);

    WP_API uint64_t hash_grid_create_device(void* context, int dim_x, int dim_y, int dim_z);
    WP_API void hash_grid_reserve_device(uint64_t id, int num_points);
    WP_API void hash_grid_destroy_device(uint64_t id);
    WP_API void hash_grid_update_device(uint64_t id, float cell_width, const wp::vec3* positions, int num_points);

    WP_API uint64_t volume_create_host(void* buf, uint64_t size);
    WP_API void volume_get_buffer_info_host(uint64_t id, void** buf, uint64_t* size);
    WP_API void volume_get_tiles_host(uint64_t id, void** buf, uint64_t* size);
    WP_API void volume_destroy_host(uint64_t id);

    WP_API uint64_t volume_create_device(void* context, void* buf, uint64_t size);
    WP_API uint64_t volume_f_from_tiles_device(void* context, void* points, int num_points, float voxel_size, float bg_value, float tx, float ty, float tz, bool points_in_world_space);
    WP_API uint64_t volume_v_from_tiles_device(void* context, void* points, int num_points, float voxel_size, float bg_value_x, float bg_value_y, float bg_value_z, float tx, float ty, float tz, bool points_in_world_space);
    WP_API uint64_t volume_i_from_tiles_device(void* context, void* points, int num_points, float voxel_size, int bg_value, float tx, float ty, float tz, bool points_in_world_space);
    WP_API void volume_get_buffer_info_device(uint64_t id, void** buf, uint64_t* size);
    WP_API void volume_get_tiles_device(uint64_t id, void** buf, uint64_t* size);
    WP_API void volume_destroy_device(uint64_t id);

    WP_API void volume_get_voxel_size(uint64_t id, float* dx, float* dy, float* dz);
    
    WP_API uint64_t marching_cubes_create_device(void* context);
    WP_API void marching_cubes_destroy_device(uint64_t id);
    WP_API int marching_cubes_surface_device(uint64_t id, const float* field, int nx, int ny, int nz, float threshold, wp::vec3* verts, int* triangles, int max_verts, int max_tris, int* out_num_verts, int* out_num_tris);

    WP_API void array_inner_host(uint64_t a, uint64_t b, uint64_t out, int len);
    WP_API void array_sum_host(uint64_t a, uint64_t out, int len);

    WP_API void array_inner_device(uint64_t a, uint64_t b, uint64_t out, int len);
    WP_API void array_sum_device(uint64_t a, uint64_t out, int len);

    WP_API int cuda_driver_version();   // CUDA driver version
    WP_API int cuda_toolkit_version();  // CUDA Toolkit version used to build Warp

    WP_API int nvrtc_supported_arch_count();
    WP_API void nvrtc_supported_archs(int* archs);

    WP_API int cuda_device_get_count();
    WP_API void* cuda_device_primary_context_retain(int ordinal);
    WP_API void cuda_device_primary_context_release(int ordinal);
    WP_API const char* cuda_device_get_name(int ordinal);
    WP_API int cuda_device_get_arch(int ordinal);
    WP_API int cuda_device_is_uva(int ordinal);

    WP_API void* cuda_context_get_current();
    WP_API void cuda_context_set_current(void* context);
    WP_API void cuda_context_push_current(void* context);
    WP_API void cuda_context_pop_current();
    WP_API void* cuda_context_create(int device_ordinal);
    WP_API void cuda_context_destroy(void* context);
    WP_API int cuda_context_get_device_ordinal(void* context);
    WP_API int cuda_context_is_primary(void* context);
    WP_API void* cuda_context_get_stream(void* context);
    WP_API void cuda_context_set_stream(void* context, void* stream);
    WP_API int cuda_context_can_access_peer(void* context, void* peer_context);
    WP_API int cuda_context_enable_peer_access(void* context, void* peer_context);

    // ensures all device side operations have completed in the current context
    WP_API void cuda_context_synchronize(void* context);

    // return cudaError_t code
    WP_API uint64_t cuda_context_check(void* context);

    WP_API void* cuda_stream_create(void* context);
    WP_API void cuda_stream_destroy(void* context, void* stream);
    WP_API void cuda_stream_synchronize(void* context, void* stream);
    WP_API void* cuda_stream_get_current();
    WP_API void cuda_stream_wait_event(void* context, void* stream, void* event);
    WP_API void cuda_stream_wait_stream(void* context, void* stream, void* other_stream, void* event);

    WP_API void* cuda_event_create(void* context, unsigned flags);
    WP_API void cuda_event_destroy(void* context, void* event);
    WP_API void cuda_event_record(void* context, void* event, void* stream);

    WP_API void cuda_graph_begin_capture(void* context);
    WP_API void* cuda_graph_end_capture(void* context);
    WP_API void cuda_graph_launch(void* context, void* graph);
    WP_API void cuda_graph_destroy(void* context, void* graph);

    WP_API size_t cuda_compile_program(const char* cuda_src, int arch, const char* include_dir, bool debug, bool verbose, bool verify_fp, bool fast_math, const char* output_file);

    WP_API void* cuda_load_module(void* context, const char* ptx);
    WP_API void cuda_unload_module(void* context, void* module);
    WP_API void* cuda_get_kernel(void* context, void* module, const char* name);
    WP_API size_t cuda_launch_kernel(void* context, void* kernel, size_t dim, void** args);

    WP_API void cuda_set_context_restore_policy(bool always_restore);
    WP_API int cuda_get_context_restore_policy();

} // extern "C"
