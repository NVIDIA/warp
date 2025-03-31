/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// defines all crt + builtin types
#include "builtin.h"

#define WP_CURRENT_STREAM ((void*)0xffffffffffffffff)

struct timing_result_t;

// this is the core runtime API exposed on the DLL level
extern "C"
{
    WP_API int init();
    //WP_API void shutdown();

    // get error message from C++
    WP_API const char* get_error_string();

    // allow disabling error output, which is handy during tests that expect failure
    WP_API void set_error_output_enabled(int enable);
    WP_API int is_error_output_enabled();

    // whether Warp was compiled with CUDA support
    WP_API int is_cuda_enabled();
    // whether Warp was compiled with enhanced CUDA compatibility
    WP_API int is_cuda_compatibility_enabled();
    // whether Warp was compiled with MathDx support
    WP_API int is_mathdx_enabled();
    // whether Warp was compiled with debug support
    WP_API int is_debug_enabled();

    WP_API uint16_t float_to_half_bits(float x);
    WP_API float half_bits_to_float(uint16_t u);

    WP_API void* alloc_host(size_t s);
    WP_API void* alloc_pinned(size_t s);
    WP_API void* alloc_device(void* context, size_t s);  // uses cudaMallocAsync() if supported, cudaMalloc() otherwise
    WP_API void* alloc_device_default(void* context, size_t s);  // uses cudaMalloc()
    WP_API void* alloc_device_async(void* context, size_t s);  // uses cudaMallocAsync()

    WP_API void free_host(void* ptr);
    WP_API void free_pinned(void* ptr);
    WP_API void free_device(void* context, void* ptr);  // uses cudaFreeAsync() if supported, cudaFree() otherwise
    WP_API void free_device_default(void* context, void* ptr);  // uses cudaFree()
    WP_API void free_device_async(void* context, void* ptr);  // uses cudaFreeAsync()

    WP_API bool memcpy_h2h(void* dest, void* src, size_t n);
    WP_API bool memcpy_h2d(void* context, void* dest, void* src, size_t n, void* stream=WP_CURRENT_STREAM);
    WP_API bool memcpy_d2h(void* context, void* dest, void* src, size_t n, void* stream=WP_CURRENT_STREAM);
    WP_API bool memcpy_d2d(void* context, void* dest, void* src, size_t n, void* stream=WP_CURRENT_STREAM);
    WP_API bool memcpy_p2p(void* dst_context, void* dst, void* src_context, void* src, size_t n, void* stream=WP_CURRENT_STREAM);

    WP_API void memset_host(void* dest, int value, size_t n);
    WP_API void memset_device(void* context, void* dest, int value, size_t n);
    
    // takes srcsize bytes starting at src and repeats them n times at dst (writes srcsize * n bytes in total):
    WP_API void memtile_host(void* dest, const void* src, size_t srcsize, size_t n);
    WP_API void memtile_device(void* context, void* dest, const void* src, size_t srcsize, size_t n);

	WP_API uint64_t bvh_create_host(wp::vec3* lowers, wp::vec3* uppers, int num_items, int constructor_type);
	WP_API void bvh_destroy_host(uint64_t id);
    WP_API void bvh_refit_host(uint64_t id);

	WP_API uint64_t bvh_create_device(void* context, wp::vec3* lowers, wp::vec3* uppers, int num_items, int constructor_type);
	WP_API void bvh_destroy_device(uint64_t id);
    WP_API void bvh_refit_device(uint64_t id);

    // create a user-accessible copy of the mesh, it is the 
    // users responsibility to keep-alive the points/tris data for the duration of the mesh lifetime
	WP_API uint64_t mesh_create_host(wp::array_t<wp::vec3> points, wp::array_t<wp::vec3> velocities, wp::array_t<int> tris, 
        int num_points, int num_tris, int support_winding_number, int constructor_type);
	WP_API void mesh_destroy_host(uint64_t id);
    WP_API void mesh_refit_host(uint64_t id);

	WP_API uint64_t mesh_create_device(void* context, wp::array_t<wp::vec3> points, wp::array_t<wp::vec3> velocities, 
        wp::array_t<int> tris, int num_points, int num_tris, int support_winding_number, int constructor_type);
	WP_API void mesh_destroy_device(uint64_t id);
    WP_API void mesh_refit_device(uint64_t id);

    WP_API void mesh_set_points_host(uint64_t id, wp::array_t<wp::vec3> points);
    WP_API void mesh_set_points_device(uint64_t id, wp::array_t<wp::vec3> points);

    WP_API void mesh_set_velocities_host(uint64_t id, wp::array_t<wp::vec3> velocities);
    WP_API void mesh_set_velocities_device(uint64_t id, wp::array_t<wp::vec3> velocities);

    WP_API uint64_t hash_grid_create_host(int dim_x, int dim_y, int dim_z);
    WP_API void hash_grid_reserve_host(uint64_t id, int num_points);
    WP_API void hash_grid_destroy_host(uint64_t id);
    WP_API void hash_grid_update_host(uint64_t id, float cell_width, const wp::array_t<wp::vec3>* points);

    WP_API uint64_t hash_grid_create_device(void* context, int dim_x, int dim_y, int dim_z);
    WP_API void hash_grid_reserve_device(uint64_t id, int num_points);
    WP_API void hash_grid_destroy_device(uint64_t id);
    WP_API void hash_grid_update_device(uint64_t id, float cell_width, const wp::array_t<wp::vec3>* points);

    WP_API uint64_t volume_create_host(void* buf, uint64_t size, bool copy, bool owner);
    WP_API void volume_get_tiles_host(uint64_t id, void* buf);
    WP_API void volume_get_voxels_host(uint64_t id, void* buf);
    WP_API void volume_destroy_host(uint64_t id);

    WP_API uint64_t volume_create_device(void* context, void* buf, uint64_t size, bool copy, bool owner);
    WP_API void volume_get_tiles_device(uint64_t id, void* buf);
    WP_API void volume_get_voxels_device(uint64_t id, void* buf);
    WP_API void volume_destroy_device(uint64_t id);
    
    WP_API uint64_t volume_from_tiles_device(void* context, void* points, int num_points, float transform[9], float translation[3], bool points_in_world_space, const void* bg_value, uint32_t bg_value_size, const char* bg_value_type);
    WP_API uint64_t volume_index_from_tiles_device(void* context, void* points, int num_points, float transform[9], float translation[3], bool points_in_world_space);
    WP_API uint64_t volume_from_active_voxels_device(void* context, void* points, int num_points, float transform[9], float translation[3], bool points_in_world_space);

    WP_API void volume_get_buffer_info(uint64_t id, void** buf, uint64_t* size);
    WP_API void volume_get_voxel_size(uint64_t id, float* dx, float* dy, float* dz);
    WP_API void volume_get_tile_and_voxel_count(uint64_t id, uint32_t& tile_count, uint64_t& voxel_count);
    WP_API const char* volume_get_grid_info(uint64_t id, uint64_t *grid_size, uint32_t *grid_index, uint32_t *grid_count, float translation[3], float transform[9], char type_str[16]);
    WP_API uint32_t volume_get_blind_data_count(uint64_t id);
    WP_API const char* volume_get_blind_data_info(uint64_t id, uint32_t data_index, void** buf, uint64_t* value_count, uint32_t* value_size, char type_str[16]);
    
    WP_API uint64_t marching_cubes_create_device(void* context);
    WP_API void marching_cubes_destroy_device(uint64_t id);
    WP_API int marching_cubes_surface_device(uint64_t id, const float* field, int nx, int ny, int nz, float threshold, wp::vec3* verts, int* triangles, int max_verts, int max_tris, int* out_num_verts, int* out_num_tris);

    // generic copy supporting non-contiguous arrays
    WP_API bool array_copy_host(void* dst, void* src, int dst_type, int src_type, int elem_size);
    WP_API bool array_copy_device(void* context, void* dst, void* src, int dst_type, int src_type, int elem_size);

    // generic fill for non-contiguous arrays
    WP_API void array_fill_host(void* arr, int arr_type, const void* value, int value_size);
    WP_API void array_fill_device(void* context, void* arr, int arr_type, const void* value, int value_size);

    WP_API void array_inner_float_host(uint64_t a, uint64_t b, uint64_t out, int count, int stride_a, int stride_b, int type_len);
    WP_API void array_inner_double_host(uint64_t a, uint64_t b, uint64_t out, int count, int stride_a, int stride_b, int type_len);
    WP_API void array_inner_float_device(uint64_t a, uint64_t b, uint64_t out, int count, int stride_a, int stride_b, int type_len);
    WP_API void array_inner_double_device(uint64_t a, uint64_t b, uint64_t out, int count, int stride_a, int stride_b, int type_len);

    WP_API void array_sum_float_device(uint64_t a, uint64_t out, int count, int stride, int type_len);
    WP_API void array_sum_float_host(uint64_t a, uint64_t out, int count, int stride, int type_len);
    WP_API void array_sum_double_host(uint64_t a, uint64_t out, int count, int stride, int type_len);
    WP_API void array_sum_double_device(uint64_t a, uint64_t out, int count, int stride, int type_len);

    WP_API void array_scan_int_host(uint64_t in, uint64_t out, int len, bool inclusive);
    WP_API void array_scan_float_host(uint64_t in, uint64_t out, int len, bool inclusive);

    WP_API void array_scan_int_device(uint64_t in, uint64_t out, int len, bool inclusive);
    WP_API void array_scan_float_device(uint64_t in, uint64_t out, int len, bool inclusive);

    WP_API void radix_sort_pairs_int_host(uint64_t keys, uint64_t values, int n);
    WP_API void radix_sort_pairs_int_device(uint64_t keys, uint64_t values, int n);

    WP_API void radix_sort_pairs_float_host(uint64_t keys, uint64_t values, int n);
    WP_API void radix_sort_pairs_float_device(uint64_t keys, uint64_t values, int n);

    WP_API void radix_sort_pairs_int64_host(uint64_t keys, uint64_t values, int n);
    WP_API void radix_sort_pairs_int64_device(uint64_t keys, uint64_t values, int n);

    WP_API void segmented_sort_pairs_float_host(uint64_t keys, uint64_t values, int n, uint64_t segment_start_indices, uint64_t segment_end_indices, int num_segments);
    WP_API void segmented_sort_pairs_float_device(uint64_t keys, uint64_t values, int n, uint64_t segment_start_indices, uint64_t segment_end_indices, int num_segments);

    WP_API void segmented_sort_pairs_int_host(uint64_t keys, uint64_t values, int n, uint64_t segment_start_indices, uint64_t segment_end_indices, int num_segments);
    WP_API void segmented_sort_pairs_int_device(uint64_t keys, uint64_t values, int n, uint64_t segment_start_indices, uint64_t segment_end_indices, int num_segments);

    WP_API void runlength_encode_int_host(uint64_t values, uint64_t run_values, uint64_t run_lengths, uint64_t run_count, int n);
    WP_API void runlength_encode_int_device(uint64_t values, uint64_t run_values, uint64_t run_lengths, uint64_t run_count, int n);

    WP_API void bsr_matrix_from_triplets_float_host(
        int rows_per_block,
        int cols_per_block,
        int row_count,
        int tpl_nnz,
        int* tpl_rows,
        int* tpl_columns,
        void* tpl_values,
        bool prune_numerical_zeros,
        bool masked,
        int* bsr_offsets,
        int* bsr_columns,
        void* bsr_values,
        int* bsr_nnz,
        void* bsr_nnz_event);
    WP_API void bsr_matrix_from_triplets_double_host(
        int rows_per_block,
        int cols_per_block,
        int row_count,
        int tpl_nnz,
        int* tpl_rows,
        int* tpl_columns,
        void* tpl_values,
        bool prune_numerical_zeros,
        bool masked,
        int* bsr_offsets,
        int* bsr_columns,
        void* bsr_values,
        int* bsr_nnz,
        void* bsr_nnz_event);
    WP_API void bsr_matrix_from_triplets_float_device(
        int rows_per_block,
        int cols_per_block,
        int row_count,
        int tpl_nnz,
        int* tpl_rows,
        int* tpl_columns,
        void* tpl_values,
        bool prune_numerical_zeros,
        bool masked,
        int* bsr_offsets,
        int* bsr_columns,
        void* bsr_values,
        int* bsr_nnz,
        void* bsr_nnz_event);
    WP_API void bsr_matrix_from_triplets_double_device(
        int rows_per_block,
        int cols_per_block,
        int row_count,
        int tpl_nnz,
        int* tpl_rows,
        int* tpl_columns,
        void* tpl_values,
        bool prune_numerical_zeros,
        bool masked,
        int* bsr_offsets,
        int* bsr_columns,
        void* bsr_values,
        int* bsr_nnz,
        void* bsr_nnz_event);

    WP_API void bsr_transpose_float_host(int rows_per_block, int cols_per_block,
        int row_count, int col_count, int nnz,
        int* bsr_offsets, int* bsr_columns,
        void* bsr_values,
        int* transposed_bsr_offsets,
        int* transposed_bsr_columns,
        void* transposed_bsr_values);
    WP_API void bsr_transpose_double_host(int rows_per_block, int cols_per_block,
        int row_count, int col_count, int nnz,
        int* bsr_offsets, int* bsr_columns,
        void* bsr_values,
        int* transposed_bsr_offsets,
        int* transposed_bsr_columns,
        void* transposed_bsr_values);
    WP_API void bsr_transpose_float_device(int rows_per_block, int cols_per_block,
        int row_count, int col_count, int nnz,
        int* bsr_offsets, int* bsr_columns,
        void* bsr_values,
        int* transposed_bsr_offsets,
        int* transposed_bsr_columns,
        void* transposed_bsr_values);
    WP_API void bsr_transpose_double_device(int rows_per_block, int cols_per_block,
        int row_count, int col_count, int nnz,
        int* bsr_offsets, int* bsr_columns,
        void* bsr_values,
        int* transposed_bsr_offsets,
        int* transposed_bsr_columns,
        void* transposed_bsr_values);


    WP_API int cuda_driver_version();   // CUDA driver version
    WP_API int cuda_toolkit_version();  // CUDA Toolkit version used to build Warp
    WP_API bool cuda_driver_is_initialized();

    WP_API int nvrtc_supported_arch_count();
    WP_API void nvrtc_supported_archs(int* archs);

    WP_API int cuda_device_get_count();
    WP_API void* cuda_device_get_primary_context(int ordinal);
    WP_API const char* cuda_device_get_name(int ordinal);
    WP_API int cuda_device_get_arch(int ordinal);
    WP_API int cuda_device_get_sm_count(int ordinal);
    WP_API void cuda_device_get_uuid(int ordinal, char uuid[16]);
    WP_API int cuda_device_get_pci_domain_id(int ordinal);
    WP_API int cuda_device_get_pci_bus_id(int ordinal);
    WP_API int cuda_device_get_pci_device_id(int ordinal);
    WP_API int cuda_device_is_uva(int ordinal);
    WP_API int cuda_device_is_mempool_supported(int ordinal);
    WP_API int cuda_device_is_ipc_supported(int ordinal);
    WP_API int cuda_device_set_mempool_release_threshold(int ordinal, uint64_t threshold);
    WP_API uint64_t cuda_device_get_mempool_release_threshold(int ordinal);
    WP_API uint64_t cuda_device_get_mempool_used_mem_current(int ordinal);
    WP_API uint64_t cuda_device_get_mempool_used_mem_high(int ordinal);
    WP_API void cuda_device_get_memory_info(int ordinal, size_t* free_mem, size_t* total_mem);

    WP_API void* cuda_context_get_current();
    WP_API void cuda_context_set_current(void* context);
    WP_API void cuda_context_push_current(void* context);
    WP_API void cuda_context_pop_current();
    WP_API void* cuda_context_create(int device_ordinal);
    WP_API void cuda_context_destroy(void* context);
    WP_API int cuda_context_get_device_ordinal(void* context);
    WP_API int cuda_context_is_primary(void* context);
    WP_API void* cuda_context_get_stream(void* context);
    WP_API void cuda_context_set_stream(void* context, void* stream, int sync);

    // ensures all device side operations have completed in the current context
    WP_API void cuda_context_synchronize(void* context);

    // return cudaError_t code
    WP_API uint64_t cuda_context_check(void* context);

    // peer access
    WP_API int cuda_is_peer_access_supported(int target_ordinal, int peer_ordinal);
    WP_API int cuda_is_peer_access_enabled(void* target_context, void* peer_context);
    WP_API int cuda_set_peer_access_enabled(void* target_context, void* peer_context, int enable);
    WP_API int cuda_is_mempool_access_enabled(int target_ordinal, int peer_ordinal);
    WP_API int cuda_set_mempool_access_enabled(int target_ordinal, int peer_ordinal, int enable);

    // inter-process communication
    WP_API void cuda_ipc_get_mem_handle(void* ptr, char* out_buffer);
    WP_API void* cuda_ipc_open_mem_handle(void* context, char* handle);
    WP_API void cuda_ipc_close_mem_handle(void* ptr);
    WP_API void cuda_ipc_get_event_handle(void* context, void* event, char* out_buffer);
    WP_API void* cuda_ipc_open_event_handle(void* context, char* handle);

    WP_API void* cuda_stream_create(void* context, int priority);
    WP_API void cuda_stream_destroy(void* context, void* stream);
    WP_API int cuda_stream_query(void* stream);
    WP_API void cuda_stream_register(void* context, void* stream);
    WP_API void cuda_stream_unregister(void* context, void* stream);
    WP_API void* cuda_stream_get_current();
    WP_API void cuda_stream_synchronize(void* stream);
    WP_API void cuda_stream_wait_event(void* stream, void* event);
    WP_API void cuda_stream_wait_stream(void* stream, void* other_stream, void* event);
    WP_API int cuda_stream_is_capturing(void* stream);
    WP_API uint64_t cuda_stream_get_capture_id(void* stream);
    WP_API int cuda_stream_get_priority(void* stream);

    WP_API void* cuda_event_create(void* context, unsigned flags);
    WP_API void cuda_event_destroy(void* event);
    WP_API int cuda_event_query(void* event);
    WP_API void cuda_event_record(void* event, void* stream, bool timing=false);
    WP_API void cuda_event_synchronize(void* event);
    WP_API float cuda_event_elapsed_time(void* start_event, void* end_event);

    WP_API bool cuda_graph_begin_capture(void* context, void* stream, int external);
    WP_API bool cuda_graph_end_capture(void* context, void* stream, void** graph_ret);
    WP_API bool cuda_graph_launch(void* graph, void* stream);
    WP_API bool cuda_graph_destroy(void* context, void* graph);

    WP_API size_t cuda_compile_program(const char* cuda_src, const char* program_name, int arch, const char* include_dir, int num_cuda_include_dirs, const char** cuda_include_dirs, bool debug, bool verbose, bool verify_fp, bool fast_math, bool fuse_fp, bool lineinfo, bool compile_time_trace, const char* output_path, size_t num_ltoirs, char** ltoirs, size_t* ltoir_sizes, int* ltoir_input_types);
    WP_API bool cuda_compile_fft(const char* ltoir_output_path, const char* symbol_name, int num_include_dirs, const char** include_dirs, const char* mathdx_include_dir, int arch, int size, int elements_per_thread, int direction, int precision, int* shared_memory_size);
    WP_API bool cuda_compile_dot(const char* ltoir_output_path, const char* symbol_name, int num_include_dirs, const char** include_dirs, const char* mathdx_include_dir, int arch, int M, int N, int K, int precision_A, int precision_B, int precision_C, int type, int arrangement_A, int arrangement_B, int arrangement_C, int num_threads);
    WP_API bool cuda_compile_solver(const char* fatbin_output_path, const char* ltoir_output_path, const char* symbol_name, int num_include_dirs, const char** include_dirs, const char* mathdx_include_dir, int arch, int M, int N, int function, int precision, int fill_mode, int num_threads);

    WP_API void* cuda_load_module(void* context, const char* ptx);
    WP_API void cuda_unload_module(void* context, void* module);
    WP_API void* cuda_get_kernel(void* context, void* module, const char* name);
    WP_API size_t cuda_launch_kernel(void* context, void* kernel, size_t dim, int max_blocks, int block_dim, int shared_memory_bytes, void** args, void* stream);
    WP_API int cuda_get_max_shared_memory(void* context);
    WP_API bool cuda_configure_kernel_shared_memory(void* kernel, int size);

    WP_API void cuda_set_context_restore_policy(bool always_restore);
    WP_API int cuda_get_context_restore_policy();

    WP_API void cuda_graphics_map(void* context, void* resource);
    WP_API void cuda_graphics_unmap(void* context, void* resource);
    WP_API void cuda_graphics_device_ptr_and_size(void* context, void* resource, uint64_t* ptr, size_t* size);
    WP_API void* cuda_graphics_register_gl_buffer(void* context, uint32_t gl_buffer, unsigned int flags);
    WP_API void cuda_graphics_unregister_resource(void* context, void* resource);

    // CUDA timing
    WP_API void cuda_timing_begin(int flags);
    WP_API int cuda_timing_get_result_count();
    WP_API void cuda_timing_end(timing_result_t* results, int size);

    // graph coloring
    WP_API int graph_coloring(int num_nodes, wp::array_t<int> edges, int algorithm, wp::array_t<int> node_colors);
    WP_API float balance_coloring(int num_nodes, wp::array_t<int> edges, int num_colors, float target_max_min_ratio, wp::array_t<int> node_colors);
        
} // extern "C"
