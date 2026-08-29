// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "apic.h"
#include "apic_internal.h"
#include "cuda_util.h"
#include "error.h"
#include "sort.h"

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <cassert>
#include <mutex>
#include <unordered_map>
#include <utility>

#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

// temporary buffer for radix sort
struct TempBuffer {
    void* mem = NULL;
    size_t size = 0;
    bool is_ephemeral = false;  // not cached, released after use
    CUstream stream = NULL;  // owning stream handle, used by radix_sort_stream_release() (global cache entries only)
};

// Use unique temp buffers per CUDA stream to avoid race conditions.
// - Buffers are released when their stream is destroyed/unregistered.
// - Keyed by stream id rather than the stream handle: ids are unique for the
//   lifetime of the process, while handles can be reused after a stream is
//   destroyed (unknown external streams do not pass through
//   radix_sort_stream_release()).
using TempMapPerStreamId = std::unordered_map<uint64_t, TempBuffer>;
static TempMapPerStreamId g_temp_cache;

// Temp buffers used in graphs are capture-specific.
// - This ensures that each graph is fully independent.
// - Only side-allocated buffers are tracked here. When graph allocations
//   are possible, the buffers are ephemeral: allocated and released within
//   the graph as needed, not cached or tracked.
// - The buffers are owned by the graph and released when the graph is destroyed.
//   The caches just track the latest entry to determine if a bigger buffer needs
//   to be allocated.
// - The inner maps are keyed by stream handle, not stream id: querying the id
//   of a capturing stream is not permitted (cudaErrorStreamCaptureUnsupported),
//   and these entries are capture-scoped, so handle reuse cannot outlive them.
using TempMapPerStream = std::unordered_map<CUstream, TempBuffer>;
using TempMapPerCapture = std::unordered_map<uint64_t, TempMapPerStream>;
static TempMapPerCapture g_temp_cache_per_capture;

// Side streams for unsafe operations during capture (e.g., side allocs).
static std::unordered_map<CUcontext, CUstream> g_side_streams;

// Protect g_temp_cache, g_temp_cache_per_capture, and g_side_streams.
static std::mutex g_cache_mutex;

template <int Size> struct SortPayload {
    uint8_t data[Size];
};

static CUstream get_side_stream(void* context)
{
    if (!context) {
        context = wp_cuda_context_get_current();
    }

    auto it = g_side_streams.find(static_cast<CUcontext>(context));
    if (it != g_side_streams.end()) {
        return it->second;
    }

    CUstream stream = (CUstream)wp_cuda_stream_create(WP_CURRENT_CONTEXT, 0);
    if (stream) {
        g_side_streams[static_cast<CUcontext>(context)] = stream;
    } else {
        wp::set_error_string("Warp sort error: Failed to create side stream");
    }

    return stream;
}

static void cached_side_alloc(
    size_t size, bool async, CUstream stream, uint64_t capture_id, CaptureInfo* capture, TempBuffer& temp_ret
)
{
    std::lock_guard<std::mutex> lock(g_cache_mutex);

    // Use a capture-specific temp cache.
    // If this is a child graph capture, we should use the parent's capture->id.
    if (capture)
        capture_id = capture->id;

    TempBuffer& temp = g_temp_cache_per_capture[capture_id][stream];
    if (temp.mem && temp.size >= size) {
        // existing buffer is big enough
        temp_ret = temp;
        return;
    }

    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    void* ptr = NULL;

    // Don't free the previous cached buffer here, just allocate a bigger one.
    // All temps must be retained by the graph and freed once the graph is destroyed.
    if (async) {
        // allocate on a non-capturing side stream
        check_cuda(cudaThreadExchangeStreamCaptureMode(&mode));
        cudaStream_t side_stream = get_side_stream(WP_CURRENT_CONTEXT);
        if (side_stream) {
            ptr = wp_alloc_device_async(WP_CURRENT_CONTEXT, size, side_stream, "(native:sort)");
            wp_cuda_stream_synchronize(side_stream);
        }
        check_cuda(cudaThreadExchangeStreamCaptureMode(&mode));
    } else {
        // use synchronous cudaMalloc()
        check_cuda(cudaThreadExchangeStreamCaptureMode(&mode));
        ptr = wp_alloc_device_default(WP_CURRENT_CONTEXT, size, "(native:sort)");
        check_cuda(cudaThreadExchangeStreamCaptureMode(&mode));
    }

    temp.mem = ptr;
    temp.size = ptr ? size : 0;
    temp.is_ephemeral = false;

    // Ensure the alloc is freed when the graph is destroyed.
    if (ptr) {
        if (capture) {
            FreeInfo free_info;
            free_info.context = wp_cuda_context_get_current();
            free_info.ptr = ptr;
            free_info.is_async = async;
            capture->tmp_allocs.push_back(free_info);
        } else {
            // This is an unregistered capture without a graph destruction callback.
            // The alloc will leak, so inform user how to fix it.
            fprintf(
                stderr,
                "Warp sort warning: Allocating a temporary sort buffer during unregistered graph capture. "
                "Register external captures using wp.ScopedCapture(..., external=True) or "
                "wp.capture_begin(..., external=True) to avoid memory leaks.\n"
            );
        }
    }

    temp_ret = temp;
}

static bool acquire_temp_buffer(size_t size, TempBuffer& temp_ret)
{
    CUstream stream = static_cast<CUstream>(wp_cuda_stream_get_current());

    if (wp_cuda_stream_is_capturing(stream)) {
        int ordinal = wp_cuda_context_get_device_ordinal(WP_CURRENT_CONTEXT);
        bool mempool_supported = bool(wp_cuda_device_is_mempool_supported(ordinal));
        uint64_t capture_id = get_capture_id(stream);

        // Find the registered capture, which can be a top-level capture or the
        // parent of a child graph capture. This works for any participating
        // stream, including forked streams.
        CaptureInfo* capture = find_capture_info(capture_id);

        // Use ephemeral graph allocations when the capture is registered
        // and is not a child graph capture (where graph allocations are not allowed).
        bool use_graph_allocs = mempool_supported && capture && capture->id == capture_id;

        if (use_graph_allocs) {
            // Use ephemeral graph allocs, released after use.
            temp_ret.mem = wp_alloc_device_async(WP_CURRENT_CONTEXT, size, stream, "(native:sort)");
            temp_ret.size = temp_ret.mem ? size : 0;
            temp_ret.is_ephemeral = true;
        } else {
            cached_side_alloc(size, mempool_supported, stream, capture_id, capture, temp_ret);
        }
    } else {
        // No capture, use global temp cache.
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        TempBuffer& temp = g_temp_cache[get_stream_id(stream)];
        if (size > temp.size) {
            if (temp.mem) {
                wp_free_device(WP_CURRENT_CONTEXT, temp.mem);
            }
            temp.mem = wp_alloc_device(WP_CURRENT_CONTEXT, size, "(native:sort)");
            temp.size = temp.mem ? size : 0;
            temp.is_ephemeral = false;
        }
        temp.stream = stream;
        temp_ret = temp;
    }

    return temp_ret.mem && temp_ret.size >= size;
}

static void release_temp_buffer(TempBuffer& temp)
{
    if (temp.is_ephemeral) {
        wp_free_device_async(WP_CURRENT_CONTEXT, temp.mem, NULL);
        temp.mem = NULL;
        temp.size = 0;
    }
}

template <typename KeyType, typename ValueType>
size_t radix_sort_temp_size(void* context, int n, int begin_bit, int end_bit)
{
    ContextGuard guard(context);

    cub::DoubleBuffer<KeyType> d_keys;
    cub::DoubleBuffer<ValueType> d_values;

    CUstream stream = static_cast<CUstream>(wp_cuda_stream_get_current());

    // compute temporary memory required
    size_t sort_temp_size;
    if (check_cuda(
            cub::DeviceRadixSort::SortPairs(NULL, sort_temp_size, d_keys, d_values, n, begin_bit, end_bit, stream)
        )) {
        return sort_temp_size;
    } else {
        return 0;
    }
}

void radix_sort_stream_release(void* context, void* stream)
{
    std::lock_guard<std::mutex> lock(g_cache_mutex);

    // Release temporary buffers created for the given stream. Lookup is by the
    // stored stream handle rather than the stream id, since the id cannot be
    // queried while the stream is capturing. As a bonus, this also reclaims
    // entries leaked by externally destroyed streams once their handle is reused.
    for (auto it = g_temp_cache.begin(); it != g_temp_cache.end();) {
        if (it->second.stream == static_cast<CUstream>(stream)) {
            wp_free_device(context, it->second.mem);
            it = g_temp_cache.erase(it);
        } else {
            ++it;
        }
    }

    // release capture-specific info for the given stream
    for (auto& kv : g_temp_cache_per_capture) {
        kv.second.erase(static_cast<CUstream>(stream));
    }
}

void radix_sort_context_release(void* context)
{
    if (!context)
        context = wp_cuda_context_get_current();

    CUstream side_stream = NULL;
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto it = g_side_streams.find(static_cast<CUcontext>(context));
        if (it != g_side_streams.end()) {
            side_stream = it->second;
            g_side_streams.erase(it);
        }
    }

    // destroy outside the lock: stream destruction re-enters radix_sort_stream_release()
    if (side_stream)
        wp_cuda_stream_destroy(context, side_stream);
}

void radix_sort_end_capture(uint64_t capture_id)
{
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_temp_cache_per_capture.erase(capture_id);
}

template <typename KeyType, typename ValueType>
void radix_sort_pairs_device(void* context, KeyType* keys, ValueType* values, int n, int begin_bit, int end_bit)
{
    ContextGuard guard(context);

    cub::DoubleBuffer<KeyType> d_keys(keys, keys + n);
    cub::DoubleBuffer<ValueType> d_values(values, values + n);

    CUstream stream = static_cast<CUstream>(wp_cuda_stream_get_current());
    size_t temp_size = radix_sort_temp_size<KeyType, ValueType>(WP_CURRENT_CONTEXT, n, begin_bit, end_bit);

    TempBuffer temp;
    if (!acquire_temp_buffer(temp_size, temp)) {
        wp::set_error_string("Warp sort error: Failed to acquire radix sort buffer");
        return;
    }

    // sort
    check_cuda(cub::DeviceRadixSort::SortPairs(temp.mem, temp.size, d_keys, d_values, n, begin_bit, end_bit, stream));

    release_temp_buffer(temp);

    if (d_keys.Current() != keys)
        wp_memcpy_d2d(WP_CURRENT_CONTEXT, keys, d_keys.Current(), sizeof(KeyType) * n);

    if (d_values.Current() != values)
        wp_memcpy_d2d(WP_CURRENT_CONTEXT, values, d_values.Current(), sizeof(ValueType) * n);
}

template <typename KeyType>
void radix_sort_pairs_device_dispatch_value(
    void* context, KeyType* keys, void* values, int n, int begin_bit, int end_bit, int value_size
)
{
    if (value_size == 4) {
        radix_sort_pairs_device<KeyType, SortPayload<4>>(
            context, keys, reinterpret_cast<SortPayload<4>*>(values), n, begin_bit, end_bit
        );
    } else if (value_size == 8) {
        radix_sort_pairs_device<KeyType, SortPayload<8>>(
            context, keys, reinterpret_cast<SortPayload<8>*>(values), n, begin_bit, end_bit
        );
    } else {
        wp::set_error_string("Warp sort error: Unsupported radix sort value size %d", value_size);
        assert(false && "Unsupported radix sort value size");
    }
}

void radix_sort_pairs_device(void* context, int* keys, int* values, int n, int begin_bit, int end_bit)
{
    radix_sort_pairs_device_dispatch_value(context, keys, values, n, begin_bit, end_bit, sizeof(int));
}

void radix_sort_pairs_device(void* context, uint32_t* keys, int* values, int n, int begin_bit, int end_bit)
{
    radix_sort_pairs_device_dispatch_value(context, keys, values, n, begin_bit, end_bit, sizeof(int));
}

void radix_sort_pairs_device(void* context, float* keys, int* values, int n, int begin_bit, int end_bit)
{
    radix_sort_pairs_device_dispatch_value(context, keys, values, n, begin_bit, end_bit, sizeof(int));
}

void radix_sort_pairs_device(void* context, double* keys, int* values, int n, int begin_bit, int end_bit)
{
    radix_sort_pairs_device_dispatch_value(context, keys, values, n, begin_bit, end_bit, sizeof(int));
}

void radix_sort_pairs_device(void* context, int64_t* keys, int* values, int n, int begin_bit, int end_bit)
{
    radix_sort_pairs_device_dispatch_value(context, keys, values, n, begin_bit, end_bit, sizeof(int));
}

void radix_sort_pairs_device(void* context, uint64_t* keys, int* values, int n, int begin_bit, int end_bit)
{
    radix_sort_pairs_device_dispatch_value(context, keys, values, n, begin_bit, end_bit, sizeof(int));
}

// Record-and-execute a radix sort under CUDA APIC capture: record params into
// the byte stream, then fall through so the live sort issues onto the captured
// stream. Mirror of apic_capture_radix_sort in sort.cpp, but device-scoped and
// non-skipping (the CUDA op must execute so the driver captures it into the
// native graph; the byte stream carries it for persistent .wrp save/load).
// No-op outside a CUDA APIC capture and during graph rebuild.
static void apic_capture_radix_sort_device(
    uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size, uint8_t dtype, uint64_t key_size
)
{
    APICState* state = wp_apic_get_cuda_recording_state();
    if (!state || n <= 0)
        return;
    if (value_size != 4 && value_size != 8)
        return;
    uint64_t keys_bytes = static_cast<uint64_t>(2) * static_cast<uint64_t>(n) * key_size;
    uint64_t values_bytes = static_cast<uint64_t>(2) * static_cast<uint64_t>(n) * static_cast<uint64_t>(value_size);
    APICAddress keys_addr = apic_resolve_live_ptr(state, keys, keys_bytes);
    APICAddress values_addr = apic_resolve_live_ptr(state, values, values_bytes);
    apic_record_radix_sort(
        state, keys_addr.region_id, keys_addr.offset, values_addr.region_id, values_addr.offset,
        static_cast<uint32_t>(n), begin_bit, end_bit, value_size, dtype
    );
}

// Record-and-execute a segmented sort under CUDA APIC capture. Mirror of
// apic_capture_segmented_sort in sort.cpp, device-scoped and non-skipping.
static void apic_capture_segmented_sort_device(
    uint64_t keys, uint64_t values, int n, uint64_t segment_start, uint64_t segment_end, int num_segments, uint8_t dtype
)
{
    APICState* state = wp_apic_get_cuda_recording_state();
    if (!state || n <= 0)
        return;
    // keys/values span 2*n elements (sort scratch). Keys are int32 or float32
    // (both 4 bytes); values are always int32.
    uint64_t kv_bytes = static_cast<uint64_t>(2) * static_cast<uint64_t>(n) * sizeof(uint32_t);
    // Inferred-end captures alias segment_end into the start array one element
    // in, so the start array spans num_segments+1 entries; explicit-end captures
    // use two separate num_segments-entry arrays. Match the recorded span.
    bool inferred_end = (segment_end == segment_start + sizeof(int32_t));
    uint64_t segstart_count = static_cast<uint64_t>(num_segments) + (inferred_end ? 1u : 0u);
    uint64_t segstart_bytes = segstart_count * sizeof(int32_t);
    uint64_t segend_bytes = static_cast<uint64_t>(num_segments) * sizeof(int32_t);
    APICAddress keys_addr = apic_resolve_live_ptr(state, keys, kv_bytes);
    APICAddress values_addr = apic_resolve_live_ptr(state, values, kv_bytes);
    APICAddress segstart_addr = apic_resolve_live_ptr(state, segment_start, segstart_bytes);
    APICAddress segend_addr = apic_resolve_live_ptr(state, segment_end, segend_bytes);
    apic_record_segmented_sort(
        state, keys_addr.region_id, keys_addr.offset, values_addr.region_id, values_addr.offset,
        segstart_addr.region_id, segstart_addr.offset, segend_addr.region_id, segend_addr.offset,
        static_cast<uint32_t>(n), static_cast<uint32_t>(num_segments), dtype
    );
}

void wp_radix_sort_pairs_int_device(uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size)
{
    apic_capture_radix_sort_device(keys, values, n, begin_bit, end_bit, value_size, APIC_TYPE_INT32, sizeof(int32_t));
    radix_sort_pairs_device_dispatch_value(
        WP_CURRENT_CONTEXT, reinterpret_cast<int*>(keys), reinterpret_cast<void*>(values), n, begin_bit, end_bit,
        value_size
    );
}

void wp_radix_sort_pairs_uint_device(uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size)
{
    apic_capture_radix_sort_device(keys, values, n, begin_bit, end_bit, value_size, APIC_TYPE_UINT32, sizeof(uint32_t));
    radix_sort_pairs_device_dispatch_value(
        WP_CURRENT_CONTEXT, reinterpret_cast<uint32_t*>(keys), reinterpret_cast<void*>(values), n, begin_bit, end_bit,
        value_size
    );
}

void wp_radix_sort_pairs_float_device(uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size)
{
    apic_capture_radix_sort_device(keys, values, n, begin_bit, end_bit, value_size, APIC_TYPE_FLOAT32, sizeof(float));
    radix_sort_pairs_device_dispatch_value(
        WP_CURRENT_CONTEXT, reinterpret_cast<float*>(keys), reinterpret_cast<void*>(values), n, begin_bit, end_bit,
        value_size
    );
}

void wp_radix_sort_pairs_double_device(
    uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size
)
{
    apic_capture_radix_sort_device(keys, values, n, begin_bit, end_bit, value_size, APIC_TYPE_FLOAT64, sizeof(double));
    radix_sort_pairs_device_dispatch_value(
        WP_CURRENT_CONTEXT, reinterpret_cast<double*>(keys), reinterpret_cast<void*>(values), n, begin_bit, end_bit,
        value_size
    );
}

void wp_radix_sort_pairs_int64_device(uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size)
{
    apic_capture_radix_sort_device(keys, values, n, begin_bit, end_bit, value_size, APIC_TYPE_INT64, sizeof(int64_t));
    radix_sort_pairs_device_dispatch_value(
        WP_CURRENT_CONTEXT, reinterpret_cast<int64_t*>(keys), reinterpret_cast<void*>(values), n, begin_bit, end_bit,
        value_size
    );
}

void wp_radix_sort_pairs_uint64_device(
    uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size
)
{
    apic_capture_radix_sort_device(keys, values, n, begin_bit, end_bit, value_size, APIC_TYPE_UINT64, sizeof(uint64_t));
    radix_sort_pairs_device_dispatch_value(
        WP_CURRENT_CONTEXT, reinterpret_cast<uint64_t*>(keys), reinterpret_cast<void*>(values), n, begin_bit, end_bit,
        value_size
    );
}

template <bool IsBegin> struct ValidatedSegmentOffset {
    const int* segment_start_indices;
    const int* segment_end_indices;
    int count;

    __host__ __device__ __forceinline__ int operator()(int segment_index) const
    {
        const int start = segment_start_indices[segment_index];
        const int end = segment_end_indices[segment_index];
        if (start < 0 || end < start || end > count)
            return 0;
        return IsBegin ? start : end;
    }
};

// CUB accepts iterator-defined offsets. Map each invalid pair to [0, 0) so
// malformed device-resident metadata cannot address outside the input buffers.
// This avoids a device-to-host copy or synchronization and remains safe during
// CUDA graph capture.
auto make_validated_segment_offsets(int* segment_start_indices, int* segment_end_indices, int count)
{
    auto segment_indices = thrust::make_counting_iterator(0);
    auto begin_offsets = thrust::make_transform_iterator(
        segment_indices, ValidatedSegmentOffset<true> { segment_start_indices, segment_end_indices, count }
    );
    auto end_offsets = thrust::make_transform_iterator(
        segment_indices, ValidatedSegmentOffset<false> { segment_start_indices, segment_end_indices, count }
    );
    return std::make_pair(begin_offsets, end_offsets);
}

size_t
segmented_sort_temp_size(void* context, int n, int num_segments, int* segment_start_indices, int* segment_end_indices)
{
    ContextGuard guard(context);

    cub::DoubleBuffer<int> d_keys;
    cub::DoubleBuffer<int> d_values;

    auto segment_offsets = make_validated_segment_offsets(segment_start_indices, segment_end_indices, n);

    CUstream stream = static_cast<CUstream>(wp_cuda_stream_get_current());

    // compute temporary memory required
    size_t sort_temp_size = 0;
    if (check_cuda(
            cub::DeviceSegmentedRadixSort::SortPairs(
                NULL, sort_temp_size, d_keys, d_values, n, num_segments, segment_offsets.first, segment_offsets.second,
                0, 32, stream
            )
        )) {
        return sort_temp_size;
    } else {
        return 0;
    }
}

// segment_start_indices and segment_end_indices are arrays of length num_segments, where segment_start_indices[i] is
// the index of the first element in the i-th segment and segment_end_indices[i] is the index after the last element in
// the i-th segment https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceSegmentedRadixSort.html
void segmented_sort_pairs_device(
    void* context,
    float* keys,
    int* values,
    int n,
    int* segment_start_indices,
    int* segment_end_indices,
    int num_segments
)
{
    ContextGuard guard(context);

    cub::DoubleBuffer<float> d_keys(keys, keys + n);
    cub::DoubleBuffer<int> d_values(values, values + n);

    CUstream stream = static_cast<CUstream>(wp_cuda_stream_get_current());
    auto segment_offsets = make_validated_segment_offsets(segment_start_indices, segment_end_indices, n);
    size_t temp_size
        = segmented_sort_temp_size(WP_CURRENT_CONTEXT, n, num_segments, segment_start_indices, segment_end_indices);

    TempBuffer temp;
    if (!acquire_temp_buffer(temp_size, temp)) {
        wp::set_error_string("Warp sort error: Failed to acquire segmented sort buffer");
        return;
    }

    // sort
    check_cuda(
        cub::DeviceSegmentedRadixSort::SortPairs(
            temp.mem, temp.size, d_keys, d_values, n, num_segments, segment_offsets.first, segment_offsets.second, 0,
            32, stream
        )
    );

    release_temp_buffer(temp);

    if (d_keys.Current() != keys)
        wp_memcpy_d2d(WP_CURRENT_CONTEXT, keys, d_keys.Current(), sizeof(float) * n);

    if (d_values.Current() != values)
        wp_memcpy_d2d(WP_CURRENT_CONTEXT, values, d_values.Current(), sizeof(int) * n);
}

void wp_segmented_sort_pairs_float_device(
    uint64_t keys,
    uint64_t values,
    int n,
    uint64_t segment_start_indices,
    uint64_t segment_end_indices,
    int num_segments
)
{
    apic_capture_segmented_sort_device(
        keys, values, n, segment_start_indices, segment_end_indices, num_segments, APIC_TYPE_FLOAT32
    );
    segmented_sort_pairs_device(
        WP_CURRENT_CONTEXT, reinterpret_cast<float*>(keys), reinterpret_cast<int*>(values), n,
        reinterpret_cast<int*>(segment_start_indices), reinterpret_cast<int*>(segment_end_indices), num_segments
    );
}

// segment_indices is an array of length num_segments + 1, where segment_indices[i] is the index of the first element in
// the i-th segment The end of a segment is given by segment_indices[i+1]
// https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceSegmentedSort.html#a-simple-example
void segmented_sort_pairs_device(
    void* context, int* keys, int* values, int n, int* segment_start_indices, int* segment_end_indices, int num_segments
)
{
    ContextGuard guard(context);

    cub::DoubleBuffer<int> d_keys(keys, keys + n);
    cub::DoubleBuffer<int> d_values(values, values + n);

    CUstream stream = static_cast<CUstream>(wp_cuda_stream_get_current());
    auto segment_offsets = make_validated_segment_offsets(segment_start_indices, segment_end_indices, n);
    size_t temp_size
        = segmented_sort_temp_size(WP_CURRENT_CONTEXT, n, num_segments, segment_start_indices, segment_end_indices);

    TempBuffer temp;
    if (!acquire_temp_buffer(temp_size, temp)) {
        wp::set_error_string("Warp sort error: Failed to acquire segmented sort buffer");
        return;
    }

    // sort
    check_cuda(
        cub::DeviceSegmentedRadixSort::SortPairs(
            temp.mem, temp.size, d_keys, d_values, n, num_segments, segment_offsets.first, segment_offsets.second, 0,
            32, stream
        )
    );

    release_temp_buffer(temp);

    if (d_keys.Current() != keys)
        wp_memcpy_d2d(WP_CURRENT_CONTEXT, keys, d_keys.Current(), sizeof(int) * n);

    if (d_values.Current() != values)
        wp_memcpy_d2d(WP_CURRENT_CONTEXT, values, d_values.Current(), sizeof(int) * n);
}

void wp_segmented_sort_pairs_int_device(
    uint64_t keys,
    uint64_t values,
    int n,
    uint64_t segment_start_indices,
    uint64_t segment_end_indices,
    int num_segments
)
{
    apic_capture_segmented_sort_device(
        keys, values, n, segment_start_indices, segment_end_indices, num_segments, APIC_TYPE_INT32
    );
    segmented_sort_pairs_device(
        WP_CURRENT_CONTEXT, reinterpret_cast<int*>(keys), reinterpret_cast<int*>(values), n,
        reinterpret_cast<int*>(segment_start_indices), reinterpret_cast<int*>(segment_end_indices), num_segments
    );
}
