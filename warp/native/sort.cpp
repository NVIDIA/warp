// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "error.h"
#include "sort.h"
#include "string.h"

#include <cassert>
#include <cstdint>

template <int Size> struct SortPayload {
    uint8_t data[Size];
};

// Only integer keys (bit count 32 or 64) are supported. Floats need to get converted into int first. see
// radix_float_to_int.
template <typename KeyType, typename ValueType, typename RadixKeyType, typename KeyToRadix>
void radix_sort_pairs_host(
    KeyType* keys,
    ValueType* values,
    int n,
    int offset_to_scratch_memory,
    int begin_bit,
    int end_bit,
    KeyToRadix key_to_radix
)
{
    constexpr int keyWidth = sizeof(RadixKeyType) * 8;
    constexpr int maxPasses = (keyWidth + 15) / 16;

    if (begin_bit < 0 || end_bit <= begin_bit || end_bit > keyWidth) {
        return;
    }

    const int requestedPasses = (end_bit - begin_bit + 15) / 16;
    const int numPasses = requestedPasses < maxPasses ? requestedPasses : maxPasses;

    static thread_local int tables[maxPasses][1 << 16];
    memset(tables, 0, sizeof(tables));

    // build histograms
    for (int p = 0; p < numPasses; ++p) {
        const int shift = begin_bit + p * 16;
        const int passBits = (end_bit - shift) < 16 ? (end_bit - shift) : 16;
        const RadixKeyType mask = (RadixKeyType(1) << passBits) - 1;

        for (int i = 0; i < n; ++i) {
            const int b = (key_to_radix(keys[i]) >> shift) & mask;

            ++tables[p][b];
        }
    }

    // convert histograms to offset tables in-place
    for (int p = 0; p < numPasses; ++p) {
        const int shift = begin_bit + p * 16;
        const int passBits = (end_bit - shift) < 16 ? (end_bit - shift) : 16;
        const int bucketCount = 1 << passBits;
        int off = 0;
        for (int i = 0; i < bucketCount; ++i) {
            const int newoff = off + tables[p][i];

            tables[p][i] = off;

            off = newoff;
        }
    }

    for (int p = 0; p < numPasses; ++p) {
        int flipFlop = p % 2;
        KeyType* readKeys = keys + offset_to_scratch_memory * flipFlop;
        ValueType* readValues = values + offset_to_scratch_memory * flipFlop;
        KeyType* writeKeys = keys + offset_to_scratch_memory * (1 - flipFlop);
        ValueType* writeValues = values + offset_to_scratch_memory * (1 - flipFlop);

        // pass 1 - sort by low 16 bits
        for (int i = 0; i < n; ++i) {
            // lookup offset of input
            const KeyType k = readKeys[i];
            const ValueType v = readValues[i];

            const int shift = begin_bit + p * 16;
            const int passBits = (end_bit - shift) < 16 ? (end_bit - shift) : 16;
            const RadixKeyType mask = (RadixKeyType(1) << passBits) - 1;
            const int b = (key_to_radix(k) >> shift) & mask;

            // find offset and increment
            const int offset = tables[p][b]++;

            writeKeys[offset] = k;
            writeValues[offset] = v;
        }
    }

    if (numPasses % 2 == 1) {
        KeyType* auxKeys = keys + offset_to_scratch_memory;
        ValueType* auxValues = values + offset_to_scratch_memory;
        memcpy(keys, auxKeys, sizeof(KeyType) * n);
        memcpy(values, auxValues, sizeof(ValueType) * n);
    }
}

template <typename KeyType, typename RadixKeyType, typename KeyToRadix>
void radix_sort_pairs_host_dispatch_value(
    KeyType* keys,
    void* values,
    int n,
    int offset_to_scratch_memory,
    int begin_bit,
    int end_bit,
    int value_size,
    KeyToRadix key_to_radix
)
{
    if (value_size == 4) {
        radix_sort_pairs_host<KeyType, SortPayload<4>, RadixKeyType>(
            keys, reinterpret_cast<SortPayload<4>*>(values), n, offset_to_scratch_memory, begin_bit, end_bit,
            key_to_radix
        );
    } else if (value_size == 8) {
        radix_sort_pairs_host<KeyType, SortPayload<8>, RadixKeyType>(
            keys, reinterpret_cast<SortPayload<8>*>(values), n, offset_to_scratch_memory, begin_bit, end_bit,
            key_to_radix
        );
    } else {
        wp::set_error_string("Warp sort error: Unsupported radix sort value size %d", value_size);
        assert(false && "Unsupported radix sort value size");
    }
}

void radix_sort_pairs_host(
    int* keys, void* values, int n, int offset_to_scratch_memory, int begin_bit, int end_bit, int value_size
)
{
    radix_sort_pairs_host_dispatch_value<int, uint32_t>(
        keys, values, n, offset_to_scratch_memory, begin_bit, end_bit, value_size,
        [](int key) { return static_cast<uint32_t>(key) ^ 0x80000000u; }
    );
}

void radix_sort_pairs_host(int* keys, int* values, int n, int begin_bit, int end_bit)
{
    radix_sort_pairs_host(keys, values, n, n, begin_bit, end_bit, sizeof(int));
}

void radix_sort_pairs_host(
    uint32_t* keys, void* values, int n, int offset_to_scratch_memory, int begin_bit, int end_bit, int value_size
)
{
    radix_sort_pairs_host_dispatch_value<uint32_t, uint32_t>(
        keys, values, n, offset_to_scratch_memory, begin_bit, end_bit, value_size, [](uint32_t key) { return key; }
    );
}

void radix_sort_pairs_host(uint32_t* keys, int* values, int n, int begin_bit, int end_bit)
{
    radix_sort_pairs_host(keys, values, n, n, begin_bit, end_bit, sizeof(int));
}

void radix_sort_pairs_host(
    int64_t* keys, void* values, int n, int offset_to_scratch_memory, int begin_bit, int end_bit, int value_size
)
{
    radix_sort_pairs_host_dispatch_value<int64_t, uint64_t>(
        keys, values, n, offset_to_scratch_memory, begin_bit, end_bit, value_size,
        [](int64_t key) { return static_cast<uint64_t>(key) ^ 0x8000000000000000ull; }
    );
}

void radix_sort_pairs_host(int64_t* keys, int* values, int n, int begin_bit, int end_bit)
{
    radix_sort_pairs_host(keys, values, n, n, begin_bit, end_bit, sizeof(int));
}

void radix_sort_pairs_host(
    uint64_t* keys, void* values, int n, int offset_to_scratch_memory, int begin_bit, int end_bit, int value_size
)
{
    radix_sort_pairs_host_dispatch_value<uint64_t, uint64_t>(
        keys, values, n, offset_to_scratch_memory, begin_bit, end_bit, value_size, [](uint64_t key) { return key; }
    );
}

void radix_sort_pairs_host(uint64_t* keys, int* values, int n, int begin_bit, int end_bit)
{
    radix_sort_pairs_host(keys, values, n, n, begin_bit, end_bit, sizeof(int));
}

// http://stereopsis.com/radix.html
inline unsigned int radix_float_to_int(float f)
{
    unsigned int i;
    memcpy(&i, &f, sizeof(i));
    unsigned int mask = (unsigned int)(-(int)(i >> 31)) | 0x80000000;
    return i ^ mask;
}

inline uint64_t radix_double_to_int(double f)
{
    uint64_t i;
    memcpy(&i, &f, sizeof(i));
    uint64_t mask = (uint64_t)(-(int64_t)(i >> 63)) | 0x8000000000000000ull;
    return i ^ mask;
}

void radix_sort_pairs_host(
    float* keys, void* values, int n, int offset_to_scratch_memory, int begin_bit, int end_bit, int value_size
)
{
    radix_sort_pairs_host_dispatch_value<float, uint32_t>(
        keys, values, n, offset_to_scratch_memory, begin_bit, end_bit, value_size,
        [](float key) { return radix_float_to_int(key); }
    );
}

void radix_sort_pairs_host(float* keys, int* values, int n, int begin_bit, int end_bit)
{
    radix_sort_pairs_host(keys, values, n, n, begin_bit, end_bit, sizeof(int));
}

void radix_sort_pairs_host(
    double* keys, void* values, int n, int offset_to_scratch_memory, int begin_bit, int end_bit, int value_size
)
{
    radix_sort_pairs_host_dispatch_value<double, uint64_t>(
        keys, values, n, offset_to_scratch_memory, begin_bit, end_bit, value_size,
        [](double key) { return radix_double_to_int(key); }
    );
}

void radix_sort_pairs_host(double* keys, int* values, int n, int begin_bit, int end_bit)
{
    radix_sort_pairs_host(keys, values, n, n, begin_bit, end_bit, sizeof(int));
}

void segmented_sort_pairs_host(
    float* keys, int* values, int n, int* segment_start_indices, int* segment_end_indices, int num_segments
)
{
    for (int i = 0; i < num_segments; ++i) {
        const int start = segment_start_indices[i];
        const int end = segment_end_indices[i];
        radix_sort_pairs_host(keys + start, values + start, end - start, n, 0, 32, sizeof(int));
    }
}

void segmented_sort_pairs_host(
    int* keys, int* values, int n, int* segment_start_indices, int* segment_end_indices, int num_segments
)
{
    for (int i = 0; i < num_segments; ++i) {
        const int start = segment_start_indices[i];
        const int end = segment_end_indices[i];
        radix_sort_pairs_host(keys + start, values + start, end - start, n, 0, 32, sizeof(int));
    }
}


#if !WP_ENABLE_CUDA

void radix_sort_reserve(void* context, int n, void** mem_out, size_t* size_out, int begin_bit, int end_bit) { }

void wp_radix_sort_pairs_int_device(uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size)
{
}

void wp_radix_sort_pairs_uint_device(uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size)
{
}

void wp_radix_sort_pairs_int64_device(uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size)
{
}

void wp_radix_sort_pairs_uint64_device(
    uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size
)
{
}

void wp_radix_sort_pairs_float_device(uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size)
{
}

void wp_radix_sort_pairs_double_device(
    uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size
)
{
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
}

#endif  // !WP_ENABLE_CUDA


void wp_radix_sort_pairs_int_host(uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size)
{
    radix_sort_pairs_host(
        reinterpret_cast<int*>(keys), reinterpret_cast<void*>(values), n, n, begin_bit, end_bit, value_size
    );
}

void wp_radix_sort_pairs_uint_host(uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size)
{
    radix_sort_pairs_host(
        reinterpret_cast<uint32_t*>(keys), reinterpret_cast<void*>(values), n, n, begin_bit, end_bit, value_size
    );
}

void wp_radix_sort_pairs_int64_host(uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size)
{
    radix_sort_pairs_host(
        reinterpret_cast<int64_t*>(keys), reinterpret_cast<void*>(values), n, n, begin_bit, end_bit, value_size
    );
}

void wp_radix_sort_pairs_uint64_host(uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size)
{
    radix_sort_pairs_host(
        reinterpret_cast<uint64_t*>(keys), reinterpret_cast<void*>(values), n, n, begin_bit, end_bit, value_size
    );
}

void wp_radix_sort_pairs_float_host(uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size)
{
    radix_sort_pairs_host(
        reinterpret_cast<float*>(keys), reinterpret_cast<void*>(values), n, n, begin_bit, end_bit, value_size
    );
}

void wp_radix_sort_pairs_double_host(uint64_t keys, uint64_t values, int n, int begin_bit, int end_bit, int value_size)
{
    radix_sort_pairs_host(
        reinterpret_cast<double*>(keys), reinterpret_cast<void*>(values), n, n, begin_bit, end_bit, value_size
    );
}

void wp_segmented_sort_pairs_float_host(
    uint64_t keys,
    uint64_t values,
    int n,
    uint64_t segment_start_indices,
    uint64_t segment_end_indices,
    int num_segments
)
{
    segmented_sort_pairs_host(
        reinterpret_cast<float*>(keys), reinterpret_cast<int*>(values), n,
        reinterpret_cast<int*>(segment_start_indices), reinterpret_cast<int*>(segment_end_indices), num_segments
    );
}

void wp_segmented_sort_pairs_int_host(
    uint64_t keys,
    uint64_t values,
    int n,
    uint64_t segment_start_indices,
    uint64_t segment_end_indices,
    int num_segments
)
{
    segmented_sort_pairs_host(
        reinterpret_cast<int*>(keys), reinterpret_cast<int*>(values), n, reinterpret_cast<int*>(segment_start_indices),
        reinterpret_cast<int*>(segment_end_indices), num_segments
    );
}
