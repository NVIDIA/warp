/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tile.h"

#if defined(__clang__)
// disable warnings related to C++17 extensions on CPU JIT builds
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif

namespace wp
{


// After this threashold, using segmented_sort from cub is faster
// The threshold must be a power of 2
// The radix sort in this file is consistently slower than the bitonic sort
#define BITONIC_SORT_THRESHOLD 2048

struct UintKeyToUint
{
    inline CUDA_CALLABLE uint32_t convert(uint32 value)
    {
        return value;
    }

    inline CUDA_CALLABLE uint32_t max_possible_key_value()
    {
        return 0xFFFFFFFF;
    }
};

struct IntKeyToUint
{
    inline CUDA_CALLABLE uint32_t convert(int value)
    {
        // Flip the sign bit: ensures negative numbers come before positive numbers
        return static_cast<uint32_t>(value) ^ 0x80000000;
    }

    inline CUDA_CALLABLE int max_possible_key_value()
    {
        return 2147483647;
    }
};

struct FloatKeyToUint
{
    //http://stereopsis.com/radix.html
    inline CUDA_CALLABLE uint32_t convert(float value)
    {
        unsigned int i = reinterpret_cast<unsigned int&>(value);
        unsigned int mask = (unsigned int)(-(int)(i >> 31)) | 0x80000000;
        return i ^ mask;
    }

    inline CUDA_CALLABLE float max_possible_key_value()
    {
        return FLT_MAX;
    }
};

    
constexpr inline CUDA_CALLABLE bool is_power_of_two(int x)
{
    return (x & (x - 1)) == 0;
}

constexpr inline CUDA_CALLABLE int next_higher_pow2(int input)
{
    if (input <= 0) return 1; // Smallest power of 2 is 1

    input--; // Decrement to handle already a power of 2 cases
    input |= input >> 1;
    input |= input >> 2;
    input |= input >> 4;
    input |= input >> 8;
    input |= input >> 16;
    input++; // Next power of 2

    return input;
}


#if defined(__CUDA_ARCH__)


// Bitonic sort fast pass for small arrays

template<typename T>
inline CUDA_CALLABLE T shfl_xor(unsigned int thread_id, T* sh_mem, unsigned int lane_mask)
{
    unsigned int source_lane = thread_id ^ lane_mask;        
    return sh_mem[source_lane];
}

template<typename K, typename V, int num_loops>
inline CUDA_CALLABLE void bitonic_sort_single_stage_full_thread_block(int k, unsigned int thread_id, unsigned int stride, K* key_sh_mem, V* val_sh_mem, int length, K max_key_value, 
    K* key_register, V* val_register)
{
    __syncthreads();
    #pragma unroll
    for (int loop_id = 0; loop_id < num_loops; ++loop_id)
    {
        int thread_id2 = loop_id * WP_TILE_BLOCK_DIM + thread_id;

        key_register[loop_id] = thread_id2 < length ? key_sh_mem[thread_id2] : max_key_value;
        val_register[loop_id] = thread_id2 < length ? val_sh_mem[thread_id2] : 0;        
    }
    
    __syncthreads();

    K s_key[num_loops];
    V s_val[num_loops];
    bool swap[num_loops];
    #pragma unroll
    for (int loop_id = 0; loop_id < num_loops; ++loop_id)      
    {
        int thread_id2 = loop_id * WP_TILE_BLOCK_DIM + thread_id;

        if(thread_id2 < length)
        {
            s_key[loop_id] = shfl_xor(thread_id2, key_sh_mem, stride);
            s_val[loop_id] = shfl_xor(thread_id2, val_sh_mem, stride);
            swap[loop_id] = (((thread_id2 & stride) != 0 ? key_register[loop_id] > s_key[loop_id] : key_register[loop_id] < s_key[loop_id])) ^ ((thread_id2 & k) == 0);
        }
    }

    __syncthreads();

    #pragma unroll
    for (int loop_id = 0; loop_id < num_loops; ++loop_id)
    {
        int thread_id2 = loop_id * WP_TILE_BLOCK_DIM + thread_id;
        if (thread_id2 < length)
        {
            key_sh_mem[thread_id2] = swap[loop_id] ? s_key[loop_id] : key_register[loop_id];
            val_sh_mem[thread_id2] = swap[loop_id] ? s_val[loop_id] : val_register[loop_id];
        }
    }
    __syncthreads();
}

//stride can be 1, 2, 4, 8, 16
template<typename K, typename V>
inline CUDA_CALLABLE void bitonic_sort_single_stage_full_warp(int k, unsigned int thread_id, int stride, K& key, V& val)
{
    auto s_key = __shfl_xor_sync(0xFFFFFFFFu, key, stride);
    auto s_val = __shfl_xor_sync(0xFFFFFFFFu, val, stride);
    auto swap = (((thread_id & stride) != 0 ? key > s_key : key < s_key)) ^ ((thread_id & k) == 0);
    key = swap ? s_key : key;
    val = swap ? s_val : val;
}


//Sorts 32 elements according to keys
template<typename K, typename V>
inline CUDA_CALLABLE void bitonic_sort_single_warp(unsigned int thread_id, K& key, V& val)
{
#pragma unroll
    for (int k = 2; k <= 32; k <<= 1)
    {
#pragma unroll
        for (int stride = k / 2; stride > 0; stride >>= 1)
        {
            bitonic_sort_single_stage_full_warp(k, thread_id, stride, key, val);
        }
    }
}

template<typename K, typename V, typename KeyToUint>
inline CUDA_CALLABLE void bitonic_sort_single_warp(int thread_id, 
    K* keys_input, 
    V* values_input,
    int num_elements_to_sort)
{
    KeyToUint key_converter; 

    __syncwarp();

    K key = thread_id < num_elements_to_sort ? keys_input[thread_id] : key_converter.max_possible_key_value();
    V value;
    if(thread_id < num_elements_to_sort)
        value = values_input[thread_id];

    __syncwarp();
    bitonic_sort_single_warp(thread_id, key, value);
    __syncwarp();

    if(thread_id < num_elements_to_sort)
    {
        keys_input[thread_id] = key;
        values_input[thread_id] = value;
    }
    __syncwarp();
}


//Sorts according to keys
template<int max_num_elements, typename K, typename V>
inline CUDA_CALLABLE void bitonic_sort_pow2_length(unsigned int thread_id, K* key_sh_mem, V* val_sh_mem, int length, K key_max_possible_value)
{
    constexpr int num_loops = (max_num_elements + WP_TILE_BLOCK_DIM - 1) / WP_TILE_BLOCK_DIM;
    K key[num_loops];
    V val[num_loops];

    #pragma unroll
    for (int loop_id = 0; loop_id < num_loops; ++loop_id)
    {
        int thread_id2 = loop_id * WP_TILE_BLOCK_DIM + thread_id;
        key[loop_id] = thread_id2 < length ? key_sh_mem[thread_id2] : key_max_possible_value;
        if (thread_id2 < length)
            val[loop_id] = val_sh_mem[thread_id2];
    }

    __syncthreads();
    bool full_block_sort_active = false;    

    for (int k = 2; k <= length; k <<= 1)
    {        
        for (int stride = k / 2; stride > 0; stride >>= 1)
        {
            if (stride <= 16) //no inter-warp communication needed up to stride 16
            {
                if(full_block_sort_active)
                {
                    __syncthreads();
                    #pragma unroll
                    for (int loop_id = 0; loop_id < num_loops; ++loop_id)
                    {
                        int thread_id2 = loop_id * WP_TILE_BLOCK_DIM + thread_id;

                        //Switch from shared mem to registers
                        if (thread_id2 < length)
                        {
                            key[loop_id] = key_sh_mem[thread_id2];
                            val[loop_id] = val_sh_mem[thread_id2];
                        }
                    }
                    full_block_sort_active = false;
                    __syncthreads();
                }

                #pragma unroll
                for (int loop_id = 0; loop_id < num_loops; ++loop_id)
                {
                    int thread_id2 = loop_id * WP_TILE_BLOCK_DIM + thread_id;
                    bitonic_sort_single_stage_full_warp(k, thread_id2, stride, key[loop_id], val[loop_id]);
                }
            }
            else
            {
                if (!full_block_sort_active)
                {
                    __syncthreads();
                    #pragma unroll
                    for (int loop_id = 0; loop_id < num_loops; ++loop_id)
                    {
                        int thread_id2 = loop_id * WP_TILE_BLOCK_DIM + thread_id;

                        //Switch from registers t0 shared mem
                        if (thread_id2 < length)
                        {
                            key_sh_mem[thread_id2] = key[loop_id];
                            val_sh_mem[thread_id2] = val[loop_id];
                        }
                    }
                    full_block_sort_active = true;
                    __syncthreads();
                }
                
                bitonic_sort_single_stage_full_thread_block<K, V, num_loops>(k, thread_id, (unsigned int)stride, key_sh_mem, val_sh_mem, length, key_max_possible_value, key, val);                
            }            
        }
    }

    if (!full_block_sort_active)
    {
        #pragma unroll
        for (int loop_id = 0; loop_id < num_loops; ++loop_id)
        {
            int thread_id2 = loop_id * WP_TILE_BLOCK_DIM + thread_id;
            //Switch from registers t0 shared mem
            if (thread_id2 < length)
            {
                key_sh_mem[thread_id2] = key[loop_id];
                val_sh_mem[thread_id2] = val[loop_id];
            }
        }
        full_block_sort_active = true;
        __syncthreads();
    }
}

//Allocates shared memory to buffer the arrays that need to be sorted
template <int max_num_elements, typename K, typename V, typename KeyToUint>
inline CUDA_CALLABLE void bitonic_sort_thread_block_shared_mem(
    int thread_id, 
    K* keys_input, 
    V* values_input,
    int num_elements_to_sort)
{
    if constexpr(max_num_elements < 32)
    {
        //Fast track - single warp sort
        if (thread_id < 32)
            bitonic_sort_single_warp<K, V, KeyToUint>(thread_id, keys_input, values_input, num_elements_to_sort);
        __syncthreads();
    }
    else
    {
        KeyToUint key_converter; 
        const K key_max_possible_value = key_converter.max_possible_key_value();

        constexpr int shared_mem_count = next_higher_pow2(max_num_elements);

        __shared__ K keys_shared_mem[shared_mem_count]; //TODO: This shared memory can be avoided if keys_input is already shared memory
        __shared__ V values_shared_mem[shared_mem_count]; //TODO: This shared memory can be avoided if values_input is already shared memory

        for(int i = thread_id; i < shared_mem_count; i += WP_TILE_BLOCK_DIM)
        {
            if (i < num_elements_to_sort)
            {  
                keys_shared_mem[i] = keys_input[i];
                values_shared_mem[i] = values_input[i];
            }
            else
                keys_shared_mem[i] = key_max_possible_value;
        }
        __syncthreads();

        bitonic_sort_pow2_length<shared_mem_count, K, V>((unsigned int)thread_id, keys_shared_mem, values_shared_mem, shared_mem_count, key_max_possible_value);

        __syncthreads();

        for (int i = thread_id; i < num_elements_to_sort; i += WP_TILE_BLOCK_DIM)
        {
            keys_input[i] = keys_shared_mem[i];
            values_input[i] = values_shared_mem[i];            
        }
        __syncthreads();
    }
}


// Specialization for int keys
template <int max_num_elements, typename V>
inline CUDA_CALLABLE void bitonic_sort_thread_block_shared_mem(
    int thread_id,
    int* keys_input,
    V* values_input, 
    int num_elements_to_sort)
{
    bitonic_sort_thread_block_shared_mem<max_num_elements, int, V, IntKeyToUint>(
        thread_id, keys_input, values_input, num_elements_to_sort);
}

// Specialization for unsigned int keys
template <int max_num_elements, typename V>
inline CUDA_CALLABLE void bitonic_sort_thread_block_shared_mem(
    int thread_id,
    unsigned int* keys_input,
    V* values_input,
    int num_elements_to_sort)
{
    bitonic_sort_thread_block_shared_mem<max_num_elements, unsigned int, V, UintKeyToUint>(
        thread_id, keys_input, values_input, num_elements_to_sort);
}

// Specialization for float keys
template <int max_num_elements, typename V>
inline CUDA_CALLABLE void bitonic_sort_thread_block_shared_mem(
    int thread_id,
    float* keys_input,
    V* values_input,
    int num_elements_to_sort)
{
    bitonic_sort_thread_block_shared_mem<max_num_elements, float, V, FloatKeyToUint>(
        thread_id, keys_input, values_input, num_elements_to_sort);
}



// Ideally keys_input and values_input point into fast memory (shared memory)
template <int max_num_elements, typename K, typename V, typename KeyToUint>
inline CUDA_CALLABLE void bitonic_sort_thread_block_direct(
    int thread_id, 
    K* keys_input, 
    V* values_input,
    int num_elements_to_sort)
{
    if constexpr(max_num_elements < 32)
    {
        //Fast track - single warp sort
        if (thread_id < 32)
            bitonic_sort_single_warp<K, V, KeyToUint>(thread_id, keys_input, values_input, num_elements_to_sort);
        __syncthreads();
    }
    else
    {
        assert(num_elements_to_sort <= num_threads);

        KeyToUint key_converter; 
        const K key_max_possible_value = key_converter.max_possible_key_value();
        
        bitonic_sort_pow2_length<max_num_elements, K, V>((unsigned int)thread_id, keys_input, values_input, num_elements_to_sort, key_max_possible_value);
    }
}

// Specialization for int keys
template <int max_num_elements, typename V>
inline CUDA_CALLABLE void bitonic_sort_thread_block_direct(
    int thread_id,
    int* keys_input,
    V* values_input, 
    int num_elements_to_sort)
{
    bitonic_sort_thread_block_direct<max_num_elements, int, V, IntKeyToUint>(
        thread_id, keys_input, values_input, num_elements_to_sort);
}

// Specialization for unsigned int keys
template <int max_num_elements, typename V>
inline CUDA_CALLABLE void bitonic_sort_thread_block_direct(
    int thread_id,
    unsigned int* keys_input,
    V* values_input,
    int num_elements_to_sort)
{
    bitonic_sort_thread_block_direct<max_num_elements, unsigned int, V, UintKeyToUint>(
        thread_id, keys_input, values_input, num_elements_to_sort);
}

// Specialization for float keys
template <int max_num_elements, typename V>
inline CUDA_CALLABLE void bitonic_sort_thread_block_direct(
    int thread_id,
    float* keys_input,
    V* values_input,
    int num_elements_to_sort)
{
    bitonic_sort_thread_block_direct<max_num_elements, float, V, FloatKeyToUint>(
        thread_id, keys_input, values_input, num_elements_to_sort);
}

// End bitonic sort

inline CUDA_CALLABLE int warp_scan_inclusive(int lane, unsigned int ballot_mask) 
{
    uint32_t mask = ((1u << (lane + 1)) - 1);
    return __popc(ballot_mask & mask);
}

inline CUDA_CALLABLE int warp_scan_inclusive(int lane, unsigned int mask, bool thread_contributes_element)
{
    return warp_scan_inclusive(lane, __ballot_sync(mask, thread_contributes_element));
}

template<typename T>
inline CUDA_CALLABLE T warp_scan_inclusive(int lane, T value)
{
//Computes an inclusive cumulative sum
#pragma unroll
    for (int i = 1; i <= 32; i *= 2)
    {
        auto n = __shfl_up_sync(0xffffffffu, value, i, 32);

        if (lane >= i)
            value = value + n;
    }
    return value;
}

template<typename T>
inline CUDA_CALLABLE T warp_scan_exclusive(int lane, T value)
{
    T scan = warp_scan_inclusive(lane, value);   
    return scan - value;
}

template <int num_warps, int num_threads, typename K, typename V, typename KeyToUint>
inline CUDA_CALLABLE void radix_sort_thread_block_core(
    int thread_id, 
    K* keys_input, K* keys_tmp,
    V* values_input, V* values_tmp, 
    int num_elements_to_sort)
{
    KeyToUint key_converter;   

    int num_bits_to_sort = 32; //Sort all bits because that's what the bitonic fast pass does as well

    const int warp_id = thread_id / 32;
    const int lane_id = thread_id & 31;

    const int bits_per_pass = 4; //Higher than 5 is currently not supported - 2^5=32 is the warp size and is still just fine
    const int lowest_bits_mask = (1 << bits_per_pass) - 1;
    const int num_scan_buckets = (1 << bits_per_pass);

    const int num_warp_passes = (num_scan_buckets + num_warps - 1) / num_warps;

    __shared__ int buckets[num_scan_buckets];
    __shared__ int buckets2[num_scan_buckets];
    __shared__ int buckets_cumulative_sum[num_scan_buckets];
    __shared__ int shared_mem[num_warps][num_scan_buckets];

    const int num_passes = (num_bits_to_sort + bits_per_pass - 1) / bits_per_pass;
    const int num_inner_loops = (num_elements_to_sort + num_threads - 1) / num_threads;

    for (int pass_id = 0; pass_id < num_passes; ++pass_id)
    {
        __syncthreads();
        if (thread_id < num_scan_buckets)
        {
            buckets[lane_id] = 0;
            buckets2[lane_id] = 0;
        }
        __syncthreads();

        int shift = pass_id * bits_per_pass;

        for (int j = thread_id; j < num_inner_loops * num_threads; j += num_threads)
        {
            int digit = j < num_elements_to_sort ? (int)((key_converter.convert(keys_input[j]) >> shift) & lowest_bits_mask) : num_scan_buckets;

            for (int b = 0; b < num_scan_buckets; b++)
            {
                bool contributes = digit == b;
                int sum_per_warp = warp_scan_inclusive(lane_id, 0xFFFFFFFF, contributes);

                if (lane_id == 31)
                    shared_mem[warp_id][b] = sum_per_warp;
            }
            __syncthreads();

            for(int b=warp_id;b< num_warp_passes * num_warps;b += num_warps)
            {
                int f = lane_id < num_warps ? shared_mem[lane_id][b] : 0;
                f = warp_scan_inclusive(lane_id, f);
                if (lane_id == 31)
                    buckets[b] += f;
            }
            __syncthreads();
        }

#if VALIDATE_SORT
        if (thread_id == 0)
        {
            for (int b = 0; b < num_scan_buckets; b++)
            {
                int bucket_sum = 0;
                for (int j = 0; j < num_elements_to_sort; j++)
                {
                    int digit = j < num_elements_to_sort ? (int)((key_converter.convert(keys_input[j]) >> shift) & lowest_bits_mask) : num_scan_buckets;
                    if (digit == b)
                        ++bucket_sum;
                }
                assert(buckets[b] == bucket_sum);
            }
        }
        __syncthreads();
#endif

        if (warp_id == 0)
        {
            int value = lane_id < num_scan_buckets ? buckets[lane_id] : 0;                
            value = warp_scan_exclusive(lane_id, value);
            if (lane_id < num_scan_buckets)
                buckets_cumulative_sum[lane_id] = value;

            if (lane_id == num_scan_buckets - 1)
                assert(debug + value == num_elements_to_sort);
        }

        __syncthreads();

#if VALIDATE_SORT
        if(thread_id == 0)
        {
            for (int b = 0; b < num_scan_buckets; b++)
            {
                int bucket_sum = 0;
                for(int j=0; j<num_elements_to_sort; j++)
                {
                    int digit = j < num_elements_to_sort ? (int)((key_converter.convert(keys_input[j]) >> shift) & lowest_bits_mask) : num_scan_buckets;
                    if (digit == b)
                        ++bucket_sum;
                }
                assert(buckets[b] == bucket_sum);
            }

            int exclusive_bucket_sum = 0;
            for (int b = 0; b < num_scan_buckets; b++)
            {
                assert(exclusive_bucket_sum == buckets_cumulative_sum[b]);
                exclusive_bucket_sum += buckets[b];
            }
            assert(exclusive_bucket_sum == num_elements_to_sort);
        }
        __syncthreads();
#endif

        //Now buckets holds numBuckets inclusive cumulative sums (e. g. 16 sums for 4 bit radix sort - 2^4=16)
        //The problem is that we either store local_offset_per_thread for every element array (potentially many) or we recompute it again
        for (int j = thread_id; j < num_inner_loops * num_threads; j += num_threads)
        {
            int digit = j < num_elements_to_sort ? (int)((key_converter.convert(keys_input[j]) >> shift) & lowest_bits_mask) : num_scan_buckets;

            int local_offset_per_thread = 0;

            for (int b = 0; b < num_scan_buckets; b++)
            {
                bool contributes = digit == b;
                int sum_per_warp = warp_scan_inclusive(lane_id, 0xFFFFFFFF, contributes);
                if (lane_id == 31)
                    shared_mem[warp_id][b] = sum_per_warp;

                if (contributes)
                    local_offset_per_thread = sum_per_warp - 1; //-1 because of inclusive scan and local_offset_per_thread needs exclusive scan
            }

            for (int b = 0; b < num_scan_buckets; b++)
            {
                __syncthreads();
                int global_offset = buckets2[b];
                __syncthreads();

                int f = lane_id < num_warps ? shared_mem[lane_id][b] : 0;
                int inclusive_scan = warp_scan_inclusive(lane_id, f);
                if (lane_id == 31 && warp_id == 0)
                {
                    buckets2[b] += inclusive_scan;
                }

                int warp_offset = __shfl_sync(0xFFFFFFFF, inclusive_scan - f, warp_id); //-f because warp_offset needs to be an exclusive scan

                bool contributes = digit == b;
                if (contributes)
                {
                    local_offset_per_thread += global_offset + warp_offset; 

#if VALIDATE_SORT
                    int curr = buckets_cumulative_sum[b];
                    int next = b + 1 < num_scan_buckets ? buckets_cumulative_sum[b + 1] : num_elements_to_sort;
                    assert(local_offset_per_thread < next - curr && local_offset_per_thread >= 0);
#endif
                }
            }
            __syncthreads();

            if (j < num_elements_to_sort)
            {
                int final_offset = buckets_cumulative_sum[digit] + local_offset_per_thread;

                keys_tmp[final_offset] = keys_input[j];
                values_tmp[final_offset] = values_input[j];
            }
        }

        __syncthreads();

#if VALIDATE_SORT
        for (int j = thread_id; j < num_inner_loops * num_threads; j += num_threads)
        {
            if(j>0 && j < num_elements_to_sort)
            {
                int digit1 =  (int)((keys_tmp[j-1] >> shift) & lowest_bits_mask);
                int digit2 = (int)((keys_tmp[j] >> shift) & lowest_bits_mask);

                assert(digit1<=digit2);
            }
        }
        __syncthreads();
#endif

        auto tmp = keys_tmp;
        keys_tmp = keys_input;
        keys_input = tmp;

        auto tmp2 = values_tmp;
        values_tmp = values_input;
        values_input = tmp2;
    }

    //For odd number of passes, the result is the const& wrong array - copy it over
    if (num_passes % 2 != 0)
    {
        for (int j = thread_id; j < num_inner_loops * num_threads; j += num_threads)
        {
            if (j < num_elements_to_sort)
            {
                keys_tmp[j] = keys_input[j];
                values_tmp[j] = values_input[j];
            }
        }

        auto tmp = keys_tmp;
        keys_tmp = keys_input;
        keys_input = tmp;

        auto tmp2 = values_tmp;
        values_tmp = values_input;
        values_input = tmp2;
    }    
}




template <int num_warps, int num_threads, typename V>
inline CUDA_CALLABLE void radix_sort_thread_block(
    int thread_id, 
    int* keys_input, int* keys_tmp, 
    V* values_input, V* values_tmp, 
    int num_elements_to_sort)
{
    radix_sort_thread_block_core<num_warps, num_threads, int, V, IntKeyToUint>(
        thread_id, keys_input, keys_tmp,
        values_input, values_tmp, num_elements_to_sort);
}

template <int num_warps, int num_threads, typename V>
inline CUDA_CALLABLE void radix_sort_thread_block(
    int thread_id, 
    unsigned int* keys_input, unsigned int* keys_tmp, 
    V* values_input, V* values_tmp, 
    int num_elements_to_sort)
{
    radix_sort_thread_block_core<num_warps, num_threads, unsigned int, V, UintKeyToUint>(
        thread_id, keys_input, keys_tmp, 
        values_input, values_tmp, 
        num_elements_to_sort);
}

template <int num_warps, int num_threads, typename V>
inline CUDA_CALLABLE void radix_sort_thread_block(
    int thread_id, 
    float* keys_input, float* keys_tmp, 
    V* values_input, V* values_tmp, 
    int num_elements_to_sort)
{
    radix_sort_thread_block_core<num_warps, num_threads, float, V, FloatKeyToUint>(
        thread_id, keys_input, keys_tmp, 
        values_input, values_tmp, 
        num_elements_to_sort);
}


template <typename TileK, typename TileV>
void tile_sort(TileK& t, TileV& t2)
{ 
    using T = typename TileK::Type;
    using V = typename TileV::Type;
  
    constexpr int num_elements_to_sort = TileK::Layout::Shape::size(); 
    T* keys = &t.data(0);
    V* values = &t2.data(0);

    //Trim away the code that won't be used - possible because the number of elements to sort is known at compile time
    if constexpr (num_elements_to_sort <= BITONIC_SORT_THRESHOLD)
    {
        if constexpr(is_power_of_two(num_elements_to_sort))  
            bitonic_sort_thread_block_direct<num_elements_to_sort, V>(WP_TILE_THREAD_IDX, keys, values, num_elements_to_sort);
        else
            bitonic_sort_thread_block_shared_mem<num_elements_to_sort, V>(WP_TILE_THREAD_IDX, keys, values, num_elements_to_sort);
    }
    else
    {
        __shared__ T keys_tmp[num_elements_to_sort];
        __shared__ V values_tmp[num_elements_to_sort];

        constexpr int warp_count = (WP_TILE_BLOCK_DIM + WP_TILE_WARP_SIZE - 1)/WP_TILE_WARP_SIZE;
        
        radix_sort_thread_block<warp_count, WP_TILE_BLOCK_DIM, V>(WP_TILE_THREAD_IDX, keys, keys_tmp, 
            values, values_tmp, num_elements_to_sort);
    }

    WP_TILE_SYNC();
}

template <typename TileK, typename TileV>
void tile_sort(TileK& t, TileV& t2, int start, int length)
{ 
    using T = typename TileK::Type;
    using V = typename TileV::Type;
  
    constexpr int max_elements_to_sort = TileK::Layout::Shape::size(); 
    const int num_elements_to_sort = length; 
    T* keys = &t.data(start);
    V* values = &t2.data(start);

    if (num_elements_to_sort <= BITONIC_SORT_THRESHOLD)
    {
        if (is_power_of_two(num_elements_to_sort)) 
            bitonic_sort_thread_block_direct<max_elements_to_sort, V>(WP_TILE_THREAD_IDX, keys, values, num_elements_to_sort);
        else
            bitonic_sort_thread_block_shared_mem<max_elements_to_sort, V>(WP_TILE_THREAD_IDX, keys, values, num_elements_to_sort);    
    }
    else
    {
        if constexpr (max_elements_to_sort > BITONIC_SORT_THRESHOLD)
        {
            __shared__ T keys_tmp[max_elements_to_sort];
            __shared__ V values_tmp[max_elements_to_sort];

            constexpr int warp_count = (WP_TILE_BLOCK_DIM + WP_TILE_WARP_SIZE - 1)/WP_TILE_WARP_SIZE;
            
            radix_sort_thread_block<warp_count, WP_TILE_BLOCK_DIM, V>(WP_TILE_THREAD_IDX, keys, keys_tmp, 
                values, values_tmp, num_elements_to_sort);
        }
    }

    WP_TILE_SYNC();
}

#else

// CPU implementation

template <typename K>
void swap_elements(K& a, K& b)
{
    K tmp = a;
    a = b;
    b = tmp;
}

// length must be a power of two
template <typename K, typename V>
void bitonic_sort_pairs_pow2_length_cpu(K* keys, V* values, int length)
{
    for (int k = 2; k <= length; k *= 2) 
    {
        for (int stride = k / 2; stride > 0; stride /= 2) 
        {
            for (int i = 0; i < length; i++) 
            {
                int swap_idx = i ^ stride;
                if (swap_idx > i) 
                {
                    bool ascending = ((i & k) == 0);
                    if ((ascending && keys[i] > keys[swap_idx]) || (!ascending && keys[i] < keys[swap_idx])) 
                    {
                        swap_elements(keys[i], keys[swap_idx]);
                        swap_elements(values[i], values[swap_idx]);
                    }
                }
            }
        }
    }
}

template <typename K, typename V, int max_size, typename KeyToUint>
void bitonic_sort_pairs_general_size_cpu(K* keys, V* values, int length)
{
    constexpr int pow2_size = next_higher_pow2(max_size);

    K keys_tmp[pow2_size];
    V values_tmp[pow2_size];

    KeyToUint converter;
    K max_key = converter.max_possible_key_value();

    for(int i=0; i<pow2_size; ++i)
    {
        keys_tmp[i] = i < length ? keys[i] : max_key;
        if(i < length)
            values_tmp[i] = values[i];
    }

    bitonic_sort_pairs_pow2_length_cpu(keys_tmp, values_tmp, pow2_size);

    for(int i=0; i<length; ++i)
    {
        keys[i] = keys_tmp[i];
        values[i] = values_tmp[i];
    }
}

template <typename V, int max_size>
void bitonic_sort_pairs_general_size_cpu(unsigned int* keys, V* values, int length)
{
    bitonic_sort_pairs_general_size_cpu<unsigned int, V, max_size, UintKeyToUint>(keys, values, length);
}

template <typename V, int max_size>
void bitonic_sort_pairs_general_size_cpu(int* keys, V* values, int length)
{
    bitonic_sort_pairs_general_size_cpu<int, V, max_size, IntKeyToUint>(keys, values, length);
}

template <typename V, int max_size>
void bitonic_sort_pairs_general_size_cpu(float* keys, V* values, int length)
{
    bitonic_sort_pairs_general_size_cpu<float, V, max_size, FloatKeyToUint>(keys, values, length);
}



template <typename K, typename V, typename KeyToUint>
void radix_sort_pairs_cpu_core(K* keys, K* aux_keys, V* values, V* aux_values, int n)
{
    KeyToUint converter;
	static unsigned int tables[2][1 << 16];
	memset(tables, 0, sizeof(tables));
	
	// build histograms
	for (int i=0; i < n; ++i)
	{
		const unsigned int k = converter.convert(keys[i]);
		const unsigned short low = k & 0xffff;
		const unsigned short high = k >> 16;
		
		++tables[0][low];
		++tables[1][high];
	}
	
	// convert histograms to offset tables in-place
	unsigned int offlow = 0;
	unsigned int offhigh = 0;
	
	for (int i=0; i < 65536; ++i)
	{
		const unsigned int newofflow = offlow + tables[0][i];
		const unsigned int newoffhigh = offhigh + tables[1][i];
		
		tables[0][i] = offlow;
		tables[1][i] = offhigh;
		
		offlow = newofflow;
		offhigh = newoffhigh;
	}
		
	// pass 1 - sort by low 16 bits
	for (int i=0; i < n; ++i)
	{
		// lookup offset of input
		const K f = keys[i];
		const unsigned int k = converter.convert(f);
		const V v = values[i];
		const unsigned int b = k & 0xffff;
		
		// find offset and increment
		const unsigned int offset = tables[0][b]++;
		
		aux_keys[offset] = f;
		aux_values[offset] = v;
	}	
		
	// pass 2 - sort by high 16 bits
	for (int i=0; i < n; ++i)
	{
		// lookup offset of input
		const K f = aux_keys[i];
		const unsigned int k = converter.convert(f);
		const V v = aux_values[i];

		const unsigned int b = k >> 16;
		
		const unsigned int offset = tables[1][b]++;
		
		keys[offset] = f;
		values[offset] = v;
	}	
}

template <typename V>
inline void radix_sort_pairs_cpu(
    int* keys_input,
    int* keys_aux,
    V* values_input,
    V* values_aux,
    int num_elements_to_sort) 
{
    radix_sort_pairs_cpu_core<int, V, IntKeyToUint>(
        keys_input, keys_aux,
        values_input, values_aux,
        num_elements_to_sort);
}

template <typename V>
inline void radix_sort_pairs_cpu(
    unsigned int* keys_input,
    unsigned int* keys_aux,
    V* values_input,
    V* values_aux,
    int num_elements_to_sort)
{
    radix_sort_pairs_cpu_core<unsigned int, V, UintKeyToUint>(
        keys_input, keys_aux,
        values_input, values_aux,
        num_elements_to_sort);
}

template <typename V>
inline void radix_sort_pairs_cpu(
    float* keys_input,
    float* keys_aux,
    V* values_input,
    V* values_aux,
    int num_elements_to_sort)
{
    radix_sort_pairs_cpu_core<float, V, FloatKeyToUint>(
        keys_input, keys_aux,
        values_input, values_aux,
        num_elements_to_sort);
}



template <typename TileK, typename TileV>
void tile_sort(TileK& t, TileV& t2)
{ 
    using T = typename TileK::Type;
    using V = typename TileV::Type;
  
    constexpr int num_elements_to_sort = TileK::Layout::Shape::size(); 
    T* keys = &t.data(0);
    V* values = &t2.data(0);

    //Trim away the code that won't be used - possible because the number of elements to sort is known at compile time
    if constexpr (num_elements_to_sort <= BITONIC_SORT_THRESHOLD)
    {
        if constexpr(is_power_of_two(num_elements_to_sort))        
            bitonic_sort_pairs_pow2_length_cpu<T, V>(keys, values, num_elements_to_sort);        
        else        
            bitonic_sort_pairs_general_size_cpu<V, num_elements_to_sort>(keys, values, num_elements_to_sort);            
    }
    else
    {
        T keys_tmp[num_elements_to_sort];
        V values_tmp[num_elements_to_sort];

        radix_sort_pairs_cpu<V>(keys, keys_tmp, values, values_tmp, num_elements_to_sort);
    }

    WP_TILE_SYNC();
}

template <typename TileK, typename TileV>
void tile_sort(TileK& t, TileV& t2, int start, int length)
{ 
    using T = typename TileK::Type;
    using V = typename TileV::Type;
  
    constexpr int max_elements_to_sort = TileK::Layout::Shape::size(); 
    const int num_elements_to_sort = length; 
    T* keys = &t.data(start);
    V* values = &t2.data(start);

    if (num_elements_to_sort <= BITONIC_SORT_THRESHOLD)
    {
        if (is_power_of_two(num_elements_to_sort))        
            bitonic_sort_pairs_pow2_length_cpu<T, V>(keys, values, num_elements_to_sort);        
        else        
            bitonic_sort_pairs_general_size_cpu<V, max_elements_to_sort>(keys, values, num_elements_to_sort);        
    }
    else
    {
        if constexpr (max_elements_to_sort > BITONIC_SORT_THRESHOLD)
        {
            T keys_tmp[max_elements_to_sort];
            V values_tmp[max_elements_to_sort];

            radix_sort_pairs_cpu<V>(keys, keys_tmp, values, values_tmp, num_elements_to_sort);
        }
    }

    WP_TILE_SYNC();
}


#endif // !defined(__CUDA_ARCH__)


template <typename TileK, typename TileV>
inline void adj_tile_sort(TileK& t, TileV& t2, TileK& adj_t1, TileV& adj_t2) 
{
    // todo: general purpose sort gradients not implemented 
}

template <typename TileK, typename TileV>
inline void adj_tile_sort(TileK& t, TileV& t2, int start, int length, TileK& adj_t1, TileV& adj_t2, int adj_start, int adj_length) 
{
    // todo: general purpose sort gradients not implemented 
}

} // namespace wp

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
