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

#include "warp.h"
#include "sort.h"
#include "string.h"

#include <cstdint>

//Only integer keys (bit count 32 or 64) are supported. Floats need to get converted into int first. see radix_float_to_int.
template <typename KeyType>
void radix_sort_pairs_host(KeyType* keys, int* values, int n, int offset_to_scratch_memory)
{
	const int numPasses = sizeof(KeyType) / 2;
	static int tables[numPasses][1 << 16];
	memset(tables, 0, sizeof(tables));
	
	// build histograms
	for (int p = 0; p < numPasses; ++p)
    {
		for (int i=0; i < n; ++i)
		{
			const int shift = p * 16;
			const int b = (keys[i] >> shift) & 0xffff;

			++tables[p][b];
		}
	}
	
	// convert histograms to offset tables in-place	
	for (int p = 0; p < numPasses; ++p)
	{
		int off = 0;
		for (int i = 0; i < 65536; ++i)
		{
			const int newoff = off + tables[p][i];
			
			tables[p][i] = off;
			
			off = newoff;
		}
	}
	
    for (int p = 0; p < numPasses; ++p)
    {
		int flipFlop = p % 2;
		KeyType* readKeys = keys + offset_to_scratch_memory * flipFlop;
		int* readValues = values + offset_to_scratch_memory * flipFlop;
		KeyType* writeKeys = keys + offset_to_scratch_memory * (1 - flipFlop);
		int* writeValues = values + offset_to_scratch_memory * (1 - flipFlop);

		// pass 1 - sort by low 16 bits
		for (int i=0; i < n; ++i)
		{
			// lookup offset of input
			const KeyType k = readKeys[i];
			const int v = readValues[i];
			
			const int shift = p * 16;
			const int b = (k >> shift) & 0xffff;
			
			// find offset and increment
			const int offset = tables[p][b]++;
			
			writeKeys[offset] = k;
			writeValues[offset] = v;
		}
	}
}

void radix_sort_pairs_host(int* keys, int* values, int n)
{
	radix_sort_pairs_host<int>(keys, values, n, n);
}

void radix_sort_pairs_host(int64_t* keys, int* values, int n)
{
	radix_sort_pairs_host<int64_t>(keys, values, n, n);
}

 //http://stereopsis.com/radix.html
inline unsigned int radix_float_to_int(float f)
{
	unsigned int i = reinterpret_cast<unsigned int&>(f);
	unsigned int mask = (unsigned int)(-(int)(i >> 31)) | 0x80000000;
	return i ^ mask;
}

void radix_sort_pairs_host(float* keys, int* values, int n, int offset_to_scratch_memory)
{
	static unsigned int tables[2][1 << 16];
	memset(tables, 0, sizeof(tables));
		
	float* auxKeys = keys + offset_to_scratch_memory;
	int* auxValues = values + offset_to_scratch_memory;

	// build histograms
	for (int i=0; i < n; ++i)
	{
		const unsigned int k = radix_float_to_int(keys[i]);
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
		const float f = keys[i];
		const unsigned int k = radix_float_to_int(f);
		const int v = values[i];
		const unsigned int b = k & 0xffff;
		
		// find offset and increment
		const unsigned int offset = tables[0][b]++;
		
		auxKeys[offset] = f;
		auxValues[offset] = v;
	}	
		
	// pass 2 - sort by high 16 bits
	for (int i=0; i < n; ++i)
	{
		// lookup offset of input
		const float f = auxKeys[i];
		const unsigned int k = radix_float_to_int(f);
		const int v = auxValues[i];

		const unsigned int b = k >> 16;
		
		const unsigned int offset = tables[1][b]++;
		
		keys[offset] = f;
		values[offset] = v;
	}	
}

void radix_sort_pairs_host(float* keys, int* values, int n)
{
	radix_sort_pairs_host(keys, values, n, n);
}

void segmented_sort_pairs_host(float* keys, int* values, int n, int* segment_start_indices, int* segment_end_indices, int num_segments)
{
	for (int i = 0; i < num_segments; ++i)
	{
		const int start = segment_start_indices[i];
		const int end = segment_end_indices[i];
		radix_sort_pairs_host(keys + start, values + start, end - start, n);
	}
}

void segmented_sort_pairs_host(int* keys, int* values, int n, int* segment_start_indices, int* segment_end_indices, int num_segments)
{
	for (int i = 0; i < num_segments; ++i)
	{
		const int start = segment_start_indices[i];
		const int end = segment_end_indices[i];
		radix_sort_pairs_host(keys + start, values + start, end - start, n);
	}
}


#if !WP_ENABLE_CUDA

void radix_sort_reserve(void* context, int n, void** mem_out, size_t* size_out) {}

void radix_sort_pairs_int_device(uint64_t keys, uint64_t values, int n) {}

void radix_sort_pairs_int64_device(uint64_t keys, uint64_t values, int n) {}

void radix_sort_pairs_float_device(uint64_t keys, uint64_t values, int n) {}

void segmented_sort_pairs_float_device(uint64_t keys, uint64_t values, int n, uint64_t segment_start_indices, uint64_t segment_end_indices, int num_segments) {}

void segmented_sort_pairs_int_device(uint64_t keys, uint64_t values, int n, uint64_t segment_start_indices, uint64_t segment_end_indices, int num_segments) {}

#endif // !WP_ENABLE_CUDA


void radix_sort_pairs_int_host(uint64_t keys, uint64_t values, int n)
{
    radix_sort_pairs_host(
        reinterpret_cast<int *>(keys),
        reinterpret_cast<int *>(values), n);
}

void radix_sort_pairs_int64_host(uint64_t keys, uint64_t values, int n)
{
    radix_sort_pairs_host(
        reinterpret_cast<int64_t *>(keys),
        reinterpret_cast<int *>(values), n);
}

void radix_sort_pairs_float_host(uint64_t keys, uint64_t values, int n)
{
    radix_sort_pairs_host(
        reinterpret_cast<float *>(keys),
        reinterpret_cast<int *>(values), n);
}

void segmented_sort_pairs_float_host(uint64_t keys, uint64_t values, int n, uint64_t segment_start_indices, uint64_t segment_end_indices, int num_segments)
{
    segmented_sort_pairs_host(
        reinterpret_cast<float *>(keys),
        reinterpret_cast<int *>(values), n,
        reinterpret_cast<int *>(segment_start_indices),
        reinterpret_cast<int *>(segment_end_indices), num_segments);
}

void segmented_sort_pairs_int_host(uint64_t keys, uint64_t values, int n, uint64_t segment_start_indices, uint64_t segment_end_indices, int num_segments)
{
    segmented_sort_pairs_host(
        reinterpret_cast<int *>(keys),
        reinterpret_cast<int *>(values), n,
        reinterpret_cast<int *>(segment_start_indices),
        reinterpret_cast<int *>(segment_end_indices), num_segments);
}
