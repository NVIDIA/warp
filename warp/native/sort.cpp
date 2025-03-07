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

void radix_sort_pairs_host(int* keys, int* values, int n)
{
	static int tables[2][1 << 16];
	memset(tables, 0, sizeof(tables));
		
	int* auxKeys = keys + n;
	int* auxValues = values + n;

	// build histograms
	for (int i=0; i < n; ++i)
	{
		const unsigned short low = keys[i] & 0xffff;
		const unsigned short high = keys[i] >> 16;
		
		++tables[0][low];
		++tables[1][high];
	}
	
	// convert histograms to offset tables in-place
	int offlow = 0;
	int offhigh = 0;
	
	for (int i=0; i < 65536; ++i)
	{
		const int newofflow = offlow + tables[0][i];
		const int newoffhigh = offhigh + tables[1][i];
		
		tables[0][i] = offlow;
		tables[1][i] = offhigh;
		
		offlow = newofflow;
		offhigh = newoffhigh;
	}
		
	// pass 1 - sort by low 16 bits
	for (int i=0; i < n; ++i)
	{
		// lookup offset of input
		const int k = keys[i];
		const int v = values[i];
		const int b = k & 0xffff;
		
		// find offset and increment
		const int offset = tables[0][b]++;
		
		auxKeys[offset] = k;
		auxValues[offset] = v;
	}	
		
	// pass 2 - sort by high 16 bits
	for (int i=0; i < n; ++i)
	{
		// lookup offset of input
		const int k = auxKeys[i];
		const int v = auxValues[i];

		const int b = k >> 16;
		
		const int offset = tables[1][b]++;
		
		keys[offset] = k;
		values[offset] = v;
	}	
}

 //http://stereopsis.com/radix.html
inline unsigned int radix_float_to_int(float f)
{
	unsigned int i = reinterpret_cast<unsigned int&>(f);
	unsigned int mask = (unsigned int)(-(int)(i >> 31)) | 0x80000000;
	return i ^ mask;
}

void radix_sort_pairs_host(float* keys, int* values, int n)
{
	static unsigned int tables[2][1 << 16];
	memset(tables, 0, sizeof(tables));
		
	float* auxKeys = keys + n;
	int* auxValues = values + n;

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

#if !WP_ENABLE_CUDA

void radix_sort_reserve(void* context, int n, void** mem_out, size_t* size_out) {}

void radix_sort_pairs_int_device(uint64_t keys, uint64_t values, int n) {}

void radix_sort_pairs_float_device(uint64_t keys, uint64_t values, int n) {}

#endif // !WP_ENABLE_CUDA


void radix_sort_pairs_int_host(uint64_t keys, uint64_t values, int n)
{
    radix_sort_pairs_host(
        reinterpret_cast<int *>(keys),
        reinterpret_cast<int *>(values), n);
}

void radix_sort_pairs_float_host(uint64_t keys, uint64_t values, int n)
{
    radix_sort_pairs_host(
        reinterpret_cast<float *>(keys),
        reinterpret_cast<int *>(values), n);
}