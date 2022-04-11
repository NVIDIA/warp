/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "sort.h"
#include "string.h"

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

#if WP_DISABLE_CUDA

void radix_sort_reserve(int n) {}

#endif // WP_DISABLE_CUDA
