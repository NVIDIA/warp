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

#include "builtin.h"
#include "intersect.h"

#define BVH_LEAF_SIZE (4)
#define SAH_NUM_BUCKETS (16)
#define USE_LOAD4
#define BVH_QUERY_STACK_SIZE (32)

#define BVH_CONSTRUCTOR_SAH (0)
#define BVH_CONSTRUCTOR_MEDIAN (1)
#define BVH_CONSTRUCTOR_LBVH (2)

namespace wp
{

struct bounds3
{
	CUDA_CALLABLE inline bounds3() : lower( FLT_MAX)
						           , upper(-FLT_MAX) {}

	CUDA_CALLABLE inline bounds3(const vec3& lower, const vec3& upper) : lower(lower), upper(upper) {}

	CUDA_CALLABLE inline vec3 center() const { return 0.5f*(lower+upper); }
	CUDA_CALLABLE inline vec3 edges() const { return upper-lower; }

	CUDA_CALLABLE inline void expand(float r)
	{
		lower -= vec3(r);
		upper += vec3(r);
	}

	CUDA_CALLABLE inline void expand(const vec3& r)
	{
		lower -= r;
		upper += r;
	}

	CUDA_CALLABLE inline bool empty() const { return lower[0] >= upper[0] || lower[1] >= upper[1] || lower[2] >= upper[2]; }

	CUDA_CALLABLE inline bool overlaps(const vec3& p) const
	{
		if (p[0] < lower[0] ||
			p[1] < lower[1] ||
			p[2] < lower[2] ||
			p[0] > upper[0] ||
			p[1] > upper[1] ||
			p[2] > upper[2])
		{
			return false;
		}
		else
		{
			return true;
		}
	}

	CUDA_CALLABLE inline bool overlaps(const bounds3& b) const
	{
		if (lower[0] > b.upper[0] ||
			lower[1] > b.upper[1] ||
			lower[2] > b.upper[2] ||
			upper[0] < b.lower[0] ||
			upper[1] < b.lower[1] ||
			upper[2] < b.lower[2])
		{
			return false;
		}
		else
		{
			return true;
		}
	}

	CUDA_CALLABLE inline bool overlaps(const vec3& b_lower, const vec3& b_upper) const
	{
		if (lower[0] > b_upper[0] ||
			lower[1] > b_upper[1] ||
			lower[2] > b_upper[2] ||
			upper[0] < b_lower[0] ||
			upper[1] < b_lower[1] ||
			upper[2] < b_lower[2])
		{
			return false;
		}
		else
		{
			return true;
		}
	}

	CUDA_CALLABLE inline void add_point(const vec3& p)
	{
		lower = min(lower, p);
		upper = max(upper, p);
	}

	CUDA_CALLABLE inline void add_bounds(const vec3& lower_other, const vec3& upper_other)
	{
		// lower_other will only impact the lower of the new bounds
		// upper_other will only impact the upper of the new bounds
		// this costs only half of the computation of adding lower_other and upper_other separately
		lower = min(lower, lower_other);
		upper = max(upper, upper_other);
	}

	CUDA_CALLABLE inline float area() const
	{
		vec3 e = upper-lower;
		return 2.0f*(e[0]*e[1] + e[0]*e[2] + e[1]*e[2]);
	}

	vec3 lower;
	vec3 upper;
};

CUDA_CALLABLE inline bounds3 bounds_union(const bounds3& a, const vec3& b) 
{
	return bounds3(min(a.lower, b), max(a.upper, b));
}

CUDA_CALLABLE inline bounds3 bounds_union(const bounds3& a, const bounds3& b) 
{
	return bounds3(min(a.lower, b.lower), max(a.upper, b.upper));
}

CUDA_CALLABLE inline bounds3 bounds_intersection(const bounds3& a, const bounds3& b)
{
	return bounds3(max(a.lower, b.lower), min(a.upper, b.upper));
}

struct BVHPackedNodeHalf
{
	float x;
	float y;
	float z;
	// For non-leaf nodes:
	// - 'lower.i' represents the index of the left child node.
	// - 'upper.i' represents the index of the right child node.
	//
	// For leaf nodes:
	// - 'lower.i' indicates the start index of the primitives in 'primitive_indices'.
	// - 'upper.i' indicates the index just after the last primitive in 'primitive_indices'
	unsigned int i : 31;
	unsigned int b : 1;
};

struct BVH
{		
    BVHPackedNodeHalf* node_lowers;
    BVHPackedNodeHalf* node_uppers;

	// used for fast refits
	int* node_parents;
	int* node_counts;
	// reordered primitive indices corresponds to the ordering of leaf nodes
	int* primitive_indices;
	
	int max_depth;
	int max_nodes;
	int num_nodes;
	// since we use packed leaf nodes, the number of them is no longer the number of items, but variable
	int num_leaf_nodes;

	// pointer (CPU or GPU) to a single integer index in node_lowers, node_uppers
	// representing the root of the tree, this is not always the first node
	// for bottom-up builders
	int* root;

	// item bounds are not owned by the BVH but by the caller
    vec3* item_lowers;
	vec3* item_uppers;
	int num_items;

	// cuda context
	void* context;
};

CUDA_CALLABLE inline BVHPackedNodeHalf make_node(const vec3& bound, int child, bool leaf)
{
    BVHPackedNodeHalf n;
    n.x = bound[0];
    n.y = bound[1];
    n.z = bound[2];
    n.i = (unsigned int)child;
    n.b = (unsigned int)(leaf?1:0);

    return n;
}

// variation of make_node through volatile pointers used in build_hierarchy
CUDA_CALLABLE inline void make_node(volatile BVHPackedNodeHalf* n, const vec3& bound, int child, bool leaf)
{
    n->x = bound[0];
    n->y = bound[1];
    n->z = bound[2];
    n->i = (unsigned int)child;
    n->b = (unsigned int)(leaf?1:0);
}

#ifdef __CUDA_ARCH__
__device__ inline wp::BVHPackedNodeHalf bvh_load_node(const wp::BVHPackedNodeHalf* nodes, int index)
{
#ifdef USE_LOAD4
	//return  (const wp::BVHPackedNodeHalf&)(__ldg((const float4*)(nodes)+index));
	return  (const wp::BVHPackedNodeHalf&)(*((const float4*)(nodes)+index));
#else
	return  nodes[index];
#endif // USE_LOAD4

}
#else
inline wp::BVHPackedNodeHalf bvh_load_node(const wp::BVHPackedNodeHalf* nodes, int index)
{
	return  nodes[index];
}
#endif // __CUDACC__

CUDA_CALLABLE inline int clz(int x)
{
    int n;
    if (x == 0) return 32;
    for (n = 0; ((x & 0x80000000) == 0); n++, x <<= 1);
    return n;
}

CUDA_CALLABLE inline uint32_t part1by2(uint32_t n)
{
    n = (n ^ (n << 16)) & 0xff0000ff;
    n = (n ^ (n <<  8)) & 0x0300f00f;
    n = (n ^ (n <<  4)) & 0x030c30c3;
    n = (n ^ (n <<  2)) & 0x09249249;

    return n;
}

// Takes values in the range [0, 1] and assigns an index based Morton codes of length 3*lwp2(dim) bits 
template <int dim>
CUDA_CALLABLE inline uint32_t morton3(float x, float y, float z)
{
    uint32_t ux = clamp(int(x*dim), 0, dim-1);
    uint32_t uy = clamp(int(y*dim), 0, dim-1);
    uint32_t uz = clamp(int(z*dim), 0, dim-1);

    return (part1by2(uz) << 2) | (part1by2(uy) << 1) | part1by2(ux);
}

// making the class accessible from python

CUDA_CALLABLE inline BVH bvh_get(uint64_t id)
{
    return *(BVH*)(id);
}

CUDA_CALLABLE inline int bvh_get_num_bounds(uint64_t id)
{
	BVH bvh = bvh_get(id);
	return bvh.num_items;
}


// stores state required to traverse the BVH nodes that 
// overlap with a query AABB.
struct bvh_query_t
{
    CUDA_CALLABLE bvh_query_t()
        : bvh(),
          stack(),
          count(0),
          is_ray(false),
          input_lower(),
          input_upper(),
          bounds_nr(0),
		  primitive_counter(-1)
    {}

    // Required for adjoint computations.
    CUDA_CALLABLE inline bvh_query_t& operator+=(const bvh_query_t& other)
    {
        return *this;
    }

    BVH bvh;

	// BVH traversal stack:
	int stack[BVH_QUERY_STACK_SIZE];
	int count;

	// >= 0 if currently in a packed leaf node
	int primitive_counter;
	
    // inputs
    wp::vec3 input_lower;	// start for ray
    wp::vec3 input_upper;	// dir for ray

	int bounds_nr;
	bool is_ray;
};

CUDA_CALLABLE inline bool bvh_query_intersection_test(const bvh_query_t& query, const vec3& node_lower, const vec3& node_upper)
{
	if (query.is_ray)
	{
		float t = 0.0f;
		return intersect_ray_aabb(query.input_lower, query.input_upper, node_lower, node_upper, t);
	}
	else
	{
		return intersect_aabb_aabb(query.input_lower, query.input_upper, node_lower, node_upper);
	}
}

CUDA_CALLABLE inline bvh_query_t bvh_query(
	uint64_t id, bool is_ray, const vec3& lower, const vec3& upper)
{
	// This routine traverses the BVH tree until it finds
	// the first overlapping bound. 

	// initialize empty
	bvh_query_t query;

	query.bounds_nr = -1;

	BVH bvh = bvh_get(id);

	query.bvh = bvh;
	query.is_ray = is_ray;

	// optimization: make the latest	
	query.stack[0] = *bvh.root;
	query.count = 1;
	query.input_lower = lower;
	query.input_upper = upper;

	// Navigate through the bvh, find the first overlapping leaf node.
	while (query.count)
	{
		const int node_index = query.stack[--query.count];
		BVHPackedNodeHalf node_lower = bvh_load_node(bvh.node_lowers, node_index);
		BVHPackedNodeHalf node_upper = bvh_load_node(bvh.node_uppers, node_index);

		if (!bvh_query_intersection_test(query, (vec3&)node_lower, (vec3&)node_upper))
		{
			continue;
		}

		const int left_index = node_lower.i;
		const int right_index = node_upper.i;
		// Make bounds from this AABB
		if (node_lower.b)
		{
			// Reached a leaf node, point to its first primitive
			// Back up one level and return 
			query.primitive_counter = left_index;
			query.stack[query.count++] = node_index;
			return query;
		}
		else
		{
			query.stack[query.count++] = left_index;
			query.stack[query.count++] = right_index;
		}
	}

	return query;
}

CUDA_CALLABLE inline bvh_query_t bvh_query_aabb(
    uint64_t id, const vec3& lower, const vec3& upper)
{
	return bvh_query(id, false, lower, upper);
}


CUDA_CALLABLE inline bvh_query_t bvh_query_ray(
    uint64_t id, const vec3& start, const vec3& dir)
{
	return bvh_query(id, true, start, 1.0f / dir);
}

//Stub
CUDA_CALLABLE inline void adj_bvh_query_aabb(uint64_t id, const vec3& lower, const vec3& upper,
											   uint64_t, vec3&, vec3&, bvh_query_t&)
{
}


CUDA_CALLABLE inline void adj_bvh_query_ray(uint64_t id, const vec3& start, const vec3& dir,
											   uint64_t, vec3&, vec3&, bvh_query_t&)
{
}


CUDA_CALLABLE inline bool bvh_query_next(bvh_query_t& query, int& index)
{
	BVH bvh = query.bvh;

	if (query.primitive_counter != -1)
		// currently in a leaf node which is the last node in the stack
	{
		const int node_index = query.stack[query.count - 1];
		BVHPackedNodeHalf node_lower = bvh_load_node(bvh.node_lowers, node_index);
		BVHPackedNodeHalf node_upper = bvh_load_node(bvh.node_uppers, node_index);

		const int end = node_upper.i;
		for (int primitive_counter = query.primitive_counter; primitive_counter < end; primitive_counter++)
		{
			int primitive_index = bvh.primitive_indices[primitive_counter];
			if (bvh_query_intersection_test(query, bvh.item_lowers[primitive_index], bvh.item_uppers[primitive_index]))
			{
				if (primitive_counter < end - 1)
					// still need to come back to this leaf node for the leftover primitives
				{
					query.primitive_counter = primitive_counter + 1;
				}
				else
					// no need to come back to this leaf node
				{
					query.count--;
					query.primitive_counter = -1;
				}
				index = primitive_index;
				query.bounds_nr = primitive_index;

				return true;
			}
		}
		// if we reach here that means we have finished the current leaf node without finding intersections
		query.primitive_counter = -1;
		// remove the leaf node from the back of the stack because it is finished
		// and continue the bvh traversal
		query.count--;
	}

	// Navigate through the bvh, find the first overlapping leaf node.
	while (query.count)
	{
		const int node_index = query.stack[--query.count];
		BVHPackedNodeHalf node_lower = bvh_load_node(bvh.node_lowers, node_index);
		BVHPackedNodeHalf node_upper = bvh_load_node(bvh.node_uppers, node_index);

		const int left_index = node_lower.i;
		const int right_index = node_upper.i;

		wp::vec3 lower_pos(node_lower.x, node_lower.y, node_lower.z);
		wp::vec3 upper_pos(node_upper.x, node_upper.y, node_upper.z);
		wp::bounds3 current_bounds(lower_pos, upper_pos);

		if (!bvh_query_intersection_test(query, (vec3&)node_lower, (vec3&)node_upper))
		{
			continue;
		}

		if (node_lower.b)
		{
			// found leaf, loop through its content primitives
			const int start = left_index;
			const int end = right_index;

			for (int primitive_counter = start; primitive_counter < end; primitive_counter++)
			{
				int primitive_index = bvh.primitive_indices[primitive_counter];
				if (bvh_query_intersection_test(query, bvh.item_lowers[primitive_index], bvh.item_uppers[primitive_index]))
				{
					if (primitive_counter < end - 1)
						// still need to come back to this leaf node for the leftover primitives
					{
						query.primitive_counter = primitive_counter + 1;
						query.stack[query.count++] = node_index;
					}
					else
						// no need to come back to this leaf node
					{
						query.primitive_counter = -1;
					}
					index = primitive_index;
					query.bounds_nr = primitive_index;

					return true;
				}
			}
		}
		else
		{
			query.stack[query.count++] = left_index;
			query.stack[query.count++] = right_index;
		}
	}
	return false;
}


CUDA_CALLABLE inline int iter_next(bvh_query_t& query)
{
    return query.bounds_nr;
}

CUDA_CALLABLE inline bool iter_cmp(bvh_query_t& query)
{
    bool finished = bvh_query_next(query, query.bounds_nr);
    return finished;
}

CUDA_CALLABLE inline bvh_query_t iter_reverse(const bvh_query_t& query)
{
    // can't reverse BVH queries, users should not rely on traversal ordering
    return query;
}

CUDA_CALLABLE inline void adj_iter_reverse(const bvh_query_t& query, bvh_query_t& adj_query, bvh_query_t& adj_ret)
{
}


// stub
CUDA_CALLABLE inline void adj_bvh_query_next(bvh_query_t& query, int& index, bvh_query_t&, int&, bool&) 
{

}

CUDA_CALLABLE bool bvh_get_descriptor(uint64_t id, BVH& bvh);
CUDA_CALLABLE void bvh_add_descriptor(uint64_t id, const BVH& bvh);
CUDA_CALLABLE void bvh_rem_descriptor(uint64_t id);

#if !__CUDA_ARCH__

void bvh_create_host(vec3* lowers, vec3* uppers, int num_items,  int constructor_type, BVH& bvh);
void bvh_destroy_host(wp::BVH& bvh);
void bvh_refit_host(wp::BVH& bvh);

void bvh_destroy_device(wp::BVH& bvh);
void bvh_refit_device(uint64_t id);

#endif

} // namespace wp
