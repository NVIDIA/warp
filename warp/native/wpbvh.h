/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include "builtin.h"
#include "bvh.h"
#include "intersect.h"

#define BVH_DEBUG 0

namespace wp
{

struct Bvh
{
    vec3* lowers;
	vec3* uppers;

	bounds3* bounds;
    int num_bounds;

    BVH internal_bvh;

	void* context;
};

CUDA_CALLABLE inline Bvh bvh_get(uint64_t id)
{
    return *(Bvh*)(id);
}

CUDA_CALLABLE inline int bvh_get_num_bounds(uint64_t id)
{
	Bvh bvh = bvh_get(id);
	return bvh.num_bounds;
}


// stores state required to traverse the BVH nodes that 
// overlap with a query AABB.
struct bvh_query_t
{
    CUDA_CALLABLE bvh_query_t()
    {
    }
    CUDA_CALLABLE bvh_query_t(int)
    {
    } // for backward pass

    // Bvh Id
    Bvh bvh;
	// BVH traversal stack:
	int stack[32];
	int count;

    // inputs

	bool is_ray;
    wp::vec3 input_lower;	// start for ray
    wp::vec3 input_upper;	// dir for ray

	// Face
	int bounds_nr;
};


CUDA_CALLABLE inline bvh_query_t bvh_query(
    uint64_t id, bool is_ray, const vec3& lower, const vec3& upper)
{
    // This routine traverses the BVH tree until it finds
	// the first overlapping bound. 

    // initialize empty
	bvh_query_t query;
	query.bounds_nr = -1;

	Bvh bvh = bvh_get(id);

	query.bvh = bvh;
	query.is_ray = is_ray;
	
    // if no bvh nodes, return empty query.
    if (bvh.internal_bvh.num_nodes == 0)
    {
		query.count = 0;
		return query;
	}

    // optimization: make the latest
	
	query.stack[0] = bvh.internal_bvh.root;
	query.count = 1;
    query.input_lower = lower;
    query.input_upper = upper;

    wp::bounds3 input_bounds(query.input_lower, query.input_upper);
	
    // Navigate through the bvh, find the first overlapping leaf node.
    while (query.count)
    {
		const int nodeIndex = query.stack[--query.count];
		BVHPackedNodeHalf node_lower = bvh.internal_bvh.node_lowers[nodeIndex];
		BVHPackedNodeHalf node_upper = bvh.internal_bvh.node_uppers[nodeIndex];

		wp::vec3 lower_pos(node_lower.x, node_lower.y, node_lower.z);
		wp::vec3 upper_pos(node_upper.x, node_upper.y, node_upper.z);
        wp::bounds3 current_bounds(lower_pos, upper_pos);

		if (query.is_ray) 
		{
			float t = 0.0f;
			if (!intersect_ray_aabb(query.input_lower, query.input_upper, current_bounds.lower, current_bounds.upper, t))
			// Skip this box, it doesn't overlap with our ray.
				continue;
		}
		else {
	        if (!input_bounds.overlaps(current_bounds))
				// Skip this box, it doesn't overlap with our target box.
				continue;
		}

		const int left_index = node_lower.i;
		const int right_index = node_upper.i;

        // Make bounds from this AABB
        if (node_lower.b)
        {
			// found very first triangle index.
			// Back up one level and return 
			query.stack[query.count++] = nodeIndex;
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
	return bvh_query(id, true, start, dir);
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
    Bvh bvh = query.bvh;
	
	wp::bounds3 input_bounds(query.input_lower, query.input_upper);
    // Navigate through the bvh, find the first overlapping leaf node.
    while (query.count)
    {
        const int nodeIndex = query.stack[--query.count];
        BVHPackedNodeHalf node_lower = bvh.internal_bvh.node_lowers[nodeIndex];
        BVHPackedNodeHalf node_upper = bvh.internal_bvh.node_uppers[nodeIndex];

        wp::vec3 lower_pos(node_lower.x, node_lower.y, node_lower.z);
        wp::vec3 upper_pos(node_upper.x, node_upper.y, node_upper.z);
        wp::bounds3 current_bounds(lower_pos, upper_pos);

		if (query.is_ray) 
		{
			float t = 0.0f;
			if (!intersect_ray_aabb(query.input_lower, query.input_upper, current_bounds.lower, current_bounds.upper, t))
			// Skip this box, it doesn't overlap with our ray.
				continue;
		}
		else {
	        if (!input_bounds.overlaps(current_bounds))
				// Skip this box, it doesn't overlap with our target box.
				continue;
		}

        const int left_index = node_lower.i;
        const int right_index = node_upper.i;

        // Make bounds from this AABB
        if (node_lower.b)
        {
            // found very first triangle index
            query.bounds_nr = left_index;
			index = left_index;
            return true;
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
    // can't reverse BVH queries, users should not rely on neighbor ordering
    return query;
}


// stub
CUDA_CALLABLE inline void adj_bvh_query_next(bvh_query_t& query, int& index, bvh_query_t&, int&, bool&) 
{

}




bool bvh_get_descriptor(uint64_t id, Bvh& bvh);
void bvh_add_descriptor(uint64_t id, const Bvh& bvh);
void bvh_rem_descriptor(uint64_t id);


} // namespace wp
