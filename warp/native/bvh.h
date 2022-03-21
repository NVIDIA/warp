/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include "builtin.h"

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

	CUDA_CALLABLE inline bool empty() const { return lower.x >= upper.x || lower.y >= upper.y || lower.z >= upper.z; }

	CUDA_CALLABLE inline bool overlaps(const vec3& p) const
	{
		if (p.x < lower.x ||
			p.y < lower.y ||
			p.z < lower.z ||
			p.x > upper.x ||
			p.y > upper.y ||
			p.z > upper.z)
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
		if (lower.x > b.upper.x ||
			lower.y > b.upper.y ||
			lower.z > b.upper.z ||
			upper.x < b.lower.x ||
			upper.y < b.lower.y ||
			upper.z < b.lower.z)
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

	CUDA_CALLABLE inline float area() const
	{
		vec3 e = upper-lower;
		return 2.0f*(e.x*e.y + e.x*e.z + e.y*e.z);
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
	
	int max_depth;
	int max_nodes;
    int num_nodes;

	int root;
};

BVH bvh_create(const bounds3* bounds, int num_bounds);

void bvh_destroy_host(BVH& bvh);
void bvh_destroy_device(BVH& bvh);

void bvh_refit_host(BVH& bvh, const bounds3* bounds);
void bvh_refit_device(BVH& bvh, const bounds3* bounds);

// copy host BVH to device
BVH bvh_clone(const BVH& bvh_host);



CUDA_CALLABLE inline BVHPackedNodeHalf make_node(const vec3& bound, int child, bool leaf)
{
    BVHPackedNodeHalf n;
    n.x = bound.x;
    n.y = bound.y;
    n.z = bound.z;
    n.i = (unsigned int)child;
    n.b = (unsigned int)(leaf?1:0);

    return n;
}

// variation of make_node through volatile pointers used in BuildHierarchy
CUDA_CALLABLE inline void make_node(volatile BVHPackedNodeHalf* n, const vec3& bound, int child, bool leaf)
{
    n->x = bound.x;
    n->y = bound.y;
    n->z = bound.z;
    n->i = (unsigned int)child;
    n->b = (unsigned int)(leaf?1:0);
}

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


} // namespace wp

