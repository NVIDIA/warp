#pragma once

#include "core.h"
#include "bvh.h"
#include "intersect.h"

namespace og
{

struct Mesh
{
    vec3* points;
    int* indices;

	bounds3* bounds;

    int num_points;
    int num_tris;

    BVH bvh;
};

CUDA_CALLABLE inline Mesh mesh_get(uint64_t id)
{
    return *(Mesh*)(&id);
}

// these can be called inside kernels so need to be inline
CUDA_CALLABLE inline vec3 mesh_query_point(uint64_t id, const vec3& point, float max_dist)
{
    Mesh mesh = mesh_get(id);

	if (mesh.bvh.num_nodes == 0)
		return vec3();

	int stack[64];
    stack[0] = mesh.bvh.root;

	int count = 1;

	float min_dist = FLT_MAX;
	vec3 min_point;

	while (count)
	{
		const int nodeIndex = stack[--count];

		BVHPackedNodeHalf lower = mesh.bvh.node_lowers[nodeIndex];
		BVHPackedNodeHalf upper = mesh.bvh.node_uppers[nodeIndex];

		const float dist = distance_to_aabb(point, vec3(lower.x, lower.y, lower.z), vec3(upper.x, upper.y, upper.z));

		if (dist < min_dist)
		{
			const int left_index = lower.i;
			const int right_index = upper.i;

			if (lower.b)
			{	
                // compute closest point on tri
                int i = mesh.indices[left_index*3+0];
                int j = mesh.indices[left_index*3+1];
                int k = mesh.indices[left_index*3+2];

                vec3 p = mesh.points[i];
                vec3 q = mesh.points[j];
                vec3 r = mesh.points[k];

                float v, w;
                vec3 c = closest_point_to_triangle(p, q, r, point, v, w);

                float dist = length(c-point);

                if (dist < min_dist)
				{
                    min_dist = dist;
					min_point = c;
				}
			}
			else
			{
				stack[count++] = left_index;
				stack[count++] = right_index;
			}
		}
	}

	return min_point;
}


CUDA_CALLABLE inline float mesh_query_ray(uint64_t id, const vec3& start, const vec3& dir, float max_t, float& u, float& v, float& sign)
{
	/*
    Mesh mesh = mesh_get(id);

	if (mesh.bvh.mNumNodes == 0)
		return 0.0f;

    int stack[64]
	stack[0] = mesh.bvh.root;
	int count = 1;

	vec3 rcp_dir = vec3(1.0f/dir.x, 1.0f/dir.y, 1.0f/dir.z);

	while (count)
	{
		const int nodeIndex = stack[--count];

		// union to allow 128-bit loads
		union { BVHPackedNodeHalf lower; vec4 lowerf; };
		union {	BVHPackedNodeHalf upper; vec4 upperf; };
		
		lowerf = tex1Dfetch<vec4>(bvh.node_lowers, nodeIndex);
		upperf = tex1Dfetch<vec4>(bvh.node_uppers, nodeIndex);

		Bounds nodeBounds(vec3(&lower.x)*bvhScale, vec3(&upper.x)*bvhScale);
		nodeBounds.Expand(thickness);

		float t;
		bool hit = intersect_ray_aabb(start, rcp_dir, nodeBounds.lower, nodeBounds.upper, t);

		if (hit && t < maxT)
		{
			const int leftIndex = lower.i;
			const int rightIndex = upper.i;

			if (lower.b)
			{	
				//f(leftIndex, maxT);
			}
			else
			{
				stack[count++] = leftIndex;
				stack[count++] = rightIndex;
			}
		}
	}

	// todo:
	return 0.0f;
	*/
	return 0.0f;
}

// determine if a point is inside (ret >0 ) or outside the mesh (ret < 0)
CUDA_CALLABLE inline float mesh_query_inside(uint64_t id, const vec3& p)
{
    float u, v, sign;
    int parity = 0;

    // x-axis
    if (mesh_query_ray(id, p, vec3(1.0f, 0.0f, 0.0f), FLT_MAX, u, v, sign) < 0.0f)
        parity++;
    // y-axis
    if (mesh_query_ray(id, p, vec3(0.0f, 1.0f, 0.0f), FLT_MAX, u, v, sign) < 0.0f)
        parity++;
    // z-axis
    if (mesh_query_ray(id, p, vec3(0.0f, 0.0f, 1.0f), FLT_MAX, u, v, sign) < 0.0f)
        parity++;

    // if all 3 rays inside then return -1
    if (parity == 3)
        return -1.0f;
    else
        return 1.0f;
}


} // namespace og