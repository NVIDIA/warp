#pragma once

#include "adjoint.h"
#include "bvh.h"
#include "intersect.h"

struct Mesh
{
    og::float3* points;
    int* indices;

    int num_points;
    int num_tris;

    BVH bvh;
};


uint64_t mesh_create(const og::float3* points, int* tris);
void mesh_destroy(uint64_t id);
void mesh_update(uint64_t id, const og::float3* points, int* tris, bool refit);

Mesh mesh_get(uint64_t id)
{
    return *(Mesh*)(&id);
}

// these can be called inside kernels so need to be inline
inline og::float3 mesh_query_point(uint64_t id, const og::float3& point, float max_dist)
{
    Mesh mesh = mesh_get(id);

	if (mesh.bvh.num_nodes == 0)
		return;

	int stack[64];
    stack[0] = *mesh.bvh.root;

	int count = 1;

	while (count)
	{
		const int nodeIndex = stack[--count];

		float4 lower = mesh.bvh.node_lowers[nodeIndex];
		float4 upper = mesh.bvh.node_uppers[nodeIndex];

        float min_dist = FLT_MAX;
		const float dist = distance_to_aabb(point, lower, upper);

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

                float3 p = mesh.points[i];
                float3 q = mesh.points[j];
                float3 r = mesh.points[k];

                float v, w;
                float3 c = closest_point_to_triangle(p, q, r, point, v, w);

                float dist = length(c-point);

                if (dist < min_dist)
                    min_dist = dist;
			}
			else
			{
				stack[count++] = left_index;
				stack[count++] = right_index;
			}
		}
	}
}


inline float mesh_query_ray(uint64_t id, const og::float3& start, const og::float3& dir, float max_t, float& u, float& v, float& sign)
{
	if (bvh.mNumNodes == 0)
		return;

    int stack[64]
	stack[0] = *bvh.mRootNode;
	int count = 1;

	float3 rcp_dir = float3(1.0f/dir.x, 1.0f/dir.y, 1.0f/dir.z);

	while (count)
	{
		const int nodeIndex = stack[--count];

		// union to allow 128-bit loads
		union { BVHPackedNodeHalf lower; float4 lowerf; };
		union {	BVHPackedNodeHalf upper; float4 upperf; };
		
		lowerf = tex1Dfetch<float4>(bvh.mNodeLowersTex, nodeIndex);
		upperf = tex1Dfetch<float4>(bvh.mNodeUppersTex, nodeIndex);

		Bounds nodeBounds(float3(&lower.x)*bvhScale, float3(&upper.x)*bvhScale);
		nodeBounds.Expand(thickness);

		float t;
		bool hit = intersect_ray_aabb(start, rcp_dir, nodeBounds.lower, nodeBounds.upper, t);

		if (hit && t < maxT)
		{
			const int leftIndex = lower.i;
			const int rightIndex = upper.i;

			if (lower.b)
			{	
				f(leftIndex, maxT);
			}
			else
			{
				stack[count++] = leftIndex;
				stack[count++] = rightIndex;
			}
		}
	}
}

// determine if a point is inside (ret >0 ) or outside the mesh (ret < 0)
inline float mesh_query_inside(uint64_t id, const og::float3& point)
{
    float u, v, sign;
    int parity = 0;

    // x-axis
    if (mesh_query_ray(id, p, float3(1.0f, 0.0f, 0.0f), FLT_MAX, u, v, sign) < 0.0f)
        parity++;
    // y-axis
    if (mesh_query_ray(id, p, float3(0.0f, 1.0f, 0.0f), FLT_MAX, u, v, sign) < 0.0f)
        parity++;
    // z-axis
    if (mesh_query_ray(id, p, float3(0.0f, 0.0f, 1.0f), FLT_MAX, u, v, sign) < 0.0f)
        parity++;

    // if all 3 rays inside then return -1
    if (parity == 3)
        return -1.0f;
    else
        return 1.0f;
}

