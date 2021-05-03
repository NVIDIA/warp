#pragma once

#include "core.h"
#include "vec3.h"


CUDA_CALLABLE inline vec3 closest_point_to_aabb(const vec3& p, const vec3& lower, const vec3& upper)
{
	vec3 c;

	{
		float v = p[0];
		if (v < lower[0]) v = lower[0];
		if (v > upper[0]) v = upper[0];
		c[0] = v;
	}

	{
		float v = p[1];
		if (v < lower[1]) v = lower[1];
		if (v > upper[1]) v = upper[1];
		c[1] = v;
	}

	{
		float v = p[2];
		if (v < lower[2]) v = lower[2];
		if (v > upper[2]) v = upper[2];
		c[2] = v;
	}

	return c;
}

CUDA_CALLABLE inline vec3 closest_point_to_triangle(const vec3& a, const vec3& b, const vec3& c, const vec3& p, float& v, float& w)
{
	vec3 ab = b-a;
	vec3 ac = c-a;
	vec3 ap = p-a;
	
	float d1 = dot(ab, ap);
	float d2 = dot(ac, ap);
	if (d1 <= 0.0f && d2 <= 0.0f)
	{
		v = 0.0f;
		w = 0.0f;
		return a;
	}

	vec3 bp = p-b;
	float d3 = dot(ab, bp);
	float d4 = dot(ac, bp);
	if (d3 >= 0.0f && d4 <= d3)
	{
		v = 1.0f;
		w = 0.0f;
		return b;
	}

	float vc = d1*d4 - d3*d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
	{
		v = d1 / (d1-d3);
		w = 0.0f;
		return a + v*ab;
	}

	vec3 cp =p-c;
	float d5 = dot(ab, cp);
	float d6 = dot(ac, cp);
	if (d6 >= 0.0f && d5 <= d6)
	{
		v = 0.0f;
		w = 1.0f;
		return c;
	}

	float vb = d5*d2 - d1*d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
	{
		v = 0.0f;
		w = d2 / (d2 - d6);
		return a + w * ac;
	}

	float va = d3*d6 - d5*d4;
	if (va <= 0.0f && (d4 -d3) >= 0.0f && (d5-d6) >= 0.0f)
	{
		w = (d4-d3)/((d4-d3) + (d5-d6));
		v = 1.0f-w;		
		return b + w * (c-b);
	}

	float denom = 1.0f / (va + vb + vc);
	v = vb * denom;
	w = vc * denom;
	return a + ab*v + ac*w;
}


CUDA_CALLABLE inline bool intersect_ray_aabb(const vec3& pos, const vec3& rcp_dir, const vec3& lower, const vec3& upper, float& t)
{
	float l1, l2, lmin, lmax;

    l1 = (lower.x - pos.x) * rcp_dir.x;
    l2 = (upper.x - pos.x) * rcp_dir.x;
    lmin = min(l1,l2);
    lmax = max(l1,l2);

    l1 = (lower.y - pos.y) * rcp_dir.y;
    l2 = (upper.y - pos.y) * rcp_dir.y;
    lmin = max(min(l1,l2), lmin);
    lmax = min(max(l1,l2), lmax);

    l1 = (lower.z - pos.z) * rcp_dir.z;
    l2 = (upper.z - pos.z) * rcp_dir.z;
    lmin = max(min(l1,l2), lmin);
    lmax = min(max(l1,l2), lmax);

    bool hit = ((lmax >= 0.f) & (lmax >= lmin));
    if (hit)
        t = lmin;

    return hit;
}
