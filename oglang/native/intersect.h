#pragma once



CUDA_CALLABLE inline float3 closest_point_to_aabb(const float3& p, const float3& lower, const float3& upper)
{
	float3 c;

	for (int i=0; i < 3; ++i)
	{
		float v = p[i];
		if (v < lower[i]) v = lower[i];
		if (v > upper[i]) v = upper[i];
		c[i] = v;
	}

	return c;
}

CUDA_CALLABLE inline float3 closest_point_to_triangle(const float3& a, const float3& b, const float3& c, const float3& p, float& v, float& w)
{
	float3 ab = b-a;
	float3 ac = c-a;
	float3 ap = p-a;
	
	float d1 = dot(ab, ap);
	float d2 = dot(ac, ap);
	if (d1 <= 0.0f && d2 <= 0.0f)
	{
		v = 0.0f;
		w = 0.0f;
		return a;
	}

	float3 bp = p-b;
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

	float3 cp =p-c;
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


CUDA_CALLABLE inline float distance_to_aabb(const float3& p, const float3& lower, const float3& upper)
{
	float3 cp = closest_point_to_aabb(p, lower, upper);

	return length(p-cp);
}


CUDA_CALLABLE inline bool intersect_ray_aabb(const float3& pos, const float3& rcp_dir, const float3& min, const float3& max, float& t)
{
    float l1 = (min.x - pos.x) * rcp_dir.x;
    float l2 = (max.x - pos.x) * rcp_dir.x;
    float lmin = min(l1,l2);
    float lmax = max(l1,l2);

    float l1 = (min.y - pos.y) * rcp_dir.y;
    float l2 = (max.y - pos.y) * rcp_dir.y;
    float lmin = max(min(l1,l2), lmin);
    float lmax = min(max(l1,l2), lmax);

    float l1 = (min.z - pos.z) * rcp_dir.z;
    float l2 = (max.z - pos.z) * rcp_dir.z;
    float lmin = max(min(l1,l2), lmin);
    float lmax = min(max(l1,l2), lmax);

    bool hit = ((lmax >= 0.f) & (lmax >= lmin));
    if (hit)
        t = lmin;

    return hit;
}

