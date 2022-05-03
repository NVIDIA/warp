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

CUDA_CALLABLE inline vec2 closest_point_to_triangle(const vec3& a, const vec3& b, const vec3& c, const vec3& p)
{
	vec3 ab = b-a;
	vec3 ac = c-a;
	vec3 ap = p-a;
	
	float u, v, w;
	float d1 = dot(ab, ap);
	float d2 = dot(ac, ap);
	if (d1 <= 0.0f && d2 <= 0.0f)
	{
		v = 0.0f;
		w = 0.0f;
		u = 1.0f - v - w;
		return vec2(u, v);
	}

	vec3 bp = p-b;
	float d3 = dot(ab, bp);
	float d4 = dot(ac, bp);
	if (d3 >= 0.0f && d4 <= d3)
	{
		v = 1.0f;
		w = 0.0f;
		u = 1.0f - v - w;
		return vec2(u, v);
	}

	float vc = d1*d4 - d3*d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
	{
		v = d1 / (d1-d3);
		w = 0.0f;
		u = 1.0f - v - w;
		return vec2(u, v);
	}

	vec3 cp = p-c;
	float d5 = dot(ab, cp);
	float d6 = dot(ac, cp);
	if (d6 >= 0.0f && d5 <= d6)
	{
		v = 0.0f;
		w = 1.0f;
		u = 1.0f - v - w;
		return vec2(u, v);
	}

	float vb = d5*d2 - d1*d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
	{
		v = 0.0f;
		w = d2 / (d2 - d6);
		u = 1.0f - v - w;
		return vec2(u, v);
	}

	float va = d3*d6 - d5*d4;
	if (va <= 0.0f && (d4 -d3) >= 0.0f && (d5-d6) >= 0.0f)
	{
		w = (d4-d3)/((d4-d3) + (d5-d6));
		v = 1.0f - w;
		u = 1.0f - v - w;
		return vec2(u, v);
	}

	float denom = 1.0f / (va + vb + vc);
	v = vb * denom;
	w = vc * denom;
	u = 1.0f - v - w;
	return vec2(u, v);
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


// Moller and Trumbore's method
CUDA_CALLABLE inline bool intersect_ray_tri_moller(const vec3& p, const vec3& dir, const vec3& a, const vec3& b, const vec3& c, float& t, float& u, float& v, float& w, float& sign, vec3* normal)
{
    vec3 ab = b - a;
    vec3 ac = c - a;
    vec3 n = cross(ab, ac);

    float d = dot(-dir, n);
    float ood = 1.0f / d; // No need to check for division by zero here as infinity arithmetic will save us...
    vec3 ap = p - a;

    t = dot(ap, n) * ood;
    if (t < 0.0f)
        return false;

    vec3 e = cross(-dir, ap);
    v = dot(ac, e) * ood;
    if (v < 0.0f || v > 1.0f) // ...here...
        return false;
    w = -dot(ab, e) * ood;
    if (w < 0.0f || (v + w) > 1.0f) // ...and here
        return false;

    u = 1.0f - v - w;
    if (normal)
        *normal = n;

	sign = d;

    return true;
}


CUDA_CALLABLE inline bool intersect_ray_tri_rtcd(const vec3& p, const vec3& dir, const vec3& a, const vec3& b, const vec3& c, float& t, float& u, float& v, float& w, float& sign, vec3* normal)
{
	const vec3 ab = b-a;
	const vec3 ac = c-a;

	// calculate normal
	vec3 n = cross(ab, ac);

	// need to solve a system of three equations to give t, u, v
	float d = dot(-dir, n);

	// if dir is parallel to triangle plane or points away from triangle 
	if (d <= 0.0f)
        return false;

	vec3 ap = p-a;
	t = dot(ap, n);

	// ignores tris behind 
	if (t < 0.0f)
		return false;

	// compute barycentric coordinates
	vec3 e = cross(-dir, ap);
	v = dot(ac, e);
	if (v < 0.0f || v > d) return false;

	w = -dot(ab, e);
	if (w < 0.0f || v + w > d) return false;

	float ood = 1.0f / d;
	t *= ood;
	v *= ood;
	w *= ood;
	u = 1.0f-v-w;

	// optionally write out normal (todo: this branch is a performance concern, should probably remove)
	if (normal)
		*normal = n;

	return true;
}

#ifndef  __CUDA_ARCH__

// these are provided as built-ins by CUDA
inline float __int_as_float(int i)
{
	return *(float*)(&i);
}

inline int __float_as_int(float f)
{
	return *(int*)(&f);
}

#endif


CUDA_CALLABLE inline float xorf(float x, int y)
{
	return __int_as_float(__float_as_int(x) ^ y);
}

CUDA_CALLABLE inline int sign_mask(float x)
{
	return __float_as_int(x) & 0x80000000;
}

CUDA_CALLABLE inline int max_dim(vec3 a)
{
	float x = abs(a.x);
	float y = abs(a.y);
	float z = abs(a.z);

	return longest_axis(vec3(x, y, z));
}

// http://jcgt.org/published/0002/01/05/
CUDA_CALLABLE inline bool intersect_ray_tri_woop(const vec3& p, const vec3& dir, const vec3& a, const vec3& b, const vec3& c, float& t, float& u, float& v, float& w, float& sign, vec3* normal)
{
	// todo: precompute for ray

	int kz = max_dim(dir);
	int kx = kz+1; if (kx == 3) kx = 0;
	int ky = kx+1; if (ky == 3) ky = 0;

	if (dir[kz] < 0.0f)
	{
		float tmp = kx;
		kx = ky;
		ky = tmp;
	}

	float Sx = dir[kx]/dir[kz];
	float Sy = dir[ky]/dir[kz];
	float Sz = 1.0f/dir[kz];

	// todo: end precompute

	const vec3 A = a-p;
	const vec3 B = b-p;
	const vec3 C = c-p;
	
	const float Ax = A[kx] - Sx*A[kz];
	const float Ay = A[ky] - Sy*A[kz];
	const float Bx = B[kx] - Sx*B[kz];
	const float By = B[ky] - Sy*B[kz];
	const float Cx = C[kx] - Sx*C[kz];
	const float Cy = C[ky] - Sy*C[kz];
		
	float U = Cx*By - Cy*Bx;
	float V = Ax*Cy - Ay*Cx;
	float W = Bx*Ay - By*Ax;

	if (U == 0.0f || V == 0.0f || W == 0.0f) 
	{
		double CxBy = (double)Cx*(double)By;
		double CyBx = (double)Cy*(double)Bx;
		U = (float)(CxBy - CyBx);
		double AxCy = (double)Ax*(double)Cy;
		double AyCx = (double)Ay*(double)Cx;
		V = (float)(AxCy - AyCx);
		double BxAy = (double)Bx*(double)Ay;
		double ByAx = (double)By*(double)Ax;
		W = (float)(BxAy - ByAx);
	}

	if ((U<0.0f || V<0.0f || W<0.0f) &&	(U>0.0f || V>0.0f || W>0.0f)) 
		return false;

	float det = U+V+W;

	if (det == 0.0f) 
		return false;

	const float Az = Sz*A[kz];
	const float Bz = Sz*B[kz];
	const float Cz = Sz*C[kz];
	const float T = U*Az + V*Bz + W*Cz;

	int det_sign = sign_mask(det);
	if (xorf(T,det_sign) < 0.0f)// || xorf(T,det_sign) > hit.t * xorf(det, det_sign)) // early out if hit.t is specified
		return false;

	const float rcpDet = 1.0f/det;
	u = U*rcpDet;
	v = V*rcpDet;
	w = W*rcpDet;
	t = T*rcpDet;
	sign = det;
	
	// optionally write out normal (todo: this branch is a performance concern, should probably remove)
	if (normal)
	{
		const vec3 ab = b-a;
		const vec3 ac = c-a;

		// calculate normal
		*normal = cross(ab, ac); 
	}

	return true;
}

// Möller's method
#include "intersect_tri.h"

CUDA_CALLABLE inline int intersect_tri_tri(
    vec3& v0, vec3& v1, vec3& v2,
    vec3& u0, vec3& u1, vec3& u2)
{
	return NoDivTriTriIsect(&v0[0], &v1[0], &v2[0], &u0[0], &u1[0], &u2[0]);
}

CUDA_CALLABLE inline void adj_intersect_tri_tri(const vec3& var_v0,
												const vec3& var_v1,
												const vec3& var_v2,
												const vec3& var_u0,
												const vec3& var_u1,
												const vec3& var_u2,
												vec3& adj_v0,
												vec3& adj_v1,
												vec3& adj_v2,
												vec3& adj_u0,
												vec3& adj_u1,
												vec3& adj_u2,
												int adj_ret) {}


CUDA_CALLABLE inline void adj_closest_point_to_triangle(
	const vec3& var_a, const vec3& var_b, const vec3& var_c, const vec3& var_p,
	vec3& adj_a, vec3& adj_b, vec3& adj_c, vec3& adj_p, vec2& adj_ret)
{
    // primal vars
    vec3 var_0;
    vec3 var_1;
    vec3 var_2;
    float32 var_3;
    float32 var_4;
    const float32 var_5 = 0.0;
    bool var_6;
    bool var_7;
    bool var_8;
    const float32 var_9 = 1.0;
    vec2 var_10;
    vec3 var_11;
    float32 var_12;
    float32 var_13;
    bool var_14;
    bool var_15;
    bool var_16;
    vec2 var_17;
    vec2 var_18;
    float32 var_19;
    float32 var_20;
    float32 var_21;
    float32 var_22;
    float32 var_23;
    bool var_24;
    bool var_25;
    bool var_26;
    bool var_27;
    float32 var_28;
    vec2 var_29;
    vec2 var_30;
    vec3 var_31;
    float32 var_32;
    float32 var_33;
    bool var_34;
    bool var_35;
    bool var_36;
    vec2 var_37;
    vec2 var_38;
    float32 var_39;
    float32 var_40;
    float32 var_41;
    float32 var_42;
    float32 var_43;
    bool var_44;
    bool var_45;
    bool var_46;
    bool var_47;
    float32 var_48;
    vec2 var_49;
    vec2 var_50;
    float32 var_51;
    float32 var_52;
    float32 var_53;
    float32 var_54;
    float32 var_55;
    float32 var_56;
    float32 var_57;
    float32 var_58;
    bool var_59;
    float32 var_60;
    bool var_61;
    float32 var_62;
    bool var_63;
    bool var_64;
    float32 var_65;
    vec2 var_66;
    vec2 var_67;
    float32 var_68;
    float32 var_69;
    float32 var_70;
    float32 var_71;
    float32 var_72;
    float32 var_73;
    float32 var_74;
    vec2 var_75;
    //---------
    // dual vars
    vec3 adj_0 = 0;
    vec3 adj_1 = 0;
    vec3 adj_2 = 0;
    float32 adj_3 = 0;
    float32 adj_4 = 0;
    float32 adj_5 = 0;
    bool adj_6 = 0;
    bool adj_7 = 0;
    bool adj_8 = 0;
    float32 adj_9 = 0;
    vec2 adj_10 = 0;
    vec3 adj_11 = 0;
    float32 adj_12 = 0;
    float32 adj_13 = 0;
    bool adj_14 = 0;
    bool adj_15 = 0;
    bool adj_16 = 0;
    vec2 adj_17 = 0;
    vec2 adj_18 = 0;
    float32 adj_19 = 0;
    float32 adj_20 = 0;
    float32 adj_21 = 0;
    float32 adj_22 = 0;
    float32 adj_23 = 0;
    bool adj_24 = 0;
    bool adj_25 = 0;
    bool adj_26 = 0;
    bool adj_27 = 0;
    float32 adj_28 = 0;
    vec2 adj_29 = 0;
    vec2 adj_30 = 0;
    vec3 adj_31 = 0;
    float32 adj_32 = 0;
    float32 adj_33 = 0;
    bool adj_34 = 0;
    bool adj_35 = 0;
    bool adj_36 = 0;
    vec2 adj_37 = 0;
    vec2 adj_38 = 0;
    float32 adj_39 = 0;
    float32 adj_40 = 0;
    float32 adj_41 = 0;
    float32 adj_42 = 0;
    float32 adj_43 = 0;
    bool adj_44 = 0;
    bool adj_45 = 0;
    bool adj_46 = 0;
    bool adj_47 = 0;
    float32 adj_48 = 0;
    vec2 adj_49 = 0;
    vec2 adj_50 = 0;
    float32 adj_51 = 0;
    float32 adj_52 = 0;
    float32 adj_53 = 0;
    float32 adj_54 = 0;
    float32 adj_55 = 0;
    float32 adj_56 = 0;
    float32 adj_57 = 0;
    float32 adj_58 = 0;
    bool adj_59 = 0;
    float32 adj_60 = 0;
    bool adj_61 = 0;
    float32 adj_62 = 0;
    bool adj_63 = 0;
    bool adj_64 = 0;
    float32 adj_65 = 0;
    vec2 adj_66 = 0;
    vec2 adj_67 = 0;
    float32 adj_68 = 0;
    float32 adj_69 = 0;
    float32 adj_70 = 0;
    float32 adj_71 = 0;
    float32 adj_72 = 0;
    float32 adj_73 = 0;
    float32 adj_74 = 0;
    vec2 adj_75 = 0;
    //---------
    // forward
    var_0 = wp::sub(var_b, var_a);
    var_1 = wp::sub(var_c, var_a);
    var_2 = wp::sub(var_p, var_a);
    var_3 = wp::dot(var_0, var_2);
    var_4 = wp::dot(var_1, var_2);
    var_6 = (var_3 <= var_5);
    var_7 = (var_4 <= var_5);
    var_8 = var_6 && var_7;
    if (var_8) {
    	var_10 = wp::vec2(var_9, var_5);
    	goto label0;
    }
    var_11 = wp::sub(var_p, var_b);
    var_12 = wp::dot(var_0, var_11);
    var_13 = wp::dot(var_1, var_11);
    var_14 = (var_12 >= var_5);
    var_15 = (var_13 <= var_12);
    var_16 = var_14 && var_15;
    if (var_16) {
    	var_17 = wp::vec2(var_5, var_9);
    	goto label1;
    }
    var_18 = wp::select(var_16, var_10, var_17);
    var_19 = wp::mul(var_3, var_13);
    var_20 = wp::mul(var_12, var_4);
    var_21 = wp::sub(var_19, var_20);
    var_22 = wp::sub(var_3, var_12);
    var_23 = wp::div(var_3, var_22);
    var_24 = (var_21 <= var_5);
    var_25 = (var_3 >= var_5);
    var_26 = (var_12 <= var_5);
    var_27 = var_24 && var_25 && var_26;
    if (var_27) {
    	var_28 = wp::sub(var_9, var_23);
    	var_29 = wp::vec2(var_28, var_23);
    	goto label2;
    }
    var_30 = wp::select(var_27, var_18, var_29);
    var_31 = wp::sub(var_p, var_c);
    var_32 = wp::dot(var_0, var_31);
    var_33 = wp::dot(var_1, var_31);
    var_34 = (var_33 >= var_5);
    var_35 = (var_32 <= var_33);
    var_36 = var_34 && var_35;
    if (var_36) {
    	var_37 = wp::vec2(var_5, var_5);
    	goto label3;
    }
    var_38 = wp::select(var_36, var_30, var_37);
    var_39 = wp::mul(var_32, var_4);
    var_40 = wp::mul(var_3, var_33);
    var_41 = wp::sub(var_39, var_40);
    var_42 = wp::sub(var_4, var_33);
    var_43 = wp::div(var_4, var_42);
    var_44 = (var_41 <= var_5);
    var_45 = (var_4 >= var_5);
    var_46 = (var_33 <= var_5);
    var_47 = var_44 && var_45 && var_46;
    if (var_47) {
    	var_48 = wp::sub(var_9, var_43);
    	var_49 = wp::vec2(var_48, var_5);
    	goto label4;
    }
    var_50 = wp::select(var_47, var_38, var_49);
    var_51 = wp::mul(var_12, var_33);
    var_52 = wp::mul(var_32, var_13);
    var_53 = wp::sub(var_51, var_52);
    var_54 = wp::sub(var_13, var_12);
    var_55 = wp::sub(var_13, var_12);
    var_56 = wp::sub(var_32, var_33);
    var_57 = wp::add(var_55, var_56);
    var_58 = wp::div(var_54, var_57);
    var_59 = (var_53 <= var_5);
    var_60 = wp::sub(var_13, var_12);
    var_61 = (var_60 >= var_5);
    var_62 = wp::sub(var_32, var_33);
    var_63 = (var_62 >= var_5);
    var_64 = var_59 && var_61 && var_63;
    if (var_64) {
    	var_65 = wp::sub(var_9, var_58);
    	var_66 = wp::vec2(var_5, var_65);
    	goto label5;
    }
    var_67 = wp::select(var_64, var_50, var_66);
    var_68 = wp::add(var_53, var_41);
    var_69 = wp::add(var_68, var_21);
    var_70 = wp::div(var_9, var_69);
    var_71 = wp::mul(var_41, var_70);
    var_72 = wp::mul(var_21, var_70);
    var_73 = wp::sub(var_9, var_71);
    var_74 = wp::sub(var_73, var_72);
    var_75 = wp::vec2(var_74, var_71);
    goto label6;
    //---------
    // reverse
    label6:;
    adj_75 += adj_ret;
    wp::adj_vec2(var_74, var_71, adj_74, adj_71, adj_75);
    wp::adj_sub(var_73, var_72, adj_73, adj_72, adj_74);
    wp::adj_sub(var_9, var_71, adj_9, adj_71, adj_73);
    wp::adj_mul(var_21, var_70, adj_21, adj_70, adj_72);
    wp::adj_mul(var_41, var_70, adj_41, adj_70, adj_71);
    wp::adj_div(var_9, var_69, adj_9, adj_69, adj_70);
    wp::adj_add(var_68, var_21, adj_68, adj_21, adj_69);
    wp::adj_add(var_53, var_41, adj_53, adj_41, adj_68);
    wp::adj_select(var_64, var_50, var_66, adj_64, adj_50, adj_66, adj_67);
    if (var_64) {
    	label5:;
    	adj_66 += adj_ret;
    	wp::adj_vec2(var_5, var_65, adj_5, adj_65, adj_66);
    	wp::adj_sub(var_9, var_58, adj_9, adj_58, adj_65);
    }
    wp::adj_sub(var_32, var_33, adj_32, adj_33, adj_62);
    wp::adj_sub(var_13, var_12, adj_13, adj_12, adj_60);
    wp::adj_div(var_54, var_57, adj_54, adj_57, adj_58);
    wp::adj_add(var_55, var_56, adj_55, adj_56, adj_57);
    wp::adj_sub(var_32, var_33, adj_32, adj_33, adj_56);
    wp::adj_sub(var_13, var_12, adj_13, adj_12, adj_55);
    wp::adj_sub(var_13, var_12, adj_13, adj_12, adj_54);
    wp::adj_sub(var_51, var_52, adj_51, adj_52, adj_53);
    wp::adj_mul(var_32, var_13, adj_32, adj_13, adj_52);
    wp::adj_mul(var_12, var_33, adj_12, adj_33, adj_51);
    wp::adj_select(var_47, var_38, var_49, adj_47, adj_38, adj_49, adj_50);
    if (var_47) {
    	label4:;
    	adj_49 += adj_ret;
    	wp::adj_vec2(var_48, var_5, adj_48, adj_5, adj_49);
    	wp::adj_sub(var_9, var_43, adj_9, adj_43, adj_48);
    }
    wp::adj_div(var_4, var_42, adj_4, adj_42, adj_43);
    wp::adj_sub(var_4, var_33, adj_4, adj_33, adj_42);
    wp::adj_sub(var_39, var_40, adj_39, adj_40, adj_41);
    wp::adj_mul(var_3, var_33, adj_3, adj_33, adj_40);
    wp::adj_mul(var_32, var_4, adj_32, adj_4, adj_39);
    wp::adj_select(var_36, var_30, var_37, adj_36, adj_30, adj_37, adj_38);
    if (var_36) {
    	label3:;
    	adj_37 += adj_ret;
    	wp::adj_vec2(var_5, var_5, adj_5, adj_5, adj_37);
    }
    wp::adj_dot(var_1, var_31, adj_1, adj_31, adj_33);
    wp::adj_dot(var_0, var_31, adj_0, adj_31, adj_32);
    wp::adj_sub(var_p, var_c, adj_p, adj_c, adj_31);
    wp::adj_select(var_27, var_18, var_29, adj_27, adj_18, adj_29, adj_30);
    if (var_27) {
    	label2:;
    	adj_29 += adj_ret;
    	wp::adj_vec2(var_28, var_23, adj_28, adj_23, adj_29);
    	wp::adj_sub(var_9, var_23, adj_9, adj_23, adj_28);
    }
    wp::adj_div(var_3, var_22, adj_3, adj_22, adj_23);
    wp::adj_sub(var_3, var_12, adj_3, adj_12, adj_22);
    wp::adj_sub(var_19, var_20, adj_19, adj_20, adj_21);
    wp::adj_mul(var_12, var_4, adj_12, adj_4, adj_20);
    wp::adj_mul(var_3, var_13, adj_3, adj_13, adj_19);
    wp::adj_select(var_16, var_10, var_17, adj_16, adj_10, adj_17, adj_18);
    if (var_16) {
    	label1:;
    	adj_17 += adj_ret;
    	wp::adj_vec2(var_5, var_9, adj_5, adj_9, adj_17);
    }
    wp::adj_dot(var_1, var_11, adj_1, adj_11, adj_13);
    wp::adj_dot(var_0, var_11, adj_0, adj_11, adj_12);
    wp::adj_sub(var_p, var_b, adj_p, adj_b, adj_11);
    if (var_8) {
    	label0:;
    	adj_10 += adj_ret;
    	wp::adj_vec2(var_9, var_5, adj_9, adj_5, adj_10);
    }
    wp::adj_dot(var_1, var_2, adj_1, adj_2, adj_4);
    wp::adj_dot(var_0, var_2, adj_0, adj_2, adj_3);
    wp::adj_sub(var_p, var_a, adj_p, adj_a, adj_2);
    wp::adj_sub(var_c, var_a, adj_c, adj_a, adj_1);
    wp::adj_sub(var_b, var_a, adj_b, adj_a, adj_0);
    return;

}

} // namespace wp
