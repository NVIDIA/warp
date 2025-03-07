/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// This code is adapted from https://github.com/alecjacobson/WindingNumber/tree/1e6081e52905575d8e98fb8b7c0921274a18752f
// The original license is below:
/*
MIT License

Copyright (c) 2018 Side Effects Software Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
namespace wp
{

class SolidAngleProps{
public:
	vec3 average_p;
	vec3 normal;
	
	vec3 n_ij_diag;
    vec3 n_ijk_diag; 

    float sum_permute_n_xyz; 
    float two_n_xxy_n_yxx;
    float two_n_xxz_n_zxx;
    float two_n_yyz_n_zyy;
    float two_n_yyx_n_xyy;
    float two_n_zzx_n_xzz;
    float two_n_zzy_n_yzz; 

    float n_xy; float n_yx;
    float n_yz; float n_zy;
    float n_zx; float n_xz;	
	
	bounds3 box;
	vec3 area_P;
	float area;	
	float max_p_dist_sq;
};

CUDA_CALLABLE inline void compute_integrals(
	const vec3 &a,
	const vec3 &b,
	const vec3 &c,
	const vec3 &P,
	float *integral_ii,
	float *integral_ij,
	float *integral_ik,
	const int i)
{

	// NOTE: a, b, and c must be in order of the i axis.
	// We're splitting the triangle at the middle i coordinate.
	const vec3 oab = b - a;
	const vec3 oac = c - a;
	const vec3 ocb = b - c;
	const float t = oab[i] / oac[i];

	const int j = (i == 2) ? 0 : (i + 1);
	const int k = (j == 2) ? 0 : (j + 1);
	const float jdiff = t * oac[j] - oab[j];
	const float kdiff = t * oac[k] - oab[k];
	vec3 cross_a;
	cross_a[0] = (jdiff * oab[k] - kdiff * oab[j]);
	cross_a[1] = kdiff * oab[i];
	cross_a[2] = jdiff * oab[i];
	vec3 cross_c;
	cross_c[0] = (jdiff * ocb[k] - kdiff * ocb[j]);
	cross_c[1] = kdiff * ocb[i];
	cross_c[2] = jdiff * ocb[i];
	const float area_scale_a = length(cross_a);
	const float area_scale_c = length(cross_c);
	const float Pai = a[i] - P[i];
	const float Pci = c[i] - P[i];

	// Integral over the area of the triangle of (pi^2)dA,
	// by splitting the triangle into two at b, the a side
	// and the c side.
	const float int_ii_a = area_scale_a * (0.5f * Pai * Pai + (2.0f / 3.0f) * Pai * oab[i] + 0.25f * oab[i] * oab[i]);
	const float int_ii_c = area_scale_c * (0.5f * Pci * Pci + (2.0f / 3.0f) * Pci * ocb[i] + 0.25f * ocb[i] * ocb[i]);
	*integral_ii = int_ii_a + int_ii_c;

	int jk = j;
	float *integral = integral_ij;
	float diff = jdiff;
	while (true) // This only does 2 iterations, one for j and one for k
	{
		if (integral)
		{
			float obmidj = b[jk] + 0.5f * diff;
			float oabmidj = obmidj - a[jk];
			float ocbmidj = obmidj - c[jk];
			float Paj = a[jk] - P[jk];
			float Pcj = c[jk] - P[jk];
			// Integral over the area of the triangle of (pi*pj)dA
			const float int_ij_a = area_scale_a * (0.5f * Pai * Paj + (1.0f / 3.0f) * Pai * oabmidj + (1.0f / 3.0f) * Paj * oab[i] + 0.25f * oab[i] * oabmidj);
			const float int_ij_c = area_scale_c * (0.5f * Pci * Pcj + (1.0f / 3.0f) * Pci * ocbmidj + (1.0f / 3.0f) * Pcj * ocb[i] + 0.25f * ocb[i] * ocbmidj);
			*integral = int_ij_a + int_ij_c;
		}
		if (jk == k)
			break;
		jk = k;
		integral = integral_ik;
		diff = kdiff;
	}
};

CUDA_CALLABLE inline void my_swap(int &a, int &b)
{
	int c = a;
	a = b;
	b = c;
}

CUDA_CALLABLE inline void precompute_triangle_solid_angle_props(const vec3 &a, const vec3 &b, const vec3 &c, SolidAngleProps &my_data)
{
	const vec3 ab = b - a;
	const vec3 ac = c - a;

	// Are weighted normal
	const vec3 N = 0.5f * cross(ab, ac);
	const float area2 = length_sq(N);
	const float area = sqrtf(area2);
	const vec3 P = (a + b + c) / 3.0f;
	my_data.box.add_point(a);
	my_data.box.add_point(b);
	my_data.box.add_point(c);
	my_data.average_p = P;
	my_data.area_P = P * area;
	my_data.normal = N;

	my_data.area = area;

	// NOTE: Due to P being at the centroid, triangles have Nij = 0
	//       contributions to Nij.
	my_data.n_ij_diag = 0.0f;
	my_data.n_xy = 0.0f;
	my_data.n_yx = 0.0f;
	my_data.n_yz = 0.0f;
	my_data.n_zy = 0.0f;
	my_data.n_zx = 0.0f;
	my_data.n_xz = 0.0f;

	// If it's zero-length, the results are zero, so we can skip.
	if (area == 0)
	{
		my_data.n_ijk_diag = 0.0f;
		my_data.sum_permute_n_xyz = 0.0f;
		my_data.two_n_xxy_n_yxx = 0.0f;
		my_data.two_n_xxz_n_zxx = 0.0f;
		my_data.two_n_yyz_n_zyy = 0.0f;
		my_data.two_n_yyx_n_xyy = 0.0f;
		my_data.two_n_zzx_n_xzz = 0.0f;
		my_data.two_n_zzy_n_yzz = 0.0f;
		return;
	}

	// We need to use the NORMALIZED normal to multiply the integrals by.
	vec3 n = N / area;

	// Figure out the order of a, b, and c in x, y, and z
	// for use in computing the integrals for Nijk.
	vec3 values[3] = {a, b, c};

	int order_x[3] = {0, 1, 2};
	if (a[0] > b[0])
		my_swap(order_x[0], order_x[1]);
	if (values[order_x[0]][0] > c[0])
		my_swap(order_x[0], order_x[2]);
	if (values[order_x[1]][0] > values[order_x[2]][0])
		my_swap(order_x[1], order_x[2]);
	float dx = values[order_x[2]][0] - values[order_x[0]][0];

	int order_y[3] = {0, 1, 2};
	if (a[1] > b[1])
		my_swap(order_y[0], order_y[1]);
	if (values[order_y[0]][1] > c[1])
		my_swap(order_y[0], order_y[2]);
	if (values[order_y[1]][1] > values[order_y[2]][1])
		my_swap(order_y[1], order_y[2]);
	float dy = values[order_y[2]][1] - values[order_y[0]][1];

	int order_z[3] = {0, 1, 2};
	if (a[2] > b[2])
		my_swap(order_z[0], order_z[1]);
	if (values[order_z[0]][2] > c[2])
		my_swap(order_z[0], order_z[2]);
	if (values[order_z[1]][2] > values[order_z[2]][2])
		my_swap(order_z[1], order_z[2]);
	float dz = values[order_z[2]][2] - values[order_z[0]][2];

	float integral_xx = 0.0f;
	float integral_xy = 0.0f;
	float integral_yy = 0.0f;
	float integral_yz = 0.0f;
	float integral_zz = 0.0f;
	float integral_zx = 0.0f;
	// Note that if the span of any axis is zero, the integral must be zero,
	// since there's a factor of (p_i-P_i), i.e. value minus average,
	// and every value must be equal to the average, giving zero.
	if (dx > 0)
	{
		compute_integrals(
			values[order_x[0]], values[order_x[1]], values[order_x[2]], P,
			&integral_xx, ((dx >= dy && dy > 0) ? &integral_xy : nullptr), ((dx >= dz && dz > 0) ? &integral_zx : nullptr), 0);
	}
	if (dy > 0)
	{
		compute_integrals(
			values[order_y[0]], values[order_y[1]], values[order_y[2]], P,
			&integral_yy, ((dy >= dz && dz > 0) ? &integral_yz : nullptr), ((dx < dy && dx > 0) ? &integral_xy : nullptr), 1);
	}
	if (dz > 0)
	{
		compute_integrals(
			values[order_z[0]], values[order_z[1]], values[order_z[2]], P,
			&integral_zz, ((dx < dz && dx > 0) ? &integral_zx : nullptr), ((dy < dz && dy > 0) ? &integral_yz : nullptr), 2);
	}

	vec3 Niii(integral_xx, integral_yy, integral_zz);
	Niii = cw_mul(Niii, n);
	my_data.n_ijk_diag = Niii;
	my_data.sum_permute_n_xyz = 2.0f * (n[0] * integral_yz + n[1] * integral_zx + n[2] * integral_xy);
	float Nxxy = n[0] * integral_xy;
	float Nxxz = n[0] * integral_zx;
	float Nyyz = n[1] * integral_yz;
	float Nyyx = n[1] * integral_xy;
	float Nzzx = n[2] * integral_zx;
	float Nzzy = n[2] * integral_yz;
	my_data.two_n_xxy_n_yxx = 2.0f * Nxxy + n[1] * integral_xx;
	my_data.two_n_xxz_n_zxx = 2.0f * Nxxz + n[2] * integral_xx;
	my_data.two_n_yyz_n_zyy = 2.0f * Nyyz + n[2] * integral_yy;
	my_data.two_n_yyx_n_xyy = 2.0f * Nyyx + n[0] * integral_yy;
	my_data.two_n_zzx_n_xzz = 2.0f * Nzzx + n[0] * integral_zz;
	my_data.two_n_zzy_n_yzz = 2.0f * Nzzy + n[1] * integral_zz;
}

CUDA_CALLABLE inline void combine_precomputed_solid_angle_props(SolidAngleProps &my_data, const SolidAngleProps *left_child_data, const SolidAngleProps *right_child_data)
{
	vec3 N = left_child_data->normal;
	vec3 areaP = left_child_data->area_P;
	float area = left_child_data->area;
	if (right_child_data)
	{
		const vec3 local_N = right_child_data->normal;
		N += local_N;
		areaP += right_child_data->area_P;
		area += right_child_data->area;
	}
	my_data.normal = N;
	my_data.area_P = areaP;
	my_data.area = area;
	bounds3 box(left_child_data->box);
	if (right_child_data)
	{
		box = bounds_union(box, right_child_data->box);
	}

	// Normalize P
	vec3 averageP;
	if (area > 0)
	{
		averageP = areaP / area;
	}
	else
	{
		averageP = 0.5f * (box.lower + box.upper);
	}
	my_data.average_p = averageP;

	my_data.box = box;

	// We now have the current box's P, so we can adjust Nij and Nijk
	my_data.n_ij_diag = left_child_data->n_ij_diag;
	my_data.n_xy = 0.0f;
	my_data.n_yx = 0.0f;
	my_data.n_yz = 0.0f;
	my_data.n_zy = 0.0f;
	my_data.n_zx = 0.0f;
	my_data.n_xz = 0.0f;

	my_data.n_ijk_diag = left_child_data->n_ijk_diag;
	my_data.sum_permute_n_xyz = left_child_data->sum_permute_n_xyz;
	my_data.two_n_xxy_n_yxx = left_child_data->two_n_xxy_n_yxx;
	my_data.two_n_xxz_n_zxx = left_child_data->two_n_xxz_n_zxx;
	my_data.two_n_yyz_n_zyy = left_child_data->two_n_yyz_n_zyy;
	my_data.two_n_yyx_n_xyy = left_child_data->two_n_yyx_n_xyy;
	my_data.two_n_zzx_n_xzz = left_child_data->two_n_zzx_n_xzz;
	my_data.two_n_zzy_n_yzz = left_child_data->two_n_zzy_n_yzz;

	if (right_child_data)
	{
		my_data.n_ij_diag += right_child_data->n_ij_diag;
		my_data.n_ijk_diag += right_child_data->n_ijk_diag;
		my_data.sum_permute_n_xyz += right_child_data->sum_permute_n_xyz;
		my_data.two_n_xxy_n_yxx += right_child_data->two_n_xxy_n_yxx;
		my_data.two_n_xxz_n_zxx += right_child_data->two_n_xxz_n_zxx;
		my_data.two_n_yyz_n_zyy += right_child_data->two_n_yyz_n_zyy;
		my_data.two_n_yyx_n_xyy += right_child_data->two_n_yyx_n_xyy;
		my_data.two_n_zzx_n_xzz += right_child_data->two_n_zzx_n_xzz;
		my_data.two_n_zzy_n_yzz += right_child_data->two_n_zzy_n_yzz;
	}

	for (int i = 0; i < (right_child_data ? 2 : 1); ++i)
	{
		const SolidAngleProps &child_data = (i == 0) ? *left_child_data : *right_child_data;
		vec3 displacement = child_data.average_p - vec3(my_data.average_p);
		vec3 N = child_data.normal;

		// Adjust Nij for the change in centre P
		my_data.n_ij_diag += cw_mul(N, displacement);
		float Nxy = child_data.n_xy + N[0] * displacement[1];
		float Nyx = child_data.n_yx + N[1] * displacement[0];
		float Nyz = child_data.n_yz + N[1] * displacement[2];
		float Nzy = child_data.n_zy + N[2] * displacement[1];
		float Nzx = child_data.n_zx + N[2] * displacement[0];
		float Nxz = child_data.n_xz + N[0] * displacement[2];

		my_data.n_xy += Nxy;
		my_data.n_yx += Nyx;
		my_data.n_yz += Nyz;
		my_data.n_zy += Nzy;
		my_data.n_zx += Nzx;
		my_data.n_xz += Nxz;

		// Adjust Nijk for the change in centre P
		my_data.n_ijk_diag += 2.0f * cw_mul(displacement, child_data.n_ij_diag) + cw_mul(displacement, cw_mul(displacement, child_data.normal));
		my_data.sum_permute_n_xyz += (displacement[0] * (Nyz + Nzy) + displacement[1] * (Nzx + Nxz) + displacement[2] * (Nxy + Nyx));
		my_data.two_n_xxy_n_yxx +=
			2 * (displacement[1] * child_data.n_ij_diag[0] + displacement[0] * child_data.n_xy + N[0] * displacement[0] * displacement[1]) + 2 * child_data.n_yx * displacement[0] + N[1] * displacement[0] * displacement[0];
		my_data.two_n_xxz_n_zxx +=
			2 * (displacement[2] * child_data.n_ij_diag[0] + displacement[0] * child_data.n_xz + N[0] * displacement[0] * displacement[2]) + 2 * child_data.n_zx * displacement[0] + N[2] * displacement[0] * displacement[0];
		my_data.two_n_yyz_n_zyy +=
			2 * (displacement[2] * child_data.n_ij_diag[1] + displacement[1] * child_data.n_yz + N[1] * displacement[1] * displacement[2]) + 2 * child_data.n_zy * displacement[1] + N[2] * displacement[1] * displacement[1];
		my_data.two_n_yyx_n_xyy +=
			2 * (displacement[0] * child_data.n_ij_diag[1] + displacement[1] * child_data.n_yx + N[1] * displacement[1] * displacement[0]) + 2 * child_data.n_xy * displacement[1] + N[0] * displacement[1] * displacement[1];
		my_data.two_n_zzx_n_xzz +=
			2 * (displacement[0] * child_data.n_ij_diag[2] + displacement[2] * child_data.n_zx + N[2] * displacement[2] * displacement[0]) + 2 * child_data.n_xz * displacement[2] + N[0] * displacement[2] * displacement[2];
		my_data.two_n_zzy_n_yzz +=
			2 * (displacement[1] * child_data.n_ij_diag[2] + displacement[2] * child_data.n_zy + N[2] * displacement[2] * displacement[1]) + 2 * child_data.n_yz * displacement[2] + N[1] * displacement[2] * displacement[2];
	}

	my_data.max_p_dist_sq = length_sq(max(my_data.average_p - my_data.box.lower, my_data.box.upper - my_data.average_p));
}

CUDA_CALLABLE inline SolidAngleProps combine_precomputed_solid_angle_props(const SolidAngleProps* left_child_data, const SolidAngleProps* right_child_data)
{
	SolidAngleProps my_data;
	combine_precomputed_solid_angle_props(my_data, left_child_data, right_child_data);
	return my_data;
}

// Return whether need to
CUDA_CALLABLE inline bool evaluate_node_solid_angle(const vec3 &query_point, SolidAngleProps *current_data, float &solid_angle, const float accuracy_scale_sq)
{
	SolidAngleProps &data = *current_data;
	float max_p_sq = data.max_p_dist_sq;
	vec3 q = query_point - data.average_p;
	float qlength2 = length_sq(q);

	if (qlength2 <= max_p_sq * accuracy_scale_sq)
	{
		solid_angle = 0.0f;
		return true;
	}
	float omega_approx = 0.0f;
	// qlength2 must be non-zero, since it's strictly greater than something.
	// We still need to be careful for NaNs, though, because the 4th power might cause problems.
	float qlength_m2 = 1.0f / qlength2;
	float qlength_m1 = sqrtf(qlength_m2);

	// Normalize q to reduce issues with overflow/underflow, since we'd need the 7th power
	// if we didn't normalize, and (1e-6)^-7 = 1e42, which overflows single-precision.
	q = q * qlength_m1;

	omega_approx = -qlength_m2 * dot(q, data.normal);
	vec3 q2 = cw_mul(q, q);

	float qlength_m3 = qlength_m2 * qlength_m1;

	float omega_1 = qlength_m3 * (data.n_ij_diag[0] + data.n_ij_diag[1] + data.n_ij_diag[2] - 3.0f * (dot(q2, data.n_ij_diag) + q[0] * q[1] * (data.n_xy + data.n_yx) + q[0] * q[2] * (data.n_zx + data.n_xz) + q[1] * q[2] * (data.n_yz + data.n_zy)));
	omega_approx += omega_1;

	vec3 q3 = cw_mul(q2, q);
	float qlength_m4 = qlength_m2 * qlength_m2;
	
	vec3 temp0(data.two_n_yyx_n_xyy + data.two_n_zzx_n_xzz, data.two_n_zzy_n_yzz + data.two_n_xxy_n_yxx, data.two_n_xxz_n_zxx + data.two_n_yyz_n_zyy);
	vec3 temp1(q[1] * data.two_n_xxy_n_yxx + q[2] * data.two_n_xxz_n_zxx, q[2] * data.two_n_yyz_n_zyy + q[0] * data.two_n_yyx_n_xyy, q[0] * data.two_n_zzx_n_xzz + q[1] * data.two_n_zzy_n_yzz);

	float omega_2 = qlength_m4 * (1.5f * dot(q, 3.0f * data.n_ijk_diag + temp0) - 7.5f * (dot(q3, data.n_ijk_diag) + q[0] * q[1] * q[2] * data.sum_permute_n_xyz + dot(q2, temp1)));
	omega_approx += omega_2;

	// Safety check if not finite, need to descend instead
	if (!isfinite(omega_approx))
	{
		omega_approx = 0.0f;
		solid_angle = 0.0;
		return true;
	}

	solid_angle = omega_approx;
	return false;
}

CUDA_CALLABLE inline float robust_solid_angle(
	const vec3 &a,
	const vec3 &b,
	const vec3 &c,
	const vec3 &p)
{
	vec3 qa = a - p;
	vec3 qb = b - p;
	vec3 qc = c - p;

	const float a_length = length(qa);
	const float b_length = length(qb);
	const float c_length = length(qc);

	if (a_length == 0.0f || b_length == 0.0f || c_length == 0.0f)
		return 0.0f;

	qa = qa / a_length;
	qb = qb / b_length;
	qc = qc / c_length;

	const float numerator = dot(qa, cross(qb - qa, qc - qa));

	if (numerator == 0.0f)
		return 0.0f;

	const float denominator = 1.0f + dot(qa, qb) + dot(qa, qc) + dot(qb, qc);

	return 2.0f * atan2(numerator, denominator);
}
}
