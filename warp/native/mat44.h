/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

namespace wp
{

//----------------------------------------------------------
// mat44

struct mat44
{
    inline CUDA_CALLABLE mat44(vec4 c0, vec4 c1, vec4 c2, vec4 c3)
    {
        data[0][0] = c0.x;
        data[1][0] = c0.y;
        data[2][0] = c0.z;
        data[3][0] = c0.w;

        data[0][1] = c1.x;
        data[1][1] = c1.y;
        data[2][1] = c1.z;
        data[3][1] = c1.w;

        data[0][2] = c2.x;
        data[1][2] = c2.y;
        data[2][2] = c2.z;
        data[3][2] = c2.w;

        data[0][3] = c3.x;
        data[1][3] = c3.y;
        data[2][3] = c3.z;
        data[3][3] = c3.w;
    }

    inline CUDA_CALLABLE mat44(
                 float m00=0.0f, float m01=0.0f, float m02=0.0f, float m03=0.0f,
                 float m10=0.0f, float m11=0.0f, float m12=0.0f, float m13=0.0f,
                 float m20=0.0f, float m21=0.0f, float m22=0.0f, float m23=0.0f,
                 float m30=0.0f, float m31=0.0f, float m32=0.0f, float m33=0.0f) 
    {
        data[0][0] = m00;
        data[1][0] = m10;
        data[2][0] = m20;
        data[3][0] = m30;

        data[0][1] = m01;
        data[1][1] = m11;
        data[2][1] = m21;
        data[3][1] = m31;

        data[0][2] = m02;
        data[1][2] = m12;
        data[2][2] = m22;
        data[3][2] = m32;

        data[0][3] = m03;
        data[1][3] = m13;
        data[2][3] = m23;
        data[3][3] = m33;
    }

    inline CUDA_CALLABLE mat44(const vec3& pos, const quat& rot, const vec3& scale)
    {
        mat33 R = quat_to_matrix(rot);

        data[0][0] = R.data[0][0]*scale.x;
        data[1][0] = R.data[1][0]*scale.x;
        data[2][0] = R.data[2][0]*scale.x;
        data[3][0] = 0.0f;

        data[0][1] = R.data[0][1]*scale.y;
        data[1][1] = R.data[1][1]*scale.y;
        data[2][1] = R.data[2][1]*scale.y;
        data[3][1] = 0.0f;

        data[0][2] = R.data[0][2]*scale.z;
        data[1][2] = R.data[1][2]*scale.z;
        data[2][2] = R.data[2][2]*scale.z;
        data[3][2] = 0.0f;

        data[0][3] = pos.x;
        data[1][3] = pos.y;
        data[2][3] = pos.z;
        data[3][3] = 1.0f;        
    }

    CUDA_CALLABLE vec4 get_row(int index) const
    {
        return (vec4&)data[index]; 
    }

    CUDA_CALLABLE void set_row(int index, const vec4& v)
    {
        (vec4&)data[index] = v;
    }

    CUDA_CALLABLE vec4 get_col(int index) const
    {
        return vec4(data[0][index], data[1][index], data[2][index], data[3][index]);
    }

    CUDA_CALLABLE void set_col(int index, const vec4& v)
    {
        data[0][index] = v.x;
        data[1][index] = v.y;
        data[2][index] = v.z;
        data[3][index] = v.w;
    }

    // row major storage assumed to be compatible with PyTorch
    float data[4][4];
};

inline CUDA_CALLABLE bool operator==(const mat44& a, const mat44& b)
{
    for (int i=0; i < 4; ++i)
        for (int j=0; j < 4; ++j)
            if (a.data[i][j] != b.data[i][j])
                return false;

    return true;
}


inline CUDA_CALLABLE mat44 diag(const vec4& d) {
  return mat44(d.x, 0.f, 0.f, 0.f,
               0.f, d.y, 0.f, 0.f,
               0.f, 0.f, d.z, 0.f,
               0.f, 0.f, 0.f, d.w);
}

inline CUDA_CALLABLE void adj_diag(const vec4& d, vec4& adj_d, const mat44& adj_ret) {
  adj_d += vec4(adj_ret.data[0][0], adj_ret.data[1][1], adj_ret.data[2][2], adj_ret.data[3][3]);
}

inline CUDA_CALLABLE mat44 atomic_add(mat44 * addr, mat44 value) 
{
    mat44 m;
    
    for (int i=0; i < 4; ++i)
        for (int j=0; j < 4; ++j)
            m.data[i][j] = atomic_add(&addr->data[i][j], value.data[i][j]);

    return m;
}


inline CUDA_CALLABLE void adj_mat44(
    vec4 c0, vec4 c1, vec4 c2, vec4 c3,
    vec4& a0, vec4& a1, vec4& a2, vec4& a3,
    const mat44& adj_ret)
{
    // column constructor
    a0 += adj_ret.get_col(0);
    a1 += adj_ret.get_col(1);
    a2 += adj_ret.get_col(2);
    a3 += adj_ret.get_col(3);
}

inline bool CUDA_CALLABLE isfinite(const mat44& m)
{
    for (int i=0; i < 4; ++i)
        for (int j=0; j < 4; ++j)
            if (!::isfinite(m.data[i][j]))
                return false;
    return true;
}

inline CUDA_CALLABLE void adj_mat44(const vec3& pos, const quat& rot, const vec3& scale,
                                    vec3& adj_pos, quat& adj_rot, vec3& adj_scale, const mat44& adj_ret)
{
    mat33 R = quat_to_matrix(rot);
    mat33 adj_R = 0;

    adj_pos.x += adj_ret.data[0][3];
    adj_pos.y += adj_ret.data[1][3];
    adj_pos.z += adj_ret.data[2][3];

    adj_mul(R.data[0][0], scale.x, adj_R.data[0][0], adj_scale.x, adj_ret.data[0][0]);
    adj_mul(R.data[1][0], scale.x, adj_R.data[1][0], adj_scale.x, adj_ret.data[1][0]);
    adj_mul(R.data[2][0], scale.x, adj_R.data[2][0], adj_scale.x, adj_ret.data[2][0]);

    adj_mul(R.data[0][1], scale.y, adj_R.data[0][1], adj_scale.y, adj_ret.data[0][1]);
    adj_mul(R.data[1][1], scale.y, adj_R.data[1][1], adj_scale.y, adj_ret.data[1][1]);
    adj_mul(R.data[2][1], scale.y, adj_R.data[2][1], adj_scale.y, adj_ret.data[2][1]);

    adj_mul(R.data[0][2], scale.z, adj_R.data[0][2], adj_scale.z, adj_ret.data[0][2]);
    adj_mul(R.data[1][2], scale.z, adj_R.data[1][2], adj_scale.z, adj_ret.data[1][2]);
    adj_mul(R.data[2][2], scale.z, adj_R.data[2][2], adj_scale.z, adj_ret.data[2][2]);

    adj_quat_to_matrix(rot, adj_rot, adj_R);
}


inline CUDA_CALLABLE void adj_mat44(float m00, float m01, float m02, float m03,
                      float m10, float m11, float m12, float m13,
                      float m20, float m21, float m22, float m23,
                      float m30, float m31, float m32, float m33,
                      float& a00, float& a01, float& a02, float& a03,
                      float& a10, float& a11, float& a12, float& a13,
                      float& a20, float& a21, float& a22, float& a23,
                      float& a30, float& a31, float& a32, float& a33,
                      const mat44& adj_ret)
{
    printf("todo\n");
}

inline CUDA_CALLABLE float index(const mat44& m, int row, int col)
{
#if FP_CHECK
    if (row < 0 || row > 3)
    {
        printf("mat44 row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
    if (col < 0 || col > 3)
    {
        printf("mat44 col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif
    return m.data[row][col];
}

inline CUDA_CALLABLE vec4 index(const mat44& m, int row)
{
    return vec4(m.data[row][0], m.data[row][1], m.data[row][2], m.data[row][3]);
}

inline CUDA_CALLABLE void adj_index(const mat44& m, int row, mat44& adj_m, int& adj_row, const vec4& adj_ret)
{
    adj_m.data[row][0] += adj_ret[0];
    adj_m.data[row][1] += adj_ret[1];
    adj_m.data[row][2] += adj_ret[2];
    adj_m.data[row][3] += adj_ret[3];
}


inline CUDA_CALLABLE mat44 add(const mat44& a, const mat44& b)
{
    mat44 t;
    for (int i=0; i < 4; ++i)
    {
        for (int j=0; j < 4; ++j)
        {
            t.data[i][j] = a.data[i][j] + b.data[i][j];
        }
    }

    return t;
}

inline CUDA_CALLABLE mat44 sub(const mat44& a, const mat44& b)
{
    mat44 t;
    for (int i=0; i < 4; ++i)
    {
        for (int j=0; j < 4; ++j)
        {
            t.data[i][j] = a.data[i][j] - b.data[i][j];
        }
    }

    return t;
}


inline CUDA_CALLABLE mat44 mul(const mat44& a, float b)
{
    mat44 t;
    for (int i=0; i < 4; ++i)
    {
        for (int j=0; j < 4; ++j)
        {
            t.data[i][j] = a.data[i][j]*b;
        }
    }

    return t;   
}


inline CUDA_CALLABLE vec4 mul(const mat44& a, const vec4& b)
{
    vec4 r = a.get_col(0)*b.x +
             a.get_col(1)*b.y +
             a.get_col(2)*b.z +
             a.get_col(3)*b.w;
    
    return r;
}

inline CUDA_CALLABLE mat44 mul(const mat44& a, const mat44& b)
{
    mat44 t;
    for (int i=0; i < 4; ++i)
    {
        for (int j=0; j < 4; ++j)
        {
            for (int k=0; k < 4; ++k)
            {
                t.data[i][j] += a.data[i][k]*b.data[k][j];
            }
        }
    }

    return t;
}

inline CUDA_CALLABLE mat44 transpose(const mat44& a)
{
    mat44 t;
    for (int i=0; i < 4; ++i)
    {
        for (int j=0; j < 4; ++j)
        {
            t.data[i][j] = a.data[j][i];
        }
    }

    return t;
}

inline CUDA_CALLABLE float determinant(const mat44& m)
{
    // adapted from USD GfMatrix4f::Inverse()
    float x00, x01, x02, x03;
    float x10, x11, x12, x13;
    float x20, x21, x22, x23;
    float x30, x31, x32, x33;
    double y01, y02, y03, y12, y13, y23;
    float z00, z10, z20, z30;

    // Pickle 1st two columns of matrix into registers
    x00 = m.data[0][0];
    x01 = m.data[0][1];
    x10 = m.data[1][0];
    x11 = m.data[1][1];
    x20 = m.data[2][0];
    x21 = m.data[2][1];
    x30 = m.data[3][0];
    x31 = m.data[3][1];

    // Compute all six 2x2 determinants of 1st two columns
    y01 = x00*x11 - x10*x01;
    y02 = x00*x21 - x20*x01;
    y03 = x00*x31 - x30*x01;
    y12 = x10*x21 - x20*x11;
    y13 = x10*x31 - x30*x11;
    y23 = x20*x31 - x30*x21;

    // Pickle 2nd two columns of matrix into registers
    x02 = m.data[0][2];
    x03 = m.data[0][3];
    x12 = m.data[1][2];
    x13 = m.data[1][3];
    x22 = m.data[2][2];
    x23 = m.data[2][3];
    x32 = m.data[3][2];
    x33 = m.data[3][3];

    // Compute all six 2x2 determinants of 2nd two columns
    y01 = x02*x13 - x12*x03;
    y02 = x02*x23 - x22*x03;
    y03 = x02*x33 - x32*x03;
    y12 = x12*x23 - x22*x13;
    y13 = x12*x33 - x32*x13;
    y23 = x22*x33 - x32*x23;

    // Compute all 3x3 cofactors for 1st two columns
    z30 = x11*y02 - x21*y01 - x01*y12;
    z20 = x01*y13 - x11*y03 + x31*y01;
    z10 = x21*y03 - x31*y02 - x01*y23;
    z00 = x11*y23 - x21*y13 + x31*y12;

    // compute 4x4 determinant & its reciprocal
    double det = x30*z30 + x20*z20 + x10*z10 + x00*z00;
    return det;
}

inline CUDA_CALLABLE void adj_determinant(const mat44& m, mat44& adj_m, float adj_ret)
{
    // adapted from USD GfMatrix4f::Inverse()
    float x00, x01, x02, x03;
    float x10, x11, x12, x13;
    float x20, x21, x22, x23;
    float x30, x31, x32, x33;
    double y01, y02, y03, y12, y13, y23;
    float z00, z10, z20, z30;
    float z01, z11, z21, z31;
    double z02, z03, z12, z13, z22, z23, z32, z33;

    // Pickle 1st two columns of matrix into registers
    x00 = m.data[0][0];
    x01 = m.data[0][1];
    x10 = m.data[1][0];
    x11 = m.data[1][1];
    x20 = m.data[2][0];
    x21 = m.data[2][1];
    x30 = m.data[3][0];
    x31 = m.data[3][1];

    // Compute all six 2x2 determinants of 1st two columns
    y01 = x00*x11 - x10*x01;
    y02 = x00*x21 - x20*x01;
    y03 = x00*x31 - x30*x01;
    y12 = x10*x21 - x20*x11;
    y13 = x10*x31 - x30*x11;
    y23 = x20*x31 - x30*x21;

    // Pickle 2nd two columns of matrix into registers
    x02 = m.data[0][2];
    x03 = m.data[0][3];
    x12 = m.data[1][2];
    x13 = m.data[1][3];
    x22 = m.data[2][2];
    x23 = m.data[2][3];
    x32 = m.data[3][2];
    x33 = m.data[3][3];

    // Compute all 3x3 cofactors for 2nd two columns */
    z33 = x02*y12 - x12*y02 + x22*y01;
    z23 = x12*y03 - x32*y01 - x02*y13;
    z13 = x02*y23 - x22*y03 + x32*y02;
    z03 = x22*y13 - x32*y12 - x12*y23;
    z32 = x13*y02 - x23*y01 - x03*y12;
    z22 = x03*y13 - x13*y03 + x33*y01;
    z12 = x23*y03 - x33*y02 - x03*y23;
    z02 = x13*y23 - x23*y13 + x33*y12;

    // Compute all six 2x2 determinants of 2nd two columns
    y01 = x02*x13 - x12*x03;
    y02 = x02*x23 - x22*x03;
    y03 = x02*x33 - x32*x03;
    y12 = x12*x23 - x22*x13;
    y13 = x12*x33 - x32*x13;
    y23 = x22*x33 - x32*x23;

    // Compute all 3x3 cofactors for 1st two columns
    z30 = x11*y02 - x21*y01 - x01*y12;
    z20 = x01*y13 - x11*y03 + x31*y01;
    z10 = x21*y03 - x31*y02 - x01*y23;
    z00 = x11*y23 - x21*y13 + x31*y12;
    z31 = x00*y12 - x10*y02 + x20*y01;
    z21 = x10*y03 - x30*y01 - x00*y13;
    z11 = x00*y23 - x20*y03 + x30*y02;
    z01 = x20*y13 - x30*y12 - x10*y23;

    // Multiply all 3x3 cofactors by adjoint & transpose
    adj_m.data[0][0] += float(z00*adj_ret);
    adj_m.data[0][1] += float(z10*adj_ret);
    adj_m.data[1][0] += float(z01*adj_ret);
    adj_m.data[0][2] += float(z20*adj_ret);
    adj_m.data[2][0] += float(z02*adj_ret);
    adj_m.data[0][3] += float(z30*adj_ret);
    adj_m.data[3][0] += float(z03*adj_ret);
    adj_m.data[1][1] += float(z11*adj_ret);
    adj_m.data[1][2] += float(z21*adj_ret);
    adj_m.data[2][1] += float(z12*adj_ret);
    adj_m.data[1][3] += float(z31*adj_ret);
    adj_m.data[3][1] += float(z13*adj_ret);
    adj_m.data[2][2] += float(z22*adj_ret);
    adj_m.data[2][3] += float(z32*adj_ret);
    adj_m.data[3][2] += float(z23*adj_ret);
    adj_m.data[3][3] += float(z33*adj_ret);

}

inline CUDA_CALLABLE mat44 inverse(const mat44& m)
{
    // adapted from USD GfMatrix4f::Inverse()
    float x00, x01, x02, x03;
    float x10, x11, x12, x13;
    float x20, x21, x22, x23;
    float x30, x31, x32, x33;
    double y01, y02, y03, y12, y13, y23;
    float z00, z10, z20, z30;
    float z01, z11, z21, z31;
    double z02, z03, z12, z13, z22, z23, z32, z33;

    // Pickle 1st two columns of matrix into registers
    x00 = m.data[0][0];
    x01 = m.data[0][1];
    x10 = m.data[1][0];
    x11 = m.data[1][1];
    x20 = m.data[2][0];
    x21 = m.data[2][1];
    x30 = m.data[3][0];
    x31 = m.data[3][1];

    // Compute all six 2x2 determinants of 1st two columns
    y01 = x00*x11 - x10*x01;
    y02 = x00*x21 - x20*x01;
    y03 = x00*x31 - x30*x01;
    y12 = x10*x21 - x20*x11;
    y13 = x10*x31 - x30*x11;
    y23 = x20*x31 - x30*x21;

    // Pickle 2nd two columns of matrix into registers
    x02 = m.data[0][2];
    x03 = m.data[0][3];
    x12 = m.data[1][2];
    x13 = m.data[1][3];
    x22 = m.data[2][2];
    x23 = m.data[2][3];
    x32 = m.data[3][2];
    x33 = m.data[3][3];

    // Compute all 3x3 cofactors for 2nd two columns */
    z33 = x02*y12 - x12*y02 + x22*y01;
    z23 = x12*y03 - x32*y01 - x02*y13;
    z13 = x02*y23 - x22*y03 + x32*y02;
    z03 = x22*y13 - x32*y12 - x12*y23;
    z32 = x13*y02 - x23*y01 - x03*y12;
    z22 = x03*y13 - x13*y03 + x33*y01;
    z12 = x23*y03 - x33*y02 - x03*y23;
    z02 = x13*y23 - x23*y13 + x33*y12;

    // Compute all six 2x2 determinants of 2nd two columns
    y01 = x02*x13 - x12*x03;
    y02 = x02*x23 - x22*x03;
    y03 = x02*x33 - x32*x03;
    y12 = x12*x23 - x22*x13;
    y13 = x12*x33 - x32*x13;
    y23 = x22*x33 - x32*x23;

    // Compute all 3x3 cofactors for 1st two columns
    z30 = x11*y02 - x21*y01 - x01*y12;
    z20 = x01*y13 - x11*y03 + x31*y01;
    z10 = x21*y03 - x31*y02 - x01*y23;
    z00 = x11*y23 - x21*y13 + x31*y12;
    z31 = x00*y12 - x10*y02 + x20*y01;
    z21 = x10*y03 - x30*y01 - x00*y13;
    z11 = x00*y23 - x20*y03 + x30*y02;
    z01 = x20*y13 - x30*y12 - x10*y23;

    // compute 4x4 determinant & its reciprocal
    double det = x30*z30 + x20*z20 + x10*z10 + x00*z00;
    
    if (fabsf(float(det)) > kEps) 
    {
        mat44 invm;

        // todo: should we switch to float only?
        double rcp = 1.0 / det;

        // Multiply all 3x3 cofactors by reciprocal & transpose
        invm.data[0][0] = float(z00*rcp);
        invm.data[0][1] = float(z10*rcp);
        invm.data[1][0] = float(z01*rcp);
        invm.data[0][2] = float(z20*rcp);
        invm.data[2][0] = float(z02*rcp);
        invm.data[0][3] = float(z30*rcp);
        invm.data[3][0] = float(z03*rcp);
        invm.data[1][1] = float(z11*rcp);
        invm.data[1][2] = float(z21*rcp);
        invm.data[2][1] = float(z12*rcp);
        invm.data[1][3] = float(z31*rcp);
        invm.data[3][1] = float(z13*rcp);
        invm.data[2][2] = float(z22*rcp);
        invm.data[2][3] = float(z32*rcp);
        invm.data[3][2] = float(z23*rcp);
        invm.data[3][3] = float(z33*rcp);

        return invm;
    }
    else 
    {
        return mat44();
    }
}

inline CUDA_CALLABLE void adj_inverse(const mat44& m, mat44& adj_m, const mat44& adj_ret)
{
    // todo: how to cache this from the forward pass?
    mat44 invt = transpose(inverse(m));

    // see https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf 2.2.3
    adj_m -= mul(mul(invt, adj_ret), invt);
}

inline CUDA_CALLABLE vec3 transform_point(const mat44& m, const vec3& v)
{
    vec4 out = mul(m, vec4(v.x, v.y, v.z, 1.0));
    return vec3(out.x, out.y, out.z);
}

inline CUDA_CALLABLE vec3 transform_vector(const mat44& m, const vec3& v)
{
    vec4 out = mul(m, vec4(v.x, v.y, v.z, 0.0));
    return vec3(out.x, out.y, out.z);
}

inline CUDA_CALLABLE mat44 outer(const vec4& a, const vec4& b)
{
    return mat44(a*b.x, a*b.y, a*b.z, a*b.w);    
}


inline CUDA_CALLABLE void adj_transform_point(const mat44& m, const vec3& v, mat44& adj_m, vec3& adj_v, const vec3& adj_ret)
{
    printf("todo\n");
}
inline CUDA_CALLABLE void adj_transform_vector(const mat44& m, const vec3& v, mat44& adj_m, vec3& adj_v, const vec3& adj_ret)
{
    printf("todo\n");
}


inline void CUDA_CALLABLE adj_index(const mat44& m, int row, int col, mat44& adj_m, int& adj_row, int& adj_col, float adj_ret)
{
#if FP_CHECK
    if (row < 0 || row > 3)
    {
        printf("mat44 row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
    if (col < 0 || col > 3)
    {
        printf("mat44 col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif
    adj_m.data[row][col] += adj_ret;
}

inline CUDA_CALLABLE void adj_add(const mat44& a, const mat44& b, mat44& adj_a, mat44& adj_b, const mat44& adj_ret)
{
    for (int i=0; i < 4; ++i)
    {
        for (int j=0; j < 4; ++j)
        {
            adj_a.data[i][j] += adj_ret.data[i][j];
            adj_b.data[i][j] += adj_ret.data[i][j];
        }
    }
}

inline CUDA_CALLABLE void adj_sub(const mat44& a, const mat44& b, mat44& adj_a, mat44& adj_b, const mat44& adj_ret)
{
    for (int i=0; i < 4; ++i)
    {
        for (int j=0; j < 4; ++j)
        {
            adj_a.data[i][j] += adj_ret.data[i][j];
            adj_b.data[i][j] -= adj_ret.data[i][j];
        }
    }
}


inline CUDA_CALLABLE void adj_mul(const mat44& a, float b, mat44& adj_a, float& adj_b, const mat44& adj_ret)
{
    for (int i=0; i < 4; ++i)
    {
        for (int j=0; j < 4; ++j)
        {
            adj_a.data[i][j] += b*adj_ret.data[i][j];
            adj_b += a.data[i][j]*adj_ret.data[i][j];
        }
    }
}

inline CUDA_CALLABLE void adj_mul(const mat44& a, const vec4& b, mat44& adj_a, vec4& adj_b, const vec4& adj_ret)
{
    adj_a += outer(adj_ret, b);
    adj_b += mul(transpose(a), adj_ret);
}

inline CUDA_CALLABLE void adj_mul(const mat44& a, const mat44& b, mat44& adj_a, mat44& adj_b, const mat44& adj_ret)
{
    adj_a += mul(adj_ret, transpose(b));
    adj_b += mul(transpose(a), adj_ret);
}

inline CUDA_CALLABLE void adj_transpose(const mat44& a, mat44& adj_a, const mat44& adj_ret)
{
    adj_a += transpose(adj_ret);
}

} // namespace wp
