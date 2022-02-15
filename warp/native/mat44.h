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
    for(int i=0; i < 4; ++i)
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

inline CUDA_CALLABLE void atomic_add(mat44 * addr, mat44 value) {
    for (int i=0; i < 4; ++i)
    {
        for (int j=0; j < 4; ++j)
        atomic_add(&addr->data[i][j], value.data[i][j]);
    }
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
    return m.data[row][col];
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
