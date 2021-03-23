#pragma once

//----------------------------------------------------------
// mat22

struct mat22
{
    inline CUDA_CALLABLE mat22(float m00=0.0f, float m01=0.0f, float m10=0.0f, float m11=0.0f) 
    {
        data[0][0] = m00;
        data[1][0] = m10;
        data[0][1] = m01;
        data[1][1] = m11;
    }

    // row major storage assumed to be compatible with PyTorch
    float data[2][2];
};

#ifdef CUDA
inline __device__ void atomic_add(mat22 * addr, mat22 value) {
    // *addr += value;
    atomicAdd(&((addr -> data)[0][0]), value.data[0][0]);
    atomicAdd(&((addr -> data)[0][1]), value.data[0][1]);
    atomicAdd(&((addr -> data)[1][0]), value.data[1][0]);
    atomicAdd(&((addr -> data)[1][1]), value.data[1][1]);
}
#endif

inline CUDA_CALLABLE void adj_mat22(float m00, float m01, float m10, float m11, float& adj_m00, float& adj_m01, float& adj_m10, float& adj_m11, const mat22& adj_ret)
{
    printf("todo\n");
}

inline CUDA_CALLABLE float index(const mat22& m, int row, int col)
{
    return m.data[row][col];
}

inline CUDA_CALLABLE mat22 add(const mat22& a, const mat22& b)
{
    mat22 t;
    for (int i=0; i < 2; ++i)
    {
        for (int j=0; j < 2; ++j)
        {
            t.data[i][j] = a.data[i][j] + b.data[i][j];
        }
    }

    return t;
}


inline CUDA_CALLABLE mat22 mul(const mat22& a, float b)
{
    mat22 t;
    for (int i=0; i < 2; ++i)
    {
        for (int j=0; j < 2; ++j)
        {
            t.data[i][j] = a.data[i][j]*b;
        }
    }

    return t;
}

inline CUDA_CALLABLE mat22 mul(const mat22& a, const mat22& b)
{
    mat22 t;
    for (int i=0; i < 2; ++i)
    {
        for (int j=0; j < 2; ++j)
        {
            for (int k=0; k < 2; ++k)
            {
                t.data[i][j] += a.data[i][k]*b.data[k][j];
            }
        }
    }

    return t;
}

inline CUDA_CALLABLE mat22 transpose(const mat22& a)
{
    mat22 t;
    for (int i=0; i < 2; ++i)
    {
        for (int j=0; j < 2; ++j)
        {
            t.data[i][j] = a.data[j][i];
        }
    }

    return t;
}


inline CUDA_CALLABLE float determinant(const mat22& m)
{
    return m.data[0][0]*m.data[1][1] - m.data[1][0]*m.data[0][1];
}


inline void CUDA_CALLABLE adj_index(const mat22& m, int row, int col, mat22& adj_m, int& adj_row, int& adj_col, float adj_ret)
{
    adj_m.data[row][col] += adj_ret;
}

inline CUDA_CALLABLE void adj_add(const mat22& a, const mat22& b, mat22& adj_a, mat22& adj_b, const mat22& adj_ret)
{
    for (int i=0; i < 2; ++i)
    {
        for (int j=0; j < 2; ++j)
        {
            adj_a.data[i][j] = adj_ret.data[i][j];
            adj_b.data[i][j] = adj_ret.data[i][j];
        }
    }
}

inline CUDA_CALLABLE void adj_mul(const mat22& a, const mat22& b, mat22& adj_a, mat22& adj_b, const mat22& adj_ret)
{
    printf("todo\n");
}

inline CUDA_CALLABLE void adj_transpose(const mat22& a, mat22& adj_a, const mat22& adj_ret)
{
    printf("todo\n");
}

inline CUDA_CALLABLE void adj_determinant(const mat22& m, mat22& adj_m, float adj_ret)
{
    adj_m.data[0][0] += m.data[1][1]*adj_ret;
    adj_m.data[1][1] += m.data[0][0]*adj_ret;
    adj_m.data[0][1] -= m.data[1][0]*adj_ret;
    adj_m.data[1][0] -= m.data[0][1]*adj_ret;
}
