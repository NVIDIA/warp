/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

namespace wp
{


CUDA_CALLABLE inline int dense_index(int stride, int i, int j)
{
    return i*stride + j;
}

template <bool transpose>
CUDA_CALLABLE inline int dense_index(int rows, int cols, int i, int j)
{
    if (transpose)
        return j*rows + i;
    else
        return i*cols + j;
}



template <bool t1, bool t2, bool add>
CUDA_CALLABLE inline void dense_gemm_impl(int m, int n, int p, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
    for (int i=0; i < m; i++)
    {
        for (int j=0; j < n; ++j)
        {
            float sum = 0.0f;

            for (int k=0; k < p; ++k)
            {
                sum += A[dense_index<t1>(m, p, i, k)]*B[dense_index<t2>(p, n, k, j)];
            }
            
            if (add)
                C[i*n + j] += sum;
            else
                C[i*n + j] = sum;
        }
    }
}


template <bool add=false>
CUDA_CALLABLE inline void dense_gemm(int m, int n, int p, int t1, int t2, const array_t<float>& A, const array_t<float>& B, array_t<float>& C)
{
    if (t1 == 0 && t2 == 0)
        dense_gemm_impl<false, false, add>(m, n, p, A.data, B.data, C.data);
    else if (t1 == 1 && t2 == 0)
        dense_gemm_impl<true, false, add>(m, n, p, A.data, B.data, C.data);
    else if (t1 == 0 && t2 == 1)
        dense_gemm_impl<false, true, add>(m, n, p, A.data, B.data, C.data);
    else if (t1 == 1 && t2 == 1)
        dense_gemm_impl<true, true, add>(m, n, p, A.data, B.data, C.data);
}




void  CUDA_CALLABLE inline dense_chol(int n, const array_t<float>& A, float regularization, array_t<float>& L)
{
    for (int j=0; j < n; ++j)
    {
        float s = A.data[dense_index(n, j, j)] + regularization;

        for (int k=0; k < j; ++k)
        {
            float r = L.data[dense_index(n, j, k)];
            s -= r*r;
        }

        s = sqrt(s);
        const float invS = 1.0f/s;

        L.data[dense_index(n, j, j)] = s;

        for (int i=j+1; i < n; ++i)
        {
            s = A.data[dense_index(n, i, j)];
            
            for (int k=0; k < j; ++k)
            {
                s -= L.data[dense_index(n, i, k)]*L.data[dense_index(n, j, k)];
            }

            L.data[dense_index(n, i, j)] = s*invS;
        }
    }
}




// Solves (L*L^T)x = b given the Cholesky factor L 
CUDA_CALLABLE inline void dense_subs(int n, const array_t<float>& L, const array_t<float>& b, array_t<float>& x)
{
    // forward substitution
    for (int i=0; i < n; ++i)
    {
        float s = b.data[i];

        for (int j=0; j < i; ++j)
        {
            s -= L.data[dense_index(n, i, j)]*x.data[j];
        }

        x.data[i] = s/L.data[dense_index(n, i, i)];
    }

    // backward substitution
    for (int i=n-1; i >= 0; --i)
    {
        float s = x.data[i];

        for (int j=i+1; j < n; ++j)
        {
            s -= L.data[dense_index(n, j, i)]*x.data[j];
        }

        x.data[i] = s/L.data[dense_index(n, i, i)];
    }
}

CUDA_CALLABLE inline void dense_solve(int n, const array_t<float>& A, const array_t<float>& L, const array_t<float>& b, array_t<float>& x)
{
    dense_subs(n, L, b, x);
}


// CUDA_CALLABLE inline void print_matrix(const char* name, int m, int n, const float* data)
// {
//     printf("%s = [", name);

//     for (int i=0; i < m; ++i)
//     {
//         for (int j=0; j < n; ++j)
//         {
//             printf("%f ", data[dense_index(n, i, j)]);
//         }

//         printf(";\n");
//     }

//     printf("]\n");
// }

// adjoint methods
CUDA_CALLABLE inline void adj_dense_gemm(
    int m, int n, int p, int t1, int t2, const array_t<float>& A, const array_t<float>& B, array_t<float>& C,
    int adj_m, int adj_n, int adj_p, int adj_t1, int adj_t2, array_t<float>& adj_A, array_t<float>& adj_B, const array_t<float>& adj_C)
{

    // print_matrix("A", m, p, A);
    // print_matrix("B", p, n, B);
    // printf("t1: %d t2: %d\n", t1, t2);

    if (t1)
    {
        dense_gemm<true>(p, m, n, 0, 1, B, adj_C, adj_A);
        dense_gemm<true>(p, n, m, int(!t1), 0, A, adj_C, adj_B);
    }
    else
    {
        dense_gemm<true>(m, p, n, 0, int(!t2), adj_C, B, adj_A);
        dense_gemm<true>(p, n, m, int(!t1), 0, A, adj_C, adj_B);
    }
}


CUDA_CALLABLE inline void adj_dense_chol(
    int n, const array_t<float>& A, float regularization, array_t<float>& L,
    int adj_n, const array_t<float>& adj_A, float adj_regularization, array_t<float>& adj_L)
{
    // nop, use dense_solve to differentiate through (A^-1)b = x
}

CUDA_CALLABLE inline void adj_dense_subs(
    int n, const array_t<float>& L, const array_t<float>& b, array_t<float>& x,
    int adj_n, const array_t<float>& adj_L, const array_t<float>& adj_b, array_t<float>& adj_x)
{
    // nop, use dense_solve to differentiate through (A^-1)b = x
}


CUDA_CALLABLE inline void adj_dense_solve(int n,
    const array_t<float>& A, const array_t<float>& L, const array_t<float>& b, const array_t<float>& x,
    int adj_n, array_t<float>& adj_A, array_t<float>& adj_L, array_t<float>& adj_b, const array_t<float>& adj_x)
{
    // see https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pwp, section 2.3.1
    dense_subs(n, L, adj_x, adj_b);

    // A* = -adj_b*x^T
    for (int i=0; i < n; ++i)
    {
        for (int j=0; j < n; ++j)
        {
            adj_A.data[dense_index(n, i, j)] += -adj_b.data[i]*x.data[j];
        }
    }
}

} // namespace wp
