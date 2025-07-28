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

#include "initializer_array.h"

namespace wp
{

//----------------------------------------------------------
// mat
template<typename T>
struct quat_t;

template<unsigned Rows, unsigned Cols, typename Type>
struct mat_t
{
    inline CUDA_CALLABLE mat_t()
        : data()
    {}

    inline CUDA_CALLABLE mat_t(Type s)
    {
        for (unsigned i=0; i < Rows; ++i)
            for (unsigned j=0; j < Cols; ++j)
                data[i][j] = s;
    }
    
    template <typename OtherType>
    inline explicit CUDA_CALLABLE mat_t(const mat_t<Rows, Cols, OtherType>& other)
    {
        for (unsigned i=0; i < Rows; ++i)
            for (unsigned j=0; j < Cols; ++j)
                data[i][j] = other.data[i][j];
    }
    
    inline CUDA_CALLABLE mat_t(vec_t<2,Type> c0, vec_t<2,Type> c1)
    {
        data[0][0] = c0[0];
        data[1][0] = c0[1];

        data[0][1] = c1[0];
        data[1][1] = c1[1];
    }
    
    inline CUDA_CALLABLE mat_t(vec_t<3,Type> c0, vec_t<3,Type> c1, vec_t<3,Type> c2)
    {
        data[0][0] = c0[0];
        data[1][0] = c0[1];
        data[2][0] = c0[2];

        data[0][1] = c1[0];
        data[1][1] = c1[1];
        data[2][1] = c1[2];

        data[0][2] = c2[0];
        data[1][2] = c2[1];
        data[2][2] = c2[2];
    }

    inline CUDA_CALLABLE mat_t(vec_t<4,Type> c0, vec_t<4,Type> c1, vec_t<4,Type> c2, vec_t<4,Type> c3)
    {
        data[0][0] = c0[0];
        data[1][0] = c0[1];
        data[2][0] = c0[2];
        data[3][0] = c0[3];

        data[0][1] = c1[0];
        data[1][1] = c1[1];
        data[2][1] = c1[2];
        data[3][1] = c1[3];

        data[0][2] = c2[0];
        data[1][2] = c2[1];
        data[2][2] = c2[2];
        data[3][2] = c2[3];

        data[0][3] = c3[0];
        data[1][3] = c3[1];
        data[2][3] = c3[2];
        data[3][3] = c3[3];
    }

    inline CUDA_CALLABLE mat_t(Type m00, Type m01, Type m10, Type m11) 
    {
        data[0][0] = m00;
        data[1][0] = m10;
        data[0][1] = m01;
        data[1][1] = m11;
    }
    
    inline CUDA_CALLABLE mat_t(
        Type m00, Type m01, Type m02,
        Type m10, Type m11, Type m12,
        Type m20, Type m21, Type m22)
    {
        data[0][0] = m00;
        data[1][0] = m10;
        data[2][0] = m20;

        data[0][1] = m01;
        data[1][1] = m11;
        data[2][1] = m21;

        data[0][2] = m02;
        data[1][2] = m12;
        data[2][2] = m22;
    }

    inline CUDA_CALLABLE mat_t(
                 Type m00, Type m01, Type m02, Type m03,
                 Type m10, Type m11, Type m12, Type m13,
                 Type m20, Type m21, Type m22, Type m23,
                 Type m30, Type m31, Type m32, Type m33)
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

    // implemented in quat.h
    inline CUDA_CALLABLE mat_t(const vec_t<3,Type>& pos, const quat_t<Type>& rot, const vec_t<3,Type>& scale);


    inline CUDA_CALLABLE mat_t(const initializer_array<Rows * Cols, Type> &l)
    {
        for (unsigned i=0; i < Rows; ++i)
        {
            for (unsigned j=0; j < Cols; ++j)
            {
                data[i][j] = l[i * Cols + j];
            }
        }
    }

    inline CUDA_CALLABLE mat_t(const initializer_array<Cols, vec_t<Rows,Type> > &l)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            for (unsigned i=0; i < Rows; ++i)
            {
                data[i][j] = l[j][i];
            }
        }
    }

    CUDA_CALLABLE vec_t<Cols,Type> get_row(int index) const
    {
        return (vec_t<Cols,Type>&)data[index]; 
    }

    CUDA_CALLABLE void set_row(int index, const vec_t<Cols,Type>& v)
    {
        (vec_t<Cols,Type>&)data[index] = v;
    }

    CUDA_CALLABLE vec_t<Rows,Type> get_col(int index) const
    {
        vec_t<Rows,Type> ret;
        for( unsigned i=0;i < Rows; ++i )
        {
            ret[i] = data[i][index];
        }
        return ret;
    }

    CUDA_CALLABLE void set_col(int index, const vec_t<Rows,Type>& v)
    {
        for( unsigned i=0;i < Rows; ++i )
        {
            data[i][index] = v[i];
        }
    }

    // row major storage assumed to be compatible with PyTorch
    Type data[Rows < 1 ? 1 : Rows][Cols < 1 ? 1 : Cols];
};

template<typename Type>
inline CUDA_CALLABLE mat_t<2, 2, Type> matrix_from_cols(vec_t<2, Type> c0, vec_t<2, Type> c1)
{
    mat_t<2, 2, Type> m;

    m.data[0][0] = c0[0];
    m.data[1][0] = c0[1];

    m.data[0][1] = c1[0];
    m.data[1][1] = c1[1];

    return m;
}

template<typename Type>
inline CUDA_CALLABLE mat_t<3, 3, Type> matrix_from_cols(vec_t<3, Type> c0, vec_t<3, Type> c1, vec_t<3, Type> c2)
{
    mat_t<3, 3, Type> m;

    m.data[0][0] = c0[0];
    m.data[1][0] = c0[1];
    m.data[2][0] = c0[2];

    m.data[0][1] = c1[0];
    m.data[1][1] = c1[1];
    m.data[2][1] = c1[2];

    m.data[0][2] = c2[0];
    m.data[1][2] = c2[1];
    m.data[2][2] = c2[2];

    return m;
}

template<typename Type>
inline CUDA_CALLABLE mat_t<4, 4, Type> matrix_from_cols(vec_t<4, Type> c0, vec_t<4, Type> c1, vec_t<4, Type> c2, vec_t<4, Type> c3)
{
    mat_t<4, 4, Type> m;

    m.data[0][0] = c0[0];
    m.data[1][0] = c0[1];
    m.data[2][0] = c0[2];
    m.data[3][0] = c0[3];

    m.data[0][1] = c1[0];
    m.data[1][1] = c1[1];
    m.data[2][1] = c1[2];
    m.data[3][1] = c1[3];

    m.data[0][2] = c2[0];
    m.data[1][2] = c2[1];
    m.data[2][2] = c2[2];
    m.data[3][2] = c2[3];

    m.data[0][3] = c3[0];
    m.data[1][3] = c3[1];
    m.data[2][3] = c3[2];
    m.data[3][3] = c3[3];

    return m;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows, Cols, Type> matrix_from_cols(const initializer_array<Cols, vec_t<Rows, Type> >& l)
{
    mat_t<Rows, Cols, Type> m;
    for (unsigned j=0; j < Cols; ++j)
    {
        for (unsigned i=0; i < Rows; ++i)
        {
            m.data[i][j] = l[j][i];
        }
    }

    return m;
}

template<typename Type>
inline CUDA_CALLABLE mat_t<2, 2, Type> matrix_from_rows(vec_t<2, Type> r0, vec_t<2, Type> r1)
{
    mat_t<2, 2, Type> m;

    m.data[0][0] = r0[0];
    m.data[0][1] = r0[1];

    m.data[1][0] = r1[0];
    m.data[1][1] = r1[1];

    return m;
}

template<typename Type>
inline CUDA_CALLABLE mat_t<3, 3, Type> matrix_from_rows(vec_t<3, Type> r0, vec_t<3, Type> r1, vec_t<3, Type> r2)
{
    mat_t<3, 3, Type> m;

    m.data[0][0] = r0[0];
    m.data[0][1] = r0[1];
    m.data[0][2] = r0[2];

    m.data[1][0] = r1[0];
    m.data[1][1] = r1[1];
    m.data[1][2] = r1[2];

    m.data[2][0] = r2[0];
    m.data[2][1] = r2[1];
    m.data[2][2] = r2[2];

    return m;
}

template<typename Type>
inline CUDA_CALLABLE mat_t<4, 4, Type> matrix_from_rows(vec_t<4, Type> r0, vec_t<4, Type> r1, vec_t<4, Type> r2, vec_t<4, Type> r3)
{
    mat_t<4, 4, Type> m;

    m.data[0][0] = r0[0];
    m.data[0][1] = r0[1];
    m.data[0][2] = r0[2];
    m.data[0][3] = r0[3];

    m.data[1][0] = r1[0];
    m.data[1][1] = r1[1];
    m.data[1][2] = r1[2];
    m.data[1][3] = r1[3];

    m.data[2][0] = r2[0];
    m.data[2][1] = r2[1];
    m.data[2][2] = r2[2];
    m.data[2][3] = r2[3];

    m.data[3][0] = r3[0];
    m.data[3][1] = r3[1];
    m.data[3][2] = r3[2];
    m.data[3][3] = r3[3];

    return m;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows, Cols, Type> matrix_from_rows(const initializer_array<Rows, vec_t<Cols, Type> >& l)
{
    mat_t<Rows, Cols, Type> m;
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            m.data[i][j] = l[i][j];
        }
    }

    return m;
}

template<unsigned Rows, typename Type>
inline CUDA_CALLABLE mat_t<Rows, Rows, Type> identity()
{
    mat_t<Rows, Rows, Type> m;
    for( unsigned i=0; i < Rows; ++i )
    {
        m.data[i][i] = Type(1);
    }
    return m;
}

template<unsigned Rows, typename Type>
inline CUDA_CALLABLE void adj_identity(const mat_t<Rows, Rows, Type>& adj_ret)
{
    // nop
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE bool operator==(const mat_t<Rows,Cols,Type>& a, const mat_t<Rows,Cols,Type>& b)
{
    for (unsigned i=0; i < Rows; ++i)
        for (unsigned j=0; j < Cols; ++j)
            if (a.data[i][j] != b.data[i][j])
                return false;

    return true;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> operator - (const mat_t<Rows,Cols,Type>& x)
{
    mat_t<Rows,Cols,Type> ret;
    for (unsigned i=0; i < Rows; ++i)
        for (unsigned j=0; j < Cols; ++j)
            ret.data[i][j] = -x.data[i][j];

    return ret;
}

template<unsigned Rows, unsigned Cols, typename Type>
CUDA_CALLABLE inline mat_t<Rows,Cols,Type> pos(const mat_t<Rows,Cols,Type>& x)
{
    return x;
}

template<unsigned Rows, unsigned Cols, typename Type>
CUDA_CALLABLE inline void adj_pos(const mat_t<Rows,Cols,Type>& x, mat_t<Rows,Cols,Type>& adj_x, const mat_t<Rows,Cols,Type>& adj_ret)
{
    adj_x += adj_ret;
}

template<unsigned Rows, unsigned Cols, typename Type>
CUDA_CALLABLE inline mat_t<Rows,Cols,Type> neg(const mat_t<Rows,Cols,Type>& x)
{
    return -x;
}

template<unsigned Rows, unsigned Cols, typename Type>
CUDA_CALLABLE inline void adj_neg(const mat_t<Rows,Cols,Type>& x, mat_t<Rows,Cols,Type>& adj_x, const mat_t<Rows,Cols,Type>& adj_ret)
{
    adj_x -= adj_ret;
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> atomic_add(mat_t<Rows,Cols,Type> * addr, mat_t<Rows,Cols,Type> value) 
{
    mat_t<Rows,Cols,Type> m;
    
    for (unsigned i=0; i < Rows; ++i)
        for (unsigned j=0; j < Cols; ++j)
            m.data[i][j] = atomic_add(&addr->data[i][j], value.data[i][j]);

    return m;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> atomic_min(mat_t<Rows,Cols,Type> * addr, mat_t<Rows,Cols,Type> value) 
{
    mat_t<Rows,Cols,Type> m;
    
    for (unsigned i=0; i < Rows; ++i)
        for (unsigned j=0; j < Cols; ++j)
            m.data[i][j] = atomic_min(&addr->data[i][j], value.data[i][j]);

    return m;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> atomic_max(mat_t<Rows,Cols,Type> * addr, mat_t<Rows,Cols,Type> value) 
{
    mat_t<Rows,Cols,Type> m;
    
    for (unsigned i=0; i < Rows; ++i)
        for (unsigned j=0; j < Cols; ++j)
            m.data[i][j] = atomic_max(&addr->data[i][j], value.data[i][j]);

    return m;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_atomic_minmax(
    mat_t<Rows,Cols,Type> *addr,
    mat_t<Rows,Cols,Type> *adj_addr,
    const mat_t<Rows,Cols,Type> &value,
    mat_t<Rows,Cols,Type> &adj_value)
{
    for (unsigned i=0; i < Rows; ++i)
        for (unsigned j=0; j < Cols; ++j)
            adj_atomic_minmax(&addr->data[i][j], &adj_addr->data[i][j], value.data[i][j], adj_value.data[i][j]);
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE vec_t<Cols,Type> extract(const mat_t<Rows,Cols,Type>& m, int row)
{
    vec_t<Cols,Type> ret;

#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }

    for(unsigned i=0; i < Cols; ++i)
    {
        ret.c[i] = m.data[row][i];
    }
    return ret;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE Type extract(const mat_t<Rows,Cols,Type>& m, int row, int col)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }
    if (col < 0)
    {
        col += Cols;
    }

    return m.data[row][col];
}

template<unsigned RowSliceLength, unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<RowSliceLength, ColSliceLength, Type> extract(const mat_t<Rows,Cols,Type>& m, slice_t row_slice)
{
    static_assert(
        RowSliceLength == 0 ? ColSliceLength == 0 : ColSliceLength == Cols,
        "Expected RowSliceLength == 0 ? ColSliceLength == 0 : ColSliceLength == Cols"
    );

    mat_t<RowSliceLength, ColSliceLength, Type> ret;

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    bool is_row_reversed = row_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        for (int j = 0; j < Cols; ++j)
        {
            ret.data[ii][j] = m.data[i][j];
        }

        ++ii;
    }

    assert(ii == RowSliceLength);
    return ret;
}

template<unsigned RowSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE vec_t<RowSliceLength, Type> extract(const mat_t<Rows,Cols,Type>& m, slice_t row_slice, int col)
{
#ifndef NDEBUG
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    vec_t<RowSliceLength, Type> ret;

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    if (col < 0)
    {
        col += Cols;
    }

    bool is_row_reversed = row_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        ret.c[ii] = m.data[i][col];
        ++ii;
    }

    assert(ii == RowSliceLength);
    return ret;
}

template<unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE vec_t<ColSliceLength, Type> extract(const mat_t<Rows,Cols,Type>& m, int row, slice_t col_slice)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    vec_t<ColSliceLength, Type> ret;

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);
    assert(slice_get_length(col_slice) == ColSliceLength);

    if (row < 0)
    {
        row += Rows;
    }

    bool is_col_reversed = col_slice.step < 0;

    int ii = 0;
    for (
        int i = col_slice.start;
        is_col_reversed ? (i > col_slice.stop) : (i < col_slice.stop);
        i += col_slice.step
    )
    {
        ret.c[ii] = m.data[row][i];
        ++ii;
    }

    assert(ii == ColSliceLength);
    return ret;
}

template<unsigned RowSliceLength, unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<RowSliceLength, ColSliceLength, Type> extract(const mat_t<Rows,Cols,Type>& m, slice_t row_slice, slice_t col_slice)
{
    mat_t<RowSliceLength, ColSliceLength, Type> ret;

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);
    assert(slice_get_length(col_slice) == ColSliceLength);

    bool is_row_reversed = row_slice.step < 0;
    bool is_col_reversed = col_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        int jj = 0;
        for (
            int j = col_slice.start;
            is_col_reversed ? (j > col_slice.stop) : (j < col_slice.stop);
            j += col_slice.step
        )
        {
            ret.data[ii][jj] = m.data[i][j];
            ++jj;
        }

        assert(jj == ColSliceLength);
        ++ii;
    }

    assert(ii == RowSliceLength);
    return ret;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE vec_t<Cols, Type>* index(mat_t<Rows,Cols,Type>& m, int row)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }

    return reinterpret_cast<vec_t<Cols, Type>*>(&m.data[row]);
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE Type* index(mat_t<Rows,Cols,Type>& m, int row, int col)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }
    if (col < 0)
    {
        col += Cols;
    }

    return &m.data[row][col];
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_index(const mat_t<Rows,Cols,Type>& m, int row,
                                       const mat_t<Rows,Cols,Type>& adj_m, int adj_row, const vec_t<Cols, Type>& adj_value)
{
    // nop
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_index(const mat_t<Rows,Cols,Type>& m, int row, int col,
                                       const mat_t<Rows,Cols,Type>& adj_m, int adj_row, int adj_col, Type adj_value)
{
    // nop
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void add_inplace(mat_t<Rows,Cols,Type>& m, int row, int col, Type value)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }
    if (col < 0)
    {
        col += Cols;
    }

    m.data[row][col] += value;
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void add_inplace(mat_t<Rows,Cols,Type>& m, int row, vec_t<Cols,Type>& value)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }

    for(unsigned i=0; i < Cols; ++i)
    {
        m.data[row][i] += value[i];
    }
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void add_inplace(mat_t<Rows,Cols,Type>& m, slice_t row_slice, Type value)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);

    bool is_row_reversed = row_slice.step < 0;

    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        for (int j = 0; j < Cols; ++j)
        {
            m.data[i][j] += value;
        }
    }
}


template<unsigned RowSliceLength, unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void add_inplace(mat_t<Rows,Cols,Type>& m, slice_t row_slice, mat_t<RowSliceLength, ColSliceLength, Type>& value)
{
    static_assert(
        RowSliceLength == 0 ? ColSliceLength == 0 : ColSliceLength == Cols,
        "Expected RowSliceLength == 0 ? ColSliceLength == 0 : ColSliceLength == Cols"
    );

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    bool is_row_reversed = row_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        for (int j = 0; j < Cols; ++j)
        {
            m.data[i][j] += value.data[ii][j];
        }

        ++ii;
    }

    assert(ii == RowSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void add_inplace(mat_t<Rows,Cols,Type>& m, slice_t row_slice, int col, Type value)
{
#ifndef NDEBUG
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);

    if (col < 0)
    {
        col += Cols;
    }

    bool is_row_reversed = row_slice.step < 0;

    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        m.data[i][col] += value;
    }
}


template<unsigned RowSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void add_inplace(mat_t<Rows,Cols,Type>& m, slice_t row_slice, int col, vec_t<RowSliceLength, Type>& value)
{
#ifndef NDEBUG
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    if (col < 0)
    {
        col += Cols;
    }

    bool is_row_reversed = row_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        m.data[i][col] += value.c[ii];
        ++ii;
    }

    assert(ii == RowSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void add_inplace(mat_t<Rows,Cols,Type>& m, int row, slice_t col_slice, Type value)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);

    if (row < 0)
    {
        row += Rows;
    }

    bool is_col_reversed = col_slice.step < 0;

    for (
        int i = col_slice.start;
        is_col_reversed ? (i > col_slice.stop) : (i < col_slice.stop);
        i += col_slice.step
    )
    {
        m.data[row][i] += value;
    }
}


template<unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void add_inplace(mat_t<Rows,Cols,Type>& m, int row, slice_t col_slice, vec_t<ColSliceLength, Type>& value)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);
    assert(slice_get_length(col_slice) == ColSliceLength);

    if (row < 0)
    {
        row += Rows;
    }

    bool is_col_reversed = col_slice.step < 0;

    int ii = 0;
    for (
        int i = col_slice.start;
        is_col_reversed ? (i > col_slice.stop) : (i < col_slice.stop);
        i += col_slice.step
    )
    {
        m.data[row][i] += value.c[ii];
        ++ii;
    }

    assert(ii == ColSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void add_inplace(mat_t<Rows,Cols,Type>& m, slice_t row_slice, slice_t col_slice, Type value)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);

    bool is_row_reversed = row_slice.step < 0;
    bool is_col_reversed = col_slice.step < 0;

    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        for (
            int j = col_slice.start;
            is_col_reversed ? (j > col_slice.stop) : (j < col_slice.stop);
            j += col_slice.step
        )
        {
            m.data[i][j] += value;
        }
    }
}


template<unsigned RowSliceLength, unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void add_inplace(mat_t<Rows,Cols,Type>& m, slice_t row_slice, slice_t col_slice, mat_t<RowSliceLength, ColSliceLength, Type>& value)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);
    assert(slice_get_length(col_slice) == ColSliceLength);

    bool is_row_reversed = row_slice.step < 0;
    bool is_col_reversed = col_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        int jj = 0;
        for (
            int j = col_slice.start;
            is_col_reversed ? (j > col_slice.stop) : (j < col_slice.stop);
            j += col_slice.step
        )
        {
            m.data[i][j] += value.data[ii][jj];
            ++jj;
        }

        assert(jj == ColSliceLength);
        ++ii;
    }

    assert(ii == RowSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_add_inplace(mat_t<Rows,Cols,Type>& m, int row, int col, Type value,
                                        mat_t<Rows,Cols,Type>& adj_m, int adj_row, int adj_col, Type& adj_value)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }
    if (col < 0)
    {
        col += Cols;
    }

    adj_value += adj_m.data[row][col];
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_add_inplace(mat_t<Rows,Cols,Type>& m, int row, vec_t<Cols,Type>& value,
                                        mat_t<Rows,Cols,Type>& adj_m, int adj_row, vec_t<Cols,Type>& adj_value)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }

    for(unsigned i=0; i < Cols; ++i)
    {
        adj_value[i] += adj_m.data[row][i];
    }
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_add_inplace(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, Type value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, Type& adj_value
)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);

    bool is_row_reversed = row_slice.step < 0;

    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        for (int j = 0; j < Cols; ++j)
        {
            adj_value += adj_m.data[i][j];
        }
    }
}


template<unsigned RowSliceLength, unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_add_inplace(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, mat_t<RowSliceLength, ColSliceLength, Type>& value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, mat_t<RowSliceLength, ColSliceLength, Type>& adj_value
)
{
    static_assert(
        RowSliceLength == 0 ? ColSliceLength == 0 : ColSliceLength == Cols,
        "Expected RowSliceLength == 0 ? ColSliceLength == 0 : ColSliceLength == Cols"
    );

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    bool is_row_reversed = row_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        for (int j = 0; j < Cols; ++j)
        {
            adj_value.data[ii][j] += adj_m.data[i][j];
        }

        ++ii;
    }

    assert(ii == RowSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_add_inplace(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, int col, Type value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, int& adj_col, Type& adj_value
)
{
#ifndef NDEBUG
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);

    if (col < 0)
    {
        col += Cols;
    }

    bool is_row_reversed = row_slice.step < 0;

    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        adj_value += adj_m.data[i][col];
    }
}


template<unsigned RowSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_add_inplace(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, int col, vec_t<RowSliceLength, Type>& value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, int& adj_col, vec_t<RowSliceLength, Type>& adj_value
)
{
#ifndef NDEBUG
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    if (col < 0)
    {
        col += Cols;
    }

    bool is_row_reversed = row_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        adj_value.c[ii] += adj_m.data[i][col];
        ++ii;
    }

    assert(ii == RowSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_add_inplace(
    mat_t<Rows,Cols,Type>& m, int row, slice_t col_slice, Type value,
    mat_t<Rows,Cols,Type>& adj_m, int& adj_row, slice_t& adj_col_slice, Type& adj_value
)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);

    if (row < 0)
    {
        row += Rows;
    }

    bool is_col_reversed = col_slice.step < 0;

    for (
        int i = col_slice.start;
        is_col_reversed ? (i > col_slice.stop) : (i < col_slice.stop);
        i += col_slice.step
    )
    {
        adj_value += adj_m.data[row][i];
    }
}


template<unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_add_inplace(
    mat_t<Rows,Cols,Type>& m, int row, slice_t col_slice, vec_t<ColSliceLength, Type>& value,
    mat_t<Rows,Cols,Type>& adj_m, int& adj_row, slice_t& adj_col_slice, vec_t<ColSliceLength, Type>& adj_value
)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);
    assert(slice_get_length(col_slice) == ColSliceLength);

    if (row < 0)
    {
        row += Rows;
    }

    bool is_col_reversed = col_slice.step < 0;

    int ii = 0;
    for (
        int i = col_slice.start;
        is_col_reversed ? (i > col_slice.stop) : (i < col_slice.stop);
        i += col_slice.step
    )
    {
        adj_value.c[ii] += adj_m.data[row][i];
        ++ii;
    }

    assert(ii == ColSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_add_inplace(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, slice_t col_slice, Type value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, slice_t& adj_col_slice, Type& adj_value
)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);

    bool is_row_reversed = row_slice.step < 0;
    bool is_col_reversed = col_slice.step < 0;

    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        for (
            int j = col_slice.start;
            is_col_reversed ? (j > col_slice.stop) : (j < col_slice.stop);
            j += col_slice.step
        )
        {
            adj_value += adj_m.data[i][j];
        }
    }
}


template<unsigned RowSliceLength, unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_add_inplace(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, slice_t col_slice, mat_t<RowSliceLength, ColSliceLength, Type>& value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, slice_t& adj_col_slice, mat_t<RowSliceLength, ColSliceLength, Type>& adj_value
)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);
    assert(slice_get_length(col_slice) == ColSliceLength);

    bool is_row_reversed = row_slice.step < 0;
    bool is_col_reversed = col_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        int jj = 0;
        for (
            int j = col_slice.start;
            is_col_reversed ? (j > col_slice.stop) : (j < col_slice.stop);
            j += col_slice.step
        )
        {
            adj_value.data[ii][jj] += adj_m.data[i][j];
            ++jj;
        }

        assert(jj == ColSliceLength);
        ++ii;
    }

    assert(ii == RowSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void sub_inplace(mat_t<Rows,Cols,Type>& m, int row, int col, Type value)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }
    if (col < 0)
    {
        col += Cols;
    }

    m.data[row][col] -= value;
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void sub_inplace(mat_t<Rows,Cols,Type>& m, int row, vec_t<Cols,Type>& value)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }

    for(unsigned i=0; i < Cols; ++i)
    {
        m.data[row][i] -= value[i];
    }
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void sub_inplace(mat_t<Rows,Cols,Type>& m, slice_t row_slice, Type value)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);

    bool is_row_reversed = row_slice.step < 0;

    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        for (int j = 0; j < Cols; ++j)
        {
            m.data[i][j] -= value;
        }
    }
}


template<unsigned RowSliceLength, unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void sub_inplace(mat_t<Rows,Cols,Type>& m, slice_t row_slice, mat_t<RowSliceLength, ColSliceLength, Type>& value)
{
    static_assert(
        RowSliceLength == 0 ? ColSliceLength == 0 : ColSliceLength == Cols,
        "Expected RowSliceLength == 0 ? ColSliceLength == 0 : ColSliceLength == Cols"
    );

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    bool is_row_reversed = row_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        for (int j = 0; j < Cols; ++j)
        {
            m.data[i][j] -= value.data[ii][j];
        }

        ++ii;
    }

    assert(ii == RowSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void sub_inplace(mat_t<Rows,Cols,Type>& m, slice_t row_slice, int col, Type value)
{
#ifndef NDEBUG
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);

    if (col < 0)
    {
        col += Cols;
    }

    bool is_row_reversed = row_slice.step < 0;

    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        m.data[i][col] -= value;
    }
}


template<unsigned RowSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void sub_inplace(mat_t<Rows,Cols,Type>& m, slice_t row_slice, int col, vec_t<RowSliceLength, Type>& value)
{
#ifndef NDEBUG
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    if (col < 0)
    {
        col += Cols;
    }

    bool is_row_reversed = row_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        m.data[i][col] -= value.c[ii];
        ++ii;
    }

    assert(ii == RowSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void sub_inplace(mat_t<Rows,Cols,Type>& m, int row, slice_t col_slice, Type value)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);

    if (row < 0)
    {
        row += Rows;
    }

    bool is_col_reversed = col_slice.step < 0;

    for (
        int i = col_slice.start;
        is_col_reversed ? (i > col_slice.stop) : (i < col_slice.stop);
        i += col_slice.step
    )
    {
        m.data[row][i] -= value;
    }
}


template<unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void sub_inplace(mat_t<Rows,Cols,Type>& m, int row, slice_t col_slice, vec_t<ColSliceLength, Type>& value)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);
    assert(slice_get_length(col_slice) == ColSliceLength);

    if (row < 0)
    {
        row += Rows;
    }

    bool is_col_reversed = col_slice.step < 0;

    int ii = 0;
    for (
        int i = col_slice.start;
        is_col_reversed ? (i > col_slice.stop) : (i < col_slice.stop);
        i += col_slice.step
    )
    {
        m.data[row][i] -= value.c[ii];
        ++ii;
    }

    assert(ii == ColSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void sub_inplace(mat_t<Rows,Cols,Type>& m, slice_t row_slice, slice_t col_slice, Type value)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);

    bool is_row_reversed = row_slice.step < 0;
    bool is_col_reversed = col_slice.step < 0;

    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        for (
            int j = col_slice.start;
            is_col_reversed ? (j > col_slice.stop) : (j < col_slice.stop);
            j += col_slice.step
        )
        {
            m.data[i][j] -= value;
        }
    }
}


template<unsigned RowSliceLength, unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void sub_inplace(mat_t<Rows,Cols,Type>& m, slice_t row_slice, slice_t col_slice, mat_t<RowSliceLength, ColSliceLength, Type>& value)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);
    assert(slice_get_length(col_slice) == ColSliceLength);

    bool is_row_reversed = row_slice.step < 0;
    bool is_col_reversed = col_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        int jj = 0;
        for (
            int j = col_slice.start;
            is_col_reversed ? (j > col_slice.stop) : (j < col_slice.stop);
            j += col_slice.step
        )
        {
            m.data[i][j] -= value.data[ii][jj];
            ++jj;
        }

        assert(jj == ColSliceLength);
        ++ii;
    }

    assert(ii == RowSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_sub_inplace(mat_t<Rows,Cols,Type>& m, int row, int col, Type value,
                                        mat_t<Rows,Cols,Type>& adj_m, int adj_row, int adj_col, Type& adj_value)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }
    if (col < 0)
    {
        col += Cols;
    }

    adj_value -= adj_m.data[row][col];
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_sub_inplace(mat_t<Rows,Cols,Type>& m, int row, vec_t<Cols,Type>& value,
                                        mat_t<Rows,Cols,Type>& adj_m, int adj_row, vec_t<Cols,Type>& adj_value)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }

    for(unsigned i=0; i < Cols; ++i)
    {
        adj_value[i] -= adj_m.data[row][i];
    }
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_sub_inplace(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, Type value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, Type& adj_value
)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);

    bool is_row_reversed = row_slice.step < 0;

    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        for (int j = 0; j < Cols; ++j)
        {
            adj_value -= adj_m.data[i][j];
        }
    }
}


template<unsigned RowSliceLength, unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_sub_inplace(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, mat_t<RowSliceLength, ColSliceLength, Type>& value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, mat_t<RowSliceLength, ColSliceLength, Type>& adj_value
)
{
    static_assert(
        RowSliceLength == 0 ? ColSliceLength == 0 : ColSliceLength == Cols,
        "Expected RowSliceLength == 0 ? ColSliceLength == 0 : ColSliceLength == Cols"
    );

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    bool is_row_reversed = row_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        for (int j = 0; j < Cols; ++j)
        {
            adj_value.data[ii][j] -= adj_m.data[i][j];
        }

        ++ii;
    }

    assert(ii == RowSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_sub_inplace(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, int col, Type value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, int& adj_col, Type& adj_value
)
{
#ifndef NDEBUG
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);

    if (col < 0)
    {
        col += Cols;
    }

    bool is_row_reversed = row_slice.step < 0;

    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        adj_value -= adj_m.data[i][col];
    }
}


template<unsigned RowSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_sub_inplace(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, int col, vec_t<RowSliceLength, Type>& value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, int& adj_col, vec_t<RowSliceLength, Type>& adj_value
)
{
#ifndef NDEBUG
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    if (col < 0)
    {
        col += Cols;
    }

    bool is_row_reversed = row_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        adj_value.c[ii] -= adj_m.data[i][col];
        ++ii;
    }

    assert(ii == RowSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_sub_inplace(
    mat_t<Rows,Cols,Type>& m, int row, slice_t col_slice, Type value,
    mat_t<Rows,Cols,Type>& adj_m, int& adj_row, slice_t& adj_col_slice, Type& adj_value
)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);

    if (row < 0)
    {
        row += Rows;
    }

    bool is_col_reversed = col_slice.step < 0;

    for (
        int i = col_slice.start;
        is_col_reversed ? (i > col_slice.stop) : (i < col_slice.stop);
        i += col_slice.step
    )
    {
        adj_value -= adj_m.data[row][i];
    }
}


template<unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_sub_inplace(
    mat_t<Rows,Cols,Type>& m, int row, slice_t col_slice, vec_t<ColSliceLength, Type>& value,
    mat_t<Rows,Cols,Type>& adj_m, int& adj_row, slice_t& adj_col_slice, vec_t<ColSliceLength, Type>& adj_value
)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);
    assert(slice_get_length(col_slice) == ColSliceLength);

    if (row < 0)
    {
        row += Rows;
    }

    bool is_col_reversed = col_slice.step < 0;

    int ii = 0;
    for (
        int i = col_slice.start;
        is_col_reversed ? (i > col_slice.stop) : (i < col_slice.stop);
        i += col_slice.step
    )
    {
        adj_value.c[ii] -= adj_m.data[row][i];
        ++ii;
    }

    assert(ii == ColSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_sub_inplace(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, slice_t col_slice, Type value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, slice_t& adj_col_slice, Type& adj_value
)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);

    bool is_row_reversed = row_slice.step < 0;
    bool is_col_reversed = col_slice.step < 0;

    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        for (
            int j = col_slice.start;
            is_col_reversed ? (j > col_slice.stop) : (j < col_slice.stop);
            j += col_slice.step
        )
        {
            adj_value -= adj_m.data[i][j];
        }
    }
}


template<unsigned RowSliceLength, unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_sub_inplace(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, slice_t col_slice, mat_t<RowSliceLength, ColSliceLength, Type>& value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, slice_t& adj_col_slice, mat_t<RowSliceLength, ColSliceLength, Type>& adj_value
)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);
    assert(slice_get_length(col_slice) == ColSliceLength);

    bool is_row_reversed = row_slice.step < 0;
    bool is_col_reversed = col_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        int jj = 0;
        for (
            int j = col_slice.start;
            is_col_reversed ? (j > col_slice.stop) : (j < col_slice.stop);
            j += col_slice.step
        )
        {
            adj_value.data[ii][jj] -= adj_m.data[i][j];
            ++jj;
        }

        assert(jj == ColSliceLength);
        ++ii;
    }

    assert(ii == RowSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void assign_inplace(mat_t<Rows,Cols,Type>& m, int row, int col, Type value)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }
    if (col < 0)
    {
        col += Cols;
    }

    m.data[row][col] = value;
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void assign_inplace(mat_t<Rows,Cols,Type>& m, int row, vec_t<Cols,Type>& value)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }

    for(unsigned i=0; i < Cols; ++i)
    {
        m.data[row][i] = value[i];
    }
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void assign_inplace(mat_t<Rows,Cols,Type>& m, slice_t row_slice, Type value)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);

    bool is_row_reversed = row_slice.step < 0;

    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        for (int j = 0; j < Cols; ++j)
        {
            m.data[i][j] = value;
        }
    }
}


template<unsigned RowSliceLength, unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void assign_inplace(mat_t<Rows,Cols,Type>& m, slice_t row_slice, mat_t<RowSliceLength, ColSliceLength, Type>& value)
{
    static_assert(
        RowSliceLength == 0 ? ColSliceLength == 0 : ColSliceLength == Cols,
        "Expected RowSliceLength == 0 ? ColSliceLength == 0 : ColSliceLength == Cols"
    );

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    bool is_row_reversed = row_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        for (int j = 0; j < Cols; ++j)
        {
            m.data[i][j] = value.data[ii][j];
        }

        ++ii;
    }

    assert(ii == RowSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void assign_inplace(mat_t<Rows,Cols,Type>& m, slice_t row_slice, int col, Type value)
{
#ifndef NDEBUG
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);

    if (col < 0)
    {
        col += Cols;
    }

    bool is_row_reversed = row_slice.step < 0;

    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        m.data[i][col] = value;
    }
}


template<unsigned RowSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void assign_inplace(mat_t<Rows,Cols,Type>& m, slice_t row_slice, int col, vec_t<RowSliceLength, Type>& value)
{
#ifndef NDEBUG
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    if (col < 0)
    {
        col += Cols;
    }

    bool is_row_reversed = row_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        m.data[i][col] = value.c[ii];
        ++ii;
    }

    assert(ii == RowSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void assign_inplace(mat_t<Rows,Cols,Type>& m, int row, slice_t col_slice, Type value)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);

    if (row < 0)
    {
        row += Rows;
    }

    bool is_col_reversed = col_slice.step < 0;

    for (
        int i = col_slice.start;
        is_col_reversed ? (i > col_slice.stop) : (i < col_slice.stop);
        i += col_slice.step
    )
    {
        m.data[row][i] = value;
    }
}


template<unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void assign_inplace(mat_t<Rows,Cols,Type>& m, int row, slice_t col_slice, vec_t<ColSliceLength, Type>& value)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);
    assert(slice_get_length(col_slice) == ColSliceLength);

    if (row < 0)
    {
        row += Rows;
    }

    bool is_col_reversed = col_slice.step < 0;

    int ii = 0;
    for (
        int i = col_slice.start;
        is_col_reversed ? (i > col_slice.stop) : (i < col_slice.stop);
        i += col_slice.step
    )
    {
        m.data[row][i] = value.c[ii];
        ++ii;
    }

    assert(ii == ColSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void assign_inplace(mat_t<Rows,Cols,Type>& m, slice_t row_slice, slice_t col_slice, Type value)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);

    bool is_row_reversed = row_slice.step < 0;
    bool is_col_reversed = col_slice.step < 0;

    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        for (
            int j = col_slice.start;
            is_col_reversed ? (j > col_slice.stop) : (j < col_slice.stop);
            j += col_slice.step
        )
        {
            m.data[i][j] = value;
        }
    }
}


template<unsigned RowSliceLength, unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void assign_inplace(mat_t<Rows,Cols,Type>& m, slice_t row_slice, slice_t col_slice, mat_t<RowSliceLength, ColSliceLength, Type>& value)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);
    assert(slice_get_length(col_slice) == ColSliceLength);

    bool is_row_reversed = row_slice.step < 0;
    bool is_col_reversed = col_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        int jj = 0;
        for (
            int j = col_slice.start;
            is_col_reversed ? (j > col_slice.stop) : (j < col_slice.stop);
            j += col_slice.step
        )
        {
            m.data[i][j] = value.data[ii][jj];
            ++jj;
        }

        assert(jj == ColSliceLength);
        ++ii;
    }

    assert(ii == RowSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_assign_inplace(mat_t<Rows,Cols,Type>& m, int row, int col, Type value,
                                        mat_t<Rows,Cols,Type>& adj_m, int& adj_row, int& adj_col, Type& adj_value)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }
    if (col < 0)
    {
        col += Cols;
    }

    adj_value += adj_m.data[row][col];
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_assign_inplace(mat_t<Rows,Cols,Type>& m, int row, vec_t<Cols,Type>& value,
                                        mat_t<Rows,Cols,Type>& adj_m, int& adj_row, vec_t<Cols,Type>& adj_value)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }

    for(unsigned i=0; i < Cols; ++i)
    {
        adj_value[i] += adj_m.data[row][i];
    }
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_assign_inplace(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, Type value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, Type& adj_value
)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);

    bool is_row_reversed = row_slice.step < 0;

    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        for (int j = 0; j < Cols; ++j)
        {
            adj_value += adj_m.data[i][j];
        }
    }
}


template<unsigned RowSliceLength, unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_assign_inplace(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, mat_t<RowSliceLength, ColSliceLength, Type>& value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, mat_t<RowSliceLength, ColSliceLength, Type>& adj_value
)
{
    static_assert(
        RowSliceLength == 0 ? ColSliceLength == 0 : ColSliceLength == Cols,
        "Expected RowSliceLength == 0 ? ColSliceLength == 0 : ColSliceLength == Cols"
    );

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    bool is_row_reversed = row_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        for (int j = 0; j < Cols; ++j)
        {
            adj_value.data[ii][j] += adj_m.data[i][j];
        }

        ++ii;
    }

    assert(ii == RowSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_assign_inplace(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, int col, Type value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, int& adj_col, Type& adj_value
)
{
#ifndef NDEBUG
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);

    if (col < 0)
    {
        col += Cols;
    }

    bool is_row_reversed = row_slice.step < 0;

    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        adj_value += adj_m.data[i][col];
    }
}


template<unsigned RowSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_assign_inplace(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, int col, vec_t<RowSliceLength, Type>& value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, int& adj_col, vec_t<RowSliceLength, Type>& adj_value
)
{
#ifndef NDEBUG
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    if (col < 0)
    {
        col += Cols;
    }

    bool is_row_reversed = row_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        adj_value.c[ii] += adj_m.data[i][col];
        ++ii;
    }

    assert(ii == RowSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_assign_inplace(
    mat_t<Rows,Cols,Type>& m, int row, slice_t col_slice, Type value,
    mat_t<Rows,Cols,Type>& adj_m, int& adj_row, slice_t& adj_col_slice, Type& adj_value
)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);

    if (row < 0)
    {
        row += Rows;
    }

    bool is_col_reversed = col_slice.step < 0;

    for (
        int i = col_slice.start;
        is_col_reversed ? (i > col_slice.stop) : (i < col_slice.stop);
        i += col_slice.step
    )
    {
        adj_value += adj_m.data[row][i];
    }
}


template<unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_assign_inplace(
    mat_t<Rows,Cols,Type>& m, int row, slice_t col_slice, vec_t<ColSliceLength, Type>& value,
    mat_t<Rows,Cols,Type>& adj_m, int& adj_row, slice_t& adj_col_slice, vec_t<ColSliceLength, Type>& adj_value
)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);
    assert(slice_get_length(col_slice) == ColSliceLength);

    if (row < 0)
    {
        row += Rows;
    }

    bool is_col_reversed = col_slice.step < 0;

    int ii = 0;
    for (
        int i = col_slice.start;
        is_col_reversed ? (i > col_slice.stop) : (i < col_slice.stop);
        i += col_slice.step
    )
    {
        adj_value.c[ii] += adj_m.data[row][i];
        ++ii;
    }

    assert(ii == ColSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_assign_inplace(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, slice_t col_slice, Type value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, slice_t& adj_col_slice, Type& adj_value
)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);

    bool is_row_reversed = row_slice.step < 0;
    bool is_col_reversed = col_slice.step < 0;

    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        for (
            int j = col_slice.start;
            is_col_reversed ? (j > col_slice.stop) : (j < col_slice.stop);
            j += col_slice.step
        )
        {
            adj_value += adj_m.data[i][j];
        }
    }
}


template<unsigned RowSliceLength, unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_assign_inplace(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, slice_t col_slice, mat_t<RowSliceLength, ColSliceLength, Type>& value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, slice_t& adj_col_slice, mat_t<RowSliceLength, ColSliceLength, Type>& adj_value
)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);
    assert(slice_get_length(col_slice) == ColSliceLength);

    bool is_row_reversed = row_slice.step < 0;
    bool is_col_reversed = col_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        int jj = 0;
        for (
            int j = col_slice.start;
            is_col_reversed ? (j > col_slice.stop) : (j < col_slice.stop);
            j += col_slice.step
        )
        {
            adj_value.data[ii][jj] += adj_m.data[i][j];
            ++jj;
        }

        assert(jj == ColSliceLength);
        ++ii;
    }

    assert(ii == RowSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> assign_copy(mat_t<Rows,Cols,Type>& m, int row, int col, Type value)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }
    if (col < 0)
    {
        col += Cols;
    }

    mat_t<Rows,Cols,Type> ret(m);
    ret.data[row][col] = value;
    return ret;
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> assign_copy(mat_t<Rows,Cols,Type>& m, int row, vec_t<Cols,Type>& value)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }

    mat_t<Rows,Cols,Type> ret(m);
    for(unsigned i=0; i < Cols; ++i)
    {
        ret.data[row][i] = value[i];
    }
    return ret;
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> assign_copy(mat_t<Rows,Cols,Type>& m, slice_t row_slice, Type value)
{
    mat_t<Rows, Cols, Type> ret(m);
    assign_inplace(ret, row_slice, value);
    return ret;
}


template<unsigned RowSliceLength, unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> assign_copy(mat_t<Rows,Cols,Type>& m, slice_t row_slice, mat_t<RowSliceLength, ColSliceLength, Type>& value)
{
    mat_t<Rows, Cols, Type> ret(m);
    assign_inplace(ret, row_slice, value);
    return ret;
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> assign_copy(mat_t<Rows,Cols,Type>& m, slice_t row_slice, int col, Type value)
{
    mat_t<Rows, Cols, Type> ret(m);
    assign_inplace(ret, row_slice, col, value);
    return ret;
}


template<unsigned RowSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> assign_copy(mat_t<Rows,Cols,Type>& m, slice_t row_slice, int col, vec_t<RowSliceLength, Type>& value)
{
    mat_t<Rows, Cols, Type> ret(m);
    assign_inplace(ret, row_slice, col, value);
    return ret;
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> assign_copy(mat_t<Rows,Cols,Type>& m, int row, slice_t col_slice, Type value)
{
    mat_t<Rows, Cols, Type> ret(m);
    assign_inplace(ret, row, col_slice, value);
    return ret;
}


template<unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> assign_copy(mat_t<Rows,Cols,Type>& m, int row, slice_t col_slice, vec_t<ColSliceLength, Type>& value)
{
    mat_t<Rows, Cols, Type> ret(m);
    assign_inplace(ret, row, col_slice, value);
    return ret;
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> assign_copy(mat_t<Rows,Cols,Type>& m, slice_t row_slice, slice_t col_slice, Type value)
{
    mat_t<Rows, Cols, Type> ret(m);
    assign_inplace(ret, row_slice, col_slice, value);
    return ret;
}


template<unsigned RowSliceLength, unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> assign_copy(mat_t<Rows,Cols,Type>& m, slice_t row_slice, slice_t col_slice, mat_t<RowSliceLength, ColSliceLength, Type>& value)
{
    mat_t<Rows, Cols, Type> ret(m);
    assign_inplace(ret, row_slice, col_slice, value);
    return ret;
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_assign_copy(mat_t<Rows,Cols,Type>& m, int row, int col, Type value,
                                        mat_t<Rows,Cols,Type>& adj_m, int& adj_row, int& adj_col, Type& adj_value, const mat_t<Rows,Cols,Type>& adj_ret)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }
    if (col < 0)
    {
        col += Cols;
    }

    adj_value += adj_ret.data[row][col];
    for(unsigned i=0; i < Rows; ++i)
    {
        for(unsigned j=0; j < Cols; ++j)
        {
            if(i != row || j != col)
                adj_m.data[i][j] += adj_ret.data[i][j];
        }
    }
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_assign_copy(mat_t<Rows,Cols,Type>& m, int row, vec_t<Cols,Type>& value,
                                        mat_t<Rows,Cols,Type>& adj_m, int& adj_row, vec_t<Cols,Type>& adj_value, const mat_t<Rows,Cols,Type>& adj_ret)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }

    for(unsigned i=0; i < Rows; ++i)
    {
        for(unsigned j=0; j < Cols; ++j)
        {
            if (i==row)
                adj_value[j] += adj_ret.data[i][j];
            else
                adj_m.data[i][j] += adj_ret.data[i][j];
        }
    }
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_assign_copy(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, Type value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, Type& adj_value,
    mat_t<Rows,Cols,Type>& adj_ret
)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);

    bool is_row_reversed = row_slice.step < 0;

    for (int i = 0; i < Rows; ++i)
    {
        bool in_row_slice = is_row_reversed
            ? (i <= row_slice.start && i > row_slice.stop && (row_slice.start - i) % (-row_slice.step) == 0)
            : (i >= row_slice.start && i < row_slice.stop && (i - row_slice.start) % row_slice.step == 0);

        if (!in_row_slice)
        {
            for (int j = 0; j < Cols; ++j)
            {
                adj_m.data[i][j] += adj_ret.data[i][j];
            }
        }
        else
        {
            for (int j = 0; j < Cols; ++j)
            {
                adj_value += adj_ret.data[i][j];
            }
        }
    }
}


template<unsigned RowSliceLength, unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_assign_copy(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, mat_t<RowSliceLength, ColSliceLength, Type>& value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, mat_t<RowSliceLength, ColSliceLength, Type>& adj_value,
    mat_t<Rows,Cols,Type>& adj_ret
)
{
    static_assert(
        RowSliceLength == 0 ? ColSliceLength == 0 : ColSliceLength == Cols,
        "Expected RowSliceLength == 0 ? ColSliceLength == 0 : ColSliceLength == Cols"
    );

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    bool is_row_reversed = row_slice.step < 0;

    int ii = 0;
    for (int i = 0; i < Rows; ++i)
    {
        bool in_row_slice = is_row_reversed
            ? (i <= row_slice.start && i > row_slice.stop && (row_slice.start - i) % (-row_slice.step) == 0)
            : (i >= row_slice.start && i < row_slice.stop && (i - row_slice.start) % row_slice.step == 0);

        if (!in_row_slice)
        {
            for (int j = 0; j < Cols; ++j)
            {
                adj_m.data[i][j] += adj_ret.data[i][j];
            }
        }
        else
        {
            for (int j = 0; j < Cols; ++j)
            {
                adj_value.data[ii][j] += adj_ret.data[i][j];
            }

            ++ii;
        }
    }

    assert(ii == RowSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_assign_copy(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, int col, Type value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, int& adj_col, Type& adj_value,
    mat_t<Rows,Cols,Type>& adj_ret
)
{
#ifndef NDEBUG
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);

    if (col < 0)
    {
        col += Cols;
    }

    bool is_row_reversed = row_slice.step < 0;

    for (int i = 0; i < Rows; ++i)
    {
        bool in_row_slice = is_row_reversed
            ? (i <= row_slice.start && i > row_slice.stop && (row_slice.start - i) % (-row_slice.step) == 0)
            : (i >= row_slice.start && i < row_slice.stop && (i - row_slice.start) % row_slice.step == 0);

        if (!in_row_slice)
        {
            for (int j = 0; j < Cols; ++j)
            {
                adj_m.data[i][j] += adj_ret.data[i][j];
            }
        }
        else
        {
            for (int j = 0; j < Cols; ++j)
            {
                if (j != col)
                {
                    adj_m.data[i][j] += adj_ret.data[i][j];
                }
                else
                {
                    adj_value += adj_ret.data[i][j];
                }
            }
        }
    }
}


template<unsigned RowSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_assign_copy(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, int col, vec_t<RowSliceLength, Type>& value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, int& adj_col, vec_t<RowSliceLength, Type>& adj_value,
    mat_t<Rows,Cols,Type>& adj_ret
)
{
#ifndef NDEBUG
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    if (col < 0)
    {
        col += Cols;
    }

    bool is_row_reversed = row_slice.step < 0;

    int ii = 0;
    for (int i = 0; i < Rows; ++i)
    {
        bool in_row_slice = is_row_reversed
            ? (i <= row_slice.start && i > row_slice.stop && (row_slice.start - i) % (-row_slice.step) == 0)
            : (i >= row_slice.start && i < row_slice.stop && (i - row_slice.start) % row_slice.step == 0);

        if (!in_row_slice)
        {
            for (int j = 0; j < Cols; ++j)
            {
                adj_m.data[i][j] += adj_ret.data[i][j];
            }
        }
        else
        {
            for (int j = 0; j < Cols; ++j)
            {
                if (j != col)
                {
                    adj_m.data[i][j] += adj_ret.data[i][j];
                }
                else
                {
                    adj_value.c[ii] += adj_ret.data[i][j];
                }
            }

            ++ii;
        }
    }

    assert(ii == RowSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_assign_copy(
    mat_t<Rows,Cols,Type>& m, int row, slice_t col_slice, Type value,
    mat_t<Rows,Cols,Type>& adj_m, int& adj_row, slice_t& adj_col_slice, Type& adj_value,
    mat_t<Rows,Cols,Type>& adj_ret
)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);

    bool is_col_reversed = col_slice.step < 0;

    for (int i = 0; i < Rows; ++i)
    {
        if (i != row)
        {
            for (int j = 0; j < Cols; ++j)
            {
                adj_m.data[i][j] += adj_ret.data[i][j];
            }
        }
        else
        {
            for (int j = 0; j < Cols; ++j)
            {
                bool in_col_slice = is_col_reversed
                    ? (j <= col_slice.start && j > col_slice.stop && (col_slice.start - j) % (-col_slice.step) == 0)
                    : (j >= col_slice.start && j < col_slice.stop && (j - col_slice.start) % col_slice.step == 0);

                if (!in_col_slice)
                {
                    adj_m.data[i][j] += adj_ret.data[i][j];
                }
                else
                {
                    adj_value += adj_ret.data[i][j];
                }
            }
        }
    }
}


template<unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_assign_copy(
    mat_t<Rows,Cols,Type>& m, int row, slice_t col_slice, vec_t<ColSliceLength, Type>& value,
    mat_t<Rows,Cols,Type>& adj_m, int& adj_row, slice_t& adj_col_slice, vec_t<ColSliceLength, Type>& adj_value,
    mat_t<Rows,Cols,Type>& adj_ret
)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);
    assert(slice_get_length(col_slice) == ColSliceLength);

    bool is_col_reversed = col_slice.step < 0;

    int ii = 0;
    for (int i = 0; i < Rows; ++i)
    {
        if (i != row)
        {
            for (int j = 0; j < Cols; ++j)
            {
                adj_m.data[i][j] += adj_ret.data[i][j];
            }
        }
        else
        {
            for (int j = 0; j < Cols; ++j)
            {
                bool in_col_slice = is_col_reversed
                    ? (j <= col_slice.start && j > col_slice.stop && (col_slice.start - j) % (-col_slice.step) == 0)
                    : (j >= col_slice.start && j < col_slice.stop && (j - col_slice.start) % col_slice.step == 0);

                if (!in_col_slice)
                {
                    adj_m.data[i][j] += adj_ret.data[i][j];
                }
                else
                {
                    adj_value.c[ii] += adj_ret.data[i][j];
                    ++ii;
                }
            }
        }
    }

    assert(ii == ColSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_assign_copy(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, slice_t col_slice, Type value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, slice_t& adj_col_slice, Type& adj_value,
    mat_t<Rows,Cols,Type>& adj_ret
)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);

    bool is_row_reversed = row_slice.step < 0;
    bool is_col_reversed = col_slice.step < 0;

    for (int i = 0; i < Rows; ++i)
    {
        bool in_row_slice = is_row_reversed
            ? (i <= row_slice.start && i > row_slice.stop && (row_slice.start - i) % (-row_slice.step) == 0)
            : (i >= row_slice.start && i < row_slice.stop && (i - row_slice.start) % row_slice.step == 0);

        if (!in_row_slice)
        {
            for (int j = 0; j < Cols; ++j)
            {
                adj_m.data[i][j] += adj_ret.data[i][j];
            }
        }
        else
        {
            assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
            assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
            assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);

            for (int j = 0; j < Cols; ++j)
            {
                bool in_col_slice = is_col_reversed
                    ? (j <= col_slice.start && j > col_slice.stop && (col_slice.start - j) % (-col_slice.step) == 0)
                    : (j >= col_slice.start && j < col_slice.stop && (j - col_slice.start) % col_slice.step == 0);

                if (!in_col_slice)
                {
                    adj_m.data[i][j] += adj_ret.data[i][j];
                }
                else
                {
                    adj_value += adj_ret.data[i][j];
                }
            }
        }
    }
}


template<unsigned RowSliceLength, unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_assign_copy(
    mat_t<Rows,Cols,Type>& m, slice_t row_slice, slice_t col_slice, mat_t<RowSliceLength, ColSliceLength, Type>& value,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, slice_t& adj_col_slice, mat_t<RowSliceLength, ColSliceLength, Type>& adj_value,
    mat_t<Rows,Cols,Type>& adj_ret
)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    bool is_row_reversed = row_slice.step < 0;
    bool is_col_reversed = col_slice.step < 0;

    int ii = 0;
    for (int i = 0; i < Rows; ++i)
    {
        bool in_row_slice = is_row_reversed
            ? (i <= row_slice.start && i > row_slice.stop && (row_slice.start - i) % (-row_slice.step) == 0)
            : (i >= row_slice.start && i < row_slice.stop && (i - row_slice.start) % row_slice.step == 0);

        if (!in_row_slice)
        {
            for (int j = 0; j < Cols; ++j)
            {
                adj_m.data[i][j] += adj_ret.data[i][j];
            }
        }
        else
        {
            assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
            assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
            assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);
            assert(slice_get_length(col_slice) == ColSliceLength);

            int jj = 0;
            for (int j = 0; j < Cols; ++j)
            {
                bool in_col_slice = is_col_reversed
                    ? (j <= col_slice.start && j > col_slice.stop && (col_slice.start - j) % (-col_slice.step) == 0)
                    : (j >= col_slice.start && j < col_slice.stop && (j - col_slice.start) % col_slice.step == 0);

                if (!in_col_slice)
                {
                    adj_m.data[i][j] += adj_ret.data[i][j];
                }
                else
                {
                    adj_value.data[ii][jj] += adj_ret.data[i][j];
                    ++jj;
                }
            }

            assert(jj == ColSliceLength);
            ++ii;
        }
    }

    assert(ii == RowSliceLength);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline bool CUDA_CALLABLE isfinite(const mat_t<Rows,Cols,Type>& m)
{
    for (unsigned i=0; i < Rows; ++i)
        for (unsigned j=0; j < Cols; ++j)
            if (!isfinite(m.data[i][j]))
                return false;
    return true;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline void CUDA_CALLABLE adj_isfinite(const mat_t<Rows,Cols,Type>& m, mat_t<Rows,Cols,Type>& adj_m, const bool &adj_ret)
{
}

template<unsigned Rows, unsigned Cols, typename Type>
inline bool CUDA_CALLABLE isnan(const mat_t<Rows,Cols,Type>& m)
{
    for (unsigned i=0; i < Rows; ++i)
        for (unsigned j=0; j < Cols; ++j)
            if (isnan(m.data[i][j]))
                return true;
    return false;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline void CUDA_CALLABLE adj_isnan(const mat_t<Rows,Cols,Type>& m, mat_t<Rows,Cols,Type>& adj_m, const bool &adj_ret)
{
}

template<unsigned Rows, unsigned Cols, typename Type>
inline bool CUDA_CALLABLE isinf(const mat_t<Rows,Cols,Type>& m)
{
    for (unsigned i=0; i < Rows; ++i)
        for (unsigned j=0; j < Cols; ++j)
            if (isinf(m.data[i][j]))
                return true;
    return false;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline void CUDA_CALLABLE adj_isinf(const mat_t<Rows,Cols,Type>& m, mat_t<Rows,Cols,Type>& adj_m, const bool &adj_ret)
{
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> add(const mat_t<Rows,Cols,Type>& a, const mat_t<Rows,Cols,Type>& b)
{
    mat_t<Rows,Cols,Type> t;
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            t.data[i][j] = a.data[i][j] + b.data[i][j];
        }
    }

    return t;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> add(const mat_t<Rows,Cols,Type>& a, Type b)
{
    mat_t<Rows,Cols,Type> t;
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            t.data[i][j] = a.data[i][j] + b;
        }
    }

    return t;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> add(Type a, const mat_t<Rows,Cols,Type>& b)
{
    mat_t<Rows,Cols,Type> t;
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            t.data[i][j] = a + b.data[i][j];
        }
    }

    return t;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> sub(const mat_t<Rows,Cols,Type>& a, const mat_t<Rows,Cols,Type>& b)
{
    mat_t<Rows,Cols,Type> t;
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            t.data[i][j] = a.data[i][j] - b.data[i][j];
        }
    }

    return t;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> sub(const mat_t<Rows,Cols,Type>& a, Type b)
{
    mat_t<Rows,Cols,Type> t;
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            t.data[i][j] = a.data[i][j] - b;
        }
    }

    return t;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> sub(Type a, const mat_t<Rows,Cols,Type>& b)
{
    mat_t<Rows,Cols,Type> t;
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            t.data[i][j] = a - b.data[i][j];
        }
    }

    return t;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> div(const mat_t<Rows,Cols,Type>& a, Type b)
{
    mat_t<Rows,Cols,Type> t;
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            t.data[i][j] = a.data[i][j]/b;
        }
    }

    return t;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> div(Type b, const mat_t<Rows,Cols,Type>& a)
{
    mat_t<Rows,Cols,Type> t;
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            t.data[i][j] = b / a.data[i][j];
        }
    }

    return t;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> mul(const mat_t<Rows,Cols,Type>& a, Type b)
{
    mat_t<Rows,Cols,Type> t;
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            t.data[i][j] = a.data[i][j]*b;
        }
    }

    return t;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> mul(Type b, const mat_t<Rows,Cols,Type>& a)
{
    return mul(a,b);
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> operator*(Type b, const mat_t<Rows,Cols,Type>& a)
{
    return mul(a,b);
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> operator*( const mat_t<Rows,Cols,Type>& a, Type b)
{
    return mul(a,b);
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE vec_t<Rows,Type> mul(const mat_t<Rows,Cols,Type>& a, const vec_t<Cols,Type>& b)
{
    vec_t<Rows,Type> r = a.get_col(0)*b[0];
    for( unsigned i=1; i < Cols; ++i )
    {
        r += a.get_col(i)*b[i];
    }
    return r;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE vec_t<Cols,Type> mul(const vec_t<Rows,Type>& b, const mat_t<Rows,Cols,Type>& a)
{
    vec_t<Cols,Type> r = a.get_row(0)*b[0];
    for( unsigned i=1; i < Rows; ++i )
    {
        r += a.get_row(i)*b[i];
    }
    return r;
}

template<typename T>
inline CUDA_CALLABLE T muladd(T a, T b, T c) {
    return c + a*b;
}
template<>
inline CUDA_CALLABLE float muladd(float a, float b, float c) {
    return fmaf(a, b, c);
}
template<>
inline CUDA_CALLABLE double muladd(double a, double b, double c) {
    return fma(a, b, c);
}


template<unsigned Rows, unsigned Cols, unsigned ColsOut, typename Type>
inline CUDA_CALLABLE mat_t<Rows,ColsOut,Type> mul(const mat_t<Rows,Cols,Type>& a, const mat_t<Cols,ColsOut,Type>& b)
{
    mat_t<Rows,ColsOut,Type> t(0);
    for (unsigned i=0; i < Rows; ++i)
    {        
        for (unsigned j=0; j < ColsOut; ++j)
        {
            Type sum(0.0);

            for (unsigned k=0; k < Cols; ++k)
            {
                sum = muladd<Type>(a.data[i][k], b.data[k][j], sum);
            }

            t.data[i][j] = sum;
        }
    }
    
    return t;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> mod(const mat_t<Rows,Cols,Type>& a, Type b)
{
    mat_t<Rows,Cols,Type> t;
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            t.data[i][j] = mod(a.data[i][j], b);
        }
    }

    return t;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE Type ddot(const mat_t<Rows,Cols,Type>& a, const mat_t<Rows,Cols,Type>& b)
{
    // double dot product between a and b:
    Type r(0);
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            r = muladd<Type>(a.data[i][j], b.data[i][j], r);
        }
    }
    return r;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE Type tensordot(const mat_t<Rows,Cols,Type>& a, const mat_t<Rows,Cols,Type>& b)
{
    // corresponds to `np.tensordot()` with all axes being contracted
    return ddot(a, b);
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Cols,Rows,Type> transpose(const mat_t<Rows,Cols,Type>& a)
{
    mat_t<Cols,Rows,Type> t;
    for (unsigned i=0; i < Cols; ++i)
    {
        for (unsigned j=0; j < Rows; ++j)
        {
            t.data[i][j] = a.data[j][i];
        }
    }

    return t;
}

// Only implementing determinants for 2x2, 3x3 and 4x4 matrices for now...
template<typename Type>
inline CUDA_CALLABLE Type determinant(const mat_t<2,2,Type>& m)
{
    return m.data[0][0]*m.data[1][1] - m.data[1][0]*m.data[0][1];
}

template<typename Type>
inline CUDA_CALLABLE Type determinant(const mat_t<3,3,Type>& m)
{
    return dot(
        vec_t<3,Type>(m.data[0][0],m.data[0][1],m.data[0][2]),
        cross(
            vec_t<3,Type>(m.data[1][0],m.data[1][1],m.data[1][2]),
            vec_t<3,Type>(m.data[2][0],m.data[2][1],m.data[2][2])
        )
    );
}

template<typename Type>
inline CUDA_CALLABLE Type determinant(const mat_t<4,4,Type>& m)
{
    // adapted from USD GfMatrix4f::Inverse()
    Type x00, x01, x02, x03;
    Type x10, x11, x12, x13;
    Type x20, x21, x22, x23;
    Type x30, x31, x32, x33;
    double y01, y02, y03, y12, y13, y23;
    Type z00, z10, z20, z30;

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

template<unsigned Rows, typename Type>
inline CUDA_CALLABLE Type trace(const mat_t<Rows,Rows,Type>& m)
{
    Type ret = m.data[0][0];
    for( unsigned i=1; i < Rows; ++i )
    {
        ret += m.data[i][i];
    }
    return ret;
}

template<unsigned Rows, typename Type>
inline CUDA_CALLABLE vec_t<Rows, Type> get_diag(const mat_t<Rows,Rows,Type>& m)
{
    vec_t<Rows, Type> ret;
    for( unsigned i=0; i < Rows; ++i )
    {
        ret[i] = m.data[i][i];
    }
    return ret;
}

// Only implementing inverses for 2x2, 3x3 and 4x4 matrices for now...
template<typename Type>
inline CUDA_CALLABLE mat_t<2,2,Type> inverse(const mat_t<2,2,Type>& m)
{
    Type det = determinant(m);
    if (det > Type(kEps) || det < -Type(kEps))
    {
        return mat_t<2,2,Type>( m.data[1][1], -m.data[0][1],
                     -m.data[1][0],  m.data[0][0])*(Type(1.0f)/det);
    }
    else
    {
        return mat_t<2,2,Type>();
    }
}

template<typename Type>
inline CUDA_CALLABLE mat_t<3,3,Type> inverse(const mat_t<3,3,Type>& m)
{
	Type det = determinant(m);

	if (det != Type(0.0f))
	{
		mat_t<3,3,Type> b;
		
		b.data[0][0] = m.data[1][1]*m.data[2][2] - m.data[1][2]*m.data[2][1]; 
		b.data[1][0] = m.data[1][2]*m.data[2][0] - m.data[1][0]*m.data[2][2]; 
		b.data[2][0] = m.data[1][0]*m.data[2][1] - m.data[1][1]*m.data[2][0]; 
		
        b.data[0][1] = m.data[0][2]*m.data[2][1] - m.data[0][1]*m.data[2][2]; 
        b.data[1][1] = m.data[0][0]*m.data[2][2] - m.data[0][2]*m.data[2][0]; 
        b.data[2][1] = m.data[0][1]*m.data[2][0] - m.data[0][0]*m.data[2][1]; 

        b.data[0][2] = m.data[0][1]*m.data[1][2] - m.data[0][2]*m.data[1][1];
        b.data[1][2] = m.data[0][2]*m.data[1][0] - m.data[0][0]*m.data[1][2];
        b.data[2][2] = m.data[0][0]*m.data[1][1] - m.data[0][1]*m.data[1][0];

		return b*(Type(1.0f)/det);
	}
	else
	{
		return mat_t<3,3,Type>();
	}
}

template<typename Type>
inline CUDA_CALLABLE mat_t<4,4,Type> inverse(const mat_t<4,4,Type>& m)
{
    // adapted from USD GfMatrix4f::Inverse()
    Type x00, x01, x02, x03;
    Type x10, x11, x12, x13;
    Type x20, x21, x22, x23;
    Type x30, x31, x32, x33;
    double y01, y02, y03, y12, y13, y23;
    Type z00, z10, z20, z30;
    Type z01, z11, z21, z31;
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
    
    if(fabs(det) > kEps) 
    {
        mat_t<4,4,Type> invm;

        double rcp = 1.0 / det;

        // Multiply all 3x3 cofactors by reciprocal & transpose
        invm.data[0][0] = Type(z00*rcp);
        invm.data[0][1] = Type(z10*rcp);
        invm.data[1][0] = Type(z01*rcp);
        invm.data[0][2] = Type(z20*rcp);
        invm.data[2][0] = Type(z02*rcp);
        invm.data[0][3] = Type(z30*rcp);
        invm.data[3][0] = Type(z03*rcp);
        invm.data[1][1] = Type(z11*rcp);
        invm.data[1][2] = Type(z21*rcp);
        invm.data[2][1] = Type(z12*rcp);
        invm.data[1][3] = Type(z31*rcp);
        invm.data[3][1] = Type(z13*rcp);
        invm.data[2][2] = Type(z22*rcp);
        invm.data[2][3] = Type(z32*rcp);
        invm.data[3][2] = Type(z23*rcp);
        invm.data[3][3] = Type(z33*rcp);

        return invm;
    }
    else 
    {
        return mat_t<4,4,Type>();
    }
}

template<unsigned Rows,typename Type>
inline CUDA_CALLABLE mat_t<Rows,Rows,Type> diag(const vec_t<Rows,Type>& d)
{
    mat_t<Rows,Rows,Type> ret(Type(0));
    for (unsigned i=0; i < Rows; ++i)
    {
        ret.data[i][i] = d[i];
    }
    return ret;
}

template<unsigned Rows,unsigned Cols,typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> outer(const vec_t<Rows,Type>& a, const vec_t<Cols,Type>& b)
{
    // col 0 = a * b[0] etc...
    mat_t<Rows,Cols,Type> ret;
    for (unsigned row=0; row < Rows; ++row)
    {
        for (unsigned col=0; col < Cols; ++col) // columns
        {
            ret.data[row][col] = a[row] * b[col];
        }
    }
    return ret;
}

template<unsigned Cols,typename Type>
inline CUDA_CALLABLE vec_t<Cols,Type> outer(Type a, const vec_t<Cols,Type>& b)
{
    return mul(a, b);
}

template<unsigned Rows,typename Type>
inline CUDA_CALLABLE vec_t<Rows,Type> outer(const vec_t<Rows,Type>& a, Type b)
{
    return mul(a, b);
}

template<typename Type>
inline CUDA_CALLABLE mat_t<3,3,Type> skew(const vec_t<3,Type>& a)
{
    mat_t<3,3,Type> out(
        Type(0), -a[2],   a[1],
        a[2],   Type(0), -a[0],
        -a[1],   a[0],   Type(0)
    );

    return out;
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> cw_mul(const mat_t<Rows,Cols,Type>& a, const mat_t<Rows,Cols,Type>& b)
{
    mat_t<Rows,Cols,Type> t;
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            t.data[i][j] = a.data[i][j] * b.data[i][j];
        }
    }

    return t;
}


template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE mat_t<Rows,Cols,Type> cw_div(const mat_t<Rows,Cols,Type>& a, const mat_t<Rows,Cols,Type>& b)
{
    mat_t<Rows,Cols,Type> t;
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            t.data[i][j] = a.data[i][j] / b.data[i][j];
        }
    }

    return t;
}

template<typename Type>
inline CUDA_CALLABLE vec_t<3,Type> transform_point(const mat_t<4,4,Type>& m, const vec_t<3,Type>& v)
{
    vec_t<4,Type> out = mul(m, vec_t<4,Type>(v[0], v[1], v[2], Type(1)));
    return vec_t<3,Type>(out[0], out[1], out[2]);
}

template<typename Type>
inline CUDA_CALLABLE vec_t<3,Type> transform_vector(const mat_t<4,4,Type>& m, const vec_t<3,Type>& v)
{
    vec_t<4,Type> out = mul(m, vec_t<4,Type>(v[0], v[1], v[2], 0.f));
    return vec_t<3,Type>(out[0], out[1], out[2]);
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_extract(const mat_t<Rows,Cols,Type>& m, int row, mat_t<Rows,Cols,Type>& adj_m, int& adj_row, const vec_t<Cols,Type>& adj_ret)
{
    for( unsigned col=0; col < Cols; ++col )
        adj_m.data[row][col] += adj_ret[col];
}

template<unsigned Rows, unsigned Cols, typename Type>
inline void CUDA_CALLABLE adj_extract(const mat_t<Rows,Cols,Type>& m, int row, int col, mat_t<Rows,Cols,Type>& adj_m, int& adj_row, int& adj_col, Type adj_ret)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (row < 0)
    {
        row += Rows;
    }
    if (col < 0)
    {
        col += Cols;
    }

    adj_m.data[row][col] += adj_ret;
}

template<unsigned RowSliceLength, unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_extract(
    const mat_t<Rows,Cols,Type>& m, slice_t row_slice,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice,
    const mat_t<RowSliceLength, ColSliceLength, Type>& adj_ret
)
{
    static_assert(
        RowSliceLength == 0 ? ColSliceLength == 0 : ColSliceLength == Cols,
        "Expected RowSliceLength == 0 ? ColSliceLength == 0 : ColSliceLength == Cols"
    );

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    bool is_row_reversed = row_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        for (int j = 0; j < Cols; ++j)
        {
            adj_m.data[i][j] += adj_ret.data[ii][j];
        }

        ++ii;
    }

    assert(ii == RowSliceLength);
}

template<unsigned RowSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_extract(
    const mat_t<Rows,Cols,Type>& m, slice_t row_slice, int col,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, int& adj_col,
    const vec_t<RowSliceLength, Type>& adj_ret
)
{
#ifndef NDEBUG
    if (col < -(int)Cols || col >= (int)Cols)
    {
        printf("mat col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    if (col < 0)
    {
        col += Cols;
    }

    bool is_row_reversed = row_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        adj_m.data[i][col] += adj_ret.c[ii];
        ++ii;
    }

    assert(ii == RowSliceLength);
}

template<unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_extract(
    const mat_t<Rows,Cols,Type>& m, int row, slice_t col_slice,
    mat_t<Rows,Cols,Type>& adj_m, int& adj_row, slice_t& adj_col_slice,
    const vec_t<ColSliceLength, Type>& adj_ret
)
{
#ifndef NDEBUG
    if (row < -(int)Rows || row >= (int)Rows)
    {
        printf("mat row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
#endif

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);
    assert(slice_get_length(col_slice) == ColSliceLength);

    if (row < 0)
    {
        row += Rows;
    }

    bool is_col_reversed = col_slice.step < 0;

    int ii = 0;
    for (
        int i = col_slice.start;
        is_col_reversed ? (i > col_slice.stop) : (i < col_slice.stop);
        i += col_slice.step
    )
    {
        adj_m.data[row][i] += adj_ret.c[ii];
        ++ii;
    }

    assert(ii == ColSliceLength);
}

template<unsigned RowSliceLength, unsigned ColSliceLength, unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_extract(
    const mat_t<Rows,Cols,Type>& m, slice_t row_slice, slice_t col_slice,
    mat_t<Rows,Cols,Type>& adj_m, slice_t& adj_row_slice, slice_t& adj_col_slice,
    const mat_t<RowSliceLength, ColSliceLength, Type>& adj_ret
)
{
    assert(row_slice.start >= 0 && row_slice.start <= (int)Rows);
    assert(row_slice.stop >= -1 && row_slice.stop <= (int)Rows);
    assert(row_slice.step != 0 && row_slice.step < 0 ? row_slice.start >= row_slice.stop : row_slice.start <= row_slice.stop);
    assert(slice_get_length(row_slice) == RowSliceLength);

    assert(col_slice.start >= 0 && col_slice.start <= (int)Cols);
    assert(col_slice.stop >= -1 && col_slice.stop <= (int)Cols);
    assert(col_slice.step != 0 && col_slice.step < 0 ? col_slice.start >= col_slice.stop : col_slice.start <= col_slice.stop);
    assert(slice_get_length(col_slice) == ColSliceLength);

    bool is_row_reversed = row_slice.step < 0;
    bool is_col_reversed = col_slice.step < 0;

    int ii = 0;
    for (
        int i = row_slice.start;
        is_row_reversed ? (i > row_slice.stop) : (i < row_slice.stop);
        i += row_slice.step
    )
    {
        int jj = 0;
        for (
            int j = col_slice.start;
            is_col_reversed ? (j > col_slice.stop) : (j < col_slice.stop);
            j += col_slice.step
        )
        {
            adj_m.data[i][j] += adj_ret.data[ii][jj];
            ++jj;
        }

        assert(jj == ColSliceLength);
        ++ii;
    }

    assert(ii == RowSliceLength);
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_outer(const vec_t<Rows,Type>& a, const vec_t<Cols,Type>& b, vec_t<Rows,Type>& adj_a, vec_t<Cols,Type>& adj_b, const mat_t<Rows,Cols,Type>& adj_ret)
{
  adj_a += mul(adj_ret, b);
  adj_b += mul(transpose(adj_ret), a);
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_add(const mat_t<Rows,Cols,Type>& a, const mat_t<Rows,Cols,Type>& b, mat_t<Rows,Cols,Type>& adj_a, mat_t<Rows,Cols,Type>& adj_b, const mat_t<Rows,Cols,Type>& adj_ret)
{
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            adj_a.data[i][j] += adj_ret.data[i][j];
            adj_b.data[i][j] += adj_ret.data[i][j];
        }
    }
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_add(
    const mat_t<Rows,Cols,Type>& a, Type b,
    mat_t<Rows,Cols,Type>& adj_a, Type& adj_b,
    const mat_t<Rows,Cols,Type>& adj_ret
)
{
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            adj_a.data[i][j] += adj_ret.data[i][j];
            adj_b += adj_ret.data[i][j];
        }
    }
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_add(
    Type a, const mat_t<Rows,Cols,Type>& b,
    Type& adj_a, mat_t<Rows,Cols,Type>& adj_b,
    const mat_t<Rows,Cols,Type>& adj_ret
)
{
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            adj_a += adj_ret.data[i][j];
            adj_b.data[i][j] += adj_ret.data[i][j];
        }
    }
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_sub(const mat_t<Rows,Cols,Type>& a, const mat_t<Rows,Cols,Type>& b, mat_t<Rows,Cols,Type>& adj_a, mat_t<Rows,Cols,Type>& adj_b, const mat_t<Rows,Cols,Type>& adj_ret)
{
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            adj_a.data[i][j] += adj_ret.data[i][j];
            adj_b.data[i][j] -= adj_ret.data[i][j];
        }
    }
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_sub(
    const mat_t<Rows,Cols,Type>& a, Type b,
    mat_t<Rows,Cols,Type>& adj_a, Type& adj_b,
    const mat_t<Rows,Cols,Type>& adj_ret
)
{
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            adj_a.data[i][j] += adj_ret.data[i][j];
            adj_b -= adj_ret.data[i][j];
        }
    }
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_sub(
    Type a, const mat_t<Rows,Cols,Type>& b,
    Type& adj_a, mat_t<Rows,Cols,Type>& adj_b,
    const mat_t<Rows,Cols,Type>& adj_ret
)
{
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            adj_a += adj_ret.data[i][j];
            adj_b.data[i][j] -= adj_ret.data[i][j];
        }
    }
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_div(const mat_t<Rows,Cols,Type>& a, Type s, mat_t<Rows,Cols,Type>& adj_a, Type& adj_s, const mat_t<Rows,Cols,Type>& adj_ret)
{
    adj_s -= tensordot(a , adj_ret)/ (s * s); // - a / s^2

    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            adj_a.data[i][j] += adj_ret.data[i][j] / s;
        }
    }
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_div(Type s, const mat_t<Rows,Cols,Type>& a, Type& adj_s, mat_t<Rows,Cols,Type>& adj_a, const mat_t<Rows,Cols,Type>& adj_ret)
{
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            Type inv = Type(1) / a.data[i][j];
            adj_a.data[i][j] -= s * adj_ret.data[i][j] * inv * inv;
            adj_s += adj_ret.data[i][j] * inv;
        }
    }
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_mul(const mat_t<Rows,Cols,Type>& a, Type b, mat_t<Rows,Cols,Type>& adj_a, Type& adj_b, const mat_t<Rows,Cols,Type>& adj_ret)
{
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            adj_a.data[i][j] += b*adj_ret.data[i][j];
            adj_b += a.data[i][j]*adj_ret.data[i][j];
        }
    }
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_mul(Type b, const mat_t<Rows,Cols,Type>& a, Type& adj_b, mat_t<Rows,Cols,Type>& adj_a, const mat_t<Rows,Cols,Type>& adj_ret)
{
    adj_mul(a, b, adj_a, adj_b, adj_ret);
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_ddot(mat_t<Rows,Cols,Type> a, mat_t<Rows,Cols,Type> b, mat_t<Rows,Cols,Type>& adj_a, mat_t<Rows,Cols,Type>& adj_b, const Type adj_ret)
{
    adj_a += b*adj_ret;
    adj_b += a*adj_ret;
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_mul(const mat_t<Rows,Cols,Type>& a, const vec_t<Cols,Type>& b, mat_t<Rows,Cols,Type>& adj_a, vec_t<Cols,Type>& adj_b, const vec_t<Rows,Type>& adj_ret)
{
    adj_a += outer(adj_ret, b);
    adj_b += mul(transpose(a), adj_ret);
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_mul(const vec_t<Rows,Type>& b, const mat_t<Rows,Cols,Type>& a, vec_t<Rows,Type>& adj_b, mat_t<Rows,Cols,Type>& adj_a, const vec_t<Cols,Type>& adj_ret)
{
    adj_a += outer(b, adj_ret);
    adj_b += mul(adj_ret, transpose(a));
}

template<unsigned Rows, unsigned Cols, unsigned ColsOut, typename Type>
inline CUDA_CALLABLE void adj_mul(const mat_t<Rows,Cols,Type>& a, const mat_t<Cols,ColsOut,Type>& b, mat_t<Rows,Cols,Type>& adj_a, mat_t<Cols,ColsOut,Type>& adj_b, const mat_t<Rows,ColsOut,Type>& adj_ret)
{
    adj_a += mul(adj_ret, transpose(b));
    adj_b += mul(transpose(a), adj_ret);
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_mod(
    const mat_t<Rows,Cols,Type>& a, Type b,
    mat_t<Rows,Cols,Type>& adj_a, Type& adj_b,
    const mat_t<Rows,Cols,Type>& adj_ret
)
{
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_transpose(const mat_t<Rows,Cols,Type>& a, mat_t<Rows,Cols,Type>& adj_a, const mat_t<Cols,Rows,Type>& adj_ret)
{
    adj_a += transpose(adj_ret);
}

template<unsigned Rows, typename Type>
inline CUDA_CALLABLE void adj_trace(const mat_t<Rows,Rows,Type>& m, mat_t<Rows,Rows,Type>& adj_m, Type adj_ret)
{
    for (unsigned i=0; i < Rows; ++i)
        adj_m.data[i][i] += adj_ret;
}

template<unsigned Rows, typename Type>
inline CUDA_CALLABLE void adj_diag(const vec_t<Rows,Type>& d, vec_t<Rows,Type>& adj_d, const mat_t<Rows,Rows,Type>& adj_ret)
{
    for (unsigned i=0; i < Rows; ++i)
        adj_d[i] += adj_ret.data[i][i];
}

template<unsigned Rows, typename Type>
inline CUDA_CALLABLE void adj_get_diag(const mat_t<Rows,Rows,Type>& m, mat_t<Rows,Rows,Type>& adj_m, const vec_t<Rows,Type>& adj_ret)
{
    for (unsigned i=0; i < Rows; ++i)
        adj_m.data[i][i] += adj_ret[i];
}

template<typename Type>
inline CUDA_CALLABLE void adj_determinant(const mat_t<2,2,Type>& m, mat_t<2,2,Type>& adj_m, Type adj_ret)
{
    adj_m.data[0][0] += m.data[1][1]*adj_ret;
    adj_m.data[1][1] += m.data[0][0]*adj_ret;
    adj_m.data[0][1] -= m.data[1][0]*adj_ret;
    adj_m.data[1][0] -= m.data[0][1]*adj_ret;
}

template<typename Type>
inline CUDA_CALLABLE void adj_determinant(const mat_t<3,3,Type>& m, mat_t<3,3,Type>& adj_m, Type adj_ret)
{
    (vec_t<3,Type>&)adj_m.data[0] += cross(m.get_row(1), m.get_row(2))*adj_ret;
    (vec_t<3,Type>&)adj_m.data[1] += cross(m.get_row(2), m.get_row(0))*adj_ret;
    (vec_t<3,Type>&)adj_m.data[2] += cross(m.get_row(0), m.get_row(1))*adj_ret;
}

template<typename Type>
inline CUDA_CALLABLE void adj_determinant(const mat_t<4,4,Type>& m, mat_t<4,4,Type>& adj_m, Type adj_ret)
{
    // adapted from USD GfMatrix4f::Inverse()
    Type x00, x01, x02, x03;
    Type x10, x11, x12, x13;
    Type x20, x21, x22, x23;
    Type x30, x31, x32, x33;
    double y01, y02, y03, y12, y13, y23;
    Type z00, z10, z20, z30;
    Type z01, z11, z21, z31;
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
    adj_m.data[0][0] += Type(z00*adj_ret);
    adj_m.data[1][0] += Type(z10*adj_ret);
    adj_m.data[0][1] += Type(z01*adj_ret);
    adj_m.data[2][0] += Type(z20*adj_ret);
    adj_m.data[0][2] += Type(z02*adj_ret);
    adj_m.data[3][0] += Type(z30*adj_ret);
    adj_m.data[0][3] += Type(z03*adj_ret);
    adj_m.data[1][1] += Type(z11*adj_ret);
    adj_m.data[2][1] += Type(z21*adj_ret);
    adj_m.data[1][2] += Type(z12*adj_ret);
    adj_m.data[3][1] += Type(z31*adj_ret);
    adj_m.data[1][3] += Type(z13*adj_ret);
    adj_m.data[2][2] += Type(z22*adj_ret);
    adj_m.data[3][2] += Type(z32*adj_ret);
    adj_m.data[2][3] += Type(z23*adj_ret);
    adj_m.data[3][3] += Type(z33*adj_ret);
}

template<unsigned Rows, typename Type>
inline CUDA_CALLABLE void adj_inverse(const mat_t<Rows,Rows,Type>& m, mat_t<Rows,Rows,Type>& ret, mat_t<Rows,Rows,Type>& adj_m, const mat_t<Rows,Rows,Type>& adj_ret)
{
    // todo: how to cache this from the forward pass?
    mat_t<Rows,Rows,Type> invt = transpose(ret);

    // see https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf 2.2.3
    adj_m -= mul(mul(invt, adj_ret), invt);
}

template<typename Type>
inline CUDA_CALLABLE void adj_transform_point(const mat_t<4,4,Type>& m, const vec_t<3,Type>& v, mat_t<4,4,Type>& adj_m, vec_t<3,Type>& adj_v, const vec_t<3,Type>& adj_ret)
{
    vec_t<4,Type> out = vec_t<4,Type>(v[0], v[1], v[2], 1.f);
    adj_m = add(adj_m, transpose(mat_t<4,4,Type>(adj_ret[0] * out, adj_ret[1] * out, adj_ret[2] * out, vec_t<4,Type>())));
    adj_v[0] += dot(vec_t<3,Type>(m.data[0][0], m.data[1][0], m.data[2][0]), adj_ret);
    adj_v[1] += dot(vec_t<3,Type>(m.data[0][1], m.data[1][1], m.data[2][1]), adj_ret);
    adj_v[2] += dot(vec_t<3,Type>(m.data[0][2], m.data[1][2], m.data[2][2]), adj_ret);
}

template<typename Type>
inline CUDA_CALLABLE void adj_transform_vector(const mat_t<4,4,Type>& m, const vec_t<3,Type>& v, mat_t<4,4,Type>& adj_m, vec_t<3,Type>& adj_v, const vec_t<3,Type>& adj_ret)
{
    vec_t<4,Type> out = vec_t<4,Type>(v[0], v[1], v[2], 0.f);
    adj_m = add(adj_m, transpose(mat_t<4,4,Type>(adj_ret[0] * out, adj_ret[1] * out, adj_ret[2] * out, vec_t<4,Type>())));
    adj_v[0] += dot(vec_t<3,Type>(m.data[0][0], m.data[1][0], m.data[2][0]), adj_ret);
    adj_v[1] += dot(vec_t<3,Type>(m.data[0][1], m.data[1][1], m.data[2][1]), adj_ret);
    adj_v[2] += dot(vec_t<3,Type>(m.data[0][2], m.data[1][2], m.data[2][2]), adj_ret);
}

template<typename Type>
inline CUDA_CALLABLE void adj_skew(const vec_t<3,Type>& a, vec_t<3,Type>& adj_a, const mat_t<3,3,Type>& adj_ret)
{
    adj_a[0] += adj_ret.data[2][1] - adj_ret.data[1][2];
    adj_a[1] += adj_ret.data[0][2] - adj_ret.data[2][0];
    adj_a[2] += adj_ret.data[1][0] - adj_ret.data[0][1];
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_cw_mul(const mat_t<Rows,Cols,Type>& a, const mat_t<Rows,Cols,Type>& b, mat_t<Rows,Cols,Type>& adj_a, mat_t<Rows,Cols,Type>& adj_b, const mat_t<Rows,Cols,Type>& adj_ret)
{
  adj_a += cw_mul(b, adj_ret);
  adj_b += cw_mul(a, adj_ret);
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_cw_div(const mat_t<Rows,Cols,Type>& a, const mat_t<Rows,Cols,Type>& b, mat_t<Rows,Cols,Type>& ret, mat_t<Rows,Cols,Type>& adj_a, mat_t<Rows,Cols,Type>& adj_b, const mat_t<Rows,Cols,Type>& adj_ret)
{
  adj_a += cw_div(adj_ret, b);
  adj_b -= cw_mul(adj_ret, cw_div(ret, b));
}

// adjoint for the constant constructor:
template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_mat_t(Type s, Type& adj_s, const mat_t<Rows, Cols, Type>& adj_ret)
{
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            adj_s += adj_ret.data[i][j];
        }
    }
}

// adjoint for the casting constructor:
template<unsigned Rows, unsigned Cols, typename Type, typename OtherType>
inline CUDA_CALLABLE void adj_mat_t(const mat_t<Rows, Cols, OtherType>& other, mat_t<Rows, Cols, OtherType>& adj_other, const mat_t<Rows, Cols, Type>& adj_ret)
{
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            adj_other.data[i][j] += adj_ret.data[i][j];
        }
    }
}

// adjoint for the initializer_array scalar constructor:
template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_mat_t(const initializer_array<Rows * Cols, Type> &cmps, const initializer_array<Rows * Cols, Type*> &adj_cmps, const mat_t<Rows, Cols, Type>& adj_ret)
{
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            *adj_cmps[i * Cols + j] += adj_ret.data[i][j];
        }
    }
}

template<typename Type>
inline CUDA_CALLABLE void adj_mat_t(Type m00, Type m01, Type m10, Type m11, Type& adj_m00, Type& adj_m01, Type& adj_m10, Type& adj_m11, const mat_t<2, 2, Type>& adj_ret)
{
    adj_m00 += adj_ret.data[0][0];
    adj_m01 += adj_ret.data[0][1];
    adj_m10 += adj_ret.data[1][0];
    adj_m11 += adj_ret.data[1][1];
}

template<typename Type>
inline CUDA_CALLABLE void adj_mat_t(Type m00, Type m01, Type m02,
                      Type m10, Type m11, Type m12,
                      Type m20, Type m21, Type m22,
                      Type& a00, Type& a01, Type& a02,
                      Type& a10, Type& a11, Type& a12,
                      Type& a20, Type& a21, Type& a22,
                      const mat_t<3, 3, Type>& adj_ret)
{
    a00 += adj_ret.data[0][0];
    a01 += adj_ret.data[0][1];
    a02 += adj_ret.data[0][2];
    a10 += adj_ret.data[1][0];
    a11 += adj_ret.data[1][1];
    a12 += adj_ret.data[1][2];
    a20 += adj_ret.data[2][0];
    a21 += adj_ret.data[2][1];
    a22 += adj_ret.data[2][2];
}


template<typename Type>
inline CUDA_CALLABLE void adj_mat_t(Type m00, Type m01, Type m02, Type m03,
                      Type m10, Type m11, Type m12, Type m13,
                      Type m20, Type m21, Type m22, Type m23,
                      Type m30, Type m31, Type m32, Type m33,
                      Type& a00, Type& a01, Type& a02, Type& a03,
                      Type& a10, Type& a11, Type& a12, Type& a13,
                      Type& a20, Type& a21, Type& a22, Type& a23,
                      Type& a30, Type& a31, Type& a32, Type& a33,
                      const mat_t<4, 4, Type>& adj_ret)
{
    a00 += adj_ret.data[0][0];
    a01 += adj_ret.data[0][1];
    a02 += adj_ret.data[0][2];
    a03 += adj_ret.data[0][3];

    a10 += adj_ret.data[1][0];
    a11 += adj_ret.data[1][1];
    a12 += adj_ret.data[1][2];
    a13 += adj_ret.data[1][3];

    a20 += adj_ret.data[2][0];
    a21 += adj_ret.data[2][1];
    a22 += adj_ret.data[2][2];
    a23 += adj_ret.data[2][3];

    a30 += adj_ret.data[3][0];
    a31 += adj_ret.data[3][1];
    a32 += adj_ret.data[3][2];
    a33 += adj_ret.data[3][3];
}



// adjoint for the initializer_array vector constructor:
template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_mat_t(const initializer_array<Cols, vec_t<Rows,Type> > &cmps, const initializer_array<Cols, vec_t<Rows,Type>* > &adj_cmps, const mat_t<Rows, Cols, Type>& adj_ret)
{
    for (unsigned j=0; j < Cols; ++j)
    {
        for (unsigned i=0; i < Rows; ++i)
        {
            (*adj_cmps[j])[i] += adj_ret.data[i][j];
        }
    }
}

template<typename Type>
inline CUDA_CALLABLE void adj_mat_t(const vec_t<2,Type> &cmps0, const vec_t<2,Type> &cmps1, vec_t<2,Type> &adj_cmps0, vec_t<2,Type> &adj_cmps1, const mat_t<2, 2, Type>& adj_ret)
{
    for (unsigned i=0; i < 2; ++i)
    {
        adj_cmps0[i] += adj_ret.data[i][0];
        adj_cmps1[i] += adj_ret.data[i][1];
    }
}

template<typename Type>
inline CUDA_CALLABLE void adj_mat_t(const vec_t<3,Type> &cmps0, const vec_t<3,Type> &cmps1, const vec_t<3,Type> &cmps2, vec_t<3,Type> &adj_cmps0, vec_t<3,Type> &adj_cmps1, vec_t<3,Type> &adj_cmps2, const mat_t<3, 3, Type>& adj_ret)
{
    for (unsigned i=0; i < 3; ++i)
    {
        adj_cmps0[i] += adj_ret.data[i][0];
        adj_cmps1[i] += adj_ret.data[i][1];
        adj_cmps2[i] += adj_ret.data[i][2];
    }
}

template<typename Type>
inline CUDA_CALLABLE void adj_mat_t(const vec_t<4,Type> &cmps0, const vec_t<4,Type> &cmps1, const vec_t<4,Type> &cmps2, const vec_t<4,Type> &cmps3, vec_t<4,Type> &adj_cmps0, vec_t<4,Type> &adj_cmps1, vec_t<4,Type> &adj_cmps2, vec_t<4,Type> &adj_cmps3, const mat_t<4, 4, Type>& adj_ret)
{
    for (unsigned i=0; i < 4; ++i)
    {
        adj_cmps0[i] += adj_ret.data[i][0];
        adj_cmps1[i] += adj_ret.data[i][1];
        adj_cmps2[i] += adj_ret.data[i][2];
        adj_cmps3[i] += adj_ret.data[i][3];
    }
}

template<typename Type>
inline CUDA_CALLABLE void adj_matrix_from_cols(
    const vec_t<2, Type>& c0, const vec_t<2, Type>& c1,
    vec_t<2, Type>& adj_c0, vec_t<2, Type>& adj_c1,
    const mat_t<2, 2, Type>& adj_ret
)
{
    for (unsigned i=0; i < 2; ++i)
    {
        adj_c0[i] += adj_ret.data[i][0];
        adj_c1[i] += adj_ret.data[i][1];
    }
}

template<typename Type>
inline CUDA_CALLABLE void adj_matrix_from_cols(
    const vec_t<3, Type>& c0, const vec_t<3, Type>& c1, const vec_t<3, Type>& c2,
    vec_t<3, Type>& adj_c0, vec_t<3, Type>& adj_c1, vec_t<3, Type>& adj_c2,
    const mat_t<3, 3, Type>& adj_ret
)
{
    for (unsigned i=0; i < 3; ++i)
    {
        adj_c0[i] += adj_ret.data[i][0];
        adj_c1[i] += adj_ret.data[i][1];
        adj_c2[i] += adj_ret.data[i][2];
    }
}

template<typename Type>
inline CUDA_CALLABLE void adj_matrix_from_cols(
    const vec_t<4, Type>& c0, const vec_t<4, Type>& c1, const vec_t<4, Type>& c2, const vec_t<4, Type>& c3,
    vec_t<4, Type>& adj_c0, vec_t<4, Type>& adj_c1, vec_t<4, Type>& adj_c2, vec_t<4, Type>& adj_c3,
    const mat_t<4, 4, Type>& adj_ret
)
{
    for (unsigned i=0; i < 4; ++i)
    {
        adj_c0[i] += adj_ret.data[i][0];
        adj_c1[i] += adj_ret.data[i][1];
        adj_c2[i] += adj_ret.data[i][2];
        adj_c3[i] += adj_ret.data[i][3];
    }
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_matrix_from_cols(
    const initializer_array<Cols, vec_t<Rows, Type> >& l,
    const initializer_array<Cols, vec_t<Rows, Type>* >& adj_l,
    const mat_t<Rows, Cols, Type>& adj_ret
)
{
    for (unsigned j=0; j < Cols; ++j)
    {
        for (unsigned i=0; i < Rows; ++i)
        {
            (*adj_l[j])[i] += adj_ret.data[i][j];
        }
    }
}

template<typename Type>
inline CUDA_CALLABLE void adj_matrix_from_rows(
    const vec_t<2, Type>& r0, const vec_t<2, Type>& r1,
    vec_t<2, Type>& adj_r0, vec_t<2, Type>& adj_r1,
    const mat_t<2, 2, Type>& adj_ret
)
{
    for (unsigned j=0; j < 2; ++j)
    {
        adj_r0[j] += adj_ret.data[0][j];
        adj_r1[j] += adj_ret.data[1][j];
    }
}

template<typename Type>
inline CUDA_CALLABLE void adj_matrix_from_rows(
    const vec_t<3, Type>& c0, const vec_t<3, Type>& c1, const vec_t<3, Type>& c2,
    vec_t<3, Type>& adj_c0, vec_t<3, Type>& adj_c1, vec_t<3, Type>& adj_c2,
    const mat_t<3, 3, Type>& adj_ret
)
{
    for (unsigned j=0; j < 3; ++j)
    {
        adj_c0[j] += adj_ret.data[0][j];
        adj_c1[j] += adj_ret.data[1][j];
        adj_c2[j] += adj_ret.data[2][j];
    }
}

template<typename Type>
inline CUDA_CALLABLE void adj_matrix_from_rows(
    const vec_t<4, Type>& c0, const vec_t<4, Type>& c1, const vec_t<4, Type>& c2, const vec_t<4, Type>& c3,
    vec_t<4, Type>& adj_c0, vec_t<4, Type>& adj_c1, vec_t<4, Type>& adj_c2, vec_t<4, Type>& adj_c3,
    const mat_t<4, 4, Type>& adj_ret
)
{
    for (unsigned j=0; j < 4; ++j)
    {
        adj_c0[j] += adj_ret.data[0][j];
        adj_c1[j] += adj_ret.data[1][j];
        adj_c2[j] += adj_ret.data[2][j];
        adj_c3[j] += adj_ret.data[3][j];
    }
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_matrix_from_rows(
    const initializer_array<Rows, vec_t<Cols, Type> >& l,
    const initializer_array<Rows, vec_t<Cols, Type>* >& adj_l,
    const mat_t<Rows, Cols, Type>& adj_ret
)
{
    for (unsigned i=0; i < Rows; ++i)
    {
        for (unsigned j=0; j < Cols; ++j)
        {
            (*adj_l[i])[j] += adj_ret.data[i][j];
        }
    }
}

template<unsigned Rows, unsigned Cols, typename Type>
CUDA_CALLABLE inline mat_t<Rows, Cols, Type> lerp(const mat_t<Rows, Cols, Type>& a, const mat_t<Rows, Cols, Type>& b, Type t)
{
    return a*(Type(1)-t) + b*t;
}

template<unsigned Rows, unsigned Cols, typename Type>
CUDA_CALLABLE inline void adj_lerp(const mat_t<Rows, Cols, Type>& a, const mat_t<Rows, Cols, Type>& b, Type t, mat_t<Rows, Cols, Type>& adj_a, mat_t<Rows, Cols, Type>& adj_b, Type& adj_t, const mat_t<Rows, Cols, Type>& adj_ret)
{
    adj_a += adj_ret*(Type(1)-t);
    adj_b += adj_ret*t;
    adj_t += tensordot(b, adj_ret) - tensordot(a, adj_ret);
}

// for integral types we do not accumulate gradients
template<unsigned Rows, unsigned Cols> CUDA_CALLABLE inline void adj_atomic_add(mat_t<Rows, Cols, int8>* buf, const mat_t<Rows, Cols, int8> &value) { }
template<unsigned Rows, unsigned Cols> CUDA_CALLABLE inline void adj_atomic_add(mat_t<Rows, Cols, uint8>* buf, const mat_t<Rows, Cols, uint8> &value) { }
template<unsigned Rows, unsigned Cols> CUDA_CALLABLE inline void adj_atomic_add(mat_t<Rows, Cols, int16>* buf, const mat_t<Rows, Cols, int16> &value) { }
template<unsigned Rows, unsigned Cols> CUDA_CALLABLE inline void adj_atomic_add(mat_t<Rows, Cols, uint16>* buf, const mat_t<Rows, Cols, uint16> &value) { }
template<unsigned Rows, unsigned Cols> CUDA_CALLABLE inline void adj_atomic_add(mat_t<Rows, Cols, int32>* buf, const mat_t<Rows, Cols, int32> &value) { }
template<unsigned Rows, unsigned Cols> CUDA_CALLABLE inline void adj_atomic_add(mat_t<Rows, Cols, uint32>* buf, const mat_t<Rows, Cols, uint32> &value) { }
template<unsigned Rows, unsigned Cols> CUDA_CALLABLE inline void adj_atomic_add(mat_t<Rows, Cols, int64>* buf, const mat_t<Rows, Cols, int64> &value) { }
template<unsigned Rows, unsigned Cols> CUDA_CALLABLE inline void adj_atomic_add(mat_t<Rows, Cols, uint64>* buf, const mat_t<Rows, Cols, uint64> &value) { }

using mat22h = mat_t<2,2,half>;
using mat33h = mat_t<3,3,half>;
using mat44h = mat_t<4,4,half>;

using mat22 = mat_t<2,2,float>;
using mat33 = mat_t<3,3,float>;
using mat44 = mat_t<4,4,float>;

using mat22f = mat_t<2,2,float>;
using mat33f = mat_t<3,3,float>;
using mat44f = mat_t<4,4,float>;

using mat22d = mat_t<2,2,double>;
using mat33d = mat_t<3,3,double>;
using mat44d = mat_t<4,4,double>;

inline CUDA_CALLABLE void adj_mat22(vec2 c0, vec2 c1,
                      vec2& a0, vec2& a1,
                      const mat22& adj_ret)
{
    a0 += adj_ret.get_col(0);
    a1 += adj_ret.get_col(1);
}

inline CUDA_CALLABLE void adj_mat22(float m00, float m01, float m10, float m11, float& adj_m00, float& adj_m01, float& adj_m10, float& adj_m11, const mat22& adj_ret)
{
    adj_m00 += adj_ret.data[0][0];
    adj_m01 += adj_ret.data[0][1];
    adj_m10 += adj_ret.data[1][0];
    adj_m11 += adj_ret.data[1][1];
}

inline CUDA_CALLABLE void adj_mat33(vec3 c0, vec3 c1, vec3 c2,
                      vec3& a0, vec3& a1, vec3& a2,
                      const mat33& adj_ret)
{
    // column constructor
    a0 += adj_ret.get_col(0);
    a1 += adj_ret.get_col(1);
    a2 += adj_ret.get_col(2);

}

inline CUDA_CALLABLE void adj_mat33(float m00, float m01, float m02,
                      float m10, float m11, float m12,
                      float m20, float m21, float m22,
                      float& a00, float& a01, float& a02,
                      float& a10, float& a11, float& a12,
                      float& a20, float& a21, float& a22,
                      const mat33& adj_ret)
{
    a00 += adj_ret.data[0][0];
    a01 += adj_ret.data[0][1];
    a02 += adj_ret.data[0][2];
    a10 += adj_ret.data[1][0];
    a11 += adj_ret.data[1][1];
    a12 += adj_ret.data[1][2];
    a20 += adj_ret.data[2][0];
    a21 += adj_ret.data[2][1];
    a22 += adj_ret.data[2][2];
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
    a00 += adj_ret.data[0][0];
    a01 += adj_ret.data[0][1];
    a02 += adj_ret.data[0][2];
    a03 += adj_ret.data[0][3];

    a10 += adj_ret.data[1][0];
    a11 += adj_ret.data[1][1];
    a12 += adj_ret.data[1][2];
    a13 += adj_ret.data[1][3];

    a20 += adj_ret.data[2][0];
    a21 += adj_ret.data[2][1];
    a22 += adj_ret.data[2][2];
    a23 += adj_ret.data[2][3];

    a30 += adj_ret.data[3][0];
    a31 += adj_ret.data[3][1];
    a32 += adj_ret.data[3][2];
    a33 += adj_ret.data[3][3];
}

template<unsigned Rows, unsigned Cols, typename Type>
CUDA_CALLABLE inline int len(const mat_t<Rows,Cols,Type>& x)
{
    return Rows;
}

template<unsigned Rows, unsigned Cols, typename Type>
CUDA_CALLABLE inline void adj_len(const mat_t<Rows,Cols,Type>& x, mat_t<Rows,Cols,Type>& adj_x, const int& adj_ret)
{
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void expect_near(const mat_t<Rows,Cols,Type>& actual, const mat_t<Rows,Cols,Type>& expected, const Type& tolerance)
{
    Type diff(0);
    for (unsigned i = 0; i < Rows; ++i)
    {
        for (unsigned j = 0; j < Cols; ++j)
        {
            diff = max(diff, abs(actual.data[i][j] - expected.data[i][j]));
        }
    }
    if (diff > tolerance)
    {
        printf("Error, expect_near() failed with tolerance "); print(tolerance);
        printf("    Expected: "); print(expected);
        printf("    Actual: "); print(actual);
        printf("    Max absolute difference: "); print(diff);
    }
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_expect_near(const mat_t<Rows,Cols,Type>& actual, const mat_t<Rows,Cols,Type>& expected, Type tolerance, mat_t<Rows,Cols,Type>& adj_actual, mat_t<Rows,Cols,Type>& adj_expected, Type adj_tolerance)
{
    // nop
}

} // namespace wp
