#include "warp.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace
{

// Specialized is_zero and accumulation function for common block sizes
// Rely on compiler to unroll loops when block size is known

template <int N, typename T> bool bsr_fixed_block_is_zero(const T* val, int value_size)
{
    return std::all_of(val, val + N, [](float v) { return v == T(0); });
}

template <typename T> bool bsr_dyn_block_is_zero(const T* val, int value_size)
{
    return std::all_of(val, val + value_size, [](float v) { return v == T(0); });
}

template <int N, typename T> void bsr_fixed_block_accumulate(const T* val, T* sum, int value_size)
{
    for (int i = 0; i < N; ++i, ++val, ++sum)
    {
        *sum += *val;
    }
}

template <typename T> void bsr_dyn_block_accumulate(const T* val, T* sum, int value_size)
{
    for (int i = 0; i < value_size; ++i, ++val, ++sum)
    {
        *sum += *val;
    }
}

template <int Rows, int Cols, typename T>
void bsr_fixed_block_transpose(const T* src, T* dest, int row_count, int col_count)
{
    for (int r = 0; r < Rows; ++r)
    {
        for (int c = 0; c < Cols; ++c)
        {
            dest[c * Rows + r] = src[r * Cols + c];
        }
    }
}

template <typename T> void bsr_dyn_block_transpose(const T* src, T* dest, int row_count, int col_count)
{
    for (int r = 0; r < row_count; ++r)
    {
        for (int c = 0; c < col_count; ++c)
        {
            dest[c * row_count + r] = src[r * col_count + c];
        }
    }
}

} // namespace

template <typename T>
int bsr_matrix_from_triplets_host(const int rows_per_block, const int cols_per_block, const int row_count,
                                  const int nnz, const int* tpl_rows, const int* tpl_columns, const T* tpl_values,
                                  const bool prune_numerical_zeros, int* bsr_offsets, int* bsr_columns, T* bsr_values)
{

    // get specialized accumulator for common block sizes (1,1), (1,2), (1,3),
    // (2,2), (2,3), (3,3)
    const int block_size = rows_per_block * cols_per_block;
    void (*block_accumulate_func)(const T*, T*, int);
    bool (*block_is_zero_func)(const T*, int);
    switch (block_size)
    {
    case 1:
        block_accumulate_func = bsr_fixed_block_accumulate<1, T>;
        block_is_zero_func = bsr_fixed_block_is_zero<1, T>;
        break;
    case 2:
        block_accumulate_func = bsr_fixed_block_accumulate<2, T>;
        block_is_zero_func = bsr_fixed_block_is_zero<2, T>;
        break;
    case 3:
        block_accumulate_func = bsr_fixed_block_accumulate<3, T>;
        block_is_zero_func = bsr_fixed_block_is_zero<3, T>;
        break;
    case 4:
        block_accumulate_func = bsr_fixed_block_accumulate<4, T>;
        block_is_zero_func = bsr_fixed_block_is_zero<4, T>;
        break;
    case 6:
        block_accumulate_func = bsr_fixed_block_accumulate<6, T>;
        block_is_zero_func = bsr_fixed_block_is_zero<6, T>;
        break;
    case 9:
        block_accumulate_func = bsr_fixed_block_accumulate<9, T>;
        block_is_zero_func = bsr_fixed_block_is_zero<9, T>;
        break;
    default:
        block_accumulate_func = bsr_dyn_block_accumulate<T>;
        block_is_zero_func = bsr_dyn_block_is_zero<T>;
    }

    std::vector<int> block_indices(nnz);
    std::iota(block_indices.begin(), block_indices.end(), 0);

    // remove zero blocks  and invalid row indices
    block_indices.erase(std::remove_if(block_indices.begin(), block_indices.end(),
                                       [&](int i)
                                       {
                                           return tpl_rows[i] < 0 || tpl_rows[i] >= row_count ||
                                                  (prune_numerical_zeros && tpl_values &&
                                                   block_is_zero_func(tpl_values + i * block_size, block_size));
                                       }),
                        block_indices.end());

    // sort block indices according to lexico order
    std::sort(block_indices.begin(), block_indices.end(), [tpl_rows, tpl_columns](int i, int j) -> bool
              { return tpl_rows[i] < tpl_rows[j] || (tpl_rows[i] == tpl_rows[j] && tpl_columns[i] < tpl_columns[j]); });

    // accumulate blocks at same locations, count blocks per row
    std::fill_n(bsr_offsets, row_count + 1, 0);

    int current_row = -1;
    int current_col = -1;

    // so that we get back to the start for the first block
    if (bsr_values)
    {
        bsr_values -= block_size;
    }

    for (int i = 0; i < block_indices.size(); ++i)
    {
        int idx = block_indices[i];
        int row = tpl_rows[idx];
        int col = tpl_columns[idx];
        const T* val = tpl_values + idx * block_size;

        if (row == current_row && col == current_col)
        {
            if (bsr_values)
            {
                block_accumulate_func(val, bsr_values, block_size);
            }
        }
        else
        {
            *(bsr_columns++) = col;

            if (bsr_values)
            {
                bsr_values += block_size;
                std::copy_n(val, block_size, bsr_values);
            }

            bsr_offsets[row + 1]++;

            current_row = row;
            current_col = col;
        }
    }

    // build postfix sum of row counts
    std::partial_sum(bsr_offsets, bsr_offsets + row_count + 1, bsr_offsets);

    return bsr_offsets[row_count];
}

template <typename T>
void bsr_transpose_host(int rows_per_block, int cols_per_block, int row_count, int col_count, int nnz_up,
                        const int* bsr_offsets, const int* bsr_columns, const T* bsr_values,
                        int* transposed_bsr_offsets, int* transposed_bsr_columns, T* transposed_bsr_values)
{
    const int nnz = bsr_offsets[row_count];
    const int block_size = rows_per_block * cols_per_block;

    void (*block_transpose_func)(const T*, T*, int, int) = bsr_dyn_block_transpose<T>;
    switch (rows_per_block)
    {
    case 1:
        switch (cols_per_block)
        {
        case 1:
            block_transpose_func = bsr_fixed_block_transpose<1, 1, T>;
            break;
        case 2:
            block_transpose_func = bsr_fixed_block_transpose<1, 2, T>;
            break;
        case 3:
            block_transpose_func = bsr_fixed_block_transpose<1, 3, T>;
            break;
        }
        break;
    case 2:
        switch (cols_per_block)
        {
        case 1:
            block_transpose_func = bsr_fixed_block_transpose<2, 1, T>;
            break;
        case 2:
            block_transpose_func = bsr_fixed_block_transpose<2, 2, T>;
            break;
        case 3:
            block_transpose_func = bsr_fixed_block_transpose<2, 3, T>;
            break;
        }
        break;
    case 3:
        switch (cols_per_block)
        {
        case 1:
            block_transpose_func = bsr_fixed_block_transpose<3, 1, T>;
            break;
        case 2:
            block_transpose_func = bsr_fixed_block_transpose<3, 2, T>;
            break;
        case 3:
            block_transpose_func = bsr_fixed_block_transpose<3, 3, T>;
            break;
        }
        break;
    }

    std::vector<int> block_indices(nnz), bsr_rows(nnz);
    std::iota(block_indices.begin(), block_indices.end(), 0);

    // Fill row indices from offsets
    for (int row = 0; row < row_count; ++row)
    {
        std::fill(bsr_rows.begin() + bsr_offsets[row], bsr_rows.begin() + bsr_offsets[row + 1], row);
    }

    // sort block indices according to (transposed) lexico order
    std::sort(
        block_indices.begin(), block_indices.end(), [&bsr_rows, bsr_columns](int i, int j) -> bool
        { return bsr_columns[i] < bsr_columns[j] || (bsr_columns[i] == bsr_columns[j] && bsr_rows[i] < bsr_rows[j]); });

    // Count blocks per column and transpose blocks
    std::fill_n(transposed_bsr_offsets, col_count + 1, 0);

    for (int i = 0; i < nnz; ++i)
    {
        int idx = block_indices[i];
        int row = bsr_rows[idx];
        int col = bsr_columns[idx];

        ++transposed_bsr_offsets[col + 1];
        transposed_bsr_columns[i] = row;

        if (transposed_bsr_values != nullptr)
        {
            const T* src_block = bsr_values + idx * block_size;
            T* dst_block = transposed_bsr_values + i * block_size;
            block_transpose_func(src_block, dst_block, rows_per_block, cols_per_block);
        }
    }

    // build postfix sum of column counts
    std::partial_sum(transposed_bsr_offsets, transposed_bsr_offsets + col_count + 1, transposed_bsr_offsets);
}

WP_API void bsr_matrix_from_triplets_float_host(int rows_per_block, int cols_per_block, int row_count, int nnz,
                                                int* tpl_rows, int* tpl_columns, void* tpl_values,
                                                bool prune_numerical_zeros, int* bsr_offsets, int* bsr_columns,
                                                void* bsr_values, int* bsr_nnz, void* bsr_nnz_event)
{
    bsr_matrix_from_triplets_host<float>(rows_per_block, cols_per_block, row_count, nnz, tpl_rows, tpl_columns,
                                         static_cast<const float*>(tpl_values), prune_numerical_zeros, bsr_offsets,
                                         bsr_columns, static_cast<float*>(bsr_values));
    if (bsr_nnz)
    {
        *bsr_nnz = bsr_offsets[row_count];
    }
}

WP_API void bsr_matrix_from_triplets_double_host(int rows_per_block, int cols_per_block, int row_count, int nnz,
                                                 int* tpl_rows, int* tpl_columns, void* tpl_values,
                                                 bool prune_numerical_zeros, int* bsr_offsets, int* bsr_columns,
                                                 void* bsr_values, int* bsr_nnz, void* bsr_nnz_event)
{
    bsr_matrix_from_triplets_host<double>(rows_per_block, cols_per_block, row_count, nnz, tpl_rows, tpl_columns,
                                          static_cast<const double*>(tpl_values), prune_numerical_zeros, bsr_offsets,
                                          bsr_columns, static_cast<double*>(bsr_values));
    if (bsr_nnz)
    {
        *bsr_nnz = bsr_offsets[row_count];
    }
}

WP_API void bsr_transpose_float_host(int rows_per_block, int cols_per_block, int row_count, int col_count, int nnz,
                                     int* bsr_offsets, int* bsr_columns, void* bsr_values, int* transposed_bsr_offsets,
                                     int* transposed_bsr_columns, void* transposed_bsr_values)
{
    bsr_transpose_host(rows_per_block, cols_per_block, row_count, col_count, nnz, bsr_offsets, bsr_columns,
                       static_cast<const float*>(bsr_values), transposed_bsr_offsets, transposed_bsr_columns,
                       static_cast<float*>(transposed_bsr_values));
}

WP_API void bsr_transpose_double_host(int rows_per_block, int cols_per_block, int row_count, int col_count, int nnz,
                                      int* bsr_offsets, int* bsr_columns, void* bsr_values, int* transposed_bsr_offsets,
                                      int* transposed_bsr_columns, void* transposed_bsr_values)
{
    bsr_transpose_host(rows_per_block, cols_per_block, row_count, col_count, nnz, bsr_offsets, bsr_columns,
                       static_cast<const double*>(bsr_values), transposed_bsr_offsets, transposed_bsr_columns,
                       static_cast<double*>(transposed_bsr_values));
}

#if !WP_ENABLE_CUDA
WP_API void bsr_matrix_from_triplets_float_device(int rows_per_block, int cols_per_block, int row_count, int nnz,
                                                   int* tpl_rows, int* tpl_columns, void* tpl_values,
                                                   bool prune_numerical_zeros, int* bsr_offsets, int* bsr_columns,
                                                   void* bsr_values, int* bsr_nnz, void* bsr_nnz_event)
{
}

WP_API void bsr_matrix_from_triplets_double_device(int rows_per_block, int cols_per_block, int row_count, int nnz,
                                                   int* tpl_rows, int* tpl_columns, void* tpl_values,
                                                   bool prune_numerical_zeros, int* bsr_offsets, int* bsr_columns,
                                                   void* bsr_values, int* bsr_nnz, void* bsr_nnz_event)
{
}

WP_API void bsr_transpose_float_device(int rows_per_block, int cols_per_block, int row_count, int col_count, int nnz,
                                       int* bsr_offsets, int* bsr_columns, void* bsr_values,
                                       int* transposed_bsr_offsets, int* transposed_bsr_columns,
                                       void* transposed_bsr_values)
{
}

WP_API void bsr_transpose_double_device(int rows_per_block, int cols_per_block, int row_count, int col_count, int nnz,
                                        int* bsr_offsets, int* bsr_columns, void* bsr_values,
                                        int* transposed_bsr_offsets, int* transposed_bsr_columns,
                                        void* transposed_bsr_values)
{
}

#endif
