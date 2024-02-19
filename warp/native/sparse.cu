#include "cuda_util.h"
#include "warp.h"

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>

namespace {

// Combined row+column value that can be radix-sorted with CUB
using BsrRowCol = uint64_t;

CUDA_CALLABLE BsrRowCol bsr_combine_row_col(uint32_t row, uint32_t col) {
  return (static_cast<uint64_t>(row) << 32) | col;
}

CUDA_CALLABLE uint32_t bsr_get_row(const BsrRowCol &row_col) {
  return row_col >> 32;
}

CUDA_CALLABLE uint32_t bsr_get_col(const BsrRowCol &row_col) {
  return row_col & INT_MAX;
}

// Cached temporary storage
struct BsrFromTripletsTemp {
  
  int *count_buffer = NULL;
  cudaEvent_t host_sync_event = NULL;

  BsrFromTripletsTemp()
    : count_buffer(static_cast<int*>(alloc_pinned(sizeof(int))))
  {
    cudaEventCreateWithFlags(&host_sync_event, cudaEventDisableTiming);
  }
  
  ~BsrFromTripletsTemp()
  {
    cudaEventDestroy(host_sync_event);
    free_pinned(count_buffer);
  }

  BsrFromTripletsTemp(const BsrFromTripletsTemp&) = delete;
  BsrFromTripletsTemp& operator=(const BsrFromTripletsTemp&) = delete;

};

// map temp buffers to CUDA contexts
static std::unordered_map<void *, BsrFromTripletsTemp> g_bsr_from_triplets_temp_map;

template <typename T> struct BsrBlockIsNotZero {
  int block_size;
  const T *values;

  CUDA_CALLABLE_DEVICE bool operator()(int i) const {
    const T *val = values + i * block_size;
    for (int i = 0; i < block_size; ++i, ++val) {
      if (*val != T(0))
        return true;
    }
    return false;
  }
};

__global__ void bsr_fill_block_indices(int nnz, int *block_indices) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nnz)
    return;

  block_indices[i] = i;
}

__global__ void bsr_fill_row_col(const int *nnz, const int *block_indices,
                                 const int *tpl_rows, const int *tpl_columns,
                                 BsrRowCol *tpl_row_col) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= *nnz)
    return;

  const int block = block_indices[i];

  BsrRowCol row_col = bsr_combine_row_col(tpl_rows[block], tpl_columns[block]);
  tpl_row_col[i] = row_col;
}

template <typename T>
__global__ void
bsr_merge_blocks(int nnz, int block_size, const int *block_offsets,
                 const int *sorted_block_indices,
                 const BsrRowCol *unique_row_cols, const T *tpl_values,
                 int *bsr_row_counts, int *bsr_cols, T *bsr_values)

{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nnz)
    return;

  const int beg = i ? block_offsets[i - 1] : 0;
  const int end = block_offsets[i];

  BsrRowCol row_col = unique_row_cols[i];

  bsr_cols[i] = bsr_get_col(row_col);
  atomicAdd(bsr_row_counts + bsr_get_row(row_col) + 1, 1);

  if (bsr_values == nullptr)
    return;

  T *bsr_val = bsr_values + i * block_size;
  const T *tpl_val = tpl_values + sorted_block_indices[beg] * block_size;

  for (int k = 0; k < block_size; ++k) {
    bsr_val[k] = tpl_val[k];
  }

  for (int cur = beg + 1; cur != end; ++cur) {
    const T *tpl_val = tpl_values + sorted_block_indices[cur] * block_size;
    for (int k = 0; k < block_size; ++k) {
      bsr_val[k] += tpl_val[k];
    }
  }
}

template <typename T>
int bsr_matrix_from_triplets_device(const int rows_per_block,
                                    const int cols_per_block,
                                    const int row_count, const int nnz,
                                    const int *tpl_rows, const int *tpl_columns,
                                    const T *tpl_values, int *bsr_offsets,
                                    int *bsr_columns, T *bsr_values) {
  const int block_size = rows_per_block * cols_per_block;

  void *context = cuda_context_get_current();
  ContextGuard guard(context);

  // Per-context cached temporary buffers
  BsrFromTripletsTemp &bsr_temp = g_bsr_from_triplets_temp_map[context];

  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream_get_current());

  ScopedTemporary<int> block_indices(context, 2*nnz);
  ScopedTemporary<BsrRowCol> combined_row_col(context, 2*nnz);

  cub::DoubleBuffer<int> d_keys(block_indices.buffer(),
                                block_indices.buffer() + nnz);
  cub::DoubleBuffer<BsrRowCol> d_values(combined_row_col.buffer(),
                                        combined_row_col.buffer() + nnz);

  int *p_nz_triplet_count = bsr_temp.count_buffer;

  wp_launch_device(WP_CURRENT_CONTEXT, bsr_fill_block_indices, nnz,
                   (nnz, d_keys.Current()));

  if (tpl_values) {

    // Remove zero blocks
    {
      size_t buff_size = 0;
      BsrBlockIsNotZero<T> isNotZero{block_size, tpl_values};
      check_cuda(cub::DeviceSelect::If(nullptr, buff_size, d_keys.Current(),
                                      d_keys.Alternate(), p_nz_triplet_count,
                                      nnz, isNotZero, stream));
      ScopedTemporary<> temp(context, buff_size);
      check_cuda(cub::DeviceSelect::If(
          temp.buffer(), buff_size, d_keys.Current(), d_keys.Alternate(),
          p_nz_triplet_count, nnz, isNotZero, stream));
    }
    cudaEventRecord(bsr_temp.host_sync_event, stream);

    // switch current/alternate in double buffer
    d_keys.selector ^= 1;

  } else {
    *p_nz_triplet_count = nnz;
  }

  // Combine rows and columns so we can sort on them both
  wp_launch_device(WP_CURRENT_CONTEXT, bsr_fill_row_col, nnz,
                   (p_nz_triplet_count, d_keys.Current(), tpl_rows, tpl_columns,
                    d_values.Current()));

  if (tpl_values) {
    // Make sure count is available on host
    cudaEventSynchronize(bsr_temp.host_sync_event);
  }

  const int nz_triplet_count = *p_nz_triplet_count;

  // Sort
  {
    size_t buff_size = 0;
    check_cuda(cub::DeviceRadixSort::SortPairs(
        nullptr, buff_size, d_values, d_keys, nz_triplet_count, 0, 64, stream));
    ScopedTemporary<> temp(context, buff_size);
    check_cuda(cub::DeviceRadixSort::SortPairs(temp.buffer(), buff_size,
                                              d_values, d_keys, nz_triplet_count,
                                              0, 64, stream));
  }

  // Runlength encode row-col sequences
  {
    size_t buff_size = 0;
    check_cuda(cub::DeviceRunLengthEncode::Encode(
        nullptr, buff_size, d_values.Current(), d_values.Alternate(),
        d_keys.Alternate(), p_nz_triplet_count, nz_triplet_count, stream));
    ScopedTemporary<> temp(context, buff_size);
    check_cuda(cub::DeviceRunLengthEncode::Encode(
        temp.buffer(), buff_size, d_values.Current(), d_values.Alternate(),
        d_keys.Alternate(), p_nz_triplet_count, nz_triplet_count, stream));
  }

  cudaEventRecord(bsr_temp.host_sync_event, stream);

  // Now we have the following:
  // d_values.Current(): sorted block row-col
  // d_values.Alternate(): sorted unique row-col
  // d_keys.Current(): sorted block indices
  // d_keys.Alternate(): repeated block-row count

  // Scan repeated block counts
  {
    size_t buff_size = 0;
    check_cuda(cub::DeviceScan::InclusiveSum(
        nullptr, buff_size, d_keys.Alternate(), d_keys.Alternate(),
        nz_triplet_count, stream));
    ScopedTemporary<> temp(context, buff_size);
    check_cuda(cub::DeviceScan::InclusiveSum(
        temp.buffer(), buff_size, d_keys.Alternate(), d_keys.Alternate(),
        nz_triplet_count, stream));
  }

  // While we're at it, zero the bsr offsets buffer
  memset_device(WP_CURRENT_CONTEXT, bsr_offsets, 0,
                (row_count + 1) * sizeof(int));

  // Wait for number of compressed blocks
  cudaEventSynchronize(bsr_temp.host_sync_event);
  const int compressed_nnz = *p_nz_triplet_count;

  // We have all we need to accumulate our repeated blocks
  wp_launch_device(WP_CURRENT_CONTEXT, bsr_merge_blocks, compressed_nnz,
                   (compressed_nnz, block_size, d_keys.Alternate(),
                    d_keys.Current(), d_values.Alternate(), tpl_values,
                    bsr_offsets, bsr_columns, bsr_values));

  // Last, prefix sum the row block counts
  {
    size_t buff_size = 0;
    check_cuda(cub::DeviceScan::InclusiveSum(nullptr, buff_size, bsr_offsets,
                                            bsr_offsets, row_count + 1, stream));
    ScopedTemporary<> temp(context, buff_size);
    check_cuda(cub::DeviceScan::InclusiveSum(temp.buffer(), buff_size,
                                            bsr_offsets, bsr_offsets,
                                            row_count + 1, stream));
  }

  return compressed_nnz;
}

__global__ void bsr_transpose_fill_row_col(const int nnz, const int row_count,
                                           const int *bsr_offsets,
                                           const int *bsr_columns,
                                           int *block_indices,
                                           BsrRowCol *transposed_row_col,
                                           int *transposed_bsr_offsets) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nnz)
    return;

  block_indices[i] = i;

  // Binary search for row
  int lower = 0;
  int upper = row_count - 1;

  while (lower < upper) {
    int mid = lower + (upper - lower) / 2;

    if (bsr_offsets[mid + 1] <= i) {
      lower = mid + 1;
    } else {
      upper = mid;
    }
  }

  const int row = lower;
  const int col = bsr_columns[i];
  BsrRowCol row_col = bsr_combine_row_col(col, row);
  transposed_row_col[i] = row_col;

  atomicAdd(transposed_bsr_offsets + col + 1, 1);
}

template <int Rows, int Cols, typename T> struct BsrBlockTransposer {
  void CUDA_CALLABLE_DEVICE operator()(const T *src, T *dest) const {
    for (int r = 0; r < Rows; ++r) {
      for (int c = 0; c < Cols; ++c) {
        dest[c * Rows + r] = src[r * Cols + c];
      }
    }
  }
};

template <typename T> struct BsrBlockTransposer<-1, -1, T> {

  int row_count;
  int col_count;

  void CUDA_CALLABLE_DEVICE operator()(const T *src, T *dest) const {
    for (int r = 0; r < row_count; ++r) {
      for (int c = 0; c < col_count; ++c) {
        dest[c * row_count + r] = src[r * col_count + c];
      }
    }
  }
};

template <int Rows, int Cols, typename T>
__global__ void
bsr_transpose_blocks(const int nnz, const int block_size,
                     BsrBlockTransposer<Rows, Cols, T> transposer,
                     const int *block_indices,
                     const BsrRowCol *transposed_indices, const T *bsr_values,
                     int *transposed_bsr_columns, T *transposed_bsr_values) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nnz)
    return;

  const int src_idx = block_indices[i];

  transposer(bsr_values + src_idx * block_size,
             transposed_bsr_values + i * block_size);

  transposed_bsr_columns[i] = bsr_get_col(transposed_indices[i]);
}

template <typename T>
void
launch_bsr_transpose_blocks(const int nnz, const int block_size,
                     const int rows_per_block, const int cols_per_block,
                     const int *block_indices,
                     const BsrRowCol *transposed_indices, 
                     const T *bsr_values,
                     int *transposed_bsr_columns, T *transposed_bsr_values) {

  switch (rows_per_block) {
  case 1:
    switch (cols_per_block) {
    case 1:
      wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_blocks, nnz,
                       (nnz, block_size, BsrBlockTransposer<1, 1, T>{},
                        block_indices, transposed_indices, bsr_values,
                        transposed_bsr_columns, transposed_bsr_values));
      return;
    case 2:
      wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_blocks, nnz,
                       (nnz, block_size, BsrBlockTransposer<1, 2, T>{},
                        block_indices, transposed_indices, bsr_values,
                        transposed_bsr_columns, transposed_bsr_values));
      return;
    case 3:
      wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_blocks, nnz,
                       (nnz, block_size, BsrBlockTransposer<1, 3, T>{},
                        block_indices, transposed_indices, bsr_values,
                        transposed_bsr_columns, transposed_bsr_values));
      return;
    }
  case 2:
    switch (cols_per_block) {
    case 1:
      wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_blocks, nnz,
                       (nnz, block_size, BsrBlockTransposer<2, 1, T>{},
                        block_indices, transposed_indices, bsr_values,
                        transposed_bsr_columns, transposed_bsr_values));
      return;
    case 2:
      wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_blocks, nnz,
                       (nnz, block_size, BsrBlockTransposer<2, 2, T>{},
                        block_indices, transposed_indices, bsr_values,
                        transposed_bsr_columns, transposed_bsr_values));
      return;
    case 3:
      wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_blocks, nnz,
                       (nnz, block_size, BsrBlockTransposer<2, 3, T>{},
                        block_indices, transposed_indices, bsr_values,
                        transposed_bsr_columns, transposed_bsr_values));
      return;
    }
  case 3:
    switch (cols_per_block) {
    case 1:
      wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_blocks, nnz,
                       (nnz, block_size, BsrBlockTransposer<3, 1, T>{},
                        block_indices, transposed_indices, bsr_values,
                        transposed_bsr_columns, transposed_bsr_values));
      return;
    case 2:
      wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_blocks, nnz,
                       (nnz, block_size, BsrBlockTransposer<3, 2, T>{},
                        block_indices, transposed_indices, bsr_values,
                        transposed_bsr_columns, transposed_bsr_values));
      return;
    case 3:
      wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_blocks, nnz,
                       (nnz, block_size, BsrBlockTransposer<3, 3, T>{},
                        block_indices, transposed_indices, bsr_values,
                        transposed_bsr_columns, transposed_bsr_values));
      return;
    }
  }

  wp_launch_device(
      WP_CURRENT_CONTEXT, bsr_transpose_blocks, nnz,
      (nnz, block_size,
       BsrBlockTransposer<-1, -1, T>{rows_per_block, cols_per_block},
       block_indices, transposed_indices, bsr_values, transposed_bsr_columns,
       transposed_bsr_values));
}

template <typename T>
void bsr_transpose_device(int rows_per_block, int cols_per_block, int row_count,
                          int col_count, int nnz, const int *bsr_offsets,
                          const int *bsr_columns, const T *bsr_values,
                          int *transposed_bsr_offsets,
                          int *transposed_bsr_columns,
                          T *transposed_bsr_values) {

  const int block_size = rows_per_block * cols_per_block;

  void *context = cuda_context_get_current();
  ContextGuard guard(context);

  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream_get_current());

  // Zero the transposed offsets
  memset_device(WP_CURRENT_CONTEXT, transposed_bsr_offsets, 0,
                (col_count + 1) * sizeof(int));

  ScopedTemporary<int> block_indices(context, 2*nnz);
  ScopedTemporary<BsrRowCol> combined_row_col(context, 2*nnz);

  cub::DoubleBuffer<int> d_keys(block_indices.buffer(),
                                block_indices.buffer() + nnz);
  cub::DoubleBuffer<BsrRowCol> d_values(combined_row_col.buffer(),
                                        combined_row_col.buffer() + nnz);

  wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_fill_row_col, nnz,
                   (nnz, row_count, bsr_offsets, bsr_columns, d_keys.Current(),
                    d_values.Current(), transposed_bsr_offsets));

  // Sort blocks
  {
    size_t buff_size = 0;
    check_cuda(cub::DeviceRadixSort::SortPairs(nullptr, buff_size, d_values,
                                              d_keys, nnz, 0, 64, stream));
    ScopedTemporary<> temp(context, buff_size);
    check_cuda(cub::DeviceRadixSort::SortPairs(
        temp.buffer(), buff_size, d_values, d_keys, nnz, 0, 64, stream));
  }

  // Prefix sum the transposed row block counts
  {
    size_t buff_size = 0;
    check_cuda(cub::DeviceScan::InclusiveSum(
        nullptr, buff_size, transposed_bsr_offsets, transposed_bsr_offsets,
        col_count + 1, stream));
    ScopedTemporary<> temp(context, buff_size);
    check_cuda(cub::DeviceScan::InclusiveSum(
        temp.buffer(), buff_size, transposed_bsr_offsets,
        transposed_bsr_offsets, col_count + 1, stream));
  }

  // Move and transpose individual blocks
  launch_bsr_transpose_blocks(
       nnz, block_size,
       rows_per_block, cols_per_block,
       d_keys.Current(), d_values.Current(), bsr_values, transposed_bsr_columns,
       transposed_bsr_values);
}

} // namespace

int bsr_matrix_from_triplets_float_device(
    int rows_per_block, int cols_per_block, int row_count, int nnz,
    uint64_t tpl_rows, uint64_t tpl_columns, uint64_t tpl_values,
    uint64_t bsr_offsets, uint64_t bsr_columns, uint64_t bsr_values) {
  return bsr_matrix_from_triplets_device<float>(
      rows_per_block, cols_per_block, row_count, nnz,
      reinterpret_cast<const int *>(tpl_rows),
      reinterpret_cast<const int *>(tpl_columns),
      reinterpret_cast<const float *>(tpl_values),
      reinterpret_cast<int *>(bsr_offsets),
      reinterpret_cast<int *>(bsr_columns),
      reinterpret_cast<float *>(bsr_values));
}

int bsr_matrix_from_triplets_double_device(
    int rows_per_block, int cols_per_block, int row_count, int nnz,
    uint64_t tpl_rows, uint64_t tpl_columns, uint64_t tpl_values,
    uint64_t bsr_offsets, uint64_t bsr_columns, uint64_t bsr_values) {
  return bsr_matrix_from_triplets_device<double>(
      rows_per_block, cols_per_block, row_count, nnz,
      reinterpret_cast<const int *>(tpl_rows),
      reinterpret_cast<const int *>(tpl_columns),
      reinterpret_cast<const double *>(tpl_values),
      reinterpret_cast<int *>(bsr_offsets),
      reinterpret_cast<int *>(bsr_columns),
      reinterpret_cast<double *>(bsr_values));
}

void bsr_transpose_float_device(int rows_per_block, int cols_per_block,
                                int row_count, int col_count, int nnz,
                                uint64_t bsr_offsets, uint64_t bsr_columns,
                                uint64_t bsr_values,
                                uint64_t transposed_bsr_offsets,
                                uint64_t transposed_bsr_columns,
                                uint64_t transposed_bsr_values) {
  bsr_transpose_device(rows_per_block, cols_per_block, row_count, col_count,
                       nnz, reinterpret_cast<const int *>(bsr_offsets),
                       reinterpret_cast<const int *>(bsr_columns),
                       reinterpret_cast<const float *>(bsr_values),
                       reinterpret_cast<int *>(transposed_bsr_offsets),
                       reinterpret_cast<int *>(transposed_bsr_columns),
                       reinterpret_cast<float *>(transposed_bsr_values));
}

void bsr_transpose_double_device(int rows_per_block, int cols_per_block,
                                 int row_count, int col_count, int nnz,
                                 uint64_t bsr_offsets, uint64_t bsr_columns,
                                 uint64_t bsr_values,
                                 uint64_t transposed_bsr_offsets,
                                 uint64_t transposed_bsr_columns,
                                 uint64_t transposed_bsr_values) {
  bsr_transpose_device(rows_per_block, cols_per_block, row_count, col_count,
                       nnz, reinterpret_cast<const int *>(bsr_offsets),
                       reinterpret_cast<const int *>(bsr_columns),
                       reinterpret_cast<const double *>(bsr_values),
                       reinterpret_cast<int *>(transposed_bsr_offsets),
                       reinterpret_cast<int *>(transposed_bsr_columns),
                       reinterpret_cast<double *>(transposed_bsr_values));
}
