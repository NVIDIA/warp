# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
from typing import Any, Generic, Optional, Tuple, TypeVar, Union

import warp as wp
import warp.types
import warp.utils
from warp.types import (
    Array,
    Cols,
    Rows,
    Scalar,
    Vector,
    is_array,
    scalar_types,
    type_is_matrix,
    type_length,
    type_repr,
    type_scalar_type,
    type_to_warp,
    types_equal,
)

# typing hints

_BlockType = TypeVar("BlockType")  # noqa: PLC0132


class _MatrixBlockType(Generic[Rows, Cols, Scalar]):
    pass


class _ScalarBlockType(Generic[Scalar]):
    pass


BlockType = Union[_MatrixBlockType[Rows, Cols, Scalar], _ScalarBlockType[Scalar]]

_struct_cache = {}


class BsrMatrix(Generic[_BlockType]):
    """Untyped base class for BSR and CSR matrices.

    Should not be constructed directly but through functions such as :func:`bsr_zeros`.

    Attributes:
        nrow (int): Number of rows of blocks.
        ncol (int): Number of columns of blocks.
        nnz (int):  Upper bound for the number of non-zero blocks, used for
          dimensioning launches. The exact number is at ``offsets[nrow-1]``.
          See also :meth:`nnz_sync`.
        offsets (Array[int]): Array of size at least ``1 + nrow`` such that the
          start and end indices of the blocks of row ``r`` are ``offsets[r]``
          and ``offsets[r+1]``, respectively.
        columns (Array[int]): Array of size at least equal to ``nnz`` containing
          block column indices.
        values (Array[BlockType]): Array of size at least equal to ``nnz``
          containing block values.
    """

    @property
    def scalar_type(self) -> Scalar:
        """Scalar type for individual block coefficients. For CSR matrices, this is the same as the block type."""
        return type_scalar_type(self.values.dtype)

    @property
    def block_shape(self) -> Tuple[int, int]:
        """Shape of the individual blocks."""
        return getattr(self.values.dtype, "_shape_", (1, 1))

    @property
    def block_size(self) -> int:
        """Size of the individual blocks, i.e. number of rows per block times number of columns per block."""
        return type_length(self.values.dtype)

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the matrix, i.e. number of rows/columns of blocks times number of rows/columns per block."""
        block_shape = self.block_shape
        return (self.nrow * block_shape[0], self.ncol * block_shape[1])

    @property
    def dtype(self) -> type:
        """Data type for individual block values."""
        return self.values.dtype

    @property
    def device(self) -> wp.context.Device:
        """Device on which ``offsets``, ``columns``, and ``values`` are allocated -- assumed to be the same for all three arrays."""
        return self.values.device

    @property
    def scalar_values(self) -> wp.array:
        """Accesses the ``values`` array as a 3d scalar array."""
        if self.block_shape == (1, 1):
            return self.values.reshape((self.nnz, 1, 1))

        def _as_3d_array(arr):
            return wp.array(
                ptr=arr.ptr,
                capacity=arr.capacity,
                device=arr.device,
                dtype=self.scalar_type,
                shape=(self.nnz, *self.block_shape),
                grad=None if arr.grad is None else _as_3d_array(arr.grad),
            )

        values_view = _as_3d_array(self.values)
        values_view._ref = self.values  # keep ref in case we're garbage collected
        return values_view

    def uncompress_rows(self, out: wp.array = None) -> wp.array:
        """Compute the row index for each non-zero block from the compressed row offsets."""
        if out is None:
            out = wp.empty(self.nnz, dtype=int, device=self.device)

        wp.launch(
            kernel=_bsr_get_block_row,
            device=self.device,
            dim=self.nnz,
            inputs=[self.nrow, self.offsets, out],
        )
        return out

    def nnz_sync(self):
        """Ensure that any ongoing transfer of the exact nnz number from the device offsets array to the host has completed
        and update the nnz upper bound.

        See also :meth:`copy_nnz_async`.
        """

        if self._is_nnz_transfer_setup():
            if self.device.is_cuda:
                wp.synchronize_event(self._nnz_event)
            self.nnz = int(self._nnz_buf.numpy()[0])
        return self.nnz

    def copy_nnz_async(self, known_nnz: Optional[int] = None) -> None:
        """
        Start the asynchronous transfer of the exact nnz from the device offsets array to host and records an event for completion.

        Needs to be called whenever the offsets array has been modified from outside ``warp.sparse``.

        See also :meth:`nnz_sync`.
        """
        if known_nnz is not None:
            self.nnz = int(known_nnz)
        else:
            self._setup_nnz_transfer()

        # If a transfer is already ongoing, or if the actual nnz is unknown, schedule a new transfer
        if self._is_nnz_transfer_setup():
            stream = wp.get_stream(self.device) if self.device.is_cuda else None
            wp.copy(src=self.offsets, dest=self._nnz_buf, src_offset=self.nrow, count=1, stream=stream)
            if self.device.is_cuda:
                stream.record_event(self._nnz_event)

    def _setup_nnz_transfer(self):
        if self._is_nnz_transfer_setup():
            return

        BsrMatrix.__setattr__(
            self, "_nnz_buf", wp.empty(dtype=int, shape=(1,), device="cpu", pinned=self.device.is_cuda)
        )
        if self.device.is_cuda:
            BsrMatrix.__setattr__(self, "_nnz_event", wp.Event(self.device))

    def _is_nnz_transfer_setup(self):
        return hasattr(self, "_nnz_buf")

    def _nnz_transfer_buf_and_event(self):
        self._setup_nnz_transfer()

        if not self.device.is_cuda:
            return self._nnz_buf, ctypes.c_void_p(None)
        return self._nnz_buf, self._nnz_event.cuda_event

    # Overloaded math operators
    def __add__(self, y):
        return bsr_axpy(y, bsr_copy(self))

    def __iadd__(self, y):
        return bsr_axpy(y, self)

    def __radd__(self, x):
        return bsr_axpy(x, bsr_copy(self))

    def __sub__(self, y):
        return bsr_axpy(y, bsr_copy(self), alpha=-1.0)

    def __rsub__(self, x):
        return bsr_axpy(x, bsr_copy(self), beta=-1.0)

    def __isub__(self, y):
        return bsr_axpy(y, self, alpha=-1.0)

    def __mul__(self, y):
        return _BsrScalingExpression(self, y)

    def __rmul__(self, x):
        return _BsrScalingExpression(self, x)

    def __imul__(self, y):
        return bsr_scale(self, y)

    def __matmul__(self, y):
        if isinstance(y, wp.array):
            return bsr_mv(self, y)

        return bsr_mm(self, y)

    def __rmatmul__(self, x):
        if isinstance(x, wp.array):
            return bsr_mv(self, x, transpose=True)

        return bsr_mm(x, self)

    def __imatmul__(self, y):
        return bsr_mm(self, y, self)

    def __truediv__(self, y):
        return _BsrScalingExpression(self, 1.0 / y)

    def __neg__(self):
        return _BsrScalingExpression(self, -1.0)

    def transpose(self):
        """Return a transposed copy of this matrix."""
        return bsr_transposed(self)


def bsr_matrix_t(dtype: BlockType):
    dtype = type_to_warp(dtype)

    if not type_is_matrix(dtype) and dtype not in scalar_types:
        raise ValueError(f"BsrMatrix block type must be either warp matrix or scalar; got {type_repr(dtype)}")

    class BsrMatrixTyped(BsrMatrix):
        nrow: int
        """Number of rows of blocks."""
        ncol: int
        """Number of columns of blocks."""
        nnz: int
        """Upper bound for the number of non-zeros."""
        offsets: wp.array(dtype=int)
        """Array of size at least ``1 + nrow``."""
        columns: wp.array(dtype=int)
        """Array of size at least equal to ``nnz``."""
        values: wp.array(dtype=dtype)

    module = wp.get_module(BsrMatrix.__module__)

    if hasattr(dtype, "_shape_"):
        type_str = f"{type_scalar_type(dtype).__name__}_{dtype._shape_[0]}_{dtype._shape_[1]}"
    else:
        type_str = dtype.__name__
    key = f"{BsrMatrix.__qualname__}_{type_str}"

    if key not in _struct_cache:
        _struct_cache[key] = wp.codegen.Struct(
            cls=BsrMatrixTyped,
            key=key,
            module=module,
        )

    return _struct_cache[key]


def bsr_zeros(
    rows_of_blocks: int,
    cols_of_blocks: int,
    block_type: BlockType,
    device: wp.context.Devicelike = None,
) -> BsrMatrix:
    """Construct and return an empty BSR or CSR matrix with the given shape.

    Args:
        bsr: The BSR or CSR matrix to set to zero.
        rows_of_blocks: Number of rows of blocks.
        cols_of_blocks: Number of columns of blocks.
        block_type: Type of individual blocks.
          For CSR matrices, this should be a scalar type.
          For BSR matrices, this should be a matrix type (e.g. from :func:`warp.mat`).
        device: Device on which to allocate the matrix arrays.
    """

    bsr = bsr_matrix_t(block_type)()

    bsr.nrow = int(rows_of_blocks)
    bsr.ncol = int(cols_of_blocks)
    bsr.nnz = 0
    bsr.columns = wp.empty(shape=(0,), dtype=int, device=device)
    bsr.values = wp.empty(shape=(0,), dtype=block_type, device=device)
    bsr.offsets = wp.zeros(shape=(bsr.nrow + 1,), dtype=int, device=device)

    return bsr


def _bsr_ensure_fits(bsr: BsrMatrix, nrow: Optional[int] = None, nnz: Optional[int] = None) -> None:
    if nrow is None:
        nrow = bsr.nrow
    if nnz is None:
        nnz = bsr.nnz
    else:
        # update nnz upper bound
        bsr.nnz = int(nnz)

    if bsr.offsets.size < nrow + 1:
        bsr.offsets = wp.empty(shape=(nrow + 1,), dtype=int, device=bsr.offsets.device)
    if bsr.columns.size < nnz:
        bsr.columns = wp.empty(shape=(nnz,), dtype=int, device=bsr.columns.device)
    if bsr.values.size < nnz:
        bsr.values = wp.empty(shape=(nnz,), dtype=bsr.values.dtype, device=bsr.values.device)


def bsr_set_zero(
    bsr: BsrMatrix,
    rows_of_blocks: Optional[int] = None,
    cols_of_blocks: Optional[int] = None,
):
    """Set a BSR matrix to zero, possibly changing its size.

    Args:
        bsr: The BSR or CSR matrix to set to zero.
        rows_of_blocks: If not ``None``, the new number of rows of blocks.
        cols_of_blocks: If not ``None``, the new number of columns of blocks.
    """

    if rows_of_blocks is not None:
        bsr.nrow = int(rows_of_blocks)
    if cols_of_blocks is not None:
        bsr.ncol = int(cols_of_blocks)

    _bsr_ensure_fits(bsr, nnz=0)
    bsr.offsets.zero_()
    bsr.copy_nnz_async(known_nnz=0)


def bsr_set_from_triplets(
    dest: BsrMatrix[BlockType[Rows, Cols, Scalar]],
    rows: "Array[int]",
    columns: "Array[int]",
    values: Optional["Array[Union[Scalar, BlockType[Rows, Cols, Scalar]]]"] = None,
    prune_numerical_zeros: bool = True,
    masked: bool = False,
):
    """Fill a BSR matrix with values defined by coordinate-oriented (COO) triplets, discarding existing blocks.

    The first dimension of the three input arrays must match and indicates the number of COO triplets.

    Args:
        dest: Sparse matrix to populate.
        rows: Row index for each non-zero.
        columns: Columns index for each non-zero.
        values: Block values for each non-zero. Must be either a one-dimensional array with data type identical
          to the ``dest`` matrix's block type, or a 3d array with data type equal to the ``dest`` matrix's scalar type.
          If ``None``, the values array of the resulting matrix will be allocated but uninitialized.
        prune_numerical_zeros: If ``True``, will ignore the zero-valued blocks.
        masked: If ``True``, ignore blocks that are not existing non-zeros of ``dest``.
    """

    if rows.device != columns.device or rows.device != dest.device:
        raise ValueError("All arguments must reside on the same device")

    if rows.shape[0] != columns.shape[0]:
        raise ValueError("All triplet arrays must have the same length")

    # Accept either array1d(dtype) or contiguous array3d(scalar_type) as values
    if values is not None:
        if values.device != rows.device:
            raise ValueError("All arguments must reside on the same device")

        if values.shape[0] != rows.shape[0]:
            raise ValueError("All triplet arrays must have the same length")

        if values.ndim == 1:
            if values.dtype != dest.values.dtype:
                raise ValueError("Values array type must correspond to that of dest matrix")
        elif values.ndim == 3:
            if values.shape[1:] != dest.block_shape:
                raise ValueError(
                    f"Last two dimensions in values array ({values.shape[1:]}) should correspond to matrix block shape {(dest.block_shape)})"
                )

            if type_scalar_type(values.dtype) != dest.scalar_type:
                raise ValueError("Scalar type of values array should correspond to that of matrix")

            if not values.is_contiguous:
                raise ValueError("Multi-dimensional values array should be contiguous")
        else:
            raise ValueError("Number of dimension for values array should be 1 or 3")

    nnz = rows.shape[0]
    if nnz == 0:
        bsr_set_zero(dest)
        return

    # Increase dest array sizes if needed
    if not masked:
        _bsr_ensure_fits(dest, nnz=nnz)

    device = dest.values.device
    scalar_type = dest.scalar_type
    from warp.context import runtime

    if device.is_cpu:
        if scalar_type == wp.float32:
            native_func = runtime.core.bsr_matrix_from_triplets_float_host
        elif scalar_type == wp.float64:
            native_func = runtime.core.bsr_matrix_from_triplets_double_host
    else:
        if scalar_type == wp.float32:
            native_func = runtime.core.bsr_matrix_from_triplets_float_device
        elif scalar_type == wp.float64:
            native_func = runtime.core.bsr_matrix_from_triplets_double_device

    if not native_func:
        raise NotImplementedError(f"bsr_from_triplets not implemented for scalar type {scalar_type}")

    nnz_buf, nnz_event = dest._nnz_transfer_buf_and_event()

    with wp.ScopedDevice(device):
        native_func(
            dest.block_shape[0],
            dest.block_shape[1],
            dest.nrow,
            nnz,
            ctypes.cast(rows.ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(columns.ptr, ctypes.POINTER(ctypes.c_int32)),
            None if values is None else ctypes.cast(values.ptr, ctypes.c_void_p),
            prune_numerical_zeros,
            masked,
            ctypes.cast(dest.offsets.ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(dest.columns.ptr, ctypes.POINTER(ctypes.c_int32)),
            None if values is None else ctypes.cast(dest.values.ptr, ctypes.c_void_p),
            ctypes.cast(nnz_buf.ptr, ctypes.POINTER(ctypes.c_int32)),
            nnz_event,
        )


def bsr_from_triplets(
    rows_of_blocks: int,
    cols_of_blocks: int,
    rows: "Array[int]",
    columns: "Array[int]",
    values: "Array[Union[Scalar, BlockType[Rows, Cols, Scalar]]]",
    prune_numerical_zeros: bool = True,
):
    """Constructs a BSR matrix with values defined by coordinate-oriented (COO) triplets.

    The first dimension of the three input arrays must match and indicates the number of COO triplets.

    Args:
        rows_of_blocks: Number of rows of blocks.
        cols_of_blocks: Number of columns of blocks.
        rows: Row index for each non-zero.
        columns: Columns index for each non-zero.
        values: Block values for each non-zero. Must be either a one-dimensional array with data type identical
          to the ``dest`` matrix's block type, or a 3d array with data type equal to the ``dest`` matrix's scalar type.
        prune_numerical_zeros: If ``True``, will ignore the zero-valued blocks.
    """

    if values.ndim == 3:
        block_type = wp.mat(shape=values.shape[1:], dtype=values.dtype)
    else:
        block_type = values.dtype

    A = bsr_zeros(
        rows_of_blocks=rows_of_blocks, cols_of_blocks=cols_of_blocks, block_type=block_type, device=values.device
    )
    bsr_set_from_triplets(A, rows, columns, values, prune_numerical_zeros=prune_numerical_zeros)
    return A


class _BsrExpression(Generic[_BlockType]):
    pass


class _BsrScalingExpression(_BsrExpression):
    def __init__(self, mat, scale):
        self.mat = mat
        self.scale = scale

    def eval(self):
        return bsr_copy(self)

    @property
    def nrow(self) -> int:
        return self.mat.nrow

    @property
    def ncol(self) -> int:
        return self.mat.ncol

    @property
    def nnz(self) -> int:
        return self.mat.nnz

    @property
    def offsets(self) -> wp.array:
        return self.mat.offsets

    @property
    def columns(self) -> wp.array:
        return self.mat.columns

    @property
    def scalar_type(self) -> Scalar:
        return self.mat.scalar_type

    @property
    def block_shape(self) -> Tuple[int, int]:
        return self.mat.block_shape

    @property
    def block_size(self) -> int:
        return self.mat.block_size

    @property
    def shape(self) -> Tuple[int, int]:
        return self.mat.shape

    @property
    def dtype(self) -> type:
        return self.mat.dtype

    @property
    def device(self) -> wp.context.Device:
        return self.mat.device

    # Overloaded math operators
    def __add__(self, y):
        return bsr_axpy(y, bsr_copy(self.mat), alpha=self.scale)

    def __radd__(self, x):
        return bsr_axpy(x, bsr_copy(self.mat), beta=self.scale)

    def __sub__(self, y):
        return bsr_axpy(y, bsr_copy(self.mat), alpha=-self.scale)

    def __rsub__(self, x):
        return bsr_axpy(x, bsr_copy(self.mat), beta=-self.scale)

    def __mul__(self, y):
        return _BsrScalingExpression(self.mat, y * self.scale)

    def __rmul__(self, x):
        return _BsrScalingExpression(self.mat, x * self.scale)

    def __matmul__(self, y):
        if isinstance(y, wp.array):
            return bsr_mv(self.mat, y, alpha=self.scale)

        return bsr_mm(self.mat, y, alpha=self.scale)

    def __rmatmul__(self, x):
        if isinstance(x, wp.array):
            return bsr_mv(self.mat, x, alpha=self.scale, transpose=True)

        return bsr_mm(x, self.mat, alpha=self.scale)

    def __truediv__(self, y):
        return _BsrScalingExpression(self.mat, self.scale / y)

    def __neg__(self):
        return _BsrScalingExpression(self.mat, -self.scale)

    def transpose(self):
        """Returns a transposed copy of this matrix"""
        return _BsrScalingExpression(self.mat.transpose(), self.scale)


BsrMatrixOrExpression = Union[BsrMatrix[_BlockType], _BsrExpression[_BlockType]]


def _extract_matrix_and_scale(bsr: BsrMatrixOrExpression):
    if isinstance(bsr, BsrMatrix):
        return bsr, 1.0
    if isinstance(bsr, _BsrScalingExpression):
        return bsr.mat, bsr.scale

    raise ValueError("Argument cannot be interpreted as a BsrMatrix")


@wp.func
def _bsr_row_index(
    offsets: wp.array(dtype=int),
    row_count: int,
    block: int,
):
    """Index of the row containing a block, or -1 if non-existing."""
    return wp.where(block < offsets[row_count], wp.lower_bound(offsets, 0, row_count + 1, block + 1), 0) - 1


@wp.func
def _bsr_block_index(
    row: int,
    col: int,
    bsr_offsets: wp.array(dtype=int),
    bsr_columns: wp.array(dtype=int),
):
    """Index of the block at block-coordinates (row, col), or -1 if non-existing.
    Assumes bsr_columns is sorted.
    """

    if row < 0:
        return -1

    mask_row_beg = bsr_offsets[row]
    mask_row_end = bsr_offsets[row + 1]

    if mask_row_beg == mask_row_end:
        return -1

    block_index = wp.lower_bound(bsr_columns, mask_row_beg, mask_row_end, col)
    return wp.where(bsr_columns[block_index] == col, block_index, -1)


@wp.kernel(enable_backward=False)
def _bsr_assign_list_blocks(
    src_subrows: int,
    src_subcols: int,
    dest_subrows: int,
    dest_subcols: int,
    src_row_count: int,
    src_offsets: wp.array(dtype=int),
    src_columns: wp.array(dtype=int),
    dest_rows: wp.array(dtype=int),
    dest_cols: wp.array(dtype=int),
):
    block, subrow, subcol = wp.tid()
    dest_block = (block * src_subcols + subcol) * src_subrows + subrow

    row = _bsr_row_index(src_offsets, src_row_count, block)
    if row == -1:
        dest_rows[dest_block] = row  # invalid
        dest_cols[dest_block] = row
    else:
        dest_subrow = row * src_subrows + subrow
        dest_subcol = src_columns[block] * src_subcols + subcol
        dest_rows[dest_block] = dest_subrow // dest_subrows
        dest_cols[dest_block] = dest_subcol // dest_subcols


@wp.kernel
def _bsr_assign_copy_blocks(
    scale: Any,
    src_subrows: int,
    src_subcols: int,
    dest_subrows: int,
    dest_subcols: int,
    src_row_count: int,
    src_offsets: wp.array(dtype=int),
    src_columns: wp.array(dtype=int),
    src_values: wp.array3d(dtype=Any),
    dest_offsets: wp.array(dtype=int),
    dest_columns: wp.array(dtype=int),
    dest_values: wp.array3d(dtype=Any),
):
    src_block = wp.tid()
    src_block, subrow, subcol = wp.tid()

    src_row = _bsr_row_index(src_offsets, src_row_count, src_block)
    if src_row == -1:
        return

    src_col = src_columns[src_block]

    dest_subrow = src_row * src_subrows + subrow
    dest_subcol = src_col * src_subcols + subcol
    dest_row = dest_subrow // dest_subrows
    dest_col = dest_subcol // dest_subcols

    dest_block = _bsr_block_index(dest_row, dest_col, dest_offsets, dest_columns)
    if dest_block == -1:
        return

    split_row = dest_subrow - dest_subrows * dest_row
    split_col = dest_subcol - dest_subcols * dest_col

    rows_per_subblock = src_values.shape[1] // src_subrows
    cols_per_subblock = src_values.shape[2] // src_subcols

    dest_base_i = split_row * rows_per_subblock
    dest_base_j = split_col * cols_per_subblock

    src_base_i = subrow * rows_per_subblock
    src_base_j = subcol * cols_per_subblock

    for i in range(rows_per_subblock):
        for j in range(cols_per_subblock):
            dest_values[dest_block, i + dest_base_i, j + dest_base_j] = dest_values.dtype(
                scale * src_values[src_block, i + src_base_i, j + src_base_j]
            )


def bsr_assign(
    dest: BsrMatrix[BlockType[Rows, Cols, Scalar]],
    src: BsrMatrixOrExpression[BlockType[Any, Any, Any]],
    structure_only: bool = False,
    masked: bool = False,
):
    """Copy the content of the ``src`` BSR matrix to ``dest``.

    Args:
      src: Matrix to be copied.
      dest: Destination matrix. May have a different block shape or scalar type
        than ``src``, in which case the required casting will be performed.
      structure_only: If ``True``, only the non-zeros indices are copied, and uninitialized value storage is allocated
        to accommodate at least ``src.nnz`` blocks. If ``structure_only`` is ``False``, values are also copied with implicit
        casting if the two matrices use distinct scalar types.
      masked: If ``True``, prevent the assignment operation from adding new non-zeros blocks to ``dest``.
    """

    src, src_scale = _extract_matrix_and_scale(src)

    if dest.values.device != src.values.device:
        raise ValueError("Source and destination matrices must reside on the same device")

    if src.block_shape[0] >= dest.block_shape[0]:
        src_subrows = src.block_shape[0] // dest.block_shape[0]
        dest_subrows = 1
    else:
        dest_subrows = dest.block_shape[0] // src.block_shape[0]
        src_subrows = 1

    if src_subrows * dest.block_shape[0] != src.block_shape[0] * dest_subrows:
        raise ValueError(
            f"Incompatible dest and src block shapes; block rows must evenly divide one another (Got {src.block_shape[0]}, {dest.block_shape[0]})"
        )

    if src.block_shape[1] >= dest.block_shape[1]:
        src_subcols = src.block_shape[1] // dest.block_shape[1]
        dest_subcols = 1
    else:
        dest_subcols = dest.block_shape[1] // src.block_shape[1]
        src_subcols = 1

    if src_subcols * dest.block_shape[1] != src.block_shape[1] * dest_subcols:
        raise ValueError(
            f"Incompatible dest and src block shapes; block columns must evenly divide one another (Got {src.block_shape[1]}, {dest.block_shape[1]})"
        )

    dest_nrow = (src.nrow * src_subrows) // dest_subrows
    dest_ncol = (src.ncol * src_subcols) // dest_subcols

    if src.nrow * src_subrows != dest_nrow * dest_subrows or src.ncol * src_subcols != dest_ncol * dest_subcols:
        raise ValueError("The requested block shape does not evenly divide the source matrix")

    nnz_alloc = src.nnz * src_subrows * src_subcols
    if masked:
        if dest_nrow != dest.nrow or dest_ncol != dest.ncol:
            raise ValueError(
                f"Incompatible destination matrix size, expected ({dest_nrow}, {dest_ncol}), got ({dest.nrow}, {dest.ncol})"
            )
    else:
        dest.nrow = dest_nrow
        dest.ncol = dest_ncol
        _bsr_ensure_fits(dest, nnz=nnz_alloc)

    if dest.block_shape == src.block_shape and not masked:
        # Direct copy

        wp.copy(dest=dest.offsets, src=src.offsets, count=src.nrow + 1)
        dest.copy_nnz_async()

        if nnz_alloc > 0:
            wp.copy(dest=dest.columns, src=src.columns, count=nnz_alloc)

            if not structure_only:
                warp.utils.array_cast(out_array=dest.values, in_array=src.values, count=nnz_alloc)
                bsr_scale(dest, src_scale)

    else:
        # Masked and/or multiple src blocks per dest block, go through COO format

        # Compute destination rows and columns
        dest_rows = wp.empty(nnz_alloc, dtype=int, device=dest.device)
        dest_cols = wp.empty(nnz_alloc, dtype=int, device=dest.device)
        wp.launch(
            _bsr_assign_list_blocks,
            dim=(src.nnz, src_subrows, src_subcols),
            device=dest.device,
            inputs=[
                src_subrows,
                src_subcols,
                dest_subrows,
                dest_subcols,
                src.nrow,
                src.offsets,
                src.columns,
                dest_rows,
                dest_cols,
            ],
        )

        # Compute destination offsets from triplets
        from warp.context import runtime

        if dest.device.is_cpu:
            native_func = runtime.core.bsr_matrix_from_triplets_float_host
        else:
            native_func = runtime.core.bsr_matrix_from_triplets_float_device

        nnz_buf, nnz_event = dest._nnz_transfer_buf_and_event()
        with wp.ScopedDevice(dest.device):
            native_func(
                dest.block_shape[0],
                dest.block_shape[1],
                dest.nrow,
                nnz_alloc,
                ctypes.cast(dest_rows.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(dest_cols.ptr, ctypes.POINTER(ctypes.c_int32)),
                0,
                False,
                masked,
                ctypes.cast(dest.offsets.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(dest.columns.ptr, ctypes.POINTER(ctypes.c_int32)),
                0,
                ctypes.cast(nnz_buf.ptr, ctypes.POINTER(ctypes.c_int32)),
                nnz_event,
            )

        # merge block values
        if not structure_only:
            dest.values.zero_()
            wp.launch(
                _bsr_assign_copy_blocks,
                dim=(src.nnz, src_subrows, src_subcols),
                device=dest.device,
                inputs=[
                    src.scalar_type(src_scale),
                    src_subrows,
                    src_subcols,
                    dest_subrows,
                    dest_subcols,
                    src.nrow,
                    src.offsets,
                    src.columns,
                    src.scalar_values,
                    dest.offsets,
                    dest.columns,
                    dest.scalar_values,
                ],
            )


def bsr_copy(
    A: BsrMatrixOrExpression,
    scalar_type: Optional[Scalar] = None,
    block_shape: Optional[Tuple[int, int]] = None,
    structure_only: bool = False,
):
    """Return a copy of matrix ``A``, possibly changing its scalar type.

    Args:
       A: Matrix to be copied.
       scalar_type: If provided, the returned matrix will use this scalar type instead of the one from ``A``.
       block_shape: If provided, the returned matrix will use blocks of this shape instead of the one from ``A``.
         Both dimensions of ``block_shape`` must be either a multiple or an exact divider of the ones from ``A``.
       structure_only: If ``True``, only the non-zeros indices are copied, and uninitialized value storage is allocated
         to accommodate at least ``src.nnz`` blocks. If ``structure_only`` is ``False``, values are also copied with implicit
         casting if the two matrices use distinct scalar types.
    """
    if scalar_type is None:
        scalar_type = A.scalar_type
    if block_shape is None:
        block_shape = A.block_shape

    if block_shape == (1, 1):
        block_type = scalar_type
    else:
        block_type = wp.mat(shape=block_shape, dtype=scalar_type)

    copy = bsr_zeros(
        rows_of_blocks=A.nrow,
        cols_of_blocks=A.ncol,
        block_type=block_type,
        device=A.device,
    )
    bsr_assign(dest=copy, src=A, structure_only=structure_only)
    return copy


def bsr_set_transpose(
    dest: BsrMatrix[BlockType[Cols, Rows, Scalar]],
    src: BsrMatrixOrExpression[BlockType[Rows, Cols, Scalar]],
):
    """Assign the transposed matrix ``src`` to matrix ``dest``."""

    src, src_scale = _extract_matrix_and_scale(src)

    if dest.values.device != src.values.device:
        raise ValueError("All arguments must reside on the same device")

    if dest.scalar_type != src.scalar_type:
        raise ValueError("All arguments must have the same scalar type")

    transpose_block_shape = src.block_shape[::-1]

    if dest.block_shape != transpose_block_shape:
        raise ValueError(f"Destination block shape must be {transpose_block_shape}")

    nnz = src.nnz
    dest.nrow = src.ncol
    dest.ncol = src.nrow

    if nnz == 0:
        bsr_set_zero(dest)
        return

    # Increase dest array sizes if needed
    _bsr_ensure_fits(dest, nnz=nnz)

    from warp.context import runtime

    if dest.values.device.is_cpu:
        if dest.scalar_type == wp.float32:
            native_func = runtime.core.bsr_transpose_float_host
        elif dest.scalar_type == wp.float64:
            native_func = runtime.core.bsr_transpose_double_host
    else:
        if dest.scalar_type == wp.float32:
            native_func = runtime.core.bsr_transpose_float_device
        elif dest.scalar_type == wp.float64:
            native_func = runtime.core.bsr_transpose_double_device

    if not native_func:
        raise NotImplementedError(f"bsr_set_transpose not implemented for scalar type {dest.scalar_type}")

    with wp.ScopedDevice(dest.device):
        native_func(
            src.block_shape[0],
            src.block_shape[1],
            src.nrow,
            src.ncol,
            nnz,
            ctypes.cast(src.offsets.ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(src.columns.ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(src.values.ptr, ctypes.c_void_p),
            ctypes.cast(dest.offsets.ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(dest.columns.ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(dest.values.ptr, ctypes.c_void_p),
        )

    dest.copy_nnz_async()
    bsr_scale(dest, src_scale)


def bsr_transposed(A: BsrMatrixOrExpression) -> BsrMatrix:
    """Return a copy of the transposed matrix ``A``."""

    if A.block_shape == (1, 1):
        block_type = A.values.dtype
    else:
        block_type = wp.mat(shape=A.block_shape[::-1], dtype=A.scalar_type)

    transposed = bsr_zeros(
        rows_of_blocks=A.ncol,
        cols_of_blocks=A.nrow,
        block_type=block_type,
        device=A.device,
    )
    bsr_set_transpose(dest=transposed, src=A)
    return transposed


@wp.kernel
def _bsr_get_diag_kernel(
    scale: Any,
    A_offsets: wp.array(dtype=int),
    A_columns: wp.array(dtype=int),
    A_values: wp.array(dtype=Any),
    out: wp.array(dtype=Any),
):
    row = wp.tid()

    diag = _bsr_block_index(row, row, A_offsets, A_columns)
    if diag != -1:
        out[row] = scale * A_values[diag]


def bsr_get_diag(A: BsrMatrixOrExpression[BlockType], out: "Optional[Array[BlockType]]" = None) -> "Array[BlockType]":
    """Return the array of blocks that constitute the diagonal of a sparse matrix.

    Args:
        A: The sparse matrix from which to extract the diagonal.
        out: If provided, the array into which to store the diagonal blocks.
    """

    A, scale = _extract_matrix_and_scale(A)

    dim = min(A.nrow, A.ncol)

    if out is None:
        out = wp.zeros(shape=(dim,), dtype=A.values.dtype, device=A.values.device)
    else:
        if out.dtype != A.values.dtype:
            raise ValueError(f"Output array must have type {A.values.dtype}")
        if out.device != A.values.device:
            raise ValueError(f"Output array must reside on device {A.values.device}")
        if out.shape[0] < dim:
            raise ValueError(f"Output array must be of length at least {dim}")

    wp.launch(
        kernel=_bsr_get_diag_kernel,
        dim=dim,
        device=A.values.device,
        inputs=[A.scalar_type(scale), A.offsets, A.columns, A.values, out],
    )

    return out


@wp.kernel(enable_backward=False)
def _bsr_set_diag_kernel(
    nnz: int,
    A_offsets: wp.array(dtype=int),
    A_columns: wp.array(dtype=int),
):
    row = wp.tid()
    A_offsets[row] = wp.min(row, nnz)
    if row < nnz:
        A_columns[row] = row


def bsr_set_diag(
    A: BsrMatrix[BlockType],
    diag: "Union[BlockType, Array[BlockType]]",
    rows_of_blocks: Optional[int] = None,
    cols_of_blocks: Optional[int] = None,
) -> None:
    """Set ``A`` as a block-diagonal matrix.

    Args:
        A: The sparse matrix to modify.
        diag: Specifies the values for diagonal blocks. Can be one of:

          - A Warp array of type ``A.values.dtype``: Each element defines one block of the diagonal
          - A constant value of type ``A.values.dtype``: This value is assigned to all diagonal blocks
          - ``None``: Diagonal block values are left uninitialized

        rows_of_blocks: If not ``None``, the new number of rows of blocks.
        cols_of_blocks: If not ``None``, the new number of columns of blocks.

    The shape of the matrix will be defined one of the following, in this order:

    - ``rows_of_blocks`` and ``cols_of_blocks``, if provided.
      If only one is given, the second is assumed equal.
    - The first dimension of ``diag``, if ``diag`` is an array
    - The current dimensions of ``A`` otherwise
    """

    if rows_of_blocks is None and cols_of_blocks is not None:
        rows_of_blocks = cols_of_blocks
    if cols_of_blocks is None and rows_of_blocks is not None:
        cols_of_blocks = rows_of_blocks

    if is_array(diag):
        if rows_of_blocks is None:
            rows_of_blocks = diag.shape[0]
            cols_of_blocks = diag.shape[0]

    if rows_of_blocks is not None:
        A.nrow = rows_of_blocks
        A.ncol = cols_of_blocks

    nnz = min(A.nrow, A.ncol)
    _bsr_ensure_fits(A, nnz=nnz)

    wp.launch(
        kernel=_bsr_set_diag_kernel,
        dim=nnz + 1,
        device=A.offsets.device,
        inputs=[nnz, A.offsets, A.columns],
    )

    if is_array(diag):
        wp.copy(src=diag, dest=A.values, count=nnz)
    elif diag is not None:
        A.values.fill_(diag)

    A.copy_nnz_async(known_nnz=nnz)


def bsr_diag(
    diag: Optional[Union[BlockType, Array[BlockType]]] = None,
    rows_of_blocks: Optional[int] = None,
    cols_of_blocks: Optional[int] = None,
    block_type: Optional[BlockType] = None,
    device=None,
) -> BsrMatrix["BlockType"]:
    """Create and return a block-diagonal BSR matrix from an given block value or array of block values.

    Args:
        diag: Specifies the values for diagonal blocks. Can be one of:

          - A Warp array of type ``A.values.dtype``: Each element defines one block of the diagonal
          - A constant value of type ``A.values.dtype``: This value is assigned to all diagonal blocks
        rows_of_blocks: If not ``None``, the new number of rows of blocks
        cols_of_blocks: If not ``None``, the new number of columns of blocks
        block_type: If ``diag`` is ``None``, block type of the matrix. Otherwise deduced from ``diag``
        device: If ``diag`` is not a Warp array, device on which to allocate the matrix. Otherwise deduced from ``diag``

    The shape of the matrix will be defined one of the following, in this order:

    - ``rows_of_blocks`` and ``cols_of_blocks``, if provided.
      If only one is given, the second is assumed equal.
    - The first dimension of ``diag`` if ``diag`` is an array.
    """

    if rows_of_blocks is None and cols_of_blocks is not None:
        rows_of_blocks = cols_of_blocks
    if cols_of_blocks is None and rows_of_blocks is not None:
        cols_of_blocks = rows_of_blocks

    if is_array(diag):
        if rows_of_blocks is None:
            rows_of_blocks = diag.shape[0]
            cols_of_blocks = diag.shape[0]

        block_type = diag.dtype
        device = diag.device
    else:
        if rows_of_blocks is None:
            raise ValueError(
                "rows_of_blocks and/or cols_of_blocks must be provided for constructing a diagonal matrix with uniform diagonal"
            )

    if block_type is None:
        if diag is None:
            raise ValueError("Either `diag` or `block_type` needs to be provided")

        block_type = type(diag)
        if not type_is_matrix(block_type) and len(getattr(diag, "shape", ())) == 2:
            block_type = wp.mat(shape=diag.shape, dtype=diag.dtype)

    A = bsr_zeros(rows_of_blocks, cols_of_blocks, block_type=block_type, device=device)
    bsr_set_diag(A, diag)
    return A


def bsr_set_identity(A: BsrMatrix, rows_of_blocks: Optional[int] = None) -> None:
    """Set ``A`` as the identity matrix.

    Args:
        A: The sparse matrix to modify.
        rows_of_blocks: If provided, the matrix will be resized as a square
          matrix with ``rows_of_blocks`` rows and columns.
    """

    if A.block_shape == (1, 1):
        identity = A.scalar_type(1.0)
    else:
        from numpy import eye

        identity = eye(A.block_shape[0])

    bsr_set_diag(A, diag=identity, rows_of_blocks=rows_of_blocks, cols_of_blocks=rows_of_blocks)


def bsr_identity(
    rows_of_blocks: int,
    block_type: BlockType[Rows, Rows, Scalar],
    device: wp.context.Devicelike = None,
) -> BsrMatrix[BlockType[Rows, Rows, Scalar]]:
    """Create and return a square identity matrix.

    Args:
        rows_of_blocks: Number of rows and columns of blocks in the created matrix.
        block_type: Block type for the newly created matrix. Must be square
        device: Device onto which to allocate the data arrays
    """
    A = bsr_zeros(
        rows_of_blocks=rows_of_blocks,
        cols_of_blocks=rows_of_blocks,
        block_type=block_type,
        device=device,
    )
    bsr_set_identity(A)
    return A


@wp.kernel
def _bsr_scale_kernel(
    alpha: Any,
    values: wp.array(dtype=Any),
):
    values[wp.tid()] = alpha * values[wp.tid()]


def bsr_scale(x: BsrMatrixOrExpression, alpha: Scalar) -> BsrMatrix:
    """Perform the operation ``x := alpha * x`` on BSR matrix ``x`` and return ``x``."""

    x, scale = _extract_matrix_and_scale(x)
    alpha *= scale

    if alpha != 1.0 and x.nnz > 0:
        if alpha == 0.0:
            bsr_set_zero(x)
        else:
            alpha = x.scalar_type(alpha)

            wp.launch(
                kernel=_bsr_scale_kernel,
                dim=x.nnz,
                device=x.values.device,
                inputs=[alpha, x.values],
            )

    return x


@wp.kernel(enable_backward=False)
def _bsr_get_block_row(row_count: int, bsr_offsets: wp.array(dtype=int), rows: wp.array(dtype=int)):
    block = wp.tid()
    rows[block] = _bsr_row_index(bsr_offsets, row_count, block)


@wp.kernel
def _bsr_axpy_add_block(
    src_offset: int,
    scale: Any,
    rows: wp.array(dtype=int),
    cols: wp.array(dtype=int),
    dst_offsets: wp.array(dtype=int),
    dst_columns: wp.array(dtype=int),
    src_values: wp.array(dtype=Any),
    dst_values: wp.array(dtype=Any),
):
    i = wp.tid()
    row = rows[i + src_offset]
    col = cols[i + src_offset]

    block = _bsr_block_index(row, col, dst_offsets, dst_columns)
    if block != -1:
        dst_values[block] += scale * src_values[i]


class bsr_axpy_work_arrays:
    """Opaque structure for persisting :func:`bsr_axpy` temporary work buffers across calls."""

    def __init__(self):
        self._reset(None)

    def _reset(self, device):
        self.device = device
        self._sum_rows = None
        self._sum_cols = None
        self._old_y_values = None
        self._old_x_values = None

    def _allocate(self, device, y: BsrMatrix, sum_nnz: int):
        if self.device != device:
            self._reset(device)

        if self._sum_rows is None or self._sum_rows.size < sum_nnz:
            self._sum_rows = wp.empty(shape=(sum_nnz), dtype=int, device=self.device)
        if self._sum_cols is None or self._sum_cols.size < sum_nnz:
            self._sum_cols = wp.empty(shape=(sum_nnz), dtype=int, device=self.device)

        if self._old_y_values is None or self._old_y_values.size < y.nnz:
            self._old_y_values = wp.empty(shape=(y.nnz,), dtype=y.values.dtype, device=self.device)


def bsr_axpy(
    x: BsrMatrixOrExpression,
    y: Optional[BsrMatrix[BlockType[Rows, Cols, Scalar]]] = None,
    alpha: Scalar = 1.0,
    beta: Scalar = 1.0,
    masked: bool = False,
    work_arrays: Optional[bsr_axpy_work_arrays] = None,
) -> BsrMatrix[BlockType[Rows, Cols, Scalar]]:
    """
    Perform the sparse matrix addition ``y := alpha * X + beta * y`` on BSR matrices ``x`` and ``y`` and return ``y``.

    The ``x`` and ``y`` matrices are allowed to alias.

    Args:
        x: Read-only right-hand-side.
        y: Mutable left-hand-side. If ``y`` is not provided, it will be allocated and treated as zero.
        alpha: Uniform scaling factor for ``x``.
        beta: Uniform scaling factor for ``y``.
        masked: If ``True``, discard all blocks from ``x`` which are not
          existing non-zeros of ``y``.
        work_arrays: In most cases, this function will require the use of temporary storage.
          This storage can be reused across calls by passing an instance of
          :class:`bsr_axpy_work_arrays` in ``work_arrays``.
    """

    x, x_scale = _extract_matrix_and_scale(x)
    alpha *= x_scale

    if y is None:
        if masked:
            raise ValueError("Left-hand-side 'y' matrix must be provided for masked addition")

        # If not output matrix is provided, allocate it for convenience
        y = bsr_zeros(x.nrow, x.ncol, block_type=x.values.dtype, device=x.values.device)
        beta = 0.0

    x_nnz = x.nnz
    y_nnz = y.nnz

    # Handle easy cases first
    if beta == 0.0 or y_nnz == 0:
        bsr_assign(src=x, dest=y)
        return bsr_scale(y, alpha=alpha)

    if alpha == 0.0 or x_nnz == 0:
        return bsr_scale(y, alpha=beta)

    if not isinstance(alpha, y.scalar_type):
        alpha = y.scalar_type(alpha)
    if not isinstance(beta, y.scalar_type):
        beta = y.scalar_type(beta)

    if x == y:
        # Aliasing case
        return bsr_scale(y, alpha=alpha.value + beta.value)

    # General case

    if x.values.device != y.values.device:
        raise ValueError("All arguments must reside on the same device")

    if x.scalar_type != y.scalar_type or x.block_shape != y.block_shape:
        raise ValueError("Matrices must have the same block type")

    if x.nrow != y.nrow or x.ncol != y.ncol:
        raise ValueError("Matrices must have the same number of rows and columns")

    if work_arrays is None:
        work_arrays = bsr_axpy_work_arrays()

    sum_nnz = x_nnz + y_nnz
    device = y.values.device
    work_arrays._allocate(device, y, sum_nnz)

    wp.copy(work_arrays._sum_cols, y.columns, 0, 0, y_nnz)
    y.uncompress_rows(out=work_arrays._sum_rows)

    wp.copy(work_arrays._sum_cols, x.columns, y_nnz, 0, x_nnz)
    x.uncompress_rows(out=work_arrays._sum_rows[y_nnz:])

    # Save old y values before overwriting matrix
    wp.copy(dest=work_arrays._old_y_values, src=y.values, count=y_nnz)

    # Increase dest array sizes if needed
    if not masked:
        _bsr_ensure_fits(y, nnz=sum_nnz)

    from warp.context import runtime

    if device.is_cpu:
        native_func = runtime.core.bsr_matrix_from_triplets_float_host
    else:
        native_func = runtime.core.bsr_matrix_from_triplets_float_device

    old_y_nnz = y_nnz
    nnz_buf, nnz_event = y._nnz_transfer_buf_and_event()

    with wp.ScopedDevice(y.device):
        native_func(
            y.block_shape[0],
            y.block_shape[1],
            y.nrow,
            sum_nnz,
            ctypes.cast(work_arrays._sum_rows.ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(work_arrays._sum_cols.ptr, ctypes.POINTER(ctypes.c_int32)),
            0,
            False,
            masked,
            ctypes.cast(y.offsets.ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(y.columns.ptr, ctypes.POINTER(ctypes.c_int32)),
            0,
            ctypes.cast(nnz_buf.ptr, ctypes.POINTER(ctypes.c_int32)),
            nnz_event,
        )

    y.values.zero_()

    wp.launch(
        kernel=_bsr_axpy_add_block,
        device=device,
        dim=old_y_nnz,
        inputs=[
            0,
            beta,
            work_arrays._sum_rows,
            work_arrays._sum_cols,
            y.offsets,
            y.columns,
            work_arrays._old_y_values,
            y.values,
        ],
    )

    wp.launch(
        kernel=_bsr_axpy_add_block,
        device=device,
        dim=x_nnz,
        inputs=[
            old_y_nnz,
            alpha,
            work_arrays._sum_rows,
            work_arrays._sum_cols,
            y.offsets,
            y.columns,
            x.values,
            y.values,
        ],
    )

    return y


@wp.kernel(enable_backward=False)
def _bsr_mm_count_coeffs(
    y_ncol: int,
    z_nnz: int,
    x_offsets: wp.array(dtype=int),
    x_columns: wp.array(dtype=int),
    y_offsets: wp.array(dtype=int),
    y_columns: wp.array(dtype=int),
    row_min: wp.array(dtype=int),
    block_counts: wp.array(dtype=int),
):
    row = wp.tid()
    row_count = int(0)

    x_beg = x_offsets[row]
    x_end = x_offsets[row + 1]

    min_col = y_ncol
    max_col = int(0)

    for x_block in range(x_beg, x_end):
        x_col = x_columns[x_block]
        y_row_end = y_offsets[x_col + 1]
        y_row_beg = y_offsets[x_col]
        block_count = y_row_end - y_row_beg
        if block_count != 0:
            min_col = wp.min(y_columns[y_row_beg], min_col)
            max_col = wp.max(y_columns[y_row_end - 1], max_col)

        block_counts[x_block + 1] = block_count
        row_count += block_count

    if row_count > wp.max(0, max_col - min_col):
        row_min[row] = min_col
        block_counts[x_end] = max_col + 1 - min_col
        for x_block in range(x_beg, x_end - 1):
            block_counts[x_block + 1] = 0
    else:
        row_min[row] = -1

    if row == 0:
        block_counts[0] = z_nnz


@wp.kernel(enable_backward=False)
def _bsr_mm_list_coeffs(
    x_nrow: int,
    x_offsets: wp.array(dtype=int),
    x_columns: wp.array(dtype=int),
    y_offsets: wp.array(dtype=int),
    y_columns: wp.array(dtype=int),
    mm_row_min: wp.array(dtype=int),
    mm_offsets: wp.array(dtype=int),
    mm_rows: wp.array(dtype=int),
    mm_cols: wp.array(dtype=int),
):
    x_block = wp.tid()
    mm_block = mm_offsets[x_block]

    row = _bsr_row_index(x_offsets, x_nrow, x_block)
    if row == -1:
        return

    row_min_col = mm_row_min[row]
    if row_min_col != -1:
        x_col = x_columns[x_block]

        y_beg = y_offsets[x_col]
        y_end = y_offsets[x_col + 1]

        for y_block in range(y_beg, y_end):
            col = y_columns[y_block]
            mm_rows[mm_block + col - row_min_col] = row
            mm_cols[mm_block + col - row_min_col] = col

        return

    x_col = x_columns[x_block]
    y_beg = y_offsets[x_col]
    y_end = y_offsets[x_col + 1]
    for y_block in range(y_beg, y_end):
        mm_cols[mm_block] = y_columns[y_block]
        mm_rows[mm_block] = row
        mm_block += 1


@wp.kernel
def _bsr_mm_compute_values(
    alpha: Any,
    x_offsets: wp.array(dtype=int),
    x_columns: wp.array(dtype=int),
    x_values: wp.array(dtype=Any),
    y_offsets: wp.array(dtype=int),
    y_columns: wp.array(dtype=int),
    y_values: wp.array(dtype=Any),
    mm_row_count: int,
    mm_offsets: wp.array(dtype=int),
    mm_cols: wp.array(dtype=int),
    mm_values: wp.array(dtype=Any),
):
    mm_block = wp.tid()

    row = _bsr_row_index(mm_offsets, mm_row_count, mm_block)
    if row == -1:
        return

    col = mm_cols[mm_block]

    mm_val = mm_values.dtype(type(alpha)(0.0))

    x_beg = x_offsets[row]
    x_end = x_offsets[row + 1]
    for x_block in range(x_beg, x_end):
        x_col = x_columns[x_block]
        y_block = _bsr_block_index(x_col, col, y_offsets, y_columns)
        if y_block != -1:
            mm_val += x_values[x_block] * y_values[y_block]

    mm_values[mm_block] += alpha * mm_val


class bsr_mm_work_arrays:
    """Opaque structure for persisting :func:`bsr_mm` temporary work buffers across calls."""

    def __init__(self):
        self._reset(None)

    def _reset(self, device):
        self.device = device
        self._mm_row_min = None
        self._mm_block_counts = None
        self._mm_rows = None
        self._mm_cols = None
        self._old_z_values = None
        self._old_z_offsets = None
        self._old_z_columns = None
        self._mm_nnz = 0

    def _allocate_stage_1(self, device, x_nnz: int, z: BsrMatrix, beta: float, z_aliasing: bool):
        if self.device != device:
            self._reset(device)

        # Allocations that do not depend on any computation
        z_nnz = z.nnz_sync()
        self._copied_z_nnz = z_nnz if beta != 0.0 or z_aliasing else 0

        if self._mm_row_min is None or self._mm_block_counts.size < z.nrow + 1:
            self._mm_row_min = wp.empty(shape=(z.nrow + 1,), dtype=int, device=self.device)
        if self._mm_block_counts is None or self._mm_block_counts.size < x_nnz + 1:
            self._mm_block_counts = wp.empty(shape=(x_nnz + 1,), dtype=int, device=self.device)

        if self._copied_z_nnz > 0:
            if self._old_z_values is None or self._old_z_values.size < self._copied_z_nnz:
                self._old_z_values = wp.empty(shape=(self._copied_z_nnz,), dtype=z.values.dtype, device=self.device)

        if z_aliasing:
            if self._old_z_columns is None or self._old_z_columns.size < z_nnz:
                self._old_z_columns = wp.empty(shape=(z_nnz,), dtype=z.columns.dtype, device=self.device)
            if self._old_z_offsets is None or self._old_z_offsets.size < z.nrow + 1:
                self._old_z_offsets = wp.empty(shape=(z.nrow + 1,), dtype=z.offsets.dtype, device=self.device)

    def _allocate_stage_2(self, mm_nnz: int):
        # Allocations that depend on unmerged nnz estimate
        self._mm_nnz = mm_nnz
        if self._mm_rows is None or self._mm_rows.size < mm_nnz:
            self._mm_rows = wp.empty(shape=(mm_nnz,), dtype=int, device=self.device)
        if self._mm_cols is None or self._mm_cols.size < mm_nnz:
            self._mm_cols = wp.empty(shape=(mm_nnz,), dtype=int, device=self.device)


def bsr_mm(
    x: BsrMatrixOrExpression[BlockType[Rows, Any, Scalar]],
    y: BsrMatrixOrExpression[BlockType[Any, Cols, Scalar]],
    z: Optional[BsrMatrix[BlockType[Rows, Cols, Scalar]]] = None,
    alpha: Scalar = 1.0,
    beta: Scalar = 0.0,
    masked: bool = False,
    work_arrays: Optional[bsr_mm_work_arrays] = None,
    reuse_topology: bool = False,
) -> BsrMatrix[BlockType[Rows, Cols, Scalar]]:
    """
    Perform the sparse matrix-matrix multiplication ``z := alpha * x @ y + beta * z`` on BSR matrices ``x``, ``y`` and ``z``, and return ``z``.

    The ``x``, ``y`` and ``z`` matrices are allowed to alias.
    If the matrix ``z`` is not provided as input, it will be allocated and treated as zero.

    Args:
        x: Read-only left factor of the matrix-matrix product.
        y: Read-only right factor of the matrix-matrix product.
        z: Mutable left-hand-side. If ``z`` is not provided, it will be allocated and treated as zero.
        alpha: Uniform scaling factor for the ``x @ y`` product
        beta: Uniform scaling factor for ``z``
        masked: If ``True``, ignore all blocks from ``x @ y`` which are not existing non-zeros of ``y``
        work_arrays: In most cases, this function will require the use of temporary storage.
          This storage can be reused across calls by passing an instance of
          :class:`bsr_mm_work_arrays` in ``work_arrays``.
        reuse_topology: If ``True``, reuse the product topology information
          stored in ``work_arrays`` rather than recompute it from scratch.
          The matrices ``x``, ``y`` and ``z`` must be structurally similar to
          the previous call in which ``work_arrays`` were populated.
          This is necessary for ``bsr_mm`` to be captured in a CUDA graph.
    """

    x, x_scale = _extract_matrix_and_scale(x)
    alpha *= x_scale
    y, y_scale = _extract_matrix_and_scale(y)
    alpha *= y_scale

    if z is None:
        if masked:
            raise ValueError("Left-hand-side 'z' matrix must be provided for masked multiplication")

        # If not output matrix is provided, allocate it for convenience
        z_block_shape = (x.block_shape[0], y.block_shape[1])
        if z_block_shape == (1, 1):
            z_block_type = x.scalar_type
        else:
            z_block_type = wp.mat(shape=z_block_shape, dtype=x.scalar_type)
        z = bsr_zeros(x.nrow, y.ncol, block_type=z_block_type, device=x.values.device)
        beta = 0.0

    if x.values.device != y.values.device or x.values.device != z.values.device:
        raise ValueError("All arguments must reside on the same device")

    if x.scalar_type != y.scalar_type or x.scalar_type != z.scalar_type:
        raise ValueError("Matrices must have the same scalar type")

    if (
        x.block_shape[0] != z.block_shape[0]
        or y.block_shape[1] != z.block_shape[1]
        or x.block_shape[1] != y.block_shape[0]
    ):
        raise ValueError("Incompatible block sizes for matrix multiplication")

    if x.nrow != z.nrow or z.ncol != y.ncol or x.ncol != y.nrow:
        raise ValueError("Incompatible number of rows/columns for matrix multiplication")

    device = z.values.device

    if alpha == 0.0 or x.nnz == 0 or y.nnz == 0:
        # Easy case
        return bsr_scale(z, beta)

    z_aliasing = z == x or z == y

    if masked:
        # no need to copy z, scale in-place
        copied_z_nnz = 0
        mm_nnz = z.nnz

        if z_aliasing:
            raise ValueError("`masked=True` is not supported for aliased inputs")

        if beta == 0.0:
            # do not bsr_scale(0), this would not preserve topology
            z.values.zero_()
        else:
            bsr_scale(z, beta)
    elif reuse_topology:
        if work_arrays is None:
            raise ValueError("`work_arrays` must not be ``None`` in order to reuse matrix-matrix product topology")

        copied_z_nnz = work_arrays._copied_z_nnz
        mm_nnz = work_arrays._mm_nnz
    else:
        if device.is_capturing:
            raise RuntimeError("`bsr_mm` requires `reuse_topology=True` for use in graph capture")

        if work_arrays is None:
            work_arrays = bsr_mm_work_arrays()

        work_arrays._allocate_stage_1(device, x.nnz, z, beta, z_aliasing)
        copied_z_nnz = work_arrays._copied_z_nnz

        # Prefix sum of number of (unmerged) mm blocks per row
        work_arrays._mm_block_counts.zero_()
        wp.launch(
            kernel=_bsr_mm_count_coeffs,
            device=device,
            dim=z.nrow,
            inputs=[
                y.ncol,
                copied_z_nnz,
                x.offsets,
                x.columns,
                y.offsets,
                y.columns,
                work_arrays._mm_row_min,
                work_arrays._mm_block_counts,
            ],
        )
        warp.utils.array_scan(work_arrays._mm_block_counts, work_arrays._mm_block_counts)

        # Get back total counts on host -- we need a synchronization here
        # Use pinned buffer from z, we are going to need it later anyway
        nnz_buf, _ = z._nnz_transfer_buf_and_event()
        stream = wp.get_stream(device) if device.is_cuda else None
        wp.copy(dest=nnz_buf, src=work_arrays._mm_block_counts, src_offset=x.nnz, count=1, stream=stream)
        if device.is_cuda:
            wp.synchronize_stream(stream)
        mm_nnz = int(nnz_buf.numpy()[0])

        if mm_nnz == copied_z_nnz:
            # x@y = 0
            return bsr_scale(z, beta)

        work_arrays._allocate_stage_2(mm_nnz)

        # If z has a non-zero scale, save current data before overwriting it
        if copied_z_nnz > 0:
            # Copy z row and column indices
            wp.copy(dest=work_arrays._mm_cols, src=z.columns, count=copied_z_nnz)
            z.uncompress_rows(out=work_arrays._mm_rows)
            if z_aliasing:
                # If z is aliasing with x or y, need to save topology as well
                wp.copy(src=z.columns, dest=work_arrays._old_z_columns, count=copied_z_nnz)
                wp.copy(src=z.offsets, dest=work_arrays._old_z_offsets, count=z.nrow + 1)

        # Fill unmerged mm blocks rows and columns
        work_arrays._mm_rows[copied_z_nnz:].fill_(-1)
        wp.launch(
            kernel=_bsr_mm_list_coeffs,
            device=device,
            dim=x.nnz,
            inputs=[
                x.nrow,
                x.offsets,
                x.columns,
                y.offsets,
                y.columns,
                work_arrays._mm_row_min,
                work_arrays._mm_block_counts,
                work_arrays._mm_rows,
                work_arrays._mm_cols,
            ],
        )

    alpha = z.scalar_type(alpha)
    beta = z.scalar_type(beta)

    if copied_z_nnz > 0:
        # Save current z values in temporary buffer
        wp.copy(src=z.values, dest=work_arrays._old_z_values, count=copied_z_nnz)

    if not masked:
        # Increase dest array size if needed
        if z.columns.shape[0] < mm_nnz:
            z.columns = wp.empty(shape=(mm_nnz,), dtype=int, device=device)

        from warp.context import runtime

        if device.is_cpu:
            native_func = runtime.core.bsr_matrix_from_triplets_float_host
        else:
            native_func = runtime.core.bsr_matrix_from_triplets_float_device

        nnz_buf, nnz_event = z._nnz_transfer_buf_and_event()

        with wp.ScopedDevice(z.device):
            native_func(
                z.block_shape[0],
                z.block_shape[1],
                z.nrow,
                mm_nnz,
                ctypes.cast(work_arrays._mm_rows.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(work_arrays._mm_cols.ptr, ctypes.POINTER(ctypes.c_int32)),
                0,
                False,
                masked,
                ctypes.cast(z.offsets.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(z.columns.ptr, ctypes.POINTER(ctypes.c_int32)),
                0,
                ctypes.cast(nnz_buf.ptr, ctypes.POINTER(ctypes.c_int32)),
                nnz_event,
            )

        # Resize z to fit mm result if necessary
        # If we are not reusing the product topology, this needs another synchronization
        if not reuse_topology:
            work_arrays.result_nnz = z.nnz_sync()

        _bsr_ensure_fits(z, nnz=work_arrays.result_nnz)
        z.values.zero_()

        if copied_z_nnz > 0:
            # Add back original z values
            wp.launch(
                kernel=_bsr_axpy_add_block,
                device=device,
                dim=copied_z_nnz,
                inputs=[
                    0,
                    beta,
                    work_arrays._mm_rows,
                    work_arrays._mm_cols,
                    z.offsets,
                    z.columns,
                    work_arrays._old_z_values,
                    z.values,
                ],
            )

    # Add mm blocks to z values
    if (type_is_matrix(x.values.dtype) or type_is_matrix(y.values.dtype)) and not (type_is_matrix(z.values.dtype)):
        # Result block type is scalar, but operands are matrices
        # Cast result to (1x1) matrix to perform multiplication
        mm_values = z.values.view(wp.mat(shape=(1, 1), dtype=z.scalar_type))
    else:
        mm_values = z.values

    wp.launch(
        kernel=_bsr_mm_compute_values,
        device=device,
        dim=z.nnz,
        inputs=[
            alpha,
            work_arrays._old_z_offsets if x == z else x.offsets,
            work_arrays._old_z_columns if x == z else x.columns,
            work_arrays._old_z_values if x == z else x.values,
            work_arrays._old_z_offsets if y == z else y.offsets,
            work_arrays._old_z_columns if y == z else y.columns,
            work_arrays._old_z_values if y == z else y.values,
            z.nrow,
            z.offsets,
            z.columns,
            mm_values,
        ],
    )

    return z


@wp.kernel
def _bsr_mv_kernel(
    alpha: Any,
    A_offsets: wp.array(dtype=int),
    A_columns: wp.array(dtype=int),
    A_values: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    beta: Any,
    y: wp.array(dtype=Any),
):
    row = wp.tid()

    # zero-initialize with type of y elements
    scalar_zero = type(alpha)(0)
    v = y.dtype(scalar_zero)

    if alpha != scalar_zero:
        beg = A_offsets[row]
        end = A_offsets[row + 1]
        for block in range(beg, end):
            v += A_values[block] * x[A_columns[block]]
        v *= alpha

    if beta != scalar_zero:
        v += beta * y[row]

    y[row] = v


@wp.kernel
def _bsr_mv_transpose_kernel(
    alpha: Any,
    A_offsets: wp.array(dtype=int),
    A_columns: wp.array(dtype=int),
    A_values: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
):
    row = wp.tid()
    beg = A_offsets[row]
    end = A_offsets[row + 1]
    xr = alpha * x[row]
    for block in range(beg, end):
        v = wp.transpose(A_values[block]) * xr
        wp.atomic_add(y, A_columns[block], v)


def _vec_array_view(array: wp.array, dtype: type, expected_scalar_count: int) -> wp.array:
    # cast a 1d or 2d array to a 1d array with the target dtype, adjusting shape as required

    scalar_count = array.size * type_length(array.dtype)
    if scalar_count != expected_scalar_count:
        raise ValueError(f"Invalid array scalar size, expected {expected_scalar_count}, got {scalar_count}")

    if array.ndim == 1 and types_equal(array.dtype, dtype):
        return array

    if type_scalar_type(array.dtype) != type_scalar_type(dtype):
        raise ValueError(f"Incompatible scalar types, {type_repr(array.dtype)} vs {type_repr(dtype)}")

    if array.ndim > 2:
        raise ValueError(f"Incompatible array number of dimensions {array.ndim}")

    if not array.is_contiguous:
        raise ValueError("Array must be contiguous")

    vec_length = type_length(dtype)
    vec_count = scalar_count // vec_length
    if vec_count * vec_length != scalar_count:
        raise ValueError(
            f"Array of shape {array.shape} and type {type_repr(array.dtype)} cannot be reshaped to an array of type {type_repr(dtype)}"
        )

    def vec_view(array):
        return wp.array(
            data=None,
            ptr=array.ptr,
            capacity=array.capacity,
            device=array.device,
            dtype=dtype,
            shape=vec_count,
            grad=None if array.grad is None else vec_view(array.grad),
        )

    view = vec_view(array)
    view._ref = array
    return view


def bsr_mv(
    A: BsrMatrixOrExpression[BlockType[Rows, Cols, Scalar]],
    x: "Array[Vector[Cols, Scalar] | Scalar]",
    y: Optional["Array[Vector[Rows, Scalar] | Scalar]"] = None,
    alpha: Scalar = 1.0,
    beta: Scalar = 0.0,
    transpose: bool = False,
    work_buffer: Optional["Array[Vector[Rows, Scalar] | Scalar]"] = None,
) -> "Array[Vector[Rows, Scalar] | Scalar]":
    """Perform the sparse matrix-vector product ``y := alpha * A * x + beta * y`` and return ``y``.

    The ``x`` and ``y`` vectors are allowed to alias.

    Args:
        A: Read-only, left matrix factor of the matrix-vector product.
        x: Read-only, right vector factor of the matrix-vector product.
        y: Mutable left-hand-side. If ``y`` is not provided, it will be allocated and treated as zero.
        alpha: Uniform scaling factor for ``x``. If zero, ``x`` will not be read and may be left uninitialized.
        beta: Uniform scaling factor for ``y``. If zero, ``y`` will not be read and may be left uninitialized.
        transpose: If ``True``, use the transpose of the matrix ``A``. In this case the result is **non-deterministic**.
        work_buffer: Temporary storage is required if and only if ``x`` and ``y`` are the same vector.
          If provided, the ``work_buffer`` array will be used for this purpose,
          otherwise a temporary allocation will be performed.
    """

    A, A_scale = _extract_matrix_and_scale(A)
    alpha *= A_scale

    if transpose:
        block_shape = A.block_shape[1], A.block_shape[0]
        nrow, ncol = A.ncol, A.nrow
    else:
        block_shape = A.block_shape
        nrow, ncol = A.nrow, A.ncol

    if y is None:
        # If no output array is provided, allocate one for convenience
        y_vec_len = block_shape[0]
        y_dtype = A.scalar_type if y_vec_len == 1 else wp.vec(length=y_vec_len, dtype=A.scalar_type)
        y = wp.empty(shape=(nrow,), device=A.values.device, dtype=y_dtype)
        beta = 0.0

    alpha = A.scalar_type(alpha)
    beta = A.scalar_type(beta)

    if A.values.device != x.device or A.values.device != y.device:
        raise ValueError("A, x, and y must reside on the same device")

    if x.ptr == y.ptr:
        # Aliasing case, need temporary storage
        if work_buffer is None:
            work_buffer = wp.empty_like(y)
        elif work_buffer.size < y.size:
            raise ValueError(f"Work buffer size is insufficient, needs to be at least {y.size}")
        elif not types_equal(work_buffer.dtype, y.dtype):
            raise ValueError(f"Work buffer must have same data type as y, {type_repr(y.dtype)}")

        # Save old y values before overwriting vector
        wp.copy(dest=work_buffer, src=y, count=y.size)
        x = work_buffer

    # Promote scalar vectors to length-1 vecs and conversely
    if type_is_matrix(A.values.dtype):
        x_dtype = wp.vec(length=block_shape[1], dtype=A.scalar_type)
        y_dtype = wp.vec(length=block_shape[0], dtype=A.scalar_type)
    else:
        x_dtype = A.scalar_type
        y_dtype = A.scalar_type

    try:
        x_view = _vec_array_view(x, x_dtype, expected_scalar_count=ncol * block_shape[1])
    except ValueError as err:
        raise ValueError("Incompatible 'x' vector for bsr_mv") from err
    try:
        y_view = _vec_array_view(y, y_dtype, expected_scalar_count=nrow * block_shape[0])
    except ValueError as err:
        raise ValueError("Incompatible 'y' vector for bsr_mv") from err

    if transpose:
        if beta.value == 0.0:
            y.zero_()
        elif beta.value != 1.0:
            wp.launch(
                kernel=_bsr_scale_kernel,
                device=y.device,
                dim=y.shape[0],
                inputs=[beta, y],
            )
        if alpha.value != 0.0:
            wp.launch(
                kernel=_bsr_mv_transpose_kernel,
                device=A.values.device,
                dim=ncol,
                inputs=[alpha, A.offsets, A.columns, A.values, x_view, y_view],
            )
    else:
        wp.launch(
            kernel=_bsr_mv_kernel,
            device=A.values.device,
            dim=nrow,
            inputs=[alpha, A.offsets, A.columns, A.values, x_view, beta, y_view],
        )

    return y
