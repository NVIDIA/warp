from typing import Any, Generic, Optional, Tuple, TypeVar, Union

import warp as wp
import warp.types
import warp.utils
from warp.types import Array, Cols, Rows, Scalar, Vector

# typing hints

_BlockType = TypeVar("BlockType")


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
        nrow (int): Number of rows of blocks
        ncol (int): Number of columns of blocks
        nnz (int):  Number of non-zero blocks: must be equal to ``offsets[nrow-1]``, cached on host for convenience
        offsets (Array[int]): Array of size at least ``1 + nrows`` such that the start and end indices of the blocks of row ``r`` are ``offsets[r]`` and ``offsets[r+1]``, respectively.
        columns (Array[int]): Array of size at least equal to ``nnz`` containing block column indices
        values (Array[BlockType]): Array of size at least equal to ``nnz`` containing block values
    """

    @property
    def scalar_type(self) -> Scalar:
        """Scalar type for individual block coefficients. For CSR matrices, this is the same as the block type"""
        return warp.types.type_scalar_type(self.values.dtype)

    @property
    def block_shape(self) -> Tuple[int, int]:
        """Shape of the individual blocks"""
        return getattr(self.values.dtype, "_shape_", (1, 1))

    @property
    def block_size(self) -> int:
        """Size of the individual blocks, i.e. number of rows per block times number of columns per block"""
        return warp.types.type_length(self.values.dtype)

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the matrix, i.e. number of rows/columns of blocks times number of rows/columns per block"""
        block_shape = self.block_shape
        return (self.nrow * block_shape[0], self.ncol * block_shape[1])

    @property
    def dtype(self) -> type:
        """Data type for individual block values"""
        return self.values.dtype

    @property
    def device(self) -> wp.context.Device:
        """Device on which offsets, columns and values are allocated -- assumed to be the same for all three arrays"""
        return self.values.device


def bsr_matrix_t(dtype: BlockType):
    dtype = wp.types.type_to_warp(dtype)

    if not warp.types.type_is_matrix(dtype) and dtype not in warp.types.scalar_types:
        raise ValueError(
            f"BsrMatrix block type must be either warp matrix or scalar; got {warp.types.type_repr(dtype)}"
        )

    class BsrMatrixTyped(BsrMatrix):
        nrow: int
        """Number of rows of blocks"""
        ncol: int
        """Number of columns of blocks"""
        nnz: int
        """Number of non-zero blocks: equal to offsets[-1], cached on host for convenience"""
        offsets: wp.array(dtype=int)
        """Array of size at least 1 + nrows"""
        columns: wp.array(dtype=int)
        """Array of size at least equal to nnz"""
        values: wp.array(dtype=dtype)

    module = wp.get_module(BsrMatrix.__module__)

    if hasattr(dtype, "_shape_"):
        type_str = f"{warp.types.type_scalar_type(dtype).__name__}_{dtype._shape_[0]}_{dtype._shape_[1]}"
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
    """
    Constructs and returns an empty BSR or CSR matrix with the given shape

    Args:
        bsr: The BSR or CSR matrix to set to zero
        rows_of_blocks: Number of rows of blocks
        cols_of_blocks: Number of columns of blocks
        block_type: Type of individual blocks. For CSR matrices, this should be a scalar type;
                    for BSR matrices, this should be a matrix type (e.g. from :func:`warp.mat`)
        device: Device on which to allocate the matrix arrays
    """

    bsr = bsr_matrix_t(block_type)()

    bsr.nrow = rows_of_blocks
    bsr.ncol = cols_of_blocks
    bsr.nnz = 0
    bsr.columns = wp.empty(shape=(0,), dtype=int, device=device)
    bsr.values = wp.empty(shape=(0,), dtype=block_type, device=device)
    bsr.offsets = wp.zeros(shape=(bsr.nrow + 1,), dtype=int, device=device)

    return bsr


def _bsr_ensure_fits(bsr: BsrMatrix, nrow: int = None, nnz: int = None):
    if nrow is None:
        nrow = bsr.nrow
    if nnz is None:
        nnz = bsr.nnz

    if bsr.offsets.size < nrow + 1:
        bsr.offsets = wp.empty(shape=(nrow + 1,), dtype=int, device=bsr.offsets.device)
    if bsr.columns.size < nnz:
        bsr.columns = wp.empty(shape=(nnz,), dtype=int, device=bsr.columns.device)
    if bsr.values.size < nnz:
        bsr.values = wp.empty(shape=(nnz,), dtype=bsr.values.dtype, device=bsr.values.device)


def bsr_set_zero(bsr: BsrMatrix, rows_of_blocks: Optional[int] = None, cols_of_blocks: Optional[int] = None):
    """
    Sets a BSR matrix to zero, possibly changing its size

    Args:
        bsr: The BSR or CSR matrix to set to zero
        rows_of_blocks: If not ``None``, the new number of rows of blocks
        cols_of_blocks: If not ``None``, the new number of columns of blocks
    """

    if rows_of_blocks is not None:
        bsr.nrow = rows_of_blocks
    if cols_of_blocks is not None:
        bsr.ncol = cols_of_blocks
    bsr.nnz = 0
    _bsr_ensure_fits(bsr)
    bsr.offsets.zero_()


def bsr_set_from_triplets(
    dest: BsrMatrix[BlockType[Rows, Cols, Scalar]],
    rows: "Array[int]",
    columns: "Array[int]",
    values: "Array[Union[Scalar, BlockType[Rows, Cols, Scalar]]]",
):
    """
    Fills a BSR matrix with values defined by coordinate-oriented (COO) triplets, discarding existing blocks.

    The first dimension of the three input arrays must match, and determines the number of non-zeros in the constructed matrix.

    Args:
        dest: Sparse matrix to populate
        rows: Row index for each non-zero
        columns: Columns index for each non-zero
        values: Block values for each non-zero. Must be either a one-dimensional array with data type identical
          to the `dest` matrix's block type, or a 3d array with data type equal to the `dest` matrix's scalar type.
    """

    if values.device != columns.device or values.device != rows.device or values.device != dest.values.device:
        raise ValueError("All arguments must reside on the same device")

    if values.shape[0] != rows.shape[0] or values.shape[0] != columns.shape[0]:
        raise ValueError("All triplet arrays must have the same length")

    # Accept either array1d(dtype) or contiguous array3d(scalar_type) as values
    if values.ndim == 1:
        if values.dtype != dest.values.dtype:
            raise ValueError("Values array type must correspond to that of dest matrix")
    elif values.ndim == 3:
        if values.shape[1:] != dest.block_shape:
            raise ValueError(
                f"Last two dimensions in values array ({values.shape[1:]}) should correspond to matrix block shape {(dest.block_shape)})"
            )

        if warp.types.type_scalar_type(values.dtype) != dest.scalar_type:
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

    dest.nnz = native_func(
        dest.block_shape[0],
        dest.block_shape[1],
        dest.nrow,
        nnz,
        rows.ptr,
        columns.ptr,
        values.ptr,
        dest.offsets.ptr,
        dest.columns.ptr,
        dest.values.ptr,
    )


def bsr_assign(dest: BsrMatrix[BlockType[Rows, Cols, Scalar]], src: BsrMatrix[BlockType[Rows, Cols, Any]]):
    """Copies the content of the `src` matrix to `dest`, casting the block values if the two matrices use distinct scalar types."""

    if dest.values.device != src.values.device:
        raise ValueError("Source and destination matrices must reside on the same device")

    if dest.block_shape != src.block_shape:
        raise ValueError("Source and destination matrices must have the same block shape")

    dest.nrow = src.nrow
    dest.ncol = src.ncol
    dest.nnz = src.nnz

    _bsr_ensure_fits(dest)

    wp.copy(dest=dest.offsets, src=src.offsets, count=src.nrow + 1)
    if src.nnz > 0:
        wp.copy(dest=dest.columns, src=src.columns, count=src.nnz)
        warp.utils.array_cast(out_array=dest.values, in_array=src.values, count=src.nnz)


def bsr_copy(A: BsrMatrix, scalar_type: Optional[Scalar] = None):
    """Returns a copy of matrix ``A``, possibly changing its scalar type.

    Args:
       scalar_type: If provided, the returned matrix will use this scalar type instead of the one from `A`.
    """
    if scalar_type is None:
        block_type = A.values.dtype
    elif A.block_shape == (1, 1):
        block_type = scalar_type
    else:
        block_type = wp.types.matrix(shape=A.block_shape, dtype=scalar_type)

    copy = bsr_zeros(rows_of_blocks=A.nrow, cols_of_blocks=A.ncol, block_type=block_type, device=A.values.device)
    bsr_assign(dest=copy, src=A)
    return copy


def bsr_set_transpose(dest: BsrMatrix[BlockType[Cols, Rows, Scalar]], src: BsrMatrix[BlockType[Rows, Cols, Scalar]]):
    """Assigns the transposed matrix `src` to matrix `dest`"""

    if dest.values.device != src.values.device:
        raise ValueError("All arguments must reside on the same device")

    if dest.scalar_type != src.scalar_type:
        raise ValueError("All arguments must have the same scalar type")

    transpose_block_shape = src.block_shape[::-1]

    if dest.block_shape != transpose_block_shape:
        raise ValueError(f"Destination block shape must be {transpose_block_shape}")

    dest.nrow = src.ncol
    dest.ncol = src.nrow
    dest.nnz = src.nnz

    if src.nnz == 0:
        return

    # Increase dest array sizes if needed
    _bsr_ensure_fits(dest)

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

    native_func(
        src.block_shape[0],
        src.block_shape[1],
        src.nrow,
        src.ncol,
        src.nnz,
        src.offsets.ptr,
        src.columns.ptr,
        src.values.ptr,
        dest.offsets.ptr,
        dest.columns.ptr,
        dest.values.ptr,
    )


def bsr_transposed(A: BsrMatrix):
    """Returns a copy of the transposed matrix `A`"""

    if A.block_shape == (1, 1):
        block_type = A.values.dtype
    else:
        block_type = wp.types.matrix(shape=A.block_shape[::-1], dtype=A.scalar_type)

    transposed = bsr_zeros(rows_of_blocks=A.ncol, cols_of_blocks=A.nrow, block_type=block_type, device=A.values.device)
    bsr_set_transpose(dest=transposed, src=A)
    return transposed


@wp.kernel
def _bsr_get_diag_kernel(
    A_offsets: wp.array(dtype=int),
    A_columns: wp.array(dtype=int),
    A_values: wp.array(dtype=Any),
    out: wp.array(dtype=Any),
):
    row = wp.tid()
    beg = A_offsets[row]
    end = A_offsets[row + 1]

    diag = wp.lower_bound(A_columns, beg, end, row)
    if diag < end:
        if A_columns[diag] == row:
            out[row] = A_values[diag]


def bsr_get_diag(A: BsrMatrix[_BlockType], out: "Optional[Array[BlockType]]" = None) -> "Array[BlockType]":
    """Returns the array of blocks that constitute the diagonal of a sparse matrix.

    Args:
        A: the sparse matrix from which to extract the diagonal
        out: if provided, the array into which to store the diagonal blocks
    """

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
        kernel=_bsr_get_diag_kernel, dim=dim, device=A.values.device, inputs=[A.offsets, A.columns, A.values, out]
    )

    return out


@wp.kernel
def _bsr_set_diag_kernel(
    diag: wp.array(dtype=Any),
    A_offsets: wp.array(dtype=int),
    A_columns: wp.array(dtype=int),
    A_values: wp.array(dtype=Any),
):
    row = wp.tid()
    A_offsets[row + 1] = row + 1
    A_columns[row] = row
    A_values[row] = diag[row]

    if row == 0:
        A_offsets[0] = 0


@wp.kernel
def _bsr_set_diag_constant_kernel(
    diag_value: Any,
    A_offsets: wp.array(dtype=int),
    A_columns: wp.array(dtype=int),
    A_values: wp.array(dtype=Any),
):
    row = wp.tid()
    A_offsets[row + 1] = row + 1
    A_columns[row] = row
    A_values[row] = diag_value

    if row == 0:
        A_offsets[0] = 0


def bsr_set_diag(
    A: BsrMatrix[BlockType],
    diag: "Union[BlockType, Array[BlockType]]",
    rows_of_blocks: Optional[int] = None,
    cols_of_blocks: Optional[int] = None,
):
    """Sets `A` as a block-diagonal matrix

    Args:
        A: the sparse matrix to modify
        diag: Either a warp array of type ``A.values.dtype``, in which case each element will define one block of the diagonal,
              or a constant value of type ``A.values.dtype``, in which case it will get assigned to all diagonal blocks.
        rows_of_blocks: If not ``None``, the new number of rows of blocks
        cols_of_blocks: If not ``None``, the new number of columns of blocks

    The shape of the matrix will be defined one of the following, in that order:
      - `rows_of_blocks` and `cols_of_blocks`, if provided. If only one is given, the second is assumed equal.
      - the first dimension of `diag`, if `diag` is an array
      - the current dimensions of `A` otherwise
    """

    if rows_of_blocks is None and cols_of_blocks is not None:
        rows_of_blocks = cols_of_blocks
    if cols_of_blocks is None and rows_of_blocks is not None:
        cols_of_blocks = rows_of_blocks

    if warp.types.is_array(diag):
        if rows_of_blocks is None:
            rows_of_blocks = diag.shape[0]
            cols_of_blocks = diag.shape[0]

    if rows_of_blocks is not None:
        A.nrow = rows_of_blocks
        A.ncol = cols_of_blocks

    A.nnz = min(A.nrow, A.ncol)
    _bsr_ensure_fits(A)

    if warp.types.is_array(diag):
        wp.launch(
            kernel=_bsr_set_diag_kernel,
            dim=A.nnz,
            device=A.values.device,
            inputs=[diag, A.offsets, A.columns, A.values],
        )
    else:
        if not warp.types.type_is_value(type(diag)):
            # Cast to launchable type
            diag = A.values.dtype(diag)
        wp.launch(
            kernel=_bsr_set_diag_constant_kernel,
            dim=A.nnz,
            device=A.values.device,
            inputs=[diag, A.offsets, A.columns, A.values],
        )


def bsr_diag(
    diag: "Union[BlockType, Array[BlockType]]",
    rows_of_blocks: Optional[int] = None,
    cols_of_blocks: Optional[int] = None,
) -> BsrMatrix["BlockType"]:
    """Creates and returns a block-diagonal BSR matrix from an given block value or array of block values.

    Args:
        diag: Either a warp array of type ``A.values.dtype``, in which case each element will define one block of the diagonal,
              or a constant value of type ``A.values.dtype``, in which case it will get assigned to all diagonal blocks.
        rows_of_blocks: If not ``None``, the new number of rows of blocks
        cols_of_blocks: If not ``None``, the new number of columns of blocks

    The shape of the matrix will be defined one of the following, in that order:
      - `rows_of_blocks` and `cols_of_blocks`, if provided. If only one is given, the second is assumed equal.
      - the first dimension of `diag`, if `diag` is an array
    """

    if rows_of_blocks is None and cols_of_blocks is not None:
        rows_of_blocks = cols_of_blocks
    if cols_of_blocks is None and rows_of_blocks is not None:
        cols_of_blocks = rows_of_blocks

    if warp.types.is_array(diag):
        if rows_of_blocks is None:
            rows_of_blocks = diag.shape[0]
            cols_of_blocks = diag.shape[0]

        A = bsr_zeros(
            rows_of_blocks,
            cols_of_blocks,
            block_type=diag.dtype,
            device=diag.device,
        )
    else:
        if rows_of_blocks is None:
            raise ValueError(
                "rows_of_blocks and/or cols_of_blocks must be provided for constructing a diagonal matrix with uniform diagonal"
            )

        block_type = type(diag)
        if not warp.types.type_is_matrix(block_type) and len(getattr(diag, "shape", ())) == 2:
            block_type = wp.mat(shape=diag.shape, dtype=diag.dtype)

        A = bsr_zeros(
            rows_of_blocks,
            cols_of_blocks,
            block_type=block_type,
        )

    bsr_set_diag(A, diag)
    return A


def bsr_set_identity(A: BsrMatrix, rows_of_blocks: Optional[int] = None):
    """Sets `A` as the identity matrix

    Args:
        A: the sparse matrix to modify
        rows_of_blocks: if provided, the matrix will be resized as a square matrix with `rows_of_blocks` rows and columns.
    """

    if A.block_shape == (1, 1):
        identity = A.scalar_type(1.0)
    else:
        from numpy import eye

        identity = eye(A.block_shape[0])

    bsr_set_diag(A, diag=identity, rows_of_blocks=rows_of_blocks, cols_of_blocks=rows_of_blocks)


def bsr_identity(
    rows_of_blocks: int, block_type: BlockType[Rows, Rows, Scalar], device: wp.context.Devicelike = None
) -> BsrMatrix[BlockType[Rows, Rows, Scalar]]:
    """Creates and returns a square identity matrix.

    Args:
        rows_of_blocks: Number of rows and columns of blocks in the created matrix.
        block_type: Block type for the newly created matrix -- must be square
        device: Device onto which to allocate the data arrays
    """
    A = bsr_zeros(rows_of_blocks=rows_of_blocks, cols_of_blocks=rows_of_blocks, block_type=block_type, device=device)
    bsr_set_identity(A)
    return A


@wp.kernel
def _bsr_scale_kernel(
    alpha: Any,
    values: wp.array(dtype=Any),
):
    values[wp.tid()] = alpha * values[wp.tid()]


def bsr_scale(x: BsrMatrix, alpha: Scalar) -> BsrMatrix:
    """
    Performs the operation ``x := alpha * x`` on BSR matrix `x` and returns `x`
    """

    if alpha != 1.0 and x.nnz > 0:
        if alpha == 0.0:
            bsr_set_zero(x)
        else:
            if not isinstance(alpha, x.scalar_type):
                alpha = x.scalar_type(alpha)

            wp.launch(kernel=_bsr_scale_kernel, dim=x.nnz, device=x.values.device, inputs=[alpha, x.values])

    return x


@wp.kernel
def _bsr_get_block_row(dest_offset: int, bsr_offsets: wp.array(dtype=int), rows: wp.array(dtype=int)):
    i = wp.tid()

    row = wp.lower_bound(bsr_offsets, i + 1) - 1
    rows[dest_offset + i] = row


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
    beg = dst_offsets[row]
    end = dst_offsets[row + 1]

    block = wp.lower_bound(dst_columns, beg, end, col)

    dst_values[block] = dst_values[block] + scale * src_values[i]


class bsr_axpy_work_arrays:
    """Opaque structure for persisting :func:`bsr_axpy` temporary work buffers across calls"""

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
            self._old_y_values = wp.empty(shape=(y.nnz), dtype=y.values.dtype, device=self.device)


def bsr_axpy(
    x: BsrMatrix[BlockType[Rows, Cols, Scalar]],
    y: Optional[BsrMatrix[BlockType[Rows, Cols, Scalar]]] = None,
    alpha: Scalar = 1.0,
    beta: Scalar = 1.0,
    work_arrays: Optional[bsr_axpy_work_arrays] = None,
) -> BsrMatrix[BlockType[Rows, Cols, Scalar]]:
    """
    Performs the sparse matrix addition ``y := alpha * X + beta * y`` on BSR matrices `x` and `y` and returns `y`.

    The `x` and `y` matrices are allowed to alias.

    Args:
        x: Read-only right-hand-side.
        y: Mutable left-hand-side. If `y` is not provided, it will be allocated and treated as zero.
        alpha: Uniform scaling factor for `x`
        beta: Uniform scaling factor for `y`
        work_arrays: In most cases this function will require the use of temporary storage; this storage can be reused across calls by passing an instance of :class:`bsr_axpy_work_arrays` in `work_arrays`.
    """

    if y is None:
        # If not output matrix is provided, allocate it for convenience
        y = bsr_zeros(x.nrow, x.ncol, block_type=x.values.dtype, device=x.values.device)
        beta = 0.0

    # Handle easy cases first
    if beta == 0.0 or y.nnz == 0:
        bsr_assign(src=x, dest=y)
        return bsr_scale(y, alpha=alpha)

    if alpha == 0.0 or x.nnz == 0:
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

    sum_nnz = x.nnz + y.nnz
    device = y.values.device
    work_arrays._allocate(device, y, sum_nnz)

    wp.copy(work_arrays._sum_cols, y.columns, 0, 0, y.nnz)
    wp.launch(kernel=_bsr_get_block_row, device=device, dim=y.nnz, inputs=[0, y.offsets, work_arrays._sum_rows])

    wp.copy(work_arrays._sum_cols, x.columns, y.nnz, 0, x.nnz)
    wp.launch(kernel=_bsr_get_block_row, device=device, dim=x.nnz, inputs=[y.nnz, x.offsets, work_arrays._sum_rows])

    # Save old y values before overwriting matrix
    wp.copy(dest=work_arrays._old_y_values, src=y.values, count=y.nnz)

    # Increase dest array sizes if needed
    if y.columns.shape[0] < sum_nnz:
        y.columns = wp.empty(shape=(sum_nnz,), dtype=int, device=device)

    from warp.context import runtime

    if device.is_cpu:
        native_func = runtime.core.bsr_matrix_from_triplets_float_host
    else:
        native_func = runtime.core.bsr_matrix_from_triplets_float_device

    old_y_nnz = y.nnz
    y.nnz = native_func(
        y.block_shape[0],
        y.block_shape[1],
        y.nrow,
        sum_nnz,
        work_arrays._sum_rows.ptr,
        work_arrays._sum_cols.ptr,
        0,
        y.offsets.ptr,
        y.columns.ptr,
        0,
    )

    _bsr_ensure_fits(y)
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
        dim=x.nnz,
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


@wp.kernel
def _bsr_mm_count_coeffs(
    z_nnz: int,
    x_offsets: wp.array(dtype=int),
    x_columns: wp.array(dtype=int),
    y_offsets: wp.array(dtype=int),
    counts: wp.array(dtype=int),
):
    row = wp.tid()
    count = int(0)

    x_beg = x_offsets[row]
    x_end = x_offsets[row + 1]

    for x_block in range(x_beg, x_end):
        x_col = x_columns[x_block]
        count += y_offsets[x_col + 1] - y_offsets[x_col]

    counts[row + 1] = count

    if row == 0:
        counts[0] = z_nnz


@wp.kernel
def _bsr_mm_list_coeffs(
    x_offsets: wp.array(dtype=int),
    x_columns: wp.array(dtype=int),
    y_offsets: wp.array(dtype=int),
    y_columns: wp.array(dtype=int),
    mm_offsets: wp.array(dtype=int),
    mm_rows: wp.array(dtype=int),
    mm_cols: wp.array(dtype=int),
):
    row = wp.tid()
    mm_block = mm_offsets[row]

    x_beg = x_offsets[row]
    x_end = x_offsets[row + 1]

    for x_block in range(x_beg, x_end):
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
    mm_offsets: wp.array(dtype=int),
    mm_cols: wp.array(dtype=int),
    mm_values: wp.array(dtype=Any),
):
    row = wp.tid()
    mm_beg = mm_offsets[row]
    mm_end = mm_offsets[row + 1]

    x_beg = x_offsets[row]
    x_end = x_offsets[row + 1]
    for x_block in range(x_beg, x_end):
        x_col = x_columns[x_block]
        ax_val = alpha * x_values[x_block]

        y_beg = y_offsets[x_col]
        y_end = y_offsets[x_col + 1]

        for y_block in range(y_beg, y_end):
            mm_block = wp.lower_bound(mm_cols, mm_beg, mm_end, y_columns[y_block])
            mm_values[mm_block] = mm_values[mm_block] + ax_val * y_values[y_block]


class bsr_mm_work_arrays:
    """Opaque structure for persisting :func:`bsr_mm` temporary work buffers across calls"""

    def __init__(self):
        self._reset(None)

    def _reset(self, device):
        self.device = device
        self._pinned_count_buffer = None
        self._mm_row_counts = None
        self._mm_rows = None
        self._mm_cols = None
        self._old_z_values = None
        self._old_z_offsets = None
        self._old_z_columns = None

    def _allocate_stage_1(self, device, z: BsrMatrix, copied_z_nnz: int, z_aliasing: bool):
        if self.device != device:
            self._reset(device)

        # Allocations that do not depend on any computation
        if self.device.is_cuda:
            if self._pinned_count_buffer is None:
                self._pinned_count_buffer = wp.empty(shape=(1,), dtype=int, pinned=True, device="cpu")

        if self._mm_row_counts is None or self._mm_row_counts.size < z.nrow + 1:
            self._mm_row_counts = wp.empty(shape=(z.nrow + 1,), dtype=int, device=self.device)

        if copied_z_nnz > 0:
            if self._old_z_values is None or self._old_z_values.size < copied_z_nnz:
                self._old_z_values = wp.empty(shape=(copied_z_nnz,), dtype=z.values.dtype, device=self.device)

        if z_aliasing:
            if self._old_z_columns is None or self._old_z_columns.size < z.nnz:
                self._old_z_columns = wp.empty(shape=(z.nnz,), dtype=z.columns.dtype, device=self.device)
            if self._old_z_offsets is None or self._old_z_offsets.size < z.nrow + 1:
                self._old_z_offsets = wp.empty(shape=(z.nrow + 1,), dtype=z.offsets.dtype, device=self.device)

    def _allocate_stage_2(self, mm_nnz: int):
        # Allocations that depend on unmerged nnz estimate
        if self._mm_rows is None or self._mm_rows.size < mm_nnz:
            self._mm_rows = wp.empty(shape=(mm_nnz,), dtype=int, device=self.device)
        if self._mm_cols is None or self._mm_cols.size < mm_nnz:
            self._mm_cols = wp.empty(shape=(mm_nnz,), dtype=int, device=self.device)


def bsr_mm(
    x: BsrMatrix[BlockType[Rows, Any, Scalar]],
    y: BsrMatrix[BlockType[Any, Cols, Scalar]],
    z: Optional[BsrMatrix[BlockType[Rows, Cols, Scalar]]] = None,
    alpha: Scalar = 1.0,
    beta: Scalar = 0.0,
    work_arrays: Optional[bsr_mm_work_arrays] = None,
) -> BsrMatrix[BlockType[Rows, Cols, Scalar]]:
    """
    Performs the sparse matrix-matrix multiplication ``z := alpha * x * y + beta * z`` on BSR matrices `x`, `y` and `z`, and returns `z`.

    The `x`, `y` and `z` matrices are allowed to alias.
    If the matrix `z` is not provided as input, it will be allocated and treated as zero.

    Args:
        x: Read-only left factor of the matrix-matrix product.
        y: Read-only right factor of the matrix-matrix product.
        z: Mutable left-hand-side. If `z` is not provided, it will be allocated and treated as zero.
        alpha: Uniform scaling factor for the ``x * y`` product
        beta: Uniform scaling factor for `z`
        work_arrays: In most cases this function will require the use of temporary storage; this storage can be reused across calls by passing an instance of :class:`bsr_mm_work_arrays` in `work_arrays`.
    """

    if z is None:
        # If not output matrix is provided, allocate it for convenience
        z_block_shape = (x.block_shape[0], y.block_shape[1])
        if z_block_shape == (1, 1):
            z_block_type = x.scalar_type
        else:
            z_block_type = wp.types.matrix(shape=z_block_shape, dtype=x.scalar_type)
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

    if not isinstance(alpha, z.scalar_type):
        alpha = z.scalar_type(alpha)
    if not isinstance(beta, z.scalar_type):
        beta = z.scalar_type(beta)

    if work_arrays is None:
        work_arrays = bsr_mm_work_arrays()

    z_aliasing = z == x or z == y
    copied_z_nnz = z.nnz if beta != 0.0 or z_aliasing else 0

    work_arrays._allocate_stage_1(device, z, copied_z_nnz, z_aliasing)

    # Prefix sum of number of (unmerged) mm blocks per row
    wp.launch(
        kernel=_bsr_mm_count_coeffs,
        device=device,
        dim=z.nrow,
        inputs=[copied_z_nnz, x.offsets, x.columns, y.offsets, work_arrays._mm_row_counts],
    )
    warp.utils.array_scan(work_arrays._mm_row_counts, work_arrays._mm_row_counts)

    # Get back total counts on host
    if device.is_cuda:
        wp.copy(dest=work_arrays._pinned_count_buffer, src=work_arrays._mm_row_counts, src_offset=z.nrow, count=1)
        wp.synchronize_stream(wp.get_stream(device))
        mm_nnz = int(work_arrays._pinned_count_buffer.numpy()[0])
    else:
        mm_nnz = int(work_arrays._mm_row_counts.numpy()[z.nrow])

    work_arrays._allocate_stage_2(mm_nnz)

    # If z has a non-zero scale, save current data before overwriting it
    if copied_z_nnz > 0:
        # Copy z row and column indices
        wp.copy(dest=work_arrays._mm_cols, src=z.columns, count=copied_z_nnz)
        wp.launch(
            kernel=_bsr_get_block_row, device=device, dim=copied_z_nnz, inputs=[0, z.offsets, work_arrays._mm_rows]
        )
        # Save current z values in temporary buffer
        wp.copy(src=z.values, dest=work_arrays._old_z_values, count=copied_z_nnz)
        if z_aliasing:
            # If z is aliasing with x or y, need to save topology as well
            wp.copy(src=z.columns, dest=work_arrays._old_z_columns, count=copied_z_nnz)
            wp.copy(src=z.offsets, dest=work_arrays._old_z_offsets, count=z.nrow + 1)

    # Fill unmerged mm blocks rows and columns
    wp.launch(
        kernel=_bsr_mm_list_coeffs,
        device=device,
        dim=z.nrow,
        inputs=[
            x.offsets,
            x.columns,
            y.offsets,
            y.columns,
            work_arrays._mm_row_counts,
            work_arrays._mm_rows,
            work_arrays._mm_cols,
        ],
    )

    # Increase dest array size if needed
    if z.columns.shape[0] < mm_nnz:
        z.columns = wp.empty(shape=(mm_nnz,), dtype=int, device=device)

    from warp.context import runtime

    if device.is_cpu:
        native_func = runtime.core.bsr_matrix_from_triplets_float_host
    else:
        native_func = runtime.core.bsr_matrix_from_triplets_float_device

    z.nnz = native_func(
        z.block_shape[0],
        z.block_shape[1],
        z.nrow,
        mm_nnz,
        work_arrays._mm_rows.ptr,
        work_arrays._mm_cols.ptr,
        0,
        z.offsets.ptr,
        z.columns.ptr,
        0,
    )

    _bsr_ensure_fits(z)
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
    if (warp.types.type_is_matrix(x.values.dtype) or warp.types.type_is_matrix(y.values.dtype)) and not (
        warp.types.type_is_matrix(z.values.dtype)
    ):
        # Result block type is scalar, but operands are matrices
        # Cast result to (1x1) matrix to perform multiplication
        mm_values = z.values.view(wp.types.matrix(shape=(1, 1), dtype=z.scalar_type))
    else:
        mm_values = z.values

    wp.launch(
        kernel=_bsr_mm_compute_values,
        device=device,
        dim=z.nrow,
        inputs=[
            alpha,
            work_arrays._old_z_offsets if x == z else x.offsets,
            work_arrays._old_z_columns if x == z else x.columns,
            work_arrays._old_z_values if x == z else x.values,
            work_arrays._old_z_offsets if y == z else y.offsets,
            work_arrays._old_z_columns if y == z else y.columns,
            work_arrays._old_z_values if y == z else y.values,
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


def bsr_mv(
    A: BsrMatrix[BlockType[Rows, Cols, Scalar]],
    x: "Array[Vector[Cols, Scalar] | Scalar]",
    y: Optional["Array[Vector[Rows, Scalar] | Scalar]"] = None,
    alpha: Scalar = 1.0,
    beta: Scalar = 0.0,
    work_buffer: Optional["Array[Vector[Rows, Scalar] | Scalar]"] = None,
) -> "Array[Vector[Rows, Scalar] | Scalar]":
    """
    Performs the sparse matrix-vector product ``y := alpha * A * x + beta * y`` and returns `y`.

    The `x` and `y` vectors are allowed to alias.

    Args:
        A: Read-only, left matrix factor of the matrix-vector product.
        x: Read-only, right vector factor of the matrix-vector product.
        y: Mutable left-hand-side. If `y` is not provided, it will be allocated and treated as zero.
        alpha: Uniform scaling factor for `x`. If zero, `x` will not be read and may be left uninitialized.
        beta: Uniform scaling factor for `y`. If zero, `y` will not be read and may be left uninitialized.
        work_buffer: Temporary storage is required if and only if `x` and `y` are the same vector. If provided the `work_buffer` array
            will be used for this purpose, otherwise a temporary allocation will be performed.
    """

    if y is None:
        # If no output array is provided, allocate one for convenience
        y_vec_len = A.block_shape[0]
        y_dtype = A.scalar_type if y_vec_len == 1 else wp.vec(length=y_vec_len, dtype=A.scalar_type)
        y = wp.empty(shape=(A.nrow,), device=A.values.device, dtype=y_dtype)
        y.zero_()
        beta = 0.0

    if not isinstance(alpha, A.scalar_type):
        alpha = A.scalar_type(alpha)
    if not isinstance(beta, A.scalar_type):
        beta = A.scalar_type(beta)

    if A.values.device != x.device or A.values.device != y.device:
        raise ValueError("A, x and y must reside on the same device")

    if x.shape[0] != A.ncol:
        raise ValueError("Number of columns of A must match number of rows of x")
    if y.shape[0] != A.nrow:
        raise ValueError("Number of rows of A must match number of rows of y")

    if x == y:
        # Aliasing case, need temporary storage
        if work_buffer is None:
            work_buffer = wp.empty_like(y)
        elif work_buffer.size < y.size:
            raise ValueError(f"Work buffer size is insufficient, needs to be at least {y.size}")
        elif not wp.types.types_equal(work_buffer.dtype, y.dtype):
            raise ValueError(f"Work buffer must have same data type as y, {wp.types.type_repr(y.dtype)}")

        # Save old y values before overwriting vector
        wp.copy(dest=work_buffer, src=y, count=y.size)
        x = work_buffer

    # Promote scalar vectors to length-1 vecs and conversely
    if warp.types.type_is_matrix(A.values.dtype):
        if A.block_shape[0] == 1:
            if y.dtype == A.scalar_type:
                y = y.view(dtype=wp.vec(length=1, dtype=A.scalar_type))
        if A.block_shape[1] == 1:
            if x.dtype == A.scalar_type:
                x = x.view(dtype=wp.vec(length=1, dtype=A.scalar_type))
    else:
        if A.block_shape[0] == 1:
            if y.dtype != A.scalar_type:
                y = y.view(dtype=A.scalar_type)
        if A.block_shape[1] == 1:
            if x.dtype != A.scalar_type:
                x = x.view(dtype=A.scalar_type)

    wp.launch(
        kernel=_bsr_mv_kernel,
        device=A.values.device,
        dim=A.nrow,
        inputs=[alpha, A.offsets, A.columns, A.values, x, beta, y],
    )

    return y
