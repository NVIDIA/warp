# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import weakref
from functools import cache
from typing import Any, Generic, Literal, TypeVar

from numpy import eye

import warp as wp
import warp._src.utils
from warp._src.logger import log_warning
from warp._src.types import (
    Array,
    Cols,
    Rows,
    Scalar,
    Vector,
    is_array,
    scalar_types,
    type_is_matrix,
    type_repr,
    type_scalar_type,
    type_size,
    type_size_in_bytes,
    type_to_warp,
    types_equal,
)

_wp_module_name_ = "warp.sparse"

__all__ = [
    "BSR_STATUS_ROW_CAPACITY_EXCEEDED",
    "BSR_STATUS_SUCCESS",
    "BsrMatrix",
    "bsr_assign",
    "bsr_axpy",
    "bsr_axpy_work_arrays",
    "bsr_block_index",
    "bsr_compress",
    "bsr_copy",
    "bsr_diag",
    "bsr_from_triplets",
    "bsr_get_diag",
    "bsr_identity",
    "bsr_matrix_t",
    "bsr_mm",
    "bsr_mm_work_arrays",
    "bsr_mv",
    "bsr_row_index",
    "bsr_scale",
    "bsr_set_diag",
    "bsr_set_from_triplets",
    "bsr_set_identity",
    "bsr_set_transpose",
    "bsr_set_zero",
    "bsr_transposed",
    "bsr_zeros",
]


# typing hints

_BlockType = TypeVar("BlockType")  # noqa: PLC0132


class _MatrixBlockType(Generic[Rows, Cols, Scalar]):
    pass


class _ScalarBlockType(Generic[Scalar]):
    pass


BlockType = _MatrixBlockType[Rows, Cols, Scalar] | _ScalarBlockType[Scalar]

_struct_cache = {}
_transfer_buffer_cache = {}

BSR_STATUS_SUCCESS = 0
"""Sparse operation status code for success."""

BSR_STATUS_ROW_CAPACITY_EXCEEDED = 1
"""Sparse operation status code for insufficient padded row capacity."""

_BSR_STATUS_SUCCESS = BSR_STATUS_SUCCESS
_BSR_STATUS_ROW_CAPACITY_EXCEEDED = BSR_STATUS_ROW_CAPACITY_EXCEEDED


def _mark_apic_deferred_nnz_update(bsr: BsrMatrix, apic_capture) -> None:
    BsrMatrix.__setattr__(bsr, "_apic_deferred_nnz_capture", apic_capture)


def _warn_masked_arg_deprecated(func_name: str):
    log_warning(
        f"The `masked` argument to {func_name}() is deprecated; pass topology='masked' instead.",
        category=DeprecationWarning,
        stacklevel=3,
    )


class BsrMatrix(Generic[_BlockType]):
    """Untyped base class for BSR and CSR matrices.

    Should not be constructed directly but through functions such as :func:`bsr_zeros`.

    Attributes:
        nrow (int): Number of rows of blocks.
        ncol (int): Number of columns of blocks.
        nnz (int):  Upper bound for the number of stored blocks, used for
          dimensioning launches. For compact matrices this is also the number
          of active non-zero blocks. See also :meth:`nnz_sync`.
        offsets (Array[int]): Array of size at least ``1 + nrow`` such that the
          start and capacity end indices of row ``r`` are ``offsets[r]`` and
          ``offsets[r+1]``, respectively.
        row_counts (Array[int] | None): Optional array of size at least
          ``nrow`` containing the active block count of each row. Active blocks
          of row ``r`` are stored in
          ``offsets[r]:offsets[r] + row_counts[r]``. For compact matrices,
          ``row_counts`` is ``None``, in which case all storage in each row is
          active.
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
    def block_shape(self) -> tuple[int, int]:
        """Shape of the individual blocks."""
        return getattr(self.values.dtype, "_shape_", (1, 1))

    @property
    def block_size(self) -> int:
        """Size of the individual blocks, i.e. number of rows per block times number of columns per block."""
        return type_size(self.values.dtype)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the matrix, i.e. number of rows/columns of blocks times number of rows/columns per block."""
        block_shape = self.block_shape
        return (self.nrow * block_shape[0], self.ncol * block_shape[1])

    @property
    def dtype(self) -> type:
        """Data type for individual block values."""
        return self.values.dtype

    @property
    def device(self) -> wp._src.context.Device:
        """Device on which matrix arrays are allocated."""
        return self.offsets.device

    @property
    def requires_grad(self) -> bool:
        """Read-only property indicating whether the matrix participates in adjoint computations."""
        return self.values.requires_grad

    @property
    def scalar_values(self) -> wp.array:
        """Access the ``values`` array as a 3d scalar array."""
        values_view = _as_3d_array(self.values, self.block_shape)
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
            inputs=[self.nrow, self.offsets, self.row_counts, out],
        )
        return out

    def nnz_sync(self) -> int:
        """Synchronize the stored block upper bound from the device ``offsets`` array to the host.

        Ensures that any ongoing transfer of ``offsets[nrow]`` from the device offsets array to the host has completed,
        or, if none has been scheduled yet, starts a new transfer and waits for it to complete.

        Then updates the host-side nnz upper bound to match ``offsets[nrow]``, and returns it. For compact matrices,
        this is the active non-zero block count. For padded matrices, this is the total row-capacity storage size,
        not necessarily the active non-zero block count.

        See also :meth:`notify_nnz_changed`.
        """

        # A nnz readback reads the current value "now". Under a CPU APIC capture
        # the host copy/readback would otherwise be recorded (deferred), leaving
        # the readback buffer uninitialized, so pause recording around it so it
        # runs live and returns the real value.
        from warp._src.context import capture_pause, capture_resume, runtime  # noqa: PLC0415

        apic_graph = runtime._apic_graph
        paused_graph = None
        if apic_graph is not None and apic_graph.device.is_cpu and apic_graph.device == self.device:
            if getattr(self, "_apic_deferred_nnz_capture", None) is apic_graph._apic_capture:
                raise NotImplementedError(
                    "BsrMatrix.nnz_sync() cannot read a topology update recorded during CPU APIC capture. "
                    "Call nnz_sync() after graph replay or avoid the readback inside the capture."
                )
            paused_graph = capture_pause(device=self.device)
        try:
            buf, event = self._nnz_transfer_if_any()
            if buf is None:
                buf, event = self._copy_nnz_async()

            if event is not None:
                wp.synchronize_event(event)
            self.nnz = int(buf.numpy()[0])
        finally:
            if paused_graph is not None:
                capture_resume(paused_graph, device=self.device)
        return self.nnz

    def status_sync(self) -> int:
        """Return the asynchronous sparse status code, synchronizing if needed.

        The status code is sticky: sparse operations can set it, but successful
        operations do not clear it. Use :meth:`clear_status` before an operation
        when checking only that operation's status.
        """

        status = self._status_if_any()
        if status is None:
            return _BSR_STATUS_SUCCESS

        return int(status.numpy()[0])

    def status_message(self) -> str:
        """Return a human-readable message for :meth:`status_sync`."""

        return _bsr_status_message(self.status_sync())

    def clear_status(self) -> None:
        """Clear the asynchronous sparse status code."""

        status = self._status_if_any()
        if status is not None:
            status.zero_()

    def notify_nnz_changed(self, nnz: int | None = None, nnz_capacity: int | None = None) -> None:
        """Notify the matrix that sparse storage metadata changed outside :mod:`warp.sparse`.

        Call this after modifying ``offsets`` or the host-side ``nnz`` upper
        bound directly. If assigning a new offsets array, also update
        ``row_counts`` for padded matrices, or set it to ``None`` for compact
        matrices.

        Args:
            nnz: New non-zero block count upper bound. If omitted, read from
              ``offsets[nrow]`` unless `nnz_capacity` is provided. The caller
              is responsible for ensuring it is greater or equal to ``offsets[nrow]``.
            nnz_capacity: Optional storage pre-allocation size. If omitted,
              default to `nnz`. The caller is responsible for ensuring it is greater
              or equal to the true ``offsets[nrow]`` value.
        """
        self._copy_nnz_async()
        if nnz is None:
            if nnz_capacity is None:
                self.nnz_sync()
            else:
                self.nnz = nnz_capacity

        _bsr_ensure_fits(self, nnz=nnz, capacity=nnz_capacity)

    def copy_nnz_async(self) -> None:
        """Start the asynchronous transfer of ``offsets[nrow]`` from the device offsets array to host.

        Deprecated; prefer :meth:`notify_nnz_changed` instead, which will make sure to resize arrays if necessary.
        """
        log_warning(
            "The `copy_nnz_async` method is deprecated and will be removed in a future version. Prefer `notify_nnz_changed` instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        self._copy_nnz_async()

    def _copy_nnz_async(self) -> tuple[wp.array, wp.Event]:
        buf, event = self._setup_nnz_transfer()
        if buf is not None:
            stream = wp.get_stream(self.device) if self.device.is_cuda else None
            wp.copy(src=self.offsets, dest=buf, src_offset=self.nrow, count=1, stream=stream)
            if event is not None:
                stream.record_event(event, external=True)
        return buf, event

    def _setup_nnz_transfer(self) -> tuple[wp.array, wp.Event]:
        buf, event = self._nnz_transfer_if_any()
        if buf is not None:
            return buf, event

        buf, event = _allocate_transfer_buf(self.device)
        if buf is not None:
            # buf may still be None if device is currently capturing
            BsrMatrix.__setattr__(self, "_nnz_transfer", (buf, event))
            weakref.finalize(self, _redeem_transfer_buf, self.device, buf, event)

        return buf, event

    def _nnz_transfer_if_any(self) -> tuple[wp.array, wp.Event]:
        return getattr(self, "_nnz_transfer", (None, None))

    def _ensure_status(self) -> wp.array:
        status = self._status_if_any()
        if status is None or status.device != self.device:
            status = wp.zeros(shape=(1,), dtype=int, device=self.device)
            BsrMatrix.__setattr__(self, "_status", status)

        return status

    def _status_if_any(self) -> wp.array | None:
        return getattr(self, "_status", None)

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


def _allocate_transfer_buf(device):
    if device.ordinal in _transfer_buffer_cache:
        all_, pool = _transfer_buffer_cache[device.ordinal]
    else:
        all_ = []
        pool = []
        _transfer_buffer_cache[device.ordinal] = (all_, pool)

    if pool:
        return pool.pop()

    # A CUDA captured stream cannot allocate a fresh transfer buffer mid-capture
    # (the pooled path handles that). A CPU APIC capture allocates host memory
    # safely, so let it fall through and complete the readback.
    if device.is_capturing and not device.is_cpu:
        return None, None

    buf = wp.empty(dtype=int, shape=(1,), device="cpu", pinned=device.is_cuda)
    event = wp.Event(device) if device.is_cuda else None
    all_.append((buf, event))  # keep a reference to the buffer and event, prevent garbage collection before redeem
    return buf, event


def _redeem_transfer_buf(device, buf, event):
    _all, pool = _transfer_buffer_cache[device.ordinal]
    pool.append((buf, event))


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
        row_counts: wp.array(dtype=int)
        """Array of size at least ``nrow``. May be ``None`` for compact matrices."""
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
        BsrMatrixTyped.dtype = dtype  # necessary for eval_annotations
        _struct_cache[key] = wp._src.codegen.Struct(
            key=key,
            cls=BsrMatrixTyped,
            module=module,
        )

    return _struct_cache[key]


@wp.kernel(enable_backward=False)
def _bsr_fill_uniform_row_capacity(
    row_capacity: int,
    offsets: wp.array(dtype=int),
):
    row = wp.tid()
    offsets[row] = row * row_capacity


@wp.kernel(enable_backward=False)
def _bsr_fill_row_counts_from_offsets(
    row_count: int,
    offsets: wp.array(dtype=int),
    row_counts: wp.array(dtype=int),
):
    row = wp.tid()
    row_counts[row] = offsets[row + 1] - offsets[row]


@wp.kernel(enable_backward=False)
def _bsr_extend_offsets_with_empty_rows(
    old_row_count: int,
    offsets: wp.array(dtype=int),
):
    row = old_row_count + 1 + wp.tid()
    offsets[row] = offsets[old_row_count]


@wp.func
def _bsr_row_end(offsets: wp.array(dtype=int), row_counts: wp.array(dtype=int), row: int) -> int:
    if row_counts:
        return offsets[row] + row_counts[row]
    return offsets[row + 1]


@wp.func
def _bsr_row_count(offsets: wp.array(dtype=int), row_counts: wp.array(dtype=int), row: int) -> int:
    return _bsr_row_end(offsets, row_counts, row) - offsets[row]


def _bsr_set_zero_topology(
    bsr: BsrMatrix,
    topology: Literal["compact", "padded"],
    row_capacity: int | Array[int] | None,
    nnz_capacity: int | None,
    preserve_row_capacity: bool = False,
) -> None:

    if nnz_capacity is not None:
        nnz_capacity = int(nnz_capacity)
        if nnz_capacity < 0:
            raise ValueError("nnz_capacity must be nonnegative")

    if row_capacity is not None:
        if topology != "padded":
            raise ValueError("row_capacity requires topology='padded'")

        if is_array(row_capacity):
            if row_capacity.ndim != 1 or row_capacity.shape[0] != bsr.nrow:
                raise ValueError(f"row_capacity array must have shape ({bsr.nrow},), got {row_capacity.shape}")
            if row_capacity.dtype != wp.int32:
                raise TypeError("row_capacity array must have dtype int32")
            if bsr.device != row_capacity.device:
                raise ValueError(
                    f"row_capacity array must reside on the destination matrix device, got {row_capacity.device} and {bsr.device}"
                )
        else:
            row_capacity = int(row_capacity)
            if row_capacity < 0:
                raise ValueError("row_capacity must be nonnegative")

    if topology == "padded":
        if row_capacity is not None:
            if is_array(row_capacity):
                bsr.offsets[:1].zero_()
                warp._src.utils.array_scan(row_capacity, bsr.offsets[1:], inclusive=True)
                nnz = None
            else:
                wp.launch(
                    _bsr_fill_uniform_row_capacity,
                    dim=bsr.nrow + 1,
                    device=bsr.offsets.device,
                    inputs=[row_capacity, bsr.offsets],
                )
                nnz = bsr.nrow * row_capacity
        elif preserve_row_capacity:
            nnz = bsr.nnz
        else:
            bsr.offsets.zero_()
            nnz = 0
        _bsr_ensure_independent_row_counts(bsr, preserve_row_counts=False)
    else:
        bsr.offsets.zero_()
        _bsr_set_compact_row_counts(bsr)
        nnz = 0

    bsr.notify_nnz_changed(nnz=nnz, nnz_capacity=nnz_capacity)


def bsr_zeros(
    rows_of_blocks: int,
    cols_of_blocks: int,
    block_type: BlockType,
    device: wp.DeviceLike = None,
    *,
    topology: Literal["compact", "padded"] | None = None,
    row_capacity: int | Array[int] | None = None,
    nnz_capacity: int | None = None,
) -> BsrMatrix:
    """Construct and return an empty BSR or CSR matrix with the given shape.

    Args:
        rows_of_blocks: Number of rows of blocks.
        cols_of_blocks: Number of columns of blocks.
        block_type: Type of individual blocks.
          For CSR matrices, this should be a scalar type.
          For BSR matrices, this should be a matrix type (e.g. from :func:`warp.types.matrix`).
        device: Device on which to allocate the matrix arrays.
        topology: Topology policy. Defaults to ``"compact"`` unless
          ``row_capacity`` is provided, in which case it defaults to
          ``"padded"``.
        row_capacity: Optional row capacity for padded matrices. May be a
          nonnegative integer for uniform capacity, or an integer Warp array of
          shape ``(rows_of_blocks,)`` for per-row capacity. Providing this
          implies padded topology when ``topology`` is omitted.
        nnz_capacity: Optional storage allocation upper bound for ``columns``
          and ``values``. This is mainly useful with per-row ``row_capacity``
          arrays when an upper bound is already known. When ``row_capacity`` is
          an array, the caller is responsible for ensuring ``nnz_capacity`` is
          at least the total row capacity.
    """

    if topology is None:
        topology = "padded" if row_capacity is not None else "compact"
    elif topology not in ("compact", "padded"):
        raise ValueError(f"Unsupported topology policy: {topology}")

    bsr = bsr_matrix_t(block_type)()

    bsr.nrow = int(rows_of_blocks)
    bsr.ncol = int(cols_of_blocks)
    bsr.nnz = 0
    bsr.offsets = wp.empty(shape=(bsr.nrow + 1,), dtype=int, device=device)
    bsr.columns = wp.empty(shape=(0,), dtype=int, device=device)
    bsr.values = wp.empty(shape=(0,), dtype=block_type, device=device)

    _bsr_set_zero_topology(bsr, topology, row_capacity, nnz_capacity, preserve_row_capacity=False)

    return bsr


def _bsr_resize(
    bsr: BsrMatrix,
    rows_of_blocks: int | None = None,
    cols_of_blocks: int | None = None,
    preserve_offsets: bool = False,
) -> None:
    old_nrow = bsr.nrow
    old_offsets = bsr.offsets

    if rows_of_blocks is not None:
        bsr.nrow = int(rows_of_blocks)
    if cols_of_blocks is not None:
        bsr.ncol = int(cols_of_blocks)

    if bsr.offsets.size < bsr.nrow + 1:
        bsr.offsets = wp.empty(shape=(bsr.nrow + 1,), dtype=int, device=bsr.offsets.device)

    if preserve_offsets and bsr.nrow > old_nrow:
        wp.copy(dest=bsr.offsets, src=old_offsets, count=old_nrow + 1)
        wp.launch(
            _bsr_extend_offsets_with_empty_rows,
            dim=bsr.nrow - old_nrow,
            device=bsr.offsets.device,
            inputs=[old_nrow, bsr.offsets],
        )


def _bsr_has_compact_row_counts(bsr: BsrMatrix) -> bool:
    return bsr.row_counts is None


def _bsr_ensure_independent_row_counts(bsr: BsrMatrix, preserve_row_counts: bool = True) -> None:
    old_row_counts = bsr.row_counts
    if old_row_counts is not None and old_row_counts.size >= bsr.nrow:
        if not preserve_row_counts:
            old_row_counts.zero_()
        return

    bsr.row_counts = wp.empty(shape=(bsr.nrow,), dtype=int, device=bsr.offsets.device)
    if not preserve_row_counts:
        bsr.row_counts.zero_()
    elif old_row_counts is not None and bsr.nrow > 0 and old_row_counts.size >= bsr.nrow:
        wp.copy(dest=bsr.row_counts, src=old_row_counts, count=bsr.nrow)
    elif bsr.nrow > 0:
        wp.launch(
            _bsr_fill_row_counts_from_offsets,
            dim=bsr.nrow,
            device=bsr.offsets.device,
            inputs=[bsr.nrow, bsr.offsets, bsr.row_counts],
        )


def _bsr_set_compact_row_counts(bsr: BsrMatrix) -> None:
    bsr.row_counts = None


def _bsr_ensure_fits(bsr: BsrMatrix, nnz: int | None = None, capacity: int | None = None) -> None:
    if nnz is None:
        nnz = bsr.nnz
    else:
        # update nnz upper bound
        bsr.nnz = int(nnz)

    if capacity is None:
        capacity = nnz
    elif capacity < nnz:
        raise ValueError(f"capacity must be at least nnz, got {capacity} and {nnz}")

    if bsr.columns.size < capacity:
        bsr.columns = wp.empty(shape=(capacity,), dtype=int, device=bsr.columns.device)
    if bsr.values.size < capacity:
        bsr.values = wp.empty(
            shape=(capacity,), dtype=bsr.values.dtype, device=bsr.values.device, requires_grad=bsr.values.requires_grad
        )


def bsr_set_zero(
    bsr: BsrMatrix,
    rows_of_blocks: int | None = None,
    cols_of_blocks: int | None = None,
    *,
    topology: Literal["compact", "padded", "masked"] = "compact",
    row_capacity: int | Array[int] | None = None,
    nnz_capacity: int | None = None,
):
    """Set a BSR matrix to zero, possibly changing its size.

    Args:
        bsr: The BSR or CSR matrix to set to zero.
        rows_of_blocks: If not ``None``, the new number of rows of blocks.
        cols_of_blocks: If not ``None``, the new number of columns of blocks.
        topology: Topology policy. ``"compact"`` discards the active and
          capacity topology, ``"padded"`` keeps row capacity and makes every
          row empty when the matrix size is unchanged, and ``"masked"`` keeps
          active topology and zeroes values.
        row_capacity: Optional reserved block capacity for each row when
          ``topology="padded"``. May be a nonnegative integer for uniform row
          capacity, or an integer Warp array of shape ``(rows_of_blocks,)`` for
          per-row capacity. If ``None``, existing row capacity is preserved for
          retained rows, and added rows get zero capacity.
        nnz_capacity: Optional storage allocation upper bound for ``columns``
          and ``values``. When ``row_capacity`` is an array, the caller is
          responsible for ensuring ``nnz_capacity`` is at least the total row
          capacity. Ignored with ``topology="masked"``.
    """
    if topology not in ("compact", "padded", "masked"):
        raise ValueError(f"Unsupported topology policy: {topology}")
    if row_capacity is not None and topology != "padded":
        raise ValueError("row_capacity requires topology='padded'")

    if topology == "masked":
        if rows_of_blocks is not None or cols_of_blocks is not None:
            raise ValueError("Cannot resize a matrix with topology='masked'")
        bsr.values.zero_()
        return

    preserve_row_capacity = topology == "padded" and row_capacity is None
    _bsr_resize(bsr, rows_of_blocks, cols_of_blocks, preserve_offsets=preserve_row_capacity)

    _bsr_set_zero_topology(bsr, topology, row_capacity, nnz_capacity, preserve_row_capacity=preserve_row_capacity)


def _as_3d_array(arr, block_shape):
    return wp.array(
        ptr=arr.ptr,
        capacity=arr.capacity,
        device=arr.device,
        dtype=type_scalar_type(arr.dtype),
        shape=(arr.shape[0], *block_shape),
        grad=None if arr.grad is None else _as_3d_array(arr.grad, block_shape),
    )


def _as_1d_block_array(arr, block_shape, block_type):
    if arr.ndim == 1:
        if not types_equal(arr.dtype, block_type):
            raise ValueError(
                "Values array type must correspond to that of the destination matrix, got "
                f"{type_repr(arr.dtype)} and {type_repr(block_type)}"
            )
        return arr

    if arr.ndim != 3:
        raise ValueError(f"Number of dimensions for values array should be 1 or 3, got {arr.ndim}")

    if arr.shape[1:] != block_shape:
        raise ValueError(
            f"Last two dimensions in values array ({arr.shape[1:]}) should match block shape {block_shape}"
        )

    if type_scalar_type(arr.dtype) != type_scalar_type(block_type):
        raise ValueError(
            "Scalar type of values array "
            f"({type_repr(arr.dtype)}) should match destination matrix scalar type "
            f"({type_repr(type_scalar_type(block_type))})"
        )

    if not arr.is_contiguous:
        raise ValueError("Values array should be contiguous")

    values_view = wp.array(
        ptr=arr.ptr,
        capacity=arr.capacity,
        device=arr.device,
        dtype=block_type,
        shape=(arr.shape[0],),
        grad=None if arr.grad is None else _as_1d_block_array(arr.grad, block_shape, block_type),
    )
    values_view._ref = arr
    return values_view


def _optional_ctypes_pointer(array: wp.array | None, ctype):
    return None if array is None else ctypes.cast(array.ptr, ctypes.POINTER(ctype))


def _optional_ctypes_event(event: wp.Event | None):
    return None if event is None else event.cuda_event


def _bsr_status_message(status: int) -> str:
    if status == _BSR_STATUS_SUCCESS:
        return "success"
    if status == _BSR_STATUS_ROW_CAPACITY_EXCEEDED:
        return "row capacity exceeded"
    return f"unknown status {status}"


def _bsr_try_native_compress_inplace(src: BsrMatrix, prune_numerical_zeros: bool, compact: bool = False) -> str | None:
    scalar_type_code = _bsr_scalar_type_codes.get(src.scalar_type)
    if scalar_type_code is None:
        return (
            "requires native in-place compression support for scalar types float32 and float64, "
            f"got {type_repr(src.scalar_type)}"
        )

    from warp._src.context import runtime  # noqa: PLC0415

    try:
        if src.device.is_cpu:
            native_func = runtime.core.wp_bsr_compress_inplace_host
        elif src.device.is_cuda:
            native_func = runtime.core.wp_bsr_compress_inplace_device
        else:
            return f"requires native in-place compression support on CPU or CUDA devices, got {src.device}"
    except AttributeError:
        return "requires native in-place compression support"

    scalar_values = src.scalar_values
    zero_value_mask = _zero_value_masks.get(src.scalar_type, 0) if prune_numerical_zeros else 0
    nnz_buf, nnz_event = src._setup_nnz_transfer() if compact else (None, None)
    with wp.ScopedDevice(src.device):
        native_func(
            src.nrow,
            src.block_size,
            type_size_in_bytes(src.scalar_type),
            scalar_type_code,
            src.nnz,
            prune_numerical_zeros,
            zero_value_mask,
            compact,
            ctypes.cast(src.offsets.ptr, ctypes.POINTER(ctypes.c_int32)),
            _optional_ctypes_pointer(src.row_counts, ctype=ctypes.c_int32),
            ctypes.cast(src.columns.ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_void_p(scalar_values.ptr),
            True,  # compress_values
            _optional_ctypes_pointer(nnz_buf, ctype=ctypes.c_int32),
            _optional_ctypes_event(nnz_event),
        )

    return None


def _bsr_try_native_compress_indices_inplace(
    src: BsrMatrix,
    values: wp.array3d | None,
    prune_numerical_zeros: bool,
    compact: bool = False,
) -> str | None:
    if prune_numerical_zeros:
        if src.scalar_type not in _zero_value_masks:
            return (
                "requires native symbolic compression support with numerical zero pruning for scalar types with "
                f"a zero mask, got {type_repr(src.scalar_type)}"
            )
        if values is None:
            return "requires values for numerical zero pruning"
        scalar_type_code = 0
        block_size = src.block_size
        scalar_size = type_size_in_bytes(src.scalar_type)
        zero_value_mask = _zero_value_masks.get(src.scalar_type, 0)
        values_to_prune = values
    else:
        scalar_type_code = 0
        block_size = 0
        scalar_size = 0
        zero_value_mask = 0
        values_to_prune = None

    from warp._src.context import runtime  # noqa: PLC0415

    try:
        if src.device.is_cpu:
            native_func = runtime.core.wp_bsr_compress_inplace_host
        elif src.device.is_cuda:
            native_func = runtime.core.wp_bsr_compress_inplace_device
        else:
            return f"requires native symbolic compression support on CPU or CUDA devices, got {src.device}"
    except AttributeError:
        return "requires native symbolic compression support"

    nnz_buf, nnz_event = src._setup_nnz_transfer() if compact else (None, None)
    with wp.ScopedDevice(src.device):
        native_func(
            src.nrow,
            block_size,
            scalar_size,
            scalar_type_code,
            src.nnz,
            prune_numerical_zeros,
            zero_value_mask,
            compact,
            ctypes.cast(src.offsets.ptr, ctypes.POINTER(ctypes.c_int32)),
            _optional_ctypes_pointer(src.row_counts, ctype=ctypes.c_int32),
            ctypes.cast(src.columns.ptr, ctypes.POINTER(ctypes.c_int32)),
            _optional_ctypes_pointer(values_to_prune, ctype=ctypes.c_int32),
            False,  # compress_values
            _optional_ctypes_pointer(nnz_buf, ctype=ctypes.c_int32),
            _optional_ctypes_event(nnz_event),
        )

    return None


_bsr_scalar_type_codes = {
    wp.float32: 0,
    wp.float64: 1,
}


_zero_value_masks = {
    wp.float16: 0x7FFF,
    wp.bfloat16: 0x7FFF,
    wp.float32: 0x7FFFFFFF,
    wp.float64: 0x7FFFFFFFFFFFFFFF,
    wp.int8: 0xFF,
    wp.uint8: 0xFF,
    wp.int16: 0xFFFF,
    wp.uint16: 0xFFFF,
    wp.int32: 0xFFFFFFFF,
    wp.uint32: 0xFFFFFFFF,
    wp.int64: 0xFFFFFFFFFFFFFFFF,
    wp.uint64: 0xFFFFFFFFFFFFFFFF,
}


def _bsr_set_from_triplets_native(
    dest: BsrMatrix,
    rows: wp.array,
    columns: wp.array,
    values: wp.array | None,
    count: wp.array | None,
    nnz: int,
    zero_value_mask: int,
    masked: bool,
    summed_triplet_offsets: wp.array | None,
    summed_triplet_indices: wp.array | None,
    dest_offsets: wp.array,
    dest_row_counts: wp.array | None,
    dest_columns: wp.array,
    nnz_buf: wp.array | None,
    nnz_event: wp.Event | None,
) -> None:
    from warp._src.context import runtime  # noqa: PLC0415

    device = dest.device
    if device.is_cpu:
        native_func = runtime.core.wp_bsr_matrix_from_triplets_host
    else:
        native_func = runtime.core.wp_bsr_matrix_from_triplets_device

    scalar_size = type_size_in_bytes(dest.scalar_type) if values is not None else 0

    with wp.ScopedDevice(device):
        native_func(
            dest.block_size,
            scalar_size,
            dest.nrow,
            dest.ncol,
            nnz,
            _optional_ctypes_pointer(count, ctype=ctypes.c_int32),
            ctypes.cast(rows.ptr, ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(columns.ptr, ctypes.POINTER(ctypes.c_int32)),
            _optional_ctypes_pointer(values, ctype=ctypes.c_int32),
            zero_value_mask,
            masked,
            _optional_ctypes_pointer(summed_triplet_offsets, ctype=ctypes.c_int32),
            _optional_ctypes_pointer(summed_triplet_indices, ctype=ctypes.c_int32),
            ctypes.cast(dest_offsets.ptr, ctypes.POINTER(ctypes.c_int32)),
            _optional_ctypes_pointer(dest_row_counts, ctype=ctypes.c_int32),
            ctypes.cast(dest_columns.ptr, ctypes.POINTER(ctypes.c_int32)),
            _optional_ctypes_pointer(nnz_buf, ctype=ctypes.c_int32),
            _optional_ctypes_event(nnz_event),
        )


@wp.kernel(module="unique")
def _bsr_accumulate_triplet_values(
    row_count: int,
    tpl_summed_offsets: wp.array(dtype=int),
    tpl_summed_indices: wp.array(dtype=int),
    tpl_values: wp.array3d(dtype=Any),
    bsr_offsets: wp.array(dtype=int),
    bsr_values: wp.array3d(dtype=Any),
):
    block, i, j = wp.tid()

    if block >= bsr_offsets[row_count]:
        return

    bsr_values[block, i, j] = _bsr_accumulate_triplet_value(
        block, i, j, tpl_summed_offsets, tpl_summed_indices, tpl_values
    )


@wp.kernel(module="unique")
def _bsr_compress_accumulate_values(
    row_count: int,
    src_offsets: wp.array(dtype=int),
    src_row_counts: wp.array(dtype=int),
    src_columns: wp.array(dtype=int),
    src_values: wp.array3d(dtype=Any),
    dest_offsets: wp.array(dtype=int),
    dest_row_counts: wp.array(dtype=int),
    dest_columns: wp.array(dtype=int),
    dest_values: wp.array3d(dtype=Any),
):
    dest_block, i, j = wp.tid()

    dest_row = bsr_row_index(dest_offsets, row_count, dest_block, dest_row_counts)
    if dest_row == -1:
        return

    dest_col = dest_columns[dest_block]
    if dest_col < 0:
        return

    val = dest_values.dtype(0.0)
    for src_block in range(src_offsets[dest_row], _bsr_row_end(src_offsets, src_row_counts, dest_row)):
        if src_columns[src_block] == dest_col:
            val += src_values[src_block, i, j]

    dest_values[dest_block, i, j] = val


@wp.func
def _bsr_accumulate_triplet_value(
    block: int,
    i: int,
    j: int,
    tpl_summed_offsets: wp.array(dtype=int),
    tpl_summed_indices: wp.array(dtype=int),
    tpl_values: wp.array3d(dtype=Any),
):
    if block == 0:
        beg = 0
    else:
        beg = tpl_summed_offsets[block - 1]
    end = tpl_summed_offsets[block]

    val = tpl_values[tpl_summed_indices[beg], i, j]
    for k in range(beg + 1, end):
        val += tpl_values[tpl_summed_indices[k], i, j]

    return val


@wp.kernel(module="unique")
def _bsr_set_from_triplets_masked_values(
    count: wp.array(dtype=int),
    row_count: int,
    col_count: int,
    rows: wp.array(dtype=int),
    columns: wp.array(dtype=int),
    values: wp.array3d(dtype=Any),
    bsr_offsets: wp.array(dtype=int),
    bsr_row_counts: wp.array(dtype=int),
    bsr_columns: wp.array(dtype=int),
    bsr_values: wp.array3d(dtype=Any),
):
    triplet, i, j = wp.tid()

    if count and triplet >= count[0]:
        return

    row = rows[triplet]
    col = columns[triplet]
    if row < 0 or row >= row_count or col < 0 or col >= col_count:
        return

    block = bsr_block_index(row, col, bsr_offsets, bsr_columns, bsr_row_counts)
    if block != -1:
        wp.atomic_add(bsr_values, block, i, j, values[triplet, i, j])


@wp.kernel(enable_backward=False)
def _bsr_set_from_triplets_padded_scatter_compact_topology(
    compact_offsets: wp.array(dtype=int),
    compact_columns: wp.array(dtype=int),
    dest_offsets: wp.array(dtype=int),
    dest_row_counts: wp.array(dtype=int),
    dest_columns: wp.array(dtype=int),
    record_compact_blocks: bool,
    dest_compact_blocks: wp.array(dtype=int),
    status: wp.array(dtype=int),
):
    row = wp.tid()

    compact_beg = compact_offsets[row]
    compact_end = compact_offsets[row + 1]
    block_count = compact_end - compact_beg

    row_beg = dest_offsets[row]
    capacity_end = dest_offsets[row + 1]
    if row_beg + block_count > capacity_end:
        dest_row_counts[row] = 0
        wp.atomic_max(status, 0, _BSR_STATUS_ROW_CAPACITY_EXCEEDED)
        block_count = 0
    else:
        dest_row_counts[row] = block_count

    for local_block in range(block_count):
        block = row_beg + local_block
        compact_block = compact_beg + local_block
        dest_columns[block] = compact_columns[compact_block]
        if record_compact_blocks:
            dest_compact_blocks[block] = compact_block

    if record_compact_blocks:
        for block in range(row_beg + block_count, capacity_end):
            dest_compact_blocks[block] = -1


@wp.kernel(module="unique")
def _bsr_set_from_triplets_padded_accumulate_values(
    dest_compact_blocks: wp.array(dtype=int),
    dest_offsets: wp.array(dtype=int),
    dest_nrow: int,
    tpl_summed_offsets: wp.array(dtype=int),
    tpl_summed_indices: wp.array(dtype=int),
    values: wp.array3d(dtype=Any),
    dest_values: wp.array3d(dtype=Any),
):
    block, i, j = wp.tid()

    if block >= dest_offsets[dest_nrow]:
        return

    compact_block = dest_compact_blocks[block]
    if compact_block != -1:
        dest_values[block, i, j] = _bsr_accumulate_triplet_value(
            compact_block, i, j, tpl_summed_offsets, tpl_summed_indices, values
        )


def bsr_set_from_triplets(
    dest: BsrMatrix[BlockType[Rows, Cols, Scalar]],
    rows: Array[int],
    columns: Array[int],
    values: Array[Scalar | BlockType[Rows, Cols, Scalar]] | None = None,
    count: Array[int] | None = None,
    prune_numerical_zeros: bool = True,
    masked: bool = False,
    topology: Literal["compact", "masked", "padded"] | None = None,
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
        count: Single-element array indicating the number of triplets. If ``None``, the number of triplets is determined from the shape of
          ``rows`` and ``columns`` arrays.
        prune_numerical_zeros: If ``True``, will ignore the zero-valued blocks.
        masked: Deprecated. Use ``topology="masked"`` instead. If ``True``,
          ignore blocks that are not existing non-zeros of ``dest``.
        topology: Optional topology policy. ``"compact"`` keeps the existing
          compact rebuild behavior, ``"masked"`` is equivalent to
          the deprecated ``masked=True``, and ``"padded"`` writes the compacted triplet
          topology into existing destination row capacity and records capacity
          overflow in ``dest.status_sync()``.
    """
    if topology is None:
        topology = "masked" if masked else "compact"
    elif topology not in ("compact", "masked", "padded"):
        raise ValueError(f"Unsupported topology policy: {topology}")
    elif masked and topology != "masked":
        raise ValueError("Cannot pass masked=True with a non-masked topology policy")

    if masked:
        _warn_masked_arg_deprecated("bsr_set_from_triplets")

    masked = topology == "masked"

    if rows.device != columns.device or rows.device != dest.device:
        raise ValueError(
            f"Rows and columns must reside on the destination matrix device, got {rows.device}, {columns.device} and {dest.device}"
        )

    if rows.shape[0] != columns.shape[0]:
        raise ValueError(
            f"Rows and columns arrays must have the same length, got {rows.shape[0]} and {columns.shape[0]}"
        )

    if rows.dtype != wp.int32 or columns.dtype != wp.int32:
        raise TypeError("Rows and columns arrays must be of type int32")

    if count is not None:
        if count.device != rows.device:
            raise ValueError(f"Count and rows must reside on the same device, got {count.device} and {rows.device}")

        if count.shape != (1,):
            raise ValueError(f"Count array must be a single-element array, got {count.shape}")

        if count.dtype != wp.int32:
            raise TypeError("Count array must be of type int32")

    # Accept either array1d(dtype) or contiguous array3d(scalar_type) as values
    if values is not None:
        if values.device != rows.device:
            raise ValueError(f"Values and rows must reside on the same device, got {values.device} and {rows.device}")

        if values.shape[0] != rows.shape[0]:
            raise ValueError(
                f"Values and rows arrays must have the same length, got {values.shape[0]} and {rows.shape[0]}"
            )

        if values.ndim == 1:
            if not types_equal(values.dtype, dest.values.dtype):
                raise ValueError(
                    f"Values array type must correspond to that of the dest matrix, got {type_repr(values.dtype)} and {type_repr(dest.values.dtype)}"
                )
        elif values.ndim == 3:
            if values.shape[1:] != dest.block_shape:
                raise ValueError(
                    f"Last two dimensions in values array ({values.shape[1:]}) should correspond to matrix block shape {(dest.block_shape)})"
                )

            if type_scalar_type(values.dtype) != dest.scalar_type:
                raise ValueError(
                    f"Scalar type of values array ({type_repr(values.dtype)}) should correspond to that of matrix ({type_repr(dest.scalar_type)})"
                )
        else:
            raise ValueError(f"Number of dimension for values array should be 1 or 3, got {values.ndim}")

        if prune_numerical_zeros and not values.is_contiguous:
            raise ValueError("Values array should be contiguous for numerical zero pruning")

    nnz = rows.shape[0]
    if nnz == 0:
        bsr_set_zero(dest, topology=topology)
        return

    if topology == "padded":
        _bsr_ensure_independent_row_counts(dest, preserve_row_counts=False)
        status = dest._ensure_status()

        device = dest.values.device
        scalar_type = dest.scalar_type
        values_3d = _as_3d_array(values, dest.block_shape) if values is not None else None
        zero_value_mask = _zero_value_masks.get(scalar_type, 0) if prune_numerical_zeros and values is not None else 0

        from warp._src.context import runtime  # noqa: PLC0415

        apic_capture = runtime._apic_capture if device.is_cpu else None
        cpu_apic_capture = apic_capture is not None and apic_capture.device.is_cpu

        compact_offsets = wp.empty(shape=(dest.nrow + 1,), dtype=wp.int32, device=device)
        compact_columns = wp.empty(shape=(nnz,), dtype=wp.int32, device=device)
        needs_summed_triplets = values is not None or cpu_apic_capture
        summed_triplet_offsets = (
            wp.empty(shape=(nnz,), dtype=wp.int32, device=device) if needs_summed_triplets else None
        )
        summed_triplet_indices = (
            wp.empty(shape=(nnz,), dtype=wp.int32, device=device) if needs_summed_triplets else None
        )
        if cpu_apic_capture:
            apic_capture.track_array(rows)
            apic_capture.track_array(columns)
            apic_capture.track_array(values)
            apic_capture.track_array(summed_triplet_offsets)
            apic_capture.track_array(summed_triplet_indices)
            apic_capture.track_array(compact_offsets)
            apic_capture.track_array(compact_columns)
            apic_capture.track_array(count)

        _bsr_set_from_triplets_native(
            dest,
            rows,
            columns,
            values,
            count,
            nnz,
            zero_value_mask,
            False,
            summed_triplet_offsets,
            summed_triplet_indices,
            compact_offsets,
            None,
            compact_columns,
            None,
            None,
        )

        compact_blocks = (
            wp.empty(shape=(dest.nnz,), dtype=wp.int32, device=device) if values is not None else dest.columns
        )
        wp.launch(
            _bsr_set_from_triplets_padded_scatter_compact_topology,
            dim=dest.nrow,
            device=device,
            inputs=[
                compact_offsets,
                compact_columns,
                dest.offsets,
                dest.row_counts,
                dest.columns,
                values is not None,
                compact_blocks,
                status,
            ],
        )

        if values is not None:
            dest.values.zero_()
            if dest.nnz > 0:
                wp.launch(
                    _bsr_set_from_triplets_padded_accumulate_values,
                    dim=(dest.nnz, *dest.block_shape),
                    device=device,
                    inputs=[
                        compact_blocks,
                        dest.offsets,
                        dest.nrow,
                        summed_triplet_offsets,
                        summed_triplet_indices,
                        values_3d,
                        dest.scalar_values,
                    ],
                )

        return

    if masked:
        dest.values.zero_()
        if values is not None:
            wp.launch(
                _bsr_set_from_triplets_masked_values,
                dim=(nnz, *dest.block_shape),
                device=dest.device,
                inputs=[
                    count,
                    dest.nrow,
                    dest.ncol,
                    rows,
                    columns,
                    _as_3d_array(values, dest.block_shape),
                    dest.offsets,
                    dest.row_counts,
                    dest.columns,
                    dest.scalar_values,
                ],
            )
        return

    # Increase dest array sizes if needed
    _bsr_ensure_fits(dest, nnz=nnz)

    device = dest.values.device
    scalar_type = dest.scalar_type
    zero_value_mask = _zero_value_masks.get(scalar_type, 0) if prune_numerical_zeros and values is not None else 0

    apic_capture = None
    cpu_apic_capture = False
    if device.is_cpu:
        from warp._src.context import runtime  # noqa: PLC0415

        apic_capture = runtime._apic_capture
        cpu_apic_capture = apic_capture is not None and apic_capture.device.is_cpu

    nnz_buf, nnz_event = dest._setup_nnz_transfer()
    needs_summed_triplets = values is not None or cpu_apic_capture
    summed_triplet_offsets = wp.empty(shape=(nnz,), dtype=wp.int32, device=device) if needs_summed_triplets else None
    summed_triplet_indices = wp.empty(shape=(nnz,), dtype=wp.int32, device=device) if needs_summed_triplets else None

    if device.is_cpu:
        # On CPU the topology build dispatches to wp_bsr_matrix_from_triplets_host,
        # which records an APIC_OP_BSR_FROM_TRIPLETS op under capture. Track the
        # input/output base regions first so the recorded op references real
        # region IDs (matches the sort / runlength_encode tracking).
        if cpu_apic_capture:
            apic_capture.track_array(rows)
            apic_capture.track_array(columns)
            apic_capture.track_array(values)
            apic_capture.track_array(summed_triplet_offsets)
            apic_capture.track_array(summed_triplet_indices)
            apic_capture.track_array(dest.offsets)
            apic_capture.track_array(dest.columns)
            apic_capture.track_array(dest.row_counts)
            apic_capture.track_array(count)
            apic_capture.track_array(nnz_buf)
            _mark_apic_deferred_nnz_update(dest, apic_capture)

    _bsr_set_from_triplets_native(
        dest,
        rows,
        columns,
        values,
        count,
        nnz,
        zero_value_mask,
        masked,
        summed_triplet_offsets,
        summed_triplet_indices,
        dest.offsets,
        dest.row_counts,
        dest.columns,
        nnz_buf,
        nnz_event,
    )

    if values is not None:
        wp.launch(
            _bsr_accumulate_triplet_values,
            dim=(nnz, *dest.block_shape),
            device=device,
            inputs=[
                dest.nrow,
                summed_triplet_offsets,
                summed_triplet_indices,
                _as_3d_array(values, dest.block_shape),
                dest.offsets,
            ],
            outputs=[dest.scalar_values],
        )

    if not masked:
        _bsr_set_compact_row_counts(dest)


def bsr_from_triplets(
    rows_of_blocks: int,
    cols_of_blocks: int,
    rows: Array[int],
    columns: Array[int],
    values: Array[Scalar | BlockType[Rows, Cols, Scalar]],
    prune_numerical_zeros: bool = True,
):
    """Construct a BSR matrix with values defined by coordinate-oriented (COO) triplets.

    The first dimension of the three input arrays must match and indicates the number of COO triplets.
    This convenience constructor always builds compact storage. To build into reserved row capacity,
    allocate with :func:`bsr_zeros` using ``row_capacity`` and then call :func:`bsr_set_from_triplets`
    with ``topology="padded"``.

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
        block_type = wp.types.matrix(shape=values.shape[1:], dtype=values.dtype)
    else:
        block_type = values.dtype

    A = bsr_zeros(
        rows_of_blocks=rows_of_blocks, cols_of_blocks=cols_of_blocks, block_type=block_type, device=values.device
    )
    A.values.requires_grad = values.requires_grad
    bsr_set_from_triplets(A, rows, columns, values, prune_numerical_zeros=prune_numerical_zeros)
    return A


def bsr_compress(
    src: BsrMatrix[BlockType[Rows, Cols, Scalar]],
    prune_numerical_zeros: bool = True,
    inplace: bool = False,
    topology: Literal["compact", "padded"] | None = None,
) -> BsrMatrix[BlockType[Rows, Cols, Scalar]]:
    """Compress the active blocks of ``src`` and return ``src``.

    Slack entries outside ``offsets[row]:offsets[row] + row_counts[row]`` are ignored. When
    ``inplace=False``, the topology is rebuilt natively from active row and
    column data, and values are accumulated with differentiable Warp kernels.
    When ``inplace=True``, entries are sorted and coalesced independently
    within each active row using native in-place compression without
    ``O(nnz)`` temporary allocation, but this path is not differentiable.
    In-place compression requires native support and currently supports
    matrices whose scalar type is ``float32`` or ``float64``;
    ``topology="padded"`` preserves the existing row capacity instead of
    packing active blocks into compact row storage. Compact compression sets
    ``src.row_counts`` to ``None`` and does not update any previously attached
    row-count array. Callers should not rely on buffer identity being
    preserved for any ``inplace`` value.

    Args:
        src: Matrix to compress.
        prune_numerical_zeros: If ``True``, zero-valued input blocks are ignored before coalescing.
        inplace: If ``True``, sort and coalesce each active row range directly
          in ``src`` using native in-place compression without ``O(nnz)``
          temporary allocation. This path is not differentiable. With
          ``topology="compact"``, active blocks are also packed into compact
          row storage. If ``False``, use differentiable value accumulation.
        topology: Topology policy for the compressed arrays. Defaults to
          ``"compact"``. ``"padded"`` preserves existing row capacity; it
          cannot report row-capacity overflow because compression never
          increases the active block count.
    """

    if not isinstance(src, BsrMatrix):
        raise ValueError("bsr_compress() requires a concrete BsrMatrix")

    if topology is None:
        topology = "compact"
    elif topology not in ("compact", "padded"):
        raise ValueError(f"Unsupported topology policy: {topology}")

    if inplace:
        compact = topology == "compact"
        if not compact:
            _bsr_ensure_independent_row_counts(src)
        native_error = (
            _bsr_try_native_compress_inplace(src, prune_numerical_zeros, compact=compact) if src.nrow > 0 else None
        )
        if native_error is not None:
            raise RuntimeError(f"bsr_compress(..., inplace=True) {native_error}")

        if compact:
            _bsr_set_compact_row_counts(src)
        return src

    if topology == "padded":
        _bsr_ensure_independent_row_counts(src)
    old_offsets = wp.clone(src.offsets)
    old_row_counts = None if src.row_counts is None else wp.clone(src.row_counts)
    old_columns = wp.clone(src.columns[: src.nnz])
    old_values = wp.clone(src.values[: src.nnz])
    old_scalar_values = _as_3d_array(old_values, src.block_shape)

    compact = topology == "compact"
    native_error = (
        _bsr_try_native_compress_indices_inplace(
            src,
            old_scalar_values,
            prune_numerical_zeros,
            compact=compact,
        )
        if src.nrow > 0
        else None
    )
    if native_error is not None:
        raise RuntimeError(f"bsr_compress(..., inplace=False) {native_error}")

    if compact:
        _bsr_set_compact_row_counts(src)

    if src.nnz > 0:
        wp.launch(
            _bsr_compress_accumulate_values,
            dim=(src.nnz, *src.block_shape),
            device=src.device,
            inputs=[
                src.nrow,
                old_offsets,
                old_row_counts,
                old_columns,
                old_scalar_values,
                src.offsets,
                src.row_counts,
                src.columns,
            ],
            outputs=[src.scalar_values],
        )

    return src


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
    def row_counts(self) -> wp.array:
        return self.mat.row_counts

    @property
    def columns(self) -> wp.array:
        return self.mat.columns

    @property
    def scalar_type(self) -> Scalar:
        return self.mat.scalar_type

    @property
    def block_shape(self) -> tuple[int, int]:
        return self.mat.block_shape

    @property
    def block_size(self) -> int:
        return self.mat.block_size

    @property
    def shape(self) -> tuple[int, int]:
        return self.mat.shape

    @property
    def dtype(self) -> type:
        return self.mat.dtype

    @property
    def requires_grad(self) -> bool:
        return self.mat.requires_grad

    @property
    def device(self) -> wp._src.context.Device:
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
        """Return a transposed copy of this matrix."""
        return _BsrScalingExpression(self.mat.transpose(), self.scale)


BsrMatrixOrExpression = BsrMatrix[_BlockType] | _BsrExpression[_BlockType]


def _extract_matrix_and_scale(bsr: BsrMatrixOrExpression):
    if isinstance(bsr, BsrMatrix):
        return bsr, 1.0
    if isinstance(bsr, _BsrScalingExpression):
        return bsr.mat, bsr.scale

    raise ValueError("Argument cannot be interpreted as a BsrMatrix")


@wp.func
def bsr_row_index(
    offsets: wp.array(dtype=int),
    row_count: int,
    block_index: int,
) -> int:
    """Return the row containing a block in compact BSR/CSR storage, or -1 if no such row exists.

    This overload searches the storage range defined by ``offsets``. For matrices with padded row
    capacity, pass ``row_counts`` to the row-capacity overload to search active blocks only.

    Args:
        offsets: Array of size at least ``1 + row_count`` containing the offsets of the blocks in each row.
        row_count: Number of rows of blocks.
        block_index: Index of the block.
    """
    return wp.where(block_index < offsets[row_count], wp.lower_bound(offsets, 0, row_count + 1, block_index + 1), 0) - 1


@wp.func
def bsr_row_index(
    offsets: wp.array(dtype=int),
    row_count: int,
    block_index: int,
    row_counts: wp.array(dtype=int),
) -> int:
    """Return the row containing an active block in a capacity-aware BSR matrix, or -1 if no such row exists.

    This row-capacity overload searches active row ranges
    ``offsets[row]:offsets[row] + row_counts[row]`` and ignores slack storage.
    If ``row_counts`` is ``None``, it falls back to compact storage rows
    defined by ``offsets``.
    """

    row = bsr_row_index(offsets, row_count, block_index)
    if row_counts and row >= 0:
        if block_index >= offsets[row] + row_counts[row]:
            row = -1

    return row


@wp.func
def _bsr_row_block_index(
    col: int,
    row_beg: int,
    row_end: int,
    bsr_columns: wp.array(dtype=int),
) -> int:
    if row_beg == row_end:
        return -1

    block_index = wp.lower_bound(bsr_columns, row_beg, row_end, col)
    if block_index == row_end:
        return -1
    return wp.where(bsr_columns[block_index] == col, block_index, -1)


@wp.func
def bsr_block_index(
    row: int,
    col: int,
    bsr_offsets: wp.array(dtype=int),
    bsr_columns: wp.array(dtype=int),
) -> int:
    """Return the index of the block at block-coordinates (row, col), or -1 if no such block exists.

    Assumes that the segments of ``bsr_columns`` corresponding to each row are sorted. This overload
    searches the storage row ``bsr_offsets[row]:bsr_offsets[row + 1]``. For matrices with padded row
    capacity, pass ``bsr_row_counts`` to search active blocks only.

    Args:
        row: Row of the block.
        col: Column of the block.
        bsr_offsets: Array of size at least ``1 + row`` containing the offsets of the blocks in each row.
        bsr_columns: Array of size at least equal to ``bsr_offsets[row + 1]`` containing the column indices of the blocks.
    """

    if row < 0:
        return -1

    row_beg = bsr_offsets[row]
    row_end = bsr_offsets[row + 1]

    return _bsr_row_block_index(col, row_beg, row_end, bsr_columns)


@wp.func
def bsr_block_index(
    row: int,
    col: int,
    bsr_offsets: wp.array(dtype=int),
    bsr_columns: wp.array(dtype=int),
    bsr_row_counts: wp.array(dtype=int),
) -> int:
    """Return the active block index in a capacity-aware BSR matrix, or -1 if no such block exists.

    This row-capacity overload searches
    ``bsr_offsets[row]:bsr_offsets[row] + bsr_row_counts[row]`` and ignores
    slack storage. If ``bsr_row_counts`` is ``None``, it falls back to compact
    storage rows defined by ``bsr_offsets``.
    """

    if row < 0:
        return -1

    row_beg = bsr_offsets[row]
    row_end = _bsr_row_end(bsr_offsets, bsr_row_counts, row)

    return _bsr_row_block_index(col, row_beg, row_end, bsr_columns)


@wp.kernel(enable_backward=False)
def _bsr_assign_list_blocks(
    src_subrows: int,
    src_subcols: int,
    dest_subrows: int,
    dest_subcols: int,
    src_row_count: int,
    src_offsets: wp.array(dtype=int),
    src_row_counts: wp.array(dtype=int),
    src_columns: wp.array(dtype=int),
    dest_rows: wp.array(dtype=int),
    dest_cols: wp.array(dtype=int),
):
    block, subrow, subcol = wp.tid()
    dest_block = (block * src_subcols + subcol) * src_subrows + subrow

    row = bsr_row_index(src_offsets, src_row_count, block, src_row_counts)
    if row == -1:
        dest_rows[dest_block] = row  # invalid
        dest_cols[dest_block] = row
    else:
        dest_subrow = row * src_subrows + subrow
        dest_subcol = src_columns[block] * src_subcols + subcol
        dest_rows[dest_block] = dest_subrow // dest_subrows
        dest_cols[dest_block] = dest_subcol // dest_subcols


@wp.kernel(module="unique")
def _bsr_assign_copy_blocks(
    scale: Any,
    src_subrows: int,
    src_subcols: int,
    dest_subrows: int,
    dest_subcols: int,
    src_row_count: int,
    src_offsets: wp.array(dtype=int),
    src_row_counts: wp.array(dtype=int),
    src_columns: wp.array(dtype=int),
    src_values: wp.array3d(dtype=Any),
    dest_offsets: wp.array(dtype=int),
    dest_row_counts: wp.array(dtype=int),
    dest_columns: wp.array(dtype=int),
    dest_values: wp.array3d(dtype=Any),
):
    src_block, subrow, subcol = wp.tid()

    src_row = bsr_row_index(src_offsets, src_row_count, src_block, src_row_counts)
    if src_row == -1:
        return

    src_col = src_columns[src_block]

    dest_subrow = src_row * src_subrows + subrow
    dest_subcol = src_col * src_subcols + subcol
    dest_row = dest_subrow // dest_subrows
    dest_col = dest_subcol // dest_subcols

    dest_block = bsr_block_index(dest_row, dest_col, dest_offsets, dest_columns, dest_row_counts)
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


@wp.kernel(enable_backward=False)
def _bsr_assign_padded_row_ranges(
    src_offsets: wp.array(dtype=int),
    src_row_counts: wp.array(dtype=int),
    dest_offsets: wp.array(dtype=int),
    dest_row_counts: wp.array(dtype=int),
    status: wp.array(dtype=int),
):
    row = wp.tid()

    src_beg = src_offsets[row]
    src_end = _bsr_row_end(src_offsets, src_row_counts, row)
    src_count = src_end - src_beg

    dest_beg = dest_offsets[row]
    dest_capacity_end = dest_offsets[row + 1]
    dest_count = dest_capacity_end - dest_beg

    if src_count > dest_count:
        dest_row_counts[row] = 0
        wp.atomic_max(status, 0, _BSR_STATUS_ROW_CAPACITY_EXCEEDED)
        return

    dest_row_counts[row] = src_count


@wp.func
def _bsr_reblock_row_next_col(
    min_dest_col: int,
    dest_subcols: int,
    src_subcols: int,
    src_row_beg: int,
    src_row_end: int,
    src_columns: wp.array(dtype=int),
) -> int:
    src_col = (min_dest_col * dest_subcols) // src_subcols
    src_block = wp.lower_bound(src_columns, src_row_beg, src_row_end, src_col)
    if src_block == src_row_end:
        return -1

    src_subcol_first = src_columns[src_block] * src_subcols
    src_subcol_end = src_subcol_first + src_subcols
    dest_col_beg = src_subcol_first // dest_subcols
    dest_col_end = (src_subcol_end + dest_subcols - 1) // dest_subcols
    dest_col = wp.max(dest_col_beg, min_dest_col)

    if dest_col < dest_col_end:
        return dest_col

    return -1


@wp.kernel(enable_backward=False)
def _bsr_assign_padded_reblock_topology(
    src_subrows: int,
    src_subcols: int,
    dest_subrows: int,
    dest_subcols: int,
    src_row_count: int,
    dest_col_count: int,
    src_offsets: wp.array(dtype=int),
    src_row_counts: wp.array(dtype=int),
    src_columns: wp.array(dtype=int),
    dest_offsets: wp.array(dtype=int),
    dest_row_counts: wp.array(dtype=int),
    dest_columns: wp.array(dtype=int),
    status: wp.array(dtype=int),
):
    dest_row = wp.tid()

    dest_subrow_first = dest_row * dest_subrows
    src_row_first = dest_subrow_first // src_subrows
    src_row_end = wp.min(src_row_count, (dest_subrow_first + dest_subrows + src_subrows - 1) // src_subrows)

    dest_beg = dest_offsets[dest_row]
    capacity_end = dest_offsets[dest_row + 1]
    dest_block = dest_beg
    overflow = bool(False)

    previous_col = int(-1)
    searching = bool(True)
    while searching and not overflow:
        min_dest_col = previous_col + 1
        next_col = dest_col_count

        src_row = src_row_first
        while src_row < src_row_end:
            candidate = _bsr_reblock_row_next_col(
                min_dest_col,
                dest_subcols,
                src_subcols,
                src_offsets[src_row],
                _bsr_row_end(src_offsets, src_row_counts, src_row),
                src_columns,
            )
            if candidate != -1 and candidate < next_col:
                next_col = candidate
            src_row += 1

        if next_col == dest_col_count:
            searching = False
        else:
            if dest_block >= capacity_end:
                overflow = True
            else:
                dest_columns[dest_block] = next_col
                dest_block += 1
                previous_col = next_col

    if overflow:
        dest_row_counts[dest_row] = 0
        wp.atomic_max(status, 0, _BSR_STATUS_ROW_CAPACITY_EXCEEDED)
    else:
        dest_row_counts[dest_row] = dest_block - dest_beg


@wp.kernel(enable_backward=False)
def _bsr_copy_reblocked_capacity_counts(
    src_subrows: int,
    src_subcols: int,
    dest_subrows: int,
    src_row_count: int,
    src_offsets: wp.array(dtype=int),
    dest_offsets: wp.array(dtype=int),
):
    dest_row = wp.tid()

    if dest_row == 0:
        dest_offsets[0] = 0

    dest_subrow_first = dest_row * dest_subrows
    src_row = dest_subrow_first // src_subrows
    src_row_end = wp.min(src_row_count, (dest_subrow_first + dest_subrows + src_subrows - 1) // src_subrows)
    block_count = int(0)

    while src_row < src_row_end:
        block_count += (src_offsets[src_row + 1] - src_offsets[src_row]) * src_subcols
        src_row += 1

    dest_offsets[dest_row + 1] = block_count


@wp.kernel(module="unique")
def _bsr_assign_padded_copy_blocks(
    scale: Any,
    structure_only: bool,
    row_count: int,
    src_offsets: wp.array(dtype=int),
    src_row_counts: wp.array(dtype=int),
    src_columns: wp.array(dtype=int),
    src_values: wp.array3d(dtype=Any),
    dest_offsets: wp.array(dtype=int),
    dest_row_counts: wp.array(dtype=int),
    dest_columns: wp.array(dtype=int),
    dest_values: wp.array3d(dtype=Any),
):
    src_block, br, bc = wp.tid()

    src_row = bsr_row_index(src_offsets, row_count, src_block, src_row_counts)
    if src_row == -1:
        return

    dest_block = dest_offsets[src_row] + src_block - src_offsets[src_row]
    if dest_block >= _bsr_row_end(dest_offsets, dest_row_counts, src_row):
        return

    if br == 0 and bc == 0:
        dest_columns[dest_block] = src_columns[src_block]

    if not structure_only:
        dest_values[dest_block, br, bc] = dest_values.dtype(scale * src_values[src_block, br, bc])


def _bsr_assign_padded_same_block(
    dest: BsrMatrix,
    src: BsrMatrix,
    status: wp.array,
    src_scale: float = 1.0,
    structure_only: bool = False,
):
    if dest.block_shape != src.block_shape:
        raise ValueError("Padded same-block assignment requires matching block shapes")

    wp.launch(
        _bsr_assign_padded_row_ranges,
        dim=dest.nrow,
        device=dest.device,
        inputs=[
            src.offsets,
            src.row_counts,
            dest.offsets,
            dest.row_counts,
            status,
        ],
    )

    wp.launch(
        _bsr_assign_padded_copy_blocks,
        dim=(src.nnz, *src.block_shape),
        device=dest.device,
        inputs=[
            src.scalar_type(src_scale),
            structure_only,
            dest.nrow,
            src.offsets,
            src.row_counts,
            src.columns,
            src.scalar_values,
            dest.offsets,
            dest.row_counts,
            dest.columns,
            dest.scalar_values,
        ],
    )


def _bsr_assign_padded_reblock(
    dest: BsrMatrix,
    src: BsrMatrix,
    src_scale: float,
    src_subrows: int,
    src_subcols: int,
    dest_subrows: int,
    dest_subcols: int,
    structure_only: bool,
    status: wp.array,
):
    wp.launch(
        _bsr_assign_padded_reblock_topology,
        dim=dest.nrow,
        device=dest.device,
        inputs=[
            src_subrows,
            src_subcols,
            dest_subrows,
            dest_subcols,
            src.nrow,
            dest.ncol,
            src.offsets,
            src.row_counts,
            src.columns,
            dest.offsets,
            dest.row_counts,
            dest.columns,
            status,
        ],
    )

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
                src.row_counts,
                src.columns,
                src.scalar_values,
                dest.offsets,
                dest.row_counts,
                dest.columns,
                dest.scalar_values,
            ],
        )


def bsr_assign(
    dest: BsrMatrix[BlockType[Rows, Cols, Scalar]],
    src: BsrMatrixOrExpression[BlockType[Any, Any, Any]],
    structure_only: bool = False,
    masked: bool = False,
    topology: Literal["compact", "masked", "padded"] | None = None,
):
    """Copy the content of the ``src`` BSR matrix to ``dest``.

    Args:
      src: Matrix to be copied.
      dest: Destination matrix. May have a different block shape or scalar type
        than ``src``, in which case the required casting will be performed.
      structure_only: If ``True``, only the non-zero indices are copied, and uninitialized value storage is allocated
        to accommodate at least ``src.nnz`` blocks. If ``structure_only`` is ``False``, values are also copied with implicit
        casting if the two matrices use distinct scalar types.
      masked: Deprecated. Use ``topology="masked"`` instead. If ``True``, keep
        the non-zero topology of ``dest`` unchanged.
      topology: Optional topology policy. ``"compact"`` keeps the existing
        compact rebuild behavior, ``"masked"`` is equivalent to the deprecated ``masked=True``,
        and ``"padded"`` copies each source row into existing destination row
        capacity without changing ``dest.offsets`` and records capacity
        overflow in ``dest.status_sync()``.
    """

    src, src_scale = _extract_matrix_and_scale(src)

    if dest.values.device != src.values.device:
        raise ValueError("Source and destination matrices must reside on the same device")

    if topology is None:
        topology = "masked" if masked else "compact"
    elif topology not in ("compact", "masked", "padded"):
        raise ValueError(f"Unsupported topology policy: {topology}")
    elif masked and topology != "masked":
        raise ValueError("Cannot pass masked=True with a non-masked topology policy")

    if masked:
        _warn_masked_arg_deprecated("bsr_assign")

    masked = topology == "masked"

    if src.block_shape[0] >= dest.block_shape[0]:
        src_subrows = src.block_shape[0] // dest.block_shape[0]
        dest_subrows = 1
    else:
        dest_subrows = dest.block_shape[0] // src.block_shape[0]
        src_subrows = 1

    if src_subrows * dest.block_shape[0] != src.block_shape[0] * dest_subrows:
        raise ValueError(
            f"Incompatible dest and src block shapes; block rows must evenly divide one another (Got {dest.block_shape[0]}, {src.block_shape[0]})"
        )

    if src.block_shape[1] >= dest.block_shape[1]:
        src_subcols = src.block_shape[1] // dest.block_shape[1]
        dest_subcols = 1
    else:
        dest_subcols = dest.block_shape[1] // src.block_shape[1]
        src_subcols = 1

    if src_subcols * dest.block_shape[1] != src.block_shape[1] * dest_subcols:
        raise ValueError(
            f"Incompatible dest and src block shapes; block columns must evenly divide one another (Got {dest.block_shape[1]}, {src.block_shape[1]})"
        )

    dest_nrow = (src.nrow * src_subrows) // dest_subrows
    dest_ncol = (src.ncol * src_subcols) // dest_subcols

    if src.nrow * src_subrows != dest_nrow * dest_subrows or src.ncol * src_subcols != dest_ncol * dest_subcols:
        raise ValueError(
            f"The requested block shape {dest.block_shape} does not evenly divide the source matrix of total size {src.shape}"
        )

    if topology == "padded":
        status = dest._ensure_status()
        if dest_nrow != dest.nrow or dest_ncol != dest.ncol:
            raise ValueError(
                f"Incompatible destination matrix size, expected ({dest_nrow}, {dest_ncol}), got ({dest.nrow}, {dest.ncol})"
            )

        if dest == src:
            if not structure_only and src_scale != 1.0:
                bsr_scale(dest, src_scale)
            return

        _bsr_ensure_independent_row_counts(dest)

        if dest.block_shape != src.block_shape:
            _bsr_assign_padded_reblock(
                dest=dest,
                src=src,
                src_scale=src_scale,
                src_subrows=src_subrows,
                src_subcols=src_subcols,
                dest_subrows=dest_subrows,
                dest_subcols=dest_subcols,
                structure_only=structure_only,
                status=status,
            )
            return

        _bsr_assign_padded_same_block(
            dest=dest,
            src=src,
            src_scale=src_scale,
            structure_only=structure_only,
            status=status,
        )
        return

    nnz_alloc = src.nnz * src_subrows * src_subcols
    if masked:
        if dest_nrow != dest.nrow or dest_ncol != dest.ncol:
            raise ValueError(
                f"Incompatible destination matrix size, expected ({dest_nrow}, {dest_ncol}), got ({dest.nrow}, {dest.ncol})"
            )
    else:
        _bsr_resize(dest, rows_of_blocks=dest_nrow, cols_of_blocks=dest_ncol)

    if dest == src and not masked:
        # Self-assignment
        if not structure_only and src_scale != 1.0:
            bsr_scale(dest, src_scale)

    else:
        if not masked:
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
                    src.row_counts,
                    src.columns,
                    dest_rows,
                    dest_cols,
                ],
            )

            _bsr_ensure_fits(dest, nnz=nnz_alloc)

            # Compute destination offsets from triplets
            nnz_buf, nnz_event = dest._setup_nnz_transfer()
            _bsr_set_from_triplets_native(
                dest=dest,
                rows=dest_rows,
                columns=dest_cols,
                values=None,
                count=None,
                nnz=nnz_alloc,
                zero_value_mask=0,
                masked=masked,
                summed_triplet_offsets=None,
                summed_triplet_indices=None,
                dest_offsets=dest.offsets,
                dest_row_counts=None,
                dest_columns=dest.columns,
                nnz_buf=nnz_buf,
                nnz_event=nnz_event,
            )
            _bsr_set_compact_row_counts(dest)

        # copy block values
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
                    src.row_counts,
                    src.columns,
                    src.scalar_values,
                    dest.offsets,
                    dest.row_counts,
                    dest.columns,
                    dest.scalar_values,
                ],
            )


def bsr_copy(
    A: BsrMatrixOrExpression,
    scalar_type: Scalar | None = None,
    block_shape: tuple[int, int] | None = None,
    structure_only: bool = False,
    topology: Literal["compact", "padded"] = "compact",
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
       topology: Topology policy for the copy. Supported values are
         ``"compact"``, which rebuilds compact storage, and ``"padded"``,
         which preserves row capacity. ``"masked"`` is not meaningful for a
         newly allocated copy; use :func:`bsr_assign` with an existing
         destination matrix when preserving an active destination topology.
    """
    src, src_scale = _extract_matrix_and_scale(A)

    if topology not in ("compact", "padded"):
        raise ValueError(f"Unsupported topology policy: {topology}")

    if scalar_type is None:
        scalar_type = src.scalar_type
    if block_shape is None:
        block_shape = src.block_shape

    if block_shape == (1, 1):
        block_type = scalar_type
    else:
        block_type = wp.types.matrix(shape=block_shape, dtype=scalar_type)

    if topology == "padded":
        if src.block_shape[0] >= block_shape[0]:
            src_subrows = src.block_shape[0] // block_shape[0]
            dest_subrows = 1
        else:
            dest_subrows = block_shape[0] // src.block_shape[0]
            src_subrows = 1

        if src_subrows * block_shape[0] != src.block_shape[0] * dest_subrows:
            raise ValueError(
                f"Incompatible dest and src block shapes; block rows must evenly divide one another (Got {block_shape[0]}, {src.block_shape[0]})"
            )

        if src.block_shape[1] >= block_shape[1]:
            src_subcols = src.block_shape[1] // block_shape[1]
            dest_subcols = 1
        else:
            dest_subcols = block_shape[1] // src.block_shape[1]
            src_subcols = 1

        if src_subcols * block_shape[1] != src.block_shape[1] * dest_subcols:
            raise ValueError(
                f"Incompatible dest and src block shapes; block columns must evenly divide one another (Got {block_shape[1]}, {src.block_shape[1]})"
            )

        copy_nrow = (src.nrow * src_subrows) // dest_subrows
        copy_ncol = (src.ncol * src_subcols) // dest_subcols

        if src.nrow * src_subrows != copy_nrow * dest_subrows or src.ncol * src_subcols != copy_ncol * dest_subcols:
            raise ValueError(
                f"The requested block shape {block_shape} does not evenly divide the source matrix of total size {src.shape}"
            )

        copy = bsr_zeros(
            rows_of_blocks=copy_nrow,
            cols_of_blocks=copy_ncol,
            block_type=block_type,
            device=src.device,
        )
        copy.values.requires_grad = src.requires_grad

        if block_shape == src.block_shape:
            _bsr_ensure_fits(copy, nnz=src.nnz)
            wp.copy(dest=copy.offsets, src=src.offsets, count=src.nrow + 1)
            _bsr_ensure_independent_row_counts(copy, preserve_row_counts=False)
            if src.nrow > 0:
                if src.row_counts is None:
                    wp.launch(
                        _bsr_fill_row_counts_from_offsets,
                        dim=src.nrow,
                        device=src.device,
                        inputs=[src.nrow, src.offsets, copy.row_counts],
                    )
                else:
                    wp.copy(dest=copy.row_counts, src=src.row_counts, count=src.nrow)
            if src.nnz > 0:
                wp.copy(dest=copy.columns, src=src.columns, count=src.nnz)

                if not structure_only:
                    if types_equal(copy.values.dtype, src.values.dtype):
                        wp.copy(dest=copy.values, src=src.values, count=src.nnz)
                    else:
                        warp._src.utils.array_cast(out_array=copy.values, in_array=src.values, count=src.nnz)

                    if src_scale != 1.0:
                        bsr_scale(copy, src_scale)
        else:
            status = copy._ensure_status()
            max_nnz = src.nnz * src_subrows * src_subcols
            _bsr_ensure_fits(copy, nnz=max_nnz)
            if copy.nrow > 0:
                wp.launch(
                    _bsr_copy_reblocked_capacity_counts,
                    dim=copy.nrow,
                    device=copy.device,
                    inputs=[
                        src_subrows,
                        src_subcols,
                        dest_subrows,
                        src.nrow,
                        src.offsets,
                        copy.offsets,
                    ],
                )
                warp._src.utils.array_scan(copy.offsets, copy.offsets, inclusive=True)
            _bsr_ensure_independent_row_counts(copy, preserve_row_counts=False)
            _bsr_assign_padded_reblock(
                dest=copy,
                src=src,
                src_scale=src_scale,
                src_subrows=src_subrows,
                src_subcols=src_subcols,
                dest_subrows=dest_subrows,
                dest_subcols=dest_subcols,
                structure_only=structure_only,
                status=status,
            )
    else:
        copy = bsr_zeros(
            rows_of_blocks=src.nrow,
            cols_of_blocks=src.ncol,
            block_type=block_type,
            device=src.device,
        )
        copy.values.requires_grad = src.requires_grad
        bsr_assign(dest=copy, src=A, structure_only=structure_only, topology=topology)

    return copy


@wp.kernel(module="unique")
def _bsr_transpose_values(
    col_count: int,
    scale: Any,
    bsr_offsets: wp.array(dtype=int),
    bsr_row_counts: wp.array(dtype=int),
    bsr_columns: wp.array(dtype=int),
    bsr_values: wp.array3d(dtype=Any),
    block_index_map: wp.array(dtype=int),
    transposed_bsr_offsets: wp.array(dtype=int),
    transposed_bsr_row_counts: wp.array(dtype=int),
    transposed_bsr_columns: wp.array(dtype=int),
    transposed_bsr_values: wp.array3d(dtype=Any),
):
    block, i, j = wp.tid()

    if block >= transposed_bsr_offsets[col_count]:
        return

    if block_index_map:
        src_block = block_index_map[block]
        if src_block < 0:
            return
    else:
        row = bsr_row_index(transposed_bsr_offsets, col_count, block, transposed_bsr_row_counts)
        col = transposed_bsr_columns[block]
        src_block = bsr_block_index(col, row, bsr_offsets, bsr_columns, bsr_row_counts)
        if src_block == -1:
            return

    transposed_bsr_values[block, i, j] = bsr_values[src_block, j, i] * scale


def bsr_set_transpose(
    dest: BsrMatrix[BlockType[Cols, Rows, Scalar]],
    src: BsrMatrixOrExpression[BlockType[Rows, Cols, Scalar]],
    masked: bool = False,
    topology: Literal["compact", "masked", "padded"] | None = None,
):
    """Assign the transposed matrix ``src`` to matrix ``dest``.

    Args:
        dest: Sparse matrix to populate.
        src: Sparse matrix to transpose.
        masked: Deprecated. Use ``topology="masked"`` instead. If ``True``,
          keep the non-zero topology of ``dest`` unchanged.
        topology: Optional topology policy. ``"compact"`` keeps the existing
          compact rebuild behavior, ``"masked"`` is equivalent to
          the deprecated ``masked=True``, and ``"padded"`` writes the transposed active
          topology into existing destination row capacity and records capacity
          overflow in ``dest.status_sync()``.
    """

    src, src_scale = _extract_matrix_and_scale(src)

    if topology is None:
        topology = "masked" if masked else "compact"
    elif topology not in ("compact", "masked", "padded"):
        raise ValueError(f"Unsupported topology policy: {topology}")
    elif masked and topology != "masked":
        raise ValueError("Cannot pass masked=True with a non-masked topology policy")

    if masked:
        _warn_masked_arg_deprecated("bsr_set_transpose")

    masked = topology == "masked"

    if dest.values.device != src.values.device:
        raise ValueError(
            f"All arguments must reside on the same device, got {dest.values.device} and {src.values.device}"
        )

    if dest.scalar_type != src.scalar_type:
        raise ValueError(f"All arguments must have the same scalar type, got {dest.scalar_type} and {src.scalar_type}")

    transpose_block_shape = src.block_shape[::-1]

    if dest.block_shape != transpose_block_shape:
        raise ValueError(f"Destination block shape must be {transpose_block_shape}, got {dest.block_shape}")

    if topology == "padded":
        status = dest._ensure_status()
        if dest.nrow != src.ncol or dest.ncol != src.nrow:
            raise ValueError(
                f"Destination matrix must have {src.ncol} rows and {src.nrow} columns, got {dest.nrow} and {dest.ncol}"
            )

        nnz = src.nnz
        if nnz == 0:
            bsr_set_zero(dest, topology="padded")
            return

        _bsr_ensure_independent_row_counts(dest)

        from warp._src.context import runtime  # noqa: PLC0415

        if dest.values.device.is_cpu:
            native_func = runtime.core.wp_bsr_transpose_host
        else:
            native_func = runtime.core.wp_bsr_transpose_device

        dest_nnz = dest.columns.size
        block_index_map = wp.empty(shape=max(dest_nnz, 2 * nnz), dtype=int, device=src.device)
        if dest.values.device.is_cpu:
            apic_capture = runtime._apic_capture
            if apic_capture is not None and apic_capture.device.is_cpu:
                apic_capture.track_array(src.offsets)
                if src.row_counts is not None:
                    apic_capture.track_array(src.row_counts)
                apic_capture.track_array(src.columns)
                apic_capture.track_array(dest.offsets)
                apic_capture.track_array(dest.row_counts)
                apic_capture.track_array(dest.columns)
                apic_capture.track_array(block_index_map)
                apic_capture.track_array(status)
        with wp.ScopedDevice(dest.device):
            native_func(
                src.nrow,
                src.ncol,
                nnz,
                ctypes.cast(src.offsets.ptr, ctypes.POINTER(ctypes.c_int32)),
                _optional_ctypes_pointer(src.row_counts, ctype=ctypes.c_int32),
                ctypes.cast(src.columns.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(dest.offsets.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(dest.row_counts.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(dest.columns.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(block_index_map.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(status.ptr, ctypes.POINTER(ctypes.c_int32)),
            )

        wp.launch(
            _bsr_transpose_values,
            dim=(dest_nnz, *dest.block_shape),
            device=dest.device,
            inputs=[
                src.ncol,
                dest.scalar_type(src_scale),
                src.offsets,
                src.row_counts,
                src.columns,
                src.scalar_values,
                block_index_map,
                dest.offsets,
                dest.row_counts,
                dest.columns,
            ],
            outputs=[dest.scalar_values],
        )
        return

    if masked:
        if dest.nrow != src.ncol or dest.ncol != src.nrow:
            raise ValueError(
                f"Destination matrix must have {src.ncol} rows and {src.nrow} columns, got {dest.nrow} and {dest.ncol}"
            )
        block_index_map = None
        dest.values.zero_()
    else:
        _bsr_resize(dest, rows_of_blocks=src.ncol, cols_of_blocks=src.nrow)

        nnz = src.nnz
        if nnz == 0:
            bsr_set_zero(dest)
            return

        # Increase dest array sizes if needed
        _bsr_ensure_fits(dest, nnz=nnz)

        from warp._src.context import runtime  # noqa: PLC0415

        if dest.values.device.is_cpu:
            native_func = runtime.core.wp_bsr_transpose_host
        else:
            native_func = runtime.core.wp_bsr_transpose_device

        block_index_map = wp.empty(shape=2 * nnz, dtype=int, device=src.device)

        if dest.values.device.is_cpu:
            # On CPU the transpose dispatches to wp_bsr_transpose_host, which
            # records an APIC_OP_BSR_TRANSPOSE op under capture. Track the
            # input/output base regions first so the recorded op references real
            # region IDs.
            apic_capture = runtime._apic_capture
            if apic_capture is not None and apic_capture.device.is_cpu:
                apic_capture.track_array(src.offsets)
                if src.row_counts is not None:
                    apic_capture.track_array(src.row_counts)
                apic_capture.track_array(src.columns)
                apic_capture.track_array(dest.offsets)
                if dest.row_counts is not None:
                    apic_capture.track_array(dest.row_counts)
                apic_capture.track_array(dest.columns)
                apic_capture.track_array(block_index_map)
                _mark_apic_deferred_nnz_update(dest, apic_capture)

        with wp.ScopedDevice(dest.device):
            native_func(
                src.nrow,
                src.ncol,
                nnz,
                ctypes.cast(src.offsets.ptr, ctypes.POINTER(ctypes.c_int32)),
                _optional_ctypes_pointer(src.row_counts, ctype=ctypes.c_int32),
                ctypes.cast(src.columns.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(dest.offsets.ptr, ctypes.POINTER(ctypes.c_int32)),
                None,  # transposed row counts
                ctypes.cast(dest.columns.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(block_index_map.ptr, ctypes.POINTER(ctypes.c_int32)),
                None,  # status
            )

            dest._copy_nnz_async()
            _bsr_set_compact_row_counts(dest)

    wp.launch(
        _bsr_transpose_values,
        dim=(dest.nnz, *dest.block_shape),
        device=dest.device,
        inputs=[
            src.ncol,
            dest.scalar_type(src_scale),
            src.offsets,
            src.row_counts,
            src.columns,
            src.scalar_values,
            block_index_map,
            dest.offsets,
            dest.row_counts,
            dest.columns,
        ],
        outputs=[dest.scalar_values],
    )


def bsr_transposed(A: BsrMatrixOrExpression) -> BsrMatrix:
    """Return a copy of the transposed matrix ``A``."""

    if A.block_shape == (1, 1):
        block_type = A.values.dtype
    else:
        block_type = wp.types.matrix(shape=A.block_shape[::-1], dtype=A.scalar_type)

    transposed = bsr_zeros(
        rows_of_blocks=A.ncol,
        cols_of_blocks=A.nrow,
        block_type=block_type,
        device=A.device,
    )
    transposed.values.requires_grad = A.requires_grad
    bsr_set_transpose(dest=transposed, src=A)
    return transposed


@wp.kernel(module="unique")
def _bsr_get_diag_kernel(
    scale: Any,
    A_offsets: wp.array(dtype=int),
    A_row_counts: wp.array(dtype=int),
    A_columns: wp.array(dtype=int),
    A_values: wp.array3d(dtype=Any),
    out: wp.array3d(dtype=Any),
):
    row, br, bc = wp.tid()

    diag = bsr_block_index(row, row, A_offsets, A_columns, A_row_counts)
    if diag != -1:
        out[row, br, bc] = scale * A_values[diag, br, bc]


def bsr_get_diag(A: BsrMatrixOrExpression[BlockType], out: Array[BlockType] | None = None) -> Array[BlockType]:
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
        if not types_equal(out.dtype, A.values.dtype):
            raise ValueError(f"Output array must have type {A.values.dtype}, got {out.dtype}")
        if out.device != A.values.device:
            raise ValueError(f"Output array must reside on device {A.values.device}, got {out.device}")
        if out.shape[0] < dim:
            raise ValueError(f"Output array must be of length at least {dim}, got {out.shape[0]}")
        out.zero_()

    wp.launch(
        kernel=_bsr_get_diag_kernel,
        dim=(dim, *A.block_shape),
        device=A.values.device,
        inputs=[
            A.scalar_type(scale),
            A.offsets,
            A.row_counts,
            A.columns,
            A.scalar_values,
            _as_3d_array(out, A.block_shape),
        ],
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
    diag: BlockType | Array[BlockType],
    rows_of_blocks: int | None = None,
    cols_of_blocks: int | None = None,
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
        _bsr_resize(A, rows_of_blocks, cols_of_blocks)

    nnz = min(A.nrow, A.ncol)
    _bsr_ensure_fits(A, nnz=nnz)

    wp.launch(
        kernel=_bsr_set_diag_kernel,
        dim=A.nrow + 1,
        device=A.offsets.device,
        inputs=[nnz, A.offsets, A.columns],
    )
    _bsr_set_compact_row_counts(A)

    A.notify_nnz_changed(nnz=nnz)  # notify change of offsets

    if is_array(diag):
        wp.copy(src=diag, dest=A.values, count=nnz)
    elif diag is not None:
        A.values.fill_(diag)


def bsr_diag(
    diag: BlockType | Array[BlockType] | None = None,
    rows_of_blocks: int | None = None,
    cols_of_blocks: int | None = None,
    block_type: BlockType | None = None,
    device=None,
) -> BsrMatrix[BlockType]:
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
            block_type = wp.types.matrix(shape=diag.shape, dtype=diag.dtype)

    A = bsr_zeros(rows_of_blocks, cols_of_blocks, block_type=block_type, device=device)
    if is_array(diag):
        A.values.requires_grad = diag.requires_grad
    bsr_set_diag(A, diag)
    return A


def bsr_set_identity(A: BsrMatrix, rows_of_blocks: int | None = None) -> None:
    """Set ``A`` as the identity matrix.

    Args:
        A: The sparse matrix to modify.
        rows_of_blocks: If provided, the matrix will be resized as a square
          matrix with ``rows_of_blocks`` rows and columns.
    """

    if A.block_shape == (1, 1):
        identity = A.scalar_type(1.0)
    else:
        identity = eye(A.block_shape[0])

    bsr_set_diag(A, diag=identity, rows_of_blocks=rows_of_blocks, cols_of_blocks=rows_of_blocks)


def bsr_identity(
    rows_of_blocks: int,
    block_type: BlockType[Rows, Rows, Scalar],
    device: wp.DeviceLike = None,
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


@wp.kernel(module="unique")
def _bsr_scale_1d_kernel(
    alpha: Any,
    values: wp.array(dtype=Any),
):
    row = wp.tid()
    values[row] = alpha * values[row]


@wp.kernel(module="unique")
def _bsr_scale_3d_kernel(
    alpha: Any,
    values: wp.array3d(dtype=Any),
):
    row, br, bc = wp.tid()
    values[row, br, bc] = alpha * values[row, br, bc]


def bsr_scale(x: BsrMatrixOrExpression, alpha: Scalar) -> BsrMatrix:
    """Perform the operation ``x := alpha * x`` on BSR matrix ``x`` and return ``x``."""

    x, scale = _extract_matrix_and_scale(x)
    alpha *= scale

    if alpha != 1.0 and x.nnz > 0:
        if alpha == 0.0:
            x.values.zero_()
        else:
            alpha = x.scalar_type(alpha)

            wp.launch(
                kernel=_bsr_scale_3d_kernel,
                dim=(x.nnz, *x.block_shape),
                device=x.values.device,
                inputs=[alpha, x.scalar_values],
            )

    return x


@wp.kernel(enable_backward=False)
def _bsr_get_block_row(
    row_count: int, bsr_offsets: wp.array(dtype=int), bsr_row_counts: wp.array(dtype=int), rows: wp.array(dtype=int)
):
    block = wp.tid()
    rows[block] = bsr_row_index(bsr_offsets, row_count, block, bsr_row_counts)


@wp.kernel(module="unique")
def _bsr_axpy_add_block(
    src_offset: int,
    scale: Any,
    rows: wp.array(dtype=int),
    cols: wp.array(dtype=int),
    dst_offsets: wp.array(dtype=int),
    dst_row_counts: wp.array(dtype=int),
    dst_columns: wp.array(dtype=int),
    src_values: wp.array3d(dtype=Any),
    dst_values: wp.array3d(dtype=Any),
):
    i, br, bc = wp.tid()
    row = rows[i + src_offset]
    col = cols[i + src_offset]

    block = bsr_block_index(row, col, dst_offsets, dst_columns, dst_row_counts)
    if block != -1:
        dst_values[block, br, bc] += scale * src_values[i, br, bc]


@wp.kernel(module="unique")
def _bsr_axpy_masked(
    alpha: Any,
    row_count: int,
    src_offsets: wp.array(dtype=int),
    src_row_counts: wp.array(dtype=int),
    src_columns: wp.array(dtype=int),
    src_values: wp.array3d(dtype=Any),
    dst_offsets: wp.array(dtype=int),
    dst_row_counts: wp.array(dtype=int),
    dst_columns: wp.array(dtype=int),
    dst_values: wp.array3d(dtype=Any),
):
    block, br, bc = wp.tid()

    row = bsr_row_index(dst_offsets, row_count, block, dst_row_counts)
    if row == -1:
        return

    col = dst_columns[block]
    src_block = bsr_block_index(row, col, src_offsets, src_columns, src_row_counts)
    if src_block != -1:
        dst_values[block, br, bc] += alpha * src_values[src_block, br, bc]


@wp.kernel(enable_backward=False)
def _bsr_axpy_padded_count(
    x_offsets: wp.array(dtype=int),
    x_row_counts: wp.array(dtype=int),
    x_columns: wp.array(dtype=int),
    y_offsets: wp.array(dtype=int),
    y_row_counts: wp.array(dtype=int),
    y_columns: wp.array(dtype=int),
    row_block_counts: wp.array(dtype=int),
    status: wp.array(dtype=int),
):
    row = wp.tid()

    x_block = int(x_offsets[row])
    x_end = _bsr_row_end(x_offsets, x_row_counts, row)
    y_block = int(y_offsets[row])
    y_end = _bsr_row_end(y_offsets, y_row_counts, row)

    block_count = int(0)

    while x_block < x_end and y_block < y_end:
        x_col = x_columns[x_block]
        y_col = y_columns[y_block]

        block_count += 1
        if x_col == y_col:
            x_block += 1
            y_block += 1
        elif x_col < y_col:
            x_block += 1
        else:
            y_block += 1

    block_count += x_end - x_block
    block_count += y_end - y_block

    if y_offsets[row] + block_count > y_offsets[row + 1]:
        row_block_counts[row] = -1
        wp.atomic_max(status, 0, _BSR_STATUS_ROW_CAPACITY_EXCEEDED)
    else:
        row_block_counts[row] = block_count


@wp.kernel(module="unique")
def _bsr_axpy_padded_fill(
    alpha: Any,
    beta: Any,
    x_offsets: wp.array(dtype=int),
    x_row_counts: wp.array(dtype=int),
    x_columns: wp.array(dtype=int),
    x_values: wp.array3d(dtype=Any),
    y_offsets: wp.array(dtype=int),
    y_row_counts: wp.array(dtype=int),
    y_columns: wp.array(dtype=int),
    y_values: wp.array3d(dtype=Any),
    row_block_counts: wp.array(dtype=int),
):
    row, br, bc = wp.tid()

    block_count = row_block_counts[row]
    if block_count < 0:
        return

    x_beg = x_offsets[row]
    x_block = _bsr_row_end(x_offsets, x_row_counts, row) - 1
    y_beg = y_offsets[row]
    y_block = _bsr_row_end(y_offsets, y_row_counts, row) - 1
    write = int(y_beg + block_count - 1)

    while write >= y_beg:
        use_x = bool(False)
        use_y = bool(False)
        col = int(0)

        if x_block >= x_beg and y_block >= y_beg:
            x_col = x_columns[x_block]
            y_col = y_columns[y_block]
            if x_col == y_col:
                use_x = True
                use_y = True
                col = x_col
            elif x_col > y_col:
                use_x = True
                col = x_col
            else:
                use_y = True
                col = y_col
        elif x_block >= x_beg:
            use_x = True
            col = x_columns[x_block]
        else:
            use_y = True
            col = y_columns[y_block]

        value = y_values.dtype(0.0)
        if use_x:
            value += alpha * x_values[x_block, br, bc]
            x_block -= 1
        if use_y:
            value += beta * y_values[y_block, br, bc]
            y_block -= 1

        if br == 0 and bc == 0:
            y_columns[write] = col
        y_values[write, br, bc] = value

        write -= 1


@wp.kernel(enable_backward=False)
def _bsr_axpy_padded_finalize(
    y_row_counts: wp.array(dtype=int),
    row_block_counts: wp.array(dtype=int),
):
    row = wp.tid()

    block_count = row_block_counts[row]

    if block_count < 0:
        y_row_counts[row] = 0
    else:
        y_row_counts[row] = block_count


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
            self._old_y_values = wp.empty_like(y.values[: y.nnz])


def bsr_axpy(
    x: BsrMatrixOrExpression,
    y: BsrMatrix[BlockType[Rows, Cols, Scalar]] | None = None,
    alpha: Scalar = 1.0,
    beta: Scalar = 1.0,
    masked: bool = False,
    work_arrays: bsr_axpy_work_arrays | None = None,
    topology: Literal["compact", "masked", "padded"] | None = None,
) -> BsrMatrix[BlockType[Rows, Cols, Scalar]]:
    """Perform the sparse matrix addition ``y := alpha * X + beta * y`` on BSR matrices ``x`` and ``y`` and return ``y``.

    The ``x`` and ``y`` matrices are allowed to alias.

    Args:
        x: Read-only first operand.
        y: Mutable second operand and output matrix. If ``y`` is not provided, it will be allocated and treated as zero.
        alpha: Uniform scaling factor for ``x``.
        beta: Uniform scaling factor for ``y``.
        masked: Deprecated. Use ``topology="masked"`` instead. If ``True``,
          keep the non-zero topology of ``y`` unchanged.
        work_arrays: In most cases, this function will require the use of temporary storage.
          This storage can be reused across calls by passing an instance of
          :class:`bsr_axpy_work_arrays` in ``work_arrays``.
        topology: Optional topology policy. ``"compact"`` keeps the existing
          compact rebuild behavior, ``"masked"`` is equivalent to
          the deprecated ``masked=True``, and ``"padded"`` writes the result topology into
          existing destination row capacity and records capacity overflow in
          ``y.status_sync()``.
    """

    x, x_scale = _extract_matrix_and_scale(x)
    alpha *= x_scale

    if topology is None:
        topology = "masked" if masked else "compact"
    elif topology not in ("compact", "masked", "padded"):
        raise ValueError(f"Unsupported topology policy: {topology}")
    elif masked and topology != "masked":
        raise ValueError("Cannot pass masked=True with a non-masked topology policy")

    if masked:
        _warn_masked_arg_deprecated("bsr_axpy")

    masked = topology == "masked"

    if y is None:
        if masked or topology == "padded":
            raise ValueError("Left-hand-side 'y' matrix must be provided for this topology policy")

        # If not output matrix is provided, allocate it for convenience
        y = bsr_zeros(x.nrow, x.ncol, block_type=x.values.dtype, device=x.values.device)
        y.values.requires_grad = x.requires_grad
        beta = 0.0

    x_nnz = x.nnz
    y_nnz = y.nnz

    if topology == "padded":
        if x.values.device != y.values.device:
            raise ValueError(
                f"All arguments must reside on the same device, got {x.values.device} and {y.values.device}"
            )

        if x.scalar_type != y.scalar_type or x.block_shape != y.block_shape:
            raise ValueError(
                f"Matrices must have the same block type, got ({x.block_shape}, {x.scalar_type}) and ({y.block_shape}, {y.scalar_type})"
            )

        if x.nrow != y.nrow or x.ncol != y.ncol:
            raise ValueError(
                f"Matrices must have the same number of rows and columns, got ({x.nrow}, {x.ncol}) and ({y.nrow}, {y.ncol})"
            )

        status = y._ensure_status()
        if beta == 0.0:
            if x == y:
                return bsr_scale(y, alpha=alpha)

            _bsr_ensure_independent_row_counts(y)
            _bsr_assign_padded_same_block(dest=y, src=x, src_scale=alpha, status=status)
            return y

        if alpha == 0.0 or x_nnz == 0:
            return bsr_scale(y, alpha=beta)

        if x == y:
            return bsr_scale(y, alpha=alpha + beta)

        _bsr_ensure_independent_row_counts(y)

        if not isinstance(alpha, y.scalar_type):
            alpha = y.scalar_type(alpha)
        if not isinstance(beta, y.scalar_type):
            beta = y.scalar_type(beta)

        if work_arrays is None:
            work_arrays = bsr_axpy_work_arrays()

        work_arrays._allocate(y.device, y, max(x_nnz + y_nnz, y.nrow))
        row_block_counts = work_arrays._sum_rows

        wp.launch(
            _bsr_axpy_padded_count,
            dim=y.nrow,
            device=y.device,
            inputs=[
                x.offsets,
                x.row_counts,
                x.columns,
                y.offsets,
                y.row_counts,
                y.columns,
                row_block_counts,
                status,
            ],
        )

        wp.launch(
            _bsr_axpy_padded_fill,
            dim=(y.nrow, *y.block_shape),
            device=y.device,
            inputs=[
                alpha,
                beta,
                x.offsets,
                x.row_counts,
                x.columns,
                x.scalar_values,
                y.offsets,
                y.row_counts,
                y.columns,
                y.scalar_values,
                row_block_counts,
            ],
        )

        wp.launch(
            _bsr_axpy_padded_finalize,
            dim=y.nrow,
            device=y.device,
            inputs=[
                y.row_counts,
                row_block_counts,
            ],
        )
        return y

    # Handle easy cases first
    if beta == 0.0 or y_nnz == 0:
        bsr_assign(src=x, dest=y, topology=topology)
        return bsr_scale(y, alpha=alpha)

    if alpha == 0.0 or x_nnz == 0:
        return bsr_scale(y, alpha=beta)

    if x == y:
        # Aliasing case
        return bsr_scale(y, alpha=alpha + beta)

    # General case
    if not isinstance(alpha, y.scalar_type):
        alpha = y.scalar_type(alpha)
    if not isinstance(beta, y.scalar_type):
        beta = y.scalar_type(beta)

    if x.values.device != y.values.device:
        raise ValueError(f"All arguments must reside on the same device, got {x.values.device} and {y.values.device}")

    if x.scalar_type != y.scalar_type or x.block_shape != y.block_shape:
        raise ValueError(
            f"Matrices must have the same block type, got ({x.block_shape}, {x.scalar_type}) and ({y.block_shape}, {y.scalar_type})"
        )

    if x.nrow != y.nrow or x.ncol != y.ncol:
        raise ValueError(
            f"Matrices must have the same number of rows and columns, got ({x.nrow}, {x.ncol}) and ({y.nrow}, {y.ncol})"
        )

    device = y.values.device
    if masked:
        bsr_scale(y, alpha=beta.value)
        wp.launch(
            kernel=_bsr_axpy_masked,
            device=device,
            dim=(y_nnz, y.block_shape[0], y.block_shape[1]),
            inputs=[
                alpha,
                x.nrow,
                x.offsets,
                x.row_counts,
                x.columns,
                x.scalar_values,
                y.offsets,
                y.row_counts,
                y.columns,
                y.scalar_values,
            ],
        )

    else:
        if work_arrays is None:
            work_arrays = bsr_axpy_work_arrays()

        sum_nnz = x_nnz + y_nnz
        work_arrays._allocate(device, y, sum_nnz)

        wp.copy(work_arrays._sum_cols, y.columns, 0, 0, y_nnz)
        y.uncompress_rows(out=work_arrays._sum_rows)

        wp.copy(work_arrays._sum_cols, x.columns, y_nnz, 0, x_nnz)
        x.uncompress_rows(out=work_arrays._sum_rows[y_nnz:])

        # Save old y values before overwriting matrix
        wp.copy(dest=work_arrays._old_y_values, src=y.values, count=y.nnz)

        # Increase dest array sizes if needed
        _bsr_ensure_fits(y, nnz=sum_nnz)

        old_y_nnz = y_nnz
        nnz_buf, nnz_event = y._setup_nnz_transfer()

        _bsr_set_from_triplets_native(
            dest=y,
            rows=work_arrays._sum_rows,
            columns=work_arrays._sum_cols,
            values=None,
            count=None,
            nnz=sum_nnz,
            zero_value_mask=0,
            masked=masked,
            summed_triplet_offsets=None,
            summed_triplet_indices=None,
            dest_offsets=y.offsets,
            dest_row_counts=None,
            dest_columns=y.columns,
            nnz_buf=nnz_buf,
            nnz_event=nnz_event,
        )
        _bsr_set_compact_row_counts(y)

        y.values.zero_()

        wp.launch(
            kernel=_bsr_axpy_add_block,
            device=device,
            dim=(old_y_nnz, y.block_shape[0], y.block_shape[1]),
            inputs=[
                0,
                beta,
                work_arrays._sum_rows,
                work_arrays._sum_cols,
                y.offsets,
                y.row_counts,
                y.columns,
                _as_3d_array(work_arrays._old_y_values, y.block_shape),
                y.scalar_values,
            ],
        )

        wp.launch(
            kernel=_bsr_axpy_add_block,
            device=device,
            dim=(x_nnz, y.block_shape[0], y.block_shape[1]),
            inputs=[
                old_y_nnz,
                alpha,
                work_arrays._sum_rows,
                work_arrays._sum_cols,
                y.offsets,
                y.row_counts,
                y.columns,
                x.scalar_values,
                y.scalar_values,
            ],
        )

    return y


@cache
def make_bsr_mm_count_coeffs(tile_size):

    @wp.kernel(module="unique")
    def bsr_mm_count_coeffs(
        y_ncol: int,
        z_nnz: int,
        x_offsets: wp.array(dtype=int),
        x_row_counts: wp.array(dtype=int),
        x_columns: wp.array(dtype=int),
        y_offsets: wp.array(dtype=int),
        y_row_counts: wp.array(dtype=int),
        y_columns: wp.array(dtype=int),
        row_min: wp.array(dtype=int),
        block_counts: wp.array(dtype=int),
    ):
        row, lane = wp.tid()
        row_count = int(0)

        x_beg = x_offsets[row]
        x_end = _bsr_row_end(x_offsets, x_row_counts, row)

        min_col = y_ncol
        max_col = int(0)

        for x_block in range(x_beg + lane, x_end, tile_size):
            x_col = x_columns[x_block]
            y_row_beg = y_offsets[x_col]
            y_row_end = _bsr_row_end(y_offsets, y_row_counts, x_col)
            block_count = y_row_end - y_row_beg
            if block_count != 0:
                min_col = wp.min(y_columns[y_row_beg], min_col)
                max_col = wp.max(y_columns[y_row_end - 1], max_col)

            block_counts[x_block + 1] = block_count
            row_count += block_count

        if wp.static(tile_size) > 1:
            row_count = wp.tile_sum(wp.tile(row_count))[0]
            min_col = wp.tile_min(wp.tile(min_col))[0]
            max_col = wp.tile_max(wp.tile(max_col))[0]
        col_range_size = wp.max(0, max_col - min_col + 1)

        if row_count > col_range_size:
            # Optimization for deep products.
            # Do not store the whole whole list of src product terms, they would be highly redundant
            # Instead just mark a range in the output matrix

            if lane == 0:
                row_min[row] = min_col
                block_counts[x_end] = col_range_size

            for x_block in range(x_beg + lane, x_end - 1, tile_size):
                block_counts[x_block + 1] = 0
        elif lane == 0:
            row_min[row] = -1

        if lane == 0 and row == 0:
            block_counts[0] = z_nnz

    return bsr_mm_count_coeffs


@wp.kernel(enable_backward=False)
def _bsr_mm_list_coeffs(
    copied_z_nnz: int,
    mm_nnz: int,
    x_nrow: int,
    x_offsets: wp.array(dtype=int),
    x_row_counts: wp.array(dtype=int),
    x_columns: wp.array(dtype=int),
    y_offsets: wp.array(dtype=int),
    y_row_counts: wp.array(dtype=int),
    y_columns: wp.array(dtype=int),
    mm_row_min: wp.array(dtype=int),
    mm_offsets: wp.array(dtype=int),
    mm_rows: wp.array(dtype=int),
    mm_cols: wp.array(dtype=int),
    mm_src_blocks: wp.array(dtype=int),
):
    mm_block = wp.tid() + copied_z_nnz

    x_nnz = x_offsets[x_nrow]

    x_block = bsr_row_index(mm_offsets, x_nnz, mm_block)

    if x_block == -1:
        mm_cols[mm_block] = -1
        mm_rows[mm_block] = -1
        return

    if mm_block + 1 == mm_nnz and mm_nnz < mm_offsets[x_nnz]:
        wp.printf(
            "Number of potential `bsr_mm` blocks (%d) exceeded `max_nnz` (%d)\n",
            mm_offsets[x_nnz] - copied_z_nnz,
            mm_nnz - copied_z_nnz,
        )

    pos = mm_block - mm_offsets[x_block]

    row = bsr_row_index(x_offsets, x_nrow, x_block, x_row_counts)

    row_min_col = mm_row_min[row]
    if row_min_col == -1:
        x_col = x_columns[x_block]
        y_beg = y_offsets[x_col]
        y_block = y_beg + pos
        col = y_columns[y_block]
        src_block = x_block
    else:
        col = row_min_col + pos
        src_block = -1

    mm_cols[mm_block] = col
    mm_rows[mm_block] = row
    mm_src_blocks[mm_block] = src_block


@wp.func
def _bsr_mm_use_triplets(
    row: int,
    mm_block: int,
    mm_row_min: wp.array(dtype=int),
    row_offsets: wp.array(dtype=int),
    row_counts: wp.array(dtype=int),
    summed_triplet_offsets: wp.array(dtype=int),
):
    x_beg = row_offsets[row]
    x_end = _bsr_row_end(row_offsets, row_counts, row)

    if mm_row_min:
        if mm_row_min[row] == -1:
            if mm_block == 0:
                block_beg = 0
            else:
                block_beg = summed_triplet_offsets[mm_block - 1]
            block_end = summed_triplet_offsets[mm_block]

            if x_end - x_beg > 3 * (block_end - block_beg):
                return True, block_beg, block_end

    return False, x_beg, x_end


@wp.kernel(enable_backward=False, module="unique")
def _bsr_mm_compute_values(
    alpha: Any,
    x_offsets: wp.array(dtype=int),
    x_row_counts: wp.array(dtype=int),
    x_columns: wp.array(dtype=int),
    x_values: wp.array(dtype=Any),
    y_offsets: wp.array(dtype=int),
    y_row_counts: wp.array(dtype=int),
    y_columns: wp.array(dtype=int),
    y_values: wp.array(dtype=Any),
    mm_row_min: wp.array(dtype=int),
    summed_triplet_offsets: wp.array(dtype=int),
    summed_triplet_src_blocks: wp.indexedarray(dtype=int),
    mm_row_count: int,
    mm_offsets: wp.array(dtype=int),
    mm_row_counts: wp.array(dtype=int),
    mm_cols: wp.array(dtype=int),
    mm_values: wp.array(dtype=Any),
):
    mm_block = wp.tid()

    row = bsr_row_index(mm_offsets, mm_row_count, mm_block, mm_row_counts)
    if row == -1:
        return

    use_triplets, block_beg, block_end = _bsr_mm_use_triplets(
        row, mm_block, mm_row_min, x_offsets, x_row_counts, summed_triplet_offsets
    )

    mm_val = mm_values.dtype(type(alpha)(0.0))
    col = mm_cols[mm_block]
    if use_triplets:
        for tpl_idx in range(block_beg, block_end):
            x_block = summed_triplet_src_blocks[tpl_idx]
            x_col = x_columns[x_block]
            if x_block != -1:
                y_block = bsr_block_index(x_col, col, y_offsets, y_columns, y_row_counts)
                mm_val += x_values[x_block] * y_values[y_block]
    else:
        for x_block in range(block_beg, block_end):
            x_col = x_columns[x_block]
            y_block = bsr_block_index(x_col, col, y_offsets, y_columns, y_row_counts)
            if y_block != -1:
                mm_val += x_values[x_block] * y_values[y_block]

    mm_values[mm_block] += alpha * mm_val


@cache
def make_bsr_mm_compute_values_tiled_outer(subblock_rows, subblock_cols, block_depth, scalar_type, tile_size):
    mm_type = wp.types.matrix(dtype=scalar_type, shape=(subblock_rows, subblock_cols))

    x_col_vec_t = wp.types.vector(dtype=scalar_type, length=subblock_rows)
    y_row_vec_t = wp.types.vector(dtype=scalar_type, length=subblock_cols)

    @wp.func
    def _outer_product(
        x_values: wp.array2d(dtype=Any),
        y_values: wp.array2d(dtype=Any),
        brow_off: int,
        bcol_off: int,
        block_col: int,
        brow_count: int,
        bcol_count: int,
    ):
        x_col_vec = x_col_vec_t()
        y_row_vec = y_row_vec_t()

        for k in range(brow_count):
            x_col_vec[k] = x_values[brow_off + k, block_col]
        for k in range(bcol_count):
            y_row_vec[k] = y_values[block_col, bcol_off + k]

        return wp.outer(x_col_vec, y_row_vec)

    @wp.kernel(enable_backward=False, module="unique")
    def bsr_mm_compute_values(
        alpha: Any,
        x_offsets: wp.array(dtype=int),
        x_row_counts: wp.array(dtype=int),
        x_columns: wp.array(dtype=int),
        x_values: wp.array3d(dtype=Any),
        y_offsets: wp.array(dtype=int),
        y_row_counts: wp.array(dtype=int),
        y_columns: wp.array(dtype=int),
        y_values: wp.array3d(dtype=Any),
        mm_row_min: wp.array(dtype=int),
        summed_triplet_offsets: wp.array(dtype=int),
        summed_triplet_src_blocks: wp.indexedarray(dtype=int),
        mm_row_count: int,
        mm_offsets: wp.array(dtype=int),
        mm_row_counts: wp.array(dtype=int),
        mm_cols: wp.array(dtype=int),
        mm_values: wp.array3d(dtype=Any),
    ):
        mm_block, subrow, subcol, lane = wp.tid()

        brow_off = subrow * wp.static(subblock_rows)
        bcol_off = subcol * wp.static(subblock_cols)

        brow_count = wp.min(mm_values.shape[1] - brow_off, subblock_rows)
        bcol_count = wp.min(mm_values.shape[2] - bcol_off, subblock_cols)

        mm_row = bsr_row_index(mm_offsets, mm_row_count, mm_block, mm_row_counts)
        if mm_row == -1:
            return

        lane_val = mm_type()

        use_triplets, block_beg, block_end = _bsr_mm_use_triplets(
            mm_row, mm_block, mm_row_min, x_offsets, x_row_counts, summed_triplet_offsets
        )

        col_count = (block_end - block_beg) * block_depth

        mm_col = mm_cols[mm_block]
        if use_triplets:
            for col in range(lane, col_count, tile_size):
                tpl_block = col // wp.static(block_depth)
                block_col = col - tpl_block * wp.static(block_depth)
                tpl_block += block_beg

                x_block = summed_triplet_src_blocks[tpl_block]
                if x_block != -1:
                    x_col = x_columns[x_block]
                    y_block = bsr_block_index(x_col, mm_col, y_offsets, y_columns, y_row_counts)
                    lane_val += _outer_product(
                        x_values[x_block], y_values[y_block], brow_off, bcol_off, block_col, brow_count, bcol_count
                    )
        else:
            for col in range(lane, col_count, tile_size):
                x_block = col // wp.static(block_depth)
                block_col = col - x_block * wp.static(block_depth)
                x_block += block_beg

                x_col = x_columns[x_block]
                y_block = bsr_block_index(x_col, mm_col, y_offsets, y_columns, y_row_counts)

                if y_block != -1:
                    lane_val += _outer_product(
                        x_values[x_block], y_values[y_block], brow_off, bcol_off, block_col, brow_count, bcol_count
                    )

        mm_val = wp.tile_sum(wp.tile(lane_val, preserve_type=True))[0]

        for coef in range(lane, wp.static(subblock_cols * subblock_rows), tile_size):
            br = coef // subblock_cols
            bc = coef - br * subblock_cols
            if br < brow_count and bc < bcol_count:
                mm_values[mm_block, br + brow_off, bc + bcol_off] += mm_val[br, bc] * alpha

    return bsr_mm_compute_values


@wp.kernel(enable_backward=False)
def _bsr_mm_padded_topology(
    beta_nonzero: bool,
    col_count: int,
    x_offsets: wp.array(dtype=int),
    x_row_counts: wp.array(dtype=int),
    x_columns: wp.array(dtype=int),
    y_offsets: wp.array(dtype=int),
    y_row_counts: wp.array(dtype=int),
    y_columns: wp.array(dtype=int),
    z_offsets: wp.array(dtype=int),
    old_z_row_counts: wp.array(dtype=int),
    old_z_columns: wp.array(dtype=int),
    z_row_counts: wp.array(dtype=int),
    z_columns: wp.array(dtype=int),
    y_cursors: wp.array(dtype=int),
    status: wp.array(dtype=int),
):
    row = wp.tid()

    row_beg = z_offsets[row]
    capacity_end = z_offsets[row + 1]
    write = row_beg
    searching = bool(True)
    overflow = bool(False)
    previous_col = int(-1)
    x_beg = x_offsets[row]
    x_end = _bsr_row_end(x_offsets, x_row_counts, row)
    old_z_end = _bsr_row_end(z_offsets, old_z_row_counts, row)
    old_z_block = row_beg

    for x_block in range(x_beg, x_end):
        y_cursors[x_block] = y_offsets[x_columns[x_block]]

    while searching:
        next_col = col_count

        if beta_nonzero:
            while old_z_block < old_z_end and old_z_columns[old_z_block] <= previous_col:
                old_z_block += 1
            if old_z_block < old_z_end:
                col = old_z_columns[old_z_block]
                if col < next_col:
                    next_col = col

        for x_block in range(x_beg, x_end):
            x_col = x_columns[x_block]
            y_row_end = _bsr_row_end(y_offsets, y_row_counts, x_col)
            y_block = y_cursors[x_block]
            while y_block < y_row_end and y_columns[y_block] <= previous_col:
                y_block += 1
            y_cursors[x_block] = y_block
            if y_block < y_row_end:
                col = y_columns[y_block]
                if col > previous_col and col < next_col:
                    next_col = col

        if next_col == col_count:
            searching = False
        elif write >= capacity_end:
            overflow = True
            searching = False
        else:
            z_columns[write] = next_col
            write += 1
            previous_col = next_col

    if overflow:
        z_row_counts[row] = 0
        wp.atomic_max(status, 0, _BSR_STATUS_ROW_CAPACITY_EXCEEDED)
    else:
        z_row_counts[row] = write - row_beg


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
        self._mm_src_blocks = None
        self._old_z_values = None
        self._old_z_offsets = None
        self._old_z_row_counts = None
        self._old_z_columns = None
        self._mm_y_cursors = None
        self._mm_nnz = 0

    def _allocate_stage_1(self, device, x_nnz: int, z: BsrMatrix, beta: float, z_aliasing: bool):
        if self.device != device:
            self._reset(device)

        # Allocations that do not depend on any computation
        self._copied_z_nnz = z.nnz if beta != 0.0 or z_aliasing else 0

        if self._mm_row_min is None or self._mm_row_min.size < z.nrow + 1:
            self._mm_row_min = wp.empty(shape=(z.nrow + 1,), dtype=int, device=self.device)
        if self._mm_block_counts is None or self._mm_block_counts.size < x_nnz + 1:
            self._mm_block_counts = wp.empty(shape=(x_nnz + 1,), dtype=int, device=self.device)

        if self._copied_z_nnz > 0:
            if self._old_z_values is None or self._old_z_values.size < self._copied_z_nnz:
                self._old_z_values = wp.empty(shape=(self._copied_z_nnz,), dtype=z.values.dtype, device=self.device)

        if z_aliasing:
            if self._old_z_columns is None or self._old_z_columns.size < z.nnz:
                self._old_z_columns = wp.empty(shape=(z.nnz,), dtype=z.columns.dtype, device=self.device)
            if self._old_z_offsets is None or self._old_z_offsets.size < z.nrow + 1:
                self._old_z_offsets = wp.empty(shape=(z.nrow + 1,), dtype=z.offsets.dtype, device=self.device)
            if self._old_z_row_counts is None or self._old_z_row_counts.size < z.nrow:
                self._old_z_row_counts = wp.empty(shape=(z.nrow,), dtype=int, device=self.device)

    def _allocate_snapshot(self, device, z: BsrMatrix, rows: bool = False):
        if self.device != device:
            self._reset(device)

        if self._old_z_values is None or self._old_z_values.size < z.nnz:
            self._old_z_values = wp.empty(shape=(z.nnz,), dtype=z.values.dtype, device=self.device)
        if self._old_z_columns is None or self._old_z_columns.size < z.nnz:
            self._old_z_columns = wp.empty(shape=(z.nnz,), dtype=z.columns.dtype, device=self.device)
        if self._old_z_row_counts is None or self._old_z_row_counts.size < z.nrow:
            self._old_z_row_counts = wp.empty(shape=(z.nrow,), dtype=int, device=self.device)
        if rows and (self._mm_rows is None or self._mm_rows.size < z.nnz):
            self._mm_rows = wp.empty(shape=(z.nnz,), dtype=int, device=self.device)

    def _allocate_padded_topology(self, device, x_nnz: int):
        if self.device != device:
            self._reset(device)

        if self._mm_y_cursors is None or self._mm_y_cursors.size < x_nnz:
            self._mm_y_cursors = wp.empty(shape=(x_nnz,), dtype=int, device=self.device)

    def _allocate_stage_2(self, mm_nnz: int):
        # Allocations that depend on unmerged nnz estimate
        self._mm_nnz = mm_nnz
        if self._mm_rows is None or self._mm_rows.size < mm_nnz:
            self._mm_rows = wp.empty(shape=(mm_nnz,), dtype=int, device=self.device)
        if self._mm_cols is None or self._mm_cols.size < mm_nnz:
            self._mm_cols = wp.empty(shape=(mm_nnz,), dtype=int, device=self.device)
        if self._mm_src_blocks is None or self._mm_src_blocks.size < mm_nnz:
            self._mm_src_blocks = wp.empty(shape=(mm_nnz,), dtype=int, device=self.device)


def _bsr_mm_add_scaled_existing_values(
    z: BsrMatrix,
    beta,
    block_count: int,
    rows: wp.array,
    columns: wp.array,
    values: wp.array,
) -> None:
    if block_count == 0:
        return

    wp.launch(
        kernel=_bsr_axpy_add_block,
        device=z.device,
        dim=(block_count, z.block_shape[0], z.block_shape[1]),
        inputs=[
            0,
            beta,
            rows,
            columns,
            z.offsets,
            z.row_counts,
            z.columns,
            _as_3d_array(values, z.block_shape),
            z.scalar_values,
        ],
    )


def _bsr_mm_compute_values_dispatch(
    x: BsrMatrix,
    y: BsrMatrix,
    z: BsrMatrix,
    alpha,
    tile_size: int,
    x_offsets: wp.array,
    x_row_counts: wp.array,
    x_columns: wp.array,
    x_values: wp.array,
    y_offsets: wp.array,
    y_row_counts: wp.array,
    y_columns: wp.array,
    y_values: wp.array,
    mm_row_min,
    summed_triplet_offsets,
    summed_triplet_src_blocks,
) -> None:
    device = z.device
    max_subblock_dim = 12
    if tile_size > 0:
        use_tiles = True
    elif tile_size < 0:
        use_tiles = False
    else:
        # Heuristic for using tiled variant: few or very large blocks
        tile_size = 64
        max_tiles_per_sm = 2048 // tile_size  # assume 64 resident warps per SM
        use_tiles = device.is_cuda and (
            max(x.block_size, y.block_size, z.block_size) > max_subblock_dim**2
            or z.nnz < max_tiles_per_sm * device.sm_count
        )

    if use_tiles:
        subblock_rows = min(max_subblock_dim, z.block_shape[0])
        subblock_cols = min(max_subblock_dim, z.block_shape[1])

        wp.launch(
            kernel=make_bsr_mm_compute_values_tiled_outer(
                subblock_rows, subblock_cols, x.block_shape[1], z.scalar_type, tile_size
            ),
            device=device,
            dim=(
                z.nnz,
                (z.block_shape[0] + subblock_rows - 1) // subblock_rows,
                (z.block_shape[1] + subblock_cols - 1) // subblock_cols,
                tile_size,
            ),
            block_dim=tile_size,
            inputs=[
                alpha,
                x_offsets,
                x_row_counts,
                x_columns,
                _as_3d_array(x_values, x.block_shape),
                y_offsets,
                y_row_counts,
                y_columns,
                _as_3d_array(y_values, y.block_shape),
                mm_row_min,
                summed_triplet_offsets,
                summed_triplet_src_blocks,
                z.nrow,
                z.offsets,
                z.row_counts,
                z.columns,
                z.scalar_values,
            ],
        )
        return

    if (type_is_matrix(x.values.dtype) or type_is_matrix(y.values.dtype)) and not (type_is_matrix(z.values.dtype)):
        # Result block type is scalar, but operands are matrices
        # Cast result to (1x1) matrix to perform multiplication
        mm_values = z.values.view(wp.types.matrix(shape=(1, 1), dtype=z.scalar_type))
    else:
        mm_values = z.values

    wp.launch(
        kernel=_bsr_mm_compute_values,
        device=device,
        dim=z.nnz,
        inputs=[
            alpha,
            x_offsets,
            x_row_counts,
            x_columns,
            x_values,
            y_offsets,
            y_row_counts,
            y_columns,
            y_values,
            mm_row_min,
            summed_triplet_offsets,
            summed_triplet_src_blocks,
            z.nrow,
            z.offsets,
            z.row_counts,
            z.columns,
            mm_values,
        ],
    )


def bsr_mm(
    x: BsrMatrixOrExpression[BlockType[Rows, Any, Scalar]],
    y: BsrMatrixOrExpression[BlockType[Any, Cols, Scalar]],
    z: BsrMatrix[BlockType[Rows, Cols, Scalar]] | None = None,
    alpha: Scalar = 1.0,
    beta: Scalar = 0.0,
    masked: bool = False,
    work_arrays: bsr_mm_work_arrays | None = None,
    reuse_topology: bool = False,
    tile_size: int = 0,
    max_new_nnz: int | None = None,
    topology: Literal["compact", "masked", "padded"] | None = None,
) -> BsrMatrix[BlockType[Rows, Cols, Scalar]]:
    """Perform the sparse matrix-matrix multiplication ``z := alpha * x @ y + beta * z`` on BSR matrices ``x``, ``y`` and ``z``, and return ``z``.

    The ``x``, ``y`` and ``z`` matrices are allowed to alias.
    If the matrix ``z`` is not provided as input, it will be allocated and treated as zero.

    This method can be graph-captured if either:
     - ``topology="masked"``
     - ``reuse_topology=True``
     - ``max_new_nnz`` is provided
     - ``topology="padded"`` is used with supplied ``work_arrays``

    Args:
        x: Read-only left operand of the matrix-matrix product.
        y: Read-only right operand of the matrix-matrix product.
        z: Mutable affine operand and result matrix. If ``z`` is not provided, it will be allocated and treated as zero.
        alpha: Uniform scaling factor for the ``x @ y`` product
        beta: Uniform scaling factor for ``z``
        masked: Deprecated. Use ``topology="masked"`` instead. If ``True``,
          keep the non-zero topology of ``z`` unchanged.
        work_arrays: In most cases, this function will require the use of temporary storage.
          This storage can be reused across calls by passing an instance of
          :class:`bsr_mm_work_arrays` in ``work_arrays``.
        reuse_topology: If ``True``, reuse the compact product topology
          information stored in ``work_arrays`` rather than recompute it from
          scratch. Only supported with ``topology="compact"``. The matrices
          ``x``, ``y`` and ``z`` must be structurally similar to the previous
          call in which ``work_arrays`` were populated.
        max_new_nnz: If provided, the maximum number of non-zeros for the matrix-matrix product result
           (not counting the existing non-zeros in ``z``).
        tile_size: If a positive integer, use tiles of this size to compute the matrix-matrix product.
          If negative, disable tile-based computation. Defaults to ``0``, which determines whether to
          use tiles using using an heuristic based on the matrix shape and number of non-zeros..
        topology: Optional topology policy. ``"compact"`` keeps the existing
          compact rebuild behavior, ``"masked"`` is equivalent to
          the deprecated ``masked=True``, and ``"padded"`` writes the result topology into
          existing destination row capacity and records capacity overflow in
          ``z.status_sync()``.
    """

    x, x_scale = _extract_matrix_and_scale(x)
    alpha *= x_scale
    y, y_scale = _extract_matrix_and_scale(y)
    alpha *= y_scale

    if topology is None:
        topology = "masked" if masked else "compact"
    elif topology not in ("compact", "masked", "padded"):
        raise ValueError(f"Unsupported topology policy: {topology}")
    elif masked and topology != "masked":
        raise ValueError("Cannot pass masked=True with a non-masked topology policy")

    if masked:
        _warn_masked_arg_deprecated("bsr_mm")

    if topology == "masked":
        masked = True
    elif topology != "compact" and reuse_topology:
        raise ValueError("reuse_topology is only supported with topology='compact'")

    if z is None:
        if masked or topology == "padded":
            raise ValueError("Left-hand-side 'z' matrix must be provided for this topology policy")

        # If not output matrix is provided, allocate it for convenience
        z_block_shape = (x.block_shape[0], y.block_shape[1])
        if z_block_shape == (1, 1):
            z_block_type = x.scalar_type
        else:
            z_block_type = wp.types.matrix(shape=z_block_shape, dtype=x.scalar_type)
        z = bsr_zeros(x.nrow, y.ncol, block_type=z_block_type, device=x.values.device)
        z.values.requires_grad = x.requires_grad or y.requires_grad
        beta = 0.0

    if x.values.device != y.values.device or x.values.device != z.values.device:
        raise ValueError(
            f"All arguments must reside on the same device, got {x.values.device}, {y.values.device} and {z.values.device}"
        )

    if x.scalar_type != y.scalar_type or x.scalar_type != z.scalar_type:
        raise ValueError(
            f"Matrices must have the same scalar type, got {x.scalar_type}, {y.scalar_type} and {z.scalar_type}"
        )

    if (
        x.block_shape[0] != z.block_shape[0]
        or y.block_shape[1] != z.block_shape[1]
        or x.block_shape[1] != y.block_shape[0]
    ):
        raise ValueError(
            f"Incompatible block sizes for matrix multiplication, got ({x.block_shape}, {y.block_shape}) and ({z.block_shape})"
        )

    if x.nrow != z.nrow or z.ncol != y.ncol or x.ncol != y.nrow:
        raise ValueError(
            f"Incompatible number of rows/columns for matrix multiplication, got ({x.nrow}, {x.ncol}) and ({y.nrow}, {y.ncol})"
        )

    device = z.values.device

    if topology == "padded":
        status = z._ensure_status()

        if alpha == 0.0 or x.nnz == 0 or y.nnz == 0:
            return bsr_scale(z, beta)

        if not isinstance(alpha, z.scalar_type):
            alpha = z.scalar_type(alpha)
        if not isinstance(beta, z.scalar_type):
            beta = z.scalar_type(beta)

        beta_nonzero = beta != z.scalar_type(0.0)
        x_aliasing = z == x
        y_aliasing = z == y
        z_aliasing = x_aliasing or y_aliasing

        _bsr_ensure_independent_row_counts(z)

        if work_arrays is None:
            work_arrays = bsr_mm_work_arrays()

        snapshot_z = beta_nonzero or z_aliasing

        if snapshot_z:
            work_arrays._allocate_snapshot(device, z, rows=beta_nonzero)
            if beta_nonzero:
                z.uncompress_rows(out=work_arrays._mm_rows)
            wp.copy(dest=work_arrays._old_z_row_counts, src=z.row_counts, count=z.nrow)
            wp.copy(dest=work_arrays._old_z_columns, src=z.columns, count=z.nnz)
            wp.copy(dest=work_arrays._old_z_values, src=z.values, count=z.nnz)
            old_z_row_counts = work_arrays._old_z_row_counts
            old_z_columns = work_arrays._old_z_columns
            old_z_values = work_arrays._old_z_values
        else:
            old_z_row_counts = z.row_counts
            old_z_columns = z.columns
            old_z_values = z.values

        x_row_counts = old_z_row_counts if x_aliasing else x.row_counts
        x_columns = old_z_columns if x_aliasing else x.columns
        x_values = old_z_values if x_aliasing else x.values

        y_row_counts = old_z_row_counts if y_aliasing else y.row_counts
        y_columns = old_z_columns if y_aliasing else y.columns
        y_values = old_z_values if y_aliasing else y.values

        work_arrays._allocate_padded_topology(device, x.nnz)

        wp.launch(
            _bsr_mm_padded_topology,
            dim=z.nrow,
            device=device,
            inputs=[
                beta_nonzero,
                z.ncol,
                x.offsets,
                x_row_counts,
                x_columns,
                y.offsets,
                y_row_counts,
                y_columns,
                z.offsets,
                old_z_row_counts,
                old_z_columns,
                z.row_counts,
                z.columns,
                work_arrays._mm_y_cursors,
                status,
            ],
        )

        z.values.zero_()
        if beta_nonzero:
            _bsr_mm_add_scaled_existing_values(
                z=z,
                beta=beta,
                block_count=z.nnz,
                rows=work_arrays._mm_rows,
                columns=old_z_columns,
                values=old_z_values,
            )

        _bsr_mm_compute_values_dispatch(
            x=x,
            y=y,
            z=z,
            alpha=alpha,
            tile_size=tile_size,
            x_offsets=x.offsets,
            x_row_counts=x_row_counts,
            x_columns=x_columns,
            x_values=x_values,
            y_offsets=y.offsets,
            y_row_counts=y_row_counts,
            y_columns=y_columns,
            y_values=y_values,
            mm_row_min=None,
            summed_triplet_offsets=None,
            summed_triplet_src_blocks=None,
        )
        return z

    if alpha == 0.0 or x.nnz == 0 or y.nnz == 0:
        # Easy case
        return bsr_scale(z, beta)

    z_aliasing = z == x or z == y

    if masked:
        # no need to copy z, scale in-place
        copied_z_nnz = 0
        mm_nnz = z.nnz

        if z_aliasing:
            raise ValueError("topology='masked' is not supported for aliased inputs")

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
        if work_arrays is None:
            work_arrays = bsr_mm_work_arrays()

        if max_new_nnz is None:
            if device.is_capturing:
                raise RuntimeError(
                    "`bsr_mm` requires either `reuse_topology=True`, `topology='masked'` or `max_new_nnz` to be set for use in graph capture"
                )
            z.nnz_sync()

        work_arrays._allocate_stage_1(device, x.nnz, z, beta, z_aliasing)
        copied_z_nnz = work_arrays._copied_z_nnz

        # Prefix sum of number of (unmerged) mm blocks per row
        # Use either a thread or a block per row depending on avg nnz/row
        work_arrays._mm_block_counts.zero_()
        count_tile_size = 32
        if not device.is_cuda or x.nnz < 3 * count_tile_size * x.nrow:
            count_tile_size = 1

        wp.launch(
            kernel=make_bsr_mm_count_coeffs(count_tile_size),
            device=device,
            dim=(z.nrow, count_tile_size),
            block_dim=count_tile_size if count_tile_size > 1 else 256,
            inputs=[
                y.ncol,
                copied_z_nnz,
                x.offsets,
                x.row_counts,
                x.columns,
                y.offsets,
                y.row_counts,
                y.columns,
                work_arrays._mm_row_min,
                work_arrays._mm_block_counts,
            ],
        )
        warp._src.utils.array_scan(work_arrays._mm_block_counts[: x.nnz + 1], work_arrays._mm_block_counts[: x.nnz + 1])

        if max_new_nnz is not None:
            mm_nnz = max_new_nnz + copied_z_nnz
        else:
            # Get back total counts on host -- we need a synchronization here
            # Use pinned buffer from z, we are going to need it later anyway
            nnz_buf, _ = z._setup_nnz_transfer()
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
            work_arrays._mm_src_blocks[:copied_z_nnz].fill_(-1)
            if z_aliasing:
                # If z is aliasing with x or y, need to save topology as well
                wp.copy(src=z.columns, dest=work_arrays._old_z_columns, count=copied_z_nnz)
                wp.copy(src=z.offsets, dest=work_arrays._old_z_offsets, count=z.nrow + 1)
                if z.row_counts is None:
                    wp.launch(
                        _bsr_fill_row_counts_from_offsets,
                        dim=z.nrow,
                        device=z.device,
                        inputs=[z.nrow, z.offsets, work_arrays._old_z_row_counts],
                    )
                else:
                    wp.copy(src=z.row_counts, dest=work_arrays._old_z_row_counts, count=z.nrow)

        # Fill unmerged mm blocks rows and columns
        wp.launch(
            kernel=_bsr_mm_list_coeffs,
            device=device,
            dim=mm_nnz - copied_z_nnz,
            inputs=[
                copied_z_nnz,
                mm_nnz,
                x.nrow,
                x.offsets,
                x.row_counts,
                x.columns,
                y.offsets,
                y.row_counts,
                y.columns,
                work_arrays._mm_row_min,
                work_arrays._mm_block_counts,
                work_arrays._mm_rows,
                work_arrays._mm_cols,
                work_arrays._mm_src_blocks,
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

        nnz_buf, nnz_event = z._setup_nnz_transfer()
        summed_triplet_offsets = wp.empty(shape=(mm_nnz,), dtype=wp.int32, device=device)
        summed_triplet_indices = wp.empty(shape=(mm_nnz,), dtype=wp.int32, device=device)

        _bsr_set_from_triplets_native(
            dest=z,
            rows=work_arrays._mm_rows,
            columns=work_arrays._mm_cols,
            values=None,
            count=None,
            nnz=mm_nnz,
            zero_value_mask=0,
            masked=False,
            summed_triplet_offsets=summed_triplet_offsets,
            summed_triplet_indices=summed_triplet_indices,
            dest_offsets=z.offsets,
            dest_row_counts=None,
            dest_columns=z.columns,
            nnz_buf=nnz_buf,
            nnz_event=nnz_event,
        )
        _bsr_set_compact_row_counts(z)

        # Resize z to fit mm result if necessary
        # If we are not reusing the product topology, this needs another synchronization
        if not reuse_topology:
            work_arrays.result_nnz = z.nnz_sync() if max_new_nnz is None else mm_nnz

        _bsr_ensure_fits(z, nnz=work_arrays.result_nnz)
        z.values.zero_()

        if copied_z_nnz > 0:
            # Add back original z values
            _bsr_mm_add_scaled_existing_values(
                z=z,
                beta=beta,
                block_count=copied_z_nnz,
                rows=work_arrays._mm_rows,
                columns=work_arrays._mm_cols,
                values=work_arrays._old_z_values,
            )

    if masked:
        mm_row_min = None
        summed_triplet_offsets_arg = None
        summed_triplet_src_blocks = None
    else:
        mm_row_min = work_arrays._mm_row_min
        summed_triplet_offsets_arg = summed_triplet_offsets
        summed_triplet_src_blocks = work_arrays._mm_src_blocks[summed_triplet_indices]

    _bsr_mm_compute_values_dispatch(
        x=x,
        y=y,
        z=z,
        alpha=alpha,
        tile_size=tile_size,
        x_offsets=work_arrays._old_z_offsets if x == z else x.offsets,
        x_row_counts=work_arrays._old_z_row_counts if x == z else x.row_counts,
        x_columns=work_arrays._old_z_columns if x == z else x.columns,
        x_values=work_arrays._old_z_values if x == z else x.values,
        y_offsets=work_arrays._old_z_offsets if y == z else y.offsets,
        y_row_counts=work_arrays._old_z_row_counts if y == z else y.row_counts,
        y_columns=work_arrays._old_z_columns if y == z else y.columns,
        y_values=work_arrays._old_z_values if y == z else y.values,
        mm_row_min=mm_row_min,
        summed_triplet_offsets=summed_triplet_offsets_arg,
        summed_triplet_src_blocks=summed_triplet_src_blocks,
    )

    return z


@cache
def make_bsr_mv_kernel(block_cols: int):

    @wp.kernel(enable_backward=False, module="unique")
    def bsr_mv_kernel(
        alpha: Any,
        A_offsets: wp.array(dtype=int),
        A_row_counts: wp.array(dtype=int),
        A_columns: wp.array(dtype=int),
        A_values: wp.array3d(dtype=Any),
        x: wp.array(dtype=Any),
        beta: Any,
        y: wp.array(dtype=Any),
    ):
        row, subrow = wp.tid()

        block_rows = A_values.shape[1]

        yi = row * block_rows + subrow

        # zero-initialize with type of y elements
        scalar_zero = type(alpha)(0)
        v = scalar_zero

        if alpha != scalar_zero:
            beg = A_offsets[row]
            end = _bsr_row_end(A_offsets, A_row_counts, row)
            for block in range(beg, end):
                xs = A_columns[block] * block_cols
                for col in range(wp.static(block_cols)):
                    v += A_values[block, subrow, col] * x[xs + col]
            v *= alpha

        if beta != scalar_zero:
            v += beta * y[yi]

        y[yi] = v

    return bsr_mv_kernel


@cache
def make_bsr_mv_tiled_kernel(tile_size: int):

    @wp.kernel(enable_backward=False, module="unique")
    def bsr_mv_tiled_kernel(
        alpha: Any,
        A_offsets: wp.array(dtype=int),
        A_row_counts: wp.array(dtype=int),
        A_columns: wp.array(dtype=int),
        A_values: wp.array3d(dtype=Any),
        x: wp.array(dtype=Any),
        beta: Any,
        y: wp.array(dtype=Any),
    ):
        row, subrow, lane = wp.tid()

        scalar_zero = type(alpha)(0)
        block_rows = A_values.shape[1]
        block_cols = A_values.shape[2]

        yi = row * block_rows + subrow

        if beta == scalar_zero:
            subrow_sum = wp.tile_zeros(shape=(1,), dtype=y.dtype)
        else:
            subrow_sum = beta * wp.tile_load(y, 1, yi)

        if alpha != scalar_zero:
            block_beg = A_offsets[row]
            col_count = _bsr_row_count(A_offsets, A_row_counts, row) * block_cols

            col = lane
            lane_sum = y.dtype(0)

            for col in range(lane, col_count, tile_size):
                block = col // block_cols
                block_col = col - block * block_cols
                block += block_beg

                xi = x[A_columns[block] * block_cols + block_col]
                lane_sum += A_values[block, subrow, block_col] * xi

            lane_sum *= alpha
            subrow_sum += wp.tile_sum(wp.tile(lane_sum))

        wp.tile_store(y, subrow_sum, yi)

    return bsr_mv_tiled_kernel


@cache
def make_bsr_mv_transpose_kernel(block_rows: int):

    @wp.kernel(enable_backward=False, module="unique")
    def bsr_mv_transpose_kernel(
        alpha: Any,
        A_row_count: int,
        A_offsets: wp.array(dtype=int),
        A_row_counts: wp.array(dtype=int),
        A_columns: wp.array(dtype=int),
        A_values: wp.array3d(dtype=Any),
        x: wp.array(dtype=Any),
        y: wp.array(dtype=Any),
    ):
        block, subcol = wp.tid()

        row = bsr_row_index(A_offsets, A_row_count, block, A_row_counts)
        if row == -1:
            return

        block_cols = A_values.shape[2]

        A_block = A_values[block]

        col_sum = type(alpha)(0)
        for subrow in range(wp.static(block_rows)):
            col_sum += A_block[subrow, subcol] * x[row * block_rows + subrow]

        wp.atomic_add(y, A_columns[block] * block_cols + subcol, alpha * col_sum)

    return bsr_mv_transpose_kernel


def _vec_array_view(array: wp.array, dtype: type, expected_scalar_count: int) -> wp.array:
    # cast a 1d or 2d array to a 1d array with the target dtype, adjusting shape as required

    scalar_count = array.size * type_size(array.dtype)
    if scalar_count != expected_scalar_count:
        raise ValueError(f"Invalid array scalar size, expected {expected_scalar_count}, got {scalar_count}")

    if array.ndim == 1 and types_equal(array.dtype, dtype):
        return array

    if type_scalar_type(array.dtype) != type_scalar_type(dtype):
        raise ValueError(f"Incompatible scalar types, expected {type_repr(array.dtype)}, got {type_repr(dtype)}")

    if array.ndim > 2:
        raise ValueError(f"Incompatible array number of dimensions, expected 1 or 2, got {array.ndim}")

    if not array.is_contiguous:
        raise ValueError("Array must be contiguous")

    vec_length = type_size(dtype)
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
    x: Array[Vector[Scalar, Cols] | Scalar],
    y: Array[Vector[Scalar, Rows] | Scalar] | None = None,
    alpha: Scalar = 1.0,
    beta: Scalar = 0.0,
    transpose: bool = False,
    work_buffer: Array[Vector[Scalar, Rows] | Scalar] | None = None,
    tile_size: int = 0,
) -> Array[Vector[Scalar, Rows] | Scalar]:
    """Perform the sparse matrix-vector product ``y := alpha * A * x + beta * y`` and return ``y``.

    The ``x`` and ``y`` vectors are allowed to alias.

    Args:
        A: Read-only, left matrix operand of the matrix-vector product.
        x: Read-only, right vector operand of the matrix-vector product.
        y: Mutable affine operand and result vector. If ``y`` is not provided, it will be allocated and treated as zero.
        alpha: Uniform scaling factor for ``x``. If zero, ``x`` will not be read and may be left uninitialized.
        beta: Uniform scaling factor for ``y``. If zero, ``y`` will not be read and may be left uninitialized.
        transpose: If ``True``, use the transpose of the matrix ``A``. In this case the result is **non-deterministic**.
        work_buffer: Temporary storage is required if and only if ``x`` and ``y`` are the same vector.
          If provided, the ``work_buffer`` array will be used for this purpose,
          otherwise a temporary allocation will be performed.
        tile_size: If a positive integer, use tiles of this size to compute the matrix-matrix product.
          If negative, disable tile-based computation. Defaults to ``0``, which determines whether to
          use tiles using using an heuristic based on the matrix shape and number of non-zeros..
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
        y_dtype = A.scalar_type if y_vec_len == 1 else wp.types.vector(length=y_vec_len, dtype=A.scalar_type)
        y = wp.empty(shape=(nrow,), device=A.values.device, dtype=y_dtype, requires_grad=x.requires_grad)
        beta = 0.0

    alpha = A.scalar_type(alpha)
    beta = A.scalar_type(beta)

    device = A.values.device
    if A.values.device != x.device or A.values.device != y.device:
        raise ValueError(
            f"A, x, and y must reside on the same device, got {A.values.device}, {x.device} and {y.device}"
        )

    if x.ptr == y.ptr:
        # Aliasing case, need temporary storage
        if work_buffer is None:
            work_buffer = wp.empty_like(y)
        elif work_buffer.size < y.size:
            raise ValueError(f"Work buffer size is insufficient, needs to be at least {y.size}, got {work_buffer.size}")
        elif not types_equal(work_buffer.dtype, y.dtype):
            raise ValueError(
                f"Work buffer must have same data type as y, {type_repr(y.dtype)} vs {type_repr(work_buffer.dtype)}"
            )

        # Save old y values before overwriting vector
        wp.copy(dest=work_buffer, src=y, count=y.size)
        x = work_buffer

    try:
        x_view = _vec_array_view(x, A.scalar_type, expected_scalar_count=ncol * block_shape[1])
    except ValueError as err:
        raise ValueError("Incompatible 'x' vector for bsr_mv") from err
    try:
        y_view = _vec_array_view(y, A.scalar_type, expected_scalar_count=nrow * block_shape[0])
    except ValueError as err:
        raise ValueError("Incompatible 'y' vector for bsr_mv") from err

    # heuristic to use tiled version for long rows
    if tile_size > 0:
        use_tiles = True
    elif tile_size < 0:
        use_tiles = False
    else:
        tile_size = 64
        use_tiles = device.is_cuda and A.nnz * A.block_size > 2 * tile_size * A.shape[0]

    if transpose:
        if beta.value == 0.0:
            y.zero_()
        elif beta.value != 1.0:
            wp.launch(
                kernel=_bsr_scale_1d_kernel,
                device=y.device,
                dim=y_view.shape[0],
                inputs=[beta, y_view],
            )
        if alpha.value != 0.0:
            wp.launch(
                kernel=make_bsr_mv_transpose_kernel(block_rows=block_shape[1]),
                device=A.values.device,
                dim=(A.nnz, block_shape[0]),
                inputs=[alpha, A.nrow, A.offsets, A.row_counts, A.columns, A.scalar_values, x_view, y_view],
            )
    elif use_tiles:
        wp.launch(
            kernel=make_bsr_mv_tiled_kernel(tile_size),
            device=A.values.device,
            dim=(nrow, block_shape[0], tile_size),
            block_dim=tile_size,
            inputs=[alpha, A.offsets, A.row_counts, A.columns, A.scalar_values, x_view, beta, y_view],
        )
    else:
        wp.launch(
            kernel=make_bsr_mv_kernel(block_cols=block_shape[1]),
            device=A.values.device,
            dim=(nrow, block_shape[0]),
            inputs=[alpha, A.offsets, A.row_counts, A.columns, A.scalar_values, x_view, beta, y_view],
        )

    return y
