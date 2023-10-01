from typing import Any, Tuple

import warp as wp
from warp.utils import radix_sort_pairs, runlength_encode, array_scan
from warp.fem.cache import borrow_temporary, borrow_temporary_like, TemporaryStore, Temporary


@wp.func
def generalized_outer(x: Any, y: Any):
    """Generalized outer product allowing for the first argument to be a scalar"""
    return wp.outer(x, y)


@wp.func
def generalized_outer(x: wp.float32, y: wp.vec2):
    return x * y


@wp.func
def generalized_outer(x: wp.float32, y: wp.vec3):
    return x * y


@wp.func
def generalized_inner(x: Any, y: Any):
    """Generalized inner product allowing for the first argument to be a tensor"""
    return wp.dot(x, y)


@wp.func
def generalized_inner(x: wp.mat22, y: wp.vec2):
    return x[0] * y[0] + x[1] * y[1]


@wp.func
def generalized_inner(x: wp.mat33, y: wp.vec3):
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]


@wp.func
def unit_element(template_type: Any, coord: int):
    """Returns a instance of `template_type` with a single coordinate set to 1 in the canonical basis"""

    t = type(template_type)(0.0)
    t[coord] = 1.0
    return t


@wp.func
def unit_element(template_type: wp.float32, coord: int):
    return 1.0


@wp.func
def unit_element(template_type: wp.mat22, coord: int):
    t = wp.mat22(0.0)
    row = coord // 2
    col = coord - 2 * row
    t[row, col] = 1.0
    return t


@wp.func
def unit_element(template_type: wp.mat33, coord: int):
    t = wp.mat33(0.0)
    row = coord // 3
    col = coord - 3 * row
    t[row, col] = 1.0
    return t


@wp.func
def symmetric_part(x: Any):
    """Symmetric part of a square tensor"""
    return 0.5 * (x + wp.transpose(x))


@wp.func
def skew_part(x: wp.mat22):
    """Skew part of a 2x2 tensor as corresponding rotation angle"""
    return 0.5 * (x[1, 0] - x[0, 1])


@wp.func
def skew_part(x: wp.mat33):
    """Skew part of a 3x3 tensor as the corresponding rotation vector"""
    a = 0.5 * (x[2, 1] - x[1, 2])
    b = 0.5 * (x[0, 2] - x[2, 0])
    c = 0.5 * (x[1, 0] - x[0, 1])
    return wp.vec3(a, b, c)


def compress_node_indices(
    node_count: int, node_indices: wp.array(dtype=int), temporary_store: TemporaryStore = None
) -> Tuple[Temporary, Temporary, int, Temporary]:
    """
    Compress an unsorted list of node indices into:
     - a node_offsets array, giving for each node the start offset of corresponding indices in sorted_array_indices
     - a sorted_array_indices array, listing the indices in the input array corresponding to each node
     - the number of unique node indices
     - a unique_node_indices array containg the sorted list of unique node indices (i.e. the list of indices i for which node_offsets[i] < node_offsets[i+1])
    """

    index_count = node_indices.size

    sorted_node_indices_temp = borrow_temporary(
        temporary_store, shape=2 * index_count, dtype=int, device=node_indices.device
    )
    sorted_array_indices_temp = borrow_temporary_like(sorted_node_indices_temp, temporary_store)

    sorted_node_indices = sorted_node_indices_temp.array
    sorted_array_indices = sorted_array_indices_temp.array

    wp.copy(dest=sorted_node_indices, src=node_indices, count=index_count)

    indices_per_element = 1 if node_indices.ndim == 1 else node_indices.shape[-1]
    wp.launch(
        kernel=_iota_kernel,
        dim=index_count,
        inputs=[sorted_array_indices, indices_per_element],
        device=sorted_array_indices.device,
    )

    # Sort indices
    radix_sort_pairs(sorted_node_indices, sorted_array_indices, count=index_count)

    # Build prefix sum of number of elements per node
    unique_node_indices_temp = borrow_temporary(
        temporary_store, shape=index_count, dtype=int, device=node_indices.device
    )
    node_element_counts_temp = borrow_temporary(
        temporary_store, shape=index_count, dtype=int, device=node_indices.device
    )

    unique_node_indices = unique_node_indices_temp.array
    node_element_counts = node_element_counts_temp.array

    unique_node_count_dev = borrow_temporary(temporary_store, shape=(1,), dtype=int, device=sorted_node_indices.device)
    runlength_encode(
        sorted_node_indices,
        unique_node_indices,
        node_element_counts,
        value_count=index_count,
        run_count=unique_node_count_dev.array,
    )

    # Transfer unique node count to host
    if node_indices.device.is_cuda:
        unique_node_count_host = borrow_temporary(temporary_store, shape=(1,), dtype=int, pinned=True, device="cpu")
        wp.copy(src=unique_node_count_dev.array, dest=unique_node_count_host.array, count=1)
        wp.synchronize_stream(wp.get_stream())
        unique_node_count_dev.release()
        unique_node_count = int(unique_node_count_host.array.numpy()[0])
        unique_node_count_host.release()
    else:
        unique_node_count = int(unique_node_count_dev.array.numpy()[0])
        unique_node_count_dev.release()

    # Scatter seen run counts to global array of element count per node
    node_offsets_temp = borrow_temporary(
        temporary_store, shape=(node_count + 1), device=node_element_counts.device, dtype=int
    )
    node_offsets = node_offsets_temp.array

    node_offsets.zero_()
    wp.launch(
        kernel=_scatter_node_counts,
        dim=unique_node_count,
        inputs=[node_element_counts, unique_node_indices, node_offsets],
        device=node_offsets.device,
    )

    # Prefix sum of number of elements per node
    array_scan(node_offsets, node_offsets, inclusive=True)

    sorted_node_indices_temp.release()
    node_element_counts_temp.release()

    return node_offsets_temp, sorted_array_indices_temp, unique_node_count, unique_node_indices_temp


def masked_indices(
    mask: wp.array, missing_index=-1, temporary_store: TemporaryStore = None
) -> Tuple[Temporary, Temporary]:
    """
    From an array of boolean masks (must be either 0 or 1), returns:
      - The list of indices for which the mask is 1
      - A map associating to each element of the input mask array its local index if non-zero, or missing_index if zero.
    """

    offsets_temp = borrow_temporary_like(mask, temporary_store)
    offsets = offsets_temp.array

    wp.utils.array_scan(mask, offsets, inclusive=True)

    # Get back total counts on host
    if offsets.device.is_cuda:
        masked_count_temp = borrow_temporary(temporary_store, shape=1, dtype=int, pinned=True, device="cpu")
        wp.copy(dest=masked_count_temp.array, src=offsets, src_offset=offsets.shape[0] - 1, count=1)
        wp.synchronize_stream(wp.get_stream())
        masked_count = int(masked_count_temp.array.numpy()[0])
        masked_count_temp.release()
    else:
        masked_count = int(offsets.numpy()[-1])

    # Convert counts to indices
    indices_temp = borrow_temporary(temporary_store, shape=masked_count, device=mask.device, dtype=int)

    wp.launch(
        kernel=_masked_indices_kernel,
        dim=offsets.shape,
        inputs=[missing_index, mask, offsets, indices_temp.array, offsets],
        device=mask.device,
    )

    return indices_temp, offsets_temp


def array_axpy(x: wp.array, y: wp.array, alpha: float = 1.0, beta: float = 1.0):
    """Performs y = alpha*x + beta*y"""

    dtype = wp.types.type_scalar_type(x.dtype)

    alpha = dtype(alpha)
    beta = dtype(beta)

    if not wp.types.types_equal(x.dtype, y.dtype) or x.shape != y.shape or x.device != y.device:
        raise ValueError("x and y arrays must have same dat atype, shape and device")

    wp.launch(kernel=_array_axpy_kernel, dim=x.shape, device=x.device, inputs=[x, y, alpha, beta])


@wp.kernel
def _iota_kernel(indices: wp.array(dtype=int), divisor: int):
    indices[wp.tid()] = wp.tid() // divisor


@wp.kernel
def _scatter_node_counts(
    unique_counts: wp.array(dtype=int), unique_node_indices: wp.array(dtype=int), node_counts: wp.array(dtype=int)
):
    i = wp.tid()
    node_counts[1 + unique_node_indices[i]] = unique_counts[i]


@wp.kernel
def _masked_indices_kernel(
    missing_index: int,
    mask: wp.array(dtype=int),
    offsets: wp.array(dtype=int),
    masked_to_global: wp.array(dtype=int),
    global_to_masked: wp.array(dtype=int),
):
    i = wp.tid()

    if mask[i] == 0:
        global_to_masked[i] = missing_index
    else:
        masked_idx = offsets[i] - 1
        global_to_masked[i] = masked_idx
        masked_to_global[masked_idx] = i


@wp.kernel
def _array_axpy_kernel(x: wp.array(dtype=Any), y: wp.array(dtype=Any), alpha: Any, beta: Any):
    i = wp.tid()
    y[i] = beta * y[i] + alpha * x[i]
