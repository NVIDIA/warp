# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import cProfile
import sys
import timeit
import warnings
from typing import Any

import numpy as np

import warp as wp
import warp.types


def warp_showwarning(message, category, filename, lineno, file=None, line=None):
    """Version of warnings.showwarning that always prints to sys.stdout."""
    msg = warnings.WarningMessage(message, category, filename, lineno, sys.stdout, line)
    warnings._showwarnmsg_impl(msg)


def warn(message, category=None, stacklevel=1):
    with warnings.catch_warnings():
        warnings.simplefilter("default")  # Change the filter in this process
        warnings.showwarning = warp_showwarning
        warnings.warn(message, category, stacklevel + 1)  # Increment stacklevel by 1 since we are in a wrapper


# expand a 7-vec to a tuple of arrays
def transform_expand(t):
    return wp.transform(np.array(t[0:3]), np.array(t[3:7]))


@wp.func
def quat_between_vectors(a: wp.vec3, b: wp.vec3) -> wp.quat:
    """
    Compute the quaternion that rotates vector a to vector b
    """
    a = wp.normalize(a)
    b = wp.normalize(b)
    c = wp.cross(a, b)
    d = wp.dot(a, b)
    q = wp.quat(c[0], c[1], c[2], 1.0 + d)
    return wp.normalize(q)


def array_scan(in_array, out_array, inclusive=True):
    if in_array.device != out_array.device:
        raise RuntimeError("Array storage devices do not match")

    if in_array.size != out_array.size:
        raise RuntimeError("Array storage sizes do not match")

    if in_array.dtype != out_array.dtype:
        raise RuntimeError("Array data types do not match")

    if in_array.size == 0:
        return

    from warp.context import runtime

    if in_array.device.is_cpu:
        if in_array.dtype == wp.int32:
            runtime.core.array_scan_int_host(in_array.ptr, out_array.ptr, in_array.size, inclusive)
        elif in_array.dtype == wp.float32:
            runtime.core.array_scan_float_host(in_array.ptr, out_array.ptr, in_array.size, inclusive)
        else:
            raise RuntimeError("Unsupported data type")
    elif in_array.device.is_cuda:
        if in_array.dtype == wp.int32:
            runtime.core.array_scan_int_device(in_array.ptr, out_array.ptr, in_array.size, inclusive)
        elif in_array.dtype == wp.float32:
            runtime.core.array_scan_float_device(in_array.ptr, out_array.ptr, in_array.size, inclusive)
        else:
            raise RuntimeError("Unsupported data type")


def radix_sort_pairs(keys, values, count: int):
    if keys.device != values.device:
        raise RuntimeError("Array storage devices do not match")

    if count == 0:
        return

    if keys.size < 2 * count or values.size < 2 * count:
        raise RuntimeError("Array storage must be large enough to contain 2*count elements")

    from warp.context import runtime

    if keys.device.is_cpu:
        if keys.dtype == wp.int32 and values.dtype == wp.int32:
            runtime.core.radix_sort_pairs_int_host(keys.ptr, values.ptr, count)
        else:
            raise RuntimeError("Unsupported data type")
    elif keys.device.is_cuda:
        if keys.dtype == wp.int32 and values.dtype == wp.int32:
            runtime.core.radix_sort_pairs_int_device(keys.ptr, values.ptr, count)
        else:
            raise RuntimeError("Unsupported data type")


def runlength_encode(values, run_values, run_lengths, run_count=None, value_count=None):
    if run_values.device != values.device or run_lengths.device != values.device:
        raise RuntimeError("Array storage devices do not match")

    if value_count is None:
        value_count = values.size

    if run_values.size < value_count or run_lengths.size < value_count:
        raise RuntimeError("Output array storage sizes must be at least equal to value_count")

    if values.dtype != run_values.dtype:
        raise RuntimeError("values and run_values data types do not match")

    if run_lengths.dtype != wp.int32:
        raise RuntimeError("run_lengths array must be of type int32")

    # User can provide a device output array for storing the number of runs
    # For convenience, if no such array is provided, number of runs is returned on host
    if run_count is None:
        if value_count == 0:
            return 0
        run_count = wp.empty(shape=(1,), dtype=int, device=values.device)
        host_return = True
    else:
        if run_count.device != values.device:
            raise RuntimeError("run_count storage device does not match other arrays")
        if run_count.dtype != wp.int32:
            raise RuntimeError("run_count array must be of type int32")
        if value_count == 0:
            run_count.zero_()
            return 0
        host_return = False

    from warp.context import runtime

    if values.device.is_cpu:
        if values.dtype == wp.int32:
            runtime.core.runlength_encode_int_host(
                values.ptr, run_values.ptr, run_lengths.ptr, run_count.ptr, value_count
            )
        else:
            raise RuntimeError("Unsupported data type")
    elif values.device.is_cuda:
        if values.dtype == wp.int32:
            runtime.core.runlength_encode_int_device(
                values.ptr, run_values.ptr, run_lengths.ptr, run_count.ptr, value_count
            )
        else:
            raise RuntimeError("Unsupported data type")

    if host_return:
        return int(run_count.numpy()[0])


def array_sum(values, out=None, value_count=None, axis=None):
    if value_count is None:
        if axis is None:
            value_count = values.size
        else:
            value_count = values.shape[axis]

    if axis is None:
        output_shape = (1,)
    else:

        def output_dim(ax, dim):
            return 1 if ax == axis else dim

        output_shape = tuple(output_dim(ax, dim) for ax, dim in enumerate(values.shape))

    type_length = wp.types.type_length(values.dtype)
    scalar_type = wp.types.type_scalar_type(values.dtype)

    # User can provide a device output array for storing the number of runs
    # For convenience, if no such array is provided, number of runs is returned on host
    if out is None:
        host_return = True
        out = wp.empty(shape=output_shape, dtype=values.dtype, device=values.device)
    else:
        host_return = False
        if out.device != values.device:
            raise RuntimeError("out storage device should match values array")
        if out.dtype != values.dtype:
            raise RuntimeError(f"out array should have type {values.dtype.__name__}")
        if out.shape != output_shape:
            raise RuntimeError(f"out array should have shape {output_shape}")

    if value_count == 0:
        out.zero_()
        if axis is None and host_return:
            return out.numpy()[0]
        return out

    from warp.context import runtime

    if values.device.is_cpu:
        if scalar_type == wp.float32:
            native_func = runtime.core.array_sum_float_host
        elif scalar_type == wp.float64:
            native_func = runtime.core.array_sum_double_host
        else:
            raise RuntimeError("Unsupported data type")
    elif values.device.is_cuda:
        if scalar_type == wp.float32:
            native_func = runtime.core.array_sum_float_device
        elif scalar_type == wp.float64:
            native_func = runtime.core.array_sum_double_device
        else:
            raise RuntimeError("Unsupported data type")

    if axis is None:
        stride = wp.types.type_size_in_bytes(values.dtype)
        native_func(values.ptr, out.ptr, value_count, stride, type_length)

        if host_return:
            return out.numpy()[0]
    else:
        stride = values.strides[axis]
        for idx in np.ndindex(output_shape):
            out_offset = sum(i * s for i, s in zip(idx, out.strides))
            val_offset = sum(i * s for i, s in zip(idx, values.strides))

            native_func(
                values.ptr + val_offset,
                out.ptr + out_offset,
                value_count,
                stride,
                type_length,
            )

        if host_return:
            return out


def array_inner(a, b, out=None, count=None, axis=None):
    if a.size != b.size:
        raise RuntimeError("Array storage sizes do not match")

    if a.device != b.device:
        raise RuntimeError("Array storage devices do not match")

    if a.dtype != b.dtype:
        raise RuntimeError("Array data types do not match")

    if count is None:
        if axis is None:
            count = a.size
        else:
            count = a.shape[axis]

    if axis is None:
        output_shape = (1,)
    else:

        def output_dim(ax, dim):
            return 1 if ax == axis else dim

        output_shape = tuple(output_dim(ax, dim) for ax, dim in enumerate(a.shape))

    type_length = wp.types.type_length(a.dtype)
    scalar_type = wp.types.type_scalar_type(a.dtype)

    # User can provide a device output array for storing the number of runs
    # For convenience, if no such array is provided, number of runs is returned on host
    if out is None:
        host_return = True
        out = wp.empty(shape=output_shape, dtype=scalar_type, device=a.device)
    else:
        host_return = False
        if out.device != a.device:
            raise RuntimeError("out storage device should match values array")
        if out.dtype != scalar_type:
            raise RuntimeError(f"out array should have type {scalar_type.__name__}")
        if out.shape != output_shape:
            raise RuntimeError(f"out array should have shape {output_shape}")

    if count == 0:
        if axis is None and host_return:
            return 0.0
        out.zero_()
        return out

    from warp.context import runtime

    if a.device.is_cpu:
        if scalar_type == wp.float32:
            native_func = runtime.core.array_inner_float_host
        elif scalar_type == wp.float64:
            native_func = runtime.core.array_inner_double_host
        else:
            raise RuntimeError("Unsupported data type")
    elif a.device.is_cuda:
        if scalar_type == wp.float32:
            native_func = runtime.core.array_inner_float_device
        elif scalar_type == wp.float64:
            native_func = runtime.core.array_inner_double_device
        else:
            raise RuntimeError("Unsupported data type")

    if axis is None:
        stride_a = wp.types.type_size_in_bytes(a.dtype)
        stride_b = wp.types.type_size_in_bytes(b.dtype)
        native_func(a.ptr, b.ptr, out.ptr, count, stride_a, stride_b, type_length)

        if host_return:
            return out.numpy()[0]
    else:
        stride_a = a.strides[axis]
        stride_b = b.strides[axis]

        for idx in np.ndindex(output_shape):
            out_offset = sum(i * s for i, s in zip(idx, out.strides))
            a_offset = sum(i * s for i, s in zip(idx, a.strides))
            b_offset = sum(i * s for i, s in zip(idx, b.strides))

            native_func(
                a.ptr + a_offset,
                b.ptr + b_offset,
                out.ptr + out_offset,
                count,
                stride_a,
                stride_b,
                type_length,
            )

        if host_return:
            return out


@wp.kernel
def _array_cast_kernel(
    dest: Any,
    src: Any,
):
    i = wp.tid()
    dest[i] = dest.dtype(src[i])


def array_cast(in_array, out_array, count=None):
    if in_array.device != out_array.device:
        raise RuntimeError("Array storage devices do not match")

    in_array_data_shape = getattr(in_array.dtype, "_shape_", ())
    out_array_data_shape = getattr(out_array.dtype, "_shape_", ())

    if in_array.ndim != out_array.ndim or in_array_data_shape != out_array_data_shape:
        # Number of dimensions or data type shape do not match.
        # Flatten arrays and do cast at the scalar level
        in_array = in_array.flatten()
        out_array = out_array.flatten()

        in_array_data_length = warp.types.type_length(in_array.dtype)
        out_array_data_length = warp.types.type_length(out_array.dtype)
        in_array_scalar_type = wp.types.type_scalar_type(in_array.dtype)
        out_array_scalar_type = wp.types.type_scalar_type(out_array.dtype)

        in_array = wp.array(
            data=None,
            ptr=in_array.ptr,
            capacity=in_array.capacity,
            owner=False,
            device=in_array.device,
            dtype=in_array_scalar_type,
            shape=in_array.shape[0] * in_array_data_length,
        )

        out_array = wp.array(
            data=None,
            ptr=out_array.ptr,
            capacity=out_array.capacity,
            owner=False,
            device=out_array.device,
            dtype=out_array_scalar_type,
            shape=out_array.shape[0] * out_array_data_length,
        )

        if count is not None:
            count *= in_array_data_length

    if count is None:
        count = in_array.size

    if in_array.ndim == 1:
        dim = count
    elif count < in_array.size:
        raise RuntimeError("Partial cast is not supported for arrays with more than one dimension")
    else:
        dim = in_array.shape

    if in_array.dtype == out_array.dtype:
        # Same data type, can simply copy
        wp.copy(dest=out_array, src=in_array, count=count)
    else:
        wp.launch(kernel=_array_cast_kernel, dim=dim, inputs=[out_array, in_array], device=out_array.device)


# code snippet for invoking cProfile
# cp = cProfile.Profile()
# cp.enable()
# for i in range(1000):
#     self.state = self.integrator.forward(self.model, self.state, self.sim_dt)

# cp.disable()
# cp.print_stats(sort='tottime')
# exit(0)


# helper kernels for initializing NVDB volumes from a dense array
@wp.kernel
def copy_dense_volume_to_nano_vdb_v(volume: wp.uint64, values: wp.array(dtype=wp.vec3, ndim=3)):
    i, j, k = wp.tid()
    wp.volume_store_v(volume, i, j, k, values[i, j, k])


@wp.kernel
def copy_dense_volume_to_nano_vdb_f(volume: wp.uint64, values: wp.array(dtype=wp.float32, ndim=3)):
    i, j, k = wp.tid()
    wp.volume_store_f(volume, i, j, k, values[i, j, k])


@wp.kernel
def copy_dense_volume_to_nano_vdb_i(volume: wp.uint64, values: wp.array(dtype=wp.int32, ndim=3)):
    i, j, k = wp.tid()
    wp.volume_store_i(volume, i, j, k, values[i, j, k])


# represent an edge between v0, v1 with connected faces f0, f1, and opposite vertex o0, and o1
# winding is such that first tri can be reconstructed as {v0, v1, o0}, and second tri as { v1, v0, o1 }
class MeshEdge:
    def __init__(self, v0, v1, o0, o1, f0, f1):
        self.v0 = v0  # vertex 0
        self.v1 = v1  # vertex 1
        self.o0 = o0  # opposite vertex 1
        self.o1 = o1  # opposite vertex 2
        self.f0 = f0  # index of tri1
        self.f1 = f1  # index of tri2


class MeshAdjacency:
    def __init__(self, indices, num_tris):
        # map edges (v0, v1) to faces (f0, f1)
        self.edges = {}
        self.indices = indices

        for index, tri in enumerate(indices):
            self.add_edge(tri[0], tri[1], tri[2], index)
            self.add_edge(tri[1], tri[2], tri[0], index)
            self.add_edge(tri[2], tri[0], tri[1], index)

    def add_edge(self, i0, i1, o, f):  # index1, index2, index3, index of triangle
        key = (min(i0, i1), max(i0, i1))
        edge = None

        if key in self.edges:
            edge = self.edges[key]

            if edge.f1 != -1:
                print("Detected non-manifold edge")
                return
            else:
                # update other side of the edge
                edge.o1 = o
                edge.f1 = f
        else:
            # create new edge with opposite yet to be filled
            edge = MeshEdge(i0, i1, o, -1, f, -1)

        self.edges[key] = edge


def mem_report(): #pragma: no cover
    def _mem_report(tensors, mem_type):
        """Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation"""
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel * element_size / 1024 / 1024  # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            # print('%s\t\t%s\t\t%.2f' % (
            #     element_type,
            #     size,
            #     mem) )
        print("Type: %s Total Tensors: %d \tUsed Memory Space: %.2f MBytes" % (mem_type, total_numel, total_mem))

    import gc

    import torch

    gc.collect()

    LEN = 65
    objects = gc.get_objects()
    # print('%s\t%s\t\t\t%s' %('Element type', 'Size', 'Used MEM(MBytes)') )
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, "GPU")
    _mem_report(host_tensors, "CPU")
    print("=" * LEN)



class ScopedDevice:
    def __init__(self, device):
        self.device = wp.get_device(device)

    def __enter__(self):
        # save the previous default device
        self.saved_device = self.device.runtime.default_device

        # make this the default device
        self.device.runtime.default_device = self.device

        # make it the current CUDA device so that device alias "cuda" will evaluate to this device
        self.device.context_guard.__enter__()

        return self.device

    def __exit__(self, exc_type, exc_value, traceback):
        # restore original CUDA context
        self.device.context_guard.__exit__(exc_type, exc_value, traceback)

        # restore original target device
        self.device.runtime.default_device = self.saved_device


class ScopedStream:
    def __init__(self, stream):
        self.stream = stream
        if stream is not None:
            self.device = stream.device
            self.device_scope = ScopedDevice(self.device)

    def __enter__(self):
        if self.stream is not None:
            self.device_scope.__enter__()
            self.saved_stream = self.device.stream
            self.device.stream = self.stream

        return self.stream

    def __exit__(self, exc_type, exc_value, traceback):
        if self.stream is not None:
            self.device.stream = self.saved_stream
            self.device_scope.__exit__(exc_type, exc_value, traceback)


# timer utils
class ScopedTimer:
    indent = -1

    enabled = True

    def __init__(
        self,
        name,
        active=True,
        print=True,
        detailed=False,
        dict=None,
        use_nvtx=False,
        color="rapids",
        synchronize=False,
    ):
        """Context manager object for a timer

        Parameters:
            name (str): Name of timer
            active (bool): Enables this timer
            print (bool): At context manager exit, print elapsed time to sys.stdout
            detailed (bool): Collects additional profiling data using cProfile and calls ``print_stats()`` at context exit
            dict (dict): A dictionary of lists to which the elapsed time will be appended using ``name`` as a key
            use_nvtx (bool): If true, timing functionality is replaced by an NVTX range
            color (int or str): ARGB value (e.g. 0x00FFFF) or color name (e.g. 'cyan') associated with the NVTX range
            synchronize (bool): Synchronize the CPU thread with any outstanding CUDA work to return accurate GPU timings

        Attributes:
            elapsed (float): The duration of the ``with`` block used with this object
        """
        self.name = name
        self.active = active and self.enabled
        self.print = print
        self.detailed = detailed
        self.dict = dict
        self.use_nvtx = use_nvtx
        self.color = color
        self.synchronize = synchronize
        self.elapsed = 0.0

        if self.dict is not None:
            if name not in self.dict:
                self.dict[name] = []

    def __enter__(self):
        if self.active:
            if self.synchronize:
                wp.synchronize()

            if self.use_nvtx:
                import nvtx

                self.nvtx_range_id = nvtx.start_range(self.name, color=self.color)
                return

            self.start = timeit.default_timer()
            ScopedTimer.indent += 1

            if self.detailed:
                self.cp = cProfile.Profile()
                self.cp.clear()
                self.cp.enable()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.active:
            if self.synchronize:
                wp.synchronize()

            if self.use_nvtx:
                import nvtx

                nvtx.end_range(self.nvtx_range_id)
                return

            if self.detailed:
                self.cp.disable()
                self.cp.print_stats(sort="tottime")

            self.elapsed = (timeit.default_timer() - self.start) * 1000.0

            if self.dict is not None:
                self.dict[self.name].append(self.elapsed)

            indent = ""
            for i in range(ScopedTimer.indent):
                indent += "\t"

            if self.print:
                print("{}{} took {:.2f} ms".format(indent, self.name, self.elapsed))

            ScopedTimer.indent -= 1


# helper kernels for adj_matmul
@wp.kernel
def add_kernel_2d(x: wp.array2d(dtype=Any), acc: wp.array2d(dtype=Any), beta: Any):
    i, j = wp.tid()

    x[i,j] = x[i,j] + beta * acc[i,j]

@wp.kernel
def add_kernel_3d(x: wp.array3d(dtype=Any), acc: wp.array3d(dtype=Any), beta: Any):
    i, j, k = wp.tid()

    x[i,j,k] = x[i,j,k] + beta * acc[i,j,k]
