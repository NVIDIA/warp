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

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *
from warp.utils import check_p2p


class Capturable:
    def __init__(self, use_graph=True, stream=None):
        self.use_graph = use_graph
        self.stream = stream

    def __enter__(self):
        if self.use_graph:
            # preload module before graph capture
            wp.load_module(device=wp.get_device())
            wp.capture_begin(stream=self.stream, force_module_load=False)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.use_graph:
            try:
                # need to call capture_end() to terminate the CUDA stream capture
                graph = wp.capture_end(stream=self.stream)
            except Exception:
                # capture_end() will raise if there was an error during capture, but we squash it here
                # if we already had an exception so that the original exception percolates to the caller
                if exc_type is None:
                    raise
            else:
                # capture can succeed despite some errors during capture (e.g. cudaInvalidValue during copy)
                # but if we had an exception during capture, don't launch the graph
                if exc_type is None:
                    wp.capture_launch(graph, stream=self.stream)


@wp.kernel
def inc(a: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = a[tid] + 1.0


def test_async_empty(test, device, use_mempools, use_graph):
    with wp.ScopedDevice(device), wp.ScopedMempool(device, use_mempools):
        n = 100

        with Capturable(use_graph):
            a = wp.empty(n, dtype=float)

        test.assertIsInstance(a, wp.array)
        test.assertIsNotNone(a.ptr)
        test.assertEqual(a.size, n)
        test.assertEqual(a.dtype, wp.float32)
        test.assertEqual(a.device, device)


def test_async_zeros(test, device, use_mempools, use_graph):
    with wp.ScopedDevice(device), wp.ScopedMempool(device, use_mempools):
        n = 100

        with Capturable(use_graph):
            a = wp.zeros(n, dtype=float)

        assert_np_equal(a.numpy(), np.zeros(n, dtype=np.float32))


def test_async_zero_v1(test, device, use_mempools, use_graph):
    with wp.ScopedDevice(device), wp.ScopedMempool(device, use_mempools):
        n = 100

        with Capturable(use_graph):
            a = wp.empty(n, dtype=float)
            a.zero_()

        assert_np_equal(a.numpy(), np.zeros(n, dtype=np.float32))


def test_async_zero_v2(test, device, use_mempools, use_graph):
    with wp.ScopedDevice(device), wp.ScopedMempool(device, use_mempools):
        n = 100

        a = wp.empty(n, dtype=float)

        with Capturable(use_graph):
            a.zero_()

        assert_np_equal(a.numpy(), np.zeros(n, dtype=np.float32))


def test_async_full(test, device, use_mempools, use_graph):
    with wp.ScopedDevice(device), wp.ScopedMempool(device, use_mempools):
        n = 100
        value = 42

        with Capturable(use_graph):
            a = wp.full(n, value, dtype=float)

        assert_np_equal(a.numpy(), np.full(n, value, dtype=np.float32))


def test_async_fill_v1(test, device, use_mempools, use_graph):
    with wp.ScopedDevice(device), wp.ScopedMempool(device, use_mempools):
        n = 100
        value = 17

        with Capturable(use_graph):
            a = wp.empty(n, dtype=float)
            a.fill_(value)

        assert_np_equal(a.numpy(), np.full(n, value, dtype=np.float32))


def test_async_fill_v2(test, device, use_mempools, use_graph):
    with wp.ScopedDevice(device), wp.ScopedMempool(device, use_mempools):
        n = 100
        value = 17

        a = wp.empty(n, dtype=float)

        with Capturable(use_graph):
            a.fill_(value)

        assert_np_equal(a.numpy(), np.full(n, value, dtype=np.float32))


def test_async_kernels_v1(test, device, use_mempools, use_graph):
    with wp.ScopedDevice(device), wp.ScopedMempool(device, use_mempools):
        n = 100
        num_iters = 10

        with Capturable(use_graph):
            a = wp.zeros(n, dtype=float)
            for _i in range(num_iters):
                wp.launch(inc, dim=a.size, inputs=[a])

        assert_np_equal(a.numpy(), np.full(n, num_iters, dtype=np.float32))


def test_async_kernels_v2(test, device, use_mempools, use_graph):
    with wp.ScopedDevice(device), wp.ScopedMempool(device, use_mempools):
        n = 100
        num_iters = 10

        a = wp.zeros(n, dtype=float)

        with Capturable(use_graph):
            for _i in range(num_iters):
                wp.launch(inc, dim=a.size, inputs=[a])

        assert_np_equal(a.numpy(), np.full(n, num_iters, dtype=np.float32))


class TestAsync(unittest.TestCase):
    pass


# get all CUDA devices
cuda_devices = wp.get_cuda_devices()

# get CUDA devices that support mempools
cuda_devices_with_mempools = []
for d in cuda_devices:
    if d.is_mempool_supported:
        cuda_devices_with_mempools.append(d)

# get a pair of CUDA devices that support mempool access
cuda_devices_with_mempool_access = []
for target_device in cuda_devices_with_mempools:
    for peer_device in cuda_devices_with_mempools:
        if peer_device != target_device:
            if wp.is_mempool_access_supported(target_device, peer_device):
                cuda_devices_with_mempool_access = [target_device, peer_device]
                break
    if cuda_devices_with_mempool_access:
        break


def add_test_variants(
    func,
    device_count=1,
    graph_allocs=False,
    requires_mempool_access_with_graph=False,
):
    # test that works with default allocators
    if not graph_allocs and device_count <= len(cuda_devices):
        devices = cuda_devices[:device_count]

        def func1(t, d):
            return func(t, *devices, False, False)

        def func2(t, d):
            return func(t, *devices, False, True)

        name1 = f"{func.__name__}_DefaultAlloc_NoGraph"
        name2 = f"{func.__name__}_DefaultAlloc_WithGraph"
        if device_count == 1:
            add_function_test(TestAsync, name1, func1, devices=devices)
            add_function_test(TestAsync, name2, func2, devices=devices)
        else:
            add_function_test(TestAsync, name1, func1)
            add_function_test(TestAsync, name2, func2)

    # test that works with mempool allocators
    if device_count <= len(cuda_devices_with_mempools):
        devices = cuda_devices_with_mempools[:device_count]

        def func3(t, d):
            return func(t, *devices, True, False)

        name3 = f"{func.__name__}_MempoolAlloc_NoGraph"
        if device_count == 1:
            add_function_test(TestAsync, name3, func3, devices=devices)
        else:
            add_function_test(TestAsync, name3, func3)

    # test that requires devices with mutual mempool access during graph capture (e.g., p2p memcpy limitation)
    if requires_mempool_access_with_graph:
        suitable_devices = cuda_devices_with_mempool_access
    else:
        suitable_devices = cuda_devices_with_mempools

    if device_count <= len(suitable_devices):
        devices = suitable_devices[:device_count]

        def func4(t, d):
            return func(t, *devices, True, True)

        name4 = f"{func.__name__}_MempoolAlloc_WithGraph"
        if device_count == 1:
            add_function_test(TestAsync, name4, func4, devices=devices)
        else:
            add_function_test(TestAsync, name4, func4)


add_test_variants(test_async_empty, graph_allocs=True)
add_test_variants(test_async_zeros, graph_allocs=True)
add_test_variants(test_async_zero_v1, graph_allocs=True)
add_test_variants(test_async_zero_v2, graph_allocs=False)
add_test_variants(test_async_full, graph_allocs=True)
add_test_variants(test_async_fill_v1, graph_allocs=True)
add_test_variants(test_async_fill_v2, graph_allocs=False)
add_test_variants(test_async_kernels_v1, graph_allocs=True)
add_test_variants(test_async_kernels_v2, graph_allocs=False)


# =================================================================================
# wp.copy() tests
# =================================================================================


def as_contiguous_array(data, device=None, grad_data=None):
    a = wp.array(data=data, device=device, copy=True)
    if grad_data is not None:
        a.grad = as_contiguous_array(grad_data, device=device)
    return a


def as_strided_array(data, device=None, grad_data=None):
    a = wp.array(data=data, device=device)
    # make a copy with non-contiguous strides
    strides = (*a.strides[:-1], 2 * a.strides[-1])
    strided_a = wp.zeros(shape=a.shape, strides=strides, dtype=a.dtype, device=device)
    wp.copy(strided_a, a)
    if grad_data is not None:
        strided_a.grad = as_strided_array(grad_data, device=device)
    return strided_a


def as_indexed_array(data, device=None, **kwargs):
    a = wp.array(data=data, device=device)
    # allocate double the elements so we can index half of them
    shape = (*a.shape[:-1], 2 * a.shape[-1])
    big_a = wp.zeros(shape=shape, dtype=a.dtype, device=device)
    indices = wp.array(data=np.arange(0, shape[-1], 2, dtype=np.int32), device=device)
    indexed_a = big_a[indices]
    wp.copy(indexed_a, a)
    return indexed_a


def as_fabric_array(data, device=None, **kwargs):
    from warp.tests.test_fabricarray import _create_fabric_array_interface

    a = wp.array(data=data, device=device)
    iface = _create_fabric_array_interface(a, "foo")
    fa = wp.fabricarray(data=iface, attrib="foo")
    fa._iface = iface  # save data reference
    return fa


def as_indexed_fabric_array(data, device=None, **kwargs):
    from warp.tests.test_fabricarray import _create_fabric_array_interface

    a = wp.array(data=data, device=device)
    shape = (*a.shape[:-1], 2 * a.shape[-1])
    # allocate double the elements so we can index half of them
    big_a = wp.zeros(shape=shape, dtype=a.dtype, device=device)
    indices = wp.array(data=np.arange(0, shape[-1], 2, dtype=np.int32), device=device)
    iface = _create_fabric_array_interface(big_a, "foo", copy=True)
    fa = wp.fabricarray(data=iface, attrib="foo")
    fa._iface = iface  # save data reference
    indexed_fa = fa[indices]
    wp.copy(indexed_fa, a)
    return indexed_fa


class CopyParams:
    def __init__(
        self,
        with_grad=False,  # whether to use arrays with gradients (contiguous and strided only)
        src_use_mempool=False,  # whether to enable memory pool on source device
        dst_use_mempool=False,  # whether to enable memory pool on destination device
        access_dst_src=False,  # whether destination device has access to the source mempool
        access_src_dst=False,  # whether source device has access to the destination mempool
        stream_device=None,  # the device for the stream (None for default behaviour)
        use_graph=False,  # whether to use a graph
        value_offset=0,  # unique offset for generated data values per test
    ):
        self.with_grad = with_grad
        self.src_use_mempool = src_use_mempool
        self.dst_use_mempool = dst_use_mempool
        self.access_dst_src = access_dst_src
        self.access_src_dst = access_src_dst
        self.stream_device = stream_device
        self.use_graph = use_graph
        self.value_offset = value_offset


def copy_template(test, src_ctor, dst_ctor, src_device, dst_device, n, params: CopyParams):
    # activate the given memory pool configuration
    with wp.ScopedMempool(src_device, params.src_use_mempool), wp.ScopedMempool(
        dst_device, params.dst_use_mempool
    ), wp.ScopedMempoolAccess(dst_device, src_device, params.access_dst_src), wp.ScopedMempoolAccess(
        src_device, dst_device, params.access_src_dst
    ):
        # make sure the data are different between tests by adding a unique offset
        # this avoids aliasing issues with older memory
        src_data = np.arange(params.value_offset, params.value_offset + n, dtype=np.float32)
        dst_data = np.zeros(n, dtype=np.float32)

        if params.with_grad:
            src_grad_data = -np.arange(params.value_offset, params.value_offset + n, dtype=np.float32)
            dst_grad_data = np.zeros(n, dtype=np.float32)
        else:
            src_grad_data = None
            dst_grad_data = None

        # create Warp arrays for the copy
        src = src_ctor(src_data, device=src_device, grad_data=src_grad_data)
        dst = dst_ctor(dst_data, device=dst_device, grad_data=dst_grad_data)

        # determine the stream argument to pass to wp.copy()
        if params.stream_device is not None:
            stream_arg = wp.Stream(params.stream_device)
        else:
            stream_arg = None

        # determine the actual stream used for the copy
        if stream_arg is not None:
            stream = stream_arg
        else:
            if dst_device.is_cuda:
                stream = dst_device.stream
            elif src_device.is_cuda:
                stream = src_device.stream
            else:
                stream = None

        # check if an exception is expected given the arguments and system configuration
        expected_error_type = None
        expected_error_regex = None

        # restrictions on copying between different devices during graph capture
        if params.use_graph and src_device != dst_device:
            # errors with allocating staging buffer on source device
            if not src.is_contiguous:
                if src_device.is_cuda and not src_device.is_mempool_enabled:
                    # can't allocate staging buffer using default CUDA allocator during capture
                    expected_error_type, expected_error_regex = RuntimeError, r"^Failed to allocate"
                elif src_device.is_cpu:
                    # can't allocate CPU staging buffer during capture
                    expected_error_type, expected_error_regex = RuntimeError, r"^Failed to allocate"

            # errors with allocating staging buffer on destination device
            if expected_error_type is None:
                if not dst.is_contiguous:
                    if dst_device.is_cuda and not dst_device.is_mempool_enabled:
                        # can't allocate staging buffer using default CUDA allocator during capture
                        expected_error_type, expected_error_regex = RuntimeError, r"^Failed to allocate"
                    elif dst_device.is_cpu and src_device.is_cuda:
                        # can't allocate CPU staging buffer during capture
                        expected_error_type, expected_error_regex = RuntimeError, r"^Failed to allocate"

            # p2p copies and mempool access
            if expected_error_type is None and src_device.is_cuda and dst_device.is_cuda:
                # If the source is a contiguous mempool allocation or a non-contiguous array
                # AND the destination is a contiguous mempool allocation or a non-contiguous array,
                # then memory pool access needs to be enabled EITHER from src_device to dst_device
                # OR from dst_device to src_device.
                if (
                    ((src.is_contiguous and params.src_use_mempool) or not src.is_contiguous)
                    and ((dst.is_contiguous and params.dst_use_mempool) or not dst.is_contiguous)
                    and not wp.is_mempool_access_enabled(src_device, dst_device)
                    and not wp.is_mempool_access_enabled(dst_device, src_device)
                ):
                    expected_error_type, expected_error_regex = RuntimeError, r"^Warp copy error"

        # synchronize before test
        wp.synchronize()

        if expected_error_type is not None:
            # disable error output from Warp if we expect an exception
            try:
                saved_error_output_enabled = wp.context.runtime.core.is_error_output_enabled()
                wp.context.runtime.core.set_error_output_enabled(False)
                with test.assertRaisesRegex(expected_error_type, expected_error_regex):
                    with Capturable(use_graph=params.use_graph, stream=stream):
                        wp.copy(dst, src, stream=stream_arg)
            finally:
                wp.context.runtime.core.set_error_output_enabled(saved_error_output_enabled)
                wp.synchronize()

                # print(f"SUCCESSFUL ERROR PREDICTION: {expected_error_regex}")

        else:
            with Capturable(use_graph=params.use_graph, stream=stream):
                wp.copy(dst, src, stream=stream_arg)

            # synchronize the stream where the copy was running (None for h2h copies)
            if stream is not None:
                wp.synchronize_stream(stream)

            assert_np_equal(dst.numpy(), src.numpy())

            if params.with_grad:
                assert_np_equal(dst.grad.numpy(), src.grad.numpy())

            # print("SUCCESSFUL COPY")


array_constructors = {
    "contiguous": as_contiguous_array,
    "strided": as_strided_array,
    "indexed": as_indexed_array,
    "fabric": as_fabric_array,
    "indexedfabric": as_indexed_fabric_array,
}

array_type_codes = {
    "contiguous": "c",
    "strided": "s",
    "indexed": "i",
    "fabric": "f",
    "indexedfabric": "fi",
}

device_pairs = {}
cpu = None
cuda0 = None
cuda1 = None
cuda2 = None
if wp.is_cpu_available():
    cpu = wp.get_device("cpu")
    device_pairs["h2h"] = (cpu, cpu)
if wp.is_cuda_available():
    cuda0 = wp.get_device("cuda:0")
    device_pairs["d2d"] = (cuda0, cuda0)
    if wp.is_cpu_available():
        device_pairs["h2d"] = (cpu, cuda0)
        device_pairs["d2h"] = (cuda0, cpu)
if wp.get_cuda_device_count() > 1:
    cuda1 = wp.get_device("cuda:1")
    device_pairs["p2p"] = (cuda0, cuda1)
if wp.get_cuda_device_count() > 2:
    cuda2 = wp.get_device("cuda:2")

num_copy_elems = 1000000
num_copy_tests = 0


def add_copy_test(test_name, src_ctor, dst_ctor, src_device, dst_device, n, params):
    def test_func(
        test,
        device,
        src_ctor=src_ctor,
        dst_ctor=dst_ctor,
        src_device=src_device,
        dst_device=dst_device,
        n=n,
        params=params,
    ):
        return copy_template(test, src_ctor, dst_ctor, src_device, dst_device, n, params)

    add_function_test(TestAsync, test_name, test_func, check_output=False)


# Procedurally add tests with argument combinations supported by the system.
for src_type, src_ctor in array_constructors.items():
    for dst_type, dst_ctor in array_constructors.items():
        copy_type = f"{array_type_codes[src_type]}2{array_type_codes[dst_type]}"

        for transfer_type, device_pair in device_pairs.items():
            # skip p2p tests if not supported (e.g., IOMMU is enabled on Linux)
            if transfer_type == "p2p" and not check_p2p():
                continue

            src_device = device_pair[0]
            dst_device = device_pair[1]

            # basic copy arguments
            copy_args = (src_ctor, dst_ctor, src_device, dst_device, num_copy_elems)

            if src_device.is_cuda and src_device.is_mempool_supported:
                src_mempool_flags = [False, True]
            else:
                src_mempool_flags = [False]

            if dst_device.is_cuda and dst_device.is_mempool_supported:
                dst_mempool_flags = [False, True]
            else:
                dst_mempool_flags = [False]

            # stream options
            if src_device.is_cuda:
                if dst_device.is_cuda:
                    if src_device == dst_device:
                        # d2d
                        assert src_device == cuda0 and dst_device == cuda0
                        if cuda1 is not None:
                            stream_devices = [None, cuda0, cuda1]
                        else:
                            stream_devices = [None, cuda0]
                    else:
                        # p2p
                        assert src_device == cuda0 and dst_device == cuda1
                        if cuda2 is not None:
                            stream_devices = [None, cuda0, cuda1, cuda2]
                        else:
                            stream_devices = [None, cuda0, cuda1]
                else:
                    # d2h
                    assert src_device == cuda0
                    if cuda1 is not None:
                        stream_devices = [None, cuda0, cuda1]
                    else:
                        stream_devices = [None, cuda0]
            else:
                if dst_device.is_cuda:
                    # h2d
                    assert dst_device == cuda0
                    if cuda1 is not None:
                        stream_devices = [None, cuda0, cuda1]
                    else:
                        stream_devices = [None, cuda0]
                else:
                    # h2h
                    stream_devices = [None]

            # gradient options (only supported with contiguous and strided arrays)
            if src_type in ("contiguous", "strided") and dst_type in ("contiguous", "strided"):
                grad_flags = [False, True]
            else:
                grad_flags = [False]

            # graph capture options (only supported with CUDA devices)
            if src_device.is_cuda or dst_device.is_cuda:
                graph_flags = [False, True]
            else:
                graph_flags = [False]

            # access from destination device to source mempool
            if wp.is_mempool_access_supported(dst_device, src_device):
                access_dst_src_flags = [False, True]
            else:
                access_dst_src_flags = [False]

            # access from source device to destination mempool
            if wp.is_mempool_access_supported(src_device, dst_device):
                access_src_dst_flags = [False, True]
            else:
                access_src_dst_flags = [False]

            for src_use_mempool in src_mempool_flags:
                for dst_use_mempool in dst_mempool_flags:
                    for stream_device in stream_devices:
                        for access_dst_src in access_dst_src_flags:
                            for access_src_dst in access_src_dst_flags:
                                for with_grad in grad_flags:
                                    for use_graph in graph_flags:
                                        test_name = f"test_copy_{copy_type}_{transfer_type}"

                                        if src_use_mempool:
                                            test_name += "_SrcPoolOn"
                                        else:
                                            test_name += "_SrcPoolOff"

                                        if dst_use_mempool:
                                            test_name += "_DstPoolOn"
                                        else:
                                            test_name += "_DstPoolOff"

                                        if stream_device is None:
                                            test_name += "_NoStream"
                                        elif stream_device == cuda0:
                                            test_name += "_Stream0"
                                        elif stream_device == cuda1:
                                            test_name += "_Stream1"
                                        elif stream_device == cuda2:
                                            test_name += "_Stream2"
                                        else:
                                            raise AssertionError

                                        if with_grad:
                                            test_name += "_Grad"
                                        else:
                                            test_name += "_NoGrad"

                                        if use_graph:
                                            test_name += "_Graph"
                                        else:
                                            test_name += "_NoGraph"

                                        if access_dst_src and access_src_dst:
                                            test_name += "_AccessBoth"
                                        elif access_dst_src and not access_src_dst:
                                            test_name += "_AccessDstSrc"
                                        elif not access_dst_src and access_src_dst:
                                            test_name += "_AccessSrcDst"
                                        else:
                                            test_name += "_AccessNone"

                                        copy_params = CopyParams(
                                            src_use_mempool=src_use_mempool,
                                            dst_use_mempool=dst_use_mempool,
                                            access_dst_src=access_dst_src,
                                            access_src_dst=access_src_dst,
                                            stream_device=stream_device,
                                            with_grad=with_grad,
                                            use_graph=use_graph,
                                            value_offset=num_copy_tests,
                                        )

                                        add_copy_test(test_name, *copy_args, copy_params)

                                        num_copy_tests += 1

# Specify individual test(s) for debugging purposes
# add_copy_test("test_a", as_contiguous_array, as_strided_array, cuda0, cuda1, num_copy_elems,
#               CopyParams(
#                     src_use_mempool=True,
#                     dst_use_mempool=True,
#                     access_dst_src=False,
#                     access_src_dst=False,
#                     stream_device=cuda0,
#                     with_grad=False,
#                     use_graph=True,
#                     value_offset=0))

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
