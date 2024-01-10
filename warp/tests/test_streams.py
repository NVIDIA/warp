# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

wp.init()


@wp.kernel
def inc(a: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = a[tid] + 1.0


@wp.kernel
def inc_new(src: wp.array(dtype=float), dst: wp.array(dtype=float)):
    tid = wp.tid()
    dst[tid] = src[tid] + 1.0


@wp.kernel
def sum(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]


# number of elements to use for testing
N = 10 * 1024 * 1024


def test_stream_arg_implicit_sync(test, device):
    # wp.zeros() and array.numpy() should not require explicit sync

    a = wp.zeros(N, dtype=float, device=device)
    b = wp.empty(N, dtype=float, device=device)
    c = wp.empty(N, dtype=float, device=device)

    new_stream = wp.Stream(device)

    # Exercise code path
    wp.set_stream(new_stream, device)

    test.assertTrue(wp.get_device(device).has_stream)

    # launch work on new stream
    wp.launch(inc, dim=a.size, inputs=[a], stream=new_stream)
    wp.copy(b, a, stream=new_stream)
    wp.launch(inc, dim=a.size, inputs=[a], stream=new_stream)
    wp.copy(c, a, stream=new_stream)
    wp.launch(inc, dim=a.size, inputs=[a], stream=new_stream)

    assert_np_equal(a.numpy(), np.full(N, fill_value=3.0))
    assert_np_equal(b.numpy(), np.full(N, fill_value=1.0))
    assert_np_equal(c.numpy(), np.full(N, fill_value=2.0))


def test_stream_scope_implicit_sync(test, device):
    # wp.zeros() and array.numpy() should not require explicit sync

    with wp.ScopedDevice(device):
        a = wp.zeros(N, dtype=float)
        b = wp.empty(N, dtype=float)
        c = wp.empty(N, dtype=float)

        old_stream = wp.get_stream()
        new_stream = wp.Stream()

        # launch work on new stream
        with wp.ScopedStream(new_stream):
            assert wp.get_stream() == new_stream

            wp.launch(inc, dim=a.size, inputs=[a])
            wp.copy(b, a)
            wp.launch(inc, dim=a.size, inputs=[a])
            wp.copy(c, a)
            wp.launch(inc, dim=a.size, inputs=[a])

        assert wp.get_stream() == old_stream

        assert_np_equal(a.numpy(), np.full(N, fill_value=3.0))
        assert_np_equal(b.numpy(), np.full(N, fill_value=1.0))
        assert_np_equal(c.numpy(), np.full(N, fill_value=2.0))


def test_stream_arg_synchronize(test, device):
    a = wp.zeros(N, dtype=float, device=device)
    b = wp.empty(N, dtype=float, device=device)
    c = wp.empty(N, dtype=float, device=device)
    d = wp.empty(N, dtype=float, device=device)

    stream1 = wp.get_stream(device)
    stream2 = wp.Stream(device)
    stream3 = wp.Stream(device)

    wp.launch(inc, dim=N, inputs=[a], device=device)

    # b and c depend on a
    wp.synchronize_stream(stream1)
    wp.launch(inc_new, dim=N, inputs=[a, b], stream=stream2)
    wp.launch(inc_new, dim=N, inputs=[a, c], stream=stream3)

    # d depends on b and c
    wp.synchronize_stream(stream2)
    wp.synchronize_stream(stream3)
    wp.launch(sum, dim=N, inputs=[b, c, d], device=device)

    assert_np_equal(a.numpy(), np.full(N, fill_value=1.0))
    assert_np_equal(b.numpy(), np.full(N, fill_value=2.0))
    assert_np_equal(c.numpy(), np.full(N, fill_value=2.0))
    assert_np_equal(d.numpy(), np.full(N, fill_value=4.0))


def test_stream_arg_wait_event(test, device):
    a = wp.zeros(N, dtype=float, device=device)
    b = wp.empty(N, dtype=float, device=device)
    c = wp.empty(N, dtype=float, device=device)
    d = wp.empty(N, dtype=float, device=device)

    stream1 = wp.get_stream(device)
    stream2 = wp.Stream(device)
    stream3 = wp.Stream(device)

    event1 = wp.Event(device)
    event2 = wp.Event(device)
    event3 = wp.Event(device)

    wp.launch(inc, dim=N, inputs=[a], stream=stream1)
    stream1.record_event(event1)

    # b and c depend on a
    stream2.wait_event(event1)
    stream3.wait_event(event1)
    wp.launch(inc_new, dim=N, inputs=[a, b], stream=stream2)
    stream2.record_event(event2)
    wp.launch(inc_new, dim=N, inputs=[a, c], stream=stream3)
    stream3.record_event(event3)

    # d depends on b and c
    stream1.wait_event(event2)
    stream1.wait_event(event3)
    wp.launch(sum, dim=N, inputs=[b, c, d], stream=stream1)

    assert_np_equal(a.numpy(), np.full(N, fill_value=1.0))
    assert_np_equal(b.numpy(), np.full(N, fill_value=2.0))
    assert_np_equal(c.numpy(), np.full(N, fill_value=2.0))
    assert_np_equal(d.numpy(), np.full(N, fill_value=4.0))


def test_stream_arg_wait_stream(test, device):
    a = wp.zeros(N, dtype=float, device=device)
    b = wp.empty(N, dtype=float, device=device)
    c = wp.empty(N, dtype=float, device=device)
    d = wp.empty(N, dtype=float, device=device)

    stream1 = wp.get_stream(device)
    stream2 = wp.Stream(device)
    stream3 = wp.Stream(device)

    wp.launch(inc, dim=N, inputs=[a], stream=stream1)

    # b and c depend on a
    stream2.wait_stream(stream1)
    stream3.wait_stream(stream1)
    wp.launch(inc_new, dim=N, inputs=[a, b], stream=stream2)
    wp.launch(inc_new, dim=N, inputs=[a, c], stream=stream3)

    # d depends on b and c
    stream1.wait_stream(stream2)
    stream1.wait_stream(stream3)
    wp.launch(sum, dim=N, inputs=[b, c, d], stream=stream1)

    assert_np_equal(a.numpy(), np.full(N, fill_value=1.0))
    assert_np_equal(b.numpy(), np.full(N, fill_value=2.0))
    assert_np_equal(c.numpy(), np.full(N, fill_value=2.0))
    assert_np_equal(d.numpy(), np.full(N, fill_value=4.0))


def test_stream_scope_synchronize(test, device):
    with wp.ScopedDevice(device):
        a = wp.zeros(N, dtype=float)
        b = wp.empty(N, dtype=float)
        c = wp.empty(N, dtype=float)
        d = wp.empty(N, dtype=float)

        stream2 = wp.Stream()
        stream3 = wp.Stream()

        wp.launch(inc, dim=N, inputs=[a])

        # b and c depend on a
        wp.synchronize_stream()
        with wp.ScopedStream(stream2):
            wp.launch(inc_new, dim=N, inputs=[a, b])
        with wp.ScopedStream(stream3):
            wp.launch(inc_new, dim=N, inputs=[a, c])

        # d depends on b and c
        wp.synchronize_stream(stream2)
        wp.synchronize_stream(stream3)
        wp.launch(sum, dim=N, inputs=[b, c, d])

        assert_np_equal(a.numpy(), np.full(N, fill_value=1.0))
        assert_np_equal(b.numpy(), np.full(N, fill_value=2.0))
        assert_np_equal(c.numpy(), np.full(N, fill_value=2.0))
        assert_np_equal(d.numpy(), np.full(N, fill_value=4.0))


def test_stream_scope_wait_event(test, device):
    with wp.ScopedDevice(device):
        a = wp.zeros(N, dtype=float)
        b = wp.empty(N, dtype=float)
        c = wp.empty(N, dtype=float)
        d = wp.empty(N, dtype=float)

        stream2 = wp.Stream()
        stream3 = wp.Stream()

        event1 = wp.Event()
        event2 = wp.Event()
        event3 = wp.Event()

        wp.launch(inc, dim=N, inputs=[a])
        wp.record_event(event1)

        # b and c depend on a
        with wp.ScopedStream(stream2):
            wp.wait_event(event1)
            wp.launch(inc_new, dim=N, inputs=[a, b])
            wp.record_event(event2)
        with wp.ScopedStream(stream3):
            wp.wait_event(event1)
            wp.launch(inc_new, dim=N, inputs=[a, c])
            wp.record_event(event3)

        # d depends on b and c
        wp.wait_event(event2)
        wp.wait_event(event3)
        wp.launch(sum, dim=N, inputs=[b, c, d])

        assert_np_equal(a.numpy(), np.full(N, fill_value=1.0))
        assert_np_equal(b.numpy(), np.full(N, fill_value=2.0))
        assert_np_equal(c.numpy(), np.full(N, fill_value=2.0))
        assert_np_equal(d.numpy(), np.full(N, fill_value=4.0))


def test_stream_scope_wait_stream(test, device):
    with wp.ScopedDevice(device):
        a = wp.zeros(N, dtype=float)
        b = wp.empty(N, dtype=float)
        c = wp.empty(N, dtype=float)
        d = wp.empty(N, dtype=float)

        stream1 = wp.get_stream()
        stream2 = wp.Stream()
        stream3 = wp.Stream()

        wp.launch(inc, dim=N, inputs=[a])

        # b and c depend on a
        with wp.ScopedStream(stream2):
            wp.wait_stream(stream1)
            wp.launch(inc_new, dim=N, inputs=[a, b])
        with wp.ScopedStream(stream3):
            wp.wait_stream(stream1)
            wp.launch(inc_new, dim=N, inputs=[a, c])

        # d depends on b and c
        wp.wait_stream(stream2)
        wp.wait_stream(stream3)
        wp.launch(sum, dim=N, inputs=[b, c, d])

        assert_np_equal(a.numpy(), np.full(N, fill_value=1.0))
        assert_np_equal(b.numpy(), np.full(N, fill_value=2.0))
        assert_np_equal(c.numpy(), np.full(N, fill_value=2.0))
        assert_np_equal(d.numpy(), np.full(N, fill_value=4.0))


devices = get_unique_cuda_test_devices()


class TestStreams(unittest.TestCase):
    def test_stream_exceptions(self):
        cpu_device = wp.get_device("cpu")

        # Can't set the stream on a CPU device
        with self.assertRaises(RuntimeError):
            stream0 = wp.Stream()
            cpu_device.stream = stream0

        # Can't create a stream on the CPU
        with self.assertRaises(RuntimeError):
            wp.Stream(device="cpu")

        # Can't create an event with CPU device
        with self.assertRaises(RuntimeError):
            wp.Event(device=cpu_device)

        # Can't get the stream on a CPU device
        with self.assertRaises(RuntimeError):
            cpu_stream = cpu_device.stream  # noqa: F841

    @unittest.skipUnless(len(wp.get_cuda_devices()) > 1, "Requires at least two CUDA devices")
    def test_stream_arg_graph_mgpu(self):
        wp.load_module(device="cuda:0")
        wp.load_module(device="cuda:1")

        # resources on GPU 0
        stream0 = wp.get_stream("cuda:0")
        a0 = wp.zeros(N, dtype=float, device="cuda:0")
        b0 = wp.empty(N, dtype=float, device="cuda:0")
        c0 = wp.empty(N, dtype=float, device="cuda:0")

        # resources on GPU 1
        stream1 = wp.get_stream("cuda:1")
        a1 = wp.zeros(N, dtype=float, device="cuda:1")

        # start recording on stream0
        wp.capture_begin(stream=stream0, force_module_load=False)
        try:
            # branch into stream1
            stream1.wait_stream(stream0)

            # launch concurrent kernels on each stream
            wp.launch(inc, dim=N, inputs=[a0], stream=stream0)
            wp.launch(inc, dim=N, inputs=[a1], stream=stream1)

            # wait for stream1 to finish
            stream0.wait_stream(stream1)

            # copy values from stream1
            wp.copy(b0, a1, stream=stream0)

            # compute sum
            wp.launch(sum, dim=N, inputs=[a0, b0, c0], stream=stream0)
        finally:
            # finish recording on stream0
            g = wp.capture_end(stream=stream0)

        # replay
        num_iters = 10
        for _ in range(num_iters):
            wp.capture_launch(g, stream=stream0)

        # check results
        assert_np_equal(c0.numpy(), np.full(N, fill_value=2 * num_iters))

    @unittest.skipUnless(len(wp.get_cuda_devices()) > 1, "Requires at least two CUDA devices")
    def test_stream_scope_graph_mgpu(self):
        wp.load_module(device="cuda:0")
        wp.load_module(device="cuda:1")

        # resources on GPU 0
        with wp.ScopedDevice("cuda:0"):
            stream0 = wp.get_stream()
            a0 = wp.zeros(N, dtype=float)
            b0 = wp.empty(N, dtype=float)
            c0 = wp.empty(N, dtype=float)

        # resources on GPU 1
        with wp.ScopedDevice("cuda:1"):
            stream1 = wp.get_stream()
            a1 = wp.zeros(N, dtype=float)

        # capture graph
        with wp.ScopedDevice("cuda:0"):
            # start recording
            wp.capture_begin(force_module_load=False)
            try:
                with wp.ScopedDevice("cuda:1"):
                    # branch into stream1
                    wp.wait_stream(stream0)

                    wp.launch(inc, dim=N, inputs=[a1])

                wp.launch(inc, dim=N, inputs=[a0])

                # wait for stream1 to finish
                wp.wait_stream(stream1)

                # copy values from stream1
                wp.copy(b0, a1)

                # compute sum
                wp.launch(sum, dim=N, inputs=[a0, b0, c0])
            finally:
                # finish recording
                g = wp.capture_end()

        # replay
        with wp.ScopedDevice("cuda:0"):
            num_iters = 10
            for _ in range(num_iters):
                wp.capture_launch(g)

        # check results
        assert_np_equal(c0.numpy(), np.full(N, fill_value=2 * num_iters))


add_function_test(TestStreams, "test_stream_arg_implicit_sync", test_stream_arg_implicit_sync, devices=devices)
add_function_test(TestStreams, "test_stream_scope_implicit_sync", test_stream_scope_implicit_sync, devices=devices)

add_function_test(TestStreams, "test_stream_arg_synchronize", test_stream_arg_synchronize, devices=devices)
add_function_test(TestStreams, "test_stream_arg_wait_event", test_stream_arg_wait_event, devices=devices)
add_function_test(TestStreams, "test_stream_arg_wait_stream", test_stream_arg_wait_stream, devices=devices)
add_function_test(TestStreams, "test_stream_scope_synchronize", test_stream_scope_synchronize, devices=devices)
add_function_test(TestStreams, "test_stream_scope_wait_event", test_stream_scope_wait_event, devices=devices)
add_function_test(TestStreams, "test_stream_scope_wait_stream", test_stream_scope_wait_stream, devices=devices)


if __name__ == "__main__":
    wp.build.clear_kernel_cache()
    unittest.main(verbosity=2)
