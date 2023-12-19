import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

wp.init()


def test_basic(test, device):
    snippet = """
    out[tid] = a * x[tid] + y[tid];
    """
    adj_snippet = """
    adj_a = x[tid] * adj_out[tid];
    adj_x[tid] = a * adj_out[tid];
    adj_y[tid] = adj_out[tid];
    """

    @wp.func_native(snippet, adj_snippet)
    def saxpy(
        a: wp.float32,
        x: wp.array(dtype=wp.float32),
        y: wp.array(dtype=wp.float32),
        out: wp.array(dtype=wp.float32),
        tid: int,
    ):
        ...

    @wp.kernel
    def saxpy_cu(
        a: wp.float32, x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)
    ):
        tid = wp.tid()
        saxpy(a, x, y, out, tid)

    @wp.kernel
    def saxpy_py(
        a: wp.float32, x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)
    ):
        tid = wp.tid()
        out[tid] = a * x[tid] + y[tid]

    N = 128

    a1 = 2.0
    x1 = wp.array(np.arange(N, dtype=np.float32), dtype=wp.float32, device=device, requires_grad=True)
    y1 = wp.zeros_like(x1)
    out1 = wp.array(np.arange(N, dtype=np.float32), dtype=wp.float32, device=device)
    adj_out1 = wp.array(np.ones(N, dtype=np.float32), dtype=wp.float32, device=device)

    a2 = 2.0
    x2 = wp.array(np.arange(N, dtype=np.float32), dtype=wp.float32, device=device, requires_grad=True)
    y2 = wp.zeros_like(x2)
    out2 = wp.array(np.arange(N, dtype=np.float32), dtype=wp.float32, device=device)
    adj_out2 = wp.array(np.ones(N, dtype=np.float32), dtype=wp.float32, device=device)

    tape = wp.Tape()

    with tape:
        wp.launch(kernel=saxpy_cu, dim=N, inputs=[a1, x1, y1], outputs=[out1], device=device)
        wp.launch(kernel=saxpy_py, dim=N, inputs=[a2, x2, y2], outputs=[out2], device=device)

    tape.backward(grads={out1: adj_out1, out2: adj_out2})

    # test forward snippet
    assert_np_equal(out1.numpy(), out2.numpy())

    # test backward snippet
    assert_np_equal(x1.grad.numpy(), a1 * np.ones(N, dtype=np.float32))
    assert_np_equal(x1.grad.numpy(), x2.grad.numpy())

    assert_np_equal(y1.grad.numpy(), np.ones(N, dtype=np.float32))
    assert_np_equal(y1.grad.numpy(), y2.grad.numpy())


def test_shared_memory(test, device):
    snippet = """
        __shared__ int s[128];

        s[tid] = d[tid];
        __syncthreads();
        d[tid] = s[N - tid - 1];
        """

    @wp.func_native(snippet)
    def reverse(d: wp.array(dtype=int), N: int, tid: int):
        return

    @wp.kernel
    def reverse_kernel(d: wp.array(dtype=int), N: int):
        tid = wp.tid()
        reverse(d, N, tid)

    N = 128
    x = wp.array(np.arange(N, dtype=int), dtype=int, device=device)
    y = np.arange(127, -1, -1, dtype=int)

    wp.launch(kernel=reverse_kernel, dim=N, inputs=[x, N], device=device)

    assert_np_equal(x.numpy(), y)


def test_cpu_snippet(test, device):
    snippet = """
    int inc = 1;
    out[tid] = x[tid] + inc;
    """

    @wp.func_native(snippet)
    def increment_snippet(
        x: wp.array(dtype=wp.int32),
        out: wp.array(dtype=wp.int32),
        tid: int,
    ):
        ...

    @wp.kernel
    def increment(x: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
        tid = wp.tid()
        increment_snippet(x, out, tid)

    N = 128
    x = wp.array(np.arange(N, dtype=np.int32), dtype=wp.int32, device=device)
    out = wp.zeros(N, dtype=wp.int32, device=device)

    wp.launch(kernel=increment, dim=N, inputs=[x], outputs=[out], device=device)

    assert_np_equal(out.numpy(), np.arange(1, N + 1, 1, dtype=np.int32))


class TestSnippets(unittest.TestCase):
    pass


add_function_test(TestSnippets, "test_basic", test_basic, devices=get_unique_cuda_test_devices())
add_function_test(TestSnippets, "test_shared_memory", test_shared_memory, devices=get_unique_cuda_test_devices())
add_function_test(TestSnippets, "test_cpu_snippet", test_cpu_snippet, devices=["cpu"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
