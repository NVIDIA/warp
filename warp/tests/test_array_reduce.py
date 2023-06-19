import numpy as np
import warp as wp

from warp.utils import array_sum, array_inner
from warp.tests.test_base import *

wp.init()


def make_test_array_sum(dtype):
    N = 1000

    def test_array_sum(test, device):
        cols = wp.types.type_length(dtype)

        values_np = np.random.rand(N, cols)
        values = wp.array(values_np, device=device, dtype=dtype)

        vsum = array_sum(values)
        ref_vsum = values_np.sum(axis=0)

        assert_np_equal(vsum / N, ref_vsum / N, 0.0001)

    return test_array_sum


def make_test_array_sum_axis(dtype):
    I = 5
    J = 10
    K = 2

    N = I * J * K

    def test_array_sum(test, device):
        values_np = np.random.rand(I, J, K)
        values = wp.array(values_np, shape=(I, J, K), device=device, dtype=dtype)

        for axis in range(3):
            vsum = array_sum(values, axis=axis)
            ref_vsum = values_np.sum(axis=axis)

            assert_np_equal(vsum.numpy() / N, ref_vsum / N, 0.0001)

    return test_array_sum


def make_test_array_inner(dtype):
    N = 1000

    def test_array_inner(test, device):
        cols = wp.types.type_length(dtype)

        a_np = np.random.rand(N, cols)
        b_np = np.random.rand(N, cols)

        a = wp.array(a_np, device=device, dtype=dtype)
        b = wp.array(b_np, device=device, dtype=dtype)

        ab = array_inner(a, b)
        ref_ab = np.dot(a_np.flatten(), b_np.flatten())

        test.assertAlmostEqual(ab / N, ref_ab / N, places=5)

    return test_array_inner


def make_test_array_inner_axis(dtype):
    I = 5
    J = 10
    K = 2

    N = I * J * K

    def test_array_inner(test, device):
        a_np = np.random.rand(I, J, K)
        b_np = np.random.rand(I, J, K)

        a = wp.array(a_np, shape=(I, J, K), device=device, dtype=dtype)
        b = wp.array(b_np, shape=(I, J, K), device=device, dtype=dtype)

        ab = array_inner(a, b, axis=0)
        ref_ab = np.einsum(a_np, [0, 1, 2], b_np, [0, 1, 2], [1, 2])
        assert_np_equal(ab.numpy() / N, ref_ab / N, 0.0001)

        ab = array_inner(a, b, axis=1)
        ref_ab = np.einsum(a_np, [0, 1, 2], b_np, [0, 1, 2], [0, 2])
        assert_np_equal(ab.numpy() / N, ref_ab / N, 0.0001)

        ab = array_inner(a, b, axis=2)
        ref_ab = np.einsum(a_np, [0, 1, 2], b_np, [0, 1, 2], [0, 1])
        assert_np_equal(ab.numpy() / N, ref_ab / N, 0.0001)

    return test_array_inner


def register(parent):
    devices = get_test_devices()

    class TestArraySym(parent):
        pass

    add_function_test(TestArraySym, "test_array_sum_double", make_test_array_sum(wp.float64), devices=devices)
    add_function_test(TestArraySym, "test_array_sum_vec3", make_test_array_sum(wp.vec3), devices=devices)
    add_function_test(TestArraySym, "test_array_sum_axis_float", make_test_array_sum_axis(wp.float32), devices=devices)
    add_function_test(TestArraySym, "test_array_inner_double", make_test_array_inner(wp.float64), devices=devices)
    add_function_test(TestArraySym, "test_array_inner_vec3", make_test_array_inner(wp.vec3), devices=devices)
    add_function_test(
        TestArraySym, "test_array_inner_axis_float", make_test_array_inner_axis(wp.float32), devices=devices
    )

    return TestArraySym


if __name__ == "__main__":
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
