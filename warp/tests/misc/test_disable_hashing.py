import os

os.environ["WARP_DISABLE_HASHING_PREFIX"] = "warp.tests.misc"

import unittest

import warp as wp
from warp.tests.misc.add_kernel import add_kernel
from warp.tests.unittest_utils import *

wp.config.cuda_output = "ptx"
wp.config.kernel_cache_dir = "warp/tests/misc"


class TestDisableHashing(unittest.TestCase):
    pass


def test_disable_hashing(test, device):
    a_np = np.random.randint(0, 10, size=(10,))
    b_np = np.random.randint(0, 10, size=(10,))
    a = wp.from_numpy(a_np, dtype=wp.int32, device=device)
    b = wp.from_numpy(b_np, dtype=wp.int32, device=device)
    true_res_np = a_np + b_np

    res = wp.zeros((10,), dtype=wp.int32, device=device)
    wp.launch(add_kernel, dim=(10,), inputs=[a, b, res], device=device)

    res_np = res.numpy()
    test.assertTrue(np.all(res_np == true_res_np))


devices = wp.get_devices()
add_function_test(TestDisableHashing, "test_disable_hashing", test_disable_hashing, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
