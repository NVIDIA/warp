import numpy as np
import warp as wp

from warp.utils import runlength_encode
from warp.tests.test_base import *

wp.init()


def test_runlength_encode_int(test, device):
    n = 1000

    values_np = np.sort(np.random.randint(-10, 10, n, dtype=int))

    unique_values_np, unique_counts_np = np.unique(values_np, return_counts=True)

    values = wp.array(values_np, device=device, dtype=int)

    unique_values = wp.empty_like(values)
    unique_counts = wp.empty_like(values)

    run_count = runlength_encode(values, unique_values, unique_counts)

    assert run_count == len(unique_values_np)
    assert (unique_values.numpy()[:run_count] == unique_values_np[:run_count]).all()
    assert (unique_counts.numpy()[:run_count] == unique_counts_np[:run_count]).all()


def register(parent):
    devices = get_test_devices()

    class TestRunlengthEncode(parent):
        pass

    add_function_test(TestRunlengthEncode, "test_runlength_encode_int", test_runlength_encode_int, devices=devices)

    return TestRunlengthEncode


if __name__ == "__main__":
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
