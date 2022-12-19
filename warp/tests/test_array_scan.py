import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
import numpy as np

wp.config.mode = "release"
wp.config.verbose = True
wp.config.verify_cuda = True

wp.init()

n = 100000
num_runs = 16

def test_for_type(dtype, device):
    dtype_str = dtype.__name__
    if dtype == int:
        values = np.random.randint(-1e6, 1e6, n, dtype=dtype)
    else:
        values = np.random.uniform(-1, 1, n)
    
    results_ref = np.cumsum(values)

    in_values = wp.array(values, dtype=dtype, device=device)
    out_values_inc = wp.zeros(len(values), dtype=dtype, device=device)
    out_values_exc = wp.zeros(len(values), dtype=dtype, device=device)

    wp.utils.array_scan(in_values, out_values_inc, True)
    wp.utils.array_scan(in_values, out_values_exc, False)

    tolerance = 0 if dtype == int else 1e-3

    results_inc = out_values_inc.numpy().squeeze()
    results_exc = out_values_exc.numpy().squeeze()
    error_inc = np.max(np.abs(results_inc - results_ref)) / abs(results_ref[-1])
    error_exc = max(np.max(np.abs(results_exc[1:] - results_ref[:-1])), abs(results_exc[0])) / abs(results_ref[-2])
    if error_inc > tolerance:
       print(f"FAIL! Max error in inclusive scan for {dtype_str}: {error_inc}")
    else:
       print(f"PASS! Max error in inclusive scan for {dtype_str}: {error_inc}")

    if error_exc > tolerance:
        print(f"FAIL! Max error in exclusive scan for {dtype_str}: {error_exc}")
    # else:
    #     print(f"PASS! Max error in exclusive scan for {dtype_str}: {error_exc}")
    

np.random.seed(1008)
for device in ("cuda", "cpu"):
    print(f"\n\nTesting {device}")
    for i in range(num_runs):

        print(f"Run: {i+1}")
        print("---------")

        test_for_type(int, device)
        test_for_type(float, device)
