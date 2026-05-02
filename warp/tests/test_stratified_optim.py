import warp as wp
import numpy as np
import unittest

class TestStratifiedOptim(unittest.TestCase):
    def test_stratified_accumulator(self):
        wp.init()
        
        # We must define the kernel AND register it with Warp
        source = """
#include "stratified_math.h"

def verify_stratified(values: wp.array(dtype=float), res: wp.array(dtype=float)):
    acc = wp.StratifiedAccumulator()
    for i in range(len(values)):
        acc.add(values[i])
    res[0] = acc.get()
"""
        # This line is the missing link—it compiles the source string
        module = wp.get_module(source)
        verify_stratified = module.verify_stratified

        # Create test data
        acc_test = wp.array([1.0e-7] * 10, dtype=wp.float32)
        result = wp.zeros(1, dtype=wp.float32)
        
        # Actually launch the kernel
        wp.launch(verify_stratified, dim=1, inputs=[acc_test, result])
        
        # Synchronize to ensure the GPU/CPU finished the math
        wp.synchronize()
        
        expected = 1.0e-6
        # We use a slightly wider tolerance for float32 precision
        self.assertAlmostEqual(float(result.numpy()[0]), expected, places=6)

if __name__ == "__main__":
    unittest.main()
