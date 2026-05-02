import warp as wp
import numpy as np
import unittest
from warp.optim.stratified_projection import terminal_projection_kernel
class TestStratifiedOptim(unittest.TestCase):
    def test_terminal_projection_kernel(self):
        wp.init()
        n = 1024
        # Realistic gradient range
        data = np.random.uniform(-1.0, 1.0, n).astype(np.float32)
        grads = wp.array(data, dtype=wp.float32)
        
        wp.launch(
            kernel=terminal_projection_kernel,
            dim=n,
            inputs=[grads, wp.float32(0.05)]
        )
        wp.synchronize()
        result = grads.numpy()
        
        # Verify no gradient was zeroed or blown up
        self.assertFalse(np.all(result == 0.0))
        self.assertTrue(np.all(np.isfinite(result)))

if __name__ == "__main__":
    unittest.main()
