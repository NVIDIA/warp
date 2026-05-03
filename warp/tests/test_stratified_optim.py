import unittest

import numpy as np

import warp as wp
from warp.optim.stratified_projection import terminal_projection_kernel


class TestStratifiedOptim(unittest.TestCase):
    def test_terminal_projection_kernel(self):
        wp.init()
        # Test range including values that trigger the 1.5707963f clamp
        data = np.array([-200.0, -1.0, 0.0, 1.0, 200.0], dtype=np.float32)
        grads = wp.array(data, dtype=wp.float32)
        delta = 0.05

        wp.launch(kernel=terminal_projection_kernel, dim=len(data), inputs=[grads, wp.float32(delta)])
        wp.synchronize_device()

        result = grads.numpy()

        # Ensure all constants stay in float32 to match kernel precision
        delta_f32 = np.float32(delta)
        angles = np.clip((data * delta_f32) / np.float32(4.0), np.float32(-1.5707963), np.float32(1.5707963))
        expected = data * np.cos(angles).astype(np.float32)

        # Numerical admissibility check
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
        self.assertTrue(np.all(np.isfinite(result)))


def register(parent):
    # Only register the test class if CUDA is available
    if wp.is_cuda_available():
        parent.add_class(TestStratifiedOptim)


if __name__ == "__main__":
    wp.init()
    unittest.main()
