import unittest

import numpy as np

import warp as wp
from warp.optim.stratified_projection import terminal_projection_kernel


class TestStratifiedOptim(unittest.TestCase):
    def test_terminal_projection_kernel(self):
        # Test range including small values and values > 125.7 to test the clamp
        data = np.array([-200.0, -1.0, 0.0, 1.0, 200.0], dtype=np.float32)
        grads = wp.array(data, dtype=wp.float32)
        delta = 0.05

        wp.launch(kernel=terminal_projection_kernel, dim=len(data), inputs=[grads, wp.float32(delta)])
        wp.synchronize_device()

        result = grads.numpy()

        # Manually compute expected with pi/2 clamp (1.5707963)
        angles = np.clip((data * delta) / 4.0, -1.5707963, 1.5707963)
        expected = data * np.cos(angles)

        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
        self.assertTrue(np.all(np.isfinite(result)))


def register(parent):
    conf = wp.get_config()
    if conf.cuda_enabled:
        parent.add_class(TestStratifiedOptim, name="TestStratifiedOptim")


if __name__ == "__main__":
    wp.init()
    unittest.main()
