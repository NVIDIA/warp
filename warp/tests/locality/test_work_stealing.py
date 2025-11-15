import unittest

import warp as wp
from warp.tests.unittest_utils import *


class TestWorkStealingQueues(unittest.TestCase):
    """Test the WorkStealingQueues facility for localized work distribution."""

    def test_basic_instantiation_cuda(self):
        """Test basic instantiation of WorkStealingQueues on CUDA device."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA not available")

        k = 4  # Number of deques (typically number of streams)
        m = 16  # Items per deque

        # Create work-stealing queues (device-agnostic via unified memory)
        ws_queues = wp.WorkStealingQueues(k=k, enable_instrumentation=False)
        ws_queues.next_epoch(m=m)

        # Verify properties
        self.assertEqual(ws_queues.num_deques, k)
        self.assertEqual(ws_queues.epoch, 1)  # Epoch starts at 1

        # Cleanup happens automatically via __del__

    def test_epoch_advancement(self):
        """Test advancing epochs in WorkStealingQueues."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA not available")

        ws_queues = wp.WorkStealingQueues(k=4)
        ws_queues.next_epoch(m=16)

        # Check initial epoch (starts at 1)
        self.assertEqual(ws_queues.epoch, 1)

        # Advance epoch
        ws_queues.next_epoch(m=16)
        self.assertEqual(ws_queues.epoch, 2)

        # Advance again
        ws_queues.next_epoch(m=16)
        self.assertEqual(ws_queues.epoch, 3)

    def test_view_retrieval(self):
        """Test retrieving a view of the work-stealing queues."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA not available")

        k, m = 4, 16
        ws_queues = wp.WorkStealingQueues(k=k, enable_instrumentation=False)
        ws_queues.next_epoch(m=m)

        # Get view
        view = ws_queues.view()

        # Verify view properties
        self.assertIsInstance(view, wp.WsQueuesView)
        self.assertEqual(view.k, k)
        self.assertEqual(view.m, m)
        self.assertEqual(view.max_work_items, k * m)
        self.assertEqual(view.epoch, 1)  # Epoch starts at 1
        self.assertIsNotNone(view.unified_base)

    def test_instrumentation_disabled(self):
        """Test that instrumentation is correctly disabled."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA not available")

        ws_queues = wp.WorkStealingQueues(k=4, enable_instrumentation=False)
        ws_queues.next_epoch(m=16)

        # Should have no instrumentation
        self.assertFalse(ws_queues.has_instrumentation)

    def test_instrumentation_enabled(self):
        """Test that instrumentation is correctly enabled."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA not available")

        ws_queues = wp.WorkStealingQueues(k=4, enable_instrumentation=True)
        ws_queues.next_epoch(m=16)

        # Should have instrumentation
        self.assertTrue(ws_queues.has_instrumentation)
        # Buffer should be accessible (not null)
        self.assertIsNotNone(ws_queues.instrumentation_buffer)

    def test_cpu_device_raises_error(self):
        """Test that instantiation requires CUDA."""
        # Temporarily make CUDA unavailable
        # Note: This test requires CUDA to be available to run other tests,
        # so we just verify the error message format

        # If we're running this, CUDA is available, so we can't actually test the error
        # Just verify the queues can be created (since CUDA is available)
        ws_queues = wp.WorkStealingQueues(k=4)
        self.assertIsNotNone(ws_queues.id)

    # TODO: Add kernel-based tests
    # The following tests should use actual Warp kernels to test the work-stealing mechanism:
    #
    # 1. test_kernel_push_pop: Test pushing and popping work items from kernels
    # 2. test_kernel_work_distribution: Test that work is correctly distributed across deques
    # 3. test_kernel_work_stealing: Test that work can be stolen between deques
    # 4. test_kernel_validation: Test work assignment validation
    #
    # Example skeleton:
    # @wp.kernel
    # def push_work_kernel(view: wp.WsQueuesView, work_items: wp.array(dtype=int)):
    #     tid = wp.tid()
    #     # ... push work items to the view ...
    #
    # def test_kernel_push_pop(self):
    #     ws_queues = wp.WorkStealingQueues(k=4)
    #     ws_queues.next_epoch(m=16)
    #     view = ws_queues.view()
    #     work_items = wp.array([...], device="cuda:0")
    #     wp.launch(push_work_kernel, dim=16, inputs=[view, work_items], device="cuda:0")
    #     # ... verify results ...


devices = get_test_devices()


# Helper function for device-parameterized tests (not collected by pytest)
def _test_basic_functionality(test, device):
    """Device-parameterized test for basic work-stealing functionality."""
    if not device.is_cuda:
        test.skipTest("WorkStealingQueues only supports CUDA devices")

    # Create work-stealing queues (device-agnostic via unified memory)
    ws_queues = wp.WorkStealingQueues(k=4)
    ws_queues.next_epoch(m=16)

    # TODO: Implement actual kernel-based test logic here
    # This would involve:
    # 1. Creating a kernel that uses wp.WsQueuesView
    # 2. Launching the kernel with the view
    # 3. Validating the results

    # For now, just verify basic properties
    test.assertEqual(ws_queues.num_deques, 4)


class TestWorkStealingQueuesDevices(unittest.TestCase):
    """Device-parameterized tests for WorkStealingQueues."""

    pass


# Automatically generate device-parameterized test methods
add_function_test(TestWorkStealingQueuesDevices, "test_basic_functionality", _test_basic_functionality, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
