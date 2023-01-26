# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import unittest

import warp as wp
from warp.tests.test_base import *

wp.init()


@wp.kernel
def inc(a: wp.array(dtype=float)):

    tid = wp.tid()
    a[tid] = a[tid] + 1.0


def test_dlpack_warp_to_warp(test, device):

    a1 = wp.array(data=np.arange(10, dtype=np.float32), device=device)

    a2 = wp.from_dlpack(wp.to_dlpack(a1))

    test.assertEqual(a1.ptr, a2.ptr)
    test.assertEqual(a1.device, a2.device)
    test.assertEqual(a1.dtype, a2.dtype)
    test.assertEqual(a1.shape, a2.shape)
    test.assertEqual(a1.strides, a2.strides)

    assert_np_equal(a1.numpy(), a2.numpy())

    wp.launch(inc, dim=a2.size, inputs=[a2], device=device)

    assert_np_equal(a1.numpy(), a2.numpy())


def test_dlpack_warp_to_torch(test, device):

    import torch.utils.dlpack

    a = wp.array(data=np.arange(10, dtype=np.float32), device=device)

    t = torch.utils.dlpack.from_dlpack(wp.to_dlpack(a))

    item_size = wp.types.type_size_in_bytes(a.dtype)

    test.assertEqual(a.ptr, t.data_ptr())
    test.assertEqual(a.device, wp.device_from_torch(t.device))
    test.assertEqual(a.dtype, wp.torch.dtype_from_torch(t.dtype))
    test.assertEqual(a.shape, tuple(t.shape))
    test.assertEqual(a.strides, tuple(s * item_size for s in t.stride()))

    assert_np_equal(a.numpy(), t.cpu().numpy())

    wp.launch(inc, dim=a.size, inputs=[a], device=device)

    assert_np_equal(a.numpy(), t.cpu().numpy())

    t += 1

    assert_np_equal(a.numpy(), t.cpu().numpy())


def test_dlpack_torch_to_warp(test, device):

    import torch
    import torch.utils.dlpack

    t = torch.arange(10, dtype=torch.float32, device=wp.device_to_torch(device))

    a = wp.from_dlpack(torch.utils.dlpack.to_dlpack(t))

    item_size = wp.types.type_size_in_bytes(a.dtype)

    test.assertEqual(a.ptr, t.data_ptr())
    test.assertEqual(a.device, wp.device_from_torch(t.device))
    test.assertEqual(a.dtype, wp.torch.dtype_from_torch(t.dtype))
    test.assertEqual(a.shape, tuple(t.shape))
    test.assertEqual(a.strides, tuple(s * item_size for s in t.stride()))

    assert_np_equal(a.numpy(), t.cpu().numpy())

    wp.launch(inc, dim=a.size, inputs=[a], device=device)

    assert_np_equal(a.numpy(), t.cpu().numpy())

    t += 1

    assert_np_equal(a.numpy(), t.cpu().numpy())


def test_dlpack_warp_to_jax(test, device):

    import jax
    import jax.dlpack

    a = wp.array(data=np.arange(10, dtype=np.float32), device=device)

    # use generic dlpack conversion
    j1 = jax.dlpack.from_dlpack(wp.to_dlpack(a))

    # use jax wrapper
    j2 = wp.to_jax(a)

    test.assertEqual(a.ptr, j1.unsafe_buffer_pointer())
    test.assertEqual(a.ptr, j2.unsafe_buffer_pointer())
    test.assertEqual(a.device, wp.device_from_jax(j1.device()))
    test.assertEqual(a.device, wp.device_from_jax(j2.device()))
    test.assertEqual(a.shape, j1.shape)
    test.assertEqual(a.shape, j2.shape)

    assert_np_equal(a.numpy(), np.asarray(j1))
    assert_np_equal(a.numpy(), np.asarray(j2))

    wp.launch(inc, dim=a.size, inputs=[a], device=device)
    wp.synchronize_device(device)

    # HACK? Run a no-op operation so that Jax flags the arrays as dirty
    # and gets the latest values, which were modified by Warp.
    j1 += 0
    j2 += 0

    assert_np_equal(a.numpy(), np.asarray(j1))
    assert_np_equal(a.numpy(), np.asarray(j2))


def test_dlpack_jax_to_warp(test, device):

    import jax
    import jax.dlpack

    with jax.default_device(wp.device_to_jax(device)):

        j = jax.numpy.arange(10, dtype=jax.numpy.float32)

        # use generic dlpack conversion
        a1 = wp.from_dlpack(jax.dlpack.to_dlpack(j))

        # use jax wrapper
        a2 = wp.from_jax(j)

        test.assertEqual(a1.ptr, j.unsafe_buffer_pointer())
        test.assertEqual(a2.ptr, j.unsafe_buffer_pointer())
        test.assertEqual(a1.device, wp.device_from_jax(j.device()))
        test.assertEqual(a2.device, wp.device_from_jax(j.device()))
        test.assertEqual(a1.shape, j.shape)
        test.assertEqual(a2.shape, j.shape)

        assert_np_equal(a1.numpy(), np.asarray(j))
        assert_np_equal(a2.numpy(), np.asarray(j))

        wp.launch(inc, dim=a1.size, inputs=[a1], device=device)
        wp.synchronize_device(device)

        # HACK? Run a no-op operation so that Jax flags the array as dirty
        # and gets the latest values, which were modified by Warp.
        j += 0

        assert_np_equal(a1.numpy(), np.asarray(j))
        assert_np_equal(a2.numpy(), np.asarray(j))


def register(parent):

    class TestDLPack(parent):
        pass

    devices = wp.get_devices()

    add_function_test(TestDLPack, "test_dlpack_warp_to_warp", test_dlpack_warp_to_warp, devices=devices)

    # torch interop via dlpack
    try:
        import torch
        import torch.utils.dlpack

        add_function_test(TestDLPack, "test_dlpack_warp_to_torch", test_dlpack_warp_to_torch, devices=devices)
        add_function_test(TestDLPack, "test_dlpack_torch_to_warp", test_dlpack_torch_to_warp, devices=devices)

    except ImportError:
        pass

    # jax interop via dlpack
    try:
        import jax
        import jax.dlpack

        add_function_test(TestDLPack, "test_dlpack_warp_to_jax", test_dlpack_warp_to_jax, devices=devices)
        add_function_test(TestDLPack, "test_dlpack_jax_to_warp", test_dlpack_jax_to_warp, devices=devices)

    except ImportError:
        pass

    return TestDLPack


if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
