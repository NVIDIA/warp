# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import unittest
import weakref
from typing import ClassVar

import numpy as np

import warp as wp


@wp.kernel
def increment_traced_array(array: wp.array[float]):
    i = wp.tid()
    array[i] += 1.0


class ObservedArray(wp.array):
    calls: ClassVar[list] = []

    @classmethod
    def __warp_function__(cls, func, types, args=(), kwargs=None):
        result = func(*args, **({} if kwargs is None else kwargs))
        cls.calls.append(func)
        return result


class DLPackSource:
    def __init__(self, array):
        self.array = array
        self.calls = []
        self.device_requests = 0
        self.capsule_requests = 0
        self.capsule = None

    @classmethod
    def __warp_function__(cls, func, types, args=(), kwargs=None):
        if func is not wp.from_dlpack:
            return NotImplemented
        source = args[0] if args else kwargs["source"]
        source.calls.append(func)
        return func(*args, **({} if kwargs is None else kwargs))

    def __dlpack_device__(self):
        self.device_requests += 1
        return self.array.__dlpack_device__()

    def __dlpack__(self, stream=None):
        self.capsule_requests += 1
        self.capsule = self.array.__dlpack__(stream=stream)
        return self.capsule


class TensorWrapper:
    calls: ClassVar[list] = []

    def __init__(self, tensor):
        self.tensor = tensor

    @classmethod
    def __warp_function__(cls, func, types, args=(), kwargs=None):
        if func is not wp.from_torch or any(t is not TensorWrapper for t in types):
            return NotImplemented
        cls.calls.append((func, types))
        adapted_args = tuple(value.tensor if isinstance(value, cls) else value for value in args)
        adapted_kwargs = {
            name: value.tensor if isinstance(value, cls) else value for name, value in (kwargs or {}).items()
        }
        return func(*adapted_args, **adapted_kwargs)


class TestFunctionDispatchInterop(unittest.TestCase):
    def setUp(self):
        self.addCleanup(
            setattr, wp.config, "enable_array_function_overrides", wp.config.enable_array_function_overrides
        )
        wp.config.enable_array_function_overrides = True
        ObservedArray.calls.clear()
        TensorWrapper.calls.clear()

    def import_torch(self):
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            self.skipTest("PyTorch is not installed")
        return torch

    def test_dlpack_delegation_consumes_once_and_retains_storage(self):
        array = wp.array([1.0, 2.0], dtype=wp.float32, device="cpu")
        source = DLPackSource(array)

        converted = wp.from_dlpack(source=source)

        self.assertEqual(source.calls, [wp.from_dlpack])
        self.assertEqual(source.device_requests, 1)
        self.assertEqual(source.capsule_requests, 1)
        self.assertEqual(converted.ptr, array.ptr)
        with self.assertRaisesRegex(TypeError, "valid DLPack capsule"):
            wp.from_dlpack(source.capsule)

        wp.copy(converted, wp.array([3.0, 4.0], dtype=wp.float32, device="cpu"))
        np.testing.assert_allclose(array.numpy(), [3.0, 4.0])

        owner = weakref.ref(array)
        del source, array
        gc.collect()
        self.assertIsNotNone(owner())
        np.testing.assert_allclose(converted.numpy(), [3.0, 4.0])

    def test_dlpack_export_delegation_preserves_subclass_storage(self):
        array = ObservedArray([1.0, 2.0], dtype=wp.float32, device="cpu")
        capsule = wp.to_dlpack(array)

        self.assertEqual(ObservedArray.calls, [wp.to_dlpack])
        converted = wp.from_dlpack(capsule)
        self.assertEqual(converted.ptr, array.ptr)
        wp.copy(converted, wp.array([5.0, 6.0], dtype=wp.float32, device="cpu"))
        np.testing.assert_allclose(array.numpy(), [5.0, 6.0])
        self.assertEqual(ObservedArray.calls, [wp.to_dlpack])

    def test_torch_import_wrapper_adapts_source_and_gradient(self):
        torch = self.import_torch()
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        gradient = torch.tensor([3.0, 4.0])

        array = wp.from_torch(TensorWrapper(tensor), grad=TensorWrapper(gradient), requires_grad=True)

        self.assertEqual(TensorWrapper.calls, [(wp.from_torch, (TensorWrapper,))])
        self.assertEqual(array.ptr, tensor.data_ptr())
        self.assertEqual(array.grad.ptr, gradient.data_ptr())
        self.assertTrue(array.requires_grad)
        wp.copy(array, wp.array([5.0, 6.0], dtype=wp.float32, device="cpu"))
        wp.copy(array.grad, wp.array([7.0, 8.0], dtype=wp.float32, device="cpu"))
        np.testing.assert_allclose(tensor.detach().numpy(), [5.0, 6.0])
        np.testing.assert_allclose(gradient.numpy(), [7.0, 8.0])

        owner = weakref.ref(tensor)
        gradient_owner = weakref.ref(gradient)
        del tensor, gradient
        gc.collect()
        self.assertIsNotNone(owner())
        self.assertIsNotNone(gradient_owner())
        np.testing.assert_allclose(array.numpy(), [5.0, 6.0])

    def test_torch_import_wrapper_preserves_descriptor_variant(self):
        torch = self.import_torch()
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        gradient = torch.tensor([3.0, 4.0])

        descriptor = wp.from_torch(
            t=TensorWrapper(tensor), grad=TensorWrapper(gradient), requires_grad=True, return_ctype=True
        )

        self.assertEqual(TensorWrapper.calls, [(wp.from_torch, (TensorWrapper,))])
        self.assertEqual(descriptor.data, tensor.data_ptr())
        self.assertEqual(descriptor.grad, gradient.data_ptr())
        self.assertEqual(descriptor.ndim, 1)
        self.assertEqual(descriptor.shape[0], 2)
        owner = weakref.ref(tensor)
        gradient_owner = weakref.ref(gradient)
        del tensor, gradient
        gc.collect()
        self.assertIsNotNone(owner())
        self.assertIsNotNone(gradient_owner())

    def test_torch_export_delegation_preserves_storage_and_gradient(self):
        self.import_torch()
        array = ObservedArray([1.0, 2.0], dtype=wp.float32, device="cpu", requires_grad=True)
        array.grad.assign([3.0, 4.0])

        tensor = wp.to_torch(array)

        self.assertEqual(ObservedArray.calls, [wp.to_torch])
        self.assertEqual(tensor.data_ptr(), array.ptr)
        self.assertEqual(tensor.grad.data_ptr(), array.grad.ptr)
        wp.copy(array, wp.array([5.0, 6.0], dtype=wp.float32, device="cpu"))
        wp.copy(array.grad, wp.array([7.0, 8.0], dtype=wp.float32, device="cpu"))
        np.testing.assert_allclose(tensor.detach().numpy(), [5.0, 6.0])
        np.testing.assert_allclose(tensor.grad.numpy(), [7.0, 8.0])
        owner = weakref.ref(array)
        gradient_owner = weakref.ref(array.grad)
        del array
        gc.collect()
        self.assertIsNotNone(owner())
        self.assertIsNotNone(gradient_owner())
        np.testing.assert_allclose(tensor.detach().numpy(), [5.0, 6.0])

    def test_traced_launch_and_torch_conversion_preserve_metadata_explicitly(self):
        self.import_torch()

        class TracedArray(ObservedArray):
            @classmethod
            def __warp_function__(cls, func, types, args=(), kwargs=None):
                result = super().__warp_function__(func, types, args, kwargs)
                if func is wp.to_torch:
                    wrapped = TensorWrapper(result)
                    wrapped.trace = args[0].trace
                    return wrapped
                return result

        array = TracedArray([1.0, 2.0], dtype=wp.float32, device="cpu")
        array.trace = object()

        # Input operands can be mutated even without an outputs sequence.
        wp.launch(increment_traced_array, dim=2, inputs=[array], device="cpu")
        wrapped = wp.to_torch(array)
        converted = wp.from_torch(wrapped)

        self.assertEqual(ObservedArray.calls, [wp.launch, wp.to_torch])
        self.assertIs(wrapped.trace, array.trace)
        self.assertEqual(converted.ptr, array.ptr)
        np.testing.assert_allclose(converted.numpy(), [2.0, 3.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
