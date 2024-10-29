# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import unittest
from typing import Any

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def test_print_kernel():
    wp.print(1.0)
    wp.print("this is a string")
    wp.printf("this is a float %f\n", 457.5)
    wp.printf("this is an int %d\n", 123)


@wp.kernel
def test_print_numeric_kernel(value: int):
    # signed ints
    wp.print(wp.int8(value))
    wp.print(wp.int16(value))
    wp.print(wp.int32(value))
    wp.print(wp.int64(value))
    # unsigned ints
    wp.print(wp.uint8(value))
    wp.print(wp.uint16(value))
    wp.print(wp.uint32(value))
    wp.print(wp.uint64(value))
    # floats
    wp.print(wp.float16(value))
    wp.print(wp.float32(value))
    wp.print(wp.float64(value))


@wp.kernel
def test_print_boolean_kernel(value: wp.bool):
    wp.print(value)
    wp.print(not value)


def test_print(test, device):
    wp.load_module(device=device)
    capture = StdOutCapture()
    capture.begin()
    wp.launch(kernel=test_print_kernel, dim=1, inputs=[], device=device)
    wp.synchronize_device(device)
    s = capture.end()

    # We skip the win32 comparison for now since the capture sometimes is an empty string
    if sys.platform != "win32":
        test.assertRegex(
            s,
            rf"1{os.linesep}"
            rf"this is a string{os.linesep}"
            rf"this is a float 457\.500000{os.linesep}"
            rf"this is an int 123",
        )


def test_print_numeric(test, device):
    wp.load_module(device=device)

    capture = StdOutCapture()
    capture.begin()
    wp.launch(kernel=test_print_numeric_kernel, dim=1, inputs=[17], device=device)
    wp.synchronize_device(device)
    s = capture.end()

    # We skip the win32 comparison for now since the capture sometimes is an empty string
    if sys.platform != "win32":
        test.assertRegex(
            s,
            rf"17{os.linesep}"
            rf"17{os.linesep}"
            rf"17{os.linesep}"
            rf"17{os.linesep}"
            rf"17{os.linesep}"
            rf"17{os.linesep}"
            rf"17{os.linesep}"
            rf"17{os.linesep}"
            rf"17{os.linesep}"
            rf"17{os.linesep}"
            rf"17{os.linesep}",
        )

    capture = StdOutCapture()
    capture.begin()
    wp.launch(kernel=test_print_numeric_kernel, dim=1, inputs=[-1], device=device)
    wp.synchronize_device(device)
    s = capture.end()

    # We skip the win32 comparison for now since the capture sometimes is an empty string
    if sys.platform != "win32":
        test.assertRegex(
            s,
            rf"-1{os.linesep}"
            rf"-1{os.linesep}"
            rf"-1{os.linesep}"
            rf"-1{os.linesep}"
            rf"255{os.linesep}"
            rf"65535{os.linesep}"
            rf"4294967295{os.linesep}"
            rf"18446744073709551615{os.linesep}"
            rf"-1{os.linesep}"
            rf"-1{os.linesep}"
            rf"-1{os.linesep}",
        )


def test_print_boolean(test, device):
    wp.load_module(device=device)
    capture = StdOutCapture()
    capture.begin()
    wp.launch(kernel=test_print_boolean_kernel, dim=1, inputs=[True], device=device)
    wp.synchronize_device(device)
    s = capture.end()

    # We skip the win32 comparison for now since the capture sometimes is an empty string
    if sys.platform != "win32":
        test.assertRegex(s, rf"True{os.linesep}False{os.linesep}")


@wp.kernel
def generic_print_kernel(x: Any):
    print(x)


@wp.struct
class SimpleStruct:
    x: float
    y: float


generic_print_types = [*wp.types.scalar_types]
for scalar_type in wp.types.scalar_types:
    generic_print_types.append(wp.types.vector(2, scalar_type))
    generic_print_types.append(wp.types.vector(3, scalar_type))
    generic_print_types.append(wp.types.vector(4, scalar_type))
    generic_print_types.append(wp.types.matrix((2, 2), scalar_type))
    generic_print_types.append(wp.types.matrix((3, 3), scalar_type))
    generic_print_types.append(wp.types.matrix((4, 4), scalar_type))
generic_print_types.append(wp.bool)
generic_print_types.append(SimpleStruct)
generic_print_types.append(wp.array(dtype=float))

for T in generic_print_types:
    wp.overload(generic_print_kernel, [T])


def test_print_adjoint(test, device):
    for scalar_type in wp.types.scalar_types:
        # scalar
        capture = StdOutCapture()
        capture.begin()
        wp.launch(
            generic_print_kernel,
            dim=1,
            inputs=[scalar_type(17)],
            adj_inputs=[scalar_type(42)],
            adjoint=True,
            device=device,
        )
        wp.synchronize_device(device)
        s = capture.end()

        # We skip the win32 comparison for now since the capture sometimes is an empty string
        if sys.platform != "win32":
            test.assertRegex(s, rf"17{os.linesep}adj: 42{os.linesep}")

        for dim in (2, 3, 4):
            # vector
            vec_type = wp.types.vector(dim, scalar_type)
            vec_data = np.arange(vec_type._length_, dtype=wp.dtype_to_numpy(scalar_type))
            v = vec_type(vec_data)
            adj_v = vec_type(vec_data[::-1])

            capture = StdOutCapture()
            capture.begin()
            wp.launch(generic_print_kernel, dim=1, inputs=[v], adj_inputs=[adj_v], adjoint=True, device=device)
            wp.synchronize_device(device)
            s = capture.end()

            # We skip the win32 comparison for now since the capture sometimes is an empty string
            if sys.platform != "win32":
                expected_forward = " ".join(str(int(x)) for x in v) + " "
                expected_adjoint = " ".join(str(int(x)) for x in adj_v)
                test.assertRegex(s, rf"{expected_forward}{os.linesep}adj: {expected_adjoint}{os.linesep}")

            # matrix
            mat_type = wp.types.matrix((dim, dim), scalar_type)
            mat_data = np.arange(mat_type._length_, dtype=wp.dtype_to_numpy(scalar_type))
            m = mat_type(mat_data)
            adj_m = mat_type(mat_data[::-1])

            capture = StdOutCapture()
            capture.begin()
            wp.launch(generic_print_kernel, dim=1, inputs=[m], adj_inputs=[adj_m], adjoint=True, device=device)
            wp.synchronize_device(device)
            s = capture.end()

            # We skip the win32 comparison for now since the capture sometimes is an empty string
            if sys.platform != "win32":
                expected_forward = ""
                expected_adjoint = ""
                for row in range(dim):
                    if row == 0:
                        adj_prefix = "adj: "
                    else:
                        adj_prefix = "     "
                    expected_forward += " ".join(str(int(x)) for x in m[row]) + f" {os.linesep}"
                    expected_adjoint += adj_prefix + " ".join(str(int(x)) for x in adj_m[row]) + f"{os.linesep}"
                test.assertRegex(s, rf"{expected_forward}{expected_adjoint}")

    # Booleans
    capture = StdOutCapture()
    capture.begin()
    wp.launch(generic_print_kernel, dim=1, inputs=[True], adj_inputs=[False], adjoint=True, device=device)
    wp.synchronize_device(device)
    s = capture.end()

    # We skip the win32 comparison for now since the capture sometimes is an empty string
    if sys.platform != "win32":
        test.assertRegex(s, rf"True{os.linesep}adj: False{os.linesep}")

    # structs, not printable yet
    capture = StdOutCapture()
    capture.begin()
    wp.launch(
        generic_print_kernel, dim=1, inputs=[SimpleStruct()], adj_inputs=[SimpleStruct()], adjoint=True, device=device
    )
    wp.synchronize_device(device)
    s = capture.end()

    # We skip the win32 comparison for now since the capture sometimes is an empty string
    if sys.platform != "win32":
        test.assertRegex(
            s, rf"<type without print implementation>{os.linesep}adj: <type without print implementation>{os.linesep}"
        )

    # arrays, not printable
    capture = StdOutCapture()
    capture.begin()
    a = wp.ones(10, dtype=float, device=device)
    adj_a = wp.zeros(10, dtype=float, device=device)
    wp.launch(generic_print_kernel, dim=1, inputs=[a], adj_inputs=[adj_a], adjoint=True, device=device)
    wp.synchronize_device(device)
    s = capture.end()

    # We skip the win32 comparison for now since the capture sometimes is an empty string
    if sys.platform != "win32":
        test.assertRegex(
            s, rf"<type without print implementation>{os.linesep}adj: <type without print implementation>{os.linesep}"
        )


class TestPrint(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(TestPrint, "test_print", test_print, devices=devices, check_output=False)
add_function_test(TestPrint, "test_print_numeric", test_print_numeric, devices=devices, check_output=False)
add_function_test(TestPrint, "test_print_boolean", test_print_boolean, devices=devices, check_output=False)
add_function_test(TestPrint, "test_print_adjoint", test_print_adjoint, devices=devices, check_output=False)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
