# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import unittest
from typing import Any

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def test_print_kernel():
    wp.print(1.0)
    wp.print("this is a string")
    wp.printf("this is another string\n")
    wp.printf("this is a float %f\n", 457.5)
    wp.printf("this is an int %d\n", 123)
    # fmt: off
    wp.printf(
        "0=%d, 1=%d, 2=%d, 3=%d, 4=%d, 5=%d, 6=%d, 7=%d, "
        "8=%d, 9=%d, 10=%d, 11=%d, 12=%d, 13=%d, 14=%d, 15=%d, "
        "16=%d, 17=%d, 18=%d, 19=%d, 20=%d, 21=%d, 22=%d, 23=%d, "
        "24=%d, 25=%d, 26=%d, 27=%d, 28=%d, 29=%d, 30=%d, 31=%d"
        "\n",
         0,  1,  2,  3,  4,  5,  6,  7,
         8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31,
    )
    # fmt: on


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
            rf"this is another string{os.linesep}"
            rf"this is a float 457\.500000{os.linesep}"
            rf"this is an int 123{os.linesep}"
            rf"0=0, 1=1, 2=2, 3=3, 4=4, 5=5, 6=6, 7=7, "
            rf"8=8, 9=9, 10=10, 11=11, 12=12, 13=13, 14=14, 15=15, "
            rf"16=16, 17=17, 18=18, 19=19, 20=20, 21=21, 22=22, 23=23, "
            rf"24=24, 25=25, 26=26, 27=27, 28=28, 29=29, 30=30, 31=31{os.linesep}",
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


def test_print_error_variadic_arg_count(test, device):
    @wp.kernel
    def kernel():
        # fmt: off
        wp.printf(
            "0=%d, 1=%d, 2=%d, 3=%d, 4=%d, 5=%d, 6=%d, 7=%d, "
            "8=%d, 9=%d, 10=%d, 11=%d, 12=%d, 13=%d, 14=%d, 15=%d, "
            "16=%d, 17=%d, 18=%d, 19=%d, 20=%d, 21=%d, 22=%d, 23=%d, "
            "24=%d, 25=%d, 26=%d, 27=%d, 28=%d, 29=%d, 30=%d, 31=%d, "
            "32=%d\n",
            0,  1,  2,  3,  4,  5,  6,  7,
            8,  9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
            32,
        )
        # fmt: on

    with test.assertRaisesRegex(
        RuntimeError,
        r"the maximum number of variadic arguments that can be passed to `printf` is 32$",
    ):
        wp.launch(
            kernel,
            dim=1,
            device=device,
        )


class TestPrint(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(TestPrint, "test_print", test_print, devices=devices, check_output=False)
add_function_test(TestPrint, "test_print_numeric", test_print_numeric, devices=devices, check_output=False)
add_function_test(TestPrint, "test_print_boolean", test_print_boolean, devices=devices, check_output=False)
add_function_test(TestPrint, "test_print_adjoint", test_print_adjoint, devices=devices, check_output=False)
add_function_test(
    TestPrint,
    "test_print_error_variadic_arg_count",
    test_print_error_variadic_arg_count,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
