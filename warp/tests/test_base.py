# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest
import os
import sys

import numpy as np
import warp as wp

# redirects and captures all stdout output (including from C-libs)
class StdOutCapture:

    def begin(self):
        
        # save original
        self.saved = sys.stdout
        self.target = os.dup(self.saved.fileno())
        
        # create temporary capture stream
        import io, tempfile
        self.tempfile = io.TextIOWrapper(
                            tempfile.TemporaryFile(buffering=0),
                            encoding="utf-8",
                            errors="replace",
                            newline="",
                            write_through=True)

        os.dup2(self.tempfile.fileno(), self.saved.fileno())
        
        sys.stdout = self.tempfile

    def end(self):

        os.dup2(self.target, self.saved.fileno())
        os.close(self.target)
        
        self.tempfile.seek(0)
        res = self.tempfile.buffer.read()
        self.tempfile.close()

        sys.stdout = self.saved

        return str(res.decode("utf-8"))


class CheckOutput:

    def __init__(self, test):
        self.test = test

    def __enter__(self):
        wp.force_load()

        self.capture = StdOutCapture()
        self.capture.begin()


    def __exit__(self, exc_type, exc_value, traceback):
        
        # ensure any stdout output is flushed
        wp.synchronize()

        s = self.capture.end()
        if (s != ""):
            print(s)
            
        # fail if kernel produces any stdout (e.g.: from wp.expect_eq() builtins)
        self.test.assertEqual(s, "")


def assert_array_equal(result, expect):

    a = result.numpy()
    b = expect.numpy()

    if ((a == b).all() == False):
        raise AssertionError(f"Unexpected result, got: {a} expected: {b}")

def assert_np_equal(result, expect, tol=0.0):

    a = result.flatten()
    b = expect.flatten()

    if tol == 0.0:

        if ((a == b).all() == False):
            raise AssertionError(f"Unexpected result, got: {a} expected: {b}")

    else:

        delta = a-b
        err = np.max(np.abs(delta))
        if err > tol:
            raise AssertionError(f"Maximum expected error exceeds tolerance got: {a}, expected: {b}, with err: {err} > {tol}")


def create_test_func(func, device, **kwargs):

    # pass args to func
    def test_func(self):
        func(self, device, **kwargs)

    return test_func


def add_function_test(cls, name, func, devices=["cpu"], **kwargs):
    
    for device in devices:
        setattr(cls, name + "_" + device, create_test_func(func, device, **kwargs))


def add_kernel_test(cls, kernel, dim, name=None, expect=None, inputs=None, devices=["cpu"]):
    
    for device in devices:

        def test_func(self):

            args = []
            if (inputs):
                args.extend(inputs)

            if (expect):
                # allocate outputs to match results
                result = wp.array(expect, dtype=int, device=device)
                output = wp.zeros_like(result)

                args.append(output)

            # force load so that we don't generate any log output during launch
            kernel.module.load(device)

            capture = StdOutCapture()
            capture.begin()

            with CheckOutput(self):
                wp.launch(kernel, dim=dim, inputs=args, device=device)
            
            s = capture.end()

            # fail if kernel produces any stdout (e.g.: from wp.expect_eq() builtins)
            self.assertEqual(s, "")

            # check output values
            if expect:
                assert_array_equal(output, result)

        # register test func with class
        if (name == None):
            name = kernel.key

        setattr(cls, name + "_" + device, test_func)
