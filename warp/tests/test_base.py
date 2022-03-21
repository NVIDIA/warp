# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest
import os
import sys

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


def assert_array_equal(result, expect):

    a = result.numpy()
    b = expect.numpy()

    if ((a == b).all() == False):
        raise AssertionError(f"Unexpected result, got: {a} expected: {b}")

def assert_np_equal(result, expect):

    a = result
    b = expect

    if ((a == b).all() == False):
        raise AssertionError(f"Unexpected result, got: {a} expected: {b}")



def add_function_test(cls, name, func, devices=["cpu"], **kwargs):
    
    for device in devices:

        # pass args to func
        def test_func(self):
            func(self, device, **kwargs)
        
        setattr(cls, name + "_" + device, test_func)


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
            if (kernel.module.loaded == False):
                kernel.module.load()

            capture = StdOutCapture()
            capture.begin()

            wp.launch(kernel, dim=dim, inputs=args, device=device)
            wp.synchronize()

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
