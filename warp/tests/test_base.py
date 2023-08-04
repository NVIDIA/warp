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

# default test mode (see get_test_devices())
#   "basic" - only run on CPU and first GPU device
#   "unique" - run on CPU and all unique GPU arches
#   "all" - run on all devices
test_mode = "basic"


def get_test_devices(mode=None):
    if mode is None:
        global test_mode
        mode = test_mode

    devices = []

    # only run on CPU and first GPU device
    if mode == "basic":
        if wp.is_cpu_available():
            devices.append(wp.get_device("cpu"))
        if wp.is_cuda_available():
            devices.append(wp.get_device("cuda:0"))

    # run on CPU and all unique GPU arches
    elif mode == "unique":
        if wp.is_cpu_available():
            devices.append(wp.get_device("cpu"))

        cuda_devices = wp.get_cuda_devices()

        unique_cuda_devices = {}
        for d in cuda_devices:
            if d.arch not in unique_cuda_devices:
                unique_cuda_devices[d.arch] = d

        devices.extend(list(unique_cuda_devices.values()))

    # run on all devices
    elif mode == "all":
        devices = wp.get_devices()

    return devices


# redirects and captures all stdout output (including from C-libs)
class StdOutCapture:
    def begin(self):
        # save original
        self.saved = sys.stdout
        self.target = os.dup(self.saved.fileno())

        # create temporary capture stream
        import io, tempfile

        self.tempfile = io.TextIOWrapper(
            tempfile.TemporaryFile(buffering=0), encoding="utf-8", errors="replace", newline="", write_through=True
        )

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
        # wp.force_load()

        self.capture = StdOutCapture()
        self.capture.begin()

    def __exit__(self, exc_type, exc_value, traceback):
        # ensure any stdout output is flushed
        wp.synchronize()

        s = self.capture.end()
        if s != "":
            print(s.rstrip())

        # fail if test produces unexpected output (e.g.: from wp.expect_eq() builtins)
        # we allow strings starting of the form "Module xxx load on device xxx"
        # for lazy loaded modules
        if s != "" and not s.startswith("Module"):
            self.test.fail(f"Unexpected output:\n'{s.rstrip()}'")


def assert_array_equal(result, expect):
    a = result.numpy()
    b = expect.numpy()

    if (a == b).all() == False:
        raise AssertionError(f"Unexpected result, got: {a} expected: {b}")


def assert_np_equal(result, expect, tol=0.0):
    a = result.flatten()
    b = expect.flatten()

    if tol == 0.0:
        if (a == b).all() == False:
            raise AssertionError(f"Unexpected result, got: {a} expected: {b}")

    else:
        delta = a - b
        err = np.max(np.abs(delta))
        if err > tol:
            raise AssertionError(
                f"Maximum expected error exceeds tolerance got: {a}, expected: {b}, with err: {err} > {tol}"
            )


def create_test_func(func, device, **kwargs):
    # pass args to func
    def test_func(self):
        with CheckOutput(self):
            func(self, device, **kwargs)

    return test_func


def sanitize_identifier(s):
    """replace all non-identifier characters with '_'"""

    s = str(s)
    if s.isidentifier():
        return s
    else:
        import re

        return re.sub("\W|^(?=\d)", "_", s)


def add_function_test(cls, name, func, devices=None, **kwargs):
    if devices is None:
        setattr(cls, name, create_test_func(func, None, **kwargs))
    else:
        for device in devices:
            setattr(cls, name + "_" + sanitize_identifier(device), create_test_func(func, device, **kwargs))


def add_kernel_test(cls, kernel, dim, name=None, expect=None, inputs=None, devices=None):
    def test_func(self, device):
        args = []
        if inputs:
            args.extend(inputs)

        if expect:
            # allocate outputs to match results
            result = wp.array(expect, dtype=int, device=device)
            output = wp.zeros_like(result)

            args.append(output)

        # force load so that we don't generate any log output during launch
        kernel.module.load(device)

        with CheckOutput(self):
            wp.launch(kernel, dim=dim, inputs=args, device=device)

        # check output values
        if expect:
            assert_array_equal(output, result)

    if name is None:
        name = kernel.key

    # device is required for kernel tests, so use all devices if none were given
    if devices is None:
        devices = get_test_devices()

    # register test func with class for the given devices
    for d in devices:
        # use a lambda to forward the device to the inner test function
        test_lambda = lambda test, device=d: test_func(test, device)
        setattr(cls, name + "_" + sanitize_identifier(d), test_lambda)


# helper that first calls the test function to generate all kernel permuations
# so that compilation is done in one-shot instead of per-test
def add_function_test_register_kernel(cls, name, func, devices=None, **kwargs):
    func(None, None, **kwargs, register_kernels=True)
    add_function_test(cls, name, func, devices=devices, **kwargs)
