# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import ctypes
import ctypes.util
import os
import sys
import unittest

import numpy as np

import warp as wp

try:
    import pxr  # noqa: F401

    USD_AVAILABLE = True
except ImportError as e:
    USD_AVAILABLE = False
    print(f"Skipping USD tests for reason: {e}")

# default test mode (see get_test_devices())
#   "basic" - only run on CPU and first GPU device
#   "unique" - run on CPU and all unique GPU arches
#   "all" - run on all devices
test_mode = "unique"

try:
    if sys.platform == "win32":
        LIBC = ctypes.CDLL("ucrtbase.dll")
    else:
        LIBC = ctypes.CDLL(ctypes.util.find_library("c"))
except OSError:
    print("Failed to load the standard C library")
    LIBC = None


def get_unique_cuda_test_devices(mode=None):
    """Returns a list of unique CUDA devices according to the CUDA arch.

    If ``mode`` is ``None``, the ``global test_mode`` value will be used and
    this list will be a subset of the devices returned from ``get_test_devices()``.
    """

    if mode is None:
        global test_mode
        mode = test_mode

    if mode == "basic":
        cuda_devices = [wp.get_device("cuda:0")]
    else:
        cuda_devices = wp.get_cuda_devices()

    unique_cuda_devices = {}
    for d in cuda_devices:
        if d.arch not in unique_cuda_devices:
            unique_cuda_devices[d.arch] = d

    return list(unique_cuda_devices.values())


def get_test_devices(mode=None):
    """Returns a list of devices based on the mode selected.

    Args:
        mode (str, optional): The testing mode to specify which devices to include. If not provided or ``None``, the
          ``global test_mode`` value will be used.
          "basic" (default): Returns the CPU and the first GPU device when available.
          "unique": Returns the CPU and all unique GPU architectures.
          "all": Returns all available devices.
    """
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

        devices.extend(get_unique_cuda_test_devices())

    # run on all devices
    elif mode == "all":
        devices = wp.get_devices()

    return devices


# redirects and captures all stdout output (including from C-libs)
class StdOutCapture:
    def begin(self):
        # Flush the stream buffers managed by libc.
        # This is needed at the moment due to Carbonite not flushing the logs
        # being printed out when extensions are starting up.
        if LIBC is not None:
            LIBC.fflush(None)

        # save original
        self.saved = sys.stdout
        self.target = os.dup(self.saved.fileno())

        # create temporary capture stream
        import io
        import tempfile

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


def assert_array_equal(result: wp.array, expect: wp.array):
    np.testing.assert_equal(result.numpy(), expect.numpy())


def assert_np_equal(result, expect, tol=0.0):
    a = result.flatten()
    b = expect.flatten()

    if tol == 0.0:
        if not (a == b).all():
            raise AssertionError(f"Unexpected result, got: {a} expected: {b}")

    else:
        delta = a - b
        err = np.max(np.abs(delta))
        if err > tol:
            raise AssertionError(
                f"Maximum expected error exceeds tolerance got: {a}, expected: {b}, with err: {err} > {tol}"
            )


# if check_output is True any output to stdout will be treated as an error
def create_test_func(func, device, check_output, **kwargs):
    # pass args to func
    def test_func(self):
        if check_output:
            with CheckOutput(self):
                func(self, device, **kwargs)
        else:
            func(self, device, **kwargs)

    return test_func


def skip_test_func(self):
    # A function to use so we can tell unittest that the test was skipped.
    self.skipTest("No suitable devices to run the test.")


def sanitize_identifier(s):
    """replace all non-identifier characters with '_'"""

    s = str(s)
    if s.isidentifier():
        return s
    else:
        import re

        return re.sub(r"\W|^(?=\d)", "_", s)


def add_function_test(cls, name, func, devices=None, check_output=True, **kwargs):
    if devices is None:
        setattr(cls, name, create_test_func(func, None, check_output, **kwargs))
    elif isinstance(devices, list):
        if not devices:
            # No devices to run this test
            setattr(cls, name, skip_test_func)
        else:
            for device in devices:
                setattr(
                    cls,
                    name + "_" + sanitize_identifier(device),
                    create_test_func(func, device, check_output, **kwargs),
                )
    else:
        setattr(cls, name + "_" + sanitize_identifier(devices), create_test_func(func, devices, check_output, **kwargs))


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
        # use a function to forward the device to the inner test function
        def test_func_wrapper(test, device=d):
            test_func(test, device)

        setattr(cls, name + "_" + sanitize_identifier(d), test_func_wrapper)


# helper that first calls the test function to generate all kernel permutations
# so that compilation is done in one-shot instead of per-test
def add_function_test_register_kernel(cls, name, func, devices=None, **kwargs):
    func(None, None, **kwargs, register_kernels=True)
    add_function_test(cls, name, func, devices=devices, **kwargs)


class TeamCityTestResult(unittest.TextTestResult):
    """This class will report each test result to TeamCity"""

    def __init__(self, stream, descriptions, verbosity):
        super(TeamCityTestResult, self).__init__(stream, descriptions, verbosity)

    def addSuccess(self, test):
        super(TeamCityTestResult, self).addSuccess(test)
        self.reportSuccess(test)

    def addError(self, test, err):
        super(TeamCityTestResult, self).addError(test, err)
        self.reportFailure(test)

    def addFailure(self, test, err):
        super(TeamCityTestResult, self).addFailure(test, err)
        self.reportFailure(test)

    def addSkip(self, test, reason):
        super(TeamCityTestResult, self).addSkip(test, reason)

    def addExpectedFailure(self, test, err):
        super(TeamCityTestResult, self).addExpectedFailure(test, err)
        self.reportSuccess(test)

    def addUnexpectedSuccess(self, test):
        super(TeamCityTestResult, self).addUnexpectedSuccess(test)
        self.reportFailure(test)

    def reportSuccess(self, test):
        test_id = test.id()
        print(f"##teamcity[testStarted name='{test_id}']")
        print(f"##teamcity[testFinished name='{test_id}']")

    def reportFailure(self, test):
        test_id = test.id()
        print(f"##teamcity[testStarted name='{test_id}']")
        print(f"##teamcity[testFailed name='{test_id}']")
        print(f"##teamcity[testFinished name='{test_id}']")


class TeamCityTestRunner(unittest.TextTestRunner):
    """Test runner that will report test results to TeamCity if running in TeamCity"""

    def __init__(self, **kwargs):
        self.running_in_teamcity = os.environ.get("TEAMCITY_VERSION") is not None
        if self.running_in_teamcity:
            kwargs["resultclass"] = TeamCityTestResult
        super(TeamCityTestRunner, self).__init__(**kwargs)

    def run(self, test, name):
        if self.running_in_teamcity:
            print(f"##teamcity[testSuiteStarted name='{name}']")

        result = super(TeamCityTestRunner, self).run(test)

        if self.running_in_teamcity:
            print(f"##teamcity[testSuiteFinished name='{name}']")
            if not result.wasSuccessful():
                print("##teamcity[buildStatus status='FAILURE']")

        return result
