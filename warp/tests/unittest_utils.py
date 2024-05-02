# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import ctypes
import ctypes.util
import importlib
import os
import sys
import time
import unittest
from typing import Optional

import numpy as np

import warp as wp

pxr = importlib.util.find_spec("pxr")
USD_AVAILABLE = pxr is not None

# default test mode (see get_test_devices())
#   "basic" - only run on CPU and first GPU device
#   "unique" - run on CPU and all unique GPU arches
#   "unique_or_2x" - run on CPU and all unique GPU arches. If there is a single GPU arch, add a second GPU if it exists.
#   "all" - run on all devices
test_mode = "unique_or_2x"

coverage_enabled = False
coverage_temp_dir = None
coverage_branch = None

try:
    if sys.platform == "win32":
        LIBC = ctypes.CDLL("ucrtbase.dll")
    else:
        LIBC = ctypes.CDLL(ctypes.util.find_library("c"))
except OSError:
    print("Failed to load the standard C library")
    LIBC = None


def get_selected_cuda_test_devices(mode: Optional[str] = None):
    """Returns a list of CUDA devices according the selected ``mode`` behavior.

    If ``mode`` is ``None``, the ``global test_mode`` value will be used and
    this list will be a subset of the devices returned from ``get_test_devices()``.

    Args:
        mode: ``"basic"``, returns a list containing up to a single CUDA device.
          ``"unique"``, returns a list containing no more than one device of
          every CUDA architecture on the system.
          ``"unique_or_2x"`` behaves like ``"unique"`` but adds up to one
          additional CUDA device if the system only devices of a single CUDA
          architecture.
    """

    if mode is None:
        global test_mode
        mode = test_mode

    if mode == "basic":
        if wp.is_cuda_available():
            return [wp.get_device("cuda:0")]
        else:
            return []

    cuda_devices = wp.get_cuda_devices()
    first_cuda_devices = {}

    for d in cuda_devices:
        if d.arch not in first_cuda_devices:
            first_cuda_devices[d.arch] = d

    selected_cuda_devices = list(first_cuda_devices.values())

    if mode == "unique_or_2x" and len(selected_cuda_devices) == 1 and len(cuda_devices) > 1:
        for d in cuda_devices:
            if d not in selected_cuda_devices:
                selected_cuda_devices.append(d)
                break

    return selected_cuda_devices


def get_test_devices(mode: Optional[str] = None):
    """Returns a list of devices based on the mode selected.

    Args:
        mode: The testing mode to specify which devices to include. If not provided or ``None``, the
          ``global test_mode`` value will be used.
          "basic": Returns the CPU and the first GPU device when available.
          "unique": Returns the CPU and all unique GPU architectures.
          "unique_or_2x" (default): Behaves like "unique" but adds up to one additional CUDA device
            if the system only devices of a single CUDA architecture.
          "all": Returns all available devices.
    """
    if mode is None:
        global test_mode
        mode = test_mode

    devices = []

    if mode == "basic":
        # only run on CPU and first GPU device
        if wp.is_cpu_available():
            devices.append(wp.get_device("cpu"))
        if wp.is_cuda_available():
            devices.append(wp.get_device("cuda:0"))
    elif mode == "unique" or mode == "unique_or_2x":
        # run on CPU and a subset of GPUs
        if wp.is_cpu_available():
            devices.append(wp.get_device("cpu"))
        devices.extend(get_selected_cuda_test_devices(mode))
    elif mode == "all":
        # run on all devices
        devices = wp.get_devices()
    else:
        raise ValueError(f"Unknown test mode selected: {mode}")

    return devices


def get_cuda_test_devices(mode=None):
    devices = get_test_devices(mode=mode)
    return [d for d in devices if d.is_cuda]


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
            tempfile.TemporaryFile(buffering=0),
            encoding="utf-8",
            errors="replace",
            newline="",
            write_through=True,
        )

        os.dup2(self.tempfile.fileno(), self.saved.fileno())

        sys.stdout = self.tempfile

    def end(self):
        # The following sleep doesn't seem to fix the test_print failure on Windows
        # if sys.platform == "win32":
        #    # Workaround for what seems to be a Windows-specific bug where
        #    # the output of CUDA's `printf` is not being immediately flushed
        #    # despite the context synchronisation.
        #    time.sleep(0.01)

        if LIBC is not None:
            LIBC.fflush(None)

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


def assert_np_equal(result: np.ndarray, expect: np.ndarray, tol=0.0):
    if tol != 0.0:
        # TODO: Get all tests working without the .flatten()
        np.testing.assert_allclose(result.flatten(), expect.flatten(), atol=tol, equal_nan=True)
    else:
        # TODO: Get all tests working with strict=True
        np.testing.assert_array_equal(result, expect)


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
        setattr(
            cls,
            name + "_" + sanitize_identifier(devices),
            create_test_func(func, devices, check_output, **kwargs),
        )


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


def write_junit_results(
    outfile: str,
    test_records: list,
    tests_run: int,
    tests_failed: int,
    tests_errored: int,
    tests_skipped: int,
    test_duration: float,
):
    """Write a JUnit XML from our report data

    The report file is needed for GitLab to add test reports in merge requests.
    """

    import xml.etree.ElementTree as ET

    root = ET.Element(
        "testsuites",
        name="Warp Tests",
        failures=str(tests_failed),
        errors=str(tests_errored),
        skipped=str(tests_skipped),
        tests=str(tests_run),
        time=f"{test_duration:.3f}",
    )

    for test_data in test_records:
        test = test_data[0]
        test_duration = test_data[1]
        test_status = test_data[2]

        test_case = ET.SubElement(
            root,
            "testcase",
            classname=test.__class__.__name__,
            name=test._testMethodName,
            time=f"{test_duration:.3f}",
        )

        if test_status == "FAIL":
            failure = ET.SubElement(test_case, "failure", message=str(test_data[3]))
            failure.text = str(test_data[4])  # Stacktrace
        elif test_status == "ERROR":
            error = ET.SubElement(test_case, "error")
            error.text = str(test_data[4])  # Stacktrace
        elif test_status == "SKIP":
            skip = ET.SubElement(test_case, "skipped")
            skip.text = str(test_data[3])  # The skip reason

    tree = ET.ElementTree(root)

    if hasattr(ET, "indent"):
        ET.indent(root)  # Pretty-printed XML output, Python 3.9 required

    tree.write(outfile, encoding="utf-8", xml_declaration=True)


class ParallelJunitTestResult(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity):
        stream = type(stream)(sys.stderr)
        self.test_record = []
        super().__init__(stream, descriptions, verbosity)

    def startTest(self, test):
        if self.showAll:
            self.stream.writeln(f"{self.getDescription(test)} ...")
            self.stream.flush()
        self.start_time = time.perf_counter_ns()
        super(unittest.TextTestResult, self).startTest(test)

    def _add_helper(self, test, dots_message, show_all_message):
        if self.showAll:
            self.stream.writeln(f"{self.getDescription(test)} ... {show_all_message}")
        elif self.dots:
            self.stream.write(dots_message)
        self.stream.flush()

    def _record_test(self, test, code, message=None, details=None):
        duration = round((time.perf_counter_ns() - self.start_time) * 1e-9, 3)  # [s]
        self.test_record.append((test, duration, code, message, details))

    def addSuccess(self, test):
        super(unittest.TextTestResult, self).addSuccess(test)
        self._add_helper(test, ".", "ok")
        self._record_test(test, "OK")

    def addError(self, test, err):
        super(unittest.TextTestResult, self).addError(test, err)
        self._add_helper(test, "E", "ERROR")
        self._record_test(test, "ERROR", str(err[1]), self._exc_info_to_string(err, test))

    def addFailure(self, test, err):
        super(unittest.TextTestResult, self).addFailure(test, err)
        self._add_helper(test, "F", "FAIL")
        self._record_test(test, "FAIL", str(err[1]), self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super(unittest.TextTestResult, self).addSkip(test, reason)
        self._add_helper(test, "s", f"skipped {reason!r}")
        self._record_test(test, "SKIP", reason)

    def addExpectedFailure(self, test, err):
        super(unittest.TextTestResult, self).addExpectedFailure(test, err)
        self._add_helper(test, "x", "expected failure")
        self._record_test(test, "OK", "expected failure")

    def addUnexpectedSuccess(self, test):
        super(unittest.TextTestResult, self).addUnexpectedSuccess(test)
        self._add_helper(test, "u", "unexpected success")
        self._record_test(test, "FAIL", "unexpected success")

    def addSubTest(self, test, subtest, err):
        super(unittest.TextTestResult, self).addSubTest(test, subtest, err)
        if err is not None:
            self._add_helper(test, "E", "ERROR")
            # err is (class, error, traceback)
            self._record_test(test, "FAIL", str(err[1]), self._exc_info_to_string(err, test))

    def printErrors(self):
        pass


def _tc_escape(s):
    """Modifies strings so they can be used in TeamCity log messages."""
    s = s.replace("|", "||")
    s = s.replace("\n", "|n")
    s = s.replace("\r", "|r")
    s = s.replace("'", "|'")
    s = s.replace("[", "|[")
    s = s.replace("]", "|]")
    return s


class ParallelTeamCityTestResult(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity):
        stream = type(stream)(sys.stderr)
        super().__init__(stream, descriptions, verbosity)
        self.test_record = []

    def startTest(self, test):
        if self.showAll:
            self.stream.writeln(f"{self.getDescription(test)} ...")
            self.stream.flush()
        self.start_time = time.perf_counter_ns()
        super(unittest.TextTestResult, self).startTest(test)

    def _add_helper(self, test, dots_message, show_all_message):
        if self.showAll:
            self.stream.writeln(f"{self.getDescription(test)} ... {show_all_message}")
        elif self.dots:
            self.stream.write(dots_message)
        self.stream.flush()

    def addSuccess(self, test):
        super(unittest.TextTestResult, self).addSuccess(test)
        self._add_helper(test, ".", "ok")
        self.reportSuccess(test)

    def addError(self, test, err):
        super(unittest.TextTestResult, self).addError(test, err)
        self._add_helper(test, "E", "ERROR")
        self.reportFailure(test, err)

    def addFailure(self, test, err):
        super(unittest.TextTestResult, self).addFailure(test, err)
        self._add_helper(test, "F", "FAIL")
        self.reportFailure(test, err)

    def addSkip(self, test, reason):
        super(unittest.TextTestResult, self).addSkip(test, reason)
        self._add_helper(test, "s", f"skipped {reason!r}")
        self.reportIgnored(test, reason)

    def addExpectedFailure(self, test, err):
        super(unittest.TextTestResult, self).addExpectedFailure(test, err)
        self._add_helper(test, "x", "expected failure")
        self.reportSuccess(test)

    def addUnexpectedSuccess(self, test):
        super(unittest.TextTestResult, self).addUnexpectedSuccess(test)
        self._add_helper(test, "u", "unexpected success")
        self.reportFailure(test, "unexpected success")

    def addSubTest(self, test, subtest, err):
        super(unittest.TextTestResult, self).addSubTest(test, subtest, err)
        if err is not None:
            self._add_helper(test, "E", "ERROR")
            self.reportSubTestFailure(test, err)

    def printErrors(self):
        pass

    def reportIgnored(self, test, reason):
        test_id = test.id()
        self.stream.writeln(f"##teamcity[testIgnored name='{test_id}' message='{_tc_escape(str(reason))}']")
        self.stream.flush()

    def reportSuccess(self, test):
        duration = round((time.perf_counter_ns() - self.start_time) / 1e6)  # [ms]
        test_id = test.id()
        self.stream.writeln(f"##teamcity[testStarted name='{test_id}']")
        self.stream.writeln(f"##teamcity[testFinished name='{test_id}' duration='{duration}']")
        self.stream.flush()

    def reportFailure(self, test, err):
        test_id = test.id()
        self.stream.writeln(f"##teamcity[testStarted name='{test_id}']")
        self.stream.writeln(
            f"##teamcity[testFailed name='{test_id}' message='{_tc_escape(str(err[1]))}' details='{_tc_escape(self._exc_info_to_string(err, test))}']"
        )
        self.stream.writeln(f"##teamcity[testFinished name='{test_id}']")
        self.stream.flush()

    def reportSubTestFailure(self, test, err):
        test_id = test.id()
        self.stream.writeln(f"##teamcity[testStarted name='{test_id}']")
        self.stream.writeln(
            f"##teamcity[testFailed name='{test_id}' message='{_tc_escape(str(err[1]))}' details='{_tc_escape(self._exc_info_to_string(err, test))}']"
        )
        self.stream.writeln(f"##teamcity[testFinished name='{test_id}']")
        self.stream.flush()
