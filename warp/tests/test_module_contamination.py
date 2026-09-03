# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test that a kernel that fails to build fails its module, and says why."""

import os
import tempfile
import unittest
import uuid
from importlib import util

import warp as wp
from warp.tests.unittest_utils import *


def _import_module(code: str):
    """Import a throwaway Warp module from source.

    The kernels under test have to live in a module of their own. Defining them
    in this file would fail the test file's own module, and the point of these
    tests is what a build failure does to the module that contains it.
    """
    name = f"_test_module_contamination_{uuid.uuid4().hex[:12]}"
    file, file_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(file, "w") as f:
            f.write(code)

        spec = util.spec_from_file_location(name, file_path)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.remove(file_path)

    return module


_VALIDATION_FAILURE_SOURCE = """\
import warp as wp

@wp.func
def bad_return_type(x: int) -> tuple[int, int, int]:
    # annotated as 3 elements, returns 2
    return (x + x, x * x)

@wp.kernel
def bad_kernel():
    _x, _y, _z = bad_return_type(123)

@wp.kernel
def sibling_kernel(a: wp.array[float]):
    i = wp.tid()
    a[i] = 7.0
"""

_NATIVE_FAILURE_SOURCE = '''\
import warp as wp

snippet = """
    not valid C++ #### ;;; @@@
"""

@wp.func_native(snippet)
def broken_native(a: wp.array[float], tid: int):
    ...

@wp.kernel
def bad_kernel(a: wp.array[float]):
    tid = wp.tid()
    broken_native(a, tid)

@wp.kernel
def sibling_kernel(a: wp.array[float]):
    i = wp.tid()
    a[i] = 7.0
'''

_ANNOTATION_ERROR = r"has its return type annotated as a tuple of 3 elements but the code returns 2 values"


def test_validation_failure_names_the_annotation(test, device):
    """Verify a failed return-type validation reports the annotation, not C++ fallout.

    Validation runs during build(), before any C++ referencing the function is
    emitted, so the error names the bad annotation. Emitting the call first would
    surface this as an undeclared identifier from the native compiler instead.
    """
    module = _import_module(_VALIDATION_FAILURE_SOURCE)

    with test.assertRaisesRegex(wp.WarpCodegenError, _ANNOTATION_ERROR):
        wp.launch(module.bad_kernel, dim=1, device=device)


def test_validation_failure_fails_the_module(test, device):
    """Verify a sibling of a kernel that failed validation reports that failure.

    A module is the compilation unit, so a kernel that cannot build fails the
    module rather than being dropped from it. Launching any other kernel in that
    module has to report the failure instead of returning as though it had run.
    """
    module = _import_module(_VALIDATION_FAILURE_SOURCE)
    a = wp.zeros(4, dtype=float, device=device)

    with test.assertRaisesRegex(wp.WarpCodegenError, _ANNOTATION_ERROR):
        wp.launch(module.bad_kernel, dim=1, device=device)

    with test.assertRaisesRegex(wp.WarpCodegenError, _ANNOTATION_ERROR):
        wp.launch(module.sibling_kernel, dim=4, inputs=[a], device=device)


def test_validation_failure_repeats_on_relaunch(test, device):
    """Verify a build failure keeps failing rather than reporting only once.

    The failure is reported from the recorded error on every later launch, so a
    kernel relaunched in a loop cannot quietly start running partial state.
    """
    module = _import_module(_VALIDATION_FAILURE_SOURCE)

    for _ in range(2):
        with test.assertRaisesRegex(wp.WarpCodegenError, _ANNOTATION_ERROR):
            wp.launch(module.bad_kernel, dim=1, device=device)


def test_native_failure_fails_the_module(test, device):
    """Verify a native compile failure behaves the same way as a codegen failure.

    The snippet is not valid C++, so Warp's own codegen succeeds and the native
    compiler fails instead. Both classes fail the module, so a sibling reports
    the failure rather than skipping.
    """
    module = _import_module(_NATIVE_FAILURE_SOURCE)
    a = wp.zeros(4, dtype=float, device=device)

    with test.assertRaises(Exception) as caught:
        wp.launch(module.bad_kernel, dim=4, inputs=[a], device=device)

    with test.assertRaises(type(caught.exception)):
        wp.launch(module.sibling_kernel, dim=4, inputs=[a], device=device)


class TestModuleContamination(unittest.TestCase):
    pass


devices = get_test_devices()

for _func in (
    test_validation_failure_names_the_annotation,
    test_validation_failure_fails_the_module,
    test_validation_failure_repeats_on_relaunch,
    test_native_failure_fails_the_module,
):
    add_function_test(TestModuleContamination, func=_func, name=_func.__name__, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
