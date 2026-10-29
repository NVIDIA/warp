# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test that a kernel that fails to build fails its module, says why, and can be fixed."""

import os
import tempfile
import unittest
import uuid
import weakref
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


def _redefine(module, code: str):
    """Run more source in an imported module's namespace, as running a notebook cell again does.

    Warp reads a kernel's source from its file when the kernel is defined, so the
    code has to come from a real file, which can be removed afterwards.
    """
    file, file_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(file, "w") as f:
            f.write(code)

        exec(compile(code, file_path, "exec"), module.__dict__)
    finally:
        os.remove(file_path)


def _copy_frame_locals(error: Exception) -> list[dict]:
    """Copy the local variables of every frame in an error's traceback.

    IPython keeps the last exception for ``%debug``, and showing its traceback
    keeps a dictionary of each frame's local variables as well.
    """
    frame_locals = []
    tb = error.__traceback__
    while tb is not None:
        frame_locals.append(dict(tb.tb_frame.f_locals))
        tb = tb.tb_next

    return frame_locals


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


_REDEFINITION_SOURCE = """\
import warp as wp

@wp.kernel
def exercise_kernel(a: wp.array[float]):
    i = wp.tid()
    a[i] = None  # unfinished, fails codegen

@wp.kernel
def sibling_kernel(a: wp.array[float]):
    i = wp.tid()
    a[i] = 7.0
"""

_STILL_FAILING_EXERCISE_SOURCE = """\
@wp.kernel
def exercise_kernel(a: wp.array[float]):
    i = wp.tid()
    a[i] = undefined_name
"""

_FIXED_EXERCISE_SOURCE = """\
@wp.kernel
def exercise_kernel(a: wp.array[float]):
    i = wp.tid()
    a[i] = 2.0 * float(i)
"""

_EXERCISE_ERROR = r'Error while parsing function "exercise_kernel"'


def test_redefinition_recovers_failed_module(test, device):
    """Verify redefining a kernel that failed to build lets its module build again.

    This is the notebook workflow: a cell's kernel fails to build, and running
    the cell again with each edit replaces it. IPython keeps each error and the
    local variables of its frames, which must not keep a replaced definition in
    the module. Otherwise the module keeps reporting the first error.
    """
    module = _import_module(_REDEFINITION_SOURCE)
    a = wp.zeros(4, dtype=float, device=device)
    kept = []

    try:
        wp.launch(module.exercise_kernel, dim=4, inputs=[a], device=device)
    except wp.WarpCodegenTypeError as error:
        kept.append((error, _copy_frame_locals(error)))

    test.assertEqual(len(kept), 1)

    _redefine(module, _STILL_FAILING_EXERCISE_SOURCE)

    try:
        wp.launch(module.exercise_kernel, dim=4, inputs=[a], device=device)
    except wp.WarpCodegenKeyError as error:
        test.assertRegex(str(error), "undefined_name")
        kept.append((error, _copy_frame_locals(error)))

    test.assertEqual(len(kept), 2)

    _redefine(module, _FIXED_EXERCISE_SOURCE)

    wp.launch(module.exercise_kernel, dim=4, inputs=[a], device=device)
    assert_np_equal(a.numpy(), np.array([0.0, 2.0, 4.0, 6.0]))

    wp.launch(module.sibling_kernel, dim=4, inputs=[a], device=device)
    assert_np_equal(a.numpy(), np.full(4, 7.0))


def _check_release(test, module, call):
    """Check that a kept error from a failed build does not keep the failed kernel alive.

    ``call`` builds ``module``, whose ``exercise_kernel`` fails to build. The
    error and its frames' locals are kept, as IPython keeps them, and then the
    kernel is redefined. The failed definition must be gone, so the next build
    leaves it out.
    """
    failed_definition = weakref.ref(module.exercise_kernel)
    kept = []

    try:
        call()
    except Exception as error:
        # get_cuda_kernel_properties raises its own error from the build error
        test.assertIsInstance(error.__cause__ or error, wp.WarpCodegenTypeError)
        kept.append((error, _copy_frame_locals(error)))
    else:
        test.fail("expected the build to fail")

    _redefine(module, _FIXED_EXERCISE_SOURCE)
    test.assertIsNone(failed_definition())


def test_launch_releases_failed_kernel(test, device):
    """Verify a kept error from wp.launch() releases the kernel that failed to build."""
    module = _import_module(_REDEFINITION_SOURCE)
    a = wp.zeros(1, dtype=float, device=device)
    _check_release(test, module, lambda: wp.launch(module.exercise_kernel, dim=1, inputs=[a], device=device))


def test_launch_tiled_releases_failed_kernel(test, device):
    """Verify a kept error from wp.launch_tiled() releases the kernel that failed to build."""
    module = _import_module(_REDEFINITION_SOURCE)
    a = wp.zeros(1, dtype=float, device=device)
    _check_release(
        test, module, lambda: wp.launch_tiled(module.exercise_kernel, dim=1, inputs=[a], block_dim=1, device=device)
    )


def test_load_module_releases_failed_kernel(test, device):
    """Verify a kept error from wp.load_module() releases the kernel that failed to build."""
    module = _import_module(_REDEFINITION_SOURCE)
    _check_release(test, module, lambda: wp.load_module(module, device=device))


def test_get_cuda_kernel_properties_releases_failed_kernel(test, device):
    """Verify a kept error from wp.get_cuda_kernel_properties() releases the kernel that failed to build."""
    module = _import_module(_REDEFINITION_SOURCE)
    _check_release(test, module, lambda: wp.get_cuda_kernel_properties(module.exercise_kernel, device))


def test_get_suggested_block_size_releases_failed_kernel(test, device):
    """Verify a kept error from wp.get_suggested_block_size() releases the kernel that failed to build."""
    module = _import_module(_REDEFINITION_SOURCE)
    _check_release(test, module, lambda: wp.get_suggested_block_size(module.exercise_kernel, device))


def test_get_cuda_max_cluster_dim_releases_failed_kernel(test, device):
    """Verify a kept error from wp.get_cuda_max_cluster_dim() releases the kernel that failed to build."""
    if device.arch < 90:
        # the query returns 1 without building the module
        test.skipTest("Device arch < 90")

    module = _import_module(_REDEFINITION_SOURCE)
    _check_release(test, module, lambda: wp.get_cuda_max_cluster_dim(module.exercise_kernel, device))


class TestModuleContamination(unittest.TestCase):
    pass


devices = get_test_devices()
cuda_devices = get_cuda_test_devices()

for _func in (
    test_validation_failure_names_the_annotation,
    test_validation_failure_fails_the_module,
    test_validation_failure_repeats_on_relaunch,
    test_native_failure_fails_the_module,
    test_redefinition_recovers_failed_module,
    test_launch_releases_failed_kernel,
    test_launch_tiled_releases_failed_kernel,
    test_load_module_releases_failed_kernel,
):
    add_function_test(TestModuleContamination, func=_func, name=_func.__name__, devices=devices)

# the kernel queries only build the module on CUDA devices
for _func in (
    test_get_cuda_kernel_properties_releases_failed_kernel,
    test_get_suggested_block_size_releases_failed_kernel,
    test_get_cuda_max_cluster_dim_releases_failed_kernel,
):
    add_function_test(TestModuleContamination, func=_func, name=_func.__name__, devices=cuda_devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
