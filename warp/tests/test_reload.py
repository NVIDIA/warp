# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import importlib
import os
import unittest

import numpy as np

import warp as wp

# dummy modules used for testing reload with dependencies
import warp.tests.aux_test_dependent as test_dependent
import warp.tests.aux_test_reference as test_reference
import warp.tests.aux_test_reference_reference as test_reference_reference

# dummy module used for testing reload
import warp.tests.aux_test_square as test_square
from warp.tests.unittest_utils import *


def reload_module(module):
    # Clearing the .pyc file associated with a module is a necessary workaround
    # for `importlib.reload` to work as expected when run from within Kit.
    cache_file = importlib.util.cache_from_source(module.__file__)
    os.remove(cache_file)
    importlib.reload(module)


def test_redefine(test, device):
    # --------------------------------------------
    # first pass

    @wp.kernel
    def basic(x: wp.array(dtype=float)):
        tid = wp.tid()

        x[tid] = float(tid) * 1.0

    n = 32

    x = wp.zeros(n, dtype=float, device=device)

    wp.launch(kernel=basic, dim=n, inputs=[x], device=device)

    # --------------------------------------------
    # redefine kernel, should trigger a recompile

    @wp.kernel
    def basic(x: wp.array(dtype=float)):
        tid = wp.tid()

        x[tid] = float(tid) * 2.0

    y = wp.zeros(n, dtype=float, device=device)

    wp.launch(kernel=basic, dim=n, inputs=[y], device=device)

    assert_np_equal(np.arange(0, n, 1), x.numpy())
    assert_np_equal(np.arange(0, n, 1) * 2.0, y.numpy())


square_two = """import warp as wp


@wp.func
def sqr(x: float):
    return x * x


@wp.kernel
def kern(expect: float):
    wp.expect_eq(sqr(2.0), expect)


def run(expect, device):
    wp.launch(kern, dim=1, inputs=[expect], device=device)
"""

square_four = """import warp as wp


@wp.func
def multiply(x: float):
    return x * x


@wp.kernel
def kern(expect: float):
    wp.expect_eq(multiply(4.0), expect)


def run(expect, device):
    wp.launch(kern, dim=1, inputs=[expect], device=device)
"""


def test_reload(test, device):
    # write out the module python and import it
    f = open(os.path.abspath(os.path.join(os.path.dirname(__file__), "aux_test_square.py")), "w")
    f.writelines(square_two)
    f.flush()
    f.close()

    reload_module(test_square)
    test_square.run(expect=4.0, device=device)  # 2*2=4

    f = open(os.path.abspath(os.path.join(os.path.dirname(__file__), "aux_test_square.py")), "w")
    f.writelines(square_four)
    f.flush()
    f.close()

    # reload module, this should trigger all of the funcs / kernels to be updated
    reload_module(test_square)
    test_square.run(expect=16.0, device=device)  # 4*4 = 16


def test_reload_class(test, device):
    def test_func():
        import importlib as imp

        import warp.tests.aux_test_class_kernel
        from warp.tests.aux_test_class_kernel import ClassKernelTest

        imp.reload(warp.tests.aux_test_class_kernel)

        ctest = ClassKernelTest(device)
        expected = np.zeros((10, 3, 3), dtype=np.float32)
        expected[:] = np.eye(3)
        assert_np_equal(expected, ctest.identities.numpy())

    test_func()
    test_func()


template_ref = """# This file is used to test reloading module references.

import warp as wp
import warp.tests.aux_test_reference_reference as refref


@wp.func
def magic():
    return {} * refref.more_magic()
"""

template_refref = """# This file is used to test reloading module references.

import warp as wp


@wp.func
def more_magic():
    return {}
"""


def test_reload_references(test, device):
    path_ref = os.path.abspath(os.path.join(os.path.dirname(__file__), "aux_test_reference.py"))
    path_refref = os.path.abspath(os.path.join(os.path.dirname(__file__), "aux_test_reference_reference.py"))

    # rewrite both dependency modules and reload them
    with open(path_ref, "w") as f:
        f.writelines(template_ref.format(1.0))
    importlib.reload(test_reference)

    with open(path_refref, "w") as f:
        f.writelines(template_refref.format(1.0))
    importlib.reload(test_reference_reference)

    test_dependent.run(expect=1.0, device=device)  # 1 * 1 = 1

    # rewrite and reload the first dependency module
    with open(path_ref, "w") as f:
        f.writelines(template_ref.format(2.0))
    importlib.reload(test_reference)

    test_dependent.run(expect=2.0, device=device)  # 2 * 1 = 1

    # rewrite and reload the second dependency module
    with open(path_refref, "w") as f:
        f.writelines(template_refref.format(2.0))
    importlib.reload(test_reference_reference)

    test_dependent.run(expect=4.0, device=device)  # 2 * 2 = 4


devices = get_test_devices()


class TestReload(unittest.TestCase):
    pass


add_function_test(TestReload, "test_redefine", test_redefine, devices=devices)
add_function_test(TestReload, "test_reload", test_reload, devices=devices)
add_function_test(TestReload, "test_reload_class", test_reload_class, devices=devices)
add_function_test(TestReload, "test_reload_references", test_reload_references, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=False)
