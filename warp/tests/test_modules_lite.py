# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import unittest

import warp as wp
import warp.utils
from warp.tests.unittest_utils import *

devices = get_test_devices()


# This kernel is needed to ensure this test module is registered as a Warp module.
# wp.load_module() requires the module to contain at least one Warp kernel, function, or struct.
@wp.kernel
def print_values():
    i = wp.tid()
    wp.print(i)


class TestModuleLite(unittest.TestCase):
    def test_module_lite_load(self):
        # Load current module
        wp.load_module()

        # Load named module
        wp.load_module(warp.utils)

        # Load named module (string)
        wp.load_module("warp.utils", recursive=True)

    def test_module_lite_options(self):
        wp.set_module_options({"max_unroll": 8})
        module_options = wp.get_module_options()
        self.assertIsInstance(module_options, dict)
        self.assertEqual(module_options["max_unroll"], 8)

    def test_module_lite_load_nonexistent(self):
        # Test that loading a non-existent module raises RuntimeError
        with self.assertRaises(RuntimeError) as context:
            wp.load_module("nonexistent_module_that_does_not_exist")

        self.assertIn("does not contain any Warp kernels, functions, or structs", str(context.exception))
        self.assertIn("nonexistent_module_that_does_not_exist", str(context.exception))

    def test_module_lite_load_no_warp_content(self):
        # Test that loading a module without Warp content raises RuntimeError
        # Use a standard library module that definitely has no Warp kernels
        with self.assertRaises(RuntimeError) as context:
            wp.load_module(unittest)

        self.assertIn("does not contain any Warp kernels, functions, or structs", str(context.exception))
        self.assertIn("unittest", str(context.exception))

    def test_module_lite_eager_source_registration(self):
        # The source-to-public Warp module declarations live at the top of ``warp/__init__.py``
        # and run before any ``warp._src`` submodule can be imported. So importing a source
        # module directly — without importing its public package — still registers its
        # constructs under the public Warp module name (no migration needed). Run in a
        # subprocess so the import ordering is deterministic regardless of what the test
        # process already imported.
        code = (
            "import warp as wp\n"
            # Import the source module directly, WITHOUT importing warp.optim first.
            "import warp._src.optim.linear\n"
            # Constructs resolve to the public name even though only the source was imported.
            "pub = wp.get_module('warp.optim.linear')\n"
            "assert pub.name == 'warp.optim.linear', pub.name\n"
            "k, f = len(pub.kernels), len(pub.functions)\n"
            "assert k > 0 and f > 0, ('public module not populated', k, f)\n"
            "assert pub.options['enable_backward'] is False, pub.options['enable_backward']\n"
            # A lookup by the source name resolves to the same public module.
            "src = wp.get_module('warp._src.optim.linear')\n"
            "assert src is pub, 'source-name lookup did not resolve to the public module'\n"
            "print('OK')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK", result.stdout)

    def test_module_lite_placeholder_then_source(self):
        # A bare ``get_module(public_name)`` lookup performed before the source module is
        # imported creates an empty module under the public name. Because constructs always
        # resolve to that same public name, importing the source afterwards must populate that
        # very module rather than stranding the constructs elsewhere. Run in a subprocess so
        # the import ordering is deterministic.
        code = (
            "import warp as wp\n"
            # Bare public-name lookup BEFORE the source is imported: empty placeholder.
            "placeholder = wp.get_module('warp.optim.linear')\n"
            "assert not placeholder.kernels and not placeholder.functions, 'placeholder is not empty'\n"
            # Importing the source populates that same placeholder (constructs resolve to it).
            "import warp._src.optim.linear\n"
            "pub = wp.get_module('warp.optim.linear')\n"
            "assert pub is placeholder, 'source import did not populate the existing public module'\n"
            "assert pub.name == 'warp.optim.linear', pub.name\n"
            "assert len(pub.kernels) > 0 and len(pub.functions) > 0, 'constructs were stranded'\n"
            "print('OK')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
