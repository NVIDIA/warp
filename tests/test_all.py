import unittest
import sys
import os
from unittest import runner

import warp as wp

import test_codegen
import test_mesh_unit_tests

tests = unittest.TestSuite()
result = unittest.TestResult()

tests.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_codegen))
tests.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_mesh_unit_tests))

# load all modules
wp.force_load()

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2, failfast=False)
    runner.run(tests)

