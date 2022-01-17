import unittest
import sys
import os
from unittest import runner

import warp as wp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'tests')))

import test_codegen
import test_mesh_unit_tests

tests = unittest.TestSuite()
result = unittest.TestResult()

tests.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_codegen))
tests.addTests(unittest.defaultTestLoader.loadTestsFromModule(test_mesh_unit_tests))

# load all modules
wp.force_load()

if __name__ == '__main__':
    #unittest.main(verbosity=2)
    runner = unittest.TextTestRunner(verbosity=2, failfast=False)
    runner.run(tests)


    #tests.run(result, verbosity=2)
