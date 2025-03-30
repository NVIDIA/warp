Contribution Guide
==================

Some ways to contribute to the development of Warp include:

* Reporting bugs and requesting new features on `GitHub <https://github.com/NVIDIA/warp/issues>`__.
* Asking questions, sharing your work, or participating in discussion threads on
  `GitHub <https://github.com/NVIDIA/warp/discussions>`__ (preferred) or
  `Discord <https://discord.com/invite/nvidiaomniverse>`__. 
* Adding new examples to the Warp repository.
* Documentation improvements.
* Contributing bug fixes or new features.

Code Contributions
------------------

Code contributions from the community are welcome.
Rather than requiring a formal Contributor License Agreement (CLA), we use the
`Developer Certificate of Origin <https://developercertificate.org/>`__ to
ensure contributors have the right to submit their contributions to this project.
Please ensure that all commits have a
`sign-off <https://git-scm.com/docs/git-commit#Documentation/git-commit.txt--s>`__ 
added with an email address that matches the commit author
to agree to the DCO terms for each particular contribution.

The full text of the DCO is as follows:

.. code-block:: text

    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.

Contributors are encouraged to first open an issue on GitHub to discuss proposed
feature contributions and gauge potential interest.

Overview
^^^^^^^^

#. Create a fork of the Warp GitHub repository by visiting https://github.com/NVIDIA/warp/fork
#. Clone your fork on your local machine, e.g. ``git clone git@github.com:username/warp.git``.
#. Create a branch to develop your contribution on, e.g. ``git checkout -b mmacklin/cuda-bvh-optimizations``.

   Use the following naming conventions for the branch name:

   * New features: ``username/feature-name``
   * Bug fixes: ``bugfix/feature-name``

#. Make your desired changes.

   * Please familiarize yourself with the :ref:`coding-guidelines`.
   * Ensure that code changes pass :ref:`linting and formatting checks <linting-and-formatting>`.
   * Test cases should be written to verify correctness (:ref:`testing-warp`).
   * Documentation should be added for new features (:ref:`building-docs`).
   * Add an entry to the unreleased section at the top of the
     `CHANGELOG.md <https://github.com/NVIDIA/warp/blob/main/CHANGELOG.md>`__ describing the changes.

#. Push your branch to your GitHub fork, e.g. ``git push origin username/feature-name``.
#. Submit a pull request on GitHub to the ``main`` branch (:ref:`pull-requests`).
   Work with reviewers to ensure the pull request is in a state suitable for merging.

.. _coding-guidelines:

General Coding Guidelines
^^^^^^^^^^^^^^^^^^^^^^^^^

* Follow `PEP 8 <https://peps.python.org/pep-0008/>`__ as the baseline for coding style, but prioritize matching the
  existing style and conventions of the file being modified to maintain consistency.
* Use `snake case <https://en.wikipedia.org/wiki/Snake_case>`__ for all function names.
* Use `Google-style docstrings <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`__
  for Python code.
* Include the NVIDIA copyright header on all newly created files, updating the year to current year at the time of
  the initial file creation.
* Aim for consistency in variable and function names.

  * Use the existing terminology when possible when naming new functions (e.g. use ``points`` instead of ``vertex_buffer``).
  * Don't introduce new abbreviations if one already exists in the code base.
  * Also be mindful of consistency and clarity when naming local function variables.

* Avoid generic function names like ``get_data()``.
* Follow the existing style conventions in any CUDA C++ files being modified.
* Use both ``inputs`` and ``outputs`` parameters in ``wp.launch()`` in functions that are expected to be used in
  differentiable programming applications to aid in visualization and debugging tools.

.. _linting-and-formatting:

Linting and Formatting
^^^^^^^^^^^^^^^^^^^^^^

`Ruff <https://docs.astral.sh/ruff/>`__ is used as the linter and code formatter for Python code in the Warp repository.
The contents of pull requests will automatically be checked to ensure adherence to our formatting and linting standards.

We recommend first running Ruff locally on your branch prior to opening a pull request.
From the project root, run:

.. code-block:: bash

    pip install pre-commit
    pre-commit run --all

This command will attempt to fix any lint violations and then format the code.

To run Ruff checks at the same time as ``git commit``, pre-commit hooks can be installed by running this command in the project root:

.. code-block:: bash

    pre-commit install

.. _building-docs:

Building the Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Sphinx documentation can be built by running the following from the project root:

.. code-block:: bash

    pip install -r docs/requirements.txt
    python build_docs.py

This command also regenerates the stub file (``warp/stubs.py``) and the reStructuredText file for the
:doc:`functions` page. After building the documentation, it is recommended to run a ``git status`` to
check if your changes have modified these files. If so, please commit the modified files to your branch.

.. note:: In the future, Warp needs to be built at least once prior to building the documentation.

.. _pull-requests:

Pull Request Guidelines
^^^^^^^^^^^^^^^^^^^^^^^

* Ensure your pull request has a descriptive title that clearly states the purpose of the changes.
* Include a brief description covering:

  * Summary of changes.
  * Areas affected by the changes.
  * The problem being solved.
  * Any limitations or non-handled areas in the changes.
  * Any existing GitHub issues being addressed by the changes.

.. _testing-warp:

Testing Warp
------------

Running the Test Suite
^^^^^^^^^^^^^^^^^^^^^^

Warp's test suite uses the `unittest <https://docs.python.org/3/library/unittest.html>`__ unit testing framework,
along with `unittest-parallel <https://github.com/craigahobbs/unittest-parallel>`__ to run tests in parallel.

The majority of the Warp tests are located in the `warp/tests <https://github.com/NVIDIA/warp/tree/main/warp/tests>`__
directory. As part of the test suite, most examples in the ``warp/examples`` subdirectories are tested via
`test_examples.py <https://github.com/NVIDIA/warp/blob/main/warp/tests/test_examples.py>`__.

After building and installing Warp (``pip install -e .`` from the project root), run the test suite using
``python -m warp.tests``. The tests should take 5–10 minutes to run. By default, only the test modules
defined in ``default_suite()`` (in ``warp/tests/unittest_suites.py``) are run. To run the test suite
using `test discovery <https://docs.python.org/3/library/unittest.html#test-discovery>`__, use
``python -m warp.tests -s autodetect``, which will discover tests in modules matching the path
``warp/tests/test*.py``.

Running a subset of tests
"""""""""""""""""""""""""

Instead of running the full test suite, there are two main ways to select a subset of tests to run.
These options must be used with the ``-s autodetect`` option.

Use ``-p PATTERN`` to define a pattern to match test files.
For example, to run only tests that have ``mesh`` in the file name, use:

.. code-block:: bash

    python -m warp.tests -s autodetect -p '*mesh*.py'

Use ``-k TESTNAMEPATTERNS`` to define `wildcard test name patterns <https://docs.python.org/3/library/unittest.html#unittest.TestLoader.testNamePatterns>`__.
This option can be used multiple times.
For example, to run only tests that have either ``mgpu`` or ``cuda`` in their name, use:

.. code-block:: bash

    python -m warp.tests -s autodetect -k 'mgpu' -k 'cuda'

Adding New Tests
^^^^^^^^^^^^^^^^

For tests that should be run on multiple devices, e.g. ``"cpu"``, ``"cuda:0"``, and ``"cuda:1"``, we recommend
first defining a test function at the module scope and then using ``add_function_test()`` to add multiple
test methods (a separate method for each device) to a test class.

.. code-block:: python

    # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    # SPDX-License-Identifier: Apache-2.0
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    # http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    import unittest

    import warp as wp
    from warp.tests.unittest_utils import *


    def test_amazing_code_test_one(test, device):
        pass

    devices = get_test_devices()


    class TestAmazingCode(unittest.TestCase):
        pass

    add_function_test(TestAmazingCode, "test_amazing_code_test_one", test_amazing_code_test_one, devices=devices)


    if __name__ == "__main__":
        wp.clear_kernel_cache()
        unittest.main(verbosity=2)

If we directly run this module, we get the following output:

.. code-block:: bash

    python test_amazing_code.py 
    Warp 1.3.1 initialized:
    CUDA Toolkit 12.6, Driver 12.6
    Devices:
        "cpu"      : "x86_64"
        "cuda:0"   : "NVIDIA GeForce RTX 3090" (24 GiB, sm_86, mempool enabled)
        "cuda:1"   : "NVIDIA GeForce RTX 3090" (24 GiB, sm_86, mempool enabled)
    CUDA peer access:
        Supported fully (all-directional)
    Kernel cache:
        /home/nvidia/.cache/warp/1.3.1
    test_amazing_code_test_one_cpu (__main__.TestAmazingCode) ... ok
    test_amazing_code_test_one_cuda_0 (__main__.TestAmazingCode) ... ok
    test_amazing_code_test_one_cuda_1 (__main__.TestAmazingCode) ... ok

    ----------------------------------------------------------------------
    Ran 3 tests in 0.001s

    OK

Note that the output indicated that three tests were run, despite us only writing a single test function called
``test_amazing_code_test_one()``.
A closer inspection reveals that the test function was run on three separate devices: ``"cpu"``, ``"cuda:0"``, and
``cuda:1``. This is a result of calling ``add_function_test()`` in our test script with the `devices=devices` argument.
``add_function_test()`` is defined in ``warp/tests/unittest_utils.py``.

A caveat of using ``add_function_test()`` is that this by itself is not sufficient to ensure that the registered test
function (e.g. `test_amazing_code_test_one()`) is run on different devices. It is up to the body of the test to make use
of the ``device`` argument in ensuring that data is allocated on and kernels are run on the intended ``device`` for the
test, e.g.

.. code-block:: python

    def test_amazing_code_test_one(test, device):
        with wp.ScopedDevice(device):
            score = wp.zeros(1, dtype=float, requires_grad=True)

or

.. code-block:: python

    def test_amazing_code_test_one(test, device):
        score = wp.zeros(1, dtype=float, requires_grad=True, device=device)

Checking for Expected Behaviors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Due to the use of the test-registration function ``add_function_test()``, the ``test`` parameter actually refers to the
instance of the test class, which always subclasses ``unittest.TestCase``.

The ``unittest`` library also provides methods to check that assertions are raised, as it is also important to test code
paths that trigger errors. The `assertRaises() <https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertRaises>`__
and `assertRaisesRegex() <https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertRaisesRegex>`__
methods can be used to test that a block of code correctly raises an exception.

Sometimes we need to compare the contents of a Warp array with an expected result.
Some functions that are helpful include:

* ``assert_np_equal()``: Accepts two NumPy arrays as input parameters along with an optional absolute tolerance ``tol``
  defaulted to 0. If the tolerance is 0, the arrays are compared using ``np.testing.assert_array_equal()``. Otherwise,
  both NumPy arrays are flattened and compared with ``np.testing.assert_allclose()``.
* ``assert_array_equal()``: Accepts two Warp arrays as input parameters, converts each array to a NumPy array on the
  CPU, and then compares the arrays using ``np.testing.assert_equal()``.
* ``wp.expect_eq()``: Unlike the previous two functions, the array(s) are to be compared by running a Warp kernel
  so the data can remain in the GPU. This is important if the array is particularly large that an element-wise
  comparison on the CPU would be prohibitively slow.

Skipping Tests
^^^^^^^^^^^^^^

Warp needs to be tested on multiple operating systems including macOS, on which NVIDIA GPUs are not supported.
When it is not possible for a particular test to be executed on *any* devices, there are some mechanisms to mark the
test as *skipped*.

``unittest`` provides some `methods <https://docs.python.org/3/library/unittest.html#skipping-tests-and-expected-failures>`__
to skip a test.

If the test function is added to a test class using ``add_function_test()``, we can pass an empty list as the argument
to the ``device`` parameter.

The final common technique is to avoid calling ``add_function_test`` on a test function in order to skip it.
Examples are `test_torch.py <https://github.com/NVIDIA/warp/blob/main/warp/tests/test_torch.py>`__,
`test_jax.py <https://github.com/NVIDIA/warp/blob/main/warp/tests/test_jax.py>`__, and
`test_dlpack.py <https://github.com/NVIDIA/warp/blob/main/warp/tests/test_dlpack.py>`__.
This technique is discouraged because the test is not marked as skipped in the ``unittest`` framework.
Instead, the test is treated as if it does not exist.
This can create a situation in which we are unaware that a test is being skipped because it does not show up under the
skipped tests count (it doesn't show up under the passed tests count, either).

Besides the situation in which a test requires CUDA, some examples for skipping tests are:

* ``usd-core`` is not installed in the current environment.
* The installed JAX version is too old.
* The system does not have at least two CUDA devices available (e.g. required for a multi-GPU test).

Tests Without a Device
^^^^^^^^^^^^^^^^^^^^^^

Recall that we previously discussed the use of ``add_function_test()`` to register a test function so that it can be
run on different devices (e.g. ``"cpu"`` and ``"cuda:0"``).
Sometimes, a test function doesn't make use of a specific device and we only want to run it a single time.

If we still want to use ``add_function_test()`` to register the test, we can pass ``devices=None`` to indicate that the
function does not make use of devices. In this case, the function will be registered only a single time to the test
class passed to ``add_function_test()``.

An alternative is to avoid the use of ``add_function_test()`` altogether and define the test function inside the
test class *directly*.
Taking our previous example with ``TestAmazingCode``, instead of the class body simply being
``pass``, we can add a device-agnostic function:

.. code-block:: python

    class TestAmazingCode(unittest.TestCase):
        def test_amazing_code_no_device(self):
            self.assertEqual(True, True)

This technique can be more readable to some developers because it avoids the obfuscation of
``add_function_test(..., device=None)``.
After all, ``add_function_test()`` is used to facilitate the execution of a single test function on different devices
instead of having to define a separate function for each device.
