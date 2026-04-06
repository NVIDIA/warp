Contribution Guide
==================

Some ways to contribute to the development of Warp include:

* Reporting bugs and requesting new features on `GitHub <https://github.com/NVIDIA/warp/issues>`__.
* Asking questions, sharing your work, or participating in discussion threads on
  `GitHub <https://github.com/NVIDIA/warp/discussions>`__.
* Adding new examples to the Warp repository.
* Documentation improvements.
* Contributing bug fixes or new features.
* Adding your work to the :doc:`publications list </user_guide/publications>`.

.. _before-you-start:

Before You Start
----------------

For small fixes such as typos, broken links, or minor documentation improvements,
feel free to open a pull request directly.

For bug fixes, feature contributions, or any non-trivial change:

* **Check existing issues first.** Search `GitHub Issues <https://github.com/NVIDIA/warp/issues>`__
  to see if someone is already working on it. If an issue is already assigned to
  someone else, coordinate in the issue thread before starting your own implementation.
* **Propose new features before implementing them.** Open a
  `GitHub Issue <https://github.com/NVIDIA/warp/issues/new>`__ or
  `Discussion <https://github.com/NVIDIA/warp/discussions>`__ to describe what you
  want to build and get feedback. Not all features are a good fit for Warp, and
  early discussion avoids wasted effort.
* **Gauge the complexity of what you're picking up.** Some areas of the codebase
  (e.g. JAX/PyTorch interop, code generation, CUDA runtime internals) involve subtle
  interactions that may not be apparent from a GitHub issue description alone. If you
  are not already familiar with the subsystem, comment on the issue to ask for context
  before investing time in a pull request.

.. _code-contributions:

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

Overview
^^^^^^^^

#. Create a fork of the Warp GitHub repository by visiting https://github.com/NVIDIA/warp/fork
#. Clone your fork on your local machine, e.g. ``git clone git@github.com:username/warp.git``.
#. Create a ``username/short-description`` branch for your contribution.

#. Make your desired changes.

   * Please familiarize yourself with the :ref:`coding-guidelines`.
   * Ensure that code changes pass :ref:`linting and formatting checks <linting-and-formatting>`.
   * Test cases should be written to verify correctness (:ref:`testing-warp`).
   * Documentation should be added for new features (:ref:`building-docs`).
   * Add an entry to the unreleased section at the top of the
     `CHANGELOG.md <https://github.com/NVIDIA/warp/blob/main/CHANGELOG.md>`__ describing the changes.

#. Prepare your commits.

   * Use imperative mood in commit messages (e.g. "Fix array bounds check", not "Fixed array bounds check").
   * Keep the subject line under ~50 characters. Use the body to explain *why* the change
     was made, not *what* changed (the diff shows that).
   * Reference related GitHub issues in the commit message, e.g. ``(GH-1234)``.
   * Sign off every commit with ``git commit --signoff`` (or ``-s``) to certify the
     :ref:`Developer Certificate of Origin <code-contributions>`.
   * **Clean up your commit history** before the pull request can be merged.
     Pull requests naturally accumulate merge commits, review fixups, and
     work-in-progress saves. These clutter ``git log`` and make
     ``git bisect`` and ``git cherry-pick`` less effective on ``main``.

     Organize your branch into a small number of logical, self-contained commits
     (often just one). Each commit should represent a coherent unit of change,
     pass tests on its own, and have a clear commit message. A multi-commit branch
     is fine when the commits tell a meaningful story (e.g. "Add new API" followed
     by "Add tests for new API"), but there is no reason to keep fixup or merge
     commits around.

     The simplest way to clean up a branch is to squash everything into one commit:

     .. code-block:: bash

        # Squash all commits on your branch into one
        git fetch https://github.com/NVIDIA/warp.git main
        git reset --soft FETCH_HEAD
        git commit -s

#. Push your branch to your GitHub fork, e.g. ``git push origin username/feature-name``.
#. Submit a pull request on GitHub to the ``main`` branch (:ref:`pull-requests`).
   Work with reviewers to ensure the pull request is in a state suitable for merging.

.. _quality-expectations:

Contribution Quality Expectations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To keep the review process efficient and the codebase healthy, please ensure your
contribution meets these expectations:

* **Include tests.** Bug fixes should include tests that reproduce the failure without
  the fix and pass with it. New features need tests that cover the core functionality
  and important edge cases. Pull requests that modify core library code without
  adequate test coverage will not be merged. That said, every test adds to the suite
  runtime, so focus on meaningful tests that verify real behavior rather than
  exhaustively testing trivial variations. See :ref:`testing-warp` for guidance.
* **Follow the pull request template.** Fill out all relevant sections, including
  a test plan that explains how you verified the changes. This also helps reviewers
  reproduce and confirm your results. See :ref:`pull-requests`.
* **Keep changes focused.** A pull request should address a single bug fix, feature, or
  improvement. Do not bundle unrelated changes; split them into separate PRs.
* **Minimize review cost.** Well-tested, clearly described contributions with clean
  commit history get reviewed faster. Contributions that require substantial reviewer
  effort to verify correctness may be deprioritized or declined.

.. _coding-guidelines:

Coding Guidelines
^^^^^^^^^^^^^^^^^

General Guidelines
""""""""""""""""""

* Include the NVIDIA SPDX copyright header on all newly created files, updating the year to the current year at the
  time of initial file creation. Use the two-line form::

    # SPDX-FileCopyrightText: Copyright (c) <YEAR> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    # SPDX-License-Identifier: Apache-2.0

  (Use ``//`` instead of ``#`` for C/C++/CUDA files.)
* Aim for consistency in variable and function names.

  * Use the existing terminology when possible when naming new functions (e.g. use ``points`` instead of ``vertex_buffer``).
  * Don't introduce new abbreviations if one already exists in the code base.
  * Also be mindful of consistency and clarity when naming local function variables.

* Avoid generic function names like ``get_data()``.
* Prioritize matching the existing style and conventions of the file being modified to maintain consistency.
* Avoid introducing new dependencies. New required dependencies are not accepted. Optional
  dependencies may be considered on a case-by-case basis (e.g. for examples or interop),
  but even optional dependencies require an internal license review of the package and all
  of its transitive dependencies to ensure compatibility with Warp's license. Discuss in
  the issue or PR before adding one.

Python Guidelines
"""""""""""""""""

* Follow `PEP 8 <https://peps.python.org/pep-0008/>`__ as the baseline for coding style.
* Use `snake case <https://en.wikipedia.org/wiki/Snake_case>`__ for all function names.
* Use `Google-style docstrings <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`__.
* Use both ``inputs`` and ``outputs`` parameters in :func:`wp.launch() <warp.launch>` in functions that are expected to be used in
  differentiable programming applications to aid in visualization and debugging tools.

C++ Guidelines
""""""""""""""

* Follow the clang-format style configuration in ``.clang-format`` (WebKit-based style with 120-character line limit).

  * WebKit brace style: Function braces on new line, control statement braces on same line
  * Left-aligned pointers and references: ``Type* var`` not ``Type *var``
  * No indentation inside namespaces
  * Automatic include sorting with ``warp.h`` always first

* **Symbol Naming**: All exported C/C++ symbols (functions, global variables) must be prefixed with ``wp_``
  to prevent namespace pollution. This is enforced by a GitLab CI job that inspects the compiled libraries.

  * Good: ``wp_init()``, ``wp_cuda_launch_kernel()``, ``wp_graph_coloring()``
  * Bad: ``init()``, ``cuda_launch_kernel()``, ``graph_coloring()``
  * Note: Symbols inside the ``wp`` namespace in header files don't need this prefix, but C symbols and ``extern "C"``
    functions exported from implementation files must use the ``wp_`` prefix

.. _linting-and-formatting:

Linting and Formatting
^^^^^^^^^^^^^^^^^^^^^^

`Ruff <https://docs.astral.sh/ruff/>`__ is used as the linter and code formatter for Python code in the Warp repository.
The contents of pull requests will automatically be checked to ensure adherence to our formatting and linting standards.

We recommend running the linters and formatters locally on your branch before opening a pull request.
From the project root, run:

.. code-block:: bash

    uvx pre-commit run --all-files

This command will attempt to fix any lint violations and then format the code.
Some lint violations cannot be `fixed automatically <https://docs.astral.sh/ruff/linter/#fix-safety>`__
and will require manual resolution.

To run linting and formatting checks automatically at ``git commit`` time, install the
pre-commit hooks:

.. code-block:: bash

    uvx pre-commit install

C++ Code Formatting
^^^^^^^^^^^^^^^^^^^

C++ code in the Warp repository must adhere to the clang-format style defined in ``.clang-format``.
The CI/CD pipeline will automatically check C++ formatting compliance for all pull requests.

**Using clang-format**

The same pre-commit setup handles both Python (Ruff) and C++ (clang-format) code:

.. code-block:: bash

    # Run all formatters and linters (Python and C++)
    uvx pre-commit run --all-files

    # Run only clang-format on C++ files
    uvx pre-commit run clang-format --all-files

**Optional: Installing clang-format locally**

Pre-commit automatically downloads and uses the correct clang-format version, so manual
installation is not required. However, if you want to install clang-format locally for IDE
integration (e.g., VS Code format-on-save), install the version specified in ``.pre-commit-config.yaml``:

.. code-block:: bash

    # Ubuntu/Debian - Install from apt.llvm.org
    # See https://apt.llvm.org/ for repository setup instructions
    # Check .pre-commit-config.yaml for the current version
    sudo apt-get install clang-format-21

For other platforms, consult the LLVM documentation for installation instructions. We recommend using pre-commit
for formatting, which handles version management automatically across all platforms.

**Running clang-format directly:**

If you have clang-format installed locally, you can run it directly from the command line:

.. code-block:: bash

    # Format all C++ files in warp/ (from repository root)
    # Note: nanovdb files are automatically skipped due to DisableFormat in nanovdb/.clang-format
    clang-format -i warp/**/*.{h,cpp,cu}

    # Format a specific file
    clang-format -i warp/native/mesh.h

**Disabling clang-format for specific code blocks:**

In rare cases where automatic formatting would break critical formatting (e.g., dependency-sensitive include order,
carefully aligned matrices), you can disable clang-format for a specific section:

.. code-block:: cpp

    // clang-format off
    #include "vec.h"
    #include "mat.h"
    #include "quat.h"
    // clang-format on

Use this sparingly. Add a comment explaining why if the reason isn't obvious. See ``warp/native/builtin.h`` for examples.

**Note:** The NanoVDB directory (``warp/native/nanovdb/``) is third-party code and automatically excluded via its own ``.clang-format`` configuration.

.. _building-docs:

Building the Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Warp library should first be built locally by running ``build_lib.py`` before building the Sphinx documentation.
The documentation can then be built by running the following from the project root:

.. code-block:: bash

    uv run --extra docs build_docs.py

The default behavior skips running the
`doctest tests <https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html>`__,
which take approximately 2 minutes to run.
If your changes modify core library functionality, it can be a good idea to run ``build_docs.py``
with the ``--doctest`` flag to ensure that the documentation code snippets are still up to date.
The ``--no-html`` flag can also be used to skip building the HTML documentation, e.g.

.. code-block:: bash

    # Run only the doctest tests
    uv run --extra docs build_docs.py --no-html --doctest

    # Build the HTML documentation AND run the doctest tests
    uv run --extra docs build_docs.py --doctest

Running ``build_docs.py`` also regenerates both the stub file (``warp/__init__.pyi``) and the reStructuredText files for the
reference pages. After building the documentation, it is recommended to run a ``git status`` to
check if your changes have modified these files. If so, please commit the modified files to your branch.

.. _pull-requests:

Pull Request Guidelines
^^^^^^^^^^^^^^^^^^^^^^^

* Use the pull request template provided by the repository. Fill out all
  applicable sections and delete the ones that do not apply.
* Ensure your pull request has a descriptive title that clearly states the purpose of the changes.
* Include a brief description covering:

  * Summary of changes.
  * Areas affected by the changes.
  * The problem being solved.
  * Any limitations or non-handled areas in the changes.
  * Any existing GitHub issues being addressed by the changes (use "closes #1234" syntax).

Design Documents
^^^^^^^^^^^^^^^^

For complex features (new user-facing APIs, architectural changes, or features with
non-obvious design trade-offs), consider adding a design document in the ``design/``
directory at the repository root. See ``design/README.md`` for guidelines and the
template.

Reviewers may request a design document during the review process for changes that
would benefit from one.

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

After building the Warp library (``uv run build_lib.py`` from the project root), run the test suite using
``uv run --extra dev -m warp.tests``. The tests should take approximately 10–20 minutes to run. By default, only the test modules
defined in ``default_suite()`` (in ``warp/tests/unittest_suites.py``) are run. To run the test suite
using `test discovery <https://docs.python.org/3/library/unittest.html#test-discovery>`__, use
``uv run --extra dev -m warp.tests -s autodetect``, which will discover tests in modules matching the path
``warp/tests/test*.py``.

Running a subset of tests
"""""""""""""""""""""""""

Instead of running the full test suite, there are two main ways to select a subset of tests to run.
These options must be used with the ``-s autodetect`` option.

Use ``-p PATTERN`` to define a pattern to match test files.
For example, to run only tests that have ``mesh`` in the file name, use:

.. code-block:: bash

    uv run --extra dev -m warp.tests -s autodetect -p '*mesh*.py'

Use ``-k TESTNAMEPATTERNS`` to define `wildcard test name patterns <https://docs.python.org/3/library/unittest.html#unittest.TestLoader.testNamePatterns>`__.
This option can be used multiple times.
For example, to run only tests that have either ``mgpu`` or ``cuda`` in their name, use:

.. code-block:: bash

    uv run --extra dev -m warp.tests -s autodetect -k 'mgpu' -k 'cuda'

Adding New Tests
^^^^^^^^^^^^^^^^

For tests that should be run on multiple devices, e.g. ``"cpu"``, ``"cuda:0"``, and ``"cuda:1"``, we recommend
first defining a test function at the module scope and then using ``add_function_test()`` to add multiple
test methods (a separate method for each device) to a test class.

Always add new test modules to ``default_suite()`` in ``warp/tests/unittest_suites.py``
so they are included in the default test run.

.. important::

   Never call ``wp.clear_kernel_cache()`` or ``wp.clear_lto_cache()`` in test files —
   not in ``__main__`` blocks, test methods, or at module scope. Cache clearing is not
   multi-process-safe; concurrent clears cause LLVM crashes. The test suite runner and
   ``build_lib.py`` already handle cache management.

.. code-block:: python

    # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    # SPDX-License-Identifier: Apache-2.0

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
        unittest.main(verbosity=2)

If we directly run this module, we get the following output:

.. code-block:: bash

    uv run test_amazing_code.py
    Warp 1.11.1 initialized:
    CUDA Toolkit 12.9, Driver 13.1
    Devices:
        "cpu"      : "x86_64"
        "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (48 GiB, sm_89, mempool enabled)
        "cuda:1"   : "NVIDIA RTX 6000 Ada Generation" (48 GiB, sm_89, mempool enabled)
    CUDA peer access:
        Supported fully (all-directional)
    Kernel cache:
        /home/nvidia/.cache/warp/1.11.1
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
Examples are `test_torch.py <https://github.com/NVIDIA/warp/blob/main/warp/tests/interop/test_torch.py>`__,
`test_jax.py <https://github.com/NVIDIA/warp/blob/main/warp/tests/interop/test_jax.py>`__, and
`test_dlpack.py <https://github.com/NVIDIA/warp/blob/main/warp/tests/interop/test_dlpack.py>`__.
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

.. _benchmarks:

Benchmarks
----------

Warp uses `airspeed velocity (ASV) <https://asv.readthedocs.io/>`__ for performance benchmarking.
Benchmark scripts live in the ``asv/benchmarks/`` directory, and the ASV configuration is in
``asv.conf.json``.

Running Benchmarks
^^^^^^^^^^^^^^^^^^

To run all benchmarks against the current commit:

.. code-block:: bash

    uvx --python 3.12 asv run -e --launch-method spawn HEAD^!

To run a specific benchmark against the current commit:

.. code-block:: bash

    uvx --python 3.12 asv run -e --launch-method spawn -b BenchmarkClassName HEAD^!

To run benchmarks across a range of commits (from ``older_commit`` to
``newer_commit``, inclusive):

.. code-block:: bash

    uvx --python 3.12 asv run -e --launch-method spawn -b BenchmarkClassName older_commit..newer_commit

On Linux, ASV defaults to ``forkserver``, which isolates each benchmark run with
``os.fork()`` and terminates the child via ``os._exit()``. Because ``os._exit()``
bypasses Python's normal interpreter shutdown, ``TemporaryDirectory`` finalizers
never run, and NVRTC precompiled-header directories (``/tmp/wp_pch_*``,
``/tmp/__nvrtc_auto_pch_*``) accumulate in ``/tmp``. These are cleaned up on
reboot, so this is mainly a concern on long-running systems. Pass
``--launch-method spawn`` to avoid the issue.
