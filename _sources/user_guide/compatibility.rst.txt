########################
Compatibility & Support
########################

This page is organized into two main sections:

1. **Platform compatibility**: What platforms, operating systems, and hardware Warp currently supports
2. **Support policies**: How Warp evolves over time, including versioning, deprecation, and lifecycle management

Platform compatibility
======================

This section describes the platforms, operating systems, and GPU architectures that Warp supports, 
along with the requirements for building and running Warp.

Supported platforms
-------------------

Warp supports a range of operating systems and CPU architectures, with GPU acceleration available 
on platforms with NVIDIA CUDA support. The following table summarizes platform support:

.. table:: Platform Compatibility Matrix
    :align: left

    +---------------------+--------------+----------------------+--------------------+---------------------+
    | Operating System    | Architecture | Platform / Device    | Platform Support   | Accelerator Support |
    +=====================+==============+======================+====================+=====================+
    | **Windows** 10/11   | ``x86-64``   | PC                   | Supported ✅       | CUDA                |
    +---------------------+--------------+----------------------+--------------------+---------------------+
    | **Windows** 11      | ``arm64``    | PC                   | Planned            | CUDA (Planned)      |
    +---------------------+--------------+----------------------+--------------------+---------------------+
    | **Linux**           | ``x86-64``   | PC / Server          | Supported ✅       | CUDA                |
    +---------------------+--------------+----------------------+--------------------+---------------------+
    | **Linux**           | ``aarch64``  | Jetson / Server      | Supported ✅       | CUDA                |
    +---------------------+--------------+----------------------+--------------------+---------------------+
    | **macOS**           | ``arm64``    | Apple Silicon        | Supported ✅       | CPU Only            |
    +---------------------+--------------+----------------------+--------------------+---------------------+
    | **macOS**           | ``x86-64``   | Intel-based Mac      | Discontinued       | None                |
    +---------------------+--------------+----------------------+--------------------+---------------------+

Runtime requirements
--------------------

The following requirements apply when running Warp:

**Python**: Version 3.10 or newer

**Dependencies**: `NumPy <https://numpy.org>`__ (required)

**Operating Systems**:

* Windows 10/11
* Linux (GLIBC 2.28+ for ``x86-64``, GLIBC 2.34+ for ``aarch64``)
* macOS 11.0+

**GPU Acceleration**:

.. table:: GPU Compute Capability Requirements
    :align: left

    +----------------------------+---------------------+------------------------+------------------------------------------+
    | CUDA Toolkit Version       | Minimum GPU         | Compute Capability     | Example GPUs                             |
    +============================+=====================+========================+==========================================+
    | CUDA 12.x                  | Maxwell             | 5.2 (``sm_52``)        | GeForce GTX 9xx series                   |
    +----------------------------+---------------------+------------------------+------------------------------------------+
    | CUDA 13.x                  | Turing              | 7.5 (``sm_75``)        | GeForce RTX 20xx series                  |
    +----------------------------+---------------------+------------------------+------------------------------------------+

To determine your GPU's compute capability, see `NVIDIA CUDA GPUs <https://developer.nvidia.com/cuda-gpus>`__.

* **Driver Requirements**: The driver requirements are determined by the CUDA Toolkit version used to build the Warp
  library, not the version installed on the system when running Warp.

  * Warp packages built with CUDA Toolkit 12.x require NVIDIA driver 525 or newer.
  * Warp packages built with CUDA Toolkit 13.x require NVIDIA driver 580 or newer.
  * PyPI wheels for Windows/Linux are currently built with CUDA 12.9.

* **Advanced Features**: Half-precision (``float16``) atomic operations require compute capability 7.0+ (Volta). 
  On older GPUs, these operations return zero.

* **CPU-only execution** is supported on all platforms for users without a GPU.

Building from source
--------------------

To build Warp from source, you need:

**Compilers:**

* Windows: Microsoft Visual Studio 2019+
* Linux: GCC 9.4+ (C++17 support)
* macOS: Xcode Command Line Tools

**Core Tools:**

* Python 3.10+
* `Git LFS <https://git-lfs.com>`__ (for assets used in tests, examples, and documentation)
* NumPy

**Optional:**

* CUDA Toolkit (required to build Warp with GPU support, not available on macOS):

  * Minimum: CUDA Toolkit 12.0
  * For full libmathdx support: CUDA Toolkit 12.6.3+
  * For conditional graph node support: CUDA Toolkit 12.4+

* libmathdx (auto-fetched via Packman by default)
* LLVM/Clang (auto-fetched via Packman by default)

**Building:**

.. code-block:: console

    $ python build_lib.py
    $ pip install -e .

Run ``python build_lib.py --help`` for build options like ``--cuda-path``, ``--mode``, and ``--verbose``.

For detailed instructions, see :doc:`installation`.

Build tools
~~~~~~~~~~~

The build script (``build_lib.py``) and related build infrastructure are developer-facing tools for building
Warp from source. These scripts have no versioning guarantees:

* **No API Stability**: Command-line options, arguments, and programmatic interfaces may change between
  releases without advance notice or deprecation warnings.
* **No Backward Compatibility**: Build processes that work with one version may require adjustments for
  the next version.
* **Self-Documentation**: Use ``python build_lib.py --help`` to see available options for your version.

Users building Warp from source should expect build script interfaces to change between releases and
use ``--help`` to discover current options.

Support policies
================

This section describes how Warp manages change over time, including versioning, API stability, 
deprecation practices, and platform support lifecycle. Understanding these policies helps you plan 
upgrades and anticipate breaking changes.

Versioning
----------

Warp library versions take the format X.Y.Z, but Warp does not follow
`Semantic Versioning <https://semver.org/>`__. In practice, only the Y and Z numbers are bumped:

* **X** is a "marketing" number reserved for major reworks of the Warp library causing disruptive
  incompatibility.
* **Y** is bumped for a **feature release**, published on a regular monthly cadence. Feature releases
  introduce new features and are the only releases that may contain deprecations, breaking changes,
  and removals.
* **Z** is bumped for a **bugfix release**. Bugfix releases are not regularly scheduled and are issued
  only when an important issue cannot wait for the next feature release. They do not contain new
  features, deprecations, or removals.

**Prerelease versions**

In addition to stable releases, Warp uses the following prerelease version formats:

* **Development builds** (``X.Y.Z.dev0``): The version string used in the source code on the main branch between 
  stable releases (e.g., ``1.11.0.dev0``).
* **Release candidates** (``X.Y.ZrcN``): Pre-release versions used for QA testing before a stable
  release, starting with ``rc1`` and incrementing (e.g., ``1.10.0rc1``). Release candidates are not
  regularly published externally, but may be at our discretion.
* **Nightly builds** (``X.Y.Z.devYYYYMMDD``): Automated builds from the main branch published on the 
  `NVIDIA PyPI index <https://pypi.nvidia.com>`__ with the build date appended (e.g., ``1.11.0.dev20251030``).

Prerelease versions should be considered unstable and are not subject to the same compatibility guarantees as stable releases.

Component states
----------------

All supported components (e.g. API functions, Python versions, GPU architectures) exist in one of the following states:

* **Experimental**: A new feature that is still under active development.
  It is available for early adopters to test and provide feedback, but its design may change with little or no notice,
  including in bugfix releases.
* **Stable**: The default state for most features. Changes to stable features follow our deprecation timeline 
  and backward compatibility guarantees.
* **Deprecated**: A feature that is scheduled for removal in a future release.
  It remains fully functional for the time being.
* **Removed**: A feature that is no longer part of the library after its deprecation period has ended.
  Attempting to use a removed feature will result in an error.

Deprecation timeline
--------------------

A deprecated feature will typically be maintained for **at least 4 months** after its deprecation
is announced. At the monthly feature release cadence, that gives users roughly 4 feature releases
during which they will see a ``DeprecationWarning`` and can migrate. Deprecations and removals occur
only in feature releases, never in bugfix releases.

**Example timeline** (assuming the monthly feature release cadence):

* ``v1.8.0`` (month 0): A feature is deprecated. Its deprecation is noted in the changelog and using
  it will result in a ``DeprecationWarning`` (if applicable).
* ``v1.8.1`` (ad-hoc bugfix release, if issued): The deprecated feature remains fully functional.
* ``v1.9.0`` (month 1): The deprecated feature remains fully functional.
* ``v1.10.0`` (month 2): The deprecated feature remains fully functional.
* ``v1.11.0`` (month 3): The deprecated feature remains fully functional. The release announcement
  flags the upcoming removal in the next release.
* ``v1.12.0`` (month 4): Feature is removed. Using it will result in an error.

While we strive to abide by the above deprecation timeline, there may be circumstances in which a feature is removed
with a briefer or longer deprecation period.

How deprecations are communicated
---------------------------------

We employ multiple channels to ensure users are aware of deprecated and removed features:

**CHANGELOG.md**
    The primary source of truth for all deprecations and removals. Each release's changelog includes
    dedicated **Deprecated** and **Removed** sections. The changelog is available in the GitHub repository
    and at :doc:`changelog`.

**Runtime Warnings**
    When you use a deprecated feature in your code, Warp will emit a ``DeprecationWarning`` to
    stdout the first time the feature is used in your Python session. These warnings include
    information about the deprecated feature and, when applicable, suggest an alternative. To
    include source file and line number information in the warning output, set
    ``warp.config.verbose_warnings = True`` before calling into Warp.

**API Documentation**
    Deprecated features are marked with ``.. deprecated:: X.Y`` directives in the API documentation,
    indicating the version in which deprecation was introduced.

**GitHub Releases**
    Release announcements on https://github.com/NVIDIA/warp/releases. When a deprecated feature is
    scheduled for removal in the next feature release, the preceding release's announcement will flag
    the upcoming removal.

What to do when you see a deprecation warning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Read the warning message for the suggested replacement, if any.
2. Check the **Deprecated** section of :doc:`changelog` to find the release in which the
   deprecation was first announced. Combined with the deprecation timeline above, this tells you
   roughly when the feature will be removed.
3. Migrate your code to the replacement API. The deprecated feature will remain functional for at
   least 4 months, giving you multiple feature releases to complete the migration.
4. If migration is blocked by a gap in the replacement, raise a `GitHub issue
   <https://github.com/NVIDIA/warp/issues>`__.

Your installed Warp version is available via ``warp.config.version`` (or ``warp.__version__``).

Release support policy
----------------------

Only the latest feature release line is actively maintained:

* **Active Support**: Only the most recent feature release line (e.g., ``v1.10.x``) is eligible to
  receive bugfix releases. Bugfix releases are issued ad-hoc, only when an important issue cannot
  wait for the next feature release.
* **Limited Backporting**: By default, fixes are not backported to earlier feature release lines.
  In exceptional cases where a user cannot upgrade, we may backport a critical fix on a
  case-by-case basis. If this applies to you, raise a GitHub issue.
* **Upgrade Path**: Users who encounter bugs or need fixes should upgrade to the latest feature
  release. Because bugfix releases are not regularly scheduled, the next feature release is often
  the fastest route to a fix. We document breaking changes and follow the deprecation timeline to
  make upgrades predictable.

Component-specific policies
---------------------------

The following subsections detail support policies for specific components of the Warp ecosystem, 
including the API surface, Python versions, dependencies, CUDA toolkit versions, GPU architectures, 
and operating systems.

API
~~~

Warp's **public API** consists of symbols that are documented in the :doc:`API reference </user_guide/runtime>`
and accessible without underscore prefixes (e.g., ``warp.function_name()``, ``warp.fem.ClassName``).

The public API is designed to be stable and backward compatible. Changes follow the deprecation
timeline described earlier.

**Private APIs**, such as anything under ``warp._src.*`` or any symbols with underscore prefixes 
(e.g., ``warp._private_function()``), can change or be removed without following the deprecation timeline. 
Only the public API is subject to the deprecation policies described in this document.

Python versions
~~~~~~~~~~~~~~~

Warp aims to support Python versions that are fully released (in "bugfix" or "security" status according
to the `Python version lifecycle <https://devguide.python.org/versions/>`__). 

Support for newly released Python versions is added in the next Warp feature release after the
Python version's stable release (e.g., if Python 3.15.0 is released during the Warp 1.10.x series,
support will be added in Warp 1.11.0).

When a Python version reaches end-of-life, we follow the deprecation timeline described above
before dropping support, allowing users time to migrate.

Dependencies
~~~~~~~~~~~~

**Required dependencies**

NumPy is the only required runtime dependency for Warp. While we do not follow a strict version 
support policy such as `SPEC 0 <https://scientific-python.org/specs/spec-0000/>`__, we generally 
support a wide range of NumPy versions. The minimum supported NumPy version may be updated in
Warp feature releases following our standard deprecation practices.

**Optional dependencies**

Warp includes optional features and examples that depend on additional packages (e.g., ``usd-core``, 
``matplotlib``, ``warp-lang[examples]``). For these optional dependencies:

* We aim to test against recent versions in our CI/CD pipeline.
* Version support may be adjusted if specific package versions cause CI/CD issues or compatibility 
  problems on certain platforms.
* Users can typically use a range of versions; minimum version requirements are documented in 
  ``pyproject.toml`` when applicable.

**Interoperability libraries**

Warp provides interoperability with machine learning frameworks (PyTorch, JAX, Paddle). For these 
integrations:

* **Target Support**: We aim to support the latest stable release of each framework at the time of 
  a Warp release.
* **Backward Compatibility**: Older framework versions are supported where practical, but support
  may be dropped over time (particularly for rapidly evolving frameworks like JAX). Any changes to
  supported versions are noted in the changelog.
* **Testing**: Our CI/CD pipeline tests against recent versions of these frameworks, though we cannot
  exhaustively test every version combination.

Users experiencing compatibility issues should consult the changelog and GitHub issues for known
problems and workarounds.

CUDA versions
~~~~~~~~~~~~~

Warp supports the two most recent CUDA major versions. Features with specific CUDA version
requirements are noted in the documentation.

Pre-built wheels are available for the two supported CUDA major versions, although the specific minor version
used is set at our discretion.

The specific CUDA minor version used for PyPI wheels is selected at our discretion based on what we
believe is most useful for the average user.

NVIDIA driver compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NVIDIA driver requirements are determined by the CUDA Toolkit version used to build Warp, not by Warp itself.
We follow NVIDIA's driver compatibility policies:

* **Minimum Driver Requirements**: Each CUDA Toolkit version has a minimum required driver version. Warp 
  packages built with a specific CUDA Toolkit inherit that minimum driver requirement. See the Driver 
  Requirements bullet in the Runtime requirements section for current minimums.

* **Forward Compatibility**: NVIDIA drivers are forward-compatible with older CUDA Toolkit versions. This 
  means a system with a newer driver (e.g., R580 for CUDA 13.x) can run Warp packages built with older CUDA 
  Toolkit versions (e.g., CUDA 12.8). However, a system with an older driver (e.g., R525 for CUDA 12.x) 
  cannot run Warp packages built with newer CUDA Toolkit versions (e.g., CUDA 13.x) that require a higher 
  minimum driver version.

* **No Backward Compatibility**: CUDA Toolkit versions are not backward-compatible with older drivers. Your 
  driver must meet the minimum requirement for the CUDA Toolkit version used to build your Warp installation. 
  Warp reports both the CUDA Toolkit version used to compile the loaded library and the CUDA Toolkit version 
  associated with your driver during initialization.

For the most up-to-date driver compatibility information, consult the 
`CUDA Compatibility Guide <https://docs.nvidia.com/deploy/cuda-compatibility/>`__.

GPU architectures
~~~~~~~~~~~~~~~~~

Warp's GPU architecture support is determined by the CUDA Toolkit version used to build the Warp library. 
The NVRTC (NVIDIA Runtime Compilation) component within the CUDA Toolkit establishes the minimum supported 
compute capability, while PTX (Parallel Thread Execution) forward compatibility allows Warp to target newer 
GPU architectures that were not explicitly compiled as cubin binaries. For more details on the CUDA 
compilation process, see the 
`CUDA Compiler Driver NVCC documentation <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/>`__.

**Architecture support mechanism**

* **Minimum Architecture**: The version of NVRTC in the CUDA Toolkit used to build Warp determines the 
  minimum GPU architecture that can run Warp kernels. See the GPU Compute Capability Requirements table 
  in the Runtime requirements section for current minimums.

* **Forward Compatibility**: Warp leverages PTX virtual architecture compilation to ensure compatibility 
  with newer GPU architectures. When Warp encounters a GPU with a compute capability higher than those 
  explicitly compiled as cubins, the CUDA driver can JIT-compile the PTX intermediate representation to 
  native code for that architecture.

* **Cubin Compilation**: Pre-built Warp packages include cubin binaries for a select set of common GPU
  architectures to minimize JIT compilation overhead. The specific architectures included are determined
  by build configuration and may change between feature releases. Within a feature release line, the
  architecture list is stable, except when a bugfix release must adjust it to work around a compiler
  or library issue affecting a specific architecture.

**Changes to architecture support**

GPU architecture support can change in two ways:

1. **Dropping CUDA Toolkit Support**: When Warp drops support for an older CUDA Toolkit major version 
   (typically coinciding with adding support for a new CUDA major version), the minimum supported compute 
   capability may increase to match the new minimum toolkit's requirements. Since CUDA Toolkit release 
   timelines are beyond Warp's control, such changes may occur with limited advance notice, but will
   always happen in a Warp feature release, never in a bugfix release.

2. **Changing PyPI Wheel Build Configuration**: When we decide to change the CUDA Toolkit version used to 
   build PyPI wheels, this affects the minimum GPU architecture supported by those wheels. Because this 
   change is within our control, we announce it in the release notes of the prior release, giving users 
   advance notice before the new wheels are published.

Users relying on older GPU architectures should monitor Warp's changelog and release announcements for 
information about upcoming architecture support changes.

Operating systems
~~~~~~~~~~~~~~~~~

Warp's supported operating systems are primarily determined by the platforms supported by NVIDIA CUDA:

* **Linux**: We support the distributions listed in the 
  `CUDA Installation Guide for Linux <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/>`__. 
  To ensure broad compatibility across Linux distributions, Warp wheels are built against the 
  `manylinux <https://github.com/pypa/manylinux>`__ standard. This approach provides binary 
  compatibility across a wide range of Linux distributions that meet the minimum GLIBC requirements 
  specified in the runtime requirements (GLIBC 2.28+ for ``x86-64``, GLIBC 2.34+ for ``aarch64``).

* **Windows**: We support the Windows versions listed in the 
  `CUDA Installation Guide for Windows <https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/>`__.

* **macOS**: Support follows the platform requirements outlined in the Platform compatibility 
  Matrix and Runtime requirements sections above.
