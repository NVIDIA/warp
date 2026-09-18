Installation
============

Warp requires Python 3.10 or newer. We publish ``warp-lang`` wheels on PyPI for Windows (x86-64), Linux (x86-64 and AArch64), and macOS (Apple Silicon). The Windows x86-64 and Linux wheels support CPU execution and CUDA acceleration. The macOS wheels support CPU execution but not Metal acceleration.

PyPI and nightly wheels for Linux and Windows use CUDA Toolkit 13.4. They require an
NVIDIA R580-series or newer driver and a Turing (``sm_75``) or newer GPU for CUDA acceleration.
For CUDA 12 environments, download a ``+cu12`` wheel from :ref:`GitHub Releases <github-release-wheels>`
or :ref:`build Warp from source with CUDA 12 <building-from-source>`.

The easiest way to install Warp is from `PyPI <https://pypi.org/project/warp-lang>`_:

.. code-block:: sh

    $ pip install warp-lang

.. _GitHub Installation:

Nightly Builds
--------------

Nightly builds of Warp from the ``main`` branch are available on the `NVIDIA Package Index <https://pypi.nvidia.com/warp-lang/>`_.

To install the latest nightly build, use the following command:

.. code-block:: sh

    $ pip install -U --pre warp-lang --extra-index-url=https://pypi.nvidia.com/

Nightly builds use CUDA Toolkit 13.4 and have the same :ref:`cuda-requirements` as the default PyPI wheels.
They also include the CPU-only macOS Apple Silicon wheel.

If you plan to install nightly builds regularly, you can simplify future installations by adding NVIDIA's package
repository as an extra index via the ``PIP_EXTRA_INDEX_URL`` environment variable. For example:

.. code-block:: text

    export PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com"

This ensures the index is automatically used for ``pip`` commands, avoiding the need to specify it explicitly.

Conda Installation
------------------

Conda packages for Warp are also available on the `conda-forge <https://anaconda.org/conda-forge/warp-lang>`__ channel.
By default, the CUDA variant with the latest toolkit version is installed:

.. code-block:: sh

    $ conda install conda-forge::warp-lang

To install a specific variant, use a build string filter:

.. code-block:: sh

    # CPU-only (no CUDA dependencies)
    $ conda install conda-forge::warp-lang=*=*cpu*

    # CUDA 12.9
    $ conda install conda-forge::warp-lang=*=*cuda129*

For more information, see the community-maintained feedstock for Warp
`here <https://github.com/conda-forge/warp-lang-feedstock>`__.

.. _github-release-wheels:

Installing from GitHub Releases
-------------------------------

`GitHub Releases <https://github.com/NVIDIA/warp/releases>`_ provides both the default release wheels
and CUDA 12 compatibility wheels. The default Linux and Windows wheels use CUDA Toolkit 13.4
and have no CUDA suffix. CUDA 12 wheels have a ``+cu12`` version suffix.

Choose a ``warp_lang-<version>+cu12-py3-none-<platform>.whl`` asset for your operating system
and CPU architecture. Copy its URL from the release page and replace ``<wheel-url>`` below:

.. code-block:: sh

    $ pip install "<wheel-url>"

The platform tags are ``manylinux_2_34_aarch64`` for Linux AArch64,
``manylinux_2_28_x86_64`` for Linux x86-64, and ``win_amd64`` for Windows x86-64.
See :ref:`cuda-requirements` for driver requirements and :doc:`compatibility` for GPU architecture support.

The ``--force-reinstall`` option may need to be used to overwrite a previous installation.

.. _cuda-requirements:

CUDA Requirements
-----------------

* The default PyPI and nightly Linux and Windows wheels use CUDA Toolkit 13.4 and require
  an NVIDIA R580-series or newer driver and a Turing (``sm_75``) or newer GPU for CUDA acceleration.
* Warp packages built with CUDA Toolkit 12.x, including the ``+cu12`` wheels on
  :ref:`GitHub Releases <github-release-wheels>`, require an NVIDIA R525-series or newer driver.
* Warp packages built with CUDA Toolkit 13.x require an NVIDIA R580-series or newer driver
  and a Turing (``sm_75``) or newer GPU.

The CUDA Toolkit used to build Warp determines these requirements. Pre-built wheels include the CUDA
components that Warp needs, so they do not require a system CUDA Toolkit.
See :doc:`compatibility` for the GPU architecture requirements of each CUDA major version.

Note that building Warp with the ``--quick`` flag changes the driver requirements.
The quick build skips CUDA backward compatibility, so the minimum required driver is determined by the CUDA Toolkit version.
Refer to the `latest CUDA Toolkit release notes <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>`_
to find the minimum required driver for different CUDA Toolkit versions
(e.g., `this table from CUDA Toolkit 12.9 <https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html#id7>`_).

Warp checks the installed driver during initialization and will report a warning if the driver is not suitable, e.g.:

.. code-block:: text

    Warp UserWarning:
       Insufficient CUDA driver version.
       The minimum required CUDA driver version is 13.0, but the installed CUDA driver version is 12.9.
       Visit https://nvidia.github.io/warp/stable/user_guide/installation.html for guidance.

This will make CUDA devices unavailable, but the CPU can still be used.

To remedy the situation there are a few options:

* Update the driver.
* Install a ``+cu12`` compatibility wheel from :ref:`GitHub Releases <github-release-wheels>` if your system meets its requirements.
* :ref:`Build Warp from source <building-from-source>` using a CUDA Toolkit that's compatible with the installed driver and GPU.

Also note that full support for tile-based MathDx features requires CUDA version 12.6.3 or later. See :ref:`mathdx` for more information.

.. _cuda-12-arm-limitation:

CUDA 12.9 limitation on Linux ARM platforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When building Warp with CUDA 12.9 on Linux AArch64, the default architecture set omits
the ``sm_52``, ``sm_60``, ``sm_61``, and ``sm_70`` targets due to a
`bug <https://github.com/NVIDIA/cccl/issues/4967>`__ in the CUDA 12.9 Toolkit which limits the number of architectures that
can be compiled at once.

This also applies to Linux AArch64 ``+cu12`` wheels built with CUDA 12.9.
The Jetson targets ``sm_53``, ``sm_62``, and ``sm_72`` remain in the default architecture set;
running Warp on these devices still requires compatible platform and driver support.
If support for the omitted targets is required, build Warp using CUDA Toolkit 12.0 through 12.8.
CUDA 13.x requires Turing (``sm_75``) or newer for CUDA acceleration on all platforms.

Dependencies
------------

Warp supports Python versions 3.10 onwards. Note that :ref:`some optional dependencies may not support the latest version of Python<conda>`.

`NumPy <https://numpy.org>`_ must be installed.

The ``warp-lang[examples]`` extra installs dependencies used by many Warp examples. Install it with:

.. code-block:: sh

    $ pip install "warp-lang[examples]"

.. _openusd-dependencies:

**OpenUSD dependencies**

Some Warp examples and USD rendering features import the OpenUSD ``pxr`` modules. If those modules
are missing, Python may report ``ModuleNotFoundError: No module named 'pxr'``. The error reports the
import name; the PyPI distributions that provide these modules are ``usd-core`` and ``usd-exchange``.

The ``warp-lang[examples]`` extra uses platform markers to install ``usd-core`` where supported and
``usd-exchange`` on supported platforms without a compatible ``usd-core`` wheel.

``usd-exchange`` includes its own OpenUSD runtime and ``pxr`` modules. Do not install it alongside
``usd-core``. Both packages install files to the same locations, which can cause import or runtime
failures.

``usd-exchange`` has its own release version, separate from the version of the OpenUSD runtime it
includes. After installation, call ``pxr.Usd.GetVersion()`` to query the OpenUSD version.

Some examples need extra packages. Check the example's source file to see what else you need to
install. For JAX and PyTorch, follow the official
`JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`__ or
`PyTorch installation guide <https://pytorch.org/get-started/locally/>`__ to choose a version and
build for your system.

.. _building-from-source:

Building from Source
--------------------

For developers who want to build the library themselves, the following tools are required:

* (Windows) Microsoft Visual Studio, minimum version 2019
* (Linux) GCC, minimum version 9.4
* (macOS) Xcode Command Line Tools
* `Git Large File Storage <https://git-lfs.com>`_

A CUDA Toolkit is not required for a CPU-only build. CUDA-enabled builds on Windows and Linux require
`CUDA Toolkit <https://developer.nvidia.com/cuda/toolkit>`_ 12.0 or newer.
Building from source with CUDA 12 remains supported. Choose a toolkit compatible with your
driver and GPU; see :ref:`cuda-requirements` and the :ref:`cuda-12-arm-limitation`.

After cloning the repository, users should run:

.. code-block:: console

    $ python build_lib.py

Upon success, the script will output platform-specific binary files in ``warp/bin/``.

Unless a CUDA Toolkit path is provided explicitly, ``build_lib.py`` searches for one in this order:

#. ``WARP_CUDA_PATH``, ``CUDA_HOME``, then ``CUDA_PATH``
#. The CUDA Toolkit containing ``nvcc`` found on ``PATH``
#. The standard CUDA installation locations for the operating system

If no CUDA Toolkit is found, ``build_lib.py`` builds Warp without CUDA support.

By default, CUDA libraries (cudart, NVRTC, nvJitLink, MathDx) are linked statically
to produce self-contained binaries. To link against shared CUDA libraries instead,
pass ``--use-dynamic-cuda``:

.. code-block:: console

    $ python build_lib.py --use-dynamic-cuda

The corresponding shared libraries must be available at runtime when using this option.

After building, the Warp package should be installed using:

.. code-block:: console

    $ pip install -e .

The ``-e`` option is optional but ensures that subsequent modifications to the
library will be reflected in the Python package.

CMake build
~~~~~~~~~~~

Developers who have CMake installed can use the alternate CMake build path.
This builds the native Warp libraries in place, like ``build_lib.py``, while
letting CMake and the selected generator handle parallel and incremental
rebuilds.
The commands shown below require CMake 3.24 or newer and Ninja. CMake also uses
a Python 3.10+ environment with NumPy installed to regenerate derived native
headers. By default, CMake may fetch LLVM and ``libmathdx`` through Packman
unless explicit paths are provided or the corresponding features are disabled
(``-DWARP_BUILD_CLANG=OFF`` for LLVM, ``-DWARP_USE_LIBMATHDX=OFF`` for
``libmathdx``, or ``-DWARP_ENABLE_CUDA=OFF`` for a CPU-only build).

From the repository root, the recommended path uses
`uv <https://docs.astral.sh/uv/>`__ to prepare the Python environment before
configuring and building:

.. code-block:: console

    $ uv sync --no-install-project
    $ cmake -S . -B _build/cmake -G Ninja
    $ cmake --build _build/cmake --parallel

Without ``uv``, use a Python environment you manage and install NumPy before
running the CMake commands:

.. code-block:: console

    $ python -m pip install numpy
    $ cmake -S . -B _build/cmake -G Ninja
    $ cmake --build _build/cmake --parallel

Upon success, the CMake build writes the native libraries to ``warp/bin/``.
The default CMake build enables CUDA on Linux and Windows, disables CUDA on
macOS, and builds both ``warp`` and ``warp-clang``. Pass
``-DWARP_ENABLE_CUDA=OFF`` for a CPU-only CMake build. CUDA builds default to a
single PTX target for fast local builds; use ``CMAKE_CUDA_ARCHITECTURES`` to
select different GPU architectures. Use ``build_lib.py`` for release builds
that need Warp's full GPU architecture coverage.

To use a specific CUDA Toolkit:

.. code-block:: console

    $ cmake -S . -B _build/cmake -G Ninja -DWARP_CUDA_PATH=/usr/local/cuda
    $ cmake --build _build/cmake --parallel

To link against shared CUDA libraries in the CMake build:

.. code-block:: console

    $ cmake -S . -B _build/cmake -G Ninja -DWARP_USE_DYNAMIC_CUDA=ON
    $ cmake --build _build/cmake --parallel

The corresponding shared CUDA libraries, including ``libnvptxcompiler``, must
be available at runtime when using this option.

To use an existing LLVM installation for ``warp-clang``:

.. code-block:: console

    $ cmake -S . -B _build/cmake -G Ninja -DWARP_LLVM_PATH=/opt/llvm
    $ cmake --build _build/cmake --parallel

To use an existing ``libmathdx`` installation:

.. code-block:: console

    $ cmake -S . -B _build/cmake -G Ninja -DWARP_LIBMATHDX_PATH=/path/to/libmathdx
    $ cmake --build _build/cmake --parallel

To build for specific CUDA architectures:

.. code-block:: console

    $ cmake -S . -B _build/cmake -G Ninja -DCMAKE_CUDA_ARCHITECTURES="86;89"
    $ cmake --build _build/cmake --parallel

After building, verify that the libraries can be loaded:

.. code-block:: console

    $ uv run python -c "import warp; warp.print_diagnostics()"

The CMake path is a native library build path only. It does not replace the
Python package build backend used for wheels.

.. _conda:

Conda Environments
------------------

Some modules, such as ``usd-core``, don't support the latest Python version.
To manage running Warp and other projects on different Python versions one can
make use of an environment management system such as
`Conda <https://docs.conda.io/en/latest/>`__.

.. warning::

    When building and running Warp in a different environment, make sure
    the build environment has the same C++ runtime library version, or an older
    one, than the execution environment. Otherwise Warp's shared libraries may end
    up looking for a newer runtime library version than the one available in the
    execution environment. For example, on Linux this error could occur::

        OSError: <...>/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by <...>/warp/warp/bin/warp.so)

    This can be solved by installing a newer C++ runtime version in the runtime
    conda environment using ``conda install -c conda-forge libstdcxx-ng=12.1`` or
    newer.
    
    Alternatively, the build environment's C++ toolchain can be downgraded using
    ``conda install -c conda-forge libstdcxx-ng=8.5``. Or, one can ``activate`` or
    ``deactivate`` conda environments as needed for building vs. running Warp.

Using Warp in Docker
--------------------

Docker containers can be useful for developing and deploying applications that use Warp.
They provide build environment isolation and consistency benefits.

In order to have Warp detect GPUs from inside a Docker container, the
`NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html>`__
should be installed.
Pass the ``--gpus all`` flag to the ``docker run`` command to make all GPUs available to the container.

Building Warp from source in Docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build Warp from source in Docker, you should ensure that the container has either ``curl`` or ``wget`` installed.
This is required so that Packman can download dependencies like libmathdx and LLVM/Clang from the internet
when building Warp.

We recommend using one of the NVIDIA CUDA images from `nvidia/cuda <https://hub.docker.com/r/nvidia/cuda>`__ as a base
image.
Choose a ``devel`` flavor that matches your desired CUDA Toolkit version.
The source-build examples below use CUDA 13.0; the default published wheels use CUDA Toolkit 13.4.
The host driver and GPU must meet the :ref:`cuda-requirements` of the Warp build inside the container.

The following Dockerfile clones the Warp repository, builds Warp, and installs it into the system Python
environment:

.. code-block:: dockerfile

    FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

    RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        curl \
        python3 \
        python3-pip \
        && rm -rf /var/lib/apt/lists/*

    WORKDIR /warp

    RUN git clone https://github.com/NVIDIA/warp.git . && \
        git lfs pull && \
        python3 -m pip install --break-system-packages numpy && \
        python3 build_lib.py && \
        python3 -m pip install --break-system-packages .

If we put the contents of this file in a file called ``Dockerfile``, we can build an image using a command like:

.. code-block:: sh

    docker build -t warp-github-clone:example .

After building the image, you can test it with:

.. code-block:: sh

    docker run --rm --gpus all warp-github-clone:example python3 -c "import warp as wp; wp.init()"

The ``--rm`` flag tells Docker to remove the container after the command finishes.
This will output something like:

.. code-block:: text

    ==========
    == CUDA ==
    ==========

    CUDA Version 13.0.0

    Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    This container image and its contents are governed by the NVIDIA Deep Learning Container License.
    By pulling and using the container, you accept the terms and conditions of this license:
    https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

    A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

    Warp 1.10.0.dev0 initialized:
    CUDA Toolkit 13.0, Driver 13.0
    Devices:
        "cpu"      : "x86_64"
        "cuda:0"   : "NVIDIA L40S" (47 GiB, sm_89, mempool enabled)
    Kernel cache:
      /root/.cache/warp/1.10.0.dev0

An interactive session can be started with:

.. code-block:: sh

    docker run -it --rm --gpus all warp-github-clone:example

To build a modified version of Warp from your local repository, you can use the following Dockerfile as a starting
point.
Place it at the root of your repository.

.. code-block:: dockerfile

    FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

    # Install dependencies
    RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        python3 \
        python3-pip \
        && rm -rf /var/lib/apt/lists/*

    COPY warp /warp/warp
    COPY deps /warp/deps
    COPY tools/packman /warp/tools/packman
    COPY build_lib.py build_llvm.py pyproject.toml setup.py VERSION.md /warp/

    WORKDIR /warp

    RUN python3 -m pip install --break-system-packages numpy && \
        python3 build_lib.py && \
        python3 -m pip install --break-system-packages .

The resulting image produced by either of the above Dockerfile examples can be quite large due to the inclusion of
various dependencies that are no longer needed once Warp has been built.

For production use, consider a multi-stage build employing both the ``devel`` and ``runtime`` CUDA container images
to reduce the image size significantly by excluding unnecessary build tools and development dependencies from the
runtime environment.

In the builder stage, we compile Warp similar to the previous examples, but we also build a wheel file.
The runtime stage uses the lighter ``nvidia/cuda:13.0.0-runtime-ubuntu24.04`` base image and installs the wheel
produced by the builder stage into a Python virtual environment.

The following example also uses `uv <https://docs.astral.sh/uv/>`__ for Python package management, creating virtual
environments, and building the wheel file.

.. code-block:: dockerfile

    # Build stage
    FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS builder

    COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

    RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        && rm -rf /var/lib/apt/lists/*

    COPY warp /warp/warp
    COPY deps /warp/deps
    COPY tools/packman /warp/tools/packman
    COPY build_lib.py build_llvm.py pyproject.toml setup.py VERSION.md /warp/

    WORKDIR /warp

    RUN uv venv && \
        uv pip install numpy && \
        uv run --no-project build_lib.py && \
        uv build --wheel --out-dir /wheels

    # Runtime stage
    FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

    COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

    RUN uv venv /opt/venv
    # Use the virtual environment automatically
    ENV VIRTUAL_ENV=/opt/venv
    # Place entry points in the environment at the front of the path
    ENV PATH="/opt/venv/bin:$PATH"

    RUN uv pip install numpy

    # Copy and install the wheel from builder stage
    COPY --from=builder /wheels/*.whl /tmp/
    RUN uv pip install /tmp/*.whl && \
        rm -rf /tmp/*.whl

After building the image with ``docker build -t warp-prod:example .``, we can use ``docker image ls`` to compare the
image sizes.
``warp-prod:example`` is about 3.18 GB, while ``warp-github-clone:example`` is 9.03 GB!

Using Warp in Omniverse
-----------------------

Omniverse extensions for Warp are available in the extension registry inside Omniverse Kit.

The ``omni.warp.core`` extension installs Warp into Omniverse Kit's Python environment,
which allows users to import the module in their scripts and nodes.

Please see the
`Omniverse Warp Documentation <https://docs.omniverse.nvidia.com/extensions/latest/ext_warp.html>`_
for more details on how to use Warp in Omniverse.
