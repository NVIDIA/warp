Installation
============

Python version 3.9 or newer is required. Warp can run on x86-64 and ARMv8 CPUs on Windows and Linux. macOS requires Apple Silicon (ARM64). GPU support requires a CUDA-capable NVIDIA GPU and driver (minimum GeForce GTX 9xx).

.. note::
   Intel-based macOS (x86_64) is no longer supported. Users with Intel Macs should use Warp version 1.9.x or earlier.

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

Note that the nightly builds are built with the CUDA 12 runtime and are not published for macOS.

If you plan to install nightly builds regularly, you can simplify future installations by adding NVIDIA's package
repository as an extra index via the ``PIP_EXTRA_INDEX_URL`` environment variable. For example:

.. code-block:: text

    export PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com"

This ensures the index is automatically used for ``pip`` commands, avoiding the need to specify it explicitly.

Conda Installation
------------------

Conda packages for Warp are also available on the `conda-forge <https://anaconda.org/conda-forge/warp-lang>`__ channel.

.. code-block:: sh

    # Install warp-lang specifically built against CUDA Toolkit 12.6
    $ conda install conda-forge::warp-lang=*=*cuda126*

For more information, see the community-maintained feedstock for Warp
`here <https://github.com/conda-forge/warp-lang-feedstock>`__.

Installing from GitHub Releases
-------------------------------

The binaries hosted on PyPI are currently built with the CUDA 12 runtime.
We also provide binaries built with the CUDA 13.0 runtime on the `GitHub Releases <https://github.com/NVIDIA/warp/releases>`_ page.
Copy the URL of the appropriate wheel file (``warp-lang-{ver}+cu13-py3-none-{platform}.whl``) and pass it to
the ``pip install`` command, e.g.

.. list-table:: 
   :header-rows: 1

   * - Platform
     - Install Command
   * - Linux aarch64
     - ``pip install https://github.com/NVIDIA/warp/releases/download/v1.11.1/warp_lang-1.11.1+cu13-py3-none-manylinux_2_34_aarch64.whl``
   * - Linux x86-64
     - ``pip install https://github.com/NVIDIA/warp/releases/download/v1.11.1/warp_lang-1.11.1+cu13-py3-none-manylinux_2_28_x86_64.whl``
   * - Windows x86-64
     - ``pip install https://github.com/NVIDIA/warp/releases/download/v1.11.1/warp_lang-1.11.1+cu13-py3-none-win_amd64.whl``

The ``--force-reinstall`` option may need to be used to overwrite a previous installation.

CUDA Requirements
-----------------

* Warp packages built with CUDA Toolkit 12.x require NVIDIA driver 525 or newer.
* Warp packages built with CUDA Toolkit 13.x require NVIDIA driver 580 or newer.

This applies to pre-built packages distributed on PyPI and GitHub and also when building Warp from source.

Note that building Warp with the ``--quick`` flag changes the driver requirements.
The quick build skips CUDA backward compatibility, so the minimum required driver is determined by the CUDA Toolkit version.
Refer to the `latest CUDA Toolkit release notes <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>`_
to find the minimum required driver for different CUDA Toolkit versions
(e.g., `this table from CUDA Toolkit 12.9 <https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html#id7>`_).

Warp checks the installed driver during initialization and will report a warning if the driver is not suitable, e.g.:

.. code-block:: text

    Warp UserWarning:
       Insufficient CUDA driver version.
       The minimum required CUDA driver version is 12.0, but the installed CUDA driver version is 11.8.
       Visit https://github.com/NVIDIA/warp/blob/main/README.md#installing for guidance.

This will make CUDA devices unavailable, but the CPU can still be used.

To remedy the situation there are a few options:

* Update the driver.
* Install a compatible pre-built Warp package.
* Build Warp from source using a CUDA Toolkit that's compatible with the installed driver.

Also note that full support for tile-based MathDx features requires CUDA version 12.6.3 or later. See :ref:`mathdx` for more information.

CUDA 12.9 limitation on Linux ARM platforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When building Warp from source with CUDA 12.9 on a Linux ARM platform (including NVIDIA Jetson platforms),
the resulting binary will not support Maxwell, Pascal, or Volta GPU architectures due to a
`bug <https://github.com/NVIDIA/cccl/issues/4967>`__ in the CUDA 12.9 Toolkit which limits the number of architectures that
can be compiled at once.

If support for these architectures is required, build Warp using a CUDA Toolkit prior to 12.9.
Note that CUDA 13.0 dropped support for the same architectures entirely.

Dependencies
------------

Warp supports Python versions 3.9 onwards. Note that :ref:`some optional dependencies may not support the latest version of Python<conda>`.

`NumPy <https://numpy.org>`_ must be installed.

The following optional dependencies are required to support certain features:

* `usd-core <https://pypi.org/project/usd-core>`_: Required for some Warp examples, tests, and the :class:`warp.render.UsdRenderer`.
  On Linux aarch64 systems where ``usd-core`` wheels are not available,
  `usd-exchange <https://pypi.org/project/usd-exchange>`_ can be installed as a drop-in replacement.
  The ``[examples]`` extra handles this automatically.
* `pyglet <https://pyglet.org/>`_: Required for some Warp examples and the :class:`warp.render.OpenGLRenderer`.
* `JAX <https://jax.readthedocs.io/en/latest/installation.html>`_: Required for JAX interoperability (see :ref:`jax-interop`).
* `PyTorch <https://pytorch.org/get-started/locally/>`_: Required for PyTorch interoperability (see :ref:`pytorch-interop`).
* `Paddle <https://github.com/PaddlePaddle/Paddle>`_: Required for Paddle interoperability (see :ref:`paddle-interop`).
* `NVTX for Python <https://github.com/NVIDIA/NVTX#python>`_: Required to use :class:`wp.ScopedTimer(use_nvtx=True) <warp.ScopedTimer>`.
* `psutil <https://psutil.readthedocs.io/en/latest/>`_: Required to query CPU memory info (`get_device("cpu").total_memory`, `get_device("cpu").free_memory`).

Building from Source
--------------------

For developers who want to build the library themselves the following tools are required:

* Microsoft Visual Studio (Windows), minimum version 2019
* GCC (Linux), minimum version 9.4
* `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_, minimum version 12.0
* `Git Large File Storage <https://git-lfs.com>`_

After cloning the repository, users should run:

.. code-block:: console

    $ python build_lib.py

Upon success, the script will output platform-specific binary files in ``warp/bin/``.
The build script will look for the CUDA Toolkit in its default installation path.
This path can be overridden by setting the ``CUDA_PATH`` environment variable. Alternatively,
the path to the CUDA Toolkit can be passed to the build command as
``--cuda-path="..."``. After building, the Warp package should be installed using:

.. code-block:: console

    $ pip install -e .

The ``-e`` option is optional but ensures that subsequent modifications to the
library will be reflected in the Python package.

.. _conda:

Conda Environments
------------------

Some modules, such as ``usd-core``, don't support the latest Python version.
To manage running Warp and other projects on different Python versions one can
make use of an environment management system such as
`Conda <https://docs.conda.io/>`__.

.. warning::

    When building and running Warp in a different environment, make sure
    the build environment has the same C++ runtime library version, or an older
    one, than the execution environment. Otherwise Warp's shared libraries may end
    up looking for a newer runtime library version than the one available in the
    execution environment. For example, on Linux this error could occur::

        OSError: <...>/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by <...>/warp/warp/bin/warp.so)

    This can be solved by installing a newer C++ runtime version in the runtime
    Conda environment using ``conda install -c conda-forge libstdcxx-ng=12.1`` or
    newer.
    
    Alternatively, the build environment's C++ toolchain can be downgraded using
    ``conda install -c conda-forge libstdcxx-ng=8.5``. Or, one can ``activate`` or
    ``deactivate`` Conda environments as needed for building vs. running Warp.

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
