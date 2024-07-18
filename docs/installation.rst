Installation
============

Python version 3.9 or newer is *recommended*. Warp can run on x86-64 and ARMv8 CPUs on Windows, Linux, and macOS. GPU support requires a CUDA-capable NVIDIA GPU and driver (minimum GeForce GTX 9xx).

The easiest way to install Warp is from `PyPI <https://pypi.org/project/warp-lang>`_:

.. code-block:: sh

    $ pip install warp-lang

.. _GitHub Installation:

Installing from GitHub Releases
-------------------------------

The binaries hosted on PyPI are currently built with the CUDA 12.5 runtime.
We also provide binaries built with the CUDA 11.8 runtime on the `GitHub Releases <https://github.com/NVIDIA/warp/releases>`_ page.
Copy the URL of the appropriate wheel file (``warp-lang-{ver}+cu11-py3-none-{platform}.whl``) and pass it to
the ``pip install`` command, e.g.

.. code-block:: sh

    pip install https://github.com/NVIDIA/warp/releases/download/v1.3.0/warp_lang-1.3.0+cu11-py3-none-manylinux2014_x86_64.whl

The ``--force-reinstall`` option may need to be used to overwrite a previous installation.

CUDA Requirements
-----------------

* Warp packages built with CUDA Toolkit 11.x require NVIDIA driver 470 or newer.
* Warp packages built with CUDA Toolkit 12.x require NVIDIA driver 525 or newer.

This applies to pre-built packages distributed on PyPI and GitHub and also when building Warp from source.

Note that building Warp with the ``--quick`` flag changes the driver requirements.  The quick build skips CUDA backward compatibility, so the minimum required driver is determined by the CUDA Toolkit version.  Refer to the `latest CUDA Toolkit release notes <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>`_ to find the minimum required driver for different CUDA Toolkit versions (e.g., `this table from CUDA Toolkit 12.5 <https://docs.nvidia.com/cuda/archive/12.5.0/cuda-toolkit-release-notes/index.html#id4>`_).

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

Dependencies
------------

Warp supports Python versions 3.7 onwards, with 3.9 or newer recommended for full functionality. Note that :ref:`some optional dependencies may not support the latest version of Python<conda>`.

`NumPy <https://numpy.org>`_ must be installed.

The following optional dependencies are required to support certain features:

* `usd-core <https://pypi.org/project/usd-core>`_: Required for some Warp examples, ``warp.sim.parse_usd()``, and ``warp.render.UsdRenderer``.
* `JAX <https://jax.readthedocs.io/en/latest/installation.html>`_: Required for JAX interoperability (see :ref:`jax-interop`).
* `PyTorch <https://pytorch.org/get-started/locally/>`_: Required for PyTorch interoperability (see :ref:`pytorch-interop`).
* `NVTX for Python <https://github.com/NVIDIA/NVTX#python>`_: Required to use :class:`wp.ScopedTimer(use_nvtx=True) <warp.ScopedTimer>`.

Building the Warp documentation requires:

* `Sphinx <https://www.sphinx-doc.org>`_
* `Furo <https://github.com/pradyunsg/furo>`_
* `Sphinx-copybutton <https://sphinx-copybutton.readthedocs.io/en/latest/index.html>`_

Building from Source
--------------------

For developers who want to build the library themselves the following tools are required:

* Microsoft Visual Studio (Windows), minimum version 2019
* GCC (Linux), minimum version 9.4
* `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_, minimum version 11.5
* `Git Large File Storage <https://git-lfs.com>`_

After cloning the repository, users should run:

.. code-block:: console

    $ python build_lib.py

Upon success, the script will output platform-specific binary files in ``warp/bin/``.
The build script will look for the CUDA Toolkit in its default installation path.
This path can be overridden by setting the ``CUDA_PATH`` environment variable. Alternatively,
the path to the CUDA Toolkit can be passed to the build command as
``--cuda_path="..."``. After building, the Warp package should be installed using:

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
