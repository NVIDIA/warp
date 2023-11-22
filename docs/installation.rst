Installation
============

The easiest way is to install Warp is from `PyPi <https://pypi.org/project/warp-lang>`_:

.. code-block:: sh

    $ pip install warp-lang

Pre-built binary packages for Windows, Linux and macOS are also available on the `Releases <https://github.com/NVIDIA/warp/releases>`__ page. To install in your local Python environment extract the archive and run the following command from the root directory:

.. code-block:: sh

    $ pip install .

Dependencies
------------

Warp supports Python versions 3.7 or later and requires `NumPy <https://numpy.org>`_ to be installed.

The following optional dependencies are required to support certain features:

* `usd-core <https://pypi.org/project/usd-core>`_: Required for some Warp examples, ``warp.sim.parse_usd()``, and ``warp.render.UsdRenderer``.
* `JAX <https://jax.readthedocs.io/en/latest/installation.html>`_: Required for JAX interoperability (see :ref:`jax-interop`).
* `PyTorch <https://pytorch.org/get-started/locally/>`_: Required for PyTorch interoperability (see :ref:`pytorch-interop`).
* `NVTX for Python <https://github.com/NVIDIA/NVTX#python>`_: Required to use :class:`wp.ScopedTimer(use_nvtx=True) <warp.ScopedTimer>`.

Building the Warp documentation requires:

* `Sphinx <https://www.sphinx-doc.org>`_
* `Furo <https://github.com/pradyunsg/furo>`_
* `Sphinx-copybutton <https://sphinx-copybutton.readthedocs.io/en/latest/index.html>`_

Building from source
--------------------

For developers who want to build the library themselves the following tools are required:

* Microsoft Visual Studio (Windows), minimum version 2019
* GCC (Linux), minimum version 7.2
* `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_, minimum version 11.5
* `Git Large File Storage <https://git-lfs.com>`_

If you are cloning from Windows, please first ensure that you have
enabled “Developer Mode” in Windows settings and symlinks in Git:

.. code-block:: console

    $ git config --global core.symlinks true

This will ensure symlinks inside ``exts/omni.warp.core`` work upon cloning.

After cloning the repository, users should run:

.. code-block:: console

    $ python build_lib.py

This will generate the ``warp.dll`` / ``warp.so`` core library
respectively. When building manually, users should ensure that their
``CUDA_PATH`` environment variable is set, otherwise Warp will be built
without CUDA support. Alternatively, the path to the CUDA Toolkit can be
passed to the build command as ``--cuda_path="..."``. After building, the
Warp package should be installed using:

.. code-block:: console

    $ pip install -e .

Which ensures that subsequent modifications to the library will be
reflected in the Python package.

Conda environments
------------------

Some modules, such as ``usd-core``, don't support the latest Python version.
To manage running Warp and other projects on different Python versions one can
make use of an environment management system such as
`Conda <https://docs.conda.io/>`_.

**WARNING:** When building and running Warp in a different environment, make sure
the build environment has the same C++ runtime library version, or an older
one, than the execution environment. Otherwise Warp's shared libraries may end
up looking for a newer runtime library version than the one available in the
execution environment. For example on Linux this error could occur:

``OSError: <...>/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by <...>/warp/warp/bin/warp.so)``

This can be solved by installing a newer C++ runtime version in the runtime
conda environment using ``conda install -c conda-forge libstdcxx-ng=12.1`` or
newer. Or, the build environment's C++ toolchain can be downgraded using
``conda install -c conda-forge libstdcxx-ng=8.5``. Or, one can ``activate`` or
``deactivate`` conda environments as needed for building vs. running Warp.
