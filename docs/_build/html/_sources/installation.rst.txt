Installation
============

Python version
--------------

Warp supports Python versions 3.7 and newer.

Dependencies
------------

Warp requires the following dependencies to be installed:

* `NumPy`_

.. _NumPy: https://numpy.org

Optional dependencies
~~~~~~~~~~~~~~~~~~~~~

The following dependencies may be required to enable certain features:

* JAX
* PyTorch: Required for PyTorch interoperability (see :ref:`pytorch-interop`)
* NVTX for Python

Building the Warp documentation requires the following dependencies:

* `Sphinx`_
* `Furo`_

.. _Sphinx: https://www.sphinx-doc.org
.. _Furo: https://github.com/pradyunsg/furo

Installing from PyPI
--------------------

The easiest way is to install from PyPI:

.. code-block:: sh

    $ pip install warp-lang

Pre-built binary packages for Windows and Linux are also available on the Releases page.
To install in your local Python environment extract the archive and run the following command from the root directory:

.. code-block:: sh

    $ pip install .

Building from source
-----------------------------------

For developers who want to build the library themselves the following
tools are required:

-  Microsoft Visual Studio 2019 upwards (Windows)
-  GCC 7.2 upwards (Linux)
-  `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`__ 11.5 or higher
-  `Git Large File Storage <https://git-lfs.com>`__

After cloning the repository, users should run:

.. code-block:: sh

   $ python build_lib.py

This will generate the ``warp.dll`` / ``warp.so`` core library
respectively. When building manually users should ensure that their
CUDA_PATH environment variable is set, otherwise Warp will be built
without CUDA support. Alternatively, the path to the CUDA toolkit can be
passed to the build command as ``--cuda_path="..."``. After building the
Warp package should be installed using:

.. code-block:: sh

   $ pip install -e .

Which ensures that subsequent modifications to the library will be
reflected in the Python package.

If you are cloning from Windows, please first ensure that you have
enabled “Developer Mode” in Windows settings and symlinks in git:

.. code-block:: sh

   $ git config --global core.symlinks true

This will ensure symlinks inside ``exts/omni.warp`` work upon cloning.
