Installation
============

Python Version
--------------

Warp supports Python versions 3.7 and newer.

Dependencies
------------

Warp will not function without the following dependencies:

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
