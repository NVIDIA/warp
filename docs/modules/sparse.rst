Sparse Linear Algebra Reference
===============================

.. currentmodule:: warp.sparse

..
   .. toctree::
   :maxdepth: 2

Warp includes a sparse linear algebra module ``warp.sparse`` that implements a few common operations for manipulating sparse matrices.

.. note:: The sparse module is under construction and should be expected to change rapidly, please treat this section as work in progress.
.. warning:: This module provide naive implementations of useful routines for convenience. For performance, if possible prefer optimized implementations such as provided by cuSPARSE.


Blocked Sparse Row (BSR) matrices
---------------------------------

At the moment, `warp.sparse` only supports BSR matrix, including the special case of 1x1 blocks, Compressed Sparse Row (CSR) matrices.

.. automodule:: warp.sparse
   :members:

