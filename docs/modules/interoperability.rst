Interoperability
================

Warp can interop with other Python-based frameworks such as NumPy through standard interface protocols.

NumPy
-----

Warp arrays may be converted to a NumPy array through the ``warp.array.numpy()`` method. When the Warp array lives on
the ``cpu`` device this will return a zero-copy view onto the underlying Warp allocation. If the array lives on a
``cuda`` device then it will first be copied back to a temporary buffer and copied to NumPy.

Warp CPU arrays also implement  the ``__array_interface__`` protocol and so can be used to construct NumPy arrays
directly::

   w = wp.array([1.0, 2.0, 3.0], dtype=float, device="cpu")
   a = np.array(w)
   print(a)   
   > [1. 2. 3.]

.. _pytorch-interop:

PyTorch
-------

Warp provides helper functions to convert arrays to/from PyTorch. Please see the ``warp.torch`` module for more details. Example usage is shown below::

   import warp.torch

   w = wp.array([1.0, 2.0, 3.0], dtype=float, device="cpu")

   # convert to Torch tensor
   t = warp.to_torch(w)

   # convert from Torch tensor
   w = warp.from_torch(t)

.. automodule:: warp.torch
    :members:
    :undoc-members:

CuPy/Numba
----------

Warp GPU arrays support the ``__cuda_array_interface__`` protocol for sharing data with other Python GPU frameworks.
Currently this is one-directional, so that Warp arrays can be used as input to any framework that also supports the
``__cuda_array_interface__`` protocol, but not the other way around.

JAX
---

Interop with JAX arrays is not currently well supported, although it is possible to first convert arrays to a Torch
tensor and then to JAX via. the dlpack mechanism.

.. automodule:: warp.jax
    :members:
    :undoc-members:

DLPack
------

.. automodule:: warp.dlpack
    :members:
    :undoc-members:
