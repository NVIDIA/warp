warp.jax
========

.. automodule:: warp.jax
   :no-members:

.. currentmodule:: warp.jax

.. toctree::
   :hidden:

   warp_jax_custom_call
   warp_jax_ffi

Additional Submodules
---------------------

These modules must be explicitly imported (e.g., ``import warp.jax.custom_call``).

- :mod:`warp.jax.custom_call`
- :mod:`warp.jax.ffi`

JAX Array Interop
-----------------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   device_from_jax
   device_to_jax
   dtype_from_jax
   dtype_to_jax
   from_jax
   to_jax

JAX Callable Interop
--------------------

- :obj:`GraphMode <warp.jax.ffi.GraphMode>`
- :obj:`clear_jax_callable_graph_cache <warp.jax.ffi.clear_jax_callable_graph_cache>`
- :obj:`get_jax_callable_default_graph_cache_max <warp.jax.ffi.get_jax_callable_default_graph_cache_max>`
- :obj:`jax_callable <warp.jax.ffi.jax_callable>`
- :obj:`jax_kernel <warp.jax.ffi.jax_kernel>`
- :obj:`register_ffi_callback <warp.jax.ffi.register_ffi_callback>`
- :obj:`set_jax_callable_default_graph_cache_max <warp.jax.ffi.set_jax_callable_default_graph_cache_max>`
