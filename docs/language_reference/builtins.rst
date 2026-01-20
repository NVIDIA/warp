Built-Ins
=========

.. automodule:: warp._src.lang
   :no-members:

.. currentmodule:: warp._src.lang

Scalar Math
-----------

.. autosummary::
   :nosignatures:
   :toctree: _generated
   :template: builtins.rst

   abs
   acos
   asin
   atan
   atan2
   cbrt
   ceil
   clamp
   cos
   cosh
   degrees
   erf
   erfc
   erfcinv
   erfinv
   exp
   floor
   frac
   isfinite
   isinf
   isnan
   log
   log2
   log10
   max
   min
   nonzero
   pow
   radians
   rint
   round
   sign
   sin
   sinh
   sqrt
   step
   tan
   tanh
   trunc

Vector Math
-----------

.. autosummary::
   :nosignatures:
   :toctree: _generated
   :template: builtins.rst

   argmax
   argmin
   cross
   cw_div
   cw_mul
   ddot
   determinant
   diag
   dot
   eig3
   get_diag
   identity
   inverse
   length
   length_sq
   matrix
   matrix_from_cols
   matrix_from_rows
   norm_huber
   norm_l1
   norm_l2
   norm_pseudo_huber
   normalize
   outer
   qr3
   skew
   smooth_normalize
   svd2
   svd3
   trace
   transpose
   vector

Quaternion Math
---------------

.. autosummary::
   :nosignatures:
   :toctree: _generated
   :template: builtins.rst

   quat_from_axis_angle
   quat_from_matrix
   quat_identity
   quat_inverse
   quat_rotate
   quat_rotate_inv
   quat_rpy
   quat_slerp
   quat_to_axis_angle
   quat_to_matrix
   quaternion

Transformations
---------------

.. autosummary::
   :nosignatures:
   :toctree: _generated
   :template: builtins.rst

   transform_compose
   transform_decompose
   transform_from_matrix
   transform_get_rotation
   transform_get_translation
   transform_identity
   transform_inverse
   transform_multiply
   transform_point
   transform_set_rotation
   transform_set_translation
   transform_to_matrix
   transform_vector
   transformation

Spatial Math
------------

.. autosummary::
   :nosignatures:
   :toctree: _generated
   :template: builtins.rst

   spatial_adjoint
   spatial_bottom
   spatial_cross
   spatial_cross_dual
   spatial_dot
   spatial_jacobian
   spatial_mass
   spatial_top
   spatial_vector

Tile Primitives
---------------

.. autosummary::
   :nosignatures:
   :toctree: _generated
   :template: builtins.rst

   tile
   tile_arange
   tile_argmax
   tile_argmin
   tile_assign
   tile_astype
   tile_atomic_add
   tile_atomic_add_indexed
   tile_broadcast
   tile_bvh_query_aabb
   tile_bvh_query_next
   tile_bvh_query_ray
   tile_cholesky
   tile_cholesky_inplace
   tile_cholesky_solve
   tile_cholesky_solve_inplace
   tile_diag_add
   tile_extract
   tile_fft
   tile_full
   tile_ifft
   tile_load
   tile_load_indexed
   tile_lower_solve
   tile_lower_solve_inplace
   tile_map
   tile_matmul
   tile_max
   tile_mesh_query_aabb
   tile_mesh_query_aabb_next
   tile_min
   tile_ones
   tile_randf
   tile_randi
   tile_reduce
   tile_reshape
   tile_scan_exclusive
   tile_scan_inclusive
   tile_scan_max_inclusive
   tile_scan_min_inclusive
   tile_sort
   tile_squeeze
   tile_store
   tile_store_indexed
   tile_sum
   tile_transpose
   tile_upper_solve
   tile_upper_solve_inplace
   tile_view
   tile_zeros
   untile

Geometry
--------

.. autosummary::
   :nosignatures:
   :toctree: _generated
   :template: builtins.rst

   bvh_get_group_root
   bvh_query_aabb
   bvh_query_aabb_tiled
   bvh_query_next
   bvh_query_next_tiled
   bvh_query_ray
   bvh_query_ray_tiled
   closest_point_edge_edge
   hash_grid_point_id
   hash_grid_query
   hash_grid_query_next
   intersect_tri_tri
   mesh_eval_face_normal
   mesh_eval_position
   mesh_eval_velocity
   mesh_get
   mesh_get_group_root
   mesh_get_index
   mesh_get_point
   mesh_get_velocity
   mesh_query_aabb
   mesh_query_aabb_next
   mesh_query_aabb_next_tiled
   mesh_query_aabb_tiled
   mesh_query_furthest_point_no_sign
   mesh_query_point
   mesh_query_point_no_sign
   mesh_query_point_sign_normal
   mesh_query_point_sign_parity
   mesh_query_point_sign_winding_number
   mesh_query_ray
   mesh_query_ray_anyhit
   mesh_query_ray_count_intersections

Volumes
-------

.. autosummary::
   :nosignatures:
   :toctree: _generated
   :template: builtins.rst

   volume_index_to_world
   volume_index_to_world_dir
   volume_lookup
   volume_lookup_f
   volume_lookup_i
   volume_lookup_index
   volume_lookup_v
   volume_sample
   volume_sample_f
   volume_sample_grad
   volume_sample_grad_f
   volume_sample_grad_index
   volume_sample_i
   volume_sample_index
   volume_sample_v
   volume_store
   volume_store_f
   volume_store_i
   volume_store_v
   volume_world_to_index
   volume_world_to_index_dir

Textures
--------

.. autosummary::
   :nosignatures:
   :toctree: _generated
   :template: builtins.rst

   texture_sample

Random
------

.. autosummary::
   :nosignatures:
   :toctree: _generated
   :template: builtins.rst

   curlnoise
   noise
   pnoise
   poisson
   rand_init
   randf
   randi
   randn
   randu
   sample_cdf
   sample_triangle
   sample_unit_cube
   sample_unit_disk
   sample_unit_hemisphere
   sample_unit_hemisphere_surface
   sample_unit_ring
   sample_unit_sphere
   sample_unit_sphere_surface
   sample_unit_square

Utility
-------

.. autosummary::
   :nosignatures:
   :toctree: _generated
   :template: builtins.rst

   array
   atomic_add
   atomic_and
   atomic_cas
   atomic_exch
   atomic_max
   atomic_min
   atomic_or
   atomic_sub
   atomic_xor
   block_dim
   breakpoint
   cast
   expect_near
   len
   lerp
   print
   printf
   select
   smoothstep
   tid
   where
   zeros

Other
-----

.. autosummary::
   :nosignatures:
   :toctree: _generated
   :template: builtins.rst

   lower_bound

Operators
---------

.. autosummary::
   :nosignatures:
   :toctree: _generated
   :template: builtins.rst

   add
   bit_and
   bit_or
   bit_xor
   div
   floordiv
   invert
   lshift
   mod
   mul
   neg
   pos
   rshift
   sub
   unot

Code Generation
---------------

.. autosummary::
   :nosignatures:
   :toctree: _generated
   :template: builtins.rst

   static
