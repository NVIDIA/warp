warp.geometry
=============

.. automodule:: warp.geometry
   :no-members:

.. currentmodule:: warp.geometry

Python API
----------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   IsoSurfaceBase
   IsoSurfaceMarchingCubes
   IsoSurfaceNets
   OBBMeasureType
   SweptVolumeSignMode
   delaunay_edge_flip
   lipschitz_octree
   oriented_bounding_box
   oriented_bounding_boxes
   sparse_marching_cubes
   sparse_marching_cubes_from_cells
   swept_volume_bounds
   swept_volume_field
   swept_volume_mesh
   tri_tri_adjacency

Kernel API
----------

.. autosummary::
   :nosignatures:
   :toctree: _generated

   find_triangle_neighbor_edge_index
   intersect_aabb_aabb
   intersect_seg_aabb
   intersect_seg_tri
   intersect_tri_aabb
   intersect_tri_tri
   self_intersect_tri_tri
   super_fibonacci
   swept_volume_sdf
