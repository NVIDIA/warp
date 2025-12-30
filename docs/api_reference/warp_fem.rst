warp.fem
========

.. automodule:: warp.fem
   :no-members:

.. currentmodule:: warp.fem

.. toctree::
   :hidden:

   warp_fem_cache
   warp_fem_field
   warp_fem_geometry
   warp_fem_linalg
   warp_fem_polynomial
   warp_fem_space
   warp_fem_utils

Submodules
----------

These modules are automatically available when you ``import warp.fem``.

- :mod:`warp.fem.cache`
- :mod:`warp.fem.field`
- :mod:`warp.fem.geometry`
- :mod:`warp.fem.linalg`
- :mod:`warp.fem.polynomial`
- :mod:`warp.fem.space`
- :mod:`warp.fem.utils`

API
---

.. autosummary::
   :nosignatures:
   :toctree: _generated

   AdaptiveNanogrid
   BasisSpace
   BoundarySides
   CellBasedGeometryPartition
   Cells
   Coords
   DiscreteField
   DofMapper
   Domain
   Element
   ElementBasis
   ElementIndex
   ElementKind
   ExplicitGeometryPartition
   ExplicitQuadrature
   Field
   FieldLike
   FrontierSides
   FunctionSpace
   Geometry
   GeometryDomain
   GeometryField
   GeometryPartition
   Grid2D
   Grid3D
   Hexmesh
   ImplicitField
   Integrand
   LinearGeometryPartition
   Nanogrid
   NodalQuadrature
   NodeIndex
   NonconformingField
   Operator
   PicQuadrature
   PointBasisSpace
   Polynomial
   Quadmesh2D
   Quadmesh3D
   Quadrature
   QuadraturePointIndex
   RegularQuadrature
   Sample
   ShapeBasisSpace
   ShapeFunction
   Sides
   SkewSymmetricTensorMapper
   SpacePartition
   SpaceRestriction
   SpaceTopology
   Subdomain
   SymmetricTensorMapper
   Temporary
   TemporaryStore
   Tetmesh
   Trimesh2D
   Trimesh3D
   UniformField
   D
   adaptive_nanogrid_from_field
   adaptive_nanogrid_from_hierarchy
   at_node
   average
   borrow_temporary
   borrow_temporary_like
   cells
   curl
   deformation_gradient
   degree
   div
   div_outer
   element_closest_point
   element_coordinates
   element_index
   element_partition_index
   grad
   grad_average
   grad_jump
   grad_outer
   inner
   integrand
   integrate
   interpolate
   jump
   lookup
   make_collocated_function_space
   make_contravariant_function_space
   make_covariant_function_space
   make_discrete_field
   make_element_based_space_topology
   make_element_shape_function
   make_free_sample
   make_polynomial_basis_space
   make_polynomial_space
   make_restriction
   make_space_partition
   make_space_restriction
   make_test
   make_trial
   measure
   measure_ratio
   node_count
   node_index
   node_inner_weight
   node_inner_weight_gradient
   node_outer_weight
   node_outer_weight_gradient
   node_partition_index
   normal
   normalize_dirichlet_projector
   outer
   partition_lookup
   position
   project_linear_system
   project_system_matrix
   project_system_rhs
   set_default_temporary_store
   to_cell_side
   to_inner_cell
   to_outer_cell
   NULL_ELEMENT_INDEX
   NULL_NODE_INDEX
   NULL_QP_INDEX
   OUTSIDE
