warp.fem
========

.. currentmodule:: warp.fem

The ``warp.fem`` module is designed to facilitate solving physical systems described as differential 
equations. For example, it can solve PDEs for diffusion, convection, fluid flow, and elasticity problems 
using finite-element-based (FEM) Galerkin methods and allows users to quickly experiment with various FEM
formulations and discretization schemes.

Integrands
----------

The core functionality of the FEM toolkit is the ability to integrate constant, linear, and bilinear forms 
over various domains and using arbitrary interpolation basis.

The main mechanism is the :py:func:`.integrand` decorator, for instance: ::

    @integrand
    def linear_form(
        s: Sample,
        domain: Domain,
        v: Field,
    ):
        x = domain(s)
        return v(s) * wp.max(0.0, 1.0 - wp.length(x))


    @integrand
    def diffusion_form(s: Sample, u: Field, v: Field, nu: float):
        return nu * wp.dot(
            grad(u, s),
            grad(v, s),
        )

Integrands are normal Warp kernels, meaning that they may contain arbitrary Warp functions. 
However, they accept a few special parameters:

  - :class:`.Sample` contains information about the current integration sample point, such as the element index and coordinates in element.
  - :class:`.Field` designates an abstract field, which will be replaced at call time by the actual field type: for instance a :class:`.DiscreteField`, :class:`.field.TestField` or :class:`.field.TrialField` defined over an arbitrary :class:`.FunctionSpace`.
    A field `u` can be evaluated at a given sample `s` using the :func:`.inner` operator, i.e, ``inner(u, s)``, or as a shortcut using the usual call operator, ``u(s)``.
    Several other operators are available, such as :func:`.grad`; see the :ref:`Operators` section.
  - :class:`.Domain` designates an abstract integration domain. Several operators are also provided for domains, for example in the example below evaluating the normal at the current sample position: ::
    
            @integrand
            def boundary_form(
                s: Sample,
                domain: Domain,
                u: Field,
            ):
                nor = normal(domain, s)
                return wp.dot(u(s), nor)

Integrands cannot be used directly with :func:`warp.launch`, but must be called through :func:`.integrate` or :func:`.interpolate` instead.
The root integrand (`integrand` argument passed to :func:`integrate` or :func:`interpolate` call) will automatically get passed :class:`.Sample` and :class:`.Domain` parameters.
:class:`.Field` parameters must be passed as a dictionary in the `fields` argument of the launcher function, and all other standard Warp types arguments must be
passed as a dictionary in the `values` argument of the launcher function, for instance: ::
    
    integrate(diffusion_form, fields={"u": trial, "v": test}, values={"nu": viscosity})


Basic Workflow
--------------

The typical steps for solving a linear PDE are as follow:

 - Define a :class:`.Geometry` (grid, mesh, etc). At the moment, 2D and 3D regular grids, NanoVDB volumes, and triangle, quadrilateral, tetrahedron and hexahedron unstructured meshes are supported.
 - Define one or more :class:`.FunctionSpace`, by equipping the geometry elements with shape functions. See :func:`.make_polynomial_space`. At the moment, continuous/discontinuous Lagrange (:math:`P_{k[d]}, Q_{k[d]}`) and Serendipity (:math:`S_k`) shape functions of order :math:`k \leq 3` are supported.
 - Define an integration :class:`.GeometryDomain`, for instance the geometry's cells (:class:`.Cells`) or boundary sides (:class:`.BoundarySides`).
 - Integrate linear forms to build the system's right-hand-side. Define a test function over the function space using :func:`.make_test`,
   a :class:`.Quadrature` formula (or let the module choose one based on the function space degree), and call :func:`.integrate` with the linear form integrand.
   The result is a :class:`warp.array` containing the integration result for each of the function space degrees of freedom.
 - Integrate bilinear forms to build the system's left-hand-side. Define a trial function over the function space using :func:`.make_trial`,
   then call :func:`.integrate` with the bilinear form integrand.
   The result is a :class:`warp.sparse.BsrMatrix` containing the integration result for each pair of test and trial function space degrees of freedom.
   Note that the trial and test functions do not have to be defined over the same function space, so that Mixed FEM is supported.
 - Solve the resulting linear system using the solver of your choice


The following excerpt from the introductory example ``warp/examples/fem/example_diffusion.py`` outlines this procedure: ::

    # Grid geometry
    geo = Grid2D(n=50, cell_size=2)

    # Domain and function spaces
    domain = Cells(geometry=geo)
    scalar_space = make_polynomial_space(geo, degree=3)

    # Right-hand-side (forcing term)
    test = make_test(space=scalar_space, domain=domain)
    rhs = integrate(linear_form, fields={"v": test})

    # Weakly-imposed boundary conditions on Y sides
    boundary = BoundarySides(geo)
    bd_test = make_test(space=scalar_space, domain=boundary)
    bd_trial = make_trial(space=scalar_space, domain=boundary)
    bd_matrix = integrate(y_mass_form, fields={"u": bd_trial, "v": bd_test})

    # Diffusion form
    trial = make_trial(space=scalar_space, domain=domain)
    matrix = integrate(diffusion_form, fields={"u": trial, "v": test}, values={"nu": viscosity})

    # Assemble linear system (add diffusion and boundary condition matrices)
    bsr_axpy(x=bd_matrix, y=matrix, alpha=boundary_strength, beta=1)

    # Solve linear system using Conjugate Gradient
    x = wp.zeros_like(rhs)
    bsr_cg(matrix, b=rhs, x=x)


.. note::
    The :func:`.integrate` function does not check that the passed integrands are actually linear or bilinear forms; it is up to the user to ensure that they are.
    To solve non-linear PDEs, one can use an iterative procedure and pass the current value of the studied function :class:`.DiscreteField` argument to the integrand, on which
    arbitrary operations are permitted. However, the result of the form must remain linear in the test and trial fields.

Introductory Examples
---------------------

``warp.fem`` ships with a list of examples in the ``warp/examples/fem`` directory illustrating common model problems.

 - ``example_diffusion.py``: 2D diffusion with homogeneous Neumann and Dirichlet boundary conditions
     * ``example_diffusion_3d.py``: 3D variant of the diffusion problem
 - ``example_convection_diffusion.py``: 2D convection-diffusion using semi-Lagrangian advection
     * ``example_convection_diffusion_dg.py``: 2D convection-diffusion using Discontinuous Galerkin with upwind transport and Symmetric Interior Penalty
 - ``example_burgers.py``: 2D inviscid Burgers using Discontinuous Galerkin with upwind transport and slope limiter
 - ``example_stokes.py``: 2D incompressible Stokes flow using mixed :math:`P_k/P_{k-1}` or :math:`Q_k/P_{(k-1)d}` elements
 - ``example_navier_stokes.py``: 2D Navier-Stokes flow using mixed :math:`P_k/P_{k-1}` elements
 - ``example_mixed_elasticity.py``: 2D nonlinear elasticity using mixed continuous/discontinuous :math:`S_k/P_{(k-1)d}` elements
 - ``example_streamlines.py``: Using the :func:`lookup` operator to trace through a velocity field


Advanced Usages
---------------

High-order (curved) geometries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is possible to convert any :class:`.Geometry` (grids and explicit meshes) into a curved, high-order variant by deforming them 
with an arbitrary-order displacement field using the :meth:`~.DiscreteField.make_deformed_geometry` method. 
The process looks as follows::

   # Define a base geometry
   base_geo = fem.Grid3D(res=resolution)

   # Define a displacement field on the base geometry
   deformation_space = fem.make_polynomial_space(base_geo, degree=deformation_degree, dtype=wp.vec3)
   deformation_field = deformation_space.make_field()

   # Populate the field value by interpolating an expression
   fem.interpolate(deformation_field_expr, dest=deformation_field)

   # Construct the deformed geometry from the displacement field
   deform_geo = deformation_field.make_deformed_geometry()

   # Define new function spaces on the deformed geometry
   scalar_space = fem.make_polynomial_space(deformed_geo, degree=scalar_space_degree)

See also ``example_deformed_geometry.py`` for a complete example.

Particle-based quadrature
^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`.PicQuadrature` provides a way to define Particle-In-Cell quadratures from a set or arbitrary particles,
which can be helpful to develop MPM-like methods.
The particles are automatically bucketed to the geometry cells when the quadrature is initialized.
This is illustrated by the ``example_stokes_transfer.py`` and ``example_apic_fluid.py`` examples.

Partitioning
^^^^^^^^^^^^

The FEM toolkit makes it possible to perform integration on a subset of the domain elements, 
possibly re-indexing degrees of freedom so that the linear system contains the local ones only.
This is useful for distributed computation (see ``warp/examples/fem/example_diffusion_mgpu.py``), or simply to limit the simulation domain to a subset of active cells (see ``warp/examples/fem/example_stokes_transfer.py``).

A partition of the simulation geometry can be defined using subclasses of :class:`.GeometryPartition`
such as :class:`.LinearGeometryPartition`  or :class:`.ExplicitGeometryPartition`.

Function spaces can then be partitioned according to the geometry partition using :func:`.make_space_partition`. 
The resulting :class:`.SpacePartition` object allows translating between space-wide and partition-wide node indices, 
and differentiating interior, frontier and exterior nodes.

Memory management
^^^^^^^^^^^^^^^^^

Several ``warp.fem`` functions require allocating temporary buffers to perform their computations. 
If such functions are called many times in a tight loop, those many allocations and de-allocations may degrade performance.
To overcome this issue, a :class:`.cache.TemporaryStore` object may be created to persist and reuse temporary allocations across calls,
either globally using :func:`set_default_temporary_store` or at a per-function granularity using the corresponding argument.

.. _Operators:

Operators
---------
.. autofunction:: position(domain: Domain, s: Sample)
.. autofunction:: normal(domain: Domain, s: Sample)
.. autofunction:: lookup(domain: Domain, x)
.. autofunction:: measure(domain: Domain, s: Sample)
.. autofunction:: measure_ratio(domain: Domain, s: Sample)
.. autofunction:: deformation_gradient(domain: Domain, s: Sample)

.. autofunction:: degree(f: Field)
.. autofunction:: inner(f: Field, s: Sample)
.. autofunction:: outer(f: Field, s: Sample)
.. autofunction:: grad(f: Field, s: Sample)
.. autofunction:: grad_outer(f: Field, s: Sample)
.. autofunction:: div(f: Field, s: Sample)
.. autofunction:: div_outer(f: Field, s: Sample)
.. autofunction:: at_node(f: Field, s: Sample)

.. autofunction:: D(f: Field, s: Sample)
.. autofunction:: curl(f: Field, s: Sample)
.. autofunction:: jump(f: Field, s: Sample)
.. autofunction:: average(f: Field, s: Sample)
.. autofunction:: grad_jump(f: Field, s: Sample)
.. autofunction:: grad_average(f: Field, s: Sample)

.. autofunction:: warp.fem.operator.operator

Integration
-----------

.. autofunction:: integrate
.. autofunction:: interpolate

.. autofunction:: integrand

.. class:: Sample

   Per-sample point context for evaluating fields and related operators in integrands.

.. autoclass:: Field 

.. autoclass:: Domain 

Geometry
--------

.. autoclass:: Grid2D
   :show-inheritance:

.. autoclass:: Trimesh2D
   :show-inheritance:

.. autoclass:: Quadmesh2D
   :show-inheritance:

.. autoclass:: Grid3D
   :show-inheritance:

.. autoclass:: Tetmesh
   :show-inheritance:

.. autoclass:: Hexmesh
   :show-inheritance:

.. autoclass:: Nanogrid
   :show-inheritance:

.. autoclass:: LinearGeometryPartition

.. autoclass:: ExplicitGeometryPartition

.. autoclass:: Cells
   :show-inheritance:

.. autoclass:: Sides
   :show-inheritance:

.. autoclass:: BoundarySides
   :show-inheritance:

.. autoclass:: FrontierSides
   :show-inheritance:

.. autoclass:: Subdomain
   :show-inheritance:

.. autoclass:: Polynomial
   :members:

.. autoclass:: RegularQuadrature
   :show-inheritance:

.. autoclass:: NodalQuadrature
   :show-inheritance:

.. autoclass:: ExplicitQuadrature
   :show-inheritance:

.. autoclass:: PicQuadrature
   :show-inheritance:

Function Spaces
---------------

.. autofunction:: make_polynomial_space

.. autofunction:: make_polynomial_basis_space

.. autofunction:: make_collocated_function_space

.. autofunction:: make_space_partition

.. autofunction:: make_space_restriction

.. autoclass:: ElementBasis
   :members:

.. autoclass:: SymmetricTensorMapper
   :show-inheritance:

.. autoclass:: SkewSymmetricTensorMapper
   :show-inheritance:

.. autoclass:: PointBasisSpace
   :show-inheritance:

Fields
------

.. autofunction:: make_test

.. autofunction:: make_trial

.. autofunction:: make_restriction

Boundary Conditions
-------------------

.. autofunction:: normalize_dirichlet_projector

.. autofunction:: project_linear_system

Memory Management
-----------------

.. autofunction:: set_default_temporary_store

.. autofunction:: borrow_temporary

.. autofunction:: borrow_temporary_like


Interfaces
----------

Interface classes are not meant to be constructed directly, but can be derived from extend the built-in functionality.

.. autoclass:: Geometry
   :members: cell_count, side_count, boundary_side_count

.. autoclass:: GeometryPartition
   :members: cell_count, side_count, boundary_side_count, frontier_side_count

.. autoclass:: GeometryDomain
   :members: ElementKind, element_kind, dimension, element_count

.. autoclass:: Quadrature
   :members: domain, total_point_count

.. autoclass:: FunctionSpace
   :members: dtype, topology, geometry, dimension, degree, trace, make_field

.. autoclass:: SpaceTopology
   :members: dimension, geometry, node_count, element_node_indices, trace

.. autoclass:: BasisSpace
   :members: topology, geometry, node_positions

.. autoclass:: warp.fem.space.shape.ShapeFunction

.. autoclass:: SpacePartition
   :members: node_count, owned_node_count, interior_node_count, space_node_indices

.. autoclass:: SpaceRestriction
   :members: node_count

.. autoclass:: DofMapper

.. autoclass:: FieldLike

.. autoclass:: DiscreteField
   :show-inheritance:
   :members: dof_values, trace, make_deformed_geometry

.. autoclass:: warp.fem.field.FieldRestriction

.. autoclass:: warp.fem.field.SpaceField
   :show-inheritance:

.. autoclass:: warp.fem.field.TestField
   :show-inheritance:

.. autoclass:: warp.fem.field.TrialField
   :show-inheritance:

.. autoclass:: TemporaryStore
   :members: clear

.. autoclass:: warp.fem.cache.Temporary
   :members: array, detach, release
