FEM Toolkit
===========

.. currentmodule:: warp.fem
.. py:currentmodule:: warp.fem

The :mod:`warp.fem` module is designed to facilitate solving physical systems described as differential 
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

  - :data:`.Sample` contains information about the current integration sample point, such as the element index and coordinates in element.
  - :class:`.Field` designates an abstract field, which will be replaced at call time by the actual field type such as a discrete field, :class:`.field.TestField` or :class:`.field.TrialField` defined over some :class:`.FunctionSpace`,
    an :class:`.ImplicitField` wrapping an arbitrary function, or any other of the available :ref:`Fields`.
    A field `u` can then be evaluated at a given sample `s` using the usual call operator as ``u(s)``.
    Several other operators are available, such as the gradient :obj:`.grad`; see the :ref:`Operators` section.
  - :class:`.Domain` designates an abstract integration domain. Evaluating a domain at a sample `s` as ``domain(s)`` yields the corresponding world position, 
    and several operators are also provided domains, for example evaluating the normal at a given sample: ::
    
            @integrand
            def boundary_form(
                s: Sample,
                domain: Domain,
                u: Field,
            ):
                nor = normal(domain, s)
                return wp.dot(u(s), nor)

Integrands cannot be used directly with :func:`warp.launch`, but must be called through :func:`.integrate` or :func:`.interpolate` instead.
The :data:`.Sample` and :class:`.Domain` arguments of the root integrand (`integrand` parameter passed to :func:`integrate` or :func:`interpolate` call) will get automatically populated.
:class:`.Field` arguments must be passed as a dictionary in the `fields` parameter of the launcher function, and all other standard Warp types arguments must be
passed as a dictionary in the `values` parameter of the launcher function, for instance: ::
    
    integrate(diffusion_form, fields={"u": trial, "v": test}, values={"nu": viscosity})


Basic Workflow
--------------

The typical steps for solving a linearized PDE with :mod:`warp.fem` are as follow:

 - Define a :class:`.Geometry` (grid, mesh, etc). At the moment, 2D and 3D regular grids, NanoVDB volumes, and triangle, quadrilateral, tetrahedron and hexahedron unstructured meshes are supported.
 - Define one or more :class:`.FunctionSpace`, by equipping the geometry elements with shape functions. See :func:`.make_polynomial_space`. At the moment, continuous/discontinuous Lagrange (:math:`P_{k[d]}, Q_{k[d]}`) and Serendipity (:math:`S_k`) shape functions of order :math:`k \leq 3` are supported, as well as linear Nédélec (first kind) and Raviart-Thomas vector-valued shape functions. B-spline shape functions (:math:`B_k`, :math:`k \leq 3`) are also available for grid-based geometries.
 - Define an integration domain, for instance the geometry's cells (:class:`.Cells`) or boundary sides (:class:`.BoundarySides`).
 - Integrate linear forms to build the system's right-hand-side. Define a test function over the function space using :func:`.make_test`,
   a :class:`.Quadrature` formula (or let the module choose one based on the function space degree), and call :func:`.integrate` with the linear form integrand.
   The result is a :class:`warp.array` containing the integration result for each of the function space degrees of freedom.
 - Integrate bilinear forms to build the system's left-hand-side. Define a trial function over the function space using :func:`.make_trial`,
   then call :func:`.integrate` with the bilinear form integrand.
   The result is a :class:`warp.sparse.BsrMatrix` containing the integration result for each pair of test and trial function space degrees of freedom.
   Note that the trial and test functions do not have to be defined over the same function space, so that Mixed FEM is supported.
 - Solve the resulting linear system using the solver of your choice, for instance one of the built-in :ref:`iterative-linear-solvers`.


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
    matrix += bd_matrix * boundary_strength

    # Solve linear system using Conjugate Gradient
    x = wp.zeros_like(rhs)
    bsr_cg(matrix, b=rhs, x=x)


.. note::
    The :func:`.integrate` function does not check that the passed integrands are actually linear or bilinear forms; it is up to the user to ensure that they are.
    To solve non-linear PDEs, one can use an iterative procedure and pass the current value of the studied function :class:`.DiscreteField` argument to the integrand, in which
    arbitrary operations are permitted. However, the result of the form must remain linear in the test and trial fields. 
    This strategy is demonstrated in the ``example_mixed_elasticity.py`` example.

Introductory Examples
---------------------

:mod:`warp.fem` ships with a list of examples in the ``warp/examples/fem`` directory demonstrating how to solve classical model problems.

 - ``example_diffusion.py``: 2D diffusion with homogeneous Neumann and Dirichlet boundary conditions
     * ``example_diffusion_3d.py``: 3D variant of the diffusion problem
 - ``example_convection_diffusion.py``: 2D convection-diffusion using semi-Lagrangian advection
     * ``example_convection_diffusion_dg.py``: 2D convection-diffusion using Discontinuous Galerkin with upwind transport and Symmetric Interior Penalty
 - ``example_burgers.py``: 2D inviscid Burgers using Discontinuous Galerkin with upwind transport and slope limiter
 - ``example_stokes.py``: 2D incompressible Stokes flow using mixed :math:`P_k/P_{k-1}` or :math:`Q_k/P_{(k-1)d}` elements
 - ``example_navier_stokes.py``: 2D Navier-Stokes flow using mixed :math:`P_k/P_{k-1}` elements
 - ``example_mixed_elasticity.py``: 2D nonlinear elasticity using mixed continuous/discontinuous :math:`S_k/P_{(k-1)d}` elements
 - ``example_distortion_energy.py``: Parameterization of a 3D surface minimizing a 2D nonlinear distortion energy
 - ``example_magnetostatics.py``: 2D magnetostatics using a curl-curl formulation
 - ``example_elastic_shape_optimization.py``: Shape optimization of a 2D elastic cantilever beam 
 - ``example_darcy_ls_optimization.py``: Level-set-based optimization of a 2D Darcy flow

Advanced Usages
---------------

High-order (curved) geometries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is possible to convert any :class:`.Geometry` (grids and explicit meshes) into a curved, high-order variant by deforming them 
with an arbitrary-order displacement field using the :meth:`~.GeometryField.make_deformed_geometry` method. 
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

See ``example_deformed_geometry.py`` for a complete example.
It is also possible to define the deformation field from an :class:`ImplicitField`, as done in ``example_magnetostatics.py``.

.. _lookups:

Particle-based quadrature and position lookups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The global :obj:`.lookup` and :obj:`.partition_lookup` operators allow generating a :data:`.Sample` from an arbitrary position; this is illustrated in 
the ``example_streamlines.py`` example for generating 3D streamlines by tracing through a velocity field.

.. note::
   Non-grid-based geometry types require building a Bounding Volume Hierarchy (BVH) acceleration structure for :obj:`.lookup` and similar operators to be functional.
   This can be done by calling :meth:`.Geometry.build_bvh` or passing ``build_bvh=True`` to the geometry constructor. 
   In case the geometry vertex positions are later modified, the BVH can be refit using :meth:`.Geometry.update_bvh`.

This operator is also leveraged by the :class:`.PicQuadrature` to provide a way to define Particle-In-Cell quadratures from a set of arbitrary particles,
making it possible to implement MPM-type methods.
The particles are automatically bucketed to the geometry cells when the quadrature is initialized.
For GIMP (Generalized Interpolation Material Point) methods where particles can span multiple cells, 
:class:`.PicQuadrature` also accepts pre-computed cell indices, coordinates, and particle fractions as a tuple of 2D arrays.
This is illustrated by the ``example_stokes_transfer.py`` and ``example_apic_fluid.py`` examples.

Nonconforming fields
^^^^^^^^^^^^^^^^^^^^

Fields defined on a given :class:`.Geometry` cannot be directly used for integrating over a distinct geometry; 
however, they may be wrapped in a :class:`.NonconformingField` for this purpose. 
This is leveraged by the ``example_nonconforming_contact.py`` to simulate contacting bodies that are discretized separately.

.. note::
   Currently :class:`.NonconformingField` does not support wrapping a trial field, so it is not yet possible to define
   bilinear forms over different geometries.

.. note::
   The mapping between the different geometries is position based, so a :class:`.NonconformingField` is not able to accurately capture discontinuous function spaces.
   Moreover, the integration domain must support the :obj:`.lookup` operator (see :ref:`lookups`).

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

The :class:`.Subdomain` class can be used to integrate over a subset of elements while keeping the full set of degrees of freedom,
i.e, without reindexing; this is illustrated in the ``example_streamlines.py`` example to define inflow and outflow boundaries.

Adaptivity
^^^^^^^^^^

While unstructured mesh refinement is currently out of scope, :mod:`warp.fem` provides an adaptive version of the sparse grid geometry, :class:`.AdaptiveNanogrid`,
with power-of-two voxel scales. Helpers for building such geometries from hierarchy of grids or a refinement oracle are also provided, see 
:func:`.adaptive_nanogrid_from_field` and :func:`.adaptive_nanogrid_from_hierarchy`.
An example is provided in ``warp/examples/fem/example_adaptive_grid.py``.

.. note::
   The existence of "T-junctions" at resolution boundaries mean that usual tri-polynomial shape functions will no longer be globally
   continuous. Discontinuous--Galerkin or similar techniques may be used to take into account the "jump" at multi-resolution faces.

Memory management
^^^^^^^^^^^^^^^^^

Several :mod:`warp.fem` functions require allocating temporary buffers to perform their computations. 
If such functions are called many times in a tight loop, those many allocations and de-allocations may degrade performance,
though this is a lot less significant when :ref:`mempool_allocators` are in use.
To overcome this issue, a :class:`.TemporaryStore` object may be created to persist and reuse temporary allocations across calls,
either globally using :func:`set_default_temporary_store` or at a per-function granularity using the corresponding argument.

.. _Fields:

Fields
------

Fields represent functions defined over a geometry. The following field types are available:

- :class:`.DiscreteField`: A field defined by interpolating values at the nodes of a function space.
- :class:`.ImplicitField`: A field wrapping an arbitrary function.
- :class:`.UniformField`: A constant field with the same value everywhere.
- :class:`.NonconformingField`: A wrapper for evaluating fields defined on a different geometry.
- :class:`.GeometryField`: A field representing the deformation of a geometry.

Fields can be evaluated at a :data:`.Sample` using the call operator, e.g., ``u(s)`` evaluates field ``u`` at sample ``s``.

Additionally, test and trial fields (:class:`.field.TestField` and :class:`.field.TrialField`) are created using
:func:`.make_test` and :func:`.make_trial` for building linear and bilinear forms.

.. _Operators:

Operators
---------

The following operators are available for use within integrands:

**Field operators:**

- :obj:`.grad`: Gradient of a field.
- :obj:`.div`: Divergence of a vector field.
- :obj:`.curl`: Curl of a vector field.
- :obj:`.D`: Generic derivative operator.

**Domain operators:**

- :obj:`.position`: World position at a sample (same as calling ``domain(s)``).
- :obj:`.normal`: Normal vector at a sample on a boundary.
- :obj:`.measure`: Integration measure (area or volume element).
- :obj:`.measure_ratio`: Ratio of measures between a domain and its reference element.
- :obj:`.deformation_gradient`: Deformation gradient of the domain.

**Discontinuous Galerkin operators** (for use on interior sides):

- :obj:`.inner`: Value on the inner side of an interface.
- :obj:`.outer`: Value on the outer side of an interface.
- :obj:`.average`: Average of values across an interface.
- :obj:`.jump`: Jump of values across an interface.
- :obj:`.grad_average`: Average of gradients across an interface.
- :obj:`.grad_jump`: Jump of gradients across an interface.
- :obj:`.grad_outer`: Gradient on the outer side of an interface.
- :obj:`.div_outer`: Divergence on the outer side of an interface.

Visualization
-------------

Most function spaces define a ``vtk_cells`` method that returns a list of VTK-compatible cell types and node indices.
This can be used to visualize discrete fields in VTK-aware viewers such as ``pyvista``, for instance:

.. code-block:: python

   import numpy as np
   import pyvista

   import warp as wp
   import warp.fem as fem


   @fem.integrand
   def ackley(s: fem.Sample, domain: fem.Domain):
      x = domain(s)
      return (
         -20.0 * wp.exp(-0.2 * wp.sqrt(0.5 * wp.length_sq(x)))
         - wp.exp(0.5 * (wp.cos(2.0 * wp.pi * x[0]) + wp.cos(2.0 * wp.pi * x[1])))
         + wp.e
         + 20.0
      )


   # Define field
   geo = fem.Grid2D(res=wp.vec2i(64, 64), bounds_lo=wp.vec2(-4.0, -4.0), bounds_hi=wp.vec2(4.0, 4.0))
   space = fem.make_polynomial_space(geo, degree=3)
   field = space.make_field()
   fem.interpolate(ackley, dest=field)

   # Extract cells, nodes and values
   cells, types = field.space.vtk_cells()
   nodes = field.space.node_positions().numpy()
   values = field.dof_values.numpy()
   positions = np.hstack((nodes, values[:, np.newaxis]))

   # Visualize with pyvista
   grid = pyvista.UnstructuredGrid(cells, types, positions)
   grid.point_data["scalars"] = values
   plotter = pyvista.Plotter()
   plotter.add_mesh(grid)
   plotter.show()
